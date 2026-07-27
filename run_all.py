import os
# Force non-interactive Agg backend before matplotlib import to prevent IPython/Colab MPLBACKEND conflicts
os.environ['MPLBACKEND'] = 'Agg'

import json
import uuid
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from model import GatedDeltaNetLM
from dataset import get_dataloader
from train import train_model
from eval import benchmark_inference, needle_in_a_haystack_test, mqar_test, compute_statistical_tests

def is_seed_completed(config_dir, seed):
    """Checks if a seed configuration has already completed training and metrics logging."""
    metrics_file = os.path.join(config_dir, f"seed{seed}_metrics.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)
            # Verify required metric keys exist
            if "val_perplexity" in data and "val_loss" in data:
                return True, data
        except Exception:
            return False, None
    return False, None

def check_parallel_capacity(num_requested, scales_to_run):
    """
    Inspects free GPU memory via torch.cuda.mem_get_info() to verify if `num_requested` models
    can be trained concurrently. Applies an 85% safety threshold and falls back to 1 (sequential) if capacity is tight.
    """
    if num_requested <= 1:
        return 1
        
    if not torch.cuda.is_available():
        print("[Parallel Check] No CUDA GPU available. Defaulting PARALLEL_TRAININGS to 1 (Sequential).")
        return 1
        
    try:
        free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info()
        free_vram_gb = free_vram_bytes / (1024 ** 3)
        
        # Estimate max VRAM required per model (with our micro-batched 0.82 GB peak logits footprint)
        vram_estimates_gb = {"10M": 1.2, "30M": 2.2, "55M": 3.5, "100M": 5.5}
        max_scale_vram = max(vram_estimates_gb.get(s, 5.5) for s in scales_to_run)
        
        required_vram_gb = max_scale_vram * num_requested
        usable_vram_gb = free_vram_gb * 0.85  # 85% safety threshold
        
        if required_vram_gb > usable_vram_gb:
            print(f"\n[Parallel Check] WARNING: Insufficient VRAM! Needed ~{required_vram_gb:.1f} GB for {num_requested} parallel models, but only ~{usable_vram_gb:.1f} GB usable free VRAM is available.")
            print("  -> Defaulting gracefully to PARALLEL_TRAININGS = 1 (Sequential Execution).\n")
            return 1
        else:
            print(f"\n[Parallel Check] Sufficient GPU VRAM available (~{free_vram_gb:.1f} GB free). Running PARALLEL_TRAININGS = {num_requested}.\n")
            return num_requested
    except Exception as e:
        print(f"[Parallel Check] Could not inspect CUDA memory: {e}. Defaulting to PARALLEL_TRAININGS = 1.")
        return 1

def log_worker(worker_id, tag, message):
    """Formats log output lines with worker prefix for clear parallel monitoring."""
    prefix = f"[Worker-{worker_id} | {tag}]" if worker_id is not None else f"[{tag}]"
    print(f"{prefix} {message}")

def process_model_config(scale, variant, worker_id, run_base_dir, global_results, tracker_lock):
    """Processes all seeds for a single model configuration, then unloads model from memory."""
    tag = f"{scale}-{variant}"
    config_dir = os.path.join(run_base_dir, f"run_{variant}_{scale}")
    os.makedirs(config_dir, exist_ok=True)
    
    log_worker(worker_id, tag, f"Starting processing for Scale={scale}, Variant={variant}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_seed_metrics = []
    
    for seed in config.SEEDS:
        # Check for run resumption / completed cache
        completed, cached_metrics = is_seed_completed(config_dir, seed)
        if completed:
            log_worker(worker_id, tag, f"[CACHE HIT] Seed {seed} metrics found. Restoring metrics from disk...")
            ms_per_token = cached_metrics.get("latency_ms_per_token", float('nan'))
            tokens_per_sec = cached_metrics.get("throughput_tokens_per_sec", float('nan'))
            retrieval_acc = cached_metrics.get("retrieval_accuracy", float('nan'))
            mqar_acc = cached_metrics.get("mqar_accuracy", float('nan'))
            
            seed_res = {
                "scale": scale,
                "variant": variant,
                "seed": seed,
                "parameter_count": cached_metrics.get("parameter_count", 0),
                "val_loss": cached_metrics.get("val_loss", float('nan')),
                "val_perplexity": cached_metrics.get("val_perplexity", float('nan')),
                "latency_ms_per_token": ms_per_token,
                "throughput_tokens_per_sec": tokens_per_sec,
                "retrieval_accuracy": retrieval_acc,
                "peak_memory_mb": cached_metrics.get("peak_memory_mb", float('nan')),
                "mqar_accuracy": mqar_acc
            }
            with tracker_lock:
                global_results.append(seed_res)
            config_seed_metrics.append(seed_res)
            continue
            
        log_worker(worker_id, tag, f"Initializing seed-controlled dataloaders for Seed {seed}...")
        dataloader_train = get_dataloader("train", batch_size=config.BATCH_SIZE, seed=seed, run_dir=run_base_dir)
        dataloader_val = get_dataloader("validation", batch_size=config.BATCH_SIZE, seed=seed, run_dir=run_base_dir)
        
        model = None
        try:
            # 1. Train model
            log_worker(worker_id, tag, f"Starting Training Loop (Seed {seed})...")
            model, train_metrics = train_model(
                scale=scale,
                variant=variant,
                seed=seed,
                dataloader_train=dataloader_train,
                dataloader_val=dataloader_val,
                run_dir=config_dir
            )
            
            # 2. Benchmarking
            log_worker(worker_id, tag, f"Running inference benchmark (Seed {seed})...")
            ms_per_token, tokens_per_sec = benchmark_inference(model, device)
            log_worker(worker_id, tag, f"Throughput: {tokens_per_sec:.2f} tok/s, Latency: {ms_per_token:.2f} ms/tok")
            
            # 3. Needle-in-a-Haystack Retrieval
            log_worker(worker_id, tag, f"Running needle retrieval probe (Seed {seed})...")
            retrieval_acc = needle_in_a_haystack_test(model, device, num_trials=5 if config.SMOKE_TEST else 20)
            log_worker(worker_id, tag, f"Retrieval Accuracy: {retrieval_acc * 100:.1f}%")

            # 4. Multi-Query Associative Recall (MQAR) Test
            log_worker(worker_id, tag, f"Running MQAR test (Seed {seed})...")
            mqar_acc = mqar_test(model, device, num_pairs=3 if config.SMOKE_TEST else 5, num_trials=5 if config.SMOKE_TEST else 20)
            log_worker(worker_id, tag, f"MQAR Accuracy: {mqar_acc * 100:.1f}%")
            
            # Update metrics JSON file
            metrics_file = os.path.join(config_dir, f"seed{seed}_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    m_data = json.load(f)
                m_data.update({
                    "latency_ms_per_token": ms_per_token,
                    "throughput_tokens_per_sec": tokens_per_sec,
                    "retrieval_accuracy": retrieval_acc,
                    "mqar_accuracy": mqar_acc
                })
                with open(metrics_file, "w") as f:
                    json.dump(m_data, f, indent=4)

            seed_res = {
                "scale": scale,
                "variant": variant,
                "seed": seed,
                "parameter_count": train_metrics["parameter_count"],
                "val_loss": train_metrics["val_loss"],
                "val_perplexity": train_metrics["val_perplexity"],
                "latency_ms_per_token": ms_per_token,
                "throughput_tokens_per_sec": tokens_per_sec,
                "retrieval_accuracy": retrieval_acc,
                "peak_memory_mb": train_metrics["peak_memory_mb"],
                "mqar_accuracy": mqar_acc
            }
            with tracker_lock:
                global_results.append(seed_res)
            config_seed_metrics.append(seed_res)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log_worker(worker_id, tag, f"[WARNING] OOM encountered for Scale={scale}, Seed={seed}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            else:
                raise e
        finally:
            # Memory cleanup after seed execution
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    # Perform seed-level aggregation calculation for this specific (scale, variant)
    val_losses = [r["val_loss"] for r in config_seed_metrics]
    val_perplexities = [r["val_perplexity"] for r in config_seed_metrics]
    throughputs = [r["throughput_tokens_per_sec"] for r in config_seed_metrics]
    accuracies = [r["retrieval_accuracy"] for r in config_seed_metrics]
    memories = [r["peak_memory_mb"] for r in config_seed_metrics]
    mqars = [r["mqar_accuracy"] for r in config_seed_metrics]
    param_count = config_seed_metrics[0]["parameter_count"] if config_seed_metrics else 0
    
    def get_nan_mean(lst):
        val = np.nanmean(lst)
        return float(val) if not np.isnan(val) else 0.0
        
    def get_nan_std(lst):
        val = np.nanstd(lst)
        return float(val) if not np.isnan(val) else 0.0
    
    agg_metrics = {
        "scale": scale,
        "variant": variant,
        "parameter_count": param_count,
        "seeds_evaluated": config.SEEDS,
        "mean_val_loss": get_nan_mean(val_losses),
        "std_val_loss": get_nan_std(val_losses),
        "mean_val_perplexity": get_nan_mean(val_perplexities),
        "std_val_perplexity": get_nan_std(val_perplexities),
        "mean_throughput_tokens_per_sec": get_nan_mean(throughputs),
        "std_throughput_tokens_per_sec": get_nan_std(throughputs),
        "mean_retrieval_accuracy": get_nan_mean(accuracies),
        "std_retrieval_accuracy": get_nan_std(accuracies),
        "mean_peak_memory_mb": get_nan_mean(memories),
        "std_peak_memory_mb": get_nan_std(memories),
        "mean_mqar_accuracy": get_nan_mean(mqars),
        "std_mqar_accuracy": get_nan_std(mqars)
    }
    
    agg_metrics_path = os.path.join(config_dir, "aggregate_metrics.json")
    with open(agg_metrics_path, "w") as f:
        json.dump(agg_metrics, f, indent=4)
        
    log_worker(worker_id, tag, f"Finished all seeds. Unloaded model and cleared VRAM cache.\n")
    
    with tracker_lock:
        global_tracker_path = os.path.join(run_base_dir, "global_tracker.json")
        with open(global_tracker_path, "w") as f:
            json.dump(list(global_results), f, indent=4)

def generate_plots(results_df, run_base_dir, stats_results=None):
    """
    Generates publication-grade scaling plots with error bars (std dev)
    and statistical significance annotations. Also plots loss convergence trajectories.
    """
    agg = results_df.groupby(["scale", "variant"]).agg({
        "parameter_count": ["mean"],
        "val_perplexity": ["mean", "std"],
        "throughput_tokens_per_sec": ["mean", "std"],
        "retrieval_accuracy": ["mean", "std"],
        "peak_memory_mb": ["mean", "std"],
        "mqar_accuracy": ["mean", "std"]
    })
    agg.columns = ['_'.join(c).strip('_') for c in agg.columns]
    agg = agg.reset_index()
    
    scale_order = {s: i for i, s in enumerate(config.SCALES)}
    agg["scale_rank"] = agg["scale"].map(scale_order)
    agg = agg.sort_values("scale_rank")
    
    scalar_agg = agg[agg["variant"] == "scalar"]
    cw_agg = agg[agg["variant"] == "channel-wise"]
    
    metrics_info = [
        ("val_perplexity", "Validation Perplexity", "Validation Perplexity vs. Model Scale", "perplexity_scaling.png", False, 1.0),
        ("throughput_tokens_per_sec", "Throughput (tokens/sec)", "Decoding Throughput vs. Model Scale", "throughput_scaling.png", False, 1.0),
        ("retrieval_accuracy", "Retrieval Accuracy (%)", "Needle Retrieval Accuracy vs. Model Scale", "retrieval_scaling.png", True, 100.0),
        ("peak_memory_mb", "Peak Memory (MB)", "Peak Memory Usage vs. Model Scale", "memory_scaling.png", False, 1.0),
        ("mqar_accuracy", "MQAR Accuracy (%)", "MQAR Accuracy vs. Model Scale", "mqar_scaling.png", True, 100.0)
    ]
    
    for metric_col, ylabel, title, filename, is_pct, scale_factor in metrics_info:
        plt.figure(figsize=(7.5, 4.8))
        
        s_mean = scalar_agg[f"{metric_col}_mean"] * scale_factor
        s_std = scalar_agg[f"{metric_col}_std"].fillna(0.0) * scale_factor
        
        c_mean = cw_agg[f"{metric_col}_mean"] * scale_factor
        c_std = cw_agg[f"{metric_col}_std"].fillna(0.0) * scale_factor
        
        x_indices = np.arange(len(config.SCALES))
        
        plt.errorbar(
            x_indices, s_mean, yerr=s_std,
            marker='o', color="#1f77b4", label="Scalar Gating (A)",
            linewidth=2, linestyle='--', capsize=4, elinewidth=1.5
        )
        plt.errorbar(
            x_indices, c_mean, yerr=c_std,
            marker='s', color="#ff7f0e", label="Channel-wise Gating (B)",
            linewidth=2, capsize=4, elinewidth=1.5
        )
        
        if stats_results:
            for idx, scale in enumerate(config.SCALES):
                if scale in stats_results and metric_col in stats_results[scale]:
                    sig = stats_results[scale][metric_col].get("significance", "ns")
                    sig_label = ""
                    if "**" in sig:
                        sig_label = "**"
                    elif "*" in sig:
                        sig_label = "*"
                    
                    if sig_label:
                        y_pos = max(s_mean.iloc[idx] + s_std.iloc[idx], c_mean.iloc[idx] + c_std.iloc[idx])
                        plt.text(idx, y_pos * 1.02, sig_label, ha='center', va='bottom', fontsize=11, fontweight='bold', color='#d62728')
                        
        plt.xticks(x_indices, config.SCALES)
        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel("Model Scale", fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        if is_pct:
            plt.ylim(-5, 108)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_base_dir, filename), dpi=150)
        plt.close()
        
    for scale in config.SCALES:
        plt.figure(figsize=(7.5, 4.5))
        has_loss_data = False
        
        for variant, color, ls, label_name in [("scalar", "#1f77b4", "--", "Scalar Gating (A)"), ("channel-wise", "#ff7f0e", "-", "Channel-wise Gating (B)")]:
            config_dir = os.path.join(run_base_dir, f"run_{variant}_{scale}")
            seed_losses = []
            
            for seed in config.SEEDS:
                metrics_file = os.path.join(config_dir, f"seed{seed}_metrics.json")
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, "r") as f:
                            m = json.load(f)
                        if "train_losses" in m and len(m["train_losses"]) > 0:
                            seed_losses.append(m["train_losses"])
                    except Exception:
                        pass
                        
            if seed_losses:
                min_len = min(len(l) for l in seed_losses)
                arr = np.array([l[:min_len] for l in seed_losses])
                mean_loss = np.mean(arr, axis=0)
                std_loss = np.std(arr, axis=0)
                steps = np.arange(1, min_len + 1)
                
                plt.plot(steps, mean_loss, color=color, linestyle=ls, label=label_name, linewidth=2)
                plt.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.2)
                has_loss_data = True
                
        if has_loss_data:
            plt.title(f"Training Loss Trajectory - {scale} Scale", fontsize=12, fontweight='bold')
            plt.xlabel("Step", fontsize=10)
            plt.ylabel("Training Loss", fontsize=10)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(run_base_dir, f"loss_convergence_{scale}.png"), dpi=150)
            plt.close()

    print("Research scaling plots with error bars and loss trajectory curves saved successfully.")

def main():
    resume_id = getattr(config, "RESUME_RUN_ID", None)
    if resume_id:
        run_uuid = resume_id
        run_base_dir = os.path.join(config.OUTPUT_DIR, run_uuid if run_uuid.startswith("run_") else f"run_{run_uuid}")
        print(f"RESUMING / TARGETING specified run directory: {run_base_dir}")
    else:
        run_uuid = str(uuid.uuid4())[:8]
        run_base_dir = os.path.join(config.OUTPUT_DIR, f"run_{run_uuid}")
        
    os.makedirs(run_base_dir, exist_ok=True)
    
    print("==================================================")
    print(" GATED DELTANET GATING STUDY ORCHESTRATOR ")
    print(f" Run Identifier: {run_uuid}")
    print(f" Output Location: {run_base_dir}")
    print("==================================================")
    print(f"SMOKE_TEST Mode: {config.SMOKE_TEST}")
    print(f"Scales to run: {config.SCALES}")
    print(f"Variants to run: {config.VARIANTS}")
    print(f"Seeds to run: {config.SEEDS}")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    requested_parallel = getattr(config, "PARALLEL_TRAININGS", 1)
    parallel_jobs = check_parallel_capacity(requested_parallel, config.SCALES)
    
    tasks_list = []
    for scale in config.SCALES:
        for variant in config.VARIANTS:
            tasks_list.append((scale, variant))
            
    global_results = []
    tracker_lock = threading.Lock()
    
    if parallel_jobs > 1:
        print(f"\n[PARALLEL] Launching Parallel Worker Pool with {parallel_jobs} concurrent model jobs...\n")
        with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = []
            for idx, (scale, variant) in enumerate(tasks_list):
                worker_id = (idx % parallel_jobs) + 1
                f = executor.submit(process_model_config, scale, variant, worker_id, run_base_dir, global_results, tracker_lock)
                futures.append(f)
            for f in futures:
                f.result()
    else:
        print("\n[SEQUENTIAL] Running Model Configurations Sequentially...\n")
        for scale, variant in tasks_list:
            process_model_config(scale, variant, worker_id=1, run_base_dir=run_base_dir, global_results=global_results, tracker_lock=tracker_lock)
            
    # Post-run reporting & hypothesis testing
    df = pd.DataFrame(global_results)
    csv_path = os.path.join(run_base_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    
    print("\nExecuting multi-seed statistical hypothesis testing...")
    stats_results = compute_statistical_tests(df, run_base_dir)
    
    generate_plots(df, run_base_dir, stats_results=stats_results)
    
    def fmt_val(mean, std, scale_factor=1.0, is_pct=False):
        if pd.isna(mean):
            return "N/A"
        mean_val = mean * scale_factor
        std_val = std * scale_factor if not pd.isna(std) else None
        suffix = "%" if is_pct else ""
        if std_val is not None:
            return f"{mean_val:.2f}{suffix} ± {std_val:.2f}{suffix}" if not is_pct else f"{mean_val:.1f}{suffix} ± {std_val:.1f}{suffix}"
        return f"{mean_val:.2f}{suffix}" if not is_pct else f"{mean_val:.1f}{suffix}"
        
    summary_data = []
    grouped = df.groupby(["scale", "variant"])
    for (scale, variant), group in grouped:
        p_count = group["parameter_count"].iloc[0]
        mean_ppl = group["val_perplexity"].mean()
        std_ppl = group["val_perplexity"].std()
        mean_tps = group["throughput_tokens_per_sec"].mean()
        std_tps = group["throughput_tokens_per_sec"].std()
        mean_acc = group["retrieval_accuracy"].mean()
        std_acc = group["retrieval_accuracy"].std()
        mean_mem = group["peak_memory_mb"].mean()
        std_mem = group["peak_memory_mb"].std()
        mean_mqar = group["mqar_accuracy"].mean()
        std_mqar = group["mqar_accuracy"].std()
        
        summary_data.append({
            "Scale": scale,
            "Variant": variant,
            "Parameters": f"{p_count:,}" if not pd.isna(p_count) else "N/A",
            "Val Perplexity": fmt_val(mean_ppl, std_ppl),
            "Throughput (tok/s)": fmt_val(mean_tps, std_tps),
            "Retrieval Acc (%)": fmt_val(mean_acc, std_acc, scale_factor=100.0, is_pct=True),
            "Peak Memory (MB)": fmt_val(mean_mem, std_mem),
            "MQAR Acc (%)": fmt_val(mean_mqar, std_mqar, scale_factor=100.0, is_pct=True)
        })
        
    df_summary = pd.DataFrame(summary_data)
    md_table = df_summary.to_markdown(index=False)
    
    md_summary = f"""# Gated DeltaNet Gating Granularity Study Summary

## Execution Run Identifier: {run_uuid}
*Device: `{device}`*

### Aggregated Performance (Mean ± Std Dev)

{md_table}

---
### Statistical Reports & Scaling Artifacts Generated:
1. `statistical_report.md` (Detailed Wilcoxon signed-rank, t-test, Cohen's d, and 95% CIs)
2. `statistical_analysis.json` (Structured JSON statistical analysis)
3. `perplexity_scaling.png` (Validation Perplexity with error bars)
4. `throughput_scaling.png` (Tokens/sec with error bars)
5. `retrieval_scaling.png` (Needle retrieval accuracy with error bars)
6. `memory_scaling.png` (Peak Memory in MB with error bars)
7. `mqar_scaling.png` (MQAR accuracy with error bars)
8. `loss_convergence_*.png` (Step-wise training loss trajectories per scale)
"""
    
    summary_md_path = os.path.join(run_base_dir, "summary_table.md")
    with open(summary_md_path, "w") as f:
        f.write(md_summary)
        
    print("\n==================================================")
    print(" SUMMARY TABLE ")
    print("==================================================")
    print(md_table)
    print(f"\nAll outputs successfully saved to {run_base_dir}")
    print("==================================================")

if __name__ == "__main__":
    main()
