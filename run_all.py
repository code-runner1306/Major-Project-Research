import os
import json
import uuid
import torch
import pandas as pd
import numpy as np

# Non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from dataset import get_dataloader
from train import train_model
from eval import benchmark_inference, needle_in_a_haystack_test, mqar_test

def generate_plots(results_df, run_base_dir):
    # Calculate mean metrics grouped by scale and variant
    agg_df = results_df.groupby(["scale", "variant"]).agg({
        "parameter_count": "mean",
        "val_perplexity": "mean",
        "throughput_tokens_per_sec": "mean",
        "retrieval_accuracy": "mean",
        "peak_memory_mb": "mean",
        "mqar_accuracy": "mean"
    }).reset_index()
    
    # Sort scales logically
    scale_order = {s: i for i, s in enumerate(config.SCALES)}
    agg_df["scale_rank"] = agg_df["scale"].map(scale_order)
    agg_df = agg_df.sort_values("scale_rank")
    
    scalar_df = agg_df[agg_df["variant"] == "scalar"]
    cw_df = agg_df[agg_df["variant"] == "channel-wise"]
    
    # 1. Perplexity Plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(scalar_df["scale"], scalar_df["val_perplexity"], marker='o', color="#1f77b4", label="Scalar Gating (A)", linewidth=2)
    plt.plot(cw_df["scale"], cw_df["val_perplexity"], marker='s', color="#ff7f0e", label="Channel-wise Gating (B)", linewidth=2)
    plt.title("Validation Perplexity vs. Model Scale", fontsize=12, fontweight='bold')
    plt.xlabel("Model Scale", fontsize=10)
    plt.ylabel("Validation Perplexity", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_base_dir, "perplexity_scaling.png"), dpi=150)
    plt.close()
    
    # 2. Throughput Plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(scalar_df["scale"], scalar_df["throughput_tokens_per_sec"], marker='o', color="#1f77b4", label="Scalar Gating (A)", linewidth=2)
    plt.plot(cw_df["scale"], cw_df["throughput_tokens_per_sec"], marker='s', color="#ff7f0e", label="Channel-wise Gating (B)", linewidth=2)
    plt.title("Decoding Throughput vs. Model Scale", fontsize=12, fontweight='bold')
    plt.xlabel("Model Scale", fontsize=10)
    plt.ylabel("Throughput (tokens/second)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_base_dir, "throughput_scaling.png"), dpi=150)
    plt.close()
    
    # 3. Retrieval Accuracy Plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(scalar_df["scale"], scalar_df["retrieval_accuracy"] * 100, marker='o', color="#1f77b4", label="Scalar Gating (A)", linewidth=2)
    plt.plot(cw_df["scale"], cw_df["retrieval_accuracy"] * 100, marker='s', color="#ff7f0e", label="Channel-wise Gating (B)", linewidth=2)
    plt.title("Needle Retrieval Accuracy vs. Model Scale", fontsize=12, fontweight='bold')
    plt.xlabel("Model Scale", fontsize=10)
    plt.ylabel("Retrieval Accuracy (%)", fontsize=10)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_base_dir, "retrieval_scaling.png"), dpi=150)
    plt.close()

    # 4. Memory Footprint Plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(scalar_df["scale"], scalar_df["peak_memory_mb"], marker='o', color="#1f77b4", label="Scalar Gating (A)", linewidth=2)
    plt.plot(cw_df["scale"], cw_df["peak_memory_mb"], marker='s', color="#ff7f0e", label="Channel-wise Gating (B)", linewidth=2)
    plt.title("Peak Memory Usage vs. Model Scale", fontsize=12, fontweight='bold')
    plt.xlabel("Model Scale", fontsize=10)
    plt.ylabel("Peak Memory (MB)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_base_dir, "memory_scaling.png"), dpi=150)
    plt.close()

    # 5. MQAR Accuracy Plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(scalar_df["scale"], scalar_df["mqar_accuracy"] * 100, marker='o', color="#1f77b4", label="Scalar Gating (A)", linewidth=2)
    plt.plot(cw_df["scale"], cw_df["mqar_accuracy"] * 100, marker='s', color="#ff7f0e", label="Channel-wise Gating (B)", linewidth=2)
    plt.title("MQAR Accuracy vs. Model Scale", fontsize=12, fontweight='bold')
    plt.xlabel("Model Scale", fontsize=10)
    plt.ylabel("MQAR Accuracy (%)", fontsize=10)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_base_dir, "mqar_scaling.png"), dpi=150)
    plt.close()
    
    print("Scaling performance charts (5 figures) saved successfully.")

def main():
    # 1. Generate run-specific UUID
    run_uuid = str(uuid.uuid4())[:8]
    run_base_dir = os.path.join(config.OUTPUT_DIR, f"run_{run_uuid}")
    os.makedirs(run_base_dir, exist_ok=True)
    
    print("==================================================")
    print(" GATED DELTANET GATING STUDY ORCHESTRATOR ")
    print(f" Run UUID: {run_uuid}")
    print(f" Output Location: {run_base_dir}")
    print("==================================================")
    print(f"SMOKE_TEST Mode: {config.SMOKE_TEST}")
    print(f"Scales to run: {config.SCALES}")
    print(f"Variants to run: {config.VARIANTS}")
    print(f"Seeds to run: {config.SEEDS}")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize streaming dataloaders
    print("Initializing streaming dataloaders...")
    dataloader_train = get_dataloader("train", batch_size=config.BATCH_SIZE)
    dataloader_val = get_dataloader("validation", batch_size=config.BATCH_SIZE)
    
    results = []
    run_counter = 1
    total_runs = len(config.SCALES) * len(config.VARIANTS) * len(config.SEEDS)
    
    # Run the grid search sequentially
    for scale in config.SCALES:
        for variant in config.VARIANTS:
            # Create a directory specifically for this scale and variant
            config_dir = os.path.join(run_base_dir, f"run_{variant}_{scale}")
            os.makedirs(config_dir, exist_ok=True)
            
            config_seed_metrics = []
            
            for seed in config.SEEDS:
                print(f"\n>>> Running configuration {run_counter}/{total_runs} (Scale={scale}, Variant={variant}, Seed={seed}) <<<")
                
                # 1. Train model
                model, train_metrics = train_model(
                    scale=scale,
                    variant=variant,
                    seed=seed,
                    dataloader_train=dataloader_train,
                    dataloader_val=dataloader_val,
                    run_dir=config_dir
                )
                
                # 2. Benchmarking
                print("Running inference benchmark...")
                ms_per_token, tokens_per_sec = benchmark_inference(model, device)
                print(f"Throughput: {tokens_per_sec:.2f} tokens/sec, Latency: {ms_per_token:.2f} ms/token")
                
                # 3. Needle-in-a-Haystack Retrieval
                print("Running needle-in-a-haystack retrieval probe...")
                retrieval_acc = needle_in_a_haystack_test(model, device, num_trials=5 if config.SMOKE_TEST else 20)
                print(f"Retrieval Accuracy: {retrieval_acc * 100:.1f}%")

                # 4. Multi-Query Associative Recall (MQAR) Test
                print("Running multi-query associative recall (MQAR) test...")
                mqar_acc = mqar_test(model, device, num_pairs=3 if config.SMOKE_TEST else 5, num_trials=5 if config.SMOKE_TEST else 20)
                print(f"MQAR Accuracy: {mqar_acc * 100:.1f}%")
                
                # Record result trace
                seed_res = {
                    "run_id": run_counter,
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
                results.append(seed_res)
                config_seed_metrics.append(seed_res)
                
                # Save global tracker progressively to base folder
                global_tracker_path = os.path.join(run_base_dir, "global_tracker.json")
                with open(global_tracker_path, "w") as f:
                    json.dump(results, f, indent=4)
                    
                run_counter += 1
                
            # Perform seed-level aggregation calculation for this specific (scale, variant)
            val_losses = [r["val_loss"] for r in config_seed_metrics]
            val_perplexities = [r["val_perplexity"] for r in config_seed_metrics]
            throughputs = [r["throughput_tokens_per_sec"] for r in config_seed_metrics]
            accuracies = [r["retrieval_accuracy"] for r in config_seed_metrics]
            memories = [r["peak_memory_mb"] for r in config_seed_metrics]
            mqars = [r["mqar_accuracy"] for r in config_seed_metrics]
            param_count = config_seed_metrics[0]["parameter_count"]
            
            agg_metrics = {
                "scale": scale,
                "variant": variant,
                "parameter_count": param_count,
                "seeds_evaluated": config.SEEDS,
                "mean_val_loss": float(np.mean(val_losses)),
                "std_val_loss": float(np.std(val_losses)),
                "mean_val_perplexity": float(np.mean(val_perplexities)),
                "std_val_perplexity": float(np.std(val_perplexities)),
                "mean_throughput_tokens_per_sec": float(np.mean(throughputs)),
                "std_throughput_tokens_per_sec": float(np.std(throughputs)),
                "mean_retrieval_accuracy": float(np.mean(accuracies)),
                "std_retrieval_accuracy": float(np.std(accuracies)),
                "mean_peak_memory_mb": float(np.mean(memories)),
                "std_peak_memory_mb": float(np.std(memories)),
                "mean_mqar_accuracy": float(np.mean(mqars)),
                "std_mqar_accuracy": float(np.std(mqars))
            }
            
            agg_metrics_path = os.path.join(config_dir, "aggregate_metrics.json")
            with open(agg_metrics_path, "w") as f:
                json.dump(agg_metrics, f, indent=4)
            print(f"Aggregated statistics saved to {agg_metrics_path}")
            
    # Post-run plotting and reporting
    df = pd.DataFrame(results)
    
    # Save raw CSV in UUID base directory
    csv_path = os.path.join(run_base_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    
    # Create matplotlib scaling plots
    generate_plots(df, run_base_dir)
    
    # Group results for markdown output table
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
            "Parameters": f"{p_count:,}",
            "Val Perplexity": f"{mean_ppl:.2f} ± {std_ppl:.2f}" if not pd.isna(std_ppl) else f"{mean_ppl:.2f}",
            "Throughput (tok/s)": f"{mean_tps:.2f} ± {std_tps:.2f}" if not pd.isna(std_tps) else f"{mean_tps:.2f}",
            "Retrieval Acc (%)": f"{mean_acc * 100:.1f}% ± {std_acc * 100:.1f}%" if not pd.isna(std_acc) else f"{mean_acc * 100:.1f}%",
            "Peak Memory (MB)": f"{mean_mem:.1f} ± {std_mem:.1f}" if not pd.isna(std_mem) else f"{mean_mem:.1f}",
            "MQAR Acc (%)": f"{mean_mqar * 100:.1f}% ± {std_mqar * 100:.1f}%" if not pd.isna(std_mqar) else f"{mean_mqar * 100:.1f}%"
        })
        
    df_summary = pd.DataFrame(summary_data)
    md_table = df_summary.to_markdown(index=False)
    
    md_summary = f"""# Gated DeltaNet Gating Granularity Study Summary

## Execution Run ID: {run_uuid}
*Device: `{device}`*

### Aggregated Performance (Mean ± Std Dev)

{md_table}

---
### Scaling Charts Generated:
1. `perplexity_scaling.png` (Validation Perplexity vs. Parameters)
2. `throughput_scaling.png` (Tokens per second vs. Parameters)
3. `retrieval_scaling.png` (Needle retrieval accuracy vs. Parameters)
4. `memory_scaling.png` (Peak Memory in MB vs. Parameters)
5. `mqar_scaling.png` (MQAR accuracy vs. Parameters)
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
