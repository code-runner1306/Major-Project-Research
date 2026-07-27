import time
import random
import torch
import tiktoken
import config

def benchmark_inference(model, device, prompt_len=None, gen_len=None):
    if prompt_len is None:
        prompt_len = 4 if config.SMOKE_TEST else config.CONTEXT_LENGTH
    if gen_len is None:
        gen_len = 2 if config.SMOKE_TEST else config.CONTEXT_LENGTH
        
    model.eval()
    warmup = config.BENCHMARK_WARMUP
    repeats = config.BENCHMARK_REPEATS
    
    # Generate dummy input prompt
    x = torch.randint(0, config.VOCAB_SIZE, (1, prompt_len), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            logits, _ = model(x)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
    # Timed trials
    times = []
    for _ in range(repeats):
        curr_x = x.clone()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(gen_len):
                logits, _ = model(curr_x)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr_x = torch.cat([curr_x, next_token], dim=-1)
                
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end = time.perf_counter()
        times.append(end - start)
        
    avg_time = sum(times) / len(times)
    ms_per_token = (avg_time / gen_len) * 1000
    tokens_per_sec = gen_len / avg_time
    
    return ms_per_token, tokens_per_sec

def needle_in_a_haystack_test(model, device, context_length=None, num_trials=10):
    if context_length is None:
        context_length = config.NEEDLE_LENGTH
        
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    
    distractors = [
        "Once upon a time, there was a little girl who loved to run.",
        "The small green frog sat on a big leaf in the pond.",
        "A shiny red car drove fast down the bumpy road.",
        "The dog barked loudly at the bird sitting on the wooden fence.",
        "She picked a yellow flower and smelled its sweet scent.",
        "The blue sky was covered in fluffy white clouds today.",
        "He read a big book about dragons and castles before bedtime.",
        "A tiny mouse ate a piece of yellow cheese on the kitchen floor.",
        "They built a tall sandcastle near the water at the beach."
    ]
    
    colors = ["blue", "green", "red", "pink", "gold"]
    successes = 0
    
    for _ in range(num_trials):
        target_color = random.choice(colors)
        needle_str = f" The secret key is {target_color}."
        question_str = " What is the secret key? The secret key is"
        
        needle_tokens = tokenizer.encode(needle_str)
        question_tokens = tokenizer.encode(question_str)
        
        # Token ids for target color (could be multi-token but simple color words are usually 1 token in GPT-2)
        target_token_id = tokenizer.encode(" " + target_color)[0]
        
        # Fill haystack with distractors
        haystack_tokens = []
        while len(haystack_tokens) < context_length - len(needle_tokens) - len(question_tokens):
            phrase = random.choice(distractors)
            haystack_tokens.extend(tokenizer.encode(" " + phrase))
            
        # Truncate to make room for needle and question
        max_haystack_len = context_length - len(needle_tokens) - len(question_tokens)
        haystack_tokens = haystack_tokens[:max_haystack_len]
        
        # Insert needle at random depth (between 10% and 90% of haystack)
        depth_ratio = random.uniform(0.1, 0.9)
        insert_idx = int(len(haystack_tokens) * depth_ratio)
        
        prompt_tokens = (
            haystack_tokens[:insert_idx] +
            needle_tokens +
            haystack_tokens[insert_idx:] +
            question_tokens
        )
        
        # Trim to exact context_length
        prompt_tokens = prompt_tokens[:context_length]
        
        # Run inference
        x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(x)
            predicted_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
            
        if predicted_token_id == target_token_id:
            successes += 1
            
    accuracy = successes / num_trials
    return accuracy

def mqar_test(model, device, num_pairs=4, num_trials=10):
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    
    keys = ["blue", "red", "green", "pink", "gold", "black", "white", "yellow"]
    values = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    
    successes = 0
    for _ in range(num_trials):
        chosen_keys = random.sample(keys, num_pairs)
        chosen_values = random.sample(values, num_pairs)
        
        pair_strings = []
        for k, v in zip(chosen_keys, chosen_values):
            pair_strings.append(f" {k} is {v}")
            
        prompt = ",".join(pair_strings)
        
        query_idx = random.randint(0, num_pairs - 1)
        query_key = chosen_keys[query_idx]
        target_val = chosen_values[query_idx]
        
        prompt += f". Question: What is {query_key}? Answer: {query_key} is"
        
        prompt_tokens = tokenizer.encode(prompt)
        target_token_id = tokenizer.encode(" " + target_val)[0]
        
        x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(x)
            predicted_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
            
        if predicted_token_id == target_token_id:
            successes += 1
            
    return successes / num_trials

def compute_statistical_tests(df, run_base_dir):
    """
    Computes Wilcoxon signed-rank tests, paired t-tests, Cohen's d effect sizes,
    and 95% confidence intervals comparing Variant A (scalar) vs Variant B (channel-wise)
    across seeds for every scale.
    """
    import os
    import json
    import numpy as np
    from scipy import stats
    
    metrics_to_test = [
        "val_perplexity",
        "throughput_tokens_per_sec",
        "retrieval_accuracy",
        "mqar_accuracy",
        "peak_memory_mb"
    ]
    
    scales = df["scale"].unique()
    statistical_results = {}
    
    for scale in scales:
        scale_df = df[df["scale"] == scale]
        scale_stats = {}
        
        for metric in metrics_to_test:
            scalar_vals = scale_df[scale_df["variant"] == "scalar"].sort_values("seed")[metric].values
            cw_vals = scale_df[scale_df["variant"] == "channel-wise"].sort_values("seed")[metric].values
            
            # Ensure matching lengths and valid numeric values
            mask = ~np.isnan(scalar_vals) & ~np.isnan(cw_vals)
            s_arr = scalar_vals[mask]
            c_arr = cw_vals[mask]
            
            if len(s_arr) < 2:
                scale_stats[metric] = {
                    "scalar_mean": float(np.nanmean(scalar_vals)) if len(scalar_vals) else None,
                    "channel_wise_mean": float(np.nanmean(cw_vals)) if len(cw_vals) else None,
                    "mean_diff": None,
                    "rel_change_pct": None,
                    "cohen_d": None,
                    "ttest_p_value": None,
                    "wilcoxon_p_value": None,
                    "ci_95_lower": None,
                    "ci_95_upper": None,
                    "significance": "N/A"
                }
                continue
                
            mean_s = float(np.mean(s_arr))
            mean_c = float(np.mean(c_arr))
            diffs = c_arr - s_arr
            mean_diff = float(np.mean(diffs))
            rel_change_pct = float((mean_c - mean_s) / mean_s * 100.0) if mean_s != 0 else 0.0
            
            # Cohen's d
            std_s = np.std(s_arr, ddof=1) if len(s_arr) > 1 else 0.0
            std_c = np.std(c_arr, ddof=1) if len(c_arr) > 1 else 0.0
            pooled_std = np.sqrt((std_s**2 + std_c**2) / 2.0)
            cohen_d = float(mean_diff / pooled_std) if pooled_std > 0 else 0.0
            
            # Paired t-test
            try:
                t_stat, t_p = stats.ttest_rel(s_arr, c_arr)
                t_p = float(t_p) if not np.isnan(t_p) else 1.0
            except Exception:
                t_p = 1.0
                
            # Wilcoxon signed-rank test
            try:
                if np.all(diffs == 0):
                    w_p = 1.0
                else:
                    w_stat, w_p = stats.wilcoxon(diffs)
                    w_p = float(w_p) if not np.isnan(w_p) else 1.0
            except Exception:
                w_p = 1.0
                
            # 95% Confidence Interval for mean difference
            n = len(diffs)
            std_err = stats.sem(diffs) if n > 1 else 0.0
            if std_err > 0 and n > 1:
                ci_margin = stats.t.ppf(0.975, df=n-1) * std_err
                ci_lower = float(mean_diff - ci_margin)
                ci_upper = float(mean_diff + ci_margin)
            else:
                ci_lower = mean_diff
                ci_upper = mean_diff
                
            # Significance tag based on minimum p-value
            p_min = min(t_p, w_p)
            if p_min < 0.01:
                sig = "** (p < 0.01)"
            elif p_min < 0.05:
                sig = "* (p < 0.05)"
            else:
                sig = "ns (not significant)"
                
            scale_stats[metric] = {
                "scalar_mean": mean_s,
                "channel_wise_mean": mean_c,
                "mean_diff": mean_diff,
                "rel_change_pct": rel_change_pct,
                "cohen_d": cohen_d,
                "ttest_p_value": t_p,
                "wilcoxon_p_value": w_p,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "significance": sig
            }
            
        statistical_results[scale] = scale_stats
        
    # Save json artifact
    json_path = os.path.join(run_base_dir, "statistical_analysis.json")
    with open(json_path, "w") as f:
        json.dump(statistical_results, f, indent=4)
        
    # Generate markdown report
    report_md_path = os.path.join(run_base_dir, "statistical_report.md")
    report_lines = [
        "# Statistical Hypothesis Testing Report",
        "## Gating Granularity Study: Scalar (A) vs. Channel-wise (B)\n",
        "Tests conducted: Paired t-test, Wilcoxon Signed-Rank Test, Cohen's d effect size, 95% CIs.\n",
        "| Scale | Metric | Scalar Mean | Channel-wise Mean | Mean Diff | Rel Change (%) | Cohen's d | t-test p | Wilcoxon p | 95% CI | Significance |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for scale, metrics_dict in statistical_results.items():
        for metric, res in metrics_dict.items():
            if res["scalar_mean"] is None:
                continue
            ci_str = f"[{res['ci_95_lower']:.3f}, {res['ci_95_upper']:.3f}]"
            row = (
                f"| {scale} | {metric} | {res['scalar_mean']:.4f} | {res['channel_wise_mean']:.4f} | "
                f"{res['mean_diff']:.4f} | {res['rel_change_pct']:+.2f}% | {res['cohen_d']:.3f} | "
                f"{res['ttest_p_value']:.4f} | {res['wilcoxon_p_value']:.4f} | {ci_str} | {res['significance']} |"
            )
            report_lines.append(row)
            
    with open(report_md_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
        
    print(f"Statistical hypothesis analysis saved to {json_path} and {report_md_path}")
    return statistical_results

