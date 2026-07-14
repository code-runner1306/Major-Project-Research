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
