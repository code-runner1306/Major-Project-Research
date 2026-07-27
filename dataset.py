import os
import json
import time
import torch
from torch.utils.data import IterableDataset, DataLoader
import tiktoken
from datasets import load_dataset
import config

def save_or_load_split_manifest(seed, split="train", buffer_size=10000, run_dir=None):
    splits_dir = getattr(config, "SPLITS_DIR", os.path.join("data", "splits"))
    os.makedirs(splits_dir, exist_ok=True)
    manifest_path = os.path.join(splits_dir, f"seed_{seed}_{split}_split.json")
    
    manifest = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            print(f"Loaded existing dataset split manifest for seed {seed} from {manifest_path}")
        except Exception as e:
            print(f"Error reading split manifest {manifest_path}: {e}. Creating new manifest.")
            
    if manifest is None:
        manifest = {
            "dataset_name": "roneneldan/TinyStories",
            "split": split,
            "seed": seed,
            "buffer_size": buffer_size,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
        print(f"Saved dataset split manifest for seed {seed} to {manifest_path}")
        
    if run_dir:
        run_splits_dir = os.path.join(run_dir, "splits")
        os.makedirs(run_splits_dir, exist_ok=True)
        run_manifest_path = os.path.join(run_splits_dir, f"seed_{seed}_{split}_split.json")
        with open(run_manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
        print(f"Saved copy of dataset split manifest to {run_manifest_path}")
        
    return manifest

def load_dataset_with_retry(split, seed=None, run_dir=None, max_retries=10, delay=5):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempting to load TinyStories dataset (split: {split}, seed: {seed}), attempt {attempt}...")
            # Stream the dataset to avoid downloading the entire 1GB file
            dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
            
            if seed is not None and split == "train":
                manifest = save_or_load_split_manifest(seed, split=split, run_dir=run_dir)
                dataset = dataset.shuffle(seed=manifest["seed"], buffer_size=manifest["buffer_size"])
                
            # Try to fetch the first element to ensure network success
            next(iter(dataset))
            print(f"TinyStories dataset (split: {split}, seed: {seed}) loaded successfully.")
            return dataset
        except Exception as e:
            print(f"Error loading dataset on attempt {attempt}: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay = min(delay * 1.5, 60)  # capped exponential backoff
    raise RuntimeError("Failed to load TinyStories dataset from Hugging Face after multiple retries.")

class TinyStoriesDataset(IterableDataset):
    def __init__(self, split="train", seed=None, run_dir=None):
        super().__init__()
        self.split = split
        self.seed = seed
        self.run_dir = run_dir
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eot_token = self.tokenizer.eot_token
        
        # Load dataset with retry logic and optional seed shuffling
        self.dataset = load_dataset_with_retry(split, seed=seed, run_dir=run_dir)

        
    def __iter__(self):
        token_buffer = []
        story_count = 0
        
        for story in self.dataset:
            # Limit the number of stories processed in smoke test to keep it fast
            if config.SMOKE_TEST and config.DATASET_SUBSET_SIZE is not None:
                if story_count >= config.DATASET_SUBSET_SIZE:
                    break
            
            text = story.get("text", "")
            if not text:
                continue
                
            # Tokenize text and append EOT token
            tokens = self.tokenizer.encode(text, allowed_special="all")
            tokens.append(self.eot_token)
            
            token_buffer.extend(tokens)
            story_count += 1
            
            # Yield chunks of CONTEXT_LENGTH + 1
            while len(token_buffer) >= config.CONTEXT_LENGTH + 1:
                chunk = token_buffer[:config.CONTEXT_LENGTH + 1]
                # Slide buffer to start the next chunk exactly where the last one ended (with 1 token overlap for causality)
                token_buffer = token_buffer[config.CONTEXT_LENGTH:]
                
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

def get_dataloader(split="train", batch_size=None, seed=None, run_dir=None):
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    dataset = TinyStoriesDataset(split=split, seed=seed, run_dir=run_dir)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)


