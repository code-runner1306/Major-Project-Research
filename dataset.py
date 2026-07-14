import time
import torch
from torch.utils.data import IterableDataset, DataLoader
import tiktoken
from datasets import load_dataset
import config

def load_dataset_with_retry(split, max_retries=10, delay=5):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempting to load TinyStories dataset (split: {split}), attempt {attempt}...")
            # Stream the dataset to avoid downloading the entire 1GB file
            dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
            # Try to fetch the first element to ensure network success
            next(iter(dataset))
            print(f"TinyStories dataset (split: {split}) loaded successfully.")
            return dataset
        except Exception as e:
            print(f"Error loading dataset on attempt {attempt}: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay = min(delay * 1.5, 60)  # capped exponential backoff
    raise RuntimeError("Failed to load TinyStories dataset from Hugging Face after multiple retries.")

class TinyStoriesDataset(IterableDataset):
    def __init__(self, split="train"):
        super().__init__()
        self.split = split
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eot_token = self.tokenizer.eot_token
        
        # Load dataset with retry logic
        self.dataset = load_dataset_with_retry(split)
        
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

def get_dataloader(split="train", batch_size=None):
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    dataset = TinyStoriesDataset(split=split)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)
