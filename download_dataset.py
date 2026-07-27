import os
import json
import time
from datasets import load_dataset

def download_and_verify_dataset():
    cache_dir = os.path.join("data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    print("==================================================")
    print(" PRE-DOWNLOADING TINYSTORIES DATASET TO LOCAL CACHE ")
    print(f" Target Directory: {os.path.abspath(cache_dir)}")
    print("==================================================")
    
    start_time = time.time()
    
    # Download train split
    print("\n1. Downloading/Caching 'train' split...")
    ds_train = load_dataset("roneneldan/TinyStories", split="train", cache_dir=cache_dir)
    train_count = len(ds_train)
    print(f"   -> Train split ready: {train_count:,} stories.")
    
    # Download validation split
    print("\n2. Downloading/Caching 'validation' split...")
    ds_val = load_dataset("roneneldan/TinyStories", split="validation", cache_dir=cache_dir)
    val_count = len(ds_val)
    print(f"   -> Validation split ready: {val_count:,} stories.")
    
    elapsed = time.time() - start_time
    
    # Write dataset summary manifest
    summary = {
        "dataset_name": "roneneldan/TinyStories",
        "cache_directory": cache_dir,
        "train_stories_count": train_count,
        "validation_stories_count": val_count,
        "total_stories_count": train_count + val_count,
        "download_duration_seconds": round(elapsed, 2),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    summary_path = os.path.join(cache_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\n==================================================")
    print(f" [SUCCESS] DATASET PRE-DOWNLOAD COMPLETE in {elapsed:.2f} seconds!")
    print(f" Total Stories Cached: {train_count + val_count:,}")
    print(f" Summary Saved: {summary_path}")
    print("==================================================\n")
    return summary

if __name__ == "__main__":
    download_and_verify_dataset()
