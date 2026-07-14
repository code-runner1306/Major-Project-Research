import os
import json
import torch
import torch.optim as optim
import config
from model import GatedDeltaNetLM

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_model(model, dataloader_val, device, max_steps):
    model.eval()
    val_loss_sum = 0.0
    val_steps = 0
    iter_val = iter(dataloader_val)
    
    with torch.no_grad():
        for _ in range(max_steps):
            try:
                x, y = next(iter_val)
            except StopIteration:
                # Re-initialize validation iterator if it runs out
                iter_val = iter(dataloader_val)
                try:
                    x, y = next(iter_val)
                except StopIteration:
                    break
            
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            val_loss_sum += loss.item()
            val_steps += 1
            
            if val_steps >= max_steps:
                break
                
    model.train()
    if val_steps == 0:
        return float('inf'), float('inf')
    avg_loss = val_loss_sum / val_steps
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity

def train_model(scale, variant, seed, dataloader_train, dataloader_val, run_dir):
    print(f"\n--- Starting Run: Scale={scale}, Variant={variant}, Seed={seed} ---")
    set_seed(seed)
    
    # Device Agnosticism: Check for CUDA capabilities
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Fetch model config
    scale_config = config.SCALE_CONFIGS[scale]
    model = GatedDeltaNetLM(scale_config, variant=variant)
    model.to(device)
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {param_count:,} parameters.")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(config.BETA1, config.BETA2)
    )
    
    model.train()
    train_losses = []
    iter_train = iter(dataloader_train)
    
    for step in range(1, config.MAX_STEPS + 1):
        try:
            x, y = next(iter_train)
        except StopIteration:
            # Re-initialize dataset if we exhaust the stream
            iter_train = iter(dataloader_train)
            x, y = next(iter_train)
            
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
        
        loss_val = loss.item()
        train_losses.append(loss_val)
        
        if step % max(1, config.MAX_STEPS // 5) == 0 or step == config.MAX_STEPS:
            print(f"Step {step}/{config.MAX_STEPS} - Loss: {loss_val:.4f}")
            
    # Final validation run
    print("Running validation evaluation...")
    val_loss, val_perplexity = evaluate_model(model, dataloader_val, device, config.VAL_STEPS)
    print(f"Validation Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}")
    
    # Measure memory usage (GPU VRAM or CPU RAM fallback)
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        try:
            import psutil
            process = psutil.Process(os.getpid())
            peak_memory_mb = process.memory_info().rss / (1024 * 1024)
        except:
            peak_memory_mb = 0.0

    # Save seed-specific metrics
    metrics = {
        "scale": scale,
        "variant": variant,
        "seed": seed,
        "parameter_count": param_count,
        "train_losses": train_losses,
        "val_loss": val_loss,
        "val_perplexity": val_perplexity,
        "peak_memory_mb": peak_memory_mb
    }
    
    os.makedirs(run_dir, exist_ok=True)
    metrics_file = os.path.join(run_dir, f"seed{seed}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save config details
    run_config = {
        "batch_size": config.BATCH_SIZE,
        "max_steps": config.MAX_STEPS,
        "context_length": config.CONTEXT_LENGTH,
        "learning_rate": config.LEARNING_RATE,
        "d_model": scale_config["d_model"],
        "n_layer": scale_config["n_layer"],
        "n_head": scale_config["n_head"],
        "d_ff": scale_config["d_ff"]
    }
    config_file = os.path.join(run_dir, f"seed{seed}_config.json")
    with open(config_file, "w") as f:
        json.dump(run_config, f, indent=4)
        
    # Save checkpoint state dict
    checkpoint_file = os.path.join(run_dir, f"model_checkpoint_seed{seed}.pt")
    torch.save(model.state_dict(), checkpoint_file)
    print(f"Run completed. Outputs saved to {metrics_file}")
    
    return model, metrics
