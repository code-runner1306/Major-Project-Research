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
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
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
    
    # Cosine learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        total_steps=config.MAX_STEPS,
        pct_start=0.05,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=10.0
    )
    
    # Modern PyTorch AMP GradScaler (PyTorch 2.x+)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    grad_accum_steps = getattr(config, "GRAD_ACCUM_STEPS", 1)
    
    # Checkpointing and Early Stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_file = os.path.join(run_dir, f"model_checkpoint_seed{seed}.pt")
    
    model.train()
    train_losses = []
    iter_train = iter(dataloader_train)
    
    for step in range(1, config.MAX_STEPS + 1):
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for _ in range(grad_accum_steps):
            try:
                x, y = next(iter_train)
            except StopIteration:
                # Re-initialize dataset if we exhaust the stream
                iter_train = iter(dataloader_train)
                x, y = next(iter_train)
                
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                _, loss = model(x, y)
                loss = loss / grad_accum_steps
                
            scaler.scale(loss).backward()
            accum_loss += loss.item() * grad_accum_steps
            
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        train_losses.append(accum_loss)
        
        # Periodic output log
        if step % max(1, config.MAX_STEPS // 5) == 0 or step == config.MAX_STEPS:
            print(f"Step {step}/{config.MAX_STEPS} - Loss: {accum_loss:.4f}")
            
        # Periodic Validation check
        if step % config.EVAL_INTERVAL == 0:
            val_loss, val_perplexity = evaluate_model(model, dataloader_val, device, config.VAL_STEPS)
            print(f"Step {step}/{config.MAX_STEPS} - Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}")
            
            # Check for best validation score
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                os.makedirs(run_dir, exist_ok=True)
                torch.save(model.state_dict(), checkpoint_file)
                print(f"  [New Best Model] Saved checkpoint with Val Loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  [No Improvement] Patience counter: {patience_counter}/{config.PATIENCE}")
                if patience_counter >= config.PATIENCE:
                    print(f"  [Early Stopping] Triggered at step {step}")
                    break
                    
    # Reload best model before final evaluation
    if os.path.exists(checkpoint_file):
        print(f"Reloading best checkpoint from {checkpoint_file}...")
        model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    else:
        # Save current state as fallback if training loop ended before any eval check
        os.makedirs(run_dir, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_file)
        
    # Final evaluation run on best model
    print("Running final validation evaluation...")
    val_loss, val_perplexity = evaluate_model(model, dataloader_val, device, config.VAL_STEPS)
    print(f"Final Validation Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.4f}")
    
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
        
    print(f"Run completed. Outputs saved to {metrics_file}")
    
    return model, metrics
