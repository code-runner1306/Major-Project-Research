import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class CausalConv1d(nn.Module):
    def __init__(self, channels, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=True
        )
        
    def forward(self, x):
        # x shape: (B, T, C)
        # Pad left by kernel_size - 1 to preserve causality
        x_padded = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        out = self.conv(x_padded)
        return out.transpose(1, 2)

def recurrent_forward(q, k, v, alpha, beta, g=None):
    # q, k, v shape: (B, T, H, D)
    # alpha shape: (B, T, H, G_dim) where G_dim is 1 (scalar) or D (channel-wise)
    # beta shape: (B, T, H, 1)
    # g shape: (B, T, H, D) or None
    B, T, H, D = q.shape
    
    # Initialize the memory state matrix: (B, H, D, D) on the correct device/dtype
    S = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    outputs = []
    
    # Pre-normalize queries and keys using L2 normalization
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    
    # Step through time sequentially
    for t in range(T):
        q_t = q[:, t].unsqueeze(-1)       # (B, H, D, 1)
        k_t = k[:, t].unsqueeze(-1)       # (B, H, D, 1)
        v_t = v[:, t].unsqueeze(-1)       # (B, H, D, 1)
        
        alpha_t = alpha[:, t].unsqueeze(-1)  # (B, H, G_dim, 1)
        beta_t = beta[:, t].unsqueeze(-1)    # (B, H, 1, 1)
        
        # S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1} @ k_t) @ k_t^T
        decayed_S = alpha_t * S  # Broadcasting along rows of S if G_dim = D
        error = v_t - torch.matmul(S, k_t)
        update = beta_t * torch.matmul(error, k_t.transpose(-1, -2))
        S = decayed_S + update
        
        # Retrieve: o_t = S_t @ q_t -> (B, H, D, 1)
        o_t = torch.matmul(S, q_t)
        outputs.append(o_t.squeeze(-1))   # (B, H, D)
        
    output = torch.stack(outputs, dim=1)  # (B, T, H, D)
    
    if g is not None:
        output = output * g
        
    return output, S

class GatedDeltaNetAttention(nn.Module):
    def __init__(self, d_model, n_head, d_head, variant="scalar"):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.variant = variant
        
        # Q, K, V Projections
        self.q_proj = nn.Linear(d_model, n_head * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_head * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_head * d_head, bias=False)
        
        # Causal convolutions to capture local structure
        self.q_conv = CausalConv1d(n_head * d_head, kernel_size=4)
        self.k_conv = CausalConv1d(n_head * d_head, kernel_size=4)
        self.v_conv = CausalConv1d(n_head * d_head, kernel_size=4)
        
        # Learning rate (beta) projection - scalar per head
        self.beta_proj = nn.Linear(d_model, n_head, bias=False)
        
        # Decay (alpha) projection
        if self.variant == "scalar":
            self.alpha_proj = nn.Linear(d_model, n_head, bias=False)
        elif self.variant == "channel-wise":
            self.alpha_proj = nn.Linear(d_model, n_head * d_head, bias=False)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")
            
        # Output gate projection
        self.gate_proj = nn.Linear(d_model, n_head * d_head, bias=False)
        self.out_proj = nn.Linear(n_head * d_head, d_model, bias=False)
        
    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_head, self.d_head
        
        # Project and apply causal convs
        q = self.q_conv(self.q_proj(x))
        k = self.k_conv(self.k_proj(x))
        v = self.v_conv(self.v_proj(x))
        
        # Reshape to (B, T, H, D)
        q = q.view(B, T, H, D)
        k = k.view(B, T, H, D)
        v = v.view(B, T, H, D)
        
        # Project gates and apply activations
        # beta is a positive update gate in [0, 1]
        beta = torch.sigmoid(self.beta_proj(x)).view(B, T, H, 1)
        
        # alpha is a decay gate in [0, 1]
        if self.variant == "scalar":
            alpha = torch.sigmoid(self.alpha_proj(x)).view(B, T, H, 1)
        else:
            alpha = torch.sigmoid(self.alpha_proj(x)).view(B, T, H, D)
            
        # Output gate (SiLU gating on recurrent states)
        g = F.silu(self.gate_proj(x)).view(B, T, H, D)
        
        # Recurrent state computation
        out, _ = recurrent_forward(q, k, v, alpha, beta, g)
        
        # Reshape and project out
        out = out.reshape(B, T, H * D)
        return self.out_proj(out)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, d_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        
        self.q_proj = nn.Linear(d_model, n_head * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_head * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_head * d_head, bias=False)
        self.out_proj = nn.Linear(n_head * d_head, d_model, bias=False)
        
    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_head, self.d_head
        
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)
        
        # PyTorch built-in flash/scaled dot product attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.out_proj(out)

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class GatedDeltaNetBlock(nn.Module):
    def __init__(self, layer_idx, d_model, n_head, d_head, d_ff, variant="scalar"):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(d_model)
        
        # 3:1 ratio: every 4th layer (e.g. index 3, 7, 11...) is causal self-attention
        if layer_idx % 4 == 3:
            self.attn = CausalSelfAttention(d_model, n_head, d_head)
        else:
            self.attn = GatedDeltaNetAttention(d_model, n_head, d_head, variant=variant)
            
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, d_ff)
        
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

class GatedDeltaNetLM(nn.Module):
    def __init__(self, config_dict, variant="scalar"):
        super().__init__()
        self.vocab_size = config.VOCAB_SIZE
        self.d_model = config_dict["d_model"]
        
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.d_model)
        
        self.blocks = nn.ModuleList([
            GatedDeltaNetBlock(
                layer_idx=i,
                d_model=self.d_model,
                n_head=config_dict["n_head"],
                d_head=self.d_model // config_dict["n_head"],
                d_ff=config_dict["d_ff"],
                variant=variant
            )
            for i in range(config_dict["n_layer"])
        ])
        
        self.norm = RMSNorm(self.d_model)
        self.output = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        # Tied embeddings
        self.output.weight = self.tok_embeddings.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, tokens, targets=None):
        h = self.tok_embeddings(tokens)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        
        if targets is not None:
            logits = self.output(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            logits = self.output(h)
            return logits, None
