import token

import torch.nn as nn
import torch
from math import sqrt, ceil
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        weight:torch.Tensor = torch.empty(out_features, in_features, dtype=dtype, device=device)
        sigma = sqrt(2./(in_features+out_features))
        self.W = nn.Parameter(nn.init.trunc_normal_(weight, mean=0, std=sigma, a=-3*sigma, b=3*sigma))

    def forward(self, x:torch.Tensor):
        return einsum(x, self.W, "... input, out input -> ... out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        weight: torch.Tensor = torch.empty(num_embeddings, embedding_dim, dtype = dtype, device = device)
        self.W = nn.Parameter(data=nn.init.trunc_normal_(weight, mean = 0, std = 1, a = -3, b = 3))

    def forward(self, tokens_ids: torch.Tensor):
        return self.W[tokens_ids]

class Rmsnorm(nn.Module):
    def __init__(self, d_model:int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g = nn.Parameter(torch.ones(d_model, dtype = dtype, device = device))

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(self.eps + torch.mean(x**2, dim=-1, keepdim=True))
        return (self.g * x / rms).to(in_dtype)

class swiglu(nn.Module):
    def __init__(self, d_model: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_ff = ceil((8 * d_model / 3) / 64) * 64
        self.d_Suggestedmodel = d_model
        self.W1 = Linear(self.d_model, self.d_ff, device = device, dtype = dtype)
        self.W3 = Linear(self.d_model, self.d_ff, device = device, dtype = dtype)
        self.W2 = Linear(self.d_ff, self.d_model, device = device, dtype = dtype)

    def forward(self, x: torch.Tensor):
        w1x = self.W1.forward(x)
        return self.W2.forward(w1x * torch.sigmoid(w1x) * self.W3.forward(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None) -> None:
        super().__init__()
        self.theta:float = theta
        self.d_k:int = d_k
        assert d_k % 2==0
        self.max_seq_len:int = max_seq_len
        angle = torch.tensor([[i/theta**(2*k/d_k) for k in range(d_k//2)] for i in range(max_seq_len)])
        self.register_buffer("sins", torch.sin(angle), persistent = False)
        self.register_buffer("coss", torch.cos(angle), persistent = False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # token_positions: ... seq_len
        sins = self.sins[token_positions]
        coss = self.coss[token_positions]
        x_ = rearrange(x, "... seq_len (a b) -> ... seq_len a b", b = 2)
        x1 = x_[..., 0]
        x2 = x_[..., 1]
        y1 = x1 * coss - x2 * sins
        y2 = x1 * sins + x2 * coss
        return torch.stack((y1, y2), dim = -1).reshape_as(x)

def softmax(x: torch.Tensor, i: int):
    # preform softmax on dimension i
    x = x - torch.amax(x, i, keepdim = True)
    x = torch.exp(x)
    return x / torch.sum(x, i, keepdim = True)
