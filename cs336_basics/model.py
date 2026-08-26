import torch.nn as nn
import torch
from math import sqrt, ceil
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        weight:torch.Tensor = torch.empty(out_features, in_features, dtype=dtype, device=device)
        sigma = sqrt(2./(in_features+out_features))
        self.weight = nn.Parameter(nn.init.trunc_normal_(weight, mean=0, std=sigma, a=-3*sigma, b=3*sigma))

    def forward(self, x:torch.Tensor):
        return einsum(x, self.weight, "... input, out input -> ... out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        weight: torch.Tensor = torch.empty(num_embeddings, embedding_dim, dtype = dtype, device = device)
        self.weight = nn.Parameter(data=nn.init.trunc_normal_(weight, mean = 0, std = 1, a = -3, b = 3))

    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]

class Rmsnorm(nn.Module):
    def __init__(self, d_model:int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model, dtype = dtype, device = device))

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(self.eps + torch.mean(x**2, dim=-1, keepdim=True))
        return (self.weight * x / rms).to(in_dtype)

class positionwise_feedforward(nn.Module):
    def __init__(self, d_model: int, d_ff:int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if d_ff is not None:
            self.d_ff = d_ff
        else:
            self.d_ff = ceil((8 * d_model / 3) / 64) * 64
        self.d_model = d_model
        self.w1 = Linear(self.d_model, self.d_ff, device = device, dtype = dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device = device, dtype = dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device = device, dtype = dtype)

    def forward(self, x: torch.Tensor):
        w1x = self.w1(x)
        return self.w2(w1x * torch.sigmoid(w1x) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.theta:float = theta
        self.d_k:int = d_k
        if not d_k % 2==0:
            raise ValueError("Dimension of query and key vectors should be even.")
        self.max_seq_len:int = max_seq_len
        angle = torch.tensor([[i/theta**(2*k/d_k) for k in range(d_k//2)] for i in range(max_seq_len)], dtype = dtype,device = device)
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

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None):
    """
        Given key (K), query (Q), and value (V) tensors, return
        the output of your scaled dot product attention implementation.

        Args:
            query (Float[Tensor, " ... queries d_k"]): Query tensor
            key (Float[Tensor, " ... keys d_k"]): Key tensor
            value (Float[Tensor, " ... keys d_v"]): Values tensor
            mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
        Returns:
            Float[Tensor, " ... queries d_v"]: Output of SDPA
        """
    kq = einsum(key, query, "... seq1 d, ... seq2 d -> ... seq2 seq1")/sqrt(key.shape[-1])
    if mask is not None:
        if mask.any(dim = -2).all():
            kq += torch.where(mask, 0, -torch.inf)
        else:
            raise ValueError("Require at least one True per column.")
    return einsum(softmax(kq, -1), value, "... seq1 seq2, ... seq2 d -> ... seq1 d")

class multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        # d_k = d_v = d_model / h
        if num_heads <= 0:
            raise ValueError("Number of heads must be positive.")
        if not d_model % num_heads == 0:
            raise ValueError("Dimension of transformer block inputs must be divisible by number of heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj: Linear = Linear(d_model, d_model, device, dtype)
        self.k_proj: Linear = Linear(d_model, d_model, device, dtype)
        self.v_proj: Linear = Linear(d_model, d_model, device, dtype)
        self.output_proj: Linear = Linear(d_model, d_model, device, dtype)
    def forward(self, x: torch.Tensor)->torch.Tensor:
        q: torch.Tensor = self.q_proj(x)
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)
        q = rearrange(q, "... seq (h d) -> ... h seq d", h = self.num_heads)
        k = rearrange(k, "... seq (h d) -> ... h seq d", h = self.num_heads)
        v = rearrange(v, "... seq (h d) -> ... h seq d", h = self.num_heads)
        mask = torch.tril(torch.ones(q.shape[-2], q.shape[-2], dtype = torch.bool, device=x.device), diagonal=0)
        return self.output_proj(rearrange(scaled_dot_product_attention(q, k, v, mask), "... h seq d -> ... seq (h d)"))


class multihead_self_attention_with_rope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        # d_k = d_v = d_model / h
        if num_heads <= 0:
            raise ValueError("Number of heads must be positive.")
        if not d_model % num_heads == 0:
            raise ValueError("Dimension of transformer block inputs must be divisible by number of heads.")
        self.d_model: int = d_model
        self.num_heads: int= num_heads
        self.q_proj: Linear = Linear(d_model, d_model, device, dtype)
        self.k_proj: Linear = Linear(d_model, d_model, device, dtype)
        self.v_proj: Linear = Linear(d_model, d_model, device, dtype)
        self.output_proj: Linear = Linear(d_model, d_model, device, dtype)
        self.rope: RotaryPositionalEmbedding = RotaryPositionalEmbedding(theta, d_model//num_heads, max_seq_len, device, dtype)
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len, dtype = torch.bool, device=device), diagonal=0), persistent=False)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None)->torch.Tensor:
        if x.shape[-2] > self.mask.shape[0]:
            raise ValueError("Sequence length longer than the context length.")
        q: torch.Tensor = self.q_proj(x)
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)
        q = rearrange(q, "... seq (h d) -> ... h seq d", h = self.num_heads)
        k = rearrange(k, "... seq (h d) -> ... h seq d", h = self.num_heads)
        if token_positions is None:
            q = self.rope(q, torch.arange(q.shape[-2], device = q.device))
            k = self.rope(k, torch.arange(k.shape[-2], device = q.device))
        else:
            token_positions = token_positions.unsqueeze(-2)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        v = rearrange(v, "... seq (h d) -> ... h seq d", h = self.num_heads)

        mask = self.mask[0:q.shape[-2]][..., 0:q.shape[-2]]
        return self.output_proj(rearrange(scaled_dot_product_attention(q, k, v, mask), "... h seq d -> ... seq (h d)"))


class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len:int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.ln1: Rmsnorm = Rmsnorm(d_model, device=device, dtype=dtype)
        self.attn: multihead_self_attention_with_rope = multihead_self_attention_with_rope(d_model, num_heads, theta, max_seq_len, device, dtype)
        self.ln2: Rmsnorm = Rmsnorm(d_model, device=device, dtype=dtype)
        self.ffn: positionwise_feedforward = positionwise_feedforward(d_model, d_ff, device, dtype)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        y = x + self.attn((self.ln1(x)), token_positions)
        return y + self.ffn(self.ln2(y))

class transformer_lm(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.Sequential(*[transformer_block(d_model, num_heads, d_ff, theta, context_length, device, dtype) for _ in range(num_layers)])
        self.ln_final = Rmsnorm(d_model, device = device, dtype = dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, x: torch.Tensor):
        return self.lm_head(self.ln_final(self.layers(self.token_embeddings(x))))
