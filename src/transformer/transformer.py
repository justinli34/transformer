import torch
import torch.nn.functional as F
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_QKV = nn.Parameter(torch.empty(d_model, d_model * 3))
        self.W_O = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.normal_(self.W_QKV, mean=0.0, std=d_model**-0.5)
        nn.init.normal_(self.W_O, mean=0.0, std=d_model**-0.5)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """(batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)"""
        batch_size, seq_len, d = X.shape
        Q, K, V = torch.split(X @ self.W_QKV, d, dim=-1)

        # each is (batch_size, n_heads, seq_len, d_k)
        Q, K, V = (t.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2) for t in (Q, K, V))

        scores = Q @ K.transpose(-1, -2) / self.d_k**0.5
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=X.device).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ V
        return out.transpose(1, 2).reshape(batch_size, seq_len, d) @ self.W_O


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W_up = nn.Parameter(torch.empty(d_model, d_ff))
        self.W_down = nn.Parameter(torch.empty(d_ff, d_model))
        nn.init.normal_(self.W_up, mean=0.0, std=d_model**-0.5)
        nn.init.normal_(self.W_down, mean=0.0, std=d_ff**-0.5)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """(batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)"""
        return F.gelu(X @ self.W_up) @ self.W_down


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """(batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)"""
        rms = X.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.gamma * X / rms


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attention = AttentionBlock(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)
        self.norm1, self.norm2 = RMSNorm(d_model), RMSNorm(d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """(batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)"""
        X = X + self.attention(self.norm1(X))
        return X + self.mlp(self.norm2(X))


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        vocab_size: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList(TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers))
        self.final_norm = RMSNorm(d_model)
        self.W_unembed = nn.Parameter(torch.empty(d_model, vocab_size))
        nn.init.normal_(self.W_unembed, mean=0.0, std=d_model**-0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(batch_size, seq_len) token ids -> (batch_size, seq_len, vocab_size) logits"""
        positions = torch.arange(tokens.size(1), device=tokens.device)
        X = self.token_embeddings(tokens) + self.pos_embeddings(positions)
        for layer in self.layers:
            X = layer(X)
        return self.final_norm(X) @ self.W_unembed
