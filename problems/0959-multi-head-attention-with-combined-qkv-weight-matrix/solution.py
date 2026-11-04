import numpy as np


def mha_combined_qkv(x, W_qkv, W_out, num_heads):
    """
    Multi-head causal self-attention using a single combined QKV weight matrix.


    Args:
        x: array of shape (batch, num_tokens, d_in)
        W_qkv: array of shape (d_in, 3 * d_out)
        W_out: array of shape (d_out, d_out)
        num_heads: int, number of attention heads (d_out % num_heads == 0)


    Returns:
        Nested list of shape (batch, num_tokens, d_out).
    """
    batch, num_tokens, d_in = x.shape
    d_out = W_out.shape[0]
    head_dim = d_out // num_heads

    qkv = x @ W_qkv
    qkv = qkv.reshape(batch, num_tokens, 3, num_heads, head_dim)
    q, k, v = qkv.transpose(2, 0, 3, 1, 4)

    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(head_dim)

    causal_mask = np.triu(np.ones((num_tokens, num_tokens), dtype=bool), k=1)
    scores = np.where(causal_mask, -np.inf, scores)

    scores -= scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / attn.sum(axis=-1, keepdims=True)

    out = attn @ v
    out = out.transpose(0, 2, 1, 3).reshape(batch, num_tokens, d_out)

    return (out @ W_out).tolist()