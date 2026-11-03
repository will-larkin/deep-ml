import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    
    return X @ W_q, X @ W_k, X @ W_v

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    attention_scores = Q @ K.T
    d_k = Q.shape[1]
    scaled_scores = attention_scores / np.sqrt(d_k)
    attention_weights = softmax(scaled_scores)

    return attention_weights @ V

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    Qh = split_heads(Q, n_heads)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    
    head_outputs = []

    for h in range(n_heads):
        output_h = self_attention(
            Qh[h], Kh[h], Vh[h]
        )
        head_outputs.append(output_h)

    return np.concatenate(head_outputs, axis=-1)

def split_heads(x, n_heads):
    seq_len, d_model = x.shape
    d_head = d_model // n_heads

    # (seq_len, d_model) -> (seq_len, n_heads, d_head)
    x = x.reshape(seq_len, n_heads, d_head)

    # -> (n_heads, seq_len, d_head)
    return x.transpose(1, 0, 2)

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)