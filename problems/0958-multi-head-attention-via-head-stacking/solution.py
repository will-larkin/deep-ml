import numpy as np

def multi_head_attention_stack(x, heads):
    """
    Multi-head causal self-attention by stacking single-head attention modules.

    Args:
        x: array-like of shape (B, T, d_in)
        heads: list of H tuples (W_q, W_k, W_v); each weight is shape (d_in, d_head)

    Returns:
        numpy.ndarray of shape (B, T, H * d_head)
    """
    # Your code here
    context_vectors = []
    x = np.asarray(x)

    for W_q, W_k, W_v in heads:
        Q, K, V = compute_qkv(x, W_q, W_k, W_v)
        context_vectors.append(self_attention(Q, K, V))
    
    return np.concatenate(context_vectors, axis=-1)

def self_attention(Q, K, V):
    B, T, d_k = Q.shape
    mask = np.tril(np.ones((T, T), dtype=bool))
    attention_scores = Q @ K.transpose(0, 2, 1)
    scaled_scores = attention_scores / np.sqrt(d_k)
    masked_scores = np.where(mask, scaled_scores, -np.inf)
    attention_weights = softmax(masked_scores)

    return attention_weights @ V

def compute_qkv(X, W_q, W_k, W_v):
    W_q = np.asarray(W_q)
    W_k = np.asarray(W_k)
    W_v = np.asarray(W_v)

    
    return X @ W_q, X @ W_k, X @ W_v

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
