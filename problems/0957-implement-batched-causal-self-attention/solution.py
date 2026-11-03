import numpy as np

def batched_causal_self_attention(X, W_query, W_key, W_value):
    """
    Batched causal self-attention.

    Args:
        X: nested list of shape (B, T, d_in)
        W_query, W_key, W_value: nested lists of shape (d_in, d_out)

    Returns:
        Nested list of shape (B, T, d_out) -- the context vectors.
    """
    X = np.asarray(X)
    W_query = np.asarray(W_query)
    W_key = np.asarray(W_key)
    W_value = np.asarray(W_value)

    Q = X @ W_query
    K = X @ W_key
    V = X @ W_value

    attention_scores = Q @ np.transpose(K, (0, 2, 1))
    d_out = Q.shape[2]
    scaled_scores = attention_scores / np.sqrt(d_out)
    
    masked_scores = np.where(
        np.triu(np.ones_like(scaled_scores, dtype=bool), k=1),
        -np.inf,
        scaled_scores,
    )

    attention_weights = softmax(masked_scores)
    context_vectors = attention_weights @ V

    return context_vectors.tolist()

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
