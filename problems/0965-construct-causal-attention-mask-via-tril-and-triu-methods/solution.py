import numpy as np

def causal_mask_attention(attn_weights: np.ndarray, method: str = 'tril') -> list:
    """Apply causal masking two ways and return the resulting attention matrix as a nested list."""
    seq_len = attn_weights.shape[-1]
    if method == 'tril':
        allowed = np.tril(np.ones((seq_len, seq_len)), k=0)

        # Zero out future positions.
        masked = attn_weights * allowed

        # Renormalize each surviving row.
        masked = masked / masked.sum(axis=-1, keepdims=True)

        result = masked

    else:
        attn_scores = np.log(attn_weights)

        future = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

        masked = np.where(future, -np.inf, attn_scores)
        masked = softmax(masked)

    return masked.tolist()

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)