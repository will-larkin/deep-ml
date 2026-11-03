import numpy as np

def attention_dropout(attn_weights, values, dropout_rate, mask):
    """Apply dropout mask to attention weights and compute context vectors."""
    if dropout_rate == 0.0:
        return attn_weights

    mask = np.asarray(mask)
    attn_weights = np.asarray(attn_weights, dtype=float)

    scale = 1 / (1 - dropout_rate)
    masked_weights = np.where(mask != 0, attn_weights * scale, 0)
    context_vectors = masked_weights @ values

    return context_vectors.tolist()
