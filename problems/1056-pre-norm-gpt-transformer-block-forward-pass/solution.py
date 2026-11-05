import numpy as np

def pre_norm_transformer_block(x, params, num_heads):
    """
    Pre-norm Transformer block forward pass.

    Args:
        x: numpy array of shape (batch, seq_len, emb_dim)
        params: dict with keys 'ln1_gamma','ln1_beta','ln2_gamma','ln2_beta',
                'W_q','W_k','W_v','W_o','W_ff1','b_ff1','W_ff2','b_ff2'
        num_heads: int, number of attention heads (emb_dim must be divisible by num_heads)

    Returns:
        numpy array of shape (batch, seq_len, emb_dim)
    """
    normalised = layer_norm(x, params['ln1_gamma'], params['ln1_beta'])

    context_vectors = multi_head_attention(normalised, num_heads, params)
    x = x + context_vectors

    normalised_2 = layer_norm(x, params['ln2_gamma'], params['ln2_beta'])

    trans1 = normalised_2 @ params['W_ff1'] + params['b_ff1']
    activated = gelu(trans1)
    trans2 = activated @ params['W_ff2'] + params['b_ff2']

    return x + trans2

def gelu(x):
    scaling_constant = np.sqrt(2 / np.pi)
    polynomial_arg = x + 0.044715 * x**3

    return 0.5 * x * (1 + np.tanh(scaling_constant * polynomial_arg))

def layer_norm(x, gamma, beta):
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    eps = 10**-5
    x_hat = (x - mean) / np.sqrt(variance + eps)

    return gamma * x_hat + beta

def multi_head_attention(X, num_heads, weights):
    Q, K, V = calculate_qkv(X, weights)

    Qh = split_heads(Q, num_heads)
    Kh = split_heads(K, num_heads)
    Vh = split_heads(V, num_heads)
    
    scores = Qh @ Kh.swapaxes(-1, -2) / np.sqrt(Qh.shape[-1])

    scores_seq_len = scores.shape[-1]
    future = np.triu(np.ones((scores_seq_len, scores_seq_len), dtype=bool), k=1)
    masked_scores = np.where(future, -np.inf, scores)

    attention = softmax(masked_scores, axis=-1)
    attended = attention @ Vh
    batch_size, num_heads, seq_len, d_head = attended.shape

    concatenated = attended.transpose(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_heads * d_head
    )

    return concatenated @ weights["W_o"]

def calculate_qkv(X, weights):
    W_q = weights['W_q']
    W_k = weights['W_k']
    W_v = weights['W_v']

    return X @ W_q, X @ W_k, X @ W_v

def split_heads(x, n_heads):
    batch_size, seq_len, d_model = x.shape
    d_head = d_model // n_heads

    # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_head)
    x = x.reshape(batch_size, seq_len, n_heads, d_head)

    # -> (batch_size, n_heads, seq_len, d_head)
    return x.transpose(0, 2, 1, 3)

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

