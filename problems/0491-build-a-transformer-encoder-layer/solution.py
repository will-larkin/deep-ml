import numpy as np

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    context_vectors = multi_head_attention(X, num_heads, weights)

    X = X + context_vectors

    normalised = layer_norm(X, weights['gamma1'], weights['beta1'], eps)

    trans1 = normalised @ weights['W1'] + weights['b1']
    activated = relu(trans1)
    trans2 = activated @ weights['W2'] + weights['b2']

    residual = normalised + trans2
    output = layer_norm(residual, weights['gamma2'], weights['beta2'], eps)

    return output

def layer_norm(X, gamma, beta, eps):
    mean = X.mean(axis=-1, keepdims=True)
    variance = X.var(axis=-1, keepdims=True)
    X_hat = (X - mean) / np.sqrt(variance + eps)

    return gamma * X_hat + beta
    

def multi_head_attention(X, num_heads, weights):
    Q, K, V = calculate_qkv(X, weights)

    Qh = split_heads(Q, num_heads)
    Kh = split_heads(K, num_heads)
    Vh = split_heads(V, num_heads)
    
    scores = Qh @ Kh.swapaxes(-1, -2) / np.sqrt(Qh.shape[-1])

    attention = softmax(scores, axis=-1)
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

def relu(x):
    return np.maximum(0, x)