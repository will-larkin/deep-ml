import numpy as np

def mlp_forward_loss(X, Y, C, W1, b1, W2, b2):
    """
    Forward pass of an MLP character-level language model.
    Returns mean cross-entropy loss as a float.
    """
    N = X.shape[0]

    # 1. Embedding lookup: (N, block_size, emb_dim)
    emb = C[X]

    # 2. Flatten each example's context embeddings: (N, block_size * emb_dim)
    flat = emb.reshape(N, -1)

    # 3. Hidden layer: (N, hidden)
    h = np.tanh(flat @ W1 + b1)

    # 4. Output logits: (N, vocab_size)
    logits = h @ W2 + b2

    # 5. Numerically stable softmax
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # 6. Mean cross-entropy loss
    loss = -np.log(probs[np.arange(N), Y]).mean()

    return loss
