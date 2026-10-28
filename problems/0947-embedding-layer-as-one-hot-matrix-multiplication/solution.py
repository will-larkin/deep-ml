import numpy as np

def embedding_via_one_hot(token_ids, W):
    """
    Compute token embeddings via one-hot encoding and matrix multiplication.

    Args:
        token_ids: list or 1D array of integer token IDs
        W: numpy array of shape (vocab_size, embed_dim)

    Returns:
        numpy array of shape (len(token_ids), embed_dim)
    """
    token_ids = np.asarray(token_ids)
    vocab_size = W.shape[0]

    one_hot = np.eye(vocab_size)[token_ids]
    embeddings = one_hot @ W
    return embeddings