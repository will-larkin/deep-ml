import numpy as np


def learned_positional_encoding(
    token_embeddings: np.ndarray,
    position_embedding_table: np.ndarray,
    start_pos: int = 0
) -> np.ndarray:
    """
    Apply learned positional embeddings to token embeddings.

    Args:
        token_embeddings: (batch_size, seq_len, d_model) array of token embeddings
        position_embedding_table: (max_seq_len, d_model) learned positional embedding lookup table
        start_pos: Starting position index (default is 0)

    Returns:
        Array of shape (batch_size, seq_len, d_model) with positional information applied
    """
    batch_size, seq_len, d_model = token_embeddings.shape
    max_seq_len, table_d_model = position_embedding_table.shape


    positions = np.arange(start_pos, start_pos + seq_len)
    positional_embeddings = position_embedding_table[positions]  # (seq_len, d_model)

    return token_embeddings + positional_embeddings[None, :, :]