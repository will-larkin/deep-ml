def pad_and_truncate(sequences, pad_token_id, max_length=None):
    """
    Pad and/or truncate tokenized sequences to a uniform length.

    Args:
        sequences: list of lists of int token ids
        pad_token_id: int, token id used for padding
        max_length: int or None, target length. If None, use the longest sequence length.

    Returns:
        list of lists of int token ids, each of length max_length.
    """
    if not sequences:
        return sequences

    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    for seq in sequences:
        if len(seq) < max_length:
            seq.extend([pad_token_id] * (max_length - len(seq)))
        else:
            seq[:] = seq[:max_length]


    return sequences
