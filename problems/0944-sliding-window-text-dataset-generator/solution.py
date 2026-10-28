def sliding_window_dataset(token_ids: list[int], max_length: int, stride: int) -> list:
    """
    Generate (input, target) training pairs using a sliding window.

    Args:
        token_ids: List of integer token IDs
        max_length: Length of each input/target chunk (context window size)
        stride: Step size between consecutive windows

    Returns:
        List of (input_chunk, target_chunk) tuples, each chunk as a list of ints.
    """
    output = []
    i = 0
    while i + max_length < len(token_ids):
        input_chunk = token_ids[i:i + max_length]
        target_chunk = token_ids[i+1:i + max_length + 1]
        output.append((input_chunk, target_chunk))
        i += stride

    return output