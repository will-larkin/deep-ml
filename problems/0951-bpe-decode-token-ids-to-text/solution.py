def bpe_decode(ids, vocab):
    """
    Args:
        ids: list[int] - token IDs to decode
        vocab: dict[int, str] - mapping from token ID to token string
    Returns:
        str - the decoded text
    """
    if not ids:
        return ''

    tokens = []

    for i in ids:
        token = vocab[i]

        if token.startswith('G'):
            token = ' ' + token[1:]
        
        tokens.append(token)
    
    return "".join(tokens)