def bpe_encode(text, token_to_id, merges):
    # Your code here
    if not text:
        return []

    token_ids = [token_to_id[char] for char in text]
    changed = True

    while changed:
        changed = False
        i = 0

        while i < len(token_ids) - 1:
            pair = (token_ids[i], token_ids[i + 1])

            if pair in merges:
                token_ids[i] = merges[pair]
                del token_ids[i + 1]
                changed = True

                # Stay at i: token_ids[i] may now merge with the next token
            else:
                i += 1

    return token_ids