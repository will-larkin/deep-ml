def train_tokenizer(corpus, vocab_size):
    """
    Learn a character-based BPE tokenizer.

    Args:
        corpus: list[str], containing training names.
        vocab_size: maximum vocabulary size; must be >= 2.

    Returns:
        encode: str -> list[int]
        decode: list[int] -> str
    """
    from collections import Counter, defaultdict

    if not isinstance(vocab_size, int) or vocab_size < 2:
        raise ValueError("vocab_size must be an integer >= 2")

    names = Counter(corpus)

    # IDs 0 and 1 encode fallback bytes as bits. All other IDs represent
    # observed characters or learned multi-character tokens.
    char_counts = Counter()
    for text, weight in names.items():
        for char in text:
            char_counts[char] += weight

    # Reserve room for merges even when the corpus has many characters.
    alphabet_limit = (vocab_size - 1) // 2
    alphabet = sorted(
        char_counts,
        key=lambda char: (-char_counts[char], char),
    )[:alphabet_limit]

    char_ids = {char: i + 2 for i, char in enumerate(alphabet)}
    token_bytes = [None, None] + [
        char.encode("utf-8", "surrogatepass") for char in alphabet
    ]

    def initial_tokens(text):
        result = []
        for char in text:
            token = char_ids.get(char)
            if token is not None:
                result.append(token)
            else:
                for byte in char.encode("utf-8", "surrogatepass"):
                    result.extend(
                        (byte >> shift) & 1
                        for shift in range(7, -1, -1)
                    )
        return result

    sequences = [tuple(initial_tokens(text)) for text in names]
    weights = list(names.values())

    pair_counts = Counter()
    pair_words = defaultdict(set)

    def local_pairs(seq):
        # Keep fallback bits separate from learned text tokens.
        return Counter(
            (a, b)
            for a, b in zip(seq, seq[1:])
            if a >= 2 and b >= 2
        )

    for index, seq in enumerate(sequences):
        for pair, count in local_pairs(seq).items():
            pair_counts[pair] += count * weights[index]
            pair_words[pair].add(index)

    def replace_pair(seq, pair, token):
        result = []
        i = 0
        while i < len(seq):
            if (
                i + 1 < len(seq)
                and (seq[i], seq[i + 1]) == pair
            ):
                result.append(token)
                i += 2
            else:
                result.append(seq[i])
                i += 1
        return tuple(result)

    # Token IDs also give the ordering of learned merge rules.
    merge_ids = {}

    while len(token_bytes) < vocab_size and pair_counts:
        pair = max(
            pair_counts,
            key=lambda p: (pair_counts[p], -p[0], -p[1]),
        )

        # Avoid spending vocabulary on pairs observed only once.
        if pair_counts[pair] < 2:
            break

        token = len(token_bytes)
        token_bytes.append(
            token_bytes[pair[0]] + token_bytes[pair[1]]
        )
        merge_ids[pair] = token

        # Update only names affected by this merge.
        for index in tuple(pair_words[pair]):
            old_seq = sequences[index]
            new_seq = replace_pair(old_seq, pair, token)

            for old_pair, count in local_pairs(old_seq).items():
                pair_counts[old_pair] -= count * weights[index]
                pair_words[old_pair].discard(index)

                if pair_counts[old_pair] == 0:
                    del pair_counts[old_pair]
                    del pair_words[old_pair]

            for new_pair, count in local_pairs(new_seq).items():
                pair_counts[new_pair] += count * weights[index]
                pair_words[new_pair].add(index)

            sequences[index] = new_seq

    def encode(text):
        seq = tuple(initial_tokens(text))

        # Apply the earliest learned merge currently available.
        while len(seq) > 1:
            pair = min(
                zip(seq, seq[1:]),
                key=lambda p: merge_ids.get(p, vocab_size),
            )
            token = merge_ids.get(pair)
            if token is None:
                break

            seq = replace_pair(seq, pair, token)

        return list(seq)

    def decode(ids):
        result = bytearray()
        byte = 0
        bits = 0

        for token in ids:
            if (
                not isinstance(token, int)
                or not 0 <= token < len(token_bytes)
            ):
                raise ValueError("Invalid token ID")

            if token < 2:
                byte = (byte << 1) | token
                bits += 1

                if bits == 8:
                    result.append(byte)
                    byte = 0
                    bits = 0
            else:
                if bits:
                    raise ValueError("Incomplete fallback byte")
                result.extend(token_bytes[token])

        if bits:
            raise ValueError("Incomplete fallback byte")

        return result.decode("utf-8", "surrogatepass")

    return encode, decode