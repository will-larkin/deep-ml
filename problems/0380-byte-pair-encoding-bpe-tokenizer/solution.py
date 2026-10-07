def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.
    
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """

    merges = []
    merged_corpus = corpus.copy()

    for _ in range(num_merges):
        most_frequent = count_freqs(merged_corpus)
        merges.append(most_frequent[0])
        merged_corpus = merge(most_frequent[0], merged_corpus)

    return merges

def merge(pair, corpus) -> list:
    new_corpus = {}

    for seq, freq in corpus.items():
        tokens = seq.split(" ")
        i = 0

        while i < len(tokens) - 1:
            if (tokens[i], tokens[i+1]) == pair:
                tokens[i] = tokens[i] + tokens[i+1]
                del tokens[i+1]
            else:
                i += 1

        new_sequence = " ".join(tokens)
        new_corpus[new_sequence] = freq

    return new_corpus

def count_freqs(corpus):
    token_freqs = {}
    most_frequent = (None, 0)

    for seq, freq in corpus.items():
        tokens = seq.split(" ")
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])

            if pair in token_freqs:
                token_freqs[pair] += freq
            else:
                token_freqs[pair] = freq

            if token_freqs[pair] > most_frequent[1]:
                most_frequent = (pair, token_freqs[pair])

    return most_frequent



