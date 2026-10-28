import numpy as np

def bigram_counts(words):
    vocab = ['.'] + sorted(set(''.join(words)) - {'.'})
    stoi = {char: i for i, char in enumerate(vocab)}

    v = len(vocab)
    matrix = np.zeros((v, v), dtype=int)

    for word in words:
        chars = ['.'] + list(word) + ['.']

        for first, second in zip(chars, chars[1:]):
            matrix[stoi[first], stoi[second]] += 1

    return matrix.tolist()