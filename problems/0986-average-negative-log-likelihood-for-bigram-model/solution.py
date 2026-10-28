import numpy as np
from math import log

def bigram_avg_nll(P, words, stoi):
    """Average negative log-likelihood of words under a bigram model."""
    if not words:
        return 0.0

    log_p = 0
    n = 0

    for word in words:
        seq = ['.'] + list(word) + ['.']

        for c1, c2 in zip(seq, seq[1:]):
            log_p += log(P[stoi[c1], stoi[c2]])
            n += 1
    return -(log_p / n)
