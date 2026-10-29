import numpy as np

def sample_mlp_lm(C, W1, b1, W2, b2, block_size, itos, seed, max_tokens):
    # Convert nested Python lists into NumPy arrays.
    C = np.asarray(C, dtype=float)
    W1 = np.asarray(W1, dtype=float)
    b1 = np.asarray(b1, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    b2 = np.asarray(b2, dtype=float)

    vocab_size = C.shape[0]
    rng = np.random.default_rng(seed)

    # Start with block_size end-token indices.
    context = [0] * block_size
    output = []

    for _ in range(max_tokens):
        # Look up one embedding per context character.
        emb = C[context]                 # shape: (block_size, emb_dim)

        # Concatenate all context embeddings into one vector.
        e = emb.reshape(-1)              # shape: (block_size * emb_dim,)

        # Hidden layer.
        h = np.tanh(e @ W1 + b1)         # shape: (hidden,)

        # Scores for every possible next character.
        logits = h @ W2 + b2             # shape: (vocab_size,)

        # Numerically stable softmax.
        shifted_logits = logits - np.max(logits)
        exp_logits = np.exp(shifted_logits)
        probs = exp_logits / np.sum(exp_logits)

        # Draw a next-character index using the predicted probabilities.
        ix = rng.choice(vocab_size, p=probs)

        # Index 0 represents the end token ".".
        if ix == 0:
            break

        output.append(itos[ix])

        # Keep only the most recent block_size character indices.
        context = context[1:] + [ix]

    return "".join(output)