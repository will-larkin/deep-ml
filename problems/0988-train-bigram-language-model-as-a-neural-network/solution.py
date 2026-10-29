import numpy as np

def train_bigram_nn(xs, ys, vocab_size, lr=10.0, num_iters=100, alpha=0.0):
    """
    Train a one-layer neural-net bigram language model with full-batch gradient descent.

    Args:
        xs: list[int] of length N - input character indices in [0, vocab_size)
        ys: list[int] of length N - target next-character indices in [0, vocab_size)
        vocab_size: int V - number of distinct characters
        lr: float - learning rate
        num_iters: int - number of gradient descent steps
        alpha: float - L2 regularization strength on mean(W**2)

    Returns:
        list[list[float]] - trained weight matrix W of shape (V, V)
    """
    weights = np.zeros((vocab_size, vocab_size))
    one_hot_x = calculate_one_hot(xs, vocab_size)
    one_hot_y = calculate_one_hot(ys, vocab_size)

    for i in range(num_iters):
        logits = one_hot_x @ weights
        probs = softmax(logits)
        data_loss = -np.mean(np.log(probs[np.arange(len(xs)), ys]))
        grad_W = calculate_gradient(probs, one_hot_x, one_hot_y, alpha, vocab_size, weights)
        weights = weights - lr * grad_W

    return weights.tolist()

def calculate_one_hot(token_ids, vocab_size):
    token_ids = np.asarray(token_ids)

    one_hot = np.eye(vocab_size)[token_ids]
    return one_hot

def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)

def calculate_gradient(probs, one_hot_x, one_hot_target, alpha, vocab_size, weights):
    error = probs - one_hot_target
    grad_cross_entropy = (one_hot_x.T @ error) / len(one_hot_x)
    grad_l2 = (2 * alpha / (vocab_size * vocab_size)) * weights

    grad_W = grad_cross_entropy + grad_l2
    return grad_W

    