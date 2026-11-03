import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""
	attention_scores = Q @ K.T
	d_k = Q.shape[1]
	scaled_scores = attention_scores / np.sqrt(d_k)
	masked_scores = apply_mask(scaled_scores, mask)
	attention_weights = softmax(masked_scores)
	weighted_values = attention_weights @ V

	return (weighted_values, attention_weights)

def apply_mask(scores, mask):
	if mask is None:
		return scores
	
	return np.where(mask == 1, scores, -np.inf)

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
