import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	# Your code here
	X_mean = X.mean(axis=-1, keepdims=True)
	X_variance = X.var(axis=-1, keepdims=True)
	normalised = (X - X_mean) / np.sqrt(X_variance + epsilon)
	normalised = gamma * normalised + beta

	return normalised