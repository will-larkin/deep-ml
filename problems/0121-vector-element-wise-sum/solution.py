def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	# Return the element-wise sum of vectors 'a' and 'b'.
	# If vectors have different lengths, return -1.

	if len(a) != len(b):
		return -1
	
	new_vec = []

	for c_a, c_b in zip(a, b):
		new_vec.append(c_a + c_b)
	
	return new_vec