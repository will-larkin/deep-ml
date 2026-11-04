import numpy as np

def pos_encoding(position: int, d_model: int):
	# Your code here
    if position == 0 or d_model <= 0:
        return -1

    # Position indices: shape (position, 1)
    pos = np.arange(position, dtype=np.float32)[:, None]

    # Dimension indices: shape (1, d_model)
    i = np.arange(d_model, dtype=np.float32)[None, :]

    # Frequencies: 1 / 10000^(2i / d_model)
    angle_rates = 1.0 / np.power(
        10000.0,
        (2 * (i // 2)) / np.float32(d_model)
    )

    # Position-dependent angles
    angles = pos * angle_rates

    # Even dimensions use sine; odd dimensions use cosine
    pe = np.empty((position, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe.astype(np.float16)