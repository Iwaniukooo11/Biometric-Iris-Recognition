import numpy as np

def hamming_distance(code1, code2):
    """
    Calculate the Hamming distance between two binary iris codes.
    """
    if code1.shape != code2.shape:
        raise ValueError("Iris codes must have the same shape.")
    return np.sum(code1 != code2) / code1.size

