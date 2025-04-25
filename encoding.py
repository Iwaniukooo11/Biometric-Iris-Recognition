import numpy as np

def resample_stripe(stripe, new_length=128):
    """Resample 1D stripe data to specified length using linear interpolation"""
    old_length = len(stripe)
    if old_length == new_length:
        return np.array(stripe)
    
    x_old = np.linspace(0, 1, old_length)
    x_new = np.linspace(0, 1, new_length)
    return np.interp(x_new, x_old, stripe)

def encode_iris(feature_vectors, f=0.1):
    """
    Encodes normalized iris stripes using Gabor wavelets and phase quantization
    
    Parameters:
        feature_vectors (list of lists): Output from normalization function
        f (float): Gabor wavelet frequency parameter
        
    Returns:
        np.ndarray: 2048-bit iris code as binary array
    """
    sigma = (1 / 2) * np.pi * f  # Relationship from Daugman's formula
    iris_code = []
    
    for stripe in feature_vectors:
        # 1. Resample stripe to 128 points
        stripe_resampled = resample_stripe(stripe, 128)
        
        # 2. Create coordinate grids
        x_i = np.arange(128)
        X_i, X_k = np.meshgrid(x_i, x_i)
        
        # 3. Compute Gabor wavelet components
        gaussian = np.exp(-(X_i - X_k)**2 / sigma**2)
        complex_exp = np.exp(-1j * 2 * np.pi * f * X_i)
        
        # 4. Calculate complex coefficients
        I = stripe_resampled.astype(np.float32)
        coefficients = np.sum(I * gaussian * complex_exp, axis=1)
        
        # 5. Convert phase to bits
        phases = np.angle(coefficients)
        phases[phases < 0] += 2 * np.pi  # Convert to [0, 2π) range
        
        for phase in phases:
            if phase < np.pi/2:
                iris_code.extend([0, 0])
            elif phase < np.pi:
                iris_code.extend([0, 1])
            elif phase < 3*np.pi/2:
                iris_code.extend([1, 1])
            else:
                iris_code.extend([1, 0])
    
    return np.array(iris_code[:2048], dtype=np.uint8)  # Ensure exact 2048 bits


import numpy as np

def encode_iris_book(feature_vectors, f=0.15):
    """
    Encodes normalized iris stripes using Gabor wavelets with original resolution
    according to the methodology from the book excerpt
    
    Parameters:
        feature_vectors (list of lists): Output from normalization function
        f (float): Gabor wavelet frequency parameter
        
    Returns:
        np.ndarray: Iris code as 16x128 binary matrix
    """
    sigma = (1 / 2) * np.pi * f  # From Daugman's relationship
    num_stripes = len(feature_vectors)
    encoding = np.zeros((16, 128), dtype=np.uint8)
    
    for stripe_idx, stripe in enumerate(feature_vectors):
        original_length = len(stripe)
        # Select 128 evenly spaced points from original data
        step = max(1, original_length // 128)
        x_k_indices = np.arange(0, original_length, step)[:128]
        
        # Prepare coordinate grids
        x_i = np.arange(original_length)
        x_k = x_i[x_k_indices]
        
        # Compute Gabor coefficients using original resolution
        X_i, X_k = np.meshgrid(x_i, x_k)
        gaussian = np.exp(-(X_i - X_k)**2 / sigma**2)
        complex_exp = np.exp(-1j * 2 * np.pi * f * X_i)
        coefficients = np.dot(gaussian * complex_exp, stripe)
        
        # Convert phase to 2-bit encoding
        phases = np.angle(coefficients)
        phases[phases < 0] += 2 * np.pi  # Convert to [0, 2π)
        
        # Calculate bit pairs for current stripe
        row_start = stripe_idx * 2
        for col, phase in enumerate(phases):
            if phase < np.pi/2:
                bits = [0, 0]
            elif phase < np.pi:
                bits = [0, 1]
            elif phase < 3*np.pi/2:
                bits = [1, 1]
            else:
                bits = [1, 0]
            
            encoding[row_start:row_start+2, col] = bits
    
    return encoding

import numpy as np
from skimage.exposure import equalize_hist

def enhanced_iris_encoder(feature_vectors, base_freq=0.2, num_scales=5):
    """
    Generates randomized iris codes using multi-scale Gabor filters and phase dithering
    
    Parameters:
        feature_vectors (list): Normalized iris stripes
        base_freq (float): Base Gabor frequency (0.1-0.5 works best)
        num_scales (int): Number of frequency scales to use
        
    Returns:
        np.ndarray: 16x128 randomized iris code
    """
    encoding = np.zeros((16, 128), dtype=np.uint8)
    num_stripes = len(feature_vectors)
    rng = np.random.default_rng()

    for stripe_idx, stripe in enumerate(feature_vectors):
        # 1. Preprocess stripe
        # stripe = equalize_hist(np.array(stripe)) * 255  # Enhance contrast
        original_length = len(stripe)
        
        # 2. Multi-scale Gabor processing
        all_coefficients = []
        # Select num_scales frequencies around base_freq
        frequencies = np.linspace(base_freq * (2/3), base_freq * (3/2), num_scales)
        for scale in range(num_scales):
            # Vary frequency randomly around base
            f = frequencies[scale]
            sigma = (1/2)*np.pi*f
            
            # # Random spatial shift for decorrelation
            # shift = rng.integers(0, original_length)
            # shifted_stripe = np.roll(stripe, shift)
            shifted_stripe = stripe
            
            # Calculate coefficients
            x_k = np.linspace(0, original_length-1, 128, dtype=int)
            X_i, X_k = np.meshgrid(np.arange(original_length), x_k)
            gabor = np.exp(-(X_i - X_k)**2/sigma**2) * \
                    np.exp(-1j*2*np.pi*f*(X_i))
            coefficients = np.dot(gabor, shifted_stripe)
            all_coefficients.append(coefficients)
        
        # 3. Combine multi-scale responses
        combined = np.mean(all_coefficients, axis=0)
        
        # 4. Randomized phase quantization
        phases = np.angle(combined)
        # dithering_coef = 20
        # phases += rng.uniform(-np.pi/dithering_coef, np.pi/dithering_coef, phases.shape)  # Dithering
        
        # Quantize with adaptive thresholds
        q1 = np.percentile(phases, 25)
        q2 = np.percentile(phases, 50)
        q3 = np.percentile(phases, 75)
        
        # row_start = stripe_idx * 2
        # for col, phase in enumerate(phases):
        #     if phase < q1:
        #         bits = [0,0]
        #     elif phase < q2:
        #         bits = [0,1]
        #     elif phase < q3:
        #         bits = [1,1]
        #     else:
        #         bits = [1,0]
            
        #     encoding[row_start:row_start+2, col] = bits
        phases[phases < 0] += 2 * np.pi
        row_start = stripe_idx * 2
        for col, phase in enumerate(phases):
            if phase < np.pi/2:
                bits = [0, 0]
            elif phase < np.pi:
                bits = [0, 1]
            elif phase < 3*np.pi/2:
                bits = [1, 1]
            else:
                bits = [1, 0]
            encoding[row_start:row_start+2, col] = bits

    return encoding