from pipeline import IrisPipeline
from encoding import enhanced_iris_encoder
from normalization import daugman_normalization_modified
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_iris_code(path, verbose = True, base_freq=0.5, num_scales=5):
    """
    Given an image path, this function normalizes the iris region and encodes it into a binary format.
    
    Parameters:
        path (str): Path to the input image.
        verbose (bool): If True, displays the normalized iris image.
        
    Returns:
        np.ndarray: Encoded iris code.
    """
    # Initialize the pipeline
    pipeline = IrisPipeline(path)
    pipeline.run_pipeline()
    results_1 = pipeline.get_results()

    normalized_matrix_1 = daugman_normalization_modified(
                pipeline.gray, results_1["pupil_center"], results_1["pupil_radius"], results_1["iris_radius"]
            )
    
    # Encode the normalized iris
    iris_code = enhanced_iris_encoder(normalized_matrix_1, base_freq=base_freq, num_scales=num_scales)
    
    if verbose:
        # Show the original image
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(pipeline.gray, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        # Show the normalized iris code
        plt.subplot(1, 2, 2)
        plt.imshow(iris_code, cmap='gray')
        plt.title('Iris Code for Your Image')
        plt.axis('off')
        plt.show()
    
    return iris_code