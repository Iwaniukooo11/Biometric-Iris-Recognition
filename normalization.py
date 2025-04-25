import numpy as np
import cv2
import matplotlib.pyplot as plt

def daugman_normalization(image, pupil_center, pupil_radius, iris_radius):
    """
    Normalizes the iris region into a rectangular polar form and extracts features using Gaussian-weighted column means.
    
    Parameters:
        image (numpy.ndarray): Grayscale input image.
        pupil_center (tuple): (x, y) coordinates of the pupil center.
        pupil_radius (int): Radius of the pupil.
        iris_radius (int): Radius of the iris.
        
    Returns:
        numpy.ndarray: Matrix with 8 rows (one per stripe) and columns corresponding to angular resolution.
    """
    # Validate input
    if pupil_radius >= iris_radius:
        raise ValueError("Iris radius must be larger than pupil radius.")
    
    # Step 1: Convert iris ring to polar coordinates
    radial_samples = iris_radius - pupil_radius
    theta_samples = int(2 * np.pi * iris_radius)  # Angular resolution based on circumference
    
    # Generate radial and angular grids
    r = np.linspace(pupil_radius, iris_radius, radial_samples, endpoint=False)
    theta = np.linspace(0, 2 * np.pi, theta_samples, endpoint=False)
    rr, tt = np.meshgrid(r, theta, indexing='ij')  # Shape: (radial_samples, theta_samples)
    
    # Compute Cartesian coordinates for sampling
    x = pupil_center[0] + rr * np.cos(tt)
    y = pupil_center[1] + rr * np.sin(tt)
    
    # Clip coordinates to image boundaries
    x = np.clip(x, 0, image.shape[1] - 1)
    y = np.clip(y, 0, image.shape[0] - 1)
    
    # Remap image to polar coordinates
    map_x = x.astype(np.float32)
    map_y = y.astype(np.float32)
    polar_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    
    # Step 2: Split into 8 horizontal stripes (along radial direction)
    stripes = np.array_split(polar_image, 8, axis=0)
    
    # Step 3: Process each stripe with Gaussian-weighted column means
    feature_matrix = []
    for stripe in stripes:
        if stripe.size == 0:
            continue  # Skip empty stripes (unlikely with correct parameters)
        # DEBUG
        print(stripe.shape)
        H, W = stripe.shape
        
        # Generate Gaussian weights for the stripe's height
        x_coord = np.arange(H)
        mu = (H - 1) / 2.0  # Center of the stripe
        sigma = max(H / 4.0, 1e-5)  # Prevent division by zero
        weights = np.exp(-(x_coord - mu)**2 / (2 * sigma**2))
        weights /= weights.sum()  # Normalize
        
        # Calculate weighted means for each column
        weighted_means = np.dot(weights, stripe)
        feature_matrix.append(weighted_means)
    
    # Handle edge case where all stripes are empty
    if not feature_matrix:
        return np.array([], dtype=np.float32).reshape(0, 0)
    
    return np.vstack(feature_matrix)


def daugman_normalization_modified(image, pupil_center, pupil_radius, iris_radius, debug = False):
    """
    Normalizes the iris region into 8 concentric stripes with angular constraints and extracts Gaussian-weighted features.
    
    Parameters:
        image (numpy.ndarray): Grayscale input image.
        pupil_center (tuple): (x, y) coordinates of the pupil center.
        pupil_radius (int): Radius of the pupil.
        iris_radius (int): Radius of the iris.
        
    Returns:
        list of lists: Each sublist represents a stripe's feature vector.
    """
    if pupil_radius > iris_radius:
        raise ValueError("Iris radius must be larger than pupil radius.")
    elif pupil_radius == iris_radius:
        print("Warning: Pupil radius is equal to iris radius. Syntheticly increasing iris radius by 8 pixels.")
        iris_radius += 8
    
    debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if debug else None
    num_rings = 8
    delta_r = (iris_radius - pupil_radius) / num_rings
    feature_vectors = []

    # Calculate adaptive resolutions
    # angular_resolution = int(2 * np.pi * iris_radius)  # 1 sample per pixel at outer edge
    radial_resolution = int(iris_radius - pupil_radius)  # 1 sample per pixel radially
    
    for ring in range(num_rings):
        # Radial range for the current ring
        r_start = pupil_radius + ring * delta_r
        r_end = pupil_radius + (ring + 1) * delta_r
        radial_samples = int(radial_resolution / num_rings)
        r = np.linspace(r_start, r_end, radial_samples, endpoint=False)
        
        # Determine angular regions (in degrees)
        if ring < 4:
            # 360° - 30° gap at bottom (150°-180° in standard coords)
            segments = [
                (105, 360),  # 105° to 360°
                (0, 75)      # 0° to 75°
            ]
        elif 4 <= ring < 6:
            # Middle rings: 124-237° + 303-360° + 0-56°
            segments = [
                (124, 237),  # 113°
                (303, 360),  # 57°
                (0, 56)      # 56°
            ]
        else:
            # Outer rings: 135-225° + 315-360° + 0-45°
            segments = [
                (135, 225),  # 90°
                (315, 360),  # 45°
                (0, 45)      # 45°
            ]
        color = (0, 255, 0) if ring < 4 else (0, 0, 255) if ring < 6 else (255, 0, 0)

        # Angular resolution based on current ring's outer radius
        angular_resolution = int(2 * np.pi * r_end)  # 1 sample per pixel circumferentially

        # Calculate angular samples for each segment
        theta_deg = np.array([], dtype=np.float32)
        for start, end in segments:
            arc_length = (end - start) if end > start else (360 - start + end)
            samples = max(1, int(angular_resolution * arc_length / 360))
            theta_deg = np.concatenate([
                theta_deg,
                np.linspace(start, end, samples, endpoint=False)
            ])
        
        # Convert degrees to radians
        theta = np.deg2rad(theta_deg)

        # DEBUG
        r_vis = np.array([r_start, r_end])  # For visualization
        if debug and debug_image is not None:
            # Draw angle markers for visualization
            for angle in theta_deg:
                for radius in r_vis:
                    x = int(pupil_center[0] + radius * np.cos(np.deg2rad(angle)))
                    y = int(pupil_center[1] + radius * np.sin(np.deg2rad(angle)))
                    cv2.circle(debug_image, (x, y), 2, color, -1)
        
        # Generate meshgrid for polar coordinates
        rr, tt = np.meshgrid(r, theta, indexing='ij')  # Shape: (radial_samples, theta_samples)
        
        # Convert to Cartesian coordinates
        x = pupil_center[0] + rr * np.cos(tt)
        y = pupil_center[1] + rr * np.sin(tt)
        
        # Clip coordinates to image boundaries
        x = np.clip(x, 0, image.shape[1] - 1)
        y = np.clip(y, 0, image.shape[0] - 1)
        
        # Remap the image to polar coordinates
        map_x = x.astype(np.float32)
        map_y = y.astype(np.float32)
        stripe = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        
        # Skip empty stripes
        if stripe.size == 0:
            feature_vectors.append([])
            continue
        
        # DEBUG
        # Cut the stripe to length 300, if shorter then pad with 0
        if debug:
            if stripe.shape[1] < 300:
                stripe_cut = np.pad(stripe, ((0, 0), (0, 300 - stripe.shape[1])), mode='constant', constant_values=0)
            elif stripe.shape[1] > 300:
                stripe_cut = stripe[:, :300]
            # Imshow the stripe
            plt.figure(figsize=(6, 2))
            plt.imshow(stripe_cut, cmap='gray')
            plt.title(f"Stripe {ring + 1}")
            plt.axis('off')
            plt.show()
            print(f"Stripe {ring + 1} shape: ", stripe_cut.shape)
            
        
        # Gaussian-weighted column means
        H, W = stripe.shape
        x_coord = np.arange(H)
        mu = (H - 1) / 2.0
        sigma = max(H / 4.0, 1.0)
        weights = np.exp(-(x_coord - mu)**2 / (2 * sigma**2))
        weights /= weights.sum()  # Normalize
        
        # Compute weighted means for each column
        weighted_means = np.dot(weights, stripe)
        feature_vectors.append(weighted_means.tolist())

    if debug and debug_image is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(debug_image)
        plt.title("Iris Division into 8 Stripes with Angular Constraints")
        plt.axis('off')
        plt.show()
    
    return feature_vectors