# filepath: /home/iwaniukooo/Documents/Projects/biometria-iris/Biometric-Iris-Recognition/iris_pipeline.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

class IrisPipeline:
    def __init__(self, path, debug = False):
        self.path = path
        self.img_original = None
        self.img_rgb = None
        self.gray = None
        self.binary = None
        self.new_center = None
        self.new_radius = None
        self.average_projection = None
        self.pupil_radius = None
        self.iris_rect = None
        self.pupil_threshold = 20
        self.mean_annular = None
        self.debug = debug

    def load_image(self):
        # If the image is supplied insteaad of path, use it directly
        if isinstance(self.path, np.ndarray):
            self.img_original = self.path
        else:
            self.img_original = cv2.imread(self.path)

        # Load the image and convert it to RGB
        self.img_rgb = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)

        # Center crop the image
        h, w, _ = self.img_rgb.shape
        crop_size = int(min(h, w) * 0.75)  # Crop to 60% of the smaller dimension
        center_x, center_y = w // 2, h // 2
        x1 = max(center_x - crop_size // 2, 0)
        y1 = max(center_y - crop_size // 2, 0)
        x2 = min(center_x + crop_size // 2, w)
        y2 = min(center_y + crop_size // 2, h)
        self.img_rgb = self.img_rgb[y1:y2, x1:x2]

    def preprocess(self):
        # Convert to grayscale
        self.gray = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2GRAY)
        # Binarize to find the pupil
        _, binary_inv = cv2.threshold(self.gray, self.pupil_threshold, 255, cv2.THRESH_BINARY_INV)
        # Morphological closing
        kernel_close = np.ones((5,5), np.uint8)
        closed = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        # Remove small noise
        kernel_open = np.ones((3,3), np.uint8)
        self.binary = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=10)

    def find_pupil(self):
        # Projections to find approximate center
        h, w = self.binary.shape
        vertical_projection = np.sum(self.binary, axis=1)
        horizontal_projection = np.sum(self.binary, axis=0)

        vertical_center = np.argmax(vertical_projection)
        horizontal_center = np.argmax(horizontal_projection)

        v_max_1, v_min_1 = vertical_center, vertical_center
        h_max_1, h_min_1 = horizontal_center, horizontal_center

        # Find the boundary in vertical direction
        for i in range(vertical_center, h):
            if self.binary[i, horizontal_center] == 0:
                v_max_1 = i
                break
        for i in range(vertical_center, -1, -1):
            if self.binary[i, horizontal_center] == 0:
                v_min_1 = i
                break

        # Same for horizontal direction
        for j in range(horizontal_center, w):
            if self.binary[vertical_center, j] == 0:
                h_max_1 = j
                break
        for j in range(horizontal_center, -1, -1):
            if self.binary[vertical_center, j] == 0:
                h_min_1 = j
                break

        # Another way to find final min/max
        vertical_max_freq = vertical_projection[vertical_center]
        horizonal_max_freq = horizontal_projection[horizontal_center]
        v_max, v_min, h_max, h_min = v_max_1, v_min_1, h_max_1, h_min_1

        for j in range(vertical_center, h):
            if vertical_projection[j] < vertical_max_freq * 0.2:
                v_max = j
                break
        for j in range(vertical_center, -1, -1):
            if vertical_projection[j] < vertical_max_freq * 0.2:
                v_min = j
                break
        for j in range(horizontal_center, w):
            if horizontal_projection[j] < horizonal_max_freq * 0.2:
                h_max = j
                break
        for j in range(horizontal_center, -1, -1):
            if horizontal_projection[j] < horizonal_max_freq * 0.2:
                h_min = j
                break

        # Compute radius and center
        self.new_radius = (v_max - v_min + h_max - h_min) // 4
        self.new_center = ((h_max + h_min) // 2, (v_max + v_min) // 2)

    def enhance_image(self):
        # Create mask and set inside circle to white
        mask = np.zeros_like(self.img_rgb)
        cv2.circle(mask, self.new_center, self.new_radius+2, (255,255,255), -1)
        img_with_pupil = self.img_rgb.copy()
        img_with_pupil[mask > 0] = 255
        #plot img_with_pupil
        if self.debug:
            plt.imshow(img_with_pupil)
            plt.title("Image with Pupil")
            plt.axis('off')
            plt.show()
        
        #enhanced_image is img_with_pupil minus mean + 128 and clip
        mean = np.mean(img_with_pupil)
        enhanced_img = img_with_pupil.astype(np.float32) - mean + 128
        enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
        
        gamma = 0.75
        enhanced_img = np.power(enhanced_img / 255.0, gamma) * 255
        enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)
        if self.debug:
            plt.imshow(enhanced_img)
            plt.title("Enhanced Image")
            plt.axis('off')
            plt.show()

        return enhanced_img
        
        # return img_with_pupil

    def calculate_mean_annular(self, enhanced_img):
        if len(enhanced_img.shape) == 3:  # If the image has 3 channels (color)
            enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2GRAY)
        _new_radius = self.new_radius+5
        # Create a mask for the larger circle (new_radius + 2)
        h, w = enhanced_img.shape
        y, x = np.ogrid[:h, :w]
        distance_from_center = np.sqrt((x - self.new_center[0])**2 + (y - self.new_center[1])**2)

        # Mask for the larger circle
        mask_large = distance_from_center <= (_new_radius +10)

        # Mask for the smaller circle
        mask_small = distance_from_center <= _new_radius

        # Subtract the smaller circle mask from the larger circle mask to get the annular region
        mask_annular = mask_large & ~mask_small

        # Calculate the mean pixel value in the annular region
        mean_annular = np.mean(enhanced_img[mask_annular])
        self.mean_annular = mean_annular

    def threshold_and_projection(self, enhanced_img):
        # Use the pupil center and radius calculated in find_pupil
        cx, cy = self.new_center
        r = self.new_radius
        self.calculate_mean_annular(enhanced_img)
        # mean_val = self._calculate_mean_on_horizontal_line(enhanced_img, cx, cy, r)
        _, binary_thresh = cv2.threshold(
            enhanced_img,
            # min(
            # 5 * mean_val, 220),
            self.mean_annular+12,
            255,
            cv2.THRESH_BINARY
        )

        # kernel2 = np.ones((2,2), np.uint8)
        # closing = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, kernel2, iterations=4)
        # gray_bin = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)

        # UNCOMMENT TO SEE THRESHOLD EFFECT
        # self.calculate_mean_annular(binary_thresh)
        if self.debug:
            plt.imshow(binary_thresh, cmap='gray')
            plt.title(f"Binary Threshold Image {self.mean_annular}")
            plt.axis('off')
            plt.show()

        
        # TODO: check these angles on image (should be on the bottom IMO)
        # angle_start, angle_end = 120, 60
        # angle_start, angle_end = 20, 40
        # angle_start, angle_end = 30, 50
        angle_start, angle_end = 90, 50
        
        binary_thresh=cv2.cvtColor(binary_thresh, cv2.COLOR_BGR2GRAY)
        projection = self._calculate_projection(binary_thresh, self.new_center, angle_start, angle_end)
        # self.average_projection = int(np.mean(projection))
        self.average_projection=int(np.max(projection))
        # projection=sorted(projection,reverse=True)
        # projection_10=projection[5]
        # self.average_projection=projection_10
        return binary_thresh


    def _calculate_mean_on_horizontal_line(self, img, cx, cy, r):
        outer_radius = int(1.2 * (r + 2))
        inner_radius = r + 2
        x_min = max(0, cx - outer_radius)
        x_max = min(img.shape[1], cx + outer_radius)
        pixel_values = []
        for x in range(x_min, x_max):
            distance = abs(x - cx)
            if inner_radius < distance <= outer_radius:
                pixel_values.append(img[cy, x])
        return np.mean(pixel_values) if pixel_values else 0

    def _calculate_projection(self, gray_image, center, angle_start, angle_end, num_lines=100):
        h, w = gray_image.shape
        projection = []
        angles = np.linspace(np.radians(angle_start), np.radians(angle_end), num_lines)
        for angle in angles:
            dx = np.cos(angle)
            dy = np.sin(angle)
            line_sum = 0
            for r in range(max(h, w)):
                x = int(center[0] + r * dx)
                y = int(center[1] + r * dy)
                distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                
                if 0 <= x < w and 0 <= y < h and distance <= self.new_radius *2.5:
                    line_sum += 1 if gray_image[y, x] == 0 else 0
                else:
                    break
            projection.append(line_sum)
        return projection

    def create_rectangular_iris(self):
        # Construct the IrisProcessor-like functionality
        pupil_radius = self.new_radius
        iris_radius = self.new_radius + self.average_projection
        # Ensure grayscale
        if len(self.img_original.shape) > 2:
            working_img = self.img_original[:,:,0]
        else:
            working_img = self.img_original
        self.iris_rect = np.ones((1,1)) # placeholder

        # Create mask for iris
        rows, cols = working_img.shape
        y_grid, x_grid = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((x_grid - self.new_center[0])**2 + (y_grid - self.new_center[1])**2)
        mask_iris = (dist_from_center >= pupil_radius) & (dist_from_center <= iris_radius)

        width = int(2 * np.pi * iris_radius)
        height = int(iris_radius - pupil_radius)
        height = max(height, 1)
        self.iris_rect = np.ones((height, width)) * 128

        coords = np.argwhere(mask_iris)
        for y, x in coords:
            r = np.sqrt((x - self.new_center[0])**2 + (y - self.new_center[1])**2)
            dx = (x - self.new_center[0]) / r if r else 0
            dy = (y - self.new_center[1]) / r if r else 0
            theta = np.arctan2(dy, dx)
            if theta < 0: theta += 2*np.pi
            rect_y = int(r - pupil_radius)
            rect_y = max(min(height - 1, rect_y), 0)
            rect_x = int(theta * iris_radius)
            rect_x = max(min(width - 1, rect_x), 0)
            self.iris_rect[rect_y, rect_x] = working_img[y, x]

    def run_pipeline(self):
        self.load_image()
        self.preprocess()
        self.find_pupil()
        enhanced = self.enhance_image()
        self.threshold_and_projection(enhanced)
        self.create_rectangular_iris()
        
    def get_circle_img(self):
        # Draw the pupil circle on the original image
        img_with_circle = self.img_rgb.copy()
        cv2.circle(img_with_circle, self.new_center, self.new_radius, (255, 0, 0), 2)
        cv2.circle(img_with_circle, self.new_center, self.new_radius + self.average_projection, (0, 255, 0), 2)
        return img_with_circle
    
    @staticmethod
    def compare_rectangles(rect_1, rect_2):
        """
        Compare two iris rectangles by resizing them to the same dimensions and calculating MSE.
        
        :param rect_1: First iris rectangle (2D numpy array).
        :param rect_2: Second iris rectangle (2D numpy array).
        :return: Mean Squared Error (MSE) between the two rectangles.
        """
        # Get the dimensions of both rectangles
        height_1, width_1 = rect_1.shape
        height_2, width_2 = rect_2.shape

        # Determine the maximum width and height
        max_height = max(height_1, height_2)
        max_width = max(width_1, width_2)
        
        print(f"Max height: {max_height}, Max width: {max_width}")

        # Function to pad an image to the target size
        def pad_to_size(image, target_height, target_width):
            padded_image = np.full((target_height, target_width), 128, dtype=image.dtype)  # Fill with gray (128)
            padded_image[:image.shape[0], :image.shape[1]] = image
            return padded_image

        # Pad both rectangles to the same size
        rect_1_padded = pad_to_size(rect_1, max_height, max_width)
        rect_2_padded = pad_to_size(rect_2, max_height, max_width)

        # Calculate the Mean Squared Error (MSE)
        mse = np.mean((rect_1_padded - rect_2_padded) ** 2)
        return mse
    @staticmethod
    def compare_rectangles_fourier(rect_1, rect_2):
        """
        Compare two iris rectangles using Fourier Transform similarity.
        :param rect_1: First iris rectangle (2D numpy array).
        :param rect_2: Second iris rectangle (2D numpy array).
        :return: Normalized cross-correlation of Fourier magnitudes.
        """
        # Compute the Fourier Transform of both rectangles
        f1 = np.fft.fft2(rect_1)
        f2 = np.fft.fft2(rect_2)

        # Compute the magnitude spectra
        mag1 = np.abs(f1)
        mag2 = np.abs(f2)

        # Normalize the magnitude spectra
        mag1 = mag1 / np.max(mag1)
        mag2 = mag2 / np.max(mag2)

        # Compute the normalized cross-correlation
        correlation = np.sum(mag1 * mag2) / np.sqrt(np.sum(mag1**2) * np.sum(mag2**2))
        return correlation

    def get_results(self):
        return {
            "pupil_center": self.new_center,
            "pupil_radius": self.new_radius,
            "iris_radius": self.new_radius + self.average_projection,
            "iris_rectangle": self.iris_rect
        }