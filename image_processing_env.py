import gym
from gym import spaces
import numpy as np
import cv2
from collections import deque

class ImageProcessingEnv(gym.Env):
    def __init__(self, images, ground_truths):
        super(ImageProcessingEnv, self).__init__()

        self.images = images
        self.ground_truths = ground_truths
        self.current_image_index = 0
        self.image = self.images[self.current_image_index]
        self.ground_truth = self.ground_truths[self.current_image_index]
        
        # Define action space and observation space
        # Example: 4 discrete actions (filter type, contrast adjustment, thresholding, segmentation method)
        self.action_space = spaces.Discrete(4)  # Change based on your actual action set size

        # Observation space: Image is the observation (assuming RGB image)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.image.shape, dtype=np.uint8)

        # Initialize history (for some cases, you might want to have a rolling window)
        self.history = deque(maxlen=5)

    def reset(self):
        # Reset the environment to the initial state (first image and ground truth)
        self.current_image_index = 0
        self.image = self.images[self.current_image_index]
        self.ground_truth = self.ground_truths[self.current_image_index]
        return self.image

    def step(self, action):
        # Apply the action (this should be defined in your processing pipeline)
        self.image = self.apply_action(self.image, action)

        # Calculate reward (compare processed image to ground truth)
        reward = self.calculate_reward(self.image, self.ground_truth)

        # Define done condition (e.g., if you've reached a maximum number of steps or a good result)
        done = False  # Set termination logic if necessary (e.g., after 1000 steps or good performance)

        # Info can include additional information you may want for analysis (optional)
        info = {}

        # Move to the next image (if applicable)
        self.current_image_index += 1
        if self.current_image_index >= len(self.images):
            self.current_image_index = 0  # Or stop the episode if it's the last image

        return self.image, reward, done, info

    def render(self):
        # Display or visualize the current image
        cv2.imshow('Image Processing', self.image)
        cv2.waitKey(1)

    def apply_action(self, image, action):
        # Implement the logic for applying the action to the image
        # This should apply the filtering, contrast adjustment, thresholding, and segmentation based on the action.
        # Example:
        if action == 0:
            image = self.apply_filter(image, 'Gaussian', 1.5)  # Apply Gaussian filter with sigma=1.5
        elif action == 1:
            image = self.apply_contrast(image, 'CLAHE', 2.0)  # Apply CLAHE
        elif action == 2:
            image = self.apply_threshold(image, 'Otsu', 0)  # Apply Otsu thresholding
        elif action == 3:
            image = self.apply_segmentation(image, 'Canny', [100, 200])  # Apply Canny segmentation
        
        return image

    def calculate_reward(self, processed_image, ground_truth):
        # Implement a reward function, e.g., comparing processed image to ground truth using similarity metrics
        edges_processed = cv2.Canny(processed_image, 50, 150)
        edges_truth = cv2.Canny(ground_truth, 50, 150)
        
        # Example: measure similarity between edges (you can use other metrics such as IoU or Dice coefficient)
        similarity = np.sum(edges_processed == edges_truth) / edges_processed.size
        reward = similarity  # Higher similarity means higher reward
        return reward

    def apply_filter(self, image, filter_type, param):
        if filter_type == 'Gaussian':
            return cv2.GaussianBlur(image, (5, 5), param)
        elif filter_type == 'Median':
            return cv2.medianBlur(image, param)
        elif filter_type == 'Mean':
            return cv2.blur(image, (param, param))
        return image

    def apply_contrast(self, image, contrast_type, param):
        if contrast_type == 'CLAHE':
            clahe = cv2.createCLAHE(clipLimit=param, tileGridSize=(8, 8))
            return clahe.apply(image)
        return image

    def apply_threshold(self, image, threshold_type, param):
        if threshold_type == 'Otsu':
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        return image

    def apply_segmentation(self, image, method, param):
        if method == 'Canny':
            return cv2.Canny(image, param[0], param[1])
        return image
