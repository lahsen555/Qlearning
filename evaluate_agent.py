import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from image_processing_env import ImageProcessingEnv
import cv2

# Load a new image and ground truth for evaluation
image = cv2.imread('path/to/new/image.jpg', cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread('path/to/new/ground_truth.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize the environment
env = ImageProcessingEnv(image, ground_truth)
env = DummyVecEnv([lambda: env])

# Load the trained model
model = DQN.load("dqn_image_processing")

# Test the agent
obs = env.reset()
total_rewards = 0
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    total_rewards += rewards
    if dones:
        break

print(f"Total Reward: {total_rewards}")