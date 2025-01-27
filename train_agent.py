import gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import cv2
import numpy as np

# Assuming ImageProcessingEnv is defined in image_processing_env.py
from image_processing_env import ImageProcessingEnv

# Load a sample image and ground truth (ensure images are in the correct format, e.g., grayscale or RGB)
images = cv2.imread('path/to/your/images', cv2.IMREAD_COLOR)  # Modify based on image type (color or grayscale)
ground_truths = cv2.imread('path/to/your/ground_truths', cv2.IMREAD_COLOR)

# Initialize the environment
env = ImageProcessingEnv(images, ground_truths)
check_env(env)  # Check the environment

# Wrap the environment for stable baselines
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# Initialize the DQN agent
model = DQN('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the trained model
model.save("dqn_image_processing")

# Test the trained agent
obs = env.reset()
for i in range(1000):  # Adjust the number of steps based on your problem
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        break
