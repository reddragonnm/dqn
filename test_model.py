import torch
from torch import nn
import gymnasium as gym
import ale_py
import cv2
from matplotlib import pyplot as plt
from collections import deque
import random
import copy

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="human")


def preprocess_image(image, crop_top=20):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.0
    image = cv2.resize(image, (84, 110))
    return image[crop_top : crop_top + 84, :]  # 84x84 grayscale


class DQN(nn.Module):
    def __init__(self, num_actions=4):  # inp shape: (b, 4, 84, 84)
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),  # 84x84 - 20x20
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # 20x20 - 9x9
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        return self.layers(x)


model = DQN().to("cpu")
model.load_state_dict(torch.load("models/best_model.pth", map_location="cpu"))

model.eval()

for episode in range(10):
    episode_reward = 0
    episode_frames = 0

    obs, info = env.reset()
    frame_stack = deque([preprocess_image(obs)] * 4, maxlen=4)

    done = False

    while not done:
        if episode_frames % 100 == 0:
            action = 1
            print("FIRE!")
        else:
            with torch.no_grad():
                inp = torch.tensor(frame_stack, dtype=torch.float32).unsqueeze(
                    0
                )  # add batch dim
                action = torch.argmax(model(inp), dim=1).item()

        obs, reward, terminated, truncated, info = env.step(action)

        # print(f"Selected action: {action} info: {info}")

        episode_reward += reward
        episode_frames += 1
        done = terminated or truncated

        frame_stack.append(preprocess_image(obs))
