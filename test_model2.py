import torch
from torch import nn
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

import ale_py
import cv2
from matplotlib import pyplot as plt
from collections import deque

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="human")
env = AtariPreprocessing(
    env, grayscale_obs=True, scale_obs=True, frame_skip=1, terminal_on_life_loss=True
)


def preprocess_image(image, crop_top=20):
    image = cv2.resize(image, (84, 110))
    return image[crop_top : crop_top + 84, :]


class DQN(nn.Module):
    def __init__(self, num_actions=3):  # inp shape: (b, 4, 84, 84)
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
model.load_state_dict(torch.load("models2/best_model.pth", map_location="cpu"))

model.eval()

ACTION_MAP = [0, 2, 3]  # NOOP, RIGHT, LEFT

for episode in range(10):
    obs, info = env.reset()
    env.step(1)  # FIRE
    env.step(1)  # FIRE

    frame_stack = deque([preprocess_image(obs)] * 4, maxlen=4)

    done = False

    while not done:
        with torch.no_grad():
            inp = torch.tensor(frame_stack, dtype=torch.float32).unsqueeze(
                0
            )  # add batch dim
            action_idx = torch.argmax(model(inp), dim=1).item()

        action = ACTION_MAP[action_idx]
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        frame_stack.append(preprocess_image(obs))
