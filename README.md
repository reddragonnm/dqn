# Atari Breakout DQN Agent

This repository contains code for training and testing Deep Q-Network (DQN) agents on Atari Breakout using [Gymnasium](https://gymnasium.farama.org/) and [PyTorch](https://pytorch.org/). It supports both automated agent play and human play via keyboard controls.

## Features

- **DQN Training:** Train agents using experience replay and target networks.
- **Model Saving:** Periodically saves model checkpoints during training.
- **Testing:** Evaluate trained models in human-rendered mode.
- **Human Play:** Play Breakout interactively using keyboard controls.
- **Frame Preprocessing:** Converts frames to grayscale and crops/resizes for input to the neural network.

## Directory Structure

```
.
├── main.ipynb          # DQN training notebook
├── test_model.py       # Test trained DQN agent (4 actions)
├── test_model2.py      # Test trained DQN agent (3 actions)
├── play_human.py       # Play Breakout with keyboard
├── models/             # Saved model checkpoints (DQN)
├── models2/            # Saved model checkpoints (alternate DQN)
├── runs/               # Training logs (optional)
├── README.md           # This file
└── .gitignore
```

## Getting Started

### 1. Install Dependencies

```sh
pip install torch gymnasium[atari] ale-py opencv-python matplotlib pygame
```

You may need to install Atari ROMs using [AutoROM](https://github.com/Farama-Foundation/AutoROM):

```sh
pip install autorom
AutoROM --accept-license
```

### 2. Training

Open and run main.ipynb to train a DQN agent. Model checkpoints are saved in models as `dqn_model_episode_{N}.pth`.

### 2a. Visualize Training with TensorBoard

Training metrics such as loss curves and episode rewards are logged using TensorBoard. To view these graphs:

```sh
tensorboard --logdir runs
```

Then open the displayed URL in your browser to explore the training progress interactively.

### 3. Testing

To test a trained model:

```sh
python test_model.py
```

or

```sh
python test_model2.py
```

**Note:**

- `test_model.py` runs a full episode until the game ends (all lives lost).
- `test_model2.py` terminates the episode immediately upon life loss (for more granular evaluation).

Edit the model path in the script to select a specific checkpoint.

### 4. Human Play

Play Breakout using your keyboard:

```sh
python play_human.py
```

**Controls:**

- Left / A: Move left
- Right / D: Move right
- Space: FIRE (launch ball)
- Esc / Window close: Quit

## Model Architecture

See `DQN` and `DQN` for details. The agent uses a convolutional neural network with stacked grayscale frames as input.

## Preprocessing

Frames are converted to grayscale, resized to $84 \times 110$, and cropped to $84 \times 84$ using `preprocess_image`.

## Referenced Papers

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — Mnih et al., 2013
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Mnih et al., 2015

## License

MIT License

---

For questions or issues, please open an issue or discussion.
