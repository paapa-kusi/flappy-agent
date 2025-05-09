# Flappy Bird DQN Agent

This project is a reinforcement learning implementation of a Deep Q-Network (DQN) agent trained to play a clone of the classic **Flappy Bird** game using PyTorch and Pygame.

## Objective

Train an agent to learn how to play Flappy Bird from scratch through trial and error using the DQN algorithm.

---

## Features

- **Deep Q-Learning** with experience replay and a target network
- **Custom Flappy Bird environment** using Pygame
- **Layer Normalization** and **Kaiming weight initialization** for stable training
- **Epsilon-Greedy Exploration** with decay
- **Reward shaping** to encourage staying near pipe center and surviving
- **GPU Acceleration** (optional)

---

## Learning Strategy

The agent receives:
- A small reward for staying alive
- A bonus reward for passing through pipes
- A penalty for collisions
- Extra shaping based on how well-aligned it is with the pipe gap center

Over time, the agent learns to balance exploration and exploitation to maximize its score.

---

## 📁 Project Structure

```bash
├── assets/                 # Sprites and sound files
├── objects/               # Game object classes (Bird, Pipe, etc.)
├── flappy_env.py         # Game environment wrapper for training
├── train.py              # DQN training script
├── test.py               # Trained agent tester
├── README.md             # Project overview
```

## Running the Code

### Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pygame
- Matplotlib

Install dependencies:

```bash
pip install torch pygame numpy matplotlib
```
---
### Training the Agent

To train the DQN agent:

```bash
python train.py
```

This will launch the training loop using a headless Pygame environment. The model will be saved as `flappy_bird_dqn.pth` whenever it reaches a new best average reward over the past 100 episodes. The number of episodes can be modified in the function parameters. Reward and penalty values can be modified in the `flappy_env.py` script.

### Testing the Agent

To visualize the performance of the trained agent:

```bash
python test.py
```

This will render the Flappy Bird game and have the trained model play rounds until a certain reward threshold is reached. This threshold value can be changed in the function parameters.


## License & Attribution

The Flappy Bird clone used in this project is originally sourced from a repository from this user.

> **[mehmetemineker/flappy-bird](https://github.com/mehmetemineker/flappy-bird)**  


This project and its modifications are intended for **educational, personal, and experimental purposes only**.

---

## Demo

[![Flappy Bird DQN Agent Demo](https://img.youtube.com/vi/5swwiNZEHMk/maxresdefault.jpg)](https://youtu.be/5swwiNZEHMk)
---

## 🙌 Acknowledgments

- The open-source contributor of the Flappy Bird clone (Apache 2.0)
