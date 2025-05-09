import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from flappy_env import FlappyBirdGame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dqn network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),

            nn.Linear(64, output_size)
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
    def forward(self, x):
        return self.model(x)


# DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.001
        self.batch_size = 32

        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.update_target_net()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.policy_net(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        q_values = self.policy_net(states).gather(1, actions).squeeze()

        next_q_values = self.target_net(next_states).max(1)[0].detach()
        q_target = rewards.view(-1) + (1 - dones.view(-1)) * self.gamma * next_q_values
        loss = self.criterion(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train(episodes=5000):
    env = FlappyBirdGame(headless=True)
    state_size = 4
    action_size = 2
    rewards = []

    agent = DQNAgent(state_size, action_size)
    best = float('-inf')
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()

        rewards.append(total_reward)
        agent.decay_epsilon()
        if episode >= 100:
            avg_reward = np.mean(rewards[-100:])
            if avg_reward > best:
                best = avg_reward
                torch.save(agent.policy_net.state_dict(), "flappy_bird_dqn.pth")
                print(f"New best average reward: {avg_reward:.2f}")
        if episode % 10 == 0:
            agent.update_target_net()
            print(f"Episode {episode} - Score: {total_reward:.2f} - Epsilon: {agent.epsilon:.2f}")
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.grid(True)
    plt.show()
    env.close()

if __name__ == "__main__":
    train()
