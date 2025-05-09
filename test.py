import torch
import time
from flappy_env import FlappyBirdGame
from train import DQN

def test_agent(model_path="flappy_bird_dqn.pth", render_speed=60, target_reward=180):
    env = FlappyBirdGame()
    state_size = 4
    action_size = 2
    model = DQN(state_size, action_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    attempt = 0
    while True:
        state = env.reset()
        done = False
        total_reward = 0
        attempt += 1
        print(f"\nAttempt #{attempt}")

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

            state, reward, done, _ = env.step(action)
            env.render()
            time.sleep(1 / render_speed)
            total_reward += reward

        print(f"Total Reward: {total_reward:.2f}")
        if total_reward >= target_reward:
            print(f"Target reward of {target_reward} reached in attempt #{attempt}")
            break

    env.close()

if __name__ == "__main__":
    test_agent()
