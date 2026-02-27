import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Environment
env = gym.make('Taxi-v3')

# Hyperparameters
alpha = 0.001
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.1
num_episodes = 2000  # Increased for better learning
max_steps = 100
batch_size = 32
memory_size = 5000
train_every = 5

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One-hot encoding function
def one_hot(state, size):
    vec = np.zeros(size)
    vec[state] = 1
    return torch.tensor(vec, dtype=torch.float32).to(device)

# Q-network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# Model, optimizer, loss
state_size = env.observation_space.n
action_size = env.action_space.n
model = QNetwork(state_size, action_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=alpha)
loss_fn = nn.MSELoss()

# Replay buffer
memory = deque(maxlen=memory_size)

# Epsilon-greedy action
def choose_action(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    state_tensor = one_hot(state, state_size)
    with torch.no_grad():
        return torch.argmax(model(state_tensor)).item()

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.append((state, action, reward, next_state, done))
        state = next_state

        if step % train_every == 0 and len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            s, a, r, s2, d = zip(*batch)

            s = torch.stack([one_hot(st, state_size) for st in s])
            a = torch.tensor(list(a), dtype=torch.int64).view(-1, 1).to(device)
            r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(device)
            s2 = torch.stack([one_hot(st, state_size) for st in s2])
            d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(device)

            q_vals = model(s).gather(1, a)
            with torch.no_grad():
                q_next = model(s2).max(1)[0].unsqueeze(1)
                q_target = r + gamma * q_next * (1 - d)

            loss = loss_fn(q_vals, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("✅ Training complete")


# Testing
env = gym.make('Taxi-v3', render_mode='human')
for ep in range(20):
    state, _ = env.reset()
    done = False
    print(f"\n🚕 Episode {ep + 1}")
    for _ in range(max_steps):
        env.render()
        state_tensor = one_hot(state, state_size)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
            print(f"State: {state}, Q-values: {q_values.cpu().numpy()}, Action: {action}")

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        if done:
            print(f"🎯 Finished with reward {reward}")
            break
env.close()
