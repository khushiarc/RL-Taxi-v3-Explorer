# RL-Taxi-v3-Explorer
RL-Taxi-v3 Explorer uses Deep Q-Learning to solve the Gymnasium Taxi-v3 task. It features an ANN-based agent that learns to navigate, pick up, and drop off passengers efficiently. The project highlights $\epsilon$-greedy exploration, reward optimization, and data-driven performance tracking.

An implementation of Deep Reinforcement Learning to solve the Gymnasium Taxi-v3 environment.

## 📌 Project Overview
The **Taxi-v3** environment is a classic reinforcement learning problem where an agent must navigate a 5x5 grid to pick up a passenger at one of four locations and drop them off at another. This project, **RL-Taxi-v3 Explorer**, implements an autonomous agent using **Deep Q-Networks (DQN)** to master the environment's rewards system and movement logic.

### Objectives:
* Implement an epsilon-greedy strategy for balanced Exploration vs. Exploitation.
* Utilize a Neural Network to approximate Q-values (Function Approximation).
* Optimize the agent to achieve a consistent positive reward over 100+ episodes.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **RL Framework:** Gymnasium (v29+)
* **Deep Learning:** TensorFlow / Keras (or PyTorch)

## 🚀 Key Features
* **Neural Q-Learning:** Unlike basic Q-tables, this uses an ANN to predict the best possible actions.
* **Performance Tracking:** Generates graphs showing the "Reward per Episode" to visualize the agent's learning curve.
* **Optimized Movement:** Successfully handles the -10 penalty for illegal pick-ups/drop-offs and the -1 per-step penalty.

## 📊 Results
After training for [X] episodes, the agent achieves:
* **Average Reward:** +7.2 to +9.7
* **Success Rate:** 98% to 100%
* **Average Steps per Trip:** 12 to 15 steps 

## 📂 Structure
* `main.py`: The Neural Network architecture and training loop.

## 📝 How to Run
1. Clone the repo: `git clone `
2. Install dependencies: `pip install gymnasium numpy tensorflow`
3. Run the trainer: `python main.py`
