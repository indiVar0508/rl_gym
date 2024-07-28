"""
Solved Frozen lake puzzle for deterministic environment(non-slippery)
"""

import gymnasium as gym
import numpy as np
import pickle

env = gym.make(
    "FrozenLake-v1", render_mode="human", desc=None, map_name="4x4", is_slippery=False
)
observation, info = env.reset()


class QAgent:
    def __init__(
        self, states: int = 16, actions: int = 4, lr: float = 0.6, gamma: float = 0.25
    ) -> None:
        self.possible_states = states
        self.possible_actions = actions
        try:
            with open("brain.pkl", "rb") as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            # self.q_table = np.random.randn(states, actions)
            self.q_table = np.zeros((states, actions))
        self.lr = lr
        self.gamma = gamma

    def act(self, state: int, episode: int) -> int:
        return np.argmax(
            # choose your best wisdom
            self.q_table[state, :]
            # Add noise to your wisdom                   damp the noise with time
            # + np.random.randn(1, self.possible_actions) * (1 / (episode + 1))
        )

    def learn(self, state, action, reward, new_state):
        # Q(s, a) += lr *(future best reward - Current known value)
        self.q_table[state, action] += self.lr * (
            reward
            # Temporal Difference
            + self.gamma * np.max(self.q_table[new_state, :])
            - self.q_table[state, action]
        )


agent = QAgent()
attempt = 0
for episode in range(1_000):
    action = agent.act(observation, episode)
    old_observation = observation
    observation, reward, terminated, truncated, info = env.step(action)
    if reward == 0:
        reward -= 0.05
    if truncated or terminated:
        if observation != 15:
            reward -= 0.1

    agent.learn(old_observation, action, reward, observation)

    if terminated or truncated:
        attempt += 1
        if reward == 1:
            # FIXME: Add logic if continously reaching goal then stop and save the brain
            print("Success at, ", sep="", end="")
        print(f"game attempt {attempt}")
        observation, info = env.reset()

with open("brain.pkl", "wb") as f:
    pickle.dump(agent.q_table, f)
env.close()
