"""
Solves Frozen lake puzzle for deterministic environment(non-slippery)
"""

import gymnasium as gym
import numpy as np
import pickle
import argparse
import matplotlib
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2)

class QAgent:
    def __init__(
        self, states: int = 16, actions: int = 4, lr: float = 0.6, gamma: float = 0.25
    ) -> None:
        self.possible_states = states
        self.possible_actions = actions
        self.q_table = np.zeros((states, actions))
        self.lr = lr
        self.gamma = gamma

    def act(self, state: int, attempt: int, with_noise: bool=True) -> int:
        noise = 0
        if with_noise:
            noise = np.random.randn(1, self.possible_actions) * (1 / (attempt + 1))
        return np.argmax(
            # choose your best wisdom
            self.q_table[state, :] + noise
        )

    def learn(self, state, action, reward, new_state):
        # Q(s, a) += lr *(future best reward - Current known value)
        self.q_table[state, action] += self.lr * (
            reward
            # Temporal Difference
            + self.gamma * np.max(self.q_table[new_state, :])
            - self.q_table[state, action]
        )

def run(is_slippery: bool, noise: bool, row: int, col: int) -> list:
    
    env = gym.make(
        "FrozenLake-v1",
        # render_mode="human",
        desc=None,
        map_name="4x4",
        is_slippery=is_slippery,
    )
    state, info = env.reset()

    # Make an agent
    agent = QAgent()
    # track episodes
    episode = 0
    # counter tracking how many times reached terminal 
    success_counter = 0
    rewards = []
    cummulative_reward_per_episode = 0
    for attempt in range(30_000):
        # Based on a state decide on a action
        action = agent.act(state, attempt, noise)
        old_state = state
        # Execute the action
        state, reward, terminated, truncated, info = env.step(action)
        # make reward
        if reward == 0:
            reward -= 0.05
        if truncated or terminated:
            if state != 15: # if not at success terminated state
                reward -= 0.1
        # update the q-table based on information gathered leveraging "Temporal Difference"
        cummulative_reward_per_episode += reward
        agent.learn(old_state, action, reward, state)

        # reset the environment if agent is no longer actionable
        if terminated or truncated:
            rewards.append(cummulative_reward_per_episode)    
            cummulative_reward_per_episode = 0 
            episode += 1
            if reward == 1:
                # print("Success at, ", sep="", end="")
                success_counter += 1
            else:
                success_counter = 0
            # print(f"game attempt {episode}")
            state, info = env.reset()

        if success_counter == 10:
            break
    print(episode, len(rewards))
    axs[i, j].plot(rewards)
    axs[i, j].set_title(f"slippery : {is_slippery} Noise : {noise}")

if __name__ == "__main__":
    for i, is_slippery in enumerate([True, False]):
        for j, noise in enumerate([True, False]):
            run(is_slippery, noise, i, j)
    plt.tight_layout()
    plt.savefig('combo_run.png')