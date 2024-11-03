import os
import pickle
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Hyperparameters:
    learning_rate: float = 0.9
    discount_factor: float = 0.9
    epsilon: float = 1.0
    decay_rate: float = 0.00001
    decay_type: str = "linear"


def setup_environment(render: bool = False) -> gym.Env:
    return gym.make(
        "FrozenLake-v1",
        map_name="8x8",
        is_slippery=False,
        render_mode="human" if render else None,
    )


def initialize_q_table(
    env: gym.Env, file_path: str, is_training: bool = True
) -> np.ndarray:
    assert isinstance(
        env.observation_space, gym.spaces.Discrete
    ), "Observation space must be Discrete."
    assert isinstance(
        env.action_space, gym.spaces.Discrete
    ), "Action space must be Discrete."

    if is_training:
        return np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open(file_path, "rb") as f:
            return pickle.load(f)


def choose_action(
    q: np.ndarray,
    state: int,
    env: gym.Env,
    epsilon: float,
    rng: np.random.Generator,
    is_training: bool = True,
) -> int:
    if is_training and rng.random() < epsilon:
        return int(env.action_space.sample())
    else:
        return int(np.argmax(q[state, :]))


def update_q_value(
    q: np.ndarray,
    state: int,
    action: int,
    reward: float,
    new_state: int,
    hp: Hyperparameters,
) -> float:
    return q[state, action] + hp.learning_rate * (
        reward + hp.discount_factor * np.max(q[new_state, :]) - q[state, action]
    )


def run_episode(
    env: gym.Env,
    q: np.ndarray,
    hp: Hyperparameters,
    rng: np.random.Generator,
    is_training: bool = True,
) -> float:
    state = env.reset()[0]
    terminated = truncated = False
    episode_reward = 0.0

    while not terminated and not truncated:
        action = choose_action(q, state, env, hp.epsilon, rng, is_training)
        new_state, reward, terminated, truncated, _ = env.step(action)
        reward = float(reward)

        if is_training:
            q[state, action] = update_q_value(q, state, action, reward, new_state, hp)

        state = new_state
        episode_reward += reward

    return episode_reward


def decay_epsilon(hp: Hyperparameters, episode: int) -> float:
    if hp.decay_type == "exponential":
        return hp.epsilon * np.exp(-hp.decay_rate * episode)
    elif hp.decay_type == "linear":
        return max(hp.epsilon - hp.decay_rate, 0)
    else:
        raise ValueError("Invalid decay_type. Choose either 'linear' or 'exponential'.")


def plot_results(
    rewards_per_episode: np.ndarray, graph_resolution: int, plot_file_path: str
) -> None:
    rolling_avg_rewards = np.convolve(
        rewards_per_episode, np.ones(graph_resolution) / graph_resolution, mode="valid"
    )

    plt.plot(rolling_avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel(f"Average Reward (Last {graph_resolution} Episodes)")
    plt.savefig(plot_file_path)
    plt.close()


def save_q_table(q: np.ndarray, file_path: str) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(q, f)


def run(
    episodes: int,
    hp: Hyperparameters,
    save_table: str,
    save_plot: str,
    is_training: bool = True,
    render: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    q_table_path = os.path.join(current_directory, save_table)
    plot_path = os.path.join(current_directory, save_plot)

    env = setup_environment(render)
    q = initialize_q_table(env, q_table_path, is_training)
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    graph_resolution = max(1, episodes // 100)

    for i in range(episodes):
        rewards_per_episode[i] = run_episode(env, q, hp, rng, is_training)
        hp.epsilon = decay_epsilon(hp, i)

        if (i + 1) % graph_resolution == 0 and verbose:
            print(
                f"Episode {i + 1}, Cumulative Successes: {np.sum(rewards_per_episode[:i + 1])}"
            )

    env.close()

    if is_training:
        plot_results(rewards_per_episode, graph_resolution, plot_path)
        save_q_table(q, q_table_path)

    return q, rewards_per_episode


if __name__ == "__main__":
    hp = Hyperparameters(
        learning_rate=0.9,
        discount_factor=0.9,
        epsilon=1.0,
        decay_rate=0.00001,
        decay_type="linear",
    )

    q, rewards = run(
        100000,
        hp,
        save_table="qLearning.pkl",
        save_plot="qLearning.png",
        is_training=True,
        render=False,
        verbose=True,
    )
    # For evaluation
    # q, rewards = run(1, hp, save_table="qLearning.pkl", save_plot="qLearning.png", is_training=False, render=True, verbose=False)
