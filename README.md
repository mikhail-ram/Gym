# Reinforcement Learning with Gymnasium: A Collection of Examples

This repository contains a collection of Python scripts demonstrating various reinforcement learning (RL) algorithms applied to different environments using the Gymnasium library.  The examples showcase both tabular methods (Q-learning, SARSA) and a deep learning approach (Deep Q-Network).

## Project Structure

The project is organized into subdirectories, each focusing on a specific environment and RL algorithm:

* **`RL/Gym/`**: The root directory containing all RL-related code.
    * **`validation.py`**: A simple script to test the Gymnasium library's functionality with the MsPacman environment.  This is a basic example showcasing environment interaction, not an RL algorithm implementation.
    * **`FrozenLake/`**: Contains implementations for the FrozenLake environment.
        * **`QLearning/qLearning.py`**: Implements Q-learning for the FrozenLake-v1 environment.
        * **`SARSA/SARSA.py`**: Implements SARSA (State-Action-Reward-State-Action) for the FrozenLake-v1 environment.
    * **`Taxi/`**: Contains implementations for the Taxi environment.
        * **`QLearning/qLearning.py`**: Implements Q-learning for the Taxi-v3 environment.
    * **`FlappyBird/`**: Contains code related to the FlappyBird environment (likely a custom environment).
        * **`DQN/dqn.py`**: Defines the neural network architecture (Deep Q-Network) for a FlappyBird agent using PyTorch.  This file only provides the network; a separate script would be needed to implement the complete DQN training loop.


## Dependencies

The project relies on several libraries:

* **Gymnasium:** The primary RL environment library.
* **ALE-Py:**  Used for interacting with Arcade Learning Environment (ALE) games (MsPacman).  Only used in `validation.py`.
* **NumPy:** For numerical computations.
* **Matplotlib:** For plotting results.
* **Pickle:** For saving and loading Q-tables.
* **PyTorch:** For the Deep Q-Network implementation in `dqn.py`.


To install the necessary dependencies, create a virtual environment and run:

```bash
pip install gymnasium ale-py numpy matplotlib pickle torch
```

## Usage

Each script within the subdirectories is largely self-contained.  They typically involve:

1. **Environment Setup:** Creating an instance of the specified Gymnasium environment.
2. **Agent Initialization:**  Initializing the Q-table (for Q-learning and SARSA) or a neural network (for DQN).
3. **Training Loop:** Running multiple episodes of the environment, updating the agent's policy (Q-table or network weights) based on the chosen RL algorithm.
4. **Evaluation:** (Optional) Evaluating the trained agent's performance on the environment.
5. **Saving Results:** Saving the trained agent's policy and/or plots of the learning curve.


You can run each script individually using Python:

```bash
python RL/Gym/validation.py
python RL/Gym/FrozenLake/QLearning/qLearning.py
# ... and so on
```

The `qLearning.py` and `SARSA.py` scripts within the `FrozenLake` and `Taxi` directories have command-line arguments to control the training process (e.g., number of episodes, learning rate, exploration rate). Refer to the individual files for details.  The `dqn.py` file requires additional code to be a complete, runnable DQN agent; it only defines the neural network architecture.


## Code Description (Detailed)

### `validation.py`

This file serves as a simple demonstration of the Gymnasium library's usage. It creates a MsPacman environment, performs a single episode with random actions, and closes the environment. It doesn't implement any learning algorithm; it is purely for testing and showcasing basic environment interaction.

### `FrozenLake/QLearning/qLearning.py` and `FrozenLake/SARSA/SARSA.py`

These files implement Q-learning and SARSA, respectively, for the FrozenLake-v1 environment. They share a similar structure, using functions for environment setup, Q-table initialization/updating, action selection, episode running, hyperparameter management, plotting, and saving/loading Q-tables.  The core difference lies in the `update_q_value` function, reflecting the distinct update rules of Q-learning and SARSA.

### `Taxi/QLearning/qLearning.py`

This file mirrors the structure of the FrozenLake Q-learning implementation, but uses the Taxi-v3 environment.  This showcases the reusability of the code structure across different environments.

### `FlappyBird/DQN/dqn.py`

This file defines a Deep Q-Network (DQN) architecture using PyTorch. It's a separate module defining the neural network but does *not* include the training loop or interaction with an environment.  It would need to be integrated with a separate script to create a complete Flappy Bird DQN agent.


## Future Improvements

* **Code Refactoring:**  The Q-learning implementations across different environments have high code redundancy.  Refactoring into a more generalized RL framework would improve maintainability and reusability.
* **Complete DQN Implementation:**  Creating a full DQN training script for the FlappyBird environment using the `dqn.py` network.
* **Hyperparameter Tuning:**  Implementing a systematic approach to hyperparameter tuning for improved performance.
* **Modular Environment Handling:** Creating a more flexible mechanism for switching between different environments without modifying core algorithm code.


This README provides a comprehensive overview of the project.  Refer to the individual Python files for detailed code comments and explanations.


---
Generated with ❤️ using [GitDocs](https://github.com/mikhail-ram/gitdocs).