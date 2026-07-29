# Q-learning agent with epsilon-greedy exploration
import random
import numpy as np

from src.environment import State


class Agent:
    # Initialize Q-learning agent with hyperparameters and empty Q-table
    def __init__(
        self,
        actions: list[int],
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
    ):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: dict[State, np.ndarray] = {}

    # Get or initialize Q-values for a state
    def _get_q(self, state: State) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions), dtype=np.float32)
        return self.q_table[state]

    # Select action using epsilon-greedy exploration
    def select_action(self, state: State) -> int:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q = self._get_q(state)
        return int(np.argmax(q))

    # Update Q-value using temporal difference learning
    def update_q_state_action(self, state: State, action: int, reward: float, next_state: State):
        max_q_next_state = np.max(self._get_q(next_state))
        td_target = reward + self.gamma * max_q_next_state
        td_error = td_target - self._get_q(state)[action]
        self._get_q(state)[action] += self.alpha * td_error

    # Reduce exploration rate after each episode
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
