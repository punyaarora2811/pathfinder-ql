# Training loop orchestration with episode management
import numpy as np

from src.environment import Environment, State
from src.agent import Agent


class Trainer:
    # Initialize trainer with environment, agent, and episode tracking
    def __init__(
            self,
            environment: Environment,
            agent: Agent,
            max_steps_per_episode: int,
            reward_avg_window: int
    ):
        self.env: Environment = environment
        self.agent: Agent = agent
        self.agent_state = self.env.start
        self.trajectory = [self.agent_state]
        self.episode_index: int = 0
        self.episode_return: float = 0.0
        self.episode_steps: int = 0
        self.returns_history: list[float] = []
        self.avg_returns: list[float] = []
        self.max_steps_per_episode: int = max_steps_per_episode
        self.reward_avg_window: int = reward_avg_window

    # Compute next state and reward given an action
    def _get_next_state_and_reward(self, action: int) -> tuple[State, float]:
        reward = self.env.step_reward
        dx, dy = self.env.action_to_delta[action]
        next_state = (self.agent_state[0] + dx, self.agent_state[1] + dy)
        if not self.env.state_is_valid(next_state):
            reward += self.env.invalid_move_penalty
            next_state = self.agent_state
        if next_state == self.env.goal:
            reward += self.env.goal_reward
        return next_state, reward

    # Execute one action and update agent state and Q-values
    def _take_action(self) -> tuple[State, float]:
        action = self.agent.select_action(self.agent_state)
        next_state, reward = self._get_next_state_and_reward(action)
        self.agent.update_q_state_action(self.agent_state, action, reward, next_state)
        self.agent_state = next_state
        return next_state, reward

    # Update episode statistics after each step
    def _update_episode_info(self, reward: float) -> None:
        self.episode_return += reward
        self.episode_steps += 1
        self.trajectory.append(self.agent_state)

    # Check episode termination and reset if needed
    def _check_end_episode(self, next_state: State) -> None:
        end_episode = next_state == self.env.goal or self.episode_steps >= self.max_steps_per_episode
        if end_episode:
            self.returns_history.append(self.episode_return)
            self.avg_returns.append(float(np.mean(self.returns_history[-self.reward_avg_window:])))
            self.agent.decay_epsilon()
            self.episode_index += 1
            self.episode_return = 0.0
            self.episode_steps = 0
            self.agent_state = self.env.start
            self.trajectory = [self.agent_state]

    # Execute one training step
    def take_one_step(self) -> None:
        next_state, reward = self._take_action()
        self._update_episode_info(reward)
        self._check_end_episode(next_state)
