# Load and parse configuration from config.yml file
import yaml
from dataclasses import dataclass
from src.environment import State

# Environment configuration dataclass
@dataclass
class EnvironmentConfig:
    actions: list[int]
    action_to_delta: dict[int, State]
    grid_size: int
    obstacle_probability: float
    step_reward: float
    goal_reward: float
    invalid_move_penalty: float

# Agent configuration dataclass
@dataclass
class AgentConfig:
    alpha: float
    gamma: float
    epsilon: float
    epsilon_min: float
    epsilon_decay: float

# Trainer configuration dataclass
@dataclass
class TrainerConfig:
    max_steps_per_episode: int
    reward_avg_window: int

# Root configuration container
@dataclass
class Config:
    environment: EnvironmentConfig
    agent: AgentConfig
    trainer: TrainerConfig

# Load configuration from YAML file and parse into Config object
def load_config(path="config.yml") -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(
        environment=EnvironmentConfig(**data["environment"]),
        agent=AgentConfig(**data["agent"]),
        trainer=TrainerConfig(**data["trainer"])
    )