# Import core components
from src.models.environment import Environment
from src.models.agent import Agent
from src.services.trainer import Trainer
from src.services.visualizer import Visualizer
from src.core.config import load_config


def main():
    # Load configuration from config.yml
    config = load_config()
    
    # Initialize the grid environment with obstacle generation
    env = Environment(**config.environment.__dict__)
    
    # Initialize the Q-learning agent
    agent = Agent(actions=config.environment.actions, **config.agent.__dict__)
    
    # Initialize the trainer with the environment and agent
    trainer = Trainer(environment=env, agent=agent, **config.trainer.__dict__)

    # Set up the PyGame visualizer and print control instructions
    vis = Visualizer(env, agent, trainer)
    print("Controls: SPACE pause/resume, UP increase speed, DOWN decrease speed")
    
    # Start training with real-time visualization
    vis.run(trainer)


if __name__ == "__main__":
    main()