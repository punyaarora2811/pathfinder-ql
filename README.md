<div align="center">
  <h1>🤖 Pathfinder QL</h1>
  <p><strong>A from-scratch implementation of Q-learning for navigating randomly generated 2D grid environments.</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.8--3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
  [![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](#)
  [![PyGame](https://img.shields.io/badge/PyGame-2.6.1-green?style=for-the-badge&logo=python&logoColor=white)](#)
  [![PyYAML](https://img.shields.io/badge/PyYAML-black?style=for-the-badge)](#)
</div>

<br />

## 🌟 Overview

Pathfinder QL is a custom-built, lightweight Reinforcement Learning environment. An agent learns to reach a target destination while avoiding randomly generated obstacles. This project uses pure Q-learning (tabular reinforcement learning) without relying on external RL frameworks.

![Pathfinder QL Demo](media/git.gif)

## ✨ Features

- **Pure Q-Learning Implementation:** Custom, from-scratch tabular Q-learning algorithm.
- **Dynamic Environments:** Randomly generated obstacle grids with a BFS-backed guarantee that a valid path to the goal always exists.
- **Real-Time Visualization:** A robust PyGame dashboard featuring a 4-panel display:
  - Live episode trajectory
  - The current learned greedy policy
  - A state-action max Q-value heatmap
  - A moving average of episode returns
- **Highly Configurable:** Easily tweak hyperparameters, rewards, and grid properties using a simple YAML configuration file.

## 📂 Project Structure

```text
Pathfinder_QL/
├── media/
│   └── git.gif                   # Demonstration GIF
├── src/
│   ├── agent.py                  # Q-learning agent logic
│   ├── config.py                 # Configuration loader and parser
│   ├── environment.py            # Grid generation and state management
│   ├── trainer.py                # Training loop orchestration
│   └── visualizer.py             # PyGame UI and 4-panel dashboard
├── .gitignore
├── config.yml                    # Hyperparameter configuration
├── LICENSE
├── main.py                       # Application entry point
├── README.md
└── requirements.txt              # Python dependencies
```

## 🚀 Getting Started

Follow these instructions to set up and run the Q-Learning agent on your local machine.

### Prerequisites

- **Python 3.8 – 3.12** — pygame does not yet ship pre-built wheels for Python 3.13+. Python 3.11 is recommended.
  - Download: https://www.python.org/downloads/release/python-3119/

### Installation

1. Clone the repository:
```bash
git clone https://github.com/punyaarora2811/pathfinder-ql.git
cd Pathfinder_QL
```

2. Create and activate a virtual environment:

**On Windows (recommended — using Python 3.11 explicitly):**
```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```powershell
python -m pip install -r requirements.txt
```

### Usage

Run the training loop and launch the visualization:

**Windows:**
```powershell
.\venv\Scripts\python.exe main.py
```

**macOS/Linux:**
```bash
python main.py
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume training |
| `↑` Up Arrow | Increase simulation speed |
| `↓` Down Arrow | Decrease simulation speed |
| Close window | Quit |

## 🧠 How It Works

1. **Initialization**: A grid is generated with random obstacles (verified by BFS to ensure solvability), and a Q-table is initialized to zero.
2. **Action Selection**: The agent selects actions using an $\epsilon$-greedy strategy.
3. **Learning Step**: After each move, the Q-table is updated via the Temporal Difference (TD) target, utilizing the maximum Q-value of the subsequent state.
4. **Episode Management**: Episodes terminate when the agent reaches the goal or exhausts its step limit.
5. **Real-time Feedback**: The visualizer dynamically updates to display the agent's real-time exploration, exploitation, and learning progress.

## ⚙️ Configuration

You can fully customize the learning parameters by editing the `config.yml` file.

```yaml
environment:
  grid_size: 20
  obstacle_probability: 0.3
  actions: [0, 1, 2, 3] # 0: Up, 1: Right, 2: Down, 3: Left
  action_to_delta: { 0: [0, -1], 1: [1, 0], 2: [0, 1], 3: [-1, 0] }
  step_reward: -5.0
  goal_reward: 500.0
  invalid_move_penalty: -10.0

agent:
  alpha: 0.2
  gamma: 0.995
  epsilon: 1.0
  epsilon_min: 0.002
  epsilon_decay: 0.995

trainer:
  max_steps_per_episode: 200
  reward_avg_window: 100
```

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
