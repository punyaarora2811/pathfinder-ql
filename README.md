<div align="center">
  <h1>🤖 Pathfinder_QL</h1>
  <p><strong>An AI Q-Learning Agent and Environment Visualizer</strong></p>
  
  [![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/punyaarora2811/pathfinder-ql)
  [![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
  [![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
</div>

<br>

**Pathfinder_QL** is a completely free, local-first reinforcement learning visualizer built with Python and PyGame. It trains an autonomous agent to navigate dynamic environments using Q-Learning, avoiding obstacles and finding optimal paths securely on your local machine.

---

## 🚀 Key Features

- **Reinforcement Learning:** Custom NumPy Q-Learning implementation running locally on your CPU.
- **Dynamic Environments:** Configurable grid layouts with obstacles for the agent to navigate.
- **Real-Time Visualization:** Built with PyGame for robust 2D rendering and live tracking of the agent's training.
- **Modern Tech Stack:** 
  - **Backend Core:** Pure Python (Lightning-fast and dependency-light)
  - **Frontend / Visualizer:** PyGame (Robust 2D rendering)

---

## 💻 Local Setup Instructions

**Prerequisites:**
- **Python** (v3.8 - v3.11)

### 1. Clone the repository
```bash
git clone https://github.com/punyaarora2811/pathfinder-ql.git
cd pathfinder-ql
```

### 2. Create a virtual environment
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies
Ensure you have the required dependencies listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the example environment file:
```bash
# On Linux/macOS
cp .env.example .env
# On Windows
copy .env.example .env
```
*(No API keys needed! This project is 100% local.)*

### 5. Run the Application
Use the provided runner scripts to launch the PyGame visualizer.

**On Windows:**
```cmd
run.bat
```

**On macOS/Linux:**
```bash
chmod +x run.sh
./run.sh
```

---

## 🧪 End-to-End Testing

To verify the core logic and models:
```bash
pytest tests/
```

---

## 🎥 Demo

![Pathfinder Q-Learning Demo](media/git.gif)

1. **Initialize the app:** Run `run.bat` or `run.sh` to initialize the visualizer.
2. **Watch the agent:** Watch the agent start exploring randomly, discover the goal, and quickly optimize its path using the Q-Table.
3. **Controls:** Use `SPACE` to pause/resume, `UP` to increase speed, and `DOWN` to decrease speed during training.
