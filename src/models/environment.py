# Grid environment with random obstacle generation
import random
from collections import deque
from typing import TypeAlias

# Type alias for grid coordinates
State: TypeAlias = tuple[int, int]


class Environment:
    # Initialize grid with start/goal positions and randomly generated obstacles
    def __init__(
            self,
            grid_size: int,
            obstacle_probability: float,
            actions: list[int],
            action_to_delta: dict[int, State],
            step_reward: float,
            goal_reward: float,
            invalid_move_penalty: float
    ):
        self.actions: list[int] = actions
        self.action_to_delta: dict[int, State] = action_to_delta
        self.grid_size: int = grid_size
        self.obstacle_probability: float = obstacle_probability
        self.start: State = (0, grid_size - 1)
        self.goal: State = (grid_size - 1, 0)
        self.obstacles: set[State] = self._generate_obstacles()
        self.step_reward: float = step_reward
        self.goal_reward: float = goal_reward
        self.invalid_move_penalty: float = invalid_move_penalty

    # Generate random obstacles ensuring a path always exists from start to goal
    def _generate_obstacles(self) -> set[State]:
        max_attempts = 1000
        for _ in range(max_attempts):
            obstacles = set()
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    if (x, y) == self.start or (x, y) == self.goal:
                        continue
                    if random.random() < self.obstacle_probability:
                        obstacles.add((x, y))
            if self._exists_path(obstacles):
                return obstacles
        return set()

    # Use BFS to check if a valid path exists from start to goal
    def _exists_path(self, obstacles: set) -> bool:
        visited = {self.start}
        queue = deque([self.start])
        while queue:
            visiting_state = queue.popleft()
            for dx, dy in self.action_to_delta.values():
                next_x = visiting_state[0] + dx
                next_y = visiting_state[1] + dy
                if not (0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size):
                    continue
                if (next_x, next_y) in obstacles:
                    continue
                if (next_x, next_y) in visited:
                    continue
                if (next_x, next_y) == self.goal:
                    return True
                visited.add((next_x, next_y))
                queue.append((next_x, next_y))
        return False

    # Check if state is within bounds and not blocked by obstacle
    def state_is_valid(self, state: State) -> bool:
        if any(coord < 0 or coord >= self.grid_size for coord in state):
            return False
        if state in self.obstacles:
            return False
        return True
