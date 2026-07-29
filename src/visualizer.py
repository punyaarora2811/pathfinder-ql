# PyGame visualization for Q-learning training
import math
import sys
import numpy as np
import pygame

from src.environment import Environment, State
from src.agent import Agent
from src.trainer import Trainer

# Layout and color constants
MARGIN = 10
FONT_NAME = "Arial"
TITLE_H = 18
STATS_H = 16

WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
GRAY = (180, 180, 180)
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (90, 90, 90)
BLUE = (70, 130, 180)
GREEN = (60, 179, 113)
RED = (220, 20, 60)
ORANGE = (255, 140, 0)
PURPLE = (138, 43, 226)


class Visualizer:
    # Initialize PyGame window and visualization
    def __init__(self, env: Environment, agent: Agent, trainer: Trainer):
        pygame.init()
        self.env = env
        self.agent = agent
        self.trainer = trainer
        info = pygame.display.Info()
        screen_w = info.current_w
        screen_h = info.current_h
        self.CELL_SIZE = round((min(screen_w, screen_h) * 0.45) / self.env.grid_size)
        pygame.display.set_caption("Q-Learning")
        self.subplot_px = self.env.grid_size * self.CELL_SIZE
        self.title_h = TITLE_H
        self.stats_h = STATS_H
        self.width = MARGIN + self.subplot_px + MARGIN + self.subplot_px + MARGIN
        self.height = MARGIN + (self.title_h + self.stats_h) + self.subplot_px + MARGIN + self.title_h + self.subplot_px + MARGIN
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font_small = pygame.font.SysFont(FONT_NAME, 14)
        self.font = pygame.font.SysFont(FONT_NAME, 18)
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        self.steps_per_frame = 4

    # Main event loop for visualization and training
    def run(self, trainer):
        self.running = True
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_UP:
                        self.steps_per_frame = min(64, self.steps_per_frame * 2)
                    elif event.key == pygame.K_DOWN:
                        self.steps_per_frame = max(1, self.steps_per_frame // 2)
            if not self.paused:
                for _ in range(self.steps_per_frame):
                    trainer.take_one_step()
            self.draw(trainer.trajectory, trainer.episode_index, trainer.episode_return, trainer.avg_returns)
            self.clock.tick(60)
        pygame.quit()
        sys.exit(0)

    # Render four-panel dashboard with training info
    def draw(self, trajectory: list[State], episode: int, ep_return: float, avg_returns: list[float]):
        self.screen.fill(WHITE)
        tl = (MARGIN, MARGIN + self.title_h + self.stats_h)
        tr = (MARGIN + self.subplot_px + MARGIN, MARGIN + self.title_h + self.stats_h)
        bl = (MARGIN, tl[1] + self.subplot_px + MARGIN + self.title_h)
        br = (MARGIN + self.subplot_px + MARGIN, tl[1] + self.subplot_px + MARGIN + self.title_h)

        # Top-left: Environment with current episode trajectory
        self._draw_grid(tl)
        self._draw_obstacles(tl)
        self._draw_start_goal(tl)
        self._draw_trajectory(tl, trajectory)
        self._draw_agent(tl, self.trainer.agent_state)
        header = f"Ep: {episode}  γ: {self.agent.gamma:.2f}  α: {self.agent.alpha:.2f}  ε: {self.agent.epsilon:.3f}  Cumulative Episode Return: {ep_return:.1f}"
        self._label(header, (tl[0] + 8, tl[1] - (self.title_h + self.stats_h) - 7))
        self._label("Environment + Trajectory", (tl[0] + 8, tl[1] - self.stats_h - 7))

        # Top-right: Learned greedy policy with arrows
        self._draw_grid(tr)
        self._draw_obstacles(tr)
        self._draw_start_goal(tr)
        self._draw_policy(tr)
        self._label("Current Policy (Greedy)", (tr[0] + 8, tr[1] - self.stats_h - 7))

        # Bottom-left: Training progress chart
        self._draw_chart_bg(bl)
        self._draw_avg_return(bl, avg_returns)
        self._label(f"Average Return (last {self.trainer.reward_avg_window} episodes)",(bl[0] + 8, bl[1] - self.title_h - 5))

        # Bottom-right: Q-value heatmap
        self._draw_max_q_heatmap(br)
        self._label("Max Q-value per State", (br[0] + 8, br[1] - self.title_h - 5))
        pygame.display.flip()

    # Text rendering helper
    def _label(self, text: str, pos: State, color: tuple[int, int, int] = BLACK):
        surf = self.font.render(text, True, color)
        self.screen.blit(surf, pos)

    # Draw grid background and lines
    def _draw_grid(self, origin: State):
        ox, oy = origin
        s = self.subplot_px
        pygame.draw.rect(self.screen, LIGHT_GRAY, pygame.Rect(ox, oy, s, s))
        for x in range(self.env.grid_size + 1):
            pygame.draw.line(self.screen, GRAY, (ox + x * self.CELL_SIZE, oy), (ox + x * self.CELL_SIZE, oy + s), 1)
        for y in range(self.env.grid_size + 1):
            pygame.draw.line(self.screen, GRAY, (ox, oy + y * self.CELL_SIZE), (ox + s, oy + y * self.CELL_SIZE), 1)
        pygame.draw.rect(self.screen, DARK_GRAY, pygame.Rect(ox, oy, s, s), 2)

    # Get pixel rectangle for a grid cell
    def _cell_rect(self, origin: State, st: State) -> pygame.Rect:
        ox, oy = origin
        return pygame.Rect(ox + st[0] * self.CELL_SIZE, oy + st[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)

    # Draw obstacle blocks
    def _draw_obstacles(self, origin: State):
        for x, y in self.env.obstacles:
            pygame.draw.rect(self.screen, DARK_GRAY, self._cell_rect(origin, (x, y)))

    # Draw start (S) and goal (G) positions
    def _draw_start_goal(self, origin: State):
        r = self._cell_rect(origin, self.env.start)
        pygame.draw.rect(self.screen, BLUE, r)
        self.screen.blit(self.font_small.render("S", True, WHITE), (r.x + 4, r.y + 4))
        r = self._cell_rect(origin, self.env.goal)
        pygame.draw.rect(self.screen, GREEN, r)
        self.screen.blit(self.font_small.render("G", True, BLACK), (r.x + 4, r.y + 4))

    # Draw agent at current position
    def _draw_agent(self, origin: State, state: State):
        r = self._cell_rect(origin, state)
        pad = 8
        pygame.draw.rect(self.screen, ORANGE, pygame.Rect(r.x + pad, r.y + pad, r.w - 2 * pad, r.h - 2 * pad), border_radius=6)

    # Draw episode trajectory as line
    def _draw_trajectory(self, origin: State, traj: list[State]):
        if len(traj) < 2:
            return
        ox, oy = origin
        points: list[State] = []
        for s in traj:
            cx = ox + s[0] * self.CELL_SIZE + self.CELL_SIZE // 2
            cy = oy + s[1] * self.CELL_SIZE + self.CELL_SIZE // 2
            points.append((cx, cy))
        pygame.draw.lines(self.screen, PURPLE, False, points, 3)

    # Draw policy arrows showing best action per state
    def _draw_policy(self, origin: State):
        for y in range(self.env.grid_size):
            for x in range(self.env.grid_size):
                if (x, y) in self.env.obstacles:
                    continue
                q = self.agent.q_table.get((x, y))
                if q is None or np.allclose(q, 0.0):
                    continue
                greedy = int(np.argmax(q))
                self._draw_arrow(self._cell_center(origin, (x, y)), greedy)

    # Get cell center coordinates
    def _cell_center(self, origin: State, st: State) -> State:
        r = self._cell_rect(origin, st)
        return r.x + r.w // 2, r.y + r.h // 2

    # Draw directional arrow for action (0=up, 1=right, 2=down, 3=left)
    def _draw_arrow(self, center: State, action: int, color: tuple[int, int, int] = RED):
        cx, cy = center
        size = self.CELL_SIZE // 3
        if action == 0:
            pygame.draw.line(self.screen, color, (cx, cy + size), (cx, cy - size), 3)
            pygame.draw.polygon(self.screen, color,[(cx, cy - size - 2), (cx - 6, cy - size + 10), (cx + 6, cy - size + 10)])
        elif action == 1:
            pygame.draw.line(self.screen, color, (cx - size, cy), (cx + size, cy), 3)
            pygame.draw.polygon(self.screen, color, [(cx + size + 2, cy), (cx + size - 10, cy - 6), (cx + size - 10, cy + 6)])
        elif action == 2:
            pygame.draw.line(self.screen, color, (cx, cy - size), (cx, cy + size), 3)
            pygame.draw.polygon(self.screen, color, [(cx, cy + size + 2), (cx - 6, cy + size - 10), (cx + 6, cy + size - 10)])
        else:
            pygame.draw.line(self.screen, color, (cx + size, cy), (cx - size, cy), 3)
            pygame.draw.polygon(self.screen, color, [(cx - size - 2, cy), (cx - size + 10, cy - 6), (cx - size + 10, cy + 6)])

    # Draw chart background and border
    def _draw_chart_bg(self, origin: State):
        ox, oy = origin
        s = self.subplot_px
        pygame.draw.rect(self.screen, LIGHT_GRAY, pygame.Rect(ox, oy, s, s))
        pygame.draw.rect(self.screen, DARK_GRAY, pygame.Rect(ox, oy, s, s), 2)

    # Draw line chart of moving average returns
    def _draw_avg_return(self, origin: State, avg_returns: list[float]):
        ox, oy = origin
        s = self.subplot_px
        pad = 40
        plot = pygame.Rect(ox + pad, oy + pad, s - 2 * pad, s - 2 * pad)
        pygame.draw.rect(self.screen, WHITE, plot)
        pygame.draw.rect(self.screen, GRAY, plot, 1)
        if len(avg_returns) < 2:
            return
        ys = avg_returns
        xs = list(range(len(ys)))
        y_min, y_max = min(ys), max(ys)
        if math.isclose(y_min, y_max):
            y_max = y_min + 1.0
        self.screen.blit(self.font_small.render(f"{y_min:.1f}", True, BLACK),(plot.x - 34, plot.bottom - 8))
        self.screen.blit(self.font_small.render(f"{y_max:.1f}", True, BLACK), (plot.x - 34, plot.y - 8))
        pts: list[State] = []
        for i, y in enumerate(ys):
            tx = plot.x + int(i / (len(xs) - 1) * (plot.w - 1))
            ty = plot.bottom - int((y - y_min) / (y_max - y_min) * (plot.h - 1))
            pts.append((tx, ty))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, ORANGE, False, pts, 2)

    # Draw Q-value heatmap with color gradient
    def _draw_max_q_heatmap(self, origin: State):
        ox, oy = origin
        s = self.subplot_px
        pygame.draw.rect(self.screen, LIGHT_GRAY, pygame.Rect(ox, oy, s, s))
        values: list[float] = []
        max_q_grid: dict[State, float] = {}
        for y in range(self.env.grid_size):
            for x in range(self.env.grid_size):
                if (x, y) in self.env.obstacles:
                    continue
                q = self.agent.q_table.get((x, y))
                v = float(np.max(q)) if q is not None and q.size > 0 else 0.0
                max_q_grid[(x, y)] = v
                values.append(v)
        if len(values) == 0:
            values = [0.0]
        v_min, v_max = min(values), max(values)
        if math.isclose(v_min, v_max):
            v_max = v_min + 1.0
        for y in range(self.env.grid_size):
            for x in range(self.env.grid_size):
                rect = self._cell_rect(origin, (x, y))
                if (x, y) in self.env.obstacles:
                    pygame.draw.rect(self.screen, DARK_GRAY, rect)
                    continue
                v = max_q_grid.get((x, y), 0.0)
                color = Visualizer._colormap_three(v, v_min, v_max)
                pygame.draw.rect(self.screen, color, rect)
                self.screen.blit(self.font_small.render(f"{v:.1f}", True, BLACK), (rect.x + 4, rect.y + 4))
        for x in range(self.env.grid_size + 1):
            pygame.draw.line(self.screen, GRAY, (ox + x * self.CELL_SIZE, oy), (ox + x * self.CELL_SIZE, oy + s), 1)
        for y in range(self.env.grid_size + 1):
            pygame.draw.line(self.screen, GRAY, (ox, oy + y * self.CELL_SIZE), (ox + s, oy + y * self.CELL_SIZE), 1)
        pygame.draw.rect(self.screen, DARK_GRAY, pygame.Rect(ox, oy, s, s), 2)

    # Linear color interpolation between two RGB colors
    @staticmethod
    def _lerp_color(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
        t = max(0.0, min(1.0, t))
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t)
        )

    # Three-color gradient: red → yellow → green
    @staticmethod
    def _colormap_three(v: float, v_min: float, v_max: float) -> tuple[int, int, int]:
        if v_max <= v_min:
            return 200, 200, 200
        t = (v - v_min) / (v_max - v_min)
        t = max(0.0, min(1.0, t))
        mid = 0.5
        if t <= mid:
            tt = t / mid
            return Visualizer._lerp_color((200, 30, 30), (255, 220, 60), tt)
        else:
            tt = (t - mid) / (1.0 - mid)
            return Visualizer._lerp_color((255, 220, 60), (0, 160, 0), tt)
