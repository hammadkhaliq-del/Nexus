"""
NEXUS VISUALIZER (Dark Mode / Cyberpunk Theme)
High-fidelity rendering with alpha blending, glow effects, and modern UI.
"""
import pygame
import sys
from pathlib import Path

# Fix imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.city import City
from core.simulation import Simulation

# --- THEME CONFIGURATION ---
TILE_SIZE = 32
FPS = 60  # Smoother framerate

# Cyberpunk / Dark UI Palette
THEME = {
    "BACKGROUND": (10, 10, 14),      # Almost Black
    "GRID_LINES": (25, 25, 30),      # Very faint grey
    
    # Tiles
    "ROAD": (30, 32, 38),            # Dark Slate
    "BUILDING": (15, 15, 20),        # Darker Block
    "BUILDING_RIM": (60, 60, 80),    # Slight highlight
    "GRASS": (20, 40, 30),           # Dark Forest Green
    "TRAFFIC_JAM": (200, 40, 40),    # Neon Red Warning
    
    # UI
    "UI_BG": (20, 20, 25, 230),      # Semi-transparent dark grey
    "UI_BORDER": (100, 100, 100),
    "TEXT_MAIN": (220, 220, 220),
    "TEXT_ACCENT": (0, 255, 200),    # Neon Cyan
    
    # Agents (Neon Colors)
    "AGENTS": [
        (0, 255, 255),  # Cyan
        (255, 0, 128),  # Magenta
        (255, 200, 0),  # Amber
        (50, 255, 50),  # Lime
    ]
}

class Visualizer:
    def __init__(self, simulation: Simulation):
        self.sim = simulation
        self.width = simulation.city.width * TILE_SIZE
        self.height = simulation.city.height * TILE_SIZE
        self.ui_height = 80
        
        pygame.init()
        # Enable alpha channel for transparency
        self.screen = pygame.display.set_mode((self.width, self.height + self.ui_height))
        pygame.display.set_caption("NEXUS // CORE ENGINE")
        
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_small = pygame.font.SysFont("Consolas", 12)
        self.font_main = pygame.font.SysFont("Verdana", 14)
        self.font_header = pygame.font.SysFont("Impact", 24)
        
        self.finished = False
        self.paused = False

    def draw_map(self):
        """Draws the grid with a modern, flat look."""
        self.screen.fill(THEME["BACKGROUND"])
        
        for r in range(self.sim.city.height):
            for c in range(self.sim.city.width):
                val = self.sim.city.get_value(r, c)
                
                # Determine Base Color
                if val == 0: color = THEME["ROAD"]
                elif val == 1: color = THEME["BUILDING"]
                elif val == 2: color = THEME["GRASS"]
                else: color = (0,0,0)
                
                # Dynamic Events (Traffic)
                if not self.sim.city.is_walkable(r, c) and val == 0:
                    color = THEME["TRAFFIC_JAM"]

                # Draw Tile
                rect = (c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                
                # Add "3D" Rim to buildings
                if val == 1:
                    pygame.draw.rect(self.screen, THEME["BUILDING_RIM"], rect, 1)
                else:
                    # Subtle grid lines for roads
                    pygame.draw.rect(self.screen, THEME["GRID_LINES"], rect, 1)

    def draw_agents(self):
        """Draws agents with glow effects."""
        for i, agent in enumerate(self.sim.agents):
            base_color = THEME["AGENTS"][i % len(THEME["AGENTS"])]
            
            # Coordinates
            px = int(agent.position[1] * TILE_SIZE + TILE_SIZE / 2)
            py = int(agent.position[0] * TILE_SIZE + TILE_SIZE / 2)
            
            # 1. Draw Path (Thin line)
            if len(agent.path) > agent.path_index:
                points = [(px, py)]
                for wp in agent.path[agent.path_index:]:
                    wx = int(wp[1] * TILE_SIZE + TILE_SIZE / 2)
                    wy = int(wp[0] * TILE_SIZE + TILE_SIZE / 2)
                    points.append((wx, wy))
                if len(points) > 1:
                    # Draw faintly
                    pygame.draw.lines(self.screen, (*base_color, 100), False, points, 1)

            # 2. Draw Glow (Sensor Range)
            # We use a separate surface to support Alpha transparency
            radius = agent.sensor_range * TILE_SIZE
            glow_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            # Gradient-like circle (faint fill)
            pygame.draw.circle(glow_surf, (*base_color, 15), (radius, radius), radius)
            # Inner rim
            pygame.draw.circle(glow_surf, (*base_color, 40), (radius, radius), radius - 5, 1)
            self.screen.blit(glow_surf, (px - radius, py - radius))

            # 3. Draw Agent Core
            pygame.draw.circle(self.screen, (0,0,0), (px, py), 6) # Outline
            pygame.draw.circle(self.screen, base_color, (px, py), 4) # Dot

            # 4. Agent Label
            label = self.font_small.render(agent.name.upper(), True, (255,255,255))
            self.screen.blit(label, (px + 8, py - 8))

    def draw_ui(self):
        """Draws the HUD."""
        # Create Glass Panel
        ui_surf = pygame.Surface((self.width, self.ui_height), pygame.SRCALPHA)
        ui_surf.fill(THEME["UI_BG"])
        self.screen.blit(ui_surf, (0, self.height))
        
        # Draw Top Border
        pygame.draw.line(self.screen, THEME["UI_BORDER"], (0, self.height), (self.width, self.height), 2)

        # Status Logic
        if self.finished:
            status_txt = "SYSTEM OFFLINE"
            status_col = (255, 50, 50)
        elif self.paused:
            status_txt = "SYSTEM PAUSED"
            status_col = (255, 200, 0)
        else:
            status_txt = "SYSTEM ACTIVE"
            status_col = THEME["TEXT_ACCENT"]

        # Render Text
        self.screen.blit(self.font_header.render(status_txt, True, status_col), (20, self.height + 10))
        
        # Stats
        stats = self.sim.get_statistics()
        s1 = f"STEP: {self.sim.current_step}   AGENTS: {len(self.sim.agents)}"
        s2 = f"EVENTS: {stats['events_triggered']}   DIST: {stats['total_distance_traveled']:.1f}"
        
        self.screen.blit(self.font_main.render(s1, True, THEME["TEXT_MAIN"]), (20, self.height + 45))
        self.screen.blit(self.font_main.render(s2, True, THEME["TEXT_MAIN"]), (250, self.height + 45))
        
        # Controls
        ctrl = self.font_small.render("[SPACE] PAUSE/RESUME", True, (100, 100, 100))
        self.screen.blit(ctrl, (self.width - 150, self.height + 50))

    def run(self):
        self.sim.start()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.finished:
                        self.paused = not self.paused

            if not self.paused and not self.finished:
                keep_going = self.sim.step()
                if not keep_going:
                    self.finished = True
                    print("Visualizer: Simulation signaled stop.")

            self.draw_map()
            self.draw_agents()
            self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(FPS)

if __name__ == "__main__":
    print("Please run 'run_visuals.py' instead.")