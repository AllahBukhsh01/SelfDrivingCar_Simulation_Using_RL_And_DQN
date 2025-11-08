# car_env.py
import math
import numpy as np
import pygame
import random
import os

SCREEN_W, SCREEN_H = 1000, 600

class CarEnv:
    def __init__(self, render_mode=True, track_width=280):
        self.render_mode = render_mode
        pygame.init()
        if self.render_mode:
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("TrackMania-Lite (RL)")
        self.clock = pygame.time.Clock()

        # Track geometry
        self.track_margin = 80
        self.track_width = track_width
        self.finish_x = SCREEN_W - self.track_margin - 40

        # Checkpoints
        self.checkpoints = [
            int(self.track_margin + 0.25 * (SCREEN_W - 2*self.track_margin)),
            int(self.track_margin + 0.50 * (SCREEN_W - 2*self.track_margin)),
            int(self.track_margin + 0.75 * (SCREEN_W - 2*self.track_margin)),
        ]
        self.passed_checkpoints = set()

        # Opponent count must be defined BEFORE reset()
        self.num_opponents = 3

        # Try loading track background
        self.assets_dir = os.path.join(os.getcwd(), "assets")
        self.track_image = None
        possible = os.path.join(self.assets_dir, "track.png")
        if os.path.exists(possible):
            try:
                img = pygame.image.load(possible)
                self.track_image = pygame.transform.scale(img, (SCREEN_W, SCREEN_H))
                print("[CarEnv] Loaded track image:", possible)
            except Exception as e:
                print("[CarEnv] Failed to load track image:", e)
                self.track_image = None
        else:
            self.track_image = None

        # ✅ FIXED: define num_opponents before reset()
        self.reset()

        # Initialize opponents
        self.opponents = []
        self._init_opponents()
        

        # Car sprite optional
        self.car_image = None
        car_sprite_path = os.path.join(self.assets_dir, "car_top.png")
        if os.path.exists(car_sprite_path):
            try:
                img = pygame.image.load(car_sprite_path).convert_alpha()
                self.car_image = pygame.transform.scale(img, (44, 28))
                print("[CarEnv] Loaded car sprite:", car_sprite_path)
            except Exception as e:
                print("[CarEnv] Failed to load car sprite:", e)
                self.car_image = None

        # Action / observation specs
        self.action_space = 5  # 0=left,1=straight,2=right,3=accelerate,4=brake
        self.observation_dim = 12  # includes 5 sensor readings now

    def _init_opponents(self):
        self.opponents = []
        lanes_y = [SCREEN_H//2 - 80, SCREEN_H//2, SCREEN_H//2 + 80]
        for i in range(self.num_opponents):
            x = random.randint(self.track_margin + 50, SCREEN_W - self.track_margin - 50)
            y = lanes_y[i % len(lanes_y)]
            # speed = random.uniform(1.8, 3.0)
            # direction = random.choice([-1, -1])  # moving leftwards (to mimic traffic)
            speed = random.uniform(0.8, 1.8)
            direction = -1  # all opponents move left slowly
            self.opponents.append({"pos": np.array([x, y], dtype=np.float64), "speed": speed, "dir": direction})

    def _sense_environment(self):
        """
        Cast 5 ray sensors around the car (front, front-left, front-right, left, right)
        Returns normalized distances [0..1], where 1 = clear path, 0 = immediate obstacle.
        """
        sensor_angles = [0, -30, 30, -60, 60]  # relative to car's facing direction
        max_range = 150.0
        readings = []

        for rel_ang in sensor_angles:
            ang = math.radians(self.car_angle + rel_ang)
            for dist in np.linspace(0, max_range, num=15):
                x = int(self.car_pos[0] + dist * math.cos(ang))
                y = int(self.car_pos[1] - dist * math.sin(ang))
                # Check if ray goes off-road or hits opponent
                if not self._point_on_track(x, y) or self._point_hits_opponent(x, y):
                    readings.append(dist / max_range)
                    break
            else:
                readings.append(1.0)
        return np.array(readings, dtype=np.float32)

    def _point_on_track(self, x, y):
        """Helper for sensors: returns False if point off road."""
        road_top = SCREEN_H // 2 - self.track_width // 2
        road_bottom = SCREEN_H // 2 + self.track_width // 2
        return (self.track_margin <= x <= SCREEN_W - self.track_margin) and (road_top <= y <= road_bottom)

    def _point_hits_opponent(self, x, y):
        """Helper for sensors: check if ray hits opponent car."""
        for o in self.opponents:
            ox, oy = o["pos"]
            if abs(x - ox) < 18 and abs(y - oy) < 14:
                return True
        return False

    def reset(self):
        # Agent starts near left of screen, center lane
        self.car_pos = np.array([self.track_margin + 30.0, SCREEN_H // 2], dtype=np.float64)
        self.car_angle = 0.0  # degrees, 0 = pointing right
        self.car_speed = 0.0
        self.max_speed = 8.0
        self.accel = 0.45
        self.brake = 0.4
        self.turn_rate = 3.0
        self.friction = 0.015

        # progress & bookkeeping
        self.done = False
        self.total_reward = 0.0
        self.steps = 0
        self._prev_dist = None
        self.passed_checkpoints = set()
        self._prev_dist = float(self.finish_x - self.car_pos[0])

        # init opponents
        self._init_opponents()

        return self._get_obs()

    def _get_obs(self):
        # nearest opponent relative position
        closest = np.array([0.0, 0.0], dtype=np.float64)
        min_dist = 1e9
        for o in self.opponents:
            dxdy = o["pos"] - self.car_pos
            d = np.linalg.norm(dxdy)
            if d < min_dist:
                min_dist = d
                closest = dxdy

        dist_to_goal = float(self.finish_x - self.car_pos[0])
        sensor_readings = self._sense_environment()  # <--- 5 normalized values

        obs = np.array([
            float(self.car_speed),
            float(self.car_angle),
            float(math.cos(math.radians(self.car_angle))),
            float(math.sin(math.radians(self.car_angle))),
            float(dist_to_goal),
            float(closest[0]),
            float(closest[1]),
            *sensor_readings
        ], dtype=np.float32)

        return obs

    def step(self, action: int):
        """
        action: 0=left,1=straight,2=right,3=accelerate,4=brake
        returns: obs, reward, done, info
        """
        if self.done:
            return self._get_obs(), 0.0, True, {}

        # Controls
        if action == 0:
            self.car_angle += self.turn_rate
        elif action == 2:
            self.car_angle -= self.turn_rate
        elif action == 3:
            self.car_speed = min(self.car_speed + self.accel, self.max_speed)
        elif action == 4:
            self.car_speed = max(self.car_speed - self.brake, 0.0)

        # friction
        self.car_speed = max(0.0, self.car_speed - self.friction)

        # movement
        dx = self.car_speed * math.cos(math.radians(self.car_angle))
        dy = -self.car_speed * math.sin(math.radians(self.car_angle))
        self.car_pos = self.car_pos + np.array([dx, dy], dtype=np.float64)

        # define road boundaries early (used by both opponent wrapping and offroad check)
        road_top = SCREEN_H//2 - self.track_width//2
        road_bottom = SCREEN_H//2 + self.track_width//2

        # clamp inside screen to avoid numeric blowup
        self.car_pos[0] = max(0.0, min(self.car_pos[0], SCREEN_W))
        self.car_pos[1] = max(0.0, min(self.car_pos[1], SCREEN_H))
        
        # === Opponent movement ===
        for o in self.opponents:
            o["pos"][0] += o["dir"] * o["speed"]  # move horizontally
            left_bound = self.track_margin + 20
            right_bound = SCREEN_W - self.track_margin - 20
            # wrap around track edges
            if o["pos"][0] < left_bound:
                o["pos"][0] = right_bound
            elif o["pos"][0] > right_bound:
                o["pos"][0] = left_bound
            # keep inside road vertically
            o["pos"][1] = max(road_top + 12, min(o["pos"][1], road_bottom - 12))
            
            
        # Reward shaping
        reward = 0.0
        # --- Reward Shaping with Sensors ---
        dist_to_goal = self.finish_x - self.car_pos[0]
        progress = (self._prev_dist - dist_to_goal)
        reward += progress * 0.8  # reward for moving forward

        self._prev_dist = dist_to_goal

        # checkpoint bonuses
        for idx, cp_x in enumerate(self.checkpoints):
            if (self.car_pos[0] >= cp_x) and (idx not in self.passed_checkpoints):
                self.passed_checkpoints.add(idx)
                reward += 25.0

        # sensor awareness reward
        sensors = self._sense_environment()
        min_sensor = np.min(sensors)
        if min_sensor < 0.15:
            reward -= 10.0  # very close to wall or car
        else:
            reward += np.mean(sensors) * 2.0  # encourage open distance

        # offroad check
        offroad = not self._is_on_track()
        if offroad:
            reward -= 25.0
            self.done = True
            print("[CarEnv] Off-road detected (sensor).")

        # collisions
        car_rect = self._get_car_rect()
        collision = any(car_rect.colliderect(pygame.Rect(int(o["pos"][0]-16), int(o["pos"][1]-12), 32, 24)) for o in self.opponents)
        if collision:
            reward -= 35.0
            self.done = True
            print("[CarEnv] Collision detected (sensor).")

        # finish condition
        if (self.car_pos[0] >= self.finish_x) and (len(self.passed_checkpoints) == len(self.checkpoints)) and (not offroad):
            reward += 300.0
            self.done = True
            print("[CarEnv] 🎯 Finish reached (with sensors)!")

        # small penalties
        if self.car_speed < 0.2:
            reward -= 0.4
        reward -= 0.05  # time penalty

        self.total_reward += reward
        self.steps += 1

        # max steps limit
        if self.steps >= 2000:
            self.done = True

        obs = self._get_obs()
        return obs, float(reward), bool(self.done), {"collision": collision, "offroad": offroad, "checkpoints": list(self.passed_checkpoints)}

    def _get_car_rect(self):
        w, h = 38, 22
        return pygame.Rect(int(self.car_pos[0] - w/2), int(self.car_pos[1] - h/2), w, h)

    def _is_on_track(self):
        """
        Returns True if the car center is inside the drivable road band.
        This is used to strictly enforce staying on track and to prevent cheating.
        """
        road_top = SCREEN_H//2 - self.track_width//2
        road_bottom = SCREEN_H//2 + self.track_width//2

        x, y = float(self.car_pos[0]), float(self.car_pos[1])

        # inside horizontal band of track (between track margins)
        inside_x = (self.track_margin <= x <= SCREEN_W - self.track_margin)
        # inside vertical band (between road_top and road_bottom)
        inside_y = (road_top <= y <= road_bottom)

        return inside_x and inside_y
    
    def _draw_curved_track(self):
        # Draw rounded rectangle style track (outer & inner)
        outer = pygame.Rect(self.track_margin, SCREEN_H//2 - self.track_width//2,
                            SCREEN_W - 2*self.track_margin, self.track_width)
        inner = outer.inflate(-60, -60)
        pygame.draw.rect(self.screen, (25, 25, 25), outer, border_radius=60)
        pygame.draw.rect(self.screen, (50, 50, 50), inner, border_radius=60)
        pygame.draw.circle(self.screen, (200, 100, 40), (self.track_margin, SCREEN_H//2), 100, 40)
        pygame.draw.circle(self.screen, (200, 100, 40), (SCREEN_W - self.track_margin, SCREEN_H//2), 100, 40)
        # lane lines
        pygame.draw.rect(self.screen, (255,255,255), (outer.left+6, outer.top+6, 2, outer.height-12))
        pygame.draw.rect(self.screen, (255,255,255), (outer.right-8, outer.top+6, 2, outer.height-12))

    def render(self):
        # Keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit()

        # background
        if self.track_image is not None:
            self.screen.blit(self.track_image, (0,0))
        else:
            # procedural curved track
            self.screen.fill((30, 30, 30))
            self._draw_curved_track()

        # finish line
        pygame.draw.line(self.screen, (255,255,255),
                         (self.finish_x, SCREEN_H//2 - self.track_width//2 - 10),
                         (self.finish_x, SCREEN_H//2 + self.track_width//2 + 10), 5)

        # draw checkpoints
        for idx, cp_x in enumerate(self.checkpoints):
            color = (200,200,40) if idx in self.passed_checkpoints else (120,120,120)
            pygame.draw.line(self.screen, color, (cp_x, SCREEN_H//2 - self.track_width//2 - 6),
                             (cp_x, SCREEN_H//2 + self.track_width//2 + 6), 4)
            # small label
            font = pygame.font.SysFont(None, 20)
            lbl = font.render(f"CP{idx+1}", True, (255,255,255))
            self.screen.blit(lbl, (cp_x-18, SCREEN_H//2 - self.track_width//2 - 28))

        # draw opponents
        for o in self.opponents:
            ox, oy = o["pos"]
            opp_rect = pygame.Rect(int(ox - 16), int(oy - 12), 32, 24)
            pygame.draw.rect(self.screen, (30, 144, 255), opp_rect)
            pygame.draw.polygon(self.screen, (10, 80, 200),
                                [(ox+14, oy), (ox+6, oy-6), (ox+6, oy+6)])

        # draw agent
        car_rect = self._get_car_rect()
        if self.car_image is not None:
            rotated = pygame.transform.rotate(self.car_image, -self.car_angle)
            new_rect = rotated.get_rect(center=(int(self.car_pos[0]), int(self.car_pos[1])))
            self.screen.blit(rotated, new_rect)
        else:
            surf = pygame.Surface((car_rect.w, car_rect.h), pygame.SRCALPHA)
            surf.fill((220, 60, 60))
            rotated = pygame.transform.rotate(surf, self.car_angle)
            r = rotated.get_rect(center=(int(self.car_pos[0]), int(self.car_pos[1])))
            self.screen.blit(rotated, r)
     
        # === SENSOR VISUALIZATION (add this block) ===
        sensor_angles = [0, -30, 30, -60, 60]
        max_range = 150.0
        for rel_ang in sensor_angles:
            ang = math.radians(self.car_angle + rel_ang)
            hit_x, hit_y = None, None
            for dist in np.linspace(0, max_range, num=15):
                x = int(self.car_pos[0] + dist * math.cos(ang))
                y = int(self.car_pos[1] - dist * math.sin(ang))
                if not self._point_on_track(x, y) or self._point_hits_opponent(x, y):
                    hit_x, hit_y = x, y
                    break
            if hit_x is None:
                hit_x = int(self.car_pos[0] + max_range * math.cos(ang))
                hit_y = int(self.car_pos[1] - max_range * math.sin(ang))
            pygame.draw.line(self.screen, (0, 255, 0), self.car_pos, (hit_x, hit_y), 2)
            pygame.draw.circle(self.screen, (255, 100, 100), (hit_x, hit_y), 3)


    
        # HUD
        font = pygame.font.SysFont(None, 22)
        txt = font.render(f"Speed: {self.car_speed:.2f}", True, (255,255,255))
        self.screen.blit(txt, (10, 10))
        txt2 = font.render(f"Pos: {int(self.car_pos[0])},{int(self.car_pos[1])}", True, (255,255,255))
        self.screen.blit(txt2, (10, 34))
        cp_text = font.render(f"Passed: {len(self.passed_checkpoints)}/{len(self.checkpoints)}", True, (255,255,255))
        self.screen.blit(cp_text, (10, 56))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()
