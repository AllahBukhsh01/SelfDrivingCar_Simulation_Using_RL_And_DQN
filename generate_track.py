import os
import random
import pygame
import math

# === CONFIG ===
TILE_FOLDER = r"D:\Main_Folder\RL_Project\RL_SelfDrivingCar\assets\Tiles\Sand road"
OUTPUT_FILE = r"D:\Main_Folder\RL_Project\RL_SelfDrivingCar\assets\generated_track.png"
MAP_SIZE = (20, 15)
TILE_SIZE = 64
CHECKPOINT_INTERVAL = 8  # every 8th tile is checkpoint

pygame.init()
pygame.display.set_mode((1, 1))  # Fix: required for convert_alpha()

print(f"[generate_track] Loading tiles from: {TILE_FOLDER}")

# === LOAD TILE IMAGES ===
tile_variants = []
for file in sorted(os.listdir(TILE_FOLDER)):
    if not file.lower().endswith((".png", ".jpg")):
        continue
    path = os.path.join(TILE_FOLDER, file)
    surf = pygame.image.load(path).convert_alpha()
    for rot in [0, 90, 180, 270]:
        rotated = pygame.transform.rotate(surf, rot)
        tile_variants.append((file, rotated, rot))

print(f"[generate_track] Loaded {len(os.listdir(TILE_FOLDER))} files -> {len(tile_variants)} variants")

# === MASK DETECTION ===
mask_variants = {}
for fname, surf, rot in tile_variants:
    arr = pygame.surfarray.array_alpha(surf)
    mask_val = arr.mean() // 16  # rough tile signature
    mask_variants.setdefault(mask_val, []).append((surf, fname, rot))

print(f"[generate_track] Distinct masks found: {list(mask_variants.keys())}")

# === LOOPED TRACK GENERATION ===
def generate_loop_path(width, height):
    path = []
    cx, cy = width // 2, height // 2
    radius_x, radius_y = width // 3, height // 3
    steps = 80  # smoother loop
    for i in range(steps):
        angle = (i / steps) * 2 * math.pi
        x = int(cx + radius_x * math.cos(angle))
        y = int(cy + radius_y * math.sin(angle))
        path.append((x, y))
    return path

track_path = generate_loop_path(*MAP_SIZE)
print(f"[generate_track] Built circular loop with {len(track_path)} tiles")

# === FUNCTION: choose variant safely ===
def choose_variant_for_mask(mask):
    # pick closest available mask variant
    closest_key = min(mask_variants.keys(), key=lambda k: abs(mask - k))
    return random.choice(mask_variants[closest_key])

# === DRAW TRACK ===
track_surface = pygame.Surface((MAP_SIZE[0] * TILE_SIZE, MAP_SIZE[1] * TILE_SIZE))
track_surface.fill((240, 218, 171))  # sand background

num_tiles = len(track_path)
for i, (x, y) in enumerate(track_path):
    mask = random.choice(list(mask_variants.keys()))
    surf, fname, rot = choose_variant_for_mask(mask)
    track_surface.blit(surf, (x * TILE_SIZE, y * TILE_SIZE))

    # === COLOR GRADIENT CHECKPOINTS (Red → Green) ===
    if i % CHECKPOINT_INTERVAL == 0:
        t = i / num_tiles
        r = int(255 * (1 - t))
        g = int(255 * t)
        color = (r, g, 0)
        pygame.draw.circle(
            track_surface,
            color,
            (x * TILE_SIZE + TILE_SIZE // 2, y * TILE_SIZE + TILE_SIZE // 2),
            6,
        )

pygame.image.save(track_surface, OUTPUT_FILE)
print(f"[generate_track] ✅ Track saved to: {OUTPUT_FILE}")
pygame.quit()
