
import sys
import pygame
import numpy as np
import torch

from boid import Flock

from utils import map_point_to_screen, map_vector_to_screen, draw_triangles, draw_flock_range, draw_text, draw_margin, fps_counter

# Initialize Pygame
pygame.init()
clock = pygame.time.Clock()
my_font = pygame.font.SysFont("monospace", 16)

# Set the window dimensions
world_size = 500
window_size = 500
margin_top = margin_bottom = 100


ww_ratio = window_size / world_size
window_size = (window_size, window_size)


is_draw_range = 0
is_draw_number = 0
is_draw_margin = 0

triangle_size = 10
fps = 30 

# set boids properties
N = 1000

init_speed=None

view_radius = 40
avoid_radius = 8

sep_factor = 0.05        # avoidfactor
align_factor = 0.05      # matchingfactor
cohe_factor = 0.0005     # centeringfactor
bias_factor = 0.0001
edge_factor = 0.2       # turnfactor

pass_through_edges = 1
bouncy_edges = 0

# Create a window
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Boids Simulation")

# Define colors
bg_color = (0, 0, 0)
color = (255, 255, 255)

# seed = 1337
# torch.manual_seed(seed)


# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
flock_1 = Flock(D=2, N=N, box_top=world_size, 
                sep_factor=sep_factor, align_factor=align_factor, cohe_factor=cohe_factor, 
                bias_factor=bias_factor, edge_factor=edge_factor, 
                pass_through_edges=pass_through_edges, bouncy_edges=bouncy_edges,
                init_speed=init_speed, 
                margin_bottom=margin_bottom, margin_top=margin_top,
                device=device)




running = True


top_margin = map_point_to_screen(flock_1.bound_top, flock_1, window_size).astype(int)
bottom_margin = map_point_to_screen(flock_1.bound_bottom, flock_1, window_size).astype(int)

print(top_margin, bottom_margin)

is_paused = False
while running:
    clock.tick(fps)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            is_paused = not is_paused
    
    
    if is_paused:
        continue
    
    # Clear the screen
    screen.fill(bg_color)
    
    game_pos = map_point_to_screen(flock_1.pos, flock_1, window_size)
    # game_vel = map_vector_to_screen(flock_1.vel, flock_1)
    game_vel_normed = map_vector_to_screen(torch.nn.functional.normalize(flock_1.vel, p=2, dim=-1), flock_1, window_size)

    # Draw origin and margin
    if is_draw_margin:
        draw_margin(screen, top_margin, bottom_margin, color=(255, 255, 255))
    
    for i in range(flock_1.N):
        # pygame.draw.circle(screen, color, (int(game_pos[i][0]), int(game_pos[i][1])), 5)
        draw_triangles(screen, color, game_pos[i], game_vel_normed[i], triangle_size)
        if is_draw_range:
            draw_flock_range(screen, game_pos[i], flock_1.avoid_radius * ww_ratio, flock_1.view_radius * ww_ratio, width=1)
        if is_draw_number:
            draw_text(screen, f"{i}", my_font, (game_pos[i][0], game_pos[i][1]), triangle_size = triangle_size)
    # draw_triangles(screen, color, flock_1.pos, flock_1.vel, 5, flock_1.N)
    
    fps_counter(screen, clock, my_font)

    # Update the display
    pygame.display.flip()
    
    # Update Flock
    flock_1.update()
    
    # print(flock_1.pos, flock_1.vel)

# Quit Pygame
pygame.quit()
sys.exit()


