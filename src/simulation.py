
import sys
import pygame
import numpy as np
import torch

from boid import Flock



# Initialize Pygame
pygame.init()
clock = pygame.time.Clock()

# Set the window dimensions
world_size = 500
window_size = 500
margin_top = margin_bottom = 150


ww_ratio = window_size / world_size
window_size = (window_size, window_size)
my_font = pygame.font.SysFont("monospace", 16)

is_draw_range = 1
is_draw_number = 1
triangle_size = 10
fps = 30 

# set boids properties
N = 50

init_speed=None

view_radius = 40
avoid_radius = 15

sep_factor = 0.05        # avoidfactor
align_factor = 0.05      # matchingfactor
cohe_factor = 0.0005     # centeringfactor
bias_factor = 0.00
edge_factor = 0.2       # turnfactor

pass_through_edges = 0
bouncy_edges = 0

# Create a window
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Boids Simulation")

# Define colors
bg_color = (0, 0, 0)
color = (255, 255, 255)

# seed = 1337
# torch.manual_seed(seed)


device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flock_1 = Flock(D=2, N=N, box_top=world_size, 
                sep_factor=sep_factor, align_factor=align_factor, cohe_factor=cohe_factor, 
                bias_factor=bias_factor, edge_factor=edge_factor, 
                pass_through_edges=pass_through_edges, bouncy_edges=bouncy_edges,
                init_speed=init_speed, 
                margin_bottom=margin_bottom, margin_top=margin_top,
                device=device)



def map_point_to_screen(point, flock: Flock):
    bottom_mat = torch.tensor(flock.box_bottom, device=flock.device) # float (D)
    top_mat = torch.tensor(flock.box_top, device=flock.device) # float (D)
    
    game_pos = (point - bottom_mat) / (top_mat - bottom_mat) * torch.tensor(window_size, device=flock.device) # float (D)
    game_pos = game_pos.clone().detach().cpu().numpy()
    
    return game_pos


def map_vector_to_screen(vector, flock: Flock):
    bottom_mat = torch.tensor(flock.box_bottom, device=flock.device) # float (D)
    top_mat = torch.tensor(flock.box_top, device=flock.device) # float (D)
    
    game_vel = vector / (top_mat - bottom_mat) * torch.tensor(window_size, device=flock.device) # float (D)
    game_vel = game_vel.clone().detach().cpu().numpy()
    
    return game_vel

    
    
    
def draw_triangles(screen, color, position, velocity, size):
    s = size / 2
    
    x1 = position[0] + s * velocity[0] * 0.7
    y1 = position[1] + s * velocity[1] * 0.7
    x2 = position[0] - s * velocity[0] * 0.5 + (-velocity[1]) * s * 0.5
    y2 = position[1] - s * velocity[1] * 0.5 + velocity[0] * s * 0.5
    x3 = position[0] - s * velocity[0] * 0.5 + velocity[1] * s * 0.5
    y3 = position[1] - s * velocity[1] * 0.5 + (-velocity[0]) * s * 0.5
    
    pygame.draw.polygon(screen, color, [(x1, y1), (x2, y2), (x3, y3)])
    

def draw_flock_range(screen, position, avoid_radius, view_radius, width=1):
    pygame.draw.circle(screen, (255, 0, 0), (int(position[0]), int(position[1])), int(avoid_radius), width)
    pygame.draw.circle(screen, (0, 255, 0), (int(position[0]), int(position[1])), int(view_radius), width)

def draw_text(screen, text, position, color = None, triangle_size = None):
    if color is None:
        color = (0, 0, 255)
    label = my_font.render(text, 1, color)
    
    if triangle_size is None:
        pos = (int(position[0]), int(position[1]))
    else:
        pos = (int(position[0]-triangle_size/2), int(position[1]-triangle_size/2))
    screen.blit(label, pos)
    
def draw_margin(screen, top_margin, bottom_margin, color = None):
    if color is None:
        color = (255, 0, 0)
    pygame.draw.rect(screen, color, ((bottom_margin[0]), bottom_margin[1], top_margin[0]-bottom_margin[0], top_margin[1]-bottom_margin[1]), width=1)

running = True


top_margin = map_point_to_screen(flock_1.bound_top, flock_1).astype(int)
bottom_margin = map_point_to_screen(flock_1.bound_bottom, flock_1).astype(int)

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
    
    game_pos = map_point_to_screen(flock_1.pos, flock_1)
    # game_vel = map_vector_to_screen(flock_1.vel, flock_1)
    game_vel_normed = map_vector_to_screen(torch.nn.functional.normalize(flock_1.vel, p=2, dim=-1), flock_1)

    # Draw origin and margin
    origin = torch.tensor([0, 0], device=flock_1.device)
    origin = map_point_to_screen(origin, flock_1)
    draw_text(screen, "origin", origin, color=(255, 0, 0))
    pygame.draw.circle(screen, (255, 0, 0), (int(origin[0]), int(origin[1])), 5)
    
    draw_margin(screen, top_margin, bottom_margin, color=(255, 255, 255))
    
    for i in range(flock_1.N):
        # pygame.draw.circle(screen, color, (int(game_pos[i][0]), int(game_pos[i][1])), 5)
        draw_triangles(screen, color, game_pos[i], game_vel_normed[i], triangle_size)
        if is_draw_range:
            draw_flock_range(screen, game_pos[i], flock_1.avoid_radius * ww_ratio, flock_1.view_radius * ww_ratio, width=1)
        if is_draw_number:
            draw_text(screen, f"{i}", (game_pos[i][0], game_pos[i][1]), triangle_size = triangle_size)
    # draw_triangles(screen, color, flock_1.pos, flock_1.vel, 5, flock_1.N)
    
    

    # Update the display
    pygame.display.flip()
    
    # Update Flock
    flock_1.update()
    
    # print(flock_1.pos, flock_1.vel)

# Quit Pygame
pygame.quit()
sys.exit()


