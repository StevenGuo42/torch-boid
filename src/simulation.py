
import sys
import pygame
import numpy as np
import torch

from boid import Flock



# Initialize Pygame
pygame.init()
clock = pygame.time.Clock()


fps = 1

# Set the window dimensions
world_size = 2000
window_size = 1000
ww_ratio = window_size / world_size
window_size = (window_size, window_size)



# Create a window
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Boids Simulation")

# Define colors
bg_color = (0, 0, 0)
color = (255, 255, 255)


device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flock_1 = Flock(D=2, N=100, box_top=world_size, 
                bias_factor=0,
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



running = True

print(flock_1.pos, flock_1.vel)

while running:
    clock.tick(fps)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill(bg_color)
    
    game_pos = map_point_to_screen(flock_1.pos, flock_1)
    # game_vel = map_vector_to_screen(flock_1.vel, flock_1)
    game_vel_normed = map_vector_to_screen(torch.nn.functional.normalize(flock_1.vel, p=2, dim=-1), flock_1)

    
    for i in range(flock_1.N):
        # pygame.draw.circle(screen, color, (int(game_pos[i][0]), int(game_pos[i][1])), 5)
        draw_triangles(screen, color, game_pos[i], game_vel_normed[i], 10)
        draw_flock_range(screen, game_pos[i], flock_1.avoid_radius * ww_ratio, flock_1.view_radius * ww_ratio, width=1)
    # draw_triangles(screen, color, flock_1.pos, flock_1.vel, 5, flock_1.N)
    
    

    # Update the display
    pygame.display.flip()
    
    # Update Flock
    flock_1.update()
    
    # print(flock_1.pos, flock_1.vel)

# Quit Pygame
pygame.quit()
sys.exit()


