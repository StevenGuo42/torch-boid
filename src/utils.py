import torch
from boid import Flock
import pygame



def map_point_to_screen(point, flock: Flock, window_size):
    # bottom_mat = torch.tensor(flock.box_bottom, device=flock.device) # float (D)
    # top_mat = torch.tensor(flock.box_top, device=flock.device) # float (D)
    
    game_pos = (point - flock.box_bottom) / (flock.box_top - flock.box_bottom) * torch.tensor(window_size, device=flock.device) # float (D)
    game_pos = game_pos.clone().detach().cpu().numpy()
    
    return game_pos

def map_vector_to_screen(vector, flock: Flock, window_size):
    # bottom_mat = torch.tensor(flock.box_bottom, device=flock.device) # float (D)
    # top_mat = torch.tensor(flock.box_top, device=flock.device) # float (D)
    
    game_vel = vector / (flock.box_top - flock.box_bottom) * torch.tensor(window_size, device=flock.device) # float (D)
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

def draw_text(screen, text, font, position, color = None, triangle_size = None):
    if color is None:
        color = (0, 0, 255)
    label = font.render(text, 1, color)
    
    if triangle_size is None:
        pos = (int(position[0]), int(position[1]))
    else:
        pos = (int(position[0]-triangle_size/2), int(position[1]-triangle_size/2))
    screen.blit(label, pos)
    
def draw_margin(screen, top_margin, bottom_margin, color = None):
    if color is None:
        color = (255, 0, 0)
    pygame.draw.rect(screen, color, ((bottom_margin[0]), bottom_margin[1], top_margin[0]-bottom_margin[0], top_margin[1]-bottom_margin[1]), width=1)

def fps_counter(screen, clock, font):
    fps = str(int(clock.get_fps()))
    fps_t = font.render(fps , 1, pygame.Color("GREEN"))
    screen.blit(fps_t,(0,0))