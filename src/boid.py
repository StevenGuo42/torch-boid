# Boids algorithm partially based on https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html#Update-position
from dataclasses import dataclass
from numbers import Number

import torch


# boid
@dataclass
class Boid:
    init_speed: float = None
    min_speed: float = 3
    max_speed: float = 6
    max_acc: float = 0.5
    
    view_radius: float = 40
    view_angle: float = None        # human: 220 deg, pigeon: 340 deg, owl: 110 deg
    
    avoid_radius: float = 8         
    avoid_view: bool = True         # only avoid boids in view angle
    
    sep_factor: float = 0.05        # avoidfactor
    align_factor: float = 0.05      # matchingfactor
    cohe_factor: float = 0.0005     # centeringfactor
    bias_factor: float = 0.005       
    edge_factor: float = 0.05        # turnfactor
    
    is_debug: bool = False
    
# D-dim flock with N boids
class Flock(Boid):
    def __init__(self, D: int=2, N: int=1000, 
                 box_bottom=0, box_top=500, 
                 margin_bottom=100, margin_top = 100, 
                 pass_through_edges=True, bouncy_edges=False,
                 device = torch.device("cpu"), **kwargs):
        super().__init__(**kwargs)
        self.device = device
        
        self.D = D
        self.N = N
        
        self.box_bottom = self.parse_to_tensor(box_bottom).to(self.device)
        self.box_top = self.parse_to_tensor(box_top).to(self.device)
        self.margin_bottom = self.parse_to_tensor(margin_bottom).to(self.device)
        self.margin_top = self.parse_to_tensor(margin_top).to(self.device)
        self.bound_bottom = self.box_bottom + self.margin_bottom
        self.bound_top = self.box_top - self.margin_top
        
        
        self.box_upper_mat = self.box_top.unsqueeze(0).expand(N, -1).to(self.device)
        self.box_lower_mat = self.box_bottom.unsqueeze(0).expand(N, -1).to(self.device)
        
        self.pass_through_edges = pass_through_edges
        self.bouncy_edges = bouncy_edges
        
        # init position and velocity of boids
        self.pos = torch.rand((N, D), device=self.device) * (self.bound_top - self.bound_bottom) + self.bound_bottom
        if self.init_speed is None:
            self.vel = torch.randn((N, D), device=self.device) * (self.max_speed - self.min_speed) + self.min_speed
        else:
            self.vel = torch.ones((N, D), device=self.device) * self.init_speed
        
        
        
        
        if self.is_debug:
            print('pos: \n', self.pos)
            print('vel: \n', self.vel)
        
    # parse input (Number or Iterable) to tensor
    def parse_to_tensor(self, x):
        if isinstance(x, Number):
            return torch.tensor([x] * self.D, device=self.device)
        elif len(x) != self.D:
            raise ValueError('input must be a number or a list of length D')
        else:
            return torch.tensor(x, device=self.device)

    # update position and velocity of boids
    def update(self):
        # region - init
        
        # print('-----------------Update--------------')
        # print('pos: \n', self.pos)
        # print('vel: \n', self.vel)
        
        pos_mat = self.pos.unsqueeze(1).expand(-1, self.N, -1)              # float (N,N,D)
        vel_mat = self.vel.unsqueeze(1).expand(-1, self.N, -1)              # float (N,N,D)
        # print('pos_mat: \n', pos_mat)
        # print('vel_mat: \n', vel_mat)
        
        # get boids position difference
        diff = pos_mat.transpose(0, 1) - pos_mat                            # float (N,N,D)
        # print('diff: \n', diff)
        # get boids distance
        sq_dist_mat = diff.pow(2).sum(dim=-1)                               # float (N,N)
        # print('sq_dist_mat: \n', sq_dist_mat)
        # endregion
        
        # region - boids view
        # get boids in view radius
        view_mat = sq_dist_mat < self.view_radius ** 2
        view_mat.fill_diagonal_(0)                                          # bool (N,N)
        
        # get boids in view angle
        if self.view_angle is not None:
            # if cosine similarity of pos diff and vel is greater than cosine of 
            # half view angle, then boid is in view angle
            view_angle_mat = torch.cosine_similarity(diff, vel_mat, dim=-1) \
                > torch.cos(self.view_angle / 2)                            # bool (N,N)
            view_mat *= view_angle_mat                                      # bool (N,N)
            
        view_mat = view_mat.unsqueeze(-1).expand(-1, -1, self.D)            # bool (N,N,D)
        # print('view_mat: \n', view_mat.int())
        # endregion


        
        # region - boids avoid
        # get boids in avoid radius
        avoid_mat = sq_dist_mat < self.avoid_radius ** 2
        avoid_mat.fill_diagonal_(0)                                         # bool (N,N)
        # print('avoid_mat: \n', avoid_mat.int())
        # only avoid boids it can see
        if self.view_angle is not None and self.avoid_view:
            avoid_mat *= view_angle_mat
        
        avoid_mat = avoid_mat.unsqueeze(-1).expand(-1, -1, self.D)          # bool (N,N,D)
        
        # get avoid direction matrix
        avoid_mat_sum = avoid_mat.sum(dim=1)
        avoid_mask = avoid_mat_sum != 0
        
        avoid_vel = torch.zeros((self.N, self.D), device=self.device)
        avoid_vel[avoid_mask] = ((avoid_mat * diff).sum(dim=0))[avoid_mask] \
                                    / avoid_mat_sum[avoid_mask]
        # avoid_vel[avoid_mask] = -((pos_mat * avoid_mat).sum(dim=0))[avoid_mask] \
        #                         / avoid_mat_sum[avoid_mask]
        # avoid_vel[avoid_mask] = -((pos_mat * avoid_mat).sum(dim=0))[avoid_mask]
        
        # avoid_vel = -(avoid_mat * diff).sum(dim=1) / avoid_mat.sum(dim=1)
        # print('avoid_vel: \n', avoid_vel)
        # get edge avoid vector
        bottom_edge = torch.le(self.pos, self.bound_bottom)                 # bool (N,D)
        top_edge = torch.ge(self.pos, self.bound_top)
        # print('bottom_edge: \n', bottom_edge.int())
        # print('top_edge: \n', top_edge.int())
        # endregion
        

        # get avg position and velocity of boids in view
        # avg_pos = (pos_mat * view_mat).sum(dim=1) / view_mat.sum(dim=1)     # float (N,D)
        # avg_vel = (vel_mat * view_mat).sum(dim=1) / view_mat.sum(dim=1)

        # only see boids in view and not in avoid radius
        view_mat = view_mat & ~avoid_mat
        
        view_mat_sum = view_mat.sum(dim=1)
        # print('view_mat_sum: \n', view_mat_sum)
        view_mask = view_mat_sum != 0
        # print('view_mask: \n', view_mask.int())
        avg_pos = torch.zeros((self.N, self.D), device=self.device)
        avg_pos[view_mask] = ((pos_mat * view_mat).sum(dim=0))[view_mask] \
                                / view_mat_sum[view_mask]
        
        avg_vel = torch.zeros((self.N, self.D), device=self.device)
        avg_vel[view_mask] = ((vel_mat * view_mat).sum(dim=0))[view_mask] \
                                / view_mat_sum[view_mask]

        # print('avg_pos: \n', avg_pos)
        # print('avg_vel: \n', avg_vel)
        
        # region - compute velocity
        # cohe align only if boids in view
        # Cohesion (move towards center of mass of boids in view)
        cohe_vel = (avg_pos - self.pos) * view_mask * self.cohe_factor      # float (N,D)
        # print('cohe_vel: \n', cohe_vel)
        # Alignment (move towards average velocity of boids in view)
        align_vel = (avg_vel - self.vel) * view_mask * self.align_factor    # float (N,D)   
        # print('align_vel: \n', align_vel)
        # sep only if boids in avoid radius
        # Separation (move away from boids in avoid radius)
        sep_vel = avoid_vel * avoid_mask * self.sep_factor                  # float (N,D)        
        # print('sep_vel: \n', sep_vel)
        # TODO: custom bias
        # Bias (move towards center of box)
        bias_vel = ((self.box_bottom + self.box_top) / 2 - self.pos) * self.bias_factor
        # print('bias_vel: \n', bias_vel)
        
        # Edge (move away from edges)
        # # move up if bottom edge, move down if top edge
        edge_vel = bottom_edge * self.edge_factor - top_edge * self.edge_factor
        # print('edge_vel: \n', edge_vel)
        # get new velocity
        sum_d_vel = cohe_vel + align_vel + sep_vel + bias_vel + edge_vel                # d_vel from boids rules
        # endregion
        
        # region - limit speed and acceleration
        # limit acceleration
        # acc = sum_vel - self.vel
        acc = sum_d_vel
        acc_mag = acc.pow(2).sum(dim=-1).sqrt()                             # float (N)
        acc_mag = torch.clamp(acc_mag, max=self.max_acc)                # float (N)
        acc = torch.nn.functional.normalize(acc, p=2, dim=-1) * acc_mag.unsqueeze(-1)    # float (N,D)
        # acc = acc / acc.norm(dim=-1, keepdim=True) * acc_mag.unsqueeze(-1)    # float (N,D)
        # acc = acc/acc_mag.unsqueeze(-1) * acc_mag.unsqueeze(-1)         # float (N,D)
        
        # limit speed
        sum_vel = self.vel + acc
        vel_mag = sum_vel.pow(2).sum(dim=-1).sqrt()                         # float (N)
        vel_clipped = torch.clamp(vel_mag, min=self.min_speed, max=self.max_speed)
        sum_vel = torch.nn.functional.normalize(sum_vel, p=2, dim=-1) * vel_clipped.unsqueeze(-1)    # float (N,D)
        # sum_vel = sum_vel/vel_mag.unsqueeze(-1) * vel_clipped.unsqueeze(-1)
        # endregion
        
        # region - update position and velocity
        self.vel = sum_vel
        self.pos = self.pos + self.vel
        
        # pass through edges
        # self.pos = torch.where(self.pos < self.box_bottom, self.box_top, self.pos)
        # self.pos = torch.where(self.pos > self.box_top, self.box_bottom, self.pos)
        
        if self.pass_through_edges:
            self.pos = torch.where(self.pos < self.box_lower_mat, self.box_upper_mat, self.pos)
            self.pos = torch.where(self.pos > self.box_upper_mat, self.box_lower_mat, self.pos)
        # solid edges
        else: 
            # flip velocity if boid hits edge
            if self.bouncy_edges:
                self.vel = torch.where(self.pos < self.box_lower_mat, -self.vel, self.vel)
                self.vel = torch.where(self.pos > self.box_upper_mat, -self.vel, self.vel)
            
            self.pos = torch.maximum(self.pos, self.box_lower_mat)
            self.pos = torch.minimum(self.pos, self.box_upper_mat)
            
        
        # self.pos = torch.max(self.pos, self.box_bottom)
        # self.pos = torch.min(self.pos, self.box_top)
        # endregion
        

        
        
        
        if self.is_debug:
            print('pos: \n', self.pos)
            print('vel: \n', self.vel)
            
        if any(torch.isnan(self.pos).flatten()):
            raise ValueError('position contains NaN')
        if any(torch.isnan(self.vel).flatten()):
            raise ValueError('velocity contains NaN')

# main
if __name__ == '__main__':
    device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flock = Flock(D=2, N=5, box_top=50, is_debug = True, device=device)
    
    # print(flock)
    
    flock.update()
    
    # print(flock)