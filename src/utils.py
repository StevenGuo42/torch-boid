import torch

def sq_dist(mat):
    diff = matmatrix.unsqueeze(1) - mat.unsqueeze(0)
    return torch.pow(diff, 2).sum(dim=-1) #2