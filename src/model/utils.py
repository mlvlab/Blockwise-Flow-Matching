import torch
from torch import Tensor
from typing import List, Tuple


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def modulate_representation(x, shift, scale):
    return x * (1 + scale) + shift


def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype

def make_grid_mask_size(batch_size, n_patch_h, n_patch_w, patch_size, device):
    grid_h = torch.arange(n_patch_h, dtype=torch.long)
    grid_h = torch.arange(n_patch_h, dtype=torch.long)
    grid_w = torch.arange(n_patch_w, dtype=torch.long)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
    grid = torch.cat(
        [grid[0].reshape(1,-1).to(device), grid[1].reshape(1,-1).to(device)], dim=0
    ).repeat(batch_size,1,1).to(dtype=torch.long)
    mask = torch.ones((batch_size, n_patch_h*n_patch_w)).to(device=device, dtype=torch.bfloat16)
    size = torch.tensor((n_patch_h, n_patch_w)).repeat(batch_size,1).to(dtype=torch.long)
    size = size[:, None, :]
    return grid, mask, size

def make_grid_mask_size_online(x, patch_size, device):
    batch_size = x.shape[0]
    n_patch_h = int((x.shape[1]**0.5))
    n_patch_w = int((x.shape[1]**0.5))
    
    #import pdb; pdb.set_trace()
    grid_h = torch.arange(n_patch_h, dtype=torch.long)
    grid_w = torch.arange(n_patch_w, dtype=torch.long)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
    grid = torch.cat(
        [grid[0].reshape(1,-1).to(device), grid[1].reshape(1,-1).to(device)], dim=0
    ).repeat(batch_size,1,1).to(dtype=torch.long)
    mask = torch.ones(batch_size, n_patch_h*n_patch_w).to(device=device, dtype=torch.bfloat16)
    size = torch.tensor((n_patch_h, n_patch_w)).repeat(batch_size,1).to(device=device, dtype=torch.long)
    size = size[:, None, :]
    return grid, mask, size