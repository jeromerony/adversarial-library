import torch
from torch import Tensor


def l1_ball_euclidean_projection(x: Tensor, maxnorm: float) -> Tensor:
    if (to_project := x.norm(p=1, dim=1) >= maxnorm).any():
        x_to_project = x[to_project].abs()
        μ = x_to_project.sort(dim=1, descending=True)[0]
        cumsum = μ.cumsum(dim=1)
        j_s = torch.arange(1, x.shape[-1] + 1, device=x.device).view(1, -1)
        ρ = (((μ * j_s) > (cumsum - maxnorm)) * j_s).argmax(dim=1, keepdim=True)
        θ = (cumsum.gather(dim=1, index=ρ) - maxnorm) / (ρ + 1)
        x[to_project] = (x_to_project - θ).clamp_min(0) * x[to_project].sign()
        return x
    else:
        return x
