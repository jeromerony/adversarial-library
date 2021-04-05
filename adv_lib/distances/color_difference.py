import math

import torch
from torch import Tensor

from adv_lib.utils.color_conversions import rgb_to_cielab


def cie94_color_difference(Lab_1: Tensor, Lab_2: Tensor, k_L: float = 1, k_C: float = 1, k_H: float = 1,
                           K_1: float = 0.045, K_2: float = 0.015, squared: bool = False, ε: float = 0) -> Tensor:
    """
    Inputs should be L*, a*, b*. Star from formulas are omitted for conciseness.

    Parameters
    ----------
    Lab_1 : Tensor
        First image in L*a*b* space. First image is intended to be the reference image.
    Lab_2 : Tensor
        Second image in L*a*b* space. Second image is intended to be the modified one.
    k_L : float
        Weighting factor for S_L.
    k_C : float
        Weighting factor for S_C.
    k_H : float
        Weighting factor for S_H.
    squared : bool
        Return the squared ΔE_94.
    ε : float
        Small value for numerical stability when computing gradients. Default to 0 for most accurate evaluation.

    Returns
    -------
    ΔE_94 : Tensor
        The CIEDE2000 color difference for each pixel.

    """
    ΔL = Lab_1.narrow(1, 0, 1) - Lab_2.narrow(1, 0, 1)
    C_1 = torch.norm(Lab_1.narrow(1, 1, 2), p=2, dim=1, keepdim=True)
    C_2 = torch.norm(Lab_2.narrow(1, 1, 2), p=2, dim=1, keepdim=True)
    ΔC = C_1 - C_2
    Δa = Lab_1.narrow(1, 1, 1) - Lab_2.narrow(1, 1, 1)
    Δb = Lab_1.narrow(1, 2, 1) - Lab_2.narrow(1, 2, 1)
    ΔH = Δa ** 2 + Δb ** 2 - ΔC ** 2
    S_L = 1
    S_C = 1 + K_1 * C_1
    S_H = 1 + K_2 * C_1
    ΔE_94_squared = (ΔL / (k_L * S_L)) ** 2 + (ΔC / (k_C * S_C)) ** 2 + ΔH / ((k_H * S_H) ** 2)
    if squared:
        return ΔE_94_squared
    return ΔE_94_squared.clamp_min(ε).sqrt()


def rgb_cie94_color_difference(input: Tensor, target: Tensor, **kwargs) -> Tensor:
    """Computes the CIEDE2000 Color-Difference from RGB inputs."""
    return cie94_color_difference(*map(rgb_to_cielab, (input, target)), **kwargs)


def cie94_loss(x1: Tensor, x2: Tensor, squared: bool = False, **kwargs) -> Tensor:
    """
    Computes the L2-norm over all pixels of the CIEDE2000 Color-Difference for two RGB inputs.

    Parameters
    ----------
    x1 : Tensor:
        First input.
    x2 : Tensor:
        Second input (of size matching x1).
    squared : bool
        Returns the squared L2-norm.

    Returns
    -------
    ΔE_00_l2 : Tensor
        The L2-norm over all pixels of the CIEDE2000 Color-Difference.

    """
    ΔE_94_squared = rgb_cie94_color_difference(x1, x2, squared=True, **kwargs).flatten(1)
    ε = kwargs.get('ε', 0)
    if squared:
        return ΔE_94_squared.sum(1)
    return ΔE_94_squared.sum(1).clamp_min(ε).sqrt()


def ciede2000_color_difference(Lab_1: Tensor, Lab_2: Tensor, k_L: float = 1, k_C: float = 1, k_H: float = 1,
                               squared: bool = False, ε: float = 0) -> Tensor:
    """
    Inputs should be L*, a*, b*. Primes from formulas in
    http://www2.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf are omitted for conciseness.
    This version is based on the matlab implementation from Gaurav Sharma
    http://www2.ece.rochester.edu/~gsharma/ciede2000/dataNprograms/deltaE2000.m modified to have non NaN gradients.

    Parameters
    ----------
    Lab_1 : Tensor
        First image in L*a*b* space. First image is intended to be the reference image.
    Lab_2 : Tensor
        Second image in L*a*b* space. Second image is intended to be the modified one.
    k_L : float
        Weighting factor for S_L.
    k_C : float
        Weighting factor for S_C.
    k_H : float
        Weighting factor for S_H.
    squared : bool
        Return the squared ΔE_00.
    ε : float
        Small value for numerical stability when computing gradients. Default to 0 for most accurate evaluation.

    Returns
    -------
    ΔE_00 : Tensor
        The CIEDE2000 color difference for each pixel.

    """

    C_star_1 = torch.norm(Lab_1.narrow(1, 1, 2), p=2, dim=1, keepdim=True)
    C_star_2 = torch.norm(Lab_2.narrow(1, 1, 2), p=2, dim=1, keepdim=True)
    C_star_mean = (C_star_1 + C_star_2) / 2
    G = 0.5 * (1 - (C_star_mean ** 7 / (C_star_mean ** 7 + 25 ** 7)).clamp_min(ε).sqrt())

    a_star_1 = Lab_1.narrow(1, 1, 1)
    a_star_2 = Lab_2.narrow(1, 1, 1)
    b_star_1 = Lab_1.narrow(1, 2, 1)
    b_star_2 = Lab_2.narrow(1, 2, 1)

    a_1 = (1 + G) * a_star_1
    a_2 = (1 + G) * a_star_2
    C_1_C_2_zero = ((a_1 == 0) & (b_star_1 == 0)) | ((a_2 == 0) & (b_star_2 == 0))
    C_1 = (a_1 ** 2 + b_star_1 ** 2).clamp_min(ε).sqrt()
    C_2 = (a_2 ** 2 + b_star_2 ** 2).clamp_min(ε).sqrt()
    h_1 = torch.atan2(b_star_1, a_1 + ε * (a_1 == 0))
    h_2 = torch.atan2(b_star_2, a_2 + ε * (a_2 == 0))

    h_1 = h_1 + 2 * math.pi * (h_1 < 0)
    h_2 = h_2 + 2 * math.pi * (h_2 < 0)

    ΔL = Lab_2.narrow(1, 0, 1) - Lab_1.narrow(1, 0, 1)
    ΔC = C_2 - C_1
    Δh = torch.where(C_1_C_2_zero, torch.zeros_like(h_1),
                     torch.where(torch.abs(h_2 - h_1) <= math.pi, h_2 - h_1,
                                 torch.where(h_2 - h_1 > math.pi, h_2 - h_1 - 2 * math.pi, h_2 - h_1 + 2 * math.pi)))

    ΔH = 2 * (C_1 * C_2).clamp_min(ε).sqrt() * torch.sin(Δh / 2)
    ΔH_squared = 4 * C_1 * C_2 * torch.sin(Δh / 2) ** 2

    L_bar = (Lab_1.narrow(1, 0, 1) + Lab_2.narrow(1, 0, 1)) / 2
    C_bar = (C_1 + C_2) / 2
    h_bar = torch.where(C_1_C_2_zero, h_1 + h_2,
                        torch.where(torch.abs(h_1 - h_2) < math.pi + 1e-6, (h_1 + h_2) / 2,
                                    torch.where((torch.abs(h_1 - h_2) > math.pi) & ((h_1 + h_2) < 2 * math.pi),
                                                (h_1 + h_2 + 2 * math.pi) / 2, (h_1 + h_2 - 2 * math.pi) / 2)))

    T = 1 - 0.17 * (h_bar - math.pi / 6).cos() + 0.24 * (2 * h_bar).cos() + \
        0.32 * (3 * h_bar + math.pi / 30).cos() - 0.20 * (4 * h_bar - 63 * math.pi / 180).cos()

    Δθ = math.pi / 6 * (torch.exp(-((180 / math.pi * h_bar - 275) / 25) ** 2))
    R_C = 2 * (C_bar ** 7 / (C_bar ** 7 + 25 ** 7)).clamp_min(ε).sqrt()
    S_L = 1 + 0.015 * (L_bar - 50) ** 2 / torch.sqrt(20 + (L_bar - 50) ** 2)
    S_C = 1 + 0.045 * C_bar
    S_H = 1 + 0.015 * C_bar * T
    R_T = -torch.sin(2 * Δθ) * R_C

    ΔE_00 = (ΔL / (k_L * S_L)) ** 2 + (ΔC / (k_C * S_C)) ** 2 + ΔH_squared / (k_H * S_H) ** 2 + \
            R_T * (ΔC / (k_C * S_C)) * (ΔH / (k_H * S_H))
    if squared:
        return ΔE_00
    return ΔE_00.clamp_min(ε).sqrt()


def rgb_ciede2000_color_difference(input: Tensor, target: Tensor, **kwargs) -> Tensor:
    """Computes the CIEDE2000 Color-Difference from RGB inputs."""
    return ciede2000_color_difference(*map(rgb_to_cielab, (input, target)), **kwargs)


def ciede2000_loss(x1: Tensor, x2: Tensor, squared: bool = False, **kwargs) -> Tensor:
    """
    Computes the L2-norm over all pixels of the CIEDE2000 Color-Difference for two RGB inputs.
    
    Parameters
    ----------
    x1 : Tensor:
        First input.
    x2 : Tensor:
        Second input (of size matching x1).
    squared : bool
        Returns the squared L2-norm.

    Returns
    -------
    ΔE_00_l2 : Tensor
        The L2-norm over all pixels of the CIEDE2000 Color-Difference.

    """
    ΔE_00 = rgb_ciede2000_color_difference(x1, x2, squared=True, **kwargs).flatten(1)
    ε = kwargs.get('ε', 0)
    if squared:
        return ΔE_00.sum(1)
    return ΔE_00.sum(1).clamp_min(ε).sqrt()
