import torch
from torch import Tensor

_ycbcr_conversions = {
    'rec_601': (0.299, 0.587, 0.114),
    'rec_709': (0.2126, 0.7152, 0.0722),
    'rec_2020': (0.2627, 0.678, 0.0593),
    'smpte_240m': (0.212, 0.701, 0.087),
}


def rgb_to_ycbcr(input: Tensor, standard: str = 'rec_2020'):
    kr, kg, kb = _ycbcr_conversions[standard]
    conversion_matrix = torch.tensor([[kr, kg, kb],
                                      [-0.5 * kr / (1 - kb), -0.5 * kg / (1 - kb), 0.5],
                                      [0.5, -0.5 * kg / (1 - kr), -0.5 * kb / (1 - kr)]], device=input.device)
    return torch.einsum('mc,nchw->nmhw', conversion_matrix, input)


def ycbcr_to_rgb(input: Tensor, standard: str = 'rec_2020'):
    kr, kg, kb = _ycbcr_conversions[standard]
    conversion_matrix = torch.tensor([[1, 0, 2 - 2 * kr],
                                      [1, -kb / kg * (2 - 2 * kb), -kr / kg * (2 - 2 * kr)],
                                      [1, 2 - 2 * kb, 0]], device=input.device)
    return torch.einsum('mc,nchw->nmhw', conversion_matrix, input)


_xyz_conversions = {
    'CIE_RGB': ((0.4887180, 0.3106803, 0.2006017),
                (0.1762044, 0.8129847, 0.0108109),
                (0.0000000, 0.0102048, 0.9897952)),
    'sRGB': ((0.4124564, 0.3575761, 0.1804375),
             (0.2126729, 0.7151522, 0.0721750),
             (0.0193339, 0.1191920, 0.9503041))
}


def rgb_to_xyz(input: Tensor, rgb_space: str = 'sRGB'):
    conversion_matrix = torch.tensor(_xyz_conversions[rgb_space], device=input.device)
    # Inverse sRGB companding
    v = torch.where(input <= 0.04045, input / 12.92, ((input + 0.055) / 1.055) ** 2.4)
    return torch.einsum('mc,nchw->nmhw', conversion_matrix, v)


_delta = 6 / 29


def cielab_func(input: Tensor) -> Tensor:
    # torch.where produces NaNs in backward if one of the choice produces NaNs or infs in backward (here .pow(1/3))
    return torch.where(input > _delta ** 3, input.clamp(min=_delta ** 3).pow(1 / 3), input / (3 * _delta ** 2) + 4 / 29)


def cielab_inverse_func(input: Tensor) -> Tensor:
    return torch.where(input > _delta, input.pow(3), 3 * _delta ** 2 * (input - 4 / 29))


_cielab_conversions = {
    'illuminant_d50': (96.4212, 100, 82.5188),
    'illuminant_d65': (95.0489, 100, 108.884),
}


def rgb_to_cielab(input: Tensor, standard: str = 'illuminant_d65') -> Tensor:
    # Convert to XYZ
    XYZ_input = rgb_to_xyz(input=input)

    Xn, Yn, Zn = _cielab_conversions[standard]
    L_star = 116 * cielab_func(XYZ_input.narrow(1, 1, 1) / Yn) - 16
    a_star = 500 * (cielab_func(XYZ_input.narrow(1, 0, 1) / Xn) - cielab_func(XYZ_input.narrow(1, 1, 1) / Yn))
    b_star = 200 * (cielab_func(XYZ_input.narrow(1, 1, 1) / Yn) - cielab_func(XYZ_input.narrow(1, 2, 1) / Zn))
    return torch.cat((L_star, a_star, b_star), 1)
