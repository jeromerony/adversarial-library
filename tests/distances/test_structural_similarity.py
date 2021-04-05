import numpy as np
import pytest
import torch
from skimage import data
from skimage.metrics import structural_similarity

from adv_lib.distances.structural_similarity import compute_ssim


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_compute_ssim_gray(dtype: np.dtype) -> None:
    # test for gray level images
    np_gray_img = data.camera().astype(dtype) / 255
    pt_gray_img = torch.as_tensor(np_gray_img)

    for sigma in [0, 0.01, 0.03, 0.1, 0.3]:
        noise = torch.randn_like(pt_gray_img) * sigma

        noisy_pt_gray_img = (pt_gray_img + noise).clamp(0, 1)
        noisy_np_gray_img = noisy_pt_gray_img.numpy()

        skimage_ssim = structural_similarity(noisy_np_gray_img, np_gray_img, win_size=11, sigma=1.5,
                                             use_sample_covariance=False, gaussian_weights=True, data_range=1)
        adv_lib_ssim = compute_ssim(noisy_pt_gray_img.unsqueeze(0).unsqueeze(1),
                                    pt_gray_img.unsqueeze(0).unsqueeze(1))
        abs_diff = abs(skimage_ssim - adv_lib_ssim.item())
        assert abs_diff < 2e-5


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_compute_ssim_color(dtype: np.dtype) -> None:
    # test for color images
    np_color_img = data.astronaut().astype(dtype) / 255
    pt_color_img = torch.as_tensor(np_color_img)

    for sigma in [0, 0.01, 0.03, 0.1, 0.3]:
        noise = torch.randn_like(pt_color_img) * sigma

        noisy_pt_color_img = (pt_color_img + noise).clamp(0, 1)
        noisy_np_color_img = noisy_pt_color_img.numpy()

        skimage_ssim = structural_similarity(noisy_np_color_img, np_color_img, win_size=11, sigma=1.5,
                                             multichannel=True, use_sample_covariance=False, gaussian_weights=True,
                                             data_range=1)
        adv_lib_ssim = compute_ssim(noisy_pt_color_img.permute(2, 0, 1).unsqueeze(0),
                                    pt_color_img.permute(2, 0, 1).unsqueeze(0))

        abs_diff = abs(skimage_ssim - adv_lib_ssim.item())
        assert abs_diff < 1e-5
