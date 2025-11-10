import numpy as np
import pytest
import torch
from skimage import data
from skimage.metrics import structural_similarity
from torch.autograd import gradcheck

from adv_lib.distances.structural_similarity import compute_ssim


@pytest.mark.parametrize('sigma', [0, 0.01, 0.03, 0.1, 0.3])
def test_compute_ssim_gray(sigma: float) -> None:
    # test for gray level images
    np_gray_img = data.camera().astype(np.float64) / 255
    pt_gray_img = torch.asarray(np_gray_img)

    noise = torch.randn_like(pt_gray_img) * sigma
    noisy_pt_gray_img = (pt_gray_img + noise).clamp(0, 1)
    noisy_np_gray_img = noisy_pt_gray_img.numpy()

    skimage_ssim = structural_similarity(
        noisy_np_gray_img, np_gray_img, win_size=11, use_sample_covariance=False, gaussian_weights=True, data_range=1
    )
    adv_lib_ssim = compute_ssim(noisy_pt_gray_img.unsqueeze(0).unsqueeze(1),
                                pt_gray_img.unsqueeze(0).unsqueeze(1)).squeeze()

    torch.testing.assert_close(torch.asarray(skimage_ssim), adv_lib_ssim)


@pytest.mark.parametrize('sigma', [0, 0.01, 0.03, 0.1, 0.3])
def test_compute_ssim_gradcheck(sigma: float) -> None:
    np_gray_img = data.camera()[:11, :11].astype(np.float64) / 255
    pt_gray_img = torch.asarray(np_gray_img).unsqueeze(0).unsqueeze(1)

    noise = torch.randn_like(pt_gray_img) * sigma
    noisy_pt_gray_img = (pt_gray_img + noise).clamp(0, 1)

    func = lambda x: compute_ssim(x, pt_gray_img)
    noisy_pt_gray_img.requires_grad_(True)
    gradcheck(func, inputs=(noisy_pt_gray_img,), raise_exception=True)


@pytest.mark.parametrize('sigma', [0, 0.01, 0.03, 0.1, 0.3])
def test_compute_ssim_color(sigma: float) -> None:
    # test for color images
    np_color_img = data.astronaut().astype(np.float64) / 255
    pt_color_img = torch.as_tensor(np_color_img)

    noise = torch.randn_like(pt_color_img) * sigma
    noisy_pt_color_img = (pt_color_img + noise).clamp(0, 1)
    noisy_np_color_img = noisy_pt_color_img.numpy()

    skimage_ssim = structural_similarity(noisy_np_color_img, np_color_img, win_size=11,
                                         multichannel=True, use_sample_covariance=False, gaussian_weights=True,
                                         data_range=1, channel_axis=2)
    adv_lib_ssim = compute_ssim(noisy_pt_color_img.permute(2, 0, 1).unsqueeze(0),
                                pt_color_img.permute(2, 0, 1).unsqueeze(0)).squeeze()

    torch.testing.assert_close(torch.asarray(skimage_ssim), adv_lib_ssim)
