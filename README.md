
[![DOI](https://zenodo.org/badge/315504148.svg)](https://zenodo.org/badge/latestdoi/315504148)

# Adversarial Library

This library contains various resources related to adversarial attacks implemented in PyTorch. It is aimed towards researchers looking for implementations of state-of-the-art attacks.

The code was written to maximize efficiency (_e.g._ by preferring low level functions from PyTorch) while retaining simplicity (_e.g._ by avoiding abstractions). As a consequence, most of the library, and especially the attacks, is implemented using **pure functions** (whenever possible).

While focused on attacks, this library also provides several utilities related to adversarial attacks: distances (SSIM, CIEDE2000, LPIPS), visdom callback, projections, losses and helper functions. Most notably the function `run_attack` from `utils/attack_utils.py` performs an attack on a model given the inputs and labels, with fixed batch size, and reports complexity related metrics (run-time and forward/backward propagations).

### Dependencies

The goal of this library is to be up-to-date with newer versions of PyTorch so the dependencies are expected to be updated regularly (possibly resulting in breaking changes).

- pytorch>=1.8.0
- torchvision>=0.9.0
- tqdm>=4.48.0
- visdom>=0.1.8

### Installation

You can either install using:

```pip install git+https://github.com/jeromerony/adversarial-library```

Or you can clone the repo and run:

```python setup.py install```

Alternatively, you can install (after cloning) the library in editable mode:

```pip install -e .```

### Usage
Attacks are implemented as functions, so they can be called directly by providing the model, samples and labels (possibly with optional arguments):
```python
from adv_lib.attacks import ddn
adv_samples = ddn(model=model, inputs=inputs, labels=labels, steps=300)
```

Classification attacks all expect the following arguments:
- `model`: the model  that produces logits (pre-softmax activations) with inputs in $[0, 1]$
- `inputs`: the samples to attack in $[0, 1]$
- `labels`: either the ground-truth labels for the samples or the targets
- `targeted`: flag indicated if the attack should be targeted or not -- defaults to `False`

Additionally, many attacks have an optional `callback` argument which accepts an `adv_lib.utils.visdom_logger.VisdomLogger` to plot data to a visdom server for monitoring purposes.

 For a more detailed example on how to use this library, you can look at this repo: https://github.com/jeromerony/augmented_lagrangian_adversarial_attacks

## Contents

### Attacks

#### Classification

Currently the following classification attacks are implemented in the `adv_lib.attacks` module:

| Name                                                                                    | Knowledge | Type    | Distance(s)                                               | ArXiv Link                                                                                           |
|-----------------------------------------------------------------------------------------|-----------|---------|-----------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| DeepFool (DF)                                                                           | White-box | Minimal | $\ell_2$, $\ell_\infty$                                   | [1511.04599](https://arxiv.org/abs/1511.04599)                                                       |
| Carlini and Wagner (C&W)                                                                | White-box | Minimal | $\ell_2$, $\ell_\infty$                                   | [1608.04644](https://arxiv.org/abs/1608.04644)                                                       |
| Projected Gradient Descent (PGD)                                                        | White-box | Budget  | $\ell_\infty$                                             | [1706.06083](https://arxiv.org/abs/1706.06083)                                                       |
| Structured Adversarial Attack (StrAttack)                                               | White-box | Minimal | $\ell_2$ + group-sparsity                                 | [1808.01664](https://arxiv.org/abs/1808.01664)                                                       |
| **Decoupled Direction and Norm (DDN)**                                                  | White-box | Minimal | $\ell_2$                                                  | [1811.09600](https://arxiv.org/abs/1811.09600)                                                       |
| Trust Region (TR)                                                                       | White-box | Minimal | $\ell_2$, $\ell_\infty$                                   | [1812.06371](https://arxiv.org/abs/1812.06371)                                                       |
| Fast Adaptive Boundary (FAB)                                                            | White-box | Minimal | $\ell_1$, $\ell_2$, $\ell_\infty$                         | [1907.02044](https://arxiv.org/abs/1907.02044)                                                       |
| Perceptual Color distance Alternating Loss (PerC-AL)                                    | White-box | Minimal | CIEDE2000                                                 | [1911.02466](https://arxiv.org/abs/1911.02466)                                                       |
| Auto-PGD (APGD)                                                                         | White-box | Budget  | $\ell_1$, $\ell_2$, $\ell_\infty$                         | [2003.01690](https://arxiv.org/abs/2003.01690) <br /> [2103.01208](https://arxiv.org/abs/2103.01208) |
| **Augmented Lagrangian Method for Adversarial (ALMA)**                                  | White-box | Minimal | $\ell_1$, $\ell_2$, SSIM, CIEDE2000, LPIPS, ...           | [2011.11857](https://arxiv.org/abs/2011.11857)                                                       |
| Folded Gaussian Attack (FGA)<br /> Voting Folded Gaussian Attack (VFGA)                 | White-box | Minimal | $\ell_0$                                                  | [2011.12423](https://arxiv.org/abs/2011.12423)                                                       |
| Fast Minimum-Norm (FMN)                                                                 | White-box | Minimal | $\ell_0$, $\ell_1$, $\ell_2$, $\ell_\infty$               | [2102.12827](https://arxiv.org/abs/2102.12827)                                                       |
| Primal-Dual Gradient Descent (PDGD)<br /> Primal-Dual Proximal Gradient Descent (PDPGD) | White-box | Minimal | $\ell_2$<br />$\ell_0$, $\ell_1$, $\ell_2$, $\ell_\infty$ | [2106.01538](https://arxiv.org/abs/2106.01538)                                                       |
| SuperDeepFool (SDF)                                                                     | White-box | Minimal | $\ell_2$                                                  | [2303.12481](https://arxiv.org/abs/2303.12481)                                                       |
| σ-zero                                                                                  | White-box | Minimal | $\ell_0$                                                  | [2402.01879](https://arxiv.org/abs/2402.01879)                                                       |

**Bold** means that this repository contains the official implementation.

_Type_ refers to the goal of the attack:
 - _Minimal_ attacks aim to find the smallest adversarial perturbation w.r.t. a given distance;
 - _Budget_ attacks aim to find an adversarial perturbation within a distance budget (and often to maximize a loss as well).

#### Segmentation

The library now includes segmentation attacks in the `adv_lib.attacks.segmentation` module. These require the following arguments:
- `model`: the model  that produces logits (pre-softmax activations) with inputs in $[0, 1]$
- `inputs`: the images to attack in $[0, 1]$. Shape: $b\times c\times h\times w$ with $b$ the batch size, $c$ the number of color channels and $h$ and $w$ the height and width of the images.
- `labels`: either the ground-truth labels for the samples or the targets. Shape: $b\times h\times w$.
- `masks`: binary mask indicating which pixels to attack, to account for unlabeled pixels (e.g. void in Pascal VOC). Shape: $b\times h\times w$
- `targeted`: flag indicated if the attack should be targeted or not -- defaults to `False`
- `adv_threshold`: fraction of the pixels to consider an attack successful -- defaults to `0.99`

The following segmentation attacks are implemented:

| Name                                                                                      | Knowledge | Type    | Distance(s)                                               | ArXiv Link                                     |
|-------------------------------------------------------------------------------------------|-----------|---------|-----------------------------------------------------------|------------------------------------------------|
| Dense Adversary Generation (DAG)                                                          | White-box | Minimal | $\ell_2$, $\ell_\infty$                                   | [1703.08603](https://arxiv.org/abs/1703.08603) |
| Adaptive Segmentation Mask Attack (ASMA)                                                  | White-box | Minimal | $\ell_2$                                                  | [1907.13124](https://arxiv.org/abs/1907.13124) |
| _Primal-Dual Gradient Descent (PDGD)<br /> Primal-Dual Proximal Gradient Descent (PDPGD)_ | White-box | Minimal | $\ell_2$<br />$\ell_0$, $\ell_1$, $\ell_2$, $\ell_\infty$ | [2106.01538](https://arxiv.org/abs/2106.01538) |
| **ALMA prox**                                                                             | White-box | Minimal | $\ell_\infty$                                             | [2206.07179](https://arxiv.org/abs/2206.07179) |

_Italic_ indicates that the attack is unofficially adapted from the classification variant.

### Distances

The following distances are available in the utils `adv_lib.distances` module:
- Lp-norms
- SSIM https://ece.uwaterloo.ca/~z70wang/research/ssim/
- MS-SSIM https://ece.uwaterloo.ca/~z70wang/publications/msssim.html
- CIEDE2000 color difference http://www2.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf
- LPIPS https://arxiv.org/abs/1801.03924

## Contributions

Suggestions and contributions are welcome :) 

## Citation

If this library has been useful for your research, you can cite it using the "Cite this repository" button in the "About" section.
