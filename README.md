# adversarial-library

This library contains various resources related to adversarial attacks implemented in PyTorch. It is aimed towards researchers looking for implementations of state-of-the-art attacks.

The code was written to maximize efficiency (_e.g._ by preferring low level functions from PyTorch) while retaining simplicity (_e.g._ by avoiding abstractions). As a consequence, most of the library, and especially the attacks, is implemented using **pure functions** (whenever possible).

While focused on attacks, this library also provides several utilities related to adversarial attacks: distances (SSIM, CIEDE2000, LPIPS), visdom callback, projections, losses and helper functions. Most notably the function `run_attack` from `utils/attack_utils.py` performs an attack on a model given the inputs and labels, with fixed batch size, and reports complexity related metrics (run-time and forward/backward propagations).

### Dependencies

The goal of this library is to be up-to-date with newer versions of PyTorch so the dependencies are expected to be updated regularly (possibly resulting in breaking changes).

- pytorch>=1.7.0
- torchvision>=0.8.0
- tqdm>=4.48.0
- visdom>=0.1.8

### Installation

You can either install using:

```pip install git+https://github.com/jeromerony/adversarial-library```

Or you can clone the repo and run:

```python setup.py install```

Alternatively, you can install (after cloning) the library in editable mode:

```pip install -e .```

### Example
 For an example on how to use this library, you can look at this repo: https://github.com/jeromerony/augmented_lagrangian_adversarial_attacks

## Contents

### Attacks

Currently the following attacks are implemented in the `adv_lib.attacks` module:

| Name                                                   | Knowledge | Type    | Distance(s)                                               | ArXiv Link                       |
|--------------------------------------------------------|-----------|---------|-----------------------------------------------------------|----------------------------------|
| Carlini and Wagner (C&W)                               | White-box | Minimal | L<sub>2</sub>, L<sub>∞</sub>                              | [1608.04644](https://arxiv.org/abs/1608.04644) |
| Projected Gradient Descent (PGD)                       | White-box | Budget  | L<sub>∞</sub>                                             | [1706.06083](https://arxiv.org/abs/1706.06083) |
| **Decoupled Direction and Norm (DDN)**                 | White-box | Minimal | L<sub>2</sub>                                             | [1811.09600](https://arxiv.org/abs/1811.09600) |
| Trust Region (TR)                                      | White-box | Minimal | L<sub>2</sub>, L<sub>∞</sub>                              | [1812.06371](https://arxiv.org/abs/1812.06371) |
| Fast Adaptive Boundary (FAB)                           | White-box | Minimal | L<sub>1</sub>, L<sub>2</sub>, L<sub>∞</sub>               | [1907.02044](https://arxiv.org/abs/1907.02044) |
| Perceptual Color distance Alternating Loss (PerC-AL)   | White-box | Minimal | CIEDE2000                                                 | [1911.02466](https://arxiv.org/abs/1911.02466) |
| Auto-PGD (APGD)                                        | White-box | Budget  | L<sub>1</sub>, L<sub>2</sub>, L<sub>∞</sub>               | [2003.01690](https://arxiv.org/abs/2003.01690) <br /> [2103.01208](https://arxiv.org/abs/2103.01208) |
| **Augmented Lagrangian Method for Adversarial (ALMA)** | White-box | Minimal | L<sub>1</sub>, L<sub>2</sub>, SSIM, CIEDE2000, LPIPS, ... | [2011.11857](https://arxiv.org/abs/2011.11857) |
| Voting Folded Gaussian Attack (VFGA)                   | White-box | Minimal | L<sub>0</sub>                                             | [2011.12423](https://arxiv.org/abs/2011.12423) |
| Fast Minimum-Norm (FMN)                                | White-box | Minimal | L<sub>0</sub>, L<sub>1</sub>, L<sub>2</sub>, L<sub>∞</sub>| [2102.12827](https://arxiv.org/abs/2102.12827) |
| Primal-Dual Gradient Descent (PDGD)<br /> Primal-Dual Proximal Gradient Descent (PDPGD)| White-box | Minimal | L<sub>2</sub><br />L<sub>0</sub>, L<sub>1</sub>, L<sub>2</sub>, L<sub>∞</sub>| [2106.01538](https://arxiv.org/abs/2106.01538) |

**Bold** means that this repository contains the official implementation.

_Type_ refers to the goal of the attack:
 - _Minimal_ attacks aim to find the smallest adversarial perturbation w.r.t. a given distance;
 - _Budget_ attacks aim to find an adversarial perturbation within a distance budget (and often to maximize a loss as well).

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

If this library has been useful for your research, you can cite it as follows:

```bibtex
@misc{rony2020adversarial,
   title={Adversarial Library},
   author={Rony, J{\'e}r{\^o}me and {Ben Ayed}, Ismail},
   year={2020},
   url={https://github.com/jeromerony/adversarial-library}
}
```
