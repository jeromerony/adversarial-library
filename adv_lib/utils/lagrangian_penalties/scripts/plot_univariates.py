from inspect import isfunction

import matplotlib.pyplot as plt
import torch
from cycler import cycler

from adv_lib.utils.lagrangian_penalties import univariate_functions

fig, ax = plt.subplots(figsize=(8, 8))
x = torch.linspace(-10, 5, 15001)

styles = (cycler(linestyle=['-', '--']) * cycler(color=plt.rcParams['axes.prop_cycle']))
ax.set_prop_cycle(styles)

for univariate in univariate_functions.__all__:
    if isfunction(univariate_functions.__dict__[univariate]):
        univariate_function = univariate_functions.__dict__[univariate]
    else:
        univariate_function = univariate_functions.__dict__[univariate]()
    y = univariate_function(x)
    ax.plot(x, y, label=univariate)

ax.set_xlim(-10, 5)
ax.set_ylim(-5, 10)
ax.legend(loc=2)
ax.set_aspect('equal')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$Univariate(x)$')
ax.grid(True, linestyle='--')

plt.tight_layout()
plt.show()
