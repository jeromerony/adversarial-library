import matplotlib.pyplot as plt
import torch
from cycler import cycler
from torch.autograd import grad

from adv_lib.utils.lagrangian_penalties.all_penalties import all_penalties

fig, ax = plt.subplots(figsize=(12, 12))
x = torch.linspace(-10, 5, 15001, requires_grad=True)
x[10000] = 0

styles = (cycler(linestyle=['-', '--', '-.']) * cycler(color=plt.rcParams['axes.prop_cycle']))
ax.set_prop_cycle(styles)

unique_penalties_idxs = list(range(len(all_penalties)))

for i in unique_penalties_idxs:
    penalty = list(all_penalties.keys())[i]
    ρ = torch.tensor(1.)
    μ = torch.tensor(1.)
    y = all_penalties[penalty](x, ρ, μ)
    if y.isnan().any():
        print('nan in y for {}'.format(penalty))
    grads = grad(y.sum(), x, only_inputs=True)[0]
    if grads.isnan().any():
        print('nan in grad for {}'.format(penalty))
    if not torch.allclose(grads[10000], μ):
        print("P'(0, ρ, μ)  = {:3g} = μ".format(grads[10000].item()))
    ax.plot(x.detach().numpy(), y.detach().numpy(),
            label=r'{}: $\nabla P(0)$:{:.3g}'.format(penalty, grads[10000].item()))

ax.set_xlim(-10, 5)
ax.set_ylim(-5, 10)
ax.legend(loc=2, prop={'size': 6})
ax.set_aspect('equal')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$Penalty(x)$')
ax.grid(True, linestyle='--')

plt.tight_layout()
plt.show()
