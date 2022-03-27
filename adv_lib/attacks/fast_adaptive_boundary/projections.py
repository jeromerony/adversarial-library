import math

import torch
from torch import Tensor
from torch.nn import functional as F


def projection_l1(points_to_project: Tensor, w_hyperplane: Tensor, b_hyperplane: Tensor) -> Tensor:
    device = points_to_project.device
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane

    c = (w * t).sum(1).sub_(b)
    ind2 = (c >= 0).float().mul_(2).sub_(1)
    w.mul_(ind2.unsqueeze(1))
    c.mul_(ind2)

    w_abs = w.abs()
    r = (1 / w_abs).clamp_(max=1e12)
    indr = torch.argsort(r, dim=1)
    indr_rev = torch.argsort(indr)

    d = (w < 0).float().sub_(t).mul_(w != 0)
    ds = torch.min(-w * t, (1 - t).mul_(w)).gather(1, indr)
    ds2 = torch.cat((c.unsqueeze(-1), ds), 1)
    s = torch.cumsum(ds2, dim=1)

    c2 = s[:, -1] < 0

    lb = torch.zeros(c2.sum(), device=device)
    ub = torch.full_like(lb, s.shape[1])
    nitermax = math.ceil(math.log2(w.shape[1]))

    s_ = s[c2]
    for counter in range(nitermax):
        counter4 = (lb + ub).mul_(0.5).floor_()
        counter2 = counter4.long().unsqueeze(1)
        c3 = s_.gather(1, counter2).squeeze(1) > 0
        lb = torch.where(c3, counter4, lb)
        ub = torch.where(c3, ub, counter4)

    lb2 = lb.long()

    if c2.any():
        indr = indr[c2].gather(1, lb2.unsqueeze(1)).squeeze(1)
        u = torch.arange(0, w.shape[0], device=device).unsqueeze(1)
        u2 = torch.arange(0, w.shape[1], device=device, dtype=torch.float).unsqueeze(0)
        alpha = s[c2, lb2].neg().div_(w[c2, indr])
        c5 = u2 < lb.unsqueeze(-1)
        u3 = c5[u[:c5.shape[0]], indr_rev[c2]]
        d[c2] *= u3
        d[c2, indr] = alpha

    return d.mul_(w_abs > 1e-8)


def projection_l2(points_to_project: Tensor, w_hyperplane: Tensor, b_hyperplane: Tensor) -> Tensor:
    device = points_to_project.device
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane

    c = (w * t).sum(1).sub_(b)
    ind2 = (c >= 0).float().mul_(2).sub_(1)
    w.mul_(ind2.unsqueeze(1))
    w_nonzero = w.abs() > 1e-8
    c.mul_(ind2)

    r = torch.maximum(t / w, (t - 1).div_(w)).clamp_(min=-1e12, max=1e12)
    r.masked_fill_(~w_nonzero, 1e12)
    r[r == -1e12] *= -1
    rs, indr = torch.sort(r, dim=1)
    rs2 = F.pad(rs[:, 1:], (0, 1))
    rs.masked_fill_(rs == 1e12, 0)
    rs2.masked_fill_(rs2 == 1e12, 0)

    w3s = w.square().gather(1, indr)
    w5 = w3s.sum(dim=1, keepdim=True)
    ws = w5 - torch.cumsum(w3s, dim=1)
    d = (r * w).neg_()
    d.mul_(w_nonzero)
    s = torch.cat((-w5 * rs[:, 0:1], torch.cumsum((rs - rs2).mul_(ws), dim=1).sub_(w5 * rs[:, 0:1])), 1)

    c4 = s[:, 0] + c < 0
    c3 = (d * w).sum(dim=1).add_(c) > 0
    c2 = ~(c4 | c3)

    lb = torch.zeros(c2.sum(), device=device)
    ub = torch.full_like(lb, w.shape[1] - 1)
    nitermax = math.ceil(math.log2(w.shape[1]))

    s_, c_ = s[c2], c[c2]
    for counter in range(nitermax):
        counter4 = (lb + ub).mul_(0.5).floor_()
        counter2 = counter4.long().unsqueeze(1)
        c3 = s_.gather(1, counter2).squeeze(1).add_(c_) > 0
        lb = torch.where(c3, counter4, lb)
        ub = torch.where(c3, ub, counter4)

    lb = lb.long()

    if c4.any():
        alpha = c[c4] / w5[c4].squeeze(-1)
        d[c4] = -alpha.unsqueeze(-1) * w[c4]

    if c2.any():
        alpha = (s[c2, lb] + c[c2]).div_(ws[c2, lb]).add_(rs[c2, lb])
        alpha[ws[c2, lb] == 0] = 0
        c5 = alpha.unsqueeze(-1) > r[c2]
        d[c2] = (d[c2] * c5).sub_((~c5).float().mul_(alpha.unsqueeze(-1)).mul_(w[c2]))

    return d.mul_(w_nonzero)


def projection_linf(points_to_project: Tensor, w_hyperplane: Tensor, b_hyperplane: Tensor) -> Tensor:
    device = points_to_project.device
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane.clone()

    sign = ((w * t).sum(1).sub_(b) >= 0).float().mul_(2).sub_(1)
    w.mul_(sign.unsqueeze(1))
    b.mul_(sign)

    a = (w < 0).float()
    d = (a - t).mul_(w != 0)

    p = (2 * a).sub_(1).mul_(t).neg_().add_(a)
    indp = torch.argsort(p, dim=1)

    b.sub_((w * t).sum(1))
    b0 = (w * d).sum(1)

    indp2 = indp.flip((1,))
    ws = w.gather(1, indp2)
    bs2 = -ws * d.gather(1, indp2)

    s = torch.cumsum(ws.abs_(), dim=1)
    sb = torch.cumsum(bs2, dim=1).add_(b0.unsqueeze(1))

    b2 = sb[:, -1] - s[:, -1] * p.gather(1, indp[:, 0:1]).squeeze(1)
    c_l = b - b2 > 0
    c2 = (b - b0 > 0) & (~c_l)
    lb = torch.zeros(c2.sum(), device=device)
    ub = torch.full_like(lb, w.shape[1] - 1)
    nitermax = math.ceil(math.log2(w.shape[1]))

    indp_, sb_, s_, p_, b_ = indp[c2], sb[c2], s[c2], p[c2], b[c2]
    for counter in range(nitermax):
        counter4 = (lb + ub).mul_(0.5).floor_()

        counter2 = counter4.long().unsqueeze(1)
        indcurr = indp_.gather(1, indp_.size(1) - 1 - counter2)
        b2 = sb_.gather(1, counter2).sub_(s_.gather(1, counter2).mul_(p_.gather(1, indcurr))).squeeze(1)
        c = b_ - b2 > 0

        lb = torch.where(c, counter4, lb)
        ub = torch.where(c, ub, counter4)

    lb = lb.long()

    if c_l.any():
        lmbd_opt = (b[c_l] - sb[c_l, -1]).div_(-s[c_l, -1]).clamp_(min=0).unsqueeze_(-1)
        d[c_l] = (2 * a[c_l]).sub_(1).mul_(lmbd_opt)

    lmbd_opt = (b[c2] - sb[c2, lb]).div_(-s[c2, lb]).clamp_(min=0).unsqueeze_(-1)
    d[c2] = torch.minimum(lmbd_opt, d[c2]).mul_(a[c2]).add_(torch.maximum(-lmbd_opt, d[c2]).mul_(1 - a[c2]))

    return d.mul_(w != 0)
