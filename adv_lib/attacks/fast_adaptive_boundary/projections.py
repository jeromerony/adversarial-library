import math

import torch
from torch import Tensor
from torch.nn import functional as F


def check_shape(x):
    return x if len(x.shape) > 0 else x.unsqueeze(0)


def original_projection_linf(points_to_project, w_hyperplane, b_hyperplane):
    device = points_to_project.device
    dtype = points_to_project.dtype
    t = points_to_project.clone()
    w = w_hyperplane.clone()
    b = b_hyperplane.clone()

    ind2 = ((w * t).sum(1) - b < 0).nonzero().squeeze()
    ind2 = check_shape(ind2)
    w[ind2] *= -1
    b[ind2] *= -1

    c5 = (w < 0).float()
    a = torch.ones(t.shape).to(device)
    d = (a * c5 - t) * (w != 0).float()
    a -= a * (1 - c5)

    p = torch.ones(t.shape).to(device) * c5 - t * (2 * c5 - 1)
    indp = torch.argsort(p, dim=1)

    b = b - (w * t).sum(1)
    b0 = (w * d).sum(1)
    b1 = b0.clone()

    counter = 0
    indp2 = indp.unsqueeze(-1).flip(dims=(1, 2)).squeeze()
    u = torch.arange(0, w.shape[0])
    ws = w[u.unsqueeze(1), indp2]
    bs2 = - ws * d[u.unsqueeze(1), indp2]

    s = torch.cumsum(ws.abs(), dim=1)
    sb = torch.cumsum(bs2, dim=1) + b0.unsqueeze(1)

    c = b - b1 > 0
    b2 = sb[u, -1] - s[u, -1] * p[u, indp[u, 0]]
    c_l = (b - b2 > 0).nonzero().squeeze()
    c2 = ((b - b1 > 0) * (b - b2 <= 0)).nonzero().squeeze()
    c_l = check_shape(c_l)
    c2 = check_shape(c2)

    lb = torch.zeros(c2.shape[0])
    ub = torch.ones(c2.shape[0]) * (w.shape[1] - 1)
    nitermax = torch.ceil(torch.log2(torch.tensor(w.shape[1]).float()))
    counter2 = torch.zeros(lb.shape).long()

    while counter < nitermax:
        counter4 = torch.floor((lb + ub) / 2)
        counter2 = counter4.long()
        indcurr = indp[c2, -counter2 - 1]
        b2 = sb[c2, counter2] - s[c2, counter2] * p[c2, indcurr]
        c = b[c2] - b2 > 0
        ind3 = c.nonzero().squeeze()
        ind32 = (~c).nonzero().squeeze()
        ind3 = check_shape(ind3)
        ind32 = check_shape(ind32)
        lb[ind3] = counter4[ind3]
        ub[ind32] = counter4[ind32]
        counter += 1

    lb = lb.long()
    counter2 = 0

    if c_l.nelement != 0:
        lmbd_opt = (torch.max((b[c_l] - sb[c_l, -1]) / (-s[c_l, -1]),
                              torch.zeros(sb[c_l, -1].shape, dtype=dtype)
                              .to(device))).unsqueeze(-1)
        d[c_l] = (2 * a[c_l] - 1) * lmbd_opt

    lmbd_opt = (torch.max((b[c2] - sb[c2, lb]) / (-s[c2, lb]),
                          torch.zeros(sb[c2, lb].shape, dtype=dtype)
                          .to(device))).unsqueeze(-1)
    d[c2] = torch.min(lmbd_opt, d[c2]) * c5[c2] \
            + torch.max(-lmbd_opt, d[c2]) * (1 - c5[c2])

    return d * (w != 0).float()


def original_projection_l2(points_to_project, w_hyperplane, b_hyperplane):
    device = points_to_project.device
    dtype = points_to_project.dtype
    t = points_to_project.clone()
    w = w_hyperplane.clone()
    b = b_hyperplane.clone()

    c = (w * t).sum(1) - b
    ind2 = (c < 0).nonzero().squeeze()
    ind2 = check_shape(ind2)
    w[ind2] *= -1
    c[ind2] *= -1

    u = torch.arange(0, w.shape[0]).unsqueeze(1)

    r = torch.max(t / w, (t - 1) / w)
    u2 = torch.ones(r.shape, dtype=dtype).to(device)
    r = torch.min(r, 1e12 * u2)
    r = torch.max(r, -1e12 * u2)
    r[w.abs() < 1e-8] = 1e12
    r[r == -1e12] = -r[r == -1e12]
    rs, indr = torch.sort(r, dim=1)
    rs2 = torch.cat((rs[:, 1:],
                     torch.zeros(rs.shape[0], 1).to(device)), 1)
    rs[rs == 1e12] = 0
    rs2[rs2 == 1e12] = 0

    w3 = w ** 2
    w3s = w3[u, indr]
    w5 = w3s.sum(dim=1, keepdim=True)
    ws = w5 - torch.cumsum(w3s, dim=1)
    d = -(r * w).clone()
    d = d * (w.abs() > 1e-8).float()
    s = torch.cat(((-w5.squeeze() * rs[:, 0]).unsqueeze(1),
                   torch.cumsum((-rs2 + rs) * ws, dim=1) -
                   w5 * rs[:, 0].unsqueeze(-1)), 1)

    c4 = (s[:, 0] + c < 0)
    c3 = ((d * w).sum(dim=1) + c > 0)
    c6 = c4.nonzero().squeeze()
    c2 = ((1 - c4.float()) * (1 - c3.float())).nonzero().squeeze()
    c6 = check_shape(c6)
    c2 = check_shape(c2)

    counter = 0
    lb = torch.zeros(c2.shape[0])
    ub = torch.ones(c2.shape[0]) * (w.shape[1] - 1)
    nitermax = torch.ceil(torch.log2(torch.tensor(w.shape[1]).float()))
    counter2 = torch.zeros(lb.shape).long()

    while counter < nitermax:
        counter4 = torch.floor((lb + ub) / 2)
        counter2 = counter4.long()
        c3 = s[c2, counter2] + c[c2] > 0
        ind3 = c3.nonzero().squeeze()
        ind32 = (~c3).nonzero().squeeze()
        ind3 = check_shape(ind3)
        ind32 = check_shape(ind32)
        lb[ind3] = counter4[ind3]
        ub[ind32] = counter4[ind32]
        counter += 1

    lb = lb.long()
    alpha = torch.zeros([1])

    if c6.nelement() != 0:
        alpha = c[c6] / w5[c6].squeeze(-1)
        d[c6] = -alpha.unsqueeze(-1) * w[c6]

    if c2.nelement() != 0:
        alpha = (s[c2, lb] + c[c2]) / ws[c2, lb] + rs[c2, lb]
        if torch.sum(ws[c2, lb] == 0) > 0:
            ind = (ws[c2, lb] == 0).nonzero().squeeze().long()
            ind = check_shape(ind)
            alpha[ind] = 0
        c5 = (alpha.unsqueeze(-1) > r[c2]).float()
        d[c2] = d[c2] * c5 - alpha.unsqueeze(-1) * w[c2] * (1 - c5)

    return d * (w.abs() > 1e-8).float()


def original_projection_l1(points_to_project, w_hyperplane, b_hyperplane):
    device = points_to_project.device
    dtype = points_to_project.dtype
    t = points_to_project.clone()
    w = w_hyperplane.clone()
    b = b_hyperplane.clone()

    c = (w * t).sum(1) - b
    ind2 = (c < 0).nonzero().squeeze()
    ind2 = check_shape(ind2)
    w[ind2] *= -1
    c[ind2] *= -1

    r = torch.max(1 / w, -1 / w)
    r = torch.min(r, 1e12 * torch.ones(r.shape, dtype=dtype).to(device))
    rs, indr = torch.sort(r, dim=1)
    _, indr_rev = torch.sort(indr)

    u = torch.arange(0, w.shape[0]).unsqueeze(1)
    u2 = torch.arange(0, w.shape[1]).repeat(w.shape[0], 1)
    c6 = (w < 0).float()
    d = (-t + c6) * (w != 0).float()
    d2 = torch.min(-w * t, w * (1 - t))
    ds = d2[u, indr]
    ds2 = torch.cat((c.unsqueeze(-1), ds), 1)
    s = torch.cumsum(ds2, dim=1)

    c4 = s[:, -1] < 0
    c2 = c4.nonzero().squeeze(-1)
    c2 = check_shape(c2)

    counter = 0
    lb = torch.zeros(c2.shape[0])
    ub = torch.ones(c2.shape[0]) * (s.shape[1])
    nitermax = torch.ceil(torch.log2(torch.tensor(s.shape[1]).float()))
    counter2 = torch.zeros(lb.shape).long()

    while counter < nitermax:
        counter4 = torch.floor((lb + ub) / 2)
        counter2 = counter4.long()
        c3 = s[c2, counter2] > 0
        ind3 = c3.nonzero().squeeze()
        ind32 = (~c3).nonzero().squeeze()
        ind3 = check_shape(ind3)
        ind32 = check_shape(ind32)
        lb[ind3] = counter4[ind3]
        ub[ind32] = counter4[ind32]
        counter += 1

    lb2 = lb.long()

    if c2.nelement() != 0:
        alpha = -s[c2, lb2] / w[c2, indr[c2, lb2]]
        c5 = u2[c2].float() < lb.unsqueeze(-1).float()
        u3 = c5[u[:c5.shape[0]], indr_rev[c2]]
        d[c2] = d[c2] * u3.float().to(device)
        d[c2, indr[c2, lb2]] = alpha

    return d * (w.abs() > 1e-8).float()


def projection_l1(points_to_project: Tensor, w_hyperplane: Tensor, b_hyperplane: Tensor) -> Tensor:
    device = points_to_project.device
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane

    c = (w * t).sum(1) - b
    ind2 = c < 0
    w[ind2] *= -1
    c[ind2] *= -1

    r = torch.max(1 / w, -1 / w).clamp_max(1e12)
    indr = torch.argsort(r, dim=1)
    indr_rev = torch.argsort(indr)

    u = torch.arange(0, w.shape[0], device=device).unsqueeze(1)
    u2 = torch.arange(0, w.shape[1], device=device).repeat(w.shape[0], 1)
    c6 = (w < 0).float()
    d = (-t + c6) * (w != 0).float()
    ds = torch.min(-w * t, w * (1 - t)).gather(1, indr)
    ds2 = torch.cat((c.unsqueeze(-1), ds), 1)
    s = torch.cumsum(ds2, dim=1)

    c2 = s[:, -1] < 0

    lb = torch.zeros(c2.sum(), device=device)
    ub = torch.full_like(lb, s.shape[1])
    nitermax = math.ceil(math.log2(w.shape[1]))

    s_ = s[c2]
    for counter in range(nitermax):
        counter4 = torch.floor((lb + ub) / 2)
        counter2 = counter4.long().unsqueeze(1)
        c3 = s_.gather(1, counter2).squeeze(1) > 0
        lb = torch.where(c3, counter4, lb)
        ub = torch.where(c3, ub, counter4)

    lb2 = lb.long()

    if c2.any():
        alpha = -s[c2, lb2] / w[c2, indr[c2, lb2]]
        c5 = u2[c2].float() < lb.unsqueeze(-1).float()
        u3 = c5[u[:c5.shape[0]], indr_rev[c2]]
        d[c2] = d[c2] * u3.float()
        d[c2, indr[c2, lb2]] = alpha

    return d * (w.abs() > 1e-8).float()


def projection_l2(points_to_project: Tensor, w_hyperplane: Tensor, b_hyperplane: Tensor) -> Tensor:
    device = points_to_project.device
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane

    c = (w * t).sum(1) - b
    ind2 = c < 0
    w[ind2] *= -1
    c[ind2] *= -1

    r = torch.max(t / w, (t - 1) / w).clamp(min=-1e12, max=1e12)
    r[w.abs() < 1e-8] = 1e12
    r[r == -1e12] *= -1
    rs, indr = torch.sort(r, dim=1)
    rs2 = F.pad(rs[:, 1:], (0, 1))
    rs[rs == 1e12] = 0
    rs2[rs2 == 1e12] = 0

    w3s = (w ** 2).gather(1, indr)
    w5 = w3s.sum(dim=1, keepdim=True)
    ws = w5 - torch.cumsum(w3s, dim=1)
    d = -(r * w)
    d.mul_((w.abs() > 1e-8).float())
    s = torch.cat(((-w5.squeeze() * rs[:, 0]).unsqueeze(1),
                   torch.cumsum((-rs2 + rs) * ws, dim=1) - w5 * rs[:, 0].unsqueeze(-1)), 1)

    c4 = s[:, 0] + c < 0
    c3 = (d * w).sum(dim=1) + c > 0
    c2 = ~c4 & ~c3

    lb = torch.zeros(c2.sum(), device=device)
    ub = torch.full_like(lb, w.shape[1] - 1)
    nitermax = math.ceil(math.log2(w.shape[1]))

    s_, c_ = s[c2], c[c2]
    for counter in range(nitermax):
        counter4 = torch.floor((lb + ub) / 2)
        counter2 = counter4.long().unsqueeze(1)
        c3 = s_.gather(1, counter2).squeeze(1) + c_ > 0
        lb = torch.where(c3, counter4, lb)
        ub = torch.where(c3, ub, counter4)

    lb = lb.long()

    if c4.any():
        alpha = c[c4] / w5[c4].squeeze(-1)
        d[c4] = -alpha.unsqueeze(-1) * w[c4]

    if c2.any():
        alpha = (s[c2, lb] + c[c2]) / ws[c2, lb] + rs[c2, lb]
        alpha[ws[c2, lb] == 0] = 0
        c5 = (alpha.unsqueeze(-1) > r[c2]).float()
        d[c2] = d[c2] * c5 - alpha.unsqueeze(-1) * w[c2] * (1 - c5)

    return d * (w.abs() > 1e-8).float()


def projection_linf(points_to_project: Tensor, w_hyperplane: Tensor, b_hyperplane: Tensor) -> Tensor:
    device = points_to_project.device
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane.clone()

    sign = 2 * ((w * t).sum(1) - b >= 0) - 1
    w.mul_(sign.unsqueeze(1))
    b.mul_(sign)

    a = (w < 0).float()
    d = (a - t) * (w != 0).float()

    p = a - t * (2 * a - 1)
    indp = torch.argsort(p, dim=1)

    b = b - (w * t).sum(1)
    b0 = (w * d).sum(1)

    indp2 = indp.flip((1,))
    ws = w.gather(1, indp2)
    bs2 = - ws * d.gather(1, indp2)

    s = torch.cumsum(ws.abs(), dim=1)
    sb = torch.cumsum(bs2, dim=1) + b0.unsqueeze(1)

    b2 = sb[:, -1] - s[:, -1] * p.gather(1, indp[:, 0].unsqueeze(1)).squeeze(1)
    c_l = b - b2 > 0
    c2 = (b - b0 > 0) & (b - b2 <= 0)
    lb = torch.zeros(c2.sum(), device=device)
    ub = torch.full_like(lb, w.shape[1] - 1)
    nitermax = math.ceil(math.log2(w.shape[1]))

    indp_, sb_, s_, p_, b_ = indp[c2], sb[c2], s[c2], p[c2], b[c2]
    for counter in range(nitermax):
        counter4 = torch.floor((lb + ub) / 2)

        counter2 = counter4.long().unsqueeze(1)
        indcurr = indp_.gather(1, indp_.size(1) - 1 - counter2)
        b2 = (sb_.gather(1, counter2) - s_.gather(1, counter2) * p_.gather(1, indcurr)).squeeze(1)
        c = b_ - b2 > 0

        lb = torch.where(c, counter4, lb)
        ub = torch.where(c, ub, counter4)

    lb = lb.long()

    if c_l.any():
        lmbd_opt = torch.clamp_min((b[c_l] - sb[c_l, -1]) / (-s[c_l, -1]), min=0).unsqueeze(-1)
        d[c_l] = (2 * a[c_l] - 1) * lmbd_opt

    lmbd_opt = torch.clamp_min((b[c2] - sb[c2, lb]) / (-s[c2, lb]), min=0).unsqueeze(-1)
    d[c2] = torch.min(lmbd_opt, d[c2]) * a[c2] + torch.max(-lmbd_opt, d[c2]) * (1 - a[c2])

    return d * (w != 0).float()


if __name__ == '__main__':
    device = torch.device('cuda:0')
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    n, d = 256, 1024 * 128
    dtype = torch.float

    x = torch.rand(n * 2, d, device=device, dtype=dtype)
    w = torch.randn(n, d, device=device, dtype=dtype).repeat(2, 1)
    b = torch.randn(n, device=device, dtype=dtype).repeat(2)

    orig_l1, l1 = original_projection_l1(x, w, b), projection_l1(x, w, b)
    diff_l1 = orig_l1 - l1
    print('l1 relative: {:.3g} - abs: {:.3g}'.format(diff_l1.div(orig_l1 + 1e-12).abs().max(), diff_l1.abs().max()))
    orig_l2, l2 = original_projection_l2(x, w, b), projection_l2(x, w, b)
    diff_l2 = orig_l2 - l2
    print('l2 relative: {:.3g} - abs: {:.3g}'.format(diff_l2.div(orig_l2 + 1e-12).abs().max(), diff_l2.abs().max()))
    orig_linf, linf = original_projection_linf(x, w, b), projection_linf(x, w, b)
    diff_linf = orig_linf - linf
    print('linf relative: {:.3g} - abs: {:.3g}'.format(diff_linf.div(orig_linf + 1e-12).abs().max(), diff_linf.abs().max()))

    projections = {
        'orig_l1': original_projection_l1,
        'orig_l2': original_projection_l2,
        'orig_linf': original_projection_linf,
        'l1': projection_l1,
        'l2': projection_l2,
        'linf': projection_linf,
    }
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for name, projection in projections.items():
        start.record()
        out = projection(x, w, b)
        end.record()
        torch.cuda.synchronize()
        time = (start.elapsed_time(end)) / 1000

        print('Time for {}: {:.3g}s'.format(name, time))