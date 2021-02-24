import torch

def complex_div(t1, t2, eps=1e-8):
    assert t1.size() == t2.size()
    assert t1.size()[-1] == 2
    assert t1.device == t2.device
    t1re = torch.index_select(t1, -1, torch.tensor([0], device=t1.device))
    t1im = torch.index_select(t1, -1, torch.tensor([1], device=t1.device))
    t2re = torch.index_select(t2, -1, torch.tensor([0], device=t1.device))
    t2im = torch.index_select(t2, -1, torch.tensor([1], device=t1.device))
    denominator = t2re ** 2 + t2im ** 2 + eps
    numeratorRe = t1re * t2re + t1im * t2im
    numeratorIm = t1im * t2re - t1re * t2im
    return torch.cat([numeratorRe / denominator, numeratorIm / denominator], -1)


def complex_mul(t1, t2):
    assert t1.size() == t2.size()
    assert t1.size()[-1] == 2
    assert t1.device == t2.device
    t1re = torch.index_select(t1, -1, torch.tensor([0], device=t1.device))
    t1im = torch.index_select(t1, -1, torch.tensor([1], device=t1.device))
    t2re = torch.index_select(t2, -1, torch.tensor([0], device=t1.device))
    t2im = torch.index_select(t2, -1, torch.tensor([1], device=t1.device))
    return torch.cat([t1re * t2re - t1im * t2im, t1re * t2im + t1im * t2re], -1)


def complex_conj(iT):
    assert iT.size()[-1] == 2
    iTre = torch.index_select(iT, -1, torch.tensor([0], device=iT.device))
    iTim = torch.index_select(iT, -1, torch.tensor([1], device=iT.device))
    return torch.cat([iTre, -iTim], -1)


def complex_abs(iT):
    assert iT.size()[-1] == 2
    iTre = torch.index_select(iT, -1, torch.tensor([0], device=iT.device))
    iTim = torch.index_select(iT, -1, torch.tensor([1], device=iT.device))
    outR = torch.sqrt(iTre ** 2 + iTim ** 2 + 1e-8)
    return torch.cat([outR, torch.zeros_like(outR)], -1)


