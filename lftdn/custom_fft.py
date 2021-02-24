import torch
from asset.utils import normalizeFFT

def custom_ifft(T, real=True):
    if real:
        return torch.ifft(T, signal_ndim=2, normalized=normalizeFFT)[...,0]
    else:
        return torch.ifft(T, signal_ndim=2, normalized=normalizeFFT)

def custom_fft(iT, real=True):
    ###make complex again --- TEMP: use complex signal
    if real:
        iTC = torch.stack([iT, torch.zeros_like(iT,requires_grad=False)], dim=-1)
        return torch.fft(iTC, signal_ndim=2, normalized=normalizeFFT)
    ###convert to modified transformation
    else:
        return torch.fft(iT, signal_ndim=2, normalized=normalizeFFT)


# def custom_ifft(T, real=True):
#     if real:
#         return torch.irfft(T, signal_ndim=2, normalized=normalizeFFT,onesided=False)
#     else:
#         return torch.ifft(T, signal_ndim=2, normalized=normalizeFFT)
#
# def custom_fft(iT, real=True):
#     ###make complex again --- TEMP: use complex signal
#     if real:
#         return torch.rfft(iT, signal_ndim=2, normalized=normalizeFFT,onesided=False)
#     ###convert to modified transformation
#     else:
#         return torch.fft(iT, signal_ndim=2, normalized=normalizeFFT)
