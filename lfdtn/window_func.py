import torch
import math

###binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

###outer product window from Nth level of Pascals triangle
def get_pascal_window(N):
    w = torch.tensor([binom(N-1,i) for i in range(N)], dtype=torch.float32).view((-1,1))
    w = w/w.max()
    return w @ w.T

###Gaussian required for approximate confined Gaussian tapering windowing
def gHelper(x, N, sigma):
    return torch.exp(-torch.pow(x - (N - 1) * 0.5, 2) / (4 * torch.pow(N*sigma, 2)))

###Approx. conf. Gaussian window, see
###https://www.researchgate.net/publication/261717241_Discrete-time_windows_with_minimal_RMS_bandwidth_for_given_RMS_temporal_width
def confinedGaussian1D(k, windowSize, sigma):
    const1 = gHelper(torch.tensor(-0.5).to(sigma.device), windowSize, sigma)
    const2 = gHelper(torch.tensor(-0.5).to(sigma.device) - windowSize, windowSize, sigma)
    const3 = gHelper(torch.tensor(-0.5).to(sigma.device) + windowSize, windowSize, sigma)
    denom = gHelper(k + windowSize, windowSize, sigma) + gHelper(k - windowSize, windowSize, sigma)
    return gHelper(k, windowSize, sigma) - const1 * denom / (const2 + const3)

###2D outer product version of the above window
def get_ACGW(windowSize, sigma):
    window_1D = confinedGaussian1D(torch.arange(windowSize).view(-1,1).to(sigma.device), windowSize, sigma)
    return window_1D @ window_1D.T

###returns a classic 2D Gaussian window
###note: renamed from return2DGaussian()
def get_2D_Gaussian(resolution, sigma, offset=0, normalized=False):
    kernel_size = resolution

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size).to(sigma.device).float()
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size).float()
    y_grid = x_grid.t().float()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = ((kernel_size - 1) + offset) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2. * variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    if normalized:
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    return gaussian_kernel.reshape(1, 1, kernel_size, kernel_size)
