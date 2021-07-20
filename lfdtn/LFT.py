from torch.functional import F
import torch.nn as nn
from asset.utils import getPhaseAdd,dmsg
from lfdtn.custom_fft import custom_ifft, custom_fft


def fold(T, res_y: int, res_x: int, y_stride: int, x_stride: int, cell_size: int, pad_size: int):
    fold_params = dict(kernel_size=(cell_size, cell_size), padding=(pad_size, pad_size), stride=(y_stride, x_stride))
    return nn.functional.fold(T, output_size=(res_y, res_x), **fold_params)


def extract_local_windows(batch, windowSize: int, y_stride: int = 1, x_stride: int = 1, padding: int = 0):
    batchSize = batch.shape[0]

    windowSizeAdjust = windowSize + 2 * padding

    ###this is the image padding, currently assumes x_stride == y_stride
    padSize = int(x_stride * ((windowSizeAdjust - 1) // x_stride))

    fold_params = dict(kernel_size=(windowSizeAdjust, windowSizeAdjust), padding=(padSize, padSize),
                       stride=(y_stride, x_stride))
    # print('image padded2 shape:', imgPadded.shape)
    result = nn.functional.unfold(batch, **fold_params).permute(0, 2, 1)
    # print('result new fold shape:', result.shape)
    return result.reshape(batchSize, -1, windowSizeAdjust, windowSizeAdjust) #DONOT remove this contiguous, otherwise you get warning:https://github.com/pytorch/pytorch/issues/42300


def LFT(batch, window, y_stride: int = 1, x_stride: int = 1, padding: int = 1):
    windowBatch = extract_local_windows(batch, window.shape[0], y_stride=y_stride, x_stride=x_stride, padding=padding)
    windowPadded = F.pad(window, (padding, padding, padding, padding))
    localImageWindowsSmoothedPadded = windowBatch * windowPadded

    return custom_fft(localImageWindowsSmoothedPadded)


def iLFT(stft2D_result, window, T, res_y: int, res_x: int, y_stride: int = 1, x_stride: int = 1, padding: int = 1,
         eps: float = 1e-8,is_inp_complex=True,move_window_according_T=True,channels=1):
    batchSize = stft2D_result.shape[0]
    cellSize = window.shape[0]

    ###this is the image padding, currently assumes x_stride == y_stride
    padSize = int(x_stride * ((cellSize - 1) // x_stride))

    cellSizeAdjust = cellSize + 2 * padding
    padSizeAdjust = int(x_stride * ((cellSizeAdjust - 1) // x_stride))

    ###the number of extracted cells along x and y axis
    ###this should be general enough to hold for different padSize
    num_windows_y = (res_y + 2 * padSizeAdjust - cellSizeAdjust) // y_stride + 1
    num_windows_x = (res_x + 2 * padSizeAdjust - cellSizeAdjust) // x_stride + 1
    num_windows_total = num_windows_y * num_windows_x

    if is_inp_complex:
        ifft_result = custom_ifft(stft2D_result)
    else:
        ifft_result = stft2D_result.clone()

    ifft_result = ifft_result.view((batchSize,channels, num_windows_y, num_windows_x, cellSizeAdjust, cellSizeAdjust))

    window_big = F.pad(window, (padding, padding, padding, padding), value=0.0)
    window_big = window_big.expand(batchSize,channels, num_windows_total, -1, -1)

    if move_window_according_T:
        window_big_Complex = custom_fft(window_big)
        window_big_Complex = getPhaseAdd(window_big_Complex, T.view_as(window_big_Complex))
        window_big = custom_ifft(window_big_Complex)

    window_big = window_big.view(batchSize,channels, num_windows_y, num_windows_x, window_big.shape[3], window_big.shape[4])

    ifft_result *= window_big

    ifft_result = ifft_result.reshape(batchSize, -1, channels*ifft_result.shape[4] * ifft_result.shape[5]).permute(0, 2, 1)
    test = fold(ifft_result, \
                res_y=res_y, res_x=res_x, y_stride=y_stride, x_stride=x_stride, cell_size=cellSizeAdjust,
                pad_size=padSizeAdjust)

    window_big = (window_big ** 2).reshape(batchSize, -1, channels*window_big.shape[4] * window_big.shape[5]).permute(0, 2, 1)
    windowTracker = fold(window_big, \
                         res_y=res_y, res_x=res_x, y_stride=y_stride, x_stride=x_stride, cell_size=cellSizeAdjust,
                         pad_size=padSizeAdjust)
                        

    windowTracker += eps
    weighted_result = test / windowTracker
    return weighted_result, windowTracker

def compact_LFT(batch, window, config):
    return LFT(batch, window, x_stride=config.stride, y_stride=config.stride, padding=config.window_padding_constrained)

def compact_iLFT(LFT_result, window, T, config,is_inp_complex=True,move_window_according_T=True,channel=1):
    return iLFT(LFT_result, window, T, res_y=config.res_y_constrained, res_x=config.res_x_constrained, y_stride=config.stride,\
                    x_stride=config.stride, padding=config.window_padding_constrained,is_inp_complex=is_inp_complex,move_window_according_T=move_window_according_T,channels=channel)