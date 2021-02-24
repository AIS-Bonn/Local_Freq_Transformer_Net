import datetime
import io
import json
import math
import os
import re
from datetime import datetime
from math import exp
import imageio
import cv2
import ipykernel
import kornia
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import wandb
from IPython.display import display
from notebook.notebookapp import list_running_servers
from torch.autograd import Variable
from torch.functional import F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import time
import pandas as pd
from lftdn.complex_ops import complex_div, complex_abs, complex_conj, complex_mul

wandbLog = {}
wandbC = 0

augmentData=False

def getIfAugmentData():
    global augmentData
    return augmentData

def setIfAugmentData(inp):
    global augmentData
    augmentData=inp

def wandblog(d, commit=False):
    global wandbLog, wandbC,junckFiles
    wandbLog.update(d)
    if commit:
        wandb.log(wandbLog, step=wandbC,commit=True)
        wandbC += 1
        wandbLog = {}
        clean_junk_files()


torch.pi = torch.acos(torch.zeros(1)).item() * 2

try:
    TOKEN = "mytoken"

    base_url = next(list_running_servers())['url']
    r = requests.get(
        url=base_url + 'api/sessions',
        headers={'Authorization': 'token {}'.format(TOKEN), })

    r.raise_for_status()
    response = r.json()

    kernel_id = re.search('kernel-(.*).json', ipykernel.connect.get_connection_file()).group(1)
    theNotebook = ({r['kernel']['id']: r['notebook']['path'] for r in response}[kernel_id]).split("/")[-1].replace(
        ".ipynb", "")
except:
    theNotebook = "Untitled"

dimension = 2
li = 0;
ui = li + 1
EPS = 0.000000001
BIGNUMBER = 10e10
normalizeFFT = False

NOTVALID = cv2.resize(cv2.imread('asset/notValid.jpg')[:, :, 1], dsize=(128 - 2, 128 - 2),
                      interpolation=cv2.INTER_CUBIC)
NOTVALID = cv2.copyMakeBorder(NOTVALID, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, 0)
NOTVALID = torch.from_numpy(NOTVALID).unsqueeze(0).float() / 255.

import inspect
from colorama import Fore


def dmsg(v=None, *args):
    insp = inspect.currentframe().f_back
    (filename, line_number, function_name, lines, index) = inspect.getframeinfo(insp)
    if '/' in filename:
        sp = filename.split('/')
        filename = sp[-1] if len(sp) < 2 else sp[-2] + '/' + sp[-1]
        filename = '.../' + filename
    vtxt = ''
    args = list(args)
    args.insert(0, v)
    for v in args:
        if v is None:
            continue
        vtxt += ' ! '
        varv = None
        rest = None
        if '.' in v:
            v, rest = v.split('.')
        try:
            varv = insp.f_globals[v]
        except:
            pass
        try:
            varv = insp.f_locals[v]
        except:
            pass
        try:
            if varv is not None:
                if rest is None:
                    try:
                        vtxt += v + ' = ' + json.dumps(varv, indent=4)
                    except:
                        vtxt += v + ' = ' + str(varv)
                else:
                    if rest == 'shape': #Dont ned json pretify!
                        vtxt += v + ' = ' + str(eval("varv" + "." + rest))
                    else:
                        try:
                            vtxt += v + "." + rest + ' = ' + json.dumps(eval("varv" + "." + rest), indent=4)
                        except:
                            vtxt += v + ' = ' + str(eval("varv" + "." + rest))
        except:
            vtxt += v + ' = ' + 'Could not find!'

    print(Fore.GREEN + datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
          , Fore.BLUE + filename, line_number, function_name, Fore.BLACK + '|', Fore.RED + vtxt + Fore.RESET)


class MovingFGOnBGDataset(Dataset):
    """Moving foreground dataset on background."""

    def __init__(self, infrencePhase, seqLength, shape, scale=2, foregroundScale=0.75, blurIt=True, subpixel=True,
                 minResultSpeed=0, maxResultSpeed=2, square=True, background="random", foreground="MNIST"):
        super(MovingFGOnBGDataset).__init__()
        self.background = background
        self.foregroundScale = foregroundScale
        self.shapeOrig = shape
        self.seqLength = seqLength
        self.blurIt = blurIt
        self.subpixel = subpixel
        self.minResultSpeed = minResultSpeed
        self.maxResultSpeed = maxResultSpeed
        self.square = square
        self.scale = int(scale)
        self.shape = int(shape * scale)
        self.foreground = foreground

        if self.foreground == "MNIST":
            self.MNIST = datasets.MNIST('data', train=not infrencePhase, download=True)
        elif self.foreground == "FMNIST":
            self.FMNIST = datasets.FashionMNIST('data', train=not infrencePhase, download=True)
        elif self.foreground == "grid":
            pass
        else:
            raise Exception("Wrong foregorund")

        if self.background == "STL10":
            self.STL10 = datasets.STL10('data', split='train' if not infrencePhase else 'test', download=True,
                                        transform=transforms.Compose(
                                            [transforms.Grayscale(1), transforms.Resize(self.shape)]))
            self.STL10Size = len(self.STL10)
        elif self.background == "random":
            pass
        else:
            raise Ecxeption("Wrong background")

    def _scaleBlur(self, arry):
        if (self.blurIt):
            arry = cv2.blur(arry[0, :, :], (self.scale, self.scale))[np.newaxis, :, :]

        if self.scale != 1:
            arry = cv2.resize(arry[0, :, :], (self.shapeOrig, self.shapeOrig), interpolation=cv2.INTER_NEAREST)[
                   np.newaxis, :, :]
        return np.clip(arry, a_min=0, a_max=1)

    def _cImg(self, image, scale, original=False, square=True):
        if original == True:
            return image

        o = image.max(axis=0) > 0
        o_columns = np.where(np.any(o, axis=0))[0]
        o_rows = np.where(np.any(o, axis=1))[0]
        if not square:
            res = image[:, min(o_rows):max(o_rows) + 1,
                  min(o_columns):max(o_columns) + 1]
        else:
            maximum = max((max(o_rows) + 1 - min(o_rows)), (max(o_columns) + 1 - min(o_columns)))
            mino_row = max(min(o_rows) - ((maximum - (max(o_rows) - min(o_rows) + 1)) // 2), 0)
            mino_col = max(min(o_columns) - ((maximum - (max(o_columns) - min(o_columns) + 1)) // 2), 0)
            res = image[:, mino_row:mino_row + maximum,
                  mino_col:mino_col + maximum]
        res = cv2.resize(res[0, :, :], (int(res.shape[2] * scale), int(res.shape[1] * scale)),
                         interpolation=cv2.INTER_AREA)[np.newaxis, :, :]
        res = np.pad(res, ((0, 0), (4, 4), (4, 4)), "constant", constant_values=0)
        res = np.pad(res, ((0, 0), (1, 1), (1, 1)), "constant", constant_values=0.5)
        res = np.pad(res, ((0, 0), (1, 1), (1, 1)), "constant", constant_values=0.75)
        res = np.pad(res, ((0, 0), (1, 1), (1, 1)), "constant", constant_values=0.5)
        return res

    def _occluded(self, img, val=1, stride=8., space=5):
        dim = len(img.shape)
        w = img.shape[dim - 2]
        maxRange = int(w / stride)
        res = img.copy()
        for i in range(0, maxRange):
            maxStep = int(i * stride + stride + 3)
            minStep = int(i * stride + 3)
            if i % space == 0 and maxStep < w:
                if (dim == 5):
                    res[:, :, :, minStep:maxStep, :] = res[:, :, minStep:maxStep, :, :] = val
                elif (dim == 4):
                    res[:, :, :, minStep:maxStep] = res[:, :, minStep:maxStep, :] = val
                elif (dim == 3):
                    res[:, :, minStep:maxStep] = res[:, minStep:maxStep, :] = val
                elif (dim == 2):
                    res[:, minStep:maxStep] = res[minStep:maxStep, :] = val
                else:
                    raise (BaseException("ERROR"))
        return res

    def _addGrid(self, img):
        img = self._occluded(img)
        return img

    def __len__(self):
        #         return 500
        if self.foreground == "MNIST":
            return len(self.MNIST)
        elif self.foreground == "FMNIST":
            return len(self.FMNIST)
        else:
            return 60000

    def __getitem__(self, idx):

        if self.foreground == "MNIST":
            foreground_obj = self.MNIST.__getitem__(idx)
            img = np.array(foreground_obj[0])[np.newaxis, :, :] / 255.
            img = self._cImg(img, self.scale * self.foregroundScale, False, square=self.square)
        elif self.foreground == "FMNIST":
            foreground_obj = self.FMNIST.__getitem__(idx)
            img = np.array(foreground_obj[0])[np.newaxis, :, :] / 255.
            img = self._cImg(img, self.scale * self.foregroundScale, False, square=self.square)
        elif self.foreground == "grid":
            gridSize = 74 * self.scale * self.foregroundScale
            img = np.zeros((1, gridSize, gridSize), dtype=np.float32)
            foreground_obj = [img, 'grid']
            img = self._addGrid(img)

        sign = np.random.choice([-1, 1], size=(1, 2))
        if self.subpixel:
            velocities = (np.random.randint(low=int(self.minResultSpeed * self.scale),
                                            high=(self.maxResultSpeed * self.scale) + 1, size=(1, 2)) * sign)
        else:
            velocities = (np.random.randint(low=int(self.minResultSpeed * self.scale), high=self.maxResultSpeed + 1,
                                            size=(1, 2)) * sign * self.scale)
        shape2 = int(self.shape / 2)
        positions = np.array(
            [[shape2 + (np.sign(-velocities[0, 0]) * (shape2 - 1 - (img.shape[1] * (-velocities[0, 0] > 0)))),
              shape2 + (np.sign(-velocities[0, 1]) * (shape2 - 1 - (img.shape[2] * (-velocities[0, 1] > 0))))]])
        if self.background == "random":
            bg = np.random.rand(1, self.shape, self.shape)
        elif self.background == "STL10":
            stl = self.STL10.__getitem__(idx % self.STL10Size)
            bg = 1 - (np.array(stl[0])[np.newaxis, :, :] / 255.)
            # bg*=0.9
            # bg+=0.05
        else:
            bg = np.zeros(1, self.shape, self.shape)
        ResFrame = np.empty((1, self.seqLength, self.shapeOrig, self.shapeOrig), dtype=np.float32)
        ResFrameFG = np.empty((1, self.seqLength, self.shapeOrig, self.shapeOrig), dtype=np.float32)
        ResFrameAlpha = np.empty((1, self.seqLength, self.shapeOrig, self.shapeOrig), dtype=np.float32)
        ResFrameBG = np.empty((1, self.seqLength, self.shapeOrig, self.shapeOrig), dtype=np.float32)

        for frame_idx in range(self.seqLength):
            frame = np.zeros((1, self.shape, self.shape), dtype=np.float32)
            frameFG = np.zeros((1, self.shape, self.shape), dtype=np.float32)
            frameAlpha = np.zeros((1, self.shape, self.shape), dtype=np.float32)
            frameBG = np.zeros((1, self.shape, self.shape), dtype=np.float32)
            frameBG = bg
            frame += bg
            ptmp = positions.copy()

            ptmp[0] += velocities[0]
            for dimen in range(2):
                if ptmp[0, dimen] < 0:
                    velocities[0, dimen] *= -1
                    raise Exception("Bounced")
                if ptmp[0, dimen] > self.shape - img.shape[dimen + 1]:
                    velocities[0, dimen] *= -1
                    raise Exception("Bounced")

            positions[0] += velocities[0]
            digit_mat = np.zeros((self.shape, self.shape, 1))
            IN = [positions[0][0],
                  positions[0][0]
                  + img.shape[1],
                  positions[0][1],
                  positions[0][1]
                  + img.shape[2]]
            if self.foreground == "grid":
                mask = img > 0
                np.place(frame[0, IN[0]:IN[1], IN[2]:IN[3]], mask, img[mask])
            else:
                frame[0, IN[0]:IN[1], IN[2]:IN[3]] = img
            frameFG[0, IN[0]:IN[1], IN[2]:IN[3]] = img
            frameAlpha[0, IN[0]:IN[1], IN[2]:IN[3]] = np.ones_like(img)

            ResFrame[0, frame_idx] = self._scaleBlur(frame)
            ResFrameFG[0, frame_idx] = self._scaleBlur(frameFG)
            ResFrameAlpha[0, frame_idx] = self._scaleBlur(frameAlpha)
            ResFrameBG[0, frame_idx] = self._scaleBlur(frameBG)
            del frame, frameFG, frameAlpha, frameBG

        # [batch * channel(# of channels of each image) * depth(# of frames) * height * width]
        result = {'GT': ResFrame, 'A': ResFrameAlpha, 'BG': ResFrameBG, 'FG': ResFrameFG,
                  'foreground': foreground_obj[1], "velocity": velocities / self.scale}
        return result


class timeit():
    def __enter__(self):
        self.tic = self.datetime.now()

    def __exit__(self, *args, **kwargs):
        print(Fore.GREEN + 'Runtime: {}'.format(self.datetime.now() - self.tic) + Fore.RESET)


# ([batch, seq,colorchannel, H, W])
def showSeq(normalize, step, caption, data, revert=False, oneD=False, dpi=1, save="",
            vmin=None, vmax=None, normType=matplotlib.colors.NoNorm(), verbose=True, show=True):
    '''
    Data should be: [batch, seq,colorchannel, H, W]
    '''
    if type(data) is not list:
        data = [data]
        
    for i in range(len(data)):
        if type(data[i]) is torch.Tensor:
            data[i] = data[i].detach().cpu().numpy()
    
    if len(data[0].shape)!=5:
        print("Wrong shape!")
        return
    
    B,S,C,H,W = data[0].shape
    L = len(data)
    
    if type(data[0]) == np.ndarray:
        for i in range(len(data)):
            SH = data[i].shape
            if type(data[i]) != np.ndarray:
                print("Not consistent data type")
                return
            elif (SH[0],SH[1],SH[3],SH[4])!=(B,S,H,W): # we can have different C!
                print("Not consistent dim")
                return
            else:
                data[i] = np.repeat(a=data[i], repeats = 4-data[i].shape[2],axis = 2) #make all of data C==3
                if 3!=data[i].shape[2]:
                    print("Not consistent data channel shape")
                    return
        C=3
    else:
        print("Undefined Type Error")
        return
    
    #Now it is [batch, seq, H, W,colorchannel]
    if verbose and show:
        for i in range(len(data)):
            print("Data[", i, "]: min and max", data[i].min(), data[i].max())
    
    #Square if we dont have W or H!
    if (W == 1 or H == 1) and not oneD:
        longD = max(H,W)
        dimsqrt = int(math.sqrt(longD))
        if (dimsqrt * dimsqrt == longD):
            H=W=dimsqrt
            for i in range(len(data)):
                data[i] = data[i].reshape((B, S,C, H, W))
        else:
            print("Error while reshaping")
            return
        

    # Normilize
    if (vmax == None and vmin == None):
        maxAbsVal = -1000000
        for i in range(len(data)):
            maxAbsVal = max(maxAbsVal, max(abs(data[i].max()), abs(data[i].min())))
            
        for i in range(len(data)):
            if normalize:
                data[i] = ((data[i] / maxAbsVal) / 2) + 0.5
            else:#Make the range between -1 and 1
                data[i] = ((data[i] / maxAbsVal))
    

    finalImg = np.stack(data,axis=1)#B,L,S,C,H,W
    padH = max(2,W//40);padW = max(2,H//40)
    finalImg = np.pad(finalImg, ((0,0),(0,0),(0,0),(0,0),(padH,padH),(padW,padW)), 'constant', constant_values=(1))#B,L,S,C,(padH+H+padH),(padW+W+padW)
    finalImg = np.moveaxis(finalImg, (2), (4))#B,L,C,(padH+H+padH),(padW+W+padW),S
    finalImg = finalImg.reshape((B,L,C,padH+H+padH,-1))#B,L,C,(padH+H+padH),(padW+W+padW)*S
    finalImg = np.moveaxis(finalImg, (2), (0))#C,B,L,(padH+H+padH),(padW+W+padW)*S
    finalImg = finalImg.reshape((C,-1,(padW+W+padW)*S))#C,B*L*(padH+H+padH),(padW+W+padW)*S

    if show:
        display(Markdown('<b><span style="color: #ff0000">' + caption + (" (N)" if normalize else "") +
                         (" (r)" if revert else "") + "</span></b>  " + (str(finalImg.shape) if verbose else "") + ''))

    cmap = 'viridis'

    if revert:
        if (vmax == None and vmin == None):
            if normalize:
                mid = 0.5
            else:
                mid = (finalImg.max() - finalImg.min()) / 2
            finalImg = ((finalImg - mid) * -1) + mid
        else:
            mid = (vmax - vmin) / 2
            finalImg = ((finalImg - mid) * -1) + mid
            
    dpi = 240. * dpi
    if verbose and show:
        print("Image: min and max", finalImg.min(), finalImg.max())

    if show or save != "":
        pltImg = np.moveaxis(finalImg, 0, -1)
        if show:
            plt.figure(figsize=None, dpi=dpi)
            plt.imshow(pltImg, cmap=cmap, norm=normType, vmin=vmin, vmax=vmax, aspect='equal')
            plt.axis('off')
            #       plt.colorbar()
            plt.show()
        if save != "":
            plt.imsave(save + "/" + caption + str(step) + '.png', pltImg, cmap=cmap)

    return [wandb.Image(torch.from_numpy(finalImg).cpu(), caption=caption)]




def gaussian_noise(ins, is_training, std=0.005):
    if is_training and std > 0:
        noise = Variable(ins.data.new(ins.size()).normal_(mean=0, std=std))
        return ins + noise
    return ins


def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
                 else x.new(torch.arange(x.size(i) - 1, -1, -1).tolist()).long()
                 for i in range(x.dim()))
    return x[inds]


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def listToTensor(inp):
    return torch.stack(inp, dim=1)


def manyListToTensor(inp):
    res = []
    for i in inp:
        res.append(listToTensor(i))
    return res


def createarray(a1, a2, b1, b2, sp):
    return np.hstack((np.linspace(a1, a2, seq_length - sp, endpoint=False), np.linspace(b1, b2, sp, endpoint=False)))


def getItemIfList(inp, indx):
    if type(inp) == np.ndarray:
        return inp[indx]
    else:
        return inp


'''
B,H,W
'''


def fillWithGaussian(inp, sigma, offset=0):
    center = ((inp.shape[1] - 2) + offset) / 2.
    inp[:, :, :].fill_(0)
    for i_ in range(inp.shape[1]):
        for j_ in range(inp.shape[2]):
            inp[:, i_, j_] += 1. / 2. / np.pi / (sigma ** 2.) * torch.exp(
                -1. / 2. * ((i_ - center - 0.5) ** 2. + (j_ - center - 0.5) ** 2.) / (sigma ** 2.))


# [colorchannel,H, W]
def showSingleImg(img, colorbar=True):
    if type(img) is torch.Tensor:
        img = img.detach().cpu().numpy()

    if type(img) is np.ndarray:
        if len(img.shape) > 3:
            raise Exception("showSingleImage is only for one image not batch!")
        elif len(img.shape) < 2:
            raise Exception("showSingleImage is for show 2D images!")
        elif len(img.shape) == 3:
            if img.shape[0] == 1:
                img = img[0, :, :]
            elif img.shape[0] == 3 or img.shape == 4:
                img = np.moveaxis(img, 0, -1)
            else:
                raise Exception("wrong number of channels!")
        else:  # img is 2D
            pass
    else:
        raise Exception("Unknown input type!")
    plt.imshow(img)
    if colorbar:
        plt.colorbar()
    plt.show()


def generateAllRot(dsAlpha, dsFG):
    return (getRot(dsAlpha[:, 0, 0, :, :], dsAlpha[:, 0, 1, :, :])
            , getRot(dsFG[:, 0, 0, :, :], dsFG[:, 0, 1, :, :])
            , getRot(dsAlpha[:, 0, 0, :, :] * dsFG[:, 0, 0, :, :], dsAlpha[:, 0, 1, :, :] * dsFG[:, 0, 1, :, :]))


def generateAllRotwithBG(dsAlpha, dsFG, dsBG):
    return (getRot(dsAlpha[:, 0, 0, :, :], dsAlpha[:, 0, 1, :, :])
            , getRot(dsFG[:, 0, 0, :, :], dsFG[:, 0, 1, :, :])
            , getRot(dsAlpha[:, 0, 0, :, :] * dsFG[:, 0, 0, :, :], dsAlpha[:, 0, 1, :, :] * dsFG[:, 0, 1, :, :]),
            getRot(dsBG[:, 0, 0, :, :], dsBG[:, 0, 1, :, :]))


def angleToRotM(inp, rotTFG):
    res = torch.zeros_like(rotTFG)
    res[:, 0] = torch.cos(inp[:, 0]).clamp(-1, 1)
    res[:, 1] = torch.sin(inp[:, 1]).clamp(-1, 1)
    return res


def getEnergy(inp0, inp1):
    s = torch.zeros_like(inp0, dtype=torch.float)
    hf0 = torch.stack((inp0, s), dim=3)
    hf0 = fft(hf0, 2, normalized=normalizeFFT)
    hf1 = torch.stack((inp1, s), dim=3)
    hf1 = fft(hf1, 2, normalized=normalizeFFT)
    R = complex_mul(hf0, complex_conj(hf1))
    return complex_abs(R)[:, :, :, 0:1].repeat(1, 1, 1, 2)


def getRot(inp0, inp1):
    hfRel1OrigAgri = getRotComplete(inp0, inp1)
    return hfRel1OrigAgri


def rotIt(inp, rotInp):
    hfP_fft = torch.stack((inp, torch.zeros_like(inp, dtype=torch.float)), dim=3)
    hfP_fft = fft(hfP_fft, 2, normalized=normalizeFFT)
    rotInp_fft = fft(rotInp, 2, normalized=normalizeFFT)
    hfN_fft = complex_mul(hfP_fft, complex_conj(rotInp_fft))
    return ifft(hfN_fft, 2, normalized=normalizeFFT)[:, :, :, 0]


def getRotComplete(inp0, inp1):
    s = torch.zeros_like(inp0, dtype=torch.float)
    hf0 = torch.stack((inp0, s), dim=3)
    hf0 = fft(hf0, 2, normalized=normalizeFFT)
    hf1 = torch.stack((inp1, s), dim=3)
    hf1 = fft(hf1, 2, normalized=normalizeFFT)
    R = complex_mul(hf0, complex_conj(hf1))
    R = complex_div(R, complex_abs(R))
    return R


def visshift(data):
    for dim in range(len(data.size()) - 3, len(data.size()) - 1):
        data = roll_n(data, axis=dim, n=data.size(dim) // 2)
    return data


def justshift(data):
    for dim in range(len(data.size()) - 3, len(data.size()) - 1):
        data = roll_n(data, axis=dim, n=data.size(dim) // 2)
    return data


def fftshift(data, dim, normalized):
    return justshift(fft(data, dim, normalized))


def ifftshift(data, dim, normalized):
    return ifft(justshift(data), dim, normalized)


def fft(data, dim, normalized):
    data = torch.fft(data, dim, normalized=normalized)
    return (data)


def ifft(data, dim, normalized):
    data = torch.ifft((data), dim, normalized=normalized)
    return data


def clmp(a, soft=False,skew=10):
    if soft:
        return torch.sigmoid(-(skew/2.) + (a * skew))
    return a.clamp(0, 1)


def showComplex(norm, step, name, a, justRI=False, show=True, dpi=1):
    tmp = visshift(a)[li:ui].permute(0, 3, 1, 2).unsqueeze(1).cpu().detach()
    res = []
    res.append(showSeq(norm, step, name + " R&I " + str(step), [tmp[:, :, 0:1, :, :], tmp[:, :, 1:2, :, :]],
                       oneD=dimension == 1, revert=True, dpi=dpi, show=show)[0])
    if justRI:
        return
    angle = torch.atan2(tmp[:, :, 1:2, :, :], tmp[:, :, 0:1, :, :])
    absol = torch.sqrt(tmp[:, :, 1:2, :, :] * tmp[:, :, 1:2, :, :] + tmp[:, :, 0:1, :, :] * tmp[:, :, 0:1, :, :] + 1e-8)
    res.append(showSeq(norm, step, name + " Abs " + str(step), absol, oneD=dimension == 1, revert=True, dpi=dpi / 2.,
                       show=show)[0])
    res.append(showSeq(norm, step, name + " Angl " + str(step), angle, oneD=dimension == 1, revert=True, dpi=dpi / 2.,
                       show=show)[0])
    return res


def niceL2S(n, l):
    res = n + ": <br>"
    res += (pd.DataFrame(l.detach().cpu().numpy()).style.format("{0:.5f}")).render()
    return res


def showReal(norm, step, name, a, vmin=None, vmax=None, show=True, dpi=0.3):
    return showSeq(norm, step, name + " " + str(step), (a)[li:ui].unsqueeze(1).unsqueeze(1).cpu().detach(),
                   oneD=dimension == 1, dpi=dpi, revert=True, vmin=vmin, vmax=vmax, show=show)


# For prediction hf0 is next time step and hf1 is the previous time step!
def getPhaseDiff(hf0, hf1,energy = False):
    R = complex_mul(hf0, complex_conj(hf1))
    Energy = complex_abs(R)
    R = complex_div(R, Energy)
    if energy:
        return R,Energy[...,0:1]/(Energy.shape[-2]*Energy.shape[-3])
    return R,None


def getPhaseAdd(hf0, T):
    return complex_mul(hf0, T)


def smoothIt(rot, energy=None, gainEst=0.2, step=1, axis=0, show=False, dims=(1, 2), direction=0):
    resolution = rot.shape[1]
    oAxis = 1 if axis == 0 else 0
    if direction <= 0:
        a,_ = getPhaseDiff(rot, rot.roll(-step, dims=dims[axis]))
    if direction >= 0:
        ar,_ = getPhaseDiff(rot, rot.roll(step, dims=dims[axis]))
    if (show):
        if direction <= 0:
            showComplex(True, 1, "a", a, True)
        if direction >= 0:
            showComplex(True, 1, "ar", ar, True)
    if energy is not None:
        if direction <= 0:
            am = (a * energy.abs()).sum(dim=dims[oAxis]) / (energy.abs().sum(dim=dims[oAxis]) + 1e-8)
        if direction >= 0:
            amr = (ar * energy.abs()).sum(dim=dims[oAxis]) / (energy.abs().sum(dim=dims[oAxis]) + 1e-8)
    else:
        if direction <= 0:
            am = a.mean(dim=dims[oAxis])
        if direction >= 0:
            amr = ar.mean(dim=dims[oAxis])
    if (show):
        if direction <= 0:
            amshow = am.unsqueeze(2).repeat(1, 1, resolution, 1) if axis == 0 else am.unsqueeze(1).repeat(1, resolution,
                                                                                                          1, 1)
            showComplex(True, 1, "am", amshow, True)
        if direction >= 0:
            amrshow = amr.unsqueeze(2).repeat(1, 1, resolution, 1) if axis == 0 else am.unsqueeze(1).repeat(1,
                                                                                                            resolution,
                                                                                                            1, 1)
            showComplex(True, 1, "amr", amrshow, True)
    if direction <= 0:
        amm = am.mean(dim=1).unsqueeze(1).unsqueeze(2).repeat(1, resolution, resolution, 1)
    if direction >= 0:
        ammr = amr.mean(dim=1).unsqueeze(1).unsqueeze(2).repeat(1, resolution, resolution, 1)
    if (show):
        if direction <= 0:
            showComplex(True, 1, "amm", amm, True)
        if direction >= 0:
            showComplex(True, 1, "ammr", ammr, True)
    if direction <= 0:
        newEs = complex_mul(rot.roll(-step, dims=dims[axis]), amm)
    if direction >= 0:
        newEsr = complex_mul(rot.roll(step, dims=dims[axis]), ammr)
    assert not torch.isnan(rot).any()
    if direction <= 0:
        assert not torch.isnan(newEs).any()
    if direction >= 0:
        assert not torch.isnan(newEsr).any()
    if direction < 0:
        rot = (1 - gainEst) * rot + (gainEst) * newEs  # +1e-8
    elif direction > 0:
        rot = (1 - gainEst) * rot + (gainEst) * newEsr  # +1e-8
    else:
        rot = (1 - gainEst) * rot + (gainEst / 2.) * newEs + (gainEst / 2.) * newEsr  # +1e-8
    assert not torch.isnan(rot).any()
    rot = complex_div(rot, complex_abs(rot))
    assert not torch.isnan(rot).any()
    if (show):
        if direction <= 0:
            showComplex(True, 1, "newEs", newEs, True)
        if direction >= 0:
            showComplex(True, 1, "newEsr", newEsr, True)
    return rot


recIndex = None


def createPhaseDiff(avgRecX, avgRecY, shape):
    global recIndex
    BS, H, W, _ = shape

    angleX = avgRecX.unsqueeze(-1).unsqueeze(-1)
    angleY = avgRecY.unsqueeze(-1).unsqueeze(-1)

    reconstructIdx = False
    if recIndex is None:
        reconstructIdx = True
    else:
        if recIndex.shape[1] != H or recIndex.shape[2] != W:
            reconstructIdx = True

    if reconstructIdx:
        with torch.no_grad():
            recIndex = avgRecX.new_zeros((H, W, 2), requires_grad=False)
            xMid = recIndex.shape[0] // 2
            yMid = recIndex.shape[1] // 2
            for i in range(xMid, -1, -1):
                if i != xMid:
                    highI = (xMid - i) + xMid
                    lowI = i
                    if highI < recIndex.shape[0]:
                        recIndex[highI, yMid] = recIndex[highI - 1, yMid] + avgRecX.new_tensor([-1, 0])
                    recIndex[lowI, yMid] = recIndex[lowI + 1, yMid] + avgRecX.new_tensor([1, 0])

            for j in range(yMid, -1, -1):
                if j != yMid:
                    highJ = (yMid - j) + yMid
                    lowJ = j
                    if highJ < recIndex.shape[1]:
                        recIndex[:, highJ] = recIndex[:, highJ - 1] + avgRecX.new_tensor([0, -1]).unsqueeze(0).expand(
                            recIndex.shape[1], 2)
                    recIndex[:, lowJ] = recIndex[:, lowJ + 1] + avgRecX.new_tensor([0, 1]).unsqueeze(0).expand(
                        recIndex.shape[1], 2)

            recIndex = recIndex.unsqueeze(0)
            recIndex = justshift(recIndex)

    inter = (angleX * recIndex[:, :, :, 0] + angleY * recIndex[:, :, :, 1]).unsqueeze(-1)
    return torch.cat([torch.cos(inter), torch.sin(inter)], dim=-1)


def getAvgDiff(rot, energy=None, step=1, axis=0, dims=(1, 2), untilIndex=None,variance=False):
    assert (len(rot.shape) == 4)
    assert (dims == (1, 2))
    assert (axis == 0 or axis == 1)
    oAxis = 1 if axis == 0 else 0
    midIdx = (rot.size(dims[oAxis]) - 1) // 2
    if untilIndex is not None:
        untilIndex = max(2, untilIndex)
        rot = rot[:, 0:untilIndex, 0:untilIndex, :]
        if energy is not None:
            energy = energy[:, 0:untilIndex, 0:untilIndex, :]
    a,_ = getPhaseDiff(rot, rot.roll(-step, dims=dims[axis]))

    if variance:
        varam = a.var(dim=[-3,-2]).sum(dim=-1).unsqueeze(-1) #wiki: https://en.wikipedia.org/wiki/Complex_random_variable
        #has some bias!#TODO correct it like bellow
    else:
        varam = None

    if energy is not None:
        am = (a * energy.abs()).sum(dim=dims[oAxis]) / (energy.abs().sum(dim=dims[oAxis]) + 1e-8)
    else:
        am = a.mean(dim=dims[oAxis])

    removeMid = False
    if untilIndex is not None:  # Remove the last index (It is wrong since we croped the input)
        am = am[:, 0:-1, :]
        if untilIndex > midIdx:
            removeMid = True
    else:
        removeMid = True
    if removeMid:
        amm = torch.cat([am[:, :midIdx, :], am[:, midIdx + 1:, :]],
                        dim=1)  # The middle is wrong (discontinutity) so delete it
    else:
        amm = am


    return amm.mean(dim=1),varam


def get_delta(T, config):  # For vis
    xV = T.view(-1, T.shape[2], T.shape[3], 2)
    tmpY,_ = getAvgDiff(rot=xV, step=1, axis=0,
                      untilIndex=config.untilIndex)  # Make sure you copy and paste latest getAvgDiff from hafez branch
    tmpX,_ = getAvgDiff(rot=xV, step=1, axis=1, untilIndex=config.untilIndex)
    tmpX = complex_div(tmpX, complex_abs(tmpX))
    tmpY = complex_div(tmpY, complex_abs(tmpY))

    N_prime = config.window_size + 2 * config.window_padding_constrained
    L_x = config.num_windows_x_constrained
    L_y = config.num_windows_y_constrained
    visX = torch.atan2(tmpX[:, 1], tmpX[:, 0] + 0.0000001) / (torch.pi * 2 / N_prime)
    visY = torch.atan2(tmpY[:, 1], tmpY[:, 0] + 0.0000001) / (torch.pi * 2 / N_prime)
    return visX.view(-1, L_y, L_x), visY.view(-1, L_y, L_x)


def updateWithAvgDiff(rot, avgDiff, gainEst=0.2, step=1, axis=0, dims=(1, 2), bothDir=True):
    avgDiffShape = avgDiff.unsqueeze(1).unsqueeze(1).expand_as(rot)

    newEs = complex_mul(rot.roll(-step, dims=dims[axis]), avgDiffShape)
    if bothDir:
        newEsr = complex_mul(rot.roll(step, dims=dims[axis]), complex_conj(avgDiffShape))
        rotNew = (1 - gainEst) * rot + (gainEst / 2.) * newEs + (gainEst / 2.) * newEsr
    else:
        rotNew = (1 - gainEst) * rot + (gainEst) * newEs

    rotNew = complex_div(rotNew, complex_abs(rotNew))
    return rotNew


def smoothIt2(rot, energy=None, gainEst=0.2, step=1, axis=0, dims=(1, 2), bothDir=True):
    avgDiff,_ = getAvgDiff(rot, energy, step, axis, dims)
    rot = updateWithAvgDiff(rot, avgDiff, gainEst, step, axis, dims, bothDir)
    return rot


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


from IPython.display import Image
from IPython.display import Markdown


def _prepare_pytorch(x):
    if isinstance(x, torch.autograd.Variable):
        x = x.data
        x = x.cpu().numpy()
        return x


def make_np(x):
    """
        Args:
          x: An instance of torch tensor or caffe blob name
        Returns:
            numpy.array: Numpy array
        """
    if isinstance(x, np.ndarray):
        return x
    if np.isscalar(x):
        return np.array([x])
    if isinstance(x, torch.Tensor):
        return _prepare_pytorch(x)
    raise NotImplementedError(
        'Got {}, but numpy array, torch tensor, or caffe2 blob name are expected.'.format(type(x)))


def _prepare_video(V):
    """
    Converts a 5D tensor [batchsize, time(frame), channel(color), height, width]
    into 4D tensor with dimension [time(frame), new_width, new_height, channel].
    A batch of images are spreaded to a grid, which forms a frame.
    e.g. Video with batchsize 16 will have a 4x4 grid.
    """
    b, t, c, h, w = V.shape

    if V.dtype == np.uint8:
        V = np.float32(V) / 255.

    def is_power2(num):
        return num != 0 and ((num & (num - 1)) == 0)

    # pad to nearest power of 2, all at once
    if not is_power2(V.shape[0]):
        len_addition = int(2 ** V.shape[0].bit_length() - V.shape[0])
        V = np.concatenate(
            (V, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)

    n_rows = 2 ** ((b.bit_length() - 1) // 2)
    n_cols = V.shape[0] // n_rows

    V = np.reshape(V, newshape=(n_rows, n_cols, t, c, h, w))
    V = np.transpose(V, axes=(2, 0, 4, 1, 5, 3))
    V = np.reshape(V, newshape=(t, n_rows * h, n_cols * w, c))

    return V


def _calc_scale_factor(tensor):
    converted = tensor.numpy() if not isinstance(tensor, np.ndarray) else tensor
    return 1 if converted.dtype == np.uint8 else 255


# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def make_video(tensor, fps):
    try:
        import moviepy  # noqa: F401
    except ImportError:
        print('add_video needs package moviepy')
        return
    try:
        from moviepy import editor as mpy
    except ImportError:
        print("moviepy is installed, but can't import moviepy.editor.",
              "Some packages could be missing [imageio, requests]")
        return
    import tempfile

    t, h, w, c = tensor.shape

    # encode sequence of images into gif string
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    filename = tempfile.NamedTemporaryFile(suffix='.gif', delete=False).name
    try:  # newer version of moviepy use logger instead of progress_bar argument.
        clip.write_gif(filename, verbose=False, logger=None)
    except TypeError:
        try:  # older version of moviepy does not support progress_bar argument.
            clip.write_gif(filename, verbose=False, progress_bar=False)
        except TypeError:
            clip.write_gif(filename, verbose=False)

    with open(filename, 'rb') as f:
        tensor_string = f.read()

    junckFiles.append(filename)

    return tensor_string


def video(tensor, fps=10):
    tensor = make_np(tensor)
    tensor = _prepare_video(tensor)
    # If user passes in uint8, then we don't need to rescale by 255
    scale_factor = _calc_scale_factor(tensor)
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).astype(np.uint8)
    video = make_video(tensor, fps)
    return video


def showMultiVideo(tags, tensors, fps=10):
    display(Markdown('Tag: <span style="color: #ff0000; font-size: 12pt">' + tags + '</span>'))
    newList = []
    for t in tensors:
        newList.append(t)
        newList.append(torch.ones_like(t)[..., 0:1])
    tensor = torch.cat(newList, dim=-1)
    return display(Image(data=video(1 - tensor.clamp(0, 1)), format='gif', width=900))


def showVideo(tag, tensor, fps=10):
    display(Markdown('Tag: <span style="color: #ff0000; font-size: 12pt">' + tag + '</span>'))
    return display(Image(data=video(1 - tensor.clamp(0, 1)), format='gif', width=200))


def log_npList(figs, fps, vis_anim_string="animations"):
    tensor = torch.cat(figs, dim=1)
    tensor = make_np(tensor)
    tensor = _prepare_video(tensor)
    # If user passes in uint8, then we don't need to rescale by 255
    scale_factor = _calc_scale_factor(tensor)
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).astype(np.uint8)  # C x H x W x T
    # tensor = np.moveaxis(tensor, [0, 3], [3, 0])  # T x H x W x C
    return log_video(tensor, fps=7, vis_anim_string=vis_anim_string)


junckFiles = []


def clean_junk_files():
    global junckFiles
    for j in junckFiles:
        try:
            os.remove(j)  # should commit ==True to block the code
        except OSError:
            logging.warning('The temporary file used by moviepy cannot be deleted.')
    junckFiles = []


def log_video(tensor, fps, vis_anim_string='animations'):
    global junckFiles
    try:
        import moviepy  # noqa: F401
    except ImportError:
        print('add_video needs package moviepy')
        return
    try:
        from moviepy import editor as mpy
    except ImportError:
        print("moviepy is installed, but can't import moviepy.editor.",
              "Some packages could be missing [imageio, requests]")
        return
    import tempfile

    t, h, w, c = tensor.shape

    # encode sequence of images into gif string
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    filename = tempfile.NamedTemporaryFile(suffix='.gif', delete=False).name
    try:  # newer version of moviepy use logger instead of progress_bar argument.
        clip.write_gif(filename, verbose=False, logger=None)
    except TypeError:
        try:  # older version of moviepy does not support progress_bar argument.
            clip.write_gif(filename, verbose=False, progress_bar=False)
        except TypeError:
            clip.write_gif(filename, verbose=False)

    tensor_video = wandb.Video(filename, fps=fps, format='gif')
    wandblog({vis_anim_string: tensor_video}, commit=False)

    junckFiles.append(filename)

    return None


def logMultiVideo(tags, tensors, fps=10, vis_anim_string='animations'):
    newList = []
    for t in tensors:
        newList.append(t)
        newList.append(torch.ones_like(t)[..., 0:1])
    tensor = torch.cat(newList, dim=-1)
    tensor = 1 - tensor.clamp(0, 1)
    tensor = make_np(tensor)
    tensor = _prepare_video(tensor)
    # If user passes in uint8, then we don't need to rescale by 255
    scale_factor = _calc_scale_factor(tensor)
    tensor = tensor.astype(np.float32)
    tensor = (tensor * scale_factor).astype(np.uint8)  # C x H x W x T
    # tensor = np.moveaxis(tensor, [0, 3], [3, 0])  # T x H x W x C
    return log_video(tensor, fps, vis_anim_string)


def getDistWindow(windowSize: int):
    c = windowSize // 2
    indxT, indyT = torch.meshgrid([torch.arange(windowSize).float(), torch.arange(windowSize).float()])
    distMat = torch.sqrt((c - indxT) * (c - indxT) + (c - indyT) * (c - indyT))
    return distMat


def getFlatTopWindow(windowSize: int,
                     a_flat=torch.tensor([0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368])):
    distMat = getDistWindow(windowSize)
    adjuster = torch.tensor(2 * (windowSize) ** 2).float().to(a_flat.device)
    adjuster = torch.sqrt(adjuster)
    distMatAdjust = distMat + adjuster / 2
    if a_flat.is_cuda:
        distMatSend = (distMatAdjust / adjuster).cuda()
    else:
        distMatSend = distMatAdjust / adjuster

    return a_flat[0] - a_flat[1] * torch.cos(2 * torch.pi * distMatSend) + a_flat[2] * torch.cos(
        4 * torch.pi * distMatSend) \
           - a_flat[3] * torch.cos(6 * torch.pi * distMatSend) + a_flat[4] * torch.cos(8 * torch.pi * distMatSend)


def gH(x, N, sigma):
    return torch.exp(-(x - 0.5 * N) ** 2 / (2 * (N + 1) * sigma) ** 2)


def confinedGaussian(r, windowSize, sigma=0.1):
    const1 = gH(torch.tensor(-0.5).to(sigma.device), windowSize, sigma)
    const2 = gH(torch.tensor(0.5).to(sigma.device) + windowSize, windowSize, sigma)
    const3 = gH(torch.tensor(-1.5).to(sigma.device) - windowSize, windowSize, sigma)
    denom = gH(r + windowSize + 1, windowSize, sigma) + gH(r - windowSize - 1, windowSize, sigma)
    return gH(r, windowSize, sigma) - const1 * denom / (const2 + const3)


def getGaussianWindow(windowSize, sigma=0.12):
    distMat = getDistWindow(windowSize)

    wsAdjust = math.sqrt(windowSize ** 2 * 2)

    distMatSend = distMat.to(sigma.device)

    return confinedGaussian(distMatSend + wsAdjust / 2, wsAdjust, sigma)


###added 10.10.20
class CombinedLossDiscounted(torch.nn.Module):
    def __init__(self, alpha=0.5, window_size=9, gamma=0.9):
        super(CombinedLossDiscounted, self).__init__()
        self.alpha = alpha
        self.window_size = window_size
        self.loss_function_structural = kornia.losses.SSIM(window_size, reduction='none')
        self.loss_function_pixel_wise = nn.L1Loss(reduction='none')
        self.gamma = 0.9

    def forward(self, x, y):
        lGains = [self.gamma ** i for i in range(1, x.shape[-3] + 1, 1)]
        lGains = torch.tensor([i / sum(lGains) for i in lGains], device=x.device)
        out = (1 - self.alpha) * self.loss_function_pixel_wise(x, y).mean(-1).mean(
            -1) + self.alpha * self.loss_function_structural(x, y).mean(-1).mean(-1)
        return (out @ lGains).mean()


def generate_name():
    return datetime.now().strftime("%d-%m-%Y_%H_%M")


###Superimposes the local shifts encoded in the phase diffs on the (ground truth) image frames
###Intended usage is right at the end of the prediction process, provided that the history of estimated phase differences has been retained
# input: pd_list, a list of phase differences of len (sequence_length - 1)
# input: gt, a tensor of sequence_length frames to superimpose the local shifts onto
# input: config, the wandb config dict
# input: title, some string describing the plot for wandb, should contain train/valid phase and type of phase diff (raw, after smoothing, etc)
# output: None, but the visualization is logged to wandb, allthough it is only committed later (commit = False)
def show_phase_diff(pd_list, gt, config, title,clear_motion=False):

    gt = gt.permute(0,1,3,4,2)
    offset = config.image_pad_size_constrained - (config.window_size + 2 * config.window_padding_constrained) // 2
    figs = []
    showStillVisualization = False
    if showStillVisualization:
        figT, axT = plt.subplots(1, config.sequence_length, figsize=(30, 3), dpi=360)
    for i in range(config.sequence_length):
        img = gt[0, i].detach().cpu()
        cmap = None
        if img.shape[2]==1:
            img = img[:,:,0]
            cmap='gray'
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=360)
        ax.set_yticks([])
        ax.set_xticks([])
        if showStillVisualization:
            axT[i].set_yticks([])
            axT[i].set_xticks([])
        if i == 0:
            ax.imshow(img, cmap=cmap,vmin=0,vmax=1)
            if showStillVisualization:
                axT[i].imshow(img, cmap=cmap,vmin=0,vmax=1)
            with torch.no_grad():
                visY, visX = pd_list[0]
                if clear_motion:
                    visX=torch.zeros_like(visX)
                    visY=torch.zeros_like(visY)
            L_y = visY.shape[-2]
            L_x = visY.shape[-1]
            mmx = np.arange(0 - offset, config.stride * L_x - offset, config.stride) + 1
            mmy = np.arange(0 - offset, config.stride * L_y - offset, config.stride) + 1
            xx, yy = np.meshgrid(mmx, mmy, sparse=True)
            ax.quiver(xx, yy, visX[0].cpu(), visY[0].cpu(), alpha=0)
            if showStillVisualization:
                axT[i].quiver(xx, yy, visX[0].cpu(), visY[0].cpu(), alpha=0)
        else:
            with torch.no_grad():
                visY, visX = pd_list[i - 1]
                if clear_motion:
                    visX=torch.zeros_like(visX)
                    visY=torch.zeros_like(visY)
            ax.imshow(img, cmap=cmap,vmin=0,vmax=1)
            ax.quiver(xx, yy, visX[0].cpu(), visY[0].cpu(), color='r', scale=2.5 / config.stride,
                    units='xy')  # scale is important here, might have to adjust to each dataset #, pivot='mid'
            if showStillVisualization:
                axT[i].imshow(img, cmap=cmap,vmin=0,vmax=1)
                axT[i].quiver(xx, yy, visX[0].cpu(), visY[0].cpu(), color='r', scale=2.5 / config.stride,
                              units='xy')  # scale is important here, might have to adjust to each dataset #, pivot='mid'
            figs.append(torch.from_numpy(get_img_from_fig(fig)).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).clone())

    log_npList(figs, fps=7, vis_anim_string="GIF " + title)
    if showStillVisualization:
        wandblog({title: wandb.Image(figT, caption="T superimposed on GT \n" + "max_y = " + str(
            visY.abs().max()) + " max_x = " + str(visX.abs().max()))})
    plt.close('all')
    return None

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe



class DenseNetLikeModel(nn.Module):
    def __init__(self, inputC,outputC, layerCount=4, hiddenF=24, filterS=3,gain=1,nonlin='ReLU',lastNonLin=False):
        super().__init__()
        self.layerC = layerCount
        self.convS = []
        for c in range(self.layerC):
            filt = filterS if type(filterS) ==int else filterS[c]
            self.convS.append(nn.Conv2d(in_channels=inputC + (c * hiddenF),
                                        out_channels=hiddenF if c < (self.layerC - 1) else outputC,
                                        kernel_size=filt, stride=1, padding=filt//2))
            with torch.no_grad():
                self.convS[-1].weight.data *= 0.001
                mid = self.convS[-1].weight.shape[2] // 2
                self.convS[-1].weight.data[:, :, mid, mid] = (1. / (self.convS[-1].in_channels))*gain
                self.convS[-1].bias.data *= 0.001
        self.convS = nn.ModuleList(self.convS)
        
        self.nonLin = eval('torch.nn.'+nonlin+"()") #https://pytorch.org/docs/master/nn.html#non-linear-activations-weighted-sum-nonlinearity
        self.lastNonLin = False
    def forward(self, inp):
        res = []
        for c in range(self.layerC):
            if not self.lastNonLin and c==self.layerC-1:
                res.append(self.convS[c](torch.cat(res + [inp], dim=1)))
            else:
                res.append(self.nonLin(self.convS[c](torch.cat(res + [inp], dim=1))))

        return res[-1]