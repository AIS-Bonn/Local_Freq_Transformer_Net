import random
import torch
import cv2
import json
import os
from typing import List, Tuple
import asset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from asset.utils import dmsg,getIfAugmentData
import imgaug.augmenters as iaa
import numpy as np
from tqdm import tqdm
import random as rand
from torchvision import datasets, transforms


class MovingFGeSOnBGDataset(Dataset):
    """Moving foreground dataset on background."""

    def __init__(self, infrencePhase, seqLength, shapeX, shapeY,digitCount = 1, scale=2, foregroundScale=0.7, blurIt=True,
                 minResultSpeed=0, maxResultSpeed=2,color=False):
        super(MovingFGeSOnBGDataset).__init__()
        if digitCount>2:
            raise BaseException("Too much FG requested!")
        self.shapeXOrig = shapeX
        self.shapeYOrig = shapeY
        self.seqLength = seqLength
        self.blurIt = blurIt
        self.minResultSpeed = minResultSpeed
        self.maxResultSpeed = maxResultSpeed
        self.foregroundScale = foregroundScale
        self.digitCount = digitCount
        self.scale = int(scale)
        self.shapeX = int(shapeX * scale)
        self.shapeY = int(shapeY * scale)
        self.color = color
        self.MNIST = datasets.MNIST('data', train=not infrencePhase, download=True)
        self.STL10 = datasets.STL10('data', split='train' if not infrencePhase else 'test', download=True,
                                    transform=transforms.Compose(
                                        ([transforms.Grayscale(1)] if not self.color else [])+
                                        [ transforms.Resize([self.shapeY, self.shapeX])]))
        self.STL10Size = len(self.STL10)

    def _scaleBlur(self, arry):
        if (self.blurIt):
            arry = cv2.blur(arry, (self.scale, self.scale))

        if self.scale != 1:
            arry = cv2.resize(arry, (self.shapeYOrig, self.shapeXOrig), interpolation=cv2.INTER_NEAREST)# cv2.resize wants [shape[1],shape[0]]
            if not self.color:
                arry = arry[:,:,np.newaxis] # cv2.resize return with no channel!
        
        return np.clip(arry, a_min=0, a_max=1)

    def _cImg(self, image, scale, original=False):
        if original == True:
            return image

        res = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA) # cv2.resize wants [shape[1],shape[0]]
        if not self.color:
            res = res[:,:,np.newaxis] # cv2.resize return with no channel!

        res = np.pad(res, ((4, 4), (4, 4),(0, 0)), "constant", constant_values=0)
        res = np.pad(res, ((1, 1), (1, 1),(0, 0)), "constant", constant_values=0.5)
        res = np.pad(res, ((1, 1), (1, 1),(0, 0)), "constant", constant_values=0.75)
        res = np.pad(res, ((1, 1), (1, 1),(0, 0)), "constant", constant_values=0.5)
        return res

    def __len__(self):
        return len(self.STL10)

    def __getitem__(self, idx):
        foreground_objs = []
        random.seed(idx)
        np.random.seed(idx)
        for _ in range(self.digitCount):
            mnistdig=self.MNIST.__getitem__(random.randint(0, len(self.MNIST) - 1))[0]
            mnistdig = np.array(mnistdig)[ :, :,np.newaxis]
            if self.color:
                mnistdig = np.repeat(mnistdig, 3, axis=2)
                randC = random.randint(0,2)
                mnistdig[:,:,randC] = random.randint(0,100)
            mnistdig = self._cImg(np.moveaxis(mnistdig, 0, 1) / 255., self.scale * self.foregroundScale, False)
            foreground_objs.append(mnistdig)


        shapeX2 = self.shapeX // 2
        shapeY2 = self.shapeY // 2
        MINPOS = 2
        MAXPOSX = self.shapeX - MINPOS - foreground_objs[0].shape[1]
        MAXPOSY = self.shapeY - MINPOS - foreground_objs[0].shape[0]
        possiblePos = [
            {"p": np.array([MINPOS, MINPOS]), "corner": 'tl'},
            {"p": np.array([MINPOS, MAXPOSY]), "corner": 'dl'},
            {"p": np.array([MAXPOSX, MINPOS]), "corner": 'tr'},
            {"p": np.array([MAXPOSX, MAXPOSY]), "corner": 'dr'},
        ]
        positions = random.sample(possiblePos, 2)
        velocities = np.random.randint(low=int(self.minResultSpeed * self.scale),
                                       high=(self.maxResultSpeed * self.scale) + 1, size=(2, 2))

        if positions[0]["corner"] == 'dl':
            velocities[0][1] *= -1
        elif positions[0]["corner"] == 'tr':
            velocities[0][0] *= -1
        elif positions[0]["corner"] == 'dr':
            velocities[0][1] *= -1
            velocities[0][0] *= -1


        if positions[0]['p'][0] == positions[1]['p'][0]:
            velocities[1][0] = 0
            velocities[1][0] *= -1 if positions[1]['p'][0] > shapeX2 else 1
            velocities[1][1] = min(velocities[1][1], abs(velocities[0][0]) - 1)
            velocities[1][1] *= -1 if positions[1]['p'][1] > shapeY2 else 1
        elif positions[0]['p'][1] == positions[1]['p'][1]:
            velocities[1][1] = 0
            velocities[1][1] *= -1 if positions[1]['p'][1] > shapeY2 else 1
            velocities[1][0] = min(velocities[1][0], abs(velocities[0][1]) - 1)
            velocities[1][0] *= -1 if positions[1]['p'][0] > shapeX2 else 1
        else:
            axis = 0 if abs(velocities[0][0]) <= abs(velocities[0][1]) else 1  # random.randint(0,1)
            naxis = (axis + 1) % 2
            velocities[1][axis] = 0
            velocities[1][axis] *= -1 if positions[axis]['p'][1] > (shapeX2 if axis == 0 else shapeY2) else 1
            velocities[1][naxis] *= np.sign(positions[0]['p'][naxis] - positions[1]['p'][naxis])
        
        stl = self.STL10.__getitem__(idx)
        
        if not self.color:
            stl = np.array(stl[0])[ :, :,np.newaxis]
        else:
            stl = np.array(stl[0])
        
        bg = 1 - (np.moveaxis(stl, 0, 1)/ 255.)
        # bg *= 0

        channel = 3 if self.color else 1
        ResFrame = np.empty((self.seqLength, self.shapeXOrig, self.shapeYOrig,channel), dtype=np.float32)
        ResFrameFG = np.empty((self.seqLength, self.shapeXOrig, self.shapeYOrig,channel), dtype=np.float32)
        ResFrameAlpha = np.empty((self.seqLength, self.shapeXOrig, self.shapeYOrig,channel), dtype=np.float32)
        ResFrameBG = np.empty((self.seqLength, self.shapeXOrig, self.shapeYOrig,channel), dtype=np.float32)

        for frame_idx in range(self.seqLength):

            frame = np.zeros((self.shapeX, self.shapeY,channel), dtype=np.float32)
            frameFG = np.zeros((self.shapeX, self.shapeY,channel), dtype=np.float32)
            frameAlpha = np.zeros((self.shapeX, self.shapeY,channel), dtype=np.float32)
            frameBG = np.zeros((self.shapeX, self.shapeY,channel), dtype=np.float32)
            frameBG = bg
            frame += bg

            for ax in range(self.digitCount):
                positions[ax]['p'] += velocities[ax]

                IN = [positions[ax]['p'][0],
                      positions[ax]['p'][0]
                      + foreground_objs[ax].shape[0],
                      positions[ax]['p'][1],
                      positions[ax]['p'][1]
                      + foreground_objs[ax].shape[1]]

                frame[IN[0]:IN[1], IN[2]:IN[3],:] = foreground_objs[ax]
                frameFG[IN[0]:IN[1], IN[2]:IN[3],:] = foreground_objs[ax]
                frameAlpha[IN[0]:IN[1], IN[2]:IN[3],:] = np.ones_like(foreground_objs[ax])

            ResFrame[frame_idx] = self._scaleBlur(frame)
            ResFrameFG[frame_idx] = self._scaleBlur(frameFG)
            ResFrameAlpha[frame_idx] = self._scaleBlur(frameAlpha)
            ResFrameBG[frame_idx] = self._scaleBlur(frameBG)
            del frame, frameFG, frameAlpha, frameBG

        ResFrame = np.moveaxis(ResFrame, [0,1,2,3] , [0,3,2,1])
        ResFrameAlpha = np.moveaxis(ResFrameAlpha, [0,1,2,3] , [0,3,2,1])
        ResFrameBG = np.moveaxis(ResFrameBG, [0,1,2,3] , [0,3,2,1])
        ResFrameFG = np.moveaxis(ResFrameFG, [0,1,2,3] , [0,3,2,1])
        # [batch * depth(# of frames) * channel(# of channels of each image) * height * width]
        result = {'GT': ResFrame, 'A': ResFrameAlpha, 'BG': ResFrameBG, 'FG': ResFrameFG,
                    "velocity": velocities / self.scale}
        # result = ResFrame
        return result


sequenceFileDict = {
    'lin_3': 'canvas_down_sample_just_translation_3_18_06.pt',
    'rot_lin_2': 'canvas_down_sample_just_rotation_15_06.pt',
    'lin_1': 'canvas_down_sample_just_translation_15_06.pt',
    'challenge': 'canvas_down_sample_extreme_16_06.pt',
    'cars': 'car_sequence_257.pt',
    'small_cars': 'car_sequence_downsampled.pt',
    'more_cars': 'car_sequence_downsampled_long.pt',
    'stationary_rotation': 'canvas_down_sample_inplace_rotation_2_22_06.pt',
    'lin_2': 'canvas_down_sample_just_translation_2_21_06.pt',
    'acc_1': 'canvas_down_sample_with_acc_16_06.pt',
    'rot_lin_scale_2': 'canvas_down_sample_everything_2_24_06.pt',
    'all_cars': 'carData_inv_01_07.pt',
    'random_cars': 'carData_inv_permuted_01_07.pt',
    'augmented_cars': 'augmented_cars.pt',
    'rot_lin_2_NOSCALE': 'canvas_down_sample_no_scale_12_07.pt',
    'high_res_test': 'canvas_07_08.pt',
    'high_res_test_3': 'canvas_3_07_08.pt',
    'circle_sanity_1_px': 'canvas_circle_int_up.pt',
    'circle_sanity_2_px': 'canvas_circle_int_up_2.pt'
}

class sequenceData(Dataset):
    def __init__(self, config,key='rot_lin_scale_2', device=torch.device('cpu'), size=(65, 65), sequence_length=10,color=False):
        self.key = key
        self.dict = sequenceFileDict
        if key in self.dict:
            self.filePath = 'data/pickled_ds/' + self.dict[key]
        else:
            self.filePath = 'data/pickled_ds/' + key+'.pt'
        self.data = torch.load(self.filePath).float()
        self.len = self.data.shape[0]
        self.dev = device
        self.size = size[::-1]
        self.sequence_length = sequence_length
        self.color = color
        self.config =config
    def __getitem__(self, ind):
        if len(self.data.shape)==4:
            data = self.data[ind].unsqueeze(1).to(self.dev)
        else:
            data = self.data[ind].to(self.dev)
        res = torch.nn.functional.interpolate(data, size=self.size,
                                        mode='bilinear', align_corners=False)[:self.sequence_length]

        if res.shape[1]==1 and self.color:
            res = res.expand(res.shape[0],3,res.shape[2],res.shape[3])
        elif res.shape[1]==3 and not self.color:
            res,_ = res.max(dim=1)
            res=res.unsqueeze(1)
        return res
    def __len__(self):
        return self.len

class savedData(Dataset):
    def __init__(self,data,device=torch.device('cpu'), size=(65, 65),limit=1):
        self.data = data
        self.limit=limit
        self.dev = device
        self.size = size[::-1]
    def reshape(self,inp):
        shp=inp.shape
        if len(shp)>4:
            return torch.nn.functional.interpolate(inp.reshape(-1,1,shp[-2],shp[-1]), size=self.size,
                                            mode='bilinear', align_corners=False).reshape(shp[:-2]+self.size)
        elif len(shp)>3:
            return torch.nn.functional.interpolate(inp, size=self.size,
                                            mode='bilinear', align_corners=False)
    def __getitem__(self, ind):
        res = self.data[ind]
        if res is dict:
            for it in res.keys():
                res[it] = self.reshape(res[it])
            return res
        else:
            return self.reshape(res)

    def __len__(self):
        return int(len(self.data)*self.limit)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
def get_data_loaders(config, key='rot_lin_scale_2', size=(65, 65), ratio=[0.15,0.15], batch_size=1,test_batch_size=1, num_workers=1,
                     device=torch.device('cpu'), limit=1, sequence_length=10):

    if key == 'MotionSegmentation':
        dataset = MovingFGeSOnBGDataset(infrencePhase=False, seqLength=sequence_length, shapeX=size[0], shapeY=size[1]
                ,digitCount = config.digitCount, scale=2, foregroundScale=0.7, blurIt=True,
                 minResultSpeed=2, maxResultSpeed=4,color=config.color)
        dlen = len(dataset)
        splitTr = int((1-(ratio[0]+ratio[1]))*dlen)
        tr_ds,val_ds=torch.utils.data.random_split(dataset,
                                    [splitTr, dlen-(splitTr)])
        tr_ds = torch.utils.data.Subset(tr_ds, range(0,int(len(tr_ds)*limit)))
        val_ds = torch.utils.data.Subset(val_ds, range(0,int(len(val_ds)*limit)))
        train_loader = DataLoader(tr_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True,worker_init_fn=worker_init_fn)
        valid_loader = DataLoader(val_ds, batch_size=test_batch_size, num_workers=num_workers, shuffle=False,worker_init_fn=worker_init_fn)

        dataset = MovingFGeSOnBGDataset(infrencePhase=True, seqLength=sequence_length, shapeX=size[0], shapeY=size[1]
                ,digitCount = config.digitCount, scale=2, foregroundScale=0.7, blurIt=True,
                 minResultSpeed=2, maxResultSpeed=4,color=config.color)
        dlen = len(dataset)
        limitedDlen = int(dlen * limit)
        te_ds,_=torch.utils.data.random_split(dataset,
                                    [limitedDlen, dlen-(limitedDlen)])
        test_loader = DataLoader(te_ds, batch_size=test_batch_size, num_workers=num_workers, shuffle=False,worker_init_fn=worker_init_fn)
    elif key == 'HugeNGSIM':
        fileNames = [
            'data/NGSIM/peachtree-camera5-1245pm-0100pm.avi',
            'data/NGSIM/peachtree-camera2-1245pm-0100pm.avi',
            'data/NGSIM/nb-camera7-0400pm-0415pm.avi',
            'data/NGSIM/lankershim-camera4-0830am-0845am.avi',
            'data/NGSIM/lankershim-camera5-0830am-0845am.avi',
            'data/NGSIM/peachtree-camera1-1245pm-0100pm.avi',
            'data/NGSIM/peachtree-camera3-1245pm-0100pm.avi',
            'data/NGSIM/peachtree-camera4-1245pm-0100pm.avi',
            'data/NGSIM/nb-camera5-0400pm-0415pm.avi',
            'data/NGSIM/nb-camera6-0400pm-0415pm.avi'
        ]
        dataset = VideoLoader(fileLoc=fileNames[:-4], fCount=sequence_length,
                           sampleRate=(1, 1), size=size,color=config.color)
        dlen = len(dataset)
        limitedDlen = int(dlen * limit)
        tr_ds,_=torch.utils.data.random_split(dataset,
                                    [limitedDlen, dlen-(limitedDlen)])
        train_loader = DataLoader(tr_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True,worker_init_fn=worker_init_fn)
        dataset = VideoLoader(fileLoc=fileNames[-4:-2], fCount=sequence_length,
                           sampleRate=(1, 1), size=size,color=config.color)
        dlen = len(dataset)
        limitedDlen = int(dlen * limit)
        va_ds,_=torch.utils.data.random_split(dataset,
                                    [limitedDlen, dlen-(limitedDlen)])
        valid_loader = DataLoader(va_ds, batch_size=test_batch_size, num_workers=num_workers, shuffle=False,worker_init_fn=worker_init_fn)
        dataset = VideoLoader(fileLoc=fileNames[-2:], fCount=sequence_length,
                           sampleRate=(1, 1), size=size,color=config.color)
        dlen = len(dataset)
        limitedDlen = int(dlen * limit)
        te_ds,_=torch.utils.data.random_split(dataset,
                                    [limitedDlen, dlen-(limitedDlen)])
        test_loader = DataLoader(te_ds, batch_size=test_batch_size, num_workers=num_workers, shuffle=False,worker_init_fn=worker_init_fn)
    elif 'NGSIM' in key:
        fileNames = [
            'data/NGSIM/nb-camera5-0400pm-0415pm.avi',
            'data/NGSIM/nb-camera6-0400pm-0415pm.avi'
        ]
        dataset = VideoLoader(fileLoc=fileNames, fCount=sequence_length,
                           sampleRate=(1, 1), size=size,color=config.color)
        dlen = len(dataset)
        splitTr = int((1-(ratio[0]+ratio[1]))*dlen)
        splitVa = int(ratio[0]*dlen)
        tr_ds,val_ds,test_ds=asset.utils.random_split(dataset,
                                    [splitTr,splitVa, dlen-(splitTr+splitVa)])
        tr_ds = torch.utils.data.Subset(tr_ds, range(0,int(len(tr_ds)*limit)))
        val_ds = torch.utils.data.Subset(val_ds, range(0,int(len(val_ds)*limit)))
        test_ds = torch.utils.data.Subset(test_ds, range(0,int(len(test_ds)*limit)))
        train_loader = DataLoader(tr_ds, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, shuffle=True,worker_init_fn=worker_init_fn)
        valid_loader = DataLoader(val_ds, batch_size=test_batch_size, num_workers=num_workers,
                                  pin_memory=False, shuffle=False,worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_ds, batch_size=test_batch_size, num_workers=num_workers,
                                  pin_memory=False, shuffle=False,worker_init_fn=worker_init_fn)
    else:
        dataset = sequenceData(config,key=key, device=device, size=size, sequence_length=sequence_length,color=config.color)
        dlen = len(dataset)
        splitTr = int((1-(ratio[0]+ratio[1]))*dlen)
        splitVa = int(ratio[0]*dlen)
        tr_ds,val_ds,test_ds=torch.utils.data.random_split(dataset,
                                    [splitTr,splitVa, dlen-(splitTr+splitVa)])
        tr_ds = torch.utils.data.Subset(tr_ds, range(0,int(len(tr_ds)*limit)))
        val_ds = torch.utils.data.Subset(val_ds, range(0,int(len(val_ds)*limit)))
        test_ds = torch.utils.data.Subset(test_ds, range(0,int(len(test_ds)*limit)))

        train_loader = DataLoader(tr_ds, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, shuffle=True,worker_init_fn=worker_init_fn)
        valid_loader = DataLoader(val_ds, batch_size=test_batch_size, num_workers=num_workers,
                                  pin_memory=False, shuffle=False,worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_ds, batch_size=test_batch_size, num_workers=num_workers,
                                  pin_memory=False, shuffle=False,worker_init_fn=worker_init_fn)

    return train_loader, valid_loader, test_loader


class VideoLoader(Dataset):
    def __init__(self, fileLoc: List[str], fCount: int = 10,
                 sampleRate: Tuple[int, int] = (1, 1),
                 size: Tuple[int, int] = (160, 120),color=False):
        if fCount < 2:
            raise Exception("Input arg is not correct!")

        self.fileLoc = fileLoc
        self.fCount = fCount
        self.size = size
        self.sampleRate = sampleRate
        self.length = 0
        self.frameVideo = []
        self.caps = None
        self.color = color
        for ifl, fl in enumerate(self.fileLoc):
            self.curVCI = ifl
            cap = cv2.VideoCapture(fl)
            vidLen = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            curStart = 0
            curFinish = vidLen
            curLen = ((curFinish - curStart) // self.sampleRate[1]) - self.fCount
            self.frameVideo.append([self.length, self.length + curLen, curStart])
            self.length += curLen
            try:
                cap.release()
            except:
                pass
            del cap

        print("VideoLoader initiated!")

    def __del__(self):
        if self.caps is not None:
            for c in enumerate(self.caps):
                try:
                    c.release()
                except:
                    pass

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.caps is None:
            self.caps = [cv2.VideoCapture(vp) for vp in self.fileLoc]

        foundC = False
        for ifv, fv in enumerate(self.frameVideo):
            if idx >= fv[0] and idx < fv[1]:
                foundC = True
                if self.curVCI != ifv:
                    self.curVCI = ifv
                break
        if not foundC:
            raise IndexError()

        cap = self.caps[self.curVCI]
        idx -= self.frameVideo[self.curVCI][0]
        frames = torch.Tensor(self.fCount, 3 if self.color else 1, self.size[1], self.size[0])

        vidFrameIdx = (idx) * self.sampleRate[1] + self.frameVideo[self.curVCI][2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, vidFrameIdx)
        srate = random.randint(self.sampleRate[0], self.sampleRate[1])

        if getIfAugmentData():
            augFull = iaa.Sequential([
                iaa.Affine(scale={"x": (0.97, 1), "y": (0.97, 1.)}, rotate=(-0.4, 0.4),
                           translate_px={"x": (-3, 3), "y": (-3, 3)}, shear=(-0.3, 0.3)),
                #             iaa.MultiplyAndAddToBrightness(mul=(0.9, 1.1), add=(-10, 10)),
                iaa.AddToHueAndSaturation((-20, 20), per_channel=False),
                iaa.Fliplr(0.5),
                # iaa.Flipud(0.5),
            ])
            augFull_det = augFull.to_deterministic()


        for i in range(self.fCount):
            for _ in range(srate):
                ret, frame = cap.read()
            if not ret:
                raise Exception("Cannot read the file %s at index %d" % (self.fileLoc[self.curVCI], idx))

            if self.size[0] == self.size[1]:
                res_min = min(frame.shape[0], frame.shape[1])
                frame = frame[:res_min, :res_min]
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_LINEAR)
            
            if getIfAugmentData():
                frame = augFull_det(image=frame)

            if not self.color:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            frame = torch.from_numpy(frame / 255.)

            frame = frame.permute(2,0,1)

            frames[i] = 1 - frame

        return frames
