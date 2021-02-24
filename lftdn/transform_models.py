import torch
import torch.nn as nn
from asset.utils import getAvgDiff, createPhaseDiff, justshift, get_delta,positionalencoding2d,dmsg,DenseNetLikeModel
from lftdn.custom_fft import custom_fft


class cellTransportRefine(nn.Module):
    def __init__(self, config):
        super(cellTransportRefine, self).__init__()
        self.res_y = config.res_y_constrained
        self.res_x = config.res_x_constrained
        self.y_stride = config.stride
        self.x_stride = config.stride
        self.pS = config.window_padding_constrained
        self.N = config.window_size
        self.N_prime = self.N + 2 * self.pS

        self.padSizeAdjust = int(self.x_stride * ((self.N_prime - 1) // self.x_stride))
        self.L_y = (self.res_y + 2 * self.padSizeAdjust - self.N_prime) // self.y_stride + 1
        self.L_x = (self.res_x + 2 * self.padSizeAdjust - self.N_prime) // self.x_stride + 1

        self.mode = config.trans_mode  # 'Fc', 'Conv', 'IFc', 'IConv'
        self.Recpt = config.recpt_IConv
        self.prev_x_c = config.concat_depth
        self.untilIndex = config.untilIndex  # Can be 'None'
        self.fcMiddle = config.fcMiddle * 10
       
        self.denseNet = True
        self.pose_enc_level = 4
        moreC = self.pose_enc_level if config.pos_encoding else 0

        inpDim = 4 if config.use_variance else 2
        if self.mode == 'Fc':
            self.fcLayer1 = torch.nn.Linear(self.L_x * self.L_y * inpDim * (self.prev_x_c + 1), self.fcMiddle)
            self.fcLayer2 = torch.nn.Linear(self.fcMiddle, self.L_x * self.L_y * 2)
        elif self.mode == 'Conv':
            self.cnn = DenseNetLikeModel( inputC=(inpDim * (self.prev_x_c + 1))+moreC,
                           outputC=2, layerCount=len(config.tran_filter_sizes), hiddenF=config.tran_hidden_unit,
                               filterS=config.tran_filter_sizes, nonlin = config.tr_non_lin,lastNonLin=False)
        elif self.mode == 'IFc':
            self.fcLayer1 = torch.nn.Linear(self.L_x * self.L_y * inpDim * (self.prev_x_c + 1), self.fcMiddle)
            self.fcLayer2 = torch.nn.Linear(self.fcMiddle, self.L_x * self.L_y * self.Recpt * self.Recpt)
        elif self.mode == 'IConv':
            self.cnn = DenseNetLikeModel(inputC=(inpDim * (self.prev_x_c + 1))+moreC,
                                outputC = self.Recpt ** 2, layerCount = len(config.tran_filter_sizes),
                              hiddenF = config.tran_hidden_unit, filterS = config.tran_filter_sizes,
                                         nonlin = config.tr_non_lin,lastNonLin=False)
        self.prev_x = []
        self.config = config
        self.pos_encoding = None
    # @profile
    def forward(self, x,energy, first,throwAway):

        xV = x.view(-1, x.shape[2], x.shape[3], 2)

        if energy is not None:
            energy = energy.view(-1, x.shape[2], x.shape[3], 1)
        # energy = None
        tmpX,varianceX = getAvgDiff(rot=xV,energy=energy, step=1, axis=0, untilIndex=self.untilIndex,variance = self.config.use_variance)
        tmpY,varianceY = getAvgDiff(rot=xV,energy=energy, step=1, axis=1, untilIndex=self.untilIndex,variance = self.config.use_variance)

        tmpX = (torch.atan2(tmpX[:, 1], tmpX[:, 0] + 0.0000001) / (torch.pi * 2 / self.N_prime)).unsqueeze(-1)
        tmpY = (torch.atan2(tmpY[:, 1], tmpY[:, 0] + 0.0000001) / (torch.pi * 2 / self.N_prime)).unsqueeze(-1)


        angBefore = [-tmpX.view(-1, self.L_y, self.L_x),
                     tmpY.view(-1, self.L_y, self.L_x)]  # tmpX is actually -y, tmpY is actually x

        if throwAway:
            return x, angBefore, angBefore, 0

        if "Conv" in self.mode:
            if self.config.use_variance:
                lInp = torch.cat((tmpX, tmpY,varianceX,varianceY), dim=1).view(-1, self.L_x, self.L_y, 4)
            else:
                lInp = torch.cat((tmpX, tmpY), dim=1).view(-1, self.L_x, self.L_y, 2)
        else:
            if self.config.use_variance:
                lInp = torch.cat(((tmpX, tmpY,varianceX,varianceY)), dim=1).view(-1, self.L_x * self.L_y * 4)
            else:
                lInp = torch.cat((tmpX, tmpY), dim=1).view(-1, self.L_x * self.L_y * 2)

        if first:
            self.prev_x = [0.1 * torch.ones_like(lInp) for i in range(self.prev_x_c)]
        else:
            self.prev_x.pop(0)

        self.prev_x.append(lInp)  # TODO: think if we want the gradient for prev, or not!

        
        lInp = torch.cat(self.prev_x, dim=-1)

        angAfter = None
        if self.mode == 'Fc':
            lInp = self.nonLin(self.fcLayer1(lInp))
            lInp = self.fcLayer2(lInp)
            lInp = lInp.reshape(-1, 2)
            tmpX = lInp[:, 0:1]
            tmpY = lInp[:, 1:]
            angAfter = [-tmpX.view(-1, self.L_y, self.L_x),
                            tmpY.view(-1, self.L_y, self.L_x)]  # tmpX is actually -y, tmpY is actually x
            angAfterNorm = (tmpX.abs().mean() + tmpY.abs().mean())/2.
            tmpX = tmpX * (torch.pi * 2 / self.N_prime)
            tmpY = tmpY * (torch.pi * 2 / self.N_prime)
            x = createPhaseDiff(tmpX, tmpY, xV.shape).view_as(x)

        elif self.mode == 'Conv':
            lInp = lInp.permute(0, 3, 1, 2).contiguous()
            if self.config.pos_encoding:
                if self.pos_encoding is None:
                    self.pos_encoding = positionalencoding2d(self.pose_enc_level, lInp.shape[2], lInp.shape[3]).unsqueeze(
                        0).to(lInp.device).detach()
                lInp = torch.cat(
                    (self.pos_encoding.expand(lInp.shape[0], self.pose_enc_level, lInp.shape[2], lInp.shape[3]),lInp), dim=1)


            lInp = self.cnn(lInp)
            lInp = lInp.permute(0, 2, 3, 1)
            lInp = lInp.reshape(-1, 2)
            tmpX = lInp[:, 0:1]
            tmpY = lInp[:, 1:]
            angAfter = [-tmpX.view(-1, self.L_y, self.L_x),
                            tmpY.view(-1, self.L_y, self.L_x)]  # tmpX is actually -y, tmpY is actually x
            angAfterNorm = (tmpX.abs().mean() + tmpY.abs().mean())/2.
            tmpX = tmpX * (torch.pi * 2 / self.N_prime)
            tmpY = tmpY * (torch.pi * 2 / self.N_prime)
            x = createPhaseDiff(tmpX, tmpY, xV.shape).view_as(x)

        elif self.mode == 'IFc':
            lInp = self.nonLin(self.fcLayer1(lInp))
            lInp = self.nonLin(self.fcLayer2(lInp))
            lInp = lInp.reshape(-1, self.Recpt, self.Recpt)
            paddingT = paddingL = (self.N_prime - self.Recpt) // 2
            paddingD = paddingR = (1 + self.N_prime - self.Recpt) // 2
            lInp = torch.nn.functional.pad(lInp, (paddingT, paddingD, paddingL, paddingR))
            lInpImag = torch.zeros_like(lInp)
            lInp = torch.stack([lInp, lInpImag], dim=-1)
            lInp = justshift(lInp)
            lInp = lInp.view_as(x)
            x = custom_fft(lInp, real=False)
            angAfter = get_delta(x, self.config)
            angAfterNorm = (angAfter[0].abs().mean() + angAfter[1].abs().mean())/2.
        elif self.mode == 'IConv':
            lInp = lInp.permute(0, 3, 1, 2).contiguous()
            if self.config.pos_encoding:
                if self.pos_encoding is None:
                    self.pos_encoding = positionalencoding2d(self.pose_enc_level, lInp.shape[2], lInp.shape[3]).unsqueeze(
                        0).to(lInp.device).detach()
                lInp = torch.cat(
                    (self.pos_encoding.expand(lInp.shape[0], self.pose_enc_level, lInp.shape[2], lInp.shape[3]),lInp), dim=1)

            lInp = self.cnn(lInp)

            lInp = lInp.permute(0, 2, 3, 1)
            lInp = lInp.reshape(-1, self.Recpt, self.Recpt)
            paddingT = paddingL = (self.N_prime - self.Recpt) // 2
            paddingD = paddingR = (1 + self.N_prime - self.Recpt) // 2
            lInp = torch.nn.functional.pad(lInp, (paddingT, paddingD, paddingL, paddingR))
            lInpImag = torch.zeros_like(lInp)
            lInp = torch.stack([lInp, lInpImag], dim=-1)
            lInp = justshift(lInp)
            lInp = lInp.view_as(x)
            x = custom_fft(lInp, real=False)
            angAfter = get_delta(x, self.config)
            angAfterNorm = (angAfter[0].abs().mean() + angAfter[1].abs().mean())/2
        else:
            raise NotImplemented

        return x, angBefore, angAfter, angAfterNorm

class phaseDiffModel(nn.Module):
    def __init__(self, config):
        super(phaseDiffModel, self,).__init__()
        self.res_y = config.res_y_constrained
        self.res_x = config.res_x_constrained
        self.y_stride = config.stride
        self.x_stride = config.stride
        self.pS = config.window_padding_constrained
        self.N = config.window_size
        self.N_prime = self.N + 2 * self.pS


        self.padSizeAdjust = int(self.x_stride * ((self.N_prime - 1) // self.x_stride))
        self.L_y = (self.res_y + 2 * self.padSizeAdjust - self.N_prime) // self.y_stride + 1
        self.L_x = (self.res_x + 2 * self.padSizeAdjust - self.N_prime) // self.x_stride + 1

        self.untilIndex = config.untilIndex  # Can be 'None'
       
        self.denseNet = True

        if config.PD_model_enable:
            inpDim = 3 if config.use_variance else 1
            inpDim+= 2 if config.PD_model_use_direction else 0
            self.cnn = DenseNetLikeModel( inputC=(inpDim),
                            outputC=1, layerCount=len(config.PD_model_filter_sizes), hiddenF=config.PD_model_hidden_unit,
                                filterS=config.PD_model_filter_sizes, nonlin = config.PD_model_non_lin,lastNonLin=False)
        self.prev_x = []
        self.config = config
        self.pos_encoding = None
    def forward(self, x,energy):
        eps = 1e-8
        xV = x.view(-1, x.shape[2], x.shape[3], 2)
        if energy is not None:
            energy = energy.view(-1, x.shape[2], x.shape[3], 1)
        # energy = None
        tmpX,varianceX = getAvgDiff(rot=xV,energy=energy, step=1, axis=0, untilIndex=self.untilIndex,variance = self.config.use_variance)
        tmpY,varianceY = getAvgDiff(rot=xV,energy=energy, step=1, axis=1, untilIndex=self.untilIndex,variance = self.config.use_variance)

        tmpX = (torch.atan2(tmpX[:, 1], tmpX[:, 0] + 0.0000001) / (torch.pi * 2 / self.N_prime)).unsqueeze(-1)
        tmpY = (torch.atan2(tmpY[:, 1], tmpY[:, 0] + 0.0000001) / (torch.pi * 2 / self.N_prime)).unsqueeze(-1)
        mag = torch.sqrt((tmpX*tmpX)+(tmpY*tmpY)+eps)

        if self.config.PD_model_enable:
            if self.config.use_variance:
                if self.config.PD_model_use_direction:
                    lInp = torch.cat((tmpX,tmpY,varianceX,varianceY,mag), dim=1).view(-1, self.L_x, self.L_y, 5)
                else:
                    lInp = torch.cat((varianceX,varianceY,mag), dim=1).view(-1, self.L_x, self.L_y, 3)
            else:
                if self.config.PD_model_use_direction:
                    lInp = torch.cat((tmpX,tmpY,mag), dim=1).view(-1, self.L_x, self.L_y, 3)
                else:
                    lInp = mag.view(-1, self.L_x, self.L_y, 1)

            lInp = lInp.permute(0, 3, 1, 2).contiguous()
            lInp = self.cnn(lInp)
            lInp = lInp.permute(0, 2, 3, 1).contiguous()
        else:
            lInp=mag

        tmp2  = lInp.reshape(x.shape[0],x.shape[1],1,1)
        return tmp2.expand(x.shape[0],x.shape[1],x.shape[2],x.shape[3])
