import torch
import os
from lfdtn.LFT import LFT, iLFT
from asset.utils import getPhaseAdd, getPhaseDiff
from colorama import Fore

def LFT_unit_test(window, res_y=128, res_x=128, bS=1, avDev=torch.device("cpu"), pS=1, eps=1e-12, H=2):
    S = torch.rand((bS,1, res_y, res_x), dtype=torch.float32).to(avDev)
    print('S shape:', S.shape)
    S_fft = LFT(S + eps, window, x_stride=H, y_stride=H, padding=pS)
    # T = getPhaseDiff(S_fft, S_fft)
    T,_ = getPhaseDiff(LFT(S + eps, window, x_stride=H, y_stride=H, padding=pS), \
                     LFT(S + eps, window, x_stride=H, y_stride=H, padding=pS))
    NS_fft = getPhaseAdd(S_fft, T)
    NS_, WT = iLFT(NS_fft, window, T, res_y=res_y, res_x=res_x, y_stride=H, x_stride=H, padding=pS)
    print('mean absolute error:', (S - NS_).abs().mean())
    print('max absolute error:', (S - NS_).abs().max())


def setEvalTrain(train=True):
    global paramN, inference_phase
    for p in paramN:
        toP = (p if type(p) is str else p[0])
        if type(eval(toP)) != torch.Tensor:
            if inference_phase or not train:
                globals()[toP].eval()
            else:
                globals()[toP].train()


def loadModels(fname="Untitled",excludeLoad=""):
    global paramN, inference_phase
    excludeList = excludeLoad.split(",")
    noPrefExist = os.path.exists(fname)
    if os.path.exists("savedModels/" + fname + ".th") or noPrefExist:
        loadedModel = torch.load(("savedModels/" + fname + ".th") if not noPrefExist else fname, map_location='cpu')
        lc=0
        for p in paramN:
            try:
                if p in excludeList:
                    raise BaseException("")
                fromP = (p if type(p) is str else p[1])
                toP = (p if type(p) is str else p[0])
                eStr = 'loadedModel["' + fromP + '"]'
                if type(eval(toP)) == torch.Tensor:
                    eStr = eStr + '.to(avDev)'
                    globals()[toP] = eval(eStr)
                else:
                    eStr = toP + ".load_state_dict(" + eStr + ')'
                    exec(eStr)
                print(Fore.GREEN, fromP,'loaded!',Fore.RESET)
                lc+=1
            except:
                print(Fore.RED, fromP,('Failed to load!' if p not in excludeList else "Skipped ..."),Fore.RESET)
            
        print(lc," Params loaded from disk!")
    else:
        print("Default params loaded!")

    for p in paramN:
        toP = (p if type(p) is str else p[0])
        if type(eval(toP)) == torch.Tensor:
            globals()[toP] = globals()[toP].detach().requires_grad_()
        else:
            if inference_phase:
                globals()[toP].eval()
            else:
                globals()[toP].train()


def saveModels(fname="Untitled"):
    global paramN
    toSave = {}
    for p in paramN:
        par = eval(p)
        toSave[p] = par if type(par) == torch.Tensor else par.state_dict()
    torch.save(toSave, "savedModels/" + fname + ".th")

def pretty(d,ref, indent=1):
    for key, value in d.items():
        print('  ' * indent + str(key).ljust(25)[:25],end="\r\n" if isinstance(value, dict) else "")
        rk = ref.get(key)
        if isinstance(value, dict):
            if rk!=None:
                pretty(value, rk, indent + 1)
            else:
                pretty(value,{}, indent + 1)
        else:
            if rk!=None:
                if rk ==value:
                    print('  ' * (indent + 1) + Fore.GREEN+str(value)+Fore.RESET)
                else:
                    print('  ' * (indent + 1) + Fore.RED+str(value)+" (Diff)"+Fore.RESET)
            else:
                print('  ' * (indent + 1) + Fore.CYAN+str(value)+" (New)"+Fore.RESET)