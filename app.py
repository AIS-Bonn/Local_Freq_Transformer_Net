import sys
import os
import wandb
import kornia
import time
import ast
import random as rand
import numpy as np
from lfdtn.dataloaders import get_data_loaders
from lfdtn.train_step import predict
from lfdtn.transform_models import cellTransportRefine,phaseDiffModel
from lfdtn.window_func import get_pascal_window, get_ACGW, get_2D_Gaussian
from asset.utils import generate_name, dmsg, CombinedLossDiscounted,wandblog,DenseNetLikeModel,niceL2S,setIfAugmentData
from past.builtins import execfile
from colorama import Fore
from tqdm import tqdm
import torch
import click

execfile('lfdtn/helpers.py')  # This is instead of 'from asset.helpers import *', to have loadModels and saveModels
# access global variable.

torch.utils.backcompat.broadcast_warning.enabled = True

print("Python Version:", sys.version)
print("PyTorch Version:", torch.__version__)
print("Cuda Version:", torch.version.cuda)
print("CUDNN Version:", torch.backends.cudnn.version())

# This will make some functions faster It will automatically choose between: 
# https://github.com/pytorch/pytorch/blob/1848cad10802db9fa0aa066d9de195958120d863/aten/src/ATen/native/cudnn/Conv
# .cpp#L486-L494 
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

worker_init_fn = None

hyperparameter_defaults = dict(
    dryrun=False,
    inference=False,
    load_model='',
    limitDS=1.,
    epochs=500,
    batch_size=20,
    sequence_length=10,
    sequence_seed=5,
    seeAtBegining=3,
    max_result_speed=4,
    stride=4,  # 2 to the power of this number ##https://www.calculatorsoup.com/calculators/math/commonfactors.php
    window_size=15,
    window_type='ConfGaussian',
    lg_sigma_bias=0.1729,
    optimizer='AdamW',
    gain_update_lr=0.0008,
    refine_lr = 0.0001,
    refine_wd= 0.00001,
    refine_layer_cnt=6,
    refine_layer_cnt_a=6,
    refine_hidden_unit=8,
    refine_filter_size=3,
    ref_non_lin = 'PReLU',
    M_transform_lr=0.001,
    M_transform_wd=0.0001,
    trans_mode='Conv',
    tran_hidden_unit=16,
    tran_filters="133",
    recpt_IConv=5,
    fcMiddle=10,  # Multiplied by 10
    untilIndex=8,
    concat_depth=6,
    tr_non_lin='PReLU',
    angleNormGain=0.00004,  # 0.0001
    AEstVarGain=0.01,
    AVarGain=0.01,
    AVarTarget=0.0767,#torch.tensor([0]*23+[1,1],dtype=torch.float).var()
    ANormGain=0.01,
    AEstNormGain=0.01,
    ANormTarget=2./25.,
    data_key='MotionSegmentation',
    digitCount=1,
    res_x=65,
    res_y=65,
    comp_fair_x = -1,
    comp_fair_y = -1,
    max_loss_tol_general=0.2,
    max_loss_tol_index = 2,
    max_num_param_tol=40000,
    pos_encoding=True,
    use_variance = True,
    use_energy = True,
    lr_scheduler = 'OneCycleLR',
    oneCycleMaxLRGain= 10,
    start_T_index = 2,
    tqdm=False,
    kill_no_improve = 3,
    init_A_with_T = True,
    init_T_with_GT = True,
    PD_model_enable=False,
    PD_model_use_direction=True,
    dublicate_PD_from_tr=True,
    PD_model_filters="133",
    PD_model_non_lin = 'PReLU',
    PD_model_hidden_unit=4,
    PD_model_lr= 0.002,
    PD_model_wd= 0.0001,
    alpha_root=3,
    validate_every=1,
    allways_refine= True,
    color=False,
    num_workers=4,
    init_BG_with_GT_and_A = False,
    A_skew = 10,#10 to 20! #Paste: \frac{1}{1+e^{-\left(-9+x\cdot18\right)}} to https://www.desmos.com/calculator,
    hybridGain=0.7,
    augment_data=True,
    gainLR=1,
    excludeLoad="",
    refine_output=False,
    MRef_BG_layer_cnt=-1,
    MRef_BG_hidden_unit=-1,
    MRef_BG_filter_size=-1,
    MRef_FG_layer_cnt=-1,
    MRef_FG_hidden_unit=-1,
    MRef_FG_filter_size=-1,
    MRef_A_layer_cnt=-1,
    MRef_A_hidden_unit=-1,
    MRef_A_filter_size=-1,
    share_all_possible_models=False,
    add_A_to_M_BG=True,
    loss="Hybrid",
    test_batch_size=-1,
    segmentation=True,
    enable_autoregressive_prediction=False,
    update_T_just_by_FG = False,
    useBN=False,
    initW=True,
    cmd=""
)

try:
    print("WANDB_CONFIG_PATHS = ", os.environ["WANDB_CONFIG_PATHS"])
except:
    pass

def mytqdm(x):
    return x

for a in sys.argv:
    if '--dryrun=True' in a:
        os.environ["WANDB_MODE"] = "dryrun"
    if ('--configs' in a and "=" in a) or '.yml' in a:
        try:
            try:
                v = a
                _, v = a[2:].split("=")
            except:
                pass
            if os.path.exists(v):
                v = str(os.getcwd()) + "/" + v
                os.environ["WANDB_CONFIG_PATHS"] = v
                print("Load configs from ", v)
        except Exception as e:
            print(e)
            pass

# print(sys.argv,os.environ,hyperparameter_defaults)
wandb.init(config=hyperparameter_defaults, project="localmotionseg") 
for k in wandb.config.keys():
    if '_constrained' in str(k):
        del wandb.config._items[k]


def myType(val):
    try:
        val = ast.literal_eval(val)
    except ValueError:
        pass
    return val


for a in sys.argv:
    if '--cmd=' in a[:6]:
        wandb.config.update({'cmd': str(a[6:])}, allow_val_change=True)
        continue
    if '--' in a[:2] and "=" in a:
        try:
            k, v = a[2:].split("=")
            v = myType(v)
            # if k in wandb.config.keys():
            #     v = type(wandb.config._items[k])(v)
            wandb.config.update({k: v}, allow_val_change=True)
        except Exception as e:
            pass
config = wandb.config
wandb.save('asset/*')
wandb.save('lfdtn/*')

if  config.dublicate_PD_from_tr:
    config.update({'PD_model_filters': config.tran_filters}, allow_val_change=True)
    config.update({'PD_model_non_lin': config.tr_non_lin}, allow_val_change=True)
    config.update({'PD_model_hidden_unit': config.tran_hidden_unit}, allow_val_change=True)
    config.update({'PD_model_lr': config.M_transform_lr}, allow_val_change=True)
    config.update({'PD_model_wd': config.M_transform_wd}, allow_val_change=True)

if config.MRef_BG_layer_cnt==-1:
    config.update({'MRef_BG_layer_cnt': config.refine_layer_cnt}, allow_val_change=True)

if config.MRef_BG_hidden_unit==-1:
    config.update({'MRef_BG_hidden_unit': config.refine_hidden_unit}, allow_val_change=True)

if config.MRef_BG_filter_size==-1:
    config.update({'MRef_BG_filter_size': config.refine_filter_size}, allow_val_change=True)

if config.MRef_FG_layer_cnt==-1:
    config.update({'MRef_FG_layer_cnt': config.refine_layer_cnt}, allow_val_change=True)

if config.MRef_FG_hidden_unit==-1:
    config.update({'MRef_FG_hidden_unit': config.refine_hidden_unit}, allow_val_change=True)

if config.MRef_FG_filter_size==-1:
    config.update({'MRef_FG_filter_size': config.refine_filter_size}, allow_val_change=True)

if config.MRef_A_layer_cnt==-1:
    config.update({'MRef_A_layer_cnt': config.refine_layer_cnt}, allow_val_change=True)

if config.MRef_A_hidden_unit==-1:
    config.update({'MRef_A_hidden_unit': config.refine_hidden_unit}, allow_val_change=True)

if config.MRef_A_filter_size==-1:
    config.update({'MRef_A_filter_size': config.refine_filter_size}, allow_val_change=True)

if config.test_batch_size==-1:
    config.update({'test_batch_size': config.batch_size}, allow_val_change=True)


config.tran_filter_sizes = [int(i) for i in list(str(config.tran_filters))]
config.PD_model_filter_sizes =[int(i) for i in list(str(config.PD_model_filters))]

for k in config.keys():
    if "_lr" in str(k) and config.gainLR!=1:
        print("update",k)
        v = config[k]
        wandb.config.update({k: v*config.gainLR}, allow_val_change=True)

config.model_name_constrained = generate_name()

###all other config parameters are in some way constrained, and marked as such by their names!
###these need to be set depending on the dataset
# config.res_y_constrained,config.res_x_constrained = list(trainloader)[0].shape[2:]
config.res_x_constrained = config.res_x
config.res_y_constrained = config.res_y

###these must be calculated depending on other parameters
config.window_padding_constrained = config.max_result_speed

config.image_pad_size_old_constrained = int(config.stride * ((config.window_size - 1) // config.stride))
config.num_windows_y_old_constrained = (
                                                   config.res_y_constrained + 2 * config.image_pad_size_old_constrained - config.window_size) // config.stride + 1
config.num_windows_x_old_constrained = (
                                                   config.res_x_constrained + 2 * config.image_pad_size_old_constrained - config.window_size) // config.stride + 1
config.num_windows_total_old_constrained = config.num_windows_x_old_constrained * config.num_windows_y_old_constrained

config.image_pad_size_constrained = int(
    config.stride * (((config.window_size + 2 * config.window_padding_constrained) - 1) // config.stride))
config.num_windows_y_constrained = (
                                               config.res_y_constrained + 2 * config.image_pad_size_constrained - config.window_size - 2 * config.window_padding_constrained) // config.stride + 1
config.num_windows_x_constrained = (
                                               config.res_x_constrained + 2 * config.image_pad_size_constrained - config.window_size - 2 * config.window_padding_constrained) // config.stride + 1
config.num_windows_total_constrained = config.num_windows_x_constrained * config.num_windows_y_constrained

if ((config.res_x_constrained - 1) % config.stride != 0) or ((config.res_y_constrained - 1) % config.stride != 0):
    print("Not compatible stride", config.res_x_constrained, config.stride)
    sys.exit(0)

###select torch device
avDev = torch.device("cpu")
cuda_devices = list()
if torch.cuda.is_available():
    cuda_devices = [0]
    avDev = torch.device("cuda:" + str(cuda_devices[0]))
    if (len(cuda_devices) > 0):
        torch.cuda.set_device(cuda_devices[0])
print("avDev:", avDev)
dmsg('os.environ["CUDA_VISIBLE_DEVICES"]')
if config.tqdm:
    mytqdm=tqdm

with torch.no_grad():
    LG_Sigma = torch.zeros(1).to(avDev) + config.lg_sigma_bias
    if config.window_type == 'Pascal':
        window = get_pascal_window(config.window_size).to(avDev)
    elif config.window_type == 'ConfGaussian':
        window = get_ACGW(windowSize=config.window_size, sigma=LG_Sigma).detach()
    else:
        window = get_2D_Gaussian(resolution=config.window_size, sigma=LG_Sigma * config.window_size)[0, 0, :, :]

    unit_test_args = dict(res_y=config.res_y_constrained, res_x=config.res_x_constrained, H=config.stride, avDev=avDev,
                          pS=config.window_padding_constrained, bS=config.batch_size)
    LFT_unit_test(window, **unit_test_args)
    del (window)
    del (LG_Sigma)

inference_phase = config.inference
is_sweep = wandb.run.sweep_id is not None

print("config:{")
pretty(config._items,hyperparameter_defaults)
print("}")
critBCE = torch.nn.BCELoss()
critL1 = torch.nn.L1Loss()
critL2 = torch.nn.MSELoss()
critSSIM = kornia.losses.SSIMLoss(window_size=9, reduction='mean')
critHybrid = CombinedLossDiscounted()

LG_Sigma_A = None
LG_TeS = None
LG_UpdS = None
LG_UpdT = None
M_A = None
M_FG = None
M_BG = None
MRef_A = None
MRef_FG = None
MRef_BG = None
MRef_Out = None
M_transform = None
PD_model = None

LG_Sigma = torch.tensor(config.lg_sigma_bias, requires_grad=True, device=avDev)  ##TODO

paramN = []
minLR = []
wD = []

chennelC = 3 if config.color else 1

if config.segmentation:
    if config.init_A_with_T:
        LG_Sigma_A = torch.tensor(0.45, requires_grad=True, device=avDev)
        if "NGSIM" in config.data_key:
            LG_Sigma_A = torch.tensor(0.1278, requires_grad=True, device=avDev)
        elif "MotionSegmentation" == config.data_key:
            LG_Sigma_A = torch.tensor(0.2188, requires_grad=True, device=avDev)
        paramN.append('LG_Sigma_A')
        minLR.append(config.gain_update_lr)
        wD.append(10e-12)

        PD_model = phaseDiffModel(config).to(avDev)
        paramN.append('PD_model')
        minLR.append(config.PD_model_lr)
        wD.append(config.PD_model_wd)

    #Learnable Gain Update Spatials (A,FG,BG). It will be used with sigmoid. Default after sigmoid = 0.5
    LG_UpdS = torch.zeros(3,config.sequence_length, requires_grad=True, device=avDev)
    if "NGSIM" in config.data_key and config.sequence_length==10:
        LG_UpdS = torch.tensor([[ 4.6171,  5.2717, 10.1665, 11.3596, 12.2106,  0.0000,  0.0000,  0.0000, 0.0000,  0.0000],  [11.2458, 12.9779, 14.4671, 14.7118, 15.8778,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],  [ 9.7147,  6.8452,  0.8428,  0.5595,  0.6800,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]], requires_grad=True, device=avDev)
    elif "MotionSegmentation" == config.data_key and config.sequence_length==10:
        LG_UpdS = torch.tensor([[-1.1536,  0.1880,  7.0582,  5.2182,  7.0127,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],  [ 5.1158,  6.8903, 10.4487,  9.6647, 12.1630,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],  [-6.5552,  0.2025,  1.7952,  1.6631,  2.9506,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]], requires_grad=True, device=avDev)
    paramN.append('LG_UpdS')
    minLR.append(config.gain_update_lr)
    wD.append(10e-12)

    #Learnable Gain Update Transformation (T). It will be used with sigmoid. Default after sigmoid = 0.5
    LG_UpdT = torch.zeros(1,config.sequence_length, requires_grad=True, device=avDev)
    if "NGSIM" in config.data_key and config.sequence_length==10:
        LG_UpdT = torch.tensor([[ 0.0000,  0.1049,  1.5577,  0.1450, -0.0979,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]], requires_grad=True, device=avDev)
    elif "MotionSegmentation" == config.data_key and config.sequence_length==10:
        LG_UpdT = torch.tensor([[ 0.0000, -2.5745,  0.5905,  0.1705,  0.0219,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]], requires_grad=True, device=avDev)
    paramN.append('LG_UpdT')
    minLR.append(config.gain_update_lr)
    wD.append(0)

    #Learnable Gain to between PDFG and PDA. It will be used with sigmoid. Default after sigmoid = 0.5
    LG_TeS = torch.zeros(1,config.sequence_length, requires_grad=True, device=avDev)
    if "NGSIM" in config.data_key and config.sequence_length==10:
        LG_TeS = torch.tensor([[0.3233, 0.3762, 0.2739, 3.8217, 1.2684, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]], requires_grad=True, device=avDev)
    elif "MotionSegmentation" == config.data_key and config.sequence_length==10:
        LG_TeS = torch.tensor([[-1.1181, -5.4842, -0.2704, -0.1654, -0.9836,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]], requires_grad=True, device=avDev)
    paramN.append('LG_TeS')
    minLR.append(config.gain_update_lr)
    wD.append(0)


    M_transform = cellTransportRefine(config).to(avDev)
    paramN.append('M_transform')
    minLR.append(config.M_transform_lr)
    wD.append(config.M_transform_wd)


    MRef_A = DenseNetLikeModel(inputC=1,outputC=1,layerCount=config.MRef_A_layer_cnt, hiddenF=config.MRef_A_hidden_unit,
                                filterS=config.MRef_A_filter_size,nonlin=config.ref_non_lin,lastNonLin=False,initW=config.initW,bn=config.useBN).to(avDev)
    paramN.append('MRef_A')
    minLR.append(config.refine_lr)
    wD.append(config.refine_wd)

    MRef_FG = DenseNetLikeModel(inputC=chennelC,outputC=chennelC,layerCount=config.MRef_FG_layer_cnt, hiddenF=config.MRef_FG_hidden_unit,
                                filterS=config.MRef_FG_filter_size,nonlin=config.ref_non_lin,lastNonLin=False,initW=config.initW,bn=config.useBN).to(avDev)
    paramN.append('MRef_FG')
    minLR.append(config.refine_lr)
    wD.append(config.refine_wd)

    if config.share_all_possible_models:
        MRef_BG = MRef_FG
    else:
        MRef_BG = DenseNetLikeModel(inputC=chennelC,outputC=chennelC,layerCount=config.MRef_BG_layer_cnt, hiddenF=config.MRef_BG_hidden_unit,
                                    filterS=config.MRef_BG_filter_size,nonlin=config.ref_non_lin,lastNonLin=False,initW=config.initW,bn=config.useBN).to(avDev)
        paramN.append('MRef_BG')
        minLR.append(config.refine_lr)
        wD.append(config.refine_wd)

    
    if config.refine_output:
        if config.share_all_possible_models:
            MRef_Out = MRef_FG
        else:
            MRef_Out = DenseNetLikeModel(inputC=chennelC,outputC=chennelC,layerCount=config.refine_layer_cnt, hiddenF=config.refine_hidden_unit,
                                        filterS=config.refine_filter_size,nonlin=config.ref_non_lin,lastNonLin=False,initW=config.initW,bn=config.useBN).to(avDev)
            paramN.append('MRef_Out')
            minLR.append(config.refine_lr)
            wD.append(config.refine_wd)



    M_A = DenseNetLikeModel(inputC=chennelC+1 if config.init_A_with_T else chennelC,outputC=1,layerCount=config.refine_layer_cnt_a, hiddenF=config.refine_hidden_unit,
                            filterS=config.refine_filter_size,gain=0.4#to make sure we start with blank A
                            ,nonlin=config.ref_non_lin,lastNonLin=False,initW=config.initW,bn=config.useBN).to(avDev)
    paramN.append('M_A')
    minLR.append(config.refine_lr)
    wD.append(config.refine_wd)

    inpC = 1+chennelC if config.add_A_to_M_BG else chennelC
    if not config.add_A_to_M_BG and config.share_all_possible_models:
        M_BG = MRef_FG
    else:
        M_BG = DenseNetLikeModel(inputC=inpC,outputC=chennelC,layerCount=config.refine_layer_cnt, hiddenF=config.refine_hidden_unit,
                                filterS=config.refine_filter_size,nonlin=config.ref_non_lin,lastNonLin=False,initW=config.initW,bn=config.useBN).to(avDev)#One for input one for Alpha
        paramN.append('M_BG')
        minLR.append(config.refine_lr)
        wD.append(config.refine_wd)

else:
    if config.refine_output:
        MRef_Out = DenseNetLikeModel(inputC=chennelC,outputC=chennelC,layerCount=config.refine_layer_cnt, hiddenF=config.refine_hidden_unit,
                                    filterS=config.refine_filter_size,nonlin=config.ref_non_lin,lastNonLin=False,initW=config.initW,bn=config.useBN).to(avDev)
        paramN.append('MRef_Out')
        minLR.append(config.refine_lr)
        wD.append(config.refine_wd)

    M_transform = cellTransportRefine(config).to(avDev)
    paramN.append('M_transform')
    minLR.append(config.M_transform_lr)
    wD.append(config.M_transform_wd)


loadModels(config.load_model,config.excludeLoad)

paramList = []

max_lrs = []
for i, p in enumerate(paramN):
    par = eval(p)
    paramList.append({'params': [par] if type(par) is torch.Tensor else par.parameters(),
                      'lr': minLR[i],
                      'weight_decay': wD[i],
                      'name': p})
    max_lrs.append(minLR[i]*config.oneCycleMaxLRGain)

optimizer = eval('torch.optim.'+config.optimizer+'(paramList)')


numParam = 0
for par in optimizer.param_groups:
    numParam += sum(l.numel() for l in par["params"] if l.requires_grad)

config.parameter_number_constrained = numParam
wandblog({"numParam": numParam})


for par in optimizer.param_groups:
    print(Fore.CYAN, par["name"], sum(l.numel() for l in par["params"] if l.requires_grad), Fore.RESET)
    for l in par["params"]:
        if l.requires_grad:
            print(Fore.MAGENTA, l.shape, "  =>", l.numel(), Fore.RESET)

print("Number of trainable params: ", Fore.RED + str(numParam) + Fore.RESET)
if is_sweep and numParam > config.max_num_param_tol:
    wandblog({"cstate": 'High Param', 'sweep_metric': 1.1},commit=True)
    print(Fore.RED, "TOO high #Params ", numParam, " > ", config.max_num_param_tol, Fore.RESET)
    sys.exit(0)

trainloader, validloader, testloader = get_data_loaders(config,key=config.data_key,
                                            size=(config.res_x_constrained, config.res_y_constrained),
                                            batch_size=config.batch_size,test_batch_size=config.test_batch_size, num_workers=config.num_workers, limit=config.limitDS,
                                            sequence_length=config.sequence_length)

if len(config.cmd)>1:
    exec(config.cmd)

print(Fore.MAGENTA,'Trainloader:',len(trainloader),'Validloader:',len(validloader),'Testloader:',len(testloader),Fore.RESET)

startOptimFromIndex = 0
lGains = [i * 1.2 for i in range(config.sequence_length + 1, startOptimFromIndex + 1, -1)]
# lGains = lGains[::-1]
lGains[0]=0.4
lGains[1]=0.8
lGains = [i / sum(lGains) for i in lGains]

print(lGains)

li = 0
ui = 1
t = 0  # Reset the step counter
bestFullL2Loss = 1e25
SHOWINTER = False

print(Fore.MAGENTA + ("Sweep!" if is_sweep else "Normal Run") + Fore.RESET)
if inference_phase:
    print(Fore.CYAN + "Inference Phase!" + Fore.RESET)
    bs = config.batch_size
    inferenceRes = []
    paintEvery = 10
    paintOncePerEpoch = False
    runs = 1
else:
    print(Fore.GREEN + "Training Phase!" + Fore.RESET)
    bs = config.batch_size
    paintEvery = None
    paintOncePerEpoch = True
    runs = config.epochs if (is_sweep or config.lr_scheduler == 'OneCycleLR' )else 100000000

torch.set_grad_enabled(not inference_phase)

if not inference_phase:
    if config.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=5, threshold=0.0001,
                                                        cooldown=0, verbose=True, min_lr=0.000001)
    elif config.lr_scheduler == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lrs, total_steps=len(trainloader)*config.epochs)

    else:
        class dummyOpt():
            def step(self,inp=None):
                pass
            def get_last_lr(self):
                return [0.0]
            def get_lr(self):
                return [0.0]
        scheduler=dummyOpt()


last_improved=0
while t < runs:
    # torch.autograd.set_detect_anomaly(True)
    with torch.no_grad():
        ###setting window determines the LFT tapering window!
        if config.window_type == 'Pascal':
            window = get_pascal_window(config.window_size).to(avDev)
        elif config.window_type == 'ConfGaussian':
            window = get_ACGW(windowSize=config.window_size, sigma=LG_Sigma).detach()
        else:
            window = get_2D_Gaussian(resolution=config.window_size, sigma=LG_Sigma * config.window_size)[0, 0, :, :]
    # t starts at zero #https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232
    if t == 1 and torch.backends.cudnn.benchmark == True:
        torch.cuda.empty_cache()
    wandbLog = {}

    if not inference_phase:
        start_time = time.time()
        phase = 'Training'
        print('Going through ',phase,' set at epoch', t + 1, '...')
        train_c, train_ll,ANorm_ll,AVar_ll,AEstNorm_ll,AEstVar_ll,angAfterNorm_ll = (0,0,0,0,0,0,0)
        ANorm_t,AVar_t,AEstNorm_t,AEstVar_t,angAfterNorm_t = (0,0,0,0,0)
        setEvalTrain(True)
        setIfAugmentData(config.augment_data)
        if paintOncePerEpoch:
            paintEvery = rand.randint(1,len(trainloader))

        for mini_batch in mytqdm(trainloader):
            optimizer.zero_grad()
            train_c += 1
            if config.data_key=="MotionSegmentation":
                data =mini_batch["GT"].to(avDev)
            else:
                data =mini_batch.to(avDev)
            

            if paintOncePerEpoch:
                show_images = train_c == paintEvery
            else:
                show_images = True if train_c % paintEvery == 0 else False

            show_images = show_images and not config.dryrun
            pred_frames, angAfterNorm,ANorm,AVar,AEstNorm,AEstVar = predict(data, window, config,LG_Sigma_A,LG_TeS,LG_UpdS,LG_UpdT,M_A,M_FG,M_BG,MRef_A,MRef_FG,MRef_BG,MRef_Out, M_transform,PD_model,phase=phase, log_vis=show_images)

            if pred_frames is not False:
                netOut = pred_frames[:, startOptimFromIndex:]
                target = data[:, startOptimFromIndex:]
                ANorm_t +=ANorm.item();AVar_t +=AVar.item();AEstNorm_t +=AEstNorm.item();AEstVar_t +=AEstVar.item();angAfterNorm_t +=angAfterNorm.item()
                ANorm_loss = (ANorm-config.ANormTarget).abs()*config.ANormGain
                AVar_loss = (AVar-config.AVarTarget).abs()*config.AVarGain
                AEstNorm_loss = (AEstNorm-config.ANormTarget).abs()*config.AEstNormGain
                AEstVar_loss = (AEstVar-config.AVarTarget).abs()*config.AEstVarGain
                angAfterNorm_loss = angAfterNorm * config.angleNormGain 


                if config.loss=="Hybrid":
                    ll = 0
                    l = torch.pow((target - netOut), 2)
                    for i in range(config.sequence_length - startOptimFromIndex):
                        ll += l[:, i, :, :, :].mean() * lGains[i]
                    
                    ll = (1-config.hybridGain)*ll + (config.hybridGain)*critSSIM(netOut.view(-1,1,netOut.shape[3],netOut.shape[4]),
                        target.view(-1,1,target.shape[3],target.shape[4]))
                elif config.loss=="L2Pred":
                    ll = critL2(pred_frames[:,config.sequence_seed:], target[:,config.sequence_seed:])
                else:
                    ll = critL2(netOut,target)

                ll = ll + ANorm_loss + AVar_loss + AEstNorm_loss + AEstVar_loss + angAfterNorm_loss
                ll.backward()
                    

                optimizer.step()
                if not inference_phase and config.lr_scheduler == 'OneCycleLR':
                    scheduler.step()

                train_ll += ll.item()
                ANorm_ll += ANorm_loss.item()
                AVar_ll += AVar_loss.item()
                AEstNorm_ll += AEstNorm_loss.item()
                AEstVar_ll += AEstVar_loss.item()
                angAfterNorm_ll += angAfterNorm_loss.item()
            else:
                print(Fore.RED + "NAN found!" + Fore.RESET)
                raise BaseException("NAN error!")

            wandblog(wandbLog, commit=(not paintOncePerEpoch and show_images))#Messy steps in wandb

        wandbLog["trainLoss"] = train_ll / train_c
        wandbLog["ANormLoss"] = ANorm_ll / train_c
        wandbLog["AVarLoss"] = AVar_ll / train_c
        wandbLog["AEstNormLoss"] = AEstNorm_ll / train_c
        wandbLog["AEstVarLoss"] = AEstVar_ll / train_c
        wandbLog["angAfterNormLoss"] = angAfterNorm_ll / train_c

        wandbLog["ANorm"] = ANorm_t / train_c
        wandbLog["AVar"] = AVar_t / train_c
        wandbLog["AEstNorm"] = AEstNorm_t / train_c
        wandbLog["AEstVar"] = AEstVar_t / train_c
        wandbLog["angAfterNorm"] = angAfterNorm_t / train_c

        if config.segmentation:
            customHtml = str(paramN)+"<br>"
            # customHtml += ', '.join('{:0.5f}'.format(i) for i in (scheduler.get_lr()))+"<br>"
            customHtml += niceL2S('torch.sigmoid(LG_UpdS[0:3,0:seedNumber])',torch.sigmoid(LG_UpdS[0:3,0:config.sequence_seed]))+"<br>"
            customHtml += niceL2S('torch.sigmoid(LG_UpdT[:,1:seedNumber])',torch.sigmoid(LG_UpdT[:,1:config.sequence_seed]))+"<br>"
            customHtml += niceL2S('torch.sigmoid(LG_TeS[:,1:seedNumber])',torch.sigmoid(LG_TeS[:,1:config.sequence_seed]))+"<br>"
            wandbLog["html"] = wandb.Html(customHtml)
        
        print('...done! ',Fore.LIGHTYELLOW_EX+"Took: {:.2f}".format(time.time() - start_time)+" Sec"+Fore.RESET)

    if t%config.validate_every>0 and not inference_phase and not is_sweep:
        print(Fore.LIGHTYELLOW_EX," ==> Skip validation",(config.validate_every-(t%config.validate_every)),"!...",Fore.RESET)
        wandblog(wandbLog, commit=True)
        t=t+1
        continue

    start_time = time.time()
    tPhase = ('Validation' if not inference_phase else 'Testing')
    tloader =  validloader if not inference_phase else testloader
    print('Going through ',tPhase,' set...')


    with torch.no_grad():
        setEvalTrain(False)
        setIfAugmentData(False)
        valid_c, bceFull, bceFullMin, L1FullNet, L2FullNet, ssimFull, ssimHybrid = (0, 0, 0, 0, 0, 0, 0)
        if paintOncePerEpoch:
            paintEvery = rand.randint(1, len(tloader))
        for mini_batch in mytqdm(tloader):
            if config.data_key=="MotionSegmentation":
                data =mini_batch["GT"].to(avDev)
            else:
                data =mini_batch.to(avDev)

            if paintOncePerEpoch:
                show_images = valid_c == paintEvery
            else:
                show_images = True if valid_c % paintEvery == 0 else False
            show_images = show_images and not config.dryrun
            pred_frames, _,_,_,_,_ = predict(data, window, config,LG_Sigma_A,LG_TeS,LG_UpdS,LG_UpdT,M_A,M_FG,M_BG,MRef_A,MRef_FG,MRef_BG,MRef_Out,M_transform,PD_model, phase=tPhase, log_vis=show_images)
            netOut = pred_frames[:,  config.sequence_seed:,:,:config.comp_fair_y,:config.comp_fair_x]
            target = data[:,  config.sequence_seed:,:,:config.comp_fair_y,:config.comp_fair_x]
            valid_c += 1
            bceFull += critBCE(netOut, target)
            bceFullMin += critBCE(target, target)
            L1FullNet += critL1(netOut, target)
            L2FullNet += critL2(netOut, target)
            netOutSSIM = netOut.reshape(-1, 1, netOut.shape[3], netOut.shape[4])
            targetSSIM = target.reshape(-1, 1, target.shape[3], target.shape[4])
            ssimFull += critSSIM(netOutSSIM,targetSSIM)
            ssimHybrid += critHybrid(netOutSSIM,targetSSIM)

            wandblog(wandbLog, commit=(not paintOncePerEpoch and show_images))  # Messy steps in wandb


    if not inference_phase and config.lr_scheduler == 'ReduceLROnPlateau':
        scheduler.step(L2FullNet.item() / valid_c)
    wandbLog["hybridSSIMLoss"] = ssimHybrid.item() / valid_c
    wandbLog["L1FullLoss"] = L1FullNet.item() / valid_c
    wandbLog["L2FullLoss"] = L2FullNet.item() / valid_c
    wandbLog["bceFullMin"] = bceFullMin.item() / valid_c
    wandbLog["bceFull"] = bceFull.item() / valid_c
    wandbLog["SSIMFull"] = ssimFull.item() / valid_c

    paramGain = 1 if (
                config.parameter_number_constrained < config.max_num_param_tol / 3.) else config.parameter_number_constrained / (
                config.max_num_param_tol / 3.)
    wandbLog["paramGain"] = paramGain
    wandbLog["sweep_metric"] = wandbLog["L2FullLoss"] * paramGain

    if wandbLog["L2FullLoss"] < bestFullL2Loss and not inference_phase:
        last_improved = t
        if not config.dryrun:
            nameM = config.data_key[:6]+"_"+"{:.4f}".format(wandbLog["L2FullLoss"]).replace(".","_")+"_"+wandb.run.project+"_"+wandb.run.name+"_"+config.model_name_constrained.replace("-","_")
        else:
            nameM = config.data_key[:6]+"_"+"{:.4f}".format(wandbLog["L2FullLoss"]).replace(".","_")+"_"+config.model_name_constrained.replace("-","_")
        nameM = nameM.replace("-","_").replace(".","_")
        print('Model improved:',Fore.GREEN+str(wandbLog["L2FullLoss"])+Fore.RESET,' Model saved! :', nameM)
        bestFullL2Loss = wandbLog["L2FullLoss"]
        saveModels(nameM)
        cFile=wandb.run._settings._sync_dir+'/files/config.yaml'
        if os.path.exists(cFile):
            open('savedModels/'+nameM+ ".yml", 'wb').write(open(cFile, 'rb').read())

    if is_sweep and config.kill_no_improve>=0 and (t-last_improved)>config.kill_no_improve:
        print(Fore.RED, "No improvement!", (t-last_improved),t,last_improved, Fore.RESET)
        wandblog(wandbLog,commit=True)
        sys.exit(0)

    if t>=config.max_loss_tol_index and is_sweep and wandbLog['sweep_metric'] > config.max_loss_tol_general:
        wandbLog["cstate"]= 'High Loss'
        print(Fore.RED, "Loss too high!", wandbLog['sweep_metric'], is_sweep, Fore.RESET)
        wandblog(wandbLog,commit=True)
        sys.exit(0)

    if inference_phase:
        inferenceRes.append(
            [bceFullMin.item() / valid_c, bceFull.item() / valid_c, L1FullNet.item() / valid_c
            , L2FullNet.item() / valid_c, ssimFull.item() / valid_c])


    t = t + 1
    url = "DRY"
    if not config.dryrun:
        url=click.style(wandb.run.get_url().replace('https://',""), underline=True, fg="blue")
    print('...done! ',Fore.LIGHTYELLOW_EX+"Took: {:.2f}".format(time.time() - start_time)+" Sec"+Fore.RESET,
        'Run:',url)
    wandblog(wandbLog, commit=True)

    if (inference_phase and t >= runs):
        inferenceRes = np.array(inferenceRes)
        print("BCEMin=", inferenceRes[:, 0].mean(), " BCE=", inferenceRes[:, 1].mean(), " L1=",
              inferenceRes[:, 2].mean(), " L2=", inferenceRes[:, 3].mean(), " SSIM=", inferenceRes[:, 4].mean())
        break
wandblog({"cstate": 'Done'},commit=True)
print("Run completed!")
