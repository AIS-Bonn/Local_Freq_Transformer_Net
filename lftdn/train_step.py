import torch
import wandb
from lftdn.LFT import compact_LFT, compact_iLFT
from asset.utils import getPhaseDiff, getPhaseAdd, clmp, listToTensor, showSeq, dimension, li, ui, show_phase_diff, \
    logMultiVideo, wandblog,dmsg,getAvgDiff,manyListToTensor
from lftdn.window_func import  get_2D_Gaussian,get_ACGW
from torch.functional import F

def getOneC(inp,dim=2):
    if inp.shape[dim]==1:
        return inp
    return inp.mean(dim=dim).unsqueeze(dim)

def predict(dataSet, window, config,LG_Sigma_A,LG_TeS,LG_UpdS,LG_UpdT,M_A,M_FG,M_BG,MRef_A,MRef_FG,MRef_BG,MRef_Out, M_transform,PD_model, phase='train', log_vis=True):
    if config.segmentation:
        return predict_motionSeg(dataSet, window, config,LG_Sigma_A,LG_TeS,LG_UpdS,LG_UpdT,M_A,M_FG,M_BG,MRef_A,MRef_FG,MRef_BG,MRef_Out, M_transform,PD_model, phase, log_vis)
    else:
        return predict_noMotion(dataSet, window,config,MRef_Out,M_transform,phase, log_vis)

def predict_motionSeg(dataSet, window, config,LG_Sigma_A,LG_TeS,LG_UpdS,LG_UpdT,M_A,M_FG,M_BG,MRef_A,MRef_FG,MRef_BG,MRef_Out, M_transform,PD_model, phase='train', log_vis=True):
    eps = 1e-8
    angAfterNorm = torch.tensor(0., requires_grad=True)
    ANorm = torch.tensor(0., requires_grad=True)
    AVar = torch.tensor(0., requires_grad=True)
    AEstNorm = torch.tensor(0., requires_grad=True)
    AEstVar = torch.tensor(0., requires_grad=True)
    PRes = []
    if log_vis:
        T_hist = []
        ARes=[]
        AURes=[]
        FGRes = []
        FGURes = []
        BGRes=[]
        BGURes=[]
        CRes = []
        T_ref_hist = []
    

    AEstmiate = []
    FGEstmiate = []
    BGEstmiate = []
    oneCDS = getOneC(dataSet)
    if config.init_A_with_T:
        windowTmp =get_ACGW(windowSize=config.window_size, sigma=abs(LG_Sigma_A)+eps)

    for idx2A in range(config.seeAtBegining):
        if config.init_A_with_T:
            PDD,PDDEnergy = getPhaseDiff(compact_LFT(oneCDS[:,idx2A+1]+ eps, windowTmp, config),
                                            compact_LFT(oneCDS[:,idx2A  ]+ eps, windowTmp, config),config.use_energy)
            PDD2 = PD_model(PDD,PDDEnergy)
            AEs, _ = compact_iLFT(PDD2, windowTmp, PDD, config,is_inp_complex=False,move_window_according_T=False)
            AEs = M_A(torch.cat([dataSet[:,idx2A],AEs],dim=1))
        else:
            AEs = M_A((dataSet[:,idx2A]))
        
        AEstmiate.append(clmp(AEs,True,skew=config.A_skew))

        AEstNorm = AEstNorm+AEstmiate[-1].abs().mean()
        AEstVar = AEstVar+AEstmiate[-1].var() 
        FGEstmiate.append(AEstmiate[idx2A]*dataSet[:,idx2A])
        if not config.init_BG_with_GT_and_A:
            if config.add_A_to_M_BG:
                BGEstmiate.append(clmp(M_BG(torch.cat([((1-AEstmiate[idx2A])*dataSet[:,idx2A]),AEstmiate[idx2A]],dim=1)),False))
            else:
                BGEstmiate.append(clmp(M_BG((1-AEstmiate[idx2A])*dataSet[:,idx2A]),False))
        else:
            BGEstmiate.append((1-AEstmiate[idx2A])*dataSet[:,idx2A])

    
    if not config.init_T_with_GT:
        if not config.update_T_just_by_FG:
            PDA,PDAEnergy = getPhaseDiff(compact_LFT(AEstmiate[1]+ eps, window, config),
                                        compact_LFT(AEstmiate[0]+ eps, window, config),config.use_energy)
        
        PDFG,PDFGEnergy = getPhaseDiff(compact_LFT(getOneC(FGEstmiate[1],dim=1)+ eps, window, config),
                                       compact_LFT(getOneC(FGEstmiate[0],dim=1)+ eps, window, config),config.use_energy) #TODO: Why use FG in calculation of first T?

        if not config.update_T_just_by_FG:
            T = (PDFG*torch.sigmoid(LG_TeS[0,0])+PDA*(1-torch.sigmoid(LG_TeS[0,0])))
        else:
            T = PDFG
    else:
        T,TEnergy = getPhaseDiff(compact_LFT(oneCDS[:,1]+ eps, window, config),
                                    compact_LFT(oneCDS[:,0]+ eps, window, config),config.use_energy)

    if config.use_energy:
        if not config.init_T_with_GT:
            if not config.update_T_just_by_FG:
                Energy = (PDFGEnergy*torch.sigmoid(LG_TeS[0,0])+PDAEnergy*(1-torch.sigmoid(LG_TeS[0,0])))
            else:
                Energy = PDFGEnergy
        else:
            Energy = TEnergy
    else:
        Energy = None


    for i in range(config.sequence_length):
        ##apply transform model
        T, visB, visA, angAfterNormTmp = M_transform(T,Energy, 
                                                    first = True if i==config.start_T_index else False,
                                                    throwAway = True if i<config.start_T_index else False)

        angAfterNorm = angAfterNorm + angAfterNormTmp
        if log_vis:
            T_hist.append(visB)
            T_ref_hist.append(visA)


        if i == 0:
            NA = AEstmiate[0]
            NFG = FGEstmiate[0]
            NBG = BGEstmiate[0]
        else:
            A_fft = compact_LFT(A+ eps, window, config)
            NA_fft = getPhaseAdd(A_fft, T)
            NA_, _ = compact_iLFT(NA_fft, window, T, config)

            FG_fft = compact_LFT(FG+ eps, window, config)
            if FG.shape[1]==1:
                ET = T
            else:
                ET = T.unsqueeze(2).expand(T.shape[0],T.shape[1],FG.shape[1],T.shape[2],T.shape[3],T.shape[4]).reshape_as(FG_fft)
            NFG_fft = getPhaseAdd(FG_fft, ET)
            NFG_, _ = compact_iLFT(NFG_fft, window, ET, config,channel=FG.shape[1])

            NBG_ = BG

            if i<config.seeAtBegining:  
                gtmp = float(i)/config.seeAtBegining
                NA = AEstmiate[i]*(1-gtmp) + gtmp*NA_
                NFG = FGEstmiate[i]*(1-gtmp) + gtmp*NFG_
                NBG = BGEstmiate[i]*(1-gtmp) + gtmp*NBG_
            else:
                NA = NA_
                NFG = NFG_
                NBG = NBG_

            if i<config.sequence_seed or config.allways_refine:
                #print("NNNN",NA.shape)
                NA = clmp(MRef_A(NA),True,skew=config.A_skew)
                
                NFG = clmp(MRef_FG(NFG),False)*((NA+eps)**(1/float(config.alpha_root)))
                NBG = clmp(MRef_BG(NBG),False)
            else:
                NA = clmp(NA,True,skew=config.A_skew)
                NFG = clmp(NFG,False)*((NA+eps)**(1/float(config.alpha_root)))
                NBG = clmp(NBG,False)


        ANorm = ANorm+NA.abs().mean()    
        AVar = AVar+NA.var() 
        Res = (NA*NFG)+((1-NA)*NBG)
        if config.refine_output:
            Res= clmp(MRef_Out(Res),False)
        PRes.append(Res)
        
        if log_vis:
            ARes.append(NA.detach())
            FGRes.append(NFG.detach())
            BGRes.append(NBG.detach())

        if i<config.sequence_seed:
            inter = Res-dataSet[:,i,:,:,:]
            Agrad = getOneC(NFG*inter-(NBG*inter),dim=1)
            BGgrad = (1-NA)*inter
            FGgrad = NA*inter

            NA =  clmp((NA-Agrad*torch.sigmoid(LG_UpdS[0,i])),True,skew=config.A_skew)
            NFG = clmp(NFG-FGgrad*torch.sigmoid(LG_UpdS[1,i]),False)*((NA+eps)**(1/float(config.alpha_root)))
            NBG = clmp(NBG-BGgrad*torch.sigmoid(LG_UpdS[2,i]),False)
            correction = (NA*NFG)+((1-NA)*NBG)
        else:
            inter = torch.zeros_like(Res)
            Agrad = torch.zeros_like(Res)
            BGgrad = torch.zeros_like(Res)
            FGgrad = torch.zeros_like(Res)
            correction = torch.zeros_like(Res)
        

        if log_vis:
            AURes.append(NA.detach())
            FGURes.append(NFG.detach())
            BGURes.append(NBG.detach())
            CRes.append(correction.detach())

        if i>0:
            if i<config.sequence_seed:
                if not config.update_T_just_by_FG:
                    PDA,PDAEnergy = getPhaseDiff(compact_LFT(NA+ eps, window, config), compact_LFT(A+ eps, window, config),config.use_energy)
                PDFG,PDFGEnergy = getPhaseDiff(compact_LFT(getOneC(NFG,dim=1)+ eps, window, config), compact_LFT(getOneC(FG,dim=1)+ eps, window, config),config.use_energy)
                
                
                if config.use_energy and not config.update_T_just_by_FG:
                        Energy = (PDFGEnergy*torch.sigmoid(LG_TeS[0,i])+PDAEnergy*(1-torch.sigmoid(LG_TeS[0,i])))
                else:
                    Energy = PDFGEnergy

                if not config.update_T_just_by_FG:
                    TEstimate = (PDFG*torch.sigmoid(LG_TeS[0,i])+PDA*(1-torch.sigmoid(LG_TeS[0,i])))
                else:
                    TEstimate = PDFG

                NT = ((1-torch.sigmoid(LG_UpdT[0,i]))*T)+(torch.sigmoid(LG_UpdT[0,i])*TEstimate)
            else:
                TEstimate = torch.zeros_like(TEstimate)

            T = NT

        A = NA
        FG = NFG
        BG = NBG
        
    PRes = listToTensor(PRes)


    if (torch.isnan(PRes).any()):
        # return False,False,False
        print(torch.pow((dataSet[:, 2:] - PRes[:, 2:]).cpu(), 2).mean().item())
        print(torch.isnan(dataSet - PRes).any())
        raise Exception("NAN Exception")

    if log_vis:
        with torch.no_grad():
            ARes,AURes,FGRes,FGURes,BGRes,BGURes,CRes = manyListToTensor([ARes,AURes,FGRes,FGURes,BGRes,BGURes,CRes])
            AEstmiate,FGEstmiate,BGEstmiate = manyListToTensor([AEstmiate,FGEstmiate,BGEstmiate])
            vis_image_string = phase+" predictions"
            vis_anim_string = phase+' animations'
            header = phase+': '
            setting = {'oneD': (dimension == 1), 'revert': True, 'dpi': 2.4, 'show': False,"vmin":0,"vmax":1}
            L2loss = torch.pow((dataSet - PRes), 2)
            L1loss = 0.5 * (PRes.clamp(0, 1) - dataSet.clamp(0, 1)) + 0.5
            pic = showSeq(False, -1, "Pred,GT,L2,L1", [PRes[li:ui].detach().cpu(),
                          dataSet[li:ui].cpu(), L2loss[li:ui].detach().cpu(),L1loss[li:ui].detach().cpu()],
                          **setting)
            wandblog({vis_image_string: pic},commit=False)

            txtL = vis_image_string+" A,AU,FG,FGU,BG,BGU,Pred,Cor,GT,L2,L1"
            pic = showSeq(False,-1,txtL,[ARes[li:ui].cpu(),AURes[li:ui].cpu(),FGRes[li:ui].cpu(),FGURes[li:ui].cpu(),BGRes[li:ui].cpu(),BGURes[li:ui].cpu(),
                                        PRes[li:ui].detach().cpu(),CRes[li:ui].cpu(),dataSet[li:ui].cpu(),L2loss[li:ui].detach().cpu(),L1loss[li:ui].detach().cpu()],**setting)
            wandblog({txtL: pic},commit=False)

            txtL = vis_image_string+" AEstmiate,FGEstmiate,BGEstmiate"
            pic = showSeq(False,-1,txtL,[AEstmiate[li:ui].cpu(),FGEstmiate[li:ui].cpu(),BGEstmiate[li:ui].cpu()],**setting)
            wandblog({txtL: pic},commit=False)
            
            ####get linear shift encoded in phase diffs
            show_phase_diff(pd_list=T_hist, gt=1-PRes, config=config, title=header + "VF from Phase Diffs")

            show_phase_diff(pd_list=T_ref_hist, gt=1-PRes, config=config, title=header + "VF after M_Transform")

            
            show_phase_diff(pd_list=T_ref_hist, gt=1-AURes, config=config, title=header + "AURes after M_Transform")
            show_phase_diff(pd_list=T_ref_hist, gt=1-FGURes, config=config, title=header + "FGURes after M_Transform")
            show_phase_diff(pd_list=T_ref_hist, gt=1-BGURes, config=config, title=header + "BGURes after M_Transform",clear_motion=True)

            # print('logging stuff')
            
            pt = PRes[li:ui].detach().cpu()
            gt = dataSet[li:ui].cpu()

            logMultiVideo('Prediction, GT, Diff', [pt, gt, 0.5 * (pt.clamp(0, 1) - gt.clamp(0, 1)) + 0.5],
                          vis_anim_string=vis_anim_string)


    angAfterNorm = angAfterNorm / float(config.sequence_length)
    ANorm = ANorm/float(config.sequence_length)
    AVar = AVar/float(config.sequence_length)
    AEstNorm = AEstNorm/float(config.seeAtBegining)
    AEstVar = AEstVar/float(config.seeAtBegining)

    return PRes, angAfterNorm,ANorm,AVar,AEstNorm,AEstVar
 

def predict_noMotion(dataSet, window,config,MRef_Out,M_transform,phase='train', log_vis=True):
    eps = 1e-8
    angAfterNorm = torch.tensor(0., requires_grad=True)
    if log_vis:
        T_hist = []
        T_ref_hist = []

    oneCDS = getOneC(dataSet)
    ###seed
    for i in range(1,config.sequence_seed):
        SP = dataSet[:,i-1]
        SN = dataSet[:,i]
        SPOne = oneCDS[:,i-1]
        SNOne = oneCDS[:,i]
        S_curr = SN

        if  i==1:
            PRes = [SP,SN]
            if log_vis:
                WTs = [torch.zeros_like(SN), torch.zeros_like(SN)]
        else:
            PRes.append(SN)
            if log_vis:
                WTs.append(torch.zeros_like(SN))

        T,Energy = getPhaseDiff(compact_LFT(getOneC(SN,dim=1) + eps, window, config), \
                         compact_LFT(getOneC(SP,dim=1) + eps, window, config),config.use_energy)

        T, visB, visA, _ = M_transform(T,Energy, 
                                            first = True if i==config.start_T_index else False,
                                            throwAway = True if i<config.start_T_index else False)
        if log_vis:
            T_hist.append(visB)
            T_ref_hist.append(visA)



    ###future frame prediction loop
    for i in range(config.sequence_length - config.sequence_seed):
        ###get LFT of current frame
        S_fft = compact_LFT(S_curr + eps, window, config)

        if S_curr.shape[1]==1:
            ET = T
        else:
            ET = T.unsqueeze(2).expand(T.shape[0],T.shape[1],S_curr.shape[1],T.shape[2],T.shape[3],T.shape[4]).reshape_as(S_fft)

        ###apply transform
        NS_fft = getPhaseAdd(S_fft, ET)
        ###reconstruction, also get OLA denominator
        NS_, WT = compact_iLFT(NS_fft, window, ET, config,channel=S_curr.shape[1])
        ###apply second conv model to reconstruction result
        if config.refine_output:
            NS_ = MRef_Out(NS_)
        ###clamp to [0,1]
        NS = clmp(NS_, False)
        ###collect new frame and window overlap
        PRes.append(NS)
        if log_vis:
            WTs.append(WT)
        ###prepare for next frame prediction loop
        ###update T
        if config.enable_autoregressive_prediction:
            T,Energy = getPhaseDiff(compact_LFT(getOneC(NS,dim=1)+ eps, window, config), S_fft,config.use_energy)
        else:
            Energy = None

        ###apply transform model
        T, visB, visA, angAfterNormTmp = M_transform(T,Energy, 
                                    first = False,
                                    throwAway =False)
        
        angAfterNorm = angAfterNorm + angAfterNormTmp
        if log_vis:
            T_hist.append(visB)
            T_ref_hist.append(visA)


        S_curr = NS

    PRes = listToTensor(PRes)

    if (torch.isnan(PRes).any()):
        return False,False
        print(torch.pow((dataSet[:, :, 2:] - PRes[:, :, 2:]).cpu(), 2).mean().item())
        print(torch.isnan(dataSet - PRes).any())
        raise Exception("NAN Exception")



    if log_vis:
        with torch.no_grad():
            WTs = listToTensor(WTs)
            vis_image_string = phase+" predictions"
            vis_anim_string = phase+' animations'
            header = phase+': '
            setting = {'oneD': (dimension == 1), 'revert': True, 'dpi': 2.4, 'show': False,"vmin":0,"vmax":1}
            L2loss = torch.pow((dataSet - PRes), 2) #L1loss[li:ui]
            L1loss = 0.5 * (PRes.clamp(0, 1) - dataSet.clamp(0, 1)) + 0.5
            pic = showSeq(False, -1, "Pred,GT,L2,L1,WT", [PRes[li:ui].detach().cpu(),
                          dataSet[li:ui].cpu(), L2loss[li:ui].detach().cpu(),L1loss[li:ui], WTs[li:ui].clamp(0, 1).detach().cpu()],
                          **setting)
            ####get linear shift encoded in phase diffs
            # dmsg('xx.shape', 'yy.shape', 'visX.shape', 'visY.shape')
            show_phase_diff(pd_list=T_hist, gt=1-PRes, config=config, title=header + "VF from Phase Diffs")

            
            show_phase_diff(pd_list=T_ref_hist, gt=1-PRes, config=config, title=header + "VF after M_Transform")

            # print('logging stuff')
            wandblog({vis_image_string: pic},commit=False)

            pt = PRes[li:ui].detach().cpu()
            gt = dataSet[li:ui].cpu()
            logMultiVideo('Prediction, GT, Diff', [pt, gt, 0.5 * (pt.clamp(0, 1) - gt.clamp(0, 1)) + 0.5],
                          vis_anim_string=vis_anim_string)


    

    dummyRet= torch.tensor(0., requires_grad=True)
    angAfterNorm = angAfterNorm / float(config.sequence_length - config.sequence_seed)

    return PRes, angAfterNorm,dummyRet,dummyRet,dummyRet,dummyRet
