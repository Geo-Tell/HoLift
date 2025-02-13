import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import models.deeplabv3plus as Dplbv3p
from deepglobe import DeepGlobe
from dfc2020 import DFC2020
import os
import sys
import logging
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
np.set_printoptions(threshold=np.inf)

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--Dataset", type=str, default='deepglobe', 
                        choices=['deepglobe','dfc2020'])
    parse.add_argument('--Backbone',default='Resnet50', 
                        choices=['Resnet101','Resnet50','Resnet18'])
    parse.add_argument('--LargeASPP',action='store_true')
    parse.add_argument('--Validset',default='test.lst')
    parse.add_argument('--ModelPath',default='./res/')
    parse.add_argument('--ModelFile',default='final_state.pth')
    parse.add_argument('--PngPath',default='vis')
    parse.add_argument('--Multiscale',action='store_true')
    parse.add_argument('--SavePng',action='store_true')
    return parse.parse_args()

class MscEval(object):
    def __init__(self, bashArgs):
        self.args = bashArgs
        if self.args.Dataset == 'deepglobe':
            self.val_ds = DeepGlobe(bashArgs, bashArgs.Validset)
            self.n_cls = 6
            self.labels_dict = [{'name':'urban land','color':(0,255,255)},
                                {'name':'agriculture land','color':(255,255,0)},
                                {'name':'rangeland','color':(255,0,255)},
                                {'name':'forest land','color':(0,255,0)},
                                {'name':'water','color':(0,0,255)},
                                {'name':'barren land','color':(255,255,255)},
                                {'name':'others','color':(0,0,0)}]
        else:
            self.val_ds = DFC2020(bashArgs, bashArgs.Validset)
            self.n_cls = 8
            self.labels_dict = [{'name':'forest','color':(0,153,0)},
                                {'name':'shrubland land','color':(198,176,68)},
                                {'name':'grassland','color':(182,255,5)},
                                {'name':'wetlands','color':(39,255,135)},
                                {'name':'croplands','color':(194,79,68)},
                                {'name':'urban','color':(165,165,165)},
                                {'name':'barren','color':(249,255,164)},
                                {'name':'water','color':(28,13,255)},
                                {'name':'others','color':(0,0,0)}]
        self.input_channel = self.val_ds.input_channel
        self.val_dl = DataLoader(self.val_ds,
                                batch_size = 1,
                                shuffle = False,
                                num_workers = 2,
                                pin_memory = True,
                                drop_last = False)


    def __call__(self, net):
        ## evaluate
        if self.args.Multiscale:
            eval_scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
        else:
            eval_scales = (1,)
        net.eval()
        n_classes = self.n_cls
        ignore_label = 255
        hist = torch.zeros(n_classes, n_classes).cuda()
        ious = np.array([])
        with torch.no_grad():
            for _, batch in enumerate(tqdm(self.val_dl), 0):
                images, labels, fn = batch
                labels = labels.squeeze(1).cuda()
                N, H, W = labels.shape
                images = images.numpy()[0].transpose((1,2,0)).copy()
                probs = torch.zeros((N, n_classes, H, W)).cuda()
                probs.requires_grad = False
                for scale in eval_scales:
                    w, h = int(W * scale + 0.5), int(H * scale + 0.5)
                    new_img = cv2.resize(images,(w, h))
                    new_img = new_img.transpose((2, 0, 1))
                    new_img = np.expand_dims(new_img, axis=0)
                    new_img = torch.from_numpy(new_img)
                    logits = net(new_img)
                    logits = F.interpolate(logits,(H,W),mode='bilinear',align_corners=True)
                    prob = F.softmax(logits.cuda(), 1)
                    probs += prob
                preds = torch.argmax(probs, dim=1)
                if self.args.SavePng:
                    save_path = os.path.join(self.args.ModelPath, self.args.PngPath)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    colorPreds = preds.clone()
                    colorPreds = colorPreds.squeeze(0).unsqueeze(-1).repeat(1,1,3).cpu().numpy().astype('uint8')
                    
                    for i in range(len(self.labels_dict)):
                        colorPreds[(preds.squeeze(0).cpu().numpy()==i)]=self.labels_dict[i]['color']
                    save_color_img = Image.fromarray(colorPreds)
                    save_color_img.save(os.path.join(save_path, fn[0]+"_pred.png"))

                    colorLabel = labels.squeeze(0).unsqueeze(-1).repeat(1,1,3).cpu().numpy().astype('uint8')
                    for i in range(len(self.labels_dict)):
                        colorLabel[(labels.squeeze(0).cpu().numpy()==i)]=self.labels_dict[i]['color']
                    save_color_label = Image.fromarray(colorLabel)
                    save_color_label.save(os.path.join(save_path, fn[0]+"_label.png"))
                keep = labels != ignore_label
                hist += torch.bincount(
                    labels[keep] * n_classes + preds[keep],
                    minlength=n_classes ** 2
                    ).to(torch.float32).view(n_classes, n_classes)
        AA = (hist.diag() / hist.sum(dim=1)).mean()
        AA = AA.cpu().numpy()
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        ious = ious.cpu().numpy()
        miou = np.nanmean(ious)
        '''
        TPs = hist.diag()
        FNs = hist.sum(dim=1)-hist.diag()
        FPs = hist.sum(dim=0)-hist.diag()
        TNs = hist.sum()-hist.sum(dim=0)-hist.sum(dim=1)+TPs
        precision = TPs/(TPs+FPs)
        recall = TPs/(TPs+FNs)
        F1 = 2*precision*recall/(precision+recall)
        accuracy = (TPs+TNs)/hist.sum()
        OA = hist.diag().sum()/hist.sum()
        AA = accuracy.mean()
        p0 = hist.diag().sum()/hist.sum()
        pe = (hist.sum(dim=0)*hist.sum(dim=1)).sum()
        kappa = (p0-pe)/(1-pe)
        '''
        return miou.item(), AA.item(), ious, hist

def evaluate():
    ## setup
    args = parse_args()
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger()
    evaluator = MscEval(args)

    ## model
    logger.info('setup and restore model')
    net = Dplbv3p.Deeplab_v3plus(args, in_channel=evaluator.input_channel, out_channel=evaluator.n_cls, large_ASPP=args.LargeASPP)
    net_model = torch.load(os.path.join(args.ModelPath, args.ModelFile))
    for k,v in list(net_model.items()):
        if 'module' in k:
            net_model[k[7:]] = net_model.pop(k)
    net.load_state_dict(net_model)
    net.cuda()
    net.eval()
    net = nn.DataParallel(net)

    ## evaluator
    logger.info('compute the mIOU')
    mIOU, AA, IoU_array, hist = evaluator(net)
    logger.info('mIOU is: {:.6f}'.format(mIOU))
    logger.info('AA is: {:.6f}'.format(AA))
    f = open(os.path.join(args.ModelPath,'acc.txt'),'w')
    f.write('mIoU='+str(mIOU)+'\nAA='+str(AA)+'\nIoU='+str(IoU_array))
    f.close()
    logger.info(IoU_array)
    logger.info(hist)

if __name__ == "__main__":
    evaluate()
