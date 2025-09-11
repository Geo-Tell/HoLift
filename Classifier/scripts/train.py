import sys
import os
sys.path.append(os.getcwd())

import torch
import argparse
import time
import shutil
import my_optim
import random
import numpy as np
import torch.optim as optim
from Classifier.models import vgg
from Classifier.models import resnet50_cam as resnet50
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from Classifier.utils import AverageMeter
from Classifier.utils import L_cross, L_holift, L_focal_balance
from Classifier.utils.LoadData_DeepGlobe import DeepGlobe_Train
from Classifier.utils.LoadData_DFC2020 import DFC2020_Train
from tqdm import trange, tqdm
import torch.distributed as dist
from tensorboardX import SummaryWriter


def get_arguments():
    parser = argparse.ArgumentParser(description='The Pytorch code of HoLift')
    # data
    parser.add_argument("--Dataset", type=str, default='deepglobe', 
                        choices=['deepglobe','dfc2020'])
    parser.add_argument("--TrainImgDir", type=str, default='', 
                        help='Directory of training images')
    parser.add_argument("--TrainLabelDir", type=str, default='', 
                        help='Directory of training annotations')
    parser.add_argument("--ValImgDir", type=str, default='', 
                        help='Directory of validation images')
    parser.add_argument("--ValLabelDir", type=str, default='', 
                        help='Directory of validation annotations')
    parser.add_argument("--TrainList", type=str, default='None', 
                        help='File path of list for training')
    parser.add_argument("--ValList", type=str, default='None', 
                        help='File path of list for validation')
    parser.add_argument("--TrainAug", action='store_true', 
                        help='Data augmentation for training')
    parser.add_argument("--TestAug", action='store_true', 
                        help='Data augmentation for testing')
    # model
    parser.add_argument("--ModelName", type=str, default='vgg')
    parser.add_argument("--ReferenceNum", type=int, default=4, 
                        help='Number of reference images')
    parser.add_argument("--CAMThre", type=float, default=0.7, 
                        help='Threshold of CAM for selecting pixels in L_cross')
    # loss
    parser.add_argument("--ClsLoss", type=str, default='ce', 
                        choices=['ce','focal'])
    parser.add_argument("--IntraImg", action='store_true', 
                        help='Use transformed input image as the last reference or not')
    parser.add_argument("--LambdaCross", type=float, default=4.0,
                        help='Weight of L_cross')
    parser.add_argument("--LambdaHolift", type=float, default=1.0,
                        help='Weight of L_holift')
    parser.add_argument("--WaitEpochHolift", type=int, default=2, 
                        help='Epoch that HoLift loss join')
    parser.add_argument("--WaitEpochCross", type=int, default=10, 
                        help='Epoch that Cross-image loss join')
    # hyperparameter
    parser.add_argument("--BatchSize", type=int, default=16)
    parser.add_argument("--InputSize", type=int, default=256, 
                        help='Size of input image')
    parser.add_argument("--Lr", type=float, default=1e-3, 
                        help='Learning rate')
    parser.add_argument("--WeightDecay", type=float, default=0.0005)
    parser.add_argument("--Epoch", type=int, default=30, help='training epochs')
    parser.add_argument("--NumWorkers", type=int, default=2)
    
    parser.add_argument("--DispInterval", type=int, default=100, 
                        help='Iteration number of displaying training log')
    parser.add_argument("--SnapshotDir", type=str, default='', 
                        help='Path of saving model')
    parser.add_argument("--Resume", action='store_true')
    parser.add_argument('--local_rank', dest='local_rank', type=int, default=-1)

    return parser.parse_args()

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.SnapshotDir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.SnapshotDir, 'model_best.pth.tar'))

def get_model(args):
    if args.Dataset == 'deepglobe':
        input_channel = args.InputChannel
        num_classes = 6
    elif args.Dataset == 'dfc2020':
        input_channel = args.InputChannel
        num_classes = 8
    if args.ModelName == 'vgg':
        model = vgg.VGG_CAM(pretrained=(input_channel==3), num_classes=num_classes, input_channel=input_channel)
        model = model.cuda(args.local_rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])#, find_unused_parameters=True)
        param_groups = model.module.get_parameter_groups()
        optimizer = optim.SGD([
            {'params': param_groups[0], 'lr': args.Lr},
            {'params': param_groups[1], 'lr': 2*args.Lr},
            {'params': param_groups[2], 'lr': 10*args.Lr},
            {'params': param_groups[3], 'lr': 20*args.Lr}], 
            momentum=0.9, weight_decay=args.WeightDecay, nesterov=True)
    elif args.ModelName == 'resnet50':
        model = resnet50.Net(num_classes)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.local_rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])#, find_unused_parameters=True)
        param_groups = model.module.trainable_parameters()
        optimizer = optim.SGD([
            {'params': param_groups[0], 'lr': args.Lr},
            {'params': param_groups[1], 'lr': 2*args.Lr}], momentum=0.9, weight_decay=args.WeightDecay, nesterov=True)
    return  model, optimizer


def validate(model, val_loader):
    losses_val = AverageMeter()
    model.eval()
    with torch.no_grad():
        for idx, dat in tqdm(enumerate(val_loader)):
            _, input1, label1 = dat
            label1 = label1.cuda(non_blocking=True)
            logits, _, _ = model(input1[0][:,0])
            loss_val = F.multilabel_soft_margin_loss(logits, label1)   
            losses_val.update(loss_val.data.item(), input1[0].size()[0])
    return losses_val

def train(args):
    # fix random seed
    seed = 2
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    batch_time = AverageMeter()
    cam_area = AverageMeter()   # area of cam higher than threshold
    stat_loss = AverageMeter()
    stat_loss_cls = AverageMeter()
    stat_loss_holift = AverageMeter()
    stat_loss_cross = AverageMeter()
    stat_loss_cross_multi = []
    for i in range(args.ReferenceNum):
        stat_loss_cross_multi.append(AverageMeter())
    losses_val = AverageMeter()
    
    total_epoch = args.Epoch
    # global_counter = 0
    current_epoch = 0
    if args.Dataset == 'deepglobe':
        img_train, img_test = DeepGlobe_Train(args)
    elif args.Dataset == 'dfc2020':
        img_train, img_test = DFC2020_Train(args)
    args.InputChannel = img_train.input_channel
    sampler = torch.utils.data.distributed.DistributedSampler(img_train)
    train_loader = DataLoader(img_train, batch_size=args.BatchSize, sampler=sampler, 
                              shuffle=False, num_workers=args.NumWorkers)
    val_loader = DataLoader(img_test, batch_size=args.BatchSize, shuffle=False, 
                            num_workers=args.NumWorkers)

    steps_per_epoch = len(train_loader)
    max_step = total_epoch * steps_per_epoch
    if args.local_rank <= 0:
        print('Max step:', max_step)
    
    model, optimizer = get_model(args)
    model.train()
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    if args.Resume:
        for root, dirs, files in os.walk(args.SnapshotDir):
                dirs.sort(reverse = True)
                for tb_dirs in dirs:
                    if 'tensorboard' in tb_dirs:
                        tensorboard_log_dir = os.path.join(args.SnapshotDir,tb_dirs)
                        break
        model_state_file = os.path.join(args.ResPath,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            current_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.local_rank <= 0:
                print("=> loaded checkpoint (epoch {})"
                            .format(current_epoch))
    else:
        if args.local_rank <= 0:
            tensorboard_log_dir = os.path.join(args.SnapshotDir, 'tensorboard_'+time_str)
            if not os.path.exists(tensorboard_log_dir): os.makedirs(tensorboard_log_dir)

    if args.local_rank <= 0:
        writer_dict = {
            'writer': SummaryWriter(tensorboard_log_dir),
            'train_global_steps': 0,
            'train_global_epoch': 0,
            'valid_global_epoch': 0,
        }

    end = time.time()

    best_loss = 1e5
    cam_thre = args.CAMThre
    wait_epoch_holift = args.WaitEpochHolift
    wait_epoch_cross = args.WaitEpochCross

    if args.ClsLoss == 'focal':
        l_cls = L_focal_balance().cuda()
    else:
        l_cls = nn.MultiLabelSoftMarginLoss().cuda()
    l_cross = L_cross(cam_thre).cuda()
    l_holift = L_holift().cuda()

    while current_epoch < total_epoch:
        sampler.set_epoch(current_epoch)
        if args.local_rank <= 0:
            writer = writer_dict['writer']
        model.train()
        cam_area.reset()
        stat_loss.reset()
        stat_loss_cls.reset()
        stat_loss_holift.reset()
        stat_loss_cross.reset()
        for i in range(args.ReferenceNum):
            stat_loss_cross_multi[i].reset()

        batch_time.reset()
        steps_per_epoch = len(train_loader)
        wait_step_holift = steps_per_epoch * wait_epoch_holift
        torch.autograd.set_detect_anomaly(True)
        for idx, dat in enumerate(train_loader):
            current_iter = current_epoch * steps_per_epoch + idx
            _, _, input1, input2s, label1, label2s= dat
            label1 = label1.cuda()
            logits, cam1, feat1  = model(input1)

            if len(logits.shape) == 1:
                logits = logits.reshape(label1.shape)
            loss_ce = l_cls(logits, label1.clone())

            losses_cross = []
            for i in range(len(input2s)):
                _, cam2, feat2 = model(input2s[i])
                losses_cross.append(l_cross(cam1, cam2, feat1, feat2, label1.clone(), label2s[i].clone().cuda()))
                if i == 0:
                    mask_area = l_cross.area()
            
            loss_cross = sum(losses_cross) / len(losses_cross)
            loss_holift = l_holift(cam1, label1)
            lambda_holift = 0 if current_epoch < wait_epoch_holift else args.LambdaHolift
            lambda_cross = 0 if current_epoch < wait_epoch_cross else args.LambdaCross
            loss = loss_ce + lambda_holift*loss_holift + lambda_cross*loss_cross 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            stat_loss.update(loss.data.item(), input1.size()[0]+input2s[0].size()[0])
            stat_loss_cls.update(loss_ce.data.item(), input1.size()[0]+input2s[0].size()[0])

            if current_epoch >= wait_epoch_holift:
                stat_loss_holift.update(loss_holift.data.item(), input1.size()[0]+input2s[0].size()[0])
            
            if current_epoch >= wait_epoch_cross:
                cam_area.update(mask_area.data.item(), input1.size()[0])
                if loss_cross > 0:
                    stat_loss_cross.update(loss_cross.data.item(), input1.size()[0]+input2s[0].size()[0])
                for i in range(len(input2s)):
                    if losses_cross[i] > 0:
                        stat_loss_cross_multi[i].update(losses_cross[i].data.item(), input1.size()[0]+input2s[i].size()[0])

            batch_time.update(time.time() - end)
            end = time.time()
            
            if current_epoch >= wait_epoch_cross:
                my_optim.reduce_lr_poly(args.Lr*2, optimizer, current_iter, max_step, wait_step_holift)
            else:
                my_optim.reduce_lr_poly(args.Lr, optimizer, current_iter, max_step, wait_step_holift)
            global_counter = current_epoch * steps_per_epoch + idx + 1

            if global_counter % args.DispInterval == 0:
                if args.local_rank <= 0:
                    print('Epoch: [{}][{}/{}]\t'
                        'LR: {:.5f}\t' 
                        'Loss {loss.val:.4f}={loss_ce.val:.4f}+{loss_holift.val:.4f}+{loss_cross.val:.4f} ({loss.avg:.4f})\t'.format(
                            current_epoch, (global_counter-1)%len(train_loader)+1, len(train_loader), 
                            optimizer.param_groups[0]['lr'], loss=stat_loss, loss_ce=stat_loss_cls, loss_holift=stat_loss_holift, loss_cross=stat_loss_cross))

            if global_counter % steps_per_epoch == 0:
                if args.local_rank <= 0:
                    writer.add_scalar('train_loss', stat_loss.avg, current_epoch)
                    writer.add_scalar('train_loss_ce', stat_loss_cls.avg, current_epoch)
                    if current_epoch >= wait_epoch_holift:
                        writer.add_scalar('train_loss_holift', stat_loss_holift.avg, current_epoch)
                    if current_epoch >= wait_epoch_cross:
                        writer.add_scalar('cam_area', cam_area.avg, current_epoch)
                        writer.add_scalar('train_loss_cross', stat_loss_cross.avg, current_epoch)
                        for i in range(len(input2s)):
                            writer.add_scalar('train_loss_cross_{}'.format(i), stat_loss_cross_multi[i].avg, current_epoch)
                    writer.add_scalar('lr', optimizer.param_groups[0]['lr'] ,current_epoch)
                    writer_dict['train_global_epoch'] = current_epoch
                    print('\nvalidating ... ', flush=True, end='')
                    save_checkpoint(args,
                                    {
                                        'epoch': current_epoch,
                                        'state_dict':model.state_dict(),
                                        'optimizer':optimizer.state_dict()
                                    }, is_best=False)
                torch.distributed.barrier()

                losses_val = validate(model, val_loader)
                if args.local_rank <= 0:
                    print('val_loss:', losses_val.avg)
                    writer.add_scalar('val_loss', losses_val.avg, current_epoch)
                    writer_dict['valid_global_epoch'] = current_epoch
                    if losses_val.avg < best_loss:
                        best_loss = losses_val.avg
                        save_checkpoint(args,
                                        {
                                            'epoch': current_epoch,
                                            'state_dict':model.state_dict(),
                                            'optimizer':optimizer.state_dict()
                                        }, is_best=True,
                                        filename='%s_epoch_%d.pth' %(args.Dataset, current_epoch))
            torch.distributed.barrier()
        if current_epoch == args.Epoch-1:
            if args.local_rank <= 0:
                save_checkpoint(args,
                                {
                                    'epoch': current_epoch,
                                    'state_dict':model.state_dict(),
                                    'optimizer':optimizer.state_dict()
                                }, is_best=False,
                                filename='%s_epoch_%d.pth' %(args.Dataset, current_epoch))
        current_epoch += 1

if __name__ == '__main__':
    os.environ['TEMP']='/SDB/tmp'
    args = get_arguments()
    if args.local_rank <= 0:
        print('Running parameters:\n', args)
        if not os.path.exists(args.SnapshotDir):
            os.makedirs(args.SnapshotDir)
    train(args)
