import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import os
import os.path as osp
import logging
import time
import argparse
import numpy as np
from logger import *
from deepglobe import DeepGlobe
from dfc2020 import DFC2020
from loss import OhemCELoss
import models.deeplabv3plus as Dplbv3p
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True

def parse_args():
    parse = argparse.ArgumentParser()
    # data
    parse.add_argument("--Dataset", type=str, default='deepglobe', 
                        choices=['deepglobe','dfc2020'])
    parse.add_argument('--Trainset',default='train.lst')
    parse.add_argument('--ClassWeight',default='',type=str)
    parse.add_argument('--ResPath',default='./res')
    # model
    parse.add_argument('--Backbone',default='Resnet50', 
                        choices=['Resnet101','Resnet50','Resnet18'])
    parse.add_argument('--LargeASPP',action='store_true')
    parse.add_argument('--Loss',default='ohem')
    parse.add_argument('--Resume',action='store_true')
    # transform
    parse.add_argument('--HFTrans',action='store_true',
                        help='random horizontal flip')
    parse.add_argument('--VFTrans',action='store_true',
                        help='random vertical flip')
    parse.add_argument('--RRTrans',action='store_true',
                        help='random rotation 90')
    parse.add_argument('--RSTrans',action='store_true',
                        help='random rescale')
    parse.add_argument('--RCTrans',action='store_true',
                        help='random crop')
    # hyperparameters
    parse.add_argument('--Epoch', type=int, default=30)
    parse.add_argument("--BatchSize", type=int, default=48)
    parse.add_argument("--NumWorkers", type=int, default=48)
    parse.add_argument('--LrStart',default=3e-4,type=float,
                        help='initial learning rate after warmup')
    parse.add_argument('--WUEpoch',default=3,type=int,
                        help='warmup epoch')
    parse.add_argument('--WUStartLr',default=3e-7,type=float,
                        help='learning rate of warmup')
    parse.add_argument("--DispInterval", type=int, default=100, 
                        help='Iteration number of displaying training log')
    parse.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = -1,
            )
    return parse.parse_args()

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, warmup_start_lr, warmup_steps=0, power=0.9):
    if(warmup_steps==0):
        lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
        optimizer.param_groups[0]['lr'] = lr
    else:
        warmup_factor = (base_lr / warmup_start_lr) ** (1. / warmup_steps)
        if cur_iters <= warmup_steps:
            lr = warmup_start_lr * (warmup_factor ** cur_iters)
        else:
            lr = base_lr * ((1 - (cur_iters - warmup_steps) / (max_iters - warmup_steps)) ** power)
        optimizer.param_groups[0]['lr'] = lr

def train(verbose=True, **kwargs):
    args = kwargs['args']
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    #logger
    if args.local_rank <= 0:
        if not osp.exists(args.ResPath): os.makedirs(args.ResPath)
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        logfile = 'Deeplabv3plus-{}.log'.format(time_str)
        logfile = osp.join(args.ResPath, logfile)
        setup_logger(logfile)
        logger = logging.getLogger()
        logger.info('logger setup')

    # dataset
    if args.Dataset == 'deepglobe':
        ds = DeepGlobe(args, args.Trainset)
        n_cls = 6
    else:
        ds = DFC2020(args, args.Trainset)
        n_cls = 8

    if os.path.isfile(args.ClassWeight):
        weight = np.loadtxt(args.ClassWeight)
        class_weights = torch.FloatTensor([weight])
        # class_weights = torch.FloatTensor([[0.89090317, 0.38865106, 0.9477132,  0.87018226, 0.97332638, 0.92922393]])
        if args.local_rank <= 0:
            logger.info('weight loaded')
            f_weight = open(osp.join(args.ResPath, 'weight.txt'), 'w')
            f_weight.write(str(class_weights))
            f_weight.close()
    else:
        class_weights = torch.ones([1, n_cls])
    
    if args.local_rank <= 0:
        logger.info('dataset done')

    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size = args.BatchSize,
                    shuffle = False,
                    sampler=sampler,
                    num_workers = args.NumWorkers,
                    pin_memory = True,
                    drop_last = True)
    if args.local_rank <= 0:
        logger.info('dataset ditributed')
    
    # model
    net = Dplbv3p.Deeplab_v3plus(args, in_channel=ds.input_channel, out_channel=n_cls, large_ASPP=args.LargeASPP)
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.cuda(args.local_rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
    if args.local_rank <= 0:
        logger.info('net initialized')

    # optimizer
    if hasattr(net, 'module'):
        wd_params, non_wd_params = net.module.get_params()
    else:
        wd_params, non_wd_params = net.get_params()
    params_list = [{'params': wd_params,},
                    {'params': non_wd_params, 'weight_decay': 5e-4}]
    optimizer = torch.optim.SGD(params_list,
                                lr=args.WUStartLr,
                                momentum=0.9,
                                weight_decay=5e-4,
                                )
    if args.local_rank <= 0:
        logger.info('optimizer done')

    epoch_iters = np.int(ds.__len__() / 
                        args.BatchSize / torch.cuda.device_count())
    last_epoch = 0
    if args.Resume:
        for root, dirs, files in os.walk(args.ResPath):
                dirs.sort(reverse = True)
                for tb_dirs in dirs:
                    if 'tensorboard' in tb_dirs:
                        tensorboard_log_dir = os.path.join(args.ResPath,tb_dirs)
                        break
        model_state_file = os.path.join(args.ResPath,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.local_rank <= 0:
                logger.info("=> loaded checkpoint (epoch {})"
                            .format(checkpoint['epoch']))
    else:
        if args.local_rank <= 0:
            tensorboard_log_dir = os.path.join(args.ResPath, 'tensorboard_'+time_str)
            if not osp.exists(tensorboard_log_dir): os.makedirs(tensorboard_log_dir)
    if args.local_rank <= 0:
        writer_dict = {
            'writer': SummaryWriter(tensorboard_log_dir),
            'train_global_steps': last_epoch,
            'valid_global_steps': last_epoch,
        }
    
    end_epoch = args.Epoch
    num_iters = args.Epoch * epoch_iters
    n_min = 200*200
    ohem_thresh = 0.7
    
    if args.Loss == 'ohem':
        criteria = OhemCELoss(thresh=ohem_thresh, n_min=n_min, ignore_lb=255,
                            class_weight=class_weights).cuda()
    else:
        criteria = torch.nn.CrossEntropyLoss(ignore_index=255,
                                            weight=class_weights).cuda()
    if args.local_rank <= 0:
        logger.info('train start')
    for epoch in range(last_epoch, end_epoch):
        net.train()
        sampler.set_epoch(epoch)
        loss_avg = []
        cur_iters = epoch*epoch_iters
        if args.local_rank <= 0:
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
        st = time.time()
        for i_iter, batch in enumerate(dl,0):
            images, labels, _ = batch
            images = images.cuda()
            labels = labels.cuda()
            labels = torch.squeeze(labels, 1)
            optimizer.zero_grad()
            logits = net(images)
            loss = criteria(logits, labels)
            loss.backward()
            optimizer.step()
            # poly learning rate policy with warmup
            adjust_learning_rate(optimizer,
                                args.LrStart,
                                num_iters,
                                i_iter+cur_iters,
                                args.WUStartLr,
                                args.WUEpoch*epoch_iters)
            loss_avg.append(loss.item())
            if i_iter % args.DispInterval == 0:
                ed = time.time()
                msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                    'lr: {:.6f}, Loss: {:.6f}' .format(
                        epoch, end_epoch, i_iter, epoch_iters, 
                        ed-st, optimizer.param_groups[0]['lr'], sum(loss_avg) / len(loss_avg))
                logging.info(msg)
        if args.local_rank <= 0:
            writer.add_scalar('train_loss', sum(loss_avg) / len(loss_avg), global_steps)
            writer.add_scalar('lr',optimizer.param_groups[0]['lr'],global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
        torch.distributed.barrier()
    if args.local_rank <= 0:
        torch.save(net.state_dict(),
                    os.path.join(args.ResPath, 'final_state.pth'))
        writer_dict['writer'].close()
        logger.info('Done')

if __name__ == "__main__":
    args = parse_args()
    train(args=args)
