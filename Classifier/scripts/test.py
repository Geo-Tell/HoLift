import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from utils.LoadData_DeepGlobe import DeepGlobe_Test
from utils.LoadData_DFC2020 import DFC2020_Test
from models import vgg
from models import resnet50_cam as resnet50
from tqdm import tqdm
from PIL import Image

def get_arguments():
    parser = argparse.ArgumentParser(description='HoLift')
    parser.add_argument("--Dataset", type=str, default='deepglobe', 
                        choices=['deepglobe','dfc2020'])
    parser.add_argument("--ImgDir", type=str, default='', 
                        help='Directory of training images')
    parser.add_argument("--LabelDir", type=str, default='', 
                        help='Directory of annotations')
    parser.add_argument("--ValList", type=str, default='',
                        help='File path of list for validation')
    parser.add_argument("--RestoreFrom", type=str, default='',
                        help='Path to trained model')
    parser.add_argument("--SaveDir", type=str, default='',
                        help='Path to save results')
    parser.add_argument("--InputSize", type=int, default=256,
                        help='Size of input image')
    parser.add_argument("--ModelName", type=str, default='vgg',
                        choices=['vgg','resnet50'])
    parser.add_argument("--BatchSize", type=int, default=1)
    parser.add_argument("--NumWorkers", type=int, default=0)
    parser.add_argument("--FgThre", type=float, default=0.5,
                        help='Threshold for foreground')
    parser.add_argument("--ConflictThre", type=float, default=0.99,
                        help='Threshold for multi-class conflict')
    parser.add_argument("--TestAug", action='store_true', 
                        help='Data augmentation for testing')
    return parser.parse_args()

def get_model(args):
    if args.dataset == 'deepglobe':
        input_channel = 3
        num_classes = 6
    elif args.Dataset == 'dfc2020':
        input_channel = 12
        num_classes = 8
    if args.ModelName == 'vgg':
        model = vgg.VGG_CAM(input_channel=input_channel, num_classes=num_classes)
    elif args.ModelName == 'resnet50':
        model = resnet50.Net(num_classes)
        
    model = torch.nn.DataParallel(model).cuda()

    pretrained_dict = torch.load(args.RestoreFrom)['state_dict']
    model_dict = model.state_dict()
    
    print(model_dict.keys())
    print(pretrained_dict.keys())
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    print("Weights cannot be loaded:")
    print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def validate(args):
    print('\nvalidating ... ', flush=True, end='')
    model = get_model(args)
    model.eval()
    
    if args.Dataset == 'deepglobe':
        val_loader = DeepGlobe_Test(args)
        #       Urban land, Agriculture, Rangeland, Forest,    Water,   Barren land, Background
        cmap = [0,255,255,  255,255,0,  255,0,255,  0,255,0,  0,0,255,  255,255,255,  0,0,0]
    elif args.Dataset == 'dfc2020':
        val_loader = DFC2020_Test(args)
        #       forest,   shrubland,   grassland,   wetlands,   croplands,   urban,        barren,      water,      background
        cmap = [0,153,0,  198,176,68,  182,255,5,  39,255,135,  194,79,68,  165,165,165,  249,255,164,  28,13,255,  0,0,0]
    n_cls = len(cmap)//3-1
    cls_count = np.zeros((n_cls, ))

    cam_save_dir = os.path.join(args.SaveDir, 'cam')
    pseudo_label_save_dir = os.path.join(args.SaveDir, 'pseudo_label')
    pseudo_label_rgb_save_dir = os.path.join(args.SaveDir, 'pseudo_label_rgb')
    pseudo_label_list = os.path.join(args.SaveDir, 'pseudo_label.lst')
    cls_weight_file = os.path.join(args.SaveDir, 'cls_weight.txt')
    if not os.path.exists(cam_save_dir):
        os.makedirs(cam_save_dir)
    if not os.path.exists(pseudo_label_save_dir):
        os.makedirs(pseudo_label_save_dir)
    if not os.path.exists(pseudo_label_rgb_save_dir):
        os.makedirs(pseudo_label_rgb_save_dir)
    save_lst = open(pseudo_label_list, 'w')

    with torch.no_grad():
        for idx, dat in tqdm(enumerate(val_loader)):
            img_name, imgs, label= dat
            outputs = []
            size = imgs[0].shape[-2:]
            for img in imgs:
                _, cams, _ = model(img[0].cuda(non_blocking=True))
                if args.TestAug and cams.shape[0] == 4:
                    cams = cams[0]+cams[1].flip(-1)+cams[2].flip(-2)+cams[3].flip(-1).flip(-2)
                    cams = cams.unsqueeze(0)
                outputs.append(cams)
            highres_cam = [F.interpolate(o, size,
                            mode='bicubic', align_corners=False) for o in outputs]
            highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[0]
            
            highres_cam = F.relu(highres_cam)
            valid_class = torch.nonzero(label[0])[:, 0]
            num_class = highres_cam.shape[0]
            valid_class_onehot = torch.zeros(num_class)
            valid_class_onehot[valid_class] = 1
            highres_cam = highres_cam.cpu() * (valid_class_onehot.view(-1, 1, 1))
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
            
            # save class-wise cams
            highres_cam_vis = highres_cam * 255
            highres_cam_vis = highres_cam_vis.cpu().numpy()
            for cls in valid_class:
                cam_cls = highres_cam_vis[cls]
                if args.Dataset == 'dfc2020':
                    img_name_split = img_name[0].split('/')
                    # img_name_split[-2] = img_name_split[-2].replace('s2','lc')
                    cam_save_dir_ = os.path.join(cam_save_dir, img_name_split[-3], img_name_split[-2])
                    if not os.path.exists(cam_save_dir_):
                        os.makedirs(cam_save_dir_)
                else:
                    cam_save_dir_ = cam_save_dir
                cam_out_name = os.path.join(cam_save_dir_, os.path.basename(img_name[0])[:-4] + '_{}.png'.format(cls))
                cv2.imwrite(cam_out_name, cam_cls)
            
            # save pseudo labels
            # exclude pixels with multi-class conflict
            bg_mask = torch.sum((highres_cam > args.ConflictThre).float(), 0)
            
            highres_cam_bg = torch.cat([highres_cam, torch.ones_like(highres_cam[:1])*args.FgThre], 0)
            pseudo_labels = torch.argmax(highres_cam_bg, dim=0)
            pseudo_labels[bg_mask > 1] = 255
            pseudo_labels[pseudo_labels == num_class] = 255

            # skip if valid pixels are too few
            valid_pixel = torch.sum(pseudo_labels != 255).float()
            valid_ratio = valid_pixel / highres_cam_vis.shape[-2] / highres_cam_vis.shape[-1]
            if valid_ratio < 0.01:
                print('invalid pseudo label', img_name[0])
                continue
            
            # save pseudo labels
            if args.Dataset == 'dfc2020':
                img_name_split = img_name[0].split('/')
                # img_name_split[-2] = img_name_split[-2].replace('s2','lc')
                pseudo_label_save_dir_ = os.path.join(pseudo_label_save_dir, img_name_split[-3], img_name_split[-2])
                pseudo_label_rgb_save_dir_ = os.path.join(pseudo_label_rgb_save_dir, img_name_split[-3], img_name_split[-2])
                if not os.path.exists(pseudo_label_save_dir_):
                    os.makedirs(pseudo_label_save_dir_)
                if not os.path.exists(pseudo_label_rgb_save_dir_):
                    os.makedirs(pseudo_label_rgb_save_dir_)
            else:
                pseudo_label_save_dir_ = pseudo_label_save_dir
                pseudo_label_rgb_save_dir_ = pseudo_label_rgb_save_dir
            pseudo_out_name = os.path.join(pseudo_label_save_dir_, os.path.splitext(os.path.basename(img_name[0]))[0] + '_class.png')
            pseudo_rgb_out_name = os.path.join(pseudo_label_rgb_save_dir_, os.path.splitext(os.path.basename(img_name[0]))[0] + '_class.png')
            pseudo_labels_np = pseudo_labels.cpu().numpy()
            pseudo_labels_rgb = pseudo_labels_np.copy()
            pseudo_labels_rgb[pseudo_labels_rgb == 255] = num_class
            pseudo_labels_rgb = Image.fromarray(pseudo_labels_np.astype(np.uint8), mode='P')
            pseudo_labels_rgb.putpalette(cmap)
            cv2.imwrite(pseudo_out_name, pseudo_labels.cpu().numpy())
            pseudo_labels_rgb.save(pseudo_rgb_out_name)

            save_lst.write(os.path.abspath(img_name[0])+'\t'+os.path.abspath(pseudo_out_name)+'\n')

            count_keep = pseudo_labels_np!=255
            cls_count += np.bincount(pseudo_labels_np[count_keep], minlength=n_cls)
    save_lst.close()
    weight = 1 - cls_count / cls_count.sum()
    np.savetxt(cls_weight_file, weight, fmt='%0.7f')
            
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = get_arguments()
    print(args)
    validate(args)
