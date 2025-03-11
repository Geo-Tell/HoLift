from torchvision import transforms as transforms_pytorch
from .transforms import transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
from .imutils import RandomResizeLong, RandomCrop
import os
import torch
from PIL import Image
import random

def DeepGlobe_Train(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    
    input_size = int(args.InputSize)
    if args.TrainAug:
        tsfm_train = transforms.Compose([
                        transforms.Resize(input_size),  
                        RandomResizeLong(160, 320),
                        transforms_pytorch.RandomHorizontalFlip(0.5),
                        transforms_pytorch.RandomVerticalFlip(0.5),
                        transforms.ToTensor(),
                        RandomCrop(256),
                        transforms.Normalize(mean_vals, std_vals),
                        ])
    else:
        tsfm_train = transforms.Compose([
                        transforms.Resize(input_size),  
                        transforms.ToTensor(),
                        transforms.Normalize(mean_vals, std_vals),
                        ])
        
    tsfm_test = transforms.Compose([
                    transforms.Resize(input_size),  
                    transforms.ToTensor(),
                    transforms.Normalize(mean_vals, std_vals),
                ])
    tsfm_more = [transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomResize([int(0.75*input_size), int(input_size), 
                                            int(1.25*input_size), int(1.5*input_size), 
                                            int(2*input_size)]),  
                    transforms_pytorch.RandomHorizontalFlip(0.5),
                    transforms_pytorch.RandomVerticalFlip(0.5),
                    transforms.RandomRotation([0,90]),
                    transforms.Normalize(mean_vals, std_vals),
                ])]

    img_train = DeepGlobe_multi_reference(args.TrainList, root_dir=args.TrainImgDir, label_list=args.TrainLabelDir, 
                                          reference_num=args.ReferenceNum, intra_image=args.IntraImg,
                                          transform=tsfm_train, more_transform=tsfm_more)
    img_test = DeepGlobe_single(args.ValList, root_dir=args.ValImgDir, label_list=args.ValLabelDir, 
                                transform=tsfm_test)

    return img_train, img_test

def DeepGlobe_Test(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
       
    input_size = int(args.InputSize)

    tsfm_test = transforms.Compose([transforms.Resize(input_size),  
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean_vals, std_vals),
                                    ])
    img_test = DeepGlobe_single(args.ValList, root_dir=args.ImgDir, 
                                label_list=args.LabelDir, transform=tsfm_test, 
                                test_aug=args.TestAug)
    val_loader = DataLoader(img_test, batch_size=args.BatchSize, shuffle=False, 
                            num_workers=args.NumWorkers)

    return val_loader

def read_labeled_image_list(data_dir, data_list, label_list):
    # return 1: image path, return 2: one-hot image label
    with open(data_list, 'r') as f:
        lines = f.readlines()
    img_name_list = []
    img_labels = []
    labels = np.load(label_list, allow_pickle=True).item()
    for line in lines:
        image = line.strip()
        img_name_list.append(os.path.join(data_dir, image))
        img_labels.append(labels[image[:-4]][:-1])
    return img_name_list, img_labels

class DeepGlobe_multi_reference(Dataset):
    def __init__(self, datalist_file, root_dir, label_list, reference_num=4, intra_image=True, transform=None, more_transform=None):
        self.root_dir = root_dir
        self.label_list = label_list
        self.datalist_file =  datalist_file
        self.transform = transform
        self.more_transform = more_transform
        self.input_channel = 3

        self.image_list, self.label_list = read_labeled_image_list(self.root_dir, self.datalist_file, self.label_list)
        self.label_list=np.array(self.label_list, dtype='float32')
        same_class_index=[]
        single_class_index = []
        multi_class_index = []
        for i in range(self.label_list.shape[1]):
            same_class_index.append(np.where(self.label_list[:,i]==1)[0])   # ids of imgs that share the same label
        single_class_index = np.where(np.sum(self.label_list, 1) == 1)[0]
        single_class_index = set(single_class_index)
        for i in range(self.label_list.shape[1]):
            multi_class_index.append(list(set(same_class_index[i]) - single_class_index))
        self.same_class_index = same_class_index
        self.multi_class_index = multi_class_index
        self.sample_count = reference_num-int(intra_image)
        self.intra_image = intra_image


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img1_name =  self.image_list[idx]
        label1=self.label_list[idx]
        image1 = Image.open(img1_name).convert('RGB')
        if self.transform is not None:
            image1_torch = self.transform(image1)
        img2_names = []
        label2s = []
        image2s = []
        for i in range(self.sample_count):
            posi_index=random.choice(np.where(label1==1)[0])
            # sample reference images (prevent only 1 class exist in both input and reference images)
            if np.sum(label1) != 1:
                idx2=random.choice(self.same_class_index[posi_index])
            else:
                idx2=random.choice(self.multi_class_index[posi_index])

            img2_name = self.image_list[idx2]
            label2 = self.label_list[idx2]
            image2 = Image.open(img2_name).convert('RGB')
            if self.transform is not None:
                image2 = self.transform(image2)
            img2_names.append(img2_name)
            label2s.append(label2)
            image2s.append(image2)

        if self.more_transform is not None:
            for i in self.more_transform:
                img2_names.append(img1_name)
                label2s.append(label1)
                image2s.append(i(image1))

        return img1_name,img2_names,image1_torch,image2s,label1,label2s


class DeepGlobe_single(Dataset):
    def __init__(self, datalist_file, root_dir, label_list, transform=None, test_aug=False):
        self.root_dir = root_dir
        self.label_list = label_list
        self.datalist_file =  datalist_file
        self.transform = transform
        self.multi_aug = test_aug
        self.input_channel = 3

        self.image_list, self.label_list = read_labeled_image_list(self.root_dir, self.datalist_file, self.label_list)
        self.label_list=np.array(self.label_list, dtype='float32')

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name =  self.image_list[idx]
        label=self.label_list[idx]
        image = Image.open(img_name).convert('RGB')
        if self.transform is not None:
            image_torch = self.transform(image)

        ms_img_list = []
        if self.multi_aug:
            for scale in (1.0, 0.5, 1.5, 2.0):
                if scale == 1:
                    s_img = image_torch
                else:
                    s_img = transforms_pytorch.functional.resize(image_torch, [int(scale*image_torch.shape[-2]),int(scale*image_torch.shape[-1])])
                ms_img_list.append(torch.stack([s_img, 
                                    transforms_pytorch.functional.hflip(s_img),
                                    transforms_pytorch.functional.vflip(s_img),
                                    transforms_pytorch.functional.vflip(transforms_pytorch.functional.hflip(s_img))],
                                    dim=0))
        else:
            ms_img_list.append(image_torch.unsqueeze(0))
        
        return img_name,ms_img_list,label
