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
import rasterio

S2_BANDS_RGB = [2, 3, 4]
S2_BANDS_HR = [2, 3, 4, 8]
S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
S2_BANDS_LR = [1, 10, 11]
mean_s2 = np.array([0.15238575, 0.13202671, 0.12669834, 0.13706985, 
            0.154139, 0.19794664, 0.22118443, 0.21426206, 0.23699412, 
            0.08163001, 0.00252292, 0.22200217, 0.16355222])
std_s2 = np.array([0.07878236, 0.0803859,  0.08106566, 0.10718387, 
            0.10785178, 0.11202664, 0.12124208, 0.11836369, 0.12721946, 
            0.06226578, 0.0037218,  0.13491758, 0.11694675])
mean_s1 = np.array([0.48575834, 0.21378007])
std_s1 = np.array([0.19927819, 0.18168045])

def DFC2020_Train(args, use_hr=True, use_mr=True, use_lr=False, use_rgb=True, use_sar=True):
    bands_selected = []
    if use_hr:
        bands_selected = bands_selected + S2_BANDS_HR
    if use_mr:
        bands_selected = bands_selected + S2_BANDS_MR
    if use_lr:
        bands_selected = bands_selected + S2_BANDS_LR
    if use_rgb:
        bands_selected = bands_selected + S2_BANDS_RGB
    bands_selected = sorted(list(set(bands_selected)))
    bands_selected_np = np.array(bands_selected)
    mean_all = mean_s2[bands_selected_np-1]
    std_all = std_s2[bands_selected_np-1]
    if use_sar:
        mean_all = np.concatenate((mean_s1, mean_all))
        std_all = np.concatenate((std_s1, std_all))

    input_size = int(args.InputSize)
    if args.ModelName == 'vgg':
        tsfm_train = transforms.Compose([transforms.Normalize(mean_all, std_all),])
    tsfm_test = transforms.Compose([transforms.Normalize(mean_all, std_all),])
    tsfm_more = [transforms.Compose([
                    transforms.RandomResize([int(0.75*input_size), int(input_size), 
                                            int(1.25*input_size), int(1.5*input_size),
                                            int(2*input_size)]),  
                    transforms_pytorch.RandomHorizontalFlip(0.5),
                    transforms_pytorch.RandomVerticalFlip(0.5),
                    transforms.RandomRotation([0,90]),
                    transforms.Normalize(mean_all, std_all),
                ])]

    img_train = DFC2020_multi_reference(args.TrainList, root_dir=args.TrainImgDir, label_list=args.TrainLabelDir, bands_selected=bands_selected, use_sar=use_sar,
                                        reference_num=args.ReferenceNum, intra_image=args.IntraImg, transform=tsfm_train, more_transform=tsfm_more)
    img_test = DFC2020_single(args.ValList, root_dir=args.ValImgDir, label_list=args.ValLabelDir, bands_selected=bands_selected, use_sar=use_sar, transform=tsfm_test)

    return img_train, img_test

def DFC2020_Test(args, use_hr=True, use_mr=True, use_lr=False, use_rgb=True, use_sar=True):
    bands_selected = []
    if use_hr:
        bands_selected = bands_selected + S2_BANDS_HR
    if use_mr:
        bands_selected = bands_selected + S2_BANDS_MR
    if use_lr:
        bands_selected = bands_selected + S2_BANDS_LR
    if use_rgb:
        bands_selected = bands_selected + S2_BANDS_RGB
    bands_selected = sorted(list(set(bands_selected)))
    bands_selected_np = np.array(bands_selected)
    mean_all = mean_s2[bands_selected_np-1]
    std_all = std_s2[bands_selected_np-1]
    if use_sar:
        mean_all = np.concatenate((mean_s1, mean_all))
        std_all = np.concatenate((std_s1, std_all))

    tsfm_test = transforms.Compose([transforms.Normalize(mean_all, std_all),])
    img_test = DFC2020_single(args.ValList, root_dir=args.ImgDir, label_list=args.LabelDir, bands_selected=bands_selected, use_sar=use_sar, transform=tsfm_test)

    val_loader = DataLoader(img_test, batch_size=args.BatchSize, shuffle=False, num_workers=args.NumWorkers)
    return val_loader

def read_labeled_image_list(data_dir, data_list, label_list):
    # return 1: image path, return 2: one-hot image label
    with open(data_list, 'r') as f:
        lines = f.readlines()
    img_name_list = []
    img_labels = []
    labels = np.load(label_list, allow_pickle=True).item()
    for line in lines:
        image_pair = []
        image = line.strip()
        image_name_split = image.split('_')
        image_pair.append(os.path.join(data_dir, '_'.join(image_name_split[:2]),'_'.join(image_name_split[2:4]),image))
        image_name_split[2] = 's1'
        image_pair.append(os.path.join(data_dir, '_'.join(image_name_split[:2]),'_'.join(image_name_split[2:4]),'_'.join(image_name_split)))
        img_name_list.append(image_pair)
        image_name_split[2] = 'lc'
        label_name = '_'.join(image_name_split)
        lab = labels[label_name[:-4]]
        lab = np.concatenate((lab[1:3],lab[4:8],lab[9:])) # skip Savanna(3) and Snow/Ice(8)
        img_labels.append(lab)
    return img_name_list, img_labels

class DFC2020_multi_reference(Dataset):
    def __init__(self, datalist_file, root_dir, label_list, bands_selected, use_sar=True, reference_num=4, intra_image=True, transform=None, more_transform=None):
        self.root_dir = root_dir
        self.label_list = label_list
        self.datalist_file =  datalist_file
        self.bands_selected = bands_selected
        self.use_sar = use_sar
        self.input_channel = len(self.bands_selected)
        if self.use_sar:
            self.input_channel += 2
        self.transform = transform
        self.more_transform = more_transform

        self.image_list, self.label_list = read_labeled_image_list(self.root_dir, self.datalist_file, self.label_list)
        self.label_list=np.array(self.label_list,dtype='float32')
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
        # read Sentinel-2
        img1_name_s2 =  self.image_list[idx][0]
        with rasterio.open(img1_name_s2) as data2:
            image1 = data2.read(self.bands_selected)
        image1 = image1.astype(np.float32)
        image1 = np.clip(image1, 0, 10000)
        image1 /= 10000
        image1 = image1.astype(np.float32)
        # read Sentinel-1
        if self.use_sar:
            img1_name_s1 =  self.image_list[idx][1]
            with rasterio.open(img1_name_s1) as data1:
                image1_s1 = data1.read(list(range(1,3)))
            image1_s1 = image1_s1.astype(np.float32)
            image1_s1 = np.nan_to_num(image1_s1)
            image1_s1 = np.clip(image1_s1, -25, 0)
            image1_s1 /= 25
            image1_s1 += 1
            image1_s1 = image1_s1.astype(np.float32)

        image1 = np.concatenate((image1_s1,image1),0)
        image1 = torch.from_numpy(image1)
        if self.transform is not None:
            image1_torch = self.transform(image1)
        label1=self.label_list[idx]
        
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
            # read Sentinel-2
            img2_name_s2 =  self.image_list[idx2][0]
            label2=self.label_list[idx2]
            with rasterio.open(img2_name_s2) as data2:
                image2 = data2.read(self.bands_selected)
            image2 = image2.astype(np.float32)
            image2 = np.clip(image2, 0, 10000)
            image2 /= 10000
            image2 = image2.astype(np.float32)
            # read Sentinel-1
            if self.use_sar:
                img2_name_s1 =  self.image_list[idx2][1]
                with rasterio.open(img2_name_s1) as data1:
                    image2_s1 = data1.read(list(range(1,3)))
                image2_s1 = image2_s1.astype(np.float32)
                image2_s1 = np.nan_to_num(image2_s1)
                image2_s1 = np.clip(image2_s1, -25, 0)
                image2_s1 /= 25
                image2_s1 += 1
                image2_s1 = image2_s1.astype(np.float32)

            image2 = np.concatenate((image2_s1,image2),0)
            image2 = torch.from_numpy(image2)
            if self.transform is not None:
                image2 = self.transform(image2)
            img2_names.append(img2_name_s2)
            label2s.append(label2)
            image2s.append(image2)

        if self.more_transform is not None:
            for i in self.more_transform:
                img2_names.append(img1_name_s2)
                label2s.append(label1)
                image2s.append(i(image1))

        return img1_name_s2,img2_names,image1_torch,image2s,label1,label2s


class DFC2020_single(Dataset):
    def __init__(self, datalist_file, root_dir, label_list, bands_selected, use_sar=True, transform=None):
        self.root_dir = root_dir
        self.label_list = label_list
        self.datalist_file =  datalist_file
        self.transform = transform
        self.bands_selected = bands_selected
        self.use_sar = use_sar
        self.input_channel = len(self.bands_selected)
        if self.use_sar:
            self.input_channel += 2

        self.image_list, self.label_list = read_labeled_image_list(self.root_dir, self.datalist_file, self.label_list)
        self.label_list=np.array(self.label_list, dtype='float32')

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name_s2 =  self.image_list[idx][0]
        label=self.label_list[idx]
        with rasterio.open(img_name_s2) as data2:
            image = data2.read(self.bands_selected)
        image = image.astype(np.float32)
        image = np.clip(image, 0, 10000)
        image /= 10000
        image = image.astype(np.float32)
        if self.use_sar:
            img1_name_s1 =  self.image_list[idx][1]
            with rasterio.open(img1_name_s1) as data1:
                image_s1 = data1.read(list(range(1,3)))
            image_s1 = image_s1.astype(np.float32)
            image_s1 = np.nan_to_num(image_s1)
            image_s1 = np.clip(image_s1, -25, 0)
            image_s1 /= 25
            image_s1 += 1
            image_s1 = image_s1.astype(np.float32)

        image = np.concatenate((image_s1,image),0)
        image = torch.from_numpy(image)
        
        if self.transform is not None:
            image_torch = self.transform(image)
        return img_name_s2,[image_torch.unsqueeze(0)],label
