import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import rasterio
from transform import *


class DFC2020(Dataset):
    def __init__(self, bashArgs, dataList, mode='train', 
                use_hr=True, use_mr=True, use_lr=False, use_rgb=True, 
                use_sar=True, *args, **kwargs):
        super(DFC2020, self).__init__(*args, **kwargs)
        self.bashArgs = bashArgs
        self.dataList = dataList
        self.mode = mode
        self.n_class = 8
        self.use_sar = use_sar

        S2_BANDS_RGB = [2, 3, 4]
        S2_BANDS_HR = [2, 3, 4, 8]
        S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
        S2_BANDS_LR = [1, 10, 11]
        bands_selected = []
        if use_hr:
            bands_selected = bands_selected + S2_BANDS_HR
        if use_mr:
            bands_selected = bands_selected + S2_BANDS_MR
        if use_lr:
            bands_selected = bands_selected + S2_BANDS_LR
        if use_rgb:
            bands_selected = bands_selected + S2_BANDS_RGB
        self.bands_selected = sorted(list(set(bands_selected)))
        bands_selected_np = np.array(self.bands_selected)
        self.input_channel = len(self.bands_selected)
        
        mean_s2 = np.array([0.15238575, 0.13202671, 0.12669834, 0.13706985, 
                    0.154139, 0.19794664, 0.22118443, 0.21426206, 0.23699412, 
                    0.08163001, 0.00252292, 0.22200217, 0.16355222])
        std_s2 = np.array([0.07878236, 0.0803859,  0.08106566, 0.10718387, 
                    0.10785178, 0.11202664, 0.12124208, 0.11836369, 0.12721946, 
                    0.06226578, 0.0037218,  0.13491758, 0.11694675])
        mean_s1 = np.array([0.48575834, 0.21378007])
        std_s1 = np.array([0.19927819, 0.18168045])
        
        mean_all = mean_s2[bands_selected_np-1]
        std_all = std_s2[bands_selected_np-1]
        if use_sar:
            mean_all = np.concatenate((mean_s1, mean_all))
            std_all = np.concatenate((std_s1, std_all))
            self.input_channel += 2
        self.mean=torch.from_numpy(mean_all).float()
        self.std=torch.from_numpy(std_all).float()

        self.img_list = [line.strip().split() for line in open(dataList)]
        self.files = []
        if mode == 'test':
            for item in self.img_list:
                image_s2_path = item[0]
                name = os.path.basename(image_s2_path)
                image_name_split = name.split('_')
                image_name_split[2] = 's1'
                image_path_split = image_s2_path.split('/')
                image_s1_path = os.path.join('/'.join(image_path_split[:-3]), '_'.join(image_name_split[:2]),'_'.join(image_name_split[2:4]),'_'.join(image_name_split))
                self.files.append({
                    "img1": image_s1_path,
                    "img2": image_s2_path,
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_s2_path, label_path = item
                name = os.path.basename(image_s2_path)
                image_name_split = name.split('_')
                image_name_split[2] = 's1'
                image_path_split = image_s2_path.split('/')
                image_s1_path = os.path.join('/'.join(image_path_split[:-3]), '_'.join(image_name_split[:2]),'_'.join(image_name_split[2:4]),'_'.join(image_name_split))
                name = os.path.splitext(name)[0]
                self.files.append({
                    "img1": image_s1_path,
                    "img2": image_s2_path,
                    "label": label_path,
                    "name": name,
                })
        
        transList = []
        if(hasattr(self.bashArgs, 'HFTrans') and self.bashArgs.HFTrans):
            transList.append(HorizontalFlip())
        if(hasattr(self.bashArgs, 'VFTrans') and self.bashArgs.VFTrans):
            transList.append(VerticalFlip())
        if(hasattr(self.bashArgs, 'RRTrans') and self.bashArgs.RRTrans):
            transList.append(RandomRotate((0,90)))
        if(hasattr(self.bashArgs, 'RSTrans') and self.bashArgs.RSTrans):
            transList.append(RandomScale((0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)))
        if(hasattr(self.bashArgs, 'RCTrans') and self.bashArgs.RCTrans):
            transList.append(RandomCrop((256, 256)))
        self.trans = Compose(transList)

    def __getitem__(self, idx):
        item = self.files[idx]
        fn = item['name']
        with rasterio.open(item['img2']) as data2:
            image = data2.read(self.bands_selected)
        image = image.astype(np.float32)
        image = np.clip(image, 0, 10000)
        image /= 10000
        image = image.astype(np.float32)
        
        if self.use_sar:
            with rasterio.open(item['img1']) as data1:
                image_s1 = data1.read(list(range(1,3)))
            image_s1 = image_s1.astype(np.float32)
            image_s1 = np.nan_to_num(image_s1)
            image_s1 = np.clip(image_s1, -25, 0)
            image_s1 /= 25
            image_s1 += 1
            image_s1 = image_s1.astype(np.float32)
            image = np.concatenate((image_s1,image),0)
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, self.mean, self.std)

        if self.mode == 'test':
            return image, fn
        label = Image.open(item['label'])
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = torch.from_numpy(label)
        im_lb = dict(im = image, lb = label)
        im_lb = self.trans(im_lb)
        image, label = im_lb['im'], im_lb['lb']
        return image, label, fn

    def __len__(self):
        return len(self.files)
