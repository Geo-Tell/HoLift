
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
from transform import *

class DeepGlobe(Dataset):
    def __init__(self, bashArgs, dataList, mode='train', *args, **kwargs):
        super(DeepGlobe, self).__init__(*args, **kwargs)
        self.bashArgs = bashArgs
        self.dataList = dataList
        self.mode = mode
        self.input_channel = 3
        self.n_class = 6
        self.img_list = [line.strip().split() for line in open(dataList)]
        
        self.files = []
        if mode == 'test':
            self.mean=torch.from_numpy(np.array([0.24378921,0.34647351,0.38518094])).float()
            self.std=torch.from_numpy(np.array([0.10298847,0.12556953,0.16742616])).float()
            for item in self.img_list:
                image_path = item[0]
                name = os.path.splitext(os.path.basename(image_path))[0]
                self.files.append({
                    "img": image_path,
                    "name": name,
                })
        else:
            self.mean=torch.from_numpy(np.array([0.28036855,0.37817039,0.4080014])).float()
            self.std=torch.from_numpy(np.array([0.10472486,0.11470858,0.14728342])).float()
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(image_path))[0]
                self.files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                })
        
        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            ])
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
        img = Image.open(item['img'])
        img = self.to_tensor(img)
        image = transforms.functional.normalize(img, self.mean, self.std)
        if self.mode == 'test':
            return image, fn
        label = Image.open(item['label'])
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = torch.from_numpy(label)
        im_lb = dict(im = img, lb = label)
        im_lb = self.trans(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        return image, label, fn

    def __len__(self):
        return len(self.files)

    def input_transform(self, image):
        image = self.to_tensor(image)
        image = image.permute((1,2,0))
        image = (image - self.mean)
        image = (image / self.std)
        image = image.permute((2,0,1))
        return image
