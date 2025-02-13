import numpy as np
import os
import math
from tqdm import tqdm
from PIL import Image
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument("--DirImage", type=str, default='') 
    parser.add_argument("--SavePath", type=str, default='')
    return parser.parse_args()

def List2Names(file_list):
    file_names = []
    lines = file_list.readlines()
    file_names = [line.strip().split()[0].split('/')[-1].split('_')[0] for line in lines]
    return file_names

def LabelTrans(label):
    label_trans = np.ones((label.shape[0],label.shape[1])).astype('uint8')*255
    r,g,b = label[:,:,0],label[:,:,1],label[:,:,2]
    b_0 = b<128
    b_255 = b>=128
    g_0 = g<128
    g_255 = g>=128
    r_0 = r<128
    r_255 = r>=128
    label_trans[(r_0*g_255*b_255)]=0
    label_trans[(r_255*g_255*b_0)]=1
    label_trans[(r_255*g_0*b_255)]=2
    label_trans[(r_0*g_255*b_0)]=3
    label_trans[(r_0*g_0*b_255)]=4
    label_trans[(r_255*g_255*b_255)]=5
    return label_trans

def Split(img, save_path, img_name, patch_size, interval, name_suffix=''):
    mkdirlambda =lambda x: os.makedirs(x) if not os.path.exists(x) else True
    mkdirlambda(save_path)
    row = img.shape[0]
    col = img.shape[1]
    i_total = math.floor(row / patch_size)
    j_total = math.floor(col / patch_size)
    for i in range(i_total):
        for j in range(j_total):
            img_temp = img[i * patch_size + i * interval:(i + 1) * patch_size + i * interval,
                           j * patch_size + j * interval:(j + 1) * patch_size + j * interval]
            save_name = os.path.splitext(img_name)[0].split('_')[0] + '_' + str(patch_size) +  "_" \
                        + str(i * patch_size + i * interval) + "_" + str(j * patch_size + j * interval) \
                        + name_suffix + '.png'
            Image.fromarray(img_temp).save(os.path.join(save_path, save_name))

def ImageLabel(path, file_list, save_path):
    dict_all = {}
    n_cls = 6
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    files = os.listdir(path)
    for file in tqdm(files):
        file_name = file.split('_')[0]
        if file_name in file_list:
            label = np.array(Image.open(os.path.join(path, file)))
            img_label = np.unique(label)
            img_label[img_label == 255] = n_cls
            img_label_binary = np.bincount(img_label.flatten(), minlength=n_cls+1)
            dict_all[os.path.splitext(file)[0][:-6]] = img_label_binary
    np.save(save_path, dict_all)

def NpyDataList(npy_file, save_path):
    data = np.load(npy_file, allow_pickle=True).item()
    with open(save_path, 'w') as f:
        for k, v in data.items():
            if sum(v[:6]) == 0:
                continue
            f.write(k + '.png\n')

def ImgDataList(image_path, label_path, file_list, save_path):
    files = os.listdir(image_path)
    with open(save_path, 'w') as f:
        for file in tqdm(files):
            file_name = file.split('_')[0]
            if file_name in file_list:
                f.write(os.path.join(image_path, file) + ' ' \
                + os.path.join(label_path, os.path.splitext(file)[0]) + '_class.png\n')


if __name__ == "__main__":
    args = get_arguments()
    train_path = args.DirImage
    save_path = args.SavePath
    train_list = List2Names(open('./train.txt', 'r'))
    val_list = List2Names(open('./val.txt', 'r'))
    test_list = List2Names(open('./test.txt', 'r'))
    files = os.listdir(train_path)
    for file in tqdm(files):
        if file.endswith('.jpg'):
            image = np.array(Image.open(os.path.join(train_path, file)))
            label = np.array(Image.open(os.path.join(train_path, file.split('_')[0]+'_mask.png')))
            label = LabelTrans(label)
            Split(image, os.path.join(save_path, 'img'), file, 256, 18)
            Split(label, os.path.join(save_path, 'lb'), file, 256, 18, '_class')
    ImageLabel(os.path.join(save_path, 'lb'), train_list, os.path.join(save_path, 'train_label.npy'))
    ImageLabel(os.path.join(save_path, 'lb'), val_list, os.path.join(save_path, 'val_label.npy'))
    NpyDataList(os.path.join(save_path, 'train_label.npy'), os.path.join(save_path, 'train_list.txt'))
    NpyDataList(os.path.join(save_path, 'val_label.npy'), os.path.join(save_path, 'val_list.txt'))
    ImgDataList(os.path.join(save_path, 'img'), os.path.join(save_path, 'lb'), test_list, os.path.join(save_path, 'test_list.txt'))