import gdal, os
from PIL import Image
from tqdm import tqdm
import argparse
from glob import glob
import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument("--DirSEN12MS", type=str, default='') 
    parser.add_argument("--DirDFC2020", type=str, default='')
    parser.add_argument("--SavePath", type=str, default='')
    return parser.parse_args()

def ImageLabel(path, sub_folders, save_path, map_dict, label_symbol='lc'):
    dict_all = {}
    n_cls = 10
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    for sub_folder in sub_folders:
        for dir in glob(os.path.join(path, sub_folder, label_symbol+'_*/')):
            files = os.listdir(dir)
            for file in tqdm(files):
                if os.path.splitext(file)[-1] != '.tif':
                    continue
                data = gdal.Open(os.path.join(dir, file))
                label = data.ReadAsArray(0,0,data.RasterXSize,data.RasterYSize)
                if len(label.shape) == 3:
                    label = label[0]
                img_label = np.unique(label)
                if map_dict is not None:
                    for k, v in map_dict.items():
                        img_label[img_label==k] = v
                img_label_binary = np.bincount(img_label.flatten(), minlength=n_cls+1)
                img_label_binary[img_label_binary > 0] = 1
                dict_all[os.path.splitext(file)[0]] = img_label_binary
    np.save(save_path, dict_all)
                

def PixelLabel(path, sub_folders, save_path, map_dict, label_symbol='lc'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sub_folder in sub_folders:
        for dir in glob(os.path.join(path, sub_folder, label_symbol+'_*/')):
            files = os.listdir(dir)
            for file in tqdm(files):
                data = gdal.Open(os.path.join(dir, file))
                label = data.ReadAsArray(0,0,data.RasterXSize,data.RasterYSize)
                if len(label.shape) == 3:
                    label = label[0]
                label_copy = label.copy()
                for k, v in map_dict.items():
                    label_copy[label==k] = v
                save_dir = os.path.join(save_path, sub_folder, dir.split('/')[-2])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                lab_save = Image.fromarray(label_copy)
                lab_save.save(os.path.join(save_dir, os.path.splitext(os.path.basename(file))[0]+'.png'))

def NpyDataList(npy_file, save_path):
    data = np.load(npy_file, allow_pickle=True).item()
    with open(save_path, 'w') as f:
        for k, v in data.items():
            if v[3]+v[8] > 0 or k == 'ROIs1868_summer_lc_146_p202':
                continue
            img_name_split = k.split('_')
            img_name_split[2] = 's2'
            img_name_s2 = '_'.join(img_name_split)
            f.write(img_name_s2 + '.tif\n')

def ImgDataList(image_path, label_path, save_path):
    with open(save_path, 'w') as f:
        for label_dir, _, label_names in os.walk(label_path):
            for label_name in label_names:
                if label_name.endswith('.png'):
                    label_name_split = label_name[:-4].split('_')
                    label_name_split[2] = 's2'
                    image_name_s2 = '_'.join(label_name_split)
                    label_dir_split = os.path.relpath(label_dir, label_path).split('/')
                    label_dir_split[-1] = 's2_' + label_dir_split[-1].split('_')[1]
                    image_dir = image_path + '/'.join(label_dir_split)
                    f.write(os.path.join(image_dir, image_name_s2) +'.tif '+ os.path.join(label_dir, label_name) + '\n')

if __name__ == '__main__':
    args = get_arguments()
    train_SEN12MS_path = args.DirSEN12MS
    valtst_DFC2020_path = args.DirDFC2020
    save_path = args.SavePath
    train_subfolders = ['ROIs1158_spring','ROIs1868_summer','ROIs1970_fall','ROIs2017_winter']
    val_subfolders = ['ROIs0000_validation']
    test_subfolders = ['ROIs0000_test']
    map_SEN12MS = {1:1, 2:1, 3:1, 4:1, 5:1, 6:2, 7:2, 8:3, 9:3, 10:4 , 11:5, 12:6, 14:6, 13:7, 15:8, 16:9, 17:10}
    map_DFC2020 = {1:0, 2:1, 3:255, 4:2, 5:3, 6:4, 7:5, 8:255, 9:6, 10:7}   # skip Savanna(3) and Snow/Ice(8)
    ImageLabel(train_SEN12MS_path, train_subfolders, os.path.join(save_path, 'train_label.npy'), map_SEN12MS, 'lc')
    ImageLabel(valtst_DFC2020_path, val_subfolders, os.path.join(save_path, 'val_label.npy'), map_SEN12MS, 'lc')
    PixelLabel(valtst_DFC2020_path, test_subfolders, save_path, map_DFC2020, 'dfc')
    NpyDataList(os.path.join(save_path, 'train_label.npy'), os.path.join(save_path, 'train_list.txt'))
    NpyDataList(os.path.join(save_path, 'val_label.npy'), os.path.join(save_path, 'val_list.txt'))
    for sub_folder in test_subfolders:
        ImgDataList(os.path.join(valtst_DFC2020_path, sub_folder), os.path.join(save_path, sub_folder), os.path.join(save_path, 'test_list.txt'))