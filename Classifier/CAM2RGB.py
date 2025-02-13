import numpy as np
import os, cv2
from PIL import Image
from tqdm import tqdm
import zipfile
import argparse
from osgeo import gdal

def get_arguments():
    parser = argparse.ArgumentParser(description='HoLift')
    parser.add_argument("--Dataset", type=str, default='deepglobe', 
                        choices=['deepglobe','dfc2020'])
    parser.add_argument("--ImgDir", type=str, default='')
    parser.add_argument("--CAMDir", type=str, default='')
    parser.add_argument("--SaveDir", type=str, default='')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    for dir, _, names in os.walk(args.CAMDir):
        for name in tqdm(names):
            if name.endswith('.png'):
                CAM = np.array(Image.open(os.path.join(dir, name)))
                img_name_split = name.split('_')
                img_name = '_'.join(img_name_split[:-1])
                if args.Dataset == 'deepglobe':
                    img = np.array(Image.open(os.path.join(args.ImgDir, img_name+'.png')))
                else:
                    img_path = os.path.join(args.ImgDir, '_'.join(img_name_split[:2]), '_'.join(img_name_split[2:4]), img_name+'.tif')
                    data = gdal.Open(img_path)
                    img = data.ReadAsArray(0,0,data.RasterXSize,data.RasterYSize)
                    img = img[1:4,:,:].astype('int64')
                    img = img.transpose(1,2,0)
                    d2 = np.percentile(img, 2, axis=(0,1))
                    u98 = np.percentile(img, 98, axis=(0,1))
                    img = 255 / (u98 - d2) * (img - d2)
                    img = np.clip(img, 0, 255)
                    img = np.uint8(np.round(img[:,:,::-1]))
                heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                result = (heatmap * 0.3 + img * 0.5).astype(np.uint8)
                result_save = Image.fromarray(result)

                if args.Dataset == 'deepglobe':
                    save_path = os.path.join(args.SaveDir, name)
                else:
                    save_path = os.path.join(args.SaveDir, '_'.join(img_name_split[:2]), '_'.join(img_name_split[2:4]), name)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                result_save.save(save_path)

    with zipfile.ZipFile((args.SaveDir+'.zip'),'w',zipfile.ZIP_DEFLATED) as colorZip:
        for dir, _, names in os.walk(args.SaveDir):
            for name in tqdm(names):
                colorZip.write(os.path.join(dir, name), 
                                os.path.join(os.path.relpath(dir, os.path.dirname(args.SaveDir)), name))
        colorZip.close()
                    
