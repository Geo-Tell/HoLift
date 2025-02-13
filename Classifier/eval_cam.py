from ast import arg
import numpy as np
import cv2, os
from tqdm import tqdm
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='HoLift')
    parser.add_argument("--CAMDir", type=str, default='', 
                        help='Directory of CAMs')
    parser.add_argument("--LabelDir", type=str, default='', 
                        help='Directory of annotations')
    parser.add_argument("--FgThre", type=float, default=0.5,
                        help='Threshold for foreground')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    width, height = 256, 256
    lb_path = args.LabelDir
    files = os.listdir(args.CAMDir)

    recalls = []
    precisions = []

    for img_name in tqdm(files):
        cls = int(img_name[-5])
        lb_name = img_name[:-6] + '_class.png'
        lb = cv2.imread(os.path.join(lb_path, lb_name), 0)
        lb[lb==255] = 6
        lb_mask = (lb == cls)

        cam = cv2.imread(os.path.join(args.CAMDir, img_name), 0)
        cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_CUBIC)
        # min_value = np.min(cam)
        max_value = np.max(cam)
        # cam = (cam - min_value) / (max_value - min_value + 1e-8)
        cam = cam / (max_value + 1e-8)
        cam_mask = (cam >= args.FgThre)

        TP = cam_mask * lb_mask
        if np.sum(cam_mask) == 0:
            precisions.append(0)
        else:
            precisions.append(np.sum(TP) / np.sum(cam_mask))
        recalls.append(np.sum(TP) / np.sum(lb_mask))
    print('Recall:', np.mean(recalls))
    print('Noise:', 1 - np.mean(precisions))
