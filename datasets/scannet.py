import warnings
warnings.filterwarnings("ignore")
import torch
import cv2
import numpy as np
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from utils.utils import Resize_img, scale_intrinsics
from torch.utils.data.dataset import Dataset


class Scannet(Dataset):
    def __init__(self, data_path, pairs_path):
        self.data_path = data_path
        self.pairs_path = pairs_path
        all_pairs = np.loadtxt(self.pairs_path, dtype=str)
        self.num_pairs = all_pairs.shape[0]
 
    def __len__(self):
        return self.num_pairs

    def __getitem__(self, index):
        with open(self.pairs_path, 'r') as fr:
            lines = fr.readlines()
            line = lines[index].split()

        name0 = line[0].split('/')
        name1 = line[1].split('/')
        intrinsic1 = np.asarray(line[4:13]).astype(float).reshape(3, 3)
        intrinsic2 = np.asarray(line[13:22]).astype(float).reshape(3, 3)
        scene_name = name0[1]
        scenes_path = self.data_path + "scans/" + scene_name + "/"
        sequence_num0 = int(name0[3][6:12])
        sequence_num1 = int(name1[3][6:12])
        left_rgb_path = scenes_path + "color/" + str(sequence_num0) + ".jpg"
        right_rgb_path = scenes_path + "color/" + str(sequence_num1) + ".jpg"
        size = 640
        pose0 = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).astype(float)
        pose1 = np.asarray(line[22:]).astype(float).reshape(4, 4)
        left = cv2.imread(left_rgb_path)[:, :, [2, 1, 0]]
        h_l, w_l = left.shape[:2]
        max_shape_l = max(h_l, w_l)
        size_l = size / max_shape_l
        left = Resize_img(left, np.array([int(w_l * size_l), int(h_l * size_l)]))
        h_l2, w_l2 = left.shape[:2]
        right = cv2.imread(right_rgb_path)[:, :, [2, 1, 0]]
        h_r, w_r = right.shape[:2]
        max_shape_r = max(h_r, w_r)
        size_r = size / max_shape_r
        right = Resize_img(right, np.array([int(w_r * size_r), int(h_r * size_r)]))
        h_r2, w_r2 = right.shape[:2]
        left = cv2.copyMakeBorder(left, 0, 480 - h_l2, 0, 640 - w_l2, cv2.BORDER_CONSTANT, None, 0)
        right = cv2.copyMakeBorder(right, 0, 480 - h_r2, 0, 640 - w_r2, cv2.BORDER_CONSTANT, None, 0)
        intrinsic1 = scale_intrinsics(intrinsic1, [float(w_l) / int(w_l * size_l), float(h_l) / int(h_l * size_l)])
        intrinsic2 = scale_intrinsics(intrinsic2, [float(w_r) / int(w_r * size_r), float(h_r) / int(h_r * size_r)])

        data = {
            'image0': left,
            'image1': right,
            'K0': intrinsic1[:3,:3].astype(np.float32),
            'K1': intrinsic2[:3,:3].astype(np.float32),
            'T0': pose0.astype(np.float32),
            'T1': pose1.astype(np.float32)
        }

        return data

