import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from utils.utils import Resize_img, Get_resize_ratio
from torch.utils.data.dataset import Dataset
import cv2

class Yfcc(Dataset):
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

        intrinsic1 = np.asarray(line[4:13]).astype(float).reshape(3, 3)
        intrinsic2 = np.asarray(line[13:22]).astype(float).reshape(3, 3)
        size = 1024
        pose0 = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).astype(float)
        pose1 = np.asarray(line[22:]).astype(float).reshape(4, 4)
        left_rgb_path = self.data_path + line[0]
        right_rgb_path = self.data_path + line[1]
        left = cv2.imread(left_rgb_path)[:, :, [2, 1, 0]]
        h_l0, w_l0 = left.shape[:2]
        max_shape_l = max(h_l0, w_l0)
        size_l = size / max_shape_l
        left = Resize_img(left, np.array([int(w_l0 * size_l), int(h_l0 * size_l)]))
        h_l, w_l = left.shape[:2]
        r1, add_num1 = Get_resize_ratio(np.array([w_l0, h_l0]), np.array([w_l, h_l]))
        intrinsic1[0:3, 0:3] = r1 * intrinsic1[0:3, 0:3]
        intrinsic1[2, 2] = 1
        intrinsic1[0:2, 2] -= add_num1 * r1
        h_l2 = h_l // 32 * 32 + (1 - int(h_l % 32 == 0)) * 32
        w_l2 = w_l // 32 * 32 + (1 - int(w_l % 32 == 0)) * 32 
        right = cv2.imread(right_rgb_path)[:, :, [2, 1, 0]]
        h_r0, w_r0 = right.shape[:2]
        max_shape_r = max(h_r0, w_r0)
        size_r = size / max_shape_r
        right = Resize_img(right, np.array([int(w_r0 * size_r), int(h_r0 * size_r)]))
        h_r, w_r = right.shape[:2]
        r2, add_num2 = Get_resize_ratio(np.array([w_r0, h_r0]), np.array([w_r, h_r]))
        intrinsic2[0:3, 0:3] = r2 * intrinsic2[0:3, 0:3]
        intrinsic2[2, 2] = 1
        intrinsic2[0:2, 2] -= add_num2 * r2
        h_r2 = h_r // 32 * 32 + (1 - int(h_r % 32 == 0)) * 32
        w_r2 = w_r // 32 * 32 + (1 - int(w_r % 32 == 0)) * 32
        max_width = max(w_l2, w_r2)
        max_height = max(h_l2, h_r2)
        left = cv2.copyMakeBorder(left, 0, max_height - h_l, 0, max_width - w_l, cv2.BORDER_CONSTANT, None, 0)
        right = cv2.copyMakeBorder(right, 0, max_height - h_r, 0, max_width - w_r, cv2.BORDER_CONSTANT, None, 0)

        data = {
            'image0': left,
            'image1': right,
            'K0': intrinsic1[:3,:3].astype(np.float32),
            'K1': intrinsic2[:3,:3].astype(np.float32),
            'T0': pose0.astype(np.float32),
            'T1': pose1.astype(np.float32)
        }

        return data

