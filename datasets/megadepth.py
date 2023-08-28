from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import cv2
import h5py
import torch
import numpy as np
import cv2
import h5py
from utils.utils import Get_cameras, Resize_depth, Resize_img

def Compute_depth_label(depth0, depth1, P, num, patch_size, threshold=8):
    # patch_size is the distance between the point and the board of patch
    upper_bound = 1e7
    lower_bound = 1e-11
    row_num0 = depth0.shape[1] // patch_size // 2
    col_num0 = depth0.shape[2] // patch_size // 2
    row_num1 = depth1.shape[1] // patch_size // 2
    col_num1 = depth1.shape[2] // patch_size // 2
    # 在2×2 的区间排除无数据区域进行平均，在全部没有数据的情况下权值则全部设定为1
    row = np.arange(0, col_num0) * 2 * patch_size + patch_size - 1
    rows = row.reshape(1, col_num0).repeat(row_num0, axis=0).reshape(1, row_num0 * col_num0, 1).repeat(num, axis=0)
    col = np.arange(0, row_num0) * 2 * patch_size + patch_size - 1
    cols = col.reshape(row_num0, 1).repeat(col_num0, axis=1).reshape(1, row_num0 * col_num0, 1).repeat(num, axis=0)

    prefix = np.arange(0, num).reshape(num, 1).repeat(row_num0 * col_num0 * 4, axis=1).reshape(num, row_num0*col_num0, 4)
    rows_new = rows.astype(int)
    cols_new = cols.astype(int)
    depth_input_row = np.concatenate([rows_new, rows_new, rows_new + 1, rows_new + 1], axis=2)
    depth_input_col = np.concatenate([cols_new, cols_new + 1, cols_new, cols_new + 1], axis=2)
    d0 = depth0[prefix, depth_input_col, depth_input_row]
    d0_weights = (d0 > lower_bound).astype(int)
    d0_weights[d0.max(2) < lower_bound] = 1
    # d0_weights[d0 - d0.min(2)[0].reshape(d0.shape[0], row_num * col_num, 1).repeat(1, 1, 4) > 1] = 0
    d0 = np.average(d0, weights=d0_weights, axis=2).reshape(num, col_num0 * row_num0, 1)
    if_d0 = d0 < lower_bound
    d0[d0 < lower_bound] = upper_bound
    # 从左向右投影
    last_ones = np.ones([num, col_num0*row_num0, 1])
    point_input = np.concatenate([(rows + 1) * d0, (cols + 1) * d0, d0, last_ones], axis=2)
    point_output = np.einsum('ijk,ipk->ipj', P, point_input)
    point_output[:, :, 0] = point_output[:, :, 0] / point_output[:, :, 2]
    point_output[:, :, 1] = point_output[:, :, 1] / point_output[:, :, 2]
    correspondense = np.zeros([point_output.shape[0], point_output.shape[1], 3])
    correspondense[:, :, 0:2] = point_output[:, :, 0:2]
    # 去掉投出图像的投影
    if_outlier1 = np.logical_and(np.logical_or(point_output[:, :, 0] < 1, point_output[:, :, 0] >= depth1.shape[2] - 1), np.logical_not(if_d0.reshape(d0.shape[0], d0.shape[1])))
    if_outlier2 = np.logical_and(np.logical_or(point_output[:, :, 1] < 1, point_output[:, :, 1] >= depth1.shape[1] - 1), np.logical_not(if_d0.reshape(d0.shape[0], d0.shape[1])))
    correspondense[if_d0.repeat(3, axis=2)] = -1
    # 计算右图目标点的深度
    output_row = np.round(point_output[:, :, 0]).reshape(num, row_num0 * col_num0, 1).astype(int)
    output_row[np.logical_or(point_output[:, :, 0] < 2, point_output[:, :, 0] >= depth1.shape[2] - 2)] = int(depth1.shape[2] / 2)
    output_col = np.round(point_output[:, :, 1]).reshape(num, row_num0 * col_num0, 1).astype(int)
    output_col[np.logical_or(point_output[:, :, 1] < 2, point_output[:, :, 1] >= depth1.shape[1] - 2)] = int(depth1.shape[1] / 2)
    output_cols = output_col.repeat(3, axis=2)
    depth1_input_row = np.concatenate((output_row - 1, output_row, output_row + 1), axis=2).repeat(3, axis=2)
    depth1_input_col = np.concatenate((output_cols - 1, output_cols, output_cols + 1), axis=2)
    prefix = np.arange(num).reshape(num, 1).repeat(row_num0 * col_num0 * 9, axis=1).reshape(num, row_num0 * col_num0, 9)
    d1 = depth1[prefix, depth1_input_col, depth1_input_row]
    d1_weights = (d1 > lower_bound).astype(int)
    d1_weights[np.max(d1, axis=2) < lower_bound] = 1
    # d1_weights[d1 - np.min(d1, axis=2)[0].reshape(d1.shape[0], d1.shape[1], 1).repeat(4, axis=2) > 1] = 0
    d1 = np.average(d1, weights=d1_weights, axis=2).reshape(num, col_num0 * row_num0, 1)
    d1[d1 < lower_bound] = upper_bound
    # 重投影，右图向左图
    point_input2 = np.concatenate([output_row * d1, output_col * d1, d1, last_ones], axis=2)
    point_output2 = np.einsum('ijk,ipk->ipj', np.linalg.inv(P), point_input2)
    point_output2[:, :, 1] = point_output2[:, :, 1] / point_output2[:, :, 2]
    point_output2[:, :, 0] = point_output2[:, :, 0] / point_output2[:, :, 2]
    # 计算重投影误差
    distance_vector = np.abs(point_input / d0 - point_output2)
    distance = np.sqrt(np.power(distance_vector[:, :, 0], 2) + np.power(distance_vector[:, :, 1], 2))
    # print(distance)
    # correspondense[:, :, 2] = distance
    correspondense[:, :, 2] = d0[:, :, 0] / d1[:, :, 0]
    correspondense[distance.reshape(distance.shape[0], distance.shape[1], 1).repeat(3, axis=2) > threshold] = -1
    correspondense[:, :, 0][if_outlier1] = -upper_bound
    correspondense[:, :, 0][if_outlier2] = -upper_bound

    return correspondense


layer_config = {
    'layer1': {
        'patch_size': 32,
        'threshold': 8.0,
    },
    'layer2': {
        'patch_size': 8,
        'threshold': 4.0,
    },
    'layer3': {
        'patch_size': 2,
        'threshold': 1.0,
    }
}

def create_megadepth_label(left_K, right_K, left_depth, right_depth, lp, rp):
    P = right_K.dot(rp).dot(np.linalg.inv(left_K.dot(lp)))

    left_depth = left_depth[None]
    right_depth = right_depth[None]
    P = P[None]

    label_list = []
    label_reverse_list = []
    for i in range(3):
        config = layer_config['layer' + str(i + 1)]
        patch_size = config['patch_size']
        threshold = config['threshold']
        label = Compute_depth_label(left_depth, right_depth, P, 1, int(patch_size/2), threshold=threshold)
        label_reverse = Compute_depth_label(right_depth, left_depth, np.linalg.inv(P), 1, int(patch_size/2), threshold=threshold)
        label_list.append(label)
        label_reverse_list.append(label_reverse )

    pose = rp.dot(np.linalg.inv(lp))
    transform = np.array([[0, -pose[2, 3], pose[1, 3]], [pose[2, 3], 0, -pose[0, 3]], [-pose[1, 3], pose[0, 3], 0]])
    E = transform.dot(pose[:3, :3])
    F = np.linalg.inv(right_K[:3, :3]).transpose().dot(E).dot(np.linalg.inv(left_K[:3, :3]))

    return {
        'label': label_list,
        'label_reverse': label_reverse_list,
        'F': F,
    }

class MegaDepth(Dataset):
    def __init__(self, data_path, pairs_path, is_train=False, aug_resolution=False):
        if is_train:
            pairs = np.load(pairs_path + "/megadepth_train.npy")
            sequence = torch.randperm(int(pairs.shape[0]))
            data_pairs = pairs[sequence]
            self.all_pairs = data_pairs[:int(0.05 * pairs.shape[0])]
            self.imgs = Get_cameras(pairs_path, data_path, is_train, if_origin=False)
            self.setting = "train"
        else:
            pairs = np.load(pairs_path + "/megadepth_test.npy")
            self.all_pairs = pairs
            self.imgs = Get_cameras(pairs_path, data_path, is_train, if_origin=False)
            self.setting = "test"
        self.pairs_path = pairs_path
        self.data_path = data_path
        self.aug_resolution = aug_resolution 

    def __getitem__(self, item):
        the_pair = self.all_pairs[item]
        left_path = self.data_path + the_pair[0] + "/imgs/" + the_pair[3]
        left_path_depth = self.data_path + the_pair[0] + "/depths/" + the_pair[3].split('.')[0]
        right_path = self.data_path + the_pair[0] + "/imgs/" + the_pair[2]
        right_path_depth = self.data_path + the_pair[0] + "/depths/" + the_pair[2].split('.')[0]           
        left_img = cv2.imread(left_path)[:, :, [2, 1, 0]]
        right_img = cv2.imread(right_path)[:, :, [2, 1, 0]]
        h_l, w_l = left_img.shape[:2]
        h_r, w_r = right_img.shape[:2]
        max_shape_l = max(h_l, w_l)
        max_shape_r = max(h_r, w_r)
        left_depth = np.asfarray(h5py.File(left_path_depth + ".h5", "r")['depth']).astype(float)
        right_depth = np.asfarray(h5py.File(right_path_depth + ".h5", "r")['depth']).astype(float)
        if self.aug_resolution:
            size_l = 1600.0 / max_shape_l
            size_r = 1600.0 / max_shape_r
            left_img = Resize_img(left_img, [int(w_l * size_l), int(h_l * size_l)])
            right_img = Resize_img(right_img, [int(w_r * size_r), int(h_r * size_r)])
            h_l2, w_l2 = left_img.shape[:2]
            h_r2, w_r2 = right_img.shape[:2]
            left_img = left_img[:h_l2//32*32, :w_l2//32*32]
            right_img = right_img[:h_r2//32*32, :w_r2//32*32]
            max_width = max(w_l2//32*32, w_r2//32*32)
            max_height = max(h_l2//32*32, h_r2//32*32)

            left_img = cv2.copyMakeBorder(left_img, 0, max_height - h_l2//32*32, 0, max_width - w_l2//32*32, cv2.BORDER_CONSTANT, None, 0)
            right_img = cv2.copyMakeBorder(right_img, 0, max_height - h_r2//32*32, 0, max_width - w_r2//32*32, cv2.BORDER_CONSTANT, None, 0)

            left_depth = Resize_depth(left_depth, [int(w_l * size_l), int(h_l * size_l)])
            right_depth = Resize_depth(right_depth, [int(w_r * size_r), int(h_r * size_r)])

            left_img = left_img[:h_l2//32*32, :w_l2//32*32]
            right_img = right_img[:h_r2//32*32, :w_r2//32*32]

            left_depth = cv2.copyMakeBorder(left_depth, 0, max_height - h_l2//32*32, 0, max_width - w_l2//32*32, cv2.BORDER_CONSTANT, None, 0)
            right_depth = cv2.copyMakeBorder(right_depth, 0, max_height - h_r2//32*32, 0, max_width - w_r2//32*32, cv2.BORDER_CONSTANT, None, 0)
        else:
            left_img = Resize_img(left_img, [640, 480])
            right_img = Resize_img(right_img, [640, 480])

            left_depth = Resize_depth(left_depth, [640, 480])
            right_depth = Resize_depth(right_depth, [640, 480])

        l_cam = self.imgs[self.data_path + the_pair[0] + "/imgs/" + the_pair[3]]
        r_cam = self.imgs[self.data_path + the_pair[0] + "/imgs/" + the_pair[2]]

        # label_data = create_megadepth_label(l_cam["K"].astype(float), r_cam["K"].astype(float), 
        #     left_depth, right_depth, l_cam["P"].astype(float), r_cam["P"].astype(float))

        data = {
            'image0': left_img,
            'image1': right_img,
            'K0': l_cam['K'][:3,:3].astype(np.float32),
            'K1': r_cam['K'][:3,:3].astype(np.float32),
            'T0': l_cam['P'].astype(np.float32),
            'T1': r_cam['P'].astype(np.float32)
        }

        return data

    def __len__(self):
        return self.all_pairs.shape[0]

