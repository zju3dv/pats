from logging import critical
from numpy import ma
from numpy.core.numeric import zeros_like
from numpy.lib.function_base import average
import torch
import torch.nn.functional as F
import numpy as np
import sys
import cv2
import h5py
import copy
import math
import glob
import os
import shutil
import collections
import tensor_resize
import imagesize
import pydegensac
import tqdm
import open3d


circle_num = 0

def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)

def merge_points_second(medium_information, medium_information2, device, height=15, width=20):
    batch_num = medium_information["if_nomatching1"].shape[0]
    first_choice_num = medium_information2["if_nomatching1"].shape[0]
    if_matching = torch.zeros([batch_num, height * width, 12, 12], device=device).bool()
    if_matching[torch.logical_not(medium_information["if_nomatching1"])] = torch.logical_not(medium_information2["if_nomatching1"].bool()).reshape(first_choice_num, 12, 12)
    if_matching = if_matching.reshape(batch_num, height, width, 3, 4, 3, 4).permute(0, 1, 4, 2, 6, 3, 5).reshape(batch_num, height * 4, width * 4, 9)
    scores = medium_information2["trust_score"].reshape(first_choice_num, 3, 4, 3, 4).permute(0, 2, 4, 1, 3).reshape(-1, 16, 9)
    scores_back = torch.zeros([batch_num, height * width, 16, 9], device=device).double()
    scores_back[torch.logical_not(medium_information["if_nomatching1"])] = scores.double()
    scores_back = scores_back.reshape(batch_num, height, width, 4, 4, 9).permute(0, 1, 3, 2, 4, 5).reshape(batch_num, height * 4, width * 4, 9)
    scores_back[:, :, :, 4] -= 0.0
    for i in range(9):
        dy = i % 3 - 1
        dx = i // 3 - 1
        scores_back[:, 4 * max(dx, 0):height * 4 + min(0, dx) * 4, 4 * max(dy, 0):width * 4 + min(0, dy) * 4, i] = scores_back[:, 4 * max(-dx, 0):height * 4 + min(0, -dx) * 4, 4 * max(-dy, 0):width * 4 + min(0, -dy) * 4, i].clone()
        if_matching[:, 4 * max(dx, 0):height * 4 + min(0, dx) * 4, 4 * max(dy, 0):width * 4 + min(0, dy) * 4, i] = if_matching[:, 4 * max(-dx, 0):height * 4 + min(0, -dx) * 4, 4 * max(-dy, 0):width * 4 + min(0, -dy) * 4, i].clone()
        # if_matching[:, :, 0] = torch.logical_or(if_matching[:, :, 0], if_matching[:, :, i])
    scores_back[if_matching] -= 10000.0
    sequence_base = torch.argsort(scores_back)[:, :, :, 0]
    if_matching = torch.gather(if_matching, 3, sequence_base.reshape(batch_num, height * 4, width * 4, 1)).reshape(batch_num, -1)
    sequence2 = sequence_base.reshape(batch_num, -1, 1) + torch.arange(height * width * 16, device=device).\
        reshape(1, -1, 1).repeat(batch_num, 1, 1) * 9 - (sequence_base.reshape(batch_num, -1, 1) % 3 - 1) * 4 * 9 - \
        (sequence_base.reshape(batch_num, -1, 1) // 3 - 1) * 4 * width * 4 * 9
    

    h = torch.arange(height * width * 16, device=device).reshape(1, -1).repeat(batch_num, 1) // width // 4 - \
        (sequence_base.reshape(batch_num, -1) // 3 - 1) * 4
    w = torch.arange(height * width * 16, device=device).reshape(1, -1).repeat(batch_num, 1) % (width * 4) - \
        (sequence_base.reshape(batch_num, -1) % 3 - 1) * 4 
    critical1 = torch.logical_or(h < 0, h >= height * 4)
    critical2 = torch.logical_or(w < 0, w >= width * 4)
    critical = torch.logical_or(critical1, critical2)
    if_matching[critical] = False
    sequence2 = torch.clamp(sequence2, 0, height * width * 144 - 1)
    if_nomatching = torch.ones([batch_num, height * 4 * width * 4 * 9], device=device)
    if_nomatching = torch.scatter(if_nomatching, 1, sequence2.reshape(batch_num, -1), torch.logical_not(if_matching).float()).bool()
    return if_nomatching.reshape(batch_num, height, 4, width, 4, 3, 3).permute(0, 1, 3, 5, 2, 6, 4).reshape(batch_num, height * width, 144)[torch.logical_not(medium_information["if_nomatching1"])]

# def merge_points(medium_information, medium_information2, matches, device, height=15, width=20, length=16):
#     first_choice_num = medium_information2["if_nomatching1"].shape[0]
#     if_matching2 = torch.logical_not(medium_information2["if_nomatching1"].bool()).reshape(-1, 3 * length, 3 * length)
#     if_matching = torch.zeros([width * height, 3 * length, 3 * length], device=device).bool()
#     if_matching[torch.logical_not(medium_information["if_nomatching1"].reshape(-1))] = if_matching2
#     if_matching = if_matching.reshape(height, width, 3, length, 3, length).permute(0, 3, 1, 5, 2, 4).reshape(height * length, width * length, 9)
#     scores = medium_information2["trust_score"].reshape(-1, 3, length, 3, length).permute(0, 2, 4, 1, 3).reshape(-1, length ** 2, 9)
#     scores_back = torch.zeros([width * height, length ** 2, 9], device=device).double() + 1e4
#     scores_back[torch.logical_not(medium_information["if_nomatching1"]).reshape(-1)] = scores.double()
#     scores_back = scores_back.reshape(height, width, length, length, 9).permute(0, 2, 1, 3, 4).reshape(height * length, width * length, 9)
#     matches_back = torch.zeros([first_choice_num, (3 * length) ** 2, 4], device=device)
#     matches_back[torch.logical_not(medium_information2["if_nomatching1"].bool())] = matches.reshape(-1, 4)
#     # print(matches.reshape(-1, 16, 4)[0, :, 0:2])
#     matches_back2 = torch.zeros([width * height, (3 * length) ** 2, 4], device=device)
#     matches_back2[torch.logical_not(medium_information["if_nomatching1"].reshape(-1))] = matches_back
#     matches_back2 = matches_back2.reshape(height, width, 3, 4, 4, 3, 4, 4, 4).permute(0, 3, 4, 1, 6, 7, 8, 2, 5).reshape(height * 16, width * 16, 4, 9)
#     # scores_back[:, :, 4] -= 1000.0
#     # length = 16
#     for i in range(9):
#         dy = i % 3 - 1
#         dx = i // 3 - 1
#         scores_back[length * max(dx, 0):height * length + min(0, dx) * length, length * max(dy, 0):width * length + min(0, dy) * length, i] = scores_back[length * max(-dx, 0):height * length + min(0, -dx) * length, length * max(-dy, 0):width * length + min(0, -dy) * length, i].clone()
#         if_matching[length * max(dx, 0):height * length + min(0, dx) * length, length * max(dy, 0):width * length + min(0, dy) * length, i] = if_matching[length * max(-dx, 0):height * length + min(0, -dx) * length, length * max(-dy, 0):width * length + min(0, -dy) * length, i].clone()
#         matches_back2[16 * max(dx, 0):height * 16 + min(0, dx) * 16, 16 * max(dy, 0):width * 16 + min(0, dy) * 16, :, i] = matches_back2[16 * max(-dx, 0):height * 16 + min(0, -dx) * 16, 16 * max(-dy, 0):width * 16 + min(0, -dy) * 16, :, i].clone()
#         # if_matching[:, :, 0] = torch.logical_or(if_matching[:, :, 0], if_matching[:, :, i])
#     scores_back[if_matching] -= 1e7
#     sequence2 = torch.argsort(scores_back)[:, :, 0]
#     # if_matching = if_matching[:, :, 0].reshape(height * 4, 1, width * 4, 1).repeat(1, 8, 1, 8).reshape(480, 640)
#     batch_num = medium_information["if_nomatching1"].shape[0]
#     sequence2_new = sequence2.reshape(batch_num, -1, 1) + torch.arange(height * width * length * length, device=device).\
#         reshape(1, -1, 1).repeat(batch_num, 1, 1) * 9 - (sequence2.reshape(batch_num, -1, 1) % 3 - 1) * length * 9 - \
#         (sequence2.reshape(batch_num, -1, 1) // 3 - 1) * length * width * length * 9
#     if_matching2 = torch.gather(if_matching, 2, sequence2.reshape(height * length, width * length, 1)).reshape(batch_num, -1)
#     h = torch.arange(height * width * 256, device=device).reshape(1, -1).repeat(batch_num, 1) // width // length - \
#         (sequence2.reshape(batch_num, -1) // 3 - 1) * length * width * length
#     w = torch.arange(height * width * 256, device=device).reshape(1, -1).repeat(batch_num, 1) % (width * length) - \
#         (sequence2.reshape(batch_num, -1) % 3 - 1) * length * width * length
#     critical1 = torch.logical_or(h < 0, h >= height * length)
#     critical2 = torch.logical_or(w < 0, w >= width * length)
#     critical = torch.logical_or(critical1, critical2)
#     if_matching2[critical] = False
#     sequence2_new = torch.clamp(sequence2_new, 0, height * width * 48 * 48 - 1)
#     if_nomatching = torch.ones([batch_num, height * 16 * width * 16 * 9], device=device)
#     if_nomatching = torch.scatter(if_nomatching, 1, sequence2_new.reshape(batch_num, -1), torch.logical_not(if_matching2).float()).bool()
#     if_nomatching = if_nomatching.reshape(batch_num, height, 16, width, 16, 3, 3).permute(0, 1, 3, 5, 2, 6, 4).reshape(batch_num, height * width, 48 * 48)[torch.logical_not(medium_information["if_nomatching1"])]
#     medium_information2["if_nomatching1"] = if_nomatching
    
#     if_matching = torch.gather(if_matching, 2, sequence2.reshape(height * 16, width * 16, 1)).reshape(height * 16, 1, width * 16, 1).reshape(height * 16, width * 16)
#     matches_new = torch.gather(matches_back2, 3, sequence2.reshape(height * 16, 1, width * 16, 1, 1, 1).repeat(1, 1, 1, 1, 4, 1).reshape(height * 16, width * 16, 4, 1)).reshape(height * 16, width * 16, 4)
#     return if_matching, matches_new

def merge_points(medium_information, medium_information2, matches, device, height=15, width=20):
    first_choice_num = medium_information2["if_nomatching1"].shape[0]
    if_matching2 = torch.logical_not(medium_information2["if_nomatching1"].bool()).reshape(-1, 12, 4, 12, 4)[:, :, 0, :, 0]
    if_matching = torch.zeros([width * height, 12, 12], device=device).bool()
    if_matching[torch.logical_not(medium_information["if_nomatching1"].reshape(-1))] = if_matching2
    if_matching = if_matching.reshape(height, width, 3, 4, 3, 4).permute(0, 3, 1, 5, 2, 4).reshape(height * 4, width * 4, 9)
    scores = medium_information2["trust_score"].reshape(-1, 3, 4, 3, 4).permute(0, 2, 4, 1, 3).reshape(-1, 16, 9)
    scores_back = torch.zeros([width * height, 16, 9], device=device).double()
    scores_back[torch.logical_not(medium_information["if_nomatching1"]).reshape(-1)] = scores.double()
    scores_back = scores_back.reshape(height, width, 4, 4, 9).permute(0, 2, 1, 3, 4).reshape(height * 4, width * 4, 9)
    matches_back = torch.zeros([first_choice_num, 48 * 48, 4], device=device)
    matches_back[torch.logical_not(medium_information2["if_nomatching1"].bool())] = matches.reshape(-1, 4)
    # print(matches.reshape(-1, 16, 4)[0, :, 0:2])
    matches_back2 = torch.zeros([width * height, 48 * 48, 4], device=device)
    matches_back2[torch.logical_not(medium_information["if_nomatching1"].reshape(-1))] = matches_back
    matches_back2 = matches_back2.reshape(height, width, 3, 4, 4, 3, 4, 4, 4).permute(0, 3, 4, 1, 6, 7, 8, 2, 5).reshape(height * 16, width * 16, 4, 9)
    # scores_back[:, :, 4] -= 0.01
    for i in range(9):
        dy = i % 3 - 1
        dx = i // 3 - 1
        scores_back[4 * max(dx, 0):height * 4 + min(0, dx) * 4, 4 * max(dy, 0):width * 4 + min(0, dy) * 4, i] = scores_back[4 * max(-dx, 0):height * 4 + min(0, -dx) * 4, 4 * max(-dy, 0):width * 4 + min(0, -dy) * 4, i].clone()
        if_matching[4 * max(dx, 0):height * 4 + min(0, dx) * 4, 4 * max(dy, 0):width * 4 + min(0, dy) * 4, i] = if_matching[4 * max(-dx, 0):height * 4 + min(0, -dx) * 4, 4 * max(-dy, 0):width * 4 + min(0, -dy) * 4, i].clone()
        matches_back2[16 * max(dx, 0):height * 16 + min(0, dx) * 16, 16 * max(dy, 0):width * 16 + min(0, dy) * 16, :, i] = matches_back2[16 * max(-dx, 0):height * 16 + min(0, -dx) * 16, 16 * max(-dy, 0):width * 16 + min(0, -dy) * 16, :, i].clone()
        # if_matching[:, :, 0] = torch.logical_or(if_matching[:, :, 0], if_matching[:, :, i])
    scores_back[if_matching] -= 1000
    sequence2 = torch.argsort(scores_back)[:, :, 0]
    if_matching = torch.gather(if_matching, 2, sequence2.reshape(height * 4, width * 4, 1)).reshape(height * 4, 1, width * 4, 1).repeat(1, 4, 1, 4).reshape(height * 16, width * 16)
    matches_new = torch.gather(matches_back2, 3, sequence2.reshape(height * 4, 1, width * 4, 1, 1, 1).repeat(1, 4, 1, 4, 4, 1).reshape(height * 16, width * 16, 4, 1)).reshape(height * 16, width * 16, 4)
    # if_matching = if_matching[:, :, 0].reshape(height * 4, 1, width * 4, 1).repeat(1, 8, 1, 8).reshape(480, 640)
    return if_matching, matches_new

    

def split_patches(sum_cycle, height, width, max_once_used = 350):
    cycle_num = 0
    second_layer_set = []
    third_layer_set = []
    last_second_line = 0
    last_third_line = 0
    for i in range(height):
        num = sum_cycle[(i + 1) * width - 1]
        if num > max_once_used * (cycle_num + 1):
            if last_second_line == 0:
                origin_num = 0
            else:
                origin_num = sum_cycle[last_second_line * width - 1]
            cycle_num += 1
            second_layer_set.append([origin_num, sum_cycle[(i + 1) * width - 1]])
            third_layer_set.append([sum_cycle[last_third_line * width] - origin_num, sum_cycle[(i + 1) * width - 1] - sum_cycle[i * width - 1]])
            last_second_line = i
            last_third_line = i + 1
    if last_second_line == 0:
        origin_num = 0
    else:
        origin_num = sum_cycle[last_second_line * width - 1]
    cycle_num += 1
    second_layer_set.append([origin_num, height * width])
    if last_third_line == height:
        end_num = origin_num
    else:
        end_num = sum_cycle[last_third_line * width] 
    third_layer_set.append([end_num - origin_num, 0])
    return cycle_num, second_layer_set, third_layer_set

# 这里的average point是没有乘patch_scale的，而scale则单纯指放缩比例
# 考虑left_choice相关的问题的话，感觉就非常难搞了。。。似乎ifnomatching和avergepoint之类的东西都必须重新安排才可以，这里先假设输入都是已经修改好了的
# 在这种情况下，把左点和右点分开处理也就可以了,然后用left_choice完成交换
# 这里假设，筛选的结果会在if_nomatching上动手脚
# patch_size需要包括width和height之类的东西在内才行，[patch_size, height, width], layer_num * 3
# 现在这个结果已经是考虑到全部scaling的结果了
def get_result(batch_size, if_nomatching, average_point, scale, patch_size, left_choice, layer_num=2):
    matches_l = torch.zeros([batch_size, 2], device=average_point[0].device)
    matches_r = torch.zeros([batch_size, 2], device=average_point[0].device)
    for i in range(layer_num):
        the_matching = torch.logical_not(if_nomatching[i])
        the_point = average_point[i]
        the_scale = scale[i].reshape(scale[i].shape[0], scale[i].shape[1], 2, 1).repeat(1, 1, 1, 2)
        the_size = patch_size[i]
        the_choice = left_choice[i]
        positions_y = torch.arange(0, the_size[1] * the_size[0], the_size[0]).reshape(1, -1, 1, 1)\
            .repeat(the_matching.shape[0], 1, the_size[2], 1).to(average_point[0].device)
        positions_x = torch.arange(0, the_size[2] * the_size[0], the_size[0]).reshape(1, 1, -1, 1).\
            repeat(the_matching.shape[0], the_size[1], 1, 1).to(average_point[0].device)
        matches_l = matches_l.reshape(-1, 1, 2).repeat(1, the_matching.shape[1], 1)
        matches_r = matches_r.reshape(-1, 1, 2).repeat(1, the_matching.shape[1], 1)
        delta_l = torch.cat([positions_y, positions_x], dim=3).reshape(the_matching.shape[0], -1, 2) + 0.5 * the_size[0]
        if i < layer_num - 1:
            delta_l = delta_l - 1.5 * the_scale[:, :, 1] * the_size[0]
            delta_r = (the_point - 1.5 * the_scale[:, :, 0]) * the_size[0]
        else:
            delta_l = delta_l * the_scale[:, :, 1]
            delta_r = (the_point * the_size[0]) * the_scale[:, :, 0]
        matches_l = (matches_l + torch.where(the_choice, delta_l.permute(1,2,0), delta_r.permute(1,2,0)).permute(2,0,1))[the_matching]
        matches_r = (matches_r + torch.where(the_choice, delta_r.permute(1,2,0), delta_l.permute(1,2,0)).permute(2,0,1))[the_matching]
    return matches_l, matches_r

#这里需要改的地方就是要把本来每个patch都算一个结果出来，再根据if_matching筛选的过程改成输入每一层都直接是sp点的对应输入
def get_result_superpoint(point_num, average_point, the_scale, patch_size, layer_num=2):
    matches_r = torch.zeros([point_num, 2], device=average_point[0].device)
    for i in range(layer_num):
        the_point = average_point[i]
        the_size = patch_size[i]
        if i < layer_num - 1:
            delta_r = (the_point - 1.5 * the_scale[:, 0].reshape(-1, 1).repeat(1, 2)) * the_size[0]
        else:
            delta_r = (the_point * the_size[0]) * the_scale[:, 0].reshape(-1, 1).repeat(1, 2)
        matches_r = matches_r + delta_r
    return matches_r

# 假设这里输入的left和right的形式是torch，维度为（h，w, 3）
def matches_show(left, right, matches_l, matches_r, sequence, path, epipolar_distance=None):
    left_show = np.int16(left.cpu().numpy())[0]
    right_show = np.int16(right.cpu().numpy())[0]
    map = cv2.hconcat([left_show, right_show])
    if epipolar_distance == None:
        for j in range(matches_l.shape[0]):
            if j % 10 != 0:
                continue
            colar = np.random.randint(0, 256, [3])
            cv2.line(map, (matches_l[j, 1], matches_l[j, 0]), (matches_r[j, 1]+left_show.shape[1], matches_r[j, 0]),
                    (int(colar[0]), int(colar[1]), int(colar[2])))
    else:
        for j in range(matches_l.shape[0]):
            # if j % 10 != 0:
            #     continue
            if epipolar_distance[j] > 1:
                colar = [0, 0, 255]
            else:
                colar = [0, 255, 0]
            cv2.line(map, (matches_l[j, 1], matches_l[j, 0]), (matches_r[j, 1]+left_show.shape[1], matches_r[j, 0]),
                    (int(colar[0]), int(colar[1]), int(colar[2])))    
    cv2.imwrite(path + "test_" + str(sequence) + "_" + str(0) + ".jpg", map)

def Get_cameras(path, img_folder, is_train=False, if_origin=False):
    if is_train:
        set_file = path + "/megadepth_train_scenes.txt"
    else:
        set_file = path + "/megadepth_validation_scenes_full.txt"
    f = open(set_file, 'r')
    lines = np.array(f.readlines())
    images = {}
    for i in range(lines.shape[0]):
        if not (os.path.exists(path + lines[i][:-1])):
            continue
        dense_name = np.sort(np.array(os.listdir(path + lines[i][:-1])))
        for j in range(dense_name.shape[0]):
            path_now = path + lines[i][:-1] + "/" + dense_name[j]
            img_cam_txt_path = os.path.join(path_now, 'img_cam.txt')
            img_cam_txt_path2 = os.path.join(path_now, 'img_cam_new.txt')
            with open(img_cam_txt_path, "r") as fid:
                fid2 = open(img_cam_txt_path2, "r") 
                while True:
                    line = fid.readline()
                    line2 = fid2.readline()
                    if not line or not line2:
                        break
                    line = line.strip()
                    line2 = line2.strip()
                    if len(line) > 0 and line[0] != "#":
                        elems = line.split()
                        elems2 = line2.split()
                        image_name = elems[0]
                        img_path = os.path.join(img_folder + lines[i][:-1] + "/" + dense_name[j] + "/imgs", image_name)
                        w, h = int(elems2[1]), int(elems2[2])
                        fx, fy = float(elems2[3]), float(elems2[4])
                        cx, cy = float(elems2[5]), float(elems2[6])
                        P = np.array(elems[7:19]).reshape(3, 4)
                        last_line = np.array([0, 0, 0, 1])
                        P = np.concatenate([P, last_line.reshape(1, -1)], axis=0)
                        R = np.array(P[0:3, 0:3])
                        T = np.array(P[0:3, 3])
                        intrinsic = np.array([[fx, 0, cx],
                                              [0, fy, cy],
                                              [0, 0, 1]])
                        if if_origin==False:
                            r1, add_num1 = Get_resize_ratio(np.array([w, h]), np.array([640, 480]))
                            k1 = np.identity(4)
                            k1[0:3, 0:3] = r1 * intrinsic[0:3, 0:3]
                            k1[2, 2] = 1
                            k1[0:2, 2] -= add_num1 * r1
                            images[img_path] = {
                                "name":image_name, "K":k1, "R":R, "T":T, "P":P}
                        else:
                            # img_path2 = img_folder + lines[i][:-1] + "/" + dense_name[j] + "/imgs/" + image_name
                            max_shape = max(w, h)
                            size = 1600.0 / max_shape
                            r1, add_num1 = Get_resize_ratio(np.array([w, h]), np.array([int(w * size), int(h *  size)]))
                            # r1, add_num1 = Get_resize_ratio(np.array([w, h]), np.array([640, 480]))
                            k1 = np.identity(4)
                            k1[0:3, 0:3] = r1 * intrinsic[0:3, 0:3]
                            k1[2, 2] = 1
                            k1[0:2, 2] -= add_num1 * r1
                            images[img_path] = {
                                "name":image_name, "K":k1, "R":R, "T":T, "P":P, "h": int(h*size), "w": int(w*size)}
    return images

# def get_pose_error(kp1, kp2, intrinsic1, intrinsic2, pose, imf1, imf2):
#     # intrinsic_mean = (intrinsic1 + intrinsic2) / 2.
#     # focal = (intrinsic_mean[0, 0] + intrinsic_mean[1, 1]) / 2.
#     # pp = tuple(intrinsic_mean[0:2, 2])
#     # E, mask = cv2.findEssentialMat(kp1, kp2, focal=focal, pp=pp, method=cv2.RANSAC)
#     # F, mask = cv2.findFundamentalMat(kp1.astype(float), kp2.astype(float), cv2.LMEDS, 1.0, 0.999)
#     F, mask = cv2.findFundamentalMat(kp1.astype(float), kp2.astype(float), cv2.RANSAC, 1.0, 0.99999)
#     # F, mask = pydegensac.findFundamentalMatrix(kp1.astype(float), kp2.astype(float), 1.0, 0.9999, 1000)
#     mask = mask.reshape(-1).astype(bool)
#     # mask = 1
#     # F_gt = self.result[i]["F_gt"]
#     transform = np.array([[0, -pose[2, 3], pose[1, 3]], [pose[2, 3], 0, -pose[0, 3]], [-pose[1, 3], pose[0, 3], 0]])
#     E_gt = transform.dot(pose[:3, :3])
#     F_gt = np.linalg.inv(intrinsic2).transpose().dot(E_gt).dot(np.linalg.inv(intrinsic1))
#     third_line = np.ones([kp1.shape[0], 1])
#     p1 = np.concatenate([kp1, third_line], axis=1)
#     p2 = np.concatenate([kp2, third_line], axis=1)
#     line = np.einsum("jk,ik->ij", F_gt, p1)
#     result = np.einsum("ij,jk,ik->i", p2, F_gt, p1) / np.sqrt(line[:, 0] ** 2 + line[:, 1] ** 2)
#     result_masked = np.einsum("ij,jk,ik->i", p2[mask], F_gt, p1[mask]) / np.sqrt(line[mask][:, 0] ** 2 + line[mask][:, 1] ** 2)
#     E = intrinsic2.transpose().dot(F).dot(intrinsic1)
#     try:
#         R1, R2, t = cv2.decomposeEssentialMat(E)
#     except:
#         print(imf1, imf2)
#         print(E)
#     R_gt, t_gt = pose[:3, :3], pose[:3, 3]
#     theta_1 = np.arccos(np.clip((np.trace(R1.T.dot(R_gt)) - 1) / 2, -1, 1))
#     theta_2 = np.arccos(np.clip((np.trace(R2.T.dot(R_gt)) - 1) / 2, -1, 1))
#     theta = min(theta_1, theta_2) * 180 / np.pi
#     t = np.squeeze(t)
#     tran_cos = np.inner(t, t_gt) / (np.linalg.norm(t_gt) * np.linalg.norm(t))
#     tran1 = np.arccos(np.clip(np.abs(tran_cos), -1, 1)) * 180 / np.pi
#     tran_cos2 = np.inner(-t, t_gt) / (np.linalg.norm(t_gt) * np.linalg.norm(t))
#     tran2 = np.arccos(np.clip(np.abs(tran_cos2), -1, 1)) * 180 / np.pi
#     tran = min(tran1, tran2)
#     # print(mask.shape, mask.astype(int).sum(), theta, np.mean(np.abs(result)),
#     #           np.mean(np.abs(result[mask.reshape(-1).astype(bool)])), (np.trace(R1.T.dot(R_gt)) - 1) / 2,
#     #           (np.trace(R2.T.dot(R_gt)) - 1) / 2, np.linalg.norm(t_gt), tran)
#     return theta, tran, kp1, kp2, result_masked, result

def get_pose_error(kpts0, kpts1, K0, K1, pose, imf1, imf2, threshold, num):

    # colars = np.zeros([kpts0.shape[0], 3])

    # F, mask2 = cv2.findFundamentalMat(kpts0.astype(float), kpts1.astype(float), cv2.RANSAC, 3.0, 0.99999)
    # F, mask = pydegensac.findFundamentalMatrix(kp1.astype(float), kp2.astype(float), 1.0, 0.9999, 1000)
    # mask2 = mask2.reshape(-1).astype(bool)
    # mask = 1
    # F_gt = self.result[i]["F_gt"]
    transform = np.array([[0, -pose[2, 3], pose[1, 3]], [pose[2, 3], 0, -pose[0, 3]], [-pose[1, 3], pose[0, 3], 0]])
    E_gt = transform.dot(pose[:3, :3])
    F_gt = np.linalg.inv(K1).transpose().dot(E_gt).dot(np.linalg.inv(K0))
    third_line = np.ones([kpts0.shape[0], 1])
    p1 = np.concatenate([kpts0, third_line], axis=1)
    p2 = np.concatenate([kpts1, third_line], axis=1)
    line = np.einsum("jk,ik->ij", F_gt, p1)
    result = np.einsum("ij,jk,ik->i", p2, F_gt, p1) / np.sqrt(line[:, 0] ** 2 + line[:, 1] ** 2)
    # result_masked = np.einsum("ij,jk,ik->i", p2[mask2], F_gt, p1[mask2]) / np.sqrt(line[mask2][:, 0] ** 2 + line[mask2][:, 1] ** 2)

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = threshold / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=1-1e-5,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    R_gt = pose[:3, :3]
    t_gt = pose[:3, 3]
    if ret != None:
        R, t, _ = ret
    else:
        R, t = R, t[:, 0]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)

    return error_R, error_t, kpts0, kpts1, None, result



def CheckPlyFileExportCondition(disparityZ):
    # position conditions
    condPosZ = (0 < disparityZ and disparityZ < 1000)
    return condPosZ

def SaveWorldImageToPLY(p3d, plyFilename):
    with open(plyFilename, 'w') as f: # write to a new file (NOT APPEND)
        f.write('ply\n')
        f.write('format ascii 1.0\n')

        # count possible display points
        count = 0
        for i in range(p3d.shape[0]):
                disparityZ = p3d[i][2]
                if CheckPlyFileExportCondition(disparityZ):
                    count = count + 1
        f.write('element vertex ' + str(count) + '\n')

        # other headers
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')

        # write points
        for i in range(p3d.shape[0]):
                disparityX = p3d[i][0]
                disparityY = p3d[i][1]
                disparityZ = p3d[i][2]
                pixelB = int(p3d[i][3])
                pixelG = int(p3d[i][4])
                pixelR = int(p3d[i][5])
                if CheckPlyFileExportCondition(disparityZ):
                    f.write(format(disparityX,'.4f') + ' ' + format(disparityY,'.4f') + ' ' + format(disparityZ,'.4f') \
                            + ' ' + str(pixelR) + ' ' + str(pixelG) + ' ' + str(pixelB) + '\n')
    return

def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.round(np.trapz(r, x=e)/t, 3))
    return aucs

def Compute_accuracy(results, left_cameras, right_cameras, scale_factor=1.0, threshold=0.25):
    R_errors = []
    T_errors = []
    exist_rates_32 = []
    exist_rates_8 = []
    recall_rates_1 = []
    recall_rates_2 = []
    recall_rates_3 = []
    recall_rates_4 = []
    recall_rates_5 = []
    recall_rates_6 = []
    recall_rates_sum_32 = []
    recall_rates_sum_8 = []
    accuracy_rates_1 = []
    accuracy_rates_2 = []
    accuracy_rates_3 = []
    accuracy_rates_sum = []
    point_errs = []
    point_errs_masked = []
    for i in tqdm.tqdm(range(results.__len__()), ncols=50):
        pair = results[i]
        if pair == -1:
            R_errors.append(1000)
            T_errors.append(1000)
            exist_rates_32.append(0)
            exist_rates_8.append(0)
            recall_rates_1.append(0)
            recall_rates_2.append(0)
            recall_rates_3.append(0)
            recall_rates_4.append(0)
            recall_rates_5.append(0)
            recall_rates_6.append(0)
            accuracy_rates_1.append(0)
            accuracy_rates_2.append(0)
            accuracy_rates_3.append(0)
            recall_rates_sum_32.append(1e-10)
            recall_rates_sum_8.append(1e-10)
            accuracy_rates_sum.append(1e-10)
            continue
        kp1 = pair["matches_l"]
        kp2 = pair["matches_r"]
        if kp1.shape[0] < 15:
            R_errors.append(1000)
            T_errors.append(1000)
            exist_rates_32.append(0)
            exist_rates_8.append(0)
            recall_rates_1.append(0)
            recall_rates_2.append(0)
            recall_rates_3.append(0)
            recall_rates_4.append(0)
            recall_rates_5.append(0)
            recall_rates_6.append(0)
            accuracy_rates_1.append(0)
            accuracy_rates_2.append(0)
            accuracy_rates_3.append(0)
            recall_rates_sum_32.append(1e-10)
            recall_rates_sum_8.append(1e-10)
            accuracy_rates_sum.append(1e-10)
            continue
        kp1 = np.concatenate([kp1[:, 1].reshape(-1, 1), kp1[:, 0].reshape(-1, 1)], axis=1)
        kp2 = np.concatenate([kp2[:, 1].reshape(-1, 1), kp2[:, 0].reshape(-1, 1)], axis=1)
        img1, img2 = left_cameras[i], right_cameras[i]
        intrinsic1, intrinsic2 = copy.copy(img1['K']), copy.copy(img2['K'])
        intrinsic2[:3, :3] = scale_intrinsics(intrinsic2[:3, :3], [1.0/ scale_factor, 1.0/ scale_factor])
        if scale_factor > 1.0:
            intrinsic1[:2, 2] += np.asarray([int((scale_factor-1)*320), int((scale_factor-1)*240)])
        else:
            intrinsic2[:2, 2] += np.asarray([int((1 - scale_factor)*320), int((1 - scale_factor)*240)])
        extrinsic1, extrinsic2 = img1['P'], img2['P']
        pose = extrinsic2.astype(float).dot(np.linalg.inv(extrinsic1.astype(float)))
        R_error, T_error, kp1, kp2, result_masked, result = get_pose_error(kp1, kp2, intrinsic1[0:3, 0:3],
                intrinsic2[0:3, 0:3], pose, pair["left_path"][0], pair["right_path"][0], threshold=threshold, num=i)
        point_errs.append(np.mean(np.abs(result)))
        results[i]["R_error"] = R_error
        results[i]["T_error"] = T_error
        results[i]["epipolar_distance"] = result
        R_errors.append(max(R_error, T_error))
        T_errors.append(min(R_error, T_error))
    R_errors = np.array(R_errors)
    T_errors = np.array(T_errors)
    R_2_accuracy = np.mean(R_errors < 0.5)
    R_5_accuracy = np.mean(R_errors < 5)
    T_5_accuracy = np.mean(T_errors < 5)
    R_10_accuracy = np.mean(R_errors < 20)
    T_10_accuracy = np.mean(T_errors < 20)
    R_median = np.median(R_errors)
    T_median = np.median(T_errors)
    err_median = np.median(np.array(point_errs))
    err_ransac_median = np.median(np.array(point_errs_masked))
    aucs = pose_auc(R_errors, [5, 10, 20])
    aucs = [100.*yy for yy in aucs]
    print('R_0.5_accuracy: {}, R_5_accuracy: {}, T_5_accuracy: {}, R_20_accuracy: {}, T_20_accuracy: {},'
          ' R_median: {}, T_median: {}, err_median: {}, err_ransac_median: {}'
          .format(np.round(R_2_accuracy, 4), np.round(R_5_accuracy, 4), np.round(T_5_accuracy, 4),
                  np.round(R_10_accuracy, 4), np.round(T_10_accuracy, 4),
                  np.round(R_median, 4), np.round(T_median, 4),
                  np.round(err_median, 4), np.round(err_ransac_median, 4)))

    print('{:.3}/{:.3}/{:.3}'.format(aucs[0], aucs[1], aucs[2]))
    return results

def compute_distance_matrix(matches, mask_base, epipolar_distance, patch_size):
    matches_truncated = matches.long() // patch_size

    criterion1 = torch.logical_or(matches_truncated[:, 0] < 0, matches_truncated[:, 1] < 0)
    criterion2 = torch.logical_or(matches_truncated[:, 0] >= mask_base.shape[0], matches_truncated[:, 1] >= mask_base.shape[1])
    criterion = torch.logical_not(torch.logical_or(criterion1, criterion2))
    matches_truncated = matches_truncated[criterion]
    matches = matches[criterion]
    epipolar_distance = epipolar_distance[criterion]

    matches_truncated = matches_truncated[:, 0] * mask_base.shape[1] + matches_truncated[:, 1]
    matches = torch.cat([matches, epipolar_distance.reshape(-1, 1)], dim=1)
    distance_matirx = torch.zeros([mask_base.shape[0], mask_base.shape[1]], device=matches.device) + 1e7
    
    mask = torch.zeros([mask_base.shape[0], mask_base.shape[1]], device=matches.device).bool().reshape(-1)
    mask[matches_truncated] = True
    mask = mask.reshape(mask_base.shape[0], mask_base.shape[1])


    mask = torch.logical_and(mask, mask_base)
    matches_mask = torch.gather(mask.reshape(-1), dim=0, index=matches_truncated)
    matches = matches[matches_mask]
    matches_truncated = matches_truncated[matches_mask]
    matches_sorted, origin_sequence = torch.sort(matches_truncated)
    matches_selected = torch.zeros([matches_truncated.shape[0]], device=matches.device).bool()
    matches_selected[1:][matches_sorted[:-1] < matches_sorted[1:]] = True
    if matches_selected.shape[0] >= 1:
        matches_selected[0] = True

    matches_truncated = matches_truncated.float() + torch.min(torch.tensor(0.999, device=matches.device), epipolar_distance[matches_mask] / 1000.0)
    _, sequence_sorted = torch.sort(matches_truncated)
    sequence_sorted = sequence_sorted[matches_selected]
    _, sequence_reverse = torch.sort(origin_sequence)
    matches = matches[sequence_sorted]
    distance_matirx[mask] = matches[:, 2].float()
    return mask, distance_matirx, criterion, matches_selected[sequence_reverse]


import plotly, PIL.ImageColor
def get_color(colorscale_name, loc):
    from _plotly_utils.basevalidators import ColorscaleValidator
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...]
    colorscale = cv.validate_coerce(colorscale_name)

    if hasattr(loc, "__iter__"):
        intermediate_colors = [get_continuous_color(colorscale, x) for x in loc]
        return intermediate_colors
    return get_continuous_color(colorscale, loc)


def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.
    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:
        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]
    Others are just swatches that need to be constructed into a colorscale:
        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)
    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    hex_to_rgb = lambda c: "rgb" + str(PIL.ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    intermediate_color = plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )
    return intermediate_color

def get_plotly_colors(num_points, colorscale):
    color_steps = torch.linspace(start=0, end=1, steps=num_points).tolist()
    colors = get_color(colorscale, color_steps)
    colors = [plotly.colors.unlabel_rgb(color) for color in colors]
    colors = torch.tensor(colors, dtype=torch.float, device='cuda').view(1, num_points, 3)
    colors = colors.div(255.0).add(-0.5).mul(2)  # Map [0, 255] RGB colors to [-1, 1]
    return colors  # (1, P, 3)

#这里的label姑且先专门指代epipolar loss，不过理论上homography也不应该有问题才对
def Compute_new_img(medium_information, left, right, desc_l, desc_r, label, label_reverse):
    if_nomatching = medium_information["if_nomatching2"]
    zero = torch.tensor(0.0, device=left.device)
    average_scale = torch.where(if_nomatching, zero, medium_information["x_scale1"] * medium_information["y_scale1"]).sum(1) / \
                        (torch.logical_not(if_nomatching).float().sum(1) + 1e-7)
    left_choice = average_scale >= 1.0
    new_left = torch.where(left_choice, left.permute(1, 2, 3, 0), right.permute(1, 2, 3, 0)).permute(3, 0, 1, 2)
    new_right = torch.where(left_choice, right.permute(1, 2, 3, 0), left.permute(1, 2, 3, 0)).permute(3, 0, 1, 2)
    medium_information["x_scale1"] = torch.where(left_choice, medium_information["x_scale1"].permute(1, 0), medium_information["x_scale2"].permute(1, 0)).permute(1, 0)
    medium_information["y_scale1"] = torch.where(left_choice, medium_information["y_scale1"].permute(1, 0), medium_information["y_scale2"].permute(1, 0)).permute(1, 0)
    medium_information["if_nomatching1"] = torch.where(left_choice, medium_information["if_nomatching1"].permute(1, 0), medium_information["if_nomatching2"].permute(1, 0)).permute(1, 0)
    medium_information["average_point1"] = torch.where(left_choice, medium_information["average_point1"].permute(1, 2, 0), medium_information["average_point2"].permute(1, 2, 0)).permute(2, 0, 1)
    new_desc_l = torch.where(left_choice, desc_l.permute(1, 2, 0), desc_r.permute(1, 2, 0)).permute(2, 0, 1)
    new_label = torch.where(left_choice, label.permute(1, 2, 0), label_reverse.permute(1, 2, 0)).permute(2, 0, 1)
    new_label_reverse = torch.where(left_choice, label_reverse.permute(1, 2, 0), label.permute(1, 2, 0)).permute(2, 0, 1)
    return new_left, new_right, left_choice, new_desc_l, new_label, new_label_reverse, medium_information

def Compute_matches(glue):
    matches = glue["matches"]
    k1 = glue["keypoints0"]
    k2 = glue["keypoints1"]
    pts1 = []
    pts2 = []
    for i in range(matches.shape[0]):
        if matches[i] != -1:
            pts1.append(k1[i])
            pts2.append(k2[matches[i]])
    return (np.array(pts1), np.array(pts2))

def Compute_matches_matrix(glue):
    pts1_list = []
    pts2_list = []
    for i in range(glue.shape[0]):
        matches = glue[i]["matches"]
        k1 = glue[i]["keypoints0"]
        k2 = glue[i]["keypoints1"]
        pts1 = []
        pts2 = []
        for j in range(matches.shape[0]):
            if matches[j] != -1:
                pts1.append(k1[j])
                pts2.append(k2[matches[j]])
        pts1_list.append(np.array(pts1))
        pts2_list.append(np.array(pts2))
    return (np.array(pts1_list), np.array(pts2_list))

def Get_pair_H_matrix(H_path, pair_path, glue_path, save_path, batch_size=1000):
    pair_list = Get_pairs(pair_path)
    H_list = np.load(H_path)
    N = batch_size // H_list.shape[1]
    for i in range(int((pair_list.shape[0] - 0.1) // batch_size + 1)):
        print(i)
        glue = np.load(glue_path + "matches.npz_" + str((i + 1) * batch_size-1) + ".npz", allow_pickle=True)["arr_0"]
        (pts1, pts2) = Compute_matches_matrix(glue)
        save_name = save_path + "Label_" + str(i) + "_"
        labels_list = {}
        for j in range(1000):
            labels = Compute_label_H_matrix(np.array(pts1[j]).reshape(1, pts1[j].shape[0], 2),
                                   np.array(pts2[j].reshape(1, pts2[j].shape[0], 2)),
                                   H_list[i * N + int(j//5), j % 5].reshape(-1, 3, 3),
                                   save_name, patch_size=5, step=1)
            labels_list[j] = labels
        np.save(save_name + "1", labels_list)

def Compute_label(depth0, depth1, glue, P, patch_size=5):
    # patch_size is the distance between the point and the board of patch
    (pts1, pts2) = Compute_matches(glue)
    N = 2 * patch_size + 1
    upper_bound = 1e7
    lower_bound = 1e-2
    labels = []
    for i in range(np.array(pts1).shape[0]):
        if_success = 1
        x = np.array(pts1[i]).astype(int)
        y = np.array(pts2[i]).astype(int)
        min_x1 = -10
        min_y1 = -10
        min_x2 = -10
        min_y2 = -10
        min = sys.maxsize
        M1 = np.identity(4)
        M1[0][2] = -x[0]
        M1[1][2] = -x[1]
        M2 = np.identity(4)
        M2[0][2] = -y[0]
        M2[1][2] = -y[1]
        P_new = M2.dot(P).dot(np.linalg.inv(M1))
        for j in np.array(range(N)) - patch_size:
            for k in np.array(range(N)) - patch_size:
                l1 = upper_bound
                d1 = depth0[x[1] + j, x[0] + k]
                if d1 < lower_bound:
                    d1 = upper_bound
                p1 = np.array([d1 * k, d1 * j, d1, 1.0])
                p21 = P_new.dot(p1)
                p21 = p21 / p21[2]
                # print(p21)
                if not (p21[0] < -patch_size or p21[0] > patch_size or p21[1] < -patch_size or p21[1] > patch_size):
                    d2 = depth1[int(p21[1] + y[1]), int(p21[0] + y[0])]
                    if d2 < lower_bound:
                        d2 = upper_bound
                    p2 = [p21[0] * d2, p21[1] * d2, d2, 1]
                    p12 = np.linalg.inv(P_new).dot(p2)
                    p12 = p12 / p12[2]
                    distance = p12 - p1 / d1
                    l1 = np.sqrt(np.power(distance[0], 2) + np.power(distance[1], 2))
                    # print(l1)
                l2 = np.sqrt(np.power(j, 2) + np.power(k, 2))
                l3 = np.sqrt(np.power(p21[0], 2) + np.power(p21[1], 2))
                l = 10 * l1 + (l2 + l3)
                if l < min:
                    min = l
                    min_y1 = j
                    min_x1 = k
                    min_x2 = int(p21[0])
                    min_y2 = int(p21[1])
        if min > 20:
            if_success = -1
        label = np.array([min_x1, min_y1, min_x2, min_y2, if_success, min])
        labels.append(label)
    return labels



def Compute_label_H(glue, H, path, patch_size=5):
    # patch_size is the distance between the point and the board of patch
    (pts1, pts2) = Compute_matches(glue)
    N = 2 * patch_size + 1
    upper_bound = 1e7
    labels = []
    for i in range(np.array(pts1).shape[0]):
        if_success = 1
        x = np.array(pts1[i]).astype(int)
        y = np.array(pts2[i]).astype(int)
        min_x1 = -10
        min_y1 = -10
        min_x2 = -10
        min_y2 = -10
        min = sys.maxsize
        M1 = np.identity(3)
        M1[0][2] = -x[0]
        M1[1][2] = -x[1]
        M2 = np.identity(4)
        M2[0][2] = -y[0]
        M2[1][2] = -y[1]
        H_new = M2.dot(H).dot(np.linalg.inv(M1))
        for j in np.array(range(N)) - patch_size:
            for k in np.array(range(N)) - patch_size:
                l1 = upper_bound
                p1 = np.array([k, j, 1.0])
                p21 = H_new.dot(p1)
                p21 = p21 / p21[2]
                if not (p21[0] < -patch_size or p21[0] > patch_size or p21[1] < -patch_size or p21[1] > patch_size):
                    p2 = [p21[0], p21[1], 1]
                    p12 = np.linalg.inv(H_new).dot(p2)
                    p12 = p12 / p12[2]
                    distance = p12 - p1
                    l1 = np.sqrt(np.power(distance[0], 2) + np.power(distance[1], 2))
                l2 = np.sqrt(np.power(j, 2) + np.power(k, 2))
                l3 = np.sqrt(np.power(p21[0], 2) + np.power(p21[1], 2))
                l = 10 * l1 + (l2 + l3)
                if l < min:
                    min = l
                    min_y1 = j
                    min_x1 = k
                    min_x2 = int(p21[0])
                    min_y2 = int(p21[1])
        if min > 10:
            if_success = -1
        label = np.array([min_x1, min_y1, min_x2, min_y2, if_success, min])
        labels.append(label)
    np.save(path+'homography_labels', labels)
    return labels

def Check_homography_output(H, path, img0_name, img1_name, sample = 10):
    pts1 = np.ones((sample, 3))
    H = np.linalg.inv(H)
    pts1[:, 0] = np.random.randint(-320, 320, sample) / 320
    pts1[:, 1] = np.random.randint(-240, 240, sample) / 240
    save_path = path + "test.png"
    img0 = cv2.resize(cv2.imread(path + img0_name), (640, 480))
    img1 = cv2.imread(path + img1_name)
    img2 = cv2.hconcat([img0, img1])
    pts2 = H.dot(pts1.transpose()).transpose()
    pts2[:, 0] = (pts2[:, 0] / pts2[:, 2])
    pts2[:, 1] = (pts2[:, 1] / pts2[:, 2])
    pts2[:, 0] = pts2[:, 0] * 320 + 960
    pts2[:, 1] = pts2[:, 1] * 240 + 240
    pts1[:, 0] = pts1[:, 0] * 320 + 320
    pts1[:, 1] = pts1[:, 1] * 240 + 240
    pts1 = pts1.astype(int)
    pts2 = pts2.astype(int)
    for i in range(sample):
        cv2.line(img2, (pts1[i, 0], pts1[i, 1]), (pts2[i, 0], pts2[i, 1]),
                 [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)])
    cv2.imwrite(save_path, img2)


def Compute_label_H_matrix(pts1, pts2, H, path, shape=[480, 640], patch_size=5, step=1):
    # patch_size is the distance between the point and the board of patch
    # compute new homography matrix(batch_size, key_point_num, rows, cols)
    if pts1.shape[1] == 0:
        labels = np.array([-1, -1, -1, -1, -1, -1]).reshape(1, 1, -1)
        return labels
    H[0] = np.linalg.inv(H[0])
    N = 2 * patch_size + 1
    M1 = np.eye(3).reshape(1, 3, 3).repeat(pts1.shape[0] * pts1.shape[1], axis=0).reshape(pts1.shape[0], pts1.shape[1], 3, 3)
    M2 = np.eye(3).reshape(1, 3, 3).repeat(pts1.shape[0] * pts1.shape[1], axis=0).reshape(pts1.shape[0], pts1.shape[1], 3, 3)
    pts1[0, :, 0] = pts1[0, :, 0] / shape[1] * 2 - 1
    pts1[0, :, 1] = pts1[0, :, 1] / shape[0] * 2 - 1
    pts2[0, :, 0] = pts2[0, :, 0] / shape[1] * 2 - 1
    pts2[0, :, 1] = pts2[0, :, 1] / shape[0] * 2 - 1
    M1[0, :, 0:2, 2] = pts1
    M2[0, :, 0:2, 2] = - pts2
    H_new = np.einsum('ijkl,ilp,ijpq->ijkq', M2, H, M1)
    # compute new pts1 matrix(batch_size, key_point_num, N*N, location(3))
    first = (np.arange(N) - patch_size) * step
    second = first.reshape(1, N).repeat(N, axis=0).reshape(1, N*N) / shape[1] * 2
    third = first.reshape(N, 1).repeat(N, axis=1).reshape(1, N*N) / shape[0] * 2
    fourth = np.ones((1, N*N))
    fifith = np.concatenate((second, third, fourth), axis=0).transpose().reshape(1, N*N, 3)
    pts_new = fifith.repeat(pts1.shape[0] * pts1.shape[1], axis=0).reshape(pts1.shape[0], pts1.shape[1], N*N, 3)
    # compute the linear mapping between pictures(batch_size, key_point_num, N*N, location(3))
    p21 = np.einsum('ijkl,ijpl->ijkp', pts_new, H_new)
    p21[:, :, :, 0] = p21[:, :, :, 0] / p21[:, :, :, 2]
    p21[:, :, :, 1] = p21[:, :, :, 1] / p21[:, :, :, 2]
    # compute loss(batch_size, key_point_num, N*N)
    loss = np.abs(pts_new[:, :, :, 0] * pts_new[:, :, :, 1]) + np.abs(p21[:, :, :, 0] * p21[:, :, :, 1])
    loss_pos = np.argmin(loss, axis=2).astype(int)
    loss_result = np.min(loss, axis=2)
    # compute output(batch_size, key_point_num, 6)
    result_p1 = np.zeros((pts1.shape[0], pts1.shape[1], 2))
    result_p1[:, :, 0] = loss_pos//N - patch_size
    result_p1[:, :, 1] = loss_pos - (result_p1[:, :, 0] + patch_size) * N - patch_size
    pts1[0, :, 0] = (pts1[0, :, 0] + 1) * shape[1] / 2
    pts1[0, :, 1] = (pts1[0, :, 1] + 1) * shape[0] / 2
    pts2[0, :, 0] = (pts2[0, :, 0] + 1) * shape[1] / 2
    pts2[0, :, 1] = (pts2[0, :, 1] + 1) * shape[0] / 2
    pts1 = result_p1 * step + pts1
    p21[:, :, :, 0] = p21[:, :, :, 0] * shape[1] / 2
    p21[:, :, :, 1] = p21[:, :, :, 1] * shape[0] / 2
    pts2 = p21[0, 0, loss_pos.data, 0:2] + pts2
    if_sucess = (loss_result < N * N * step * step / 5 / shape[0] / shape[0] * 2).astype(int)
    labels = np.concatenate((pts1, pts2, if_sucess.reshape(1, -1, 1), loss_result.reshape(1, -1, 1)), axis=2)
    return labels



def Create_P(K1, K2, P1, P2):
    k1 = np.identity(4)
    k1[0:3, 0:3] = K1
    k2 = np.identity(4)
    k2[0:3, 0:3] = K2
    p1 = np.identity(4)
    p1[0:3, 0:4] = P1
    p2 = np.identity(4)
    p2[0:3, 0:4] = P2
    P = k2.dot(p2).dot(np.linalg.inv(p1)).dot(np.linalg.inv(k1))
    return P

def Get_resize_ratio(shape_origin, shape):
    w = shape_origin[0].astype(float)
    h = shape_origin[1].astype(float)
    w_new = shape[0].astype(float)
    h_new = shape[1].astype(float)
    h_w = h_new / w_new
    ratio = -1
    add_num = [0, 0]
    if w / w_new < h / h_new:
        ratio = w_new / w
        add_num[1] = (h - w * h_w) / 2
    else:
        ratio = h_new / h
        add_num[0] = (w - h / h_w) / 2
    return ratio, np.array(add_num)

def Create_K_resize(K1, K2, shape_origin1, shape_origin2, shape=np.array([640, 480])):
    r1, add_num1 = Get_resize_ratio(shape_origin1, shape)
    r2, add_num2 = Get_resize_ratio(shape_origin2, shape)
    k1 = np.identity(4)
    k1[0:3, 0:3] = r1 * K1[0:3, 0:3]
    k1[2, 2] = 1
    k1[0:2, 2] -= add_num1 * r1
    k2 = np.identity(4)
    k2[0:3, 0:3] = r2 * K2[0:3, 0:3]
    k2[2, 2] = 1
    k2[0:2, 2] -= add_num2 * r2
    return k1, k2

def Create_P_resize(K1, K2, P1, P2, shape_origin1, shape_origin2, shape=np.array([640, 480])):
    k1, k2 = Create_K_resize(K1, K2, shape_origin1, shape_origin2, shape)
    p1 = np.identity(4)
    p1[0:3, 0:4] = P1
    p2 = np.identity(4)
    p2[0:3, 0:4] = P2
    P = k2.dot(p2).dot(np.linalg.inv(p1)).dot(np.linalg.inv(k1))
    return P

def Save_training_data(img0, img1, depth0, depth1, glue, P, name, patch_size=5):
    # matches = glue["matches"]
    # k1 = glue["keypoints0"]
    # k2 = glue["keypoints1"]
    # pts1 = []
    # pts2 = []
    # patches1 = []
    # patches2 = []
    # for i in range(matches.shape[0]):
    #     if (matches[i] != -1):
    #         pts1.append(k1[i])
    #         pts2.append(k2[matches[i]])
    # pts1 = np.array(pts1).astype(int)
    # pts2 = np.array(pts2).astype(int)
    # for i in range(pts1.shape[0]):
    #     patches1.append(img0[pts1[i][1] - patch_size:pts1[i][1] + patch_size + 1, pts1[i][0] - patch_size:pts1[i][0] + patch_size + 1].reshape(-1) + 1)
    #     patches2.append(img1[pts2[i][1] - patch_size:pts2[i][1] + patch_size + 1, pts2[i][0] - patch_size:pts2[i][0] + patch_size + 1].reshape(-1) + 1)
    labels = Compute_label(depth0, depth1, glue, P, patch_size)
    # patches1 = np.array(patches1)
    # patches2 = np.array(patches2)
    labels = np.array(labels)
    # data = np.concatenate((patches1, patches2, labels), axis=1)
    np.save(name, labels)


def test_label_data(name_list, patch_size=5):
    sum = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    N = 0
    for i in range(name_list.shape[0]):
        data = np.load(name_list[i] + ".npy")
        sum += np.mean(data[:, 2 * N:2 * N + 6], axis=0)
    sum = sum / name_list.shape[0]
    print("average trx1 is " + str(sum[0]))
    print("average try1 is " + str(sum[1]))
    print("average trx2 is " + str(sum[2]))
    print("average try2 is " + str(sum[3]))
    print("average loss is " + str(sum[5]))
    print("average accuracy is " + str(sum[4]))


def Get_params(file_name):
    f = open(file_name)
    line = f.readline()
    name_list = {}
    size_list = []
    K_list = []
    P_list = []
    i = 0
    while line:
        data = np.array(line.split())
        name_list[data[0]] = i
        size_list.append([data[1], data[2]])
        K = np.identity(3)
        K[0, 0] = data[3]
        K[1, 1] = data[4]
        K[0, 2] = data[5]
        K[1, 2] = data[6]
        K_list.append(K)
        P = np.zeros((3, 4))
        P[0:3, 0:4] = data[7:19].astype(np.float64).reshape(3, 4)
        P_list.append(P)
        i += 1
        line = f.readline()
    return (name_list, size_list, K_list, P_list)


def Get_pairs(file_name):
    f = open(file_name)
    line = f.readline()
    pair_list = []
    while line:
        data = np.array(line.split())
        pair_list.append(data)
        line = f.readline()
    return np.array(pair_list)

# def Get_coco_pairs(coco_path):
#     origin_path = coco_path + "train2017/"
#     transform_path1 = coco_path + "train2017_transform1/"
#     transform_path2 = coco_path + "train2017_transform2/"
#     transform_path3 = coco_path + "train2017_transform3/"
#     transform_path4 = coco_path + "train2017_transform4/"
#     transform_path5 = coco_path + "train2017_transform5/"
#     file_name1 = np.sort(np.array(os.listdir(origin_path)))
#     array = []
#     for i in range


def Get_all_pairs(path, use_set='train'):
    # get all of the pairs in the dataset and use two new parts to distinguish them
    # file_name_1 = np.sort(np.array(os.listdir(path)))
    all_pairs = np.array([])
    if use_set=="train":
        set_file = path + "megadepth_train_scenes.txt"
    if use_set=="test":
        set_file = path + "megadepth_validation_scenes_new.txt"
    if use_set=="test2":
        set_file = path + "megadepth_validation_scenes_old.txt"
    if use_set=="test_full":
        set_file = path + "megadepth_validation_scenes_full.txt"
    f = open(set_file, 'r')
    all_pairs = np.zeros([0, 4])
    F_list = np.zeros([0, 3, 3])
    lines = np.array(f.readlines())
    length = lines.shape[0]
    # if use_set=="train":
    #     length = 1
    for i in range(length):
    # for i in range(1):
        file_name_2 = np.sort(np.array(os.listdir(path + lines[i].strip('\n'))))
        for j in range(file_name_2.shape[0]):
            the_pair = Get_pairs(path + lines[i].strip('\n') + "/" + file_name_2[j] + "/pair_list.txt").reshape(-1, 2)
            the_F = np.load(path + lines[i].strip('\n') + "/" + file_name_2[j] + "/epipolar.npy").reshape(-1, 3, 3)
            type_name = np.array([lines[i].strip('\n') + "/" + file_name_2[j]]).reshape(1, 1).repeat(the_pair.shape[0], axis=0)
            sequence = np.arange(the_pair.shape[0]).reshape(-1, 1).astype(str)
            # print(type_name.shape, sequence.shape, the_pair.shape)
            the_pair = np.concatenate((type_name, sequence, the_pair), axis=1)
            all_pairs = np.concatenate([all_pairs, the_pair], axis=0)
            F_list = np.concatenate([F_list, the_F], axis=0)
    return np.array(all_pairs), np.array(F_list)

def Resize_depth(depth, shape=[640, 480]):
    w = depth.shape[1]
    h = depth.shape[0]
    w_new = shape[0]
    h_new = shape[1]
    if w/w_new < h/h_new:
        gap = int((h - w / w_new * h_new) / 2)
        crop_depth = depth[gap:h - gap, :]
    else:
        gap = int((w - h / h_new * w_new) / 2)
        crop_depth = depth[:, gap:w - gap]
    resize_depth = cv2.resize(
        crop_depth, tuple(shape), interpolation=cv2.INTER_LINEAR)
    return resize_depth


def Resize_img(depth, shape=np.array([640, 480])):
    w = depth.shape[1]
    h = depth.shape[0]
    w_new = shape[0]
    h_new = shape[1]
    if w/w_new < h/h_new:
        gap = int((h - w / w_new * h_new) / 2)
        crop_img = depth[gap:h - gap, :, :]
    else:
        gap = int((w - h / h_new * w_new) / 2)
        crop_img = depth[:, gap:w - gap, :]
    resize_img = cv2.resize(
        crop_img, tuple(shape))
    return resize_img

def Create_pair_list_H(path):
    data_path = path + "train2017/"
    f = open(path + "pair_list.txt", "w")
    transform_path = []
    transform_path.append("train2017/")
    transform_path.append("train2017_transform1/")
    transform_path.append("train2017_transform2/")
    transform_path.append("train2017_transform3/")
    transform_path.append("train2017_transform4/")
    transform_path.append("train2017_transform5/")
    for file_path in glob.glob(data_path + '/*.jpg'):
        filename = os.path.split(file_path)[1]
        for i in range(5):
            name = transform_path[0] + filename + " " + transform_path[i + 1] + filename + "\n"
            f.write(name)
    f.close()

def test_cropway(img_origin, img_warped):
    img_warped_my = Resize_img(img_origin)
    res = np.sum(np.abs(np.array(img_warped_my) - np.array(img_warped)))
    print("the residual of two align ways is "+str(res/3/img_warped.shape[0]/img_warped.shape[1]))


def Test_depth(param_path, pair_path, img_path, depth_path, glue_path, save_path):
    (name_list, size_list, K_list, P_list) = Get_params(param_path)
    pair_list = Get_pairs(pair_path)
    for i in range(pair_list.shape[0]):
        name1 = pair_list[i][0].split('.')[0]
        name2 = pair_list[i][1].split('.')[0]
        pair_name = name1 + "_" + name2 + "_matches.npz"
        glue = np.load(glue_path + pair_name)
        p = name_list[pair_list[i][0]]
        q = name_list[pair_list[i][1]]
        P = Create_P(K_list[p], K_list[q], P_list[p], P_list[q])
        img1 = cv2.imread(img_path + pair_list[i][0])
        img2 = cv2.imread(img_path + pair_list[i][1])
        depth1 = h5py.File(depth_path + name1 + ".h5", "r")['depth']
        depth2 = h5py.File(depth_path + name2 + ".h5", "r")['depth']
        depth1 = Resize_depth(depth1)
        depth2 = Resize_depth(depth2)
        save_name = save_path + name1 + "_" + name2
        Save_training_data(img1, img2, depth1, depth2, glue, P, save_name)
        test_label_data(np.array([save_name]))

# 为了应对不同分辨率的图片，这里需要正反两个positions和ranges
#bound的顺序分别是up， down， left， right
def Iterative_expand_matrix(scores_in, scalex, scaley, limitation, ranges, positions,
        lower_bound=1e-3, upper_bound=1e7, iter_num=15, width=20, height=15, type="distance"):
    height, width = positions.shape[0] // ranges.shape[0], ranges.shape[0]
    max0 = scores_in[:, :-1, :-1].max(2).indices
    opposite_nomatching_scores = scores_in[:, -1, :-1]
    last_nomatching_scores = torch.gather(opposite_nomatching_scores, 1, max0)
    opposite_nomatching_scores = opposite_nomatching_scores.reshape(scores_in.shape[0], 1, -1).repeat(1, scores_in.shape[1] - 1, 1)
    scores = scores_in[:, :-1, :]
    point0 = torch.zeros([scores.shape[0], scores.shape[1], 2], device=scores.device)
    # print(scale.reshape(15, 20))
    point0[:, :, 0] = max0 // limitation[3]
    point0[:, :, 1] = max0 % limitation[3]
    if_nomatching = (scores[:, :, :].max(2).indices == scores.shape[1])
    scale = scalex * scaley
    ruler = positions.reshape(1, 1, width * height, 2).repeat(scores.shape[0], scores.shape[1], 1, 1).permute(2, 0, 1, 3)
    bound = torch.zeros([scores.shape[0], scores.shape[1], 4], device=scores.device, dtype=torch.long)
    bound_difference = torch.zeros([scores.shape[0], scores.shape[1], 2], device=scores.device, dtype=torch.long)
    bound[:, :, 0:2] = point0[:, :, 0].reshape(point0.shape[0], point0.shape[1], 1).repeat(1, 1, 2)
    bound[:, :, 2:4] = point0[:, :, 1].reshape(point0.shape[0], point0.shape[1], 1).repeat(1, 1, 2)
    # bound[if_nomatching] = upper_bound
    b, m, n = scores.shape
    b, m2, n2 = scalex.shape
    zero = scores.new_tensor(1e-14, device=scores.device)
    bins0 = zero.expand(b, m, 1)
    bins1 = zero.expand(b, n - 1, 1)
    # 为了避免for循环，原始的scores中必须包括大量的0,而scale只有（b, 300, 300）的大小，比scores小得更多
    expand_scores = torch.cat([scores, bins0], dim=2)
    expand_scale = scale.reshape(-1, 1, scores.shape[2] - 1).repeat(1, scores.shape[1], 1)
    expand_scale = torch.cat([expand_scale, bins0, bins0], dim=2)
    opposite_nomatching_scores = torch.cat([opposite_nomatching_scores, bins0, bins0], dim=2)
    last_sum = torch.gather(expand_scores, 2, max0.reshape(max0.shape[0], -1, 1))
    last_scale = torch.gather(expand_scale, 2, max0.reshape(max0.shape[0], -1, 1))
    bound_change = torch.tensor([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], device=scores.device)
    # sequence_base = torch.zeros([scores.shape[0], scores.shape[1], 2, width], device=scores.device)
    for i in range(iter_num):
        sequence = torch.zeros([scores.shape[0], scores.shape[1], 4 * width], device=scores.device, dtype=torch.long)
        sequence_base = ranges[bound_difference.reshape(-1)]
        sequence_base = sequence_base.reshape(-1, scores.shape[1], 2, width)
        sequence[:, :, 0:width] = sequence_base[:, :, 1] + (bound[:, :, 2] + bound[:, :, 0] * width - width).unsqueeze(2)
        sequence[:, :, width:2*width] = sequence_base[:, :, 1] + (bound[:, :, 2] + bound[:, :, 1] * width + width).unsqueeze(2)
        sequence[:, :, 2*width:3*width] = sequence_base[:, :, 0] * width + (bound[:, :, 2] + bound[:, :, 0] * width - 1).unsqueeze(2)
        sequence[:, :, 3*width:4*width] = sequence_base[:, :, 0] * width + (bound[:, :, 3] + bound[:, :, 0] * width + 1).unsqueeze(2)
        sequence = torch.where(sequence >= 0, sequence, torch.tensor(width * height + 1, device=scores.device))
        sequence = torch.where(sequence <= width * height - 1, sequence, torch.tensor(width * height + 1, device=scores.device))
        expand_sum = torch.gather(expand_scores[:, :, :], 2, sequence).reshape(scores.shape[0], -1, 4, width)
        expand_nomatching_sum = torch.gather(opposite_nomatching_scores, 2, sequence).reshape(scores.shape[0], scores.shape[1], 4, width)
        expand_nomatching_sum = torch.where(expand_sum > lower_bound, expand_nomatching_sum, zero).sum(dim=3)
        expand_sum = expand_sum.sum(dim=3)
        expand_sum[:, :, 0] = torch.where(bound[:, :, 0]==0, zero, expand_sum[:, :, 0].float())
        expand_sum[:, :, 1] = torch.where(bound[:, :, 1]==height - 1, zero, expand_sum[:, :, 1].float())
        expand_sum[:, :, 2] = torch.where(bound[:, :, 2]==0, zero, expand_sum[:, :, 2].float())
        expand_sum[:, :, 3] = torch.where(bound[:, :, 3]==width - 1, zero, expand_sum[:, :, 3].float())
        expand_scale_sum = torch.gather(expand_scale[:, 0:scores.shape[1], :], 2, sequence).reshape(scores.shape[0], scores.shape[1], 4, width).sum(dim=3)
        max_sum, argmax_sum = torch.max(expand_sum, dim=2)
        max_scale_sum = torch.gather(expand_scale_sum, 2, argmax_sum.unsqueeze(2))
        max_nomatching_sum = torch.gather(expand_nomatching_sum, 2, argmax_sum.unsqueeze(2))
        bound = torch.where(max_sum.unsqueeze(2).repeat(1, 1, 4) > lower_bound, bound + bound_change[argmax_sum], bound)
        expand_sum = torch.where(max_sum > lower_bound, max_sum, zero)        
        expand_scale_sum = torch.where(max_sum.unsqueeze(2) > lower_bound, max_scale_sum, zero)
        expand_nomatching_sum = torch.where(max_sum.unsqueeze(2) > lower_bound, max_nomatching_sum, zero).squeeze(2)
        bound_difference[:, :, 0] = bound[:, :, 1] - bound[:, :, 0]
        bound_difference[:, :, 1] = bound[:, :, 3] - bound[:, :, 2]
        last_sum = last_sum + expand_sum.unsqueeze(2)
        last_scale = last_scale + expand_scale_sum
        last_nomatching_scores = last_nomatching_scores + expand_nomatching_sum
    if_core_exist = torch.logical_and(bound_difference[:, :, 0] > 1, bound_difference[:, :, 1] > 1)
    sequence = torch.zeros([scores.shape[0], scores.shape[1], 4 * width], device=scores.device, dtype=torch.long)
    sequence[:, :, 0:width] = sequence_base[:, :, 1] + (bound[:, :, 2] + bound[:, :, 0] * width).unsqueeze(2)
    sequence[:, :, width:2*width] = sequence_base[:, :, 1] + (bound[:, :, 2] + bound[:, :, 1] * width).unsqueeze(2)
    sequence[:, :, 2*width:3*width] = sequence_base[:, :, 0] * width + (bound[:, :, 2] + bound[:, :, 0] * width).unsqueeze(2)
    sequence[:, :, 3*width:4*width] = sequence_base[:, :, 0] * width + (bound[:, :, 3] + bound[:, :, 0] * width).unsqueeze(2)
    sequence = torch.where(sequence >= 0, sequence, torch.tensor(width * height + 1, device=scores.device))
    sequence = torch.where(sequence <= width * height - 1, sequence, torch.tensor(width * height + 1, device=scores.device))
    expand_sum = torch.gather(expand_scores[:, 0:scores.shape[1], :], 2, sequence).reshape(scores.shape[0], scores.shape[1], 4, width).sum(dim=3)
    expand_scale_sum = torch.gather(expand_scale, 2, sequence).reshape(scores.shape[0], scores.shape[1], 4, width).sum(dim=3)
    criterion1 = torch.logical_and(ruler[:, :, :, 0] >= bound[:, :, 0], ruler[:, :, :, 0] <= bound[:, :, 1]).permute(1, 2, 0)
    criterion2 = torch.logical_and(ruler[:, :, :, 1] >= bound[:, :, 2], ruler[:, :, :, 1] <= bound[:, :, 3]).permute(1, 2, 0)
    criterion = torch.logical_and(criterion1, criterion2)
    # print(criterion.float().sum()/scores.shape[0]/300.0)
    # 改了bug以后的写法
    other_expand_scores_x = torch.where(criterion, ((expand_scores[:, 0:, 0:-2] + 1e-7).sqrt().permute(1, 0, 2) / scalex.squeeze(2)).permute(1, 0, 2), zero)
    other_expand_scores_y = torch.where(criterion, ((expand_scores[:, 0:, 0:-2] + 1e-7).sqrt().permute(1, 0, 2) / scaley.squeeze(2)).permute(1, 0, 2), zero)
    # 有bug的写法
    # other_expand_scores_x = torch.where(criterion, ((expand_scores[:, 0:width * height, 0:width * height] + 1e-7).sqrt().permute(2, 0, 1) / scalex.squeeze()).permute(1, 2, 0), zero)
    # other_expand_scores_y = torch.where(criterion, ((expand_scores[:, 0:width * height, 0:width * height] + 1e-7).sqrt().permute(2, 0, 1) / scaley.squeeze()).permute(1, 2, 0), zero)
    weighted_point_x = torch.einsum('ijp,p->ij', other_expand_scores_x, positions[:, 1])
    weighted_point_y = torch.einsum('ijp,p->ij', other_expand_scores_y, positions[:, 0])
    point_weight_sum_x = other_expand_scores_x.sum(2)
    point_weight_sum_y = other_expand_scores_y.sum(2)
    average_point = torch.zeros([scale.shape[0], scores.shape[1], 2], device=scores.device)
    average_point[:, :, 1] = weighted_point_x / point_weight_sum_x
    average_point[:, :, 0] = weighted_point_y / point_weight_sum_y
    # print(torch.logical_and(torch.logical_or(average_point[:, :, 0] < bound[:, :, 0], average_point[:, :, 0] > bound[:, :, 1]), max0!=300).float().sum())
    # print(torch.logical_and(torch.logical_or(average_point[:, :, 1] < bound[:, :, 2], average_point[:, :, 1] > bound[:, :, 3]), max0!=300).float().sum())
    average_point = average_point + 0.5
    # test_y = torch.where(if_nomatching, zero, average_point[:, :, 0] - point0[:, :, 0]).sum()
    # test_x = torch.where(if_nomatching, zero, average_point[:, :, 1] - point0[:, :, 1]).sum()
    # print("test_y:", test_y/(max0 != 0).sum())
    # print("test_x:", test_x/(max0 != 0).sum())
    corner1 = (bound[:, :, 0] * width + bound[:, :, 2]).unsqueeze(dim=2)
    corner2 = (bound[:, :, 0] * width + bound[:, :, 3]).unsqueeze(dim=2)
    corner3 = (bound[:, :, 1] * width + bound[:, :, 2]).unsqueeze(dim=2)
    corner4 = (bound[:, :, 1] * width + bound[:, :, 3]).unsqueeze(dim=2)
    corner = torch.cat([corner1, corner2, corner3, corner4], dim=2)
    corner = torch.where(corner >= 0, corner, torch.tensor(width * height + 1, device=scores.device))
    corner = torch.where(corner <= width * height - 1, corner, torch.tensor(width * height + 1, device=scores.device))
    corner_point_sum = torch.gather(expand_scores, 2, corner).sum(dim=2)
    corner_scale_sum = torch.gather(expand_scale, 2, corner).sum(dim=2)
    the_scale = scores.sum(2)
    core_scale_sum = the_scale - expand_scale_sum.sum(dim=2) + corner_scale_sum
    core_sum = last_sum.squeeze() - expand_sum.sum(dim=2) + corner_point_sum
    core_cost = torch.where(torch.logical_and(if_core_exist, torch.logical_not(if_nomatching)), ((core_sum - core_scale_sum) / the_scale).abs(), zero)
    if_bound_board1 = torch.logical_or(bound[:, :, 0] == 0, bound[:, :, 2] == 0)
    if_bound_board2 = torch.logical_or(bound[:, :, 1] == height, bound[:, :, 2] == width)
    if_bound_board = torch.logical_or(if_bound_board1, if_bound_board2)
    last_sum = torch.where(if_bound_board, last_sum.squeeze(), last_sum.squeeze())
    whole_cost = torch.where(if_nomatching, zero, (((the_scale - last_sum).abs() + last_nomatching_scores / 4) / the_scale))
    x_scale, y_scale = Compute_scaling(scale, other_expand_scores_x, other_expand_scores_y, bound)
    return whole_cost, core_cost, average_point, x_scale, y_scale, bound


def origin_extract(left, patch_scale, width, height, if_swap=False, average_point=None):
    pixel_index = torch.arange(0, 3 * patch_scale, device=left.device).reshape(1, 3 * patch_scale).repeat(
        3 * patch_scale, 1) + torch.arange(0, 3 * patch_scale, device=left.device).reshape(3 * patch_scale, 1).repeat(1,
        3 * patch_scale) * torch.tensor(width * patch_scale + 2 * patch_scale, device=left.device)
    if if_swap:
        # 本来应该是-1.5,考虑padding改成0.5
        patch_index = ((average_point - 0.5) * patch_scale).long()
        patch_index = patch_index[:, :, 0] * (patch_scale * width + 2 * patch_scale) + patch_index[:, :, 1]
    else:
        patch_index = torch.arange(0, width, device=left.device).reshape(1, width).repeat(height, 1) * torch.tensor(
            patch_scale, device=left.device) + torch.arange(0, height, device=left.device).reshape(height, 1)\
            .repeat(1, width) * torch.tensor(patch_scale * (patch_scale * width + 2 * patch_scale), device=left.device)
        patch_index = patch_index.reshape(1, width, height).repeat(left.shape[0], 1, 1)
    patch_index = patch_index.reshape(left.shape[0], 1, width * height, 1).repeat(1, 3, 1, 9 * patch_scale * patch_scale)
    pixel_index = pixel_index.reshape(1, 1, 1, 9 * patch_scale * patch_scale).repeat(left.shape[0], 3, width * height, 1)
    left_index = (patch_index + pixel_index).reshape(left.shape[0], 3, -1)
    left = left.reshape(left.shape[0], 3, -1)
    new_left = torch.gather(left, 2, left_index).reshape(left.shape[0], 3, width * height, patch_scale * 3, patch_scale * 3)
    return new_left

#总之，先假设它全部都是内点
def Compute_scaling(scale, other_expand_scores_x, other_expand_scores_y, bound):
    # compute the average scale
    other_expand_scores = other_expand_scores_x * other_expand_scores_y
    weighted_scale = torch.einsum('ilr,ir->il', other_expand_scores, scale.squeeze(2))
    point_weight_sum = other_expand_scores.sum(2)
    average_scale = (weighted_scale / point_weight_sum).sqrt()
    # compute the weighted matrix
    bound_difference = torch.zeros([scale.shape[0], other_expand_scores.shape[1], 2], device=scale.device, dtype=torch.long)
    bound_difference[:, :, 0] = bound[:, :, 1] - bound[:, :, 0] + 1
    bound_difference[:, :, 1] = bound[:, :, 3] - bound[:, :, 2] + 1
    # range_x = other_expand_scores_x.sum(2) / bound_difference[:, :, 0]
    # range_y = other_expand_scores_y.sum(2) / bound_difference[:, :, 1]
    # ratio = (range_y / range_x).sqrt()
    ratio = 1.0
    x_scale = average_scale / ratio
    y_scale = average_scale * ratio
    # print(x_scale, y_scale)
    # print(x_scale[average_scale > 3])
    # print(y_scale[average_scale > 3])
    return 1.0 / x_scale, 1.0 / y_scale


def Compute_imgs(x_scale, y_scale, average_point, if_nomatching, left, right, sequence_num=0, output_path=None,
                 if_view=False, margin=128, width=20, height=15, patch_scale=32):
    if if_view:
        test_left = np.int16(left[0].cpu().numpy())
        test_right = np.int16(right[0].cpu().numpy())
        test_img = cv2.hconcat([test_left, test_right])
        cv2.imwrite(output_path + "_" + str(sequence_num) + "_" + "origin" + ".png", test_img)
    bound_new = torch.zeros([x_scale.shape[0], x_scale.shape[1], 4], device=x_scale.device)
    board = torch.tensor([0.0, float(patch_scale * height - 1), 0.0, float(patch_scale * width)], device=x_scale.device)
    right_use = F.pad(right, (0, 0, margin, margin, margin, margin), "constant", 0).permute(0, 3, 1, 2).float()
    left_use = F.pad(left, (0, 0, margin, margin, margin, margin), "constant", 0).permute(0, 3, 1, 2)

    resize_source = right_use
    average_new = average_point
    x_scale_new = x_scale
    y_scale_new = y_scale

    bound_new[:, :, 0] = (average_new[:, :, 0] - y_scale_new * 3.0 / 2.0) * float(patch_scale) + margin
    bound_new[:, :, 1] = (average_new[:, :, 0] + y_scale_new * 3.0 / 2.0) * float(patch_scale) + margin
    bound_new[:, :, 2] = (average_new[:, :, 1] - x_scale_new * 3.0 / 2.0) * float(patch_scale) + margin
    bound_new[:, :, 3] = (average_new[:, :, 1] + x_scale_new * 3.0 / 2.0) * float(patch_scale) + margin
    bound_new = torch.where(bound_new >= 0, bound_new, board[0])
    bound_new[:, :, 1] = torch.where(bound_new[:, :, 1] < patch_scale * height + 2 * margin, bound_new[:, :, 1], board[1])
    bound_new[:, :, 3] = torch.where(bound_new[:, :, 3] < patch_scale * width + 2 * margin, bound_new[:, :, 3], board[3])
    x_scale_new = (bound_new[:, :, 1] - bound_new[:, :, 0] + 1) / float(3 * patch_scale)
    y_scale_new = (bound_new[:, :, 3] - bound_new[:, :, 2] + 1) / float(3 * patch_scale)
    bound_new = bound_new.long()
    average_new = torch.zeros([x_scale.shape[0], x_scale.shape[1], 2], device=x_scale.device)
    average_new[:, :, 1] = (bound_new[:, :, 1] + bound_new[:, :, 0]).float() / 2.0 - margin + 0.5
    average_new[:, :, 0] = (bound_new[:, :, 2] + bound_new[:, :, 3]).float() / 2.0 - margin + 0.5
    # 这个10000是图像序列乘以的倍数，为了让图像序列和patch序列组合在一起
    initiator = torch.arange(0, width * height, 1, device=bound_new.device).reshape(1, width * height).repeat(if_nomatching.shape[0], 1) + \
                torch.arange(0, if_nomatching.shape[0] * 10000, 10000, device=bound_new.device).reshape(-1, 1)
    # scale 小于1可以维持现状，scale大于1的则需要resize left图像
    sequence = initiator[torch.logical_not(if_nomatching)].reshape(-1, 1)
    x_scale_new = x_scale_new.unsqueeze(2)
    y_scale_new = y_scale_new.unsqueeze(2)
    x_scale_new = torch.cat([x_scale_new, torch.ones(x_scale_new.shape, device=x_scale_new.device)], dim=2)
    y_scale_new = torch.cat([y_scale_new, torch.ones(y_scale_new.shape, device=y_scale_new.device)], dim=2)
    bound_new = torch.cat((bound_new[torch.logical_not(if_nomatching)], sequence), dim=1)
    new_left = origin_extract(left_use[:, :, margin - patch_scale:-(margin - patch_scale), margin - patch_scale:-(margin - patch_scale)], patch_scale, width, height)
    new_left = new_left.permute(0, 2, 3, 4, 1)[torch.logical_not(if_nomatching)]
    resize_result = tensor_resize.tensor_resize(resize_source, bound_new).permute(0, 2, 3, 1)
    new_right = resize_result
    if if_view:
        for i in range(new_right.shape[0]):
        # print(new_right.shape, new_left.shape)
            output_img = cv2.hconcat([np.int16(new_left[i].cpu().numpy()), np.int16(new_right[i].cpu().numpy())])
            cv2.imwrite(output_path + "_" + str(sequence_num) + "_" + str(i) + ".png", output_img)
    # cv2.imwrite(output_path + str(sequence_num) + str(1) + ".png", np.int16(new_right[0].permute(1, 2, 0).cpu().numpy()))
    return new_left, new_right, x_scale_new, y_scale_new, average_new

# def Compute_imgs(output_path, x_scale, y_scale, average_point, if_nomatching, left, right, sequence_num):
#     bound_new = torch.zeros([x_scale.shape[0], x_scale.shape[1], 4], device=x_scale.device)
#     bound_difference = torch.zeros([x_scale.shape[0], x_scale.shape[1], 2], device=x_scale.device)
#     cols = torch.range(0, 479, device=x_scale.device).reshape(480, 1).repeat(1, 640).reshape(-1)
#     rows = torch.range(0, 639, device=x_scale.device).reshape(1, 640).repeat(480, 1).reshape(-1)
#     positions = torch.zeros((640*480, 2), device=x_scale.device)
#     positions[:, 0] = cols
#     positions[:, 1] = rows
#     board = torch.tensor([0.0, 479.0, 0.0, 639.0], device=x_scale.device)
#     ruler = positions.reshape(1, 1, 640*480, 2).repeat(left.shape[0], 300, 1, 1).permute(2, 0, 1, 3)
#     bound_new[:, :, 0] = (average_point[:, :, 0] - y_scale / 2.0) * 32.0
#     bound_new[:, :, 1] = (average_point[:, :, 0] + y_scale / 2.0) * 32.0
#     bound_new[:, :, 2] = (average_point[:, :, 1] - x_scale / 2.0) * 32.0
#     bound_new[:, :, 3] = (average_point[:, :, 1] + x_scale / 2.0) * 32.0
#     bound_new = torch.where(bound_new >= 0, bound_new, board[0])
#     bound_new[:, :, 1] = torch.where(bound_new[:, :, 1] < 480, bound_new[:, :, 1], board[1])
#     bound_new[:, :, 3] = torch.where(bound_new[:, :, 3] < 640, bound_new[:, :, 3], board[3])
#     bound_new = bound_new.long()
#     bound_difference[:, :, 0] = bound_new[:, :, 1] - bound_new[:, :, 0] + 1
#     bound_difference[:, :, 1] = bound_new[:, :, 3] - bound_new[:, :, 2] + 1
#     bound_difference = bound_difference.long()
#     criterion1 = torch.logical_and(ruler[:, :, :, 0] >= bound_new[:, :, 0], ruler[:, :, :, 0] <= bound_new[:, :, 1]).permute(1, 2, 0)
#     criterion2 = torch.logical_and(ruler[:, :, :, 1] >= bound_new[:, :, 2], ruler[:, :, :, 1] <= bound_new[:, :, 3]).permute(1, 2, 0)
#     criterion = torch.logical_and(criterion1, criterion2).reshape(left.shape[0], 300, 480, 640)
#     new_right = torch.zeros((left.shape[0], 480, 640, 3), device=x_scale.device)
#     for i in range(left.shape[0]):
#         for j in range(300):
#             x = j % 20
#             y = j // 20
#             if if_nomatching[i, j]:
#                 new_right[i, 0:3, y * 32 + 0:32, x * 32 + 0:32] = 0
#             else:
#                 new_right[i, y * 32:y * 32 + 32, x * 32:x * 32 + 32, 0:3] = F.interpolate(right[i, :, :, :][criterion[i,
#                         j]].reshape(bound_difference[i, j, 0], bound_difference[i, j, 1], 3).unsqueeze(0).permute(0, 3, 1, 2), size=[32, 32]).permute(0, 2, 3, 1).squeeze()
#         new_right2 = np.int16(new_right[i].cpu().numpy())
#         left_old = np.int16(left[i].cpu().numpy())
#         right_old = np.int16(right[i].cpu().numpy())
#         output_img = cv2.hconcat([left_old, new_right2, right_old])
#         output_img_rgb = output_img[:, :, [2, 1, 0]]
#         cv2.imwrite(output_path + str(sequence_num) + str(i) + ".png", output_img_rgb)
#         print(output_path + str(sequence_num) + str(i) + ".png")


def loss_function_matches(scores, gt_matches, width, height):
    max0 = scores[:, :gt_matches.shape[1], :].max(2).indices
    gt_matches = gt_matches.to(scores.device).float()
    position = (gt_matches[:, :, 0] + gt_matches[:, :, 1] * width).reshape(-1, gt_matches.shape[1], 1).long()
    position = torch.where(position < -1e-7, torch.tensor(width * height, device=scores.device), position)
    position = torch.where(position > width * height - 0.5, torch.tensor(width * height, device=scores.device), position)
    x = torch.gather(scores[:, :gt_matches.shape[1], :], 2, position.long())
    x = x.reshape(-1, gt_matches.shape[1])
    a = -torch.where(torch.logical_and(gt_matches[:, :, 0] > -0.001, position.squeeze() != max0), x, torch.tensor(0.0, device="cuda"))
    loss_matches = a / ((gt_matches[:, :, 0] > -0.001).float().sum() + 1e-7)
    return loss_matches

def Continuous_loss(if_nomatching, width = 20, height = 15):
    coefficients = torch.tensor([-1.0, -1.0, -0.01, 1.0, 1.0], device=if_nomatching.device)
    filter = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], device=if_nomatching.device).reshape(1, 9, 1)
    pd = [1, 1, 1, 1]
    input = torch.nn.functional.pad(if_nomatching.float().reshape(if_nomatching.shape[0], 1, height, width), pd, "constant", 0.5)
    unfold = torch.nn.Unfold(kernel_size=3)
    unin = unfold(input).reshape(input.shape[0], 1, 9, height, width)
    output = torch.einsum('ijkmn,jkl->ilmn', unin, filter)
    return coefficients[output.long()].reshape(input.shape[0], height * width)

def Position_loss(label, average_point, scores, max, patch_scale, width, height, weight = 1):
    zero = torch.tensor(0.0, device=scores.device)
    label_use = torch.zeros([label.shape[0], label.shape[1], label.shape[2]])
    label_use[:, :, 0] = label[:, :, 1]
    label_use[:, :, 1] = label[:, :, 0]
    label_use = label_use.to(average_point.device).float() / patch_scale
    distance = (average_point - label_use).pow(2).sum(2)
    label_type = torch.where((label+1).abs() < 0.5, (zero - 2).long(), torch.floor(label / patch_scale).long())
    loss_matches = loss_function_matches(scores, label_type, width, height)
    loss_matches = torch.where(label_type[:, :, 0] + label_type[:, :, 1] * width == max, zero, loss_matches)
    loss_matches = torch.where(torch.logical_and((label_type[:, :, 0] - max//width)<1.5, (label_type[:, :, 1] - max%width)<1.5), zero, loss_matches)
    criterion = torch.logical_and((label_type[:, :, 0] - max % width).abs() <= 1,
                                   (label_type[:, :, 1] - max // width).abs() <= 1)
    criterion = torch.logical_and(torch.logical_and(label[:, :, 0] > -0.01, max != width * height), criterion)
    num_new_satisfy = criterion.float().sum()
    # position_loss = loss_matches
    average_error = (average_point - label_use).pow(2).sum(2).sqrt()
    distance_loss = torch.where(average_error < 1.0, distance * 2, distance)
    distance_loss = torch.where(average_error < 0.5, distance_loss * 2, distance_loss)
    distance_loss = torch.where(average_error < 0.25, distance_loss * 2, distance_loss)
    position_loss = loss_matches * weight + torch.where(criterion, (weight**2) * distance_loss, zero) / (num_new_satisfy + 1e-7)
    return position_loss, average_error, average_error

def Evaluate_loss(distance, eval_scores, label, width, height, weight_positive, weight_negtive, threshold=4, if_epipolar=True):
    if if_epipolar:
        eval_info = distance
    else:
        eval_info = distance.sqrt()
    c1 = torch.logical_and(eval_info < 1, label > 100)
    c2 = torch.logical_and(eval_info > threshold, label > 100)
    c3 = label < -100
    # num = c1.reshape(-1).shape[0]
    # print(c1.float().sum()/num, c2.float().sum()/num, c3.float().sum()/num, (eval_scores[:, :, 0]).float().sum()/num)
    eval_loss = (torch.where(c1, -torch.log(eval_scores[:, :, 0]).double(), 0.0)).sum() / (c1.float().sum() + 1e-7) + \
                (torch.where(c2, -torch.log(eval_scores[:, :, 1]).double(), 0.0)).sum() / (c2.float().sum() + 1e-7) + \
                (torch.where(c3, -torch.log(eval_scores[:, :, 2]).double(), 0.0)).sum() / (c3.float().sum() + 1e-7)
    return eval_loss

def Compute_epipolar_loss(F, input_point, output_point):
    sum = torch.einsum('ijk,ijkl,ijl->ij', output_point, F, input_point).abs()
    line1 = torch.einsum('ijkl,ijl->ijk', F, input_point)
    line2 = torch.einsum('ijk,ijkl->ijl', output_point, F)
    return sum**2/((line1[:, :, :2] ** 2).sum(2) + 1e-7), sum/((line1[:, :, :2] ** 2).sum(2).sqrt() + 1e-7)

#label[:, :, 0] decides whether this point can be loss, -100 is "no matching", 0 is "not known", 100 is "match exists"
#In Epipolar_loss, the distance is the same as the average_error
#the dimension of label is (b, width * height + 9, 1)
def Epipolar_loss(F_in, label, average_point, positions, whole_num, point_scale=8, if_shrink=False):
    zero = torch.tensor(0.0, device=average_point.device).double()
    input_point = positions.unsqueeze(0).repeat(label.shape[0], 1, 1) * point_scale + point_scale // 2
    input_point_z = torch.ones([input_point.shape[0], input_point.shape[1], 1], device=label.device)
    input_point = torch.cat([input_point[:, :, 1].unsqueeze(2), input_point[:, :, 0].unsqueeze(2), input_point_z], dim=2).double()
    output_point = torch.cat([average_point[:, :, 1].unsqueeze(2) * point_scale, average_point[:, :, 0].unsqueeze(2) * point_scale, input_point_z], dim=2).double()
    if if_shrink:
        input_point[:, :, 0:2] += 16
        output_point[:, :, 0:2] += 16
    F = F_in.reshape(label.shape[0], 1, 3, 3).repeat(1, input_point.shape[1], 1, 1)
    num_new_satisfy = (label >= -100).float().sum()
    sampson_error, distance = Compute_epipolar_loss(F, input_point, output_point)
    sampson_error = torch.where(sampson_error < zero + point_scale ** 2, sampson_error, zero + point_scale ** 2)
    sampson_error = torch.where(distance < zero + point_scale / 2, sampson_error * 2, sampson_error)
    sampson_error = torch.where(distance < zero + point_scale / 4, sampson_error * 2, sampson_error)
    distance_loss = torch.where(label >= -0.01, sampson_error, zero)/ (num_new_satisfy + 1e-8)
    # distance = torch.where(label[:, :whole_num, 0] >= 0, distance, zero)
    # distance_loss = torch.where(label[:, :whole_num, 0] >= 0, distance ** 2, zero)
    return distance_loss, distance, distance

def Compute_positions_and_ranges(height, width, device):
    cols = torch.arange(0, height).reshape(height, 1).repeat(1, width).reshape(width * height)
    rows = torch.arange(0, width).reshape(1, width).repeat(height, 1).reshape(width * height)
    positions = torch.zeros((width * height, 2), device=device)
    positions[:, 0] = cols
    positions[:, 1] = rows
    max_shape = max(height, width)
    ranges = torch.zeros([max_shape, max_shape], device=device)
    for i in range(max_shape):
        ranges[i] = F.pad(torch.arange(i + 1), [0, max_shape - 1 - i], "constant", 1e7)
    return positions, ranges


def Compute_loss(scores, scalex, scaley, max0, max1, label1, label2, ranges, positions, left, right,
                 weight=[1.0, 1.0, 10.0, 10.0, 1.0], reverse_set=True, width=20, height=15, patch_scale=32,
                 iter=15, refine_mode=False, eval_scores=None, loss_type='distance', if_refine=False, if_choose=True):
    zero = torch.tensor(0.0, device=scores.device)
    scalex = scalex.reshape(scalex.shape[0], -1, 1)
    scaley = scaley.reshape(scaley.shape[0], -1, 1)
    max0 = max0[:, :-1]
    max1 = max1[:, :-1]
    limitation2 = torch.tensor([0, left.shape[1] // patch_scale, 0, left.shape[2] // patch_scale], device=scores.device)
    limitation1 = torch.tensor([0, right.shape[1] // patch_scale, 0, right.shape[2] // patch_scale], device=scores.device)
    if_nomatching1 = (max0 == (right.shape[1] // patch_scale) * (right.shape[2] // patch_scale))
    if_nomatching2 = (max1 == (left.shape[1] // patch_scale) * (left.shape[2] // patch_scale))
    num_non1 = (label1[:, :(left.shape[1] // patch_scale) * (left.shape[2] // patch_scale), 0] < -100).float().sum()
    num_non2 = (label2[:, :(right.shape[1] // patch_scale) * (right.shape[2] // patch_scale), 0] < -100).float().sum()
    num_satisfy1 = (max0 != (left.shape[1] // patch_scale) * (left.shape[2] // patch_scale)).float().sum()
    num_satisfy2 = (max1 != (right.shape[1] // patch_scale) * (right.shape[2] // patch_scale)).float().sum()
    positions1, ranges1 = Compute_positions_and_ranges(right.shape[1] // patch_scale, right.shape[2] // patch_scale, scores.device)
    positions2, ranges2 = Compute_positions_and_ranges(left.shape[1] // patch_scale, left.shape[2] // patch_scale, scores.device)
    if if_refine:
        weight_p = 25.0
    else:
        weight_p = 5.0
    if loss_type == 'distance':
        width = right.shape[2] // patch_scale
        height = right.shape[1] // patch_scale  
        weight_positive = 100 / ((label1[:, :, 0] > -0.01).float().sum(1) + 5)
        weight_negtive = 100 / ((label1[:, :, 0] < -100).float().sum(1) + 5)
        whole_cost1, core_cost1, average_point1, x_scale, y_scale, bound1 = \
            Iterative_expand_matrix(scores.exp(), scalex, scaley, limitation1,
                ranges1, positions1, left, right, if_view=False, height=height, width=width, iter=iter, lower_bound=1e-5)

        trust_score = whole_cost1.clone()
        nomatching_loss1 = - torch.where(torch.logical_and(torch.logical_not(if_nomatching1), 
            label1[:, :scores.shape[1] - 1, 0] < -100), scores[:, :-1, -1], zero).sum(1) * weight_negtive
        # mismatching_loss1 = Continuous_loss(if_nomatching1, width, height).multiply(scores[:, :-1, -1])
        position_loss1, distance1, average_error1 = Position_loss(label1[:, :scores.shape[1] - 1, :2], average_point1, scores, max0, patch_scale, width, height, weight=weight_p)
        label1_epipolar = torch.cat([label1[:, :scores.shape[1] - 1, 0].unsqueeze(2), label1[:, -3:, :].reshape(-1, 9, 1)], dim=1)
        epipolar_loss1, distance1_epipolar, _ = Epipolar_loss(label1_epipolar[:, -9:, 0], label1_epipolar[:, :width*height, 0], average_point1, positions2, scores.shape[1] -1, point_scale=32)
        position_loss1 = epipolar_loss1.sum(1) * 0 + position_loss1.sum(1) * weight_positive
        mismatching_loss1 = torch.where(torch.logical_and(if_nomatching1, label1[:, :scores.shape[1] - 1, 0] > -0.01), scores[:, :-1, -1], zero).sum(1) * 5
        # whole_cost1 = torch.where(label1[:, :scores.shape[1] - 1, 0] > -0.01, whole_cost1, zero)
        if if_choose:
            criterion = torch.logical_and(torch.logical_and(distance1_epipolar > 32, label1[:, :scores.shape[1] - 1, 0] > -100), 
                torch.logical_and(label1[:, :scores.shape[1] - 1, 0] < -0.01, torch.logical_not(if_nomatching1)))
            mismatching_loss1 += torch.where(torch.logical_and(torch.logical_and(if_nomatching1, label1[:, :scores.shape[1] - 1, 0] > -100), 
                torch.logical_and(distance1_epipolar < 16, label1[:, :scores.shape[1] - 1, 0] < -0.01)), scores[:, :-1, -1], zero).sum(1) * 0.5
            nomatching_loss1 += - torch.where(criterion, scores[:, :-1, -1], zero).sum(1) * 5
            mismatching_loss1 *= 0.1
            nomatching_loss1 *= 0.1
            position_loss1 *= 10
        whole_cost1 = torch.where(torch.logical_and(label1[:, :scores.shape[1] - 1, 0] > -0.01, distance1 < 1.0), whole_cost1, zero)
        core_cost1 = torch.where(torch.logical_and(distance1_epipolar < 16, label1[:, :scores.shape[1] - 1, 0] > -100), whole_cost1, zero)
    if loss_type == 'epipolar':
        weight_positive = 50 / ((label1[:, :width * height, 0] > -0.01).float().sum(1) + 5)
        weight_negtive = 1000 / ((label1[:, :width * height, 0] < -100).float().sum(1) + 5)
        weight_positive2 = 50 / ((label1[:, :width * height, 2] > 100).float().sum(1) + 5)
        whole_cost1, core_cost1, average_point1, x_scale, y_scale, bound1 = \
            Iterative_expand_matrix(scores.exp(), scalex, scaley, limitation1,
                ranges1, positions1, left, right, if_view=False, height=height, width=width, iter=iter, lower_bound=1e-3)
        position_loss1, _, average_error1 = Position_loss(label1[:, :scores.shape[1] - 1, :2], average_point1, scores, max0, patch_scale, width, height, weight=patch_scale)
        epipolar_loss1, distance1, _ = Epipolar_loss(label1[:, -3:, :], label1[:, :width*height, 2], average_point1, positions2, width*height, point_scale=patch_scale)
        # position_loss1 = torch.where(torch.logical_not(if_nomatching1), position_loss1, zero)
        nomatching_loss1 = -torch.where(torch.logical_and(torch.logical_not(if_nomatching1), label1[:, :width * height, 0] < -100), scores[:, :-1, -1], zero).sum(1) * weight_negtive
        # criterion = torch.logical_and(average_error1 > 1.0, label1[:, :width*height, 0] < -0.01)
        # nomatching_loss1 += -torch.where(criterion, 1.1 * scores[:, :-1, -1], zero).sum(1)
        # mismatching_loss1 = torch.where(torch.logical_and(if_nomatching1, label1[:, :width * height, 0] > 2), scores[:, :-1, -1], zero).sum(1)
        mismatching_loss1 = torch.where(torch.logical_and(if_nomatching1, label1[:, :width * height, 0] > -0.01), scores[:, :-1, -1], zero).sum(1) * weight_positive
        trust_score = whole_cost1.clone()
        if if_choose:
            position_loss1 = torch.where(whole_cost1 < 1.0, position_loss1, zero)
            epipolar_loss1 = torch.where(whole_cost1 < 1.0, epipolar_loss1, zero.double())
        whole_cost1 = torch.where(torch.logical_and(label1[:, :width*height, 0] > -0.01, average_error1 < 2.0), whole_cost1, zero)
        core_cost1 = torch.where(torch.logical_and(distance1 < 8.0, label1[:, :width*height, 0] > -100), whole_cost1, zero) 
        position_loss1 = position_loss1.sum(1) * weight_positive * 2 + epipolar_loss1.sum(1) * weight_positive2 * 2
        # position_loss1 = position_loss1 * eval_scores[0]
        # position_loss1 = torch.where(torch.logical_not(if_nomatching1), position_loss1.double(), zero.double())
    if eval_scores == None:
        eval_loss1 = 0
    else:
        # eval_loss1 = Evaluate_loss(distance1, eval_scores, label1[:, :width * height, 0], width, height, weight_positive, weight_negtive)
        eval_loss1 = 0
    # position_loss1 = loss_matches1
    # 将不满足条件的第一步骤的匹配结果也视为无匹配
    # if if_refine == False:
    #     criterion = torch.logical_or(eval_scores[:, :, 0] <= eval_scores[:, :, 1], eval_scores[:, :, 0] <= eval_scores[:, :, 2])
        # if_nomatching1 = torch.logical_or(criterion, if_nomatching1)
    Medium_information = {
        'average_point1': average_point1,
        'if_nomatching1': if_nomatching1,
        'if_nomatching2': if_nomatching2,
        'average_error1': average_error1,
        'trust_score': trust_score,
        'x_scale1': x_scale,
        'y_scale1': y_scale,
        "bound1": bound1
    }
    #这里的refine_model指的是只保留一个patch进行loss,一般用不到这个设置
    if refine_mode:
        if_nomatching1_refine = torch.ones(if_nomatching1.shape, device=left.device).bool()
        if_matching1 = torch.logical_not(if_nomatching1)
        set1 = torch.arange(0, 300, 1, device = left.device)
        for i in range(left.shape[0]):
            set2 = set1[if_matching1[i]]
            num = if_matching1[i].long().sum() - 1
            if num == -1:
                continue
            if num == 0:
                n = 0
            else:
                n = torch.randint(num, size=[1])
            if_nomatching1_refine[i, set2[n]] = False
        Medium_information['if_nomatching1'] = if_nomatching1_refine
    if reverse_set:
        reverse_scores = scores.permute(0, 2, 1)
        reverse_scale = torch.ones([scalex.shape[0], scores.shape[1] - 1, 1], device=scores.device)
        if loss_type == 'distance':
            width = left.shape[2] // patch_scale
            height = left.shape[1] // patch_scale  
            weight_positive_reverse = 100 / ((label2[:, :, 0] > -0.01).float().sum(1) + 5)
            weight_negtive_reverse = 100 / ((label2[:, :, 0] < -100).float().sum(1) + 5)
            whole_cost2, core_cost2, average_point2, x_scale2, y_scale2, bound2 = Iterative_expand_matrix(reverse_scores.exp(), reverse_scale, 
                reverse_scale, limitation2, ranges2, positions2, left, right, if_view=False, height=height, width=width, iter=iter, lower_bound=1e-5)
            nomatching_loss2 = -torch.where(torch.logical_and(torch.logical_not(if_nomatching2), label2[:, :scores.shape[2] - 1, 0] < -100), 
                reverse_scores[:, :-1, -1], zero).sum(1) * weight_negtive_reverse
            # mismatching_loss2 = Continuous_loss(if_nomatching2, width, height).multiply(reverse_scores[:, :-1, -1])
            position_loss2, distance2, average_error2 = Position_loss(label2[:, :scores.shape[2] - 1, :2], average_point2, reverse_scores, max1, patch_scale, width, height, weight=weight_p)
            label2_epipolar = torch.cat([label2[:, :scores.shape[2] - 1, 0].unsqueeze(2), label2[:, scores.shape[2] - 1:, :].reshape(-1, 9, 1)], dim=1)
            epipolar_loss2, distance2_epipolar, _ = Epipolar_loss(label2_epipolar[:, -9:, 0], label2_epipolar[:, :width*height, 0], average_point2, positions1, scores.shape[2] -1, point_scale=32)
            position_loss2 = epipolar_loss2.sum(1) * 0 + position_loss2.sum(1) * weight_positive_reverse
            mismatching_loss2 = torch.where(torch.logical_and(if_nomatching2, label2[:, :scores.shape[2] - 1, 0] > -0.01), reverse_scores[:, :-1, -1], zero).sum(1) * 5
            if if_choose:
                criterion = torch.logical_and(torch.logical_and(distance2_epipolar > 32, label2[:, :scores.shape[2] - 1, 0] > -100), 
                    torch.logical_and(label2[:, :scores.shape[2] - 1, 0] < -0.01, torch.logical_not(if_nomatching2)))
                mismatching_loss2 += torch.where(torch.logical_and(torch.logical_and(if_nomatching2, label2[:, :scores.shape[2] - 1, 0] > -100), 
                    torch.logical_and(distance2_epipolar < 16, label2[:, :scores.shape[2] - 1, 0] < -0.01)), reverse_scores[:, :-1, -1], zero).sum(1) * 0.5
                nomatching_loss2 += - torch.where(criterion, reverse_scores[:, :-1, -1], zero).sum(1) * 5
                mismatching_loss2 *= 0.1
                nomatching_loss2 *= 0.1
                position_loss2 *= 10
            # whole_cost2 = torch.where(label2[:, :scores.shape[2] - 1, 0] > -0.01, whole_cost2, zero)
            whole_cost2 = torch.where(torch.logical_and(label2[:, :scores.shape[2] - 1, 0] > -0.01, distance2 < 1.0), whole_cost2, zero)
            core_cost2 = torch.where(torch.logical_and(distance2_epipolar < 16, label2[:, :scores.shape[2] - 1, 0] > -100), core_cost2, zero)
            # mismatching_loss2 += torch.where(torch.logical_and(torch.logical_and(distance2_epipolar < 16, if_nomatching2), label2[:, :width*height, 0] > -100), reverse_scores[:, :-1, -1], zero).sum(1)
        if loss_type == 'epipolar':
            weight_positive_reverse = 50 / ((label2[:, :width * height, 0] > 100).float().sum(1) + 5)
            weight_negtive_reverse = 50 / ((label2[:, :width * height, 0] < -100).float().sum(1) + 5)
            whole_cost2, core_cost2, average_point2, x_scale2, y_scale2, bound2 = Iterative_expand_matrix(reverse_scores.exp(),
                reverse_scale, reverse_scale, limitation2, ranges2, positions2, left, right, if_view=False, height=height, width=width, iter=iter, lower_bound=1e-3)
            position_loss2, distance2, average_error2 = Epipolar_loss(label2[:, -9:, 0], label2[:, :width*height, 0], average_point2, positions1, width*height, point_scale=patch_scale)
            if if_choose:
                position_loss2 = torch.where(whole_cost2 < 1.0, position_loss2, zero.double())
            # position_loss2 = torch.where(torch.logical_not(if_nomatching2), position_loss2, zero)
            nomatching_loss2 = -torch.where(torch.logical_and(torch.logical_not(if_nomatching2), label2[:, :width*height, 0] < -10), reverse_scores[:, :-1, -1], zero).sum(1) * weight_negtive_reverse
            criterion = torch.logical_and(distance2 > 8, label2[:, :width*height, 0] > 10)
            nomatching_loss2 += -torch.where(criterion, 1.1 * reverse_scores[:, :-1, -1], zero).sum(1)
            mismatching_loss2 = torch.where(torch.logical_and(if_nomatching2, label2[:, :width*height, 0] > 10), reverse_scores[:, :-1, -1], zero).sum(1) * weight_positive_reverse
            whole_cost2 = torch.where(torch.logical_and(label2[:, :width*height, 0] > 10, distance2 < 4), whole_cost2, zero)
            core_cost2 = torch.where(torch.logical_and(distance2 < 4, label2[:, :width*height, 0] > 10), core_cost2, zero)
            position_loss2 = position_loss2.sum(1) * weight_positive_reverse * 2
            # position_loss2 = position_loss2 * eval_scores[1]
            # position_loss2 = torch.where(torch.logical_not(if_nomatching2), position_loss2.double(), zero.double())
        Medium_information['average_error2'] = average_error2
        if loss_type=='distance':
            weight_whole_loss = 1.0
        else:
            weight_whole_loss = 0.0
        whole_cost = (whole_cost1.sum() + whole_cost2.sum() * weight_whole_loss) / (num_satisfy1 + num_satisfy2 * weight_whole_loss + 1e-7)
        core_cost = (core_cost1.sum() + core_cost2.sum()) / (num_satisfy1 + num_satisfy2 + 1e-7)
        mismatching_loss = (mismatching_loss1.sum() + mismatching_loss2.sum() * 0) / (1 * width * height * scores.shape[0] + 1e-3)
        nomatching_loss = (nomatching_loss1.sum() + nomatching_loss2.sum() * 0.5) / (num_non1 + num_non2 * 0.5 + 10)
        position_loss = (position_loss1.sum() + position_loss2.sum())
        Medium_information['average_point2'] = average_point2
        Medium_information['x_scale2'] = scalex[:, :, 0]
        Medium_information['y_scale2'] = scaley[:, :, 0]
    else:
        whole_cost = (whole_cost1).sum()/(num_satisfy1 + 1e-7)
        core_cost = (core_cost1).sum()/(num_satisfy1 + 1e-7)
        mismatching_loss = (mismatching_loss1).sum()/(width * height * scores.shape[0] + 1e-7)
        nomatching_loss = (nomatching_loss1).sum()/(num_non1 + 1e-7)
        position_loss = position_loss1.sum()
    Loss = {
        'whole': whole_cost,
        'core': core_cost,
        'mismatching': mismatching_loss,
        'nomatching': nomatching_loss,
        'position': position_loss,
        'eval_loss': eval_loss1
    }
    if refine_mode:
        if_nomatching1_refine = torch.ones(if_nomatching1.shape, device=left.device).bool()
        if_nomatching2_refine = torch.ones(if_nomatching2.shape, device=left.device).bool()
        if_matching1 = torch.logical_not(if_nomatching1)
        if_matching2 = torch.logical_not(if_nomatching2)
        set1 = torch.arange(0, 300, 1, device = left.device)
        for i in range(left.shape[0]):
            set2 = set1[if_matching1[i]]
            num = if_matching1[i].long().sum() - 1
            if num == -1:
                continue
            if num == 0:
                n = torch.tensor([0])
            else:
                if num < 12:
                    n = torch.randint(num, size=[num + 1])
                else:
                    n = torch.randint(num, size=[12])
            if_nomatching1_refine[i, set2[n]] = False
        for i in range(left.shape[0]):
            set2 = set1[if_matching2[i]]
            num = if_matching2[i].long().sum() - 1
            if num == -1:
                continue
            if num == 0:
                n = torch.tensor([0])
            else: 
                if num < 12:
                    n = torch.randint(num, size=[num + 1])
                else:
                    n = torch.randint(num, size=[12])
            if_nomatching2_refine[i, set2[n]] = False
        Medium_information['if_nomatching1'] = if_nomatching1_refine
        Medium_information['if_nomatching2'] = if_nomatching2_refine
    return (Loss, Medium_information)

def sample_function_construct(device):
    sample = torch.zeros(1, 301, 301, device=device)
    # sample_sequence = torch.zeros(1, 301)
    # sample_sequence[0] = torch.randperm(301, device=device)
    scale = torch.ones(1, 300, 1, device=device)
    ranges = torch.zeros([20, 20], device=device)
    limitation = torch.tensor([0, 15, 0, 20], device=device)
    for i in range(20):
        ranges[i] = F.pad(torch.arange(i + 1), [0, 19 - i], "constant", 1e7)
    for i in range(15):
        for j in range(20):
            sample[0, i * 20 + j, i * 20 + j] = 0.2
            if j < 18:
                sample[0, i * 20 + j, i * 20 + j + 1] = 0.2
                sample[0, i * 20 + j, i * 20 + j + 2] = 0.2
            else:
                sample[0, i * 20 + j, 300] += 0.4
            if i < 13:
                sample[0, i * 20 + j, (i + 1) * 20 + j] = 0.2
                sample[0, i * 20 + j, (i + 2) * 20 + j] = 0.2
            else:
                sample[0, i * 20 + j, 300] += 0.4
            if i >= 13 or j >= 18:
                sample[0, i * 20 + j, 300] += 0.8
            else:
                sample[0, i * 20 + j, (i + 1) * 20 + j + 1] = 0.2
                sample[0, i * 20 + j, (i + 2) * 20 + j + 2] = 0.2
                sample[0, i * 20 + j, (i + 1) * 20 + j + 2] = 0.2
                sample[0, i * 20 + j, (i + 2) * 20 + j + 1] = 0.2
    cols = torch.range(0, 14, device=device).reshape(15, 1).repeat(1, 20).reshape(300)
    rows = torch.range(0, 19, device=device).reshape(1, 20).repeat(15, 1).reshape(300)
    positions = torch.zeros((300, 2), device=device)
    positions[:, 0] = cols
    positions[:, 1] = rows
    max0, max1 = sample.max(2).indices.long(), sample.max(1).indices.long()
    whole_cost1, core_cost1, average_point1 = Iterative_expand_matrix(sample[:, :-1, :], max0[:, :-1], scale, limitation, ranges, positions)
    print(whole_cost1)
    print(core_cost1)
    print(average_point1)

def CAPS_struct_change(caps_path):
    file_name_1 = np.array(os.listdir(caps_path))
    for i in np.array(range(file_name_1.shape[0])):
        prefix = caps_path + file_name_1[i]
        if os.path.isdir(prefix):
            if os.path.isdir(prefix + "/dense"):
                os.rename(prefix + "/dense", prefix + "/dense0")
            file_name_2 = np.array(os.listdir(prefix))
            for j in np.array(range(file_name_2.shape[0])):
                prefix = caps_path + file_name_1[i] + "/" + file_name_2[j]
                if os.path.isdir(prefix + "/aligned/images"):
                    file_name_3 = np.array(os.listdir(prefix + "/aligned/images"))
                    for k in range(file_name_3.shape[0]):
                        os.rename(prefix + "/aligned/images/" + file_name_3[k], caps_path+file_name_1[i]+"/"+file_name_2[j]+"/"+file_name_3[k])
                    if os.path.isfile(os.path.join(prefix, "aligned/img_cam.txt")):
                        os.rename(os.path.join(prefix, "aligned/img_cam.txt"),
                                          os.path.join(prefix, "img_cam.txt"))
                    if os.path.isfile(os.path.join(prefix, "aligned/pairs.txt")):
                        os.rename(os.path.join(prefix, "aligned/pairs.txt"),
                                          os.path.join(prefix, "pair_list.txt"))
                    if os.path.isdir(prefix + "/aligned"):
                        shutil.rmtree(prefix + "/aligned")


def CAPS_cut(caps_path, depth_path):
    file_name_1 = np.array(os.listdir(caps_path))
    print(file_name_1.shape[0])
    for i in np.array(range(file_name_1.shape[0])):
        # if file_name_1[i] != "0185":
        #     continue
        print(i, file_name_1[i])
        prefix = caps_path + file_name_1[i]
        if os.path.isdir(prefix):
            file_name_2 = np.array(os.listdir(prefix))
            for j in np.array(range(file_name_2.shape[0])):
                prefix = caps_path + file_name_1[i] + "/" + file_name_2[j]
                if os.path.isdir(prefix):
                    (name_list, size_list, K_list, P_list) = Get_params(prefix + "/img_cam.txt")
                    if os.path.exists(prefix + "/pairs.txt"):
                        pair_list = Get_pairs(prefix + "/pairs.txt")
                    else:
                        pair_list = Get_pairs(prefix + "/pair_list.txt")
                    f = open(prefix + "/pair_list_new.txt", "w")
                    for k in range(np.array(pair_list).shape[0]):
                        name1 = pair_list[k][1].split('.')[0]
                        name2 = pair_list[k][0].split('.')[0]
                        depth_prefix = depth_path + file_name_1[i] + "/" + file_name_2[j] + "/depths/"
                        if os.path.exists(depth_prefix + name1 + ".h5") and os.path.exists(depth_prefix + name2 + ".h5"):
                            f.write(str(pair_list[k][0]) + " " + str(pair_list[k][1]) + "\n")
                    f.close()

#这样操作不合适，因为affine_grid会自动进行一轮缩放，把数据彻底搞乱掉。得想想办法制止缩放。而平移这个是实际百分比的一半，不能按照实际百分比来算的。
def transform_input(right, label):
    angle = 2 * (torch.rand([1]) - 1) * math.pi
    # angle = 0
    x = (torch.rand([1]) * 32 - 16) / 320.0
    y = (torch.rand([1]) * 32 - 16) / 240.0
    x = 0
    y = 0
    theta = torch.tensor([[math.cos(angle), 3/4*math.sin(-angle) , x],
                      [4/3*math.sin(angle), math.cos(angle)  , y]], device=right.device)
    theta2 = torch.tensor([[math.cos(-angle), math.sin(angle), - x * 320],
                           [math.sin(-angle),  math.cos(-angle), - y * 240],
                           [              0,                0, 1]], device=right.device, dtype=torch.double)
    grid = F.affine_grid(theta.unsqueeze(0), right.permute(0, 3, 1, 2).size(), align_corners=False)
    output = F.grid_sample(right.float().permute(0, 3, 1, 2), grid, align_corners=False)
    new_right = output.permute(0, 2, 3, 1)
    offset = torch.tensor([320, 240, 1], device=right.device)
    label_z = torch.ones(label.shape[0:2], device=right.device).unsqueeze(2)
    new_label = torch.cat([label, label_z], dim=2)
    new_label[:, :, 0:2] = (new_label[:, :, 0:2] - offset[0:2])
    new_label = torch.einsum('ij,tkj->tki', theta2, new_label)
    new_label[:, :, 0:2] = new_label[:, :, 0:2] + offset[0:2]
    new_label[:, :, 0:2] = torch.where(torch.logical_or(new_label[:, :, 0:2] < 0, new_label[:, :, 0:2] > 639), torch.tensor(-1e3, device=right.device).double(), new_label[:, :, 0:2])
    new_label[:, :, 1] = torch.where(new_label[:, :, 1] > 479, torch.tensor(-1e3, device=right.device).double(), new_label[:, :, 1])
    new_label[:, :, 0] = torch.where(label[:, :, 0] > -1e-4, new_label[:, :, 0], label[:, :, 0])
    new_label[:, :, 1] = torch.where(label[:, :, 0] > -1e-4, new_label[:, :, 1], label[:, :, 1])
    return (new_right, new_label[:, :, 0:2])

def Compute_covariance_matrix(scale0, scale1, if_matching1, if_matching2, label, height, width):
    depth_ratio = label[:, :, 2]
    depth_ratio = depth_ratio[0].reshape(height, 4, width, 4).repeat(1, 3, 1, 3).permute(0, 2, 1, 3).reshape(1, height*width, 144)[if_matching1][if_matching2]
    new_label = label[0, :, 0].reshape(height, 4, width, 4).repeat(1, 3, 1, 3).permute(0, 2, 1, 3).reshape(1, height*width, 144)[if_matching1][if_matching2]
    scale1 = (scale1 * scale0[if_matching1].reshape(-1, 1).repeat(1, 144))[if_matching2]
    criterion = (new_label > -0.0001)
    depth_ratio = depth_ratio[criterion].log()
    scale1 = scale1[criterion].log()
    # depth_ratio_mean = depth_ratio.mean()
    # scale1_mean = scale1.mean()
    # result = [(depth_ratio * depth_ratio).mean() - depth_ratio_mean**2, 
    #     (scale1 * scale1).mean() - scale1_mean * scale1_mean, (scale1 * depth_ratio).mean() - scale1_mean * depth_ratio_mean]
    result = [depth_ratio, scale1]
    return result

