import numpy as np
import cv2
from collections import OrderedDict


def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(kp1, kp2, intrinsic1, intrinsic2, extrinsic1,
                       extrinsic2, scale_factor=1.0, threshold=0.25):
    if kp1.shape[0] < 15:
        return np.inf, np.inf

    kp1 = kp1[:, [1,0]]
    kp2 = kp2[:, [1,0]]

    intrinsic2 = scale_intrinsics(intrinsic2, [1.0/ scale_factor, 1.0/ scale_factor])
    if scale_factor > 1.0:
        intrinsic1[:2, 2] += np.asarray([int((scale_factor-1)*320), int((scale_factor-1)*240)])
    else:
        intrinsic2[:2, 2] += np.asarray([int((1 - scale_factor)*320), int((1 - scale_factor)*240)])
    relative_pose = extrinsic2.astype(float).dot(np.linalg.inv(extrinsic1.astype(float)))

    f_mean = np.mean([intrinsic1[0, 0], intrinsic2[1, 1], intrinsic1[0, 0], intrinsic2[1, 1]])
    norm_thresh = threshold / f_mean

    kp1 = (kp1 - intrinsic1[[0, 1], [2, 2]][None]) / intrinsic1[[0, 1], [0, 1]][None]
    kp2 = (kp2 - intrinsic2[[0, 1], [2, 2]][None]) / intrinsic2[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kp1, kp2, np.eye(3), threshold=norm_thresh, prob=1-1e-5,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kp1, kp2, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    R_gt = relative_pose[:3, :3]
    t_gt = relative_pose[:3, 3]
    if ret != None:
        R, t, _ = ret
    else:
        R, t = R, t[:, 0]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)

    return error_R, error_t



def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def aggregate_metrics(error_R_list, error_t_list):
    angular_thresholds = [5, 10, 20]
    error_R = np.array(error_R_list)
    error_t = np.array(error_t_list)
    pose_errors = np.max(np.stack([error_R, error_t]), axis=0)
    aucs = error_auc(pose_errors, angular_thresholds)
    return aucs