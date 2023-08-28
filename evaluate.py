import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import yaml
import numpy.random as random
from torch.utils.data.dataloader import DataLoader
from datasets.megadepth import MegaDepth
from datasets.scannet import Scannet
from datasets.yfcc import Yfcc
from models.pats import PATS
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from utils.metrics import compute_pose_error, aggregate_metrics


@torch.no_grad()
def evaluate_megadepth(model, data_path, pairs_path, scale_factor, threshold):
    dataset = MegaDepth(data_path, pairs_path, is_train=False, aug_resolution=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    error_R_list, error_t_list = [], []
    for data in tqdm(loader):
        data['image0'] = data['image0'].cuda()
        data['image1'] = data['image1'].cuda()
        result = model(data)
        error_R, error_t = compute_pose_error(result['matches_l'].cpu().numpy(),
                                              result['matches_r'].cpu().numpy(),
                                              data['K0'][0].numpy(), data['K1'][0].numpy(),
                                              data['T0'][0].numpy(), data['T1'][0].numpy(),
                                              scale_factor, threshold)
        error_R_list.append(error_R)
        error_t_list.append(error_t)
    metric = aggregate_metrics(error_R_list, error_t_list)
    print('-'*5 + 'Evaluation on MegaDepth' + '-'*5)
    for key, value in metric.items():
        print(f'{key}: {value}')


@torch.no_grad()
def evaluate_yfcc(model, data_path, pairs_path, scale_factor, threshold):
    dataset = Yfcc(data_path, pairs_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    error_R_list, error_t_list = [], []
    for data in tqdm(loader):
        data['image0'] = data['image0'].cuda()
        data['image1'] = data['image1'].cuda()
        result = model(data)
        error_R, error_t = compute_pose_error(result['matches_l'].cpu().numpy(),
                                              result['matches_r'].cpu().numpy(),
                                              data['K0'][0].numpy(), data['K1'][0].numpy(),
                                              data['T0'][0].numpy(), data['T1'][0].numpy(),
                                              scale_factor, threshold)
        error_R_list.append(error_R)
        error_t_list.append(error_t)
    metric = aggregate_metrics(error_R_list, error_t_list)
    print('-'*5 + 'Evaluation on YFCC' + '-'*5)
    for key, value in metric.items():
        print(f'{key}: {value}')



@torch.no_grad()
def evaluate_scannet(model, data_path, pairs_path, scale_factor, threshold):
    dataset = Scannet(data_path, pairs_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    error_R_list, error_t_list = [], []
    for data in tqdm(loader):
        data['image0'] = data['image0'].cuda()
        data['image1'] = data['image1'].cuda()
        result = model(data)
        error_R, error_t = compute_pose_error(result['matches_l'].cpu().numpy(),
                                              result['matches_r'].cpu().numpy(),
                                              data['K0'][0].numpy(), data['K1'][0].numpy(),
                                              data['T0'][0].numpy(), data['T1'][0].numpy(),
                                              scale_factor, threshold)
        error_R_list.append(error_R)
        error_t_list.append(error_t)
    metric = aggregate_metrics(error_R_list, error_t_list)
    print('-'*5 + 'Evaluation on ScanNet' + '-'*5)
    for key, value in metric.items():
        print(f'{key}: {value}')



if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--scale_factor', type=float, default=1.0)

    param = parser.parse_args()
    param.cur_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if param.config is not None:
        with open(param.config, 'r') as f:
            yaml_dict = yaml.safe_load(f)
            for k, v in yaml_dict.items():
                param.__dict__[k] = v

    # initialize random seed
    torch.manual_seed(param.seed)
    np.random.seed(param.seed)
    random.seed(param.seed)
    model = PATS(param)
    model.load_state_dict()
    model = model.cuda().eval()

    dataset_name = param.dataset
    if dataset_name == 'MegaDepth':
        evaluate_megadepth(model, param.data_path, param.pairs_path, param.scale_factor, param.threshold)
    elif dataset_name == 'YFCC':
        evaluate_yfcc(model, param.data_path, param.pairs_path, param.scale_factor, param.threshold)
    elif dataset_name == 'ScanNet':
        evaluate_scannet(model, param.data_path, param.pairs_path, param.scale_factor, param.threshold)

