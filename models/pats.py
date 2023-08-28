import torch.nn as nn
import torch
from models.first_layer import FirstLayer
from models.second_layer import SecondLayer
from models.third_layer import ThirdLayer
from utils.utils import get_result
from collections import OrderedDict


class PATS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.first_layer = FirstLayer()
        self.second_layer = SecondLayer()
        self.third_layer = ThirdLayer()

    def forward(self, data):
        left = data['image0']
        right = data['image1']
        width = left.shape[2] // 32
        height = left.shape[1] // 32
        matches_l = torch.zeros([0, 2], device=left.device)
        matches_r = torch.zeros([0, 2], device=left.device)

        result_first = self.first_layer(left, right, self.config.if_local)
        if torch.logical_not(result_first['if_nomatching1']).int().sum() <= 0:
            return {
                "matches_l": matches_l,
                "matches_r": matches_r,
            }
        scores_refine_iter = torch.zeros([left.shape[0], height * width, 16, 9], device=left.device).double()
        for num, input in enumerate(result_first['output_list']):
            result_second = self.second_layer(input['patches'][0], input['patches'][1], 
                        result_first['features'][0].permute(0, 2, 1)[torch.logical_not(input['if_nomatching1'])], left.shape[1:], 
                        input['if_nomatching1'], scores_refine_iter, self.config.if_outdoor, self.config.merge_new)
            scores_refine_iter = result_second["scores_back"]
            if result_first['third_layer_set'][num][1] != 0:
                result_second['if_nomatching1'][-result_first['third_layer_set'][num][1]:, :] = True
            if_ndelete = torch.logical_not(result_second['if_nomatching1']).int().sum(1).bool()
            result_second['if_nomatching1'] = result_second['if_nomatching1'][if_ndelete]
            if torch.logical_not(result_second['if_nomatching1']).float().sum() >= 0.5:
                result_second['pts'] = result_second['pts'][if_ndelete]
                result_second['trust_score'] = result_second['trust_score'][if_ndelete] 
                input['patches'][0] = input['patches'][0][if_ndelete]
                input['patches'][1] = input['patches'][1][if_ndelete]
                result_second['features'] = result_second['features'].reshape(2, -1, result_second['features'].shape[1], 
                    result_second['features'].shape[2]).permute(1, 0, 2, 3)[if_ndelete].permute(1, 0, 2, 3).reshape(-1, 
                    result_second['features'].shape[1], result_second['features'].shape[2])
                desc_new= []
                for desc in result_second['features_before']:
                    n, d, h, w = desc.shape
                    desc_new.append(desc.reshape(2, n//2, d, h, w).permute(1, 0, 2, 3, 4)\
                        [if_ndelete].permute(1, 0, 2, 3, 4).reshape(-1, d, h, w))
                result_second['features_before'] = desc_new
                input['if_nomatching1'][torch.logical_not(input['if_nomatching1'])] = torch.logical_not(if_ndelete)
                sequence = torch.arange(0, 144, device=left.device).reshape(-1, 144, 1).repeat(input['patches'][0].shape[0], 1, 1)
                third_input = torch.cat([sequence % 12 * 4 + 2, sequence // 12 * 4 + 2, torch.round(result_second['pts'] * 4)[:, :,
                    [1, 0]], torch.arange(input['patches'][0].shape[0], device=left.device).reshape(-1, 1, 1).repeat(1, 144, 1)], dim=2)
                third_input = third_input[torch.logical_not(result_second['if_nomatching1'])]
                result_third = self.third_layer(input['patches'][0], input['patches'][1], third_input[:, :2] * 2, third_input[:, 2:4] * 2, 
                    third_input[:, -1], result_second['features_before'], result_second['features'], self.config.if_outdoor)
                mkpts1 = result_third['mkpts1_f'].reshape(-1, 2)
                result_second['pts'] = result_second['pts'].reshape(-1, 144, 1, 2).repeat(1, 1, 16, 1)
                result_second['if_nomatching1'] = result_second['if_nomatching1'].reshape(-1, 144, 1).repeat(1, 1, 16)
                result_second['pts'][torch.logical_not(result_second['if_nomatching1'])] = mkpts1.float()
                label = torch.zeros_like(result_second['if_nomatching1']).float()
                label[torch.logical_not(result_second['if_nomatching1'])] = result_third["label"][:, 0]
                result_second['if_nomatching1'] = torch.logical_or(result_second['if_nomatching1'], label < -9.9)
                result_second['if_nomatching1'] = result_second['if_nomatching1'].reshape(-1, 12, 12, 4, 4).permute(0, 1, 3, 2, 4).reshape(-1, 144 * 16)
                result_second['pts'] = result_second['pts'].reshape(-1, 12, 12, 4, 4, 2).permute(0, 1, 3, 2, 4, 5).reshape(-1, 144 * 16, 2)
                if_nomatching = [input['if_nomatching1'], result_second['if_nomatching1']]
                patch_size = [[32, height, width], [2, 48, 48]]
                scale = [input['scales'][0], input['scales'][0].reshape(-1, width * height, 2)[torch.logical_not(input['if_nomatching1'])].reshape(-1, 1, 2).repeat(1, 144 * 16, 1)]
                average_point = [input['pts_new'].flip(dims=[2]) / 32.0, result_second['pts'].flip(dims=[2]) / 2.0]
                left_choice = torch.ones([left.shape[0]]).bool().cuda()
                left_choice = [left_choice, torch.ones([result_second['if_nomatching1'].shape[0]], device=left.device).bool()]
                matches_l_new, matches_r_new = get_result(left.shape[0], if_nomatching, average_point, scale, patch_size, left_choice)
                matches_l = torch.cat([matches_l, matches_l_new], dim=0)
                matches_r = torch.cat([matches_r, matches_r_new], dim=0)
        result = {
            "matches_l": matches_l,
            "matches_r": matches_r,
        }
        return result


    def load_state_dict(self):
        model_dict = torch.load(self.config.checkpoint)
        new_model_dict = OrderedDict()
        for k, v in model_dict.items():
            name = k[7:]
            new_model_dict[name] = v
        self.first_layer.load_state_dict(new_model_dict)

        model_dict2 = torch.load(self.config.checkpoint2)
        new_model_dict2 = OrderedDict()
        for k, v in model_dict2.items():
            if k[:7] != 'evaluat':
                name = k[7:]
                new_model_dict2[name] = v
        self.second_layer.load_state_dict(new_model_dict2)

        model_dict3 = torch.load(self.config.checkpoint3)
        new_model_dict3 = OrderedDict()
        for k, v in model_dict3.items():
            name = k[7:]
            new_model_dict3[name] = v
        self.third_layer.load_state_dict(new_model_dict3)


    def eval(self):
        self.first_layer = self.first_layer.eval()
        self.second_layer = self.second_layer.eval()
        if self.config.if_local:
            self.third_layer = self.third_layer.eval()
        else:
            self.third_layer = self.third_layer.train()
        
        return self