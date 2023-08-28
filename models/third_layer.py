import torch
import torch.nn as nn

import torch.nn.functional as F
from models.modules import *
from kornia.utils.grid import create_meshgrid
import torchvision.models as models

from models.resnet import ResNet2, BasicBlock


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FPN_8_2(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """
    def __init__(self):
        super().__init__()
        block_dims = [128, 192, 264]
        block_dims_before = [64, 64, 128]
        self.layer3_outconv = conv1x1(block_dims_before[2], block_dims[2])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[2]),
        )
        self.layer2_outconv = conv1x1(block_dims_before[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims_before[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )
        self.pad2 = torch.nn.ZeroPad2d(2)
        self.pad1 = torch.nn.ZeroPad2d(1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, x_before):
        # ResNet Backbone
        x3_out = self.layer3_outconv2(x) + self.layer3_outconv(x_before[2])
        x3_out_2x = self.pad1(F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=False))
        x2_out = self.pad1(self.layer2_outconv(x_before[1]))
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)
        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=False)
        x1_out = self.pad2(self.layer1_outconv(x_before[0]))
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)
        x_out = x1_out.reshape(2, -1, 128, x1_out.shape[2], x1_out.shape[3])
        return x_out[0], x_out[1]


class ThirdLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Misc
        # Modules
        self.backbone = FPN_8_2()
        # self.loftr_fine = LocalFeatureTransformer(config["fine"])
        # self.fine_matching = FineMatching(if_new=True)
        self.scale_proj = nn.Conv2d(in_channels=128, out_channels=1,
                                     kernel_size=3, padding=1, stride=1, bias=True)
        self.compress = MLP([264, 264, 264, 128])
        self.gnn = AttentionalGNN(128, ['self', 'cross'] * 5)
        # self.final_proj = nn.Conv1d(128, 128, kernel_size=1, bias=True)
        self.pad = torch.nn.ZeroPad2d(2)
        self.pad_1 = torch.nn.ConstantPad2d(2, 1e-2)
        self.sigmoid = nn.Sigmoid()
        self.kenc = KeypointEncoder(
            128, [32, 64, 128, 256, 512])
        self.descriptor_extract = ResNet2(BasicBlock, [3, 4, 6, 3])
        pretrained_dict = models.resnet34(pretrained=True).state_dict()
        model_dict = self.descriptor_extract.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.descriptor_extract.load_state_dict(model_dict)
        for p in self.descriptor_extract.parameters():
           p.requires_grad = True

        self.W = 8 # size of output's feature map
        # 额外进行了两次padding，所以特征图的大小是52*52而不是48*48，所以所有坐标要+2
        self.M = 52 # size of input's image after padding
        self.T = 5 # neighborhood size

    def forward(self, new_left, new_right, mkpts0_c, mkpts1_c, b_ids, desc_before, mdesc, outdoor):
        pic0 = torch.cat([new_left.permute(0, 3, 1, 2).float().contiguous(), new_right.permute(0, 3, 1, 2).float().contiguous()], dim=0).reshape(-1, 
            new_left.shape[3], new_left.shape[1], new_left.shape[2])
        desc_before = self.descriptor_extract.forward2(pic0)
        self.one = torch.tensor(1.0, device=mdesc.device)
        feat_f0, feat_f1 = self.backbone(mdesc[:, :, :-1].reshape(mdesc.shape[0], -1, 12, 12), desc_before)
        # 计算垃圾箱的特征
        rubbish = self.compress(mdesc[:, :, :-1].reshape(mdesc.shape[0], -1, 144))
        # 计算上一层传入的点所在的 self.W * self.W patch位置，包括rubbish
        b = b_ids.reshape(-1, 1).repeat(1, self.W*self.W)
        mkpts0_c = torch.round(mkpts0_c / 4.0).long() * 4
        x0 = (mkpts0_c[:, 0] // 2).reshape(-1, 1).expand(-1, self.W * self.W) + torch.arange(self.W, device=feat_f0.device).reshape(1, 1, self.W).repeat(b_ids.shape[0], self.W, 1).reshape(-1, self.W*self.W) - self.W / 2 + 2
        y0 = (mkpts0_c[:, 1] // 2).reshape(-1, 1).expand(-1, self.W * self.W) + torch.arange(self.W, device=feat_f0.device).reshape(1, self.W, 1).repeat(b_ids.shape[0], 1, self.W).reshape(-1, self.W*self.W) - self.W / 2 + 2
        index0 = (b * self.M * self.M + y0 * self.M + x0).long().reshape(-1, 1).expand(-1, 128)
        mkpts1_c = torch.where(mkpts1_c >= 96, torch.tensor(96 , device=feat_f0.device).float(), mkpts1_c)
        mkpts1_c = torch.where(mkpts1_c <= 0, torch.tensor(0, device=feat_f0.device).float(), mkpts1_c)
        mkpts1_c = torch.round(mkpts1_c / 4.0).long() * 4
        x1 = (mkpts1_c[:, 0] // 2).reshape(-1, 1).expand(-1, self.W*self.W) + torch.arange(self.W, device=feat_f0.device).reshape(1, 1, self.W).repeat(b_ids.shape[0], self.W, 1).reshape(-1, self.W*self.W) - self.W / 2 + 2
        y1 = (mkpts1_c[:, 1] // 2).reshape(-1, 1).expand(-1, self.W*self.W) + torch.arange(self.W, device=feat_f0.device).reshape(1, self.W, 1).repeat(b_ids.shape[0], 1, self.W).reshape(-1, self.W*self.W) - self.W / 2 + 2
        index1 = (b * self.M * self.M + y1 * self.M + x1).long().reshape(-1, 1).expand(-1, 128)

        cols = torch.arange(0, self.W).reshape(self.W, 1).repeat(1, self.W).reshape(-1) / float(self.W)
        rows = torch.arange(0, self.W).reshape(1, self.W).repeat(self.W, 1).reshape(-1) / float(self.W)
        kpts  = torch.zeros((self.W * self.W), 2).to(feat_f0.device)
        kpts [:, 0] = cols
        kpts [:, 1] = rows

        feat_f0_unfold = torch.gather(feat_f0.permute(0, 2, 3, 1).reshape(-1, feat_f0.shape[1]), 0, index0).reshape(-1, self.W*self.W, 128).permute(0, 2, 1) + self.kenc(kpts)
        feat_f1_unfold = torch.gather(feat_f1.permute(0, 2, 3, 1).reshape(-1, feat_f1.shape[1]), 0, index1).reshape(-1, self.W*self.W, 128).permute(0, 2, 1) + self.kenc(kpts)
        x2 = torch.round(mkpts0_c[:, 0] / 8.0).long()
        y2 = torch.round(mkpts0_c[:, 1] / 8.0).long()
        index2 = (b_ids * 12 * 12 + y2 * 12 + x2).long().reshape(-1, 1).expand(-1, 128)
        rubbish_unfold = torch.gather(rubbish.permute(0, 2, 1).reshape(-1, 128), 0, index2).reshape(-1, 128, 1)
        feat_f0_unfold = torch.cat([feat_f0_unfold, rubbish_unfold], dim=2)
        feat_f1_unfold = torch.cat([feat_f1_unfold, rubbish_unfold], dim=2)
        # self 与 cross attention
        feat_f0_unfold, feat_f1_unfold = self.gnn(feat_f0_unfold, feat_f1_unfold)
        # feat_f0_unfold, feat_f1_unfold = self.final_proj(feat_f0_unfold), self.final_proj(feat_f1_unfold)
        # 估计局部scale
        scale = self.scale_proj(feat_f1_unfold[:, :, :-1].reshape(-1, 128, self.W, self.W)).reshape(-1, 1, self.W*self.W)
        scale = torch.exp(self.sigmoid(scale) * math.log(256.0) - math.log(256.0) / 2)
        scale_x = (scale + 1e-8).sqrt() 
        scale_y = (scale + 1e-8).sqrt() 
        # 计算特征的相关性分数
        scores = torch.einsum('bdn,bdm->bnm', feat_f0_unfold, feat_f1_unfold)
        scores = scores / 128**.5
        scores_origin = log_optimal_transport2(0.1 * scores, self.one, scale, iters=100)
        scores = torch.exp(scores_origin)
        mkpts0_f, mkpts1_f, whole_loss = self.Compute_result(scores, self.W, self.T, scale_x, scale_y, mkpts0_c, mkpts1_c, feat_f0_unfold.device)
        label = ((torch.zeros_like(mkpts1_c[:, None, :].expand(-1, 16, -1))) + 1e8).reshape(-1, 2)
        if not outdoor:
            select1 = torch.logical_or(torch.arange(label.shape[0], device=scores.device) % 16 == 5, torch.arange(label.shape[0], device=scores.device) % 16 == 15)
            select2 = torch.logical_or(torch.arange(label.shape[0], device=scores.device) % 16 == 7, torch.arange(label.shape[0], device=scores.device) % 16 == 13)
            select = torch.logical_or(select1, select2)
            label[:, 0] = torch.where(select, label[:, 0], torch.tensor(-10.0, device=label.device))
        scores_used = scores[:, :-1, :].reshape(scores.shape[0], self.W, self.W, -1)[:, 2:6, 2:6, :].reshape(scores.shape[0], 16, -1) + 1e-8
        if_matching1 = (scores_used.max(2)[1] != self.W ** 2)
        if outdoor:
            label[:, 0] = torch.where(if_matching1.reshape(-1), label[:, 0], torch.tensor(-10.0, device=label.device))
        return {
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f,
            "label": label,
        }

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)


    def Compute_result(self, scores, W, T, scale_x, scale_y, p_s, p_t, device):
        scores_back = self.pad(scores[:, :-1, :-1].reshape(scores.shape[0], scores.shape[1] - 1, W, W)).reshape(scores.shape[0], scores.shape[1] - 1, -1)
        scores_back = scores_back.reshape(scores.shape[0], W, W, -1)[:, 2:6, 2:6, :].reshape(scores.shape[0], 16, -1)
        # 提取T*T邻域， 并计算回归坐标结果
        max0 = scores[:, :-1, :-1].max(2)[1].reshape(scores.shape[0], W, W)[:, 2:6, 2:6].reshape(scores.shape[0], 16)
        x3 = (max0 % W).unsqueeze(2).expand(-1, -1, T**2)  + torch.arange(T, device=device).reshape(1, 1, 1, T).repeat(max0.shape[0], max0.shape[1], T, 1).reshape(max0.shape[0], max0.shape[1], T**2)
        y3 = (max0 // W).unsqueeze(2).expand(-1, -1, T**2)  + torch.arange(T, device=device).reshape(1, 1, T, 1).repeat(max0.shape[0], max0.shape[1], 1, T).reshape(max0.shape[0], max0.shape[1], T**2)
        index3 = y3 * (W + 4) + x3
        # print(index3.max(), index3.min())
        # import pdb
        # pdb.set_trace()
        scale_x_new = self.pad_1(scale_x.reshape(1, -1, W, W)).reshape(scale_x.shape[0], 1, -1).expand(-1, 16, -1)
        scale_y_new = self.pad_1(scale_y.reshape(1, -1, W, W)).reshape(scale_y.shape[0], 1, -1).expand(-1, 16, -1)
        scores_unfold_x = torch.gather((scores_back + 1e-7).sqrt() / scale_x_new, 2, index3)
        scores_unfold_y = torch.gather((scores_back + 1e-7).sqrt() / scale_y_new, 2, index3)
        positions = create_meshgrid(T, T, False, device=device).reshape(-1, 2) * 2 - (T - 1)
        weighted_point_x = torch.einsum('ijp,p->ij', scores_unfold_x, positions[:, 0])
        weighted_point_y = torch.einsum('ijp,p->ij', scores_unfold_y, positions[:, 1])
        point_weight_sum_x = scores_unfold_x.sum(2)
        point_weight_sum_y = scores_unfold_y.sum(2)
        # 计算真实坐标系下的坐标点
        mkpts1_f = torch.zeros([scale_x.shape[0], 16, 2], device=scores.device)
        mkpts1_f[:, :, 0] = weighted_point_x / point_weight_sum_x + (max0 % W + 0.5 - W/2) * 2.0
        mkpts1_f[:, :, 1] = weighted_point_y / point_weight_sum_y + (max0 // W + 0.5 - W/2) * 2.0
        mkpts1_f = mkpts1_f + p_t.reshape(-1, 1, 2).repeat(1, 16, 1)
        mkpts0_f = p_s.reshape(-1, 1, 2).repeat(1, 16, 1) + create_meshgrid(4, 4, False, device=device)\
            .reshape(1, -1, 2).repeat(p_s.shape[0], 1, 1) * 2 - 3.0
        # 计算whole_loss
        scores_unfold_sum = torch.gather(scores_back, 2, index3).sum(2)
        whole_loss = (scores[:, :-1, :].reshape(scores.shape[0], W, W, -1)[:, 2:6, 2:6, :].reshape(scores.shape[0], 16, -1).sum(2) - scores_unfold_sum)
            #  + torch.gather(scores[:, -1, :], 1, max0)
        whole_loss = torch.where(whole_loss >= 1e-2, whole_loss, torch.tensor(0.0, device=device)) / ((whole_loss >= 1e-2).float().sum() + 10) / 10

        return mkpts0_f, mkpts1_f, whole_loss


if __name__ == '__main__':
    third_layer = ThirdLayer()

    new_left = torch.rand(100, 96, 96, 3)
    new_right = torch.rand(100, 96, 96, 3)
    mkpts0_c = (torch.rand(3000, 2) * 80).int() + 10
    mkpts1_c = (torch.rand(3000, 2) * 80).int() + 10
    b_ids = (torch.rand(3000) * 100).int()
    label_dense = torch.rand(3000, 16, 3)
    desc_before = [
        torch.rand(200, 64, 48, 48),
        torch.rand(200, 64, 24, 24),
        torch.rand(200, 128, 12, 12),
    ]
    mdesc = torch.rand(200, 264, 145)
    outdoor = True

    result3 = third_layer(new_left, new_right, mkpts0_c, mkpts1_c, b_ids, label_dense, desc_before, mdesc, outdoor=True)

