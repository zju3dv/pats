import math

from models.resnet import ResNet2, BasicBlock
from utils.utils import Iterative_expand_matrix, Compute_positions_and_ranges
import torchvision.models as models
from models.modules import *
import torchvision


class SecondLayer(nn.Module):
    default_config = {
        'descriptor_dim': 264,
        'weights': 'outdoor',
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
        'point_num': 144,
        'scores_ratio': [0.4, 0.3, 0.3],
        'scores_num': [20, 7, 2]
    }
    def __init__(self):
        super().__init__()
        self.row_num = 12
        self.config = {**self.default_config}
        self.descriptor_extract = ResNet2(BasicBlock, [3, 4, 6, 3])
        pretrained_dict = models.resnet34(pretrained=True).state_dict()
        model_dict = self.descriptor_extract.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.descriptor_extract.load_state_dict(model_dict)
        for p in self.descriptor_extract.parameters():
           p.requires_grad = True
        self.sigmoid = nn.Sigmoid()
        self.scalex_proj = nn.Conv2d(in_channels=self.config['descriptor_dim'], out_channels=1,
                                     kernel_size=3, padding=1, stride=1, bias=True)
        self.scaley_proj = nn.Conv2d(in_channels=self.config['descriptor_dim'], out_channels=1,
                                     kernel_size=3, padding=1, stride=1, bias=True)
        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])
        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=2)
        self.relu = torch.nn.ReLU(inplace=False)
        cols = torch.arange(0, self.row_num).reshape(self.row_num, 1).repeat(1, self.row_num).reshape(self.config['point_num'])
        rows = torch.arange(0, self.row_num).reshape(1, self.row_num).repeat(self.row_num, 1).reshape(self.config['point_num'])
        self.positions = torch.zeros((self.config['point_num'], 2))
        self.positions[:, 0] = cols
        self.positions[:, 1] = rows
        # self.ranges = self.ranges.reshape(1, 20, 20)
        assert self.config['weights'] in ['indoor', 'outdoor']
        self.upsampling = nn.Upsample(size=[12, 12], mode='bilinear', align_corners=True)
        self.avgpool = nn.AvgPool2d(2, stride=1, padding=1)
        self.compress_1 = MLP([448, 256, 128, 64, 32, 16, 8])
        self.compress_2 = MLP([448, 448, 448, self.config['descriptor_dim']])
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    # outdoor: Megadepth, yfcc, InLoc, Aachen
    # indoor: ScanNet
    def forward(self, left, right, desc_l, original_image_shape, if_nomatching1_L1, scores_back, outdoor, merge_new):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        self.one = torch.tensor(1.0, device=left.device)
        self.zeros = torch.tensor(0.0, device=left.device)
        self.positions = self.positions.to(left.device).contiguous()
        left = self.normalize(left.permute(0, 3, 1, 2).float().contiguous())
        right = self.normalize(right.permute(0, 3, 1, 2).float().contiguous())
        pic0 = torch.cat([left, right], dim=0).reshape(-1, left.shape[1], left.shape[2], left.shape[3])
        desc0_ = self.descriptor_extract.forward2(pic0)
        desc = []
        for i, feat in enumerate(desc0_):
            stride = int(8.0 / torch.pow(torch.tensor(2.0, device=left.device), i + 1))
            # feat = feat[:, :, stride*2:stride*10, stride*2:stride*10]
            if i <= 1:
                feat = self.avgpool(feat)
            index = ((self.positions.reshape(self.row_num, self.row_num, 2) + 0.5) * stride).long()
            index = (index[:, :, 0] * feat.shape[3] + index[:, :, 1]).reshape(1, 1, -1).\
                repeat(feat.shape[0], feat.shape[1], 1)
            desc.append(torch.gather(feat.reshape(feat.shape[0], feat.shape[1], -1), 2, index))
        desc = torch.cat(desc, dim=1).reshape(2, left.shape[0], 256, -1)
        # desc = desc0_[2].reshape(2, left.shape[0], 128, -1)
        title = self.compress_1(desc_l.unsqueeze(2)).repeat(2, 1, self.config['point_num']).reshape(2, left.shape[0], 8, -1)
        rubbish = self.compress_2(desc_l.unsqueeze(2)).repeat(2, 1, 1).reshape(2, left.shape[0], self.config['descriptor_dim'], 1)
        desc = torch.cat([title, desc], dim=2)
        desc = torch.cat([desc ,rubbish], dim=3)
        desc0 = desc[0]
        desc1 = desc[1]
        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)
        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        scale_x = self.scalex_proj(mdesc1[:, :, :-1].reshape(mdesc1.shape[0], -1, self.row_num, self.row_num))\
            .reshape(right.shape[0], -1, self.config['point_num'])
        scale_y = self.scaley_proj(mdesc1[:, :, :-1].reshape(mdesc1.shape[0], -1, self.row_num, self.row_num))\
            .reshape(right.shape[0], -1, self.config['point_num'])
        scale_x = torch.exp(self.sigmoid(scale_x) * math.log(256.0) - math.log(256.0) / 2)
        scale_y = torch.exp(self.sigmoid(scale_y) * math.log(256.0) - math.log(256.0) / 2)
        scale = scale_x * scale_y
        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5
        # Run the optimal transport.
        scores = log_optimal_transport2(
            0.1 * scores, self.one, scale,
            iters=self.config['sinkhorn_iterations'])
        # Get the matches with score above "match_threshold".
        if outdoor:
            scores[:, :, -1] += torch.log(self.one * 2)
            scores[:, -1, :] += torch.log(self.one * 2)
        else:
            scores[:, :, -1] += torch.log(self.one * 3)
            scores[:, -1, :] += torch.log(self.one * 3)

        patch_scale = 8
        patch_size = [96, 96]
        trust_score_L2, pts, x_scale_reproj, y_scale_reproj, if_nomatching1, if_nomatching2 = \
            self.est_position(scores, scale_x, scale_y, patch_size, patch_scale)
        patch_num = left.shape[0]
        if merge_new:
            if_nomatching1, scores_back = self.merge_patches_new(patch_num, trust_score_L2, original_image_shape, if_nomatching1_L1, if_nomatching1, scores_back)
        else:
            if_nomatching1, scores_back = self.merge_patches_old(patch_num, trust_score_L2, original_image_shape, if_nomatching1_L1, if_nomatching1, scores_back)
        return {
            'scales': [scale_x, scale_y],
            'scales_reproj': [x_scale_reproj, y_scale_reproj],
            'scores': scores,
            'features': torch.cat([mdesc0, mdesc1], dim=0),
            'features_before': desc0_,
            'pts': pts,
            'if_nomatching1': if_nomatching1,
            'if_nomatching2': if_nomatching2,
            'trust_score': trust_score_L2,
            "scores_back": scores_back
        }

    # 我们发现老的merge算法理论上是存在错误的，但是将其改正以后会造成室内位姿估计的性能降低
    def merge_patches_old(self, patch_num, trust_score, original_image_shape, if_nomatching1_L1, if_nomatching1_L2, scores_back):
        positions_x = torch.arange(0, 12, device=trust_score.device).reshape(1, 1, 12).repeat(patch_num, 12, 1)
        positions_y = torch.arange(0, 12, device=trust_score.device).reshape(1, 12, 1).repeat(patch_num, 1, 12)
        c1 = torch.logical_or(positions_x < 1, positions_x > 10)
        c2 = torch.logical_or(positions_y < 1, positions_y > 10)
        c3 = torch.logical_or(c1, c2).reshape(-1, 144)
        for i in range(3):
            d1 = torch.logical_or(positions_x < 3 - i, positions_x > 7 + i)
            d2 = torch.logical_or(positions_y < 3 - i, positions_y > 7 + i)
            d3 = torch.logical_or(d1, d2).reshape(-1, 144)
            trust_score[d3] *= 2
        if_nomatching1_L2[trust_score > 2.0] = True
        if_nomatching1_L2[c3] = True
        batch_num = if_nomatching1_L1.shape[0]
        first_choice_num = if_nomatching1_L2.shape[0]
        height = original_image_shape[0] // 32
        width = original_image_shape[1] // 32
        if_matching = torch.zeros([batch_num, height * width, 12, 12], device=trust_score.device).bool()
        if_matching[torch.logical_not(if_nomatching1_L1)] = torch.logical_not(if_nomatching1_L2.bool()).reshape(first_choice_num, 12, 12)
        if_matching = if_matching.reshape(batch_num, height, width, 3, 4, 3, 4).permute(0, 1, 4, 2, 6, 3, 5).reshape(batch_num, height * 4, width * 4, 9)
        scores = trust_score.reshape(first_choice_num, 3, 4, 3, 4).permute(0, 2, 4, 1, 3).reshape(-1, 16, 9)
        # scores_back = torch.zeros([batch_num, height * width, 16, 9], device=trust_score.device).double()
        scores_back[torch.logical_not(if_nomatching1_L1)] = scores.double()
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
        sequence2 = sequence_base.reshape(batch_num, -1, 1) + torch.arange(height * width * 16, device=trust_score.device).\
            reshape(1, -1, 1).repeat(batch_num, 1, 1) * 9 - (sequence_base.reshape(batch_num, -1, 1) % 3 - 1) * 4 * 9 - \
            (sequence_base.reshape(batch_num, -1, 1) // 3 - 1) * 4 * width * 4 * 9
        h = torch.arange(height * width * 16, device=trust_score.device).reshape(1, -1).repeat(batch_num, 1) // width // 4 - \
            (sequence_base.reshape(batch_num, -1) // 3 - 1) * 4
        w = torch.arange(height * width * 16, device=trust_score.device).reshape(1, -1).repeat(batch_num, 1) % (width * 4) - \
            (sequence_base.reshape(batch_num, -1) % 3 - 1) * 4 
        critical1 = torch.logical_or(h < 0, h >= height * 4)
        critical2 = torch.logical_or(w < 0, w >= width * 4)
        critical = torch.logical_or(critical1, critical2)
        if_matching[critical] = False
        sequence2 = torch.clamp(sequence2, 0, height * width * 144 - 1)
        if_nomatching = torch.ones([batch_num, height * 4 * width * 4 * 9], device=trust_score.device)
        if_nomatching = torch.scatter(if_nomatching, 1, sequence2.reshape(batch_num, -1), torch.logical_not(if_matching).float()).bool()
        return if_nomatching.reshape(batch_num, height, 4, width, 4, 3, 3).permute(0, 1, 3, 5, 2, 6, 4).reshape(batch_num, height * width,
         144)[torch.logical_not(if_nomatching1_L1)], torch.zeros([batch_num, height * width, 16, 9], device=trust_score.device).double()


    def merge_patches_new(self, patch_num, trust_score, original_image_shape, if_nomatching1_L1, if_nomatching1_L2, scores_back):
        positions_x = torch.arange(0, 12, device=trust_score.device).reshape(1, 1, 12).repeat(patch_num, 12, 1)
        positions_y = torch.arange(0, 12, device=trust_score.device).reshape(1, 12, 1).repeat(patch_num, 1, 12)
        c1 = torch.logical_or(positions_x < 1, positions_x > 10)
        c2 = torch.logical_or(positions_y < 1, positions_y > 10)
        c3 = torch.logical_or(c1, c2).reshape(-1, 144)
        for i in range(3):
            d1 = torch.logical_or(positions_x < 3 - i, positions_x > 7 + i)
            d2 = torch.logical_or(positions_y < 3 - i, positions_y > 7 + i)
            d3 = torch.logical_or(d1, d2).reshape(-1, 144)
            trust_score[d3] *= 2.0
        if_nomatching1_L2[trust_score > 2.0] = True
        if_nomatching1_L2[c3] = True
        trust_score[torch.logical_not(if_nomatching1_L2)] -= 10000.0
        batch_num = if_nomatching1_L1.shape[0]
        first_choice_num = if_nomatching1_L2.shape[0]
        height = original_image_shape[0] // 32
        width = original_image_shape[1] // 32
        if_matching = torch.zeros([batch_num, height * width, 12, 12], device=trust_score.device).bool()
        if_matching[torch.logical_not(if_nomatching1_L1)] = torch.logical_not(if_nomatching1_L2.bool()).reshape(first_choice_num, 12, 12)
        if_matching = if_matching.reshape(batch_num, height, width, 3, 4, 3, 4).permute(0, 1, 4, 2, 6, 3, 5).reshape(batch_num, height * 4, width * 4, 9)
        scores = trust_score.reshape(first_choice_num, 3, 4, 3, 4).permute(0, 2, 4, 1, 3).reshape(-1, 16, 9)
        scores_back[torch.logical_not(if_nomatching1_L1)] = scores.double()
        scores_back_use = scores_back.reshape(batch_num, height, width, 4, 4, 9).permute(0, 1, 3, 2, 4, 5).reshape(batch_num, height * 4, width * 4, 9)
        scores_back_use[:, :, :, 4] -= 0.0
        grid_y, grid_x = torch.meshgrid(torch.arange(height * 4, device=trust_score.device), torch.arange(width * 4, device=trust_score.device))
        bound_choice = torch.stack([grid_y, grid_x], -1)[:, :, None].repeat(1, 1, 9, 1)
        bound_choice[:, :, [0, 3, 6], 1] -= 4
        bound_choice[:, :, [2, 5, 8], 1] += 4
        bound_choice[:, :, [0, 1, 2], 0] -= 4
        bound_choice[:, :, [6, 7, 8], 0] += 4
        criterion = torch.logical_or(torch.logical_or(bound_choice[:, :, :, 0] < 0, bound_choice[:, :, :, 0] >= 4 * height),
            torch.logical_or(bound_choice[:, :, :, 1] < 0, bound_choice[:, :, :, 1] >= 4 * width))[None].expand(batch_num, -1, -1, -1)
        scores_back_use[criterion] += 100000
        scores_back_use2 = scores_back_use.clone()
        if_matching2 = if_matching.clone()
        for i in range(9):
            dy = -(i % 3 - 1)
            dx = -(i // 3 - 1)
            scores_back_use2[:, 4 * max(dx, 0):height * 4 + min(0, dx) * 4, 4 * max(dy, 0):width * 4 + min(0, dy) * 4, i] = scores_back_use[:, 4 * max(-dx, 0):height * 4 + min(0, -dx) * 4, 4 * max(-dy, 0):width * 4 + min(0, -dy) * 4, 8 - i].clone()
            if_matching2[:, 4 * max(dx, 0):height * 4 + min(0, dx) * 4, 4 * max(dy, 0):width * 4 + min(0, dy) * 4, i] = if_matching[:, 4 * max(-dx, 0):height * 4 + min(0, -dx) * 4, 4 * max(-dy, 0):width * 4 + min(0, -dy) * 4, 8 - i].clone()
        sequence_base = torch.argsort(scores_back_use)[:, :, :, 0]
        if_matching2 = torch.gather(if_matching2, 3, sequence_base.reshape(batch_num, height * 4, width * 4, 1)).reshape(batch_num, -1)
        sequence2 = 8 - sequence_base.reshape(batch_num, -1, 1) + torch.arange(height * width * 16, device=trust_score.device).\
            reshape(1, -1, 1).repeat(batch_num, 1, 1) * 9 + (sequence_base.reshape(batch_num, -1, 1) % 3 - 1) * 4 * 9 + \
            (sequence_base.reshape(batch_num, -1, 1) // 3 - 1) * 4 * width * 4 * 9
        if_nomatching = torch.ones([batch_num, height * 4 * width * 4 * 9], device=trust_score.device)
        if_nomatching = torch.scatter(if_nomatching, 1, sequence2.reshape(batch_num, -1), torch.logical_not(if_matching2).float()).bool()
        return if_nomatching.reshape(batch_num, height, 4, width, 4, 3, 3).permute(0, 1, 3, 5, 2, 6, 4).reshape(batch_num, 
            height * width, 144)[torch.logical_not(if_nomatching1_L1)], scores_back

    def est_position(self, scores, scale_x, scale_y, image_shape, patch_scale):
        H, W = image_shape
        scale_x = scale_x.reshape(scale_x.shape[0], -1, 1)
        scale_y = scale_y.reshape(scale_y.shape[0], -1, 1)
        max0, max1 = scores.max(2).indices, scores.max(1).indices
        max0 = max0[:, :-1]
        max1 = max1[:, :-1]
        limitation1 = torch.tensor([0, H // patch_scale, 0, W // patch_scale], device=scores.device)
        if_nomatching1 = (max0 == (H // patch_scale) * (W // patch_scale))
        if_nomatching2 = (max1 == (H // patch_scale) * (W // patch_scale))

        positions1, ranges1 = Compute_positions_and_ranges(H // patch_scale, W // patch_scale, scores.device)

        width = W // patch_scale
        height = H // patch_scale  
        trust_score, _, average_point1, x_scale, y_scale, _ = \
            Iterative_expand_matrix(scores.exp(), scale_x, scale_y, limitation1,
                ranges1, positions1, height=height, width=width, iter_num=8, lower_bound=1e-3)

        return trust_score, average_point1, x_scale, y_scale, if_nomatching1, if_nomatching2



if __name__ == '__main__':
    second_layer = SecondLayer()
    left_patch = torch.rand(100, 96, 96, 3)
    right_patch = torch.rand(100, 96, 96, 3)
    features = torch.rand(100, 448)
    scales_reproj = torch.rand(100)
    original_image_shape = [480, 640]
    if_nomatching1_L1 = torch.ones(1, 300).bool()
    if_nomatching1_L1[0, :100] = False
    result2 = second_layer(left_patch, right_patch, features, scales_reproj,
                           original_image_shape, if_nomatching1_L1, outdoor=True)
