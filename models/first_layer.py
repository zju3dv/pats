import math
from models.resnet import ResNet, BasicBlock
from utils.utils import Iterative_expand_matrix, Compute_positions_and_ranges, Compute_imgs, split_patches
from models.modules import *
import torchvision.transforms.functional as transforms
import torchvision


class FirstLayer(nn.Module):
    default_config = {
        'descriptor_dim': 448,
        'weights': 'outdoor',
        'keypoint_encoder': [32, 64, 128, 256, 512],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self):
        super().__init__()

        self.config = {**self.default_config}

        self.descriptor_extract = ResNet(BasicBlock, [2, 2, 2, 2])
        # pretrained_dict = models.resnet18(pretrained=True).state_dict()
        # model_dict = self.descriptor_extract.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # self.descriptor_extract.load_state_dict(model_dict)
        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])
        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=2)
        self.relu = torch.nn.ReLU()
        self.scalex_proj = nn.Conv2d(in_channels=self.config['descriptor_dim'], out_channels=1, kernel_size=3,
                                     padding=1, stride=1, bias=True)
        self.compress_0 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, stride=1, bias=True)
        self.compress_1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=0, stride=1, bias=True)
        self.compress_2 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, padding=0, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        bin_score = torch.nn.Parameter(torch.tensor(0.0))
        self.register_parameter('bin_score', bin_score)
        assert self.config['weights'] in ['indoor', 'outdoor']
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.pos_encode = PositionEncodingSine(448)

    def forward(self, left_ori, right_ori, if_local):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        left = self.normalize(left_ori.permute(0, 3, 1, 2).float().contiguous())
        right = self.normalize(right_ori.permute(0, 3, 1, 2).float().contiguous())

        left1 = transforms.resize(left, [left.shape[2] // 2, left.shape[3] // 2])
        left2 = transforms.resize(left, [left.shape[2] // 4, left.shape[3] // 4])
        
        right1 = transforms.resize(right, [right.shape[2] // 2, right.shape[3] // 2])
        right2 = transforms.resize(right, [right.shape[2] // 4, right.shape[3] // 4])

        desc0_ = self.descriptor_extract(left)
        upsampling = nn.Upsample(size=[desc0_.shape[2], desc0_.shape[3]], mode='bilinear', align_corners=True)
        desc1_ = self.descriptor_extract(left1)
        desc2_ = self.descriptor_extract(left2)
        desc1_ = upsampling(desc1_)
        desc2_ = upsampling(desc2_)
        desc0_ = self.compress_0(desc0_)
        desc1_ = self.compress_1(desc1_)
        desc2_ = self.compress_2(desc2_)
        desc = torch.cat([desc0_, desc1_, desc2_], dim=1)

        # Keypoint MLP encoder.
        cols = torch.arange(0, desc.shape[2]).reshape(desc.shape[2], 1).repeat(1, desc.shape[3]).reshape(-1) / float(desc.shape[2])
        rows = torch.arange(0, desc.shape[3]).reshape(1, desc.shape[3]).repeat(desc.shape[2], 1).reshape(-1) / float(desc.shape[3])
        kpts  = torch.zeros((desc.shape[2] * desc.shape[3]), 2).to(left.device)
        kpts [:, 0] = cols
        kpts [:, 1] = rows

        desc0 = (desc + self.kenc(kpts).reshape(-1, desc.shape[2], desc.shape[3])).reshape(left.shape[0], self.config['descriptor_dim'], -1)

        desc0_ = self.descriptor_extract(right)
        upsampling = nn.Upsample(size=[desc0_.shape[2], desc0_.shape[3]], mode='bilinear', align_corners=True)
        desc1_ = self.descriptor_extract(right1)
        desc2_ = self.descriptor_extract(right2)
        desc1_ = upsampling(desc1_)
        desc2_ = upsampling(desc2_)
        desc0_ = self.compress_0(desc0_)
        desc1_ = self.compress_1(desc1_)
        desc2_ = self.compress_2(desc2_)
        desc = torch.cat([desc0_, desc1_, desc2_], dim=1)
        cols = torch.arange(0, desc.shape[2]).reshape(desc.shape[2], 1).repeat(1, desc.shape[3]).reshape(-1) / float(desc.shape[2])
        rows = torch.arange(0, desc.shape[3]).reshape(1, desc.shape[3]).repeat(desc.shape[2], 1).reshape(-1) / float(desc.shape[3])
        kpts  = torch.zeros((desc.shape[2] * desc.shape[3]), 2).to(left.device)
        kpts [:, 0] = cols
        kpts [:, 1] = rows
        # Keypoint MLP encoder.
        desc1 = (desc + self.kenc(kpts).reshape(-1, desc.shape[2], desc.shape[3])).reshape(left.shape[0], self.config['descriptor_dim'], -1)
        
        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        scale = self.scalex_proj(mdesc1[:, :, :].reshape(mdesc1.shape[0], -1, desc0_.shape[2], desc0_.shape[3])).reshape(right.shape[0], -1, desc0_.shape[2] * desc0_.shape[3])
        scale = torch.exp(self.sigmoid(scale) * math.log(256.0) - math.log(256.0) / 2)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim'] ** .5

        # Run the optimal transport.
        scores = log_optimal_transport( 0.1 * scores, self.bin_score.abs(),
                                       scale, iters=self.config['sinkhorn_iterations'])
            
        scales = scores[:, :-1, :-1].exp().sum(1)
        scales  = torch.sqrt(scales + 1e-8)

        patch_scale = 32
        trust_score, pts, x_scale_reproj, y_scale_reproj, if_nomatching1, if_nomatching2 = \
            self.est_position(scores, scales, left.shape[2:], patch_scale)

        height = left.shape[2] // 32
        width = left.shape[3] // 32
        left = left.permute(0, 2, 3, 1)
        right = right.permute(0, 2, 3, 1)

        first_nomatching_back_up = if_nomatching1.clone()
        sum_cycle = torch.cumsum(torch.logical_not(if_nomatching1).int(), dim=1)
        if if_local:
            max_cycle = width * 2
        else:
            max_cycle = 512
        cycle_num, second_layer_set, third_layer_set = split_patches(sum_cycle[0], height, width, max_cycle)
        output_list = []
        for num in range(cycle_num):
            first_nomatching_back_up = torch.where(torch.logical_and(if_nomatching1==False, 
                torch.logical_and(sum_cycle>second_layer_set[num][0], sum_cycle<=second_layer_set[num][1])), False, True)
            (new_left, new_right, x_scale_new, y_scale_new, average_new) = Compute_imgs(x_scale_reproj,
                y_scale_reproj, pts, first_nomatching_back_up, left_ori, right_ori, width=width, height=height)
            output_list.append({'patches': [new_left, new_right],
                'scales': [x_scale_new, y_scale_new],
                'pts_new': average_new,
                'if_nomatching1': first_nomatching_back_up,
                })
        return {
            'trust_score': trust_score,
            'scales_reproj': [x_scale_reproj, y_scale_reproj],
            'scores': scores,
            'features': [mdesc0, mdesc1],
            'pts': pts,
            'if_nomatching1': if_nomatching1,
            'if_nomatching2': if_nomatching2,
            'output_list': output_list,
            'third_layer_set': third_layer_set
        }

    def est_position(self, scores, scale_src, image_shape, patch_scale):
        H, W = image_shape
        scale_src = scale_src.reshape(scale_src.shape[0], -1, 1)
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
            Iterative_expand_matrix(scores.exp(), scale_src, scale_src, limitation1,
                ranges1, positions1, height=height, width=width, iter_num=15, lower_bound=1e-5)

        return trust_score, average_point1, x_scale, y_scale, if_nomatching1, if_nomatching2

if __name__ == '__main__':
    network = FirstLayer()
    image1 = torch.rand(1, 480, 640, 3)
    image2 = torch.rand(1, 480, 640, 3)
    predition = network(image1, image2)
