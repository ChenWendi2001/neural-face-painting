import torch
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel
from . import networks
from util import morphology
from scipy.optimize import linear_sum_assignment
from PIL import Image

from .pseudo_render.painter import Painter as DiffPainter

import argparse
parser = argparse.ArgumentParser(description='STYLIZED NEURAL PAINTING')
parser.add_argument('--renderer', type=str, default='oilpaintbrush', metavar='str',
                    help='renderer: [watercolor, markerpen, oilpaintbrush, rectangle (default oilpaintbrush)')
parser.add_argument('--transfer_mode', type=int, default=1, metavar='N',
                    help='style transfer mode, 0: transfer color only, 1: transfer both color and texture, '
                         'defalt: 1')
parser.add_argument('--canvas_color', type=str, default='black', metavar='str',
                    help='canvas_color: [black, white] (default black)')
parser.add_argument('--canvas_size', type=int, default=512, metavar='str',
                    help='size of the canvas for stroke rendering')
parser.add_argument('--keep_aspect_ratio', action='store_true', default=False,
                    help='keep input aspect ratio when saving outputs')
parser.add_argument('--beta_L1', type=float, default=1.0,
                    help='weight for L1 loss (default: 1.0)')
parser.add_argument('--net_G', type=str, default='zou-fusion-net-light', metavar='str',
                    help='net_G: plain-dcgan, plain-unet, huang-net, zou-fusion-net, '
                         'or zou-fusion-net-light (default: zou-fusion-net-light)')
parser.add_argument('--renderer_checkpoint_dir', type=str, default=r'checkpoints_G_oilpaintbrush_light', metavar='str',
                    help='dir to load neu-renderer (default: ./checkpoints_G_oilpaintbrush_light)')
args = parser.parse_args(args=[])


class DiffpainterModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='null')
        parser.add_argument('--used_strokes', type=int, default=8,
                            help='actually generated strokes number')
        parser.add_argument('--num_blocks', type=int, default=3,
                            help='number of transformer blocks for stroke generator')
        parser.add_argument('--lambda_w', type=float, default=10.0, help='weight for w loss of stroke shape')
        parser.add_argument('--lambda_pixel', type=float, default=10.0, help='weight for pixel-level L1 loss')
        parser.add_argument('--lambda_gt', type=float, default=1.0, help='weight for ground-truth loss')
        parser.add_argument('--lambda_decision', type=float, default=10.0, help='weight for stroke decision loss')
        parser.add_argument('--lambda_recall', type=float, default=10.0, help='weight of recall for stroke decision loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['pixel', 'gt', 'w', 'decision']
        self.visual_names = ['old', 'render', 'rec']
        self.model_names = ['g']
        self.d = 12  # xc, yc, w, h, theta, R0, G0, B0, R2, G2, B2, A
        self.d_shape = 5

        def read_img(img_path, img_type='RGB'):
            img = Image.open(img_path).convert(img_type)
            img = np.array(img)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0).float() / 255.
            return img

        brush_large_vertical = read_img('brush/brush_large_vertical.png', 'L').to(self.device)
        brush_large_horizontal = read_img('brush/brush_large_horizontal.png', 'L').to(self.device)
        self.meta_brushes = torch.cat(
            [brush_large_vertical, brush_large_horizontal], dim=0)
        net_g = networks.Painter(self.d_shape, opt.used_strokes, opt.ngf,
                                 n_enc_layers=opt.num_blocks, n_dec_layers=opt.num_blocks)
        self.net_g = networks.init_net(net_g, opt.init_type, opt.init_gain, self.gpu_ids)
        self.old = None
        self.render = None
        self.rec = None
        self.gt_param = None
        self.pred_param = None
        self.gt_decision = None
        self.pred_decision = None
        self.patch_size = 32
        self.loss_pixel = torch.tensor(0., device=self.device)
        self.loss_gt = torch.tensor(0., device=self.device)
        self.loss_w = torch.tensor(0., device=self.device)
        self.loss_decision = torch.tensor(0., device=self.device)
        self.criterion_pixel = torch.nn.L1Loss().to(self.device)
        self.criterion_decision = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(opt.lambda_recall)).to(self.device)
        if self.isTrain:
            self.optimizer = torch.optim.Adam(self.net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
        
        self.painter = DiffPainter(args)

    def param2stroke(self, param, H, W):
        # param: b, 12
        foreground, alphas = self.painter.render(param)
        foreground = F.interpolate(foreground, (H, W))
        alphas = F.interpolate(alphas, (H, W))
        return foreground, alphas

    def set_input(self, input_dict):
        self.image_paths = input_dict['A_paths']
        with torch.no_grad():
            old_param = torch.rand(self.opt.batch_size // 4, self.opt.used_strokes, self.d, device=self.device)
            old_param[:, :, :4] = old_param[:, :, :4] * 0.5 + 0.2
            old_param[:, :, -4:-1] = old_param[:, :, -7:-4]
            old_param = old_param.view(-1, self.d).contiguous()
            foregrounds, alphas = self.param2stroke(old_param, self.patch_size * 2, self.patch_size * 2)
            foregrounds = morphology.Dilation2d(m=1)(foregrounds)
            alphas = morphology.Erosion2d(m=1)(alphas)
            foregrounds = foregrounds.view(self.opt.batch_size // 4, self.opt.used_strokes, 3, self.patch_size * 2,
                                           self.patch_size * 2).contiguous()
            alphas = alphas.view(self.opt.batch_size // 4, self.opt.used_strokes, 3, self.patch_size * 2,
                                 self.patch_size * 2).contiguous()
            old = torch.zeros(self.opt.batch_size // 4, 3, self.patch_size * 2, self.patch_size * 2, device=self.device)
            for i in range(self.opt.used_strokes):
                foreground = foregrounds[:, i, :, :, :]
                alpha = alphas[:, i, :, :, :]
                old = foreground * alpha + old * (1 - alpha)
            old = old.view(self.opt.batch_size // 4, 3, 2, self.patch_size, 2, self.patch_size).contiguous()
            old = old.permute(0, 2, 4, 1, 3, 5).contiguous()
            self.old = old.view(self.opt.batch_size, 3, self.patch_size, self.patch_size).contiguous()

            gt_param = torch.rand(self.opt.batch_size, self.opt.used_strokes, self.d, device=self.device)
            gt_param[:, :, :4] = gt_param[:, :, :4] * 0.5 + 0.2
            gt_param[:, :, -4:-1] = gt_param[:, :, -7:-4]
            self.gt_param = gt_param[:, :, :self.d_shape]
            gt_param = gt_param.view(-1, self.d).contiguous()
            foregrounds, alphas = self.param2stroke(gt_param, self.patch_size, self.patch_size)
            foregrounds = morphology.Dilation2d(m=1)(foregrounds)
            alphas = morphology.Erosion2d(m=1)(alphas)
            foregrounds = foregrounds.view(self.opt.batch_size, self.opt.used_strokes, 3, self.patch_size,
                                           self.patch_size).contiguous()
            alphas = alphas.view(self.opt.batch_size, self.opt.used_strokes, 3, self.patch_size,
                                 self.patch_size).contiguous()
            self.render = self.old.clone()
            gt_decision = torch.ones(self.opt.batch_size, self.opt.used_strokes, device=self.device)
            for i in range(self.opt.used_strokes):
                foreground = foregrounds[:, i, :, :, :]
                alpha = alphas[:, i, :, :, :]
                for j in range(i):
                    iou = (torch.sum(alpha * alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5) / (
                            torch.sum(alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5)
                    gt_decision[:, i] = ((iou < 0.75) | (~gt_decision[:, j].bool())).float() * gt_decision[:, i]
                decision = gt_decision[:, i].view(self.opt.batch_size, 1, 1, 1).contiguous()
                self.render = foreground * alpha * decision + self.render * (1 - alpha * decision)
            self.gt_decision = gt_decision

    def forward(self):
        param, decisions = self.net_g(self.render, self.old)
        # stroke_param: b, stroke_per_patch, param_per_stroke
        # decision: b, stroke_per_patch, 1
        self.pred_decision = decisions.view(-1, self.opt.used_strokes).contiguous()
        self.pred_param = param[:, :, :self.d_shape]
        param = param.view(-1, self.d).contiguous()
        foregrounds, alphas = self.param2stroke(param, self.patch_size, self.patch_size)
        foregrounds = morphology.Dilation2d(m=1)(foregrounds)
        alphas = morphology.Erosion2d(m=1)(alphas)
        # foreground, alpha: b * stroke_per_patch, 3, output_size, output_size
        foregrounds = foregrounds.view(-1, self.opt.used_strokes, 3, self.patch_size, self.patch_size)
        alphas = alphas.view(-1, self.opt.used_strokes, 3, self.patch_size, self.patch_size)
        # foreground, alpha: b, stroke_per_patch, 3, output_size, output_size
        decisions = networks.SignWithSigmoidGrad.apply(decisions.view(-1, self.opt.used_strokes, 1, 1, 1).contiguous())
        self.rec = self.old.clone()
        for j in range(foregrounds.shape[1]):
            foreground = foregrounds[:, j, :, :, :]
            alpha = alphas[:, j, :, :, :]
            decision = decisions[:, j, :, :, :]
            self.rec = foreground * alpha * decision + self.rec * (1 - alpha * decision)

    @staticmethod
    def get_sigma_sqrt(w, h, theta):
        sigma_00 = w * (torch.cos(theta) ** 2) / 2 + h * (torch.sin(theta) ** 2) / 2
        sigma_01 = (w - h) * torch.cos(theta) * torch.sin(theta) / 2
        sigma_11 = h * (torch.cos(theta) ** 2) / 2 + w * (torch.sin(theta) ** 2) / 2
        sigma_0 = torch.stack([sigma_00, sigma_01], dim=-1)
        sigma_1 = torch.stack([sigma_01, sigma_11], dim=-1)
        sigma = torch.stack([sigma_0, sigma_1], dim=-2)
        return sigma

    @staticmethod
    def get_sigma(w, h, theta):
        sigma_00 = w * w * (torch.cos(theta) ** 2) / 4 + h * h * (torch.sin(theta) ** 2) / 4
        sigma_01 = (w * w - h * h) * torch.cos(theta) * torch.sin(theta) / 4
        sigma_11 = h * h * (torch.cos(theta) ** 2) / 4 + w * w * (torch.sin(theta) ** 2) / 4
        sigma_0 = torch.stack([sigma_00, sigma_01], dim=-1)
        sigma_1 = torch.stack([sigma_01, sigma_11], dim=-1)
        sigma = torch.stack([sigma_0, sigma_1], dim=-2)
        return sigma

    def gaussian_w_distance(self, param_1, param_2):
        mu_1, w_1, h_1, theta_1 = torch.split(param_1, (2, 1, 1, 1), dim=-1)
        w_1 = w_1.squeeze(-1)
        h_1 = h_1.squeeze(-1)
        theta_1 = torch.acos(torch.tensor(-1., device=param_1.device)) * theta_1.squeeze(-1)
        trace_1 = (w_1 ** 2 + h_1 ** 2) / 4
        mu_2, w_2, h_2, theta_2 = torch.split(param_2, (2, 1, 1, 1), dim=-1)
        w_2 = w_2.squeeze(-1)
        h_2 = h_2.squeeze(-1)
        theta_2 = torch.acos(torch.tensor(-1., device=param_2.device)) * theta_2.squeeze(-1)
        trace_2 = (w_2 ** 2 + h_2 ** 2) / 4
        sigma_1_sqrt = self.get_sigma_sqrt(w_1, h_1, theta_1)
        sigma_2 = self.get_sigma(w_2, h_2, theta_2)
        trace_12 = torch.matmul(torch.matmul(sigma_1_sqrt, sigma_2), sigma_1_sqrt)
        trace_12 = torch.sqrt(trace_12[..., 0, 0] + trace_12[..., 1, 1] + 2 * torch.sqrt(
            trace_12[..., 0, 0] * trace_12[..., 1, 1] - trace_12[..., 0, 1] * trace_12[..., 1, 0]))
        return torch.sum((mu_1 - mu_2) ** 2, dim=-1) + trace_1 + trace_2 - 2 * trace_12

    def optimize_parameters(self):
        self.forward()
        self.loss_pixel = self.criterion_pixel(self.rec, self.render) * self.opt.lambda_pixel
        cur_valid_gt_size = 0
        with torch.no_grad():
            r_idx = []
            c_idx = []
            for i in range(self.gt_param.shape[0]):
                is_valid_gt = self.gt_decision[i].bool()
                valid_gt_param = self.gt_param[i, is_valid_gt]
                cost_matrix_l1 = torch.cdist(self.pred_param[i], valid_gt_param, p=1)
                pred_param_broad = self.pred_param[i].unsqueeze(1).contiguous().repeat(
                    1, valid_gt_param.shape[0], 1)
                valid_gt_param_broad = valid_gt_param.unsqueeze(0).contiguous().repeat(
                    self.pred_param.shape[1], 1, 1)
                cost_matrix_w = self.gaussian_w_distance(pred_param_broad, valid_gt_param_broad)
                decision = self.pred_decision[i]
                cost_matrix_decision = (1 - decision).unsqueeze(-1).repeat(1, valid_gt_param.shape[0])
                r, c = linear_sum_assignment((cost_matrix_l1 + cost_matrix_w + cost_matrix_decision).cpu())
                r_idx.append(torch.tensor(r + self.pred_param.shape[1] * i, device=self.device))
                c_idx.append(torch.tensor(c + cur_valid_gt_size, device=self.device))
                cur_valid_gt_size += valid_gt_param.shape[0]
            r_idx = torch.cat(r_idx, dim=0)
            c_idx = torch.cat(c_idx, dim=0)
            paired_gt_decision = torch.zeros(self.gt_decision.shape[0] * self.gt_decision.shape[1], device=self.device)
            paired_gt_decision[r_idx] = 1.
        all_valid_gt_param = self.gt_param[self.gt_decision.bool(), :]
        all_pred_param = self.pred_param.view(-1, self.pred_param.shape[2]).contiguous()
        all_pred_decision = self.pred_decision.view(-1).contiguous()
        paired_gt_param = all_valid_gt_param[c_idx, :]
        paired_pred_param = all_pred_param[r_idx, :]
        self.loss_gt = self.criterion_pixel(paired_pred_param, paired_gt_param) * self.opt.lambda_gt
        self.loss_w = self.gaussian_w_distance(paired_pred_param, paired_gt_param).mean() * self.opt.lambda_w
        self.loss_decision = self.criterion_decision(all_pred_decision, paired_gt_decision) * self.opt.lambda_decision
        loss = self.loss_pixel + self.loss_gt + self.loss_w + self.loss_decision
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
