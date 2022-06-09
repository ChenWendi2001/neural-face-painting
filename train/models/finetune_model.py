import torch
import numpy as np
from .base_model import BaseModel
from . import networks
from util import morphology
from scipy.optimize import linear_sum_assignment
from PIL import Image

from torchvision.utils import save_image

class FinetuneModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='null')
        parser.add_argument('--used_strokes', type=int, default=8,
                            help='actually generated strokes number')
        parser.add_argument('--num_blocks', type=int, default=3,
                            help='number of transformer blocks for stroke generator')
        parser.add_argument('--lambda_stroke', type=float, default=0.01, help='weight for w loss of stroke shape')
        parser.add_argument('--lambda_pixel', type=float, default=100.0, help='weight for pixel-level L1 loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['pixel', 'stroke']
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
        self.net_g = networks.Painter(self.d_shape, opt.used_strokes, opt.ngf,
                                 n_enc_layers=opt.num_blocks, n_dec_layers=opt.num_blocks)
        self.net_g.load_state_dict(torch.load("model.pth"))
        self.net_g.to(self.gpu_ids[0])
        self.net_g = torch.nn.DataParallel(self.net_g, self.gpu_ids)
        self.old = None
        self.render = None
        self.rec = None
        self.pred_param = None
        self.pred_decision = None
        self.patch_size = 32
        self.loss_pixel = torch.tensor(0., device=self.device)
        self.loss_stroke = torch.tensor(0., device=self.device)
        self.criterion_pixel = torch.nn.L1Loss().to(self.device)
        if self.isTrain:
            self.optimizer = torch.optim.Adam(self.net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

    def param2stroke(self, param, H, W):
        # param: b, 12
        b = param.shape[0]
        param_list = torch.split(param, 1, dim=1)
        x0, y0, w, h, theta = [item.squeeze(-1) for item in param_list[:5]]

        R0, G0, B0, R2, G2, B2, _ = param_list[5:]
        sin_theta = torch.sin(torch.acos(torch.tensor(-1., device=param.device)) * theta)
        cos_theta = torch.cos(torch.acos(torch.tensor(-1., device=param.device)) * theta)
        index = torch.full((b,), -1, device=param.device)
        index[h > w] = 0
        index[h <= w] = 1
        brush = self.meta_brushes[index.long()]
        alphas = torch.cat([brush, brush, brush], dim=1)
        alphas = (alphas > 0).float()
        t = torch.arange(0, brush.shape[2], device=param.device).unsqueeze(0) / brush.shape[2]
        color_map = torch.stack([R0 * (1 - t) + R2 * t, G0 * (1 - t) + G2 * t, B0 * (1 - t) + B2 * t], dim=1)
        color_map = color_map.unsqueeze(-1).repeat(1, 1, 1, brush.shape[3])
        brush = brush * color_map

        warp_00 = cos_theta / w
        warp_01 = sin_theta * H / (W * w)
        warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
        warp_10 = -sin_theta * W / (H * h)
        warp_11 = cos_theta / h
        warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
        warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
        warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
        warp = torch.stack([warp_0, warp_1], dim=1)
        
        grid = torch.nn.functional.affine_grid(warp, torch.Size((b, 3, H, W)), align_corners=False)
        brush = torch.nn.functional.grid_sample(brush, grid, align_corners=False)
        alphas = torch.nn.functional.grid_sample(alphas, grid, align_corners=False)

        return brush, alphas

    def set_input(self, input_dict):
        self.render = input_dict['img'].cuda()
        self.hog = input_dict['hog'].cuda()
        self.old = torch.zeros_like(self.render)

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

    def criterion_stroke(self, stroke, hog):
        with torch.no_grad():
            x = stroke[:, :, 0] * self.patch_size
            y = stroke[:, :, 1] * self.patch_size
            x = x.long()
            y = y.long()
        
            selected_hog = torch.zeros([x.shape[0], x.shape[1], hog.shape[-1]]).to(x.device)
            for i in range(x.shape[0]):
                selected_hog[i, :, :] = hog[i, y[i], x[i], :]
            selected_hog = selected_hog.reshape(-1, hog.shape[-1])

        w = stroke[:,:,2].flatten()
        h = stroke[:,:,3].flatten()
        theta = stroke[:,:,4].flatten().unsqueeze(-1)

        temp = selected_hog[h>w,:6]
        selected_hog[h>w, :6] = selected_hog[h>w, 6:]
        selected_hog[h>w, 6:] = temp

        print(torch.sum(selected_hog, axis=1))
        
        angle = torch.arange(12).to(x.device) / 12
        angle = angle.unsqueeze(0)
        angle = angle - theta

        cos_angle = torch.cos(torch.acos(torch.tensor(-1., device=x.device)) * angle) + 1
        weighted_cos_angle = cos_angle * selected_hog
        loss = torch.sum(weighted_cos_angle)

        return loss

    def optimize_parameters(self):
        self.forward()
        self.loss_pixel = self.criterion_pixel(self.rec, self.render) * self.opt.lambda_pixel
        self.loss_stroke = self.criterion_stroke(self.pred_param, self.hog) * self.opt.lambda_stroke
        # loss = self.loss_pixel + self.loss_stroke
        loss = self.loss_stroke
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
