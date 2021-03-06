import math
import pickle
from turtle import forward
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import os

import numpy as np
from PIL import Image

from tqdm import tqdm
import argparse

from .painter import Painter


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

painter = Painter(args)


def save_img(img, output_path):
    result = Image.fromarray(
        (img.data.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))
    result.save(output_path)


def param2stroke(param, H, W, meta_brushes):
    """
    Input a set of stroke parameters and output its corresponding foregrounds and alpha maps.
    Args:
        param: a tensor with shape n_strokes x n_param_per_stroke. Here, param_per_stroke is 8:
        x_center, y_center, width, height, theta, R, G, and B.
        H: output height.
        W: output width.
        meta_brushes: a tensor with shape 2 x 3 x meta_brush_height x meta_brush_width.
         The first slice on the batch dimension denotes vertical brush and the second one denotes horizontal brush.

    Returns:
        foregrounds: a tensor with shape n_strokes x 3 x H x W, containing color information.
        alphas: a tensor with shape n_strokes x 3 x H x W,
         containing binary information of whether a pixel is belonging to the stroke (alpha mat), for painting process.
    """
    # global painter
    foreground, alphas = painter.render(param)
    foreground = F.interpolate(foreground, (H, W))
    alphas = F.interpolate(alphas, (H, W))
    return foreground.to(torch.float32), alphas.to(torch.float32)



def param2img_parallel(param, decision, meta_brushes, cur_canvas):
    """
        Input stroke parameters and decisions for each patch, meta brushes, current canvas, frame directory,
        and whether there is a border (if intermediate painting results are required).
        Output the painting results of adding the corresponding strokes on the current canvas.
        Args:
            param: a tensor with shape batch size x patch along height dimension x patch along width dimension
             x n_stroke_per_patch x n_param_per_stroke
            decision: a 01 tensor with shape batch size x patch along height dimension x patch along width dimension
             x n_stroke_per_patch
            meta_brushes: a tensor with shape 2 x 3 x meta_brush_height x meta_brush_width.
            The first slice on the batch dimension denotes vertical brush and the second one denotes horizontal brush.
            cur_canvas: a tensor with shape batch size x 3 x H x W,
             where H and W denote height and width of padded results of original images.

        Returns:
            cur_canvas: a tensor with shape batch size x 3 x H x W, denoting painting results.
        """
    # param: b, h, w, stroke_per_patch, param_per_stroke
    # decision: b, h, w, stroke_per_patch
    b, h, w, s, _ = param.shape
    param = param.view(-1, 12).contiguous()
    decision = decision.view(-1).contiguous().bool()
    H, W = cur_canvas.shape[-2:]
    is_odd_y = h % 2 == 1
    is_odd_x = w % 2 == 1
    patch_size_y = 2 * H // h
    patch_size_x = 2 * W // w
    even_idx_y = torch.arange(0, h, 2, device=cur_canvas.device)
    even_idx_x = torch.arange(0, w, 2, device=cur_canvas.device)
    odd_idx_y = torch.arange(1, h, 2, device=cur_canvas.device)
    odd_idx_x = torch.arange(1, w, 2, device=cur_canvas.device)
    even_y_even_x_coord_y, even_y_even_x_coord_x = \
        torch.meshgrid([even_idx_y, even_idx_x])
    odd_y_odd_x_coord_y, odd_y_odd_x_coord_x = \
        torch.meshgrid([odd_idx_y, odd_idx_x])
    even_y_odd_x_coord_y, even_y_odd_x_coord_x = \
        torch.meshgrid([even_idx_y, odd_idx_x])
    odd_y_even_x_coord_y, odd_y_even_x_coord_x = \
        torch.meshgrid([odd_idx_y, even_idx_x])
    cur_canvas = F.pad(cur_canvas, [patch_size_x // 4, patch_size_x // 4,
                                    patch_size_y // 4, patch_size_y // 4, 0, 0, 0, 0])
    foregrounds = torch.zeros(
        param.shape[0], 3, patch_size_y, patch_size_x,
        dtype=torch.float32, device=cur_canvas.device)
    alphas = torch.zeros(
        param.shape[0], 3, patch_size_y, patch_size_x,
        dtype=torch.float32, device=cur_canvas.device)
    # print(torch.cuda.memory_summary())
    valid_foregrounds, valid_alphas = param2stroke(
        param[decision, :], patch_size_y, patch_size_x, meta_brushes)
    foregrounds[decision, :, :, :] = valid_foregrounds
    alphas[decision, :, :, :] = valid_alphas
    # foreground, alpha: b * h * w * stroke_per_patch, 3, patch_size_y, patch_size_x
    foregrounds = foregrounds.view(
        -1, h, w, s, 3, patch_size_y, patch_size_x).contiguous()
    alphas = alphas.view(
        -1, h, w, s, 3, patch_size_y, patch_size_x).contiguous()
    # foreground, alpha: b, h, w, stroke_per_patch, 3, render_size_y, render_size_x
    decision = decision.view(-1, h, w, s, 1, 1, 1).contiguous()
    # decision: b, h, w, stroke_per_patch, 1, 1, 1

    def partial_render(this_canvas, patch_coord_y, patch_coord_x):

        canvas_patch = F.unfold(this_canvas, (patch_size_y, patch_size_x),
                                stride=(patch_size_y // 2, patch_size_x // 2))
        # canvas_patch: b, 3 * py * px, h * w
        canvas_patch = canvas_patch.view(
            b, 3, patch_size_y, patch_size_x, h, w).contiguous()
        canvas_patch = canvas_patch.permute(0, 4, 5, 1, 2, 3).contiguous()
        # canvas_patch: b, h, w, 3, py, px
        selected_canvas_patch = canvas_patch[:,
                                             patch_coord_y, patch_coord_x, :, :, :]
        selected_foregrounds = foregrounds[:,
                                           patch_coord_y, patch_coord_x, :, :, :, :]
        selected_alphas = alphas[:, patch_coord_y, patch_coord_x, :, :, :, :]
        selected_decisions = decision[:,
                                      patch_coord_y, patch_coord_x, :, :, :, :]
        for i in range(s):
            cur_foreground = selected_foregrounds[:, :, :, i, :, :, :]
            cur_alpha = selected_alphas[:, :, :, i, :, :, :]
            cur_decision = selected_decisions[:, :, :, i, :, :, :]
            selected_canvas_patch = cur_foreground * cur_alpha * cur_decision + selected_canvas_patch * (
                1 - cur_alpha * cur_decision)
        this_canvas = selected_canvas_patch.permute(
            0, 3, 1, 4, 2, 5).contiguous()
        # this_canvas: b, 3, h_half, py, w_half, px
        h_half = this_canvas.shape[2]
        w_half = this_canvas.shape[4]
        this_canvas = this_canvas.view(
            b, 3, h_half * patch_size_y, w_half * patch_size_x).contiguous()
        # this_canvas: b, 3, h_half * py, w_half * px
        return this_canvas

    if even_idx_y.shape[0] > 0 and even_idx_x.shape[0] > 0:
        canvas = partial_render(
            cur_canvas, even_y_even_x_coord_y, even_y_even_x_coord_x)
        if not is_odd_y:
            canvas = torch.cat(
                [canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if not is_odd_x:
            canvas = torch.cat(
                [canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if odd_idx_y.shape[0] > 0 and odd_idx_x.shape[0] > 0:
        canvas = partial_render(
            cur_canvas, odd_y_odd_x_coord_y, odd_y_odd_x_coord_x)
        canvas = torch.cat(
            [cur_canvas[:, :, :patch_size_y // 2, -canvas.shape[3]:], canvas], dim=2)
        canvas = torch.cat(
            [cur_canvas[:, :, -canvas.shape[2]:, :patch_size_x // 2], canvas], dim=3)
        if is_odd_y:
            canvas = torch.cat(
                [canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if is_odd_x:
            canvas = torch.cat(
                [canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if odd_idx_y.shape[0] > 0 and even_idx_x.shape[0] > 0:
        canvas = partial_render(
            cur_canvas, odd_y_even_x_coord_y, odd_y_even_x_coord_x)
        canvas = torch.cat(
            [cur_canvas[:, :, :patch_size_y // 2, :canvas.shape[3]], canvas], dim=2)
        if is_odd_y:
            canvas = torch.cat(
                [canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if not is_odd_x:
            canvas = torch.cat(
                [canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if even_idx_y.shape[0] > 0 and odd_idx_x.shape[0] > 0:
        canvas = partial_render(
            cur_canvas, even_y_odd_x_coord_y, even_y_odd_x_coord_x)
        canvas = torch.cat(
            [cur_canvas[:, :, :canvas.shape[2], :patch_size_x // 2], canvas], dim=3)
        if not is_odd_y:
            canvas = torch.cat(
                [canvas, cur_canvas[:, :, -patch_size_y // 2:, -canvas.shape[3]:]], dim=2)
        if is_odd_x:
            canvas = torch.cat(
                [canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    cur_canvas = cur_canvas[:, :, patch_size_y // 4:-
                            patch_size_y // 4, patch_size_x // 4:-patch_size_x // 4]
    # print(torch.cuda.memory_summary())
    return cur_canvas


def read_img(img_path, img_type='RGB', h=None, w=None):
    img = Image.open(img_path).convert(img_type)
    if h is not None and w is not None:
        img = img.resize((w, h), resample=Image.NEAREST)
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = img.transpose((2, 0, 1)).astype(np.float32) / 255
    img = torch.from_numpy(img).unsqueeze(0)
    return img


def crop(img, h, w):
    H, W = img.shape[-2:]
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    remainder_h = (H - h) % 2
    remainder_w = (W - w) % 2
    img = img[:, :, pad_h:H - pad_h - remainder_h,
              pad_w:W - pad_w - remainder_w]
    return img


class Render(nn.Module):
    def __init__(self, strokes):
        super().__init__()
        self.patch_size = 32
        self.original_h = 512
        self.original_w = 512
        self.K = max(math.ceil(math.log2(max(self.original_h, self.original_w) / self.patch_size)), 0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = [p for p,d in strokes]
        self.decision = [d for p,d in strokes]

        brush_large_vertical = read_img( os.path.join(os.path.dirname(__file__),
            'brush', "brush_large_vertical.png"), 'L').to(self.device)
        brush_large_horizontal = read_img(os.path.join(os.path.dirname(__file__),
            'brush', "brush_large_horizontal.png"), 'L').to(self.device)
        self.meta_brushes = torch.cat(
            [brush_large_vertical, brush_large_horizontal], dim=0)

    def forward(self, params):
        original_img_pad_size = self.patch_size * (2 ** self.K)
        final_result = torch.zeros(
            (1, 3, original_img_pad_size, original_img_pad_size)).to(self.device)
        for layer in range(0, self.K + 1):
            param, decision = params[layer](), self.decision[layer]
            final_result = param2img_parallel(
                param, decision, self.meta_brushes, final_result)
            # print(final_result.mean())

        layer_size = self.patch_size * (2 ** self.K)
        # There are patch_num * patch_num patches in total
        patch_num = (layer_size - self.patch_size) // self.patch_size + 1
        border_size = original_img_pad_size // (2 * patch_num)
        final_result = F.pad(
            final_result, [border_size, border_size, border_size, border_size, 0, 0, 0, 0])
        param, decision = params[-1](), self.decision[-1]
        final_result = param2img_parallel(
            param, decision, self.meta_brushes, final_result)
        final_result = final_result[:, :, border_size:-
                                    border_size, border_size:-border_size]
        final_result = crop(final_result, self.original_h, self.original_w)
        return final_result[0]

class Stroke(nn.Module):
    def __init__(self, stroke):
        super().__init__()
        self.stroke = nn.Parameter(stroke)
    
    def forward(self):
        return self.stroke

class Strokes(nn.Module):
    def __init__(self, strokes):
        super().__init__()
        strokes = [p for p, d in strokes]
        self.strokes = nn.ModuleList(
            [Stroke(stroke) for stroke in strokes]
        )
    
    def forward(self):
        return self.strokes

if __name__ == '__main__':
    strokes: List[Tuple[Tensor, Tensor]] = \
        pickle.load(open("strokes.pkl", "rb"))
    r = Render(strokes)
    s = Strokes(strokes)
    result = r(s())
    print(result.shape)
    save_img(result, "output.png")