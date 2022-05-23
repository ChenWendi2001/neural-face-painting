import math
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import morphology
import network

from pseudo_render.painter import Painter as DiffPainter

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
painter = DiffPainter(args)

def save_img(img, output_path):
    result = Image.fromarray((img.data.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))
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
    # Firstly, resize the meta brushes to the required shape,
    # in order to decrease GPU memory especially when the required shape is small.
    param = F.pad(param, pad=[0, 4, 0, 0])
    print(param.shape)

    param[...,8:11] = param[...,5:8]

    foreground, alphas = painter.render(param)
    foreground = F.interpolate(foreground, (H, W))
    alphas = F.interpolate(alphas, (H, W))
    return foreground, alphas


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
    param = param.view(-1, 8).contiguous()
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
    even_y_even_x_coord_y, even_y_even_x_coord_x = torch.meshgrid([even_idx_y, even_idx_x])
    odd_y_odd_x_coord_y, odd_y_odd_x_coord_x = torch.meshgrid([odd_idx_y, odd_idx_x])
    even_y_odd_x_coord_y, even_y_odd_x_coord_x = torch.meshgrid([even_idx_y, odd_idx_x])
    odd_y_even_x_coord_y, odd_y_even_x_coord_x = torch.meshgrid([odd_idx_y, even_idx_x])
    cur_canvas = F.pad(cur_canvas, [patch_size_x // 4, patch_size_x // 4,
                                    patch_size_y // 4, patch_size_y // 4, 0, 0, 0, 0])
    foregrounds = torch.zeros(
        param.shape[0], 3, patch_size_y, patch_size_x,
        dtype=torch.float32, device=cur_canvas.device)
    alphas = torch.zeros(
        param.shape[0], 3, patch_size_y, patch_size_x,
        dtype=torch.float32, device=cur_canvas.device)
    # print(torch.cuda.memory_summary())
    valid_foregrounds, valid_alphas = param2stroke(param[decision, :], patch_size_y, patch_size_x, meta_brushes)
    foregrounds[decision, :, :, :] = valid_foregrounds
    alphas[decision, :, :, :] = valid_alphas
    # foreground, alpha: b * h * w * stroke_per_patch, 3, patch_size_y, patch_size_x
    foregrounds = foregrounds.view(-1, h, w, s, 3, patch_size_y, patch_size_x).contiguous()
    alphas = alphas.view(-1, h, w, s, 3, patch_size_y, patch_size_x).contiguous()
    # foreground, alpha: b, h, w, stroke_per_patch, 3, render_size_y, render_size_x
    decision = decision.view(-1, h, w, s, 1, 1, 1).contiguous()
    # decision: b, h, w, stroke_per_patch, 1, 1, 1

    def partial_render(this_canvas, patch_coord_y, patch_coord_x):

        canvas_patch = F.unfold(this_canvas, (patch_size_y, patch_size_x),
                                stride=(patch_size_y // 2, patch_size_x // 2))
        # canvas_patch: b, 3 * py * px, h * w
        canvas_patch = canvas_patch.view(b, 3, patch_size_y, patch_size_x, h, w).contiguous()
        canvas_patch = canvas_patch.permute(0, 4, 5, 1, 2, 3).contiguous()
        # canvas_patch: b, h, w, 3, py, px
        selected_canvas_patch = canvas_patch[:, patch_coord_y, patch_coord_x, :, :, :]
        selected_foregrounds = foregrounds[:, patch_coord_y, patch_coord_x, :, :, :, :]
        selected_alphas = alphas[:, patch_coord_y, patch_coord_x, :, :, :, :]
        selected_decisions = decision[:, patch_coord_y, patch_coord_x, :, :, :, :]
        for i in range(s):
            cur_foreground = selected_foregrounds[:, :, :, i, :, :, :]
            cur_alpha = selected_alphas[:, :, :, i, :, :, :]
            cur_decision = selected_decisions[:, :, :, i, :, :, :]
            selected_canvas_patch = cur_foreground * cur_alpha * cur_decision + \
                selected_canvas_patch * (1 - cur_alpha * cur_decision)
        this_canvas = selected_canvas_patch.permute(0, 3, 1, 4, 2, 5).contiguous()
        # this_canvas: b, 3, h_half, py, w_half, px
        h_half = this_canvas.shape[2]
        w_half = this_canvas.shape[4]
        this_canvas = this_canvas.view(b, 3, h_half * patch_size_y, w_half * patch_size_x).contiguous()
        # this_canvas: b, 3, h_half * py, w_half * px
        return this_canvas

    if even_idx_y.shape[0] > 0 and even_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, even_y_even_x_coord_y, even_y_even_x_coord_x)
        if not is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if not is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if odd_idx_y.shape[0] > 0 and odd_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, odd_y_odd_x_coord_y, odd_y_odd_x_coord_x)
        canvas = torch.cat([cur_canvas[:, :, :patch_size_y // 2, -canvas.shape[3]:], canvas], dim=2)
        canvas = torch.cat([cur_canvas[:, :, -canvas.shape[2]:, :patch_size_x // 2], canvas], dim=3)
        if is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if odd_idx_y.shape[0] > 0 and even_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, odd_y_even_x_coord_y, odd_y_even_x_coord_x)
        canvas = torch.cat([cur_canvas[:, :, :patch_size_y // 2, :canvas.shape[3]], canvas], dim=2)
        if is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if not is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if even_idx_y.shape[0] > 0 and odd_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, even_y_odd_x_coord_y, even_y_odd_x_coord_x)
        canvas = torch.cat([cur_canvas[:, :, :canvas.shape[2], :patch_size_x // 2], canvas], dim=3)
        if not is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, -canvas.shape[3]:]], dim=2)
        if is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    cur_canvas = cur_canvas[:, :, patch_size_y // 4:-patch_size_y // 4, patch_size_x // 4:-patch_size_x // 4]
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


def pad(img: torch.Tensor, H, W):
    b, c, h, w = img.shape
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    remainder_h = (H - h) % 2
    remainder_w = (W - w) % 2
    img = torch.cat([torch.zeros((b, c, pad_h, w), dtype=img.dtype, device=img.device), img,
                     torch.zeros((b, c, pad_h + remainder_h, w), dtype=img.dtype, device=img.device)], dim=-2)
    img = torch.cat([torch.zeros((b, c, H, pad_w), dtype=img.dtype, device=img.device), img,
                     torch.zeros((b, c, H, pad_w + remainder_w), dtype=img.dtype, device=img.device)], dim=-1)
    return img


def crop(img, h, w):
    H, W = img.shape[-2:]
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    remainder_h = (H - h) % 2
    remainder_w = (W - w) % 2
    img = img[:, :, pad_h:H - pad_h - remainder_h, pad_w:W - pad_w - remainder_w]
    return img


def main(input_path, model_path, output_dir, resize_h=None, resize_w=None):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    input_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, input_name)
    patch_size = 32
    stroke_num = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_g = network.Painter(5, stroke_num, 256, 8, 3, 3).to(device)
    net_g.load_state_dict(torch.load(model_path))
    net_g.eval()
    for param in net_g.parameters():
        param.requires_grad = False

    brush_large_vertical = read_img('brush/brush_large_vertical.png', 'L').to(device)
    brush_large_horizontal = read_img('brush/brush_large_horizontal.png', 'L').to(device)
    meta_brushes = torch.cat(
        [brush_large_vertical, brush_large_horizontal], dim=0)

    strokes = []

    with torch.no_grad():
        original_img = read_img(input_path, 'RGB', resize_h, resize_w).to(device)
        original_h, original_w = original_img.shape[-2:]
        K = max(math.ceil(math.log2(max(original_h, original_w) / patch_size)), 0)
        original_img_pad_size = patch_size * (2 ** K)
        original_img_pad = pad(original_img, original_img_pad_size, original_img_pad_size)
        final_result = torch.zeros_like(original_img_pad).to(device)
        print(f"[INFO]: {K + 1} layers")
        for layer in range(0, K + 1):
            layer_size = patch_size * (2 ** layer)
            img = F.interpolate(original_img_pad, (layer_size, layer_size))
            result = F.interpolate(final_result, (patch_size * (2 ** layer), patch_size * (2 ** layer)))
            img_patch = F.unfold(img, (patch_size, patch_size), stride=(patch_size, patch_size))
            result_patch = F.unfold(result, (patch_size, patch_size), stride=(patch_size, patch_size))
            # There are patch_num * patch_num patches in total
            patch_num = (layer_size - patch_size) // patch_size + 1

            # img_patch, result_patch: b, 3 * output_size * output_size, h * w
            img_patch = img_patch.permute(0, 2, 1).contiguous().view(-1, 3, patch_size, patch_size).contiguous()
            result_patch = result_patch.permute(0, 2, 1).contiguous().view(
                -1, 3, patch_size, patch_size).contiguous()
            
            n_patch = img_patch.shape[0]
            shape_param = torch.zeros([n_patch, 8, 5], dtype=torch.float32).cuda()
            stroke_decision = torch.zeros(
                [n_patch, 8, 1], dtype=torch.float32).cuda()
            batch_size = 128
            for index in tqdm(range(0, n_patch, batch_size), desc=f"Layer {layer}"):
                last = min(n_patch, index + batch_size)
                input_img = img_patch[index: last, ...]
                input_result = result_patch[index: last, ...]
                output_param, output_decision = \
                    net_g(input_img.float(), input_result.float())
                shape_param[index: last, ...] = output_param
                stroke_decision[index: last, ...] = output_decision
            # shape_param, stroke_decision = net_g(img_patch, result_patch)
            shape_param = shape_param.contiguous().to(torch.float32)
            stroke_decision = stroke_decision.contiguous()
            stroke_decision = network.SignWithSigmoidGrad.apply(stroke_decision)

            grid = shape_param[:, :, :2].view(img_patch.shape[0] * stroke_num, 1, 1, 2).contiguous()
            img_temp = img_patch.unsqueeze(1).contiguous().repeat(1, stroke_num, 1, 1, 1).view(
                img_patch.shape[0] * stroke_num, 3, patch_size, patch_size).contiguous()
            color = F.grid_sample(img_temp, 2 * grid - 1, align_corners=False).view(
                img_patch.shape[0], stroke_num, 3).contiguous()
            stroke_param = torch.cat([shape_param, color], dim=-1)
            # stroke_param: b * h * w, stroke_per_patch, param_per_stroke
            # stroke_decision: b * h * w, stroke_per_patch, 1
            param = stroke_param.view(1, patch_num, patch_num, stroke_num, 8).contiguous()
            decision = stroke_decision.view(1, patch_num, patch_num, stroke_num).contiguous().bool()
            # param: b, h, w, stroke_per_patch, 8
            # decision: b, h, w, stroke_per_patch
            param[..., :2] = param[..., :2] / 2 + 0.25
            param[..., 2:4] = param[..., 2:4] / 2

            # NOTE: save strokes
            strokes.append((param, decision))
            final_result = param2img_parallel(
                param, decision, meta_brushes, final_result)
            # print(final_result.mean())

        border_size = original_img_pad_size // (2 * patch_num)
        img = F.interpolate(original_img_pad, (patch_size * (2 ** layer), patch_size * (2 ** layer)))
        result = F.interpolate(final_result, (patch_size * (2 ** layer), patch_size * (2 ** layer)))
        img = F.pad(img, [patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2,
                        0, 0, 0, 0])
        result = F.pad(result, [patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2,
                                0, 0, 0, 0])
        img_patch = F.unfold(img, (patch_size, patch_size), stride=(patch_size, patch_size))
        result_patch = F.unfold(result, (patch_size, patch_size), stride=(patch_size, patch_size))
        final_result = F.pad(final_result, [border_size, border_size, border_size, border_size, 0, 0, 0, 0])
        h = (img.shape[2] - patch_size) // patch_size + 1
        w = (img.shape[3] - patch_size) // patch_size + 1
        # img_patch, result_patch: b, 3 * output_size * output_size, h * w
        img_patch = img_patch.permute(0, 2, 1).contiguous().view(-1, 3, patch_size, patch_size).contiguous()
        result_patch = result_patch.permute(0, 2, 1).contiguous().view(-1, 3, patch_size, patch_size).contiguous()
        
        # shape_param, stroke_decision = net_g(img_patch, result_patch)
        n_patch = img_patch.shape[0]
        shape_param = torch.zeros([n_patch, 8, 5], dtype=torch.float32).cuda()
        stroke_decision = torch.zeros(
            [n_patch, 8, 1], dtype=torch.float32).cuda()
        batch_size = 128
        for index in tqdm(range(0, n_patch, batch_size), desc=f"Final"):
            last = min(n_patch, index + batch_size)
            input_img = img_patch[index: last, ...]
            input_result = result_patch[index: last, ...]
            output_param, output_decision = \
                net_g(input_img.float(), input_result.float())
            shape_param[index: last, ...] = output_param
            stroke_decision[index: last, ...] = output_decision
        # shape_param, stroke_decision = net_g(img_patch, result_patch)
        shape_param = shape_param.contiguous().to(torch.float32)
        stroke_decision = stroke_decision.contiguous()

        grid = shape_param[:, :, :2].view(img_patch.shape[0] * stroke_num, 1, 1, 2).contiguous()
        img_temp = img_patch.unsqueeze(1).contiguous().repeat(1, stroke_num, 1, 1, 1).view(
            img_patch.shape[0] * stroke_num, 3, patch_size, patch_size).contiguous()
        color = F.grid_sample(img_temp, 2 * grid - 1, align_corners=False).view(
            img_patch.shape[0], stroke_num, 3).contiguous()
        stroke_param = torch.cat([shape_param, color], dim=-1)
        # stroke_param: b * h * w, stroke_per_patch, param_per_stroke
        # stroke_decision: b * h * w, stroke_per_patch, 1
        param = stroke_param.view(1, h, w, stroke_num, 8).contiguous()
        decision = stroke_decision.view(1, h, w, stroke_num).contiguous() > 0
        # param: b, h, w, stroke_per_patch, 8
        # decision: b, h, w, stroke_per_patch
        param[..., :2] = param[..., :2] / 2 + 0.25
        param[..., 2:4] = param[..., 2:4] / 2

        # NOTE: save strokes
        strokes.append((param, decision))
        final_result = param2img_parallel(
            param, decision, meta_brushes, final_result)

        final_result = final_result[:, :, border_size:-border_size, border_size:-border_size]

        final_result = crop(final_result, original_h, original_w)
        save_img(final_result[0], output_path)

        pickle.dump(strokes, open("strokes.pkl", "wb"))


if __name__ == '__main__':
    main(input_path='./input/bingbing.jpg',
         model_path='./model.pth',
         output_dir='./output/',
         resize_h=512,         # resize original input to this size. None means do not resize.
         resize_w=512)         # resize original input to this size. None means do not resize.
