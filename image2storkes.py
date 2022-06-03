import math
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from options import Options
from inference import main
from inference_diff import main as main_diff

if __name__ == '__main__':
    opt = Options(type="stroke").parse()
    image_file = os.path.split(opt.input_path)[1]
    image_name = "".join(image_file.split(".")[:-1])

    if opt.use_neural_render:
        main_diff(
            input_path = opt.input_path,
            model_path = os.path.join(opt.model_dir, "diff_model.pth"),
            output_dir = os.path.join(opt.output_dir, image_name),
            resize_h = opt.image_h,
            resize_w = opt.image_w
        )
    else:
        main(
            input_path = opt.input_path,
            model_path = os.path.join(opt.model_path, "stroke_model.pth"),
            output_dir = os.path.join(opt.output_dir, image_name),
            resize_h = opt.image_h,
            resize_w = opt.image_w
        )

