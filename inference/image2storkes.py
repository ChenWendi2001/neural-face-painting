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
            input_path=opt.input_path,
            model_path=os.path.join(opt.model_dir, "diff_model.pth"),
            output_dir=os.path.join(opt.output_dir, image_name),
            resize_h=opt.image_h,
            resize_w=opt.image_w,
            increasing_layers=opt.strokes_increasing_layers
        )
        os.rename(
            f"./output/{image_name}/strokes.pkl", f"./output/{image_name}/strokes_more.pkl")
        main_diff(
            input_path=opt.input_path,
            model_path=os.path.join(opt.model_dir, "diff_model.pth"),
            output_dir=os.path.join(opt.output_dir, image_name),
            resize_h=opt.image_h,
            resize_w=opt.image_w,
            increasing_layers=None
        )
        strokes = pickle.load(
            open(f"./output/{image_name}/strokes_more.pkl", "rb"))
        strokes_less = pickle.load(
            open(f"./output/{image_name}/strokes.pkl", "rb"))
        strokes[-1] = strokes_less[-1]
        strokes[-2] = strokes_less[-2]
        pickle.dump(
            strokes, open(f"./output/{image_name}/strokes.pkl", "wb"))
    else:
        main(
            input_path=opt.input_path,
            model_path=os.path.join(opt.model_dir, "stroke_model.pth"),
            output_dir=os.path.join(opt.output_dir, image_name),
            resize_h=opt.image_h,
            resize_w=opt.image_w,
            increasing_layers=opt.strokes_increasing_layers
        )
        os.rename(
            f"./output/{image_name}/strokes.pkl", f"./output/{image_name}/strokes_more.pkl")
        main(
            input_path=opt.input_path,
            model_path=os.path.join(opt.model_dir, "stroke_model.pth"),
            output_dir=os.path.join(opt.output_dir, image_name),
            resize_h=opt.image_h,
            resize_w=opt.image_w,
            increasing_layers=None
        )
        strokes = pickle.load(
            open(f"./output/{image_name}/strokes_more.pkl", "rb"))
        strokes_less = pickle.load(
            open(f"./output/{image_name}/strokes.pkl", "rb"))
        strokes[-1] = strokes_less[-1]
        strokes[-2] = strokes_less[-2]
        pickle.dump(
            strokes, open(f"./output/{image_name}/strokes.pkl", "wb"))
