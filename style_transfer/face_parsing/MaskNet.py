
import os
import torch
import datetime
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
import math

import cv2
import PIL
from .unet import unet
from .utils import generate_label, generate_label_plain
from PIL import Image


def transformer(resize, totensor, normalize, centercrop, imsize):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(160))
    if resize:
        options.append(transforms.Resize(imsize, interpolation=PIL.Image.NEAREST))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(options)
    
    return transform

class MaskNet(object):
    def __init__(self, imsize_w, imsize_h):
        self.version = "parsenet"
        self.imsize_w = imsize_w
        self.imsize_h = imsize_h
        self.pretrained_model = True
        self.model_path = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "mask_model.pth")
        
        self.transform = transformer(True, False, True, False, (self.imsize_w, self.imsize_h))
        self.build_model()
    
    def build_model(self):
        self.G = unet().cuda()
        self.G.load_state_dict(torch.load(self.model_path))
        

    def pred(self, img):
        self.G.eval()
        

        
        img = self.transform(img)
        img.unsqueeze_(0)
        img = img.cuda()

        labels_predict = self.G(img)
        return labels_predict



