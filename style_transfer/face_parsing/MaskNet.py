
import os
from regex import D
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
        options.append(transforms.Resize((imsize,imsize), interpolation=PIL.Image.NEAREST))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(options)
    
    return transform

class MaskNet(object):
    def __init__(self):
        self.version = "parsenet"
        self.imsize = 512
        self.pretrained_model = True
        self.model_name = "model.pth"
        
        self.model_save_path = os.path.join(os.path.dirname(__file__), "models")
        
        self.transform = transformer(True, False, True, False, self.imsize)
        self.build_model()
    
    def build_model(self):
        self.G = unet().cuda()
        self.G.load_state_dict(torch.load(os.path.join(self.model_save_path, self.version, self.model_name)))
        

    def pred(self, img):
        self.G.eval()
        

        
        img = self.transform(img)
        img.unsqueeze_(0)
        img = img.cuda()

        labels_predict = self.G(img)
        labels_predict_plain = generate_label_plain(labels_predict, self.imsize)
        labels_predict_color = generate_label(labels_predict, self.imsize)

        return labels_predict



