import time
import os
from PIL import Image
image_dir = os.getcwd() + '/Images/'
model_dir = os.getcwd() + '/Models/'

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torchvision.utils import save_image

import torchvision
from torchvision import transforms

from PIL import Image
from collections import OrderedDict

# pre and post processing for images
img_size = 1024
prep = transforms.Compose([transforms.Resize([img_size, img_size]),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])
prep_render =  transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                          ])
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                           ])
postpb = transforms.Compose([transforms.ToPILImage()])
def postp(tensor): # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    img = postpb(t)
    return img

def postp_gpu(tensor):
    t = postpa(tensor)
    t[t>1] = 1    
    t[t<0] = 0
    return t


#load images, ordered as [style_image, content_image]
img_dirs = [image_dir, image_dir]
img_names = ['mosaic.jpg', 'bingbing.jpg']
imgs = [Image.open(img_dirs[i] + name) for i,name in enumerate(img_names)]
imgs_torch = [prep(img) for img in imgs]
if torch.cuda.is_available():
    imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
else:
    imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
style_image, content_image = imgs_torch

# opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True) #random init
# opt_img = Variable(content_image.data.clone(), requires_grad=True)

print(content_image.shape)

from render import Render, Strokes
from pseudo_render.pseudo_render import Render as PseudoRender
import pickle

s = pickle.load(open("strokes.pkl", "rb"))
render = Render(s)
pseudo_render = PseudoRender(s)
strokes = Strokes(s)

print(render(strokes()).shape)
out_img = postp(prep_render(render(strokes()).cpu().squeeze()))
out_img.save(os.path.join(os.getcwd(), "output", "1-render.jpg"))


print(pseudo_render(strokes()).shape)
out_img_pseudo = postp(prep_render(pseudo_render(strokes()).cpu().squeeze()))

out_img_pseudo.save( os.path.join(os.getcwd(), "output", "2-postp_render.jpg"))


#run style transfer
max_iter = 200
show_iter = 5
lr = 0.002
optimizer = optim.Adam(strokes.parameters(), lr)
n_iter=[0]

from loss import VGGStyleLoss, PixelLoss
style = VGGStyleLoss(transfer_mode=1)
l1 = PixelLoss()

from  face_parsing.MaskNet import MaskNet
from face_parsing.utils import generate_label, generate_label_plain, cross_entropy2d

use_mask = True
mask_loss_lambda = 1e9

if use_mask:
    img2mask = MaskNet(img_size)
    label = torch.tensor(generate_label_plain(img2mask.pred(postp_gpu(content_image.squeeze(0))), img_size)).cuda()



while n_iter[0] <= max_iter:

    def closure():
        optimizer.zero_grad()
        pseudo_render_result = prep_render(pseudo_render(strokes())).unsqueeze(0)

        # render_result = prep_render(render(strokes())).unsqueeze(0)
        content_loss = l1(pseudo_render_result, content_image) * 1
        # loss += l1(render_result, content_image) * 1
        
        style_loss = style(pseudo_render_result, style_image) * 0.5
        
        if use_mask:
            out_img_pseudo = postp_gpu(pseudo_render_result.squeeze(0))
            predict = img2mask.pred(out_img_pseudo)
            mask_loss = cross_entropy2d(predict, label)
        loss = mask_loss * mask_loss_lambda + style_loss * 0.5 + content_loss
        loss.backward()
        n_iter[0]+=1
        #print loss
        if n_iter[0]%show_iter == (show_iter-1):
            print('Iteration: %d, loss: %f, style loss %f, mask loss %f'%(n_iter[0]+1, loss.item(), style_loss, mask_loss.item()))
            out_img = postp(prep_render(render(strokes())).cpu())
            out_img.save(os.path.join(os.getcwd(), "output", "3-final-" + str(n_iter[0]) + ".jpg"))

        return loss
    
    optimizer.step(closure)
    
#display result
out_img = postp(prep_render(render(strokes())).cpu())
out_img.save(os.path.join(os.getcwd(), "output", "3-final-%f-%f-%f.jpg" % (max_iter, mask_loss_lambda, lr)))
