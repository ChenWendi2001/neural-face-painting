from options import Options
import os
from PIL import Image

import torch 
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from torch import optim


from style_transfer.render import Render, Strokes
from pseudo_render import pseudo_render
import pickle


from style_transfer.loss import VGGStyleLoss, PixelLoss

from style_transfer.face_parsing.MaskNet import MaskNet
from style_transfer.face_parsing.utils import generate_label, generate_label_plain, cross_entropy2d

opt = Options(type="style").parse()
image_file = os.path.split(opt.input_path)[1]
image_name = "".join(image_file.split(".")[:-1])

img_size_h = opt.image_h
img_size_w = opt.image_w


prep = transforms.Compose([transforms.Resize([img_size_w, img_size_h]),
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

images_path = [opt.style_path, opt.input_path]
imgs = [Image.open(image_path) for image_path in images_path]

imgs_torch = [prep(img) for img in imgs]

if torch.cuda.is_available():
    imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
else:
    imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
style_image, content_image = imgs_torch

print("content image shape", content_image.shape)

s = pickle.load(open(os.path.join(opt.output_dir, image_name, "strokes.pkl"), "rb"))
render = Render(s)
pseudo_render = pseudo_render.Render(s)
strokes = Strokes(s)

print(render(strokes()).shape)
out_img = postp(prep_render(render(strokes()).cpu().squeeze()))
out_img.save(os.path.join(opt.output_dir, image_name, "1-render.jpg"))


print(pseudo_render(strokes()).shape)
out_img_pseudo = postp(prep_render(pseudo_render(strokes()).cpu().squeeze()))

out_img_pseudo.save( os.path.join(opt.output_dir, image_name, "2-pseudo_render.jpg"))

max_iter = opt.max_iter
show_iter = opt.show_freq
lr = opt.lr
optimizer = optim.Adam(strokes.parameters(), lr)
n_iter=[0]

style = VGGStyleLoss(transfer_mode=1)
l1 = PixelLoss()

use_mask = opt.use_mask
mask_loss_lambda = opt.mask_loss_lambda

if use_mask:
    img2mask = MaskNet(img_size_w, img_size_h)
    label = torch.tensor(generate_label_plain(img2mask.pred(postp_gpu(content_image.squeeze(0))), img_size_w, img_size_h)).cuda()

if not os.path.exists(os.path.join(opt.output_dir, image_name, "temp")):
    os.mkdir(os.path.join(opt.output_dir, image_name, "temp") )

while n_iter[0] <= max_iter:
    
    def closure():
        optimizer.zero_grad()
        pseudo_render_result = prep_render(pseudo_render(strokes())).unsqueeze(0)

        # render_result = prep_render(render(strokes())).unsqueeze(0)
        content_loss = l1(pseudo_render_result, content_image) * 1
        # loss += l1(render_result, content_image) * 1
        
        style_loss = style(pseudo_render_result, style_image) * 0.5
        
        mask_loss = torch.tensor([0]).cuda()
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
            out_img.save(os.path.join(opt.output_dir, image_name, "temp", "3-final-" + str(n_iter[0]) + ".jpg"))

        return loss
    
    optimizer.step(closure)


#display result
out_img = postp(prep_render(render(strokes())).cpu())
out_img.save(os.path.join(opt.output_dir, image_name, "3-final-%f-%f-%f.jpg" % (max_iter, mask_loss_lambda, lr)))
