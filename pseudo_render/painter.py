import os

import torch
import torch.nn.functional as F

from .morphology import *
from .renderer import *
from .networks import define_G

class Painter():
    def __init__(self, args):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.renderer_checkpoint_dir = args.renderer_checkpoint_dir
        self.rderr = Renderer(renderer=args.renderer,
                                       CANVAS_WIDTH=args.canvas_size,
                                       canvas_color=args.canvas_color)
        self.net_G = define_G(rdrr=self.rderr, netG=args.net_G).to(self.device)
        self.loadCheckpoint()

    def loadCheckpoint(self):
        path = os.path.join(os.getcwd(), "pseudo_render",self.renderer_checkpoint_dir,
             'last_ckpt.pt')
        if os.path.exists(path):
            print('loading renderer from pre-trained checkpoint...', path)
            # load the entire checkpoint
            checkpoint = torch.load(path,
                                    map_location=None if torch.cuda.is_available() else self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.to(self.device)
            self.net_G.eval()
        else:
            print("!!!!!!! no render checkpoint !!!!!!!")

    def render(self, stroke):
        # stroke = F.pad(stroke, pad=[0, 4, 0, 0])
        G_pred_foregrounds, G_pred_alphas = \
            self.net_G(stroke.unsqueeze(-1).unsqueeze(-1).float())
        G_pred_foregrounds = Dilation2d()(G_pred_foregrounds)
        G_pred_alphas = Erosion2d()(G_pred_alphas)

        return G_pred_foregrounds,  G_pred_alphas
