import os

import torch

import morphology
import renderer
from networks import define_G
import torch.nn.functional as F

class Painter():
    def __init__(self, args):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.renderer_checkpoint_dir = args.renderer_checkpoint_dir
        self.rderr = renderer.Renderer(renderer=args.renderer,
                                       CANVAS_WIDTH=args.canvas_size,
                                       canvas_color=args.canvas_color)
        self.net_G = define_G(rdrr=self.rderr, netG=args.net_G).to(self.device)
        self.loadCheckpoint()

    def loadCheckpoint(self):
        if os.path.exists((os.path.join(
                self.renderer_checkpoint_dir, 'last_ckpt.pt'))):
            print('loading renderer from pre-trained checkpoint...')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.renderer_checkpoint_dir, 'last_ckpt.pt'),
                                    map_location=None if torch.cuda.is_available() else self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.to(self.device)
            self.net_G.eval()

    def render(self, stroke):
        stroke = F.pad(stroke, pad=[0, 4, 0, 0])
        G_pred_foregrounds, G_pred_alphas = \
            self.net_G(stroke.unsqueeze(-1).unsqueeze(-1).float())
        G_pred_foregrounds = morphology.dilation(G_pred_foregrounds)
        G_pred_alphas = morphology.erosion(G_pred_alphas)

        return G_pred_foregrounds,  G_pred_alphas
