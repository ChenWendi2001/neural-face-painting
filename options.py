import argparse
import os


class Options():
    def __init__(self, type):
        self.initialized = False
        self.type = type

    def initialize(self, parser):
        cwd = os.path.dirname(__file__)
        parser.add_argument("--input_path", required=True) 
        parser.add_argument("--output_dir", required=True)
        parser.add_argument("--model_dir", type=str, default=os.path.join(cwd, "checkpoints"))
        parser.add_argument("--image_h", type=int, default=512)
        parser.add_argument("--image_w", type=int, default=512)
        parser.add_argument("--use_neural_render", action="store_true")
        if self.type == "style":
            parser.add_argument("--style_path", required=True)
            parser.add_argument("--max_iter", type=int, default=200)
            parser.add_argument("--show_freq", type=int, default=5)
            parser.add_argument("--lr", type=float, default=0.002)
            parser.add_argument("--use_mask", action="store_true")
            parser.add_argument("--mask_loss_lambda", type=float, default=1e9)
        self.initialize = True
        return parser

    def parse(self):
        if not self.initialized: 
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        
        opt = parser.parse_args()

        self.opt = opt
        return opt


