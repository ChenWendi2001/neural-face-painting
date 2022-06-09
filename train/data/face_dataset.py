from data.base_dataset import BaseDataset, get_transform
import os
import cv2
import numpy as np
import torch

from PIL import Image
from skimage import feature, exposure


class FaceDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.img_folder = os.path.join(self.root, "CelebA-HQ-img", "train")
        self.img_path = [os.path.join(self.img_folder, filename) for filename in os.listdir(self.img_folder)]

    def __getitem__(self, index):
        img = cv2.imread(self.img_path[index])[:,:,::-1]
        hog = self.get_hog(img)

        crop_x =np.random.randint(0, img.shape[0] - self.opt.crop_size)
        crop_y =np.random.randint(0, img.shape[1] - self.opt.crop_size)
        params = {
            'crop_pos': (crop_x, crop_y),
            'flip': np.random.random()>0.5
        }
        transform = get_transform(self.opt, params=params)

        img = Image.fromarray(img)
        img = transform(img)
        # hog = transform(hog)
        hog = hog[crop_y:crop_y + self.opt.crop_size, crop_x:crop_x + self.opt.crop_size, :]
        if params['flip']:
            hog = hog[:,::-1,:].copy()
        hog = torch.from_numpy(hog)
        return {'img': img, 'hog': hog}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(len(self.img_path), self.opt.max_dataset_size)

    def get_hog(self, img):
        kernel_size = 8
        hog_feature = feature.hog(img, orientations=12, pixels_per_cell=(kernel_size, kernel_size),
                            cells_per_block=(1, 1), channel_axis=2, feature_vector=False)
        
        hog_feature = hog_feature.squeeze()
        hog_feature = hog_feature.repeat(kernel_size, axis=0).repeat(kernel_size, axis=1)

        return hog_feature
        