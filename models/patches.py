import os
import random
import numpy as np 
import pandas as pd
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset 
import torch 

templates = [
            "{}.",
            "a photomicrograph showing {}.",
            "a photomicrograph of {}.",
            "an image of {}.",
            "an image showing {}.",
            "an example of {}.",
            "{} is shown.",
            "this is {}.",
            "there is {}.",
            "a histopathological image showing {}.",
            "a histopathological image of {}.",
            "a histopathological photograph of {}.",
            "a histopathological photograph showing {}.",
            "shows {}.",
            "presence of {}.",
            "{} is present.",
            "an H&E stained image of {}.",
            "an H&E stained image showing {}.",
            "an H&E image showing {}.",
            "an H&E image of {}.",
            "{}, H&E stain.",
            "{}, H&E."
]

class Patches(Dataset):
    
    def __init__(self, image_path, patch_size, transform=None):

        self.image = np.array(Image.open(image_path))
        self.label = int(image_path.split('.png')[0].split('label_')[-1])
        self.patch_size = patch_size

        self.classnames = [
                "Background",
                "Sebaceous and sweat glands",
                "Inflammation",
                "Hair follicle",
                "Hypodermis",
                "Reticular dermis",
                "Papillary dermis",
                "Epidermis", 
                "Keratin",
                "Basal cell carcinoma",
                "Squamous cell carcinom",
                "Intra-epidermal carinoma"
             ]
        self.template = templates
        self.transform = transform

    def __len__(self):
        wh = self.image.shape[0]*self.image.shape[1]
        return wh // (self.patch_size**2)

    def  __getitem__(self, idx):
        label = self.label

        n_patch_h = self.image.shape[0] // self.patch_size
        n_patch_w = self.image.shape[1] // self.patch_size
        row = idx // n_patch_h
        column = idx - (row * n_patch_w)

        patch = self.image[column*self.patch_size:(column+1)*self.patch_size, row*self.patch_size:(row+1)*self.patch_size]

        #position = [int(row), int(column)]

        if self.transform:
            patch = Image.fromarray(patch)
            patch = self.transform(patch)

        return patch, label