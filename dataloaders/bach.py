import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset 


templates = [
            "a histopathological image showing {}.",
            "{}.",
            "a photomicrograph showing {}.",
            "a photomicrograph of {}.",
            "an image of {}.",
            "an image showing {}.",
            "an example of {}.",
            "{} is shown.",
            "this is {}.",
            "there is {}.",
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

labels_dict = {"Normal": 0, "Benign": 1, 
                "InSitu": 2, "Invasive": 3}


class BACH(Dataset):
    
    dataset_dir = os.path.join("thunder", "datasets", "bach")
    
    def __init__(self, root, transform=None):

        self.dataset_dir = os.path.join(root, self.dataset_dir) 
        self.image_dir = os.path.join(self.dataset_dir, 'ICIAR2018_BACH_Challenge', 'Photos')
        self.csv_path = os.path.join(self.image_dir, 'microscopy_ground_truth.csv')

        self.classnames = ['Normal', 'Benign', 'In Situ carcinoma', 'Invasive carcinoma']

        self.template = templates
        self.transform = transform

    def __len__(self):
        csv_file = pd.read_csv(self.csv_path)
        return len(csv_file)

    def  __getitem__(self, idx):
        csv_file_idx = pd.read_csv(self.csv_path, header=None).iloc[idx]
        image_name = csv_file_idx[0]
        classname = csv_file_idx[1]
        image_path = os.path.join(self.image_dir, classname, image_name)
        image = Image.open(image_path)
        label = labels_dict[classname]

        if self.transform:
            image = Image.open(image_path)
            image = self.transform(image)

        return image, label