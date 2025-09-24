import numpy as np 
import os
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

labels_dict = {"ULCUS": 0, "TUMOR": 1, 
                "SUB_GL": 2, "SUBMUC": 3, 
                "SH_OES": 4, "SH_MAG": 5, 
                "REGR_TU": 6,
                "MUSC_PROP": 7, "MUSC_MUC": 8, 
                "LAM_PROP": 9, "ADVENT": 10}

inv_labels_dict = {0: "ULCUS", 1: "TUMOR",
                   2: "SUB_GL", 3: "SUBMUC",
                   4: "SH_OES", 5: "SH_MAG",
                   6: "REGR_TU",
                   7: "MUSC_PROP", 8: "MUSC_MUC",
                   9: "LAM_PROP", 10: "ADVENT"}


class ESCA(Dataset):
    
    dataset_dir = os.path.join("thunder", "datasets", "esca")
    
    def __init__(self, root, transform=None):

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'VALSET2_WNS')

        self.classnames = ['Ulceration', 'Tumor', 
                           'Submucosal glands', 'Submucosa', 
                           'Oesophagus mucosa', 'Gastric mucosa', 
                           'Regression tissue',
                           'Muscularis propria', 'Muscularis mucosae', 
                           'Lamina propria mucosae', 'Adventitia']

        self.template = templates
        self.transform = transform

        self.file_counts = self.count_files()

    def __len__(self):
        return np.sum(self.file_counts)
    
    def count_files(self):
        num_files = []
        for folder_name in sorted(os.listdir(self.image_dir), reverse=True):
            folder_path = os.path.join(self.image_dir, folder_name)
            if os.path.isdir(folder_path):
                num_file = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
                num_files.extend([num_file])
        return num_files
    
    def __getitem__(self, idx):
        file_counts_cumsum = np.cumsum(self.file_counts)

        # Find the label (class index) from global idx
        label = np.searchsorted(file_counts_cumsum, idx, side='right')
        classname = inv_labels_dict[label]
        class_dir = os.path.join(self.image_dir, classname)

        images_name = os.listdir(class_dir)

        # Local index within the class
        if label == 0:
            local_idx = idx
        else:
            local_idx = idx - file_counts_cumsum[label - 1]

        image_name = images_name[local_idx]
        image_path = os.path.join(class_dir, image_name)

        # Open image with PIL and apply transform if available
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label