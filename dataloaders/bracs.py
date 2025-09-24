import os
from PIL import Image
from torchvision.io import read_image
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

classname_to_label = {
    "0_N": 0,
    "1_PB": 1,
    "2_UDH": 2,
    "3_FEA": 3,
    "4_ADH": 4,
    "5_DCIS": 5, 
    "6_IC": 6
}

classname_to_classname = {
    "N": "Normal tissue",
    "PB": "Pathological Benign Breast Lesion",
    "UDH": "Usual Ductal Hyperplasia Breast Lesion",
    "FEA": "Flat Epithelial Atypia Breast Lesion",
    "ADH": "Atypical Ductal Hyperplasia Breast Lesion",
    "DCIS": "Ductal Carcinoma in Situ Breast Lesion", 
    "IC": "Invasive Carcinoma Breast Lesion"
}


class BRACS(Dataset):

    image_dir = os.path.join("data", "BRACS", "BRACS_RoI", "latest_version", "test")
    
    def __init__(self, root, transform=None):
        self.image_dir = os.path.join(root, self.image_dir)

        self.data_test = self.__getdataset__()
        
        self.classnames = [
            "Normal tissue",
            "Pathological Benign Breast Lesion",
            "Usual Ductal Hyperplasia Breast Lesion",
            "Flat Epithelial Atypia Breast Lesion",
            "Atypical Ductal Hyperplasia Breast Lesion",
            "Ductal Carcinoma in Situ Breast Lesion", 
            "Invasive Carcinoma Breast Lesion"
        ]
        self.classnames = [c.lower() for c in self.classnames]

        self.template = templates
        self.transform = transform

    def __getdataset__(self):
        data_folders = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)]
        data_files = []
        for f in data_folders:
            data_files_f = [os.path.join(self.image_dir, f, fi) for fi in os.listdir(f)]
            data_files.extend(data_files_f)
        return data_files
    
    def __len__(self):
        return len(self.data_test)

    def __getitem__(self, idx):
         
        img_path = self.data_test[idx]
        label = classname_to_label[img_path.split(os.sep)[-2]]

        # Load and optionally transform image
        if self.transform:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        else:
            image = read_image(img_path)

        return image, label