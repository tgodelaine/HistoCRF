import numpy as np 
from torch.utils.data import Dataset 


class Features(Dataset):

    def __init__(self, args, npz_path, test=False):

        self.npz_path = npz_path
        self.features = np.load(npz_path)["image_features"]
        self.labels = np.load(npz_path)["labels"]
        self.annotations = args.annotations
        self.test = test

    def __len__(self):
        if not self.test:
            return len(self.annotations)
        else:
            return len(self.features)

    def  __getitem__(self, idx):
        if not self.test: 
            features = self.features[self.annotations]
            labels = self.labels[self.annotations]
        else: 
            features = self.features
            labels = self.labels
        feature = features[idx]
        label = labels[idx]
        return feature, label