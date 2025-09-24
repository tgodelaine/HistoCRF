import os
from PIL import Image
from torch.utils.data import Dataset 


labels_dict = {
    "Adrenocortical_carcinoma": 0,       # ACC
    "Bladder_Urothelial_Carcinoma": 1,   # BLCA
    "Brain_Lower_Grade_Glioma": 2,       # LGG
    "Breast_invasive_carcinoma": 3,      # BRCA
    "Cervical_squamous_cell_carcinoma_and_endocervical_adenocarcinoma": 4,  # CESC
    "Cholangiocarcinoma": 5,             # CHOL
    "Colon_adenocarcinoma": 6,           # COAD
    "Esophageal_carcinoma": 7,           # ESCA
    "Glioblastoma_multiforme": 8,        # GBM
    "Head_and_Neck_squamous_cell_carcinoma": 9,  # HNSC
    "Kidney_Chromophobe": 10,            # KICH
    "Kidney_renal_clear_cell_carcinoma": 11,  # KIRC
    "Kidney_renal_papillary_cell_carcinoma": 12,  # KIRP
    "Liver_hepatocellular_carcinoma": 13,    # LIHC
    "Lung_adenocarcinoma": 14,           # LUAD
    "Lung_squamous_cell_carcinoma": 15,  # LUSC
    "Lymphoid_Neoplasm_Diffuse_Large_B-cell_Lymphoma": 16, 
    "Mesothelioma": 17,                  # MESO
    "Ovarian_serous_cystadenocarcinoma": 18,  # OV
    "Pancreatic_adenocarcinoma": 19,     # PAAD
    "Pheochromocytoma_and_Paraganglioma": 20,  # PCPG
    "Prostate_adenocarcinoma": 21,       # PRAD
    "Rectum_adenocarcinoma": 22,         # READ
    "Sarcoma": 23,                       # SARC
    "Skin_Cutaneous_Melanoma": 24,       # SKCM
    "Stomach_adenocarcinoma": 25,        # STAD
    "Testicular_Germ_Cell_Tumors": 26,   # TGCT
    "Thymoma": 27,                       # THYM
    "Thyroid_carcinoma": 28,             # THCA
    "Uterine_Carcinosarcoma": 29,        # UCS
    "Uterine_Corpus_Endometrial_Carcinoma": 30,  # UCEC
    "Uveal_Melanoma": 31                # UVM
}


class TCGAUniform(Dataset):
    """
    Loads H&E tumor patches from TCGA across cancer types, resolution levels, and patients.
    Expects directory structure:
        root/<cancer_type>/<resolution_level>/<TCGA_barcode>/*.jpg
    """

    dataset_dir = os.path.join("thunder", "datasets", "tcga_uniform")

    def __init__(self, root_dir, transform=None):
        self.root_dir = os.path.join(root_dir, self.dataset_dir)
        self.transform = transform

        self.samples = []
        for c in sorted(os.listdir(self.root_dir)):
            c_dir = os.path.join(self.root_dir, c)
            if not os.path.isdir(c_dir):
                continue
            for p in sorted(os.listdir(c_dir)):
                p_dir = os.path.join(self.root_dir, c, p)
                if not os.path.isdir(p_dir):
                    continue
                for s in sorted(os.listdir(p_dir)):
                    s_dir = os.path.join(self.root_dir, c, p, s)
                    if not os.path.isdir(s_dir):
                        continue
                    for img in sorted(os.listdir(s_dir)):
                        self.samples.append({
                                'path': os.path.join(s_dir, img),
                                'classname': c,
                                'label': labels_dict[c]
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)

        label = sample['label']
        return img, label