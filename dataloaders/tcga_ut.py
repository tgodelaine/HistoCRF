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
    "Adrenocortical_carcinoma": 0, 
    "Bladder_Urothelial_Carcinoma": 1,
    "Brain_Lower_Grade_Glioma": 2,
    "Breast_invasive_carcinoma": 3,
    "Cervical_squamous_cell_carcinoma_and_endocervical_adenocarcinoma": 4,
    "Cholangiocarcinoma": 5,
    "Colon_Rectum_adenocarcinoma": 6,
    "Esophageal_carcinoma": 7,
    "Glioblastoma_multiforme": 8,
    "Head_and_Neck_squamous_cell_carcinoma": 9,
    "Kidney_Chromophobe": 10,
    "Kidney_renal_clear_cell_carcinoma": 11,
    "Kidney_renal_papillary_cell_carcinoma": 12,
    "Liver_hepatocellular_carcinoma": 13,
    "Lung_adenocarcinoma": 14,
    "Lung_squamous_cell_carcinoma": 15,
    "Lymphoid_Neoplasm_Diffuse_Large_B-cell_Lymphoma": 16,
    "Mesothelioma": 17,
    "Ovarian_serous_cystadenocarcinoma": 18,
    "Pancreatic_adenocarcinoma": 19,
    "Pheochromocytoma_and_Paraganglioma": 20,
    "Prostate_adenocarcinoma": 21,
    "Sarcoma": 22,
    "Skin_Cutaneous_Melanoma": 23,
    "Stomach_adenocarcinoma": 24,
    "Testicular_Germ_Cell_Tumors": 25,
    "Thymoma": 26,
    "Thyroid_carcinoma": 27,
    "Uterine_Carcinosarcoma": 28,
    "Uterine_Corpus_Endometrial_Carcinoma": 29,
    "Uveal_Melanoma": 30
}

# https://huggingface.co/datasets/dakomura/tcga-ut
class TCGA_ut(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self.classnames = [k.replace("_", " ").lower() for k in classname_to_label.keys()]

        self.template = templates

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        label = classname_to_label[item['label']]
        return item['pixel_values'], label
    