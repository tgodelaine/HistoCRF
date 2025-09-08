# Real-time Refinement of Histopathological Predictions via Human-in-the-Loop [Submitted to ICASSP 2026]

Implementation of **[Real-time Refinement of Histopathological Predictions via Human-in-the-Loop](doi)**.

Assisting pathologists in the analysis of histopathological images has high clinical value, as it supports cancer detection and staging. In this context, histology foundation models have recently emerged, providing strong yet imperfect initial predictions. We refine these predictions by adapting Conditional Random Fields (CRFs) to histopathological applications, requiring no additional model training. We present HistoCRF, a CRF-based framework with a novel definition of the pairwise potential that includes both a diversity-promoting term and a term that leverages expert annotations. Experiments on five patch-level classification datasets covering different organs and diseases demonstrate accuracy gains of 14.3\% without annotations and 27.1\% with only 100 annotations, compared to the foundation models alone. Moreover, we highlight the value of integrating a human in the loop: by iteratively correcting the predictions, the expert guides the refinement process in real time, reaching a further gain of 32.7\% with the same number of annotations. 

**Authors**: [T. Godelaine](https://scholar.google.com/citations?user=xKcPd0oAAAAJ&hl=en&oi=ao), [M. Zanella](https://scholar.google.com/citations?user=FIoE9YIAAAAJ&hl=fr&oi=ao), [K. El Khoury](https://scholar.google.be/citations?user=UU_keGAAAAAJ&hl=fr), [S. Mahmoudi](https://scholar.google.com/citations?user=K2BAx8sAAAAJ&hl=fr), [B. Macq](https://scholar.google.be/citations?user=H9pGN70AAAAJ&hl=fr), [C. De Vleeschouwer](https://scholar.google.com/citations?user=xb3Zc3cAAAAJ&hl=fr&oi=ao)

## Contents 

- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

## Installation 

📌 **NB:** The Python version used is 3.10.4.

1. Create a virtual environment
   ```bash
   python3 -m venv histocrf_venv
   source histocrf_venv/bin/activate
   ```

   Clone the GitHub repository
   ```bash
   pip3 install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1
   git clone https://github.com/tgodelaine/HistoCRF.git
   ```
   
   Install the required packages
   ```bash
   cd HistoCRF
   pip3 install -r requirements.txt
   ```


2. Datasets downloads:

| Dataset        | 🔗 Download Link                                                                                        |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| BACH           | [📥 Link](https)                     |
| BRACS           | [📥 Link](https)                    |
| ESCA       | [📥 Link](https) |
| NCT       | [📥 Link](https)                                                          |
| SICAP       | [📥 Link](https)                                                        |

📌 Each dataset must be divided into three folders: train, val and test. Images were named following this structure: *classname_number*.
**Important**: All file paths in scripts are set with the placeholder "TO CHANGE". You will need to search for this placeholder in the cloned repository's files and replace it with the appropriate path ```/root/path/``` as specified for your system. In this setup, we have placed the different datasets inside a folder named `./data`.

## Usage 

1. Once the datasets are downloaded, you must extract the features of each patch in the dataset. By using the following command line, a .npz file will be created at ./root/data_processed. 
   ```python
   python3 extraction.py --root_dir ./root/ --data_dir ./data/ --model {model} --dataset {dataset}
   ```

2. Once the features are extracted, you can run the experiments using the following command lines. The results will be saved into a JSON file for further analysis at ./root/results. 

| Experiment             | Command line                                                                                                                      |
| -----------------------| --------------------------------------------------------------------------------------------------------------------------------- |
| **LP**  | `python3 inference_iteration_split_split.py --root_path ./root/ --model {model} --dataset {dataset} --uni_pot softmax_unlabeled_and_annotation --weight 0.1 0.01 --pair_pot model_features minus_model_features_ann --compat indicator potts --sparse_method cossim --n_affinity 0 16 --n_annotations {n_annotations} --annotation_method error -ann_iterations {ann_iterations} --N_UP -100 --N_PROP 50 --seed {seed}` |
| **HistoCRF**      | `python3 inference_iteration_split_split.py --root_path ./root/ --model {model} --dataset {dataset} --uni_pot softmax_unlabeled_and_annotation --weight 0.1 0.01 --pair_pot model_features minus_model_features_ann --compat indicator potts --sparse_method cossim --n_affinity 0 16 --n_annotations {n_annotations} --annotation_method error -ann_iterations 1 --N_UP -1 --N_PROP 50 --seed {seed}` |
| **HistoCRF-HITL**      | `python3 inference_iteration_split_split.py --root_path ./root/ --model {model} --dataset {dataset} --uni_pot softmax_unlabeled_and_annotation --weight 0.1 0.01 --pair_pot model_features minus_model_features_ann --compat indicator potts --sparse_method cossim --n_affinity 0 16 --n_annotations {n_annotations} --annotation_method error -ann_iterations {ann_iterations} --N_UP -1 --N_PROP 50 --seed {seed}` |


## Contact 

If you have any questions, you can contact us by email: [tiffanie.godelaine@uclouvain.be](mailto\:tiffanie.godelaine@uclouvain.be)

