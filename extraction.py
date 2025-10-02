import argparse
import numpy as np 
import os
import torch
from torch.utils.data import DataLoader

from dataloaders.sicap_mil import SicapMIL
from dataloaders.skincancer import SkinCancer
from dataloaders.nct import NCT
from dataloaders.lc_lung import LClung
from dataloaders.lc_colon import LCcolon
from dataloaders.tcga_ut import TCGA_ut
from dataloaders.bracs import BRACS
from dataloaders.bach import BACH
from dataloaders.esca import ESCA
from dataloaders.tcga_uniform import TCGAUniform
from dataloaders.bach_wsi import BACH_WSI
from extraction_esca import extraction_esca


dataloader_conversion = {
    'sicap_mil': SicapMIL,
    'skincancer2': SkinCancer,
    'nct': NCT,
    'lc_lung': LClung,
    'lc_colon': LCcolon,
    'tcga_ut': TCGA_ut,
    'bracs': BRACS,
    'bach': BACH, 
    'esca': ESCA,
    'tcga_uniform': TCGAUniform, 
    'bach_wsi': BACH_WSI
}


def get_variables(npz_path):
    # Load the .npz file and access the required keys
    data = np.load(npz_path)
    width = data['width']
    height = data['height']
    n_labels = data['n_labels']
    probabilities = data['probabilities']

    # Close the file
    data.close()

    return width, height, n_labels, probabilities

def extraction(args):
    # Create directory
    npz_dir = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model)
    if not os.path.exists(npz_dir): 
        os.makedirs(npz_dir)
    npz_name = f'{args.dataset}_{args.model}.npz'
    if args.dataset in ['skincancer2', 'bach_wsi']: 
        npz_name = f'{args.dataset}_{args.model}_{args.n_patient}.npz'
    npz_path = os.path.join(npz_dir, npz_name)

    # Feature extraction 
    if args.model == 'clip':
        import clip
        from models.llm import features_extraction

        # Load model and get data
        model, preprocess = clip.load(args.backbone)
        if args.dataset == 'skincancer2': 
            dataset = dataloader_conversion[args.dataset](args.root_dir, preprocess, args.n_patient)
        elif args.dataset == 'tcga-ut':
            os.environ["HF_HOME"] = env_path
            from datasets import load_dataset
            from torchvision import transforms

            ds = load_dataset("dakomura/tcga-ut", "internal")
            ds = ds["test"]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224)
            ])
            def preprocess(example):
                image = transform(example['jpg'])
                label = example['json']['label']  # Adjust based on the actual key
                return {'pixel_values': image, 'label': label}
            dataset = ds.map(preprocess)
            dataset.set_format(type='torch', columns=['pixel_values', 'label'])
            dataset = dataset.remove_columns(['__url__', '__key__', 'jpg', 'json'])
            dataset = TCGA_ut(dataset)
 
        else:
            dataset = dataloader_conversion[args.dataset](args.data_dir, preprocess)
        dataloader = DataLoader(dataset, 16)

        # Get features
        images, img_features, text_features, labels, cosine_similarities, positions = features_extraction(args, model, None, dataset, dataloader, get_position)
        n_labels = len(dataset.classnames)
        n_samples = images.size()[0]
        height = 1
        
        if get_position: 
            n_samples = torch.max(positions[0]).item() // args.patch_size +1
            height = torch.max(positions[1]).item() // args.patch_size +1

        np.savez_compressed(
            npz_path, 
            width=n_samples, 
            height=height, 
            n_labels=n_labels, 
            probabilities=cosine_similarities.cpu().numpy(), 
            features=img_features.detach().cpu().numpy(), 
            text_features=text_features.detach().cpu().numpy(),
            image_features=img_features.detach().cpu().numpy(), 
            images=images.cpu().numpy(), 
            labels=labels.cpu().numpy(),
            positions=positions
        )
        print(f"Save to {npz_path}")

    elif args.model == 'quilt':
        from open_clip import create_model_and_transforms, get_tokenizer
        from models.llm import features_extraction

        # Load model and get data
        model, preprocess, _ = create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        tokenizer = get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')
        if args.dataset == 'skincancer2': 
            dataset = dataloader_conversion[args.dataset](args.root_dir, preprocess, args.n_patient)
        elif args.dataset == 'tcga-ut':
            os.environ["HF_HOME"] = env_path
            from datasets import load_dataset
            from torchvision import transforms

            ds = load_dataset("dakomura/tcga-ut", "internal")
            ds = ds["test"]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224)
            ])
            def preprocess(example):
                image = transform(example['jpg'])
                label = example['json']['label']  # Adjust based on the actual key
                return {'pixel_values': image, 'label': label}
            dataset = ds.map(preprocess)
            dataset.set_format(type='torch', columns=['pixel_values', 'label'])
            dataset = dataset.remove_columns(['__url__', '__key__', 'jpg', 'json'])
            dataset = TCGA_ut(dataset)

        else:
            dataset = dataloader_conversion[args.dataset](args.data_dir, preprocess)
        dataloader = DataLoader(dataset, 16)
        
        # Get features
        images, img_features, text_features, labels, cosine_similarities, positions = features_extraction(args, model.cuda(), tokenizer, dataset, dataloader, get_position)
        n_labels = len(dataset.classnames)
        #n_samples = images.size()[0]
        n_samples = img_features.size()[0]
        height = 1

        if get_position:
            n_samples = torch.max(positions[0]).item() // args.patch_size +1
            height = torch.max(positions[1]).item() // args.patch_size +1

        np.savez(
            npz_path, 
            width=n_samples, 
            height=height, 
            n_labels=n_labels, 
            probabilities=cosine_similarities.cpu().numpy(), 
            features=img_features.detach().cpu().numpy(), 
            text_features=text_features.detach().cpu().numpy(),
            image_features=img_features.detach().cpu().numpy(), 
            images=images.cpu().numpy(), 
            labels=labels.cpu().numpy(),
            positions=positions
        )
        print(f"Save to {npz_path}")

    elif args.model == 'plip':
        import clip
        from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
        from models.llm import features_extraction

        # Load model and get data
        _, preprocess = clip.load(args.backbone)
        processor = AutoProcessor.from_pretrained("vinid/plip")
        model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip")
        if args.dataset == 'skincancer2': 
            dataset = dataloader_conversion[args.dataset](args.root_dir, preprocess, args.n_patient)
        elif args.dataset == 'tcga-ut':
            os.environ["HF_HOME"] = env_path
            from datasets import load_dataset
            from torchvision import transforms

            ds = load_dataset("dakomura/tcga-ut", "internal")
            ds = ds["test"]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224)
            ])
            def preprocess(example):
                image = transform(example['jpg'])
                label = example['json']['label']  # Adjust based on the actual key
                return {'pixel_values': image, 'label': label}
            dataset = ds.map(preprocess)
            dataset.set_format(type='torch', columns=['pixel_values', 'label'])
            dataset = dataset.remove_columns(['__url__', '__key__', 'jpg', 'json'])
            dataset = TCGA_ut(dataset)

        else:
            dataset = dataloader_conversion[args.dataset](args.data_dir, preprocess)
        dataloader = DataLoader(dataset, 16)

        # Get features
        images, img_features, text_features, labels, cosine_similarities, positions = features_extraction(args, model.cuda(), None, dataset, dataloader, get_position)
        n_labels = len(dataset.classnames)
        n_samples = images.size()[0]
        height = 1

        if get_position:
            n_samples = torch.max(positions[0]).item() // args.patch_size +1
            height = torch.max(positions[1]).item() // args.patch_size +1

        np.savez_compressed(
            npz_path, 
            width=n_samples, 
            height=height, 
            n_labels=n_labels, 
            probabilities=cosine_similarities.cpu().numpy(), 
            features=img_features.detach().cpu().numpy(), 
            image_features=img_features.detach().cpu().numpy(), 
            text_features=text_features.detach().cpu().numpy(),
            images=images.cpu().numpy(), 
            labels=labels.cpu().numpy(),
            positions=positions
        )
        print(f"Save to {npz_path}")


    elif args.model == 'conch':
        from CONCH.conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        from models.llm import features_extraction

        # Load model and get data
        model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="checkpoints/conch/pytorch_model.bin")
        tokenizer = get_tokenizer()
        if args.dataset == 'skincancer2': 
            dataset = dataloader_conversion[args.dataset](args.root_dir, preprocess, args.n_patient)
        elif args.dataset == 'tcga-ut':
            os.environ["HF_HOME"] = env_path
            from datasets import load_dataset
            from torchvision import transforms

            ds = load_dataset("dakomura/tcga-ut", "internal")
            ds = ds["test"]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224)
            ])
            def preprocess(example):
                image = transform(example['jpg'])
                label = example['json']['label']  # Adjust based on the actual key
                return {'pixel_values': image, 'label': label}
            dataset = ds.map(preprocess)
            dataset.set_format(type='torch', columns=['pixel_values', 'label'])
            dataset = dataset.remove_columns(['__url__', '__key__', 'jpg', 'json'])
            dataset = TCGA_ut(dataset)

        else:
            dataset = dataloader_conversion[args.dataset](args.data_dir, preprocess)
        dataloader = DataLoader(dataset, 16)

        # Get features
        images, img_features, text_features, labels, cosine_similarities, positions = features_extraction(args, model, tokenizer, dataset, dataloader, get_position)
        n_labels = len(dataset.classnames)
        n_samples = images.size()[0]
        height = 1

        if get_position:
            n_samples = torch.max(positions[0]).item() // args.patch_size +1
            height = torch.max(positions[1]).item() // args.patch_size +1
            n_samples = cosine_similarities.size()[0]
            height = 1

        np.savez_compressed(
            npz_path, 
            width=n_samples, 
            height=height, 
            n_labels=n_labels, 
            probabilities=cosine_similarities.cpu().numpy(), 
            features=img_features.detach().cpu().numpy(), 
            text_features=text_features.detach().cpu().numpy(),
            image_features=img_features.detach().cpu().numpy(), 
            images=images.cpu().numpy(), 
            labels=labels.cpu().numpy(),
            positions=positions
        )
        print(f"Save to {npz_path}")

    elif args.model in ['plipanduni2', 'conchanduni2', 'conchandgigapath', 'conchandvirshow2', 'conchandoptimus1']:
        import clip
        import timm 
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
        from CONCH.conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        from torchvision import transforms
        from models.llm import features_extraction
        
        model_names = args.model
        # Load both model2 and get data
        if args.model == 'plipanduni2':
            _, preprocess = clip.load(args.backbone)
            model1 = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip").cuda() 
            tokenizer = None
        elif args.model in ['conchanduni2', 'conchandgigapath', 'conchandvirshow2', 'conchandoptimus1']:
            model1, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="checkpoints/conch/pytorch_model.bin")
            tokenizer = get_tokenizer()
        
        if args.dataset in ['skincancer2', 'bach_wsi']: 
            dataset1 = dataloader_conversion[args.dataset](args.root_dir, preprocess, args.n_patient)
        elif args.dataset == 'tcga-ut':
            os.environ["HF_HOME"] = env_path
            from datasets import load_dataset
            from torchvision import transforms

            ds = load_dataset("dakomura/tcga-ut", "internal")
            ds = ds["test"]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224)
            ])
            def preprocess(example):
                image = transform(example['jpg'])
                label = example['json']['label']  # Adjust based on the actual key
                return {'pixel_values': image, 'label': label}
            dataset = ds.map(preprocess)
            dataset.set_format(type='torch', columns=['pixel_values', 'label'])
            dataset = dataset.remove_columns(['__url__', '__key__', 'jpg', 'json'])
            dataset1 = TCGA_ut(dataset)
        else:
            dataset1 = dataloader_conversion[args.dataset](args.data_dir, preprocess)
        dataloader1 = DataLoader(dataset1, 16)
        
        if args.model in ['conchanduni2', 'plipanduni2']:
            timm_kwargs = {
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
            model2 = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)   
            transform2 = transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )
        elif args.model == 'conchandgigapath':
            model2 = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            transform2 = transforms.Compose(
                        [
                            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ]
                    )
        elif args.model == 'conchandvirshow2':
            from timm.layers import SwiGLUPacked
            from timm.data import resolve_data_config
            model2 = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
            transform2 = create_transform(**resolve_data_config(model2.pretrained_cfg, model=model2))
        elif args.model == 'conchandoptimus1':
            model2 = timm.create_model("hf_hub:bioptimus/H-optimus-1", pretrained=True)
            transform2 = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(

                    mean=(0.707223, 0.578729, 0.703617), 
                    std=(0.211883, 0.230117, 0.177517)
                ),
            ])
        else:
            raise RuntimeError(f"Not implemented for {args.model}")
        
        if args.dataset in ['skincancer2', 'bach_wsi']: 
            dataset2 = dataloader_conversion[args.dataset](args.root_dir, transform2, args.n_patient) 
        elif args.dataset == 'tcga-ut':
            os.environ["HF_HOME"] = env_path
            from datasets import load_dataset
            from torchvision import transforms

            ds = load_dataset("dakomura/tcga-ut", "internal")
            ds = ds["test"]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(224)
            ])
            def preprocess(example):
                image = transform(example['jpg'])
                label = example['json']['label']  # Adjust based on the actual key
                return {'pixel_values': image, 'label': label}
            dataset = ds.map(preprocess)
            dataset.set_format(type='torch', columns=['pixel_values', 'label'])
            dataset = dataset.remove_columns(['__url__', '__key__', 'jpg', 'json'])
            dataset2 = TCGA_ut(dataset)
      
        else:
            dataset2 = dataloader_conversion[args.dataset](args.data_dir, transform2)
        dataloader2 = DataLoader(dataset2, 16)

        # Get features
        if model_names == 'conchandgigapath':
            args.model = 'gigapath'
        elif model_names in ['plipanduni2', 'conchanduni2']:
            args.model = 'uni2'
        elif model_names in ['conchandvirshow2']:
            args.model = 'virshow2'
        elif model_names in ['conchandoptimus1']:
            args.model = 'optimus1'
        else:
            raise RuntimeError("Not implemented yet for model {args.model}")
        
        _, img_features, __, ___, ____ = features_extraction(args, model2.cuda(), None, dataset2, dataloader2, get_position, True)
        del model2, dataloader2, dataset2
        if model_names == 'plipanduni2':
            args.model = 'plip'
        elif model_names in ['conchandgigapath', 'conchanduni2', 'conchandvirshow2', 'conchandoptimus1']:
            args.model = 'conch'
        else:
            raise RuntimeError("Not implemented yet for model {args.model}")
        images, image_features, text_features, labels, cosine_similarities, positions = features_extraction(args, model1, tokenizer, dataset1, dataloader1, get_position, False)
        n_labels = len(dataset1.classnames)
        n_samples = cosine_similarities.size()[0]
        height = 1

        if get_position:
            n_samples = torch.max(positions[0]).item() // args.patch_size +1
            height = torch.max(positions[1]).item() // args.patch_size +1
            n_samples = cosine_similarities.size()[0]
            height = 1

        img_features_ = img_features.to(dtype=torch.float16)
        cossim = (img_features_@img_features_.T).cpu().numpy()
        min_cossim_per_row = np.min(cossim, axis=1)  # shape: (num_rows,)
        indices_high_affinities = cossim < (min_cossim_per_row[:, np.newaxis] + 0.4)
        if not args.dissimilar:  #For ablation studies dissimilar = False
            min_cossim_per_row = np.max(cossim, axis=1)  # shape: (num_rows,)
            indices_high_affinities = cossim > (min_cossim_per_row[:, np.newaxis] - 0.4)

        indices_high_affinities = torch.tensor(indices_high_affinities)
        row_lengths = indices_high_affinities.sum(dim=1)        # number of True per row
        max_len = row_lengths.max().item()
        n = indices_high_affinities.shape[0]
        # Build padded tensor by repeating values
        padded = torch.empty((n, max_len), dtype=torch.long, device='cuda')
        for i in range(n):
            valid = torch.nonzero(indices_high_affinities[i]).squeeze(1)
            repeat_factor = (max_len + valid.numel() - 1) // valid.numel()  # ceil division
            repeated = valid.repeat(repeat_factor)[:max_len]                # repeat & trim
            padded[i] = repeated
    
        if get_position: 
            positions = torch.cat(positions).numpy()

        if not args.dissimilar:
            npz_path = npz_path.replace('.npz', '_similar.npz')

        np.savez_compressed(
            npz_path, 
            width=n_samples, 
            height=height, 
            n_labels=n_labels, 
            probabilities=cosine_similarities.cpu().numpy(), 
            features=img_features.detach().cpu().numpy(), 
            text_features=text_features.detach().cpu().numpy(),
            image_features=image_features.detach().cpu().numpy(), 
            images=images, 
            labels=labels.cpu().numpy(),
            positions=positions,
            cossim=padded.detach().cpu().numpy()
        )
        print(f"Save to {npz_path}")

        if args.dataset == 'esca':
            extraction_esca(args)
            
    else:
        raise RuntimeError(f"Features extraction is not implemented yet for the model {args.model}!")


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='sicap_mil', choices=['sicap_mil', 'monuseg', 'skincancer', 'panuke', 'skincancer2', 'lc_lung', 'lc_colon', 'nct', 'tcga-ut', 'bracs', 'bach', 'esca', 'tcga_uniform', 'bach_wsi'])
    parser.add_argument('--model', type=str, default='conchanduni2', choices=['clip', 'quilt', 'plip', 'conch', 'plipanduni2', 'conchanduni2', 'conchandgigapath', 'conchandvirshow2', 'conchandoptimus1'])
    parser.add_argument('--backbone', type=str, default='ViT-B/16')
    parser.add_argument('--file_ending', type=str, default='.tif')
    parser.add_argument('--patch_size', type=int, default=254)
    parser.add_argument('--n_patient', type=int, default=0)
    parser.add_argument('--dissimilar', action="store_false")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parser()
    env_path = 'path_to_HistoCRF_venv'

    if args.dataset in ['bach_wsi', 'skincancer2']: 
        get_position = True
    else:
        get_position = False

    if args.dataset == 'bach_wsi':
        data_dir_patches = os.path.join(args.data_dir, 'thunder', 'datasets', 'bach', 'ICIAR2018_BACH_Challenge', f'patches_{args.size}')
        if not os.path.isdir(data_dir_patches):
            raise RuntimeError('You need to extract patches from the whole slide images. Run the following command: python3 datasets.extract_patches_bach_wsi.py --data_dir ./data --tile_size {tile_size}')
    
    extraction(args)