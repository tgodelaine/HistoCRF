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

dataloader_conversion = {
    'sicap_mil': SicapMIL,
    'skincancer2': SkinCancer,
    'nct': NCT,
    'lc_lung': LClung,
    'lc_colon': LCcolon
}

def extraction(args):
    # Create directory
    npz_dir = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model)
    if not os.path.exists(npz_dir): 
        os.makedirs(npz_dir)
    npz_name = f'{args.dataset}_{args.model}.npz'
    if args.dataset == 'skincancer2': 
        npz_name = f'{args.dataset}_{args.model}_{args.n_patient}.npz'
    npz_path = os.path.join(npz_dir, npz_name)

    # Feature extraction 
    if args.model == 'clip':
        import clip
        from models.llm import features_extraction
        print("Create dataloader")
        # Load model and get data
        model, preprocess = clip.load(args.backbone)
        if args.dataset == 'skincancer2': 
            dataset = dataloader_conversion[args.dataset](args.root_dir, preprocess, args.n_patient)
        else:
            print("args.data_dir", args.data_dir)
            dataset = dataloader_conversion[args.dataset](args.data_dir, preprocess)
        dataloader = DataLoader(dataset, 16)
        print("Get features")
        # Get features
        images, img_features, text_features, labels, cosine_similarities, positions = features_extraction(args, model, None, dataset, dataloader, get_position)
        n_labels = len(dataset.classnames)
        n_samples = images.size()[0]
        height = 1
        
        if get_position:
            n_samples = torch.max(positions[0]).item() // 254 +1
            height = torch.max(positions[1]).item() // 254 +1

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

    elif args.model == 'quilt':
        from open_clip import create_model_and_transforms, get_tokenizer
        from models.llm import features_extraction

        # Load model and get data
        model, preprocess, _ = create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        tokenizer = get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')
        if args.dataset == 'skincancer2': 
            dataset = dataloader_conversion[args.dataset](args.root_dir, preprocess, args.n_patient)
        else:
            dataset = dataloader_conversion[args.dataset](args.data_dir, preprocess)
        dataloader = DataLoader(dataset, 16)
        
        # Get features
        images, img_features, text_features, labels, cosine_similarities, positions = features_extraction(args, model.cuda(), tokenizer, dataset, dataloader, get_position)
        n_labels = len(dataset.classnames)
        n_samples = images.size()[0]
        height = 1

        if get_position:
            n_samples = torch.max(positions[0]).item() // 254 +1
            height = torch.max(positions[1]).item() // 254 +1

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
        print(f"Saved file to {npz_path}")

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
        else:
            dataset = dataloader_conversion[args.dataset](args.data_dir, preprocess)
        dataloader = DataLoader(dataset, 16)

        # Get features
        images, img_features, text_features, labels, cosine_similarities, positions = features_extraction(args, model.cuda(), None, dataset, dataloader, get_position)
        n_labels = len(dataset.classnames)
        n_samples = images.size()[0]
        height = 1

        if get_position:
            n_samples = torch.max(positions[0]).item() // 254 +1
            height = torch.max(positions[1]).item() // 254 +1

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


    elif args.model == 'conch':
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        from models.llm import features_extraction

        # Load model and get data
        model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="checkpoints/conch/pytorch_model.bin")
        tokenizer = get_tokenizer()
        if args.dataset == 'skincancer2': 
            dataset = dataloader_conversion[args.dataset](args.root_dir, preprocess, args.n_patient)
        else:
            dataset = dataloader_conversion[args.dataset](args.data_dir, preprocess)
        dataloader = DataLoader(dataset, 16)

        # Get features
        images, img_features, text_features, labels, cosine_similarities, positions = features_extraction(args, model, tokenizer, dataset, dataloader, get_position)
        n_labels = len(dataset.classnames)
        n_samples = images.size()[0]
        height = 1

        if get_position:
            n_samples = torch.max(positions[0]).item() // 254 +1
            height = torch.max(positions[1]).item() // 254 +1


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
        
    elif args.model == 'nnunet':
        from models.nnunet import features_extraction
        features_extraction(args)

    elif args.model == 'uni':
        #from models.uni import features_extraction
        print('???TO IMPLEMENT: HOW TO OBTAIN THE PROBABILITIES AS THERE IS NO TEXT ENCODER???')
    
    elif args.model == 'nnunet_patches':
        from models.nnunet_patches import features_extraction_nnunet
        features_extraction_nnunet(args)

    elif args.model == 'nnunet_bigpatches':
        from models.nnunet_bigpatches import features_extraction_nnunet
        features_extraction_nnunet(args)

    elif args.model == 'plipanduni2':
        import clip
        import timm 
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
        from torchvision import transforms
        from models.llm import features_extraction

        # Load both model2 and get data
        _, preprocess = clip.load(args.backbone)
        processor = AutoProcessor.from_pretrained("vinid/plip")
        model1 = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip")     
        if args.dataset == 'skincancer2': 
            print("args.n_patient", args.n_patient)
            dataset1 = dataloader_conversion[args.dataset](args.root_dir, preprocess, args.n_patient)
        else:
            dataset1 = dataloader_conversion[args.dataset](args.data_dir, preprocess)
        dataloader1 = DataLoader(dataset1, 16)
        
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
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        if args.dataset == 'skincancer2': 
            dataset2 = dataloader_conversion[args.dataset](args.root_dir, transform2, args.n_patient)
        else:
            dataset2 = dataloader_conversion[args.dataset](args.data_dir, transform2)
        dataloader2 = DataLoader(dataset2, 16)

        # Get features
        args.model = 'uni2'
        print("UNI2")
        _, img_features, __, ___, ____ = features_extraction(args, model2.cuda(), None, dataset2, dataloader2, get_position, True)
        args.model = 'plip'
        print("PLIP")
        images, image_features, text_features, labels, cosine_similarities, positions = features_extraction(args, model1.cuda(), None, dataset1, dataloader1, get_position, False)
        n_labels = len(dataset1.classnames)
        n_samples = images.size()[0]
        height = 1

        if get_position:
            n_samples = torch.max(positions[0]).item() // 254 +1
            height = torch.max(positions[1]).item() // 254 +1

        np.savez_compressed(
            npz_path, 
            width=n_samples, 
            height=height, 
            n_labels=n_labels, 
            probabilities=cosine_similarities.cpu().numpy(), 
            features=img_features.detach().cpu().numpy(), 
            text_features=text_features.detach().cpu().numpy(),
            image_features=image_features.detach().cpu().numpy(), 
            images=images.cpu().numpy(), 
            labels=labels.cpu().numpy(),
            positions=positions
        )
    else:
        raise RuntimeError(f"Features extraction is not implemented yet for the model {args.model}!")


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='sicap_mil', choices=['sicap_mil', 'monuseg', 'skincancer', 'panuke', 'skincancer2', 'lc_lung', 'lc_colon', 'nct'])
    parser.add_argument('--model', type=str, default='clip', choices=['clip', 'quilt', 'plip', 'conch', 'nnunet', 'nnunet_patches', 'nnunet_bigpatches', 'plipanduni2'])
    parser.add_argument('--backbone', type=str, default='ViT-B/16')
    parser.add_argument('--file_ending', type=str, default='.tif')
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--n_patient', type=int, default=0)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parser()
    get_position = False

    extraction(args)