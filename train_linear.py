import argparse
import numpy as np 
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers.modeling_outputs import ImageClassifierOutput
import pytorch_warmup as warmup
import wandb

from dataloaders.features import Features


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]

    return acc


def infer(args, linear_layer, test_loader):
    linear_layer.eval()

    outputs = []
    with torch.no_grad():
        for i, (image_features, target) in enumerate(test_loader):
            image_features, target = image_features.cuda(), target.cuda()

            if args.model in ["clip"]:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    output = linear_layer(image_features)
            else:
                output = linear_layer(image_features)

            if isinstance(output, ImageClassifierOutput):
                output = output.logits
            outputs.append(output.cuda())
    outputs = torch.cat(outputs, dim=0)

    return outputs


def train_linear(args):
    # Fix seed
    np.random.seed(args.seed) # Set NumPy seed
    torch.manual_seed(args.seed) # Set PyTorch seed for CPU
    torch.cuda.manual_seed(args.seed) # If using CUDA

    # Get directory
    npz_dir = os.path.join(args.root_dir, 'data_processed', args.dataset, args.model)
    if not os.path.exists(npz_dir): 
        os.makedirs(npz_dir)
    npz_name = f'{args.dataset}_{args.model}.npz'
    if args.dataset in ['skincancer2', 'bach_wsi']: 
        npz_name = f'{args.dataset}_{args.model}_{args.n_patient}.npz'
    npz_path = os.path.join(npz_dir, npz_name)

    if WANDB:
        name_run = f"{args.model}_{args.dataset}_{args.lr}_{args.n_iters}_{args.seed}_{args.n_annotations}"
        wandb.init(
            project="crf_linear", name=name_run
        )
        config = wandb.config
        config.model = args.model
        config.dataset = args.dataset
        config.lr = args.lr
        config.n_iters = args.n_iters
        config.seed = args.seed
        config.n_annotations = args.n_annotations
        config.weight_decay = 1e-2
        config.beta1 = 0.9
        config.beta2 = 0.999
        config.warmup_period = 10
        config.batch_size = args.batch_size


    # Get dataloader 
    train_dataset = Features(args, npz_path, test=False)
    test_dataset = Features(args, npz_path, test=True)
    train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    num_workers=5,
                    shuffle=True,
                    pin_memory=True,
                    )
    test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.batch_size,
                    num_workers=5,
                    shuffle=False,
                    pin_memory=True,
                    )

    # Construct linear layer 
    num_classes = len(np.unique(np.load(npz_path)['labels']))
    if args.model in ["quilt", "biomedclip", "conch", "conchanduni2", "conchandgigapath", "conchandoptimus1", "plip"]:
        num_features = 512
    else:
        raise RuntimeError("Not implemented yet")
    linear_layer = nn.Sequential(
        nn.Flatten(start_dim=1), nn.Linear(num_features, num_classes)
    ).cuda()

    
    linear_layer = linear_layer.cuda()
    trainable_parameters = []
    for _, param in linear_layer.named_parameters():
        trainable_parameters.append(param)

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        lr=args.lr,
    )

    num_steps = args.n_iters 
    warmup_period = 10
    total_iters = warmup_period + num_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_steps, eta_min=1e-6
    )
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)

    # Training
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    while count_iters < total_iters:
        linear_layer.train()

        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.0
        
        for i, (images, target) in enumerate(tqdm(train_loader)):

            images, target = images.cuda(), target.cuda()
            output = linear_layer(images)
            print("output", output.size(), output[0,:])
            if isinstance(output, ImageClassifierOutput):
                output = output.logits

            loss = F.cross_entropy(output, target)
            acc_train += cls_acc(output, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period:
                    scheduler.step()

        count_iters += 1

        acc_train /= tot_samples
        loss_epoch /= tot_samples

        current_lr = scheduler.get_last_lr()[0]
        for param_group in optimizer.param_groups:
            optimizer_lr = param_group["lr"]
        print(
            " OptLR: {:.6f}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}".format(
                optimizer_lr, current_lr, acc_train, loss_epoch
            )
        )

        if WANDB:
            wandb.log({"train/loss": loss_epoch, "train/accuracy": acc_train})        

    new_probs = infer(args, linear_layer, test_loader)

    if WANDB:
        wandb.finish()

    return new_probs


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--dataset', type=str, default='sicap_mil', choices=['sicap_mil', 'monuseg', 'skincancer', 'panuke', 'skincancer2', 'lc_lung', 'lc_colon', 'nct', 'tcga-ut', 'bracs'])
    parser.add_argument('--model', type=str, default='conchanduni2', choices=['clip', 'quilt', 'plip', 'conch', 'nnunet', 'nnunet_patches', 'nnunet_bigpatches', 'plipanduni2', 'conchanduni2', 'conchandgigapath', 'conchandvirshow2', 'conchandoptimus1'])
    parser.add_argument('--patch_size', type=int, default=254)
    parser.add_argument('--n_patient', type=int, default=0)
    # Model parameter
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_iters', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    # Exp parameter
    parser.add_argument('--annotations', type=int, nargs='+', default=None)
    parser.add_argument('--n_annotations', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parser()
    WANDB = False

    train_linear(args)
