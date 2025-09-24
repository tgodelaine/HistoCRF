import numpy as np
import time
import torch
import torch.nn.functional as F

from utils_annotations import annotated_unitary_potential

device = 'cuda'


def apply_compatibility_transform(compat, Q_tiled):
    return [c @ Q_t.T for c, Q_t in zip(compat, Q_tiled)]


def compute_Q_tilde_sparse(pairwise, Q):
    #Q_t = Q.T.contiguous()
    return [torch.sparse.mm(p, Q.T) for p in pairwise]


def mean_field_iteration_split(args, npz_path, unary, pairwise, compat, variances, max_iters=10, cossim=None, data=None):
    """
    Perform mean-field inference for fully connected CRFs with PyTorch optimizations.

    Args:
        args: Argument object with temperature and other parameters.
        unary (torch.Tensor): Unary potential tensor of shape (n_classes, n_pixels).
        pairwise (torch.Tensor): Pairwise potential tensor of shape (n_pixels, n_pixels).
        compat (torch.Tensor): Compatibility matrix of shape (n_classes, n_classes).
        max_iters (int): Maximum number of iterations.

    Returns:
        torch.Tensor: Final Q distribution (n_classes, n_pixels).
    """
    n_class = len(np.unique(data["labels"]))
    labels = data["labels"]

    # Initialize Q
    Q = torch.nn.functional.softmax(-unary, dim=0).clamp(min=1e-8)
    Q_prev = torch.clone(Q) 

    for iteration in range(max_iters):
        args.crf_it = iteration 

        # Compute Q_tilde (message passing step): pairwise @ Q.T
        Q_tilde = compute_Q_tilde_sparse(pairwise, Q)

        # Compatibility transform: compat @ Q_tilde.T
        Q_hats = apply_compatibility_transform(compat, Q_tilde)

        # Automatic weight
        if args.n_annotations > 0:
            n_patch_bound_annotation = torch.unique(torch.as_tensor(args.annotated_row_positions)).numel()
            if args.n_affinity[0] == 0:
                ratios = [1, unary.size()[1]/(n_patch_bound_annotation)] 
            else:
                ratios = [1, 1, unary.size()[1]/(n_patch_bound_annotation)] 
        else:
            ratios = [1, 1, 1]
        weight_up_method = 'fix'
        if iteration == 0:
            if weight_up_method == "automatic":
                w = np.copy(args.weight)
                for i, _ in enumerate(compat):
                    w[i] = torch.abs(args.weight[i] * torch.mean(unary)/torch.mean(Q_hats[i])) 
                    if i==1: 
                        w[i] = torch.abs(args.weight[i] * n_class * torch.mean(unary)/(torch.mean(Q_hats[i])*ratios[i]))
            elif weight_up_method == "fix":
                if args.it % args.N_PROP == 0:
                    w = np.copy(args.weight)
                    for i, _ in enumerate(compat):
                        w[i] = torch.abs(args.weight[i] * torch.mean(unary)/(torch.mean(Q_hats[i])))
                        if i==1: 
                            w[i] = torch.abs(args.weight[i] * n_class * torch.mean(unary)/(torch.mean(Q_hats[i])*ratios[i]))
                    args.weight_initial = np.copy(w)
                else:
                    w = np.copy(args.weight_initial)
            elif weight_up_method == "moving_average":
                w = np.copy(args.weight)
                alpha = 0.8
                w = np.copy(args.weight)
                for i, _ in enumerate(compat):
                    w[i] = torch.abs(args.weight[i] * torch.mean(unary)/torch.mean(Q_hats[i]))
                    if i==1: 
                        w[i] = torch.abs(args.weight[i] * n_class * torch.mean(unary)/(torch.mean(Q_hats[i])*ratios[i]))
                if args.it > 0:
                    for i, _ in enumerate(compat):
                        w[i] = (1-alpha)*args.weight_initial[i] + alpha*w[i]
                args.weight_initial = np.copy(w)

        # Update Q
        Q_logit = torch.clone(unary) 
        for i, Q_h in enumerate(Q_hats): 
            Q_logit += w[i]*Q_h

        if len(args.annotations) > 0 : 
            Q_logit = annotated_unitary_potential(args, labels, Q_logit)#
   
        Q = torch.nn.functional.softmax(-Q_logit, dim=0).clamp(min=1e-8)

        if torch.isnan(Q).any():
            print(f"NaN detected in Q at iteration {iteration}.")
            break

        # Check for convergence (optional)
        kl_div = F.kl_div(Q.t().log(), Q_prev.t(), reduction='batchmean')
        if kl_div.item() < 1e-6:  # Convergence threshold e-6
            print(f"Converged at iteration {iteration}") 
            break
        
        Q_prev.copy_(Q)

    return Q


def label_prop(args, npz_path, unary, pairwise, sparse=False):
    data = np.load(npz_path)

    labels = torch.tensor(data["labels"], dtype=torch.long, device=unary.device)

    # Compute Y
    n_classes, n_samples = unary.shape
    Y = torch.zeros((n_samples, n_classes), dtype=unary.dtype, device=unary.device)
    labeled_images = args.annotations
    Y[labeled_images] = torch.nn.functional.one_hot(labels[labeled_images], num_classes=n_classes).to(unary.dtype) ## Fill in one-hot for labeled rows

    if args.N_UP == -50: # Instead of a hard matrix (only O and 1), soft matrix using probabilities
        softmax = torch.nn.Softmax(dim=1)
        Y = softmax(torch.tensor(data['probabilities'])).to(device=device, dtype=unary.dtype)   

    # Pairwise
    if sparse:
        if len(pairwise) == 1:
            A = pairwise[0].to(dtype=unary.dtype, device=unary.device).to_dense() 
        else: # Addition of non-local and annotation pairwise potential
            A = pairwise[0].to(dtype=unary.dtype, device=unary.device).to_dense() + pairwise[1].to(dtype=unary.dtype, device=unary.device).to_dense() 
        W = A + A.T
    else:
        W = pairwise[0].to(dtype=unary.dtype, device=unary.device)   

    # Compute D^(-1/2)
    I = torch.ones((W.size()[0]), dtype=unary.dtype, device=device)
    D_prim = torch.pow(((W @ I)), -0.5)
    D = torch.diag(D_prim)

    # Compute omega
    identity = torch.eye(unary.size()[1], device=device, dtype=unary.dtype)
    omega = (D @ W) @ D

    # Propagate
    alpha = 0.99
    Q = torch.linalg.inv((identity-alpha*omega).float())@Y.float()

    return Q.T 