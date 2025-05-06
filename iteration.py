import torch
import torch.nn.functional as F
import numpy as np

device = 'cuda'

def create_pairwise_mask(binary_mask, device='cpu'):
    """
    Create a mask used for the pairwise potential in a few-shot setting.
    
    Args:
        binary_mask: Binary mask of labeled regions (1 for labeled pixels), shape (n, n).
    
    Returns:
        pairwise_mask: Sparse mask (n_pixels, n_pixels).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Flatten the mask
    flattened_mask = binary_mask.flatten()

    # Step 2: Get positions where it's NOT -1 and get annotation 
    positions_annotated = np.where(flattened_mask != -1)[0]
    annotations = torch.tensor(flattened_mask[positions_annotated], dtype=torch.long, device=device)
    positions_annotated = torch.tensor(positions_annotated, device=device)
    positions_not_annotated = torch.tensor(np.where(flattened_mask == -1)[0],  device=device)

    return positions_annotated, positions_not_annotated, annotations


def mean_field_iteration(args, unary, pairwise, compat, max_iters=10):
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
    pairwise /= pairwise.max()
    #print("pairwise small values: ", pairwise[pairwise < 1e-3].size())

    # Initialize Q
    print("unary", unary.size())
    Q = torch.exp(-unary)
    Q /= Q.sum(dim=0, keepdim=True)  # Normalize Q across classes

    for iteration in range(max_iters):
        #Q_prev = Q.clone()

        # Compute Q_tilde (message passing step): pairwise @ Q.T
        Q_tilde = torch.matmul(pairwise, Q.T)  # Shape: (n_pixels, n_classes)

        # Compatibility transform: compat @ Q_tilde.T
        Q_hat = torch.matmul(compat, Q_tilde.T)  # Shape: (n_classes, n_pixels)
        Q_hat /= Q_hat.max() 

        # Update Q
        Q = torch.exp((-unary - Q_hat)) # / args.temperature)
        Q /= Q.sum(dim=0, keepdim=True)  # Normalize Q across classes

        # Debugging output and NaN check
        if torch.isnan(Q).any():
            print(f"NaN detected in Q at iteration {iteration}.")
            break

        '''
        # Check for convergence (optional)
        diff = torch.norm(Q - Q_prev, p='fro')
        if diff < 1e-5:  # Convergence threshold
            print(f"Converged in {iteration + 1} iterations.")
            break
        '''

    return Q


def mean_field_iteration_fs(args, unary, pairwise, compat, max_iters=10, labeled_mask=None):
    """
    Perform mean-field inference for fully connected CRFs with PyTorch optimizations.

    Args:
        args: Argument object with temperature and other parameters.
        unary (torch.Tensor): Unary potential tensor of shape (n_classes, n_pixels).
        pairwise (torch.Tensor): Pairwise potential tensor of shape (n_pixels, n_pixels).
        compat (torch.Tensor): Compatibility matrix of shape (n_classes, n_classes).
        max_iters (int): Maximum number of iterations.
        labeled_mask (torch.Tensor): Labeled mask (n_pixels,).

    Returns:
        torch.Tensor: Final Q distribution (n_classes, n_pixels).
    """
    pairwise /= pairwise.max()

    # Get pairwise_mask
    positions_annotated, positions_not_annotated, annotations = create_pairwise_mask(labeled_mask, device=device)

    # Create a new tensor with zeros
    new_probs = torch.zeros_like(unary[:, positions_annotated])

    # Set the max-probability index to 1 for each target pixel
    new_probs[annotations, torch.arange(len(positions_annotated))] = 1

    # Initialize Q
    Q = torch.exp(-unary)
    Q[:, positions_annotated] = new_probs
    Q /= Q.sum(dim=0, keepdim=True)  # Normalize Q across classes

    for iteration in range(max_iters):
        # Compute Q_tilde (message passing step): pairwise @ Q.T
        sigma = torch.tensor(args.sigma, device=device) 
        minus_sigma = torch.tensor(1-args.sigma, device=device) 

        pairwise_mask = pairwise.clone()
        pairwise_mask[positions_annotated, :] = 0
        pairwise_mask[:, positions_annotated] = 0

        Q_tilde = sigma * torch.matmul(pairwise_mask, Q.T) 
        del pairwise_mask
        pairwise_mask = pairwise.clone()
        pairwise_mask[positions_not_annotated, :] = 0
        pairwise_mask[:, positions_not_annotated] = 0
        Q_tilde += minus_sigma * torch.matmul(pairwise_mask, Q.T) 

        # Compatibility transform: compat @ Q_tilde.T
        Q_hat = torch.matmul(compat, Q_tilde.T)
        Q_hat /= Q_hat.max()
        del Q_tilde

        # Update Q
        Q = torch.exp((-unary -Q_hat) / args.temperature)
        Q[:, positions_annotated] = new_probs
        Q /= Q.sum(dim=0, keepdim=True)  # Normalize Q across classes
        del Q_hat

        # Debugging output and NaN check
        if torch.isnan(Q).any():
            print(f"NaN detected in Q at iteration {iteration}.")
            break
    del positions_annotated, positions_not_annotated, annotations, pairwise

    return Q


def mean_field_iteration_sparse(args, unary, pairwise, compat, max_iters=10, cos_sim=None):
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
    pairwise /= pairwise.max()
    from utils import mask_pairwise_potentials
    pairwise = mask_pairwise_potentials(args, pairwise, cos_sim)

    # Initialize Q
    Q = torch.exp(-unary)
    Q /= Q.sum(dim=0, keepdim=True)  # Normalize Q across classes

    for iteration in range(max_iters):
        #Q_prev = Q.clone()

        # Compute Q_tilde (message passing step): pairwise @ Q.T
        Q_tilde = torch.matmul(pairwise, Q.T)  # Shape: (n_pixels, n_classes)

        # Compatibility transform: compat @ Q_tilde.T
        Q_hat = torch.matmul(compat, Q_tilde.T)  # Shape: (n_classes, n_pixels)
        Q_hat /= Q_hat.max() 

        # Update Q
        Q = torch.exp((-unary - Q_hat) / args.temperature)
        Q /= Q.sum(dim=0, keepdim=True)  # Normalize Q across classes

        # Debugging output and NaN check
        if torch.isnan(Q).any():
            print(f"NaN detected in Q at iteration {iteration}.")
            break

        '''
        # Check for convergence (optional)
        diff = torch.norm(Q - Q_prev, p='fro')
        if diff < 1e-5:  # Convergence threshold
            print(f"Converged in {iteration + 1} iterations.")
            break
        '''

    return Q


def mean_field_iteration(args, npz_path, unary, pairwise, compat, variances, max_iters=10):
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
    from utils import determine_sparsity
    from sparse_potentials import pairwise_potential_from_img_and_position, pairwise_potential_from_model_features_and_position, pairwise_potential_from_prob, pairwise_potential_from_model_features

    potentials_correspondance = {
        "image_and_position": pairwise_potential_from_img_and_position,
        "model_features_and_position": pairwise_potential_from_model_features_and_position, 
        "prob": pairwise_potential_from_prob, 
        "model_features": pairwise_potential_from_model_features
    }
    get_pairwise_potential = potentials_correspondance[args.pair_pot]

    # Initialize Q
    Q = torch.exp(-unary)
    Q /= Q.sum(dim=0, keepdim=True)  # Normalize Q across classes
    Q_prev = torch.clone(Q) 

    for iteration in range(max_iters):
        # Compute Q_tilde (message passing step): pairwise @ Q.T
        Q_tilde = torch.matmul(pairwise, Q.T)  # Shape: (n_pixels, n_classes)

        # Compatibility transform: compat @ Q_tilde.T
        Q_hat = torch.matmul(compat, Q_tilde.T)  # Shape: (n_classes, n_pixels)
       # Q_hat /= Q_hat.max()
        print("Q_hat", Q_hat[:,:5])
        # Update Q
        Q = torch.exp((-unary - args.weight[0]*Q_hat))
        Q_logit = -unary - args.weight[0]*Q_hat
        Q_logit = Q_logit - Q_logit.max(dim=0, keepdim=True).values
        Q = torch.exp(Q_logit) # / args.temperature)
        Q /= Q.sum(dim=0, keepdim=True)  # Normalize Q across classes
        print("Q", Q[:,0])

        if args.sparse_method in ['random', 'nonlocalrandom']:
            img = None
            if args.sparse_method in ['nonlocalrandom']:
                img = np.load(npz_path)['images']
            all_positions, factors = determine_sparsity(args, unary.size()[1], npz_path, img=img)
            pairwise = get_pairwise_potential(args, npz_path, all_positions, factors, variances, args.weight).to(device)
                                                                                                        
        # Debugging output and NaN check
        if torch.isnan(Q).any():
            print(f"NaN detected in Q at iteration {iteration}.")
            break

        # Check for convergence (optional)
        diff = torch.norm(Q - Q_prev, p='fro')
        if diff < 1e-9:  # Convergence threshold
            print(f"Converged in {iteration + 1} iterations.")
            break

        Q_prev = torch.clone(Q)

    return Q