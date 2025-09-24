import math
import numpy as np 
import torch 


def highest_entropy(z, topk = 30):
    N,K = z.shape
    max_entropy = -torch.log(torch.tensor(1/K)+1e-7)
    z_entropy = -torch.sum(z * torch.log(z+1e-7), dim = -1)/max_entropy
    indexes_sort = torch.argsort(z_entropy)
    return indexes_sort[-topk:], z_entropy, indexes_sort


def smallest_top2_margin(z, topk = 30):
    N,K = z.shape
    sorted_z, indexes_sort = torch.sort(z, dim = -1)
    margins = sorted_z[:,1] - sorted_z[:,0]
    indexes_sort = torch.argsort(margins)
    return indexes_sort[:topk], margins, indexes_sort


def least_confident(z, topk = 30):
    N,K = z.shape
    z_pred, pred, = torch.max(z, dim = -1)
    indexes_sort = torch.argsort(z_pred)
    return indexes_sort[:topk], z_pred, indexes_sort


def get_annotation_iterative(args, npz_path, probabilities=None):
    data = np.load(npz_path)
    labels = data['labels']

    # Get annotations and their label
    annotations = args.annotations
    label_annotations = labels[annotations]
    n_annotations = len(annotations)
   
    rng = np.random.default_rng(seed=args.seed)
    if args.annotation_method == 'random':
        images_idx = np.arange(1, len(labels))
        images_idx_not_annotated = [int(i) for i in images_idx if i not in annotations]
        images_idx_selected = rng.permutation(images_idx_not_annotated)[:args.n_annotations]

    elif args.annotation_method == 'entropy':
        if probabilities is None:
            probabilities = data['probabilities']
        probabilities = torch.tensor(probabilities)
        images_idx_selected, _, _ = highest_entropy(probabilities, topk = args.n_annotations)

    elif args.annotation_method == 'margin':
        if probabilities is None:
            probabilities = data['probabilities']
        probabilities = torch.tensor(probabilities)
        images_idx_selected, _, _ = smallest_top2_margin(probabilities, topk = args.n_annotations)

    elif args.annotation_method == 'least_confident':
        if probabilities is None:
            probabilities = data['probabilities']
        probabilities = torch.tensor(probabilities)
        images_idx_selected, _, _ = least_confident(probabilities, topk = args.n_annotations)

    elif args.annotation_method == 'error':
        if probabilities is None:
            probabilities = data['probabilities']  #(n_samples, n_class)
        predicted_labels = np.argmax(probabilities, axis=1)
        gt_labels = data['labels']
        wrong_labels = np.where(predicted_labels != gt_labels)[0]
        wrong_labels = [int(i) for i in wrong_labels if i not in annotations]
        images_idx_selected = rng.permutation(wrong_labels)[:args.n_annotations]

    elif args.annotation_method == 'diverseerror':
        if probabilities is None:
            probabilities = data['probabilities']  #(n_samples, n_class)
        predicted_labels = np.argmax(probabilities, axis=1)
        gt_labels = data['labels']
        wrong_labels = np.where(predicted_labels != gt_labels)[0]
        images_idx_selected = []
        if len(wrong_labels) == 0:
            # Fallback: no errors, sample randomly
            print("Warning: no misclassified samples found, selecting random annotations instead.")
            all_indices = np.arange(len(gt_labels))
            images_idx_selected = rng.choice(all_indices, size=args.n_annotations, replace=False).tolist()
        else:
            n_class = len(np.unique(gt_labels[wrong_labels]))
            for n in range(args.n_annotations):
                class_to_select = (n_annotations + n) % n_class
                wrong_labels_class_to_select = [l for l in wrong_labels if gt_labels[l] == class_to_select]

                if len(wrong_labels_class_to_select) == 0:
                    # If no misclassified sample for this class, fallback to any wrong label
                    idx = rng.choice(wrong_labels)
                else:
                    idx = rng.permutation(wrong_labels_class_to_select)[0]

                images_idx_selected.append(idx)

    elif args.annotation_method == 'oracle':
        gt_labels = data['labels']
        n_class = len(np.unique(gt_labels))
        images_idx_selected = []
        for c in range(n_class)[:1]:
            c_labels = np.where(gt_labels == c)[0]
            c_labels = rng.permutation(c_labels)
            images_idx_selected.append(c_labels[0])

    elif args.annotation_method == 'circle':
        if args.n_patient == 0: 
            x_centers = [58, 16, 85, 34, 34] #patient 5
            y_centers = [17, 36, 44, 36, 36]
        elif args.n_patient == 9: 
            x_centers = [50, 91, 56, 103, 106] # patient 6 [-1]
            y_centers = [39, 42, 15, 25, 34]
        elif args.n_patient == 1: 
            x_centers = [96, 36, 13, 76, 76] #patient 2 [1]
            y_centers = [43, 81, 32, 37, 37]
        elif args.n_patient == 5: 
            x_centers = [9, 43, 62, 23, 62] #patient 9
            y_centers = [46, 31, 15, 29, 15]
        else: 
            raise RuntimeError("Not implemented yet for patient {args.n_patient}")

        positions = data['positions']

        x_coords = positions[:positions.shape[0]//2]
        y_coords = positions[positions.shape[0]//2:]

        # Optional: compute step (spacing) to normalize to a grid
        coords = np.stack([x_coords, y_coords], axis=1)

        # Circle definition
        x_center, y_center = x_centers[args.ann_it] * 512, y_centers[args.ann_it] * 512
        center = np.array([x_center, y_center])   # circle center
        diameter = 512 * 8
        radius = diameter / 2

        # compute distances of all points to the center
        distances = np.linalg.norm(coords - center, axis=1)

        # select all indices inside circle (and not already annotated)
        images_idx_selected = [i for i, dist in enumerate(distances) 
                            if dist <= radius and i not in annotations]

    else:
        raise RuntimeError(f"Not implemetend yet for {args.annotation_method} annotation method")

    annotations = np.concatenate((annotations, images_idx_selected))
    annotations = [int(a) for a in annotations]
    label_annotations = np.concatenate((label_annotations, labels[images_idx_selected]))

    return annotations, label_annotations 


def get_annotation(args, npz_path, return_class=False):
    data = np.load(npz_path)
    labels = data['labels']
    n_class = np.unique(labels)
    annotations = []
    label_annotations = []
    if return_class:
        for c in n_class:
            label_annotations.extend([c]*args.n_annotations)
        return label_annotations
    for c in n_class:
        rng = np.random.default_rng(seed=args.seed)
        c_labels = np.where(labels == c)[0]  #np.where(labels == c)[0] #random.shuffle(np.where(labels == c)[0])
        c_labels = rng.permutation(c_labels)
        if len(c_labels) < args.n_annotations:
            annotations.extend(c_labels)
            annotations.extend([c_labels[0]]*(args.n_annotations-len(c_labels)))
        else:
            annotations.extend(c_labels[:args.n_annotations])
        label_annotations.extend([c]*args.n_annotations)
    return annotations, label_annotations


def unitary_potential_from_softmax_unlabeled_and_annotation(args, npz_path, probabilities=None, text_features=None):
    # Load the .npz file and access the required keys 
    data = np.load(npz_path)
    image_features = data["image_features"]
    if text_features is None : text_features = data["text_features"] #(n_class, 512)
    if probabilities is None: 
        probabilities = data["probabilities"] #(n_samples, n_class)
    softmax = torch.nn.Softmax(dim=0)
    probabilities = softmax(probabilities)

    annotations = args.annotations 
    labels = np.load(npz_path)['labels']
    label_annotations = labels[annotations]
   
    n_class = np.unique(labels)

    v = np.einsum('ij,ik->jk', probabilities, image_features) / np.sum(probabilities, axis=0)[:, np.newaxis]

    if args.beta: 
        from utils import update_beta
        beta = update_beta(probabilities, alpha=0.5, soft=True)

    updated_text_features = np.copy(text_features)
    for c in n_class:
        # Labeled images contribution 
        label_c_annotations = np.where(label_annotations == c)[0]
        if len(label_c_annotations) > 0:
            annotations_c = [annotations[i] for i in label_c_annotations]
            image_features_annotations = image_features[annotations_c, :] 

        # Unlabeled images contribution 
        v_c = v[c, :]

        # Mean of contribution 
        if len(label_c_annotations) > 0:
            mean_image_feat_i = np.mean(image_features_annotations, axis=0)
            mean_image_feat_i /= np.linalg.norm(mean_image_feat_i, keepdims=True) #(512,)
        if not args.beta:
            if not len(label_c_annotations) > 0:
                updated_text_features[c, :] = np.mean(np.concatenate((np.expand_dims(text_features[c,:], axis=1), np.expand_dims(v_c, axis=1)), axis=1), axis=1)
            else:
                updated_text_features[c, :] = np.mean(np.concatenate((np.expand_dims(mean_image_feat_i.T, axis=1), np.expand_dims(text_features[c,:], axis=1), np.expand_dims(v_c, axis=1)), axis=1), axis=1)
        else:
            option = 1
            if option == 1: 
                if not len(label_c_annotations) > 0:
                    updated_text_features[c, :] = np.mean(np.concatenate((np.expand_dims(text_features[c,:], axis=1), np.expand_dims(v_c, axis=1)), axis=1), axis=1)
                else:
                    beta_part = beta[c]*mean_image_feat_i + (1-beta[c])*v_c
                    updated_text_features[c, :] = np.mean(np.concatenate((np.expand_dims(text_features[c,:], axis=1), np.expand_dims(beta_part, axis=1)), axis=1), axis=1)
            elif option == 2: 
                beta_part = (1-beta[c])*text_features[c,:] + beta[c]*v_c
                if not len(label_c_annotations) > 0:
                    updated_text_features[c, :] = beta_part
                else:
                    updated_text_features[c, :] = np.mean(np.concatenate((np.expand_dims(mean_image_feat_i, axis=1), np.expand_dims(beta_part, axis=1)), axis=1), axis=1)
        updated_text_features[c, :] /= np.linalg.norm(updated_text_features[c, :], keepdims=True) 

    probabilities = image_features @ updated_text_features.T
    probabilities = softmax(probabilities)
    unitary_potential = -torch.log(probabilities) 

    return unitary_potential.T, updated_text_features 


def annotated_unitary_potential(args, labels, Q):
    annotated_positions = args.annotations 
    classes_not_annotated =  args.classes_not_annotated

    annotated_labels = labels[annotated_positions]

    for annotated_position, annotated_label in zip(annotated_positions, annotated_labels):
        if annotated_label not in classes_not_annotated:
            new_Q = torch.ones((Q.size()[0]))*(-math.log(0.1)) #!!!!! torch.zeros
            new_Q[annotated_label] = 0
            Q[:, annotated_position] = new_Q

    return Q