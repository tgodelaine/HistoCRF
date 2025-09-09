import numpy as np
#from scipy.sparse import lil_matrix, dok_matrix
import torch
#from sklearn.decomposition import PCA 

device = 'cuda'

def compute_diff(f):
    batch_size = 1000  # Choose a suitable batch size
    num_points = f.size(0)
    diff_f = torch.zeros((num_points, num_points), dtype=torch.float16)
    for i in range(0, num_points, batch_size):
        end_i = min(i + batch_size, num_points)
        for j in range(0, num_points, batch_size):
            end_j = min(j + batch_size, num_points)
            diff_f[i:end_i, j:end_j] = torch.cdist(
                f[i:end_i].float(),
                f[j:end_j].float(),
                p=2
            ) ** 2
    return diff_f

def unitary_potential_from_softmax(args, probabilities):
    # Load the .npz file and access the required keys 
    '''
    data = np.load(npz_path)
    probabilities = softmax(torch.tensor(data['probabilities'])) # / args.temperature))
    '''
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(probabilities) # / args.temperature))
    #unitary_potential = -np.log(probabilities)
    unitary_potential = -(torch.log(probabilities)) #.clamp(min=1e-8)))
    #print("data['probabilities']", data['probabilities'].mean(), data['probabilities'].min(), data['probabilities'].max())
    #print("unitary_potential", unitary_potential.mean(), unitary_potential.min(), unitary_potential.max())
    #unitary_potential = -(torch.tensor(data['probabilities']+1))
    #print("unitary_potential", unitary_potential.size())
    return unitary_potential.T     #.reshape((n_labels, -1)) ####### !!!!! Enlever le .T


def unitary_potential_from_softmax_and_annotation(args, npz_path):
    # Load the .npz file and access the required keys 
    data = np.load(npz_path)
    n_class = len(np.unique(data["labels"]))
    image_features, text_features = data["image_features"], data["text_features"]
    annotations = args.annotations 
    n_annotation = args.n_annotations
    label_annotations = data["labels"][annotations]

    updated_text_features = np.zeros_like(text_features)
    for i in range(n_class):  #au lieu d'utiliser n_annotation, aller rechercher la classe de chacune des images annotée (pour prendre le cas où toutes les classes ne sont pas présentes sur l'image)
        idx_i = np.where(label_annotations == i)[0]
        print("idx_i", idx_i)
        annotations_i = np.array(annotations)[idx_i]
        if len(annotations_i) > 0:
            image_features_i = image_features[annotations_i, :]
            mean_image_feat_i = np.mean(image_features_i, axis=0)
            #mean_image_feat_i /= np.linalg.norm(mean_image_feat_i, keepdims=True) #(512,)
            updated_text_features[i, :] = (mean_image_feat_i + text_features[i,:])/2 #np.mean(np.concatenate((mean_image_feat_i.T, np.expand_dims(text_features[i,:], axis=1)), axis=1))
            #updated_text_features[i, :] = np.mean(np.concatenate((np.expand_dims(mean_image_feat_i.T, axis=1), np.expand_dims(text_features[i,:], axis=1)), axis=1), axis=1)
        else: 
            updated_text_features[i, :] = (text_features[i,:])/2 
        updated_text_features[i, :] /= np.linalg.norm(updated_text_features[i, :], keepdims=True) #.norm(dim=-1, keepdim=True)

    '''
    updated_text_features = np.zeros_like(text_features)
    for i in range(len(annotations) // n_annotation):  #au lieu d'utiliser n_annotation, aller rechercher la classe de chacune des images annotée (pour prendre le cas où toutes les classes ne sont pas présentes sur l'image)
        annotations_i = annotations[i:2*i+2]
        image_features_i = image_features[annotations_i, :]
        #updated_text_features[i, :] = np.mean(image_features_i.T, axis=1)
        mean_image_feat_i = np.mean(image_features_i, axis=0)
        mean_image_feat_i /= np.linalg.norm(mean_image_feat_i, keepdims=True) #(512,)
        #updated_text_features[i, :] = np.mean(np.concatenate((mean_image_feat_i.T, np.expand_dims(text_features[i,:], axis=1)), axis=1))
        updated_text_features[i, :] = np.mean(np.concatenate((np.expand_dims(mean_image_feat_i.T, axis=1), np.expand_dims(text_features[i,:], axis=1)), axis=1), axis=1)
        updated_text_features[i, :] /= np.linalg.norm(updated_text_features[i, :], keepdims=True) #.norm(dim=-1, keepdim=True)
    '''

    softmax = torch.nn.Softmax(dim=1)
    probabilities = image_features @ updated_text_features.T
    probabilities = softmax(torch.tensor(probabilities)) 
    unitary_potential = -np.log(probabilities)
    return unitary_potential.T, updated_text_features     #.reshape((n_labels, -1)) ####### !!!!! Enlever le .T


'''
def unitary_potential_from_softmax_unlabeled_and_annotation(args, npz_path):
    from utils import update_beta

    # Load the .npz file and access the required keys 
    data = np.load(npz_path)
    image_features, text_features = data["image_features"], data["text_features"]
    labels = data['labels']

    annotations = args.annotations 
    label_annotations = args.label_annotations

    n_class = np.unique(labels)
    classes_not_annotated = [1]

    updated_text_features = np.copy(text_features)
    for c in n_class:
        #img_c = np.where(labels == c)[0]
        annotations_c = np.where(label_annotations == c)[0]
        annotations_c = annotations[annotations_c] #annotations_i
        #unannotation_c =  [img for img in img_c if img not in annotations_c]
        #unannotated_features_c = image_features[unannotation_c, :]
        annotated_features_c = image_features[annotations_c, :] #image_features_i

        mean_image_feat_i = np.mean(annotated_features_c, axis=0)
        mean_image_feat_i /= np.linalg.norm(mean_image_feat_i, keepdims=True) #(512,)
        updated_text_features[c, :] = np.mean(np.concatenate((np.expand_dims(mean_image_feat_i.T, axis=1), np.expand_dims(text_features[c,:], axis=1)), axis=1), axis=1)
        updated_text_features[c, :] /= np.linalg.norm(updated_text_features[c, :], keepdims=True) 

    softmax = torch.nn.Softmax(dim=1)
    probabilities = image_features @ updated_text_features.T
    probabilities = softmax(torch.tensor(probabilities / args.temperature))
    print("Softmax output: ", probabilities[:2,:])
    unitary_potential = -np.log(probabilities)
    return unitary_potential.T   
'''


def unitary_potential_from_softmax_unlabeled_and_annotation(args, npz_path, beta=False):

    # Load the .npz file and access the required keys 
    data = np.load(npz_path)
    image_features, text_features = data["image_features"], data["text_features"]
    probabilities = data["probabilities"]

    annotations = args.annotations 
    labels = np.load(npz_path)['labels']
    label_annotations = labels[annotations]
   
    n_class = np.unique(labels)
    rng = np.random.default_rng(seed=args.seed)
    n_class_randomized = rng.permutation(n_class)
    if args.n_class_not_annotated >= len(n_class):
        classes_not_annotated = n_class-1
    else:
        classes_not_annotated = n_class_randomized[:args.n_class_not_annotated]
    args.classes_not_annotated = classes_not_annotated

    v = np.einsum('ij,ik->jk', probabilities, image_features) / np.sum(probabilities, axis=0)[:, np.newaxis]

    updated_text_features = np.copy(text_features)
    for c in n_class:
        # Labeled contribution
        if args.n_annotations > 0:
            annotations_ = np.where(label_annotations == c)[0]
            annotations_c = [annotations[i] for i in annotations_]
            annotated_features_c = image_features[annotations_c, :] 

        # Unlabeled contribution
        v_c = v[c, :]

        if args.beta: 
            from utils import update_beta
            beta = update_beta(probabilities, alpha=1.0, soft=True)

        # Mean
        if args.n_annotations > 0:
            mean_image_feat_i = np.mean(annotated_features_c, axis=0)
            mean_image_feat_i /= np.linalg.norm(mean_image_feat_i, keepdims=True) #(512,)
        if not args.beta:
            if c in classes_not_annotated or args.n_annotations == 0:
                print(f"Number of annotations = {args.n_annotations}")
                updated_text_features[c, :] = np.mean(np.concatenate((np.expand_dims(text_features[c,:], axis=1), np.expand_dims(v_c, axis=1)), axis=1), axis=1)
            else:
                updated_text_features[c, :] = np.mean(np.concatenate((np.expand_dims(2*mean_image_feat_i.T, axis=1), np.expand_dims(text_features[c,:], axis=1), np.expand_dims(v_c, axis=1)), axis=1), axis=1)
        else:
            option = 1
            if option == 1: 
                if c in classes_not_annotated or args.n_annotations == 0:
                    updated_text_features[c, :] = np.mean(np.concatenate((np.expand_dims(text_features[c,:], axis=1), np.expand_dims(v_c, axis=1)), axis=1), axis=1)
                else:
                    beta_part = beta[c]*mean_image_feat_i + (1-beta[c])*v_c
                    #beta_part = np.mean(np.concatenate(((1-beta[c])*np.expand_dims(mean_image_feat_i, axis=1), beta[c]*np.expand_dims(v_c, axis=1)), axis=1), axis=1) 
                    updated_text_features[c, :] = np.mean(np.concatenate((np.expand_dims(text_features[c,:], axis=1), np.expand_dims(beta_part, axis=1)), axis=1), axis=1)
            elif option == 2: 
                beta_part = (1-beta[c])*text_features[c,:] + beta[c]*v_c
                #beta_part = np.mean(np.concatenate((np.expand_dims((1-beta[c])*text_features[c,:], axis=1), np.expand_dims(beta[c]*v_c, axis=1)), axis=1), axis=1)
                if c in classes_not_annotated or args.n_annotations == 0:
                    updated_text_features[c, :] = beta_part
                else:
                    updated_text_features[c, :] = np.mean(np.concatenate((np.expand_dims(mean_image_feat_i, axis=1), np.expand_dims(beta_part, axis=1)), axis=1), axis=1)
        updated_text_features[c, :] /= np.linalg.norm(updated_text_features[c, :], keepdims=True) 

    softmax = torch.nn.Softmax(dim=1)
    probabilities = image_features @ updated_text_features.T
    probabilities = softmax(torch.tensor(probabilities / args.temperature))
    #print("Softmax output: ", probabilities[:2,:])
    unitary_potential = -np.log(probabilities)
    return unitary_potential.T   



'''
# !!! From StatA github https://github.com/MaxZanella/StatA/blob/main/solvers/StatA.py#L148 !!! 
def update_mu(adapter, query_features, z, beta, init_prototypes):

    mu = torch.einsum('ij,ik->jk', z, query_features) 
    mu /= torch.sum(z, dim=0).unsqueeze(-1)
    mu = mu.unsqueeze(1)
    mu /= mu.norm(dim=-1, keepdim=True)
    mu = beta.unsqueeze(-1).unsqueeze(-1) * mu + (1-beta).unsqueeze(-1).unsqueeze(-1) * init_prototypes
    mu /= mu.norm(dim=-1, keepdim=True)
    return mu
'''


def pairwise_potential_from_model_features(npz_path, variances, weight, n_components=10): 
    # Load the .npz file
    data = np.load(npz_path)

    # Access the required keys
    width = data['width']
    height = data['height'] 
    n = width * height
    features = data['features']

    # Pairwise potential matrix
    pairwise_potential = np.zeros((n,n), dtype=np.float16)
    for i in range(n):
        for j in range(n):
            f_i, f_j = features[i], features[j]
            diff = f_i @ f_j.T 
            if i == j: 
                diff = 0
            #diff = (f_i @ f_j.T).reshape((-1,1))

            #pairwise_potential[i, j] = np.exp(-0.5 * np.linalg.norm(diff)**2 / variances[0])
            pairwise_potential[i, j] = diff #np.exp(-0.5 * (1-diff) / variances[0])

    # Apply PCA to reduce the dimensions
    #pca = PCA(n_components=n_components)
    #reduced_pairwise_potential = pca.fit_transform(pairwise_potential)
    #print("reduced_pairwise_pot", reduced_pairwise_potential.shape)
    
    #pairwise_potential_flat = pairwise_potential.reshape((-1,1))
    return pairwise_potential

def pairwise_potential_from_model_features_and_position(npz_path, variances, weight): 
    # Load the .npz file
    data = np.load(npz_path)

    # Access the required keys
    width = data['width']
    height = data['height'] 
    n = width * height
    features = data['features']
    positions = data['positions'].T
    print("positions", positions.shape)

    # Pairwise potential matrix
    pairwise_potential = np.zeros((n,n), dtype=np.float16)
    for i in range(n):
        for j in range(n):
            f_i, f_j = features[i], features[j]
            p_i, p_j = positions[i], positions[j]
            diff_f = (f_i - f_j).reshape((-1,1))
            diff_p = (p_i - p_j).reshape((-1,1))

            #pairwise_potential[i, j] = np.exp(-0.5 * np.linalg.norm(diff_f)**2 / variances[0])
            pairwise_potential[i, j] = (
                + weight[0] * np.exp(-0.5 * (np.linalg.norm(diff_p) / variances[0] + np.linalg.norm(diff_f) / variances[1])) 
                + weight[1] * np.exp(-0.5 * (np.linalg.norm(diff_p) / variances[0]))
            )
    # Apply PCA to reduce the dimensions
    #pca = PCA(n_components=n_components)
    #reduced_pairwise_potential = pca.fit_transform(pairwise_potential)
    
    #pairwise_potential_flat = pairwise_potential.reshape((-1,1))
    return torch.tensor(pairwise_potential, dtype=torch.float16)

def pairwise_potential_from_img_features(npz_path, variances, weight,  device='cpu'):
    # Load the .npz file
    data = np.load(npz_path)

    # Access the required keys
    intensity = torch.tensor(data['images'], dtype=torch.float16, device=device)

    # Compute pairwise squared differences using broadcasting
    print('--- Compute diff_i ---')
    intensity = intensity.view(-1,3) 
    diff_i = torch.cdist(intensity, intensity, p=2) ** 2
    
    # Compute pairwise potential
    print("--- Compute pairwise potentials ---")
    pairwise_potential = (
        + weight[0] * torch.exp(-0.5 * (diff_i / variances[0])) 
    )

    return pairwise_potential


def pairwise_potential_from_img_and_position(npz_path, variances, weight, device='cpu'):
    # Load the .npz file
    data = np.load(npz_path)
    
    # Access the required keys
    width = int(data['width'])
    height = int(data['height'])
    intensity = torch.tensor(data['images'], dtype=torch.float16, device=device)
    
    # Create position tensor
    x_coords = torch.arange(height, dtype=torch.float16, device=device).repeat(width)
    y_coords = torch.arange(width, dtype=torch.float16, device=device).repeat_interleave(height)
    positions = torch.stack([x_coords, y_coords], dim=1)
    positions /= 255.
    
    # Flatten intensity to match positions
    intensity = intensity.view(-1, 3)  # Flatten and add a dimension for broadcasting
    intensity /= 255.

    # Compute pairwise squared differences using broadcasting
    print('--- Compute diff_p ---')
    diff_p = torch.cdist(positions, positions, p=2) ** 2
    #diff_p = compute_diff(positions)
    print('--- Compute diff_i ---')
    diff_i = torch.cdist(intensity, intensity, p=2) ** 2
    #diff_i = compute_diff(intensity)
    
    # Compute pairwise potential
    print("--- Compute pairwise potentials ---")
    #pp1 = weight[0] * torch.exp(-0.5 * (diff_p / variances[0] + diff_i / variances[1]))
    #pp2 = weight[1] * torch.exp(-0.5 * (diff_p / variances[0]))
    #pairwise_potential = pp1 + pp2
    pairwise_potential = (
        + weight[0] * torch.exp(-0.5 * (diff_p / variances[0] + diff_i / variances[1])) 
        + weight[1] * torch.exp(-0.5 * (diff_p / variances[0]))
    )
    #print("Contains NaN:", torch.isnan(pairwise_potential).any())
    #print("Contains Inf:", torch.isinf(pairwise_potential).any())

    return pairwise_potential


def pairwise_potential_from_img(npz_path, variances, weight, device='cpu'):
    # Load the .npz file
    data = np.load(npz_path)
    
    # Access the required keys
    intensity = torch.tensor(data['images'], dtype=torch.float16, device=device)
    
    # Flatten intensity to match positions
    intensity = intensity.view(-1, 3)  # Flatten and add a dimension for broadcasting
    
    # Compute pairwise squared differences using broadcasting
    print('--- Compute diff_i ---')
    diff_i = torch.cdist(intensity, intensity, p=2) ** 2
    
    # Compute pairwise potential
    print("print('--- Compute pairwise potentials ---')")
    pairwise_potential = weight[0] * torch.exp(-0.5 * (diff_i / variances[0]))
    
    return pairwise_potential


def pairwise_potential_from_hsv_image_and_position(npz_path, variances, weight, device='cpu'):
    import cv2
    # Load the .npz file
    data = np.load(npz_path)
    
    # Access the required keys
    width = int(data['width'])
    height = int(data['height'])
    n = width * height
    hsv_image = cv2.cvtColor(data['images'], cv2.COLOR_BGR2HSV)
    intensity = torch.tensor(hsv_image, dtype=torch.float16, device=device)
    
    # Create position tensor
    x_coords = torch.arange(height, device=device).repeat(width)
    y_coords = torch.arange(width, device=device).repeat_interleave(height)
    positions = torch.stack([x_coords, y_coords], dim=1)
    
    # Flatten intensity to match positions
    intensity = intensity.view(-1, 3)  # Flatten and add a dimension for broadcasting
    
    # Compute pairwise squared differences using broadcasting
    print('--- Compute diff_p ---')
    diff_p = torch.cdist(positions.float(), positions.float(), p=2) ** 2
    print('--- Compute diff_i ---')
    diff_i = torch.cdist(intensity, intensity, p=2) ** 2
    
    # Compute pairwise potential
    print("print('--- Compute pp1 ---')")
    pp1 = weight[0] * torch.exp(-0.5 * (diff_p / variances[0] + diff_i / variances[1]))
    print("print('--- Compute pp2 ---')")
    pp2 = weight[1] * torch.exp(-0.5 * (diff_p / variances[0]))
    pairwise_potential = pp1 + pp2
    
    return pairwise_potential


def pairwise_potential_from_binary_mask(npz_path, variances, weight, device='cpu'):
    import cv2

    def create_binary_mask_for_purple_elements(image):
        # Step 2: Convert to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Step 3: Define the range for purple in HSV
        # Adjust these values based on your image
        lower_purple = np.array([120, 50, 50])  # Lower bound (H, S, V)
        upper_purple = np.array([160, 255, 255])  # Upper bound (H, S, V)

        # Step 4: Create a mask for purple
        purple_mask = cv2.inRange(hsv_image, lower_purple, upper_purple)

        # Step 5: Invert the mask to turn purple regions black and the rest white
        binary_mask = cv2.bitwise_not(purple_mask)

        return binary_mask

    # Load the .npz file
    data = np.load(npz_path)
    
    # Access the required keys
    width = int(data['width'])
    height = int(data['height'])
    n = width * height
    hsv_image = create_binary_mask_for_purple_elements(data['images'])
    intensity = torch.tensor(hsv_image, dtype=torch.float16, device=device)
    
    # Create position tensor
    x_coords = torch.arange(height, device=device).repeat(width)
    y_coords = torch.arange(width, device=device).repeat_interleave(height)
    positions = torch.stack([x_coords, y_coords], dim=1)
    
    # Flatten intensity to match positions
    intensity = intensity.view(-1, 3)  # Flatten and add a dimension for broadcasting
    
    # Compute pairwise squared differences using broadcasting
    print('--- Compute diff_p ---')
    diff_p = torch.cdist(positions.float(), positions.float(), p=2) ** 2
    print('--- Compute diff_i ---')
    diff_i = torch.cdist(intensity, intensity, p=2) ** 2
    
    # Compute pairwise potential
    print("print('--- Compute pairwise_potential ---')")
    pairwise_potential = weight[0] * torch.exp(-0.5 * (diff_p / variances[0] + diff_i / variances[1]))
    
    return pairwise_potential


def pairwise_potential_from_hsl_image_and_position(npz_path, variances, weight, device='cpu'):
    import cv2
    # Load the .npz file
    data = np.load(npz_path)
    
    # Access the required keys
    width = int(data['width'])
    height = int(data['height'])
    n = width * height
    rgb_image = data['images']
    rgb_image_normalized = rgb_image.astype(np.float32) / 255.0
    hls_image = cv2.cvtColor(rgb_image_normalized, cv2.COLOR_RGB2HLS)
    intensity = torch.tensor(hls_image, dtype=torch.float16, device=device)
    
    # Create position tensor
    x_coords = torch.arange(height, device=device).repeat(width)
    y_coords = torch.arange(width, device=device).repeat_interleave(height)
    positions = torch.stack([x_coords, y_coords], dim=1)
    
    # Flatten intensity to match positions
    intensity = intensity.view(-1, 3)  # Flatten and add a dimension for broadcasting
    
    # Compute pairwise squared differences using broadcasting
    print('--- Compute diff_p ---')
    diff_p = torch.cdist(positions.float(), positions.float(), p=2) ** 2
    print('--- Compute diff_i ---')
    diff_i = torch.cdist(intensity, intensity, p=2) ** 2
    
    # Compute pairwise potential
    print("print('--- Compute pp1 ---')")
    pp1 = weight[0] * torch.exp(-0.5 * (diff_p / variances[0] + diff_i / variances[1]))
    print("print('--- Compute pp2 ---')")
    pp2 = weight[1] * torch.exp(-0.5 * (diff_p / variances[0]))
    pairwise_potential = pp1 + pp2
    
    return pairwise_potential


def pairwise_potential_from_cielab_image_and_position(npz_path, variances, weight, device='cpu'):
    import cv2
    # Load the .npz file
    data = np.load(npz_path)
    
    # Access the required keys
    width = int(data['width'])
    height = int(data['height'])
    n = width * height
    rgb_image = data['images']
    rgb_image_normalized = rgb_image.astype(np.float32) / 255.0
    lab_image = cv2.cvtColor(rgb_image_normalized, cv2.COLOR_RGB2LAB)
    intensity = torch.tensor(lab_image, dtype=torch.float16, device=device)
    
    # Create position tensor
    x_coords = torch.arange(height, device=device).repeat(width)
    y_coords = torch.arange(width, device=device).repeat_interleave(height)
    positions = torch.stack([x_coords, y_coords], dim=1)
    
    # Flatten intensity to match positions
    intensity = intensity.view(-1, 3)  # Flatten and add a dimension for broadcasting
    
    # Compute pairwise squared differences using broadcasting
    print('--- Compute diff_p ---')
    diff_p = torch.cdist(positions.float(), positions.float(), p=2) ** 2
    print('--- Compute diff_i ---')
    diff_i = torch.cdist(intensity, intensity, p=2) ** 2
    
    # Compute pairwise potential
    print("print('--- Compute pp1 ---')")
    pp1 = weight[0] * torch.exp(-0.5 * (diff_p / variances[0] + diff_i / variances[1]))
    print("print('--- Compute pp2 ---')")
    pp2 = weight[1] * torch.exp(-0.5 * (diff_p / variances[0]))
    pairwise_potential = pp1 + pp2
    
    return pairwise_potential


def pairwise_potential_from_edges(npz_path, variances, weight, device='cpu'):
    import cv2
    # Load the .npz file
    data = np.load(npz_path)
    
    # Access the required keys
    width = int(data['width'])
    height = int(data['height'])
    n = width * height
    rgb_image = data['images']
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image,200,250)
    (intensity) = torch.tensor(edges, dtype=torch.float16, device=device)
    
    # Create position tensor
    x_coords = torch.arange(height, device=device).repeat(width)
    y_coords = torch.arange(width, device=device).repeat_interleave(height)
    positions = torch.stack([x_coords, y_coords], dim=1)
    
    # Flatten intensity to match positions
    # intensity = intensity.view(-1, 3)  # Flatten and add a dimension for broadcasting
    
    # Compute pairwise squared differences using broadcasting
    print('--- Compute diff_p ---')
    diff_p = torch.cdist(positions.float(), positions.float(), p=2) ** 2
    print('--- Compute diff_i ---')
    #diff_i = torch.cdist(intensity, intensity, p=2) ** 2
    diff_i = intensity
    
    # Compute pairwise potential
    print("print('--- Compute pp1 ---')")
    pairwise_potential = weight[0] * torch.exp(-0.5 * (diff_p / variances[0]) - weight[1] * diff_i)
    
    return pairwise_potential

def pairwise_potential_from_img_features_and_cossim(npz_path, variances, weight,  device='cpu'):
    # Load the .npz file
    data = np.load(npz_path)

    # Access the required keys
    intensity = torch.tensor(data['images'], dtype=torch.float16, device=device)
    width = data['width']
    height = data['height'] 
    n_labels = data['n_labels']
    n = width * height
    probabilities = torch.tensor(data['probabilities'], dtype=torch.float16, device=device).resize(n_labels,n) #2 = n_class
    print("probabilities", probabilities.size())
    print("intensity", intensity.size())

    # Compute pairwise squared differences using broadcasting
    print('--- Compute diff_i ---')
    intensity = intensity.view(-1, 3)  # Flatten intensity to match positions
    intensity /= 255.
    diff_i = torch.cdist(intensity, intensity, p=2) ** 2

    # Compute the cosine similarities of probabilities vector 
    print('--- Compute cosine sim ---')
    cos_sim = (probabilities.T @ probabilities)

    # Compute pairwise potential
    print("--- Compute pairwise potentials ---")
    print("cos_sim", cos_sim)
    print("diff_i", diff_i)
    pairwise_potential = (
        + weight[0] * torch.exp(-0.5 * (diff_i / variances[0])) 
        + weight[1] * torch.exp(-0.5 * (cos_sim / variances[1])) 
    )
    print("pairwise_potential", pairwise_potential)
    return pairwise_potential

def pairwise_potential_from_img_patches(intensity, variances, weight):
    # Flatten intensity to match positions
    intensity = intensity.view(-1, 3)  # Flatten and add a dimension for broadcasting
    
    # Compute pairwise squared differences using broadcasting
    print('--- Compute diff_i ---')
    diff_i = torch.cdist(intensity, intensity, p=2) ** 2
    
    # Compute pairwise potential
    print("print('--- Compute pairwise potentials ---')")
    pairwise_potential = weight[0] * torch.exp(-0.5 * (diff_i / variances[0]))
    
    return pairwise_potential