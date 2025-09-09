import numpy as np
import torch

device = 'cuda'


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