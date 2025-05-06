import numpy as np

def variance_from_image(args, npz_path):
    # Load the .npz file
    data = np.load(npz_path)

    # Access the required keys
    image = data['images']

    variances = np.var(image, axis=2)
    return variances 


def variance_from_features(args, npz_path):
    # Load the .npz file
    data = np.load(npz_path)

    # Access the required keys
    features = data['features']

    variances = np.var(features, axis=0)
    return variances 


def custom_variance(args, npz_path):
    variances = args.custom_var
    if variances is None:
        raise RuntimeError("Missing argument for --custom_var.")
    return variances