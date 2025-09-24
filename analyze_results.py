import argparse
import json
import numpy as np
import os 
import pandas as pd


def analyze_results_iteration(args):
    # Get results json files 
    results_dir = os.path.join(args.root_dir, 'results', args.dataset, args.model) # Define the directory containing the results
    results_paths = [f for f in os.listdir(results_dir) if f.endswith(".json")] # Get all result file paths that end with .npz
    
    # Initialize an empty DataFrame to store results
    results = pd.DataFrame(columns=[
                                    "N_PROP", "N_UP", "ann_iterations", 
                                    "n_iterations", "weight", "var", "pair_pot", "compat", "temperature", 
                                    "sparse_method", "n_affinity", "n_annotations", "seed", "annotation_method", "linear",
                                    "initial_accuracy", "initial_annotation_accuracy", "annotation_accuracies", "map_accuracies", "accuracies_gain",
                                    "annotation_balanced_accuracies", "map_balanced_accuracies", "accuracies_balanced_gain",
                                    "initial_balanced_accuracy", "initial_annotated_balanced_accuracy"
                                    ])

    # Process each result file
    for results_path in results_paths:
       
        if 'random' in results_path or 'entropy' in results_path or 'error' in results_path:

            # Load the .npz file and extract the accuracy
            with open(os.path.join(results_dir, results_path), 'r') as file:
                data = json.load(file)

            # Append the data to the DataFrame
            new_results = pd.DataFrame([{
                #"npz_name": str(npz_name),
                "N_PROP": str(data["N_PROP"]),
                "N_UP": int(data["N_UP"]),
                "ann_iterations": int(data["ann_iterations"]),
                "n_iterations": int(data["n_iterations"]),
                "weight": str(data["weight"]),
                "var": str(data["var"]),
                "pair_pot": str(data["pp"]),
                "compat": str(data["compat"]),
                "temperature": str(data["temperature"]),
                "sparse_method": str(data["sparse_method"]),
                "n_affinity": str(data["n_affinity"]),
                "n_annotations": int(data["n_annotations"]),
                "seed": int(data["seed"]),
                "annotation_method": str(data["annotation_method"]),
                "linear": str(data["linear"]), 
                "initial_accuracy": data["initial_accuracy"],
                "initial_annotation_accuracy": data["initial_annotation_accuracy"],
                "annotation_accuracies": data["annotation_accuracies"],
                "map_accuracies": data["map_accuracies"],
                "accuracies_gain": data["accuracies_gain"],
                "annotation_balanced_accuracies": data["annotation_balanced_accuracies"], 
                "map_balanced_accuracies": data["map_balanced_accuracies"], 
                "accuracies_balanced_gain": data["accuracies_balanced_gain"], 
                "initial_balanced_accuracy": data["initial_balanced_accuracy"], 
                "initial_annotated_balanced_accuracy": data["initial_annotated_balanced_accuracy"],
                #"final_label": data["final_label"]

                "map_accuracies_std": data["map_accuracies"],
                "map_balanced_accuracies_std": data["map_balanced_accuracies"],
            }])
            results = pd.concat([results, new_results])

    def mean_of_lists(series):
        # Stack the lists into a 2D array and take the mean across rows
        return np.mean(np.stack(series), axis=0).tolist()

    def std_of_lists(series):
        # Stack the lists into a 2D array and take the mean across rows
        return np.std(np.stack(series), axis=0).tolist()

    mean_results = (
        results
        .groupby(["N_PROP", "N_UP", "ann_iterations", "n_iterations", "weight", "var", "pair_pot", "compat", "temperature",
                "sparse_method", "n_affinity", "n_annotations", "annotation_method", "linear"], as_index=False)
        .agg({
            "initial_accuracy": "mean",
            "initial_annotation_accuracy": "mean",
            "annotation_accuracies": mean_of_lists,
            "map_accuracies": mean_of_lists,
            "accuracies_gain": mean_of_lists,
            "annotation_balanced_accuracies": mean_of_lists,
            "map_balanced_accuracies": mean_of_lists,
            "accuracies_balanced_gain": mean_of_lists,
            "initial_balanced_accuracy": "mean",
            "initial_annotated_balanced_accuracy": mean_of_lists,
            "map_accuracies_std": std_of_lists,
            "map_balanced_accuracies_std": std_of_lists
        })
    )

    # Save the DataFrame to a CSV file
    csv_name = f'iteration_{args.dataset}_{args.model}_results_summary.csv'
    results_csv_path = os.path.join(results_dir, csv_name)
    results.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

    csv_name = f'iteration_{args.dataset}_{args.model}_results_mean_summary.csv'
    results_csv_path = os.path.join(results_dir, csv_name)
    mean_results.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}") 


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--dataset', type=str, default='sicap_mil', choices=['sicap_mil', 'monuseg', 'panuke', 'skincancer2', 'lc_lung', 'lc_colon', 'nct', 'bracs', 'bach', 'esca', 'tcga_uniform', 'bach_wsi'])
    parser.add_argument('--model', type=str, default='clip', choices=['clip', 'quilt', 'conch', 'plip', 'nnunet', 'plipanduni2', 'conchanduni2', 'conchandgigapath', 'conchandoptimus1'])
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parser()

    analyze_results_iteration(args)
