import argparse
import ast
import json
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import numpy as np
import os 
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops


def reconstruct(positions, labels):
    # Ensure positions are (N, 2)
    if positions.shape[0] == 2 and positions.shape[1] != 2:
        positions = positions.T  # Convert from (2, N) to (N, 2)

    positions = positions.astype(int)
    labels = np.array(labels)

    # Normalize label IDs to integers if needed 
    unique_labels, label_indices = np.unique(labels, return_inverse=True)
    label_indices = labels

    # Get the pixel positions in units (e.g., pixels or tile origin)
    x_coords = positions[:positions.shape[0]//2]
    y_coords = positions[positions.shape[0]//2:]


    # Optional: compute step (spacing) to normalize to a grid
    dy = np.gcd.reduce(np.diff(np.unique(y_coords)))
    dx = np.gcd.reduce(np.diff(np.unique(x_coords)))

    # Normalize positions to grid indices
    grid_y = (y_coords - y_coords.min()) // dy
    grid_x = (x_coords - x_coords.min()) // dx

    # Build label map
    height = grid_y.max() + 1
    width = grid_x.max() + 1
    label_map = np.full((height, width), fill_value=4, dtype=int)

    for y, x, label_idx in zip(grid_y, grid_x, label_indices):
        label_map[y, x] = label_idx

    unique_labels = [0, 1, 2, 3, 4]
    
    return unique_labels, label_map


def create_df_from_json(args, datasets, models, shots):
    results = pd.DataFrame(columns=[
        "model", "dataset", "shot", "exp",
        "zs_accuracy", "zs_balanced_accuracy", 
        "annotation_accuracy", "annotation_balanced_accuracy", 
        "map_accuracy", "map_balanced_accuracy", "std_map_accuracy"
        ])
    for dataset in datasets:
        if dataset in ['bracs']:
            n = 50
            i = -1
            w = '0.1,0.01'
        elif dataset in [ 'nct', 'sicap_mil', 'bach']:
            n = 50
            i = -1 
            w = '0.1,0.01'
        elif dataset in ['esca', 'bach_wsi']:
            n = 50
            i = -1 
            w = '0.1,0.01'

        else: 
            raise RuntimeError(f"Not implemented for dataset {dataset}")

        for model in models:
            # Define the directory containing the results
            if dataset in ['bach', 'tcga_uniform', 'esca', 'skincancer2', 'bach_wsi']:
                results_dir = os.path.join(args.root_dir.replace('CECI/home', 'globalscratch'), 'results', dataset, model)
            else:
                results_dir = os.path.join(args.root_dir, 'results', dataset, model)
            
            # Load dataframe
            csv_name = f'iteration_{dataset}_{model}_results_mean_summary.csv'
            df = pd.read_csv(os.path.join(results_dir, csv_name))
            df = df.replace(np.nan, 'None')

            for shot in shots:
                print(f"-------------- Dataset {dataset}, model {model}, shot {shot} --------------")
                
                df_rows = df[
                    (df["N_PROP"] == n) & 
                    (df["sparse_method"] == str(args.sparse_method)) &
                    (df["pair_pot"] == str(args.pair_pot)) &
                    (df["compat"] == str(args.compat)) & 
                    (df["annotation_method"] == 'error') &
                    (df["n_iterations"] == int(args.it)) &
                    (df["n_affinity"] == "[0, 16]") &
                    (df["weight"] == w) & 
                    (df["ann_iterations"] == int(args.ann_iterations)) &
                    (df["N_UP"] == -1) & # No prototype
                    (df["linear"] == False) & # No linear 
                    (df["n_annotations"] == int(shot))
                ] 
                try: 
                    new_results = pd.DataFrame([{
                        "model": model,
                        "dataset": dataset,
                        "shot": int(shot), 
                        "exp": "fs_error",

                        "zs_accuracy": float(df_rows["initial_accuracy"].iloc[0]),
                        "zs_balanced_accuracy": float(df_rows["initial_balanced_accuracy"].iloc[0]),
            
                        "annotation_accuracy": float(ast.literal_eval(df_rows["annotation_accuracies"].iloc[0])[0]), 
                        "annotation_balanced_accuracy": float(ast.literal_eval(df_rows["annotation_balanced_accuracies"].iloc[0])[0]),
            
                        "map_accuracy": float(ast.literal_eval(df_rows["map_accuracies"].iloc[0])[i]),
                        "map_balanced_accuracy": float(ast.literal_eval(df_rows["map_balanced_accuracies"].iloc[0])[i]),
                        "std_map_accuracy": float(ast.literal_eval(df_rows["map_accuracies_std"].iloc[0])[i]),                    
                    }])
                    results = pd.concat([results, new_results])
                except:
                    pass
                
                df_rows = df[
                    (df["N_PROP"] == n) & 
                    (df["sparse_method"] == str(args.sparse_method)) &
                    (df["pair_pot"] == str(args.pair_pot)) &
                    (df["compat"] == str(args.compat)) & 
                    (df["annotation_method"] == 'random') &
                    (df["n_iterations"] == int(args.it)) &
                    (df["n_affinity"] == "[0, 16]") &
                    (df["weight"] == w) & 
                    (df["ann_iterations"] == int(args.ann_iterations)) &
                    (df["N_UP"] == -1) & # No prototype
                    (df["linear"] == False) & # No linear 
                    (df["n_annotations"] == int(shot))
                ] 
                try: 
                    new_results = pd.DataFrame([{
                        "model": model,
                        "dataset": dataset,
                        "shot": int(shot), 
                        "exp": "fs_random",

                        "zs_accuracy": float(df_rows["initial_accuracy"].iloc[0]),
                        "zs_balanced_accuracy": float(df_rows["initial_balanced_accuracy"].iloc[0]),
            
                        "annotation_accuracy": float(ast.literal_eval(df_rows["annotation_accuracies"].iloc[0])[0]), 
                        "annotation_balanced_accuracy": float(ast.literal_eval(df_rows["annotation_balanced_accuracies"].iloc[0])[0]),
            
                        "map_accuracy": float(ast.literal_eval(df_rows["map_accuracies"].iloc[0])[i]),
                        "map_balanced_accuracy": float(ast.literal_eval(df_rows["map_balanced_accuracies"].iloc[0])[i]),
                        "std_map_accuracy": float(ast.literal_eval(df_rows["map_accuracies_std"].iloc[0])[i]),                    
                    }])
                    results = pd.concat([results, new_results])
                except: 
                    pass
                
                if DIVERSEERROR:
                    df_rows = df[
                        (df["N_PROP"] == n) & 
                        (df["sparse_method"] == str(args.sparse_method)) &
                        (df["pair_pot"] == str(args.pair_pot)) &
                        (df["compat"] == str(args.compat)) & 
                        (df["annotation_method"] == 'diverseerror') &
                        (df["n_iterations"] == int(args.it)) &
                        (df["n_affinity"] == "[0, 16]") &
                        (df["weight"] == w) & 
                        (df["ann_iterations"] == int(args.ann_iterations)) &
                        (df["N_UP"] == -1) & # No prototype
                        (df["linear"] == False) & # No linear 
                        (df["n_annotations"] == int(shot))
                    ] 
                    try: 
                        new_results = pd.DataFrame([{
                            "model": model,
                            "dataset": dataset,
                            "shot": int(shot), 
                            "exp": "fs_diverseerror",

                            "zs_accuracy": float(df_rows["initial_accuracy"].iloc[0]),
                            "zs_balanced_accuracy": float(df_rows["initial_balanced_accuracy"].iloc[0]),
                
                            "annotation_accuracy": float(ast.literal_eval(df_rows["annotation_accuracies"].iloc[0])[0]), 
                            "annotation_balanced_accuracy": float(ast.literal_eval(df_rows["annotation_balanced_accuracies"].iloc[0])[0]),
                
                            "map_accuracy": float(ast.literal_eval(df_rows["map_accuracies"].iloc[0])[i]),
                            "map_balanced_accuracy": float(ast.literal_eval(df_rows["map_balanced_accuracies"].iloc[0])[i]),
                            "std_map_accuracy": float(ast.literal_eval(df_rows["map_accuracies_std"].iloc[0])[i]),                    
                        }])
                        results = pd.concat([results, new_results])
                    except: 
                        pass


                df_rows_hitl = df[
                    (df["N_PROP"] == n) & 
                    (df["sparse_method"] == str(args.sparse_method)) &
                    (df["pair_pot"] == str(args.pair_pot)) &
                    (df["compat"] == str(args.compat)) & 
                    (df["annotation_method"] == 'error') &
                    (df["n_iterations"] == int(args.it)) &
                    (df["n_affinity"] == "[0, 16]") &
                    (df["weight"] == w) & 
                    (df["N_UP"] == -1) & # No prototype
                    (df["linear"] == False) & # No linear 
                    (df["n_annotations"] == 5) &
                    (df["ann_iterations"] == 20) 
                ]
                try: 
                    if shot > 0: 
                        x = shot // 5 * 50 + i
                        new_results = pd.DataFrame([{
                            "model": model,
                            "dataset": dataset,
                            "shot": int(shot), 
                            "exp": "hitl_error",

                            "zs_accuracy": float(df_rows["initial_accuracy"].iloc[0]),
                            "zs_balanced_accuracy": float(df_rows["initial_balanced_accuracy"].iloc[0]),
                
                            "annotation_accuracy": float(ast.literal_eval(df_rows_hitl["annotation_accuracies"].iloc[0])[x]), 
                            "annotation_balanced_accuracy": float(ast.literal_eval(df_rows_hitl["annotation_balanced_accuracies"].iloc[0])[x]),
                
                            "map_accuracy": float(ast.literal_eval(df_rows_hitl["map_accuracies"].iloc[0])[x]),
                            "map_balanced_accuracy": float(ast.literal_eval(df_rows_hitl["map_balanced_accuracies"].iloc[0])[x]),
                            "std_map_accuracy": float(ast.literal_eval(df_rows_hitl["map_accuracies_std"].iloc[0])[x]),                    
                        }])
                    else: 
                        x = shot // 5 * 50 + i
                        new_results = pd.DataFrame([{
                            "model": model,
                            "dataset": dataset,
                            "shot": int(shot), 
                            "exp": "hitl_error",

                            "zs_accuracy": float(df_rows["initial_accuracy"].iloc[0]),
                            "zs_balanced_accuracy": float(df_rows["initial_balanced_accuracy"].iloc[0]),
                
                            "annotation_accuracy": float(ast.literal_eval(df_rows["annotation_accuracies"].iloc[0])[x]), 
                            "annotation_balanced_accuracy": float(ast.literal_eval(df_rows["annotation_balanced_accuracies"].iloc[0])[x]),
                
                            "map_accuracy": float(ast.literal_eval(df_rows["map_accuracies"].iloc[0])[x]),
                            "map_balanced_accuracy": float(ast.literal_eval(df_rows["map_balanced_accuracies"].iloc[0])[x]),
                            "std_map_accuracy": float(ast.literal_eval(df_rows["map_accuracies_std"].iloc[0])[x]),                    
                        }])
                    results = pd.concat([results, new_results])
                except: 
                    pass
                
                if DIVERSEERROR:
                    df_rows_hitl = df[
                        (df["N_PROP"] == n) & 
                        (df["sparse_method"] == str(args.sparse_method)) &
                        (df["pair_pot"] == str(args.pair_pot)) &
                        (df["compat"] == str(args.compat)) & 
                        (df["annotation_method"] == 'diverseerror') &
                        (df["n_iterations"] == int(args.it)) &
                        (df["n_affinity"] == "[0, 16]") &
                        (df["weight"] == w) & 
                        (df["N_UP"] == -1) & # No prototype
                        (df["linear"] == False) & # No linear 
                        (df["n_annotations"] == 5) &
                        (df["ann_iterations"] == 20) 
                    ]
                    try: 
                        if shot > 0: 
                            x = shot // 5 * 50 + i
                            new_results = pd.DataFrame([{
                                "model": model,
                                "dataset": dataset,
                                "shot": int(shot), 
                                "exp": "hitl_diverseerror",

                                "zs_accuracy": float(df_rows["initial_accuracy"].iloc[0]),
                                "zs_balanced_accuracy": float(df_rows["initial_balanced_accuracy"].iloc[0]),
                    
                                "annotation_accuracy": float(ast.literal_eval(df_rows_hitl["annotation_accuracies"].iloc[0])[x]), 
                                "annotation_balanced_accuracy": float(ast.literal_eval(df_rows_hitl["annotation_balanced_accuracies"].iloc[0])[x]),
                    
                                "map_accuracy": float(ast.literal_eval(df_rows_hitl["map_accuracies"].iloc[0])[x]),
                                "map_balanced_accuracy": float(ast.literal_eval(df_rows_hitl["map_balanced_accuracies"].iloc[0])[x]),
                                "std_map_accuracy": float(ast.literal_eval(df_rows_hitl["map_accuracies_std"].iloc[0])[x]),                    
                            }])
                        else: 
                            x = shot // 5 * 50 + i
                            new_results = pd.DataFrame([{
                                "model": model,
                                "dataset": dataset,
                                "shot": int(shot), 
                                "exp": "hitl_diverseerror",

                                "zs_accuracy": float(df_rows["initial_accuracy"].iloc[0]),
                                "zs_balanced_accuracy": float(df_rows["initial_balanced_accuracy"].iloc[0]),
                    
                                "annotation_accuracy": float(ast.literal_eval(df_rows["annotation_accuracies"].iloc[0])[x]), 
                                "annotation_balanced_accuracy": float(ast.literal_eval(df_rows["annotation_balanced_accuracies"].iloc[0])[x]),
                    
                                "map_accuracy": float(ast.literal_eval(df_rows["map_accuracies"].iloc[0])[x]),
                                "map_balanced_accuracy": float(ast.literal_eval(df_rows["map_balanced_accuracies"].iloc[0])[x]),
                                "std_map_accuracy": float(ast.literal_eval(df_rows["map_accuracies_std"].iloc[0])[x]),                    
                            }])
                        results = pd.concat([results, new_results])
                    except: 
                        pass
                

                if shot == 0: 
                    if dataset not in ['esca', 'bach_wsi']:
                        new_results = pd.DataFrame([{
                            "model": model,
                            "dataset": dataset,
                            "shot": int(shot), 
                            "exp": "baseline_error",

                            "zs_accuracy": float(df_rows["initial_accuracy"].iloc[0]),
                            "zs_balanced_accuracy": float(df_rows["initial_balanced_accuracy"].iloc[0]),
                
                            "annotation_accuracy": -1, 
                            "annotation_balanced_accuracy": -1,
                
                            "map_accuracy": -1,
                            "map_balanced_accuracy": -1,
                            "std_map_accuracy": -1,                    
                        }])
                    else:
                        new_results = pd.DataFrame([{
                            "model": model,
                            "dataset": dataset,
                            "shot": int(shot), 
                            "exp": "baseline_error",

                            "zs_accuracy": -1,
                            "zs_balanced_accuracy": -1,
                
                            "annotation_accuracy": -1, 
                            "annotation_balanced_accuracy": -1,
                
                            "map_accuracy": -1,
                            "map_balanced_accuracy": -1,
                            "std_map_accuracy": -1,                    
                        }])
                    results = pd.concat([results, new_results])
                if shot == 0: 
                    if dataset not in ['esca', 'bach_wsi']:
                        new_results = pd.DataFrame([{
                            "model": model,
                            "dataset": dataset,
                            "shot": int(shot), 
                            "exp": "baseline_random",

                            "zs_accuracy": float(df_rows["initial_accuracy"].iloc[0]),
                            "zs_balanced_accuracy": float(df_rows["initial_balanced_accuracy"].iloc[0]),
                
                            "annotation_accuracy": -1, 
                            "annotation_balanced_accuracy": -1,
                
                            "map_accuracy": -1,
                            "map_balanced_accuracy": -1,
                            "std_map_accuracy": -1,                    
                        }])
                    else:
                        new_results = pd.DataFrame([{
                            "model": model,
                            "dataset": dataset,
                            "shot": int(shot), 
                            "exp": "baseline_random",

                            "zs_accuracy": -1,
                            "zs_balanced_accuracy": -1,
                
                            "annotation_accuracy": -1, 
                            "annotation_balanced_accuracy": -1,
                
                            "map_accuracy": -1,
                            "map_balanced_accuracy": -1,
                            "std_map_accuracy": -1,                    
                        }])
                    results = pd.concat([results, new_results])
                if DIVERSEERROR:
                    if shot == 0: 
                        if dataset not in ['esca', 'bach_wsi']:
                            new_results = pd.DataFrame([{
                                "model": model,
                                "dataset": dataset,
                                "shot": int(shot), 
                                "exp": "baseline_error",

                                "zs_accuracy": float(df_rows["initial_accuracy"].iloc[0]),
                                "zs_balanced_accuracy": float(df_rows["initial_balanced_accuracy"].iloc[0]),
                    
                                "annotation_accuracy": -1, 
                                "annotation_balanced_accuracy": -1,
                    
                                "map_accuracy": -1,
                                "map_balanced_accuracy": -1,
                                "std_map_accuracy": -1,                    
                            }])
                        else:
                            new_results = pd.DataFrame([{
                                "model": model,
                                "dataset": dataset,
                                "shot": int(shot), 
                                "exp": "baseline_error",

                                "zs_accuracy": -1,
                                "zs_balanced_accuracy": -1,
                    
                                "annotation_accuracy": -1, 
                                "annotation_balanced_accuracy": -1,
                    
                                "map_accuracy": -1,
                                "map_balanced_accuracy": -1,
                                "std_map_accuracy": -1,                    
                            }])
                        results = pd.concat([results, new_results])
                df_rows_baseline = df[
                    (df["N_PROP"] == 1) & 
                    (df["sparse_method"] == str(args.sparse_method)) & #args.sparse_method
                    (df["pair_pot"] == 'minusmodelfeatures,minusmodelfeaturesann') & #minusmodelfeatures
                    (df["compat"] == str(args.compat)) & 
                    (df["annotation_method"] == 'error') &
                    (df["n_iterations"] == int(args.it)) &
                    (df["n_affinity"] == "[0, 16]") &
                    (df["weight"] == w) & 
                    (df["N_UP"] == -100) & # Get baseline
                    (df["linear"] == False) & # No linear 
                    (df["n_annotations"] == int(shot)) &
                    (df["ann_iterations"] == 1) 
                ]
                try: 
                    new_results = pd.DataFrame([{
                        "model": model,
                        "dataset": dataset,
                        "shot": int(shot), 
                        "exp": "baseline_error",

                        "zs_accuracy": float(df_rows_baseline["initial_accuracy"].iloc[0]),
                        "zs_balanced_accuracy": float(df_rows_baseline["initial_balanced_accuracy"].iloc[0]),
            
                        "annotation_accuracy": float(ast.literal_eval(df_rows_baseline["annotation_accuracies"].iloc[0])[0]), 
                        "annotation_balanced_accuracy": float(ast.literal_eval(df_rows_baseline["annotation_balanced_accuracies"].iloc[0])[0]),
            
                        "map_accuracy": float(ast.literal_eval(df_rows_baseline["map_accuracies"].iloc[0])[0]),
                        "map_balanced_accuracy": float(ast.literal_eval(df_rows_baseline["map_balanced_accuracies"].iloc[0])[0]),
                        "std_map_accuracy": float(ast.literal_eval(df_rows_baseline["map_accuracies_std"].iloc[0])[0]),                    
                    }])
                    results = pd.concat([results, new_results])

                except:
                    pass
                
                df_rows_baseline = df[
                    (df["N_PROP"] == 1) & 
                    (df["sparse_method"] == str(args.sparse_method)) & #args.sparse_method
                    (df["pair_pot"] == 'minusmodelfeatures,minusmodelfeaturesann') & #minusmodelfeatures
                    (df["compat"] == str(args.compat)) & 
                    (df["annotation_method"] == 'random') &
                    (df["n_iterations"] == int(args.it)) &
                    (df["n_affinity"] == "[0, 16]") &
                    (df["weight"] == w) & 
                    (df["N_UP"] == -100) & # Get baseline
                    (df["linear"] == False) & # No linear 
                    (df["n_annotations"] == int(shot)) &
                    (df["ann_iterations"] == 1) 
                ]
                try: 
                    new_results = pd.DataFrame([{
                        "model": model,
                        "dataset": dataset,
                        "shot": int(shot), 
                        "exp": "baseline_random",

                        "zs_accuracy": float(df_rows_baseline["initial_accuracy"].iloc[0]),
                        "zs_balanced_accuracy": float(df_rows_baseline["initial_balanced_accuracy"].iloc[0]),
            
                        "annotation_accuracy": float(ast.literal_eval(df_rows_baseline["annotation_accuracies"].iloc[0])[0]), 
                        "annotation_balanced_accuracy": float(ast.literal_eval(df_rows_baseline["annotation_balanced_accuracies"].iloc[0])[0]),
            
                        "map_accuracy": float(ast.literal_eval(df_rows_baseline["map_accuracies"].iloc[0])[0]),
                        "map_balanced_accuracy": float(ast.literal_eval(df_rows_baseline["map_balanced_accuracies"].iloc[0])[0]),
                        "std_map_accuracy": float(ast.literal_eval(df_rows_baseline["map_accuracies_std"].iloc[0])[0]),                    
                    }])
                    results = pd.concat([results, new_results])

                except:
                    pass
                
                if DIVERSEERROR:
                    df_rows_baseline = df[
                        (df["N_PROP"] == 1) & 
                        (df["sparse_method"] == str(args.sparse_method)) & #args.sparse_method
                        (df["pair_pot"] == 'minusmodelfeatures,minusmodelfeaturesann') & #minusmodelfeatures
                        (df["compat"] == str(args.compat)) & 
                        (df["annotation_method"] == 'diverseerror') &
                        (df["n_iterations"] == int(args.it)) &
                        (df["n_affinity"] == "[0, 16]") &
                        (df["weight"] == w) & 
                        (df["N_UP"] == -100) & # Get baseline
                        (df["linear"] == False) & # No linear 
                        (df["n_annotations"] == int(shot)) &
                        (df["ann_iterations"] == 1) 
                    ]
                    try: 
                        new_results = pd.DataFrame([{
                            "model": model,
                            "dataset": dataset,
                            "shot": int(shot), 
                            "exp": "baseline_diverseerror",

                            "zs_accuracy": float(df_rows_baseline["initial_accuracy"].iloc[0]),
                            "zs_balanced_accuracy": float(df_rows_baseline["initial_balanced_accuracy"].iloc[0]),
                
                            "annotation_accuracy": float(ast.literal_eval(df_rows_baseline["annotation_accuracies"].iloc[0])[0]), 
                            "annotation_balanced_accuracy": float(ast.literal_eval(df_rows_baseline["annotation_balanced_accuracies"].iloc[0])[0]),
                
                            "map_accuracy": float(ast.literal_eval(df_rows_baseline["map_accuracies"].iloc[0])[0]),
                            "map_balanced_accuracy": float(ast.literal_eval(df_rows_baseline["map_balanced_accuracies"].iloc[0])[0]),
                            "std_map_accuracy": float(ast.literal_eval(df_rows_baseline["map_accuracies_std"].iloc[0])[0]),                    
                        }])
                        results = pd.concat([results, new_results])

                    except:
                        pass
                    
    # Average along datasets
    accuracy_cols = [
        "zs_accuracy", "zs_balanced_accuracy",
        "annotation_accuracy", "annotation_balanced_accuracy",
        "map_accuracy", "map_balanced_accuracy", "std_map_accuracy"
    ]      
    avg_df = (
        results
        .groupby(["model", "shot", "exp"], as_index=False)[accuracy_cols]
        .mean()
    )

    # Step 3: Add a marker for dataset column to indicate this is an average
    avg_df["dataset"] = "avg_datasets"

    # Step 4: Append back to results
    results_with_avg = pd.concat([results, avg_df], ignore_index=True)

    return results_with_avg


dataset_names = {
    'bach': 'BACH \cite{datasetbach}',
    'sicap_mil': 'SICAP\_MIL \cite{datasetsicapmil}',
    'nct': 'NCT \cite{datasetnct}',
    'bracs': 'BRACS \cite{datasetbracs}', 
    'esca': 'ESCA \cite{datasetesca}', 
    'bach_wsi': 'BACH\_WSI \cite{datasetbach}',
    'avg_datasets': 'AVERAGE'
}


method_names = {
    'baseline_random': 'LP \cite{Zhou2003} (random)',
    'baseline_error': 'LP \cite{Zhou2003} (error)',
    'baseline_diverseerror': 'LP \cite{Zhou2003} (diverse error)',
    'fs_random': 'Ours (random)',
    'fs_error': 'Ours (error)',
    'fs_diverseerror': 'Ours (diverse error)',
    'hitl_error': 'Ours (error)',
    'hitl_diverseerror': 'Ours (diverse error)',
}

ann_names = {
    'baseline_random': 'One-time',
    'baseline_diverseerror': 'One-time',
    'baseline_error': 'One-time',
    'fs_random': 'One-time',
    'fs_diverseerror': 'One-time',
    'fs_error': 'One-time',
    'hitl_error': 'HITL',
    'hitl_diverseerror': 'HITL'
}

def create_table_from_df(args, shots, df):
    # Pivot to get map_accuracy
    pivoted = df.pivot_table(
        index=["dataset", "exp"],
        columns="shot",
        values="map_accuracy"
    ).reset_index()

    pivoted = pivoted.fillna(0)
    
    # Get zero-shot accuracy for shot = 0 only (assuming zs_accuracy is constant per (dataset, exp))
    zs_acc = df[df["shot"] == 0][["dataset", "exp", "zs_accuracy"]].drop_duplicates()

    # Merge
    final_df = pd.merge(zs_acc, pivoted, on=["dataset", "exp"], how="left")
    final_df = final_df.sort_values(by=["dataset", "exp"])
    final_df = final_df.replace(-1, np.nan)

    # Construct LaTeX table string
    latex_lines = []
    latex_lines.append("\\begin{table*}[]")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\caption{Accuracy of the proposed method in one-time and HITL settings across five patch-level datasets, for different number of annotations $k$.}")
    
    shot_headers = " & ".join([f"\\textbf{{k={s}}}" for s in shots])
    latex_lines.append(f"    \\resizebox{{\\textwidth}}{{!}}{{\\begin{{tabular}}{{lcc|c{'c' * len(shots)}}}")
    latex_lines.append(f"        \\textbf{{Dataset}} & \\textbf{{Method}} & \\textbf{{Ann}} & \\textbf{{ZS}} & {shot_headers} \\\\")
    latex_lines.append("        \\hline")

    # Column headers
    current_dataset = None
    for _, row in final_df.iterrows():
        dataset = row["dataset"]
        exp = row["exp"]
        zs = f"{row['zs_accuracy']*100:.1f}"
        if exp != 'hitl':
            shots_values = " & ".join(
                f"{row.get(shot, float('nan'))*100:.1f}" if pd.notna(row.get(shot)) else "-" for shot in shots 
            )
        else: 
            shots_values = " & ".join(
                f"\cellcolor[HTML]{{CCEFD0}}{row.get(shot, float('nan'))*100:.1f}" if pd.notna(row.get(shot)) else "-" for shot in shots 
            )
        if dataset != current_dataset:
            latex_lines.append(f"        \\textbf{{{dataset_names[dataset]}}} & \\textbf{{{method_names[exp]}}}  & \\textbf{{{ann_names[exp]}}} & {zs} & {shots_values} \\\\")
            current_dataset = dataset
        else:
            latex_lines.append(f"        & \\textbf{{{method_names[exp]}}} & \\textbf{{{ann_names[exp]}}} & {zs} & {shots_values} \\\\")
            latex_lines.append("     \\hline")

    latex_lines.append("    \\end{tabular}}")
    latex_lines.append("    \\label{tab:patch}")
    latex_lines.append("\\end{table*}")

    # Write to file
    latex_table_path = os.path.join(args.root_dir, 'results', 'final_latex_table.txt')
    with open(latex_table_path, "w") as f:
        for line in latex_lines:
            f.write(line + "\n")

    return f'Table saved in txt file {latex_table_path}'


def plot_FS(args):
    # Define the directory containing the results
    results_dir = os.path.join(args.root_dir, 'results', args.dataset, args.model)

    # Load dataframe
    csv_name = f'iteration_{args.dataset}_{args.model}_results_mean_summary.csv'

    #try:
    df = pd.read_csv(os.path.join(results_dir, csv_name))

    n = 50
    if args.dataset in ['bracs']:
        n = 50 
        i = -1
    elif args.dataset in [ 'nct', 'sicap_mil', 'bach']:
        n = 50
        i = -1 
    elif args.dataset in [ 'bach']:
        n = 50
        i = -1 
    else: 
        raise RuntimeError(f"Not implemented yet for dataset {args.dataset}")
    df_rows = df[
                (df["N_PROP"] == n) &
                (df["weight"] == str(args.w)) &
                (df["N_UP"] == args.N_UP) &
                (df["linear"] == args.linear) &
                (df["ann_iterations"] == args.ann_iterations) & 
                (df["sparse_method"] == str(args.sparse_method)) &
                (df["pair_pot"] == str(args.pair_pot)) &
                (df["compat"] == str(args.compat)) & 
                (df["annotation_method"] == str(args.annotation_method)) &
                (df["n_iterations"] == int(args.it)) & 
                (df["n_affinity"] == "[0, 16]") 
            ] 

    num_plots = 1
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 3), sharey=False)

    shots =  [0, 5, 10, 15, 20, 25, 30, 50, 100]
    # Get the colormap
    cmap = plt.get_cmap("PuBuGn") #Wistia, PuBuGn ##Greens
    num_colors = len(shots) + 5
    colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)] # Normalize values to range [0, 1] to sample colormap

    it = 5 
    y_max, y_min = 0, 100
    y2_max, y2_min = 0, 100
    for _, row in df_rows.iterrows():
        # Plot 1 
        y = [row["initial_annotation_accuracy"]]
        if y[0] == -1:
            y[0] = row["initial_accuracy"]

        y[0] = ast.literal_eval(row["annotation_accuracies"])[0]
        n_annotations = row["n_annotations"]
        if n_annotations in shots:
            annotation_map_accuracies = ast.literal_eval(row["map_accuracies"]) 
            if np.max(annotation_map_accuracies) > y_max:
                y_max = np.max(annotation_map_accuracies)
            if y[0] < y_min:
                y_min = y[0]
            y.extend(annotation_map_accuracies[:i])
            x = list(range(len(y)))
            if n_annotations == 0: 
                axes.plot(x, y, linestyle='-', c=colors[it], marker=".", markersize=4, markevery=5, label='k={0, 5, 10, 15, 20, 25, 30, 50, 100}') #axes[0]
            else: 
                axes.plot(x, y, linestyle='-', c=colors[it], marker=".", markersize=4, markevery=5) #axes[0]

            # Plot 2
            if num_plots == 2:
                y2 = [row["initial_annotated_balanced_accuracy"]]
                if y2[0] == -1:
                    y2[0] = row["initial_balanced_accuracy"]
                annotation_map_accuracies2 = ast.literal_eval(row["map_balanced_accuracies"]) 

                y2[0] = ast.literal_eval(row["annotation_balanced_accuracies"])[0]
                if np.max(annotation_map_accuracies2) > y2_max:
                    y2_max = np.max(annotation_map_accuracies2)
                if y2[0] < y2_min:
                    y2_min = y2[0]
                y2.extend(annotation_map_accuracies2[:i])
                x2 = list(range(len(y2)))
                axes[1].plot(x2, y2, label=f"{n_annotations}", linestyle='-', c=colors[it], marker=".")

            it += 1

    # Plot 1
    axes.hlines(float(row["initial_accuracy"]), xmin=x[0], xmax=x[-1], label='ZS', colors='r', linestyle='--')
    axes.set_xlabel("Label Propagation Iteration")
    axes.set_ylabel("Accuracy")
    '''
    axes.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),  # X=0.5 centers it, Y<0 puts it below
        ncol=4,  # Number of columns (adjust to fit your legend items)
        frameon=False  # Optional: removes the legend frame
    )
    '''
    axes.legend()
    axes.grid(True, axis='y')
    axes.set_xticks(np.arange(0, len(y)+1, step=5))

    if num_plots == 2:
        # Plot 2 
        axes[1].hlines(float(row["initial_balanced_accuracy"]), xmin=x2[0], xmax=x2[-1], label='ZS - No prop', colors='r', linestyle='--')
        axes[1].set_xlabel("Label Propagation Iteration")
        axes[1].set_ylabel("Balanced ccuracy")
        axes[1].legend()
        axes[1].grid(True, axis='y')
        axes[1].set_xticks(np.arange(0, len(y2)+1, step=5))

    plt.tight_layout()
    save_path = os.path.join(results_dir, f'plotFS_{args.w}_{args.ann_iterations}_{args.PROP}_{args.N_UP}_{args.sparse_method}_{args.dataset}_{args.model}_{args.pair_pot}_{args.compat}_{args.linear}.png')
    plt.savefig(save_path)


def mask_to_centers(binary_mask):
    # Label connected components
    labeled = label(binary_mask)

    # Create an empty mask for centers
    centers_mask = np.zeros_like(binary_mask, dtype=bool)

    # Extract centroid for each component
    for region in regionprops(labeled):
        cy, cx = map(int, region.centroid)  # centroid (row, col)
        centers_mask[cy, cx] = True

    return centers_mask


# Custom palette
my_colors = [
    "#f1c4d5",  
    "#4ec15b", 
    "#e77bf7", 
    "#480848",  
    "#e0e0e0",  
]
cmap3 = mcolors.ListedColormap(my_colors)




def visualize(args):
    # Define the directory containing the features 
    data_dir = os.path.join(raw_data_dir, 'data_processed', args.dataset, args.model)
    npz_name = f'{args.dataset}_{args.model}_{args.n_patient}' 
    wsi_path = os.path.join(raw_data_dir, 'thunder', 'datasets', 'bach', 'ICIAR2018_BACH_Challenge', 'WSI', 'thumbnails', 'A0{str(args.n_patient)}_thumb.png')
    #wsi_path = f'/auto/globalscratch/users/t/g/tgodelai/thunder/datasets/bach/ICIAR2018_BACH_Challenge/WSI/thumbnails/A0{str(args.n_patient)}_thumb.png'

    npz_path = os.path.join(data_dir, npz_name+'.npz')

    # Read npz
    data = np.load(npz_path, allow_pickle=True)

    positions = data['positions']
    x_coords = positions[:positions.shape[0]//2]
    y_coords = positions[positions.shape[0]//2:]
    dy, dx = 512, 512
    grid_y = (y_coords - y_coords.min()) // dy
    grid_x = (x_coords - x_coords.min()) // dx

    labels = data['labels']
    probabilities = np.argmax(data['probabilities'], axis=1)

    # Define the directory containing the results
    results_dir = os.path.join(args.root_dir, 'results', args.dataset, args.model)
    json_name = f'{args.N_PROP}_{args.N_UP}_{str(args.ann_iterations)}_{str(args.it)}_{str(args.w)}_{args.var}_{args.pair_pot}_{args.compat}_{str(args.temperature)}_{args.sparse_method}_{str(args.n_affinity)}_{str(args.n_annotations)}_{str(args.n_class_not_annotated)}_{str(args.seed)}_{args.annotation_method}_{args.linear}_{str(npz_name)}.json'
    
    # Read json
    with open(os.path.join(results_dir, json_name), 'r') as file:
        data = json.load(file)

    final_label = data["final_label"]
    final_label = [f for i,f in enumerate(final_label) if i in [0, 1, 2, 3]]# [0, 7]]

    fig, axs = plt.subplots(1, len(final_label)+2+1, figsize=(40*(len(final_label)+2), 40)) #8 au lieu de 40

    unique_labels, label_map = reconstruct(positions, labels)
    _, ZSpred_map = reconstruct(positions, probabilities)

    for i, it in enumerate(range(len(final_label))):
        final_label_it = final_label[it]

        _, prop_map = reconstruct(positions, final_label_it)

        # Plot second label map
        #cmap3 = plt.get_cmap('tab20', len(unique_labels))
        im3 = axs[i+2].imshow(prop_map, cmap=cmap3, interpolation='nearest')
        axs[i+2].set_title("Propagation")
        axs[i+2].axis('off')

    # Plot image label map
    wsi = Image.open(wsi_path)
    im1 = axs[0].imshow(wsi)
    axs[0].set_title("WSI")
    axs[0].axis('off')

    # Plot first label map
    im1 = axs[-1].imshow(label_map, cmap=cmap3, interpolation='nearest')
    axs[-1].set_title("Ground truth")
    axs[-1].axis('off')
    nrows, ncols = label_map.shape
    #cbar3 = fig.colorbar(im1, ax=axs[-1], ticks=np.arange(len(unique_labels)))
    #cbar3.set_label('Label Index')
    #for i in range(nrows):
    #    for j in range(ncols):
    #        axs[-1].text(j, i, f"({j},{i})", ha='center', va='center', fontsize=6, color='black')

    selected_indices = data["annotations"]
    nrows, ncols = label_map.shape
    mask = np.zeros_like(label_map, dtype=bool) 
    for idx in selected_indices:
        row = grid_y[idx]
        col = grid_x[idx] 
        if 0 <= row < nrows and 0 <= col < ncols:  # safety check
            mask[row, col] = True
    centers_mask = mask_to_centers(mask)
    # axs[-1].imshow(mask, cmap="gray", alpha=0.8, interpolation='nearest')
    #axs[-1].imshow(centers_mask, cmap="gray", alpha=0.8, interpolation='nearest')
    ys, xs = np.where(centers_mask)
    # Overlay red crosses
    axs[-1].plot(xs, ys, 'rx', markersize=10, mew=2)  # 'rx' = red 'x'

    # Plot first label map
    im2 = axs[1].imshow(ZSpred_map, cmap=cmap3, interpolation='nearest')
    axs[1].set_title("ZS prediction")
    axs[1].axis('off')
    

    plt.tight_layout()
    #plt.show()
    save_path = os.path.join(results_dir, f'plot_{args.n_annotations}_{args.ann_iterations}_{args.PROP}_{args.N_UP}_{args.sparse_method}_{args.dataset}_{args.model}_{args.pair_pot}_{args.compat}_{args.seed}_{npz_name}.png')
    #plt.savefig(save_path)

    for i, ax in enumerate(axs):
        # Get the bounding box of the subplot in figure coordinates
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        
        # Save just this subplot
        save_path_subfig = save_path.replace('.png', f'_{i}.pdf')
        fig.savefig(save_path_subfig, bbox_inches=extent)

    return 


def create_table_ablation1(args, datasets, models, sparsity, shots):
    # Get df 
    results = pd.DataFrame(columns=[
        "model", "dataset", "sparsity", "shot",
        "zs_accuracy", "zs_balanced_accuracy", 
        "annotation_accuracy", "annotation_balanced_accuracy", 
        "map_accuracy", "map_balanced_accuracy", "std_map_accuracy"
        ])
    for dataset in datasets:
        n, i = 50, -1
        for model in models:
            # Define the directory containing the results
            if dataset in ['bach', 'tcga_uniform', 'esca', 'skincancer2', 'bach_wsi']:
                results_dir = os.path.join(args.root_dir.replace('CECI/home', 'globalscratch'), 'results', dataset, model)
            else:
                results_dir = os.path.join(args.root_dir, 'results', dataset, model)
            
            # Load dataframe
            csv_name = f'iteration_{dataset}_{model}_results_mean_summary.csv'
            df = pd.read_csv(os.path.join(results_dir, csv_name))
            df = df.replace(np.nan, 'None')

            for s in sparsity:
                for shot in shots: 
                    print(f"-------------- Dataset {dataset}, model {model}, sparsity {s}, shots {shot} --------------")
                    affinity = f'[0, {s}]'
                    df_rows = df[
                        (df["N_PROP"] == n) & 
                        (df["sparse_method"] == str(args.sparse_method)) &
                        (df["pair_pot"] == str(args.pair_pot)) &
                        (df["compat"] == str(args.compat)) & 
                        (df["annotation_method"] == str(args.annotation_method)) &
                        (df["n_iterations"] == int(args.it)) &
                        (df["n_affinity"] == affinity) & 
                        (df["weight"] == str(args.w)) & 
                        (df["ann_iterations"] == int(args.ann_iterations)) &
                        (df["N_UP"] == -1) & # No prototype
                        (df["linear"] == False) & # No linear 
                        (df["n_annotations"] == int(shot))
                    ] 
                    try: 
                        new_results = pd.DataFrame([{
                            "model": model,
                            "dataset": dataset,
                            "sparsity": s, 
                            "shot": int(shot), 

                            "zs_accuracy": float(df_rows["initial_accuracy"].iloc[0]),
                            "zs_balanced_accuracy": float(df_rows["initial_balanced_accuracy"].iloc[0]),
                
                            "annotation_accuracy": float(ast.literal_eval(df_rows["annotation_accuracies"].iloc[0])[0]), 
                            "annotation_balanced_accuracy": float(ast.literal_eval(df_rows["annotation_balanced_accuracies"].iloc[0])[0]),
                
                            "map_accuracy": float(ast.literal_eval(df_rows["map_accuracies"].iloc[0])[i]),
                            "map_balanced_accuracy": float(ast.literal_eval(df_rows["map_balanced_accuracies"].iloc[0])[i]),
                            "std_map_accuracy": float(ast.literal_eval(df_rows["map_accuracies_std"].iloc[0])[i]),                    
                        }])
                        results = pd.concat([results, new_results])
                        print("No empty dataframe")
                    except: 
                        print(f"Empty dataframe")
                        pass

    # Pivot to get map_accuracy
    df = results
    pivoted = df.pivot_table(
        index=["dataset", "sparsity"],
        columns="shot",
        values="map_accuracy"
    ).reset_index()

    pivoted = pivoted.fillna(0)
    
    # Get zero-shot accuracy for shot = 0 only (assuming zs_accuracy is constant per (dataset, exp))
    zs_acc = df[df["shot"] == 0][["dataset", "sparsity", "zs_accuracy"]].drop_duplicates()
    print("zs_acc", zs_acc)

    # Merge
    final_df = pd.merge(zs_acc, pivoted, on=["dataset", "sparsity"], how="left")
    final_df = final_df.sort_values(by=["dataset", "sparsity"])
    final_df = final_df.replace(-1, np.nan)
    print(final_df)

    # Construct LaTeX table string
    latex_lines = []
    latex_lines.append("\\begin{table*}[]")
    latex_lines.append("    \\centering")
    
    shot_headers = " & ".join([f"\\textbf{{k={s}}}" for s in shots])
    latex_lines.append(f"    \\resizebox{{\\textwidth}}{{!}}{{\\begin{{tabular}}{{lc|c{'c' * len(shots)}}}")
    latex_lines.append(f"        \\textbf{{Dataset}} & \\textbf{{Ann}} & \\textbf{{ZS}} & {shot_headers} \\\\")
    latex_lines.append("        \\hline")

    # Column headers
    current_dataset = None
    for _, row in final_df.iterrows():
        dataset = row["dataset"]
        sparsity = row["sparsity"]
        zs = f"{row['zs_accuracy']*100:.1f}"
        shots_values = " & ".join(
            f"{row.get(shot, float('nan'))*100:.1f}" if pd.notna(row.get(shot)) else "-" for shot in shots 
        )
        if dataset != current_dataset:
            latex_lines.append(f"        \\textbf{{{dataset_names[dataset]}}} & \\textbf{{{sparsity}}} & {zs} & {shots_values} \\\\")
            current_dataset = dataset
        else:
            latex_lines.append(f"        & \\textbf{{{sparsity}}} & {zs} & {shots_values} \\\\")
            latex_lines.append("     \\hline")

    latex_lines.append("    \\end{tabular}}")
    latex_lines.append("    \\caption{First table}")
    latex_lines.append("    \\label{tab:1}")
    latex_lines.append("\\end{table*}")

    # Write to file
    latex_table_path = os.path.join(args.root_dir, 'results', 'final_latex_table_ablation1.txt')
    with open(latex_table_path, "w") as f:
        for line in latex_lines:
            f.write(line + "\n")

    return f'Table saved in txt file {latex_table_path}'


def create_table_ablation2(args, datasets, models, weights):
    # Get df 
    results = pd.DataFrame(columns=[
        "model", "dataset", "weights", "shot",
        "zs_accuracy", "zs_balanced_accuracy", 
        "annotation_accuracy", "annotation_balanced_accuracy", 
        "map_accuracy", "map_balanced_accuracy", "std_map_accuracy"
        ])
    for dataset in datasets:
        n, i = 50, -1
        for model in models:
            # Define the directory containing the results
            if dataset in ['bach', 'tcga_uniform', 'esca', 'skincancer2', 'bach_wsi']:
                results_dir = os.path.join(args.root_dir.replace('CECI/home', 'globalscratch'), 'results', dataset, model)
            else:
                results_dir = os.path.join(args.root_dir, 'results', dataset, model)
            
            # Load dataframe
            csv_name = f'iteration_{dataset}_{model}_results_mean_summary.csv'
            df = pd.read_csv(os.path.join(results_dir, csv_name))
            df = df.replace(np.nan, 'None')

            for w in weights: 
                print(f"-------------- Dataset {dataset}, model {model}, weights {w} --------------")
                affinity = f'[0, 16]'
                weight = f'{w},1.0'
                df_rows = df[
                    (df["N_PROP"] == n) & 
                    (df["sparse_method"] == str(args.sparse_method)) &
                    (df["pair_pot"] == str(args.pair_pot)) &
                    (df["compat"] == str(args.compat)) & 
                    (df["annotation_method"] == str(args.annotation_method)) &
                    (df["n_iterations"] == int(args.it)) &
                    (df["n_affinity"] == affinity) & 
                    (df["weight"] == weight) & 
                    (df["ann_iterations"] == int(args.ann_iterations)) &
                    (df["N_UP"] == -1) & # No prototype
                    (df["linear"] == False) & # No linear 
                    (df["n_annotations"] == 0)
                ] 
                print("w",w)
                try: 
                    new_results = pd.DataFrame([{
                        "model": model,
                        "dataset": dataset,
                        "weights": float(w), 
                        "shot": int(0),

                        "zs_accuracy": float(df_rows["initial_accuracy"].iloc[0]),
                        "zs_balanced_accuracy": float(df_rows["initial_balanced_accuracy"].iloc[0]),
            
                        "annotation_accuracy": float(ast.literal_eval(df_rows["annotation_accuracies"].iloc[0])[0]), 
                        "annotation_balanced_accuracy": float(ast.literal_eval(df_rows["annotation_balanced_accuracies"].iloc[0])[0]),
            
                        "map_accuracy": float(ast.literal_eval(df_rows["map_accuracies"].iloc[0])[i]),
                        "map_balanced_accuracy": float(ast.literal_eval(df_rows["map_balanced_accuracies"].iloc[0])[i]),
                        "std_map_accuracy": float(ast.literal_eval(df_rows["map_accuracies_std"].iloc[0])[i]),                    
                    }])
                    results = pd.concat([results, new_results])
                    print("No empty dataframe")
                except: 
                    print(f"Empty dataframe")
                    pass

    shots = [0, 0]
    # Pivot to get map_accuracy
    df = results
    pivoted = df.pivot_table(
        index=["dataset", "weights"],
        columns="shot",
        values="map_accuracy"
    ).reset_index()
    pivoted = pivoted.fillna(0)
    
    # Get zero-shot accuracy for shot = 0 only (assuming zs_accuracy is constant per (dataset, exp))
    zs_acc = df[df["shot"] == 0][["dataset", "weights", "zs_accuracy"]].drop_duplicates()

    # Merge
    final_df = pd.merge(zs_acc, pivoted, on=["dataset", "weights"], how="left")
    final_df = final_df.sort_values(by=["dataset", "weights"])
    final_df = final_df.replace(-1, np.nan)
    print("final_df", final_df)

    # Construct LaTeX table string
    latex_lines = []
    latex_lines.append("\\begin{table*}[]")
    latex_lines.append("    \\centering")
    
    shot_headers = " & ".join([f"\\textbf{{k={s}}}" for s in shots])
    latex_lines.append(f"    \\resizebox{{\\textwidth}}{{!}}{{\\begin{{tabular}}{{lc|c{'c' * len(shots)}}}")
    latex_lines.append(f"        \\textbf{{Dataset}} & \\textbf{{Ann}} & \\textbf{{ZS}} & {shot_headers} \\\\")
    latex_lines.append("        \\hline")

    # Column headers
    current_dataset = None
    for _, row in final_df.iterrows():
        dataset = row["dataset"]
        weight = row["weights"]
        zs = f"{row['zs_accuracy']*100:.1f}"
        shots_values = " & ".join(
            f"{row.get(shot, float('nan'))*100:.1f}" if pd.notna(row.get(shot)) else "-" for shot in shots 
        )
        if dataset != current_dataset:
            latex_lines.append(f"        \\textbf{{{dataset_names[dataset]}}} & \\textbf{{{weight}}} & {zs} & {shots_values} \\\\")
            current_dataset = dataset
        else:
            latex_lines.append(f"        & \\textbf{{{weight}}} & {zs} & {shots_values} \\\\")
            latex_lines.append("     \\hline")

    latex_lines.append("    \\end{tabular}}")
    latex_lines.append("    \\caption{First table}")
    latex_lines.append("    \\label{tab:1}")
    latex_lines.append("\\end{table*}")

    # Write to file
    latex_table_path = os.path.join(args.root_dir, 'results', 'final_latex_table_ablation2.txt')
    with open(latex_table_path, "w") as f:
        for line in latex_lines:
            f.write(line + "\n")

    return f'Table saved in txt file {latex_table_path}'


def create_table_ablation3(args, datasets, models, weights, shots):
    # Get df 
    results = pd.DataFrame(columns=[
        "model", "dataset", "weights", "shot",
        "zs_accuracy", "zs_balanced_accuracy", 
        "annotation_accuracy", "annotation_balanced_accuracy", 
        "map_accuracy", "map_balanced_accuracy", "std_map_accuracy"
        ])
    for dataset in datasets:
        n, i = 50, -1
        for model in models:
            # Define the directory containing the results
            if dataset in ['bach', 'tcga_uniform', 'esca', 'skincancer2', 'bach_wsi']:
                results_dir = os.path.join(args.root_dir.replace('CECI/home', 'globalscratch'), 'results', dataset, model)
            else:
                results_dir = os.path.join(args.root_dir, 'results', dataset, model)
            
            # Load dataframe
            csv_name = f'iteration_{dataset}_{model}_results_mean_summary.csv'
            df = pd.read_csv(os.path.join(results_dir, csv_name))
            df = df.replace(np.nan, 'None')

            for w in weights: 
                for shot in shots: 
                    print(f"-------------- Dataset {dataset}, model {model}, weights {w}, shot {shot} --------------")
                    affinity = f'[0, 16]'
                    #weight = f'0.075,{w}'
                    weight = f'0.1,{w}'
                    df_rows = df[
                        (df["N_PROP"] == n) & 
                        (df["sparse_method"] == str(args.sparse_method)) &
                        (df["pair_pot"] == str(args.pair_pot)) &
                        (df["compat"] == str(args.compat)) & 
                        (df["annotation_method"] == str(args.annotation_method)) &
                        (df["n_iterations"] == int(args.it)) &
                        (df["n_affinity"] == affinity) & 
                        (df["weight"] == weight) & 
                        (df["ann_iterations"] == int(args.ann_iterations)) &
                        (df["N_UP"] == -1) & # No prototype
                        (df["linear"] == False) & # No linear 
                        (df["n_annotations"] == int(shot))
                    ] 
                    try: 
                        new_results = pd.DataFrame([{
                            "model": model,
                            "dataset": dataset,
                            "weights": float(w), 
                            "shot": int(shot),

                            "zs_accuracy": float(df_rows["initial_accuracy"].iloc[0]),
                            "zs_balanced_accuracy": float(df_rows["initial_balanced_accuracy"].iloc[0]),
                
                            "annotation_accuracy": float(ast.literal_eval(df_rows["annotation_accuracies"].iloc[0])[0]), 
                            "annotation_balanced_accuracy": float(ast.literal_eval(df_rows["annotation_balanced_accuracies"].iloc[0])[0]),
                
                            "map_accuracy": float(ast.literal_eval(df_rows["map_accuracies"].iloc[0])[i]),
                            "map_balanced_accuracy": float(ast.literal_eval(df_rows["map_balanced_accuracies"].iloc[0])[i]),
                            "std_map_accuracy": float(ast.literal_eval(df_rows["map_accuracies_std"].iloc[0])[i]),                    
                        }])
                        results = pd.concat([results, new_results])
                        print("No empty dataframe")
                    except: 
                        print(f"Empty dataframe")
                        pass

    # Pivot to get map_accuracy
    df = results
    pivoted = df.pivot_table(
        index=["dataset", "weights"],
        columns="shot",
        values="map_accuracy"
    ).reset_index()
    pivoted = pivoted.fillna(0)
    
    # Get zero-shot accuracy for shot = 0 only (assuming zs_accuracy is constant per (dataset, exp))
    zs_acc = df[df["shot"] == 0][["dataset", "weights", "zs_accuracy"]].drop_duplicates()

    # Merge
    final_df = pd.merge(zs_acc, pivoted, on=["dataset", "weights"], how="left")
    final_df = final_df.sort_values(by=["dataset", "weights"])
    final_df = final_df.replace(-1, np.nan)
    print("final_df", final_df)

    # Construct LaTeX table string
    latex_lines = []
    latex_lines.append("\\begin{table*}[]")
    latex_lines.append("    \\centering")
    
    shot_headers = " & ".join([f"\\textbf{{k={s}}}" for s in shots])
    latex_lines.append(f"    \\resizebox{{\\textwidth}}{{!}}{{\\begin{{tabular}}{{lc|c{'c' * len(shots)}}}")
    latex_lines.append(f"        \\textbf{{Dataset}} & \\textbf{{Ann}} & \\textbf{{ZS}} & {shot_headers} \\\\")
    latex_lines.append("        \\hline")

    # Column headers
    current_dataset = None
    for _, row in final_df.iterrows():
        dataset = row["dataset"]
        weight = row["weights"]
        zs = f"{row['zs_accuracy']*100:.1f}"
        shots_values = " & ".join(
            f"{row.get(shot, float('nan'))*100:.1f}" if pd.notna(row.get(shot)) else "-" for shot in shots 
        )
        if dataset != current_dataset:
            latex_lines.append(f"        \\textbf{{{dataset_names[dataset]}}} & \\textbf{{{weight}}} & {zs} & {shots_values} \\\\")
            current_dataset = dataset
        else:
            latex_lines.append(f"        & \\textbf{{{weight}}} & {zs} & {shots_values} \\\\")
            latex_lines.append("     \\hline")

    latex_lines.append("    \\end{tabular}}")
    latex_lines.append("    \\caption{First table}")
    latex_lines.append("    \\label{tab:1}")
    latex_lines.append("\\end{table*}")

    # Write to file
    latex_table_path = os.path.join(args.root_dir, 'results', 'final_latex_table_ablation3.txt')
    with open(latex_table_path, "w") as f:
        for line in latex_lines:
            f.write(line + "\n")

    return f'Table saved in txt file {latex_table_path}'


def create_table_ablation4(args, datasets, models, shots, compat, pairwises):
    # Get df 
    results = pd.DataFrame(columns=[
        "model", "dataset", "shot", "pairwises", 
        "zs_accuracy", "zs_balanced_accuracy", 
        "annotation_accuracy", "annotation_balanced_accuracy", 
        "map_accuracy", "map_balanced_accuracy", "std_map_accuracy"
        ])
    for dataset in datasets:
        n, i = 50, -1
        for model in models:
            # Define the directory containing the results
            if dataset in ['bach', 'tcga_uniform', 'esca', 'skincancer2', 'bach_wsi']:
                results_dir = os.path.join(args.root_dir.replace('CECI/home', 'globalscratch'), 'results', dataset, model)
            else:
                results_dir = os.path.join(args.root_dir, 'results', dataset, model)
            
            # Load dataframe
            csv_name = f'iteration_{dataset}_{model}_results_mean_summary.csv'
            df = pd.read_csv(os.path.join(results_dir, csv_name))
            df = df.replace(np.nan, 'None')
 
            for p, c in zip(pairwises, compat): 
                for shot in shots: 
                    print(f"-------------- Dataset {dataset}, model {model}, shot {shot}, pairwise {p} --------------")
                    affinity = f'[0, 16]'
                    #weight = f'0.075,0.0075'
                    df_rows = df[
                        (df["N_PROP"] == n) & 
                        (df["sparse_method"] == str(args.sparse_method)) &
                        (df["pair_pot"] == p) &
                        (df["compat"] == c) &
                        (df["annotation_method"] == str(args.annotation_method)) &
                        (df["n_iterations"] == int(args.it)) &
                        (df["n_affinity"] == affinity) & 
                        (df["weight"] == str(args.w)) & 
                        (df["ann_iterations"] == int(args.ann_iterations)) &
                        (df["N_UP"] == -1) & # No prototype
                        (df["linear"] == False) & # No linear 
                        (df["n_annotations"] == int(shot))
                    ] 
                    try: 
                        new_results = pd.DataFrame([{
                            "model": model,
                            "dataset": dataset,
                            "shot": int(shot),
                            "pairwises": str(p),

                            "zs_accuracy": float(df_rows["initial_accuracy"].iloc[0]),
                            "zs_balanced_accuracy": float(df_rows["initial_balanced_accuracy"].iloc[0]),
                
                            "annotation_accuracy": float(ast.literal_eval(df_rows["annotation_accuracies"].iloc[0])[0]), 
                            "annotation_balanced_accuracy": float(ast.literal_eval(df_rows["annotation_balanced_accuracies"].iloc[0])[0]),
                
                            "map_accuracy": float(ast.literal_eval(df_rows["map_accuracies"].iloc[0])[i]),
                            "map_balanced_accuracy": float(ast.literal_eval(df_rows["map_balanced_accuracies"].iloc[0])[i]),
                            "std_map_accuracy": float(ast.literal_eval(df_rows["map_accuracies_std"].iloc[0])[i]),                    
                        }])
                        results = pd.concat([results, new_results])
                        print("No empty dataframe")
                    except: 
                        print(f"Empty dataframe")
                        pass

    # Pivot to get map_accuracy
    df = results
    pivoted = df.pivot_table(
        index=["dataset", "pairwises"],
        columns="shot",
        values="map_accuracy"
    ).reset_index()
    pivoted = pivoted.fillna(0)
    
    # Get zero-shot accuracy for shot = 0 only (assuming zs_accuracy is constant per (dataset, exp))
    zs_acc = df[df["shot"] == 0][["dataset", "pairwises", "zs_accuracy"]].drop_duplicates()

    # Merge
    final_df = pd.merge(zs_acc, pivoted, on=["dataset", "pairwises"], how="left")
    final_df = final_df.sort_values(by=["dataset", "pairwises"])
    final_df = final_df.replace(-1, np.nan)
    print("final_df", final_df)

    # Construct LaTeX table string
    latex_lines = []
    latex_lines.append("\\begin{table*}[]")
    latex_lines.append("    \\centering")
    
    shot_headers = " & ".join([f"\\textbf{{k={s}}}" for s in shots])
    latex_lines.append(f"    \\resizebox{{\\textwidth}}{{!}}{{\\begin{{tabular}}{{lc|c{'c' * len(shots)}}}")
    latex_lines.append(f"        \\textbf{{Dataset}} & \\textbf{{Ann}} & \\textbf{{ZS}} & {shot_headers} \\\\")
    latex_lines.append("        \\hline")

    # Column headers
    current_dataset = None
    for _, row in final_df.iterrows():
        dataset = row["dataset"]
        pairwise = row["pairwises"]
        zs = f"{row['zs_accuracy']*100:.1f}"
        shots_values = " & ".join(
            f"{row.get(shot, float('nan'))*100:.1f}" if pd.notna(row.get(shot)) else "-" for shot in shots 
        )
        if dataset != current_dataset:
            latex_lines.append(f"        \\textbf{{{dataset_names[dataset]}}} & \\textbf{{{pairwise}}} & {zs} & {shots_values} \\\\")
            current_dataset = dataset
        else:
            latex_lines.append(f"        & \\textbf{{{pairwise}}} & {zs} & {shots_values} \\\\")
            latex_lines.append("     \\hline")

    latex_lines.append("    \\end{tabular}}")
    latex_lines.append("    \\caption{First table}")
    latex_lines.append("    \\label{tab:1}")
    latex_lines.append("\\end{table*}")

    # Write to file
    latex_table_path = os.path.join(args.root_dir, 'results', 'final_latex_table_ablation4.txt')
    with open(latex_table_path, "w") as f:
        for line in latex_lines:
            f.write(line + "\n")

    return f'Table saved in txt file {latex_table_path}'


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/CECI/home/users/t/g/tgodelai/miccai25/')
    parser.add_argument('--dataset', type=str, default='sicap_mil', choices=['sicap_mil', 'monuseg', 'panuke', 'skincancer2', 'lc_lung', 'lc_colon', 'nct', 'bracs', 'bach', 'esca', 'tcga_uniform', 'bach_wsi'])
    parser.add_argument('--model', type=str, default='clip', choices=['clip', 'quilt', 'conch', 'plip', 'nnunet', 'plipanduni2', 'conchanduni2', 'conchandgigapath', 'conchandoptimus1'])
    parser.add_argument('--n_patient', type=int, default=0)
    parser.add_argument('--it', type=int, default=10)
    parser.add_argument('--w', type=str, default='0.1n0.01')
    parser.add_argument('--var', type=str, default='')
    parser.add_argument('--pair_pot', type=str, default='minusmodelfeatures,modelfeaturesann')
    parser.add_argument('--compat', type=str, default='indicator,potts')
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--sparse_method', type=str, default='cossim')
    parser.add_argument('--n_affinity', type=int, nargs='+', default=[0, 16])
    parser.add_argument('--n_annotations', type=int, default=1)
    parser.add_argument('--ann_iterations', type=int, default=1)
    parser.add_argument('--annotation_method', type=str, default='error')
    parser.add_argument('--n_class_not_annotated', type=int, default=0)
    parser.add_argument('--N_UP', type=int, default=-1)
    parser.add_argument('--PROP', type=bool, default=True) #, default=True)
    parser.add_argument('--N_PROP', type=int, default=1) #, default=True)
    parser.add_argument('--seed', type=int, default=42) #, default=True)
    parser.add_argument('--linear', type=bool, default=False)

    parser.add_argument('--task', type=str, choices=['table', 'plot', 'visualize', 'plotFS', 'plotHITL', 'tablefinal', 'tablefinal_bachwsi', 'tablefinal_ablation1', 'tablefinal_ablation2', 'tablefinal_ablation3', 'tablefinal_ablation4']) #, default=True)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parser()

    DIVERSEERROR = False
    raw_data_dir = '/auto/globalscratch/users/t/g/tgodelai/miccai25'

    datasets = ['bach', 'sicap_mil', 'bracs', 'nct'] #'sicap_mil', 'nct', 'lc_lung', 'bracs'] #, 'skincancer']
    models = ['conchanduni2']
    shots = [0, 5, 10, 15, 20, 25, 30, 50, 100]

    if args.task == 'plotFS':
        plot_FS(args)
    elif args.task == 'tablefinal':
        df_results = create_df_from_json(args, datasets, models, shots)
        print("results", df_results["map_accuracy"])
        create_table_from_df(args, shots, df_results)
    elif args.task == 'visualize':
        visualize(args)
    elif args.task == 'tablefinal_bachwsi':
        datasets = ['bach_wsi']
        models = ['conchanduni2']
        df_results = create_df_from_json(args, datasets, models, shots)
        create_table_from_df(args, shots, df_results)
    elif args.task == 'tablefinal_ablation1': # Ablation on sparsity 
        datasets = ['bracs']
        models = ['conchanduni2']
        sparsity = [4, 8, 16, 50, 100]
        shots = [0, 10, 50, 100]
        create_table_ablation1(args, datasets, models, sparsity, shots)
    elif args.task == 'tablefinal_ablation2': # Ablation on alpha
        datasets = ['bracs']
        models = ['conchanduni2']
        weights = [1.0, 0.5, 0.1, 0.05, 0.01]
        create_table_ablation2(args, datasets, models, weights)
    elif args.task == 'tablefinal_ablation3': # Ablation on beta
        datasets = ['bracs']
        models = ['conchanduni2']
        #weights = [0.075, 0.0075, 0.0375, 0.0, 0.00075]
        weights = [0.1, 0.01, 0.05, 0.0, 0.001, 0.005]
        shots = [0, 10, 50, 100]
        create_table_ablation3(args, datasets, models, weights, shots)
    elif args.task == 'tablefinal_ablation4': # Ablation on the definition of the pairwise potentials 
        datasets = ['bracs']
        models = ['conchanduni2']
        shots = [0, 10, 50, 100]
        compat = ['potts,potts', 'indicator,potts']
        pairwises = ['minusmodelfeatures,minusmodelfeaturesann', 'modelfeatures,minusmodelfeaturesann']
        create_table_ablation4(args, datasets, models, shots, compat, pairwises)
