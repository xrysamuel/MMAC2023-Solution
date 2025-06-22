import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import numpy as np
from sklearn.metrics import auc # Import auc to calculate if needed, though usually read from the curve directly

def plot_roc_curves(output_dir: str):
    """
    Loads ROC data from CSV files in the specified output directory
    and plots the ROC curve for each class.

    Args:
        output_dir (str): The base directory where the 'roc' folder is located.
                          (e.g., 'model_outputs')
    """
    roc_data_dir = os.path.join(output_dir, "roc")

    if not os.path.exists(roc_data_dir):
        print(f"Error: ROC data directory not found at {roc_data_dir}")
        print("Please ensure your training script has generated the ROC CSV files.")
        return

    csv_files = [f for f in os.listdir(roc_data_dir) if f.startswith("roc-class_") and f.endswith(".csv")]

    if not csv_files:
        print(f"No ROC CSV files found in {roc_data_dir}. Exiting.")
        return

    plt.figure(figsize=(10, 8))
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)') # Dashed diagonal for reference
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)

    for csv_file in sorted(csv_files):
        class_label = csv_file.replace("roc-class_", "").replace(".csv", "")
        file_path = os.path.join(roc_data_dir, csv_file)

        try:
            df = pd.read_csv(file_path)
            fpr = df['fpr'].values
            tpr = df['tpr'].values
            thresholds = df['threshold'].values

            # Calculate AUC directly from FPR and TPR
            current_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f'Class {class_label} (AUC = {current_auc:.2f})')
        except Exception as e:
            print(f"Warning: Could not load or plot data from {csv_file}: {e}")
            continue

    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ROC curves from CSV data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_outputs", # Default to the same output_dir as your training script
        help="The base directory where the 'roc' subdirectory containing CSV files is located.",
    )
    args = parser.parse_args()

    plot_roc_curves(args.output_dir)