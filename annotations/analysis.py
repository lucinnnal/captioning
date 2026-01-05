
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

def analyze_annotations(file_path):
    # Define categories based on user requests
    # Request 3: Exclude 'Accurate'.
    # Request 4: This list is used to ensure all categories appear, even with a count of 0.
    fine_grained_categories = [
        "Object", "Text", "Number", "Color", "Action", 
        "Temporal", "Camera Panning", "Overthinking", "Position"
    ]
    hallucination_categories = [
        "Object", "Text", "Number", "Color", "Action", "Temporal", 
        "Camera Panning", "Overthinking", "Position"
    ]
    coarse_categories = ["Accurate", "Ambiguous", "Hallucination"]

    # Load the JSON data
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return

    # Extract all labels from the data
    all_labels = []
    if data:
        for item in data:
            if "labels" in item and isinstance(item["labels"], list):
                for label_item in item["labels"]:
                    if "labels" in label_item and isinstance(label_item["labels"], list):
                        all_labels.extend(label_item["labels"])

    # Count the occurrences of each label
    label_counts = Counter(all_labels)

    # --- Fine-grained statistics ---
    # Request 4: Use .get(cat, 0) to include categories with a count of 0.
    fine_grained_stats = {cat: label_counts.get(cat, 0) for cat in fine_grained_categories}

    # --- Coarse-grained statistics ---
    hallucination_total = sum(label_counts.get(cat, 0) for cat in hallucination_categories)
    coarse_grained_stats = {
        "Accurate": label_counts.get("Accurate", 0),
        "Ambiguous": label_counts.get("Ambiguous", 0),
        "Hallucination": hallucination_total
    }
    
    # --- Create DataFrames for display ---
    fine_grained_df = pd.DataFrame(list(fine_grained_stats.items()), columns=['Category', 'Count']).sort_values(by='Count', ascending=False).reset_index(drop=True)
    coarse_grained_df = pd.DataFrame(list(coarse_grained_stats.items()), columns=['Category', 'Count'])
    # Set coarse_grained_df Category as index to enforce order
    coarse_grained_df = coarse_grained_df.set_index('Category').reindex(coarse_categories).reset_index()


    print("--- Detailed Category Statistics (without Accurate) ---")
    print(fine_grained_df.to_string())
    print("\n" + "="*50 + "\n")
    print("--- Aggregated Statistics (Accurate, Ambiguous, Hallucination) ---")
    print(coarse_grained_df.to_string())

    # --- Visualization ---
    output_dir = "analysis_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot for fine-grained statistics
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Count', y='Category', data=fine_grained_df.sort_values(by='Category', ascending=True), ax=ax, palette='viridis')
    ax.set_title('Detailed Category Counts', fontsize=16)
    ax.set_xlabel('Count', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    plt.tight_layout()
    fine_grained_plot_path = os.path.join(output_dir, 'fine_grained_stats.png')
    plt.savefig(fine_grained_plot_path)
    print(f"\nSaved detailed statistics plot to: {fine_grained_plot_path}")

    # Plot for coarse-grained statistics
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Category', y='Count', data=coarse_grained_df, ax=ax, palette='mako', order=coarse_categories)
    ax.set_title('Coarse Statistics', fontsize=16)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    plt.tight_layout()
    coarse_grained_plot_path = os.path.join(output_dir, 'coarse_grained_stats.png')
    plt.savefig(coarse_grained_plot_path)
    print(f"Saved aggregated statistics plot to: {coarse_grained_plot_path}")


if __name__ == "__main__":
    # Corrected the file name based on user's first prompt, can be changed if needed.
    file_to_analyze = "internvl_annotation.json"
    analyze_annotations(file_to_analyze)