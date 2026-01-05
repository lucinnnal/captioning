
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from urllib.parse import urlparse, parse_qs

def create_difficulty_map(base_dir):
    """Walks through the directory and creates a map of filename to difficulty."""
    difficulty_map = {}
    if not os.path.exists(base_dir):
        print(f"Warning: Directory '{base_dir}' not found.")
        return difficulty_map
        
    for difficulty in ['easy', 'medium', 'hard']:
        folder_path = os.path.join(base_dir, difficulty)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.mp4'):
                    difficulty_map[filename] = difficulty
    return difficulty_map

def generate_and_save_stats(difficulty, label_counts, output_dir):
    """Generates and saves statistics and plots for a given difficulty level."""
    print(f"\n{'='*20} Statistics for: {difficulty.upper()} {'='*20}")

    # Define categories
    fine_grained_categories = [
        "Ambiguous", "Object", "Text", "Number", "Color", "Action", 
        "Temporal", "Camera Panning", "Overthinking", "Position", "Hallucination"
    ]
    hallucination_categories = [
        "Object", "Text", "Number", "Color", "Action", "Temporal", 
        "Camera Panning", "Overthinking", "Position", "Hallucination"
    ]
    coarse_categories = ["Accurate", "Ambiguous", "Hallucination"]

    # --- Fine-grained statistics ---
    fine_grained_stats = {cat: label_counts.get(cat, 0) for cat in fine_grained_categories}
    fine_grained_df = pd.DataFrame(list(fine_grained_stats.items()), columns=['Category', 'Count']).sort_values(by='Count', ascending=False).reset_index(drop=True)

    # --- Coarse-grained statistics ---
    hallucination_total = sum(label_counts.get(cat, 0) for cat in hallucination_categories)
    coarse_grained_stats = {
        "Accurate": label_counts.get("Accurate", 0),
        "Ambiguous": label_counts.get("Ambiguous", 0),
        "Hallucination": hallucination_total
    }
    coarse_grained_df = pd.DataFrame(list(coarse_grained_stats.items()), columns=['Category', 'Count']).set_index('Category').reindex(coarse_categories).reset_index()

    # --- Print Tables ---
    print("\n--- Detailed Category Statistics (without Accurate) ---")
    print(fine_grained_df.to_string())
    print("\n--- Aggregated Statistics (Accurate, Ambiguous, Hallucination) ---")
    print(coarse_grained_df.to_string())

    # --- Visualization ---
    # Fine-grained plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Count', y='Category', data=fine_grained_df.sort_values(by='Category', ascending=True), ax=ax, palette='viridis', ci=None)
    ax.set_title(f'Detailed Counts for {difficulty.capitalize()}', fontsize=16)
    ax.set_xlabel('Count', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    plt.tight_layout()
    fine_grained_plot_path = os.path.join(output_dir, f'{difficulty}_fine_grained_stats.png')
    plt.savefig(fine_grained_plot_path)
    print(f"\nSaved detailed plot to: {fine_grained_plot_path}")

    # Coarse-grained plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Category', y='Count', data=coarse_grained_df, ax=ax, palette='mako', order=coarse_categories, ci=None)
    ax.set_title(f'Aggregated Stats for {difficulty.capitalize()}', fontsize=16)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    plt.tight_layout()
    coarse_grained_plot_path = os.path.join(output_dir, f'{difficulty}_coarse_grained_stats.png')
    plt.savefig(coarse_grained_plot_path)
    print(f"Saved aggregated plot to: {coarse_grained_plot_path}")

def plot_aggregated_charts(stats_by_difficulty, output_dir):
    """Creates and saves aggregated charts comparing difficulties."""
    print(f"\n{'='*20} Generating Aggregated Plots {'='*20}")
    
    fine_grained_categories = [
        "Ambiguous", "Object", "Text", "Number", "Color", "Action", 
        "Temporal", "Camera Panning", "Overthinking", "Position", "Hallucination"
    ]
    hallucination_categories = [
        "Object", "Text", "Number", "Color", "Action", "Temporal", 
        "Camera Panning", "Overthinking", "Position", "Hallucination"
    ]
    coarse_categories = ["Accurate", "Ambiguous", "Hallucination"]

    plot_data = []
    for difficulty, counts in stats_by_difficulty.items():
        if difficulty == "unknown":
            continue
        
        # Coarse data
        hallucination_total = sum(counts.get(cat, 0) for cat in hallucination_categories)
        plot_data.append({'Category': 'Accurate', 'Count': counts.get('Accurate', 0), 'Difficulty': difficulty})
        plot_data.append({'Category': 'Ambiguous', 'Count': counts.get('Ambiguous', 0), 'Difficulty': difficulty})
        plot_data.append({'Category': 'Hallucination', 'Count': hallucination_total, 'Difficulty': difficulty})

        # Fine-grained data
        for cat in fine_grained_categories:
            plot_data.append({'Category': cat, 'Count': counts.get(cat, 0), 'Difficulty': difficulty})

    df = pd.DataFrame(plot_data)

    # --- Aggregated Fine-Grained Plot ---
    df_fine = df[df['Category'].isin(fine_grained_categories)]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.barplot(data=df_fine, x='Category', y='Count', hue='Difficulty', ax=ax, palette='YlGnBu')
    ax.set_title('Aggregated Detailed Category Counts by Difficulty', fontsize=16)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    agg_fine_path = os.path.join(output_dir, 'aggregated_fine_grained.png')
    plt.savefig(agg_fine_path)
    print(f"\nSaved aggregated fine-grained plot to: {agg_fine_path}")

    # --- Aggregated Coarse-Grained Plot ---
    df_coarse = df[df['Category'].isin(coarse_categories)]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=df_coarse, x='Category', y='Count', hue='Difficulty', ax=ax, palette='YlOrRd')
    ax.set_title('Aggregated Coarse Counts by Difficulty', fontsize=16)
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()
    agg_coarse_path = os.path.join(output_dir, 'aggregated_coarse_grained.png')
    plt.savefig(agg_coarse_path)
    print(f"Saved aggregated coarse-grained plot to: {agg_coarse_path}")


def analyze_by_difficulty(annotation_file, video_base_dir):
    # Create the mapping from filename to difficulty
    difficulty_map = create_difficulty_map(video_base_dir)
    if not difficulty_map:
        print("Could not create a difficulty map. Aborting.")
        return

    # Load the annotation data
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{annotation_file}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{annotation_file}' is not a valid JSON file.")
        return

    # Aggregate labels by difficulty
    stats_by_difficulty = {
        "easy": Counter(),
        "medium": Counter(),
        "hard": Counter(),
        "unknown": Counter()
    }

    for item in data:
        video_url = item.get("video_url")
        if not video_url:
            continue
            
        # Extract filename from '/data/local-files/?d=/path/to/filename.mp4'
        try:
            parsed_url = urlparse(video_url)
            query_params = parse_qs(parsed_url.query)
            video_path = query_params.get('d', [None])[0]
            if not video_path:
                continue
            filename = os.path.basename(video_path)
        except Exception:
            # Fallback for simpler paths
            filename = os.path.basename(video_url)

        difficulty = difficulty_map.get(filename, "unknown")
        
        item_labels = []
        if "labels" in item and isinstance(item["labels"], list):
            for label_item in item["labels"]:
                if "labels" in label_item and isinstance(label_item["labels"], list):
                    item_labels.extend(label_item["labels"])
        
        stats_by_difficulty[difficulty].update(item_labels)

    # --- Generate reports for each difficulty ---
    output_dir = "analysis_by_difficulty"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for difficulty, label_counts in stats_by_difficulty.items():
        if sum(label_counts.values()) > 0: # Only process if there are labels
            generate_and_save_stats(difficulty, label_counts, output_dir)
            
    # --- Generate and save aggregated plots ---
    plot_aggregated_charts(stats_by_difficulty, output_dir)

if __name__ == "__main__":
    annotation_file = "internvl_annotation.json"
    video_base_dir = "/Users/kipyokim/학부연구/video_evaluation/captioning/ori-mini-sample"
    analyze_by_difficulty(annotation_file, video_base_dir)
