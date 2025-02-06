import os
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import entropy, skew, kurtosis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def get_gaze_coordinates(heatmap):
    """
    Convert a 64×64 heatmap to a single (x, y) coordinate based on the maximum probability.
    Returns (32, 32) if heatmap is None or sum is zero.
    """
    if heatmap is None:
        return (32, 32)
    h = np.array(heatmap)
    if h.sum() == 0:
        return (32, 32)
    y, x = np.unravel_index(np.argmax(h), h.shape)
    return (x, y)

def define_region(x, y):
    """
    Define region based on gaze coordinates.
    Returns 'left', 'center', or 'right'.
    """
    if x < 21:
        return 'left'
    elif x > 43:
        return 'right'
    else:
        return 'center'

def calculate_entropy_distribution(directions, bins=12):
    """
    Calculate entropy based on the distribution of movement directions.
    """
    if len(directions) == 0:
        return 0
    direction_hist, _ = np.histogram(directions, bins=np.linspace(-np.pi, np.pi, bins+1), density=True)
    direction_hist = direction_hist[direction_hist > 0]  # Remove zero entries to avoid log(0)
    return entropy(direction_hist)

def extract_features_from_group(group):
    """
    Extract features from a grouped DataFrame (per person per label).
    """
    features = {}
    if group.empty:
        # Handle empty groups
        return features

    # Spatial Features
    features['mean_x'] = group['gaze_x'].mean()
    features['mean_y'] = group['gaze_y'].mean()
    features['std_x'] = group['gaze_x'].std()
    features['std_y'] = group['gaze_y'].std()

    # Gaze Entropy
    all_heatmaps = np.stack(group['heatmap'].apply(np.array))
    flattened_heatmaps = all_heatmaps.reshape(all_heatmaps.shape[0], -1)
    entropy_values = [entropy(hm / hm.sum()) if hm.sum() != 0 else 0 for hm in flattened_heatmaps]
    features['gaze_entropy'] = np.mean(entropy_values)

    # Distance Metrics
    center_x, center_y = 32, 32
    distances = np.sqrt((group['gaze_x'] - center_x)**2 + (group['gaze_y'] - center_y)**2)
    features['max_distance'] = distances.max()
    features['avg_distance'] = distances.mean()

    # Movement Metrics
    group = group.copy()
    group['gaze_shift'] = np.sqrt(group['gaze_x'].diff()**2 + group['gaze_y'].diff()**2)
    features['avg_speed'] = group['gaze_shift'].mean()
    features['total_movement'] = group['gaze_shift'].sum()
    fixation_threshold = 2
    saccade_threshold = 5
    features['fixation_count'] = (group['gaze_shift'] < fixation_threshold).sum()
    features['saccade_count'] = (group['gaze_shift'] > saccade_threshold).sum()

    # Statistical Moments
    features['skew_x'] = skew(group['gaze_x'])
    features['skew_y'] = skew(group['gaze_y'])
    features['kurt_x'] = kurtosis(group['gaze_x'])
    features['kurt_y'] = kurtosis(group['gaze_y'])

    # Gaze Concentration
    unique_gazes = group[['gaze_x', 'gaze_y']].drop_duplicates()
    features['gaze_concentration'] = len(unique_gazes) / len(group)

    # Central Region Proportion
    central_region = ((group['gaze_x'] > 16) & (group['gaze_x'] < 48) &
                      (group['gaze_y'] > 16) & (group['gaze_y'] < 48))
    features['central_proportion'] = central_region.mean()

    # Gaze Transition Frequency
    group['region'] = group.apply(lambda row: define_region(row['gaze_x'], row['gaze_y']), axis=1)
    transitions = group['region'].shift(1) != group['region']
    features['transition_frequency'] = transitions.sum()

    # Clustering Metrics
    num_clusters = 3
    if len(group[['gaze_x', 'gaze_y']].dropna()) >= num_clusters:
        try:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            labels = kmeans.fit_predict(group[['gaze_x', 'gaze_y']])
            features['silhouette_score'] = silhouette_score(group[['gaze_x', 'gaze_y']], labels)
        except:
            features['silhouette_score'] = np.nan
    else:
        features['silhouette_score'] = np.nan

    # Gaze Stability Ratio
    stability_ratio = (group['gaze_shift'] < fixation_threshold).mean()
    features['stability_ratio'] = stability_ratio

    # Entropy of Gaze Shifts
    # Calculate movement directions
    group['direction'] = np.arctan2(group['gaze_y'].diff(), group['gaze_x'].diff())
    directions = group['direction'].dropna()
    features['direction_entropy'] = calculate_entropy_distribution(directions)

    # Frequency of Gaze Reversions
    # Count how often the gaze returns to the previous region
    group['previous_region'] = group['region'].shift(1)
    revisits = (group['region'] == group['previous_region']) & (group['region'].notna())
    total_transitions = transitions.sum()
    if total_transitions > 0:
        features['gaze_reversion_frequency'] = revisits.sum() / total_transitions
    else:
        features['gaze_reversion_frequency'] = 0

    return features

def extract_gaze_features(gaze_output_dir, output_csv='gaze_features.csv'):
    """
    Traverse the gaze_output_dir, extract features for each video, and save to a CSV.
    """
    records = []
    person_dirs = sorted([d for d in os.listdir(gaze_output_dir) if os.path.isdir(os.path.join(gaze_output_dir, d))])

    for person in tqdm(person_dirs, desc="Processing Persons"):
        person_path = os.path.join(gaze_output_dir, person)
        for label in ['scripted', 'spontaneous']:
            data_dir = os.path.join(person_path, label, 'data')
            if not os.path.exists(data_dir):
                print(f"Skipping {person} - {label}: 'data' directory does not exist.")
                continue

            frame_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith('.json')])
            if not frame_files:
                print(f"Skipping {person} - {label}: No JSON files found in 'data' directory.")
                continue

            gaze_data = []
            for frame_file in frame_files:
                frame_json_path = os.path.join(data_dir, frame_file)
                try:
                    with open(frame_json_path, 'r') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Error reading {frame_json_path}: {e}")
                    continue

                heatmap = data.get("heatmap", None)
                gaze_x, gaze_y = get_gaze_coordinates(heatmap)
                inout_score = data.get("inout_score", None)
                in_frame = data.get("in_frame", None)
                head_box = data.get("head_box", [None, None, None, None])

                gaze_data.append({
                    "gaze_x": gaze_x,
                    "gaze_y": gaze_y,
                    "heatmap": heatmap,
                    "inout_score": inout_score,
                    "in_frame": in_frame,
                    "head_box_xmin": head_box[0],
                    "head_box_ymin": head_box[1],
                    "head_box_xmax": head_box[2],
                    "head_box_ymax": head_box[3],
                })

            if not gaze_data:
                print(f"No valid gaze data for {person} - {label}.")
                continue

            df = pd.DataFrame(gaze_data)
            features = extract_features_from_group(df)

            if features:
                features['person_id'] = person
                features['label'] = label
                records.append(features)
            else:
                print(f"No features extracted for {person} - {label}.")

    # Create a DataFrame from the records
    features_df = pd.DataFrame(records)

    # Reorder columns for better readability
    cols = ['person_id', 'label'] + [col for col in features_df.columns if col not in ['person_id', 'label']]
    features_df = features_df[cols]

    # Save to CSV
    features_df.to_csv(output_csv, index=False)
    print(f"Feature extraction completed. Saved to {output_csv}.")

if __name__ == "__main__":
    # Define your gaze_output_dir path
    gaze_output_directory = r"C:\Users\muham\PycharmProjects\gazelle\gaze_output"  # Update as needed

    # Define output CSV path
    output_csv_path = r"C:\Users\muham\PycharmProjects\gazelle\gaze_output\gaze_features.csv"  # Update as needed

    extract_gaze_features(gaze_output_directory, output_csv=output_csv_path)
