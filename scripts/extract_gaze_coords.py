# scripts/extract_gaze_coords.py

import os
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path


def get_gaze_coordinates(heatmap):
    """
    Convert a 64×64 heatmap to a single (x, y) coordinate
    based on the maximum probability.

    Args:
        heatmap (list of list or np.ndarray): 2D [64,64] grid of probabilities.

    Returns:
        (int, int): (x, y) coordinate (column, row).
                    Returns (32, 32) if heatmap is None.
    """
    if heatmap is None:
        # Default center if missing
        return (32, 32)
    h = np.array(heatmap)
    y, x = np.unravel_index(np.argmax(h), h.shape)
    return (x, y)


def process_frame_json(frame_json_path):
    """
    Process a single JSON file that contains:
    {
      "heatmap": [... 64×64 array ...],
      "inout_score": float,
      "in_frame": bool,
      "head_box": [xmin, ymin, xmax, ymax]
    }
    and extract relevant info, including the gaze coordinate.
    """
    with open(frame_json_path, 'r') as f:
        data = json.load(f)

    # Extract fields
    heatmap = data.get("heatmap", None)
    inout_score = data.get("inout_score", None)
    in_frame = data.get("in_frame", None)
    head_box = data.get("head_box", [None, None, None, None])

    # Compute gaze coordinate (x, y) from the heatmap
    gaze_x, gaze_y = get_gaze_coordinates(heatmap)

    # Distance from the 64×64 center, optional
    center_x, center_y = 32, 32
    distance_from_center = math.sqrt((gaze_x - center_x) ** 2 + (gaze_y - center_y) ** 2)

    # Return dictionary of extracted data
    return {
        "gaze_x": gaze_x,
        "gaze_y": gaze_y,
        "inout_score": inout_score,
        "in_frame": in_frame,
        "head_box_xmin": head_box[0],
        "head_box_ymin": head_box[1],
        "head_box_xmax": head_box[2],
        "head_box_ymax": head_box[3],
        "distance_from_center": distance_from_center,
        "json_path": frame_json_path
    }


def extract_gaze_coordinates(gaze_output_dir, output_csv):
    """
    Traverse the gaze_output directory, find JSON files under each
    person/label/data subfolder, extract gaze coordinates, and
    save them to a CSV (one row per frame).
    """
    records = []

    # e.g. person_1, person_2, ...
    for person in os.listdir(gaze_output_dir):
        person_path = os.path.join(gaze_output_dir, person)
        if not os.path.isdir(person_path):
            continue

        # For each label: 'scripted', 'spontaneous'
        for label in ['scripted', 'spontaneous']:
            data_dir = os.path.join(person_path, label, 'data')
            if not os.path.isdir(data_dir):
                print(f"Skipping {person} - {label}: no data directory.")
                continue

            # JSON files in data_dir
            frame_files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith('.json'))
            if not frame_files:
                print(f"No JSON files in {data_dir}.")
                continue

            for frame_file in frame_files:
                frame_json_path = os.path.join(data_dir, frame_file)
                # Extract info
                info = process_frame_json(frame_json_path)
                # Add metadata
                info["person_id"] = person
                info["label"] = label
                info["frame_file"] = frame_file
                records.append(info)
                print(f"Processed {person}/{label}/{frame_file}")

    # Convert all records to a DataFrame & save CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Extraction completed! CSV saved to: {output_csv}")


if __name__ == "__main__":
    # Adjust these paths as needed
    gaze_output_directory = r"C:\Users\muham\PycharmProjects\gazelle\gaze_output"
    output_csv_path = r"C:\Users\muham\PycharmProjects\gazelle\gaze_output\gaze_coordinates.csv"

    extract_gaze_coordinates(
        gaze_output_dir=gaze_output_directory,
        output_csv=output_csv_path
    )
