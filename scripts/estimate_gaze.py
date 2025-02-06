# scripts/estimate_gaze.py
import os
import json
import logging
from pathlib import Path

import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from gazelle.model import get_gazelle_model

# Configure logging
logging.basicConfig(
    filename='gaze_estimation.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

#############################################
# 1. Load Gaze-LLE Model
#############################################
model_name = "gazelle_dinov2_vitl14_inout"  # Update as needed
checkpoint_path = r"C:\Users\muham\PycharmProjects\gazelle\model\gazelle_dinov2_vitl14_inout.pt"  # Update path

try:
    model, transform = get_gazelle_model(model_name)
except Exception as e:
    logging.critical(f"Error loading Gaze-LLE model: {e}")
    raise e

if not os.path.exists(checkpoint_path):
    logging.critical(f"Checkpoint not found at {checkpoint_path}.")
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

try:
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_gazelle_state_dict(state_dict)
except Exception as e:
    logging.critical(f"Error loading state dict: {e}")
    raise e

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logging.info(f"Model loaded and moved to {device}.")


#############################################
# 2. Visualization Helper
#############################################
def visualize_heatmap(image_path, heatmap, bbox=None, inout_score=None, save_path=None):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Error opening image for visualization {image_path}: {e}")
        return

    width, height = image.size
    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    if heatmap is not None:
        heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        heatmap_resized = np.array(heatmap_resized).astype(float) / 255.0
        plt.imshow(heatmap_resized, cmap='jet', alpha=0.5, extent=(0, width, height, 0))

    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        rect = plt.Rectangle(
            (xmin * width, ymin * height),
            (xmax - xmin) * width,
            (ymax - ymin) * height,
            linewidth=2,
            edgecolor='lime',
            facecolor='none'
        )
        plt.gca().add_patch(rect)

    if inout_score is not None:
        plt.text(
            10, 10,
            f"In-Frame: {inout_score:.2f}",
            color='white',
            bbox=dict(facecolor='black', alpha=0.5)
        )

    plt.axis('off')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            logging.info(f"Saved visualization to {save_path}")
        except Exception as e:
            logging.error(f"Error saving visualization {save_path}: {e}")
    plt.close()


#############################################
# 3. Gaze Inference Utilities
#############################################
def prepare_input(image_path, head_box):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Error opening image {image_path}: {e}")
        return None

    input_image = transform(image).unsqueeze(0).to(device)
    img_width, img_height = image.size
    xmin, ymin, xmax, ymax = head_box
    normalized_bbox = [(xmin / img_width, ymin / img_height, xmax / img_width, ymax / img_height)]

    input_dict = {
        "images": input_image,
        "bboxes": [normalized_bbox]
    }
    return input_dict


def run_gaze_estimation(image_path, head_box, sigma=1.0):
    input_data = prepare_input(image_path, head_box)
    if input_data is None:
        return None, None
    with torch.no_grad():
        try:
            output = model(input_data)
        except Exception as e:
            logging.error(f"Model inference error on {image_path}: {e}")
            return None, None

    heatmap = output["heatmap"][0][0].cpu().numpy() if "heatmap" in output else None
    inout_score = output["inout"][0][0].cpu().numpy() if "inout" in output else None

    if heatmap is not None and sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)
    return heatmap, float(inout_score) if inout_score is not None else None


def smooth_inout_scores(inout_scores, sigma=1):
    try:
        smoothed = gaussian_filter1d(np.array(inout_scores), sigma=sigma)
        return smoothed.tolist()
    except Exception as e:
        logging.error(f"Error smoothing inout scores: {e}")
        return inout_scores


#############################################
# 4. Processing Function
#############################################
def process_gaze_estimation(processed_dir, gaze_dir, threshold=0.5, sigma=1.0):
    os.makedirs(gaze_dir, exist_ok=True)

    person_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    if not person_dirs:
        logging.warning("No person directories found.")
        return

    for person in tqdm(person_dirs, desc="Processing Persons"):
        person_path = os.path.join(processed_dir, person)
        for label in ["scripted", "spontaneous"]:
            frames_dir = os.path.join(person_path, label, "frames")
            head_boxes_dir = os.path.join(person_path, label, "head_boxes")
            if not os.path.exists(frames_dir) or not os.path.exists(head_boxes_dir):
                logging.warning(f"Skipping {person} - {label}: Missing frames or head boxes.")
                continue

            logging.info(f"Processing {person} - {label}")
            # Create separate directories for heatmaps and data
            heatmap_dir = os.path.join(gaze_dir, person, label, "heatmaps")
            data_dir = os.path.join(gaze_dir, person, label, "data")
            os.makedirs(heatmap_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)

            frame_files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.png')))
            if not frame_files:
                logging.warning(f"No frames found for {person} - {label}")
                continue

            for frame_file in tqdm(frame_files, desc=f"{person}-{label}", leave=False):
                frame_path = os.path.join(frames_dir, frame_file)
                frame_stem = Path(frame_file).stem
                head_box_path = os.path.join(head_boxes_dir, f"{frame_stem}.json")

                if not os.path.exists(head_box_path):
                    logging.warning(f"No head box for frame {frame_file}")
                    continue

                try:
                    with open(head_box_path, "r") as f:
                        head_boxes = json.load(f)
                except Exception as e:
                    logging.error(f"Error reading {head_box_path}: {e}")
                    continue

                if not head_boxes:
                    logging.warning(f"No bounding boxes in {head_box_path}")
                    continue

                main_head_box = head_boxes[0]
                heatmap, inout_score = run_gaze_estimation(frame_path, main_head_box, sigma=sigma)
                if heatmap is None or inout_score is None:
                    continue

                in_frame = (inout_score >= threshold)

                # Prepare JSON data for this frame
                json_data = {
                    "heatmap": heatmap.tolist(),
                    "inout_score": inout_score,
                    "in_frame": in_frame,
                    "head_box": main_head_box
                }
                json_out_path = os.path.join(data_dir, f"{frame_stem}_gaze.json")
                try:
                    with open(json_out_path, "w") as jf:
                        json.dump(json_data, jf, indent=2)
                except Exception as e:
                    logging.error(f"Error saving JSON {json_out_path}: {e}")

                # Save heatmap visualization
                heatmap_out_path = os.path.join(heatmap_dir, f"{frame_stem}_heatmap.png")

                # Compute normalized bounding box correctly
                width, height = Image.open(frame_path).size
                xmin, ymin, xmax, ymax = main_head_box
                normalized_bbox = [xmin / width, ymin / height, xmax / width, ymax / height]

                visualize_heatmap(
                    image_path=frame_path,
                    heatmap=heatmap,
                    bbox=normalized_bbox,
                    inout_score=inout_score,
                    save_path=heatmap_out_path
                )

            logging.info(f"Completed {person} - {label}")


#############################################
# 5. Main Entry Point
#############################################
def main():
    processed_directory = r"C:\Users\muham\PycharmProjects\gazelle\processed"  # Update as needed
    gaze_output_directory = r"C:\Users\muham\PycharmProjects\gazelle\gaze_output"  # Update as needed

    process_gaze_estimation(
        processed_dir=processed_directory,
        gaze_dir=gaze_output_directory,
        threshold=0.5,  # Adjust as desired
        sigma=1.0  # Adjust as desired
    )


if __name__ == "__main__":
    main()
