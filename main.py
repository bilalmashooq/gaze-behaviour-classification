import sys
import os
import cv2
import shutil
import csv
import re
import json
import logging
from pathlib import Path
import gdown
import requests
import time
from PyQt5.QtWidgets import QInputDialog
import joblib
from sklearn.exceptions import NotFittedError

import mediapipe as mp
import numpy as np
import torch
import pandas as pd
from scipy.stats import entropy, skew, kurtosis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from tqdm import tqdm

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QSizePolicy,
    QProgressDialog
)

# -----------------------------
# Import Your UI and Resources
# -----------------------------
from gui1 import Ui_MainWindow    # Make sure gui.py is generated from .ui
from qt_resource_rc import *     # Resource file (icons, images, etc.)

# -----------------------------
# Import Your Gaze Model Loader
# -----------------------------
from gazelle.model import get_gazelle_model  # Ensure this is correct


# -----------------------------
# Logging Setup
# -----------------------------
def setup_logging(log_file='app.log'):
    """
    Sets up logging configuration to both file and console.
    """
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# Initialize logging
setup_logging()


# -----------------------------
# Blink Detection Functions
# -----------------------------
def eye_aspect_ratio(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR) for a given eye.
    eye_landmarks: Nx2 array of (x, y) eye landmark coords.
    """
    try:
        vertical1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        return (vertical1 + vertical2) / (2.0 * horizontal)
    except Exception as e:
        logging.error(f"Error calculating EAR: {e}")
        return None

def detect_blinks(video_path, ear_threshold=0.18, consecutive_frames=2):
    """
    Detect blinks in a video and compute (blink_count, blink_rate).
    """
    logging.info(f"Starting blink detection on: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        return 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps
    duration_minutes = duration_seconds / 60

    logging.info(f"Video FPS: {fps}, Frames: {total_frames}, Duration(min): {duration_minutes}")

    blink_count = 0
    consecutive_low_ear = 0
    in_blink = False

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if frame_number % 100 == 0:
                logging.info(f"Processing frame {frame_number}/{total_frames}")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                try:
                    # We'll assume just the first face
                    landmarks = results.multi_face_landmarks[0].landmark
                    h, w, _ = frame.shape

                    left_eye = np.array([
                        (landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE_INDICES
                    ])
                    right_eye = np.array([
                        (landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE_INDICES
                    ])

                    ear_left = eye_aspect_ratio(left_eye)
                    ear_right = eye_aspect_ratio(right_eye)
                    if ear_left is not None and ear_right is not None:
                        ear = (ear_left + ear_right) / 2.0
                        # Blink logic
                        if ear < ear_threshold:
                            consecutive_low_ear += 1
                            if not in_blink and consecutive_low_ear >= consecutive_frames:
                                blink_count += 1
                                in_blink = True
                                logging.info(f"Blink detected at frame={frame_number}")
                        else:
                            consecutive_low_ear = 0
                            in_blink = False

                except Exception as e:
                    logging.error(f"Error in blink detection at frame {frame_number}: {e}")
                    continue

    cap.release()
    blink_rate = blink_count / duration_minutes if duration_minutes > 0 else 0
    logging.info(f"Blinks={blink_count}, BlinkRate={blink_rate} bpm")
    return blink_count, round(blink_rate, 2)


# -----------------------------
# Gaze Estimation Helper Functions
# -----------------------------
def prepare_input(image_path, head_box, transform, device, model):
    """
    Prepare input dict for Gaze-LLE model from an image + head box.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Error opening image {image_path}: {e}")
        return None

    input_img = transform(image).unsqueeze(0).to(device)
    img_w, img_h = image.size
    x1, y1, x2, y2 = head_box

    normalized_bbox = [
        (
            x1 / img_w,
            y1 / img_h,
            x2 / img_w,
            y2 / img_h
        )
    ]
    input_dict = {
        "images": input_img,
        "bboxes": [normalized_bbox]
    }
    return input_dict

def visualize_heatmap(image_path, heatmap, bbox=None, inout_score=None, save_path=None):
    """
    Visualize a heatmap over the original image and optionally draw bounding box + inout score.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Error opening image for heatmap {image_path}: {e}")
        return

    w, h = image.size
    plt.figure(figsize=(8, 8))
    plt.imshow(image)

    if heatmap is not None:
        # Convert float array [0..1] to [0..255] for resizing with PIL
        scaled_heatmap = (heatmap * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(scaled_heatmap).resize(
            (w, h), resample=Image.Resampling.BILINEAR
        )
        final_hmap = np.array(heatmap_img).astype(float) / 255.0
        plt.imshow(final_hmap, cmap='jet', alpha=0.5, extent=(0, w, h, 0))

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle(
            (x1, y1),
            (x2 - x1),
            (y2 - y1),
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
            logging.info(f"Saved heatmap visualization to {save_path}")
        except Exception as e:
            logging.error(f"Error saving heatmap to {save_path}: {e}")
    plt.close()

def compute_gaze_point(heatmap):
    """
    Compute a (x, y) gaze point from a 2D heatmap using weighted averaging.
    """
    if heatmap is None:
        return None
    try:
        arr = np.array(heatmap)
        total = arr.sum()
        if total == 0:
            return None
        y_indices, x_indices = np.indices(arr.shape)
        x = (x_indices * arr).sum() / total
        y = (y_indices * arr).sum() / total
        return (x, y)
    except Exception as e:
        logging.error(f"Error computing gaze point: {e}")
        return None
# -----------------------------
# Gaze Feature Extraction Helpers
# -----------------------------
def get_gaze_coordinates(heatmap):
    """
    Convert a 64×64 heatmap to a single (x, y) coordinate
    based on the maximum probability.
    Returns (32, 32) if heatmap is None or sum is zero.
    """
    if heatmap is None:
        return (32, 32)
    arr = np.array(heatmap)
    if arr.sum() == 0:
        return (32, 32)
    y, x = np.unravel_index(np.argmax(arr), arr.shape)
    return (x, y)

def define_region(x, y):
    """Define region based on (x, y) in a 64×64 map."""
    if x < 21:
        return 'left'
    elif x > 43:
        return 'right'
    else:
        return 'center'

def calculate_entropy_distribution(directions, bins=12):
    """
    Calculate entropy based on distribution of movement directions.
    directions are array-like angles in [-π, π].
    """
    if len(directions) == 0:
        return 0
    # Create histogram from -π to π
    direction_hist, _ = np.histogram(
        directions, bins=np.linspace(-np.pi, np.pi, bins+1), density=True
    )
    direction_hist = direction_hist[direction_hist > 0]  # Remove zeros to avoid log(0)
    return entropy(direction_hist)

def extract_features_from_group(df):
    """
    Given a DataFrame `df` that has columns:
      - gaze_x, gaze_y, heatmap, inout_score, in_frame, ...
    this function computes a dictionary of features.
    """
    features = {}
    if df.empty:
        return features

    # Spatial features
    features['mean_x'] = df['gaze_x'].mean()
    features['mean_y'] = df['gaze_y'].mean()
    features['std_x'] = df['gaze_x'].std()
    features['std_y'] = df['gaze_y'].std()

    # Gaze Entropy
    # shape: (#frames, 64, 64)
    all_heatmaps = np.stack(df['heatmap'].apply(np.array))
    flattened_heatmaps = all_heatmaps.reshape(all_heatmaps.shape[0], -1)
    entropies = []
    for hm in flattened_heatmaps:
        total = hm.sum()
        if total > 0:
            entropies.append(entropy(hm / total))
        else:
            entropies.append(0.0)
    features['gaze_entropy'] = np.mean(entropies)

    # Distance from center
    center_x, center_y = 32, 32
    distances = np.sqrt((df['gaze_x'] - center_x) ** 2 + (df['gaze_y'] - center_y) ** 2)
    features['max_distance'] = distances.max()
    features['avg_distance'] = distances.mean()

    # Movement metrics
    df = df.copy()
    df['gaze_shift'] = np.sqrt(df['gaze_x'].diff()**2 + df['gaze_y'].diff()**2)
    features['avg_speed'] = df['gaze_shift'].mean()
    features['total_movement'] = df['gaze_shift'].sum()
    fixation_threshold = 2
    saccade_threshold = 5
    features['fixation_count'] = (df['gaze_shift'] < fixation_threshold).sum()
    features['saccade_count'] = (df['gaze_shift'] > saccade_threshold).sum()

    # Statistical moments (skew, kurtosis)
    features['skew_x'] = skew(df['gaze_x'].dropna())
    features['skew_y'] = skew(df['gaze_y'].dropna())
    features['kurt_x'] = kurtosis(df['gaze_x'].dropna())
    features['kurt_y'] = kurtosis(df['gaze_y'].dropna())

    # Gaze concentration (unique gaze coords)
    unique_gazes = df[['gaze_x', 'gaze_y']].drop_duplicates()
    features['gaze_concentration'] = len(unique_gazes) / len(df)

    # Central region proportion
    central_region = (
        (df['gaze_x'] > 16) & (df['gaze_x'] < 48) &
        (df['gaze_y'] > 16) & (df['gaze_y'] < 48)
    )
    features['central_proportion'] = central_region.mean()

    # Gaze transitions
    df['region'] = df.apply(lambda row: define_region(row['gaze_x'], row['gaze_y']), axis=1)
    transitions = df['region'].shift(1) != df['region']
    features['transition_frequency'] = transitions.sum()

    # Clustering metrics
    coords = df[['gaze_x', 'gaze_y']].dropna()
    num_clusters = 3
    if len(coords) >= num_clusters:
        try:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            labels = kmeans.fit_predict(coords)
            features['silhouette_score'] = silhouette_score(coords, labels)
        except:
            features['silhouette_score'] = np.nan
    else:
        features['silhouette_score'] = np.nan

    # Gaze stability ratio
    stability_ratio = (df['gaze_shift'] < fixation_threshold).mean()
    features['stability_ratio'] = stability_ratio

    # Direction entropy
    df['direction'] = np.arctan2(df['gaze_y'].diff(), df['gaze_x'].diff())
    directions = df['direction'].dropna()
    features['direction_entropy'] = calculate_entropy_distribution(directions)

    # Gaze reversion frequency
    df['previous_region'] = df['region'].shift(1)
    revisits = (df['region'] == df['previous_region']) & (df['region'].notna())
    total_transitions = transitions.sum()
    if total_transitions > 0:
        features['gaze_reversion_frequency'] = revisits.sum() / total_transitions
    else:
        features['gaze_reversion_frequency'] = 0

    return features

# -----------------------------
# Prediction Helpers
# -----------------------------
def load_model_and_encoder(self, model_path, label_encoder_path):
    """Load the trained classifier pipeline and label encoder."""
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(label_encoder_path):
        logging.error(f"Label encoder file not found at: {label_encoder_path}")
        raise FileNotFoundError(f"Label encoder file not found at: {label_encoder_path}")

    try:
        pipeline = joblib.load(model_path)
        le = joblib.load(label_encoder_path)
        logging.info("Successfully loaded the model pipeline and label encoder.")
        return pipeline, le
    except Exception as e:
        logging.error(f"Error loading model or label encoder: {e}")
        raise e


def load_new_data(self, gaze_features_path, blink_results_path):
    """Load and merge gaze_features and blink_results datasets."""
    if not os.path.exists(gaze_features_path):
        logging.error(f"Gaze features file not found at: {gaze_features_path}")
        raise FileNotFoundError(f"Gaze features file not found at: {gaze_features_path}")
    if not os.path.exists(blink_results_path):
        logging.error(f"Blink results file not found at: {blink_results_path}")
        raise FileNotFoundError(f"Blink results file not found at: {blink_results_path}")

    gaze_features = pd.read_csv(gaze_features_path)
    blink_results = pd.read_csv(blink_results_path)

    # Ensure 'person_id' exists in both DataFrames
    if 'person_id' not in gaze_features.columns or 'person_id' not in blink_results.columns:
        logging.error("Both gaze_features and blink_results must contain 'person_id' column for merging.")
        raise KeyError("Missing 'person_id' column in one of the input files.")

    # Drop 'label' from blink_results if present, since it's unknown during prediction
    if 'label' in blink_results.columns:
        blink_results = blink_results.drop('label', axis=1)

    merged_data = pd.merge(gaze_features, blink_results, on=['person_id'], how='inner')

    if merged_data.empty:
        logging.error("Merged data is empty. Check if 'person_id's match between the two files.")
        raise ValueError("No matching 'person_id's found between gaze_features and blink_results.")

    logging.info("Successfully loaded and merged the new data.")
    return merged_data


def define_features_for_prediction(self, merged_data):
    """Define feature matrix X for prediction."""
    # Drop columns that are not features. During training, 'person_id', 'label', and 'label_encoded' were dropped.
    # Since 'label' is not available, ensure these columns are excluded if present.
    columns_to_drop = ['person_id']
    for col in columns_to_drop:
        if col in merged_data.columns:
            merged_data = merged_data.drop(col, axis=1)

    X = merged_data.copy()
    return X


def make_prediction(self, pipeline, le, X):
    """Make prediction using the pipeline and label encoder."""
    try:
        # Predict the label
        y_pred = pipeline.predict(X)

        # Predict the probability
        y_prob = pipeline.predict_proba(X)[:, 1]  # Probability for the positive class

        # Decode the label
        y_pred_label = le.inverse_transform(y_pred)

        # Create a result DataFrame
        results = pd.DataFrame({
            'predicted_label': y_pred_label,
            'confidence_score': y_prob
        })

        return results
    except NotFittedError as e:
        logging.error(f"Model is not fitted: {e}")
        raise e
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise e


def save_results(self, results, output_path):
    """Save the prediction results to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        results.to_csv(output_path, index=False)
        logging.info(f"Prediction results saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error saving prediction results: {e}")
        raise e

# -----------------------------
# YOLOv5 Head Detection Thread
# -----------------------------
class HeadDetectionThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, frames_dir, detecthead_dir, model_path, conf_threshold=0.4):
        super().__init__()
        self.frames_dir = frames_dir
        self.detecthead_dir = detecthead_dir
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._is_running = True

    def run(self):
        try:
            self.model = torch.hub.load(
                'ultralytics/yolov5', 'custom',
                path=self.model_path,
                source='local'
            )
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"YOLOv5 loaded from {self.model_path} on {self.device}")

            frame_files = sorted([
                f for f in os.listdir(self.frames_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            total = len(frame_files)
            if total == 0:
                msg = "No frames found for head detection."
                logging.warning(msg)
                self.finished_signal.emit(False, msg)
                return

            for idx, frame_file in enumerate(frame_files):
                if not self._is_running:
                    msg = "Head detection canceled by user."
                    logging.info(msg)
                    self.finished_signal.emit(False, msg)
                    return

                frame_path = os.path.join(self.frames_dir, frame_file)
                head_boxes = self.detect_heads_yolov5(frame_path)

                stem = Path(frame_file).stem
                json_path = os.path.join(self.detecthead_dir, f"{stem}.json")
                try:
                    with open(json_path, 'w') as f:
                        json.dump(head_boxes, f)
                except Exception as e:
                    logging.error(f"Failed writing head boxes {json_path}: {e}")

                progress = int(((idx + 1) / total) * 100)
                self.progress_signal.emit(progress)

            msg = "Head detection completed successfully."
            logging.info(msg)
            self.finished_signal.emit(True, msg)
        except Exception as e:
            msg = f"Head detection failed: {e}"
            logging.error(msg)
            self.finished_signal.emit(False, msg)

    def detect_heads_yolov5(self, frame_path):
        try:
            image = Image.open(frame_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error opening {frame_path}: {e}")
            return []

        try:
            results = self.model(image, size=640)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logging.error("Out of Memory error in YOLOv5 inference.")
                torch.cuda.empty_cache()
            else:
                logging.error(f"Runtime error: {e}")
            return []
        except Exception as e:
            logging.error(f"Inference error: {e}")
            return []

        detections = results.xyxy[0]
        boxes = []
        for *box, conf, cls_idx in detections:
            if int(cls_idx.item()) == 0 and conf.item() >= self.conf_threshold:
                x1, y1, x2, y2 = box
                boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
        return boxes

    def stop(self):
        self._is_running = False
        self.terminate()


# -----------------------------
# Gaze Estimation Thread
# -----------------------------
class GazeEstimationThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, processed_dir, gaze_dir, model_path, threshold=0.5, sigma=1.0):
        super().__init__()
        self.processed_dir = processed_dir
        self.gaze_dir = gaze_dir
        self.model_path = model_path
        self.threshold = threshold
        self.sigma = sigma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self._is_running = True

    def run(self):
        try:
            model_name = "gazelle_dinov2_vitl14_inout"
            if not os.path.exists(self.model_path):
                msg = f"Checkpoint not found: {self.model_path}"
                logging.critical(msg)
                self.finished_signal.emit(False, msg)
                return

            self.model, self.transform = get_gazelle_model(model_name)
            state_dict = torch.load(self.model_path, map_location=self.device)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.model.load_gazelle_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)
            logging.info(f"Gaze-LLE loaded from {self.model_path} on {self.device}.")

            os.makedirs(self.gaze_dir, exist_ok=True)
            heatmap_dir = os.path.join(self.gaze_dir, "heatmaps")
            data_dir = os.path.join(self.gaze_dir, "data")
            os.makedirs(heatmap_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)

            frames_dir = os.path.join(self.processed_dir, "frames")
            head_dir = os.path.join(self.processed_dir, "detecthead")
            if not os.path.exists(frames_dir) or not os.path.exists(head_dir):
                msg = "Missing frames or detecthead directories."
                logging.warning(msg)
                self.finished_signal.emit(False, msg)
                return

            frame_files = sorted([
                f for f in os.listdir(frames_dir)
                if f.lower().endswith(('.jpg', '.png'))
            ])
            total = len(frame_files)
            if total == 0:
                msg = "No frames found for gaze estimation."
                logging.warning(msg)
                self.finished_signal.emit(False, msg)
                return

            summary_data = {"gaze_estimations": []}

            for idx, frame_file in enumerate(frame_files):
                if not self._is_running:
                    msg = "Gaze estimation canceled by user."
                    logging.info(msg)
                    self.finished_signal.emit(False, msg)
                    return

                frame_path = os.path.join(frames_dir, frame_file)
                stem = Path(frame_file).stem
                head_box_path = os.path.join(head_dir, f"{stem}.json")

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

                main_box = head_boxes[0]
                input_data = prepare_input(frame_path, main_box, self.transform, self.device, self.model)
                if input_data is None:
                    continue

                with torch.no_grad():
                    try:
                        output = self.model(input_data)
                    except Exception as e:
                        logging.error(f"Inference error on {frame_path}: {e}")
                        continue

                heatmap = output.get("heatmap", [None])[0]
                inout = output.get("inout", [None])[0]

                if heatmap is not None:
                    heatmap = heatmap[0].cpu().numpy()
                inout_score = None
                if inout is not None:
                    inout_score = float(inout[0].cpu().numpy())

                if heatmap is not None and self.sigma > 0:
                    heatmap = gaussian_filter(heatmap, sigma=self.sigma)
                in_frame = (inout_score >= self.threshold) if inout_score is not None else False

                # Save JSON
                json_data = {
                    "heatmap": heatmap.tolist() if heatmap is not None else None,
                    "inout_score": inout_score,
                    "in_frame": in_frame,
                    "head_box": main_box
                }
                data_out = os.path.join(data_dir, f"{stem}_gaze.json")
                try:
                    with open(data_out, "w") as jf:
                        json.dump(json_data, jf, indent=2)
                    logging.info(f"Saved gaze data to {data_out}")
                except Exception as e:
                    logging.error(f"Error saving gaze data {data_out}: {e}")

                # Save heatmap image
                heatmap_out = os.path.join(heatmap_dir, f"{stem}_heatmap.png")
                visualize_heatmap(
                    image_path=frame_path,
                    heatmap=heatmap,
                    bbox=main_box,
                    inout_score=inout_score,
                    save_path=heatmap_out
                )

                if heatmap is not None and inout_score is not None:
                    gp = compute_gaze_point(heatmap)
                    summary_data["gaze_estimations"].append({
                        "frame": frame_file,
                        "gaze_point": gp,
                        "inout_score": inout_score
                    })

                progress = int(((idx + 1) / total) * 100)
                self.progress_signal.emit(progress)

            summary_path = os.path.join(self.gaze_dir, "gaze_summary.json")
            try:
                with open(summary_path, "w") as sf:
                    json.dump(summary_data, sf, indent=2)
                logging.info(f"Saved gaze summary to {summary_path}")
            except Exception as e:
                logging.error(f"Error saving gaze summary: {e}")

            msg = "Gaze estimation completed successfully."
            logging.info(msg)
            self.finished_signal.emit(True, msg)
        except Exception as e:
            msg = f"Gaze estimation failed: {e}"
            logging.error(msg)
            self.finished_signal.emit(False, msg)

    def stop(self):
        self._is_running = False
        self.terminate()


# -----------------------------
# Main Window Class
# -----------------------------
class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.current_theme = "light"

        # Button connections
        self.tabWidget.setCurrentIndex(0)
        self.pushButton.clicked.connect(self.analysis)
        self.pushButton_2.clicked.connect(self.results)
        self.pushButton_13.clicked.connect(self.quit)
        self.pushButton_14.clicked.connect(self.theme)

        # Analysis tab
        self.pushButton_3.clicked.connect(self.uploadVideo)
        self.pushButton_4.clicked.connect(self.uploadViaLink)
        self.pushButton_5.clicked.connect(self.blinkDetection)
        self.pushButton_6.clicked.connect(self.estimateGaze)
        self.pushButton_7.clicked.connect(self.extractFeatures)
        self.pushButton_8.clicked.connect(self.prediction)
        self.pushButton_9.clicked.connect(self.save)
        self.pushButton_10.clicked.connect(self.showResults)

        # Video display
        self.videoLabel = QLabel()
        self.videoLabel.setAlignment(Qt.AlignCenter)
        size_policy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.videoLabel.setSizePolicy(size_policy)

        self.frame_5_layout = QVBoxLayout(self.frame_5)
        self.frame_5_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_5_layout.setSpacing(0)
        self.frame_5.setLayout(self.frame_5_layout)
        self.frame_5_layout.addWidget(self.videoLabel)

        self.video_capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_frame)

        self.head_detection_thread = None
        self.gaze_estimation_thread = None

        self.yolov5_model_path = "model/yolov5n.pt"
        self.gaze_model_path = "model/gazelle_dinov2_vitl14_inout.pt"

        if not os.path.exists(self.yolov5_model_path):
            logging.error(f"YOLOv5 not found at {self.yolov5_model_path}")
            QMessageBox.critical(self, "Model Not Found",
                                 f"YOLOv5 not found: {self.yolov5_model_path}")
            sys.exit(1)

        if not os.path.exists(self.gaze_model_path):
            logging.error(f"Gaze model not found at {self.gaze_model_path}")
            QMessageBox.critical(self, "Model Not Found",
                                 f"Gaze model not found: {self.gaze_model_path}")
            sys.exit(1)

        self.current_video_path = None

    # Tab switching
    def analysis(self):
        self.tabWidget.setCurrentIndex(1)

    def results(self):
        self.tabWidget.setCurrentIndex(2)

    # Theme
    def theme(self):
        if self.current_theme == "light":
            self.applyTheme("dark")
            self.current_theme = "dark"
        else:
            self.applyTheme("light")
            self.current_theme = "light"
        logging.info(f"Theme changed to {self.current_theme}")

    def applyTheme(self, theme_name):
        theme_path = f"styles/{theme_name}.qss"
        try:
            with open(theme_path, "r") as f:
                style_sheet = f.read()
                self.setStyleSheet(style_sheet)
        except FileNotFoundError:
            self.showErrorMessage(f"Theme file not found: {theme_path}")

    # Video
    def uploadVideo(self):
        self.stop_video_playback()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_path:
            db_folder = "database"
            if not os.path.exists(db_folder):
                os.makedirs(db_folder)

            file_name = os.path.basename(file_path)
            vid_name = os.path.splitext(file_name)[0]
            vid_folder = os.path.join(db_folder, vid_name)
            if not os.path.exists(vid_folder):
                os.makedirs(vid_folder)

            save_path = os.path.join(vid_folder, file_name)
            try:
                shutil.copy(file_path, save_path)
                logging.info(f"Video copied to: {save_path}")
            except Exception as e:
                self.showErrorMessage(f"Failed to copy video: {e}")
                return

            self.load_video(save_path)
            self.current_video_path = save_path
        else:
            logging.info("No video selected.")

    def load_video(self, file_path):
        self.stop_video_playback()
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        self.video_capture = cv2.VideoCapture(file_path)
        if not self.video_capture.isOpened():
            self.showErrorMessage("Could not open video file.")
            return

        self.timer.start(30)
        self.videoLabel.setText("Playing Video...")
        self.videoLabel.show()
        logging.info(f"Playing video: {file_path}")

    def update_video_frame(self):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_image.data, w, h,
                                bytes_per_line, QImage.Format_RGB888)
                scaled_img = qt_img.scaled(
                    self.videoLabel.width(),
                    self.videoLabel.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.videoLabel.setPixmap(QPixmap.fromImage(scaled_img))
            else:
                self.timer.stop()
                self.video_capture.release()
                self.video_capture = None
                self.videoLabel.clear()
                self.videoLabel.setText("Video Ended.")
                logging.info("Video playback ended.")

    def stop_video_playback(self):
        if self.video_capture is not None:
            self.timer.stop()
            self.video_capture.release()
            self.video_capture = None
            self.videoLabel.clear()
            self.videoLabel.setText("No video playing.")
            logging.info("Video playback stopped.")

    # Blink Detection
    def blinkDetection(self):
        if self.current_video_path and os.path.exists(self.current_video_path):
            try:
                count, rate = detect_blinks(self.current_video_path)
                logging.info(f"Blinks Detected: {count}, Rate: {rate} bpm")
                self.save_blink_results(self.current_video_path, count, rate)
            except Exception as e:
                self.showErrorMessage(f"Blink detection failed: {e}")
        else:
            QMessageBox.information(
                self, "Blink Detection",
                "No valid video available for blink detection.",
                QMessageBox.Ok
            )
            logging.info("No valid video for blink detection.")

    # Gaze Estimation
    def estimateGaze(self):
        logging.info("Estimate Gaze initiated.")
        QMessageBox.information(
            self,
            "Estimate Gaze",
            "Estimate Gaze initiated. Please wait...",
            QMessageBox.Ok
        )
        if self.current_video_path and os.path.exists(self.current_video_path):
            try:
                vid_dir = os.path.dirname(self.current_video_path)
                frames_dir = os.path.join(vid_dir, "frames")
                detecthead_dir = os.path.join(vid_dir, "detecthead")

                if not os.path.exists(frames_dir):
                    os.makedirs(frames_dir)
                    logging.info(f"Created frames at: {frames_dir}")

                success = self.extract_frames(self.current_video_path,
                                              frames_dir, desired_fps=30)
                if success:
                    QMessageBox.information(
                        self,
                        "Estimate Gaze",
                        f"Frames extracted to:\n{frames_dir}",
                        QMessageBox.Ok
                    )
                    logging.info(f"Frames extracted: {frames_dir}")
                else:
                    QMessageBox.warning(
                        self,
                        "Estimate Gaze",
                        "Failed to extract frames.",
                        QMessageBox.Ok
                    )
                    return

                if not os.path.exists(detecthead_dir):
                    os.makedirs(detecthead_dir)
                    logging.info(f"Created detecthead at: {detecthead_dir}")

                # Start head detection
                self.head_detection_thread = HeadDetectionThread(
                    frames_dir=frames_dir,
                    detecthead_dir=detecthead_dir,
                    model_path=self.yolov5_model_path,
                    conf_threshold=0.40
                )
                self.head_detection_thread.progress_signal.connect(self.update_progress)
                self.head_detection_thread.finished_signal.connect(self.head_detection_finished)

                self.progress_dialog = QProgressDialog("Detecting heads...", "Cancel",
                                                       0, 100, self)
                self.progress_dialog.setWindowTitle("Head Detection Progress")
                self.progress_dialog.setWindowModality(Qt.WindowModal)
                self.progress_dialog.setMinimumDuration(0)
                self.progress_dialog.setValue(0)
                self.progress_dialog.canceled.connect(self.cancel_head_detection)

                self.head_detection_thread.start()
                self.progress_dialog.show()

                logging.info("Head detection thread started.")
            except Exception as e:
                self.showErrorMessage(f"Estimate Gaze failed: {e}")
                logging.error(f"Estimate Gaze error: {e}")
        else:
            QMessageBox.information(
                self,
                "Estimate Gaze",
                "No valid video source available for gaze estimation.",
                QMessageBox.Ok
            )
            logging.info("Invalid video for gaze estimation.")

    def head_detection_finished(self, success, message):
        if self.progress_dialog:
            self.progress_dialog.close()

        if success:
            QMessageBox.information(
                self, "Head Detection", message,
                QMessageBox.Ok
            )
            logging.info(message)
            self.initiate_gaze_estimation()
        else:
            QMessageBox.warning(
                self, "Head Detection", message,
                QMessageBox.Ok
            )
            logging.warning(message)

    def initiate_gaze_estimation(self):
        try:
            vid_dir = os.path.dirname(self.current_video_path)
            gaze_output_dir = os.path.join(vid_dir, "gaze_output")

            self.gaze_estimation_thread = GazeEstimationThread(
                processed_dir=vid_dir,
                gaze_dir=gaze_output_dir,
                model_path=self.gaze_model_path,
                threshold=0.5,
                sigma=1.0
            )
            self.gaze_estimation_thread.progress_signal.connect(self.update_progress)
            self.gaze_estimation_thread.finished_signal.connect(self.gaze_estimation_finished)

            self.progress_dialog = QProgressDialog("Estimating gaze...", "Cancel",
                                                   0, 100, self)
            self.progress_dialog.setWindowTitle("Gaze Estimation Progress")
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.setValue(0)
            self.progress_dialog.canceled.connect(self.cancel_gaze_estimation)

            self.gaze_estimation_thread.start()
            self.progress_dialog.show()
            logging.info("Gaze estimation thread started.")
        except Exception as e:
            self.showErrorMessage(f"Failed to initiate gaze estimation: {e}")
            logging.error(f"Gaze estimation error: {e}")

    def overlay_heatmap_on_frame(self, frame, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlays a heatmap onto a video frame.

        Parameters:
        - frame: Original video frame (BGR format).
        - heatmap: Heatmap image (grayscale or single-channel).
        - alpha: Transparency factor for the heatmap.
        - colormap: OpenCV colormap to apply to the heatmap.

        Returns:
        - Overlayed frame.
        """
        if heatmap is None:
            return frame

        # Ensure heatmap is in uint8 format
        heatmap = np.array(heatmap, dtype=np.uint8)

        # Apply colormap
        heatmap_color = cv2.applyColorMap(heatmap, colormap)

        # Resize heatmap to match frame size if necessary
        if heatmap_color.shape[:2] != frame.shape[:2]:
            heatmap_color = cv2.resize(heatmap_color, (frame.shape[1], frame.shape[0]))

        # Overlay heatmap on frame
        overlayed_frame = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)

        return overlayed_frame

    def create_overlayed_video(self, original_video_path, heatmaps_dir, output_video_path, alpha=0.5):
        """
        Creates a new video with heatmaps overlayed on the original frames.

        Parameters:
        - original_video_path: Path to the original video.
        - heatmaps_dir: Directory containing heatmap images.
        - output_video_path: Path to save the overlayed video.
        - alpha: Transparency factor for the heatmap.

        Returns:
        - success (bool): True if video is created successfully, else False.
        """
        cap = cv2.VideoCapture(original_video_path)
        if not cap.isOpened():
            self.showErrorMessage("Cannot open original video for overlaying.")
            logging.error("Cannot open original video for overlaying.")
            return False

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs if needed
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_number = 0
        success = True

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            heatmap_filename = f"frame_{frame_number:05d}_heatmap.png"
            heatmap_path = os.path.join(heatmaps_dir, heatmap_filename)
            if os.path.exists(heatmap_path):
                heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
            else:
                heatmap = None

            # Overlay heatmap on frame
            overlayed_frame = self.overlay_heatmap_on_frame(frame, heatmap, alpha=alpha)

            # Write the frame to the output video
            out.write(overlayed_frame)

            frame_number += 1
            if frame_number % 100 == 0:
                logging.info(f"Overlayed and wrote {frame_number} frames.")

        cap.release()
        out.release()
        logging.info(f"Overlayed video saved to {output_video_path}")
        return success

    def gaze_estimation_finished(self, success, message):
        if self.progress_dialog:
            if success:
                # Disconnect the canceled signal to prevent unintended cancellation
                try:
                    self.progress_dialog.canceled.disconnect(self.cancel_gaze_estimation)
                except TypeError:
                    # The signal was already disconnected
                    pass
            self.progress_dialog.close()

        if success:
            QMessageBox.information(
                self, "Gaze Estimation",
                message, QMessageBox.Ok
            )
            logging.info(message)

            try:
                # Paths setup
                vid_dir = os.path.dirname(self.current_video_path)
                gaze_output_dir = os.path.join(vid_dir, "gaze_output")
                heatmaps_dir = os.path.join(gaze_output_dir, "heatmaps")
                overlayed_video_path = os.path.join(gaze_output_dir, "overlayed_video.mp4")

                # Create overlayed video
                creation_success = self.create_overlayed_video(
                    original_video_path=self.current_video_path,
                    heatmaps_dir=heatmaps_dir,
                    output_video_path=overlayed_video_path,
                    alpha=0.5  # Adjust transparency as needed
                )

                if creation_success:
                    QMessageBox.information(
                        self, "Overlayed Video",
                        f"Overlayed video created at:\n{overlayed_video_path}",
                        QMessageBox.Ok
                    )
                    logging.info(f"Overlayed video created: {overlayed_video_path}")

                    # Load and display the overlayed video in frame_5
                    self.load_video(overlayed_video_path)
                else:
                    QMessageBox.warning(
                        self, "Overlayed Video",
                        "Failed to create overlayed video.",
                        QMessageBox.Ok
                    )
                    logging.error("Failed to create overlayed video.")
            except Exception as e:
                self.showErrorMessage(f"Failed during overlay and playback: {e}")
                logging.error(f"Overlay and playback error: {e}")
        else:
            QMessageBox.warning(
                self, "Gaze Estimation",
                message, QMessageBox.Ok
            )
            logging.warning(message)

    def cancel_gaze_estimation(self):
        if self.gaze_estimation_thread and self.gaze_estimation_thread.isRunning():
            self.gaze_estimation_thread.stop()
            self.gaze_estimation_thread = None
            logging.info("Gaze estimation canceled by user.")
            QMessageBox.information(
                self,
                "Gaze Estimation",
                "Gaze estimation has been canceled.",
                QMessageBox.Ok
            )

    def update_progress(self, value):
        if self.progress_dialog:
            self.progress_dialog.setValue(value)

    def cancel_head_detection(self):
        if self.head_detection_thread and self.head_detection_thread.isRunning():
            self.head_detection_thread.stop()
            self.head_detection_thread = None
            logging.info("Head detection canceled by user.")
            QMessageBox.information(
                self,
                "Head Detection",
                "Head detection has been canceled.",
                QMessageBox.Ok
            )

    # Save blink results
    def save_blink_results(self, video_path, blink_count, blink_rate):
        video_file = os.path.basename(video_path)
        vid_dir = os.path.dirname(video_path)
        blink_csv_path = os.path.join(vid_dir, "blink_detection.csv")

        person_id, label = self.extract_person_info(video_path)
        try:
            with open(blink_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['person_id', 'label', 'blink_count', 'blink_rate_bpm'])
                writer.writerow([person_id, label, blink_count, blink_rate])
            logging.info(f"Blink detection saved to {blink_csv_path}")
            QMessageBox.information(
                self,
                "Blink Detection Results",
                f"Blink Count: {blink_count}\nBlink Rate: {blink_rate} bpm",
                QMessageBox.Ok
            )
        except Exception as e:
            self.showErrorMessage(f"Failed to save blink detection results: {e}")
            logging.error(f"Error saving blink detection: {e}")

    # Extract person info from filename
    def extract_person_info(self, video_path):
        video_file = os.path.basename(video_path)
        pattern = re.compile(r'(\d+)([RS])', re.IGNORECASE)
        match = pattern.match(os.path.splitext(video_file)[0])
        if match:
            number = match.group(1)
            code = match.group(2).upper()
            pid = f"person_{number}"
            label = 'scripted' if code == 'R' else 'spontaneous'
            return pid, label
        return "unknown_person", "unknown_label"

    # Frame Extraction
    def extract_frames(self, video_path, frames_dir, desired_fps=30):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Cannot open video: {video_path}")
            return False

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps == 0:
            orig_fps = 30
        frame_interval = int(round(orig_fps / desired_fps))
        if frame_interval == 0:
            frame_interval = 1

        logging.info(f"Orig FPS={orig_fps}, Desired={desired_fps}, Interval={frame_interval}")
        frame_count = 0
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                out_name = os.path.join(frames_dir, f"frame_{saved_count:05d}.jpg")
                try:
                    cv2.imwrite(out_name, frame)
                    logging.info(f"Saved {out_name}")
                    saved_count += 1
                except Exception as e:
                    logging.error(f"Error saving frame {out_name}: {e}")

            frame_count += 1
        cap.release()
        logging.info(f"Total frames extracted: {saved_count}")
        return saved_count > 0

    def uploadViaLink(self):
        """
        Handles the 'Upload via Link' button click.
        Prompts the user for a Google Drive URL, downloads the video,
        saves it in a subfolder of database named after the video,
        and then plays the video.
        """
        logging.info("Upload via link initiated.")

        # Prompt the user to enter the video URL
        url, ok = QInputDialog.getText(
            self,
            "Upload Video via Link",
            "Enter the Google Drive video URL:"
        )

        if not ok or not url:
            logging.info("Upload via link canceled or no URL provided.")
            return

        logging.info(f"User provided URL: {url}")

        try:
            # 1) Validate/extract the file ID from the Google Drive URL
            file_id = self.extract_gdrive_file_id(url)
            if not file_id:
                raise ValueError("Invalid Google Drive URL format.")

            # 2) Prepare the database folder
            db_folder = "database"
            if not os.path.exists(db_folder):
                os.makedirs(db_folder)
                logging.info(f"Created database directory at: {db_folder}")

            # 3) Optionally retrieve the file name using gdown metadata
            file_name = self.get_gdrive_file_name(file_id)
            if not file_name:
                # Fallback if we can't retrieve an original file name
                file_name = f"video_{int(time.time())}.mp4"
                logging.warning("Could not retrieve file name from Google Drive. Using default name.")

            # 4) Extract base name (no extension) for subfolder
            base_name = os.path.splitext(file_name)[0]

            # 5) Create subfolder under database with the base name
            vid_folder = os.path.join(db_folder, base_name)
            if not os.path.exists(vid_folder):
                os.makedirs(vid_folder)
                logging.info(f"Created subfolder for video: {vid_folder}")

            # 6) Define the final download path (inside vid_folder)
            download_path = os.path.join(vid_folder, file_name)

            # 7) Download the file using gdown
            logging.info(f"Starting download of file ID: {file_id}")
            gdown.download(id=file_id, output=download_path, quiet=False)
            logging.info(f"Downloaded video to: {download_path}")

            # 8) Update the current video path and load the video
            self.current_video_path = download_path
            self.load_video(self.current_video_path)
            logging.info(f"Set current_video_path to: {self.current_video_path}")

            QMessageBox.information(
                self,
                "Upload Successful",
                f"Video downloaded and saved to:\n{download_path}",
                QMessageBox.Ok
            )

        except Exception as e:
            logging.error(f"Error during upload via link: {e}")
            QMessageBox.critical(
                self,
                "Upload Failed",
                f"An error occurred while uploading the video:\n{str(e)}",
                QMessageBox.Ok
            )

    def extract_gdrive_file_id(self, url):
        """
        Extracts the Google Drive file ID from the provided URL.

        Supports various Google Drive URL formats.
        """
        # Regular expressions for different Google Drive URL formats
        patterns = [
            r'https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',
            r'https?://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)',
            r'https?://drive\.google\.com/uc\?export=download&id=([a-zA-Z0-9_-]+)'
        ]

        for pattern in patterns:
            match = re.match(pattern, url)
            if match:
                return match.group(1)
        return None

    def get_gdrive_file_name(self, file_id):
        """
        Retrieves the file name from Google Drive using the file ID.
        Requires the file to be publicly accessible or you have access permissions.
        """
        try:
            # Construct the API URL to fetch file metadata
            api_url = f"https://drive.google.com/uc?export=download&id={file_id}"

            # Send a HEAD request to get the content-disposition header
            response = requests.head(api_url, allow_redirects=True)

            if 'Content-Disposition' in response.headers:
                content_disposition = response.headers['Content-Disposition']
                # Extract the filename from the header
                filename_match = re.findall('filename="(.+)"', content_disposition)
                if filename_match:
                    return filename_match[0]
        except Exception as e:
            logging.error(f"Error retrieving file name from Google Drive: {e}")

        return None

    def extractFeatures(self):
        """
        Extract gaze-based features from the JSON data files in <video_folder>/gaze_output/data
        and save them in <video_folder>/gaze_output/gaze_features.csv.

        Note: We no longer set any 'label' column. Only 'person_id' is kept if needed.
        """
        logging.info("Extract Features clicked")

        # 1) Check if we have a video
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            QMessageBox.warning(self, "Feature Extraction",
                                "No valid video found. Please upload a video first.",
                                QMessageBox.Ok)
            logging.warning("No valid video for feature extraction.")
            return

        # 2) Check if gaze output data exist
        vid_dir = os.path.dirname(self.current_video_path)
        data_dir = os.path.join(vid_dir, "gaze_output", "data")
        if not os.path.exists(data_dir):
            QMessageBox.warning(self, "Feature Extraction",
                                "No gaze_output/data folder found. Run Gaze Estimation first.",
                                QMessageBox.Ok)
            logging.warning(f"No gaze_output/data folder in {vid_dir}")
            return

        json_files = sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith(".json") and "_gaze" in f
        ])
        if not json_files:
            QMessageBox.warning(self, "Feature Extraction",
                                "No gaze .json files found in gaze_output/data. Run Gaze Estimation first.",
                                QMessageBox.Ok)
            logging.warning(f"No .json data files found in {data_dir}")
            return

        # 3) Build a DataFrame from all frame JSON
        gaze_data = []
        for jf in json_files:
            json_path = os.path.join(data_dir, jf)
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                logging.error(f"Error reading {json_path}: {e}")
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
            QMessageBox.warning(self, "Feature Extraction",
                                "Gaze data is empty. Nothing to process.",
                                QMessageBox.Ok)
            logging.warning("No valid gaze data collected.")
            return

        df = pd.DataFrame(gaze_data)

        # 4) Compute features on the entire DataFrame
        features = extract_features_from_group(df)
        if not features:
            QMessageBox.warning(self, "Feature Extraction",
                                "Could not extract any features (empty).",
                                QMessageBox.Ok)
            logging.warning("Feature dictionary is empty.")
            return

        # Optionally, retrieve a person_id from the video filename
        # or set to something generic if not needed
        person_id, _ = self.extract_person_info(self.current_video_path)
        features['person_id'] = person_id

        # 5) Convert features dict => DataFrame => CSV
        features_df = pd.DataFrame([features])  # single row
        # Put person_id at the front
        cols = ['person_id'] + [c for c in features_df.columns if c not in ['person_id']]
        features_df = features_df[cols]

        csv_path = os.path.join(vid_dir, "gaze_output", "gaze_features.csv")
        try:
            features_df.to_csv(csv_path, index=False)
            QMessageBox.information(
                self, "Feature Extraction",
                f"Feature extraction completed.\nSaved to:\n{csv_path}",
                QMessageBox.Ok
            )
            logging.info(f"Feature extraction completed. Saved to {csv_path}")
        except Exception as e:
            QMessageBox.critical(
                self, "Feature Extraction Error",
                f"Failed to save features: {e}",
                QMessageBox.Ok
            )
            logging.error(f"Error saving features CSV: {e}")

    # Prediction Method
    def prediction(self):
        """
        Perform prediction using the blink_detection.csv and gaze_features.csv
        for the current video. Overlay the prediction on the video and display
        the results in a dialog box.
        """
        logging.info("Prediction clicked")

        # 1. Check if video, blink detection, and gaze estimation have been processed
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            QMessageBox.warning(self, "Prediction",
                                "No valid video found. Please upload a video first.",
                                QMessageBox.Ok)
            logging.warning("No valid video for prediction.")
            return

        vid_dir = os.path.dirname(self.current_video_path)
        blink_csv_path = os.path.join(vid_dir, "blink_detection.csv")
        features_csv_path = os.path.join(vid_dir, "gaze_output", "gaze_features.csv")

        if not os.path.exists(blink_csv_path):
            QMessageBox.warning(self, "Prediction",
                                "Blink detection results not found. Please run Blink Detection first.",
                                QMessageBox.Ok)
            logging.warning(f"Blink detection CSV not found at: {blink_csv_path}")
            return

        if not os.path.exists(features_csv_path):
            QMessageBox.warning(self, "Prediction",
                                "Gaze feature results not found. Please run Feature Extraction first.",
                                QMessageBox.Ok)
            logging.warning(f"Gaze features CSV not found at: {features_csv_path}")
            return

        # 2. Load the trained model and label encoder
        model_path = os.path.join("model", "behavior_classifier_pipeline.joblib")
        label_encoder_path = os.path.join("model", "label_encoder.joblib")

        try:
            pipeline, le = self.load_model_and_encoder(model_path, label_encoder_path)
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error",
                                 f"Failed to load model and label encoder:\n{e}",
                                 QMessageBox.Ok)
            logging.error(f"Failed to load model and label encoder: {e}")
            return

        # 3. Load and merge data
        try:
            merged_data = self.load_new_data(features_csv_path, blink_csv_path)
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error",
                                 f"Failed to load and merge data:\n{e}",
                                 QMessageBox.Ok)
            logging.error(f"Failed to load and merge data: {e}")
            return

        # 4. Define features for prediction
        X = self.define_features_for_prediction(merged_data)

        # 5. Make prediction
        try:
            results = self.make_prediction(pipeline, le, X)
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error",
                                 f"Failed during prediction:\n{e}",
                                 QMessageBox.Ok)
            logging.error(f"Failed during prediction: {e}")
            return

        # 6. Save prediction results
        prediction_output_path = os.path.join(vid_dir, "gaze_output", "prediction_result.csv")
        try:
            self.save_results(results, prediction_output_path)
        except Exception as e:
            QMessageBox.critical(
                self, "Prediction Error",
                f"Failed to save prediction results:\n{e}",
                QMessageBox.Ok
            )
            logging.error(f"Failed to save prediction results: {e}")
            return

        # 7. Overlay prediction on video
        predicted_label = results['predicted_label'].iloc[0]
        confidence_score = results['confidence_score'].iloc[0]

        try:
            overlayed_video_path = os.path.join(vid_dir, "gaze_output", "overlayed_prediction_video.mp4")
            self.create_prediction_overlayed_video(
                original_video_path=self.current_video_path,
                predicted_label=predicted_label,
                confidence_score=confidence_score,
                output_video_path=overlayed_video_path
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Prediction Error",
                f"Failed to overlay prediction on video:\n{e}",
                QMessageBox.Ok
            )
            logging.error(f"Failed to overlay prediction on video: {e}")
            return

        # 8. Play the overlayed video
        self.load_video(overlayed_video_path)
        logging.info(f"Overlayed video with prediction loaded: {overlayed_video_path}")

        # 9. Show dialog box with prediction results
        QMessageBox.information(
            self,
            "Prediction Results",
            f"Predicted Behavior: {predicted_label}\nConfidence Score: {confidence_score:.2f}",
            QMessageBox.Ok
        )
        logging.info(f"Prediction Results - Label: {predicted_label}, Confidence: {confidence_score:.2f}")
    # Prediction Helpers
    def load_model_and_encoder(self, model_path, label_encoder_path):
        """Load the trained classifier pipeline and label encoder."""
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(label_encoder_path):
            logging.error(f"Label encoder file not found at: {label_encoder_path}")
            raise FileNotFoundError(f"Label encoder file not found at: {label_encoder_path}")

        try:
            pipeline = joblib.load(model_path)
            le = joblib.load(label_encoder_path)
            logging.info("Successfully loaded the model pipeline and label encoder.")
            return pipeline, le
        except Exception as e:
            logging.error(f"Error loading model or label encoder: {e}")
            raise e

    def load_new_data(self, gaze_features_path, blink_results_path):
        """Load and merge gaze_features and blink_results datasets."""
        if not os.path.exists(gaze_features_path):
            logging.error(f"Gaze features file not found at: {gaze_features_path}")
            raise FileNotFoundError(f"Gaze features file not found at: {gaze_features_path}")
        if not os.path.exists(blink_results_path):
            logging.error(f"Blink results file not found at: {blink_results_path}")
            raise FileNotFoundError(f"Blink results file not found at: {blink_results_path}")

        gaze_features = pd.read_csv(gaze_features_path)
        blink_results = pd.read_csv(blink_results_path)

        # Ensure 'person_id' exists in both DataFrames
        if 'person_id' not in gaze_features.columns or 'person_id' not in blink_results.columns:
            logging.error("Both gaze_features and blink_results must contain 'person_id' column for merging.")
            raise KeyError("Missing 'person_id' column in one of the input files.")

        # Drop 'label' from blink_results if present, since it's unknown during prediction
        if 'label' in blink_results.columns:
            blink_results = blink_results.drop('label', axis=1)

        merged_data = pd.merge(gaze_features, blink_results, on=['person_id'], how='inner')

        if merged_data.empty:
            logging.error("Merged data is empty. Check if 'person_id's match between the two files.")
            raise ValueError("No matching 'person_id's found between gaze_features and blink_results.")

        logging.info("Successfully loaded and merged the new data.")
        return merged_data

    def define_features_for_prediction(self, merged_data):
        """Define feature matrix X for prediction."""
        # Drop columns that are not features. During training, 'person_id', 'label', and 'label_encoded' were dropped.
        # Since 'label' is not available, ensure these columns are excluded if present.
        columns_to_drop = ['person_id']
        for col in columns_to_drop:
            if col in merged_data.columns:
                merged_data = merged_data.drop(col, axis=1)

        X = merged_data.copy()
        return X

    def make_prediction(self, pipeline, le, X):
        """Make prediction using the pipeline and label encoder."""
        try:
            # Predict the label
            y_pred = pipeline.predict(X)

            # Predict the probability
            y_prob = pipeline.predict_proba(X)[:, 1]  # Probability for the positive class

            # Decode the label
            y_pred_label = le.inverse_transform(y_pred)

            # Create a result DataFrame
            results = pd.DataFrame({
                'predicted_label': y_pred_label,
                'confidence_score': y_prob
            })

            return results
        except NotFittedError as e:
            logging.error(f"Model is not fitted: {e}")
            raise e
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise e

    def save_results(self, results, output_path):
        """Save the prediction results to a CSV file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            results.to_csv(output_path, index=False)
            logging.info(f"Prediction results saved to: {output_path}")
        except Exception as e:
            logging.error(f"Error saving prediction results: {e}")
            raise e

    # Prediction Overlayed Video
    def create_prediction_overlayed_video(self, original_video_path, predicted_label, confidence_score, output_video_path, alpha=0.5):
        """
        Creates a new video with prediction label and confidence score overlaid on each frame.

        Parameters:
        - original_video_path: Path to the original video.
        - predicted_label: Predicted behavior label (e.g., 'scripted' or 'spontaneous').
        - confidence_score: Confidence score of the prediction (float between 0 and 1).
        - output_video_path: Path to save the overlayed video.
        - alpha: Transparency factor for any overlays (not used here but kept for extensibility).
        """
        cap = cv2.VideoCapture(original_video_path)
        if not cap.isOpened():
            self.showErrorMessage("Cannot open original video for overlaying.")
            logging.error("Cannot open original video for overlaying.")
            raise IOError("Cannot open original video for overlaying.")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose other codecs if needed
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Define the text to overlay
            text = f"Predicted: {predicted_label} ({confidence_score*100:.2f}%)"

            # Choose font, scale, color, and thickness
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 255, 0) if predicted_label.lower() == 'scripted' else (0, 0, 255)
            thickness = 2

            # Get the text size
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

            # Set the text start position (e.g., bottom-left corner)
            x, y = 10, height - 10

            # Add a rectangle background for better visibility
            cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), (0, 0, 0), -1)

            # Put the text on the frame
            cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

            # Write the frame to the output video
            out.write(frame)

            frame_number += 1
            if frame_number % 100 == 0:
                logging.info(f"Overlayed and wrote {frame_number} frames.")

        cap.release()
        out.release()
        logging.info(f"Overlayed video saved to {output_video_path}")

    def save(self):
        """
        Handles the 'Save' button click.
        Displays a dialog box indicating that the prediction results have been saved,
        along with their file paths.
        """
        logging.info("Save clicked")

        # Ensure a video has been processed
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            QMessageBox.warning(
                self,
                "Save Results",
                "No video has been processed yet. Please perform prediction first.",
                QMessageBox.Ok
            )
            logging.warning("Save attempted without a processed video.")
            return

        vid_dir = os.path.dirname(self.current_video_path)
        gaze_output_dir = os.path.join(vid_dir, "gaze_output")
        prediction_csv = os.path.join(gaze_output_dir, "prediction_result.csv")
        overlayed_video = os.path.join(gaze_output_dir, "overlayed_prediction_video.mp4")

        # Check if prediction results exist
        if os.path.exists(prediction_csv) and os.path.exists(overlayed_video):
            message = (
                f"Prediction results have been saved successfully!\n\n"
                f"Prediction CSV: {prediction_csv}\n"
                f"Overlayed Video: {overlayed_video}"
            )
            QMessageBox.information(
                self,
                "Save Results",
                message,
                QMessageBox.Ok
            )
            logging.info(f"Results saved at {prediction_csv} and {overlayed_video}")
        else:
            QMessageBox.warning(
                self,
                "Save Results",
                "Prediction results not found. Please run Prediction first.",
                QMessageBox.Ok
            )
            logging.warning("Attempted to save results, but prediction files are missing.")


    # -----------------------------
    # show results

    def showResults(self):
        """
        Handles the 'Show Results' button click.
        Loads and plays the overlayed_prediction_video.mp4, displaying the prediction label
        and confidence score on the video.
        """
        logging.info("Show Results clicked")

        # Ensure a video has been processed
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            QMessageBox.warning(
                self,
                "Show Results",
                "No video has been processed yet. Please perform prediction first.",
                QMessageBox.Ok
            )
            logging.warning("Show Results attempted without a processed video.")
            return

        vid_dir = os.path.dirname(self.current_video_path)
        gaze_output_dir = os.path.join(vid_dir, "gaze_output")
        overlayed_video = os.path.join(gaze_output_dir, "overlayed_prediction_video.mp4")

        # Check if the overlayed prediction video exists
        if os.path.exists(overlayed_video):
            self.load_video(overlayed_video)
            QMessageBox.information(
                self,
                "Show Results",
                f"Playing the overlayed prediction video:\n{overlayed_video}",
                QMessageBox.Ok
            )
            logging.info(f"Playing overlayed prediction video: {overlayed_video}")
        else:
            QMessageBox.warning(
                self,
                "Show Results",
                "Overlayed prediction video not found. Please run Prediction first.",
                QMessageBox.Ok
            )
            logging.warning("Overlayed prediction video not found for playback.")

    # Quit logic
    def quit(self):
        reply = QMessageBox.question(
            self,
            "Quit Application",
            "Are you sure you want to quit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            logging.info("Quitting application...")
            QApplication.quit()
        else:
            logging.info("Quit canceled.")

    # Close event
    def closeEvent(self, event):
        try:
            if self.head_detection_thread and self.head_detection_thread.isRunning():
                self.head_detection_thread.stop()
                self.head_detection_thread = None
                logging.info("Head detection thread terminated.")
        except Exception as e:
            logging.error(f"Error terminating head detection thread: {e}")

        try:
            if self.gaze_estimation_thread and self.gaze_estimation_thread.isRunning():
                self.gaze_estimation_thread.stop()
                self.gaze_estimation_thread = None
                logging.info("Gaze estimation thread terminated.")
        except Exception as e:
            logging.error(f"Error terminating gaze estimation thread: {e}")

        try:
            if self.video_capture is not None:
                self.timer.stop()
                self.video_capture.release()
                logging.info("Video capture released on close.")
        except Exception as e:
            logging.error(f"Error releasing video capture on close: {e}")

        event.accept()


# -----------------------------
# Main Entry
# -----------------------------
if __name__ == "__main__":
    app = QApplication([])
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
