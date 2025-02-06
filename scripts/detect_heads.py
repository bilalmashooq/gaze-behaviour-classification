# # scripts/detect_heads.py
#
# import torch
# import os
# import json
# from pathlib import Path
# from PIL import Image
#
# def detect_heads_yolov5(frame_path, model, device):
#     """
#     Detect heads in a given frame using the YOLOv5 model.
#
#     Args:
#         frame_path (str): Path to the image frame.
#         model (torch.nn.Module): Loaded YOLOv5 model.
#         device (torch.device): Device to run inference on.
#
#     Returns:
#         list: A list of bounding boxes [x1, y1, x2, y2] for each detected head.
#     """
#     image = Image.open(frame_path).convert('RGB')
#     results = model(image, size=640)  # Adjust size as needed
#     detections = results.xyxy[0]  # Bounding boxes
#     head_boxes = []
#     for *box, conf, cls in detections:
#         if int(cls.item()) == 0:  # 'person' class in COCO is usually 0
#             x1, y1, x2, y2 = box
#             head_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
#     return head_boxes
#
# def process_head_detection(processed_dir, model, device):
#     """
#     Process all videos in the processed directory to detect heads in each frame.
#
#     Args:
#         processed_dir (str): Directory containing processed frames.
#         model (torch.nn.Module): Loaded YOLOv5 model.
#         device (torch.device): Device to run inference on.
#     """
#     for person_dir in os.listdir(processed_dir):
#         person_path = os.path.join(processed_dir, person_dir)
#         if not os.path.isdir(person_path):
#             continue
#         for label in ['scripted', 'spontaneous']:
#             frames_dir = os.path.join(person_path, label, 'frames')
#             head_boxes_dir = os.path.join(person_path, label, 'head_boxes')
#             os.makedirs(head_boxes_dir, exist_ok=True)
#             for frame_file in os.listdir(frames_dir):
#                 if frame_file.endswith(('.jpg', '.png')):
#                     frame_path = os.path.join(frames_dir, frame_file)
#                     head_boxes = detect_heads_yolov5(frame_path, model, device)
#                     # Save head boxes to JSON
#                     frame_name = Path(frame_file).stem  # e.g., 'frame_0000'
#                     with open(os.path.join(head_boxes_dir, f"{frame_name}.json"), 'w') as f:
#                         json.dump(head_boxes, f)
#             print(f"Processed head detection for {person_dir} - {label}")
#
# if __name__ == "__main__":
#     # Path to your downloaded YOLOv5n model
#     custom_model_path = r"C:\Users\muham\PycharmProjects\gazelle\model\yolov5n.pt"
#
#     # Check if the custom model file exists
#     if not os.path.exists(custom_model_path):
#         raise FileNotFoundError(f"YOLOv5 model not found at {custom_model_path}. Please check the path.")
#
#     # Load YOLOv5n Model as a custom model
#     yolo_model = torch.hub.load('yolov5', 'custom', path=custom_model_path, source='local')
#     # 'source' can be 'local' if the model is stored locally
#
#     # Set the model to evaluation mode
#     yolo_model.eval()
#
#     # Move Model to CPU (since CUDA is not available)
#     device = torch.device("cpu")
#     yolo_model.to(device)
#
#     # Adjust confidence threshold if needed
#     yolo_model.conf = 0.25  # Default is 0.25
#
#     # Define Processed Directory
#     processed_directory = r"C:\Users\muham\PycharmProjects\gazelle\processed"
#
#     # Run Head Detection
#     process_head_detection(processed_directory, yolo_model, device)


# scripts/detect_heads.py

import torch
import os
import json
from pathlib import Path
from PIL import Image
import logging
from tqdm import tqdm


def setup_logging(log_file='detect_heads.log'):
    """
    Sets up the logging configuration.

    Args:
        log_file (str): Path to the log file.
    """
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def load_yolov5_model(model_path, device):
    """
    Loads the YOLOv5 model from the specified path.

    Args:
        model_path (str): Path to the YOLOv5 model weights (.pt file).
        device (torch.device): Device to load the model onto.

    Returns:
        model (torch.nn.Module): Loaded YOLOv5 model.
    """
    try:
        # Load the YOLOv5 model as a custom model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, source='local')
        model.to(device)
        model.eval()
        logging.info(f"YOLOv5 model loaded from {model_path} on {device}.")
        return model
    except Exception as e:
        logging.critical(f"Failed to load YOLOv5 model from {model_path}: {e}")
        raise e


def detect_heads_yolov5(frame_path, model, device, conf_threshold=0.25, classes=[0]):
    """
    Detect heads in a given frame using the YOLOv5 model.

    Args:
        frame_path (str): Path to the image frame.
        model (torch.nn.Module): Loaded YOLOv5 model.
        device (torch.device): Device to run inference on.
        conf_threshold (float): Confidence threshold for detections.
        classes (list): List of class indices to filter detections (0 for 'person').

    Returns:
        list: A list of bounding boxes [x1, y1, x2, y2] for each detected head.
    """
    try:
        image = Image.open(frame_path).convert('RGB')
    except Exception as e:
        logging.error(f"Error opening image {frame_path}: {e}")
        return []

    try:
        results = model(image, size=640)  # Adjust size as needed
    except RuntimeError as e:
        if 'out of memory' in str(e):
            logging.error(f"Out of Memory error during inference on {frame_path}: {e}")
            torch.cuda.empty_cache()
        else:
            logging.error(f"Runtime error during inference on {frame_path}: {e}")
        return []
    except Exception as e:
        logging.error(f"Error during inference on {frame_path}: {e}")
        return []

    detections = results.xyxy[0]  # Bounding boxes
    head_boxes = []
    for *box, conf, cls in detections:
        if int(cls.item()) in classes and conf.item() >= conf_threshold:
            x1, y1, x2, y2 = box
            head_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])

    return head_boxes


def process_head_detection(processed_dir, model, device, start_index=1, conf_threshold=0.25):
    """
    Process all videos in the processed directory to detect heads in each frame,
    starting from the specified person index.

    Args:
        processed_dir (str): Directory containing processed frames.
        model (torch.nn.Module): Loaded YOLOv5 model.
        device (torch.device): Device to run inference on.
        start_index (int): Index of the person directory to start processing from (0-based).
        conf_threshold (float): Confidence threshold for detections.

    Returns:
        None
    """
    person_dirs = sorted([d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))])

    if not person_dirs:
        logging.warning("No person directories found in the processed directory.")
        return

    if start_index >= len(person_dirs):
        logging.warning(f"Start index {start_index} is out of range. No persons to process.")
        return

    # Slice the list to start from the specified index
    persons_to_process = person_dirs[start_index:]

    logging.info(
        f"Starting head detection for persons {start_index + 1} to {len(person_dirs)} out of {len(person_dirs)}.")

    for person_dir in persons_to_process:
        person_path = os.path.join(processed_dir, person_dir)
        for label in ['scripted', 'spontaneous']:
            frames_dir = os.path.join(person_path, label, 'frames')
            head_boxes_dir = os.path.join(person_path, label, 'head_boxes')
            if not os.path.exists(frames_dir):
                logging.warning(f"Frames directory missing for {person_dir} - {label}. Skipping.")
                continue
            if not os.path.exists(head_boxes_dir):
                os.makedirs(head_boxes_dir, exist_ok=True)

            frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if not frame_files:
                logging.warning(f"No frames found in {frames_dir}. Skipping {person_dir} - {label}.")
                continue

            logging.info(f"Processing {len(frame_files)} frames for {person_dir} - {label}...")

            for frame_file in tqdm(frame_files, desc=f"{person_dir} - {label}", unit="frame"):
                frame_path = os.path.join(frames_dir, frame_file)
                head_boxes = detect_heads_yolov5(frame_path, model, device, conf_threshold=conf_threshold)
                frame_name = Path(frame_file).stem  # e.g., 'frame_0000'
                json_path = os.path.join(head_boxes_dir, f"{frame_name}.json")
                try:
                    with open(json_path, 'w') as f:
                        json.dump(head_boxes, f)
                except Exception as e:
                    logging.error(f"Failed to write head boxes for {frame_path}: {e}")

        logging.info(f"Completed head detection for {person_dir}.")

    logging.info("Head detection completed for all specified persons.")


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Path to your downloaded YOLOv5n model
    custom_model_path = r"C:\Users\muham\PycharmProjects\gazelle\model\yolov5n.pt"

    # Check if the custom model file exists
    if not os.path.exists(custom_model_path):
        logging.critical(f"YOLOv5 model not found at {custom_model_path}. Please check the path.")
        raise FileNotFoundError(f"YOLOv5 model not found at {custom_model_path}.")

    # Set device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load YOLOv5n Model
    try:
        yolo_model = load_yolov5_model(custom_model_path, device)
    except Exception as e:
        logging.critical(f"Exiting due to model loading failure: {e}")
        exit(1)

    # Define Processed Directory
    processed_directory = r"C:\Users\muham\PycharmProjects\gazelle\processed"

    # Specify the starting person index (0-based).
    starting_person_index = 1

    # Run Head Detection
    process_head_detection(
        processed_dir=processed_directory,
        model=yolo_model,
        device=device,
        start_index=starting_person_index,
        conf_threshold=0.40  # Adjust confidence threshold as needed
    )
