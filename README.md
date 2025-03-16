# Eye Gaze Tracking and Classification System for Interview Analysis

**Master of Biometrics and Intelligent Vision**  


---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Head Detection](#head-detection)
  - [Blink Detection](#blink-detection)
  - [Gaze Estimation](#gaze-estimation)
  - [Feature Extraction](#feature-extraction)
  - [Behavior Classification](#behavior-classification)
  - [Prediction](#prediction)
  - [Desktop Application](#desktop-application)
- [Results and Evaluation](#results-and-evaluation)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [References](#references)

---

## Overview

This repository contains the code and documentation for an end-to-end system designed to analyze interview videos. The system tracks eye gaze and detects blink patterns to determine whether a candidate is reading from a script or speaking spontaneously. It integrates several advanced computer vision techniques such as:

- **Head Detection:** Using a pre-trained YOLOv5 model.
- **Gaze Estimation:** Leveraging the Gaze-LLE model (built on a DINOv2 backbone).
- **Blink Detection:** Using MediaPipe Face Mesh to compute the Eye Aspect Ratio (EAR).

The extracted features are used to train a machine learning classifier that distinguishes between scripted and spontaneous behavior. The entire pipeline is integrated into a desktop application built with PyQt5.

---

## Features

- **Data Collection & Preprocessing:**  
  - Organized dataset with videos recorded under scripted and spontaneous conditions.
  - Automated frame extraction using OpenCV.

- **Head Detection:**  
  - YOLOv5-based detection to localize heads in frames.
  - Saving bounding box coordinates as JSON.

- **Blink Detection:**  
  - MediaPipe Face Mesh for detecting blinks. 
  - Calculation of blink rate (blinks per minute).

- **Gaze Estimation:**  
  - Gaze-LLE model to produce a gaze heatmap. https://github.com/fkryan/gazelle
  - Smoothing using Gaussian filters and computation of gaze coordinates.

- **Feature Extraction:**  
  - Extraction of spatial, movement, and statistical features from gaze data.
  - Aggregation of gaze features (e.g., gaze entropy, fixation/saccade counts).

- **Behavior Classification:**  
  - Machine learning pipeline (feature selection, scaling, and Logistic Regression).
  - Performance evaluation using ROC AUC, confusion matrix, and classification report.
  - Analysis of feature importance.
    
![Gaze Tracking Results](results/plot_2025-01-26%2015-03-47_1.png)
![Gaze Tracking Results](results/plot_2025-01-26%2015-03-47_2.png).


- **Prediction:**  
  - Prediction on new data with visual overlay of results on video.
  - Saving predictions in CSV format.

- **Desktop Application:**  
  - Integrated PyQt5 application for video upload, processing, and result visualization.
  - Multi-threaded processing for head detection and gaze estimation.

---

## Repository Structure

```
├── main.py                         # Main PyQt5 application integrating all modules
├── scripts/                        # Folder containing all processing scripts
│   ├── extract_frames.py           # Script for extracting frames from video files
│   ├── detect_heads.py             # Script for head detection using YOLOv5
│   ├── blink_detection.py          # (Optional) Script for blink detection using MediaPipe Face Mesh
│   ├── estimate_gaze.py            # Script for gaze estimation using Gaze-LLE
│   ├── feature_extraction.py       # Script for extracting features from gaze data
│   ├── train_and_save_model.py     # Script to train the classifier and save the model pipeline
│   ├── predict_behavior.py         # Script to load the trained model and perform predictions on new data
├── requirements.txt               # List of Python dependencies
├── model/                         # Folder containing YOLOv5 and Gaze-LLE model checkpoints
├── data/                          # Folder for any additional data (e.g., new_data.csv for prediction)

```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- [Conda](https://docs.conda.io/) (recommended for environment management)
- CUDA-enabled GPU (for acceleration, optional but recommended)

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/bilalmashooq/eye-gaze-tracking.git
   cd eye-gaze-tracking
   ```

2. **Create and activate a Conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate gazelle
   ```

3. **Install additional dependencies (e.g., xformers for speed, if applicable):**
   ```bash
   pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install the Gaze-LLE package (from the cloned repository):**
   ```bash
   pip install -e .
   ```

5. **Verify that you have the required model checkpoints in the `model/` directory:**
   - YOLOv5 checkpoint (e.g., `yolov5n.pt`)
   - Gaze-LLE checkpoint (e.g., `gazelle_dinov2_vitl14_inout.pt`)

---

## Usage

### Data Preparation
- Organize your dataset as follows:
  ```
  dataset/
      person_01/
          scripted/
              video.mp4
          spontaneous/
              video.mp4
      ...
  ```
- Run `extract_frames.py` to extract frames from videos:
  ```bash
  python extract_frames.py
  ```

### Head Detection
- Run the head detection script to process frames:
  ```bash
  python detect_heads.py
  ```
- This script saves head bounding box data as JSON files.

### Blink Detection
- Execute the blink detection module (if separate) to generate blink statistics:
  ```bash
  python blink_detection.py
  ```
- Results are saved in a CSV file (e.g., `blink_detection.csv`).

### Gaze Estimation
- Run `estimate_gaze.py` to generate gaze heatmaps and corresponding JSON files:
  ```bash
  python estimate_gaze.py
  ```

### Feature Extraction
- Use the feature extraction script to process gaze JSON files and produce `gaze_features.csv`:
  ```bash
  python feature_extraction.py
  ```

### Behavior Classification
- Train the classifier using the combined gaze and blink features:
  ```bash
  python model.py
  ```
- The trained model and label encoder will be saved in the `model/` directory.

### Prediction
- To predict behavior on new data, run:
  ```bash
  python prediction.py
  ```

### Desktop Application
- Launch the integrated PyQt5 application:
  ```bash
  python main.py
  ```
- The GUI allows you to:
  - Upload videos (via file or Google Drive links)
  - Process videos (frame extraction, head detection, blink detection, gaze estimation)
  - Extract features and run predictions
  - View overlayed results on video playback

---

## Results and Evaluation

- **Classifier Performance:**  
  The system achieved a ROC AUC of approximately 0.70, indicating strong discriminative power.
  
- **Confusion Matrix and Classification Report:**  
  Detailed evaluation metrics (precision, recall, F1-score) are generated during training.
  
  
- **Feature Importance:**  
  Analysis identified that gaze entropy, average gaze distance, and fixation/saccade counts significantly impact classification performance.


---

## Future Work

- **Enhancements to Real-Time Processing:**  
  Optimizing the pipeline for live video analysis.
  
- **Incorporating Additional Modalities:**  
  Integrating facial expression analysis to complement gaze and blink data.
  
- **User Studies:**  
  Further evaluation in real-world interview settings to validate system performance.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes with clear messages.
4. Open a pull request describing your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

We thank Prof. Amine NAIT-ALI for his invaluable guidance. We also acknowledge the contributions of the open source communities behind YOLOv5, MediaPipe, and Gaze-LLE.

---

## References

1. Ryan, F., Bati, A., Lee, S., Bolya, D., Hoffman, J., \& Rehg, J. M. (2024). *Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders*. arXiv preprint arXiv:2412.09586.
2. Jocher, G. et al. (2020). *YOLOv5*. [GitHub repository](https://github.com/ultralytics/yolov5).
3. Google. (n.d.). *MediaPipe Face Mesh*. Retrieved from [https://google.github.io/mediapipe/solutions/face_mesh.html](https://google.github.io/mediapipe/solutions/face_mesh.html).
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
5. Ryan, F. (2024). Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders. GitHub repository. https://github.com/fkryan/gazelle
---
