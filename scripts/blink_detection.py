# # scripts to detect blinks in videos using MediaPipe Face Mesh
# # and calculate blink rate in blinks per minute (bpm)
#
# import os
# import csv
# import cv2
# import mediapipe as mp
# import numpy as np
# import matplotlib.pyplot as plt
#
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
#                                  max_num_faces=1,
#                                  refine_landmarks=True,
#                                  min_detection_confidence=0.5,
#                                  min_tracking_confidence=0.5)
#
# # Updated eye landmarks (MediaPipe 0.10.1+)
# LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
#
# def eye_aspect_ratio(eye_landmarks):
#     """Improved EAR calculation with error handling"""
#     try:
#         vertical1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
#         vertical2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
#         horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
#         return (vertical1 + vertical2) / (2.0 * horizontal)
#     except Exception as e:
#         return None
#
# def detect_blinks(video_path, ear_threshold=0.18, consecutive_frames=2, debug=False):
#     """Enhanced blink detection with visualization and debugging"""
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return 0, 0, []
#
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     blink_count = 0
#     ear_history = []
#     consecutive_low_ear = 0
#     in_blink = False
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(frame_rgb)
#
#         ear = None
#         if results.multi_face_landmarks:
#             try:
#                 landmarks = results.multi_face_landmarks[0].landmark
#
#                 # Get eye coordinates in image space
#                 left_eye = np.array([(landmarks[i].x * width, landmarks[i].y * height)
#                                    for i in LEFT_EYE_INDICES])
#                 right_eye = np.array([(landmarks[i].x * width, landmarks[i].y * height)
#                                     for i in RIGHT_EYE_INDICES])
#
#                 ear_left = eye_aspect_ratio(left_eye)
#                 ear_right = eye_aspect_ratio(right_eye)
#
#                 if ear_left and ear_right:
#                     ear = (ear_left + ear_right) / 2.0
#                     ear_history.append(ear)
#
#                     # Blink state machine
#                     if ear < ear_threshold:
#                         consecutive_low_ear += 1
#                         if not in_blink and consecutive_low_ear >= consecutive_frames:
#                             blink_count += 1
#                             in_blink = True
#                     else:
#                         consecutive_low_ear = 0
#                         in_blink = False
#
#                 # Visual debug
#                 if debug:
#                     for point in left_eye:
#                         cv2.circle(frame, tuple(map(int, point)), 1, (0,255,0), -1)
#                     for point in right_eye:
#                         cv2.circle(frame, tuple(map(int, point)), 1, (0,255,0), -1)
#                     cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
#                     cv2.imshow("Debug", frame)
#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         break
#
#             except Exception as e:
#                 continue
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     # Calculate blink rate
#     duration = len(ear_history) / fps
#     blink_rate = blink_count / (duration / 60) if duration > 0 else 0
#
#     return blink_count, round(blink_rate, 2), ear_history
#
# def process_videos_folder(input_folder, output_csv, debug=False):
#     """Enhanced processing with EAR visualization"""
#     video_files = [f for f in os.listdir(input_folder)
#                   if f.lower().endswith(('.mp4', '.avi', '.mov'))]
#
#     with open(output_csv, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Video File', 'Blink Count', 'Blink Rate (bpm)', 'EAR Plot'])
#
#         for video_file in video_files:
#             video_path = os.path.join(input_folder, video_file)
#             print(f"\nProcessing: {video_file}")
#
#             blink_count, blink_rate, ear_history = detect_blinks(
#                 video_path,
#                 ear_threshold=0.18,  # Lower threshold for subtle blinks
#                 consecutive_frames=2,  # Fewer frames for faster blinks
#                 debug=debug
#             )
#
#             # Generate EAR plot
#             plot_path = f"ear_plots/{video_file}_ear.png"
#             plt.figure(figsize=(10, 4))
#             plt.plot(ear_history)
#             plt.axhline(y=0.18, color='r', linestyle='--')
#             plt.title(f"EAR History - {video_file}")
#             plt.ylabel("EAR")
#             plt.xlabel("Frames")
#             plt.savefig(plot_path)
#             plt.close()
#
#             writer.writerow([video_file, blink_count, blink_rate, plot_path])
#
#     print(f"\nProcessing complete. Results saved to {output_csv}")
#
# # Usage
# if __name__ == "__main__":
#     input_folder = r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos"
#     output_csv = "blink_results_1.csv"
#
#     # Create directory for EAR plots
#     os.makedirs("ear_plots", exist_ok=True)
#
#     # Set debug=True to visualize processing
#     process_videos_folder(input_folder, output_csv, debug=False)


# Script to detect blinks in videos using MediaPipe Face Mesh
# and calculate blink rate in blinks per minute (bpm)
# Outputs a CSV with person_id, label, blink_count, and blink_rate_bpm

import os
import csv
import cv2
import mediapipe as mp
import numpy as np
import re

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define eye landmarks (MediaPipe 0.10.1+)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]


def eye_aspect_ratio(eye_landmarks):
    """
    Calculate the Eye Aspect Ratio (EAR) for a given eye.

    Parameters:
    - eye_landmarks: numpy array of shape (6, 2) containing the (x, y) coordinates of the eye landmarks.

    Returns:
    - EAR value or None if calculation fails.
    """
    try:
        vertical1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        return (vertical1 + vertical2) / (2.0 * horizontal)
    except Exception:
        return None


def detect_blinks(video_path, ear_threshold=0.18, consecutive_frames=2):
    """
    Detect blinks in a video and calculate blink count and blink rate (BPM).

    Parameters:
    - video_path: Path to the video file.
    - ear_threshold: EAR below which a blink is considered to have started.
    - consecutive_frames: Number of consecutive frames the EAR must be below the threshold to count as a blink.

    Returns:
    - blink_count: Total number of blinks detected.
    - blink_rate: Blink rate in blinks per minute (bpm).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default to 30 if FPS is not available
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = frame_count / fps
    duration_minutes = duration_seconds / 60

    blink_count = 0
    consecutive_low_ear = 0
    in_blink = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            try:
                landmarks = results.multi_face_landmarks[0].landmark

                # Extract eye landmarks
                left_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in LEFT_EYE_INDICES])
                right_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in RIGHT_EYE_INDICES])

                # Convert normalized coordinates to pixel coordinates
                height, width, _ = frame.shape
                left_eye *= [width, height]
                right_eye *= [width, height]

                # Calculate EAR for both eyes
                ear_left = eye_aspect_ratio(left_eye)
                ear_right = eye_aspect_ratio(right_eye)

                if ear_left and ear_right:
                    ear = (ear_left + ear_right) / 2.0

                    # Blink detection logic
                    if ear < ear_threshold:
                        consecutive_low_ear += 1
                        if not in_blink and consecutive_low_ear >= consecutive_frames:
                            blink_count += 1
                            in_blink = True
                    else:
                        consecutive_low_ear = 0
                        in_blink = False

            except Exception:
                # If there's an error processing landmarks, skip this frame
                continue

    cap.release()

    # Calculate blink rate
    blink_rate = blink_count / duration_minutes if duration_minutes > 0 else 0

    return blink_count, round(blink_rate, 2)


def process_videos_folder(input_folder, output_csv):
    """
    Process all videos in the input folder to detect blinks and save the results to a CSV.

    Parameters:
    - input_folder: Path to the folder containing video files.
    - output_csv: Path to the output CSV file.
    """
    # List of supported video file extensions
    video_extensions = ('.mp4', '.avi', '.mov')

    # Compile regex pattern to extract person_id and label
    pattern = re.compile(r'(\d+)([RS])', re.IGNORECASE)

    # Initialize list to hold results
    results = []

    # Iterate over all files in the input folder
    for video_file in os.listdir(input_folder):
        if not video_file.lower().endswith(video_extensions):
            continue  # Skip non-video files

        video_path = os.path.join(input_folder, video_file)
        print(f"Processing: {video_file}")

        # Extract person_id and label from filename
        match = pattern.match(os.path.splitext(video_file)[0])
        if match:
            person_number = match.group(1)
            label_code = match.group(2).upper()
            person_id = f"person_{person_number}"
            label = 'scripted' if label_code == 'R' else 'spontaneous'
        else:
            print(f"Filename {video_file} does not match the expected pattern. Skipping.")
            continue

        # Detect blinks
        blink_count, blink_rate = detect_blinks(
            video_path,
            ear_threshold=0.18,  # Lower threshold for subtle blinks
            consecutive_frames=2  # Fewer frames for faster blinks
        )

        # Append the result
        results.append([person_id, label, blink_count, blink_rate])

    # Write results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['person_id', 'label', 'blink_count', 'blink_rate_bpm'])
        writer.writerows(results)

    print(f"\nProcessing complete. Results saved to {output_csv}")


# Usage
if __name__ == "__main__":
    input_folder = r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos"
    output_csv = "blink_results.csv"

    process_videos_folder(input_folder, output_csv)
