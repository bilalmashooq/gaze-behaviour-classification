# scripts/extract_frames.py

import cv2
import os
from pathlib import Path


def extract_frames(video_path, output_dir, fps=5):
    os.makedirs(output_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print(f"Warning: FPS not detected for {video_path}. Defaulting to 5 FPS.")
        video_fps = fps
    frame_interval = int(video_fps / fps)
    count = 0
    saved_count = 0
    success, image = vidcap.read()
    while success:
        if count % frame_interval == 0:
            frame_filename = f"frame_{saved_count:04d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, image)
            saved_count += 1
        count += 1
        success, image = vidcap.read()
    vidcap.release()
    print(f"Extracted {saved_count} frames from {video_path}.")


if __name__ == "__main__":
    video_files = [
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\1R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\1S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\2R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\2S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\3R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\3S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\4R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\4S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\5R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\5S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\6R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\6S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\7R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\7S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\8R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\8S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\9R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\9S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\10R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\10S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\11R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\11S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\12R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\13R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\13S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\14R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\14S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\15R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\15S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\16R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\16S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\17R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\17S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\18R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\18S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\19R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\19S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\20R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\20S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\21R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\21S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\22R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\22S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\23R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\23S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\24R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\24S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\25R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\25S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\26R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\26S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\27R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\27S.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\28R.mp4",
        r"C:\Users\muham\PycharmProjects\Project_eye_gaze\videos\28S.mp4",
    ]

    for video_path in video_files:
        video_name = Path(video_path).stem  # e.g., '5S'
        label = 'scripted' if video_name.endswith('S') else 'spontaneous'
        person_id = f"person_{video_name[:-1]}"  # e.g., 'person_5'
        output_dir = f"../processed/{person_id}/{label}/frames"
        extract_frames(video_path, output_dir, fps=5)
print(f"Saving frames to: {output_dir}")
print(f"Extracting frames from: {video_path}")