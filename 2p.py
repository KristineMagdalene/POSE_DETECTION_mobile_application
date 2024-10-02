import cv2
import mediapipe as mp
import numpy as np
from moviepy.editor import VideoFileClip

# Setup MediaPipe instance
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize Pose with slightly increased confidence thresholds and the simplest model complexity
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7,
    model_complexity=0
)

def process_frame_for_pose(image):
    # Use a balanced resolution
    image = cv2.resize(image, (640, 360))  # Adjusted resolution
    height, width, _ = image.shape
    left_half = image[:, :width//2]
    right_half = image[:, width//2:]

    processed_left = process_single_half(left_half)
    processed_right = process_single_half(right_half)

    # Concatenate processed halves back together
    processed_frame = cv2.hconcat([processed_left, processed_right])
    return processed_frame

def process_single_half(half):
    half = np.array(half, copy=True)
    results = pose.process(cv2.cvtColor(half, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(half, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return half

def process_video(video_file):
    clip = VideoFileClip(video_file)
    original_fps = clip.fps
    processed_clip = clip.fl_image(process_frame_for_pose)
    # Maintain the original frame rate
    processed_clip.write_videofile("processed_output.mp4", fps=original_fps, audio_codec='aac')

video_path = 'JOTAprototypeHD - green.mp4'
process_video(video_path)
