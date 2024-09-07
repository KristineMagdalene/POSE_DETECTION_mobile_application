import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    if a is None or b is None or c is None or len(a) != 2 or len(b) != 2 or len(c) != 2:
        print("One of the input arrays is None or does not contain exactly two elements.")
        return None
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def compare_poses(landmarks1, landmarks2):
    if len(landmarks1) < mp_pose.PoseLandmark.LEFT_WRIST.value or len(landmarks2) < mp_pose.PoseLandmark.LEFT_WRIST.value:
        print("Not enough landmarks present in one of the videos.")
        return False
    pose1 = [landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
             landmarks1[mp_pose.PoseLandmark.LEFT_ELBOW.value],
             landmarks1[mp_pose.PoseLandmark.LEFT_WRIST.value]]
    pose2 = [landmarks2[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
             landmarks2[mp_pose.PoseLandmark.LEFT_ELBOW.value],
             landmarks2[mp_pose.PoseLandmark.LEFT_WRIST.value]]

    for pt in pose1 + pose2:
        if pt.x is None or pt.y is None:
            print("One of the points in a pose is None.")
            return False

    pose1 = [(pt.x, pt.y) for pt in pose1]
    pose2 = [(pt.x, pt.y) for pt in pose2]

    angle1 = calculate_angle(pose1[0], pose1[1], pose1[2])
    angle2 = calculate_angle(pose2[0], pose2[1], pose2[2])
    if angle1 is not None and angle2 is not None and abs(angle1 - angle2) < 10:
        return True
    return False

# Setup video captures
cap1 = cv2.VideoCapture('justdance.mp4')
cap2 = cv2.VideoCapture('justdance2.mp4')

# Get the original frame rates of both videos
fps1 = cap1.get(cv2.CAP_PROP_FPS)
fps2 = cap2.get(cv2.CAP_PROP_FPS)

# Set a fixed speed of 500 milliseconds per frame
wait_time = int(200 / min(fps1, fps2))

# Setup Mediapipe Pose instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("End of video or failed to grab frame")
            break

        # Process first video
        image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        image1.flags.writeable = False
        results1 = pose.process(image1)

        # Process second video
        image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        image2.flags.writeable = False
        results2 = pose.process(image2)

        # Draw pose landmarks
        if results1.pose_landmarks:
            mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        if results2.pose_landmarks:
            mp_drawing.draw_landmarks(frame2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        if results1.pose_landmarks and results2.pose_landmarks:
            landmarks1 = [pt for pt in results1.pose_landmarks.landmark]
            landmarks2 = [pt for pt in results2.pose_landmarks.landmark]

            if compare_poses(landmarks1, landmarks2):
                print("YOU DID GREAT")

        # Display the frames
        cv2.imshow('Video 1', frame1)
        cv2.imshow('Video 2', frame2)

        # Use a fixed wait time of 500 milliseconds
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
