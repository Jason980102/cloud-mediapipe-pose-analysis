import cv2
import mediapipe as mp
import os
import math

def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def point_speed(curr, prev, dt=1.0):
    return euclidean(curr, prev) / dt

def leg_extension(hip, ankle):
    return euclidean(hip, ankle)

def toward_opponent_score(curr_ankle, prev_ankle, opponent_center):
    prev_dist = euclidean(prev_ankle, opponent_center)
    curr_dist = euclidean(curr_ankle, opponent_center)
    return prev_dist - curr_dist


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

input_video_path = r"C:\Users\tc\gitRestore\cloud-mediapipe-pose-analysis\data\input1.mp4"
output_video_path = r"C:\Users\tc\gitRestore\cloud-mediapipe-pose-analysis\results\output_pose.mp4"

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Cannot open input video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define output writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = pose.process(rgb_frame)

    # Draw pose landmarks on frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    # Write processed frame to output video
    out.write(frame)
    frame_count += 1

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
pose.close()

print(f"Done! Output saved to: {output_video_path}")
