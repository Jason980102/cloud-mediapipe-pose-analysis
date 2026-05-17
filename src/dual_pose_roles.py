import cv2
import mediapipe as mp
import math
from collections import deque

# =========================
# Paths
# =========================
input_video_path = r"C:\Users\tc\gitRestore\cloud-mediapipe-pose-analysis\data\input1.mp4"
output_video_path = r"C:\Users\tc\gitRestore\cloud-mediapipe-pose-analysis\results\output_dual_pose_roles.mp4"

# =========================
# MediaPipe setup
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose_left = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose_right = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# Landmark indices
# =========================
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# =========================
# Utility functions
# =========================
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def point_speed(curr, prev, dt=1.0):
    if curr is None or prev is None:
        return 0.0
    return euclidean(curr, prev) / dt

def midpoint(p1, p2):
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

def toward_opponent_score(curr_ankle, prev_ankle, opponent_center):
    if curr_ankle is None or prev_ankle is None or opponent_center is None:
        return 0.0
    prev_dist = euclidean(prev_ankle, opponent_center)
    curr_dist = euclidean(curr_ankle, opponent_center)
    return prev_dist - curr_dist  # positive = moving toward opponent

def extract_pose_points(results, roi_offset_x=0, roi_offset_y=0):
    """
    Extract selected landmarks from MediaPipe result and convert them
    into absolute frame coordinates.
    """
    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark

    def get_xy(idx):
        return (
            lm[idx].x,  # normalized within ROI for now
            lm[idx].y,
            lm[idx].visibility
        )

    points_norm = {
        "left_shoulder": get_xy(LEFT_SHOULDER),
        "right_shoulder": get_xy(RIGHT_SHOULDER),
        "left_hip": get_xy(LEFT_HIP),
        "right_hip": get_xy(RIGHT_HIP),
        "left_knee": get_xy(LEFT_KNEE),
        "right_knee": get_xy(RIGHT_KNEE),
        "left_ankle": get_xy(LEFT_ANKLE),
        "right_ankle": get_xy(RIGHT_ANKLE),
    }

    return points_norm

def to_abs_points(points_norm, roi_x, roi_y, roi_w, roi_h, visibility_threshold=0.4):
    """
    Convert normalized ROI coordinates to absolute frame coordinates.
    """
    if points_norm is None:
        return None

    points_abs = {}
    for key, (x, y, vis) in points_norm.items():
        if vis < visibility_threshold:
            points_abs[key] = None
        else:
            abs_x = int(roi_x + x * roi_w)
            abs_y = int(roi_y + y * roi_h)
            points_abs[key] = (abs_x, abs_y)

    return points_abs

def get_body_center(points):
    if points is None:
        return None

    shoulder_center = midpoint(points.get("left_shoulder"), points.get("right_shoulder"))
    hip_center = midpoint(points.get("left_hip"), points.get("right_hip"))

    if shoulder_center and hip_center:
        return midpoint(shoulder_center, hip_center)
    return shoulder_center or hip_center

def leg_extension(points, side="left"):
    if points is None:
        return 0.0

    hip = points.get(f"{side}_hip")
    ankle = points.get(f"{side}_ankle")
    if hip is None or ankle is None:
        return 0.0
    return euclidean(hip, ankle)

def compute_kick_score(curr_points, prev_points, opponent_center):
    if curr_points is None or prev_points is None:
        return 0.0

    left_ankle_speed = point_speed(curr_points.get("left_ankle"), prev_points.get("left_ankle"))
    right_ankle_speed = point_speed(curr_points.get("right_ankle"), prev_points.get("right_ankle"))
    max_ankle_speed = max(left_ankle_speed, right_ankle_speed)

    left_knee_speed = point_speed(curr_points.get("left_knee"), prev_points.get("left_knee"))
    right_knee_speed = point_speed(curr_points.get("right_knee"), prev_points.get("right_knee"))
    max_knee_speed = max(left_knee_speed, right_knee_speed)

    curr_ext_left = leg_extension(curr_points, "left")
    curr_ext_right = leg_extension(curr_points, "right")
    prev_ext_left = leg_extension(prev_points, "left")
    prev_ext_right = leg_extension(prev_points, "right")

    max_extension_change = max(
        curr_ext_left - prev_ext_left,
        curr_ext_right - prev_ext_right
    )

    toward_score_left = toward_opponent_score(
        curr_points.get("left_ankle"),
        prev_points.get("left_ankle"),
        opponent_center
    )
    toward_score_right = toward_opponent_score(
        curr_points.get("right_ankle"),
        prev_points.get("right_ankle"),
        opponent_center
    )
    max_toward_score = max(toward_score_left, toward_score_right)

    # slightly penalize large torso movement
    curr_center = get_body_center(curr_points)
    prev_center = get_body_center(prev_points)
    torso_speed = point_speed(curr_center, prev_center)

    kick_score = (
        0.40 * max_ankle_speed +
        0.30 * max_knee_speed +
        0.20 * max_extension_change +
        0.10 * max_toward_score -
        0.10 * torso_speed
    )

    return kick_score

def draw_role_label(frame, text, position, color):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA
    )

# =========================
# Video setup
# =========================
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Cannot open input video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Sliding window for smoothing
window_size = 10
left_scores = deque(maxlen=window_size)
right_scores = deque(maxlen=window_size)

prev_left_points = None
prev_right_points = None

stable_left_role = "UNKNOWN"
stable_right_role = "UNKNOWN"

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    h, w, _ = frame.shape

    # Split frame into left and right ROIs
    mid_x = w // 2

    left_roi = frame[:, :mid_x]
    right_roi = frame[:, mid_x:]

    # MediaPipe expects RGB
    left_rgb = cv2.cvtColor(left_roi, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_roi, cv2.COLOR_BGR2RGB)

    left_results = pose_left.process(left_rgb)
    right_results = pose_right.process(right_rgb)

    # Draw landmarks onto ROIs
    if left_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            left_roi,
            left_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    if right_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            right_roi,
            right_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # Extract key points in absolute frame coords
    left_points_norm = extract_pose_points(left_results)
    right_points_norm = extract_pose_points(right_results)

    left_points = to_abs_points(left_points_norm, 0, 0, mid_x, h)
    right_points = to_abs_points(right_points_norm, mid_x, 0, w - mid_x, h)

    left_center = get_body_center(left_points)
    right_center = get_body_center(right_points)

    left_score = compute_kick_score(left_points, prev_left_points, right_center)
    right_score = compute_kick_score(right_points, prev_right_points, left_center)

    left_scores.append(left_score)
    right_scores.append(right_score)

    avg_left_score = sum(left_scores) / len(left_scores) if left_scores else 0.0
    avg_right_score = sum(right_scores) / len(right_scores) if right_scores else 0.0

    # Hysteresis: only switch roles if difference is meaningful
    score_margin = 5.0

    if avg_left_score - avg_right_score > score_margin:
        stable_left_role = "KICKER"
        stable_right_role = "PARTNER"
    elif avg_right_score - avg_left_score > score_margin:
        stable_left_role = "PARTNER"
        stable_right_role = "KICKER"
    # else keep previous stable roles

    # Draw split line
    cv2.line(frame, (mid_x, 0), (mid_x, h), (255, 255, 255), 2)

    # Draw role labels
    draw_role_label(
        frame,
        f"LEFT: {stable_left_role} | score={avg_left_score:.1f}",
        (20, 40),
        (0, 255, 0) if stable_left_role == "KICKER" else (0, 200, 255)
    )

    draw_role_label(
        frame,
        f"RIGHT: {stable_right_role} | score={avg_right_score:.1f}",
        (mid_x + 20, 40),
        (0, 255, 0) if stable_right_role == "KICKER" else (0, 200, 255)
    )

    # Save previous frame points
    prev_left_points = left_points
    prev_right_points = right_points

    out.write(frame)

    if frame_idx % 30 == 0:
        print(f"Processed {frame_idx} frames... "
              f"left_score={avg_left_score:.2f}, right_score={avg_right_score:.2f}")

cap.release()
out.release()
pose_left.close()
pose_right.close()

print("Done!")
print(f"Output video saved to: {output_video_path}")