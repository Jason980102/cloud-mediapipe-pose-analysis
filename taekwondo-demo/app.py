import json
import time
import boto3
import oci
import uuid
import urllib.request
from pathlib import Path

import os
import boto3
import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import subprocess
import imageio_ffmpeg
import base64
from botocore.config import Config
import streamlit.components.v1 as components

BUCKET_NAME = "jc-taekwondo-cloud-demo-2026"
REGION = "us-east-2"

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "outputs"

# MediaPipe Tasks model used for multi-person pose detection.
# This is separate from the Random Forest model artifacts.
POSE_MODEL_DIR = BASE_DIR / "mediapipe_models"
POSE_MODEL_PATH = POSE_MODEL_DIR / "pose_landmarker_heavy.task"
POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
POSE_MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILES = [
    "taekwondo_motion_classifier_random_forest.joblib",
    "taekwondo_label_encoder.joblib",
    "taekwondo_feature_columns.json",
    "taekwondo_model_summary.json",
    "taekwondo_confusion_matrix.png",
]

LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def download_model_from_s3():
    s3 = boto3.client("s3", region_name=REGION)

    for filename in MODEL_FILES:
        local_path = MODEL_DIR / filename
        if not local_path.exists():
            s3.download_file(BUCKET_NAME, f"model/{filename}", str(local_path))


@st.cache_resource
def load_model_objects():
    download_model_from_s3()

    model = joblib.load(MODEL_DIR / "taekwondo_motion_classifier_random_forest.joblib")
    encoder = joblib.load(MODEL_DIR / "taekwondo_label_encoder.joblib")

    with open(MODEL_DIR / "taekwondo_feature_columns.json", "r") as f:
        feature_columns = json.load(f)

    with open(MODEL_DIR / "taekwondo_model_summary.json", "r") as f:
        summary = json.load(f)

    return model, encoder, feature_columns, summary


def display_label(label):
    mapping = {
        "axe_kick": "Axe Kick / High Kick",
        "front_kick": "Front Kick",
        "roundhouse_kick": "Roundhouse Kick",
        "side_kick": "Side Kick",
    }
    return mapping.get(str(label), str(label))


# ============================================================
# Multi-person MediaPipe extraction and main actor selection
# ============================================================

LEG_LANDMARK_IDX = [25, 26, 27, 28, 29, 30, 31, 32]


def download_pose_landmarker_model():
    """
    Download MediaPipe Pose Landmarker model if it is not already available.
    This enables two-person detection, which mp.solutions.pose does not support.
    """
    if POSE_MODEL_PATH.exists() and POSE_MODEL_PATH.stat().st_size > 0:
        return POSE_MODEL_PATH

    urllib.request.urlretrieve(POSE_LANDMARKER_URL, str(POSE_MODEL_PATH))
    return POSE_MODEL_PATH


def empty_keypoint_row(video_path, frame_idx, timestamp_ms):
    row = {
        "video_name": Path(video_path).name,
        "video_path": str(video_path),
        "frame_idx": frame_idx,
        "timestamp_ms": timestamp_ms,
    }

    for name in LANDMARK_NAMES:
        row[f"{name}_x"] = np.nan
        row[f"{name}_y"] = np.nan
        row[f"{name}_z"] = np.nan
        row[f"{name}_visibility"] = np.nan

    return row


def fill_row_with_landmarks(row, landmarks):
    if landmarks is None:
        return row

    for i, lm in enumerate(landmarks):
        if i >= len(LANDMARK_NAMES):
            break

        name = LANDMARK_NAMES[i]
        row[f"{name}_x"] = lm.x
        row[f"{name}_y"] = lm.y
        row[f"{name}_z"] = lm.z
        row[f"{name}_visibility"] = getattr(lm, "visibility", np.nan)

    return row


def pose_to_array(landmarks):
    if landmarks is None:
        return None

    arr = []

    for lm in landmarks:
        arr.append([
            lm.x,
            lm.y,
            lm.z,
            getattr(lm, "visibility", 1.0),
        ])

    return np.array(arr, dtype=float)


def get_pose_center(landmarks, visibility_threshold=0.3):
    if landmarks is None:
        return None

    xs = []
    ys = []

    for lm in landmarks:
        visibility = getattr(lm, "visibility", 1.0)

        if visibility > visibility_threshold:
            xs.append(lm.x)
            ys.append(lm.y)

    if len(xs) == 0:
        return None

    return np.array([float(np.mean(xs)), float(np.mean(ys))])


def assign_poses_to_tracks(poses, prev_centers):
    """
    Assign up to two detected poses to two stable tracks.

    The first frame initializes track IDs from left to right. This is only
    for stable tracking IDs; it is not the final trainer selection.
    Later frames are assigned by nearest previous center.
    """
    assigned = [None, None]
    pose_infos = []

    for pose in poses:
        center = get_pose_center(pose)

        if center is not None:
            pose_infos.append({
                "pose": pose,
                "center": center,
            })

    if len(pose_infos) == 0:
        return assigned

    if prev_centers[0] is None and prev_centers[1] is None:
        pose_infos.sort(key=lambda item: item["center"][0])
        assigned[0] = pose_infos[0]["pose"]

        if len(pose_infos) > 1:
            assigned[1] = pose_infos[1]["pose"]

        return assigned

    used_pose_indices = set()

    for track_id in [0, 1]:
        if prev_centers[track_id] is None:
            continue

        best_idx = None
        best_dist = float("inf")

        for i, info in enumerate(pose_infos):
            if i in used_pose_indices:
                continue

            dist = np.linalg.norm(info["center"] - prev_centers[track_id])

            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is not None:
            assigned[track_id] = pose_infos[best_idx]["pose"]
            used_pose_indices.add(best_idx)

    for i, info in enumerate(pose_infos):
        if i in used_pose_indices:
            continue

        if assigned[0] is None:
            assigned[0] = info["pose"]
        elif assigned[1] is None:
            assigned[1] = info["pose"]

    return assigned


def compute_kicking_actor_score(track_arrays, visibility_threshold=0.3):
    """
    Compute a kicking-related score for one tracked person.

    The score favors the person with stronger lower-body motion, higher
    foot extension from the hip, wider leg spread, and upward kick height.
    This is used to avoid drawing the pad holder when two people appear.
    """
    total_leg_motion = 0.0
    max_leg_speed = 0.0
    max_foot_extension = 0.0
    max_leg_spread = 0.0
    max_kick_height = 0.0
    valid_frames = 0

    prev = None

    for arr in track_arrays:
        if arr is None:
            prev = None
            continue

        valid_frames += 1

        hip_points = []

        for idx in [23, 24]:
            x, y, _, v = arr[idx]

            if not np.isnan(x) and not np.isnan(y) and v > visibility_threshold:
                hip_points.append(np.array([x, y]))

        hip_center = np.mean(hip_points, axis=0) if len(hip_points) > 0 else None

        foot_points = []

        for idx in [27, 28, 29, 30, 31, 32]:
            x, y, _, v = arr[idx]

            if not np.isnan(x) and not np.isnan(y) and v > visibility_threshold:
                point = np.array([x, y])
                foot_points.append(point)

                if hip_center is not None:
                    extension = float(np.linalg.norm(point - hip_center))
                    max_foot_extension = max(max_foot_extension, extension)

                    # In image coordinates, smaller y means higher body position.
                    kick_height = float(hip_center[1] - point[1])
                    max_kick_height = max(max_kick_height, kick_height)

        if len(foot_points) >= 2:
            for i in range(len(foot_points)):
                for j in range(i + 1, len(foot_points)):
                    spread = float(np.linalg.norm(foot_points[i] - foot_points[j]))
                    max_leg_spread = max(max_leg_spread, spread)

        if prev is not None:
            for idx in LEG_LANDMARK_IDX:
                x1, y1, _, v1 = prev[idx]
                x2, y2, _, v2 = arr[idx]

                valid = (
                    not np.isnan(x1) and not np.isnan(y1)
                    and not np.isnan(x2) and not np.isnan(y2)
                    and v1 > visibility_threshold
                    and v2 > visibility_threshold
                )

                if valid:
                    distance = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
                    total_leg_motion += distance
                    max_leg_speed = max(max_leg_speed, distance)

        prev = arr

    score = (
        total_leg_motion * 1.0
        + max_leg_speed * 10.0
        + max_foot_extension * 4.0
        + max_leg_spread * 3.0
        + max_kick_height * 4.0
        + valid_frames * 0.002
    )

    return score


def extract_pose_keypoints(
    video_path,
    frame_stride=3,
    max_frames=240,
    actor_selection_mode="auto_kicking_motion",
):
    """
    Extract MediaPipe keypoints from an uploaded video.

    This version uses MediaPipe Tasks PoseLandmarker with num_poses=2.
    It tracks up to two people and selects the trainer / performer using
    kicking-related lower-body motion. Optional track override is provided
    for demo recovery when automatic selection is visually incorrect.
    """
    video_path = Path(video_path)
    pose_model_path = download_pose_landmarker_model()

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)

    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_records = []
    track_arrays = [[], []]
    prev_centers = [None, None]

    frame_idx = 0
    processed = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            if processed >= max_frames:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb,
            )

            timestamp_ms = int((frame_idx / fps) * 1000)

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            poses = result.pose_landmarks if result.pose_landmarks else []

            assigned = assign_poses_to_tracks(
                poses=poses,
                prev_centers=prev_centers,
            )

            for track_id in [0, 1]:
                pose = assigned[track_id]
                arr = pose_to_array(pose)

                track_arrays[track_id].append(arr)

                center = get_pose_center(pose)

                if center is not None:
                    prev_centers[track_id] = center

            frame_records.append({
                "frame_idx": frame_idx,
                "timestamp_ms": timestamp_ms,
                "track0": assigned[0],
                "track1": assigned[1],
            })

            processed += 1
            frame_idx += 1

    cap.release()

    track0_score = compute_kicking_actor_score(track_arrays[0])
    track1_score = compute_kicking_actor_score(track_arrays[1])

    if actor_selection_mode == "track_0":
        selected_track_id = 0
        actor_selection_method = "manual_track_0"
    elif actor_selection_mode == "track_1":
        selected_track_id = 1
        actor_selection_method = "manual_track_1"
    else:
        selected_track_id = 0 if track0_score >= track1_score else 1
        actor_selection_method = "auto_kicking_motion_score"

    print("Main Actor Selection")
    print("Track 0 kicking score:", track0_score)
    print("Track 1 kicking score:", track1_score)
    print("Actor selection mode:", actor_selection_mode)
    print("Selected track:", selected_track_id)

    rows = []
    detected = 0

    for rec in frame_records:
        row = empty_keypoint_row(
            video_path=video_path,
            frame_idx=rec["frame_idx"],
            timestamp_ms=rec["timestamp_ms"],
        )

        selected_pose = rec[f"track{selected_track_id}"]

        if selected_pose is not None:
            detected += 1
            row = fill_row_with_landmarks(row, selected_pose)

        rows.append(row)

    df = pd.DataFrame(rows)

    meta = {
        "total_video_frames": total_video_frames,
        "processed_frames": processed,
        "detected_pose_frames": detected,
        "pose_detection_rate": detected / processed if processed > 0 else 0,
        "fps": fps,
        "width": width,
        "height": height,
        "selected_track_id": selected_track_id,
        "track0_kicking_score": track0_score,
        "track1_kicking_score": track1_score,
        "actor_selection_method": actor_selection_method,
    }

    return df, meta


def convert_video_for_browser(input_path, output_path):
    """
    Convert uploaded video such as AVI/MOV to browser-playable MP4.
    This is only for frontend preview. The original video is still used for AI analysis.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(input_path),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "28",
        "-an",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    st.warning("Video preview conversion failed. The AI analysis can still run.")
    return input_path

def extract_features_from_keypoints(df):
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.interpolate(limit_direction="both")
    numeric_df = numeric_df.fillna(0)

    features = {}
    features["n_frames"] = len(numeric_df)
    features["n_numeric_cols"] = numeric_df.shape[1]

    for col in numeric_df.columns:
        s = numeric_df[col].astype(float)

        features[f"{col}_mean"] = s.mean()
        features[f"{col}_std"] = s.std()
        features[f"{col}_min"] = s.min()
        features[f"{col}_max"] = s.max()
        features[f"{col}_range"] = s.max() - s.min()
        features[f"{col}_median"] = s.median()

        motion = s.diff().fillna(0).abs()
        features[f"{col}_motion_mean"] = motion.mean()
        features[f"{col}_motion_max"] = motion.max()
        features[f"{col}_motion_std"] = motion.std()

    body_pairs = [
        ("right_foot_index_x", "right_foot_index_y"),
        ("left_foot_index_x", "left_foot_index_y"),
        ("right_ankle_x", "right_ankle_y"),
        ("left_ankle_x", "left_ankle_y"),
        ("right_knee_x", "right_knee_y"),
        ("left_knee_x", "left_knee_y"),
        ("right_wrist_x", "right_wrist_y"),
        ("left_wrist_x", "left_wrist_y"),
        ("right_hip_x", "right_hip_y"),
        ("left_hip_x", "left_hip_y"),
    ]

    for x_col, y_col in body_pairs:
        if x_col in numeric_df.columns and y_col in numeric_df.columns:
            dx = numeric_df[x_col].diff().fillna(0)
            dy = numeric_df[y_col].diff().fillna(0)
            speed = np.sqrt(dx**2 + dy**2)

            part = x_col.replace("_x", "")
            features[f"{part}_speed_mean"] = speed.mean()
            features[f"{part}_speed_max"] = speed.max()
            features[f"{part}_speed_std"] = speed.std()
            features[f"{part}_x_range"] = numeric_df[x_col].max() - numeric_df[x_col].min()
            features[f"{part}_y_range"] = numeric_df[y_col].max() - numeric_df[y_col].min()

    return features


def predict_motion(features, model, encoder, feature_columns):
    X = pd.DataFrame([features])

    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_columns]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    pred_encoded = model.predict(X)[0]
    pred_label = encoder.inverse_transform([pred_encoded])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(X)[0]))

    return pred_label, confidence


def upload_outputs_to_s3(job_id, files):
    s3 = boto3.client("s3", region_name=REGION)
    uris = []

    for file_path in files:
        file_path = Path(file_path)
        key = f"demo-results/{job_id}/{file_path.name}"
        s3.upload_file(str(file_path), BUCKET_NAME, key)
        uris.append(f"s3://{BUCKET_NAME}/{key}")

    return uris

def upload_outputs_to_oracle(job_id, files):
    oracle_access_key = os.getenv("ORACLE_ACCESS_KEY")
    oracle_secret_key = os.getenv("ORACLE_SECRET_KEY")
    oracle_namespace = os.getenv("ORACLE_NAMESPACE")
    oracle_region = os.getenv("ORACLE_REGION")
    oracle_bucket = os.getenv("ORACLE_BUCKET")

    endpoint_url = (
        f"https://{oracle_namespace}.compat.objectstorage."
        f"{oracle_region}.oraclecloud.com"
    )

    oracle_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=oracle_access_key,
        aws_secret_access_key=oracle_secret_key,
        region_name=oracle_region,
        config=Config(
            signature_version="s3v4",
            s3={
                "addressing_style": "path",
                "payload_signing_enabled": False,
            },
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )

    oracle_uris = {}

    for name, file_path in files.items():
        if file_path is None:
            continue

        file_path = Path(file_path)
        if not file_path.exists():
            continue

        key = f"demo-results/{job_id}/{file_path.name}"
        body = file_path.read_bytes()

        oracle_client.put_object(
            Bucket=oracle_bucket,
            Key=key,
            Body=body,
            ContentLength=len(body),
        )
                
        
        
        oracle_uris[name] = f"oci://{oracle_bucket}/{key}"

    return oracle_uris

st.set_page_config(
    page_title="Taekwondo Motion Classification Demo",
    layout="wide",
)

st.title("Cloud-Based Intelligent Taekwondo Motion Classification System")
st.caption("AWS EC2 frontend + MediaPipe pose extraction + Random Forest AI inference + AWS S3 storage")

model, encoder, feature_columns, summary = load_model_objects()

st.subheader("Cloud Computing Architecture")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("### AWS EC2")
    st.write("Hosts the frontend web application and runs the AI inference pipeline.")

with c2:
    st.markdown("### MediaPipe Pose")
    st.write("Extracts body landmarks and converts video frames into keypoints.")

with c3:
    st.markdown("### AI Classifier")
    st.write("Random Forest model classifies taekwondo techniques from motion features.")

with c4:
    st.markdown("### AWS S3")
    st.write("Stores model artifacts, uploaded videos, keypoints CSV files, and prediction JSON outputs.")

st.divider()

# ============================================================
# MediaPipe Front Kick Example Section
# ============================================================

EXAMPLE_VIDEO_DIR = Path(__file__).parent / "assets" / "examples"

ORIGINAL_VIDEO_PATH = EXAMPLE_VIDEO_DIR / "original_taekwondo_example.mp4"
MEDIAPIPE_OVERLAY_VIDEO_PATH = EXAMPLE_VIDEO_DIR / "mediapipe_pose_overlay_example.mp4"

def render_looping_video(video_path, height=360):
    if not video_path.exists():
        st.warning(f"Video not found: {video_path}")
        return

    video_bytes = video_path.read_bytes()
    encoded_video = base64.b64encode(video_bytes).decode()

    video_html = f"""
    <video
        width="100%"
        height="{height}"
        autoplay
        loop
        muted
        playsinline
        controls
        style="border-radius: 12px;"
    >
        <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """

    components.html(video_html, height=height + 30)

st.header("MediaPipe Pose Analysis Demo")

st.markdown("""
This built-in example demonstrates how the system analyzes a **front kick** video using MediaPipe Pose.

The original video contains a trainer and a partner. To avoid switching between the two people, the system tracks up to two pose candidates and selects the main actor based on lower-body movement.

The person with the stronger leg motion score is treated as the trainer, and only the selected trainer's skeleton is displayed in the MediaPipe overlay video.
""")

st.markdown("### Front Kick Example: Original Video vs. MediaPipe Pose Overlay")

col_original, col_overlay = st.columns(2)

with col_original:
    st.subheader("Original Front Kick Video")

    render_looping_video(ORIGINAL_VIDEO_PATH, height=360)

    st.caption("Original taekwondo front kick training video before MediaPipe processing.")

with col_overlay:
    st.subheader("MediaPipe Pose Overlay")

    render_looping_video(MEDIAPIPE_OVERLAY_VIDEO_PATH, height=360)

    st.caption("MediaPipe pose overlay after selecting the trainer based on leg motion score.")

with st.expander("How does the system select the trainer?"):
    st.markdown("""
When two people appear in the same video, the system does not simply draw every detected skeleton.

Instead, it follows this logic:

1. MediaPipe detects up to two human poses in each frame.
2. The system tracks both pose candidates across the video.
3. It calculates a leg motion score using lower-body landmarks such as knees, ankles, heels, and foot indexes.
4. The person with stronger lower-body movement is selected as the main actor / trainer.
5. The final overlay video displays only the selected trainer's skeleton.

This makes the visualization more stable and prevents the overlay from jumping between the trainer and the partner.
""")

st.markdown("---")

st.subheader("Current Model Status")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Model Source", "AWS S3")

with m2:
    st.metric("Weak-label Accuracy", f"{summary.get('accuracy', 0):.4f}")

with m3:
    st.metric("Training Samples", summary.get("num_samples", "N/A"))

with m4:
    st.metric("Classes", summary.get("num_classes", "N/A"))

st.info(summary.get("accuracy_note", "Accuracy is based on the current validation setup."))

if (MODEL_DIR / "taekwondo_confusion_matrix.png").exists():
    with st.expander("Show Confusion Matrix"):
        st.image(str(MODEL_DIR / "taekwondo_confusion_matrix.png"))

st.divider()

st.subheader("Upload a Taekwondo Video")


def create_mediapipe_pose_preview(video_path, keypoints_df, output_path, max_preview_frames=160):
    """
    Create a browser-playable MP4 preview with MediaPipe keypoints drawn on the video.
    This is only for frontend visualization.
    """
    import cv2
    import pandas as pd
    import subprocess
    import imageio_ffmpeg
    from pathlib import Path

    video_path = Path(video_path)
    output_path = Path(output_path)

    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    if keypoints_df is None or len(keypoints_df) == 0:
        return None

    if "frame_idx" in keypoints_df.columns:
        frame_col = "frame_idx"
    elif "frame" in keypoints_df.columns:
        frame_col = "frame"
    else:
        return None

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 720)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    raw_path = output_path.with_name(output_path.stem + "_raw.mp4")

    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        min(fps, 20),
        (width, height),
    )

    pose_connections = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("left_ankle", "left_foot_index"),
        ("right_ankle", "right_foot_index"),
    ]

    landmark_names = sorted(set([x for pair in pose_connections for x in pair] + ["nose"]))

    def get_point(row, name):
        x_col = f"{name}_x"
        y_col = f"{name}_y"
        v_col = f"{name}_visibility"

        if x_col not in row or y_col not in row:
            return None

        x = row[x_col]
        y = row[y_col]
        v = row[v_col] if v_col in row else 1.0

        if pd.isna(x) or pd.isna(y):
            return None

        if not pd.isna(v) and float(v) < 0.3:
            return None

        return int(float(x) * width), int(float(y) * height)

    df_preview = keypoints_df.sort_values(frame_col).head(max_preview_frames)

    for _, row in df_preview.iterrows():
        frame_idx = int(row[frame_col])

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        for a, b in pose_connections:
            p1 = get_point(row, a)
            p2 = get_point(row, b)

            if p1 is not None and p2 is not None:
                cv2.line(frame, p1, p2, (0, 255, 0), 3)

        for name in landmark_names:
            p = get_point(row, name)

            if p is not None:
                cv2.circle(frame, p, 5, (0, 0, 255), -1)

        writer.write(frame)

    cap.release()
    writer.release()

    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return None

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(raw_path),
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "28",
        str(output_path),
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        raw_path.unlink()
    except Exception:
        pass

    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    return None


uploaded_file = st.file_uploader(
    "Upload a short video clip, preferably 5-15 seconds.",
    type=["mp4", "mov", "avi", "mkv"]
)

frame_stride = st.slider("Frame stride", 1, 10, 3)
max_frames = st.slider("Maximum frames to process", 60, 600, 240, 30)

actor_selection_mode = st.selectbox(
    "Main actor selection for uploaded videos",
    options=[
        "auto_kicking_motion",
        "track_0",
        "track_1",
    ],
    format_func=lambda value: {
        "auto_kicking_motion": "Auto: select performer by kicking motion",
        "track_0": "Manual override: use Track 0",
        "track_1": "Manual override: use Track 1",
    }[value],
    help=(
        "Use Auto for normal analysis. If the preview draws the pad holder instead "
        "of the performer, rerun with Track 0 or Track 1 as a visual/debug override."
    ),
)

if uploaded_file is not None:
    # Save uploaded video temporarily for browser preview
    preview_input_dir = OUTPUT_DIR / "browser_previews"
    preview_input_dir.mkdir(exist_ok=True)

    preview_input_path = preview_input_dir / uploaded_file.name

    with open(preview_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    browser_preview_path = convert_video_for_browser(
        preview_input_path,
        preview_input_dir / f"{Path(uploaded_file.name).stem}_browser_preview.mp4"
    )

    st.subheader("Uploaded Video Preview")
    st.video(str(browser_preview_path))

    if st.button("Run Cloud AI Analysis", type="primary"):
        job_id = str(uuid.uuid4())[:8]
        job_dir = OUTPUT_DIR / f"job_{job_id}"
        job_dir.mkdir(parents=True, exist_ok=True)

        video_path = job_dir / uploaded_file.name

        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        start = time.time()

        with st.spinner("Processing video on AWS EC2..."):
            keypoints_df, video_meta = extract_pose_keypoints(
                video_path,
                frame_stride=frame_stride,
                max_frames=max_frames,
                actor_selection_mode=actor_selection_mode,
            )

            keypoints_path = job_dir / "extracted_keypoints.csv"
            keypoints_df.to_csv(keypoints_path, index=False)
            # Create MediaPipe pose overlay preview video
            pose_preview_path = create_mediapipe_pose_preview(
                video_path=video_path,
                keypoints_df=keypoints_df,
                output_path=job_dir / "mediapipe_pose_preview.mp4",
            )

            features = extract_features_from_keypoints(keypoints_df)
            pred_label, confidence = predict_motion(features, model, encoder, feature_columns)

            elapsed = time.time() - start

            result = {
                "job_id": job_id,
                "predicted_label": str(pred_label),
                "display_label": display_label(pred_label),
                "confidence": confidence,
                "processing_time_sec": elapsed,
                "video_metadata": video_meta,
                "cloud_frontend": "AWS EC2",
                "cloud_storage": "AWS S3",
                "model_source": f"s3://{BUCKET_NAME}/model/",
            }

            result_path = job_dir / "prediction_result.json"
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

            s3_uris = upload_outputs_to_s3(
                job_id,
                [video_path, keypoints_path, result_path],
            )
            
            oracle_uris = upload_outputs_to_oracle(
                job_id,
                {
                    "uploaded_video": video_path,
                    "extracted_keypoints": keypoints_path,
                    "prediction_result": result_path,
                    "mediapipe_pose_preview": pose_preview_path,
                }
            )
            
            

            result["s3_output_uris"] = s3_uris

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

        st.success("Cloud AI Analysis Completed")

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.metric("Predicted Technique", result["display_label"])

        with r2:
            st.metric("Confidence", f"{confidence:.2%}" if confidence is not None else "N/A")

        with r3:
            st.metric("Pose Detection Rate", f"{video_meta['pose_detection_rate']:.2%}")

        with r4:
            st.metric("Processing Time", f"{elapsed:.2f}s")

        st.caption(
            "Main actor selection: "
            f"{video_meta.get('actor_selection_method', 'N/A')} | "
            f"Selected track: {video_meta.get('selected_track_id', 'N/A')} | "
            f"Track 0 score: {video_meta.get('track0_kicking_score', 0):.3f} | "
            f"Track 1 score: {video_meta.get('track1_kicking_score', 0):.3f}"
        )

        st.subheader("MediaPipe Pose Preview")

        if pose_preview_path is not None:
            st.video(str(pose_preview_path))

            pose_preview_s3_key = f"demo-results/{job_id}/mediapipe_pose_preview.mp4"

            try:
                s3_client = boto3.client("s3", region_name=REGION)
                s3_client.upload_file(str(pose_preview_path), BUCKET_NAME, pose_preview_s3_key)
                st.code(f"s3://{BUCKET_NAME}/{pose_preview_s3_key}")
            except Exception as e:
                st.warning(f"MediaPipe pose preview was generated, but upload to S3 failed: {e}")
        else:
            st.info("MediaPipe pose preview could not be generated.")
        st.subheader("AWS S3 Output")
        st.code("\n".join(s3_uris))

        st.subheader("Oracle Cloud Backup")

        for name, uri in oracle_uris.items():
            st.write(f"{name}: {uri}")
        
        
        
        st.subheader("Extracted Keypoints Preview")
        st.dataframe(keypoints_df.head(10))

        st.subheader("Prediction JSON")
        st.json(result)
else:
    st.info("Upload a video to start the cloud AI analysis.")
