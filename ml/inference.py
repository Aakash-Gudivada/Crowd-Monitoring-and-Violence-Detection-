
# import cv2
# import torch
# import numpy as np
# from ultralytics import YOLO
# from ml.pose_tsm import PoseTSM

# CLIP_LEN = 16
# STRIDE = 4
# BATCH_SIZE = 32

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # -------------------------------
# # Load models
# # -------------------------------
# pose_model = YOLO("models/yolov8s-pose.pt")

# tsm_model = PoseTSM(num_classes=2)

# # ✅ FIX: load checkpoint correctly
# checkpoint = torch.load(
#     "models/TSM_Violence_Model_Final.pth",
#     map_location=device
# )

# tsm_model.load_state_dict(checkpoint["state_dict"])
# tsm_model.to(device)
# tsm_model.eval()

# # -------------------------------
# # 1️⃣ Extract pose keypoints
# # -------------------------------
# def extract_keypoints_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frames_buffer = []
#     all_kpts = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frames_buffer.append(frame)

#         if len(frames_buffer) == BATCH_SIZE:
#             results = pose_model(frames_buffer, verbose=False)

#             for r in results:
#                 if r.keypoints is not None and len(r.keypoints.xy) > 0:
#                     k = r.keypoints.xy[0].cpu().numpy()
#                 else:
#                     k = np.zeros((17, 2))
#                 all_kpts.append(k)

#             frames_buffer = []

#     if frames_buffer:
#         results = pose_model(frames_buffer, verbose=False)
#         for r in results:
#             if r.keypoints is not None and len(r.keypoints.xy) > 0:
#                 k = r.keypoints.xy[0].cpu().numpy()
#             else:
#                 k = np.zeros((17, 2))
#             all_kpts.append(k)

#     cap.release()

#     if len(all_kpts) == 0:
#         return None

#     return np.array(all_kpts)  # (T,17,2)

# # -------------------------------
# # 2️⃣ Create clips (same as Colab)
# # -------------------------------
# def make_clips_for_inference(keypoints):
#     T = keypoints.shape[0]
#     clips = []

#     if T < CLIP_LEN:
#         return clips

#     for start in range(0, T - CLIP_LEN + 1, STRIDE):
#         clips.append(keypoints[start:start + CLIP_LEN])

#     return clips

# # -------------------------------
# # 3️⃣ Final prediction
# # -------------------------------
# def predict_video(video_path):
#     keypoints = extract_keypoints_video(video_path)

#     if keypoints is None or keypoints.shape[0] < CLIP_LEN:
#         return None

#     clips = make_clips_for_inference(keypoints)
#     probs = []

#     with torch.no_grad():
#         for clip in clips:
#             clip_t = (
#                 torch.tensor(clip, dtype=torch.float32)
#                 .unsqueeze(0)
#                 .to(device)
#             )

#             logits = tsm_model(clip_t)
#             softmax = torch.softmax(logits, dim=1)[0]
#             probs.append(softmax.cpu().numpy())

#     probs = np.array(probs)
#     avg_prob = probs.mean(axis=0)

#     return {
#         "non_violence": float(avg_prob[0]),
#         "violence": float(avg_prob[1])
#     }
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ml.pose_tsm import PoseTSM

# -------------------------------
# Config
# -------------------------------
CLIP_LEN = 16
STRIDE = 4
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Load models
# -------------------------------
pose_model = YOLO("models/yolov8s-pose.pt")

tsm_model = PoseTSM(num_classes=2)

checkpoint = torch.load(
    "models/TSM_Violence_Model_Final.pth",
    map_location=device
)

tsm_model.load_state_dict(checkpoint["state_dict"])
tsm_model.to(device)
tsm_model.eval()

# -------------------------------
# 1️⃣ Extract pose keypoints
# -------------------------------
def extract_keypoints_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames_buffer = []
    all_kpts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_buffer.append(frame)

        if len(frames_buffer) == BATCH_SIZE:
            results = pose_model(frames_buffer, verbose=False)

            for r in results:
                if r.keypoints is not None and len(r.keypoints.xy) > 0:
                    k = r.keypoints.xy[0].cpu().numpy()
                else:
                    k = np.zeros((17, 2))
                all_kpts.append(k)

            frames_buffer = []

    if frames_buffer:
        results = pose_model(frames_buffer, verbose=False)
        for r in results:
            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                k = r.keypoints.xy[0].cpu().numpy()
            else:
                k = np.zeros((17, 2))
            all_kpts.append(k)

    cap.release()

    if len(all_kpts) == 0:
        return None

    return np.array(all_kpts)  # (T, 17, 2)

# -------------------------------
# 2️⃣ Create clips
# -------------------------------
def make_clips_for_inference(keypoints):
    T = keypoints.shape[0]
    clips = []

    if T < CLIP_LEN:
        return clips

    for start in range(0, T - CLIP_LEN + 1, STRIDE):
        clips.append(keypoints[start:start + CLIP_LEN])

    return clips

# -------------------------------
# 3️⃣ Final prediction (MODIFIED)
# -------------------------------
def predict_video(video_path):
    keypoints = extract_keypoints_video(video_path)

    if keypoints is None or keypoints.shape[0] < CLIP_LEN:
        return None

    clips = make_clips_for_inference(keypoints)

    probs = []
    violence_scores = []

    with torch.no_grad():
        for clip in clips:
            clip_t = (
                torch.tensor(clip, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )

            logits = tsm_model(clip_t)
            softmax = torch.softmax(logits, dim=1)[0]

            probs.append(softmax.cpu().numpy())
            violence_scores.append(float(softmax[1]))  # violence class

    probs = np.array(probs)
    avg_prob = probs.mean(axis=0)

    # 🔑 Find clip with highest violence
    max_clip_idx = int(np.argmax(violence_scores))

    # 🔑 Map clip index → frame index
    # clip starts at (clip_idx * STRIDE)
    # take middle frame of the clip
    max_violence_frame = max_clip_idx * STRIDE + CLIP_LEN // 2

    return {
        "non_violence": float(avg_prob[0]),
        "violence": float(avg_prob[1]),
        "max_violence_frame": max_violence_frame
    }
