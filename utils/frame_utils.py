import cv2
import os
import uuid

FRAME_DIR = "uploads/frames"
os.makedirs(FRAME_DIR, exist_ok=True)


def extract_frame(video_path, frame_index):
    """
    Extract a single frame from video safely
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Cannot open video:", video_path)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Clamp frame index safely
    frame_index = max(0, min(frame_index, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to read frame")
        return None

    frame_name = f"frame_{uuid.uuid4().hex}.jpg"
    frame_path = os.path.join(FRAME_DIR, frame_name)

    cv2.imwrite(frame_path, frame)

    return frame_path
