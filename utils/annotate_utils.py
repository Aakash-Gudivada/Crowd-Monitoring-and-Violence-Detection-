import cv2
import os
import uuid
from ultralytics import YOLO

# Load YOLOv8 POSE model (same one used in inference)
pose_model = YOLO("models/yolov8s-pose.pt")

ANNOTATED_DIR = "uploads/annotated"
os.makedirs(ANNOTATED_DIR, exist_ok=True)


def get_box_color(violence_prob):
    """
    Decide bounding box color based on violence probability
    """
    if violence_prob >= 0.85:
        return (0, 0, 255)      # 🔴 Red (high violence)
    elif violence_prob >= 0.70:
        return (0, 165, 255)    # 🟠 Orange (medium)
    else:
        return (0, 255, 0)      # 🟢 Green (low)


def annotate_boxes_on_frame(frame_path, violence_prob):
    """
    Draw bounding boxes (NO skeletons) on a single frame
    using YOLOv8-POSE and save annotated image.
    """

    img = cv2.imread(frame_path)
    if img is None:
        print("❌ Failed to read frame for annotation:", frame_path)
        return None

    # Run YOLO pose on ONE frame only
    results = pose_model(img, verbose=False)

    annotated_img = img.copy()
    box_color = get_box_color(violence_prob)

    for r in results:
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)

                # Draw bounding box
                cv2.rectangle(
                    annotated_img,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )

                # Draw label
                cv2.putText(
                    annotated_img,
                    f"Person {conf:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

    out_name = f"boxed_{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(ANNOTATED_DIR, out_name)

    cv2.imwrite(out_path, annotated_img)

    return out_path
