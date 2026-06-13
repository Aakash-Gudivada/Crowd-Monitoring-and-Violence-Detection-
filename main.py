# from fastapi import FastAPI, UploadFile, File, BackgroundTasks
# import shutil
# import os

# from ml.inference import predict_video
# from utils.email_service import send_violence_alert

# app = FastAPI()

# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)


# # ✅ FIX: root endpoint
# @app.get("/")
# def home():
#     return {
#         "message": "Violence Detection API is running",
#         "usage": "Open /docs to test the API"
#     }


# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...),
#     background_tasks: BackgroundTasks = None
# ):
#     video_path = os.path.join(UPLOAD_DIR, file.filename)

#     with open(video_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     result = predict_video(video_path)

#     if result is None:
#         return {"error": "Video too short or no pose detected"}

#     violence_prob = result["violence"]
#     non_violence_prob = result["non_violence"]

#     if violence_prob >= 0.70:
#         label = "Violence"
#         background_tasks.add_task(
#             send_violence_alert,
#             file.filename,
#             violence_prob
#         )
#     elif violence_prob <= 0.60:
#         label = "Non-Violence"
#     else:
#         label = "Uncertain"

#     return {
#         "label": label,
#         "violence_probability": round(violence_prob, 3),
#         "non_violence_probability": round(non_violence_prob, 3)
#     }
# from utils.frame_utils import extract_frame
# from fastapi import FastAPI, UploadFile, File, Request, BackgroundTasks
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# import os
# import shutil
# from utils.annotate_utils import annotate_boxes_on_frame



# from ml.inference import predict_video
# from utils.email_service import send_violence_alert

# app = FastAPI()

# # Templates
# templates = Jinja2Templates(directory="templates")

# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)


# # -------------------------------
# # HOME PAGE (shows index.html)
# # -------------------------------
# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     return templates.TemplateResponse(
#         "index.html",
#         {"request": request}
#     )



# @app.post("/analyze", response_class=HTMLResponse)
# async def analyze_video(
#     request: Request,
#     video: UploadFile = File(...),
#     background_tasks: BackgroundTasks = BackgroundTasks()
# ):
#     video_path = os.path.join(UPLOAD_DIR, video.filename)

#     with open(video_path, "wb") as buffer:
#         shutil.copyfileobj(video.file, buffer)

#     result = predict_video(video_path)

#     if result is None:
#         message = "❌ Video too short or no pose detected."
#         return templates.TemplateResponse(
#             "index.html",
#             {"request": request, "message": message}
#         )

#     violence_prob = result["violence"]
#     non_violence_prob = result["non_violence"]

#     frame_path = None
#     annotated_frame_path = None

#     if violence_prob >= 0.70:
#         label = "🚨 Violence Detected"

#         frame_path = extract_frame(
#             video_path=video_path,
#             frame_index=result["max_violence_frame"]
#         )

#         print("DEBUG frame_path:", frame_path)

#         # 🟢 ANNOTATION STEP
#         if frame_path:
#             annotated_frame_path = annotate_pose_on_frame(frame_path)
#             print("DEBUG annotated_frame_path:", annotated_frame_path)

#         background_tasks.add_task(
#             send_violence_alert,
#             video.filename,
#             violence_prob,
#             annotated_frame_path if annotated_frame_path else frame_path
#         )

#     elif violence_prob <= 0.60:
#         label = "✅ Non-Violence"
#     else:
#         label = "⚠️ Keep monitoring continuously for a period of time"

#     message = f"""
#     <b>Result:</b> {label}<br>
#     <b>Violence Probability:</b> {violence_prob:.3f}<br>
#     <b>Non-Violence Probability:</b> {non_violence_prob:.3f}
#     """

#     return templates.TemplateResponse(
#         "index.html",
#         {
#             "request": request,
#             "message": message
#         }
#     )
from fastapi import FastAPI, UploadFile, File, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import shutil

# 🔹 ML inference
from ml.inference import predict_video

# 🔹 Utilities
from utils.frame_utils import extract_frame
from utils.annotate_utils import annotate_boxes_on_frame
from utils.email_service import send_violence_alert

app = FastAPI()
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# 🔥 THIS MUST COME BEFORE @app.get("/")
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------
# Templates
# -------------------------------
templates = Jinja2Templates(directory="templates")

# -------------------------------
# Upload directory
# -------------------------------
UPLOAD_DIR = "uploads/videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------
# HOME PAGE
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# -------------------------------
# ANALYZE VIDEO
# -------------------------------
@app.post("/analyze", response_class=HTMLResponse)
async def analyze_video(
    request: Request,
    video: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    # 1️⃣ Save uploaded video
    video_path = os.path.join(UPLOAD_DIR, video.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # 2️⃣ Run ML inference
    result = predict_video(video_path)

    if result is None:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "message": "❌ Video too short or no pose detected."
            }
        )

    violence_prob = result["violence"]
    non_violence_prob = result["non_violence"]

    frame_path = None
    annotated_frame_path = None

    # 3️⃣ Decision logic
    if violence_prob >= 0.70:
        label = "🚨 Violence Detected"

        # 🔹 Extract ONE key frame
        frame_path = extract_frame(
            video_path=video_path,
            frame_index=result["max_violence_frame"]
        )

        print("DEBUG frame_path:", frame_path)

        # 🔹 Draw COLORED bounding boxes on that frame only
        if frame_path:
            annotated_frame_path = annotate_boxes_on_frame(
                frame_path,
                violence_prob
            )
            print("DEBUG annotated_frame_path:", annotated_frame_path)

        # 🔹 Send email with annotated frame
        background_tasks.add_task(
            send_violence_alert,
            video.filename,
            violence_prob,
            annotated_frame_path if annotated_frame_path else frame_path
        )

    elif violence_prob <= 0.60:
        label = "✅ Non-Violence"
    else:
        label = "⚠️ Keep monitoring continuously for a period of time"

    # 4️⃣ Prepare UI message
    message = f"""
    <b>Result:</b> {label}<br>
    <b>Violence Probability:</b> {violence_prob:.3f}<br>
    <b>Non-Violence Probability:</b> {non_violence_prob:.3f}
    """

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "message": message
        }
    )
