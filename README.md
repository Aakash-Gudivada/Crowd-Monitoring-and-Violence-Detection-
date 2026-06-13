# Crowd Monitoring and Violence Detection System

## Overview

Crowd Monitoring and Violence Detection System is an AI-powered surveillance solution that automatically detects violent activities in video footage. The system uses computer vision and deep learning techniques to analyze human poses and classify activities as violent or non-violent, enabling faster response to potential security threats.

## Features

* Video upload and analysis
* Human pose estimation using YOLOv8 Pose
* Violence classification using Temporal Shift Module (TSM)
* Real-time prediction and monitoring
* FastAPI-based web interface
* Automated alert generation for detected incidents

## Technology Stack

### Backend

* FastAPI
* Python

### Deep Learning and Computer Vision

* PyTorch
* YOLOv8 Pose
* Temporal Shift Module (TSM)
* OpenCV

### Frontend

* HTML
* CSS
* JavaScript
* Jinja2 Templates

## Project Structure

```text
Crowd-Monitoring-and-Violence-Detection/
│
├── ml/
│   ├── inference.py
│   └── pose_tsm.py
│
├── models/
│   └── README.md
│
├── static/
├── templates/
├── utils/
│
├── main.py
├── requirements.txt
└── README.md
```

## System Workflow

```text
Video Upload
      ↓
YOLOv8 Pose Detection
      ↓
Pose Keypoint Extraction
      ↓
TSM-Based Violence Classification
      ↓
Violent / Non-Violent Prediction
      ↓
Alert Generation
```

## Model Performance

| Metric                 | Value  |
| ---------------------- | ------ |
| Validation Accuracy    | 90.4%  |
| Training Accuracy      | 90.5%  |
| Lowest Validation Loss | 0.2486 |
| Final Validation Loss  | 0.2499 |

The model demonstrates strong generalization with minimal overfitting, maintaining nearly identical training and validation accuracy.

## Dataset

The model was trained on the Real-Life Violence Situations Dataset containing violent and non-violent video samples.

Dataset:
https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Crowd-Monitoring-and-Violence-Detection.git
cd Crowd-Monitoring-and-Violence-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Model Weights

The trained model files are not included in this repository due to GitHub storage limitations.

Download the model package from the link provided in:

```text
models/README.md
```

After downloading, place the extracted files inside the `models` directory.

## Future Enhancements

* Real-time CCTV stream monitoring
* Multi-camera surveillance support
* SMS and Email alert integration
* Violence localization and tracking
* Edge device deployment

## License

This project is intended for educational, research, and demonstration purposes.

---

## 👨‍💻 Author

**Aakash Gudivada**

B.Tech Computer Science Engineering

Interested in:
- Artificial Intelligence
- Machine Learning
- Computer Vision
- Full-Stack Development

LinkedIn: https://www.linkedin.com/in/aakashgudivada/



---

## ⭐ If you found this project useful, consider giving it a star.
