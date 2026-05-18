# Road Damage Detection System

## Overview

The **Road Damage Detection System** is a Computer Vision project developed using **YOLOv8** and **Streamlit**.
The system detects different types of road damages such as:

* Potholes
* Cracks
* Surface damages

The project applies Object Detection techniques to improve road inspection and infrastructure monitoring.

---

# Features

 Detect road damages from uploaded images
 Draw bounding boxes around detected damages
 Display confidence scores
 Count total detected damages
 Estimate damage severity level
 Analyze overall road condition
 Interactive web application using Streamlit

---

# Technologies Used

* Python
* YOLOv8
* Ultralytics
* Streamlit
* OpenCV
* PIL (Python Imaging Library)

---

# Project Structure

```bash
RoadDamageDetection/
│
├── deployment.py               
├── best.pt               
├── data.yaml        
├── README.md            
└── Final_PR.ipynb         
```

---

# Dataset

The model was trained on a road damage dataset containing annotated images of:

* Potholes
* Cracks
* Road surface defects

The dataset was prepared in YOLO format with:

* Images
* Labels
* Bounding box annotations

---

# Training Configuration

The model was trained using YOLOv8 with different training settings such as:
- Epochs
- Image size
- Data augmentation

Training was performed on a custom road damage dataset to improve detection accuracy and model performance.

---

# Deployment

The system is deployed using **Streamlit**.

---

# Detection Pipeline

The system performs the following steps:

1. Upload road image
2. Preprocess image
3. Run YOLOv8 detection
4. Draw bounding boxes
5. Display confidence scores
6. Count detected damages
7. Estimate severity level
8. Show final road condition analysis

---

# Severity Estimation

Damage severity is estimated based on the detected bounding box area.

| Area Size | Severity |
| --------- | -------- |
| Small     | Low      |
| Medium    | Medium   |
| Large     | High     |

---

# Output Example

The application displays:

* Detected damages
* Bounding boxes
* Confidence scores
* Damage count
* Severity level
* Road condition analysis

---

# Evaluation Metrics

The model performance was evaluated using:

* mAP
* Precision
* Recall

---

# Author

Developed by:  
**Fady Thabet**

Student at:  
Badr University in Assiut  
School of Artificial Intelligence and Data Management

---

# Notes

* The project uses Deep Learning and Computer Vision techniques.
* YOLOv8 was selected because of its high detection accuracy and real-time performance.
* Streamlit was used to build the deployment interface.
