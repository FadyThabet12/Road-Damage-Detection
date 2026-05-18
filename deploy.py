
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2
import numpy as np
#loading model as best .pt
@st.cache_resource
def load_model():
    return YOLO("C:/Users/user/Downloads/PR2/best.pt")

model = load_model()

st.set_page_config(
    page_title="Road Damage Detection",
    layout="centered"
)

st.title("🚧 Road Damage Detection System")

st.write("""
Upload a road image to detect:
""")
uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True
    )

    if st.button("Run Detection"):

        with st.spinner("Detecting Road Damages..."):

       
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as tmp:

                image.save(tmp.name)
                temp_path = tmp.name

            results = model.predict(
                source=temp_path,
                conf=0.25,
                imgsz=640
            )

            result = results[0]

            plotted_image = result.plot()

            st.image(
                plotted_image,
                caption="Detection Results",
                use_column_width=True
            )
            boxes = result.boxes
            damage_count = 0
            st.subheader(" Detection Details")

            if boxes is not None and len(boxes) > 0:

                damage_count = len(boxes)

                #  Display Total Damages
                
                st.success(f"Total Damages Detected: {damage_count}")

                #  Loop Through All Detections
                
                for i, box in enumerate(boxes):

                    #Extract Class and Confidence
                   
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Extract Bounding Box Coordinates
                
                    x1, y1, x2, y2 = box.xyxy[0]

                    width = x2 - x1
                    height = y2 - y1

                    # ***** Calculate Area
                   
                    area = width * height

                    # Severity Estimation
                    # 
                    if area > 50000:
                        severity = "🔴 High"

                    elif area > 15000:
                        severity = "🟠 Medium"

                    else:
                        severity = "🟢 Low"

                    st.write(f"""
                    ### Damage {i+1}
                    - Class ID: {cls}
                    - Confidence Score: {conf:.2f}
                    - Severity Level: {severity}
                    - Bounding Box Area: {int(area)}
                    """)

                #  Overall Road Condition
                # 
                st.subheader(" Road Condition Analysis")

                if damage_count >= 10:
                    st.error("Road condition is VERY POOR")

                elif damage_count >= 5:
                    st.warning("Road condition is MODERATE")

                else:
                    st.success("Road condition is ACCEPTABLE")

            else:

                st.success(" No road damages detected.")
            os.remove(temp_path)

st.markdown("---")

st.caption("""
Thank YOU
""")