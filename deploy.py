import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os


@st.cache_resource
def load_model():
    return YOLO("C:/Users/user/Downloads/PR2/best.pt")

model = load_model()

st.set_page_config(page_title="Road Damage Detection", layout="centered")

st.title("🚧 Road Damage Detection System")
st.write("Upload an image to detect potholes, cracks, and road damage.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Detecting..."):
            # Save temp image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                temp_path = tmp.name

            
            results = model.predict(source=temp_path, conf=0.25, imgsz=640)

          
            result_img = results[0].plot()
            st.image(result_img, caption="Detection Result", use_column_width=True)

            
            boxes = results[0].boxes
            if boxes is not None:
                st.subheader("Detections:")
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    st.write(f"Object {i+1}: Class = {cls}, Confidence = {conf:.2f}")

            
            os.remove(temp_path)

st.markdown("---")
st.caption("Built with YOLOv8 + Streamlit ")
