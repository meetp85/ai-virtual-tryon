import streamlit as st
import cv2
import numpy as np

st.title("AI Virtual Try-On")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.write("Camera error")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # TODO: call your try-on function here
    # frame = tryon_process(frame)

    FRAME_WINDOW.image(frame)

else:
    st.write("Camera stopped")