import streamlit as st
import numpy as np

st.set_page_config(page_title="Emotion Detection App")

st.title("🎭 Emotion Detection App")
st.write("Upload a 5-second clip to predict emotion")

uploaded_file = st.file_uploader(
    "Upload your clip",
    type=["mp3", "wav", "mp4", "avi"]
)

# 👇 THIS BLOCK RUNS ONLY AFTER FILE IS UPLOADED
if uploaded_file is not None:
    st.success("File uploaded!")

    # ✅ SHOW VIDEO PREVIEW HERE
    st.video(uploaded_file)

    # Predict button
    if st.button("Predict Emotion"):
        emotions = ["Happy 😊", "Sad 😢", "Angry 😠", "Neutral 😐"]
        predicted_emotion = np.random.choice(emotions)

        st.subheader("🎯 Predicted Emotion")
        st.markdown(f"## {predicted_emotion}")
