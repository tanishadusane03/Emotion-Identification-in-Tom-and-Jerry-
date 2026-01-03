# app.py
import streamlit as st
import backend
import tempfile
import os

st.set_page_config(
    page_title="Emotion Detection App",
    layout="centered"
)

st.title("🎭 Emotion Detection App")
st.write("Upload a video to analyze emotions")

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader(
    "Upload your video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:
    st.success("✅ Video uploaded")
    st.video(uploaded_file)

    if st.button("🔍 Analyze Emotion"):

        with st.spinner("Processing video..."):

            # ---------- SAVE TEMP VIDEO ----------
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                video_path = tmp.name

            # ---------- SHARDING ----------
            st.subheader("🎬 Extracting audio clips")
            shard_bar = st.progress(0)

            clips = backend.shard_video(
                video_path,
                progress_cb=lambda p: shard_bar.progress(p)
            )

            # ---------- PREDICTION ----------
            st.subheader("🧠 Running emotion analysis")
            pred_bar = st.progress(0)

            df = backend.predict(
                clips,
                progress_cb=lambda p: pred_bar.progress(p)
            )

        st.success("✅ Analysis complete")

        # ---------- RESULTS ----------
        st.subheader("📄 Emotion Results")
        st.dataframe(df, use_container_width=True)

        st.subheader("🔢 Emotion Counts")

        cols = st.columns(len(df["emotion"].unique()))

        for col, (emotion, count) in zip(cols, df["emotion"].value_counts().items()):
            col.metric(
                label=emotion.capitalize(),
                value=count
            )
        # ---------- DOWNLOAD CSV ----------
        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False),
            file_name="emotion_results.csv",
            mime="text/csv"
        )

        # ---------- PLOTS ----------
        st.subheader("📊 Emotion Distribution")
        fig1, fig2 = backend.emotion_plots(df)
        st.pyplot(fig1)
        st.pyplot(fig2)

        # ---------- CLEANUP ----------
        os.remove(video_path)
