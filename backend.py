# backend.py
import os
import joblib
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import xgboost as xgb
import matplotlib.pyplot as plt

from moviepy.video.io.VideoFileClip import VideoFileClip

# ================= CONFIG =================
SR = 16000
DURATION = 5
TARGET_LEN = SR * DURATION

MODEL_DIR = "models"
CLIP_DIR = "audio_clips"

os.makedirs(CLIP_DIR, exist_ok=True)

# ================= LOAD MODELS =================
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
le     = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
svm    = joblib.load(f"{MODEL_DIR}/svm.pkl")
rf     = joblib.load(f"{MODEL_DIR}/rf.pkl")
xgb_b  = joblib.load(f"{MODEL_DIR}/xgb_base.pkl")
meta   = joblib.load(f"{MODEL_DIR}/meta_model.pkl")
class_names = joblib.load(f"{MODEL_DIR}/class_names.pkl")

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

# ================= FEATURE EXTRACTION =================
def extract_features(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y = y[:TARGET_LEN] if len(y) >= TARGET_LEN else np.pad(y, (0, TARGET_LEN-len(y)))
    y = y / (np.max(np.abs(y)) + 1e-6)

    _, emb, _ = yamnet(tf.convert_to_tensor(y, tf.float32))
    yam = tf.reduce_mean(emb, axis=0).numpy()

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
    mfcc = np.hstack([mfcc.mean(1), mfcc.std(1)])

    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64)
    mel = librosa.power_to_db(mel)
    mel = np.hstack([mel.mean(1)[:12], mel.std(1)[:12]])

    zcr = librosa.feature.zero_crossing_rate(y).mean()
    sc  = librosa.feature.spectral_centroid(y=y, sr=SR).mean()

    return np.hstack([yam, mfcc, mel, [zcr, sc]]).astype(np.float32)

# ================= SHARD VIDEO =================
def shard_video(video_path, progress_cb=None):
    video = VideoFileClip(video_path)
    duration = int(video.duration)

    clips = []
    steps = list(range(0, duration, DURATION))

    for i, start in enumerate(steps):
        end = min(start + DURATION, duration)
        name = f"clip_{start:05d}_{end:05d}.wav"
        path = os.path.join(CLIP_DIR, name)

        sub = video.subclipped(start, end)
        sub.audio.write_audiofile(path, fps=SR, codec="pcm_s16le")

        clips.append({"clip": name, "start": start, "end": end})

        if progress_cb:
            progress_cb((i + 1) / len(steps))

    video.close()
    return clips

# ================= PREDICT =================
def predict(clips, progress_cb=None):
    X, rows = [], []

    for i, c in enumerate(clips):
        feats = extract_features(os.path.join(CLIP_DIR, c["clip"]))
        X.append(feats)
        rows.append(c)

        if progress_cb:
            progress_cb((i + 1) / len(clips))

    X = scaler.transform(np.array(X))

    p_svm = svm.predict_proba(X)
    p_rf  = rf.predict_proba(X)
    p_xgb = xgb_b.predict(xgb.DMatrix(X))

    stacked = np.hstack([p_svm, p_rf, p_xgb])
    y_pred = np.argmax(meta.predict(xgb.DMatrix(stacked)), axis=1)

    emotions = le.inverse_transform(y_pred)
    for i, e in enumerate(emotions):
        rows[i]["emotion"] = e

    return pd.DataFrame(rows)

# ================= PLOTS =================
def emotion_plots(df):
    fig1, ax1 = plt.subplots(figsize=(14,4))
    mapping = {e:i for i,e in enumerate(sorted(df["emotion"].unique()))}
    ax1.plot(df["start"], df["emotion"].map(mapping), marker="o")
    ax1.set_yticks(list(mapping.values()))
    ax1.set_yticklabels(list(mapping.keys()))
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Emotion")
    ax1.set_title("Emotion Timeline")
    ax1.grid(True)

    fig2, ax2 = plt.subplots(figsize=(6,4))
    df["emotion"].value_counts().plot(kind="bar", ax=ax2)
    ax2.set_title("Emotion Distribution")
    ax2.grid(axis="y")

    return fig1, fig2
