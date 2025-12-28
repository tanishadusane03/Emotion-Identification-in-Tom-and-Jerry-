# 🔹 Step 1: Install dependencies
!pip install moviepy pandas

# 🔹 Step 2: Import required libraries
from moviepy.video.io.VideoFileClip import VideoFileClip
import pandas as pd
import os
from google.colab import files
import zipfile

# 🔹 Step 3: Upload your MP4 video
print("📤 Upload your MP4 file:")
uploaded = files.upload()

# Get uploaded filename
mp4_file = list(uploaded.keys())[0]
print(f"✅ Uploaded: {mp4_file}")

# 🔹 Step 4: Function to split video and create CSV
def split_video_with_csv(mp4_path, output_folder="video_clips", clip_length=5):
    os.makedirs(output_folder, exist_ok=True)
    video = VideoFileClip(mp4_path)
    duration = int(video.duration)

    clip_data = []  # For CSV rows

    for start in range(0, duration, clip_length):
        end = min(start + clip_length, duration)

        # NEW: Updated naming format: tj2_start_end.mp4
        clip_name = f"tj2_{start:05d}_{end:05d}.mp4"
        clip_path = os.path.join(output_folder, clip_name)

        clip = video.subclip(start, end)
        clip.write_videofile(
            clip_path,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None
        )
        print(f"🎞️ Saved: {clip_path}")

        # Add to CSV
        clip_data.append({
            "Clip Name": clip_name,
            "Start Time (s)": start,
            "End Time (s)": end
        })

    video.close()

    # Create CSV metadata file
    csv_path = os.path.join(output_folder, "clips_metadata.csv")
    df = pd.DataFrame(clip_data)
    df.to_csv(csv_path, index=False)
    print(f"\n🧾 Metadata CSV created: {csv_path}")

    print(f"✅ Done! {len(clip_data)} clips saved in '{output_folder}' folder.")
    return csv_path

# 🔹 Step 5: Run the split + CSV creation
csv_file = split_video_with_csv(mp4_file, output_folder="video_clips", clip_length=5)

# 🔹 Step 6: Zip and download all clips + CSV
zip_filename = "video_clips.zip"
with zipfile.ZipFile(zip_filename, "w") as zipf:
    for root, _, files_in_dir in os.walk("video_clips"):
        for file in files_in_dir:
            zipf.write(os.path.join(root, file))

print(f"\n📦 All clips + CSV zipped as '{zip_filename}' — ready to download.")
files.download(zip_filename)
