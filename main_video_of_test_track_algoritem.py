import cv2
import pandas as pd
import os
from natsort import natsorted
import re

# 🗂️ נתיבים
frames_dir = r"C:\videos_lsm_test_yolov_8\frames_lsm\kk fly5 sr1"
tracking_csv = r"C:\files_of_csv\simple_tracks_kk fly5 sr1_good.csv"
output_video = r'C:\videos_try\simple_tracks_kk fly5 sr1_good.mp4'

# 📖 קריאת תוצאות המעקב
df = pd.read_csv(tracking_csv)

# 📷 קבצי תמונה ממיונים לפי סדר טבעי
frame_files = natsorted([
    f for f in os.listdir(frames_dir)
    if f.startswith('frame_') and f.endswith(('.jpg', '.png'))
])

# 📐 קביעת גודל הפריים הראשון
first_frame_path = os.path.join(frames_dir, frame_files[0])
first_frame = cv2.imread(first_frame_path)
if first_frame is None:
    raise FileNotFoundError(f"שגיאה: לא ניתן לקרוא את התמונה הראשונה ({first_frame_path})")

height, width, _ = first_frame.shape

# 🎞️ יצירת קובץ וידאו
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 1
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 🖼️ מעבר על כל פריים לפי שם הקובץ
for filename in frame_files:
    frame_path = os.path.join(frames_dir, filename)
    frame = cv2.imread(frame_path)

    # חילוץ מספר הפריים מתוך שם הקובץ
    match = re.search(r'frame_(\d+)', filename)
    if not match:
        print(f"⚠️ לא נמצא מספר פריים עבור {filename}")
        continue

    frame_number = int(match.group(1))

    # ציור התיבות הרלוונטיות לפריים הנוכחי
    frame_tracks = df[df['frame'] == frame_number]
    for _, row in frame_tracks.iterrows():
        x1, y1, x2, y2 = map(int, [row['x1'], row['y1'], row['x2'], row['y2']])
        track_id = int(row['track_id'])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out.write(frame)

out.release()
print(f"\n🎬 הווידאו נוצר בהצלחה: {output_video}")