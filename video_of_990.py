import os
import cv2
import csv

# ⚙️ נתיבים:
frames_dir = r'C:\imgae_of_yolov8\image_Protamine_6h_fly1_sr2'
tracks_csv = r'yolo_output\sort_tracks.csv'
output_video = r'yolo_output\sperm_990_tracking.mp4'
target_id = 990

# קריאת פרטי track_id = 990
track_data = {}
with open(tracks_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if int(row['track_id']) != target_id:
            continue
        frame = int(row['frame'])
        x1 = int(float(row['x1']))
        y1 = int(float(row['y1']))
        x2 = int(float(row['x2']))
        y2 = int(float(row['y2']))
        track_data[frame] = (x1, y1, x2, y2)

# הגדרת גודל וידאו
first_frame_path = os.path.join(frames_dir, f"frame_{min(track_data.keys()):04d}.png")
sample = cv2.imread(first_frame_path)
height, width, _ = sample.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, 10, (width, height))

# יצירת הסרטון
for frame_num in sorted(track_data.keys()):
    filename = f"frame_{frame_num:04d}.png"
    img_path = os.path.join(frames_dir, filename)

    if not os.path.exists(img_path):
        print(f"⚠️ קובץ לא נמצא: {filename}")
        continue

    frame = cv2.imread(img_path)
    x1, y1, x2, y2 = track_data[frame_num]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f'ID {target_id}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    video_writer.write(frame)

video_writer.release()
print(f"🎉 הסרטון של זרעון 990 נוצר בהצלחה: {output_video}")
