import csv
from sort import Sort  # ודא שיש לך את sort.py בסביבה שלך
import numpy as np

# נתיב לקובץ שנוצר מה-YOLO
input_csv = r'yolo_output\sort_input_Protamine_6h_fly1_sr1.csv'
output_csv = r'yolo_output\sort_tracks_Protamine_6h_fly1_sr1.csv'

# הכנה לקריאה
detections_per_frame = {}

with open(input_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        frame = int(row['frame'])
        det = [
            float(row['x1']),
            float(row['y1']),
            float(row['x2']),
            float(row['y2']),
            float(row['confidence'])
        ]
        detections_per_frame.setdefault(frame, []).append(det)

# הכנת SORT
tracker = Sort()
output_rows = []

# ריצה על פריימים לפי סדר
for frame in sorted(detections_per_frame.keys()):
    dets = np.array(detections_per_frame[frame])
    tracks = tracker.update(dets)

    for track in tracks:
        x1, y1, x2, y2, track_id = track
        output_rows.append([
            frame, int(track_id), x1, y1, x2, y2
        ])

# שמירת הפלט
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'track_id', 'x1', 'y1', 'x2', 'y2'])
    writer.writerows(output_rows)

print(f"✅ הקובץ עם תוצאות העקיבה נוצר בהצלחה: {output_csv}")
