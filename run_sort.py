import csv
from sort import Sort
import numpy as np

input_csv = r'C:\tracformer_modle\trackformer-sperm\progect_yolov8\yolo_output\sort_input_image_before_tagging.csv'
output_csv = r'yolo_output\try_sort_tracks_image_before_tagging.csv'

# קובץ הקלט → מיפוי לפי פריים
detections_per_frame = {}
with open(input_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        frame = int(row['frame'])
        confidence = float(row['confidence'])
        if confidence < 0.3:
            continue  # אל תכלול זיהויים חלשים

        det = [
            float(row['x1']),
            float(row['y1']),
            float(row['x2']),
            float(row['y2']),
            confidence
        ]
        detections_per_frame.setdefault(frame, []).append(det)

# יצירת הטראקר עם פרמטרים מותאמים
tracker = Sort(iou_threshold=0.4, max_age=7, min_hits=2)
output_rows = []

for frame in sorted(detections_per_frame.keys()):
    frame_dets = detections_per_frame[frame]
    dets = np.array(frame_dets)

    if dets.size == 0:
        print(f"⚠️ פריים {frame} בלי זיהויים")
        dets = np.empty((0, 5))

    tracks = tracker.update(dets)

    print(f"\n🎞️ פריים {frame} | קלט: {len(dets)} תיבות | טראקים פעילים: {len(tracks)}")
    for track in tracks:
        x1, y1, x2, y2, track_id = track
        print(f"   🟢 Track ID {int(track_id)} -> Box: ({x1:.1f},{y1:.1f}) to ({x2:.1f},{y2:.1f})")
        output_rows.append([
            frame, int(track_id), x1, y1, x2, y2
        ])

# שמירת תוצאות
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'track_id', 'x1', 'y1', 'x2', 'y2'])
    writer.writerows(output_rows)

print(f"\n✅ הסתיים! התוצאות נשמרו בקובץ: {output_csv}")
