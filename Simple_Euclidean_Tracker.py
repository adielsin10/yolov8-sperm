import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


input_csv = r"C:\tracformer_modle\trackformer-sperm\progect_yolov8\yolo_output\sort_input_kk fly5 sr1.csv"
output_csv = r'yolo_output\simple_tracks_kk fly5 sr1.csv'
# ⚙️ פרמטרים
distance_threshold = 30  # מרחק מירבי לשיוך אובייקט

# 📥 קריאה
df = pd.read_csv(input_csv)
df['x_center'] = (df['x1'] + df['x2']) / 2
df['y_center'] = (df['y1'] + df['y2']) / 2
df = df.sort_values(by='frame')

# 🔄 משתנים
tracks = []
next_track_id = 1
results = []

# 🔁 מעבר פריים-פריים
for frame, group in df.groupby('frame'):
    detections = group[['x_center', 'y_center']].values
    assigned = [False] * len(detections)

    # רק טראקים מהפריים הקודם (המשכיות בלבד!)
    active_tracks = [
        t for t in tracks if not t['locked'] and (frame - t['last_seen_frame'] == 1)
    ]

    if active_tracks and len(detections) > 0:
        track_centers = np.array([t['center'] for t in active_tracks])
        dists = cdist(track_centers, detections)

        used_detections = set()
        for t_idx, track in enumerate(active_tracks):
            d_idx = np.argmin(dists[t_idx])
            if (
                dists[t_idx][d_idx] < distance_threshold
                and not assigned[d_idx]
                and d_idx not in used_detections
            ):
                x1, y1, x2, y2 = group.iloc[d_idx][['x1', 'y1', 'x2', 'y2']]
                results.append([frame, track['id'], x1, y1, x2, y2])
                track['center'] = detections[d_idx]
                track['last_seen_frame'] = frame
                track['age'] += 1
                assigned[d_idx] = True
                used_detections.add(d_idx)

    # יצירת טראקים חדשים עבור אובייקטים לא משויכים
    for idx, det in enumerate(detections):
        if not assigned[idx]:
            x1, y1, x2, y2 = group.iloc[idx][['x1', 'y1', 'x2', 'y2']]
            results.append([frame, next_track_id, x1, y1, x2, y2])
            tracks.append({
                'id': next_track_id,
                'center': det,
                'last_seen_frame': frame,
                'age': 1,
                'locked': False
            })
            next_track_id += 1

    # נועל כל טראק שלא הופיע בפריים הזה בדיוק
    for trk in tracks:
        if not trk['locked'] and (frame - trk['last_seen_frame'] > 0):
            trk['locked'] = True

# 💾 שמירה
results_df = pd.DataFrame(results, columns=['frame', 'track_id', 'x1', 'y1', 'x2', 'y2'])
results_df.to_csv(output_csv, index=False)
print(f"✅ הסתיים! נשמר אל: {output_csv}")
