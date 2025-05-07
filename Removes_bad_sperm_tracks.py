import pandas as pd
import numpy as np
import os

# נתיב לקובץ המקורי
csv_path = r"C:\tracformer_modle\trackformer-sperm\progect_yolov8\yolo_output\simple_tracks_kk fly5 sr1.csv"

# נתיב לשמירת הקובץ המסונן
output_path = os.path.join(r"C:\files_of_csv", "simple_tracks_kk fly5 sr1_good.csv")

# טען את הנתונים
df = pd.read_csv(csv_path)

# חישוב מרכז התיבה
df["x_center"] = (df["x1"] + df["x2"]) / 2
df["y_center"] = (df["y1"] + df["y2"]) / 2

# סף זווית
MAX_ANGLE = 120
valid_tracks = []

def cut_track_by_angle(points, group_sorted, angle_thresh=MAX_ANGLE):
    deltas = np.diff(points, axis=0)
    speeds = np.linalg.norm(deltas, axis=1)
    unit_deltas = deltas / (speeds[:, None] + 1e-6)

    for i in range(len(unit_deltas) - 1):
        dot = np.dot(unit_deltas[i], unit_deltas[i+1])
        angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        if angle_deg > angle_thresh:
            return group_sorted.iloc[:i + 2]  # חותך עד לנקודה שלפני הזווית החריפה
    return group_sorted  # אם אין זווית חריפה – מחזיר את כולו

# עיבוד לפי track_id
for track_id, group in df.groupby("track_id"):
    group_sorted = group.sort_values("frame")
    points = group_sorted[["x_center", "y_center"]].to_numpy()

    if len(points) < 3:
        continue

    partial_track = cut_track_by_angle(points, group_sorted)

    if len(partial_track) >= 3:  # שומר רק מסלולים משמעותיים
        valid_tracks.append(partial_track)

# שמירה לקובץ חדש
if valid_tracks:
    clean_df = pd.concat(valid_tracks)
    clean_df.to_csv(output_path, index=False)
    print(f"✅ הקובץ נשמר לאחר חיתוך לפי זוויות בנתיב:\n{output_path}")
else:
    print("⚠️ לא נשמרו מסלולים – ייתכן שכולם נחתכו מוקדם מדי.")
