import pandas as pd
import numpy as np

# ⚙️ עדכן את הנתיב לקובץ הפלט של SORT
csv_path = r'yolo_output\try_sort_tracks_Protamine_6h_fly1_sr1.csv'

# ⚙️ הגדרת סף קפיצה: כמה פיקסלים ייחשבו כקפיצה חשודה
jump_threshold = 80

# טען את הנתונים
df = pd.read_csv(csv_path)

# מחשב את מרכז התיבה (x_center, y_center)
df['x_center'] = (df['x1'] + df['x2']) / 2
df['y_center'] = (df['y1'] + df['y2']) / 2

# סידור לפי track_id ואז לפי frame
df_sorted = df.sort_values(by=['track_id', 'frame'])

# שמירה של קפיצות חריגות
jumps = []

# מעבר על כל track_id בנפרד
for track_id, group in df_sorted.groupby('track_id'):
    group = group.sort_values(by='frame')
    coords = group[['x_center', 'y_center']].values
    frames = group['frame'].values
    for i in range(1, len(coords)):
        dist = np.linalg.norm(coords[i] - coords[i - 1])
        if dist > jump_threshold:
            jumps.append({
                'track_id': track_id,
                'frame_from': frames[i - 1],
                'frame_to': frames[i],
                'distance': round(dist, 2)
            })

# הצגת התוצאה
jumps_df = pd.DataFrame(jumps)
if not jumps_df.empty:
    print("🔍 קפיצות חשודות שהתגלו:")
    print(jumps_df)
else:
    print("✅ לא נמצאו קפיצות חריגות מעל", jump_threshold, "פיקסלים.")
