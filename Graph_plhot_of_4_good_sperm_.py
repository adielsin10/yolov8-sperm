import pandas as pd
import matplotlib.pyplot as plt

# 🔁 נתיב לקובץ התוצאה:
csv_path = r"C:\tracformer_modle\trackformer-sperm\progect_yolov8\yolo_output\hun_tracks_Protamine_6h_fly1_sr1.csv"

# 📥 קריאת הקובץ
df = pd.read_csv(csv_path)

# 🎯 חישוב מספר פריימים לכל track_id
track_lengths = df.groupby('track_id')['frame'].nunique()

# מזהים שצריך להכריח להיכלל
forced_ids = [733, 939,571,636]

# בדיקה אם הם קיימים בכלל בדאטה
existing_forced_ids = [tid for tid in forced_ids if tid in track_lengths.index]

# הסרתם מהטבלה הזמנית
remaining_tracks = track_lengths.drop(labels=existing_forced_ids, errors='ignore')

# בחירת שני זרעונים נוספים עם הכי הרבה פריימים
additional_ids = remaining_tracks.sort_values(ascending=False).head(4 - len(existing_forced_ids)).index.tolist()

# רשימת כל הארבעה
top_4_ids = existing_forced_ids + additional_ids
print("Top 4 selected IDs:", top_4_ids)

# ➗ חישוב מרכז התיבה
df['x_center'] = (df['x1'] + df['x2']) / 2
df['y_center'] = (df['y1'] + df['y2']) / 2

# 🎨 צבעים מותאמים אישית: מספיק ל-4 זרעונים
custom_colors = ['red', 'green', 'blue', 'black']

# ציור המסלולים
plt.figure(figsize=(10, 8))
for i, track_id in enumerate(top_4_ids):
    sub_df = df[df['track_id'] == track_id].sort_values(by='frame')
    plt.plot(
        sub_df['x_center'],
        sub_df['y_center'],
        marker='o',
        linestyle='-',
        color=custom_colors[i],
        label=f'ID {track_id}'
    )

plt.title("Selected 4 Sperm Tracks (including ID 733 & 939)")
plt.xlabel("X Center Position (pixels)")
plt.ylabel("Y Center Position (pixels)")
plt.gca().invert_yaxis()
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
