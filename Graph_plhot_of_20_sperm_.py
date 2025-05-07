import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# 🔁 נתיב לקובץ התוצאה:
csv_path = r"C:\tracformer_modle\trackformer-sperm\progect_yolov8\yolo_output\simple_tracks_FKY1_SR1.csv"

# 📥 קריאת הקובץ
df = pd.read_csv(csv_path)
print(df.columns)

# 🎯 חישוב מספר פריימים לכל track_id
track_lengths = df.groupby('track_id')['frame'].nunique()

# ✨ בחירת 20 הזרעונים שהיו הכי הרבה זמן בתנועה
top_20_ids = track_lengths.sort_values(ascending=False).head(20).index

# חישוב מרכז התיבה
df['x_center'] = (df['x1'] + df['x2']) / 2
df['y_center'] = (df['y1'] + df['y2']) / 2

# 🎨 הגדרת צבעים שונים מתוך colormap
colors = cm.get_cmap('tab20', len(top_20_ids))  # colormap עם עד 20 צבעים מובחנים

# ציור המסלולים
plt.figure(figsize=(10, 8))
for i, track_id in enumerate(top_20_ids):
    sub_df = df[df['track_id'] == track_id].sort_values(by='frame')
    plt.plot(
        sub_df['x_center'],
        sub_df['y_center'],
        marker='o',
        linestyle='-',
        color=colors(i),
        label=f'ID {track_id}'
    )

plt.title("Top 20 Longest-Moving Sperm Cells")
plt.xlabel("X Center Position (pixels)")
plt.ylabel("Y Center Position (pixels)")
plt.gca().invert_yaxis()
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
