import pandas as pd
import matplotlib.pyplot as plt

# 🔁 נתיב לקובץ התוצאה:
csv_path = r"C:\tracformer_modle\trackformer-sperm\progect_yolov8\yolo_output\simple_tracks_Protamine_6h_fly1_sr1.csv"

# 🎯 מזהה הזרעון
target_id = 1  # ← שנה את המספר לפי ה-ID שברצונך לראות

# קריאת הקובץ וסינון לפי ה-ID
df = pd.read_csv(csv_path)
df = df[df['track_id'] == target_id].sort_values(by='frame')

# חישוב מרכז התיבה בכל פריים
df['x_center'] = (df['x1'] + df['x2']) / 2
df['y_center'] = (df['y1'] + df['y2']) / 2

# ציור המסלול
plt.figure(figsize=(8, 6))
plt.plot(df['x_center'], df['y_center'], marker='o', linestyle='-', color='blue', label=f'Track ID {target_id}')
plt.title(f"Trajectory of sperm cell ID {target_id}")
plt.xlabel("X Center Position (pixels)")
plt.ylabel("Y Center Position (pixels)")
plt.gca().invert_yaxis()  # הפוך את ציר Y כי תמונות מתחילות למעלה
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
