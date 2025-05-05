import csv
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# קובץ העקיבה
csv_path = r'yolo_output\sort_tracks.csv'

# אוגרים את כל נקודות התנועה
track_coords = defaultdict(list)
track_frames = Counter()

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        track_id = int(row['track_id'])
        frame = int(row['frame'])
        x1 = float(row['x1'])
        y1 = float(row['y1'])
        x2 = float(row['x2'])
        y2 = float(row['y2'])

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        track_coords[track_id].append((frame, x_center, y_center))
        track_frames[track_id] += 1

# מיון לפי אורך הופעה (כמות פריימים)
top_10_tracks = [track_id for track_id, _ in track_frames.most_common(10)]
top_9_tracks=[x for x in top_10_tracks if x==990 ]
# ציור
plt.figure(figsize=(10, 10))
for track_id in top_9_tracks:
    # ממיינים לפי מספר פריים כדי שהתנועה תצא רציפה
    sorted_coords = sorted(track_coords[track_id], key=lambda x: x[0])
    xs = [x for _, x, _ in sorted_coords]
    ys = [y for _, _, y in sorted_coords]
    plt.plot(xs, ys, marker='o', label=f'ID {track_id}')

plt.gca().invert_yaxis()
plt.title("תנועת 10 הזרעונים הארוכים ביותר (SORT)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
