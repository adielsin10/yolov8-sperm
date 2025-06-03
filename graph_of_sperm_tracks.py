import pandas as pd
import matplotlib.pyplot as plt

# טען את הקובץ המסונן
filtered_path = r"C:\files_of_csv\simple_tracks_kk fly5 sr2_good.csv"
df = pd.read_csv(filtered_path)

# יצירת גרף
plt.figure(figsize=(10, 8))
for track_id, group in df.groupby("track_id"):
    group_sorted = group.sort_values("frame")
    plt.plot(group_sorted["x_center"], group_sorted["y_center"], label=f"ID {track_id}")

plt.title("Filtered Sperm Tracks")
plt.xlabel("x_center")
plt.ylabel("y_center")
plt.gca().invert_yaxis()  # להפוך את כיוון ציר Y כדי שיתאים לתמונה
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

