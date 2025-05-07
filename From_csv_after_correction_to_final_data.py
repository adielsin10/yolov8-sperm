import pandas as pd
import os

def extract_tracks_summary_readable(csv_path, video_name, output_path):
    # טען את הקובץ
    df = pd.read_csv(csv_path)

    # ודא שיש את העמודות הדרושות
    required_cols = {"frame", "track_id", "x_center", "y_center"}
    if not required_cols.issubset(df.columns):
        raise ValueError("חסרות עמודות בקובץ הקלט")

    # רשימת נתונים למסלול
    output_data = []

    # קיבוץ לפי track_id
    for track_id, group in df.groupby("track_id"):
        group_sorted = group.sort_values("frame")

        # הפיכת קואורדינטות לפורמט קריא: (x,y) → ...
        coords = list(zip(group_sorted["x_center"], group_sorted["y_center"]))
        coord_str = " , ".join([f"({round(x, 1)}, {round(y, 1)})" for x, y in coords])

        duration = group_sorted["frame"].nunique()

        output_data.append({
            "track_id": track_id,
            "coordinates": coord_str,
            "duration_frames": duration,
            "video_name": video_name
        })

    # יצירת טבלה חדשה
    output_df = pd.DataFrame(output_data)

    # שמירה לקובץ
    output_df.to_csv(output_path, index=False)
    print(f"✅ הקובץ הקריא נשמר:\n{output_path}")





csv_path = r"C:\files_of_csv\simple_tracks_protamine 48h #2 sr_good.csv"
video_name = "simple_tracks_protamine 48h #2 sr_good.mp4"
output_path = r"C:\files_of_csv\final file csv\end_simple_tracks_protamine 48h #2 sr_good.csv"
extract_tracks_summary_readable(csv_path, video_name, output_path)