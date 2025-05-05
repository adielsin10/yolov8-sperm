import json
import os

# ⚙️ הגדרות - עדכן לפי הצורך:
json_path = r"C:\imgae_of_yolov8\trajectories.json"  # נתיב לקובץ הגיוון שלך (JSON)
labels_dir =r"C:\imgae_of_yolov8\50_images_for_2_model" # תיקייה לשמור בה את קבצי התיוג
image_width = 256  # רוחב התמונה בפיקסלים
image_height = 256  # גובה התמונה בפיקסלים

# יוצרים את תיקיית התיוגים אם לא קיימת
os.makedirs(labels_dir, exist_ok=True)

# טוענים את קובץ ה-JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# עוברים על כל תמונה והבוקסים שלה
for filename, boxes in data.items():
    label_filename = os.path.splitext(filename)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_filename)

    with open(label_path, 'w') as out_file:
        for box in boxes:
            x, y, w, h = box
            x_center = (x + w / 2) / image_width
            y_center = (y + h / 2) / image_height
            w_norm = w / image_width
            h_norm = h / image_height

            out_file.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("✅ סיום! קובצי התיוג בפורמט YOLOv8 נוצרו בתיקייה:", labels_dir)
