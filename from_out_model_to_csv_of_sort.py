import os
import csv

# ⚙️ נתיבים:
labels_folder = r"C:\tracformer_modle\trackformer-sperm\progect_yolov8\yolo_output\boxes_Protamine_6h_fly1_sr1\labels"  # הנתיב לקבצי ה-TXT של YOLO
output_csv = r'yolo_output\sort_input_Protamine_6h_fly1_sr1.csv'   # הנתיב לקובץ ה-CSV של SORT
image_width = 256
image_height = 256

rows = []

# עובר על כל קובץ תיוג
for filename in sorted(os.listdir(labels_folder)):
    if not filename.endswith('.txt'):
        continue

    frame_id = int(os.path.splitext(filename)[0].split('_')[-1])  # מספר פריים מתוך השם

    with open(os.path.join(labels_folder, filename), 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:5])

            # המרה מ-YOLO (מרוכז) ל-corners
            x1 = (x_center - w / 2) * image_width
            y1 = (y_center - h / 2) * image_height
            x2 = (x_center + w / 2) * image_width
            y2 = (y_center + h / 2) * image_height

            # confidence – אם לא קיים בפלט, נשים 1.0
            confidence = 1.0

            rows.append([frame_id, x1, y1, x2, y2, confidence, class_id])

# שמירה לקובץ CSV
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['frame', 'x1', 'y1', 'x2', 'y2', 'confidence', 'class'])  # כותרות
    writer.writerows(rows)

print(f"✅ קובץ SORT נוצר בהצלחה: {output_csv}")
