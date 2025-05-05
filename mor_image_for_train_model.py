import os
from ultralytics import YOLO
import cv2

# טען את המודל המאומן
model = YOLO("runs/detect/train3/weights/best.pt")

# נתיב לתיקיית התמונות
image_dir = r"C:\imgae_of_yolov8\150_image_train"
output_labels_dir = os.path.join(image_dir, "labels")
os.makedirs(output_labels_dir, exist_ok=True)

# עבור כל תמונה
for img_file in sorted(os.listdir(image_dir)):
    if not img_file.endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(image_dir, img_file)
    results = model(image_path)[0]

    # שם קובץ התוצאה
    label_file = os.path.join(output_labels_dir, os.path.splitext(img_file)[0] + ".txt")

    h, w = results.orig_img.shape[:2]
    with open(label_file, "w") as f:
        for box in results.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            # המרה ל-YOLO format (cx, cy, w, h) יחסיים
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

print("הזיהויים נשמרו בתיקיית 'labels'. אפשר לטעון את זה לתוך LabelImg.")
