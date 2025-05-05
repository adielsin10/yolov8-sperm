import os
import cv2

# ⚙️ נתיבים
images_dir = r"C:\videos_lsm\frames\Protamine 6h fly1 sr1"  # ← נתיב לתמונות (frame_0000.png וכו')
labels_dir = r"C:\tracformer_modle\trackformer-sperm\progect_yolov8\yolo_output\boxes_Protamine_6h_fly1_sr1\labels"  #
output_dir = r"C:\tracformer_modle\trackformer-sperm\progect_yolov8\check"  # ← איפה לשמור את התמונות עם תיבות

os.makedirs(output_dir, exist_ok=True)

# מעבר על כל התמונות
image_files = sorted([
    f for f in os.listdir(images_dir) if f.endswith('.png')
], key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

for img_name in image_files:
    img_path = os.path.join(images_dir, img_name)
    txt_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                _, xc, yc, bw, bh = map(float, parts[:5])  # מתעלמים מ-confidence

                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, img)

print(f"✅ נוצרו תמונות עם תיבות לבדיקה ויזואלית בתיקייה: {output_dir}")
