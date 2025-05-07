import os
import cv2
from glob import glob

# ⚙️ נתיבים — עדכן לפי הצורך
images_dir = r"C:\videos_lsm_test_yolov_8\frames_lsm\protamine 48h #2 sr" # ← תיקיית התמונות (frame_0001.png וכו')
labels_dir = r"C:\tracformer_modle\trackformer-sperm\progect_yolov8\yolo_output\protamine 48h #2 sr\labels"  # ← תיקיית קבצי YOLO.txt
output_video = r"C:\videos_lsm_test_yolov_8\out_yolov\protamine 48h #2 sr.mp4"
image_width, image_height = 256, 256  # ← גודל התמונה המקורי

# ✨ איסוף התמונות לפי סדר
image_files = sorted(glob(os.path.join(images_dir, "*.png")))
if not image_files:
    raise FileNotFoundError("❌ לא נמצאו תמונות בתיקיה.")

# הגדרת וידאו
sample_img = cv2.imread(image_files[0])
h, w, _ = sample_img.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, 10, (w, h))  # 10 FPS

# יצירת הווידאו
for img_path in image_files:
    img_name = os.path.basename(img_path)
    label_name = os.path.splitext(img_name)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_name)

    frame = cv2.imread(img_path)

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id, xc, yc, bw, bh = map(float, parts[:5])
                    # המרה לקואורדינטות מקוריות
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    # ציור תיבה
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    video_writer.write(frame)

video_writer.release()
print(f"✅ וידאו נוצר בהצלחה: {output_video}")
