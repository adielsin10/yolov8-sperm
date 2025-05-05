from ultralytics import YOLO

# טען את המודל החדש
model = YOLO("runs/detect/train4/weights/best.pt")

# בצע חיזוי (inference) על תיקייה שלמה
results = model.predict(
    source="C:\imgae_of_yolov8\image_for_test_model",  # איפה התמונות שלך נמצאות
    save=True,                                # לשמור תמונות מזוהות
    save_txt=True,                            # לשמור גם קובץ TXT של תוצאות
    imgsz=256                                 # גודל תמונה (כמו באימון)
)
