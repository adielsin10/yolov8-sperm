from ultralytics import YOLO

# טען את המודל המאומן שלך
model = YOLO("runs/detect/train4/weights/best.pt")

# הרצת חיזוי על תיקיית פריימים
model.predict(
    source=r"C:\videos_lsm\frames\FKY1 SR1",   # ← הנתיב לתיקיית התמונות
    save_txt=True,             # שומר קבצי תיוג בפורמט YOLO
    save=False,                # לא שומר תמונות עם תיבות (אם לא צריך)
    conf=0.25,                 # סף סינון לפי confidence
    imgsz=256,                 # גודל התמונות לפי מה שהתאמנת עליו
    project='yolo_output',     # תיקיית פלט
    name='FKY1 SR1',              # שם תת-תיקייה
    exist_ok=True              # מאפשר דריסה אם כבר קיימת
)

