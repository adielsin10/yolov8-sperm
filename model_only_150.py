from ultralytics import YOLO

# טען את המודל
model = YOLO('yolov8n.pt')  # או מודל אחר אם תרצה (s, m, l וכו')

# התחלת אימון עם Early Stopping וגרפים בלייב
model.train(
    data=r"C:\tracformer_modle\trackformer-sperm\progect_yolov8\dataset\data.yaml",   # 🔵 תעדכן כאן לנתיב של ה-data.yaml שלך
    epochs=100,                 # מאפשר מספיק זמן לאימון אם צריך
    imgsz=256,
    batch=16,
    patience=15,                # 🛑 תעצור אם אין שיפור במשך 15 אפוקים
    optimizer='SGD',
    lr0=0.001,
    plots=True
)
