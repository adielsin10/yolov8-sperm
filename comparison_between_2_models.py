from ultralytics import YOLO

def compare_models(model_path_1, model_path_2, data_yaml):
    model1 = YOLO(model_path_1)
    model2 = YOLO(model_path_2)

    print(f"\n🔍 Running validation for {model_path_1} ...")
    metrics1 = model1.val(data=data_yaml).results_dict

    print(f"\n🔍 Running validation for {model_path_2} ...")
    metrics2 = model2.val(data=data_yaml).results_dict

    print("\n📊 Comparison:")
    print(f"{'Metric':<20}{'Model 1':>12}{'Model 2':>12}{'Better':>10}")
    print("-" * 54)

    keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "fitness"
    ]
    names = {
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
        "metrics/mAP50(B)": "mAP@50",
        "metrics/mAP50-95(B)": "mAP@50-95",
        "fitness": "Fitness"
    }

    for k in keys:
        val1 = float(metrics1[k])
        val2 = float(metrics2[k])
        better = "Model 1" if val1 > val2 else ("Model 2" if val2 > val1 else "-")
        print(f"{names[k]:<20}{val1:>12.3f}{val2:>12.3f}{better:>10}")


compare_models(
    model_path_1='runs/detect/train4/weights/best.pt',
    model_path_2='runs/detect/train2/weights/best.pt',
    data_yaml='C:/tracformer_modle/trackformer-sperm/progect_yolov8/dataset/data.yaml'
)
