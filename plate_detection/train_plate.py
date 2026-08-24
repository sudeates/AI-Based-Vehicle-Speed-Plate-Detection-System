from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="License-plate-detection-1/data.yaml",
        epochs=30,
        imgsz=640,
        batch=16
    )