from ultralytics import YOLO
import cv2

video_path="Peugeot3008_100.MP4"
model=YOLO("yolov8n.pt")
VEHICLE_CLASS_IDS={2:"car",3:"motorcycle",5:"bus",7:"truck"}
cap=cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
speed_ms = 100 / 3.6
distance_per_frame = speed_ms / fps
positions=[]

while True:
    ret,frame=cap.read()
    if not ret:
        break #video has ended
    results=model.track(frame,persist=True,verbose=False,imgsz=1280)[0]
    for box in results.boxes:
        cls_id=int(box.cls[0])
        if box.id is not None:
            if cls_id in VEHICLE_CLASS_IDS:
                x1,y1,x2,y2=map(int,box.xyxy[0])
                cx = (x1+x2)//2
                cy = (y1+y2)//2
                positions.append((cx,cy))

cap.release()
cv2.destroyAllWindows()

print(len(positions))
print(positions[:10])

calibration_table = []
window = 10
for i in range(0, len(positions) - window, window):
    x1, y1 = positions[i]
    x2, y2 = positions[i + window]
    pixel_distance = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
    real_distance = distance_per_frame * window
    mpp = real_distance / pixel_distance
    avg_y = (y1 + y2) / 2
    calibration_table.append((avg_y, mpp))

with open("calibration100.txt", "w") as f:
    for avg_y, mpp in calibration_table:
        f.write(f"{avg_y} {mpp}\n")