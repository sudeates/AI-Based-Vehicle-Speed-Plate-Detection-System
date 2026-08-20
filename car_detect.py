from ultralytics import YOLO
import cv2

model=YOLO("yolov8n.pt")
VEHICLE_CLASS_IDS={2:"car",3:"motorcycle",5:"bus",7:"truck"}

video_path="cars.mp4"
cap=cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
meters_per_pixel =0.00764
previous_positions = {}
speed_history = {}
current_speeds = {}
while True:
    ret,frame=cap.read()
    if not ret:
        break #video has ended
    results=model.track(frame,persist=True,verbose=False,imgsz=1280)[0]
    for box in results.boxes:
        cls_id=int(box.cls[0])
        if box.id is not None:
            track_id=int(box.id)
        else:
            track_id=-1
        if track_id != -1:
            if cls_id in VEHICLE_CLASS_IDS:
                x1,y1,x2,y2=map(int,box.xyxy[0])
                cx = (x1+x2)//2
                cy = (y1+y2)//2
                speed_kmh = 0
                avg_speed = 0
                if track_id in previous_positions:
                    prev_cx, prev_cy = previous_positions[track_id]
                    distance = ((cx-prev_cx)**2 + (cy-prev_cy)**2) ** 0.5
                    distance_meters = distance * meters_per_pixel
                    speed_ms = distance_meters * fps
                    speed_kmh = speed_ms * 3.6
                    if track_id not in speed_history:
                        speed_history[track_id] = []
                    speed_history[track_id].append(speed_kmh)
                    speed_history[track_id] = speed_history[track_id][-30:]
                    avg_speed = sum(speed_history[track_id]) / len(speed_history[track_id])
                    current_speeds[track_id] = avg_speed
                previous_positions[track_id] = (cx, cy)
                conf=float(box.conf[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                label=f"{VEHICLE_CLASS_IDS[cls_id]} {track_id} {avg_speed:.2f} km/h:{conf:.2f}"
                cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    y_offset = 30
    for track_id, speed in current_speeds.items():
        text = f"ID {track_id}: {speed:.1f} km/h"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
    cv2.imshow("Vehicle Detection",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()