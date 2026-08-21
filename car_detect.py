from ultralytics import YOLO
import cv2
import numpy as np

model=YOLO("yolov8n.pt")
VEHICLE_CLASS_IDS={2:"car",3:"motorcycle",5:"bus",7:"truck"}
video_path="Peugeot3008_100.MP4"
cap=cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
previous_positions = {}
frame_counters = {}
ema_speeds = {}
alpha = 0.25
current_speeds = {}
calibration_ys = []
calibration_mpps = []
with open("calibration.txt", "r") as f:
    for line in f:
        y_val, mpp_val = line.split()
        calibration_ys.append(float(y_val))
        calibration_mpps.append(float(mpp_val))
with open("calibration100.txt", "r") as f:
    for line in f:
        y_val, mpp_val = line.split()
        calibration_ys.append(float(y_val))
        calibration_mpps.append(float(mpp_val))
sorted_pairs = sorted(zip(calibration_ys, calibration_mpps))
calibration_ys = [p[0] for p in sorted_pairs]
calibration_mpps = [p[1] for p in sorted_pairs]

while True:
    ret,frame=cap.read()
    if not ret:
        break #video has ended
    results=model.track(frame,persist=True,verbose=False,imgsz=1280)[0]

    # her frame'de sadece en guvenilir araci sec
    best_box = None
    best_conf = 0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASS_IDS:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box

    if best_box is not None:
        box = best_box
        cls_id=int(box.cls[0])
        if box.id is not None:
            track_id=int(box.id)
        else:
            track_id=-1
        if track_id != -1:
            if track_id not in frame_counters:
                frame_counters[track_id] = 0
            frame_counters[track_id] += 1

            x1,y1,x2,y2=map(int,box.xyxy[0])
            cx = (x1+x2)//2
            cy = (y1+y2)//2
            avg_speed = 0
            speed_text = "olculuyor..."

            if track_id in current_speeds:
                speed_text = f"{current_speeds[track_id]:.2f} km/h"

            N=5
            if track_id not in previous_positions:
                previous_positions[track_id] = (cx, cy)
                frame_counters[track_id] = 0
            elif frame_counters[track_id] >= N:
                prev_cx, prev_cy = previous_positions[track_id]
                distance = ((cx-prev_cx)**2 + (cy-prev_cy)**2) ** 0.5
                current_mpp = np.interp(cy, calibration_ys, calibration_mpps)
                distance_meters = distance * current_mpp
                speed_ms = distance_meters * fps / frame_counters[track_id]
                speed_kmh = speed_ms * 3.6

                is_outlier = False
                if track_id in ema_speeds:
                    if abs(speed_kmh - ema_speeds[track_id]) > ema_speeds[track_id] * 0.5:
                        is_outlier = True

                if not is_outlier:
                    if track_id not in ema_speeds:
                        ema_speeds[track_id] = speed_kmh
                    else:
                        ema_speeds[track_id] = alpha * speed_kmh + (1 - alpha) * ema_speeds[track_id]
                    avg_speed = ema_speeds[track_id]
                    current_speeds[track_id] = avg_speed
                    speed_text = f"{avg_speed:.2f} km/h"

                previous_positions[track_id] = (cx, cy)
                frame_counters[track_id] = 0

            conf=float(box.conf[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            label = f"{VEHICLE_CLASS_IDS[cls_id]} {track_id} {speed_text}:{conf:.2f}"
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