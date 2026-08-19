from ultralytics import YOLO
import cv2

model=YOLO("yolov8n.pt")
VEHICLE_CLASS_IDS={2:"car",3:"motorcycle",5:"bus",7:"truck"}

video_path="test.mp4"
cap=cv2.VideoCapture(video_path)

while True:
    ret,frame=cap.read()
    if not ret:
        break #video has ended
    results=model(frame,verbose=False)[0]
    for box in results.boxes:
        cls_id=int(box.cls[0])
        if cls_id in VEHICLE_CLASS_IDS:
            x1,y1,x2,y2=map(int,box.xyxy[0])
            conf=float(box.conf[0])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            label=f"{VEHICLE_CLASS_IDS[cls_id]}:{conf:.2f}"
            cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.imshow("Vehicle Detection",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()