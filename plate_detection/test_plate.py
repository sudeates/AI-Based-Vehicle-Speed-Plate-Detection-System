from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train-3/weights/best.pt")


image = cv2.imread("image.png")
results = model(image)[0]

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

cv2.imshow("Plate Detection Test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()