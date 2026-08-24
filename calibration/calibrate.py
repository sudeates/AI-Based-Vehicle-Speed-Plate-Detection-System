import cv2

video_path = "Peugeot3008_58.MP4"
cap = cv2.VideoCapture(video_path)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"Frame: {frame_number} - Clicked at: ({x},{y})")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti.")
        break

    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", click_event)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('n'):
        continue
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()