import cv2

video_path="cars.mp4"
cap=cv2.VideoCapture(video_path)
ret,frame=cap.read()

if not ret:
    print("Failed to read video")
    exit()
def click_event(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x},{y})")
cv2.imshow("Frame",frame)
cv2.setMouseCallback("Frame",click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
