from ultralytics import YOLO
import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer


def vote_plate(readings):
    if not readings:
        return None
    lengths = [len(r) for r in readings]
    most_common_length = max(set(lengths), key=lengths.count)
    filtered = [r for r in readings if len(r) == most_common_length]
    result = ""
    for i in range(most_common_length):
        chars_at_position = [r[i] for r in filtered]
        best_char = max(set(chars_at_position), key=chars_at_position.count)
        result += best_char
    return result


def blur_score(img_gray):
    # Laplacian varyansi: yuksek = keskin/net, dusuk = bulanik (motion blur)
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()


# fast-plate-ocr dusuk sharpness skorlarinda bile dogru okudugu icin
# bunu artik sert bir filtre olarak degil, sadece cok kucuk/anlamsiz
# crop'lari elemek icin dusuk bir esik olarak kullaniyoruz.
SHARPNESS_THRESHOLD = 20
SHARPNESS_DEBUG = False

# Sharpness skorunun mesafeye gore degil sadece bulanikliga gore
# degismesi icin plaka crop'u her zaman ayni hedef genislige buyutuluyor
# (sabit fx/fy orani yerine).
TARGET_PLATE_WIDTH = 300
MIN_NATIVE_PLATE_WIDTH = 15  # bundan kucuk native crop'larda OCR denemeye deger yok


model = YOLO("yolov8n.pt")
plate_model = YOLO("runs/detect/train-3/weights/best.pt")
plate_reader = LicensePlateRecognizer('cct-s-v2-global-model')
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
video_path = "videos/Peugeot3008_100.MP4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

previous_positions = {}
frame_counters = {}
ema_speeds = {}
alpha = 0.25
current_speeds = {}
plate_readings = {}

calibration_ys = []
calibration_mpps = []
with open("calibration/calibration55.txt", "r") as f:
    for line in f:
        y_val, mpp_val = line.split()
        calibration_ys.append(float(y_val))
        calibration_mpps.append(float(mpp_val))
with open("calibration/calibration100.txt", "r") as f:
    for line in f:
        y_val, mpp_val = line.split()
        calibration_ys.append(float(y_val))
        calibration_mpps.append(float(mpp_val))
sorted_pairs = sorted(zip(calibration_ys, calibration_mpps))
calibration_ys = [p[0] for p in sorted_pairs]
calibration_mpps = [p[1] for p in sorted_pairs]

while True:
    ret, frame = cap.read()
    if not ret:
        break  # video has ended
    results = model.track(frame, persist=True, verbose=False, imgsz=1280)[0]

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
        cls_id = int(box.cls[0])
        if box.id is not None:
            track_id = int(box.id)
        else:
            track_id = -1
        if track_id != -1:
            if track_id not in frame_counters:
                frame_counters[track_id] = 0
            frame_counters[track_id] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            avg_speed = 0
            speed_text = "olculuyor..."

            if track_id in current_speeds:
                speed_text = f"{current_speeds[track_id]:.2f} km/h"

            min_y = min(calibration_ys)
            max_y = max(calibration_ys)
            if cy < min_y or cy > max_y:
                continue

            N = 5
            if track_id not in previous_positions:
                previous_positions[track_id] = (cx, cy)
                frame_counters[track_id] = 0
            elif frame_counters[track_id] >= N:
                prev_cx, prev_cy = previous_positions[track_id]
                distance = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
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

            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size > 0:
                plate_results = plate_model(vehicle_crop, verbose=False)[0]

                # sadece en yuksek konfidansli plaka kutusunu isle,
                # coklu/gurultulu tespitlerin voting'i kirletmesini onle
                best_plate_box = None
                best_plate_conf = 0
                for plate_box in plate_results.boxes:
                    pconf = float(plate_box.conf[0])
                    if pconf > best_plate_conf:
                        best_plate_conf = pconf
                        best_plate_box = plate_box

                if best_plate_box is not None:
                    plate_box = best_plate_box
                    px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                    plate_conf = float(plate_box.conf[0])

                    real_px1 = x1 + px1
                    real_py1 = y1 + py1
                    real_px2 = x1 + px2
                    real_py2 = y1 + py2

                    cv2.rectangle(frame, (real_px1, real_py1), (real_px2, real_py2), (255, 0, 0), 2)

                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                    native_w = plate_crop.shape[1] if plate_crop.size > 0 else 0

                    if plate_crop.size > 0 and native_w >= MIN_NATIVE_PLATE_WIDTH:
                        # sabit hedef genislige buyut, boylece sharpness skoru
                        # mesafeden degil sadece bulaniklikdan etkilensin
                        scale = TARGET_PLATE_WIDTH / native_w
                        plate_crop_big = cv2.resize(plate_crop, None, fx=scale, fy=scale,
                                                     interpolation=cv2.INTER_CUBIC)
                        gray = cv2.cvtColor(plate_crop_big, cv2.COLOR_BGR2GRAY)

                        sharpness = blur_score(gray)
                        if SHARPNESS_DEBUG:
                            print(f"[SHARPNESS] track={track_id} native_w={native_w} score={sharpness:.1f}")

                        if sharpness >= SHARPNESS_THRESHOLD:
                            pred = plate_reader.run(plate_crop_big, return_confidence=True)[0]
                            cleaned = ''.join(c for c in pred.plate if c.isalnum()).upper()
                            if len(cleaned) >= 6:
                                if track_id not in plate_readings:
                                    plate_readings[track_id] = []
                                plate_readings[track_id].append(cleaned)

                    cv2.putText(frame, f"plaka {plate_conf:.2f}", (real_px1, real_py1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            label = f"{VEHICLE_CLASS_IDS[cls_id]} {track_id} {speed_text}:{conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    y_offset = 30
    for track_id, speed in current_speeds.items():
        text = f"ID {track_id}: {speed:.1f} km/h"
        if track_id in plate_readings and len(plate_readings[track_id]) > 0:
            best_plate = vote_plate(plate_readings[track_id])
            text += f" | {best_plate}"
        cv2.putText(frame, text, (11, y_offset + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25

    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\n--- Video bitti ---")
for track_id in current_speeds:
    plate = vote_plate(plate_readings.get(track_id))
    print(f"ID {track_id}: {current_speeds[track_id]:.1f} km/h | plaka: {plate}")

cap.release()
cv2.destroyAllWindows()