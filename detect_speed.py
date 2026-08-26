"""
detect_speed.py
-----------------
PROJENIN ANA SCRIPTI. Tek bir video uzerinde:
  1. YOLOv8n + ByteTrack ile araclari tespit edip takip eder,
  2. Her aracin "Sanal Radar" (virtual_loop.py) ile hizini olcer,
  3. Aracin plakasini tespit edip OCR ile okur (plate_reader.py),
  4. Sonuclari ekranda gosterir ve terminale RESULT satiri olarak yazdirir
     (bu satirlar batch_evaluate.py tarafindan otomatik okunur).

Kullanim:
    python detect_speed.py [video_yolu] [--headless]

--headless verilirse ekran penceresi acilmaz (batch_evaluate.py bunu kullanir).
"""

import sys

import cv2
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

from config import (
    DEFAULT_VIDEO_PATH,
    VEHICLE_MODEL_PATH,
    PLATE_MODEL_PATH,
    PLATE_OCR_MODEL,
    VEHICLE_CLASS_IDS,
    SPEED_LIMIT_KMH,
)
from plate_reader import PlateReader, find_best_box, vote_plate
from virtual_loop import VirtualLoop


def process_video(video_path, headless=False):
    vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    plate_model = YOLO(PLATE_MODEL_PATH)
    plate_reader = PlateReader(plate_model, LicensePlateRecognizer(PLATE_OCR_MODEL))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[VIDEO] fps={fps}")

    loop = VirtualLoop(fps, debug=True)

    plate_readings = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = vehicle_model.track(frame, persist=True, verbose=False, imgsz=1280)[0]
        best_box = find_best_box(results.boxes, allowed_class_ids=VEHICLE_CLASS_IDS)

        if best_box is not None and best_box.id is not None:
            track_id = int(best_box.id)
            cls_id = int(best_box.cls[0])
            conf = float(best_box.conf[0])
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])

            # Referans nokta: kutu merkezi. (Alt kenar/y2 denendi ama
            # kalibrasyon tablosuyla uyumsuzluk yarattigi icin merkeze donuldu.)
            ground_y = (y1 + y2) / 2

            loop.update(track_id, frame_idx, ground_y)

            speed = loop.get(track_id)
            speed_text = f"{speed:.2f} km/h" if speed is not None else "olculuyor..."

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size > 0:
                plate_text, plate_box = plate_reader.read(vehicle_crop, x1, y1)
                if plate_text is not None:
                    plate_readings.setdefault(track_id, []).append(plate_text)
                if plate_box is not None:
                    bx1, by1, bx2, by2 = plate_box
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)

            label = f"{VEHICLE_CLASS_IDS[cls_id]} {track_id} {speed_text}:{conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        _draw_summary_overlay(frame, loop, plate_readings)

        if not headless:
            cv2.imshow("Vehicle Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    _print_final_results(loop, plate_readings)


def _draw_summary_overlay(frame, loop, plate_readings):
    """Ekranin sol ust kosesine, olculen tum araclarin hiz/plaka ozetini yazar."""
    y_offset = 30
    for track_id, speed in loop.speeds.items():
        text = f"ID {track_id}: {speed:.1f} km/h"
        plate = vote_plate(plate_readings.get(track_id))
        if plate:
            text += f" | {plate}"
        over_limit = speed > SPEED_LIMIT_KMH
        if over_limit:
            text += " !!! LIMIT ASIMI !!!"
        color = (0, 0, 255) if over_limit else (255, 255, 255)
        cv2.putText(frame, text, (11, y_offset + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25


def _print_final_results(loop, plate_readings):
    """Video bitince terminale ozet basar. RESULT satirlari batch_evaluate.py
    tarafindan regex ile parse edilir - formatini degistirirsen orayi da guncelle."""
    print("\n--- Video bitti ---")
    for track_id, speed in loop.speeds.items():
        plate = vote_plate(plate_readings.get(track_id))
        print(f"ID {track_id}: {speed:.1f} km/h | plaka: {plate}")
        print(f"RESULT track={track_id} speed_kmh={speed:.2f} plate={plate}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    is_headless = "--headless" in sys.argv
    process_video(path, headless=is_headless)
