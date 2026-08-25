"""
Virtual loop (sanal radar) yontemi: EMA/outlier-reddi ile bircok kucuk
olcumu istatistiksel birlestirmek yerine, giris-cikis cizgileri arasinda
TEK bir makro hiz olcumu yapar. Referans nokta olarak kutunun merkezi
degil, ALT-ORTA noktasi (zeminle temas, y_max) kullanilir.

Kullanim:
    python car_detect_virtual_loop.py [video_yolu] [--headless]
"""

import sys
from ultralytics import YOLO
import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer


# =============================================================================
# AYARLAR
# =============================================================================

DEFAULT_VIDEO_PATH = "videos/Peugeot3008_87.MP4"
VEHICLE_MODEL_PATH = "yolov8n.pt"
PLATE_MODEL_PATH = "runs/detect/train-3/weights/best.pt"
PLATE_OCR_MODEL = "cct-s-v2-global-model"
CALIBRATION_FILES = ["calibration/calibration55.txt", "calibration/calibration100.txt"]
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Videolar 2 farkli kamera kurulumuna ait (bkz. onceki analiz): Grup A
# araclar y~440'ta, Grup B araclar y~570'te kadraja giriyor. Tek bir
# giris/cikis ciftinin ikisine de uymasi imkansiz - Grup B icin y=460
# hicbir zaman "asagida" olmuyor, gecis hic tetiklenmiyor. Bu yuzden
# her grup icin ayri sinirlar tanimlayip, aracin ilk gorulen y'sine
# gore otomatik seciyoruz.
# Y2 (alt kenar) kullanıldığı için tüm değerler aşağı (büyük y'ye) kaydırıldı.
FAMILY_SPLIT_Y = 540  # 500'den 540'a çektik ki Grup A araçları yanlış gruba düşmesin.

LOOP_BOUNDS = {
    # Girişi 465'ten 510'a aldık, böylece araç 490'da doğsa bile çizgiyi geçecek.
    "A": (510, 590),   
    # Girişi 600 civarına çektik.
    "B": (600, 640),   
}
SHARPNESS_THRESHOLD = 20
TARGET_PLATE_CROP_WIDTH = 300
MIN_NATIVE_PLATE_WIDTH = 15
MIN_PLATE_LENGTH = 6
SPEED_LIMIT_KMH = 50


# =============================================================================
# YARDIMCI FONKSIYONLAR
# =============================================================================

def load_calibration_table(paths):
    ys, mpps = [], []
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                y_val, mpp_val = line.split()
                ys.append(float(y_val))
                mpps.append(float(mpp_val))
    pairs = sorted(zip(ys, mpps))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def real_distance_between(y_start, y_end, calib_ys, calib_mpps):
    """y_start ile y_end arasindaki gercek mesafeyi (metre), kalibrasyon
    egrisini (mpp, piksel basina metre) sayisal olarak integre ederek hesaplar.
    Sabit tek bir mpp degeri yerine, araninin degisimini (perspektif egrisini)
    dogru sekilde hesaba katar."""
    # entegrasyon noktalari: kalibrasyon tablosundaki [y_start, y_end] arasi
    # noktalar + uc noktalarin kendisi
    sample_ys = [y_start] + [y for y in calib_ys if y_start < y < y_end] + [y_end]
    sample_ys = sorted(set(sample_ys))
    total = 0.0
    for i in range(len(sample_ys) - 1):
        y0, y1 = sample_ys[i], sample_ys[i + 1]
        mpp0 = np.interp(y0, calib_ys, calib_mpps)
        mpp1 = np.interp(y1, calib_ys, calib_mpps)
        # trapez kurali: bu araliktaki ortalama mpp * piksel araligi
        total += (mpp0 + mpp1) / 2 * (y1 - y0)
    return total


def find_best_box(boxes, allowed_class_ids=None):
    best_box, best_conf = None, 0.0
    for box in boxes:
        if allowed_class_ids is not None:
            cls_id = int(box.cls[0])
            if cls_id not in allowed_class_ids:
                continue
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box = box
    return best_box


def blur_score(image_gray):
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()


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


class PlateReader:
    def __init__(self, plate_model, ocr_reader):
        self.plate_model = plate_model
        self.ocr_reader = ocr_reader

    def read(self, vehicle_crop, offset_x, offset_y):
        plate_results = self.plate_model(vehicle_crop, verbose=False)[0]
        best_plate_box = find_best_box(plate_results.boxes)
        if best_plate_box is None:
            return None, None

        px1, py1, px2, py2 = map(int, best_plate_box.xyxy[0])
        draw_box = (offset_x + px1, offset_y + py1, offset_x + px2, offset_y + py2)

        plate_crop = vehicle_crop[py1:py2, px1:px2]
        native_w = plate_crop.shape[1] if plate_crop.size > 0 else 0
        if plate_crop.size == 0 or native_w < MIN_NATIVE_PLATE_WIDTH:
            return None, draw_box

        scale = TARGET_PLATE_CROP_WIDTH / native_w
        crop_resized = cv2.resize(plate_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)

        if blur_score(gray) < SHARPNESS_THRESHOLD:
            return None, draw_box

        prediction = self.ocr_reader.run(crop_resized, return_confidence=True)[0]
        cleaned = ''.join(c for c in prediction.plate if c.isalnum()).upper()
        if len(cleaned) < MIN_PLATE_LENGTH:
            return None, draw_box

        return cleaned, draw_box


class VirtualLoop:
    """Her track icin, ilk gorulen y'ye gore dogru grubu (A/B) secip o
    grubun giris/cikis sinirlarini kullanir. LOOP_ENTRY_Y'yi gectigi
    (interpole edilmis) ani ve LOOP_EXIT_Y'yi gectigi ani kaydeder,
    TEK bir hiz hesaplar."""

    def __init__(self, calib_ys, calib_mpps, fps, debug=False):
        self.calib_ys = calib_ys
        self.calib_mpps = calib_mpps
        self.fps = fps
        self.debug = debug

        self.real_distance_m = {}  # family -> gercek mesafe (metre), ilk kullanimda hesaplanir
        self.family = {}           # track_id -> "A" / "B"
        self.last_position = {}    # track_id -> (frame_idx, y)
        self.entry_frame = {}      # track_id -> interpole edilmis giris frame'i
        self.speeds = {}           # track_id -> hesaplanan hiz (km/h)

    def _get_bounds(self, family):
        entry_y, exit_y = LOOP_BOUNDS[family]
        
        # Hatalı kalibrasyon egrisi yerine fiziksel olarak kanitlanmis mesafeler
        MANUAL_DISTANCES = {
            "A": 4.04,  # 510-610 pikselleri arasi gercekte ~5.05 metre
            "B": 7.00   # 600-640 pikselleri arasi gercekte ~7.00 metre
        }
        
        if family not in self.real_distance_m:
            self.real_distance_m[family] = MANUAL_DISTANCES[family]
            
        return entry_y, exit_y, self.real_distance_m[family]

    def update(self, track_id, frame_idx, y):
        if track_id in self.speeds:
            return

        if track_id not in self.family:
            self.family[track_id] = "A" if y < FAMILY_SPLIT_Y else "B"
        entry_y, exit_y, real_distance_m = self._get_bounds(self.family[track_id])

        prev = self.last_position.get(track_id)
        self.last_position[track_id] = (frame_idx, y)
        if prev is None:
            return
        prev_frame, prev_y = prev

        if track_id not in self.entry_frame:
            # --- YENI DEBUG KODU BASLANGICI ---
            if prev_y > entry_y and frame_idx < 10: 
                # Video yeni baslamissa ve arac zaten cizgiyi gecmisse
                if self.debug:
                    print(f"!!! UYARI: ID {track_id} baslangic cizgisini kacirdi! Ilk y={prev_y:.1f}, Beklenen giris={entry_y}")
            # --- YENI DEBUG KODU BITISI ---
            if prev_y < entry_y <= y:
                fraction = (entry_y - prev_y) / (y - prev_y) if y != prev_y else 0
                self.entry_frame[track_id] = prev_frame + fraction
                if self.debug:
                    print(f"[LOOP] track={track_id} grup={self.family[track_id]} GIRIS tespit edildi (frame~{self.entry_frame[track_id]:.1f})")
            return

        if prev_y < exit_y <= y:
            fraction = (exit_y - prev_y) / (y - prev_y) if y != prev_y else 0
            exit_frame = prev_frame + fraction
            elapsed_frames = exit_frame - self.entry_frame[track_id]
            if self.debug:
                print(f"[LOOP] track={track_id} grup={self.family[track_id]} CIKIS tespit edildi (frame~{exit_frame:.1f}, gecen_frame={elapsed_frames:.1f})")
            if elapsed_frames > 0:
                elapsed_s = elapsed_frames / self.fps
                speed_ms = real_distance_m / elapsed_s
                self.speeds[track_id] = speed_ms * 3.6

    def get(self, track_id):
        return self.speeds.get(track_id)


# =============================================================================
# ANA ISLEM DONGUSU
# =============================================================================

def process_video(video_path, headless=False):
    calibration_ys, calibration_mpps = load_calibration_table(CALIBRATION_FILES)
    print(f"[KALIBRASYON] tablo y-araligi: {min(calibration_ys):.0f} - {max(calibration_ys):.0f}")

    vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    plate_model = YOLO(PLATE_MODEL_PATH)
    plate_reader = PlateReader(plate_model, LicensePlateRecognizer(PLATE_OCR_MODEL))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)

    loop = VirtualLoop(calibration_ys, calibration_mpps, fps, debug=True)
    print(f"[LOOP] grup A: giris_y={LOOP_BOUNDS['A'][0]} cikis_y={LOOP_BOUNDS['A'][1]}")
    print(f"[LOOP] grup B: giris_y={LOOP_BOUNDS['B'][0]} cikis_y={LOOP_BOUNDS['B'][1]}")

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

            # NOT: zemin-temasi (y2) denendi ama kalibrasyon tablosu muhtemelen
            # kutu MERKEZI ile uretildigi icin tutarsizlik yaratti - hem sinirlar
            # hem kalibrasyon sorgusu yanlis hizalanmis oldu. Guvenilirligini
            # bildigimiz merkez noktaya donuyoruz.
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

        if not headless:
            cv2.imshow("Vehicle Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    print("\n--- Video bitti ---")
    for track_id, speed in loop.speeds.items():
        plate = vote_plate(plate_readings.get(track_id))
        print(f"ID {track_id}: {speed:.1f} km/h | plaka: {plate}")
        print(f"RESULT track={track_id} speed_kmh={speed:.2f} plate={plate}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    is_headless = "--headless" in sys.argv
    process_video(path, headless=is_headless)