"""
Vehicle speed + license plate detection system.

Bir yol videosundaki araclari tespit edip takip eder, hizlarini olcer,
plakalarini okur ve hiz limitini asan araclari isaretler.

Kullanim:
    python car_detect.py [video_yolu] [--headless]

    video_yolu   : isteğe bagli, verilmezse DEFAULT_VIDEO_PATH kullanilir.
    --headless   : pencere acmadan, sadece konsola sonuc yazar (toplu test icin).
"""

import sys
from ultralytics import YOLO
import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer


# =============================================================================
# AYARLAR
# =============================================================================

DEFAULT_VIDEO_PATH = "videos/Peugeot3008_100.MP4"
VEHICLE_MODEL_PATH = "yolov8n.pt"
PLATE_MODEL_PATH = "runs/detect/train-3/weights/best.pt"
PLATE_OCR_MODEL = "cct-s-v2-global-model"
CALIBRATION_FILES = ["calibration/calibration55.txt", "calibration/calibration100.txt"]

VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Hiz olcumu
MEASUREMENT_WINDOW_FRAMES = 5      # kac frame'de bir hiz ornegi alinir
POSITION_SMOOTH_WINDOW = 3         # kutu titremesini azaltmak icin hareketli ortalama penceresi
EMA_ALPHA = 0.25                   # hiz yumusatma katsayisi
EMA_WARMUP_COUNT = 3               # EMA baslamadan once biriktirilecek olcum sayisi (medyanla baslatilir)
OUTLIER_MIN_DEVIATION_KMH = 15     # outlier esiginin mutlak tabani (oransal esik cok kucukken kilitlenmeyi onler)
OUTLIER_RELATIVE_DEVIATION = 0.5   # outlier esigi: mevcut tahminin bu oranindan fazla sapma

# Plaka tespiti / OCR
SHARPNESS_THRESHOLD = 20           # bu skorun altindaki (cok bulanik) crop'lar OCR'a girmez
TARGET_PLATE_CROP_WIDTH = 300      # plaka crop'u bu genislige buyutulur (mesafeden bagimsiz sharpness icin)
MIN_NATIVE_PLATE_WIDTH = 15        # bundan kucuk native crop'larda OCR denemeye deger yok
MIN_PLATE_LENGTH = 6               # bundan kisa OCR sonuclari gecersiz sayilir

# Genel
SPEED_LIMIT_KMH = 50


# =============================================================================
# YARDIMCI FONKSIYONLAR
# =============================================================================

def load_calibration_table(paths):
    """Bir ya da daha fazla kalibrasyon dosyasini (y_piksel, metre_per_piksel
    ciftleri) okuyup, y'ye gore sirali tek bir tabloya birlestirir."""
    ys, mpps = [], []
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                y_val, mpp_val = line.split()
                ys.append(float(y_val))
                mpps.append(float(mpp_val))
    pairs = sorted(zip(ys, mpps))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def find_best_box(boxes, allowed_class_ids=None):
    """Verilen kutular arasindan en yuksek konfidansliyi dondurur.
    allowed_class_ids verilirse sadece o siniflar arasindan secer."""
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
    """Laplacian varyansi: yuksek = keskin/net, dusuk = bulanik (motion blur)."""
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()


def vote_plate(readings):
    """Birden fazla plaka okumasi arasinda, pozisyon bazinda cogunluk oyu
    ile en olasi plaka dizisini olusturur. Farkli uzunluktaki okumalar
    en sik gorulen uzunluga gore filtrelenir."""
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


class SpeedEstimator:
    """Bir track_id icin, gurultulu ham hiz olcumlerini EMA (exponential
    moving average) ile yumusatir. Ilk birkac olcumun medyanindan baslar
    (tek bir hatali ilk deger EMA'yi kalici olarak yanlis bir noktada
    kilitleyebildigi icin), sonrasinda mevcut tahminden asiri sapan
    olcumleri outlier olarak reddeder."""

    def __init__(self, alpha=EMA_ALPHA, warmup_count=EMA_WARMUP_COUNT):
        self.alpha = alpha
        self.warmup_count = warmup_count
        self.warmup_buffers = {}   # track_id -> [ham olcumler]
        self.estimates = {}        # track_id -> guncel EMA tahmini

    def update(self, track_id, raw_speed_kmh):
        """Yeni bir ham hiz olcumu ekler, guncel tahmini dondurur.
        Henuz warmup tamamlanmadiysa None doner (tahmin yok demektir)."""
        if track_id not in self.estimates:
            buffer = self.warmup_buffers.setdefault(track_id, [])
            buffer.append(raw_speed_kmh)
            if len(buffer) < self.warmup_count:
                return None
            sorted_vals = sorted(buffer)
            self.estimates[track_id] = sorted_vals[len(sorted_vals) // 2]
            return self.estimates[track_id]

        current = self.estimates[track_id]
        deviation_threshold = max(current * OUTLIER_RELATIVE_DEVIATION, OUTLIER_MIN_DEVIATION_KMH)
        if abs(raw_speed_kmh - current) > deviation_threshold:
            return current  # outlier, tahmin degismedi

        self.estimates[track_id] = self.alpha * raw_speed_kmh + (1 - self.alpha) * current
        return self.estimates[track_id]

    def get(self, track_id):
        return self.estimates.get(track_id)


class PlateReader:
    """Bir arac kutusu icinden plaka tespit edip okur. Sadece en yuksek
    konfidansli plaka kutusunu ve yeterince net (bulanik olmayan)
    crop'lari isler."""

    def __init__(self, plate_model, ocr_reader):
        self.plate_model = plate_model
        self.ocr_reader = ocr_reader

    def read(self, vehicle_crop, frame, offset_x, offset_y):
        """vehicle_crop icinde plaka arar. Bulursa (plaka_metni, cizim_kutusu)
        dondurur; bulamazsa (None, None)."""
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
            return None, draw_box  # cok bulanik, OCR'a sokma

        prediction = self.ocr_reader.run(crop_resized, return_confidence=True)[0]
        cleaned = ''.join(c for c in prediction.plate if c.isalnum()).upper()
        if len(cleaned) < MIN_PLATE_LENGTH:
            return None, draw_box

        return cleaned, draw_box


# =============================================================================
# ANA ISLEM DONGUSU
# =============================================================================

def process_video(video_path, headless=False):
    calibration_ys, calibration_mpps = load_calibration_table(CALIBRATION_FILES)
    min_calibrated_y, max_calibrated_y = min(calibration_ys), max(calibration_ys)

    vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    plate_model = YOLO(PLATE_MODEL_PATH)
    plate_reader = PlateReader(plate_model, LicensePlateRecognizer(PLATE_OCR_MODEL))
    speed_estimator = SpeedEstimator()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)

    position_history = {}      # track_id -> son POSITION_SMOOTH_WINDOW (cx, cy)
    previous_position = {}     # track_id -> (cx, cy) son hiz olcumunun yapildigi nokta
    frames_since_measurement = {}  # track_id -> son olcumden bu yana gecen frame sayisi
    plate_readings = {}        # track_id -> [okunan plaka metinleri]
    current_speeds = {}        # track_id -> en guncel hiz tahmini (km/h)

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

            # Kutu titremesini (jitter) kaynaginda azaltmak icin, son birkac
            # tespitin hareketli ortalamasi uzerinden merkez konum hesaplanir.
            raw_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            history = position_history.setdefault(track_id, [])
            history.append(raw_center)
            del history[:-POSITION_SMOOTH_WINDOW]
            cx = sum(p[0] for p in history) / len(history)
            cy = sum(p[1] for p in history) / len(history)

            speed_text = "olculuyor..."
            if track_id in current_speeds:
                speed_text = f"{current_speeds[track_id]:.2f} km/h"

            if min_calibrated_y <= cy <= max_calibrated_y:
                frames_since_measurement[track_id] = frames_since_measurement.get(track_id, 0) + 1

                if track_id not in previous_position:
                    previous_position[track_id] = (cx, cy)
                    frames_since_measurement[track_id] = 0
                elif frames_since_measurement[track_id] >= MEASUREMENT_WINDOW_FRAMES:
                    prev_cx, prev_cy = previous_position[track_id]
                    pixel_distance = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                    meters_per_pixel = np.interp(cy, calibration_ys, calibration_mpps)
                    distance_m = pixel_distance * meters_per_pixel
                    elapsed_s = frames_since_measurement[track_id] / fps
                    raw_speed_kmh = (distance_m / elapsed_s) * 3.6

                    estimate = speed_estimator.update(track_id, raw_speed_kmh)
                    if estimate is not None:
                        current_speeds[track_id] = estimate
                        speed_text = f"{estimate:.2f} km/h"

                    previous_position[track_id] = (cx, cy)
                    frames_since_measurement[track_id] = 0

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size > 0:
                plate_text, plate_box = plate_reader.read(vehicle_crop, frame, x1, y1)
                if plate_text is not None:
                    plate_readings.setdefault(track_id, []).append(plate_text)
                if plate_box is not None:
                    bx1, by1, bx2, by2 = plate_box
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                    cv2.putText(frame, "plaka", (bx1, by1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            label = f"{VEHICLE_CLASS_IDS[cls_id]} {track_id} {speed_text}:{conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        draw_speed_overlay(frame, current_speeds, plate_readings)

        if not headless:
            cv2.imshow("Vehicle Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    report_results(current_speeds, plate_readings)


def draw_speed_overlay(frame, current_speeds, plate_readings):
    """Ekranin sol ustune, tespit edilen her arac icin hiz + plaka ozetini yazar.
    Limit asan araclar kirmizi ile vurgulanir."""
    y_offset = 30
    for track_id, speed in current_speeds.items():
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


def report_results(current_speeds, plate_readings):
    """Video bitince sonuc ozetini ve limit asan araclari konsola yazar.
    'RESULT ...' satirlari batch_evaluate.py tarafindan ayristirilir,
    formatini degistirme."""
    print("\n--- Video bitti ---")
    violations = []
    for track_id, speed in current_speeds.items():
        plate = vote_plate(plate_readings.get(track_id))
        print(f"ID {track_id}: {speed:.1f} km/h | plaka: {plate}")
        print(f"RESULT track={track_id} speed_kmh={speed:.2f} plate={plate}")
        if speed > SPEED_LIMIT_KMH:
            violations.append((track_id, speed, plate))

    if violations:
        print(f"\n--- Limit asimi ({SPEED_LIMIT_KMH} km/h) yapan araclar ---")
        for track_id, speed, plate in violations:
            print(f"ID {track_id}: {speed:.1f} km/h | plaka: {plate}")


# =============================================================================
# GIRIS NOKTASI
# =============================================================================

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    is_headless = "--headless" in sys.argv
    process_video(path, headless=is_headless)