"""
plate_reader.py
----------------
Bir arac kirpiminin icinden plaka bolgesini bulma, bulanikligini kontrol
etme, OCR ile okuma ve birden fazla kare boyunca toplanan okumalari
"cogunluk oylamasi" (majority voting) ile tek bir nihai plakaya
birlestirme mantigi burada.
"""

import cv2

from config import (
    SHARPNESS_THRESHOLD,
    TARGET_PLATE_CROP_WIDTH,
    MIN_NATIVE_PLATE_WIDTH,
    MIN_PLATE_LENGTH,
)


def find_best_box(boxes, allowed_class_ids=None):
    """Bir YOLO sonucundaki kutular arasindan en yuksek guvenli (confidence)
    olani secer. allowed_class_ids verilirse sadece o siniflar arasindan bakar."""
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
    """Laplacian varyansi ile bulaniklik skoru. Dusuk deger = bulanik goruntu."""
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()


def vote_plate(readings):
    """Ayni arac icin birden fazla karede okunan plaka metinlerini,
    karakter bazinda cogunluk oylamasi yaparak tek bir sonuca birlestirir.
    Once en sik gorulen uzunluk secilir, sonra her pozisyondaki en sik
    karakter alinir. Bu, tek bir kareki OCR hatasinin sonucu bozmasini engeller."""
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
    """Bir arac kirpimi (vehicle_crop) verildiginde plakayi bulup okuyan sinif."""

    def __init__(self, plate_model, ocr_reader):
        self.plate_model = plate_model
        self.ocr_reader = ocr_reader

    def read(self, vehicle_crop, offset_x, offset_y):
        """Donus: (plaka_metni veya None, cizim_icin_kutu veya None).
        offset_x/offset_y, plaka kutusunu orijinal kare koordinatina
        cevirmek icin kullanilir (cizim amacli)."""
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
