"""
config.py
---------
Projedeki tum sabit ayarlar burada toplanir. Model yollari, video yolu,
sanal radar (virtual loop) sinirlari, hiz limiti gibi degerleri
degistirmek istediginde SADECE bu dosyaya bakman yeterli.
"""

# --- Video / Model yollari ---
DEFAULT_VIDEO_PATH = "videos/Peugeot3008_100.MP4"
VEHICLE_MODEL_PATH = "yolov8n.pt"
PLATE_MODEL_PATH = "runs/detect/train-3/weights/best.pt"
PLATE_OCR_MODEL = "cct-s-v2-global-model"

# --- Arac siniflari (YOLO/COCO class id -> isim) ---
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# --- Sanal radar (virtual loop) ayarlari ---
# Videolar 2 farkli kamera kurulumuna ait: Grup A araclar y~440'ta,
# Grup B araclar y~570'te kadraja giriyor. Tek bir giris/cikis ciftinin
# ikisine de uymasi imkansiz oldugu icin her grup icin ayri sinir
# tanimlayip, aracin ilk gorulen y'sine gore otomatik seciyoruz.
FAMILY_SPLIT_Y = 540

LOOP_BOUNDS = {
    "A": (510, 590),
    "B": (600, 640),
}

# Giris-cikis cizgileri arasindaki gercek mesafe, fiziksel olarak
# olculup elle girilmistir (grup basina).
MANUAL_DISTANCES_M = {
    "A": 4.04,
    "B": 7.00,
}

# --- Plaka okuma ayarlari ---
SHARPNESS_THRESHOLD = 20          # bu esikten bulanik plakalar elenir
TARGET_PLATE_CROP_WIDTH = 300     # OCR'a verilmeden once plaka bu genislige olceklenir
MIN_NATIVE_PLATE_WIDTH = 15       # bu pikselden dar plaka kirpimlari atlanir
MIN_PLATE_LENGTH = 6              # bu karakter sayisindan kisa OCR sonucu gecersiz sayilir

# --- Hiz ihlali ---
SPEED_LIMIT_KMH = 50
