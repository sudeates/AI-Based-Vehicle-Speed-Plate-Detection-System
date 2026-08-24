import re
from pathlib import Path
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
N = 5  # ana script ile ayni pencere boyutu

SPLIT_FILE = Path("videos/Train_valid_split.txt")
VIDEOS_DIR = Path("videos")
OUTPUT_FILE = Path("calibration/calibration_combined.txt")
BIN_SIZE = 10  # piksel - bu genislikte y-bantlarina gore medyan alinir


def load_split():
    entries = []
    with open(SPLIT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, split = line.split()
            entries.append((name, split))
    return entries


def extract_speed(name):
    m = re.search(r"_(\d+)$", name)
    return float(m.group(1)) if m else None


def find_video_path(name):
    for ext in (".MP4", ".mp4"):
        p = VIDEOS_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None


def track_vehicle(video_path):
    """Videoyu bastan sona izler, her frame'de en guvenilir arac kutusunu
    secer (ana script ile ayni best_box mantigi) ve track_id -> [(frame_idx, cx, cy), ...]
    dondurur."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    positions = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, persist=True, verbose=False, imgsz=1280)[0]
        best_box, best_conf = None, 0
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASS_IDS:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = box
        if best_box is not None and best_box.id is not None:
            track_id = int(best_box.id)
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            positions.setdefault(track_id, []).append((frame_idx, cx, cy))
        frame_idx += 1
    cap.release()
    return positions, fps


def build_points_from_video(video_path, actual_speed_kmh):
    positions, fps = track_vehicle(video_path)
    if not positions:
        return [], None
    # baskin track: en cok frame'de gorunen (tek arac varsayimi)
    track_id = max(positions, key=lambda k: len(positions[k]))
    traj = positions[track_id]
    actual_speed_ms = actual_speed_kmh / 3.6

    y_min = min(cy for _, _, cy in traj)

    points = []
    for i in range(len(traj) - N):
        f0, cx0, cy0 = traj[i]
        f1, cx1, cy1 = traj[i + N]
        if f1 - f0 != N:
            continue  # tracker bu araliktta frame atlamis, guvenilmez
        pixel_dist = ((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5
        if pixel_dist < 1:
            continue  # bolme hatasi / anlamsiz nokta
        elapsed_s = N / fps
        real_dist_m = actual_speed_ms * elapsed_s
        mpp = real_dist_m / pixel_dist
        mid_cy = (cy0 + cy1) / 2
        points.append((mid_cy, mpp))
    return points, y_min


# Videolar 2 ayri kamera kurulumuna ait gibi gorunuyor (bkz. onceki analiz):
# Grup A araclar y~436-450'de, Grup B araclar y~570-580'de kadraja giriyor,
# ikisi arasinda buyuk bir bosluk (450-570) var, hicbir video bu araliktan
# baslamiyor. Bu esigi kullanarak videoyu otomatik siniflandiriyoruz.
FAMILY_SPLIT_Y = 500


def classify_family(y_min):
    return "A" if y_min < FAMILY_SPLIT_Y else "B"


def main():
    entries = load_split()
    points_by_family = {"A": [], "B": []}

    for name, split in entries:
        if split != "train":
            continue
        actual_speed = extract_speed(name)
        video_path = find_video_path(name)
        if video_path is None or actual_speed is None:
            print(f"[ATLA] {name}: video/hiz bulunamadi")
            continue

        print(f"[ISLENIYOR] {name} ({actual_speed:.0f} km/h) ...")
        pts, y_min = build_points_from_video(video_path, actual_speed)
        if y_min is None:
            print("  -> arac bulunamadi, atlaniyor")
            continue
        family = classify_family(y_min)
        print(f"  -> {len(pts)} nokta uretildi, grup={family} (y_min={y_min})")
        points_by_family[family].extend(pts)

    for family, all_points in points_by_family.items():
        if not all_points:
            print(f"\n[UYARI] Grup {family} icin hic nokta yok, dosya yazilmadi.")
            continue

        bins = {}
        for y, mpp in all_points:
            b = int(y // BIN_SIZE) * BIN_SIZE
            bins.setdefault(b, []).append(mpp)

        output_path = Path(f"calibration/calibration_group{family}.txt")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w") as f:
            for b in sorted(bins):
                vals = sorted(bins[b])
                median = vals[len(vals) // 2]
                f.write(f"{b + BIN_SIZE / 2} {median}\n")

        print(f"\nGrup {family}: {len(all_points)} ham nokta, {len(bins)} y-bandina indirgendi.")
        print(f"Yazildi: {output_path}")


if __name__ == "__main__":
    main()