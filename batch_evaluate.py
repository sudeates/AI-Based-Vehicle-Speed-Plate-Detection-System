"""
batch_evaluate.py
------------------
videos/ klasorundeki TUM videolari sirayla detect_speed.py ile (--headless
modda) calistirir, dosya adindan gercek hizi cikarir (orn. Peugeot3008_100.MP4
-> 100 km/h), olculen hiz ile karsilastirip hata yuzdesini hesaplar.

Sonunda en sorunludan en iyiye siralanmis bir ozet tablo basar. Bu, sistemin
genel dogruluk oranini tek tek video acmadan hizlica gormek icin kullanilir.

Kullanim:
    python batch_evaluate.py
"""

import re
import subprocess
import sys
from pathlib import Path

VIDEOS_DIR = Path("videos")
SCRIPT = "detect_speed.py"

# Dosya adinin sonundaki sayiyi gercek hiz olarak okuyor,
# ornek: Peugeot3008_100.MP4 -> 100
FILENAME_SPEED_PATTERN = re.compile(r"(\d+)(?=\.\w+$)")


def extract_actual_speed(filename: str):
    match = FILENAME_SPEED_PATTERN.search(filename)
    return float(match.group(1)) if match else None

def run_one(video_path: Path):
    proc = subprocess.Popen(
        [sys.executable, "-u", SCRIPT, str(video_path), "--headless"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True, bufsize=1
    )
    readings = []  # (speed_kmh, plate)
    for line in proc.stdout:
        line = line.rstrip()
        print(f"    {line}")  # canli akis - donup donmadigini gormek icin
        if line.startswith("RESULT"):
            m = re.search(r"speed_kmh=([\d.]+)", line)
            p = re.search(r"plate=(\S+)", line)
            if m:
                speed = float(m.group(1))
                plate = p.group(1) if p else "None"
                readings.append((speed, plate))
    proc.wait(timeout=600)
    if proc.returncode != 0:
        print(f"  [HATA] {video_path.name} calisirken cokme oldu (kod {proc.returncode})")
    return readings


def pick_measurement(readings):
    """Plakasi okunmus (dolayisiyla gercek arac oldugu dogrulanmis) track'leri
    tercih eder. Birden fazlaysa en yukseginii alir. Hic plaka yoksa (eski
    davranis) en yuksek hizi alir."""
    plated = [r for r in readings if r[1] not in ("None", "")]
    pool = plated if plated else readings
    return max(s for s, _ in pool)


def main():
    videos = sorted(VIDEOS_DIR.glob("*.MP4")) + sorted(VIDEOS_DIR.glob("*.mp4"))
    videos = sorted(set(videos))
    if not videos:
        print(f"'{VIDEOS_DIR}' klasorunde video bulunamadi.")
        return

    rows = []
    for video_path in videos:
        actual = extract_actual_speed(video_path.name)
        if actual is None:
            print(f"[ATLA] {video_path.name}: dosya adindan hiz cikarilamadi")
            continue

        print(f"[CALISIYOR] {video_path.name} (gercek hiz: {actual:.0f} km/h) ...")
        readings = run_one(video_path)

        if not readings:
            rows.append((video_path.name, actual, None, None))
            print("  -> hic hiz olcumu yapilamadi")
            continue

        # plakasi dogrulanmis track'i tercih ediyoruz - birden fazla
        # track ciktiginda (orn. gecici yanlis-pozitif) daha guvenilir secim
        measured = pick_measurement(readings)
        error_pct = (measured - actual) / actual * 100
        rows.append((video_path.name, actual, measured, error_pct))
        print(f"  -> olculen: {measured:.1f} km/h  (hata: {error_pct:+.1f}%)")

    print("\n" + "=" * 70)
    print("OZET (en sorunludan en iyiye)")
    print("=" * 70)

    valid_rows = [r for r in rows if r[2] is not None]
    failed_rows = [r for r in rows if r[2] is None]

    valid_rows.sort(key=lambda r: abs(r[3]), reverse=True)

    print(f"{'video':35s} {'gercek':>8s} {'olculen':>8s} {'hata%':>8s}")
    for name, actual, measured, error_pct in valid_rows:
        flag = " <-- INCELE" if abs(error_pct) > 15 else ""
        print(f"{name:35s} {actual:8.0f} {measured:8.1f} {error_pct:+7.1f}%{flag}")

    if failed_rows:
        print("\nHic olcum alinamayan videolar:")
        for name, actual, _, _ in failed_rows:
            print(f"  {name} (gercek: {actual:.0f} km/h)")


if __name__ == "__main__":
    main()