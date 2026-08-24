calibration_ys = []
calibration_mpps = []
sources = []  # hangi dosyadan geldigini de izleyelim

with open("calibration/calibration55.txt", "r") as f:
    for line in f:
        y_val, mpp_val = line.split()
        calibration_ys.append(float(y_val))
        calibration_mpps.append(float(mpp_val))
        sources.append("55")

with open("calibration/calibration100.txt", "r") as f:
    for line in f:
        y_val, mpp_val = line.split()
        calibration_ys.append(float(y_val))
        calibration_mpps.append(float(mpp_val))
        sources.append("100")

combined = sorted(zip(calibration_ys, calibration_mpps, sources))

print(f"{'y':>8s} {'mpp':>10s} {'kaynak':>8s} {'onceki_egim':>14s}")
prev_y, prev_mpp = None, None
for y, mpp, src in combined:
    slope = ""
    if prev_y is not None and y != prev_y:
        slope = f"{(mpp - prev_mpp) / (y - prev_y):.6f}"
    print(f"{y:8.1f} {mpp:10.6f} {src:>8s} {slope:>14s}")
    prev_y, prev_mpp = y, mpp