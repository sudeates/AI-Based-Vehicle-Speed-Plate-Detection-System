import cv2
import numpy as np

image_points = np.array([
    [261, 583],
    [357, 578],
    [414, 814],
    [1458, 661]
], dtype=np.float32)

world_points = np.array([
    [0, 72.03],
    [7, 72.03],
    [0, 0],
    [7, 0]
], dtype=np.float32)

H = cv2.getPerspectiveTransform(image_points, world_points)
print(H)