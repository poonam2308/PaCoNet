import cv2
import os
import json
import numpy as np
from scipy.signal import find_peaks

def detect_vertical_axes(image_input, save_dir, output_json, apertureSize=5,
                         minLineLength=40, maxLineGap=1,
                         min_spacing=20, left_edge_thresh=0.03,
                         right_edge_thresh=0.95,method="hough"):
    os.makedirs(save_dir, exist_ok=True)
    results = []

    if os.path.isfile(image_input):
        image_files = [image_input]
    else:
        image_files = [
            os.path.join(image_input, f)
            for f in os.listdir(image_input)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    for path in image_files:
        fname = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        def detect_by_hough():
            edges = cv2.Canny(gray, 50, 150, apertureSize=apertureSize)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 4, 10,
                                    minLineLength=minLineLength, maxLineGap=maxLineGap)
            verticals = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(x1 - x2) < 6:
                        verticals.append(((x1, y1), (x2, y2)))
            x_coords = [int((a[0] + b[0]) // 2) for a, b in verticals]
            return x_coords, verticals

        def detect_by_projection():
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            vertical_sum = np.sum(binary, axis=0)
            smoothed = cv2.GaussianBlur(vertical_sum.astype(np.float32).reshape(1, -1), (25, 1), 0).flatten()
            peaks, _ = find_peaks(smoothed, distance=20, prominence=np.max(smoothed) * 0.3)
            return list(peaks)

        # Run detections
        hough_coords, _ = detect_by_hough() if method in ["hough", "combined"] else ([], [])
        proj_coords = detect_by_projection() if method in ["projection", "combined"] else []

        # Combine detections
        if method == "hough":
            final_coords = hough_coords
        elif method == "projection":
            final_coords = proj_coords
        elif method == "combined":
            combined = sorted(hough_coords + proj_coords)
            final_coords = []
            for x in combined:
                if len(final_coords) == 0 or abs(x - final_coords[-1]) > 6:
                    final_coords.append(x)

        # Filter bounding edges
        W = img.shape[1]
        final_coords = sorted(final_coords)
        filtered = [final_coords[0]] if final_coords else []
        for x in final_coords[1:]:
            if abs(x - filtered[-1]) > min_spacing:
                filtered.append(x)

        if filtered and filtered[0] < left_edge_thresh * W:
            filtered = filtered[1:]
        if filtered and filtered[-1] > right_edge_thresh * W:
            filtered = filtered[:-1]

        # Draw and save
        img_out = img.copy()
        for x in filtered:
            cv2.line(img_out, (x, 0), (x, img.shape[0]), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(save_dir, fname), img_out)

        results.append({
            "image_name": fname,
            "x_coordinates": [int(x) for x in filtered]
        })

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Vertical line data saved to {output_json}")
    return results
