import cv2
import os
import json
import numpy as np

def detect_vertical_axes(image_input, save_dir, output_json, apertureSize=5, minLineLength=40, maxLineGap=10):
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
        # img = cv2.resize(img, (600, 300))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        edges = cv2.Canny(gray, 50, 150, apertureSize=apertureSize)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength=minLineLength, maxLineGap=maxLineGap)

        verticals = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < 6:
                    verticals.append(((x1, y1), (x2, y2)))
        verticals.sort(key=lambda x: x[0][0])
        filtered = verticals[1:-1] if len(verticals) > 2 else verticals
        x_coords = [int(max(a[0], b[0])) for a, b in filtered]

        # Save image with lines
        for (x1, y1), (x2, y2) in filtered:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(save_dir, fname), img)

        results.append({
            "image_name": fname,
            "x_coordinates": x_coords
        })

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Vertical line data saved to {output_json}")
    return results

