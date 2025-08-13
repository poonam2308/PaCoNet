# line_data.py
import os
import json
import glob
import numpy as np
from xml.etree import ElementTree as ET
from pc.plot_gen.plot_utils import safe_join
from pc.plot_gen.svg_helper import _unit_to_px, _detect_chart_top, _cumulative_transform, parse_path_data, _apply_M, \
    _cluster_means

np.random.seed(0)

class LineCoordinateExtractor:

    def __init__(self, main_dir, output_file="alldata.json", normalize_y_to_plot=False):
        self.main_dir = main_dir
        self.output_file = safe_join(self.main_dir, output_file)
        self.normalize_y_to_plot = normalize_y_to_plot  # default False for image alignment

    def extract_all(self, eps_axis=0.25):
        svg_files = sorted(glob.glob(os.path.join(self.main_dir, "*.svg")))
        all_data = []

        for svg_file_path in svg_files:
            tree = ET.parse(svg_file_path)
            root = tree.getroot()
            ns = {'svg': 'http://www.w3.org/2000/svg'}
            ET.register_namespace('', ns['svg'])

            parent_map = {child: parent for parent in root.iter() for child in parent}
            to_px = _unit_to_px(root)

            # Detect chart top (record only; do not apply unless normalize_y_to_plot=True)
            chart_top = _detect_chart_top(root, parent_map, ns)

            # Collect path elements that are line marks
            path_items = []
            for elem in root.findall('.//svg:path', ns):
                role = (elem.attrib.get('aria-roledescription') or '').lower()
                if role != 'line mark':
                    continue
                d = elem.attrib.get('d')
                if not d:
                    continue
                M = _cumulative_transform(elem, parent_map)
                path_items.append((d, M))

            # Transform segments into absolute pixels
            segs_px = []
            for d, M in path_items:
                _, segs = parse_path_data(d)
                for (sx, sy), (ex, ey) in segs:
                    sx2, sy2 = _apply_M(M, sx, sy)
                    ex2, ey2 = _apply_M(M, ex, ey)
                    (sx2, sy2) = to_px(sx2, sy2)
                    (ex2, ey2) = to_px(ex2, ey2)
                    # If user chooses plot-relative Y, subtract chart_top
                    if self.normalize_y_to_plot:
                        sy2 -= chart_top
                        ey2 -= chart_top
                    segs_px.append(((sx2, sy2), (ex2, ey2)))

            if not segs_px:
                continue

            # Infer vertical axes from x endpoints and cluster near-dupes
            all_x = []
            for (sx, _), (ex, _) in segs_px:
                all_x.append(sx); all_x.append(ex)
            unique_xs = _cluster_means(all_x, eps=eps_axis)
            if len(unique_xs) < 2:
                continue

            region_filename_base = os.path.basename(svg_file_path).replace('.svg', '')

            for i in range(len(unique_xs) - 1):
                left_x_orig  = unique_xs[i]
                right_x_orig = unique_xs[i + 1]
                region_width = right_x_orig - left_x_orig

                # Keep only segments fully within the region and not perfectly vertical
                region_lines = []
                for (sx, sy), (ex, ey) in segs_px:
                    if left_x_orig <= sx <= right_x_orig and left_x_orig <= ex <= right_x_orig:
                        if abs(sx - ex) < 1e-9:
                            continue
                        region_lines.append([sx, sy, ex, ey])

                if not region_lines:
                    continue

                # Normalize X to region; Y kept as-is (absolute image Y) unless normalize_y_to_plot=True
                normalized_lines = []
                frac_offsets = []  # store sub-pixel x fractional parts (lost by int casting)
                for sx, sy, ex, ey in region_lines:
                    sx_norm = sx - left_x_orig
                    ex_norm = ex - left_x_orig
                    if self.normalize_y_to_plot:
                        # already subtracted chart_top above; no further Y change
                        sy_norm, ey_norm = sy, ey
                    else:
                        # Y unchanged; aligns to PNG crops that include margins
                        sy_norm, ey_norm = sy, ey

                    normalized_lines.append([sx_norm, sy_norm, ex_norm, ey_norm])

                    # fractional parts of x's (helpful if you need to reconstruct exact placement)
                    frac_offsets.append([
                        sx_norm - int(sx_norm),
                        ex_norm - int(ex_norm),
                    ])

                all_data.append({
                    "filename": f"{region_filename_base}_crop_{i + 1}.png",
                    "lines": [[round(a, 2) for a in line] for line in normalized_lines],

                })

        with open(self.output_file, 'w') as f:
            json.dump(all_data, f, indent=4)

        return self.output_file
