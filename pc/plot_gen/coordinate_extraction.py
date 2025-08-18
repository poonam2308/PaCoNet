import numpy as np
import xml.etree.ElementTree as ET
from typing import List
from pc.plot_gen.svg_helper import (
    _unit_to_px,
    _detect_chart_top,
    _cumulative_transform,
    parse_path_data,
    _apply_M,
    _cluster_means,
    _dedupe_sorted
)

np.random.seed(42)

class CoordinateExtraction:

    def __init__(self, normalize_y_to_plot=False):
        self.normalize_y_to_plot = normalize_y_to_plot

    def extract_line_coordinates(self, svg_file_path: str, eps_axis: float = 0.25):

        tree = ET.parse(svg_file_path)
        root = tree.getroot()
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        ET.register_namespace('', ns['svg'])

        parent_map = {child: parent for parent in root.iter() for child in parent}
        to_px = _unit_to_px(root)

        chart_top = _detect_chart_top(root, parent_map, ns)

        # collect path elements that are line marks
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

        # transform segments into absolute pixels
        segs_px = []
        for d, M in path_items:
            _, segs = parse_path_data(d)
            for (sx, sy), (ex, ey) in segs:
                sx2, sy2 = _apply_M(M, sx, sy)
                ex2, ey2 = _apply_M(M, ex, ey)
                (sx2, sy2) = to_px(sx2, sy2)
                (ex2, ey2) = to_px(ex2, ey2)
                if self.normalize_y_to_plot:
                    sy2 -= chart_top
                    ey2 -= chart_top
                segs_px.append(((sx2, sy2), (ex2, ey2)))

        if not segs_px:
            return {"lines_by_region": {}}

        # infer vertical axes from x endpoints and cluster near-dupes
        all_x = []
        for (sx, _), (ex, _) in segs_px:
            all_x.append(sx);
            all_x.append(ex)
        unique_xs = _cluster_means(all_x, eps=eps_axis)
        if len(unique_xs) < 2:
            return {"lines_by_region": {}}

        # build output per region
        lines_by_region = {}
        for i in range(len(unique_xs) - 1):
            left_x = unique_xs[i]
            right_x = unique_xs[i + 1]

            region_lines = []
            for (sx, sy), (ex, ey) in segs_px:
                if left_x <= sx <= right_x and left_x <= ex <= right_x:
                    if abs(sx - ex) < 1e-9:  # skip perfectly vertical segments
                        continue
                    # normalize X to region; Y already handled above
                    region_lines.append([
                        sx - left_x,
                        sy,
                        ex - left_x,
                        ey
                    ])

            if region_lines:
                key = f"crop_{i + 1}"
                # round to 2 decimals like the original
                lines_by_region[key] = [[round(a, 2) for a in line] for line in region_lines]

        return {"lines_by_region": lines_by_region}

    def extract_vertical_axes(self, svg_path: str, eps: float = 0.75) -> List[float]:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        parent_map = {c: p for p in root.iter() for c in p}

        # Use the same unit conversion as for lines
        to_px = _unit_to_px(root)

        xs = []
        for el in root.iter():
            tag = el.tag.split('}')[-1]
            if tag != 'line':
                continue

            role = (el.attrib.get('aria-roledescription') or '').lower()
            if role != 'rule mark':
                continue

            x1 = el.attrib.get('x1')
            x2 = el.attrib.get('x2')
            y1 = el.attrib.get('y1', '0')
            y2 = el.attrib.get('y2')

            # Must be a vertical line (x1 == x2) and have positive length
            try:
                if x1 is None: x1 = x2
                if x2 is None: x2 = x1
                if x1 is None or x2 is None:
                    continue
                if float(y2) <= float(y1):
                    continue
                if abs(float(x1) - float(x2)) > 1e-9:
                    continue
            except Exception:
                continue

            # Full cumulative transform (translate/scale/matrix) like for paths
            M = _cumulative_transform(el, parent_map)

            # Apply transform to the local x1,y1 and then to pixels
            x_local = float(x1);
            y_local = float(y1)
            x_abs, y_abs = _apply_M(M, x_local, y_local)
            x_px, _ = to_px(x_abs, y_abs)

            xs.append(x_px)

        xs.sort()
        xs = _dedupe_sorted(xs, eps=eps)
        return xs

