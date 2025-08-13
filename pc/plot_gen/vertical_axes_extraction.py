# vertical_axes_extraction.py
import re
import xml.etree.ElementTree as ET
from typing import List

from pc.plot_gen.svg_helper import _accumulate_tx, _dedupe_sorted

def extract_vertical_axes_coords(svg_path: str, *, eps: float = 0.75) -> List[float]:
    """
    Extract the x positions (in SVG pixels) of the vertical axes (Altair/Vega-Lite 'rule' marks)
    and return a deduplicated, ascending list. Near-duplicates within `eps` px are merged.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    parent = {c: p for p in root.iter() for c in p}

    xs = []
    for el in root.iter():
        tag = el.tag.split('}')[-1]  # strip namespace
        if tag != 'line':
            continue

        # keep only rule marks that are drawn as vertical lines from x=0 to x=0 (before transforms)
        # Vega-Lite draws each rule as a line from (0, 0) -> (0, y2) inside a translated group.
        if el.attrib.get('x2', None) not in ('0', '0.0', 0, 0.0, None):
            # Some exporters omit x2 when it equals 0; in that case require x1 == x2.
            x1 = el.attrib.get('x1', None)
            if x1 is None or el.attrib.get('x2') is None or str(x1) != str(el.attrib.get('x2')):
                continue

        # We need a positive length in Y to be a vertical rule.
        y2 = el.attrib.get('y2')
        y1 = el.attrib.get('y1', '0')
        try:
            if float(y2) <= float(y1):
                continue
        except Exception:
            continue

        role = el.attrib.get('aria-roledescription', '')
        if role and role.lower() != 'rule mark':
            # ignore tick marks or other line-like marks
            continue

        tx = _accumulate_tx(el, parent)
        xs.append(tx)

    xs.sort()
    xs = _dedupe_sorted(xs, eps=eps)
    return xs
