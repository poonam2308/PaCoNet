import re
from typing import List


TRANSLATE_RE = re.compile(r'translate\(\s*([+-]?\d*\.?\d+)(?:[ ,]\s*([+-]?\d*\.?\d+))?\s*\)', re.I)
MATRIX_RE    = re.compile(r'matrix\(\s*([^\)]+)\)', re.I)

def _parse_transform_x(t: str) -> float:
    """Return the net X translation contributed by a transform string."""
    if not t:
        return 0.0
    tx = 0.0
    # Sum translations. Vega-Lite typically only uses translate for positioned marks.
    for part in re.finditer(r'(translate\([^\)]*\)|matrix\([^\)]*\))', t):
        s = part.group(1)
        m = TRANSLATE_RE.match(s)
        if m:
            tx += float(m.group(1))
            continue
        m = MATRIX_RE.match(s)
        if m:
            nums = [float(x) for x in m.group(1).replace(',', ' ').split()]
            if len(nums) == 6:
                tx += nums[4]  # e in matrix(a b c d e f)
    return tx

def _accumulate_tx(el, parent_map) -> float:
    tx = 0.0
    cur = el
    # walk up the tree, accumulating x-translation
    while cur is not None:
        tx += _parse_transform_x(cur.attrib.get('transform', ''))
        cur = parent_map.get(cur)
    return tx

def _dedupe_sorted(xs: List[float], eps: float = 0.75) -> List[float]:
    """Given a sorted list, remove near-duplicates within eps pixels."""
    out = []
    for x in xs:
        if not out or abs(x - out[-1]) > eps:
            out.append(x)
    return out

# --------------------------
# Transform parsing utilities
# --------------------------
_MTX_RE = {
    "translate": re.compile(r'translate\(\s*([+-]?\d*\.?\d+)(?:[ ,]\s*([+-]?\d*\.?\d+))?\s*\)', re.I),
    "scale":     re.compile(r'scale\(\s*([+-]?\d*\.?\d+)(?:[ ,]\s*([+-]?\d*\.?\d+))?\s*\)', re.I),
    "matrix":    re.compile(r'matrix\(\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\)', re.I),
}

def _mat_mul(a, b):
    return [
        [a[0][0]*b[0][0]+a[0][1]*b[1][0], a[0][0]*b[0][1]+a[0][1]*b[1][1], a[0][0]*b[0][2]+a[0][1]*b[1][2]+a[0][2]],
        [a[1][0]*b[0][0]+a[1][1]*b[1][0], a[1][0]*b[0][1]+a[1][1]*b[1][1], a[1][0]*b[0][2]+a[1][1]*b[1][2]+a[1][2]],
        [0, 0, 1],
    ]


def _affine_for(transform_str):
    t = transform_str or ""
    M = [[1,0,0],[0,1,0],[0,0,1]]
    for chunk in re.finditer(r'(translate|scale|matrix)\([^)]*\)', t):
        s = chunk.group(0)
        if m := _MTX_RE["matrix"].match(s):
            a,b,c,d,e,f = map(float, m.groups())
            Mi = [[a, c, e], [b, d, f], [0,0,1]]
        elif m := _MTX_RE["translate"].match(s):
            tx = float(m.group(1) or 0.0); ty = float(m.group(2) or 0.0)
            Mi = [[1,0,tx],[0,1,ty],[0,0,1]]
        elif m := _MTX_RE["scale"].match(s):
            sx = float(m.group(1) or 1.0); sy = float(m.group(2) or sx)
            Mi = [[sx,0,0],[0,sy,0],[0,0,1]]
        else:
            continue
        M = _mat_mul(M, Mi)
    return M

def _cumulative_transform(node, parent_map):
    M = [[1,0,0],[0,1,0],[0,0,1]]
    cur = node
    while cur is not None:
        Mi = _affine_for(cur.attrib.get('transform', ''))
        M = _mat_mul(Mi, M)  # parent first
        cur = parent_map.get(cur)
    return M

def _apply_M(M, x, y):
    return (M[0][0]*x + M[0][1]*y + M[0][2],
            M[1][0]*x + M[1][1]*y + M[1][2])


def _cluster_means(values, eps=0.25):
    """Merge near-duplicate coordinates; returns cluster means."""
    if not values:
        return []
    vals = sorted(values)
    clusters, cur = [], [vals[0]]
    for v in vals[1:]:
        if abs(v - cur[-1]) <= eps:
            cur.append(v)
        else:
            clusters.append(sum(cur)/len(cur))
            cur = [v]
    clusters.append(sum(cur)/len(cur))
    return clusters

def _unit_to_px(root):
    """Convert viewBox units to pixels; if missing, passthrough."""
    vb = root.attrib.get('viewBox')
    W  = root.attrib.get('width')
    H  = root.attrib.get('height')
    if not (vb and W and H):
        return lambda x,y: (x,y)
    minx, miny, vbw, vbh = map(float, vb.strip().split())
    def _num(s): return float(re.sub(r'px$', '', str(s)))
    Wpx, Hpx = _num(W), _num(H)
    sx, sy = (Wpx / vbw), (Hpx / vbh)
    return lambda x,y: ((x - minx)*sx, (y - miny)*sy)

def parse_path_data(d):
    d = re.sub(r'[,\s]+', ' ', d.strip())
    tokens = re.findall(r'[MmLlHhVvZz]|-?\d*\.?\d+(?:e[+-]?\d+)?', d)

    points, lines = [], []
    cx = cy = 0.0

    def add_point(nx, ny, draw_line):
        nonlocal cx, cy
        p_prev = (cx, cy)
        p_new  = (nx, ny)
        if draw_line:
            lines.append((p_prev, p_new))
        points.append(p_new)
        cx, cy = nx, ny

    i = 0
    while i < len(tokens):
        t = tokens[i]; i += 1
        if t in 'Mm':
            rel = (t == 'm'); first = True
            while i + 1 < len(tokens) and re.match(r'[-\d]', tokens[i]) and re.match(r'[-\d]', tokens[i+1]):
                x = float(tokens[i]); y = float(tokens[i+1]); i += 2
                nx = cx + x if rel else x
                ny = cy + y if rel else y
                add_point(nx, ny, draw_line=(not first))
                first = False
        elif t in 'Ll':
            rel = (t == 'l')
            while i + 1 < len(tokens) and re.match(r'[-\d]', tokens[i]) and re.match(r'[-\d]', tokens[i+1]):
                x = float(tokens[i]); y = float(tokens[i+1]); i += 2
                add_point(cx + x if rel else x, cy + y if rel else y, draw_line=True)
        elif t in 'Hh':
            rel = (t == 'h')
            while i < len(tokens) and re.match(r'[-\d]', tokens[i]):
                x = float(tokens[i]); i += 1
                add_point(cx + x if rel else x, cy, draw_line=True)
        elif t in 'Vv':
            rel = (t == 'v')
            while i < len(tokens) and re.match(r'[-\d]', tokens[i]):
                y = float(tokens[i]); i += 1
                add_point(cx, cy + y if rel else y, draw_line=True)
        elif t in 'Zz':
            pass
    return points, lines

# --------------------------
# Helper: detect chart top (optional metadata)
# --------------------------
def _detect_chart_top(root, parent_map, ns):
    """
    Try to detect the Y translation of the main marks group (chart top).
    Returns float (pixels). If not found, returns 0.0.
    """
    top = 0.0
    for g in root.findall('.//svg:g', ns):
        role = g.attrib.get('aria-roledescription', '').lower()
        if 'group mark' in role or 'layer mark' in role:
            M = _cumulative_transform(g, parent_map)
            # translation component (e,f) is M[0][2], M[1][2]
            top = M[1][2]
            break
    return float(top)

# Add to svg_helper.py (or a new helper)
import xml.etree.ElementTree as ET

def _plot_frames(root, parent_map):
    """Return a list of plot frames [(x0,y0,w,h)] in *pixels*.
       Frames come from clipPath <rect> used by the marks group(s)."""
    ns = {"svg": root.tag.split('}')[0].strip('{')}
    to_px = _unit_to_px(root)

    # Map clipPath id -> rect element
    clip_rects = {}
    for cp in root.findall('.//svg:clipPath', ns):
        rect = None
        for child in cp:
            if child.tag.endswith('rect'):
                rect = child
                break
        if rect is not None:
            cid = cp.attrib.get('id')
            if cid:
                clip_rects[cid] = rect

    frames = []
    # Find groups that reference a clip-path
    for g in root.findall('.//svg:g', ns):
        cp_url = g.attrib.get('clip-path')
        if not cp_url or not cp_url.startswith('url(#'):
            continue
        cid = cp_url[5:-1]
        rect = clip_rects.get(cid)
        if rect is None:
            continue

        # Rect in document units
        x = float(rect.attrib.get('x', '0') or 0)
        y = float(rect.attrib.get('y', '0') or 0)
        w = float(rect.attrib.get('width', '0') or 0)
        h = float(rect.attrib.get('height', '0') or 0)

        # Apply any transforms on the rect itself
        M_rect = _cumulative_transform(rect, parent_map)
        (x0, y0) = _apply_M(M_rect, x, y)
        (x1, y1) = _apply_M(M_rect, x + w, y + h)

        # Convert to pixels
        (x0, y0) = to_px(x0, y0)
        (x1, y1) = to_px(x1, y1)

        # Normalize corners
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])
        frames.append((x_min, y_min, x_max - x_min, y_max - y_min))

    # Deduplicate nearly-identical frames (facets will yield many)
    frames.sort()
    deduped = []
    for f in frames:
        if not deduped or abs(f[0] - deduped[-1][0]) > 0.5 or abs(f[1] - deduped[-1][1]) > 0.5:
            deduped.append(f)
    return deduped
