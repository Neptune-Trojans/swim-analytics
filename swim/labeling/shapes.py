"""Pixel masks (from SAM3-LiteText) to label shapes: the pool area as polygons with few corners, each lane rope as a
straight line with a width at each end, saved as the 4 corners of that band (docs/labeling.md).

Measured on 42 frames (2026-11-16): a corner wherever the outline bends more than 8 px keeps the pool area within IoU
0.99 of the mask with about 50 corners instead of about 6,000 outline points.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from swim.labeling.labels import Point, band_corners

WATER_TOLERANCE = 8.0  # pixels: the polygon stays within this distance of the mask's outline
MIN_POOL_PIECE = 0.005  # pieces of pool smaller than this share of the frame are dropped
CLOSE_GAPS = 0.02  # gaps in the pool area thinner than this share of the frame's short side are closed
EDGE_NOTCH = 0.2  # non-pool stretches along a frame edge shorter than this share of the edge are notches to fill
PERSON_RING = 15  # pixels around a person mask that decide whether the person is in the water
MIN_ROPE_PIXELS = 40
MIN_ROPE_ELONGATION = 5.0  # length / width: blobs are not ropes
SAME_ROPE_ANGLE = 3.0  # degrees: two rope pieces this parallel and on one line are one rope


@dataclass
class Rope:
    """A lane rope fitted to a mask: a straight line between two points, with its width (pixels) at each end."""

    points: list[Point]
    widths: list[float]

    def outline(self) -> list[Point]:
        """The rope as a label: the 4 corners of its band."""
        return band_corners(self.points, self.widths)


def pool_polygons(water: np.ndarray, persons: list[np.ndarray] = (), parts: list[np.ndarray] = (),
                  tolerance: float = WATER_TOLERANCE) -> list[list[Point]]:
    """The pool area as polygons: the water mask, plus `parts` of the pool (ropes, floor lane lines), plus persons in
    the water, with holes filled.

    SAM3's "swimming pool" mask leaves swimmers and the dark floor lines out; inside the pool they become holes (filled
    here), but where they reach the frame's edge or the pool's edge they cut notches into the outline: floor lines are
    added from their own masks, a person if most of the ring around them is pool.
    """
    area = water.astype(bool).copy()
    for part in parts:
        area |= part.astype(bool)
    for person in persons:
        if person_in_water(person.astype(bool), area):
            area |= person.astype(bool)
    # Close thin gaps (a rope SAM3's masks don't fully cover splits the pool along it, with a zigzag outline)
    size = 2 * round(CLOSE_GAPS * min(area.shape) / 2) + 1
    area = cv2.morphologyEx(area.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return area_polygons(close_edge_notches(area), tolerance)


def area_polygons(area: np.ndarray, tolerance: float = WATER_TOLERANCE, min_piece: float = MIN_POOL_PIECE) -> list[list[Point]]:
    """Outer outlines of an area mask as polygons with few corners (holes are filled: our polygons have none)."""
    area = area.astype(np.uint8)
    contours, _ = cv2.findContours(area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    height, width = area.shape
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_piece * area.size:
            continue
        corners = cv2.approxPolyDP(contour, tolerance, True)[:, 0].astype(float)
        corners = along_frame_edges(corners, width, height, tolerance)
        if len(corners) >= 3:
            polygons.append([(float(x), float(y)) for x, y in corners])
    return polygons


def add_water(polygons: list[list[Point]], added: list[Point], width: int, height: int,
              tolerance: float = WATER_TOLERANCE) -> list[list[Point]]:
    """Water polygons with an area merged in, outlines simplified again (the Labels page's + Water)."""
    area = np.zeros((height, width), np.uint8)
    for polygon in polygons + [added]:
        _fill(area, polygon, 1)
    # Keep pieces drawn by hand even if small (a few pixels of pool beside a sign)
    return area_polygons(area, tolerance, min_piece=0.0005)


def _fill(area: np.ndarray, polygon: list[Point], value: int) -> None:
    cv2.fillPoly(area, [np.round(np.array(polygon, float) * 16).astype(np.int32)], value, lineType=cv2.LINE_8, shift=4)


def close_edge_notches(area: np.ndarray, max_share: float = EDGE_NOTCH) -> np.ndarray:
    """Close the pool's outline along the frame's edges, so notches that reach the edge are filled like holes.

    Along each edge of the frame, a stretch of non-pool between two stretches of pool, shorter than `max_share` of the
    edge, is something narrow in the pool running out of the picture (a swimmer, a floor marking): it is marked as
    pool on that edge row / column. Stretches touching the frame's corners are left alone (the deck usually is one).
    """
    area = area.copy()
    for line in (area[0, :], area[-1, :], area[:, 0], area[:, -1]):  # views: changes go into `area`
        pool = np.flatnonzero(line)
        if len(pool) < 2:
            continue
        for a, b in zip(pool[:-1], pool[1:]):
            if 1 < b - a <= max_share * len(line):
                line[a + 1:b] = 1
    return area


def along_frame_edges(corners: np.ndarray, width: int, height: int, tolerance: float) -> np.ndarray:
    """Corners near the frame's edge moved onto it, and corners in between two others on the same edge dropped: where
    the pool runs out of the picture, its outline is the frame's edge, with a corner at each end."""
    corners = corners.copy()
    for axis, last in ((0, width - 1), (1, height - 1)):
        corners[corners[:, axis] <= tolerance, axis] = 0
        corners[corners[:, axis] >= last - tolerance, axis] = last

    def edges(p):
        return {(axis, p[axis]) for axis, last in ((0, width - 1), (1, height - 1)) if p[axis] in (0, last)}

    keep = []
    n = len(corners)
    for k in range(n):
        shared = edges(corners[k]) & edges(corners[k - 1]) & edges(corners[(k + 1) % n])
        if not shared:
            keep.append(corners[k])
    return np.array(keep) if len(keep) >= 3 else corners


def person_in_water(person: np.ndarray, water: np.ndarray, ring: int = PERSON_RING) -> bool:
    """Whether most of the ring around a person mask is water (a swimmer), not deck (a person beside the pool)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    around = cv2.dilate(person.astype(np.uint8), kernel).astype(bool) & ~person
    return bool(around.any()) and water[around].mean() >= 0.5


def rope_from_mask(mask: np.ndarray) -> Rope | None:
    """A rope mask as a straight line through it, with its width near each end; None if the mask is not line-shaped."""
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_ROPE_PIXELS:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    direction, origin = np.array([vx, vy]), np.array([x0, y0])
    across_dir = np.array([-vy, vx])
    along = (pts - origin) @ direction
    across = np.abs((pts - origin) @ across_dir)
    lo, hi = np.percentile(along, 0.5), np.percentile(along, 99.5)
    length = hi - lo
    widths = []
    for near_end in (along <= lo + 0.15 * length, along >= hi - 0.15 * length):
        widths.append(float(2 * np.percentile(across[near_end], 90) + 1) if near_end.any() else 2.0)
    if length < MIN_ROPE_ELONGATION * max(widths):
        return None
    p1, p2 = origin + lo * direction, origin + hi * direction
    return Rope([(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))], widths)


def ropes_from_masks(masks: list[np.ndarray]) -> list[Rope]:
    """One straight line per rope: a line per mask, then pieces of the same rope joined (a swimmer crossing a rope
    cuts it in two; the model sometimes returns one rope as two)."""
    ropes = [r for r in (rope_from_mask(m.astype(bool)) for m in masks) if r is not None]
    return join_pieces(ropes)


def join_pieces(ropes: list[Rope]) -> list[Rope]:
    ropes = list(ropes)
    joined = True
    while joined:
        joined = False
        for i in range(len(ropes)):
            for j in range(i + 1, len(ropes)):
                if same_rope(ropes[i], ropes[j]):
                    ropes[i] = join(ropes[i], ropes[j])
                    del ropes[j]
                    joined = True
                    break
            if joined:
                break
    return ropes


def _unit(rope: Rope) -> np.ndarray:
    d = np.subtract(rope.points[1], rope.points[0])
    return d / max(np.linalg.norm(d), 1e-9)


def _distance_to_line(point, rope: Rope) -> float:
    d = _unit(rope)
    v = np.subtract(point, rope.points[0])
    return abs(float(v[0] * d[1] - v[1] * d[0]))


def same_rope(a: Rope, b: Rope, max_angle: float = SAME_ROPE_ANGLE) -> bool:
    """Whether two rope lines are pieces of one rope: (almost) parallel and each lying on the other's line."""
    cos = abs(float(_unit(a) @ _unit(b)))
    if cos < math.cos(math.radians(max_angle)):
        return False
    tolerance = 0.75 * max(a.widths + b.widths) + 4
    return (all(_distance_to_line(p, a) <= tolerance for p in b.points)
            and all(_distance_to_line(p, b) <= tolerance for p in a.points))


def join(a: Rope, b: Rope) -> Rope:
    """One line from the two pieces' farthest ends, keeping the width at those ends."""
    ends = list(zip(a.points + b.points, a.widths + b.widths))
    d = _unit(a if np.linalg.norm(np.subtract(*a.points)) >= np.linalg.norm(np.subtract(*b.points)) else b)
    along = [float(np.dot(p, d)) for p, _ in ends]
    first, last = ends[int(np.argmin(along))], ends[int(np.argmax(along))]
    return Rope([first[0], last[0]], [first[1], last[1]])
