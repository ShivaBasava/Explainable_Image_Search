"""
ICC++ : Image Composition Canvas (improved)
Faithful re-implementation following Madhu et al., "ICC++: Explainable Image
Retrieval for Art Historical Corpora using Image Composition Canvas".

Pipeline
--------
1. Pose estimation (YOLOv8-Pose).
2. Poseline generation with POSE FALLBACK (extend neck->nose x3 when lower
   body is missing).
3. Body-orientation estimation via bisection of (nose, neck, mid-hip) plus
   correction angle rho.
4. Direction CONES (opening omega, length scale sigma, base scale eta) per
   character; pairwise intersections give global ACTION CENTERS.
5. Normalization of poselines (image / bbox / ar-norm).
6. Retrieval: single-poseline distance -> bipartite min-matching ->
   r_hr, r_nmd, r_cr similarity metrics.

COCO-17 keypoint indices (YOLOv8-Pose):
    0 nose, 5 L-shoulder, 6 R-shoulder, 11 L-hip, 12 R-hip
(Neck is synthesized as the shoulder midpoint; COCO has no neck keypoint.)
"""

import math

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from shapely.geometry import LineString, Point, Polygon

Image.MAX_IMAGE_PIXELS = None


class ICCVisualizer:
    """Pose-driven Image Composition Canvas (ICC++) explainability view.

    Estimates character poses in an artwork, derives poselines and
    direction cones per character, and intersects the cones to recover
    global "action centers" — the compositional focal points the figures'
    body orientations converge on. Also provides ICC++ normalization and
    similarity metrics for pose-based retrieval comparisons.
    """

    # Hyperparameters (baseline values from paper, Sec. 4.2.2)
    RHO_DEG = 20.0          # correction angle  (rho)
    OMEGA_DEG = 80.0        # cone opening angle (omega)
    SIGMA = 10.0            # cone length scale factor (sigma)
    ETA = 0.0               # cone base scale factor   (eta)  -> 0 = apex at origin
    BETA = 150.0            # outlier filter threshold for retrieval (px)
    FALLBACK_MULT = 3.0     # neck->nose extension multiplier for pose fallback

    POSE_MODEL_WEIGHTS = "yolov8n-pose.pt"

    def __init__(self):
        # Lazy YOLO singleton — loaded on first call to extract_icc()
        self._pose_model = None

    def _get_pose_model(self):
        """Lazily load and cache the YOLOv8-Pose model."""
        if self._pose_model is None:
            from ultralytics import YOLO
            self._pose_model = YOLO(self.POSE_MODEL_WEIGHTS)
        return self._pose_model

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _valid(pt):
        return pt is not None and np.all(np.asarray(pt) > 0)

    @staticmethod
    def _midpoint(a, b):
        return (np.asarray(a) + np.asarray(b)) / 2.0

    @staticmethod
    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else v

    @staticmethod
    def _rotate(v, ang_rad):
        c, s = math.cos(ang_rad), math.sin(ang_rad)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

    # ------------------------------------------------------------------
    # 1+2. Poselines WITH fallback
    # ------------------------------------------------------------------
    def _build_poseline(self, kp):
        """
        kp : (17, 2) keypoint array for one person.
        Returns (poseline LineString, body_origin, body_direction unit-vector)
        or None if no usable poseline can be built.
        """
        nose = kp[0]
        neck = self._midpoint(kp[5], kp[6]) if (self._valid(kp[5]) and self._valid(kp[6])) else None

        if self._valid(kp[11]) and self._valid(kp[12]):
            mid_hip = self._midpoint(kp[11], kp[12])
        elif self._valid(kp[11]):
            mid_hip = kp[11]
        elif self._valid(kp[12]):
            mid_hip = kp[12]
        else:
            mid_hip = None

        if not self._valid(nose):
            return None

        if mid_hip is not None:
            line = LineString([nose, mid_hip])
            origin = mid_hip
        elif neck is not None:
            # POSE FALLBACK: extend neck->nose downward x FALLBACK_MULT
            d = np.asarray(nose) - np.asarray(neck)
            fallback_bottom = np.asarray(nose) - d * self.FALLBACK_MULT
            line = LineString([nose, fallback_bottom])
            origin = fallback_bottom
            mid_hip = fallback_bottom
        else:
            return None

        body_dir = self._body_direction(nose, neck, mid_hip)
        return line, np.asarray(origin), body_dir

    def _body_direction(self, nose, neck, mid_hip):
        """Bisection of vectors (origin->nose) and (origin->mid_hip) at neck,
        corrected by rho.  Falls back to poseline axis if neck is missing."""
        if neck is not None:
            v1 = self._unit(np.asarray(nose) - np.asarray(neck))
            v2 = self._unit(np.asarray(mid_hip) - np.asarray(neck))
            bisec = self._unit(v1 + v2) if np.linalg.norm(v1 + v2) > 1e-6 else v1
            bisec = self._unit(self._rotate(bisec, -math.radians(self.RHO_DEG)))
        else:
            bisec = self._unit(np.asarray(nose) - np.asarray(mid_hip))
        return bisec

    # ------------------------------------------------------------------
    # 4. Cone construction
    # ------------------------------------------------------------------
    def _build_cone(self, origin, direction, img_diag):
        """Triangle (cone) polygon from `origin` along `direction`.
        length = sigma * (img_diag / 100); half-opening = omega / 2."""
        length = self.SIGMA * (img_diag / 100.0)
        half = math.radians(self.OMEGA_DEG) / 2.0
        apex = np.asarray(origin) - direction * (self.ETA * length)
        left = apex + self._rotate(direction, half) * length
        right = apex + self._rotate(direction, -half) * length
        return Polygon([apex, left, right])

    @staticmethod
    def _action_centers_from_cones(cones, w, h):
        """Pairwise cone intersections -> centroids = global action centers."""
        centers = []
        for i in range(len(cones)):
            for j in range(i + 1, len(cones)):
                inter = cones[i].intersection(cones[j])
                if not inter.is_empty and inter.area > 0:
                    c = inter.centroid
                    if 0 <= c.x < w and 0 <= c.y < h:
                        centers.append(Point(c.x, c.y))
        return centers

    # ------------------------------------------------------------------
    # Main extraction
    # ------------------------------------------------------------------
    def extract_icc(self, image):
        """Run pose estimation and compute all ICC++ elements for a BGR cv2 image."""
        h, w = image.shape[:2]
        img_diag = math.hypot(w, h)

        results = self._get_pose_model()(image, verbose=False)
        kpts = results[0].keypoints.xy.cpu().numpy()  # (N, 17, 2)

        poselines, cones, origins, dirs = [], [], [], []
        for person in kpts:
            out = self._build_poseline(person)
            if out is None:
                continue
            line, origin, body_dir = out
            poselines.append(line)
            origins.append(origin)
            dirs.append(body_dir)
            cones.append(self._build_cone(origin, body_dir, img_diag))

        action_centers = self._action_centers_from_cones(cones, w, h) if len(cones) > 1 else []

        return {
            "poselines": poselines,
            "cones": cones,
            "action_centers": action_centers,
            "origins": origins,
            "directions": dirs,
            "size": (w, h),
        }

    # ------------------------------------------------------------------
    # 5. Normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _poseline_endpoints(line):
        (x1, y1), (x2, y2) = line.coords[0], line.coords[1]
        return np.array([x1, y1]), np.array([x2, y2])

    def normalize(self, icc, method="ar"):
        w, h = icc["size"]
        pls = icc["poselines"]
        if not pls:
            return []

        if method == "image":
            return [[t / [w, h], b / [w, h]]
                    for line in pls for t, b in [self._poseline_endpoints(line)]]

        if method == "bbox":
            pts = np.array([p for line in pls for p in self._poseline_endpoints(line)])
            mn, mx = pts.min(0), pts.max(0)
            rng = np.where((mx - mn) > 1e-6, mx - mn, 1.0)
            return [[(t - mn) / rng, (b - mn) / rng]
                    for line in pls for t, b in [self._poseline_endpoints(line)]]

        if method == "ar":
            centers = icc["action_centers"]
            if not centers:
                return self.normalize(icc, "image")
            out = []
            for c in centers:
                cx, cy = c.x, c.y
                out.append([[t - [cx, cy], b - [cx, cy]]
                            for line in pls for t, b in [self._poseline_endpoints(line)]])
            return out

        raise ValueError(method)

    # ------------------------------------------------------------------
    # 6. Retrieval metrics
    # ------------------------------------------------------------------
    @staticmethod
    def _single_poseline_distance(pq, pt):
        top = np.linalg.norm(np.asarray(pq[0]) - np.asarray(pt[0]))
        bot = np.linalg.norm(np.asarray(pq[1]) - np.asarray(pt[1]))
        return (top + bot) / 2.0

    def _bipartite_match(self, Pq, Pt):
        edges = sorted(
            [(self._single_poseline_distance(pq, pt), i, j)
             for i, pq in enumerate(Pq) for j, pt in enumerate(Pt)],
            key=lambda e: e[0],
        )
        used_q, used_t, R = set(), set(), []
        for d, i, j in edges:
            if i in used_q or j in used_t:
                continue
            used_q.add(i)
            used_t.add(j)
            R.append(d)
        return R

    def similarity(self, Pq, Pt, beta=None):
        beta = self.BETA if beta is None else beta
        if not Pq or not Pt:
            return 0.0, 0.0, 0.0
        R = self._bipartite_match(Pq, Pt)
        Rf = [d for d in R if d < beta]
        if not Rf:
            return 0.0, 0.0, 0.0
        r_hr = len(Rf) / max(len(Pq), len(Pt))
        r_md = sum(Rf) / len(Rf)
        r_nmd = (beta - r_md) / beta
        return r_hr, r_nmd, r_hr * r_nmd

    def image_similarity(self, icc_q, icc_t, method="ar"):
        nq, nt = self.normalize(icc_q, method), self.normalize(icc_t, method)
        if method != "ar":
            return self.similarity(nq, nt)
        if not nq or not nt:
            return 0.0, 0.0, 0.0
        best = (0.0, 0.0, 0.0)
        for sq in nq:
            for st_ in nt:
                s = self.similarity(sq, st_)
                if s[2] > best[2]:
                    best = s
        return best

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    @staticmethod
    def draw_icc_overlay(image, icc, draw_cones=True):
        ov = image.copy()
        for line in icc["poselines"]:
            (x1, y1), (x2, y2) = line.coords[0], line.coords[1]
            cv2.line(ov, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

        if draw_cones:
            for cone in icc["cones"]:
                pts = np.array(cone.exterior.coords, np.int32)
                cv2.polylines(ov, [pts], True, (255, 0, 255), 2)

        for c in icc["action_centers"]:
            cv2.circle(ov, (int(c.x), int(c.y)), 8, (255, 255, 0), -1)

        return ov

    # ------------------------------------------------------------------
    # Public entry point for proto_type_main.py
    # ------------------------------------------------------------------
    def get_icc_overlay(self, image: Image.Image) -> tuple:
        """
        Run ICC++ on an already-loaded artwork image and return
        (PIL overlay image, stats dict).

        stats keys: poselines (int), action_centers (int)
        """
        cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        icc = self.extract_icc(cv_img)
        overlay = self.draw_icc_overlay(cv_img, icc)
        pil_out = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        stats = {
            "poselines": len(icc["poselines"]),
            "action_centers": len(icc["action_centers"]),
        }
        return pil_out, stats

    # ------------------------------------------------------------------
    # Streamlit dialog
    # ------------------------------------------------------------------
    def show_dialog(self, title: str, image: Image.Image):
        """Compute the ICC++ overlay and open a Streamlit dialog to display it."""

        @st.dialog(f"Image Composition Canvas — {title}", width="large")
        def _render_dialog():
            with st.spinner("Computing pose composition canvas…"):
                try:
                    overlay_img, stats = self.get_icc_overlay(image)
                except Exception as e:
                    st.error(f"Could not compute ICC++ overlay: {e}")
                    return

            st.image(overlay_img, width="stretch")
            st.caption(
                "Green = poselines (nose -> mid-hip). Magenta = direction cones. "
                "Yellow = action centers (cone intersections)."
            )
            st.caption(
                f"**Detected poselines:** {stats['poselines']}  |  "
                f"**Action centers:** {stats['action_centers']}"
            )
            if stats["poselines"] == 0:
                st.info("No human figures with a usable pose were detected in this artwork.")

        _render_dialog()
