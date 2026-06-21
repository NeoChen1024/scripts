#!/usr/bin/env python
"""
pic_mono_dither.py — Interactive monochrome (1-bit) image preprocessing tool.

Loads a colour image, downscales it to a target resolution, converts to OKLAB
colour space for perceptually-uniform lightness, applies an adjustable brightness
curve, and dithers to 1-bit with a choice of algorithms.

Dependencies: PyQt6, numpy, scipy, Pillow
"""

from __future__ import annotations

import os
import sys
import threading
import time

import click
import numpy as np
from numba import jit
from PIL import Image
from PyQt6.QtCore import (
    QEvent,
    QLineF,
    QMimeData,
    QObject,
    QPointF,
    QRect,
    QRectF,
    QSettings,
    Qt,
    QThread,
    QTimer,
    QUrl,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QClipboard,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QFontMetrics,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QShortcut,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from scipy.interpolate import PchipInterpolator

# ══════════════════════════════════════════════════════════════════════════════
#  OKLAB  —  Perceptual colour space (Björn Ottosson)
# ══════════════════════════════════════════════════════════════════════════════


class Oklab:
    """sRGB → OKLAB L* (lightness) conversion."""

    _M1 = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ],
        dtype=np.float64,
    )

    @staticmethod
    def _linearize(rgb: np.ndarray) -> np.ndarray:
        mask = rgb <= 0.04045
        return np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    @staticmethod
    def lightness(rgb: np.ndarray) -> np.ndarray:
        """sRGB → OKLAB L* (lightness, in [0, 1]).

        Args:
            rgb: (H, W, 3) float64 array in [0, 1].
        Returns:
            (H, W) float64 L* clamped to [0, 1].
        """
        lin = Oklab._linearize(rgb)
        lms = lin @ Oklab._M1.T
        lms_ = np.cbrt(np.maximum(lms, 0.0))
        L = 0.2104542553 * lms_[..., 0] + 0.7936177850 * lms_[..., 1] - 0.0040720468 * lms_[..., 2]
        return np.clip(L, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# DITHERING  —  All algorithms return uint8 {0, 1}.  0 = black, 1 = white.
# ══════════════════════════════════════════════════════════════════════════════


@jit(nopython=True, cache=False, parallel=False)
def _error_diffuse_numba(
    out: np.ndarray,  # float64 (H, W)
    dx: np.ndarray,  # int64  (K,)
    dy: np.ndarray,  # int64  (K,)
    weights: np.ndarray,  # float64 (K,)
    H: int,
    W: int,
) -> None:
    """Numba-compiled serpentine error diffusion core.
    Modifies *out* in-place.
    """
    for y in range(H):
        left_to_right = y % 2 == 0
        sign = 1 if left_to_right else -1

        if left_to_right:
            x_range = range(W)
        else:
            x_range = range(W - 1, -1, -1)

        for x in x_range:
            old = out[y, x]
            new = 1.0 if old > 0.5 else 0.0
            err = old - new
            if err == 0.0:
                # Point already exactly black or white — no error to diffuse
                out[y, x] = new
                continue
            out[y, x] = new

            for k in range(len(dx)):
                nx = x + dx[k] * sign
                ny = y + dy[k]
                if 0 <= nx < W and 0 <= ny < H:
                    out[ny, nx] += err * weights[k]


class Ditherer:
    """1-bit dithering algorithms on a 2-D L* array."""

    # ── Bayer matrix ──────────────────────────────────────────────────────
    @staticmethod
    def _bayer_matrix(n: int) -> np.ndarray:
        if n == 1:
            return np.array([[0]], dtype=np.float64)
        m = Ditherer._bayer_matrix(n // 2)
        top = np.hstack([4 * m, 4 * m + 2])
        bot = np.hstack([4 * m + 3, 4 * m + 1])
        return np.vstack([top, bot])

    # ── Public methods ────────────────────────────────────────────────────

    @staticmethod
    def threshold(L: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        return (L > threshold).astype(np.uint8)

    @staticmethod
    def floyd_steinberg(L: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
        return Ditherer._error_diffuse(
            L,
            [
                ((1, 0), 7),
                ((-1, 1), 3),
                ((0, 1), 5),
                ((1, 1), 1),
            ],
            16,
            strength=strength,
        )

    @staticmethod
    def atkinson(L: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
        return Ditherer._error_diffuse(
            L,
            [
                ((1, 0), 1),
                ((2, 0), 1),
                ((-1, 1), 1),
                ((0, 1), 1),
                ((1, 1), 1),
                ((0, 2), 1),
            ],
            8,
            strength=strength,
        )

    @staticmethod
    def jarvis(L: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
        return Ditherer._error_diffuse(
            L,
            [
                ((1, 0), 7),
                ((2, 0), 5),
                ((-2, 1), 3),
                ((-1, 1), 5),
                ((0, 1), 7),
                ((1, 1), 5),
                ((2, 1), 3),
                ((-2, 2), 1),
                ((-1, 2), 3),
                ((0, 2), 5),
                ((1, 2), 3),
                ((2, 2), 1),
            ],
            48,
            strength=strength,
        )

    @staticmethod
    def stucki(L: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
        return Ditherer._error_diffuse(
            L,
            [
                ((1, 0), 8),
                ((2, 0), 4),
                ((-2, 1), 2),
                ((-1, 1), 4),
                ((0, 1), 8),
                ((1, 1), 4),
                ((2, 1), 2),
                ((-2, 2), 1),
                ((-1, 2), 2),
                ((0, 2), 4),
                ((1, 2), 2),
                ((2, 2), 1),
            ],
            42,
            strength=strength,
        )

    @staticmethod
    def sierra(L: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
        return Ditherer._error_diffuse(
            L,
            [
                ((1, 0), 5),
                ((2, 0), 3),
                ((-2, 1), 2),
                ((-1, 1), 4),
                ((0, 1), 5),
                ((1, 1), 4),
                ((2, 1), 2),
                ((-1, 2), 2),
                ((0, 2), 3),
                ((1, 2), 2),
            ],
            32,
            strength=strength,
        )

    @staticmethod
    def bayer(L: np.ndarray, *, size: int = 4) -> np.ndarray:
        matrix = Ditherer._bayer_matrix(size)
        norm = matrix / (size * size)
        H, W = L.shape
        ty = (H + size - 1) // size
        tx = (W + size - 1) // size
        tiled = np.tile(norm, (ty, tx))[:H, :W]
        return (L > tiled).astype(np.uint8)

    # ── Error diffusion core ──────────────────────────────────────────────

    @staticmethod
    def _error_diffuse(
        L: np.ndarray,
        coeffs: list[tuple[tuple[int, int], int]],
        divisor: int,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Serpentine-scan error diffusion (numba JIT).

        Row 0 runs L→R.  Odd rows run R→L with x-coefficients mirrored.
        *strength* in [0, 1] scales the error before distribution.
        """
        out = L.copy().astype(np.float64)
        H, W = out.shape

        # Extract coefficient arrays for numba
        K = len(coeffs)
        dx_arr = np.empty(K, dtype=np.int64)
        dy_arr = np.empty(K, dtype=np.int64)
        w_arr = np.empty(K, dtype=np.float64)
        for i, ((dx, dy), w) in enumerate(coeffs):
            dx_arr[i] = dx
            dy_arr[i] = dy
            w_arr[i] = (w * strength) / divisor

        _error_diffuse_numba(out, dx_arr, dy_arr, w_arr, H, W)

        return np.clip(np.round(out), 0, 1).astype(np.uint8)

    # ── Lookup ────────────────────────────────────────────────────────────

    @staticmethod
    def algorithms() -> dict[str, tuple[str, bool, bool]]:
        """{label: (method_name, needs_bayer_size, needs_strength)}."""
        return {
            "Simple Threshold": ("threshold", False, False),
            "Floyd-Steinberg": ("floyd_steinberg", False, True),
            "Atkinson": ("atkinson", False, True),
            "Jarvis-Judice-Ninke": ("jarvis", False, True),
            "Stucki": ("stucki", False, True),
            "Sierra": ("sierra", False, True),
            "Bayer Ordered": ("bayer", True, False),
        }

    @staticmethod
    def apply(label: str, L: np.ndarray, bayer_size: int = 4, strength: float = 1.0) -> np.ndarray:
        meth_name, needs_bayer, needs_strength = Ditherer.algorithms()[label]
        fn = getattr(Ditherer, meth_name)
        kwargs: dict[str, object] = {}
        if needs_bayer:
            kwargs["size"] = bayer_size
        if needs_strength:
            kwargs["strength"] = strength
        return fn(L, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
#  BRIGHTNESS CURVE  —  control-point model  (smooth PCHIP  /  piecewise linear)
# ══════════════════════════════════════════════════════════════════════════════


class BrightnessCurve:
    """Curve through adjustable control points with two interpolation modes.

    *mode* — ``"smooth"`` (monotone cubic / PCHIP, default) or ``"linear"``
             (piecewise straight segments).
    End-points are locked at x = 0 and x = 1.
    """

    def __init__(self, mode: str = "smooth") -> None:
        self.mode = mode  # "smooth" | "linear"
        self.points: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
        self._need_rebuild = True

    # ── mutation ──────────────────────────────────────────────────────────

    def add(self, x: float, y: float) -> int:
        """Insert an interior control point, keeping points sorted by x.

        Returns the index where the point was inserted.
        """
        x = float(np.clip(float(x), 0.0, 1.0))
        y = float(np.clip(float(y), 0.0, 1.0))
        # Keep endpoints fixed at index 0 and index -1; insert interior
        # points in ascending x order so dragging works correctly.
        idx = 1
        while idx < len(self.points) - 1 and self.points[idx][0] < x:
            idx += 1
        self.points.insert(idx, (x, y))
        self._need_rebuild = True
        return idx

    def remove(self, index: int) -> None:
        if 0 < index < len(self.points) - 1:
            self.points.pop(index)
            self._need_rebuild = True

    def move(self, index: int, x: float, y: float) -> None:
        x = float(np.clip(float(x), 0.0, 1.0))
        y = float(np.clip(float(y), 0.0, 1.0))
        if index == 0:
            self.points[0] = (0.0, y)
        elif index == len(self.points) - 1:
            self.points[-1] = (1.0, y)
        else:
            self.points[index] = (x, y)
        self._need_rebuild = True

    def reset(self) -> None:
        self.points = [(0.0, 0.0), (1.0, 1.0)]
        self._need_rebuild = True

    def set_mode(self, mode: str) -> None:
        """Switch interpolation mode (``"smooth"`` | ``"linear"``)."""
        self.mode = mode
        if mode == "smooth":
            self._need_rebuild = True

    # ── serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, object]:
        return {"mode": self.mode, "points": self.points}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "BrightnessCurve":
        curve = cls(mode=str(data.get("mode", "smooth")))
        pts = data.get("points")
        if isinstance(pts, list) and len(pts) >= 2:
            curve.points = [(float(p[0]), float(p[1])) for p in pts]
        return curve

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "BrightnessCurve":
        import json

        return cls.from_dict(json.loads(text))

    # ── evaluation ────────────────────────────────────────────────────────

    def evaluate(self, xs: np.ndarray) -> np.ndarray:
        if self.mode == "linear":
            return self._evaluate_linear(xs)
        return self._evaluate_smooth(xs)

    def _evaluate_smooth(self, xs: np.ndarray) -> np.ndarray:
        pts = self._sorted_deduped()
        if len(pts) < 2:
            return np.clip(xs, 0.0, 1.0)
        xs_pts = np.array([p[0] for p in pts], dtype=np.float64)
        ys_pts = np.array([p[1] for p in pts], dtype=np.float64)
        interp = PchipInterpolator(xs_pts, ys_pts)
        return np.clip(interp(xs), 0.0, 1.0)

    def _evaluate_linear(self, xs: np.ndarray) -> np.ndarray:
        pts = self._sorted_deduped()
        if len(pts) < 2:
            return np.clip(xs, 0.0, 1.0)
        xs_pts = [p[0] for p in pts]
        ys_pts = [p[1] for p in pts]
        return np.clip(np.interp(xs, xs_pts, ys_pts), 0.0, 1.0)

    def _sorted_deduped(self) -> list[tuple[float, float]]:
        pts = sorted(self.points, key=lambda p: p[0])
        deduped = []
        seen_x = set()
        for x, y in pts:
            if x not in seen_x:
                deduped.append((x, y))
                seen_x.add(x)
        return deduped


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE  —  image → OKLAB L* → curve → dither
# ══════════════════════════════════════════════════════════════════════════════


def run_pipeline(
    pil_image: Image.Image,
    w: int,
    h: int,
    curve_points: list[tuple[float, float]],
    curve_mode: str,
    algo_name: str,
    bayer_size: int,
    strength: float,
    lightness: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the mono dithering pipeline synchronously.

    Args:
        pil_image: Source image in RGB mode.
        w, h: Target output resolution.
        curve_points: Control points for the brightness curve.
        curve_mode: "smooth" or "linear".
        algo_name: Dithering algorithm name.
        bayer_size: Bayer matrix size (used only for Bayer Ordered).
        strength: Error diffusion strength in [0, 1].
        lightness: Optional cached OKLAB L* array. If provided, the downscale
            and OKLAB conversion steps are skipped.

    Returns:
        (bits, L): the 1-bit result and the OKLAB L* array that was used.
    """
    if lightness is not None:
        L = lightness
    else:
        small = pil_image.resize((w, h), Image.Resampling.LANCZOS)
        rgb = np.asarray(small, dtype=np.float64) / 255.0
        L = Oklab.lightness(rgb)

    curve = BrightnessCurve(mode=curve_mode)
    for pt in curve_points:
        curve.points.append(pt)
    if len(curve_points) >= 2:
        curve.points = list(curve_points)
    L_curved = curve.evaluate(L.ravel()).reshape(L.shape)

    bits = Ditherer.apply(algo_name, L_curved, bayer_size, strength)
    return bits, L


# ══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND WORKER  —  runs the full pipeline off the main thread
# ══════════════════════════════════════════════════════════════════════════════


class PipelineJob:
    """Immutable snapshot of parameters for one pipeline run.

    If *lightness* is provided (cached OKLAB L*), the worker skips
    the downscale + OKLAB step and goes straight to curve + dither.
    """

    __slots__ = (
        "pil_image",
        "w",
        "h",
        "curve_points",
        "curve_mode",
        "algo_name",
        "bayer_size",
        "strength",
        "generation",
        "lightness",
    )

    def __init__(
        self,
        pil_image: Image.Image,
        w: int,
        h: int,
        curve_points: list[tuple[float, float]],
        curve_mode: str,
        algo_name: str,
        bayer_size: int,
        strength: float,
        generation: int,
        lightness: np.ndarray | None = None,
    ) -> None:
        self.pil_image = pil_image
        self.w = w
        self.h = h
        self.curve_points = curve_points
        self.curve_mode = curve_mode
        self.algo_name = algo_name
        self.bayer_size = bayer_size
        self.strength = strength
        self.generation = generation
        self.lightness = lightness


class PipelineResult:
    """Result emitted from the worker thread.

    *fresh_lightness* is set only when the worker performed a full path
    (downscale + OKLAB).  The main thread should cache it.
    """

    __slots__ = ("bits", "fresh_lightness", "w", "h", "generation", "t_ms")

    def __init__(
        self, bits: np.ndarray, w: int, h: int, generation: int, t_ms: float, fresh_lightness: np.ndarray | None = None
    ) -> None:
        self.bits = bits
        self.fresh_lightness = fresh_lightness
        self.w = w
        self.h = h
        self.generation = generation
        self.t_ms = t_ms


class PipelineWorker(QThread):
    """Offloads the full pipeline to a background thread.

    Signals
    -------
    result_ready(PipelineResult) :  emitted when one pipeline pass completes.
    """

    result_ready = pyqtSignal(object)  # PipelineResult

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pending: PipelineJob | None = None
        self._condition = threading.Condition()
        self._stopping = False

    def submit(self, job: PipelineJob) -> None:
        """Replace any pending job and schedule the new one."""
        with self._condition:
            self._pending = job
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify()

    def run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                job = self._pending
                self._pending = None

            assert job is not None
            is_light = job.lightness is not None
            try:
                t0 = time.perf_counter()

                bits, L = run_pipeline(
                    job.pil_image,
                    job.w,
                    job.h,
                    job.curve_points,
                    job.curve_mode,
                    job.algo_name,
                    job.bayer_size,
                    job.strength,
                    lightness=job.lightness,
                )
                fresh_L = None if is_light else L.copy()

                t = (time.perf_counter() - t0) * 1000
                result = PipelineResult(bits, job.w, job.h, job.generation, t, fresh_lightness=fresh_L)
                self.result_ready.emit(result)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  CURVE EDITOR  —  interactive QWidget
# ══════════════════════════════════════════════════════════════════════════════


class CurveEditor(QWidget):
    """Draggable spline / linear curve editor for brightness mapping.

    Signals
    -------
    curveChanged :  emitted on every user interaction that alters the curve.
    """

    curveChanged = pyqtSignal()

    # visual constants
    _PAD_L = 48
    _PAD_R = 14
    _PAD_T = 14
    _PAD_B = 34
    _PT_RADIUS = 6
    _HIT_RADIUS = 9
    _CURVE_STEPS = 200

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.curve = BrightnessCurve()
        self._dragging: int = -1
        self._hovered: int = -1
        self._plot = QRect()  # cached plot rect (valid after first resize)

        # Relative-drag anchors (work around Wayland jump-to-edge issue)
        self._drag_start_widget: QPointF | None = None
        self._drag_start_curve: tuple[float, float] = (0.0, 0.0)
        self._drag_point_start: tuple[float, float] = (0.0, 0.0)

        self.setMinimumSize(240, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    # ── public helpers ────────────────────────────────────────────────────

    def reset_curve(self) -> None:
        self.curve.reset()
        self.curveChanged.emit()
        self.update()

    def set_mode(self, mode: str) -> None:
        self.curve.set_mode(mode)
        self.curveChanged.emit()
        self.update()

    # ── resize ────────────────────────────────────────────────────────────

    def resizeEvent(self, a0: QResizeEvent | None) -> None:  # noqa: N802, N803
        super().resizeEvent(a0)
        self._plot = self._compute_plot_rect()

    # ── painting ──────────────────────────────────────────────────────────

    def paintEvent(self, a0) -> None:  # noqa: N802, N803
        _ = a0
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(38, 38, 38))

        if self._plot.isNull() or self._plot.width() < 10 or self._plot.height() < 10:
            return  # not ready yet

        pl, pt, pr, pb = (self._plot.left(), self._plot.top(), self._plot.right(), self._plot.bottom())
        pw = pr - pl
        ph = pb - pt

        # ── grid ──────────────────────────────────────────────────────
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        for i in range(11):
            x = pl + pw * (i / 10)
            painter.drawLine(QLineF(int(x), pt, int(x), pb))
            y = pt + ph * (1.0 - i / 10)
            painter.drawLine(QLineF(pl, int(y), pr, int(y)))

        # ── axis labels ───────────────────────────────────────────────
        label_font = QFont("monospace", 9)
        painter.setFont(label_font)
        painter.setPen(QColor(160, 160, 160))
        for i in range(11):
            val = i / 10
            # x-axis
            x = pl + pw * val
            label = f"{val:.1f}"
            fm = QFontMetrics(label_font)
            painter.drawText(int(x - fm.horizontalAdvance(label) / 2), pb + 20, label)
            # y-axis
            y = pt + ph * (1.0 - val)
            label = f"{val:.1f}"
            painter.drawText(pl - 44, int(y + fm.height() / 3), label)

        painter.setPen(QColor(180, 180, 180))
        painter.setFont(QFont("sans-serif", 9))
        painter.drawText(pl + pw // 2 - 18, self.height() - 3, "Input L*")
        painter.drawText(4, pt + 12, "Out")

        # ── identity reference line ───────────────────────────────────
        painter.setPen(QPen(QColor(80, 80, 80), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QLineF(pl, pb, pr, pt))

        # ── curve ─────────────────────────────────────────────────────
        xs = np.linspace(0.0, 1.0, self._CURVE_STEPS)
        ys = self.curve.evaluate(xs)
        pts = [QPointF(pl + xs[i] * pw, pb - ys[i] * ph) for i in range(len(xs))]

        curve_col = QColor(0, 200, 255)
        painter.setPen(QPen(curve_col, 2))
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])

        # ── control points ────────────────────────────────────────────
        for idx, (cx, cy) in enumerate(self.curve.points):
            px = pl + cx * pw
            py = pb - cy * ph
            active = idx == self._dragging or idx == self._hovered
            if active:
                painter.setBrush(QBrush(QColor(255, 200, 60)))
                painter.setPen(QPen(Qt.GlobalColor.white, 2))
                r = self._PT_RADIUS + 2
            else:
                painter.setBrush(QBrush(curve_col))
                painter.setPen(QPen(QColor(180, 230, 255), 1))
                r = self._PT_RADIUS
            painter.drawEllipse(QPointF(px, py), r, r)

            # endpoint labels
            if idx == 0:
                painter.setPen(QColor(160, 160, 160))
                painter.drawText(int(px) - 10, int(py) - 12, "0")
            elif idx == len(self.curve.points) - 1:
                painter.setPen(QColor(160, 160, 160))
                painter.drawText(int(px) + 8, int(py) + 18, "1")

    # ── mouse ─────────────────────────────────────────────────────────────

    def _ensure_valid_size(self) -> bool:
        """Return True if the widget is laid out and ready for interaction."""
        return self._plot.isValid() and self._plot.width() >= 20 and self._plot.height() >= 20

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:  # noqa: N802, N803
        if a0 is None or a0.button() != Qt.MouseButton.LeftButton:
            return
        if not self._ensure_valid_size():
            return
        pos = a0.position()
        idx = self._hit_test(pos)
        if idx >= 0:
            cx, cy = self._widget_to_curve(pos)
            self._dragging = idx
            self._drag_start_widget = QPointF(pos.x(), pos.y())
            self._drag_start_curve = (cx, cy)
            self._drag_point_start = self.curve.points[idx]
        else:
            cx, cy = self._widget_to_curve(pos)
            self.curve.add(cx, cy)
            self.curveChanged.emit()
            self.update()

    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:  # noqa: N802, N803
        if a0 is None:
            return
        if self._dragging >= 0:
            if not self._ensure_valid_size():
                return
            pos = a0.position()
            cx, cy = self._widget_to_curve(pos)
            # Relative drag: move the point by the curve-space delta from the
            # press position.  This avoids jumps caused by Wayland delivering a
            # bogus position on the first move after press.
            dx = cx - self._drag_start_curve[0]
            dy = cy - self._drag_start_curve[1]
            new_x = self._drag_point_start[0] + dx
            new_y = self._drag_point_start[1] + dy
            self.curve.move(self._dragging, new_x, new_y)
            self.curveChanged.emit()
            self.update()
        else:
            idx = self._hit_test(a0.position())
            if idx != self._hovered:
                self._hovered = idx
                self.update()

    def mouseReleaseEvent(self, a0: QMouseEvent | None) -> None:  # noqa: N802, N803
        if a0 is not None and a0.button() == Qt.MouseButton.LeftButton:
            # Guard against spurious release events (observed on Wayland).
            # If the button is still reported as pressed by buttons(),
            # this is NOT a real release — ignore it.
            if Qt.MouseButton.LeftButton in a0.buttons():
                return
            self._dragging = -1
            self._drag_start_widget = None
        self.update()

    def mouseDoubleClickEvent(self, a0: QMouseEvent | None) -> None:  # noqa: N802, N803
        if a0 is None:
            return
        if not self._ensure_valid_size():
            return
        idx = self._hit_test(a0.position())
        if idx >= 0:
            self.curve.remove(idx)
            self.curveChanged.emit()
            self.update()

    # ── coordinate helpers ────────────────────────────────────────────────

    def _compute_plot_rect(self) -> QRect:
        w, h = self.width(), self.height()
        return QRect(self._PAD_L, self._PAD_T, max(w - self._PAD_L - self._PAD_R, 20), max(h - self._PAD_T - self._PAD_B, 20))

    def _widget_to_curve(self, pos: QPointF) -> tuple[float, float]:
        r = self._plot
        if r.width() < 1 or r.height() < 1:
            return (0.5, 0.5)
        x = (pos.x() - r.left()) / r.width()
        y = (r.bottom() - pos.y()) / r.height()
        return (np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0))

    def _hit_test(self, pos: QPointF) -> int:
        r = self._plot
        if r.width() < 1 or r.height() < 1:
            return -1
        best = -1
        best_d = self._HIT_RADIUS + 1
        for idx, (cx, cy) in enumerate(self.curve.points):
            px = r.left() + cx * r.width()
            py = r.bottom() - cy * r.height()
            d = np.hypot(pos.x() - px, pos.y() - py)
            if d < self._HIT_RADIUS and d < best_d:
                best_d = d
                best = idx
        return best


# ══════════════════════════════════════════════════════════════════════════════
#  HISTOGRAM WIDGET
# ══════════════════════════════════════════════════════════════════════════════


class HistogramWidget(QWidget):
    """Small live histogram of OKLAB L* lightness."""

    _BINS = 64
    _BAR_COLOR = QColor(0, 180, 255, 180)
    _BORDER = QColor(80, 80, 80)
    _BG = QColor(30, 30, 30)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._hist: np.ndarray | None = None
        self._max_val: float = 1.0
        self.setMinimumHeight(36)
        self.setMaximumHeight(48)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_lightness(self, L: np.ndarray | None) -> None:
        """Compute histogram from OKLAB L* array."""
        if L is None:
            self._hist = None
        else:
            hist, _ = np.histogram(L.ravel(), bins=self._BINS, range=(0.0, 1.0))
            self._hist = hist.astype(np.float64)
            self._max_val = float(hist.max()) if hist.max() > 0 else 1.0
        self.update()

    def paintEvent(self, a0) -> None:  # noqa: N802, N803
        _ = a0
        painter = QPainter(self)
        W, H = self.width(), self.height()
        painter.fillRect(0, 0, W, H, self._BG)

        if self._hist is None or self._hist.sum() == 0:
            return

        bar_w = max(1, int((W - 2) / self._BINS))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self._BAR_COLOR))
        for i in range(self._BINS):
            bar_h = int((self._hist[i] / self._max_val) * (H - 4))
            if bar_h > 0:
                x = 1 + i * bar_w
                painter.drawRect(x, H - 2 - bar_h, bar_w - 1, bar_h)

        # Border
        painter.setPen(QPen(self._BORDER, 1))
        painter.drawRect(0, 0, W - 1, H - 1)


class BayerSizeSpinBox(QSpinBox):
    """Spin box that only steps through valid Bayer matrix sizes."""

    _VALID_VALUES = (2, 4, 8)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRange(self._VALID_VALUES[0], self._VALID_VALUES[-1])
        self.setSingleStep(1)

    def stepBy(self, steps: int) -> None:  # noqa: N802
        current = self.value()
        values = self._VALID_VALUES
        if steps > 0:
            larger = [v for v in values if v > current]
            if not larger:
                self.setValue(values[-1])
                return
            idx = min(len(larger) - 1, steps - 1)
            self.setValue(larger[idx])
            return
        if steps < 0:
            smaller = [v for v in values if v < current]
            if not smaller:
                self.setValue(values[0])
                return
            idx = max(0, len(smaller) + steps)
            self.setValue(smaller[idx])


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════


class MainWindow(QMainWindow):
    """Application entry-point."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("pic_mono_dither — 1-bit Image Preprocessor")
        self.resize(960, 800)

        # ── processing state ───────────────────────────────────────────
        self._original_pil: Image.Image | None = None
        self._result: np.ndarray | None = None
        self._generation: int = 0
        self._current_image_path: str = ""

        # cached OKLAB L* at current target resolution
        self._lightness: np.ndarray | None = None
        self._light_w: int = 0
        self._light_h: int = 0

        # ── preview zoom state ─────────────────────────────────────────
        self._preview_fit: bool = True  # True = fit to viewport
        self._preview_scale: float = 1.0  # active when _preview_fit is False
        self._preview_min_scale: float = 0.1
        self._preview_max_scale: float = 16.0

        # ── debounce timer (batches rapid changes) ─────────────────────
        self._dirty = False
        self._full_recompute = True  # first load = full
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(8)  # ~120 fps cap; less latency
        self._debounce.timeout.connect(self._on_debounce_fire)

        # ── background worker ──────────────────────────────────────────
        self._worker = PipelineWorker()
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.start()

        self._setup_ui()
        self._setup_connections()

        # Accept drag-and-drop image files
        self.setAcceptDrops(True)

    # ═══════════════════════════════════════════════════════════════════
    #  UI construction
    # ═══════════════════════════════════════════════════════════════════

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setSpacing(8)

        # ─── file row ──────────────────────────────────────────────────
        file_row = QHBoxLayout()
        self.open_btn = QPushButton("Open Image…")
        self.help_btn = QPushButton("Keybindings")
        self.help_btn.setToolTip("Show keyboard and mouse shortcuts (F1)")
        self.file_label = QLabel("No image loaded")
        self.file_label.setStyleSheet("color: #888;")

        # Recent files dropdown attached to the open button
        self._recent_menu = QMenu(self.open_btn)
        self.open_btn.setMenu(self._recent_menu)
        self._load_recent_menu()

        file_row.addWidget(self.open_btn)
        file_row.addWidget(self.help_btn)
        file_row.addWidget(self.file_label, 1)
        vbox.addLayout(file_row)

        # ─── settings row ──────────────────────────────────────────────
        settings_row = QHBoxLayout()
        settings_row.setSpacing(12)

        res_group = QHBoxLayout()
        res_group.addWidget(QLabel("Resolution:"))
        self.w_spin = QSpinBox()
        self.w_spin.setRange(1, 99999)
        self.w_spin.setValue(384)
        self.h_spin = QSpinBox()
        self.h_spin.setRange(1, 99999)
        self.h_spin.setValue(240)
        self.lock_btn = QToolButton()
        self.lock_btn.setCheckable(True)
        self.lock_btn.setChecked(True)
        self.lock_btn.setText("🔗")
        self.lock_btn.setToolTip("Lock aspect ratio to original image")
        res_group.addWidget(self.w_spin)
        res_group.addWidget(QLabel("×"))
        res_group.addWidget(self.h_spin)
        res_group.addWidget(self.lock_btn)
        settings_row.addLayout(res_group)

        settings_row.addSpacing(24)

        self.algo_combo = QComboBox()
        for label in Ditherer.algorithms():
            self.algo_combo.addItem(label)
        self.algo_combo.setCurrentText("Floyd-Steinberg")
        settings_row.addWidget(QLabel("Dither:"))
        settings_row.addWidget(self.algo_combo)

        self.bayer_label = QLabel("Bayer size:")
        self.bayer_spin = BayerSizeSpinBox()
        self.bayer_spin.setValue(4)
        self.bayer_spin.setToolTip("Bayer matrix size (power of two)")
        settings_row.addWidget(self.bayer_label)
        settings_row.addWidget(self.bayer_spin)

        self.strength_label = QLabel("Strength:")
        self.strength_spin = QSpinBox()
        self.strength_spin.setRange(0, 100)
        self.strength_spin.setValue(100)
        self.strength_spin.setSuffix("%")
        self.strength_spin.setToolTip("Error diffusion strength. " "100% = full algorithm, 0% = simple threshold.")
        settings_row.addWidget(self.strength_label)
        settings_row.addWidget(self.strength_spin)
        settings_row.addStretch(1)
        vbox.addLayout(settings_row)

        # ─── preview area ──────────────────────────────────────────────
        preview_row = QHBoxLayout()
        preview_row.setSpacing(8)

        # original thumbnail
        orig_box = QFrame()
        orig_box.setFrameShape(QFrame.Shape.StyledPanel)
        orig_box.setStyleSheet("QFrame { background: #1e1e1e; border: 1px solid #444; }")
        orig_layout = QVBoxLayout(orig_box)
        orig_layout.setContentsMargins(6, 6, 6, 6)
        orig_layout.addWidget(QLabel("Original"), 0, Qt.AlignmentFlag.AlignCenter)
        self.orig_preview = QLabel()
        self.orig_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.orig_preview.setMinimumSize(180, 120)
        self.orig_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        orig_layout.addWidget(self.orig_preview, 1)
        preview_row.addWidget(orig_box, 1)

        # result preview
        result_box = QFrame()
        result_box.setFrameShape(QFrame.Shape.StyledPanel)
        result_box.setStyleSheet("QFrame { background: #1e1e1e; border: 1px solid #666; }")
        result_layout = QVBoxLayout(result_box)
        result_layout.setContentsMargins(6, 6, 6, 6)
        # result toolbar
        result_header = QHBoxLayout()
        self.result_info = QLabel("No result")
        self.result_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_header.addWidget(self.result_info, 1)
        self.zoom_btn = QPushButton("Fit")
        self.zoom_btn.setToolTip("Toggle fit-to-view / 1:1 (mouse wheel zooms)")
        self.compare_btn = QPushButton("A/B")
        self.compare_btn.setCheckable(True)
        self.compare_btn.setToolTip("Hold to compare with threshold baseline")
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setToolTip("Copy result to clipboard (Ctrl+Shift+C)")
        result_header.addWidget(self.zoom_btn)
        result_header.addWidget(self.compare_btn)
        result_header.addWidget(self.copy_btn)
        result_layout.addLayout(result_header, 0)

        # scrollable result preview
        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(False)
        self.result_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_preview = QLabel()
        self.result_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_preview.setStyleSheet("background: transparent;")
        self.result_scroll.setWidget(self.result_preview)
        self.result_scroll.installEventFilter(self)
        self.result_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.result_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        result_layout.addWidget(self.result_scroll, 1)

        # lightness histogram
        self.histogram = HistogramWidget()
        result_layout.addWidget(self.histogram, 0)

        preview_row.addWidget(result_box, 3)
        vbox.addLayout(preview_row, 1)

        # ─── curve editor ──────────────────────────────────────────────
        curve_row = QHBoxLayout()
        curve_row.setSpacing(8)

        curve_frame = QFrame()
        curve_frame.setFrameShape(QFrame.Shape.StyledPanel)
        curve_frame.setStyleSheet("QFrame { background: #262626; border: 1px solid #444; }")
        curve_lay = QVBoxLayout(curve_frame)
        curve_lay.setContentsMargins(6, 4, 6, 4)

        curve_toolbar = QHBoxLayout()
        curve_toolbar.addWidget(QLabel("Brightness Curve"))
        curve_toolbar.addStretch(1)
        self.curve_mode_btn = QPushButton("Smooth")
        self.curve_mode_btn.setCheckable(True)
        self.curve_mode_btn.setChecked(False)
        self.curve_mode_btn.setToolTip("Toggle between smooth (PCHIP) and linear interpolation")
        self.reset_curve_btn = QPushButton("Reset")
        curve_toolbar.addWidget(self.curve_mode_btn)
        curve_toolbar.addWidget(self.reset_curve_btn)
        curve_lay.addLayout(curve_toolbar)

        self.curve_editor = CurveEditor()
        self.curve_editor.setMinimumHeight(170)
        curve_lay.addWidget(self.curve_editor, 1)

        curve_row.addWidget(curve_frame, 1)
        vbox.addLayout(curve_row, 0)

        # ─── save buttons ──────────────────────────────────────────────
        save_row = QHBoxLayout()
        self.save_png_btn = QPushButton("Save as PNG…")
        self.save_pbm_btn = QPushButton("Save as PBM…")
        save_row.addWidget(self.save_png_btn)
        save_row.addWidget(self.save_pbm_btn)
        save_row.addStretch(1)
        vbox.addLayout(save_row)

        # ─── status bar ────────────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, 1)

    # ═══════════════════════════════════════════════════════════════════
    #  Signal wiring
    # ═══════════════════════════════════════════════════════════════════

    def _setup_connections(self) -> None:
        self.open_btn.clicked.connect(self._on_open)
        self.help_btn.clicked.connect(self._show_keybindings_help)
        self.w_spin.valueChanged.connect(self._on_res_change)
        self.h_spin.valueChanged.connect(self._on_res_change)
        self.lock_btn.toggled.connect(self._on_lock_toggle)
        self.algo_combo.currentTextChanged.connect(self._on_algo_change)
        self.bayer_spin.valueChanged.connect(self._snap_bayer)
        self.strength_spin.valueChanged.connect(self._schedule_light)
        self.curve_editor.curveChanged.connect(self._schedule_light)

        self.curve_mode_btn.toggled.connect(self._on_curve_mode_toggle)
        self.reset_curve_btn.clicked.connect(self.curve_editor.reset_curve)

        self.save_png_btn.clicked.connect(self._save_png)
        self.save_pbm_btn.clicked.connect(self._save_pbm)

        self.zoom_btn.clicked.connect(self._on_zoom_toggle)
        self.compare_btn.toggled.connect(self._on_compare_toggle)
        self.copy_btn.clicked.connect(self._copy_to_clipboard)

        # Keyboard shortcuts
        self._setup_shortcuts()

    def _setup_shortcuts(self) -> None:
        # Application actions
        open_action = QAction("Open…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open)
        self.addAction(open_action)

        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        self.addAction(quit_action)

        save_png_action = QAction("Save PNG", self)
        save_png_action.setShortcut(QKeySequence("Ctrl+S"))
        save_png_action.triggered.connect(self._save_png)
        self.addAction(save_png_action)

        save_pbm_action = QAction("Save PBM", self)
        save_pbm_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_pbm_action.triggered.connect(self._save_pbm)
        self.addAction(save_pbm_action)

        # Curve editor shortcuts
        del_shortcut = QShortcut(QKeySequence.StandardKey.Delete, self)
        del_shortcut.activated.connect(self._on_delete_curve_point)

        esc_shortcut = QShortcut(QKeySequence("Esc"), self)
        esc_shortcut.activated.connect(self._on_escape_curve)

        copy_shortcut = QShortcut(QKeySequence("Ctrl+Shift+C"), self)
        copy_shortcut.activated.connect(self._copy_to_clipboard)

        help_shortcut = QShortcut(QKeySequence("F1"), self)
        help_shortcut.activated.connect(self._show_keybindings_help)

    def _show_keybindings_help(self) -> None:
        QMessageBox.information(
            self,
            "Keybindings",
            "\n".join(
                [
                    "Keyboard shortcuts",
                    "",
                    "F1                Show this help",
                    "Ctrl+O            Open image",
                    "Ctrl+S            Save PNG",
                    "Ctrl+Shift+S      Save PBM",
                    "Ctrl+Shift+C      Copy result to clipboard",
                    "Delete            Remove hovered/selected curve point",
                    "Esc               Cancel active curve drag",
                    "",
                    "Mouse actions",
                    "",
                    "Mouse wheel       Zoom result preview",
                    "Click curve       Add a control point",
                    "Drag point        Move a control point",
                    "Double-click      Remove a control point",
                ]
            ),
        )

    def _on_delete_curve_point(self) -> None:
        """Delete the hovered or currently dragged curve point."""
        target = self.curve_editor._dragging
        if target < 0:
            target = self.curve_editor._hovered
        if target > 0 and target < len(self.curve_editor.curve.points) - 1:
            self.curve_editor.curve.remove(target)
            self.curve_editor._dragging = -1
            self.curve_editor._hovered = -1
            self.curve_editor.update()
            self._schedule_light()

    def _on_escape_curve(self) -> None:
        """Cancel an active drag and clear point hover."""
        if self.curve_editor._dragging >= 0:
            self.curve_editor._dragging = -1
            self.curve_editor.update()
        if self.curve_editor._hovered >= 0:
            self.curve_editor._hovered = -1
            self.curve_editor.update()

    # ═══════════════════════════════════════════════════════════════════
    #  Recent files
    # ═══════════════════════════════════════════════════════════════════

    _RECENT_KEY = "recent/files"
    _RECENT_MAX = 10

    def _recent_paths(self) -> list[str]:
        settings = QSettings()
        raw = settings.value(self._RECENT_KEY, [], type=list)
        return [str(p) for p in raw if isinstance(p, str) and os.path.isfile(p)]

    def _set_recent_paths(self, paths: list[str]) -> None:
        settings = QSettings()
        settings.setValue(self._RECENT_KEY, paths[: self._RECENT_MAX])

    def _load_recent_menu(self) -> None:
        self._recent_menu.clear()

        # Always include "Browse…" at the top so the file dialog is
        # reachable even when the button click shows the menu first.
        browse: QAction = self._recent_menu.addAction("Browse…")  # type: ignore[assignment]
        browse.triggered.connect(self._on_open)
        self._recent_menu.addSeparator()

        paths = self._recent_paths()
        if not paths:
            noop: QAction = self._recent_menu.addAction("(no recent files)")  # type: ignore[assignment]
            noop.setEnabled(False)
            return
        for p in paths:
            act: QAction = self._recent_menu.addAction(os.path.basename(p))  # type: ignore[assignment]
            act.setToolTip(p)
            act.triggered.connect(lambda checked=False, path=p: self._load_image(path))
        self._recent_menu.addSeparator()
        clr: QAction = self._recent_menu.addAction("Clear recent files")  # type: ignore[assignment]
        clr.triggered.connect(self._clear_recent_files)

    def _add_recent_file(self, path: str) -> None:
        paths = self._recent_paths()
        if path in paths:
            paths.remove(path)
        paths.insert(0, path)
        self._set_recent_paths(paths)
        self._load_recent_menu()

    def _clear_recent_files(self) -> None:
        self._set_recent_paths([])
        self._load_recent_menu()

    # ═══════════════════════════════════════════════════════════════════
    #  Zoom
    # ═══════════════════════════════════════════════════════════════════

    def _clear_lightness_cache(self) -> None:
        self._lightness = None
        self._light_w = 0
        self._light_h = 0
        self.histogram.set_lightness(None)

    def _on_zoom_toggle(self) -> None:
        """Toggle between fit-to-view and 1:1 display."""
        if self._preview_fit and self._result is not None:
            self._preview_fit = False
            self._preview_scale = 1.0
            self.zoom_btn.setText("1:1")
        else:
            self._preview_fit = True
            self.zoom_btn.setText("Fit")
        self._refresh_preview()

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:  # noqa: N802, N803
        """Zoom with mouse wheel on the result scroll area."""
        if a0 is self.result_scroll and a1 is not None and a1.type() == QEvent.Type.Wheel:
            wheel_event = a1  # type: ignore[assignment]
            if self._result is None:
                return False
            delta = wheel_event.angleDelta().y()  # type: ignore[union-attr]
            if delta == 0:
                return False
            factor = 1.15 if delta > 0 else 1.0 / 1.15
            new_scale = self._preview_scale * factor
            new_scale = max(self._preview_min_scale, min(self._preview_max_scale, new_scale))
            if new_scale != self._preview_scale:
                self._preview_fit = False
                self._preview_scale = new_scale
                self.zoom_btn.setText(f"{new_scale:.1f}×")
                self._refresh_preview()
            return True
        return super().eventFilter(a0, a1)

    def resizeEvent(self, a0: QResizeEvent | None) -> None:  # noqa: N802, N803
        super().resizeEvent(a0)
        if self._preview_fit:
            self._refresh_preview()

    # ═══════════════════════════════════════════════════════════════════
    #  Copy + Compare
    # ═══════════════════════════════════════════════════════════════════

    def _copy_to_clipboard(self) -> None:
        """Copy current 1-bit result to clipboard as a PNG image."""
        if self._result is None:
            return
        H, W = self._result.shape
        buf = (self._result * 255).astype(np.uint8)
        qimg = QImage(buf.tobytes(), W, H, W, QImage.Format.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        clip = QApplication.clipboard()
        if clip is not None:
            clip.setPixmap(pix)
            self.status_label.setText("Copied result to clipboard")

    def _on_compare_toggle(self, checked: bool) -> None:
        """Toggle A/B comparison: show threshold baseline instead of processed result."""
        if checked:
            # Compute the threshold baseline from cached lightness
            if self._lightness is not None:
                baseline = (self._lightness > 0.5).astype(np.uint8)
            elif self._original_pil is not None:
                W, H = self._light_w, self._light_h
                if W <= 0 or H <= 0:
                    return
                small = self._original_pil.resize((W, H), Image.Resampling.LANCZOS)
                rgb = np.asarray(small, dtype=np.float64) / 255.0
                L = Oklab.lightness(rgb)
                baseline = (L > 0.5).astype(np.uint8)
            else:
                return
            self._show_result(baseline)
            self.compare_btn.setText("A/B ✓")
        else:
            if self._result is not None:
                self._show_result(self._result)
            self.compare_btn.setText("A/B")

    # ═══════════════════════════════════════════════════════════════════
    #  Scheduling  (debounce + flag-based)
    # ═══════════════════════════════════════════════════════════════════

    def _schedule_light(self) -> None:
        """Re-run curve + dither + display (no downscale / OKLAB recompute)."""
        self._dirty = True
        if not self._debounce.isActive():
            self._debounce.start()

    def _schedule_full(self) -> None:
        """Re-run the complete pipeline (including downscale + OKLAB)."""
        self._full_recompute = True
        self._clear_lightness_cache()
        self._dirty = True
        if not self._debounce.isActive():
            self._debounce.start()

    def _on_debounce_fire(self) -> None:
        """Debounce timer fired — kick off the pipeline on the worker thread."""
        if self._original_pil is None or not self._dirty:
            return

        w = self.w_spin.value()
        h = self.h_spin.value()
        if w < 1 or h < 1:
            return

        self._dirty = False
        self._generation += 1

        # Decide whether to do a full or light pipeline pass.
        # Light = reuse cached OKLAB L* (resolution unchanged).
        need_full = self._full_recompute or self._lightness is None or self._light_w != w or self._light_h != h

        job = PipelineJob(
            pil_image=self._original_pil,
            w=w,
            h=h,
            curve_points=list(self.curve_editor.curve.points),
            curve_mode=self.curve_editor.curve.mode,
            algo_name=self.algo_combo.currentText(),
            bayer_size=self.bayer_spin.value(),
            strength=self.strength_spin.value() / 100.0,
            generation=self._generation,
            lightness=None if need_full else self._lightness,
        )
        self._worker.submit(job)

    def _on_result_ready(self, result: PipelineResult) -> None:
        """Pipeline finished — store and display."""
        if result.generation != self._generation:
            return

        self._result = result.bits

        # Cache the fresh L* from a full pipeline pass
        if result.fresh_lightness is not None:
            self._lightness = result.fresh_lightness
            self._light_w = result.w
            self._light_h = result.h
            self._full_recompute = False

        # Refresh histogram after every pass (full or light) so it
        # tracks the current curve's effect on the L* distribution.
        self._refresh_histogram()

        # Update status
        parts = []
        parts.append(f"Output: {result.w}×{result.h}")
        parts.append(f"·  {self.algo_combo.currentText()}")
        if self.algo_combo.currentText() == "Bayer Ordered":
            parts.append(f"(Bayer {self.bayer_spin.value()}×{self.bayer_spin.value()})")
        parts.append(f"·  {result.t_ms:.0f} ms")
        self.status_label.setText("  ".join(parts))

        self._show_result(result.bits)

    # ═══════════════════════════════════════════════════════════════════
    #  Slots  (UI event handlers)
    # ═══════════════════════════════════════════════════════════════════

    def _on_open(self) -> None:
        start_dir = os.path.dirname(self._current_image_path) if self._current_image_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", start_dir, "Image Files (*.png *.jpg *.jpeg *.tiff *.bmp *.webp *.gif)"
        )
        if not path:
            return
        self._load_image(path)

    def _load_image(self, path: str) -> None:
        """Load an image from disk and reset the pipeline."""
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Cannot open image:\n{exc}")
            return

        self._current_image_path = path
        self._original_pil = img
        self._clear_lightness_cache()
        self._full_recompute = True
        self.file_label.setText(f"{os.path.basename(path)}  ({img.width}×{img.height})")
        self.file_label.setStyleSheet("color: #ccc;")

        self._add_recent_file(path)

        # auto-set resolution
        aspect = img.width / img.height
        target_w = min(img.width, 384)
        target_h = max(1, int(target_w / aspect))
        self.w_spin.blockSignals(True)
        self.h_spin.blockSignals(True)
        self.w_spin.setValue(target_w)
        self.h_spin.setValue(target_h)
        self.w_spin.blockSignals(False)
        self.h_spin.blockSignals(False)

        self._update_original_thumb()
        self._schedule_full()

    def dragEnterEvent(self, a0: QDragEnterEvent | None) -> None:  # noqa: N802, N803
        if a0 is None:
            return
        mime = a0.mimeData()
        if mime is not None and mime.hasUrls():
            a0.acceptProposedAction()

    def dropEvent(self, a0: QDropEvent | None) -> None:  # noqa: N802, N803
        if a0 is None:
            return
        mime = a0.mimeData()
        if mime is None or not mime.hasUrls():
            return
        urls = mime.urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self._load_image(path)

    def _on_res_change(self) -> None:
        if self.lock_btn.isChecked() and self._original_pil is not None:
            sender = self.sender()
            aspect = self._original_pil.width / self._original_pil.height
            if sender == self.w_spin:
                new_w = self.w_spin.value()
                new_h = max(1, int(new_w / aspect))
                self.h_spin.blockSignals(True)
                self.h_spin.setValue(new_h)
                self.h_spin.blockSignals(False)
            elif sender == self.h_spin:
                new_h = self.h_spin.value()
                new_w = max(1, int(new_h * aspect))
                self.w_spin.blockSignals(True)
                self.w_spin.setValue(new_w)
                self.w_spin.blockSignals(False)
        self._schedule_full()

    def _on_lock_toggle(self, checked: bool) -> None:
        self.lock_btn.setText("🔗" if checked else "🔓")
        if checked and self._original_pil is not None:
            aspect = self._original_pil.width / self._original_pil.height
            cur_aspect = self.w_spin.value() / max(self.h_spin.value(), 1)
            if abs(cur_aspect - aspect) > 0.01:
                new_h = max(1, int(self.w_spin.value() / aspect))
                self.h_spin.blockSignals(True)
                self.h_spin.setValue(new_h)
                self.h_spin.blockSignals(False)
                self._schedule_full()

    def _on_algo_change(self) -> None:
        name = self.algo_combo.currentText()
        _, needs_bayer, needs_strength = Ditherer.algorithms()[name]
        self.bayer_label.setVisible(needs_bayer)
        self.bayer_spin.setVisible(needs_bayer)
        self.strength_label.setVisible(needs_strength)
        self.strength_spin.setVisible(needs_strength)
        self._schedule_light()

    def _snap_bayer(self, val: int) -> None:
        valid = [2, 4, 8]
        nearest = min(valid, key=lambda x: (abs(x - val), -x))
        if nearest != val:
            self.bayer_spin.blockSignals(True)
            self.bayer_spin.setValue(nearest)
            self.bayer_spin.blockSignals(False)
        self._schedule_light()

    def _on_curve_mode_toggle(self, checked: bool) -> None:
        mode = "linear" if checked else "smooth"
        self.curve_mode_btn.setText("Linear" if checked else "Smooth")
        self.curve_editor.set_mode(mode)

    # ═══════════════════════════════════════════════════════════════════
    #  Preview rendering  (HiDPI-aware)
    # ═══════════════════════════════════════════════════════════════════

    def _update_original_thumb(self) -> None:
        if self._original_pil is None:
            return
        thumb = self._original_pil.copy()
        thumb.thumbnail((280, 280), Image.Resampling.LANCZOS)
        qimg = self._pil_to_qimage(thumb)
        dpr = self.orig_preview.devicePixelRatioF()
        pix = QPixmap.fromImage(qimg)
        pix.setDevicePixelRatio(dpr)
        self.orig_preview.setPixmap(pix)

    def _show_result(self, bits: np.ndarray) -> None:
        """Display 1-bit result, respecting zoom/fit state and HiDPI.

        Uses nearest-neighbour filtering to avoid interpolation blur on
        HiDPI / Retina displays.
        """
        H, W = bits.shape

        # Build a grayscale QImage (1 byte per pixel: 0 = black, 255 = white)
        buf = (bits * 255).astype(np.uint8)
        qimg = QImage(buf.tobytes(), W, H, W, QImage.Format.Format_Grayscale8)

        # Determine logical display size based on fit/zoom mode
        if self._preview_fit:
            viewport = self.result_scroll.viewport()
            if viewport is not None:
                avail_w = max(viewport.width(), 10)
                avail_h = max(viewport.height(), 10)
            else:
                avail_w, avail_h = 300, 300
            scale = min(avail_w / W, avail_h / H)
        else:
            scale = self._preview_scale

        logical_w = max(1, int(W * scale))
        logical_h = max(1, int(H * scale))

        # Scale pixmap at physical-pixel resolution for crisp HiDPI rendering
        dpr = self.result_preview.devicePixelRatioF()
        phys_w = int(logical_w * dpr)
        phys_h = int(logical_h * dpr)

        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(phys_w, phys_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation)
        scaled.setDevicePixelRatio(dpr)

        self.result_preview.setFixedSize(logical_w, logical_h)
        self.result_preview.setPixmap(scaled)

        mode_text = "compare" if self.compare_btn.isChecked() else self.curve_editor.curve.mode
        self.result_info.setText(f"{W}×{H}  ·  {self.algo_combo.currentText()}  ·  {mode_text}")

    def _refresh_preview(self) -> None:
        """Re-render the current result (used after resize or zoom change)."""
        if self._result is None:
            return
        self._show_result(self._result)

    def _refresh_histogram(self) -> None:
        """Apply current curve to cached lightness and update histogram."""
        if self._lightness is None:
            return
        # Build a throwaway curve with the editor's current state
        curve = BrightnessCurve(mode=self.curve_editor.curve.mode)
        curve.points = list(self.curve_editor.curve.points)
        L_curved = curve.evaluate(self._lightness.ravel()).reshape(self._lightness.shape)
        self.histogram.set_lightness(L_curved)

    # ═══════════════════════════════════════════════════════════════════
    #  Save
    # ═══════════════════════════════════════════════════════════════════

    def _default_save_name(self, suffix: str, ext: str) -> str:
        """Build a default save path from the current image filename."""
        if not self._current_image_path:
            return ""
        base, _ = os.path.splitext(self._current_image_path)
        return f"{base}{suffix}{ext}"

    def _save_png(self) -> None:
        if self._result is None:
            QMessageBox.information(self, "Nothing to save", "No result yet.")
            return
        default = self._default_save_name("_mono", ".png")
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", default, "PNG Image (*.png)")
        if not path:
            return
        try:
            self._write_png(path, self._result)
        except Exception as exc:
            QMessageBox.critical(self, "Save PNG Failed", f"Could not save PNG to:\n{path}\n\n{exc}")
            self.status_label.setText(f"Failed to save PNG: {path}")
            return
        self.status_label.setText(f"Saved PNG: {path}")

    def _save_pbm(self) -> None:
        if self._result is None:
            QMessageBox.information(self, "Nothing to save", "No result yet.")
            return
        default = self._default_save_name("_mono", ".pbm")
        path, _ = QFileDialog.getSaveFileName(self, "Save PBM", default, "PBM Image (*.pbm)")
        if not path:
            return
        try:
            self._write_pbm(path, self._result)
        except Exception as exc:
            QMessageBox.critical(self, "Save PBM Failed", f"Could not save PBM to:\n{path}\n\n{exc}")
            self.status_label.setText(f"Failed to save PBM: {path}")
            return
        self.status_label.setText(f"Saved PBM: {path}")

    @staticmethod
    def _write_png(path: str, bits: np.ndarray) -> None:
        Image.fromarray((bits * 255).astype(np.uint8), mode="L").save(path)

    @staticmethod
    def _write_pbm(path: str, bits: np.ndarray) -> None:
        """Raw P4 PBM  (0 = white, 1 = black, NetPBM convention)."""
        H, W = bits.shape
        row_bytes = (W + 7) // 8
        pbm_bits = 1 - bits.astype(np.uint8)  # invert for PBM
        packed = np.zeros((H, row_bytes), dtype=np.uint8)
        for i in range(W):
            col_byte = i // 8
            col_bit = 7 - (i % 8)
            packed[:, col_byte] |= pbm_bits[:, i] << col_bit
        with open(path, "wb") as f:
            f.write(f"P4\n# pic_mono_dither\n{W} {H}\n".encode())
            f.write(packed.tobytes())

    # ═══════════════════════════════════════════════════════════════════
    #  Utilities
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _pil_to_qimage(pil_img: Image.Image) -> QImage:
        arr = np.asarray(pil_img.convert("RGB"))
        h, w, _ = arr.shape
        return QImage(arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)

    # ── cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, a0) -> None:  # noqa: N802, N803
        self._worker.stop()
        self._worker.wait(2000)
        super().closeEvent(a0)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════


def _parse_curve_value(ctx: click.Context, param: click.Parameter, value: str | None) -> list[tuple[float, float]] | None:
    """Parse a curve string such as '0.5 0.2 0.7 0.9' into (in, out) pairs."""
    if value is None:
        return None
    # Accept space or comma separated values for convenience.
    raw = value.replace(",", " ")
    try:
        nums = [float(x) for x in raw.split()]
    except ValueError as exc:
        raise click.BadParameter("Curve values must all be numeric.") from exc
    if len(nums) % 2 != 0:
        raise click.BadParameter(
            "Curve must have an even number of values (pairs of input L*, output L*).",
            ctx=ctx,
            param=param,
        )
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def _validate_dither(ctx: click.Context, param: click.Parameter, value: str) -> str:
    valid = list(Ditherer.algorithms().keys())
    if value not in valid:
        raise click.BadParameter(f"must be one of: {', '.join(valid)}")
    return value


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """1-bit mono dithering preprocessor.

    Run with no arguments to open the GUI, or use the 'process' subcommand
    for headless batch processing.
    """
    if ctx.invoked_subcommand is None:
        main_gui()


@cli.command("process")
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("--width", "-w", type=int, default=384, show_default=True, help="Output width in pixels.")
@click.option("--height", "-h", type=int, default=None, help="Output height in pixels (defaults to preserving aspect ratio).")
@click.option(
    "--dither", "-d", default="Floyd-Steinberg", show_default=True, callback=_validate_dither, help="Dithering algorithm."
)
@click.option("--bayer-size", type=int, default=4, show_default=True, help="Bayer matrix size (2, 4, or 8).")
@click.option("--strength", "-s", type=float, default=1.0, show_default=True, help="Error diffusion strength (0.0–1.0).")
@click.option(
    "--curve", "-c", "curve_pairs", type=str, callback=_parse_curve_value, help='Brightness curve as "in out in out ..." pairs.'
)
@click.option("--linear", is_flag=True, help="Use linear curve interpolation instead of smooth PCHIP.")
def cli_process(
    input: str,
    output: str,
    width: int,
    height: int | None,
    dither: str,
    bayer_size: int,
    strength: float,
    curve_pairs: list[tuple[float, float]] | None,
    linear: bool,
) -> None:
    """Process an image from the command line."""
    img = Image.open(input).convert("RGB")

    if height is None:
        aspect = img.width / img.height
        height = max(1, int(width / aspect))

    # Ensure valid Bayer size
    if dither == "Bayer Ordered" and bayer_size not in (2, 4, 8):
        bayer_size = min((2, 4, 8), key=lambda x: (abs(x - bayer_size), -x))

    curve_mode = "linear" if linear else "smooth"
    points: list[tuple[float, float]]
    if curve_pairs:
        # Ensure endpoints are present
        points = [(0.0, 0.0)] + curve_pairs + [(1.0, 1.0)]
    else:
        points = [(0.0, 0.0), (1.0, 1.0)]

    bits, _ = run_pipeline(img, width, height, points, curve_mode, dither, bayer_size, strength)

    try:
        if output.lower().endswith(".pbm"):
            MainWindow._write_pbm(output, bits)
        else:
            MainWindow._write_png(output, bits)
    except Exception as exc:
        raise click.ClickException(f"Could not save '{output}': {exc}") from exc

    click.echo(f"Saved {output} ({width}×{height}, {dither})")


def main_gui() -> None:
    """Launch the graphical interface."""
    # Enable HiDPI-per-monitor scaling (per-monitor DPI awareness)
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName("pic_mono_dither")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    cli()
