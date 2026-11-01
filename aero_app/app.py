"""Desktop wind tunnel window."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import QEvent, QSettings, QThread, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from aero_app.meshes import (
    SAMPLE_NAMES,
    areas_of,
    cell_centers_of,
    load_mesh,
    normalize_mesh,
    prepare_for_view,
    sample_mesh,
    surface_normals,
)
from aero_app.solver import (
    QUALITY,
    flow_basis,
    grid_for_mesh,
    in_domain,
    sample_vector_glyphs,
    sample_velocity,
    solid_mask_at,
    trace_streamlines,
)
from aero_app.solver import compute_flow
from aero_app.solver import compute_occupancy

MESH_FILTER = (
    "3D models (*.stl *.obj *.ply *.glb *.gltf *.3mf *.off *.dae *.vtk *.vtp *.vtu *.vtm *.3ds *.fbx *.wrl);;"
    "All files (*)"
)

VECTOR_MODES = {
    "Side slice": "side",
    "Top slice": "top",
    "Front slice": "front",
    "3D cloud": "cloud",
}

CP_BAR = {
    "title": "Cp",
    "color": "#e7eef6",
    "vertical": True,
    "position_x": 0.86,
    "position_y": 0.18,
    "height": 0.64,
    "width": 0.08,
    "title_font_size": 14,
    "label_font_size": 12,
    "fmt": "%.1f",
    "n_labels": 5,
    "shadow": False,
}

SPEED_BAR = {
    "title": "Speed / wind",
    "color": "#e7eef6",
    "vertical": True,
    "position_x": 0.02,
    "position_y": 0.18,
    "height": 0.64,
    "width": 0.08,
    "title_font_size": 14,
    "label_font_size": 12,
    "fmt": "%.1f",
    "n_labels": 5,
    "shadow": False,
}

STYLESHEET = """
QWidget { background: #12171e; color: #e6eef6; font-size: 13px; }
QLabel { background: transparent; }
QWidget#header { background: #0e141b; border-bottom: 1px solid #243140; }
QWidget#side { background: #12171e; }
QLabel#title { font-size: 22px; font-weight: 700; color: #f4fbff; }
QLabel#subtitle { color: #8eb4c9; padding-top: 4px; }
QLabel#caption { color: #93a7ba; font-size: 12px; }
QLabel#readout { font-family: Menlo, monospace; font-size: 16px; color: #f5fbff; }
QLabel#hint { color: #8ea0b3; font-size: 12px; }
QLabel#name { font-size: 14px; font-weight: 650; }
QGroupBox {
    border: 1px solid #2c3c4e;
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: #8fd0e8;
}
QPushButton {
    background: #1d2733;
    border: 1px solid #314255;
    border-radius: 8px;
    padding: 7px 12px;
}
QPushButton:hover { background: #263545; }
QPushButton#primary { background: #147a99; border: none; color: white; font-weight: 650; }
QPushButton#primary:hover { background: #1a93b5; }
QSlider::groove:horizontal { height: 4px; background: #2c3a4a; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 14px; margin: -6px 0; background: #3ddcff; border-radius: 7px;
}
QComboBox, QComboBox QAbstractItemView {
    background: #1d2733;
    color: #e6eef6;
    border: 1px solid #314255;
    border-radius: 6px;
    padding: 4px 8px;
    selection-background-color: #147a99;
}
QCheckBox { spacing: 6px; background: transparent; }
QScrollArea { border: none; background: #12171e; }
QScrollBar:vertical { background: #12171e; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #314255; border-radius: 4px; min-height: 24px; }
QStatusBar { background: #0e141b; color: #9eb0c2; }
QMenu { background: #1d2733; border: 1px solid #314255; }
QMenu::item:selected { background: #147a99; }
QToolTip { background: #243140; color: white; border: 1px solid #3ddcff; }
QMessageBox { background: #1a212b; }
QMessageBox QLabel { color: #e6eef6; }
"""


def build_icon() -> QIcon:
    pix = QPixmap(128, 128)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#0e7490"))
    painter.drawRoundedRect(4, 4, 120, 120, 28, 28)
    pen = QPen(QColor("#d8f7ff"), 7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    path = QPainterPath()
    path.moveTo(26, 80)
    path.cubicTo(48, 78, 62, 48, 104, 42)
    painter.drawPath(path)
    pen.setColor(QColor("#ffe08a"))
    pen.setWidth(5)
    painter.setPen(pen)
    path2 = QPainterPath()
    path2.moveTo(30, 92)
    path2.cubicTo(55, 70, 75, 78, 100, 60)
    painter.drawPath(path2)
    painter.end()
    return QIcon(pix)


def arrow_polydata(points, direction, length, ratio):
    points = np.asarray(points, dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)
    length = np.asarray(length, dtype=np.float32)
    ratio = np.asarray(ratio, dtype=np.float32)
    count = len(points)
    if count == 0:
        return None
    direction = direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8)
    up = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (count, 1))
    parallel = np.abs(direction[:, 2]) > 0.85
    up[parallel] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    side = np.cross(direction, up)
    side = side / np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-8)
    tip = points + direction * length[:, None]
    base = tip - direction * (length * 0.28)[:, None]
    left = base + side * (length * 0.13)[:, None]
    right = base - side * (length * 0.13)[:, None]
    verts = np.empty((count * 6, 3), dtype=np.float32)
    verts[0::6] = points
    verts[1::6] = tip
    verts[2::6] = left
    verts[3::6] = tip
    verts[4::6] = right
    verts[5::6] = tip
    cells = np.empty((count * 3, 3), dtype=np.int64)
    line_ids = np.arange(count * 3, dtype=np.int64)
    cells[:, 0] = 2
    cells[:, 1] = line_ids * 2
    cells[:, 2] = line_ids * 2 + 1
    mesh = pv.PolyData(verts, lines=cells.ravel())
    mesh["speed_ratio"] = np.repeat(ratio, 6)
    return mesh


def paths_to_polydata(paths, speeds, origin, spacing, shape, wind_speed):
    if len(paths) == 0:
        return None
    origin = np.asarray(origin, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    max_corner = origin + spacing * (np.asarray(shape) - 1)
    all_pts = []
    all_ratio = []
    cells = []
    limit = 0.03 * max(float(wind_speed), 1e-6)
    for index in range(paths.shape[0]):
        seg_p = []
        seg_r = []
        for step in range(paths.shape[1]):
            point = paths[index, step]
            if np.any(point < origin - 1e-6) or np.any(point > max_corner + 1e-6):
                break
            if step > 3 and speeds[index, step] < limit:
                break
            seg_p.append(point)
            seg_r.append(speeds[index, step] / max(float(wind_speed), 1e-6))
        if len(seg_p) < 4:
            continue
        start = len(all_pts)
        all_pts.extend(seg_p)
        all_ratio.extend(seg_r)
        cells.append(len(seg_p))
        cells.extend(range(start, start + len(seg_p)))
    if len(all_pts) < 2:
        return None
    mesh = pv.PolyData(np.asarray(all_pts, dtype=np.float32), lines=np.asarray(cells, dtype=np.int64))
    mesh["speed_ratio"] = np.asarray(all_ratio, dtype=np.float32)
    return mesh


class FlowWorker(QThread):
    finished_ok = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, job_id: int, params: dict):
        super().__init__()
        self.job_id = job_id
        self.params = params

    def run(self):
        try:
            result = compute_flow(**self.params)
            self.finished_ok.emit(self.job_id, result)
        except Exception:
            self.failed.emit(self.job_id, traceback.format_exc())


class FloatSlider(QWidget):
    changed = Signal(float)

    def __init__(self, label, low, high, value, step, suffix="", decimals=2):
        super().__init__()
        self.low = float(low)
        self.high = float(high)
        self.step = float(step)
        self.decimals = int(decimals)
        self.suffix = suffix
        self._name = QLabel(label)
        self._name.setFixedWidth(118)
        self._value = QLabel()
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._value.setFixedWidth(72)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        steps = max(1, int(round((self.high - self.low) / self.step)))
        self._slider.setRange(0, steps)
        self._slider.setValue(self._to_slider(value))
        self._refresh_text()
        self._slider.valueChanged.connect(self._on_change)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._name)
        row.addWidget(self._slider, 1)
        row.addWidget(self._value)

    def _to_slider(self, value: float) -> int:
        steps = int(round((float(value) - self.low) / self.step))
        return max(self._slider.minimum(), min(self._slider.maximum(), steps))

    def value(self) -> float:
        return self.low + self._slider.value() * self.step

    def set_value(self, value: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(self._to_slider(value))
        self._slider.blockSignals(False)
        self._refresh_text()

    def _refresh_text(self) -> None:
        text = f"{self.value():.{self.decimals}f}{self.suffix}"
        self._value.setText(text)

    def _on_change(self) -> None:
        self._refresh_text()
        self.changed.emit(self.value())


class MainWindow(QMainWindow):
    def __init__(self, start_path: str | None = None):
        super().__init__()
        self.setWindowTitle("Aero Wind Tunnel")
        self.resize(1440, 880)
        self.setMinimumSize(1100, 700)
        self.setAcceptDrops(True)
        self._settings = QSettings()
        self._ignore_settings = os.environ.get("AERO_RESET") == "1"
        self._source = None
        self._mesh = None
        self._model_name = "Wing"
        self._model_path = ""
        self._last_dir = str(Path.home())
        self._occ = None
        self._vel = None
        self._solid_grid = None
        self._origin = None
        self._spacing = None
        self._wind_hat = np.array([1.0, 0.0, 0.0])
        self._wind_speed = 8.0
        self._cp = None
        self._domain_key = None
        self._worker = None
        self._job = 0
        self._pending = False
        self._closing = False
        self._paused = False
        self._shot_queued = False
        self._time = 0.0
        self._rng = np.random.default_rng(7)
        self.pos = None
        self.trail = None
        self.trail_speed = None
        self.age = None
        self._smoke = None
        self._heads_mesh = None
        pv.set_plot_theme("dark")
        self._build_ui()
        self._build_menu()
        self._load_settings()
        self.installEventFilter(self)
        self.view.installEventFilter(self)
        if start_path:
            self.load_path(start_path)
        else:
            self._load_initial()
        if os.environ.get("AERO_SHOT"):
            QTimer.singleShot(40000, self.close)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self._scroller(self._build_left()))
        self.splitter.addWidget(self._build_view())
        self.splitter.addWidget(self._scroller(self._build_right()))
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setSizes([300, 760, 330])
        root.addWidget(self.splitter, 1)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Open a model, or try a sample shape.")
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._kick_solve)
        self._tick = QTimer(self)
        self._tick.setInterval(33)
        self._tick.timeout.connect(self._advance_smoke)
        self._tick.start()

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("header")
        header.setFixedHeight(58)
        row = QHBoxLayout(header)
        row.setContentsMargins(16, 8, 16, 8)
        title = QLabel("Aero")
        title.setObjectName("title")
        subtitle = QLabel("Wind tunnel")
        subtitle.setObjectName("subtitle")
        row.addWidget(title)
        row.addWidget(subtitle)
        row.addSpacing(16)
        open_btn = QPushButton("Open 3D model")
        open_btn.setObjectName("primary")
        open_btn.clicked.connect(self.open_model)
        row.addWidget(open_btn)
        self.sample_btn = QPushButton("Samples")
        menu = QMenu(self.sample_btn)
        for name in SAMPLE_NAMES:
            action = menu.addAction(name)
            action.triggered.connect(lambda checked=False, picked=name: self.load_sample(picked))
        self.sample_btn.setMenu(menu)
        row.addWidget(self.sample_btn)
        row.addStretch(1)
        self.pause_btn = QPushButton("Pause smoke")
        self.pause_btn.clicked.connect(self.toggle_pause)
        row.addWidget(self.pause_btn)
        reset_view = QPushButton("Reset view")
        reset_view.clicked.connect(self._frame_camera)
        row.addWidget(reset_view)
        return header

    def _scroller(self, inner: QWidget) -> QScrollArea:
        inner.setObjectName("side")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        scroll.setMinimumWidth(280)
        return scroll

    def _group(self, title: str):
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setSpacing(8)
        return box, layout

    def _build_left(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        model_box, model_layout = self._group("Model")
        self.name_label = QLabel("Wing")
        self.name_label.setObjectName("name")
        self.name_label.setWordWrap(True)
        self.tri_label = QLabel("—")
        self.scale_label = QLabel("Longest side is scaled to 1 m.")
        self.scale_label.setObjectName("hint")
        self.scale_label.setWordWrap(True)
        model_layout.addWidget(self.name_label)
        model_layout.addWidget(self.tri_label)
        model_layout.addWidget(self.scale_label)
        up_row = QHBoxLayout()
        up_row.addWidget(QLabel("Up axis"))
        self.up_axis = QComboBox()
        self.up_axis.addItems(["Z up", "Y up"])
        self.up_axis.setToolTip("If the model is lying on its side, switch this to Y.")
        self.up_axis.currentIndexChanged.connect(lambda _index: self._rebuild(True))
        up_row.addWidget(self.up_axis, 1)
        model_layout.addLayout(up_row)
        self.see_slider = FloatSlider("See-through", 0.25, 1.0, 1.0, 0.05, decimals=2)
        self.see_slider.changed.connect(self._on_see_through)
        model_layout.addWidget(self.see_slider)
        layout.addWidget(model_box)

        display_box, display_layout = self._group("Display")
        grid = QGridLayout()
        self.chk_model = QCheckBox("Model")
        self.chk_pressure = QCheckBox("Pressure colors")
        self.chk_smoke = QCheckBox("Smoke")
        self.chk_particles = QCheckBox("Particles")
        self.chk_vectors = QCheckBox("Vector arrows")
        self.chk_streams = QCheckBox("Streamlines")
        self.chk_tunnel = QCheckBox("Tunnel box")
        self.chk_floor = QCheckBox("Floor grid")
        checks = [
            self.chk_model,
            self.chk_pressure,
            self.chk_smoke,
            self.chk_particles,
            self.chk_vectors,
            self.chk_streams,
            self.chk_tunnel,
            self.chk_floor,
        ]
        for box in checks:
            box.setChecked(True)
        self.chk_streams.setChecked(True)
        for index, box in enumerate(checks):
            grid.addWidget(box, index // 2, index % 2)
        display_layout.addLayout(grid)
        self.chk_model.toggled.connect(self._apply_visibility)
        self.chk_pressure.toggled.connect(self._show_model)
        self.chk_smoke.toggled.connect(self._apply_visibility)
        self.chk_particles.toggled.connect(self._apply_visibility)
        self.chk_vectors.toggled.connect(self._on_vectors_toggle)
        self.chk_streams.toggled.connect(self._on_streams_toggle)
        self.chk_tunnel.toggled.connect(self._apply_visibility)
        self.chk_floor.toggled.connect(self._apply_visibility)
        layout.addWidget(display_box)

        numbers, numbers_layout = self._group("Estimates")
        form = QGridLayout()
        self.readouts = {}
        for row, (caption, key) in enumerate(
            (("Lift", "lift"), ("Drag", "drag"), ("Side", "side"), ("Cl", "cl"), ("Cd", "cd"), ("Peak speed", "peak"))
        ):
            cap = QLabel(caption)
            cap.setObjectName("caption")
            val = QLabel("—")
            val.setObjectName("readout")
            form.addWidget(cap, row, 0)
            form.addWidget(val, row, 1)
            self.readouts[key] = val
        numbers_layout.addLayout(form)
        hint = QLabel(
            "Blue on the model is lower pressure (suction). Red is higher pressure. "
            "Forces are estimates on the scaled 1 m model. This is a preview, not a certified wind-tunnel test."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        numbers_layout.addWidget(hint)
        layout.addWidget(numbers)
        layout.addStretch(1)
        return page

    def _build_right(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)

        wind_box, wind_layout = self._group("Wind")
        self.speed_slider = FloatSlider("Speed", 1.0, 40.0, 8.0, 0.5, " m/s", 1)
        self.pitch_slider = FloatSlider("Pitch / AoA", -30.0, 30.0, 6.0, 0.5, "°", 1)
        self.yaw_slider = FloatSlider("Yaw", -180.0, 180.0, 0.0, 1.0, "°", 0)
        self.speed_slider.setToolTip("Wind speed. Force grows with speed squared. The streamline shape stays similar.")
        self.pitch_slider.setToolTip("Angle of attack. Positive pitch lifts a normal wing.")
        self.yaw_slider.setToolTip("Sideslip. Wind comes from the side.")
        for slider in (self.speed_slider, self.pitch_slider, self.yaw_slider):
            slider.changed.connect(lambda _value: self._schedule())
            wind_layout.addWidget(slider)
        layout.addWidget(wind_box)

        flow_box, flow_layout = self._group("Flow")
        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Detail"))
        self.quality = QComboBox()
        self.quality.addItems(list(QUALITY))
        self.quality.setCurrentText("Medium")
        self.quality.currentIndexChanged.connect(self._on_quality)
        quality_row.addWidget(self.quality, 1)
        flow_layout.addLayout(quality_row)
        self.wake_slider = FloatSlider("Wake", 0.0, 100.0, 45.0, 1.0, "%", 0)
        self.wake_slider.setToolTip("Slows the smoke behind the model and adds drag.")
        self.wake_slider.changed.connect(lambda _value: self._schedule())
        self.rho_slider = FloatSlider("Air density", 0.80, 1.50, 1.225, 0.005, "", 3)
        self.rho_slider.setToolTip("Used only for the force numbers. Sea-level air is about 1.225.")
        self.rho_slider.changed.connect(lambda _value: self._schedule())
        flow_layout.addWidget(self.wake_slider)
        flow_layout.addWidget(self.rho_slider)
        layout.addWidget(flow_box)

        smoke_box, smoke_layout = self._group("Smoke and particles")
        self.count_slider = FloatSlider("Particles", 16, 800, 72, 8, "", 0)
        self.trail_slider = FloatSlider("Trail", 4, 28, 18, 1, "", 0)
        self.point_slider = FloatSlider("Size", 2, 12, 4, 1, "", 0)
        self.opacity_slider = FloatSlider("Opacity", 0.15, 1.0, 0.92, 0.05, "", 2)
        self.turb_slider = FloatSlider("Turbulence", 0.0, 1.0, 0.08, 0.05, "", 2)
        self.play_slider = FloatSlider("Playback", 0.05, 1.0, 0.22, 0.05, "×", 2)
        self.count_slider.changed.connect(self._resize_particles)
        self.trail_slider.changed.connect(self._resize_particles)
        self.point_slider.changed.connect(self._on_point_size)
        for slider in (
            self.count_slider,
            self.trail_slider,
            self.point_slider,
            self.opacity_slider,
            self.turb_slider,
            self.play_slider,
        ):
            smoke_layout.addWidget(slider)
        pattern_row = QHBoxLayout()
        pattern_row.addWidget(QLabel("Pattern"))
        self.smoke_pattern = QComboBox()
        self.smoke_pattern.addItems(["Sheet", "Full tunnel"])
        self.smoke_pattern.setToolTip("Sheet draws smoke lines in one slice. Full tunnel fills the box.")
        self.smoke_pattern.currentIndexChanged.connect(self._reset_smoke)
        pattern_row.addWidget(self.smoke_pattern, 1)
        smoke_layout.addLayout(pattern_row)
        layout.addWidget(smoke_box)

        vector_box, vector_layout = self._group("Vector field")
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Arrows"))
        self.vector_mode = QComboBox()
        self.vector_mode.addItems(list(VECTOR_MODES))
        self.vector_mode.currentIndexChanged.connect(self._on_vectors_only)
        mode_row.addWidget(self.vector_mode, 1)
        vector_layout.addLayout(mode_row)
        self.stride_slider = FloatSlider("Spacing", 1, 6, 3, 1, "", 0)
        self.stride_slider.setToolTip("Higher spacing draws fewer arrows.")
        self.stride_slider.changed.connect(self._on_vectors_only)
        vector_layout.addWidget(self.stride_slider)
        layout.addWidget(vector_box)

        recompute = QPushButton("Recompute airflow")
        recompute.clicked.connect(lambda: self._schedule(10))
        reset_smoke = QPushButton("Reset smoke")
        reset_smoke.clicked.connect(self._reset_smoke)
        reset_settings = QPushButton("Reset settings")
        reset_settings.clicked.connect(self._reset_settings)
        layout.addWidget(recompute)
        layout.addWidget(reset_smoke)
        layout.addWidget(reset_settings)
        layout.addStretch(1)
        return page

    def _build_view(self) -> QWidget:
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = QtInteractor(frame, auto_update=False, multi_samples=4)
        layout.addWidget(self.view)
        self.view.set_background("#0b1016", top="#21405c")
        try:
            self.view.enable_anti_aliasing("fxaa")
        except Exception:
            pass
        try:
            self.view.add_axes(line_width=2, color="#d5e2ef")
        except Exception:
            try:
                self.view.add_axes()
            except Exception:
                pass
        self.view.add_text(
            "Drag to orbit     Scroll to zoom     Right-drag to pan",
            position="upper_left",
            font_size=10,
            color="#9eb4c9",
            name="help",
        )
        return frame

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_action = QAction("Open 3D model…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        open_action.triggered.connect(self.open_model)
        file_menu.addAction(open_action)
        samples = file_menu.addMenu("Sample shapes")
        for name in SAMPLE_NAMES:
            action = QAction(name, self)
            action.triggered.connect(lambda checked=False, picked=name: self.load_sample(picked))
            samples.addAction(action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("View")
        reset = QAction("Reset camera", self)
        reset.setShortcut(QKeySequence("R"))
        reset.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        reset.triggered.connect(self._frame_camera)
        view_menu.addAction(reset)

        sim = self.menuBar().addMenu("Simulation")
        recompute = QAction("Recompute airflow", self)
        recompute.triggered.connect(lambda: self._schedule(10))
        sim.addAction(recompute)
        pause = QAction("Pause or play smoke", self)
        pause.setShortcut(QKeySequence("Space"))
        pause.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        pause.triggered.connect(self.toggle_pause)
        sim.addAction(pause)
        sim.addAction("Reset smoke", self._reset_smoke)

        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("How to use", self._show_help)
        help_menu.addAction("About Aero", self._show_about)

    def _show_help(self) -> None:
        QMessageBox.information(
            self,
            "How to use Aero",
            "1. Open a 3D model (STL, OBJ, PLY, GLB, and similar) or pick a sample.\n"
            "   You can also drop a file on the window.\n\n"
            "2. Drag in the view to look around. Scroll to zoom. Right-drag to pan.\n\n"
            "3. Pitch is the angle of attack. Yaw is the sideslip.\n"
            "   Speed changes the forces. Wake slows the smoke behind the model and adds drag.\n\n"
            "4. Vector arrows can be a side, top, or front slice, or a 3D cloud.\n\n"
            "5. If a model is lying on its side, set Up axis to Y.\n\n"
            "Blue on the surface means lower pressure. Red means higher pressure.",
        )

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "About Aero",
            "Aero Wind Tunnel\n\n"
            "A desktop preview of airflow around a 3D model, with smoke, particles, "
            "and a vector field.\n\n"
            "The flow is potential flow plus a simple wake. It is useful for comparing "
            "shape and angle. It is not a certified CFD result.\n\n"
            "Your model is scaled so its longest side is 1 meter before the forces are estimated.",
        )

    def _saved(self, key: str, default):
        if self._ignore_settings:
            return default
        return self._settings.value(key, default)

    def _saved_float(self, key: str, default: float) -> float:
        try:
            return float(self._saved(key, default))
        except (TypeError, ValueError):
            return default

    def _saved_bool(self, key: str, default: bool) -> bool:
        raw = self._saved(key, default)
        if isinstance(raw, str):
            return raw.lower() in {"1", "true", "yes"}
        return bool(raw)

    def _load_settings(self) -> None:
        widgets = [
            self.up_axis,
            self.quality,
            self.vector_mode,
            self.smoke_pattern,
            self.chk_model,
            self.chk_pressure,
            self.chk_smoke,
            self.chk_particles,
            self.chk_vectors,
            self.chk_streams,
            self.chk_tunnel,
            self.chk_floor,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        self.speed_slider.set_value(self._saved_float("speed", 8.0))
        self.pitch_slider.set_value(self._saved_float("pitch", 6.0))
        self.yaw_slider.set_value(self._saved_float("yaw", 0.0))
        self.wake_slider.set_value(self._saved_float("wake", 45.0))
        self.rho_slider.set_value(self._saved_float("rho", 1.225))
        self.count_slider.set_value(self._saved_float("particles", 72))
        self.trail_slider.set_value(self._saved_float("trail", 18))
        self.point_slider.set_value(self._saved_float("point", 4))
        self.opacity_slider.set_value(self._saved_float("opacity", 0.92))
        self.turb_slider.set_value(self._saved_float("turbulence", 0.08))
        self.play_slider.set_value(self._saved_float("playback", 0.22))
        self.see_slider.set_value(self._saved_float("see", 1.0))
        self.stride_slider.set_value(self._saved_float("stride", 3))
        quality = str(self._saved("quality", "Medium"))
        if quality in QUALITY:
            self.quality.setCurrentText(quality)
        mode = str(self._saved("vector_mode", "Side slice"))
        if mode in VECTOR_MODES:
            self.vector_mode.setCurrentText(mode)
        pattern = str(self._saved("smoke_pattern", "Sheet"))
        if pattern in {"Sheet", "Full tunnel"}:
            self.smoke_pattern.setCurrentText(pattern)
        up = str(self._saved("up", "Z up"))
        if up in {"Z up", "Y up"}:
            self.up_axis.setCurrentText(up)
        self.chk_model.setChecked(self._saved_bool("show_model", True))
        self.chk_pressure.setChecked(self._saved_bool("show_pressure", True))
        self.chk_smoke.setChecked(self._saved_bool("show_smoke", True))
        self.chk_particles.setChecked(self._saved_bool("show_particles", True))
        self.chk_vectors.setChecked(self._saved_bool("show_vectors", True))
        self.chk_streams.setChecked(self._saved_bool("show_streams", True))
        self.chk_tunnel.setChecked(self._saved_bool("show_tunnel", True))
        self.chk_floor.setChecked(self._saved_bool("show_floor", True))
        self._last_dir = str(self._saved("last_dir", str(Path.home())))
        for widget in widgets:
            widget.blockSignals(False)
        if not self._ignore_settings:
            geometry = self._settings.value("geometry")
            if geometry is not None:
                self.restoreGeometry(geometry)

    def _save_settings(self) -> None:
        if self._ignore_settings:
            return
        self._settings.setValue("speed", self.speed_slider.value())
        self._settings.setValue("pitch", self.pitch_slider.value())
        self._settings.setValue("yaw", self.yaw_slider.value())
        self._settings.setValue("wake", self.wake_slider.value())
        self._settings.setValue("rho", self.rho_slider.value())
        self._settings.setValue("particles", self.count_slider.value())
        self._settings.setValue("trail", self.trail_slider.value())
        self._settings.setValue("point", self.point_slider.value())
        self._settings.setValue("opacity", self.opacity_slider.value())
        self._settings.setValue("turbulence", self.turb_slider.value())
        self._settings.setValue("playback", self.play_slider.value())
        self._settings.setValue("smoke_pattern", self.smoke_pattern.currentText())
        self._settings.setValue("see", self.see_slider.value())
        self._settings.setValue("stride", self.stride_slider.value())
        self._settings.setValue("quality", self.quality.currentText())
        self._settings.setValue("vector_mode", self.vector_mode.currentText())
        self._settings.setValue("up", self.up_axis.currentText())
        self._settings.setValue("show_model", self.chk_model.isChecked())
        self._settings.setValue("show_pressure", self.chk_pressure.isChecked())
        self._settings.setValue("show_smoke", self.chk_smoke.isChecked())
        self._settings.setValue("show_particles", self.chk_particles.isChecked())
        self._settings.setValue("show_vectors", self.chk_vectors.isChecked())
        self._settings.setValue("show_streams", self.chk_streams.isChecked())
        self._settings.setValue("show_tunnel", self.chk_tunnel.isChecked())
        self._settings.setValue("show_floor", self.chk_floor.isChecked())
        self._settings.setValue("last_dir", self._last_dir)
        self._settings.setValue("last_file", self._model_path)
        self._settings.setValue("geometry", self.saveGeometry())

    def _load_initial(self) -> None:
        last = str(self._saved("last_file", "") or "")
        if last and Path(last).is_file():
            self.load_path(last)
        else:
            self.load_sample("Wing")

    def open_model(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(self, "Open a 3D model", self._last_dir, MESH_FILTER)
        if path:
            self._last_dir = str(Path(path).parent)
            self.load_path(path)

    def load_path(self, path: str) -> None:
        try:
            source = load_mesh(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not open model", f"{path}\n\n{exc}")
            if self._source is None:
                self.load_sample("Wing")
            return
        self._source = source
        self._model_name = Path(path).name
        self._model_path = str(path)
        self._rebuild(True)

    def load_sample(self, name: str) -> None:
        try:
            self._source = sample_mesh(name)
        except Exception as exc:
            QMessageBox.warning(self, "Could not build sample", str(exc))
            return
        self._model_name = name
        self._model_path = ""
        self._rebuild(True)

    def _rebuild(self, reset_camera: bool) -> None:
        if self._source is None:
            return
        try:
            mesh = self._source.copy(deep=True)
            if self.up_axis.currentText().startswith("Y"):
                mesh.rotate_x(90, inplace=True)
            mesh = normalize_mesh(mesh, 1.0)
            mesh = prepare_for_view(mesh)
        except Exception as exc:
            QMessageBox.warning(self, "Could not use this model", str(exc))
            return
        self._mesh = mesh
        self._occ = None
        self._cp = None
        self.name_label.setText(self._model_name)
        self.tri_label.setText(f"{mesh.n_cells:,} triangles")
        self._show_model()
        self._show_tunnel()
        if reset_camera:
            self._frame_camera()
        self._schedule(40)

    def _frame_camera(self) -> None:
        # Mostly side-on, tipped just enough to see the upper surface and the center slice.
        self.view.view_xz()
        self.view.set_viewup((0.0, 0.0, 1.0))
        self.view.reset_camera()
        self.view.camera.Azimuth(-28)
        self.view.camera.Elevation(16)
        self.view.reset_camera_clipping_range()
        self.view.camera.zoom(1.4)
        self.view.render()

    def _quality_pair(self):
        name = self.quality.currentText()
        return QUALITY.get(name, QUALITY["Medium"])

    def _wind_vector(self) -> np.ndarray:
        pitch = np.deg2rad(self.pitch_slider.value())
        yaw = np.deg2rad(self.yaw_slider.value())
        direction = np.array(
            [
                np.cos(pitch) * np.cos(yaw),
                np.cos(pitch) * np.sin(yaw),
                np.sin(pitch),
            ],
            dtype=np.float64,
        )
        direction /= np.linalg.norm(direction)
        return direction * float(self.speed_slider.value())

    def _schedule(self, delay: int = 160) -> None:
        if self._closing or self._mesh is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self._pending = True
            return
        self._debounce.start(delay)

    def _on_quality(self) -> None:
        self._occ = None
        self._schedule(30)

    def _ensure_occupancy(self) -> None:
        n, _iterations = self._quality_pair()
        key = (id(self._mesh), n)
        if self._occ is not None and self._occ.get("key") == key:
            return
        self.statusBar().showMessage("Reading the model shape…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            started = os.times()
            occ = compute_occupancy(self._mesh, n)
            occ["key"] = key
            self._occ = occ
            if os.environ.get("AERO_PRINT") == "1":
                elapsed = os.times()[0] + os.times()[4]
                print(f"occupancy solids {occ['solid'].mean():.2%} cpu {elapsed - (started[0] + started[4]):.2f}", flush=True)
        finally:
            QApplication.restoreOverrideCursor()

    def _kick_solve(self) -> None:
        if self._closing or self._mesh is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self._pending = True
            return
        try:
            self._ensure_occupancy()
            point_n, cell_n = surface_normals(self._mesh)
            _n, iterations = self._quality_pair()
            params = {
                "solid": self._occ["solid"],
                "origin": np.array(self._occ["origin"], dtype=np.float64, copy=True),
                "spacing": np.array(self._occ["spacing"], dtype=np.float64, copy=True),
                "wind": self._wind_vector(),
                "wake": self.wake_slider.value() / 100.0,
                "rho": self.rho_slider.value(),
                "points": np.array(self._mesh.points, dtype=np.float64, copy=True),
                "point_normals": point_n,
                "cell_centers": cell_centers_of(self._mesh),
                "cell_normals": cell_n,
                "cell_areas": areas_of(self._mesh),
                "iterations": iterations,
            }
        except Exception:
            traceback.print_exc()
            self.statusBar().showMessage("Could not read this model.")
            if os.environ.get("AERO_SHOT"):
                QTimer.singleShot(400, self.close)
            return
        self._job += 1
        self.statusBar().showMessage("Solving airflow…")
        self._worker = FlowWorker(self._job, params)
        self._worker.finished_ok.connect(self._on_flow)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _after_job(self) -> None:
        if self._closing:
            return
        if self._pending:
            self._pending = False
            self._kick_solve()

    def _on_fail(self, job_id: int, text: str) -> None:
        print(text, flush=True)
        if job_id == self._job and not self._closing:
            self.statusBar().showMessage("Could not solve the airflow.")
            if not os.environ.get("AERO_SHOT"):
                QMessageBox.warning(self, "Airflow failed", text[-1800:])
        self._after_job()
        if os.environ.get("AERO_SHOT"):
            QTimer.singleShot(400, self.close)

    def _on_flow(self, job_id: int, result: dict) -> None:
        if job_id != self._job or self._closing:
            self._after_job()
            return
        try:
            self._apply_flow(result)
        except Exception:
            traceback.print_exc()
            self.statusBar().showMessage("Could not display the airflow.")
        self._after_job()

    def _apply_flow(self, result: dict) -> None:
        self._vel = result["velocity"]
        self._solid_grid = result["solid"]
        self._origin = np.asarray(result["origin"], dtype=np.float64)
        self._spacing = np.asarray(result["spacing"], dtype=np.float64)
        self._wind_hat = np.asarray(result["wind_hat"], dtype=np.float64)
        self._wind_speed = float(result["wind_speed"])
        self._cp = np.asarray(result["cp"], dtype=np.float64)
        if len(self._cp) == self._mesh.n_points:
            self._mesh.point_data["Cp"] = self._cp
        self._show_model()
        self._show_tunnel()
        self._show_wind_arrow()
        self._show_vectors()
        self._show_streams()
        domain_key = (
            self._solid_grid.shape,
            round(float(self._origin[0]), 4),
            round(float(self._spacing[0]), 5),
            int(self.count_slider.value()),
            int(self.trail_slider.value()),
        )
        if self.pos is None or self.trail is None or self._domain_key != domain_key:
            self._allocate_particles(int(self.count_slider.value()), int(self.trail_slider.value()))
            self._rebuild_particle_actors()
            self._domain_key = domain_key
        self._upload_smoke()
        self._show_report(result["report"])
        self._set_hud()
        self._apply_visibility()
        self.view.render()
        self.statusBar().showMessage("Ready — drag to look around, scroll to zoom.")
        if os.environ.get("AERO_PRINT") == "1":
            print("REPORT", {key: round(float(value), 4) for key, value in result["report"].items()}, flush=True)
        self._maybe_shot()

    def _remove(self, name: str) -> None:
        try:
            self.view.remove_actor(name)
        except Exception:
            pass

    def _remove_bar(self, title: str) -> None:
        try:
            self.view.remove_scalar_bar(title)
        except Exception:
            pass

    def _cp_limits(self) -> list:
        values = np.asarray(self._mesh.point_data["Cp"], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < 8:
            return [-1.0, 1.0]
        low, high = np.percentile(values, [6, 94])
        span = max(float(high - low), 0.35)
        mid = 0.5 * float(high + low)
        return [mid - span * 0.62, mid + span * 0.62]

    def _show_model(self) -> None:
        if self._mesh is None:
            return
        self._remove_bar("Cp")
        common = dict(
            name="model",
            smooth_shading=True,
            specular=0.45,
            specular_power=24,
            ambient=0.32,
            diffuse=0.75,
            opacity=float(self.see_slider.value()),
            reset_camera=False,
            show_edges=False,
        )
        use_cp = self.chk_pressure.isChecked() and "Cp" in self._mesh.point_data
        if use_cp:
            try:
                self.view.add_mesh(
                    self._mesh,
                    scalars="Cp",
                    cmap="coolwarm",
                    clim=self._cp_limits(),
                    scalar_bar_args=CP_BAR,
                    **common,
                )
            except Exception:
                self.view.add_mesh(self._mesh, color="#d5dee8", **common)
        else:
            self.view.add_mesh(self._mesh, color="#d5dee8", **common)
        self._apply_visibility()

    def _show_tunnel(self) -> None:
        if self._mesh is None:
            return
        n, _iterations = self._quality_pair()
        origin, spacing = grid_for_mesh(self._mesh, n)
        max_corner = origin + spacing * (n - 1)
        bounds = (
            float(origin[0]),
            float(max_corner[0]),
            float(origin[1]),
            float(max_corner[1]),
            float(origin[2]),
            float(max_corner[2]),
        )
        box = pv.Box(bounds=bounds)
        self.view.add_mesh(
            box,
            name="tunnel",
            style="wireframe",
            color="#7fd3ea",
            opacity=0.35,
            line_width=1.5,
            reset_camera=False,
        )
        floor = pv.Plane(
            center=(
                (origin[0] + max_corner[0]) / 2.0,
                (origin[1] + max_corner[1]) / 2.0,
                float(origin[2]),
            ),
            direction=(0, 0, 1),
            i_size=float(max_corner[0] - origin[0]),
            j_size=float(max_corner[1] - origin[1]),
            i_resolution=16,
            j_resolution=16,
        )
        self.view.add_mesh(
            floor,
            name="floor",
            style="wireframe",
            color="#31475f",
            opacity=0.9,
            reset_camera=False,
        )
        self._apply_visibility()

    def _show_wind_arrow(self) -> None:
        if self._origin is None:
            return
        try:
            half = self._spacing * (np.array(self._solid_grid.shape) - 1) * 0.5
            center = self._origin + half
            _right, up = flow_basis(self._wind_hat)
            reach = float(np.min(half))
            start = center - self._wind_hat * (reach * 0.72) + up * (reach * 0.62)
            arrow = pv.Arrow(
                start=tuple(float(v) for v in start),
                direction=tuple(float(v) for v in self._wind_hat),
                scale=reach * 0.5,
                tip_length=0.28,
                tip_radius=0.07,
                shaft_radius=0.025,
                tip_resolution=16,
                shaft_resolution=16,
            )
            self.view.add_mesh(arrow, name="wind_arrow", color="#8fe7ff", smooth_shading=True, reset_camera=False)
        except Exception:
            traceback.print_exc()

    def _add_colored_lines(self, mesh, name: str, show_bar: bool, bar: dict, tubes: bool, width: float) -> None:
        kwargs = dict(
            name=name,
            scalars="speed_ratio",
            clim=[0.0, 1.7],
            line_width=width,
            render_lines_as_tubes=tubes,
            reset_camera=False,
        )
        if show_bar:
            kwargs["scalar_bar_args"] = bar
        else:
            kwargs["show_scalar_bar"] = False
        try:
            self.view.add_mesh(mesh, cmap="turbo", **kwargs)
        except Exception:
            self.view.add_mesh(mesh, cmap="viridis", **kwargs)

    def _show_vectors(self) -> None:
        self._remove_bar("Speed / wind")
        if self._vel is None or not self.chk_vectors.isChecked():
            self._remove("vectors")
            return
        mode = VECTOR_MODES.get(self.vector_mode.currentText(), "side")
        points, direction, length, ratio = sample_vector_glyphs(
            self._vel,
            self._solid_grid,
            self._origin,
            self._spacing,
            self._wind_speed,
            mode,
            int(self.stride_slider.value()),
        )
        mesh = arrow_polydata(points, direction, length, ratio)
        if mesh is None:
            self._remove("vectors")
            return
        self._add_colored_lines(mesh, "vectors", True, SPEED_BAR, len(points) <= 900, 2.0)

    def _show_streams(self) -> None:
        if self._vel is None or not self.chk_streams.isChecked():
            self._remove("streams")
            return
        wind = self._wind_hat * self._wind_speed
        paths, speeds = trace_streamlines(self._vel, self._solid_grid, self._origin, self._spacing, wind)
        mesh = paths_to_polydata(paths, speeds, self._origin, self._spacing, self._solid_grid.shape, self._wind_speed)
        if mesh is None:
            self._remove("streams")
            return
        self._add_colored_lines(mesh, "streams", False, SPEED_BAR, True, 3.0)

    def _on_vectors_toggle(self) -> None:
        self._show_vectors()
        self._apply_visibility()
        self.view.render()

    def _on_streams_toggle(self) -> None:
        self._show_streams()
        self._apply_visibility()
        self.view.render()

    def _on_vectors_only(self, _value=None) -> None:
        if self._vel is None:
            return
        self._show_vectors()
        self._apply_visibility()
        self.view.render()

    def _on_see_through(self, _value=None) -> None:
        try:
            self.view.actors["model"].GetProperty().SetOpacity(float(self.see_slider.value()))
            self.view.render()
        except Exception:
            pass

    def _on_point_size(self, _value=None) -> None:
        size = float(self.point_slider.value())
        try:
            self.view.actors["smoke"].GetProperty().SetLineWidth(max(1.2, size * 0.7))
            self.view.actors["heads"].GetProperty().SetPointSize(size * 1.6)
            self.view.render()
        except Exception:
            pass

    def _vis(self, name: str, visible: bool) -> None:
        try:
            self.view.actors[name].SetVisibility(bool(visible))
        except Exception:
            pass

    def _apply_visibility(self) -> None:
        self._vis("model", self.chk_model.isChecked())
        self._vis("vectors", self.chk_vectors.isChecked())
        self._vis("streams", self.chk_streams.isChecked())
        self._vis("tunnel", self.chk_tunnel.isChecked())
        self._vis("floor", self.chk_floor.isChecked())
        self._vis("smoke", self.chk_smoke.isChecked())
        self._vis("heads", self.chk_particles.isChecked())
        self._vis("wind_arrow", True)

    def _show_report(self, report: dict) -> None:
        def nice(value: float, digits: int) -> str:
            if not np.isfinite(value):
                return "—"
            return f"{value:.{digits}f}"

        self.readouts["lift"].setText(nice(report["lift"], 2) + " N")
        self.readouts["drag"].setText(nice(report["drag"], 2) + " N")
        self.readouts["side"].setText(nice(report["side"], 2) + " N")
        self.readouts["cl"].setText(nice(report["cl"], 3))
        self.readouts["cd"].setText(nice(report["cd"], 3))
        self.readouts["peak"].setText(nice(report["peak"], 2) + "×")

    def _set_hud(self) -> None:
        text = (
            f"Wind {self.speed_slider.value():.1f} m/s"
            f"    AoA {self.pitch_slider.value():.1f}°"
            f"    Yaw {self.yaw_slider.value():.0f}°"
        )
        try:
            self.view.remove_actor("hud")
        except Exception:
            pass
        self.view.add_text(text, name="hud", position="lower_left", font_size=12, color="#d7e6f3")

    def _allocate_particles(self, count: int, trail: int) -> None:
        self.pos = self._seed_positions(count)
        self.trail = np.empty((count, trail, 3), dtype=np.float64)
        self.trail_speed = np.empty((count, trail), dtype=np.float64)
        self.trail[:, 0] = self.pos
        wind = self._wind_hat * self._wind_speed
        step_len = float(np.mean(self._spacing)) * 0.72
        current = self.pos.copy()
        shape = self._solid_grid.shape
        for index in range(trail):
            vel = sample_velocity(self._vel, self._origin, self._spacing, current, wind, self._solid_grid)
            mag = np.linalg.norm(vel, axis=1)
            self.trail_speed[:, index] = mag
            if index + 1 == trail:
                break
            direction = vel / np.maximum(mag[:, None], 1e-8)
            nxt = current - direction * step_len
            bad = solid_mask_at(nxt, self._solid_grid, self._origin, self._spacing) | ~in_domain(
                nxt, self._origin, self._spacing, shape
            )
            nxt[bad] = current[bad]
            self.trail[:, index + 1] = nxt
            current = nxt
        self.age = np.full(count, trail, dtype=np.int32)

    def _rebuild_particle_actors(self) -> None:
        count, trail, _ = self.trail.shape
        points = np.zeros((count * trail, 3), dtype=np.float32)
        index = np.arange(count * trail, dtype=np.int64).reshape(count, trail)
        header = np.full((count, 1), trail, dtype=np.int64)
        cells = np.concatenate([header, index], axis=1).ravel()
        self._smoke = pv.PolyData(points, lines=cells)
        self._smoke["rgba"] = np.zeros((count * trail, 4), dtype=np.uint8)
        self.view.add_mesh(
            self._smoke,
            name="smoke",
            scalars="rgba",
            rgba=True,
            render_lines_as_tubes=True,
            line_width=max(1.2, float(self.point_slider.value()) * 0.7),
            reset_camera=False,
        )
        self._heads_mesh = pv.PolyData(np.zeros((count, 3), dtype=np.float32))
        self._heads_mesh["rgba"] = np.zeros((count, 4), dtype=np.uint8)
        self.view.add_mesh(
            self._heads_mesh,
            name="heads",
            scalars="rgba",
            rgba=True,
            render_points_as_spheres=True,
            point_size=float(self.point_slider.value()) * 1.6,
            reset_camera=False,
        )

    def _upload_smoke(self) -> None:
        if self._smoke is None or self.trail is None:
            return
        count, trail, _ = self.trail.shape
        ratio = np.clip(self.trail_speed / max(self._wind_speed, 1e-6), 0.0, 1.6)
        tone = np.clip(ratio / 1.35, 0.0, 1.0)[..., None]
        slow = np.array([0.55, 0.74, 0.90])
        fast = np.array([0.96, 0.98, 1.0])
        rgb = slow + (fast - slow) * tone
        fade = np.linspace(1.0, 0.12, trail) ** 1.15
        age_scale = np.clip(self.age / 6.0, 0.0, 1.0)
        opacity = float(self.opacity_slider.value())
        alpha = fade[None, :] * opacity * age_scale[:, None]
        rgba = np.empty((count, trail, 4), dtype=np.uint8)
        rgba[..., :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        rgba[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        self._smoke.points = np.ascontiguousarray(self.trail.reshape(-1, 3), dtype=np.float32)
        self._smoke.point_data["rgba"] = np.ascontiguousarray(rgba.reshape(-1, 4), dtype=np.uint8)
        head = np.empty((count, 4), dtype=np.uint8)
        head[:, :3] = rgba[:, 0, :3]
        head[:, 3] = np.clip(age_scale * max(opacity, 0.4) * 255.0, 0, 255).astype(np.uint8)
        self._heads_mesh.points = np.ascontiguousarray(self.pos, dtype=np.float32)
        self._heads_mesh.point_data["rgba"] = np.ascontiguousarray(head, dtype=np.uint8)

    def _seed_positions(self, count: int) -> np.ndarray:
        right, up = flow_basis(self._wind_hat)
        half = self._spacing * (np.array(self._solid_grid.shape) - 1) * 0.5
        center = self._origin + half
        reach = float(np.min(half))
        inlet = center - self._wind_hat * (reach * 0.82)
        along = self._rng.random(count) * reach * 1.55
        if self.smoke_pattern.currentText() == "Sheet":
            height = np.linspace(-0.48, 0.58, count) * reach
            span = (self._rng.random(count) - 0.5) * reach * 0.04
        else:
            rand = self._rng.random((count, 2))
            radius = np.sqrt(rand[:, 0]) * reach * 0.55
            theta = rand[:, 1] * (2.0 * np.pi)
            span = radius * np.cos(theta)
            height = radius * np.sin(theta)
        pos = inlet + span[:, None] * right + height[:, None] * up + self._wind_hat * along[:, None]
        blocked = solid_mask_at(pos, self._solid_grid, self._origin, self._spacing) | ~in_domain(
            pos, self._origin, self._spacing, self._solid_grid.shape
        )
        if np.any(blocked):
            pos[blocked] = self._inlet_points(int(blocked.sum()))
        return pos

    def _inlet_points(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0, 3), dtype=np.float64)
        right, up = flow_basis(self._wind_hat)
        half = self._spacing * (np.array(self._solid_grid.shape) - 1) * 0.5
        center = self._origin + half
        reach = float(np.min(half))
        inlet = center - self._wind_hat * (reach * 0.84)
        rand = self._rng.random((count, 2))
        sheet = self.smoke_pattern.currentText() == "Sheet"
        if sheet:
            span = (rand[:, 0] - 0.5) * reach * 0.04
            height = (-0.48 + rand[:, 1] * 1.06) * reach
        else:
            radius_pick = np.sqrt(rand[:, 0]) * reach * 0.55
            theta = rand[:, 1] * (2.0 * np.pi)
            span = radius_pick * np.cos(theta)
            height = radius_pick * np.sin(theta)
        return inlet + span[:, None] * right + height[:, None] * up

    def _gust(self, pos, turb: float) -> np.ndarray:
        t = self._time
        gust = np.column_stack(
            [
                np.sin(pos[:, 1] * 5.0 + t * 1.7) * np.cos(pos[:, 2] * 4.0 + t),
                np.cos(pos[:, 0] * 4.2 - t * 1.3) * np.sin(pos[:, 2] * 3.4),
                np.sin(pos[:, 0] * 3.6 + pos[:, 1] * 3.1 + t * 1.5),
            ]
        )
        gust = gust - (gust @ self._wind_hat)[:, None] * self._wind_hat
        return gust * (turb * self._wind_speed * 0.28)

    def _advance_smoke(self) -> None:
        if self._paused or self._vel is None or self.pos is None or self._closing:
            return
        dt = 0.033 * float(self.play_slider.value())
        wind = self._wind_hat * self._wind_speed
        turb = float(self.turb_slider.value())
        pos = self.pos
        head_vel = None
        for _ in range(2):
            vel = sample_velocity(self._vel, self._origin, self._spacing, pos, wind, self._solid_grid)
            if turb > 0:
                vel = vel + self._gust(pos, turb)
            step = vel * (dt / 2.0)
            mag = np.linalg.norm(step, axis=1)
            cap = 0.34 * float(np.mean(self._spacing))
            scale = np.minimum(1.0, cap / np.maximum(mag, 1e-12))
            pos = pos + step * scale[:, None]
            head_vel = vel
        self.pos = pos
        speed = np.linalg.norm(head_vel, axis=1)
        bad = (~in_domain(self.pos, self._origin, self._spacing, self._solid_grid.shape)) | solid_mask_at(
            self.pos, self._solid_grid, self._origin, self._spacing
        )
        bad = bad & (self.age > 2)
        n_bad = int(bad.sum())
        self.trail[:, 1:] = self.trail[:, :-1]
        self.trail[:, 0] = self.pos
        self.trail_speed[:, 1:] = self.trail_speed[:, :-1]
        self.trail_speed[:, 0] = speed
        if n_bad:
            self.pos[bad] = self._inlet_points(n_bad)
            self.trail[bad] = self.pos[bad, None, :]
            self.trail_speed[bad] = self._wind_speed
            self.age[bad] = 0
        self.age += 1
        self._time += dt
        if self.chk_smoke.isChecked() or self.chk_particles.isChecked():
            self._upload_smoke()
            self.view.render()

    def _resize_particles(self, _value=None) -> None:
        if self._vel is None:
            return
        self._allocate_particles(int(self.count_slider.value()), int(self.trail_slider.value()))
        self._rebuild_particle_actors()
        self._domain_key = (
            self._solid_grid.shape,
            round(float(self._origin[0]), 4),
            round(float(self._spacing[0]), 5),
            int(self.count_slider.value()),
            int(self.trail_slider.value()),
        )
        self._upload_smoke()
        self._apply_visibility()
        self.view.render()

    def _reset_smoke(self) -> None:
        if self._vel is None:
            return
        self._resize_particles()

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        self.pause_btn.setText("Play smoke" if self._paused else "Pause smoke")

    def _reset_settings(self) -> None:
        self.speed_slider.set_value(8.0)
        self.pitch_slider.set_value(6.0)
        self.yaw_slider.set_value(0.0)
        self.wake_slider.set_value(45.0)
        self.rho_slider.set_value(1.225)
        self.count_slider.set_value(72)
        self.trail_slider.set_value(18)
        self.point_slider.set_value(4)
        self.opacity_slider.set_value(0.92)
        self.turb_slider.set_value(0.08)
        self.play_slider.set_value(0.22)
        self.smoke_pattern.setCurrentText("Sheet")
        self.see_slider.set_value(1.0)
        self.stride_slider.set_value(3)
        self.quality.setCurrentText("Medium")
        self.vector_mode.setCurrentText("Side slice")
        for box in (
            self.chk_model,
            self.chk_pressure,
            self.chk_smoke,
            self.chk_particles,
            self.chk_vectors,
            self.chk_streams,
            self.chk_tunnel,
            self.chk_floor,
        ):
            box.setChecked(True)
        self._occ = None
        self._schedule(20)

    def _maybe_shot(self) -> None:
        path = os.environ.get("AERO_SHOT")
        if not path or self._shot_queued:
            return
        self._shot_queued = True
        QTimer.singleShot(1400, lambda: self._take_shot(path))

    def _take_shot(self, path: str) -> None:
        self.view.render()
        view_path = str(Path(path).with_name(Path(path).stem + "_view.png"))
        try:
            self.view.screenshot(view_path)
            print("SAVED_VIEW", view_path, flush=True)
        except Exception:
            traceback.print_exc()
        screen = self.screen()
        if screen is not None:
            screen.grabWindow(int(self.winId())).save(path)
            print("SAVED", path, flush=True)
        QTimer.singleShot(300, self.close)

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
                return True
        if kind == QEvent.Type.Drop and event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            if path:
                self.load_path(path)
            event.acceptProposedAction()
            return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event) -> None:
        self._closing = True
        self._tick.stop()
        self._save_settings()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(2500)
        try:
            self.view.close()
        except Exception:
            pass
        event.accept()


def _prepare_qt_plugins() -> None:
    try:
        import PySide6
    except ImportError:
        return
    plugins = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins"
    if plugins.is_dir():
        os.environ.setdefault("QT_PLUGIN_PATH", str(plugins))
        platforms = plugins / "platforms"
        if platforms.is_dir():
            os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platforms))


def main(argv=None) -> None:
    _prepare_qt_plugins()
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    app.setApplicationName("WindTunnel")
    app.setOrganizationName("Aero")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    app.setWindowIcon(build_icon())
    font = QFont(".AppleSystemUIFont")
    font.setPointSize(13)
    app.setFont(font)

    def _hook(exc_type, value, tb):
        text = "".join(traceback.format_exception(exc_type, value, tb))
        try:
            log = Path(__file__).resolve().parents[1] / "aero.log"
            log.write_text(text)
        except Exception:
            pass
        QMessageBox.critical(None, "Aero stopped", text[-1800:])
        sys.__excepthook__(exc_type, value, tb)

    sys.excepthook = _hook
    files = [item for item in (sys.argv[1:] if argv is None else argv) if Path(item).is_file()]
    window = MainWindow(files[0] if files else None)
    window.show()
    sys.exit(app.exec())
