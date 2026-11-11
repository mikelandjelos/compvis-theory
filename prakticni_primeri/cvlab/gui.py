"""GUI host (PySide6): pick an example, tweak auto-generated controls, see outputs live."""

from __future__ import annotations

import sys
import traceback
from typing import Any

import numpy as np

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QFont, QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
        QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
        QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget,
    )
except ImportError:  # pragma: no cover
    sys.exit("GUI needs PySide6: `poetry install --extras gui` (or --all-extras)")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from cvlab.core import Bool, Choice, Example, Float, Int, Param, load_examples
from cvlab.images import to_display

COLUMNS = 3
TILE = 360  # px, max side of image tiles


# --------------------------------------------------------------------------- param widgets


class ParamControl(QWidget):
    """Widget bound to one Param; calls `on_change` whenever the value changes."""

    def __init__(self, param: Param, on_change):
        super().__init__()
        self.param = param
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if isinstance(param, (Float, Int)):
            is_float = isinstance(param, Float)
            self.n_steps = max(1, round((param.max - param.min) / param.step))
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, self.n_steps)
            self.spin = QDoubleSpinBox() if is_float else QSpinBox()
            self.spin.setRange(param.min, param.max)
            self.spin.setSingleStep(param.step)
            if is_float:
                self.spin.setDecimals(max(0, -int(np.floor(np.log10(param.step)))) if param.step < 1 else 1)
            self.spin.setMinimumWidth(80)
            self.slider.valueChanged.connect(
                lambda i: self.spin.setValue(param.min + i * param.step) if self._idx() != i else None
            )
            self.spin.valueChanged.connect(lambda _: (self.slider.setValue(self._idx()), on_change()))
            lay.addWidget(self.slider, 1)
            lay.addWidget(self.spin)
        elif isinstance(param, Choice):
            self.combo = QComboBox()
            self.combo.addItems(list(param.options))
            self.combo.currentTextChanged.connect(lambda _: on_change())
            lay.addWidget(self.combo, 1)
        elif isinstance(param, Bool):
            self.check = QCheckBox()
            self.check.toggled.connect(lambda _: on_change())
            lay.addWidget(self.check, 1)
        else:
            raise TypeError(f"no widget for {type(param).__name__}")
        if param.help:
            self.setToolTip(param.help)
        self.set(param.default)

    def _idx(self) -> int:
        return round((self.spin.value() - self.param.min) / self.param.step)

    def set(self, value: Any) -> None:
        p = self.param
        if isinstance(p, (Float, Int)):
            self.spin.setValue(value)
            self.slider.setValue(self._idx())
        elif isinstance(p, Choice):
            self.combo.setCurrentText(value)
        else:
            self.check.setChecked(bool(value))

    def value(self) -> Any:
        p = self.param
        if isinstance(p, (Float, Int)):
            return self.spin.value()
        if isinstance(p, Choice):
            return self.combo.currentText()
        return self.check.isChecked()


# --------------------------------------------------------------------------- output tiles


def image_widget(img: np.ndarray) -> QLabel:
    img = to_display(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        qimg = QImage(img.data, w, h, img.strides[0], QImage.Format_Grayscale8)
    else:
        qimg = QImage(img.data, w, h, img.strides[0], QImage.Format_BGR888)
    pix = QPixmap.fromImage(qimg.copy())
    if max(w, h) > TILE or max(w, h) < TILE // 2:
        # upscale tiny images with nearest neighbour so pixels stay visible
        mode = Qt.FastTransformation if max(w, h) < TILE // 2 else Qt.SmoothTransformation
        pix = pix.scaled(TILE, TILE, Qt.KeepAspectRatio, mode)
    label = QLabel()
    label.setPixmap(pix)
    label.setAlignment(Qt.AlignCenter)
    return label


def output_tile(name: str, value) -> QWidget:
    box = QGroupBox(name)
    lay = QVBoxLayout(box)
    if isinstance(value, np.ndarray):
        lay.addWidget(image_widget(value))
    elif isinstance(value, Figure):
        canvas = FigureCanvasQTAgg(value)
        canvas.setMinimumSize(TILE, int(TILE * 0.8))
        lay.addWidget(canvas)
    else:
        label = QLabel(str(value))
        label.setFont(QFont("monospace"))
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setWordWrap(True)
        lay.addWidget(label)
    return box


# --------------------------------------------------------------------------- main window


class MainWindow(QMainWindow):
    def __init__(self, registry: dict[str, Example]):
        super().__init__()
        self.setWindowTitle("cvlab — computer vision exam examples")
        self.registry = registry
        self.example: Example | None = None
        self.controls: list[ParamControl] = []

        self.list = QListWidget()
        for ex in registry.values():
            item = QListWidgetItem(f"{ex.key}  {ex.title}")
            item.setData(Qt.UserRole, ex.key)
            self.list.addItem(item)
        self.list.currentItemChanged.connect(lambda cur, _: cur and self.select(cur.data(Qt.UserRole)))

        self.title = QLabel()
        self.title.setWordWrap(True)
        self.title.setTextFormat(Qt.RichText)
        self.form_box = QGroupBox("Parameters")
        self.form = QFormLayout(self.form_box)
        reset = QPushButton("Reset")
        reset.clicked.connect(self.reset)

        self.outputs = QWidget()
        self.grid = QGridLayout(self.outputs)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.outputs)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.addWidget(self.title)
        top = QHBoxLayout()
        top.addWidget(self.form_box, 1)
        top.addWidget(reset, 0, Qt.AlignTop)
        rlay.addLayout(top)
        rlay.addWidget(scroll, 1)

        split = QSplitter()
        split.addWidget(self.list)
        split.addWidget(right)
        split.setSizes([280, 1100])
        self.setCentralWidget(split)

        self.timer = QTimer(self, singleShot=True, interval=150)
        self.timer.timeout.connect(self.run)
        if registry:
            self.list.setCurrentRow(0)
        self.resize(1400, 900)

    def select(self, key: str) -> None:
        self.example = ex = self.registry[key]
        desc = f"<br><i>{ex.description}</i>" if ex.description else ""
        self.title.setText(f"<b>{ex.key} — {ex.title}</b>{desc}")
        while self.form.rowCount():
            self.form.removeRow(0)
        self.controls = [ParamControl(p, self.timer.start) for p in ex.params]
        for c in self.controls:
            self.form.addRow(c.param.display, c)
        self.run()

    def reset(self) -> None:
        for c in self.controls:
            c.set(c.param.default)
        self.run()

    def run(self) -> None:
        if not self.example:
            return
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w:
                w.setParent(None)  # detach now so stale tiles never paint over new ones
                w.deleteLater()
        try:
            outputs = self.example.run(**{c.param.name: c.value() for c in self.controls})
        except Exception as exc:  # show errors in-app instead of crashing the GUI
            traceback.print_exc()
            self.grid.addWidget(output_tile("Error", f"{type(exc).__name__}: {exc}"), 0, 0)
            self.statusBar().showMessage(str(exc))
            return
        for i, (name, value) in enumerate(outputs.items()):
            self.grid.addWidget(output_tile(name, value), i // COLUMNS, i % COLUMNS)
        self.statusBar().clearMessage()


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow(load_examples())
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
