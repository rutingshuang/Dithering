# PyQt5 GUI: Image color reduction + resolution compression + dithering
# Requirements implemented from user's spec.
# - Resizable window
# - Drag & drop or click to upload image
# - Resolution compression slider + manual width/height input
# - Reads palette from 'palette.txt' (format: "Black:[0, 0, 0]" or "*Medium Gray:[170, 170, 170]")
#   - '*' prefix => paid group (right); otherwise free group (left)
#   - each group shown as grid: 4 buttons/row, up to 6 rows visible, extra rows scrollable
# - Dithering selection: ordered_dithering (method, strength 32-128) or floyd_steinberg_dither (alpha 0-1)
#   - The user said the dithering functions already exist: this script will try to import them from
#     a module named `dithers`. If not found, fallback simple placeholders are used. Replace placeholders
#     with your actual implementations (or place them in dithers.py alongside this script).
# - Default save path: ./save
# - Preview panel and Save result
# - Uses Pillow (PIL) and numpy

import os
import sys
import math
import copy
from functools import partial

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QLocale
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QMainWindow, QFileDialog, QPushButton,
    QHBoxLayout, QVBoxLayout, QGridLayout, QSlider, QLineEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QScrollArea, QMessageBox, QSizePolicy, QGroupBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QDialog, QDialogButtonBox, QHeaderView, QAbstractItemView,
    QColorDialog
)
from PyQt5.QtGui import QPixmap, QImage, QColor

from PIL import Image, ImageQt
import numpy as np

# Try importing user's dithering functions. If unavailable, provide simple placeholders.
try:
    from dithers import ordered_dithering, floyd_steinberg_dither, barycentric_dither

    print("Loaded dithering functions from dithers.py")
except Exception as e:
    print("Could not import dithering functions (dithers.py). Using placeholders.", e)


    def barycentric_dither(image, palette, method="bayer", space="RGB", **kwargs):
        # Placeholder: 回退为普通有序抖动
        return ordered_dithering(image, palette, method=method, strength=64)


    def ordered_dithering(image, palette, method="bayer", strength=64, color_space="RGB", weights=(1.0, 1.0, 1.0)):
        # Placeholder: simply quantize the image to the palette using nearest neighbor.
        # Replace with the user's ordered_dithering implementation.
        arr = np.array(image.convert('RGB'))
        pal = np.array(palette, dtype=np.uint8)
        h, w, _ = arr.shape
        flat = arr.reshape(-1, 3).astype(np.int32)
        # compute distance and choose nearest palette color
        d = np.sum((flat[:, None, :] - pal[None, :, :]) ** 2, axis=2)
        idx = np.argmin(d, axis=1)
        out = pal[idx].reshape(h, w, 3).astype(np.uint8)
        return Image.fromarray(out, 'RGB')


    def floyd_steinberg_dither(image, palette, alpha_strength=1.0):
        # Placeholder simple quantize + basic error diffusion
        pal = np.array(palette, dtype=np.uint8)
        arr = np.array(image.convert('RGB')).astype(np.float32)
        h, w, _ = arr.shape
        out = np.zeros_like(arr, dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                old = arr[y, x]
                d = np.sum((pal - old) ** 2, axis=1)
                idx = np.argmin(d)
                new = pal[idx]
                out[y, x] = new
                err = (old - new) * alpha_strength
                if x + 1 < w:
                    arr[y, x + 1] += err * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        arr[y + 1, x - 1] += err * 3 / 16
                    arr[y + 1, x] += err * 5 / 16
                    if x + 1 < w:
                        arr[y + 1, x + 1] += err * 1 / 16
        return Image.fromarray(out.astype(np.uint8))


# 分层抖动引擎（layered_dither.py）：复刻 Krita「按明暗分层 + 逐层调参」的工作流
try:
    import layered_dither as ldl
    from layered_dither import render as run_layered_dither

    LAYERED_AVAILABLE = True
    print("Loaded layered_dither.py")
except Exception as e:
    ldl = None
    LAYERED_AVAILABLE = False
    print("Could not import layered_dither.py:", e)


    def run_layered_dither(image, palette, profile, **kw):
        return image.convert('RGB')


# 双读抖动引擎（dualview_dither.py）：一张图，远看是主图、近看是隐藏图
try:
    import dualview_dither as dvd

    DUALVIEW_AVAILABLE = True
    print("Loaded dualview_dither.py")
except Exception as e:
    dvd = None
    DUALVIEW_AVAILABLE = False
    print("Could not import dualview_dither.py:", e)


# 分层预设：界面名 → 生成函数(n_bands, colors_per_band)
def _layer_presets():
    if not LAYERED_AVAILABLE:
        return {}
    return {
        '标准（投影混色·落盘内）': lambda n_bands, colors_per_band=8:
            ldl.make_projection_profile(n_bands=n_bands, colors_per_band=colors_per_band),
    }


LAYER_PATTERNS = ('bayer', 'clustered', 'rounddot', 'diagonal', 'spiral',
                  'checker', 'noise', 'bluenoise')
LAYER_EDGE_MODES = {
    '绝对明度（保色彩）': 'linear',
    '自适应等频（保细节）': 'quantile',
    '混合': 'hybrid',
}
LAYER_SPACES = ['Lab', 'RGB']

# 双读抖动的四个预览视角
DUAL_VIEWS = ('原始 A', '隐藏图 B', '合成·远看 → A', '合成·近看 1:1 → B')
DUAL_CROP = 720   # 「近看 1:1」视图裁多大的中心方块


# Utility: read palette file
def load_palette_from_file(path='palette.txt'):
    """Reads palette file. Each line: 'Name:[r, g, b]' or '*Name:[r,g,b]'.
    Returns: list of tuples (name, (r,g,b), is_paid)
    """
    if not os.path.exists(path):
        return []
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            is_paid = False
            if line.startswith('*'):
                is_paid = True
                line = line[1:].strip()
            # split on ':' to separate name and rgb
            if ':' not in line:
                continue
            name_part, rgb_part = line.split(':', 1)
            name = name_part.strip()
            # parse [r, g, b]
            try:
                rgb_str = rgb_part.strip()
                if rgb_str.startswith('[') and rgb_str.endswith(']'):
                    rgb_str = rgb_str[1:-1]
                comps = [int(x.strip()) for x in rgb_str.split(',')]
                if len(comps) != 3:
                    continue
                lines.append((name, tuple(comps), is_paid))
            except Exception:
                continue
    return lines


# ColorButton class from test.py
class ColorButton(QPushButton):
    def __init__(self, name, rgb, parent=None):
        super().__init__(parent)
        self.name = name
        self.rgb = rgb
        self.selected = False
        self.setCheckable(True)
        self.setFixedSize(QSize(80, 48))
        self.update_style()

    def update_style(self):
        r, g, b = self.rgb
        border = "3px solid #FFD54F" if self.selected else "1px solid #555"
        text_color = "#fff" if (r * 0.299 + g * 0.587 + b * 0.114) < 140 else "#000"
        self.setStyleSheet(f"""
            QPushButton{{
                background-color: rgb({r},{g},{b});
                border: {border};
                border-radius: 6px;
                color: {text_color};
                font-size: 10px;
                text-align: center;
            }}
        """)
        self.setText(self.name)


# PalettePanel class from test.py
class PalettePanel(QWidget):
    """Panel that shows a group of ColorButtons in a scrollable grid, with select/deselect all."""

    def __init__(self, title, entries, parent=None):
        super().__init__(parent)
        self.entries = entries  # list of (name, (r,g,b))
        self.buttons = []
        self._build_ui(title)

    def _build_ui(self, title):
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        label = QLabel(title)
        select_all_btn = QPushButton("全选")
        deselect_all_btn = QPushButton("全不选")
        select_all_btn.clicked.connect(self.select_all)
        deselect_all_btn.clicked.connect(self.deselect_all)
        header.addWidget(label)
        header.addStretch()
        header.addWidget(select_all_btn)
        header.addWidget(deselect_all_btn)
        layout.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.grid = QGridLayout(container)
        self.grid.setSpacing(8)
        # add buttons
        col_count = 4
        for idx, (name, rgb) in enumerate(self.entries):
            btn = ColorButton(name, rgb)
            btn.clicked.connect(partial(self.toggle_btn, btn))
            r = idx // col_count
            c = idx % col_count
            self.grid.addWidget(btn, r, c)
            self.buttons.append(btn)

        # Fill empty slots up to at least 1 row to keep layout tidy
        # But not necessary.
        scroll.setWidget(container)
        # limit visible height to 6 rows
        row_height = 48 + 8  # approx
        max_rows_visible = 6
        scroll.setFixedHeight(min(max_rows_visible * row_height + 10, 6 * row_height + 10))
        layout.addWidget(scroll)

    def toggle_btn(self, btn):
        btn.selected = not btn.selected
        btn.setChecked(btn.selected)
        btn.update_style()

    def select_all(self):
        for b in self.buttons:
            b.selected = True
            b.setChecked(True)
            b.update_style()

    def deselect_all(self):
        for b in self.buttons:
            b.selected = False
            b.setChecked(False)
            b.update_style()

    def get_selected_palette(self):
        return [b.rgb for b in self.buttons if b.selected]


# Drag-and-drop QLabel
class DropLabel(QLabel):
    image_dropped = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setText("拖拽图片到这里或点击'打开'上传")
        self.setStyleSheet("QLabel{border: 2px dashed #aaa;}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.image_dropped.emit(path)


# 后台处理线程：把耗时的图片加载/抖动放到子线程，UI 保持响应
class ProcessWorker(QThread):
    finished = pyqtSignal(object)   # 成功结果（PIL Image 等）
    failed = pyqtSignal(str)        # 失败信息

    def __init__(self, func, args, parent=None):
        super().__init__(parent)
        self._func = func
        self._args = args
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            result = self._func(*self._args)
            if not self._cancel:
                self.finished.emit(result)
        except Exception as e:
            if not self._cancel:
                self.failed.emit(str(e))


SLOT_METHODS = ('mix', 'direct', 'ordered', 'error')
SLOT_METHOD_LABEL = {
    'mix': '重心混合',
    'direct': '直接投影',
    'ordered': '有序抖动',
    'error': '误差扩散',
}


def pil_to_pixmap(im, max_w=420):
    """PIL Image -> QPixmap（保留 alpha，超宽则按最近邻缩小保像素风）。"""
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    if max_w and im.width > max_w:
        s = max_w / im.width
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.NEAREST)
    data = im.tobytes('raw', 'RGBA')
    qim = QImage(data, im.width, im.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qim.copy())


class LayerPreviewDialog(QDialog):
    """单层预览：左原色、右抖动结果，都只显示该层像素，其余透明。"""

    def __init__(self, src_img, out_img, title='', parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'层预览 — {title}')
        self.resize(900, 500)
        canvas = QLabel()
        canvas.setAlignment(Qt.AlignCenter)
        canvas.setStyleSheet('background: #2b2b2b;')
        tile = QPixmap(canvas.size())
        grid = QGridLayout()
        for col, (name, im) in enumerate((('原色（该层）', src_img), ('抖动（该层）', out_img))):
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignCenter)
            pic = QLabel()
            pic.setAlignment(Qt.AlignCenter)
            pic.setPixmap(pil_to_pixmap(im))
            pic.setStyleSheet('background: #2b2b2b;')
            grid.addWidget(lbl, 0, col)
            grid.addWidget(pic, 1, col)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        lay = QVBoxLayout(self)
        lay.addLayout(grid)
        lay.addWidget(btns)


class LayerColorDialog(QDialog):
    """单层颜色槽编辑：增删颜色、改色号、每个颜色单独选还原方式。

    这是「自动投影」与「手工指定」之间的开关：
      - 没有颜色槽 → 回到自动（k-means 聚投影色 + 凸包混合）
      - 有颜色槽   → 跳过 k-means，层内像素归到最近的颜色槽，各槽按自己的方式还原
    """

    COLS = ['色号', '颜色', '还原方式', '强度', '删除']

    def __init__(self, band, palette, auto_colors=None, pixels=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'层颜色（该层 {pixels} 像素）')
        self.resize(640, 480)
        self.band = band
        self.palette = list(palette)
        self.auto_colors = auto_colors or []
        self.rows = []
        for c in (band.colors or []):
            if isinstance(c, dict):
                self.rows.append({'color': c.get('color', 0),
                                  'method': c.get('method', 'mix'),
                                  'strength': int(c.get('strength', 64))})
            else:
                self.rows.append({'color': c, 'method': 'mix', 'strength': 64})

        hint = QLabel(
            '没有颜色槽 = 回到自动投影；加了槽就跳过 k-means，层内像素按最近色归到各个槽。\n'
            '「重心混合」会用该色周围的调色板色把它混出来（色号本身在板内时退化为平涂）。\n'
            '「误差扩散」只能在层内像素的扫描序上传误差——同一层的像素在画面上是分散的，'
            '所以比整图误差扩散弱，容易结成条带。')
        hint.setWordWrap(True)

        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        btn_add = QPushButton('添加颜色')
        btn_add.clicked.connect(self._add_row)
        btn_import = QPushButton('导入自动结果')
        btn_import.setToolTip('把该层自动聚出的投影色配方导进来作为编辑起点')
        btn_import.clicked.connect(self._import_auto)
        btn_clear = QPushButton('清空（回到自动）')
        btn_clear.clicked.connect(self._clear)
        self.lbl_summary = QLabel()
        top = QHBoxLayout()
        top.addWidget(btn_add)
        top.addWidget(btn_import)
        top.addWidget(btn_clear)
        top.addStretch(1)
        top.addWidget(self.lbl_summary)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        lay.addWidget(hint)
        lay.addLayout(top)
        lay.addWidget(self.table)
        lay.addWidget(btns)

        self._rebuild()

    def _rebuild(self):
        self.table.setRowCount(len(self.rows))
        self._spins, self._combos, self._strengths = [], [], []
        for i, r in enumerate(self.rows):
            spin = QSpinBox()
            spin.setRange(0, len(self.palette) - 1)
            spin.setValue(int(r['color']) if isinstance(r['color'], int) else 0)
            spin.valueChanged.connect(lambda _v, i=i: self._refresh_swatch(i))
            self.table.setCellWidget(i, 0, spin)
            self._spins.append(spin)

            self.table.setItem(i, 1, QTableWidgetItem(''))
            combo = QComboBox()
            combo.addItems([SLOT_METHOD_LABEL[m] for m in SLOT_METHODS])
            combo.setCurrentText(SLOT_METHOD_LABEL.get(r['method'], '重心混合'))
            self.table.setCellWidget(i, 2, combo)
            self._combos.append(combo)

            st = QSpinBox()
            st.setRange(0, 128)
            st.setValue(int(r['strength']))
            st.setToolTip('有序抖动/误差扩散的扰动强度')
            self.table.setCellWidget(i, 3, st)
            self._strengths.append(st)

            btn = QPushButton('删除')
            btn.clicked.connect(lambda _c, i=i: self._del_row(i))
            self.table.setCellWidget(i, 4, btn)
            self._refresh_swatch(i)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.lbl_summary.setText(f'{len(self.rows)} 个槽' if self.rows else '当前：自动投影')

    def _refresh_swatch(self, i):
        idx = int(self._spins[i].value())
        c = self.palette[idx]
        item = self.table.item(i, 1)
        if item is None:                    # 已有 item 不能再 setItem，会触发 owned 警告
            item = QTableWidgetItem('')
            self.table.setItem(i, 1, item)
        item.setText(f'#{int(c[0]):02X}{int(c[1]):02X}{int(c[2]):02X}')
        item.setBackground(QtGui.QBrush(QtGui.QColor(int(c[0]), int(c[1]), int(c[2]))))
        item.setForeground(QtGui.QBrush(QtCore.Qt.white if sum(c) < 380 else QtCore.Qt.black))

    def _add_row(self):
        self.rows.append({'color': 0, 'method': 'mix', 'strength': 64})
        self._rebuild()

    def _del_row(self, i):
        if 0 <= i < len(self.rows):
            self.rows.pop(i)
            self._rebuild()

    def _import_auto(self):
        n = 0
        for c in self.auto_colors:
            rec = c.get('recipe') or []
            if rec:
                self.rows.append({'color': int(rec[0]['index']), 'method': 'mix', 'strength': 64})
                n += 1
        if n == 0:
            QMessageBox.information(self, '没有自动结果',
                                    '该层没有可导入的自动投影色（可能是空层，或已设为平涂）。')
        self._rebuild()

    def _clear(self):
        self.rows = []
        self._rebuild()

    def result_slots(self):
        return [{'color': int(self._spins[i].value()),
                 'method': SLOT_METHODS[max(0, self._combos[i].currentIndex())],
                 'strength': int(self._strengths[i].value())}
                for i in range(len(self.rows))]


class LayerTableDialog(QDialog):
    """分层参数表：一行 = 一个明度层 = Krita 里的一个图层。

    手工工作流里，作者要为每个图层分别调"抖动图案/图案大小"；
    这张表就是那几次调参的等价物，改完直接复用、可存成预设。
    """

    COLS = ['层', '明度范围', '像素占比', '图案', '阶数', '放大', '颜色', '预览', '启用']

    def __init__(self, profile, ratios=None, parent=None,
                 palette=None, image=None, preview_cb=None):
        super().__init__(parent)
        self.setWindowTitle('分层参数表（每行 = 一个明度层）')
        self.resize(900, 640)
        self.profile = profile
        self.ratios = ratios
        self.palette = palette
        self.image = image
        self.preview_cb = preview_cb

        bands = profile.bands
        n = len(bands)
        table = QTableWidget(n, len(self.COLS))
        table.setHorizontalHeaderLabels(self.COLS)
        table.verticalHeader().setVisible(False)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.pattern_boxes = []
        self.size_boxes = []
        self.cell_boxes = []
        self.enable_boxes = []
        self.color_btns = []

        for i, b in enumerate(bands):
            lo, hi = b.wide()
            table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            table.setItem(i, 1, QTableWidgetItem(f'L* {lo * 100:.1f} ~ {hi * 100:.1f}'))
            if ratios is not None and i < len(ratios):
                table.setItem(i, 2, QTableWidgetItem(f'{ratios[i] * 100:.2f}%'))
            else:
                table.setItem(i, 2, QTableWidgetItem('-'))

            cb = QComboBox()
            cb.addItems(LAYER_PATTERNS)
            cb.setCurrentText(b.pattern if b.pattern in LAYER_PATTERNS else 'bayer')
            cb.setToolTip('该层的抖动图案；clustered 容易聚成方块色团，一般用 bayer / bluenoise')
            table.setCellWidget(i, 3, cb)
            self.pattern_boxes.append(cb)

            sp_size = QSpinBox()
            sp_size.setRange(2, 64)
            sp_size.setValue(int(b.size))
            sp_size.setToolTip('图案阶数：越大网点越细密')
            table.setCellWidget(i, 4, sp_size)
            self.size_boxes.append(sp_size)

            sp_cell = QSpinBox()
            sp_cell.setRange(1, 16)
            sp_cell.setValue(int(b.cell))
            sp_cell.setToolTip('每个图案单元放大到几像素（越大图案越粗）')
            table.setCellWidget(i, 5, sp_cell)
            self.cell_boxes.append(sp_cell)

            btn_col = QPushButton('自动' if not b.colors else f'手动 {len(b.colors)}')
            btn_col.setToolTip('点开编辑该层用哪些颜色、每个颜色用哪种还原方式；改完重新投影')
            btn_col.clicked.connect(lambda _c, i=i: self._open_colors(i))
            table.setCellWidget(i, 6, btn_col)
            self.color_btns.append(btn_col)

            btn_prev = QPushButton('查看')
            btn_prev.setToolTip('只显示这一层的像素：左边原色、右边抖动结果')
            btn_prev.clicked.connect(lambda _c, i=i: self._open_preview(i))
            table.setCellWidget(i, 7, btn_prev)

            chk = QCheckBox()
            chk.setChecked(b.enabled)
            chk.setToolTip('取消勾选 = 该层不参与抖动')
            table.setCellWidget(i, 8, chk)
            self.enable_boxes.append(chk)

        table.resizeColumnsToContents()
        header = table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        hint = QLabel('提示：所有层的图案共用同一坐标原点，层与层之间纹路不会错位。\n'
                      '抖动比例由"投影色"决定，不在这里调；这里控制的是每一层纹理的样式与粗细。')
        hint.setWordWrap(True)

        lay = QVBoxLayout(self)
        lay.addWidget(hint)
        lay.addWidget(table)
        lay.addWidget(btns)

    def _sync_to_profile(self):
        """把表格里的控件值写回 profile（预览前也要同步，否则看的还是旧参数）。"""
        for i, b in enumerate(self.profile.bands):
            b.pattern = self.pattern_boxes[i].currentText()
            b.size = int(self.size_boxes[i].value())
            b.cell = int(self.cell_boxes[i].value())

    def _open_colors(self, i):
        if ldl is None or self.palette is None:
            QMessageBox.information(self, '不可用', '分层引擎或调色板不可用（请先载入图片/调色板）。')
            return
        band = self.profile.bands[i]
        auto, pixels = [], 0
        if self.image is not None:
            try:
                self._sync_to_profile()
                info = ldl.analyze_bands(self.image, self.palette, self.profile, only=i)
                if info:
                    auto = info[0]['colors']
                    pixels = info[0]['pixels']
            except Exception as e:
                print('analyze_bands failed:', e)
        dlg = LayerColorDialog(band, self.palette, auto, pixels, self)
        if dlg.exec_() == QDialog.Accepted:
            slots = dlg.result_slots()
            band.colors = slots or None
            self.color_btns[i].setText('自动' if not slots else f'手动 {len(slots)}')

    def _open_preview(self, i):
        if self.preview_cb is None:
            QMessageBox.information(self, '不可用', '需要先在主界面载入图片。')
            return
        self._sync_to_profile()
        try:
            src, out = self.preview_cb(i)
        except Exception as e:
            QMessageBox.warning(self, '预览失败', f'单层预览失败: {e}')
            return
        LayerPreviewDialog(src, out, f'第 {i + 1} 层', self).exec_()

    def apply_to_profile(self):
        """把表格里的值写回 profile.bands。"""
        self._sync_to_profile()
        for i, b in enumerate(self.profile.bands):
            b.enabled = self.enable_boxes[i].isChecked()

    def accept(self):
        self.apply_to_profile()
        super().accept()


class DualViewDialog(QDialog):
    """双读抖动的双视角对照：左边近看（B），右边远看（A）。

    收 QPixmap 而不是 PIL Image —— 复用主窗口那套已处理过行对齐的转换。
    """

    def __init__(self, near_pix, far_pix, profile, parent=None):
        super().__init__(parent)
        self.setWindowTitle('双视角预览 —— 近看是 B，远看是 A')
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        pairs = (
            ('近看（1:1 局部）→ B', near_pix, '把脸贴近屏幕，或把图放到 100% 看'),
            ('远看（缩略图）→ A', far_pix, '退后几步 / 眯眼看 / 平台生成的缩略图'),
        )
        for title, pix, tip in pairs:
            box = QVBoxLayout()
            lab_title = QLabel(title)
            lab_title.setStyleSheet('font-weight: bold;')
            box.addWidget(lab_title)
            holder = QLabel()
            holder.setPixmap(pix)
            holder.setAlignment(Qt.AlignCenter)
            holder.setStyleSheet('background:#111;')
            box.addWidget(holder)
            lab_tip = QLabel(tip)
            lab_tip.setStyleSheet('color:#888;')
            lab_tip.setWordWrap(True)
            box.addWidget(lab_tip)
            row.addLayout(box)

        lay.addLayout(row)
        info = QLabel(
            f'块尺寸 k={profile.k}，隐藏图 {profile.levels} 级，平移轴 {profile.axis}，'
            f'色彩空间 {profile.match_space}。\n'
            f'右边那格是把结果缩到 1/k 再放大 —— 一个块正好折成一个像素，'
            f'所以它就是缩略图/退后看的严格等效。')
        info.setWordWrap(True)
        lay.addWidget(info)
        btn = QDialogButtonBox(QDialogButtonBox.Close)
        btn.rejected.connect(self.reject)
        btn.accepted.connect(self.accept)
        lay.addWidget(btn)


class ImageDitherApp(QMainWindow):
    def __init__(self):
        # 修复 Qt5 bug：部分 CJK 区域（如 zh_HK）的系统默认 QLocale 会把数字格式化成
        # CJK 字形（100 → 〈〇〇），影响所有 QSpinBox。用同名 locale 显式重建可保留
        # 区域设置、同时使用拉丁数字。
        # 注意：必须在 super().__init__()（创建窗口）之前调用——子控件会继承父窗口
        # 构造时捕获的 locale，窗口建好后 setDefault 对已挂载的控件不生效。
        QLocale.setDefault(QLocale(QLocale().name()))
        super().__init__()
        self.setWindowTitle('Image Dither & Compress (PyQt5)')
        self.resize(1200, 800)  # 增加窗口宽度以容纳更多控件

        self.image_path = None
        self.original_image = None  # PIL Image
        self.processed_image = None
        self.palette_items = load_palette_from_file('palette.txt')
        # palette selection: list of (r,g,b)
        self.selected_palette = []

        self.save_folder = os.path.join(os.getcwd(), 'save')
        os.makedirs(self.save_folder, exist_ok=True)

        # 后台线程引用（防止被 GC），以及预览缩略图上限
        self._worker = None
        self.preview_cap = 4_000_000  # 预览时处理的最大**总像素数**（不是长边，避免条漫被压扁）
        self._was_preview = False  # 最近一次处理是预览还是保存

        # 重心混合：凸包投影 开/关 各存一套 权重截断/边缘保护 的值，勾选切换时互换
        self._minw_cfg = {'off': 0.0, 'on': 0.0}
        self._edge_cfg = {'off': 0, 'on': 0}
        self._gamut_state = False
        # 分层抖动：手工编辑过的 Profile（None = 按预设曲线自动生成）
        self._layer_profile = None
        self._layer_profile_edited = False   # 区分"自动缓存"与"用户手改"，切换引擎时行为不同
        # 双读抖动：隐藏图 B（PIL Image，None = 还没选）
        self.dual_hidden_image = None
        self._dual_view_cache = {}      # {'near':…, 'far':…} 由合成结果派生
        self._dual_view_active = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left: controls
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setContentsMargins(6, 6, 6, 6)

        # Upload / preview area
        upload_layout = QHBoxLayout()
        self.drop_label = DropLabel()
        self.drop_label.setFixedHeight(160)
        self.drop_label.image_dropped.connect(self.load_image_from_path)
        upload_layout.addWidget(self.drop_label)

        btn_col = QVBoxLayout()
        btn_open = QPushButton('打开图片')
        btn_open.clicked.connect(self.on_open_image)
        btn_open.setToolTip('打开并加载图片')
        btn_col.addWidget(btn_open)

        self.lbl_file = QLabel('未选择文件')
        self.lbl_file.setWordWrap(True)
        btn_col.addWidget(self.lbl_file)

        upload_layout.addLayout(btn_col)
        ctrl_layout.addLayout(upload_layout)

        # Resolution compression controls
        res_group = QWidget()
        res_layout = QGridLayout(res_group)
        res_layout.setContentsMargins(0, 0, 0, 0)
        res_layout.addWidget(QLabel('压缩比例(%)'), 0, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setValue(100)
        self.slider.valueChanged.connect(self.on_slider_change)
        res_layout.addWidget(self.slider, 0, 1)
        self.spin_percent = QSpinBox()
        self.spin_percent.setRange(1, 100)
        self.spin_percent.setValue(100)
        self.spin_percent.valueChanged.connect(self.on_spin_percent_change)
        res_layout.addWidget(self.spin_percent, 0, 2)

        res_layout.addWidget(QLabel('指定宽 (px)'), 1, 0)
        self.line_w = QLineEdit()
        self.line_w.setPlaceholderText('不填则按比例')
        self.line_w.returnPressed.connect(self.on_manual_wh)
        res_layout.addWidget(self.line_w, 1, 1)
        res_layout.addWidget(QLabel('指定高 (px)'), 1, 2)
        self.line_h = QLineEdit()
        self.line_h.setPlaceholderText('不填则按比例')
        self.line_h.returnPressed.connect(self.on_manual_wh)
        res_layout.addWidget(self.line_h, 1, 3)

        ctrl_layout.addWidget(res_group)

        # Palette button grids
        palette_label = QLabel('调色盘 (左: 免费, 右: 付费)')
        ctrl_layout.addWidget(palette_label)

        palette_area = QWidget()
        pal_layout = QHBoxLayout(palette_area)
        pal_layout.setContentsMargins(0, 0, 0, 0)

        # Create free and paid color lists
        free_entries = [(name, rgb) for name, rgb, is_paid in self.palette_items if not is_paid]
        paid_entries = [(name, rgb) for name, rgb, is_paid in self.palette_items if is_paid]

        # Free colors panel
        self.free_panel = PalettePanel("免费颜色", free_entries)
        pal_layout.addWidget(self.free_panel)

        # Paid colors panel
        self.paid_panel = PalettePanel("付费颜色", paid_entries)
        pal_layout.addWidget(self.paid_panel)

        ctrl_layout.addWidget(palette_area)

        # Dither selection
        dither_group = QGroupBox("抖动设置")
        dither_layout = QVBoxLayout(dither_group)

        # 方法选择
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel('抖动方法'))
        self.combo_dither = QComboBox()
        _methods = ['有序抖动', '误差扩散抖动', '重心混合']
        if LAYERED_AVAILABLE:
            _methods.append('分层抖动')
        if DUALVIEW_AVAILABLE:
            _methods.append('双读抖动')
        self.combo_dither.addItems(_methods)
        self.combo_dither.currentIndexChanged.connect(self.on_dither_method_change)
        method_layout.addWidget(self.combo_dither)
        dither_layout.addLayout(method_layout)

        # 有序抖动参数区域
        self.ordered_params = QWidget()
        ordered_layout = QGridLayout(self.ordered_params)

        # 强度参数
        ordered_layout.addWidget(QLabel('强度 (32-128)'), 0, 0)
        self.spin_strength = QSpinBox()
        self.spin_strength.setRange(32, 128)
        self.spin_strength.setValue(64)
        ordered_layout.addWidget(self.spin_strength, 0, 1)

        # 色彩空间选择
        ordered_layout.addWidget(QLabel('色彩空间'), 1, 0)
        self.combo_color_space = QComboBox()
        self.combo_color_space.addItems(['RGB', 'HSV', 'Lab'])
        self.combo_color_space.currentIndexChanged.connect(self.on_color_space_change)
        ordered_layout.addWidget(self.combo_color_space, 1, 1)

        # 阈值矩阵选择
        ordered_layout.addWidget(QLabel('阈值矩阵'), 2, 0)
        self.combo_matrix = QComboBox()
        self.combo_matrix.addItems(['bayer', 'clustered', 'diagonal', 'spiral'])
        ordered_layout.addWidget(self.combo_matrix, 2, 1)

        # HSV权重参数 (初始隐藏)
        self.hsv_params = QWidget()
        hsv_layout = QGridLayout(self.hsv_params)
        hsv_layout.addWidget(QLabel('H权重 (0.5-2.0)'), 0, 0)
        self.spin_h_weight = QDoubleSpinBox()
        self.spin_h_weight.setRange(0.5, 2.0)
        self.spin_h_weight.setSingleStep(0.1)
        self.spin_h_weight.setValue(1.0)
        hsv_layout.addWidget(self.spin_h_weight, 0, 1)

        hsv_layout.addWidget(QLabel('S权重 (0.5-2.0)'), 1, 0)
        self.spin_s_weight = QDoubleSpinBox()
        self.spin_s_weight.setRange(0.5, 2.0)
        self.spin_s_weight.setSingleStep(0.1)
        self.spin_s_weight.setValue(1.0)
        hsv_layout.addWidget(self.spin_s_weight, 1, 1)

        hsv_layout.addWidget(QLabel('V权重 (0.5-2.0)'), 2, 0)
        self.spin_v_weight = QDoubleSpinBox()
        self.spin_v_weight.setRange(0.5, 2.0)
        self.spin_v_weight.setSingleStep(0.1)
        self.spin_v_weight.setValue(1.0)
        hsv_layout.addWidget(self.spin_v_weight, 2, 1)

        ordered_layout.addWidget(self.hsv_params, 3, 0, 1, 2)
        self.hsv_params.setVisible(False)

        dither_layout.addWidget(self.ordered_params)

        # 误差扩散参数区域
        self.floyd_params = QWidget()
        floyd_layout = QGridLayout(self.floyd_params)
        floyd_layout.addWidget(QLabel('alpha (0.0-1.0)'), 0, 0)
        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.0, 1.0)
        self.spin_alpha.setSingleStep(0.05)
        self.spin_alpha.setValue(1.0)
        floyd_layout.addWidget(self.spin_alpha, 0, 1)

        floyd_layout.addWidget(QLabel('滤波核'), 1, 0)
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(['FS', 'Jarvis', 'Stucki'])
        self.combo_filter.setToolTip('FS 最快(4权重)；Jarvis/Stucki 观感略好但更慢')
        floyd_layout.addWidget(self.combo_filter, 1, 1)

        dither_layout.addWidget(self.floyd_params)
        self.floyd_params.setVisible(False)

        # 重心混合抖动参数区域（含可勾选的增强项）
        self.bary_params = QWidget()
        bary_layout = QGridLayout(self.bary_params)
        bary_layout.addWidget(QLabel('混合空间'), 0, 0)
        self.combo_bary_space = QComboBox()
        self.combo_bary_space.addItems(['RGB', 'Lab'])
        self.combo_bary_space.setToolTip('RGB：区域平均精确；Lab：更符合人眼感知')
        bary_layout.addWidget(self.combo_bary_space, 0, 1)
        bary_layout.addWidget(QLabel('阈值矩阵'), 1, 0)
        self.combo_bary_matrix = QComboBox()
        self.combo_bary_matrix.addItems(['bayer', 'clustered', 'diagonal', 'spiral'])
        bary_layout.addWidget(self.combo_bary_matrix, 1, 1)
        bary_layout.addWidget(QLabel('扰动强度'), 2, 0)
        self.spin_bary_strength = QSpinBox()
        self.spin_bary_strength.setRange(0, 128)
        self.spin_bary_strength.setValue(64)
        self.spin_bary_strength.setToolTip('近色/凸包外区域的纹理扰动强度；0=纯重心（这类区域会平涂，但区域色差最小）')
        bary_layout.addWidget(self.spin_bary_strength, 2, 1)

        # --- 增强项（凸包投影控制权重截断/边缘保护的默认值联动） ---
        bary_layout.addWidget(QLabel('凸包投影'), 3, 0)
        self.chk_bary_gamut = QCheckBox('启用')
        self.chk_bary_gamut.setChecked(False)
        self.chk_bary_gamut.setToolTip('凸包外像素投影到凸包表面最近点再混合，改善饱和色区域色差。投影固定走 Lab（与 ΔE76 评估口径一致），混合仍按上面的空间选项；实测权重截断/边缘保护会吃掉这份收益，默认不叠加')
        self.chk_bary_gamut.stateChanged.connect(self._on_gamut_toggled)
        bary_layout.addWidget(self.chk_bary_gamut, 3, 1)
        bary_layout.addWidget(QLabel('权重截断'), 4, 0)
        self.spin_bary_minw = QDoubleSpinBox()
        self.spin_bary_minw.setRange(0.0, 0.5)
        self.spin_bary_minw.setSingleStep(0.05)
        self.spin_bary_minw.setValue(0.0)
        self.spin_bary_minw.setToolTip('丢弃权重低于该值的顶点色再归一化，缓解像素弥散（0=不截断）')
        bary_layout.addWidget(self.spin_bary_minw, 4, 1)
        bary_layout.addWidget(QLabel('边缘保护阈值'), 5, 0)
        self.spin_bary_edge = QSpinBox()
        self.spin_bary_edge.setRange(0, 50)
        self.spin_bary_edge.setValue(0)
        self.spin_bary_edge.setToolTip('Lab色差边缘保护：边缘像素走有序路径保锐度（0=关闭）')
        bary_layout.addWidget(self.spin_bary_edge, 5, 1)
        bary_layout.addWidget(QLabel('局部混合 k'), 6, 0)
        self.spin_bary_localk = QSpinBox()
        self.spin_bary_localk.setRange(0, 12)
        self.spin_bary_localk.setValue(6)
        self.spin_bary_localk.setToolTip(
            '凸包外像素改用「最近的 k 个调色板色」构成的局部凸包来投影，0=用全局凸包（旧行为）。\n'
            '全局凸包只保留最外层约 27 个色，橄榄系全被包在里面，暖黄绿/米白只能靠 '
            'LightGreen+Black 压暗，于是冒出黑点。k=6 时含黑的面从 64% 降到 0.3%，'
            '区域色差只涨约 2%。')
        bary_layout.addWidget(self.spin_bary_localk, 6, 1)
        bary_layout.addWidget(QLabel('混合跨度上限'), 7, 0)
        self.spin_bary_mixspan = QSpinBox()
        self.spin_bary_mixspan.setRange(0, 60)
        self.spin_bary_mixspan.setValue(35)
        self.spin_bary_mixspan.setToolTip(
            "凸包内四面体的顶点亮度跨度上限，0=不限。块平均精确不等于看着干净："
            "LightGreen(L=90) 与 Olive(L=56) 混出 L=78 数值上正确，但那些深点在亮底上"
            "就是黑点。跨度超限的亮区像素退回单一近色。实测好女孩图亮区暗点 11→0、"
            "总暗点 -48%，代价是这张图区域 ΔE 涨 35%（其他图 0~3%）。")
        bary_layout.addWidget(self.spin_bary_mixspan, 7, 1)

        dither_layout.addWidget(self.bary_params)
        self.bary_params.setVisible(False)

        # ---- 分层抖动参数区域 ----
        self.layered_params = QWidget()
        ly = QGridLayout(self.layered_params)

        ly.addWidget(QLabel('预设'), 0, 0)
        self.combo_layer_preset = QComboBox()
        self.combo_layer_preset.addItems(list(_layer_presets().keys()))
        self.combo_layer_preset.setToolTip('复位到标准参数')
        self.combo_layer_preset.currentIndexChanged.connect(self._invalidate_layer_profile)
        ly.addWidget(self.combo_layer_preset, 0, 1)

        ly.addWidget(QLabel('层数'), 1, 0)
        self.spin_layer_bands = QSpinBox()
        self.spin_layer_bands.setRange(2, 64)
        self.spin_layer_bands.setValue(24)
        self.spin_layer_bands.setToolTip('按明暗分几层。层越细，少量投影色就能覆盖得越准（建议 16 起）')
        self.spin_layer_bands.valueChanged.connect(self._invalidate_layer_profile)
        ly.addWidget(self.spin_layer_bands, 1, 1)

        ly.addWidget(QLabel('分层依据'), 1, 2)
        self.combo_layer_edge = QComboBox()
        self.combo_layer_edge.addItems(list(LAYER_EDGE_MODES.keys()))
        self.combo_layer_edge.setToolTip('绝对明度：按明度等分，保色彩更好\n'
                                         '自适应等频：按图像自身明暗分布等频分带，保细节更好')
        self.combo_layer_edge.currentIndexChanged.connect(self._invalidate_layer_profile)
        ly.addWidget(self.combo_layer_edge, 1, 3)

        ly.addWidget(QLabel('每层色数'), 2, 0)
        self.spin_layer_colors = QSpinBox()
        self.spin_layer_colors.setRange(2, 32)
        self.spin_layer_colors.setValue(8)
        self.spin_layer_colors.setToolTip('每层聚出这么多种投影色（可不在调色板内）。\n'
                                          '越大投影越准、色差越低')
        self.spin_layer_colors.valueChanged.connect(self._invalidate_layer_profile)
        ly.addWidget(self.spin_layer_colors, 2, 1)

        ly.addWidget(QLabel('色彩空间'), 2, 2)
        self.combo_layer_space = QComboBox()
        self.combo_layer_space.addItems(LAYER_SPACES)
        self.combo_layer_space.setToolTip('投影聚类与调色板混合所用的色彩空间；Lab 感知更均匀（推荐）')
        self.combo_layer_space.currentIndexChanged.connect(self._invalidate_layer_profile)
        ly.addWidget(self.combo_layer_space, 2, 3)

        btn_row = QHBoxLayout()
        self.btn_layer_edit = QPushButton('编辑分层参数表…')
        self.btn_layer_edit.setToolTip('逐层调抖动图案/阶数/放大 —— 即手工逐层调参的界面')
        self.btn_layer_edit.clicked.connect(self.on_edit_layers)
        btn_row.addWidget(self.btn_layer_edit)
        self.btn_layer_load = QPushButton('载入预设…')
        self.btn_layer_load.clicked.connect(self.on_load_layer_profile)
        btn_row.addWidget(self.btn_layer_load)
        self.btn_layer_save = QPushButton('保存预设…')
        self.btn_layer_save.clicked.connect(self.on_save_layer_profile)
        btn_row.addWidget(self.btn_layer_save)
        ly.addLayout(btn_row, 3, 0, 1, 4)

        self.lbl_layer_state = QLabel('')
        self.lbl_layer_state.setWordWrap(True)
        self.lbl_layer_state.setStyleSheet('color: #888;')
        ly.addWidget(self.lbl_layer_state, 4, 0, 1, 4)

        dither_layout.addWidget(self.layered_params)
        self.layered_params.setVisible(False)

        # ---- 双读抖动参数区域 ----
        self.dual_params = QWidget()
        dv = QGridLayout(self.dual_params)

        dv.addWidget(QLabel('主图 A（远看）'), 0, 0)
        self.btn_dual_main = QPushButton('选择主图…')
        self.btn_dual_main.setToolTip('和左上角的选图是同一件事，放在这里是为了让 A / B 两个入口挨在一起')
        self.btn_dual_main.clicked.connect(self.on_choose_dual_main)
        dv.addWidget(self.btn_dual_main, 0, 1)
        self.lbl_dual_main = QLabel('未选择')
        self.lbl_dual_main.setStyleSheet('color: #888;')
        self.lbl_dual_main.setWordWrap(True)
        dv.addWidget(self.lbl_dual_main, 0, 2, 1, 2)

        dv.addWidget(QLabel('隐藏图 B（近看）'), 1, 0)
        self.btn_dual_hidden = QPushButton('选择隐藏图…')
        self.btn_dual_hidden.setToolTip('近看时看到的那张图')
        self.btn_dual_hidden.clicked.connect(self.on_choose_dual_hidden)
        dv.addWidget(self.btn_dual_hidden, 1, 1)
        self.lbl_dual_hidden = QLabel('未选择')
        self.lbl_dual_hidden.setStyleSheet('color: #888;')
        self.lbl_dual_hidden.setWordWrap(True)
        dv.addWidget(self.lbl_dual_hidden, 1, 2, 1, 2)

        dv.addWidget(QLabel('块尺寸 k'), 2, 0)
        self.spin_dual_k = QSpinBox()
        self.spin_dual_k.setRange(2, 32)
        self.spin_dual_k.setValue(8)
        self.spin_dual_k.setToolTip('块越大隐藏图越清楚，主图分辨率越低（= 输出边长 / k）\n'
                                    'k=8 在 96PPI 屏上约 2.1mm，对应人眼分辨极限约 6 米')
        self.spin_dual_k.valueChanged.connect(self._refresh_dual_state)
        dv.addWidget(self.spin_dual_k, 2, 1)

        dv.addWidget(QLabel('隐藏图级数'), 2, 2)
        self.spin_dual_levels = QSpinBox()
        self.spin_dual_levels.setRange(2, 8)
        self.spin_dual_levels.setValue(2)
        self.spin_dual_levels.setToolTip('隐藏图量化成几档明暗：2 档最清楚，档数越多层次越丰富但越难认')
        dv.addWidget(self.spin_dual_levels, 2, 3)

        dv.addWidget(QLabel('编码方式'), 3, 0)
        self.combo_dual_mode = QComboBox()
        self.combo_dual_mode.addItems(['方向交替（竖/横条纹）', '相位平移（左右）', '相位平移（上下）'])
        self.combo_dual_mode.setToolTip(
            '方向交替：低半段竖条纹、高半段横条纹。条纹方向不会被底图明暗改变，\n'
            '所以插画、照片这类复杂底图也能读出 B。\n'
            '相位平移：靠条纹位置偏移编码，只有底图是大色块、低对比时才读得出来。')
        dv.addWidget(self.combo_dual_mode, 3, 1)

        dv.addWidget(QLabel('色彩空间'), 3, 2)
        self.combo_dual_space = QComboBox()
        self.combo_dual_space.addItems(['RGB', 'Lab'])
        self.combo_dual_space.setToolTip('RGB：块混合色与缩略图严格一致\nLab：按感知匹配，色差更小但缩略图会有微小偏移')
        dv.addWidget(self.combo_dual_space, 3, 3)

        dv.addWidget(QLabel('色对接近度'), 4, 0)
        self.spin_dual_pair = QSpinBox()
        self.spin_dual_pair.setRange(0, 100)
        self.spin_dual_pair.setValue(50)
        self.spin_dual_pair.setToolTip('越高越避免跨色相配对（如品红配绿），远看越干净；\n'
                                       '0 = 只看混合精度，远看可能出现彩色噪点')
        dv.addWidget(self.spin_dual_pair, 4, 1)

        self.chk_dual_level = QCheckBox('隐藏图自动对比度')
        self.chk_dual_level.setChecked(True)
        self.chk_dual_level.setToolTip('先把隐藏图拉伸到全动态范围再量化，避免浅对比的图挤在一档里')
        dv.addWidget(self.chk_dual_level, 4, 2, 1, 2)

        self.btn_dual_view = QPushButton('双视角预览…（并排对照）')
        self.btn_dual_view.setToolTip('出一次图，并给出「近看(1:1)」与「远看(缩略图)」并排对照')
        self.btn_dual_view.clicked.connect(self.on_dual_preview)
        dv.addWidget(self.btn_dual_view, 5, 0, 1, 4)

        self.lbl_dual_state = QLabel('')
        self.lbl_dual_state.setWordWrap(True)
        self.lbl_dual_state.setStyleSheet('color: #888;')
        dv.addWidget(self.lbl_dual_state, 6, 0, 1, 4)

        dither_layout.addWidget(self.dual_params)
        self.dual_params.setVisible(False)

        ctrl_layout.addWidget(dither_group)

        # Save path and actions
        save_layout = QHBoxLayout()
        self.line_save = QLineEdit(self.save_folder)
        save_layout.addWidget(self.line_save)
        btn_browse = QPushButton('选择保存目录')
        btn_browse.clicked.connect(self.on_choose_save)
        save_layout.addWidget(btn_browse)
        ctrl_layout.addLayout(save_layout)

        action_layout = QHBoxLayout()
        self.btn_preview = QPushButton('预览处理结果')
        self.btn_preview.clicked.connect(self.on_preview)
        action_layout.addWidget(self.btn_preview)
        self.btn_save = QPushButton('保存结果')
        self.btn_save.clicked.connect(self.on_save)
        action_layout.addWidget(self.btn_save)
        self.btn_cancel = QPushButton('取消')
        self.btn_cancel.clicked.connect(self.on_cancel)
        self.btn_cancel.setVisible(False)
        action_layout.addWidget(self.btn_cancel)
        ctrl_layout.addLayout(action_layout)

        ctrl_layout.addStretch()

        # 左侧控制项很多（尤其双读抖动的参数面板），窗口不够高时会被压扁
        # 甚至挤到看不见 —— 放进滚动区，保证任何一项都能滚到
        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setMinimumWidth(360)
        ctrl_scroll.setWidget(ctrl_widget)
        main_layout.addWidget(ctrl_scroll, 0)

        # Right: image preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(6, 6, 6, 6)
        preview_layout.addWidget(QLabel('原图 / 处理后预览（单击切换）'))

        # 双读模式专用：四个视角一键切换
        self.dual_view_bar = QWidget()
        vb = QHBoxLayout(self.dual_view_bar)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.addWidget(QLabel('视角'))
        self._dual_view_btns = []
        for name in DUAL_VIEWS:
            b = QPushButton(name)
            b.setCheckable(True)
            b.clicked.connect(partial(self.on_dual_view_select, name))
            vb.addWidget(b)
            self._dual_view_btns.append(b)
        vb.addStretch()
        preview_layout.addWidget(self.dual_view_bar)
        self.dual_view_bar.setVisible(False)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setStyleSheet('QLabel{background: #111; color: white;}')
        self.preview_label.mousePressEvent = self.on_preview_click
        preview_layout.addWidget(self.preview_label)

        main_layout.addWidget(preview_widget, 1)

        # Toggle display mode: 0 original, 1 processed
        self.display_mode = 0

        # Initial state for dither widgets
        self.on_dither_method_change()
        self._refresh_layer_state()
        self._refresh_dual_state()

    def on_dither_method_change(self):
        method = self.combo_dither.currentText()
        for w in (self.ordered_params, self.floyd_params, self.bary_params,
                  self.layered_params, self.dual_params):
            w.setVisible(False)
        self.dual_view_bar.setVisible(False)
        if method == '有序抖动':
            self.ordered_params.setVisible(True)
            # 触发色彩空间变化以更新HSV权重显示
            self.on_color_space_change()
        elif method == '重心混合':
            self.bary_params.setVisible(True)
        elif method == '分层抖动':
            if LAYERED_AVAILABLE:
                self.layered_params.setVisible(True)
                self._refresh_layer_state()
            else:
                self.ordered_params.setVisible(True)
                self.on_color_space_change()
        elif method == '双读抖动':
            if DUALVIEW_AVAILABLE:
                self.dual_params.setVisible(True)
                self.dual_view_bar.setVisible(True)
                self._refresh_dual_state()
            else:
                self.ordered_params.setVisible(True)
                self.on_color_space_change()
        else:
            self.floyd_params.setVisible(True)

    def on_color_space_change(self):
        color_space = self.combo_color_space.currentText()
        if color_space == 'HSV':
            self.hsv_params.setVisible(True)
        else:
            self.hsv_params.setVisible(False)

    # ---------------- 分层抖动 ----------------
    def _invalidate_layer_profile(self, *args):
        """面板参数一变，之前手工编辑过的分层表就作废，改为按预设曲线重新生成。"""
        self._layer_profile = None
        self._layer_profile_edited = False
        self._refresh_layer_state()

    def _make_layer_profile(self):
        """按当前界面参数生成 Profile。"""
        presets = _layer_presets()
        mk = presets.get(self.combo_layer_preset.currentText())
        n = int(self.spin_layer_bands.value())
        K = int(self.spin_layer_colors.value())
        p = mk(n_bands=n, colors_per_band=K) if mk else None
        if p is None:
            return None
        p.match_space = self.combo_layer_space.currentText()
        p.edge_mode = LAYER_EDGE_MODES.get(self.combo_layer_edge.currentText(), 'linear')
        return p

    def current_layer_profile(self):
        """取当前生效的 Profile：手工编辑过的优先，否则按预设现场生成。"""
        if not LAYERED_AVAILABLE:
            return None
        if self._layer_profile is None:
            self._layer_profile = self._make_layer_profile()
        if self._layer_profile is None:
            return None
        # 色彩空间 / 分层依据始终以界面为准
        self._layer_profile.match_space = self.combo_layer_space.currentText()
        self._layer_profile.edge_mode = LAYER_EDGE_MODES.get(
            self.combo_layer_edge.currentText(), 'linear')
        return self._layer_profile

    def _refresh_layer_state(self):
        if not LAYERED_AVAILABLE:
            self.lbl_layer_state.setText('未找到 layered_dither.py，分层抖动不可用')
            return
        if not self._layer_profile_edited or self._layer_profile is None:
            title = '参数按当前设置生成；点「编辑分层参数表…」可逐层改抖动图案'
            prof = self._make_layer_profile()
        else:
            p = self._layer_profile
            title = (f'已手工编辑：共 {len(p.bands)} 层，'
                     f'启用 {sum(1 for b in p.bands if b.enabled)} 层')
            prof = p
        if prof is not None:
            title += (f'\n每层 {prof.colors_per_band} 个投影色逐个用调色板色混合，'
                      f'输出全部落在调色板内')
            # clustered 的环状阈值会被凸包混合原样映射成方块色团（暗面聚团）
            n_clu = sum(1 for b in prof.bands if b.enabled and b.pattern == 'clustered')
            if n_clu:
                title += f'\n注意：有 {n_clu} 层用 clustered，容易聚成方块色团，建议改 bayer / bluenoise'
        self.lbl_layer_state.setText(title)

    # ---------------- 双读抖动 ----------------
    def on_choose_dual_main(self):
        """选主图 A —— 与左上角的选图是同一件事，这里只是把两个入口并排放。"""
        path, _ = QFileDialog.getOpenFileName(
            self, '选择主图 (A)', '', '图片 (*.png *.jpg *.jpeg *.bmp *.webp *.tif)')
        if not path:
            return
        self.load_image_from_path(path)
        self._refresh_dual_state()

    def on_choose_dual_hidden(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择隐藏图 (B)', '', '图片 (*.png *.jpg *.jpeg *.bmp *.webp *.tif)')
        if not path:
            return
        try:
            img = Image.open(path)
            img.load()
            self.dual_hidden_image = img.convert('RGB')
        except Exception as e:
            QMessageBox.warning(self, '打开失败', f'无法打开隐藏图: {e}')
            return
        self.lbl_dual_hidden.setText(os.path.basename(path))
        self.lbl_dual_hidden.setStyleSheet('color: #ddd;')
        self._refresh_dual_state()

    def _make_dual_profile(self):
        if not DUALVIEW_AVAILABLE:
            return None
        mode = self.combo_dual_mode.currentIndex()
        return dvd.DualViewProfile(
            k=int(self.spin_dual_k.value()),
            levels=int(self.spin_dual_levels.value()),
            axis='y' if mode == 2 else 'x',
            orient=(mode == 0),
            match_space=self.combo_dual_space.currentText(),
            auto_level=self.chk_dual_level.isChecked(),
            pair_prefer=float(self.spin_dual_pair.value()),
        )

    def _refresh_dual_state(self, *args):
        if not DUALVIEW_AVAILABLE:
            self.lbl_dual_state.setText('未找到 dualview_dither.py，双读抖动不可用')
            return
        k = int(self.spin_dual_k.value())
        # 主图标签跟随实际加载的图（左上角选图 / 拖拽 / 粘贴都会走到这里）
        if self.original_image is None:
            self.lbl_dual_main.setText('未选择')
            self.lbl_dual_main.setStyleSheet('color: #888;')
        else:
            ow, oh = self.original_image.size
            name = os.path.basename(self.image_path) if self.image_path else '已加载'
            self.lbl_dual_main.setText(f'{name}  {ow}×{oh}')
            self.lbl_dual_main.setStyleSheet('color: #ddd;')

        lines = [f'远看 = 主图 A，近看 = 隐藏图 B；输出边长需为 k={k} 的整数倍（不足的边裁掉）。']
        if self.dual_hidden_image is None:
            lines.append('还没选隐藏图 B —— 点上面「选择隐藏图…」')
        elif self.original_image is not None:
            ow, oh = self.original_image.size
            lines.append(f'隐藏图网格 {max(1, ow // k)}×{max(1, oh // k)}')
        lines.append('生成后右侧会出现「视角」按钮，可切 原始 A / 隐藏图 B / 合成·远看 / 合成·近看 1:1。')
        lines.append('提示：隐藏图用文字、图标、剪影这类大色块图形最好读；'
                     '复杂照片当 B 会变成噪声点。')
        self.lbl_dual_state.setText('\n'.join(lines))

    # ---- 合成结果的双视角派生与切换 ----
    def _build_dual_views(self):
        """从合成结果派生「远看」「近看」两张视图，供右侧预览切换。"""
        self._dual_view_cache = {}
        out = self.processed_image
        if out is None or not DUALVIEW_AVAILABLE:
            return
        try:
            k = int(self.spin_dual_k.value())
            near, far = dvd.preview_pair(out, k=k, crop=DUAL_CROP)
            self._dual_view_cache = {'near': near, 'far': far}
        except Exception:
            self._dual_view_cache = {}

    def on_dual_view_select(self, name):
        self._dual_view_active = name
        for b in self._dual_view_btns:
            b.setChecked(b.text() == name)
        self._show_dual_view()

    def _show_dual_view(self):
        name = self._dual_view_active
        src = None
        if name is None or name == '原始 A':
            src = self.original_image
        elif name == '隐藏图 B':
            src = self.dual_hidden_image
        elif name.startswith('合成·远看'):
            src = self._dual_view_cache.get('far')
        elif name.startswith('合成·近看'):
            src = self._dual_view_cache.get('near')
        if src is None:
            need = '请先选择隐藏图 B' if name == '隐藏图 B' else '请先点「预览处理结果」生成合成图'
            self.statusBar().showMessage(f'{name}：还没有内容 —— {need}')
            return
        self.show_image_in_label(src)
        hint = {
            '原始 A': '主图 A 本身 —— 这是远看时该有的样子',
            '隐藏图 B': '隐藏图 B —— 这是近看时该出现的内容',
            '合成·远看 → A': '合成结果缩到 1/k 再放大，等于缩略图/退后看 —— 应该像 A',
            '合成·近看 1:1 → B': '合成结果的中心 1:1 局部 —— 贴近屏幕应该能读出 B',
        }.get(name)
        if hint:
            self.statusBar().showMessage(hint)

    def on_dual_preview(self):
        if not DUALVIEW_AVAILABLE:
            return
        if self.original_image is None:
            QMessageBox.information(self, '提示', '请先选择主图 (A)')
            return
        if self.dual_hidden_image is None:
            QMessageBox.information(self, '提示', '请先选择隐藏图 (B)')
            return
        pal = self.build_palette_list()
        if not pal:
            QMessageBox.information(self, '提示', '调色盘为空，请检查 palette.txt 或选择颜色')
            return
        prof = self._make_dual_profile()
        try:
            img = self._resize_image(self.original_image, self._capture_resize_params())
            if img is None:
                raise RuntimeError('图片为空')
            if max(img.size) > 1400:
                s = 1400 / max(img.size)
                img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))),
                                 Image.LANCZOS)
            out = dvd.render(img, pal, prof, hidden=self.dual_hidden_image, out_size=img.size)
            near, far = dvd.preview_pair(out, k=prof.k, crop=560)
        except Exception as e:
            QMessageBox.warning(self, '预览失败', f'双读抖动失败: {e}')
            return
        DualViewDialog(self.pil_image_to_qpixmap(near),
                       self.pil_image_to_qpixmap(far), prof, self).exec_()

    def on_edit_layers(self):
        p = self.current_layer_profile()
        if p is None:
            return
        ratios = None
        if self.original_image is not None:
            try:
                info = ldl.analyze_image(self.original_image, len(p.bands), p.space, p.edge_mode)
                ratios = info['ratios']
            except Exception:
                ratios = None
        dlg = LayerTableDialog(p, ratios, self,
                               palette=self.build_palette_list(),
                               image=self.original_image,
                               preview_cb=self._layer_preview)
        if dlg.exec_() == QDialog.Accepted:
            self._layer_profile = p
            self._layer_profile_edited = True
            self._refresh_layer_state()
            self.statusBar().showMessage('分层参数已更新')

    def _layer_preview(self, index):
        """给分层参数表用的单层预览：返回 (原色层, 抖动层)。"""
        p = self.current_layer_profile()
        if p is None or self.original_image is None:
            raise RuntimeError('缺少分层预设或图片')
        pal = self.build_palette_list()
        img = self.original_image
        img = self._cap_pixels(img, getattr(self, 'preview_cap', 0) or 0)
        return ldl.render_band_preview(img, pal, p, index)

    def on_save_layer_profile(self):
        p = self.current_layer_profile()
        if p is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, '保存分层预设', 'layered_preset.json', 'JSON (*.json)')
        if path:
            try:
                p.to_json(path)
                self.statusBar().showMessage(f'预设已保存：{path}')
            except Exception as e:
                QMessageBox.warning(self, '保存失败', str(e))

    def on_load_layer_profile(self):
        if not LAYERED_AVAILABLE:
            return
        path, _ = QFileDialog.getOpenFileName(self, '载入分层预设', '', 'JSON (*.json)')
        if not path:
            return
        try:
            p = ldl.Profile.from_json(path)
        except Exception as e:
            QMessageBox.warning(self, '载入失败', str(e))
            return
        widgets = (self.spin_layer_bands, self.combo_layer_edge,
                   self.combo_layer_space, self.combo_layer_preset,
                   self.spin_layer_colors)
        for w in widgets:
            w.blockSignals(True)
        self.spin_layer_bands.setValue(len(p.bands))
        for label, mode in LAYER_EDGE_MODES.items():
            if mode == p.edge_mode:
                self.combo_layer_edge.setCurrentText(label)
        if getattr(p, 'colors_per_band', None):
            self.spin_layer_colors.setValue(int(p.colors_per_band))
        if str(p.match_space).upper() in ('RGB', 'LAB'):
            self.combo_layer_space.setCurrentText(
                'RGB' if str(p.match_space).upper() == 'RGB' else 'Lab')
        for w in widgets:
            w.blockSignals(False)
        self._layer_profile = p
        self._layer_profile_edited = True
        self._refresh_layer_state()
        self.statusBar().showMessage(f'已载入分层预设（{len(p.bands)} 层）')

    def _on_gamut_toggled(self, state):
        """凸包投影勾选切换：把当前 权重截断/边缘保护 值存进旧状态的配置，
        再载入新状态对应的那套值。两套值分别保存、互不覆盖。"""
        new_state = (state == Qt.Checked)
        old_key = 'on' if self._gamut_state else 'off'
        new_key = 'on' if new_state else 'off'
        # 保存当前值到旧状态配置
        self._minw_cfg[old_key] = self.spin_bary_minw.value()
        self._edge_cfg[old_key] = self.spin_bary_edge.value()
        # 载入新状态配置
        self.spin_bary_minw.setValue(self._minw_cfg[new_key])
        self.spin_bary_edge.setValue(self._edge_cfg[new_key])
        self._gamut_state = new_state

    def build_palette_list(self):
        # Get selected colors from both panels
        free_selected = self.free_panel.get_selected_palette()
        paid_selected = self.paid_panel.get_selected_palette()
        selected = free_selected + paid_selected

        # If no colors are selected, use all colors
        if not selected:
            selected = [rgb for _, rgb, _ in self.palette_items]

        return selected

    def on_open_image(self):
        p, _ = QFileDialog.getOpenFileName(self, '打开图片', '', 'Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif)')
        if p:
            self.load_image_from_path(p)

    def load_image_from_path(self, path):
        # 大图解码很耗时，放到后台线程，避免阻塞界面
        self.lbl_file.setText('正在加载图片…')
        self.statusBar().showMessage('正在加载图片…')
        self._run_worker(self._load_task, (path,), self._on_image_loaded, self._on_image_failed)

    def _load_task(self, path):
        img = Image.open(path)
        img.load()  # 确保数据都读入，避免延迟加载导致的问题
        # 尽量保留 alpha（如果存在），否则至少转为 RGB/L
        bands = img.getbands()
        if 'A' in bands:
            img = img.convert('RGBA')
        elif img.mode == 'P':
            # 事先把调色板图像转为 RGBA（可保留透明），没有透明也不会坏
            img = img.convert('RGBA')
        elif img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        # 复制一份到内存，断开与文件句柄的关联（更安全）
        return (path, img.copy())

    def _on_image_loaded(self, result):
        path, img = result
        self.image_path = path
        self.original_image = img
        self.processed_image = None
        self._dual_view_cache = {}
        self.lbl_file.setText(os.path.basename(path))
        self.display_mode = 0
        self.show_image_in_label(self.original_image)
        self.statusBar().showMessage('图片加载完成')
        try:
            self._refresh_dual_state()
        except Exception:
            pass

    def _on_image_failed(self, err):
        self.lbl_file.setText('未选择文件')
        self.statusBar().clearMessage()
        QMessageBox.warning(self, '打开失败', f'无法打开图片: {err}')

    # 收集当前界面上的抖动参数（在主线程调用）
    def _gather_settings(self):
        settings = {'method': self.combo_dither.currentText()}
        if settings['method'] == '有序抖动':
            settings['strength'] = int(self.spin_strength.value())
            settings['color_space'] = self.combo_color_space.currentText()
            settings['matrix'] = self.combo_matrix.currentText()
            if settings['color_space'] == 'HSV':
                settings['weights'] = (
                    float(self.spin_h_weight.value()),
                    float(self.spin_s_weight.value()),
                    float(self.spin_v_weight.value()),
                )
        elif settings['method'] == '重心混合':
            settings['space'] = self.combo_bary_space.currentText()
            settings['matrix'] = self.combo_bary_matrix.currentText()
            settings['strength'] = int(self.spin_bary_strength.value())
            settings['gamut'] = self.chk_bary_gamut.isChecked()
            settings['min_weight'] = float(self.spin_bary_minw.value())
            settings['edge_guard'] = int(self.spin_bary_edge.value())
            settings['proj_local_k'] = int(self.spin_bary_localk.value())
            settings['mix_span_max'] = float(self.spin_bary_mixspan.value())
        elif settings['method'] == '分层抖动':
            settings['profile'] = self.current_layer_profile()
        elif settings['method'] == '双读抖动':
            settings['dual'] = self._make_dual_profile()
            settings['hidden'] = self.dual_hidden_image
        else:
            settings['alpha'] = float(self.spin_alpha.value())
            settings['filter'] = self.combo_filter.currentText()
        settings['resize'] = self._capture_resize_params()
        return settings

    # 后台线程里的实际处理任务：先缩放，再抖动
    # for_preview=True 时对大图先缩到预览尺寸（保持界面流畅）；for_preview=False 按完整尺寸处理
    def _dither_task(self, image, palette, settings, for_preview):
        img = self._resize_image(image, settings['resize'])
        if img is None:
            raise RuntimeError('图片为空')
        if for_preview:
            cap = self.preview_cap
            if settings['method'] == '误差扩散抖动':
                cap = 1_000_000  # 误差扩散是串行算法，预览用更低分辨率保证流畅
            img = self._cap_pixels(img, cap)
        if settings['method'] == '重心混合':
            return barycentric_dither(
                img, palette,
                method=settings['matrix'],
                space=settings.get('space', 'RGB'),
                strength=settings.get('strength', 64),
                gamut=settings.get('gamut', False),
                min_weight=settings.get('min_weight', 0.0),
                edge_guard=settings.get('edge_guard', 0),
                proj_local_k=settings.get('proj_local_k', 6),
                mix_span_max=settings.get('mix_span_max', 35.0),
            )
        elif settings['method'] == '有序抖动':
            return ordered_dithering(
                img, palette,
                method=settings['matrix'],
                strength=settings['strength'],
                color_space=settings['color_space'],
                weights=settings.get('weights', (1.0, 1.0, 1.0)),
            )
        elif settings['method'] == '分层抖动':
            prof = settings.get('profile')
            if prof is None:
                raise RuntimeError('分层抖动不可用：请确认 layered_dither.py 与 GUI.py 同目录')
            # 深拷贝：处理在后台线程跑，期间用户可能又在界面上改参数
            return run_layered_dither(img, palette, copy.deepcopy(prof))
        elif settings['method'] == '双读抖动':
            prof = settings.get('dual')
            hidden = settings.get('hidden')
            if prof is None:
                raise RuntimeError('双读抖动不可用：请确认 dualview_dither.py 与 GUI.py 同目录')
            if hidden is None:
                raise RuntimeError('双读抖动需要一张隐藏图 (B)，请点「选择图片…」')
            # out_size 用当前处理尺寸：输出与主图同尺寸，内部按 k×k 切块
            return dvd.render(img, palette, copy.deepcopy(prof), hidden=hidden,
                              out_size=img.size)
        else:
            return floyd_steinberg_dither(img, palette, alpha_strength=settings['alpha'],
                                          filter_type=settings.get('filter', 'FS'))

    def _on_process_done(self, out):
        self.processed_image = out
        self.display_mode = 1
        if self.combo_dither.currentText() == '双读抖动':
            # 双读结果在右侧按默认缩放看就是「远看」，B 看不见；
            # 所以出图后直接切到 1:1 局部，让 B 立刻可见
            self._build_dual_views()
            self.on_dual_view_select('合成·近看 1:1 → B')
        else:
            self.show_image_in_label(self.processed_image)
        self._set_processing(False)
        if self._was_preview:
            self.statusBar().showMessage('预览已生成（大图为缩略图近似，保存时按完整尺寸处理）')
        else:
            self.statusBar().showMessage('处理完成')

    def _on_process_failed(self, err):
        self._set_processing(False)
        self.statusBar().clearMessage()
        QMessageBox.warning(self, '处理失败', f'应用抖动失败: {err}')

    def _set_processing(self, active):
        self.btn_preview.setEnabled(not active)
        self.btn_save.setEnabled(not active)
        self.btn_cancel.setVisible(active)
        if active:
            self.statusBar().showMessage('正在处理…')

    def on_cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None  # 允许立即开始新的处理
        self._set_processing(False)
        self.statusBar().showMessage('已取消')

    # 通用后台任务启动器
    def _run_worker(self, func, args, on_done, on_fail=None):
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
        worker = ProcessWorker(func, args, self)
        worker.finished.connect(on_done)
        worker.failed.connect(on_fail if on_fail is not None else self._on_process_failed)
        worker.finished.connect(self._on_worker_done)
        worker.failed.connect(self._on_worker_done)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_worker_done(self, *args):
        self._worker = None

    def pil_image_to_qpixmap(self, pil_img):
        """把 PIL Image 转为 QPixmap；对多种模式做兼容处理（RGBA, RGB, L 等）。

        关键：QImage 必须显式传入 bytesPerLine = w * bpp。
        PIL 的 tobytes('raw', ...) 每行是不对齐的，若让 Qt 按对齐后的
        bytesPerLine 读取，奇数宽度的行会产生越界读，导致段错误。
        """
        if pil_img is None:
            return QPixmap()

        try:
            # 选择模式对应的格式和原始字节布局
            if pil_img.mode == 'RGBA':
                if hasattr(QImage, 'Format_RGBA8888'):
                    fmt, raw, bpp = QImage.Format_RGBA8888, 'RGBA', 4
                else:
                    fmt, raw, bpp = QImage.Format_ARGB32, 'BGRA', 4
            elif pil_img.mode == 'RGB':
                fmt, raw, bpp = QImage.Format_RGB888, 'RGB', 3
            elif pil_img.mode == 'L':
                fmt = QImage.Format_Grayscale8 if hasattr(QImage, 'Format_Grayscale8') else QImage.Format_Indexed8
                raw, bpp = 'L', 1
            else:
                # 其它模式（例如 CMYK 等）先转换为 RGBA 再处理
                pil_img = pil_img.convert('RGBA')
                if hasattr(QImage, 'Format_RGBA8888'):
                    fmt, raw, bpp = QImage.Format_RGBA8888, 'RGBA', 4
                else:
                    fmt, raw, bpp = QImage.Format_ARGB32, 'BGRA', 4

            w, h = pil_img.size
            data = pil_img.tobytes('raw', raw)
            qimg = QImage(data, w, h, w * bpp, fmt)  # 显式 bytesPerLine，避免对齐越界
            return QPixmap.fromImage(qimg)
        except Exception as e:
            print("pil->qpixmap 转换失败：", e)
            return QPixmap()

    @staticmethod
    def _cap_pixels(img, max_pixels):
        """按**总像素数**限制尺寸，而不是长边。

        条漫这类极端长宽比（如 462x10800）如果按长边缩放，宽度会被压到几十像素；
        按面积限制则与普通大图一视同仁。
        """
        if img is None or max_pixels <= 0:
            return img
        w, h = img.size
        if w * h <= max_pixels:
            return img
        s = math.sqrt(max_pixels / float(w * h))
        return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)

    def _make_display_pixmap(self, pil_img, max_pixels=4_000_000):
        """把 PIL 图像缩到适合显示的尺寸再转 QPixmap，避免大图占用海量内存/阻塞界面。"""
        if pil_img is None:
            return QPixmap()
        return self.pil_image_to_qpixmap(self._cap_pixels(pil_img, max_pixels))

    def show_image_in_label(self, pil_img):
        # Convert PIL image to QPixmap robustly, then set scaled pixmap to label
        if pil_img is None:
            self.preview_label.clear()
            return

        pix = self._make_display_pixmap(pil_img)
        if pix is None or pix.isNull():
            self.preview_label.clear()
            self.preview_label.setText("无法显示图片")
            return

        scaled = pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # refresh preview
        try:
            if self.display_mode == 0 and self.original_image:
                self.show_image_in_label(self.original_image)
            elif self.display_mode == 1 and self.processed_image:
                self.show_image_in_label(self.processed_image)
        except Exception:
            pass

    def on_slider_change(self, val):
        self.spin_percent.blockSignals(True)
        self.spin_percent.setValue(val)
        self.spin_percent.blockSignals(False)

    def on_spin_percent_change(self, val):
        self.slider.blockSignals(True)
        self.slider.setValue(val)
        self.slider.blockSignals(False)

    def on_manual_wh(self):
        # user pressed enter on width/height: set slider appropriately if original image exists
        if not self.original_image:
            return
        w_text = self.line_w.text().strip()
        h_text = self.line_h.text().strip()
        try:
            ow, oh = self.original_image.size
            if w_text and h_text:
                nw = int(w_text)
                nh = int(h_text)
                pct = int(min(100, max(1, round(min(nw / ow, nh / oh) * 100))))
                self.spin_percent.setValue(pct)
            elif w_text:
                nw = int(w_text)
                pct = int(min(100, max(1, round(nw / ow * 100))))
                self.spin_percent.setValue(pct)
            elif h_text:
                nh = int(h_text)
                pct = int(min(100, max(1, round(nh / oh * 100))))
                self.spin_percent.setValue(pct)
        except Exception:
            pass

    def on_choose_save(self):
        p = QFileDialog.getExistingDirectory(self, '选择保存目录', self.save_folder)
        if p:
            self.save_folder = p
            self.line_save.setText(p)

    def _capture_resize_params(self):
        """在主线程采集缩放参数，供后台线程使用（避免跨线程读控件）。"""
        return (self.line_w.text().strip(), self.line_h.text().strip(), self.spin_percent.value())

    def _resize_image(self, image: Image.Image, params):
        """纯函数：按参数缩放图片。params = (w_text, h_text, pct)。"""
        if image is None:
            return None
        ow, oh = image.size
        w_text, h_text, pct = params
        if w_text or h_text:
            try:
                if w_text and h_text:
                    nw, nh = int(w_text), int(h_text)
                elif w_text:
                    nw = int(w_text)
                    nh = int(round(oh * (nw / ow)))
                else:
                    nh = int(h_text)
                    nw = int(round(ow * (nh / oh)))
                if nw <= 0 or nh <= 0:
                    return image
                return image.resize((nw, nh), Image.LANCZOS)
            except Exception:
                return image
        if pct == 100:
            return image
        scale = pct / 100.0
        nw = max(1, int(round(ow * scale)))
        nh = max(1, int(round(oh * scale)))
        return image.resize((nw, nh), Image.LANCZOS)

    def apply_resize(self, image: Image.Image) -> Image.Image:
        return self._resize_image(image, self._capture_resize_params())

    def on_preview(self):
        if not self.original_image:
            QMessageBox.information(self, '提示', '请先选择图片')
            return
        if self._worker is not None and self._worker.isRunning():
            return  # 正在处理，忽略重复点击
        pal = self.build_palette_list()
        if not pal:
            QMessageBox.information(self, '提示', '调色盘为空，请检查 palette.txt 或选择颜色')
            return
        # 缩放+抖动都放到后台线程执行，避免大图阻塞界面
        settings = self._gather_settings()
        self._was_preview = True
        self._set_processing(True)
        self._run_worker(self._dither_task, (self.original_image, pal, settings, True), self._on_process_done)

    def on_save(self):
        if not self.original_image:
            QMessageBox.information(self, '提示', '请先选择图片')
            return
        if self._worker is not None and self._worker.isRunning():
            return
        pal = self.build_palette_list()
        if not pal:
            QMessageBox.information(self, '提示', '调色盘为空，请检查 palette.txt 或选择颜色')
            return
        # 保存时始终按完整尺寸处理（不缩略图），放到后台线程执行
        settings = self._gather_settings()
        self._was_preview = False
        self._set_processing(True)
        self._run_worker(self._dither_task, (self.original_image, pal, settings, False), self._on_save_done)

    def _on_save_done(self, out):
        self.processed_image = out
        if self.combo_dither.currentText() == '双读抖动':
            self._build_dual_views()
        folder = self.line_save.text().strip() or self.save_folder
        os.makedirs(folder, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.image_path or 'result'))[0]
        # find a non-conflicting filename
        i = 0
        while True:
            fname = f"{base}_dithered_{i}.png" if i else f"{base}_dithered.png"
            path = os.path.join(folder, fname)
            if not os.path.exists(path):
                break
            i += 1
        try:
            self.processed_image.save(path)
            self._set_processing(False)
            self.statusBar().showMessage('已保存')
            QMessageBox.information(self, '已保存', f'已保存到 {path}')
        except Exception as e:
            self._set_processing(False)
            self.statusBar().clearMessage()
            QMessageBox.warning(self, '保存失败', f'保存失败: {e}')

    def on_preview_click(self, event):
        # toggle between original and processed
        if self.display_mode == 0:
            if self.processed_image:
                self.display_mode = 1
                self.show_image_in_label(self.processed_image)
        else:
            if self.original_image:
                self.display_mode = 0
                self.show_image_in_label(self.original_image)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ImageDitherApp()
    win.show()
    sys.exit(app.exec_())