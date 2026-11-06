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


# 分层预设：界面名 → 生成函数(n_bands, colors_per_band)
def _layer_presets():
    if not LAYERED_AVAILABLE:
        return {}
    return {
        '标准（投影混色·落盘内）': lambda n_bands, colors_per_band=8:
            ldl.make_projection_profile(n_bands=n_bands, colors_per_band=colors_per_band),
    }


# 选预设时联动这些参数：每层色数、网点图案、阶数
LAYER_PRESET_DEFAULTS = {
    '标准（投影混色·落盘内）': (8, 'bayer', 4),
}


LAYER_PATTERNS = ('bayer', 'clustered', 'rounddot', 'diagonal', 'spiral',
                  'checker', 'noise', 'bluenoise')
LAYER_EDGE_MODES = {
    '绝对明度（保色彩）': 'linear',
    '自适应等频（保细节）': 'quantile',
    '混合': 'hybrid',
}
LAYER_SPACES = ['Lab', 'RGB']
LAYER_SIZES = [
    (2, '2×2（4 级阈值 · 色差 +77%，风格取舍）'),
    (4, '4×4（16 级 · 性价比最高，上限）'),
]
LAYER_MIX_MODES = ['4 色（重心混合，无损）', '3 色（三角形）', '2 色（线段）']
LAYER_MIX_VALUES = [4, 3, 2]


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
        self.combo_layer_preset.currentIndexChanged.connect(self.on_layer_preset_change)
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

        ly.addWidget(QLabel('网点图案'), 3, 0)
        self.combo_layer_pattern = QComboBox()
        self.combo_layer_pattern.addItems(LAYER_PATTERNS)
        self.combo_layer_pattern.setToolTip(
            'bayer：阈值均匀交错，边缘锐利但纹理细碎；'
            'rounddot：圆形网点，聚集成点、画面干净（参考图风格）；'
            '这里改会应用到所有层；网页版分层编辑器可逐层单独设定')
        self.combo_layer_pattern.currentIndexChanged.connect(self._invalidate_layer_profile)
        ly.addWidget(self.combo_layer_pattern, 3, 1)

        ly.addWidget(QLabel('图案单元'), 3, 2)
        self.combo_layer_size = QComboBox()
        for _v, _label in LAYER_SIZES:
            self.combo_layer_size.addItem(_label, _v)
        self.combo_layer_size.setCurrentIndex(1)          # 默认 4×4
        self.combo_layer_size.setToolTip(
            '图案单元越小，抖动可用的阈值档位越少：2×2 只有 4 档，区域平均色必然偏'
            '（实测 ΔE 1.51，比 8×8 差 77%），加层数也补不回来。4×4 就基本够（0.865），'
            '8×8 以上无收益。圆点图案别超过 8，否则一个点跨越多个感知块')
        self.combo_layer_size.currentIndexChanged.connect(self._invalidate_layer_profile)
        ly.addWidget(self.combo_layer_size, 3, 3)

        ly.addWidget(QLabel('混合色数'), 4, 0)
        self.combo_layer_mix = QComboBox()
        self.combo_layer_mix.addItems(LAYER_MIX_MODES)
        self.combo_layer_mix.setToolTip(
            '每个投影色用几个调色板色混出来。4 色 = 四面体重心，凸包内无损；'
            '3 色 = 三角形；2 色 = 线段。少色更"纯粹"，但端点常离目标很远、'
            '要用异色补偿（近白会掺深橄榄），实测区域平均 ΔE 从 0.85 升到 '
            '2.66（3 色）/ 3.51（2 色）')
        self.combo_layer_mix.currentIndexChanged.connect(self._invalidate_layer_profile)
        ly.addWidget(self.combo_layer_mix, 4, 1)

        btn_row = QHBoxLayout()
        self.btn_layer_load = QPushButton('载入预设…')
        self.btn_layer_load.clicked.connect(self.on_load_layer_profile)
        btn_row.addWidget(self.btn_layer_load)
        self.btn_layer_save = QPushButton('保存预设…')
        self.btn_layer_save.clicked.connect(self.on_save_layer_profile)
        btn_row.addWidget(self.btn_layer_save)
        ly.addLayout(btn_row, 5, 0, 1, 4)

        self.lbl_layer_state = QLabel('')
        self.lbl_layer_state.setWordWrap(True)
        self.lbl_layer_state.setStyleSheet('color: #888;')
        ly.addWidget(self.lbl_layer_state, 6, 0, 1, 4)

        dither_layout.addWidget(self.layered_params)
        self.layered_params.setVisible(False)

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

        # 左侧控制项很多（尤其分层抖动的参数面板），窗口不够高时会被压扁
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

    def on_dither_method_change(self):
        method = self.combo_dither.currentText()
        for w in (self.ordered_params, self.floyd_params, self.bary_params,
                  self.layered_params):
            w.setVisible(False)
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
        return self._apply_pattern(p)

    def _apply_pattern(self, p):
        """把面板上的网点图案/阶数写到所有层。"""
        if p is None or not hasattr(self, 'combo_layer_pattern'):
            return p
        pat = self.combo_layer_pattern.currentText()
        size = int(self.combo_layer_size.currentData() or 4)
        for b in p.bands:
            b.pattern = pat
            b.size = size
        if hasattr(self, 'combo_layer_mix'):
            p.mix_colors = LAYER_MIX_VALUES[self.combo_layer_mix.currentIndex()]
        return p

    def on_layer_preset_change(self):
        """切预设时把「每层色数 / 网点图案 / 阶数」一并联动过去。"""
        d = LAYER_PRESET_DEFAULTS.get(self.combo_layer_preset.currentText())
        ws = (self.spin_layer_colors, self.combo_layer_pattern, self.combo_layer_size)
        if d:
            K, pat, size = d
            for w in ws:
                w.blockSignals(True)
            self.spin_layer_colors.setValue(K)
            if self.combo_layer_pattern.findText(pat) >= 0:
                self.combo_layer_pattern.setCurrentText(pat)
            _i = self.combo_layer_size.findData(size)
            self.combo_layer_size.setCurrentIndex(_i if _i >= 0 else 1)
            for w in ws:
                w.blockSignals(False)
        self._invalidate_layer_profile()

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
        return self._apply_pattern(self._layer_profile)

    def _refresh_layer_state(self):
        if not LAYERED_AVAILABLE:
            self.lbl_layer_state.setText('未找到 layered_dither.py，分层抖动不可用')
            return
        if not self._layer_profile_edited or self._layer_profile is None:
            title = '参数按当前设置生成；逐层调整请在网页版分层编辑器里做'
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
        self.lbl_file.setText(os.path.basename(path))
        self.display_mode = 0
        self.show_image_in_label(self.original_image)
        self.statusBar().showMessage('图片加载完成')

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
        else:
            return floyd_steinberg_dither(img, palette, alpha_strength=settings['alpha'],
                                          filter_type=settings.get('filter', 'FS'))

    def _on_process_done(self, out):
        self.processed_image = out
        self.display_mode = 1
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