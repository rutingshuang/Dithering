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
from functools import partial

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QLocale
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QMainWindow, QFileDialog, QPushButton,
    QHBoxLayout, QVBoxLayout, QGridLayout, QSlider, QLineEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QScrollArea, QMessageBox, QSizePolicy, QGroupBox, QCheckBox
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
        self.preview_cap = 1600  # 预览时处理的最大边长(px)，大图用缩略图保证流畅
        self._was_preview = False  # 最近一次处理是预览还是保存

        # 重心混合：凸包投影 开/关 各存一套 权重截断/边缘保护 的值，勾选切换时互换
        self._minw_cfg = {'off': 0.0, 'on': 0.1}
        self._edge_cfg = {'off': 0, 'on': 10}
        self._gamut_state = False

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
        self.combo_dither.addItems(['有序抖动', '误差扩散抖动', '重心混合'])
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
        self.chk_bary_gamut.setToolTip('凸包外像素投影到凸包表面最近点再混合，改善饱和色区域色差。勾选后权重截断/边缘保护自动设为 0.1/10；取消则恢复之前的值')
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

        dither_layout.addWidget(self.bary_params)
        self.bary_params.setVisible(False)

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

        main_layout.addWidget(ctrl_widget, 0)

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

    def on_dither_method_change(self):
        method = self.combo_dither.currentText()
        if method == '有序抖动':
            self.ordered_params.setVisible(True)
            self.floyd_params.setVisible(False)
            self.bary_params.setVisible(False)
            # 触发色彩空间变化以更新HSV权重显示
            self.on_color_space_change()
        elif method == '重心混合':
            self.ordered_params.setVisible(False)
            self.floyd_params.setVisible(False)
            self.bary_params.setVisible(True)
        else:
            self.ordered_params.setVisible(False)
            self.floyd_params.setVisible(True)
            self.bary_params.setVisible(False)

    def on_color_space_change(self):
        color_space = self.combo_color_space.currentText()
        if color_space == 'HSV':
            self.hsv_params.setVisible(True)
        else:
            self.hsv_params.setVisible(False)

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
        cap = self.preview_cap
        if for_preview and settings['method'] == '误差扩散抖动':
            cap = 1000  # 误差扩散是串行算法，预览用更低分辨率保证流畅
        if for_preview and max(img.size) > cap:
            s = cap / max(img.size)
            img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
        if settings['method'] == '重心混合':
            return barycentric_dither(
                img, palette,
                method=settings['matrix'],
                space=settings.get('space', 'RGB'),
                strength=settings.get('strength', 64),
                gamut=settings.get('gamut', False),
                min_weight=settings.get('min_weight', 0.0),
                edge_guard=settings.get('edge_guard', 0),
            )
        elif settings['method'] == '有序抖动':
            return ordered_dithering(
                img, palette,
                method=settings['matrix'],
                strength=settings['strength'],
                color_space=settings['color_space'],
                weights=settings.get('weights', (1.0, 1.0, 1.0)),
            )
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

    def _make_display_pixmap(self, pil_img, max_dim=1600):
        """把 PIL 图像缩到适合显示的尺寸再转 QPixmap，避免大图占用海量内存/阻塞界面。"""
        if pil_img is None:
            return QPixmap()
        w, h = pil_img.size
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1.0:
            pil_img = pil_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        return self.pil_image_to_qpixmap(pil_img)

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