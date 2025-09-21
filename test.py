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
from PyQt5.QtCore import Qt, QSize
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
    from dithers import ordered_dithering, floyd_steinberg_dither

    print("Loaded dithering functions from dithers.py")
except Exception as e:
    print("Could not import dithering functions (dithers.py). Using placeholders.", e)


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


class ImageDitherApp(QMainWindow):
    def __init__(self):
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
        self.combo_dither.addItems(['有序抖动', '误差扩散抖动'])
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
        self.combo_color_space.addItems(['RGB', 'HSV'])
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

        dither_layout.addWidget(self.floyd_params)
        self.floyd_params.setVisible(False)

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
        btn_preview = QPushButton('预览处理结果')
        btn_preview.clicked.connect(self.on_preview)
        action_layout.addWidget(btn_preview)
        btn_save = QPushButton('保存结果')
        btn_save.clicked.connect(self.on_save)
        action_layout.addWidget(btn_save)
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
            # 触发色彩空间变化以更新HSV权重显示
            self.on_color_space_change()
        else:
            self.ordered_params.setVisible(False)
            self.floyd_params.setVisible(True)

    def on_color_space_change(self):
        color_space = self.combo_color_space.currentText()
        if color_space == 'HSV':
            self.hsv_params.setVisible(True)
        else:
            self.hsv_params.setVisible(False)

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
        try:
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
            img = img.copy()
        except Exception as e:
            QMessageBox.warning(self, '打开失败', f'无法打开图片: {e}')
            return

        self.image_path = path
        self.original_image = img
        self.lbl_file.setText(os.path.basename(path))
        self.display_mode = 0
        self.show_image_in_label(self.original_image)

    def pil_image_to_qpixmap(self, pil_img):
        """把 PIL Image 转为 QPixmap；对多种模式做兼容处理（RGBA, RGB, L 等）。"""
        if pil_img is None:
            return QPixmap()

        w, h = pil_img.size
        try:
            # 优先用直接字节构造 QImage（更稳健，不依赖 ImageQt 的具体实现）
            if pil_img.mode == 'RGBA':
                # 如果 Qt 支持 Format_RGBA8888 用它，否则使用 BGRA + ARGB32 作为回退
                if hasattr(QImage, 'Format_RGBA8888'):
                    data = pil_img.tobytes('raw', 'RGBA')
                    qimg = QImage(data, w, h, QImage.Format_RGBA8888)
                else:
                    # BGRA + ARGB32 是常用的回退做法（在小端平台上字节顺序匹配）
                    data = pil_img.convert('RGBA').tobytes('raw', 'BGRA')
                    qimg = QImage(data, w, h, QImage.Format_ARGB32)
            elif pil_img.mode == 'RGB':
                data = pil_img.tobytes('raw', 'RGB')
                qimg = QImage(data, w, h, QImage.Format_RGB888)
            elif pil_img.mode == 'L':
                data = pil_img.tobytes('raw', 'L')
                fmt = QImage.Format_Grayscale8 if hasattr(QImage, 'Format_Grayscale8') else QImage.Format_Indexed8
                qimg = QImage(data, w, h, fmt)
            else:
                # 其它模式（例如 CMYK 等）先转换为 RGBA 再处理
                pil2 = pil_img.convert('RGBA')
                data = pil2.tobytes('raw', 'BGRA')
                qimg = QImage(data, pil2.width, pil2.height, QImage.Format_ARGB32)

            return QPixmap.fromImage(qimg)
        except Exception as e:
            # 最后再尝试使用 PIL 的 ImageQt（若存在），作为兼容回退
            try:
                from PIL.ImageQt import ImageQt as PILImageQt
                qim = PILImageQt(pil_img)
                return QPixmap.fromImage(qim)
            except Exception as e2:
                print("pil->qpixmap 转换失败：", e, e2)
                return QPixmap()

    def show_image_in_label(self, pil_img):
        # Convert PIL image to QPixmap robustly, then set scaled pixmap to label
        if pil_img is None:
            self.preview_label.clear()
            return

        pix = self.pil_image_to_qpixmap(pil_img)
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

    def apply_resize(self, image: Image.Image) -> Image.Image:
        if image is None:
            return None
        ow, oh = image.size
        # manual width/height take precedence if provided
        w_text = self.line_w.text().strip()
        h_text = self.line_h.text().strip()
        if w_text or h_text:
            try:
                if w_text and h_text:
                    nw = int(w_text);
                    nh = int(h_text)
                elif w_text:
                    nw = int(w_text);
                    nh = int(round(oh * (nw / ow)))
                else:
                    nh = int(h_text);
                    nw = int(round(ow * (nh / oh)))
                if nw <= 0 or nh <= 0:
                    return image
                return image.resize((nw, nh), Image.LANCZOS)
            except Exception:
                return image
        # else use percent
        pct = self.spin_percent.value()
        if pct == 100:
            return image
        scale = pct / 100.0
        nw = max(1, int(round(ow * scale)))
        nh = max(1, int(round(oh * scale)))
        return image.resize((nw, nh), Image.LANCZOS)

    def on_preview(self):
        if not self.original_image:
            QMessageBox.information(self, '提示', '请先选择图片')
            return
        pal = self.build_palette_list()
        if not pal:
            QMessageBox.information(self, '提示', '调色盘为空，请检查 palette.txt 或选择颜色')
            return
        # Apply resize then dithering
        img = self.apply_resize(self.original_image)
        method = self.combo_dither.currentText()
        try:
            if method == '有序抖动':
                strength = int(self.spin_strength.value())
                color_space = self.combo_color_space.currentText()
                matrix_method = self.combo_matrix.currentText()

                if color_space == 'HSV':
                    weights = (
                        float(self.spin_h_weight.value()),
                        float(self.spin_s_weight.value()),
                        float(self.spin_v_weight.value())
                    )
                    out = ordered_dithering(img, pal, method=matrix_method, strength=strength,
                                            color_space=color_space, weights=weights)
                else:
                    out = ordered_dithering(img, pal, method=matrix_method, strength=strength,
                                            color_space=color_space)
            else:
                alpha = float(self.spin_alpha.value())
                out = floyd_steinberg_dither(img, pal, alpha_strength=alpha)
            self.processed_image = out
            self.display_mode = 1
            self.show_image_in_label(self.processed_image)
            self.statusBar().showMessage('已生成预览')
        except Exception as e:
            QMessageBox.warning(self, '处理失败', f'应用抖动失败: {e}')

    def on_save(self):
        if self.processed_image is None:
            # try to generate preview automatically
            self.on_preview()
            if self.processed_image is None:
                return
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
            QMessageBox.information(self, '已保存', f'已保存到 {path}')
        except Exception as e:
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