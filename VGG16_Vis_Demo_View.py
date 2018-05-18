# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (QAction, qApp, QTextEdit, QMainWindow, QMessageBox, QDesktopWidget, QLabel, QComboBox,
                             QPushButton, QWidget, QApplication, QMenu, QHBoxLayout, QVBoxLayout, QGridLayout,
                             QLCDNumber, QSlider, QLineEdit, QRadioButton, QGroupBox, QScrollArea, QCheckBox,
                             QInputDialog, QFrame, QColorDialog, QFileDialog, QProgressBar, QSplitter)
from PyQt5.QtGui import QFont, QIcon, QColor, QPainter, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QBasicTimer, QSize, QMargins, QThread
import numpy as np
import itertools
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from enum import Enum
import time

from VGG16_Vis_Demo_Model import VGG16_Vis_Demo_Model

BORDER_WIDTH = 10


# from DeepVis tool
def norm01(arr):
    arr = arr.copy()
    arr -= arr.min()
    arr /= arr.max() + 1e-10
    return arr


# modified from DeepVis tool
def norm0255c(arr, center):
    # Maps the input range to [0,1] such that the center value maps to .5
    arr = arr.copy()
    arr -= center
    arr /= max(2 * arr.max(), -2 * arr.min()) + 1e-10
    arr += .5
    arr *= 255
    assert arr.min() >= 0
    assert arr.max() <= 255
    return arr


# detailed unit view
class DetailedUnitViewWidget(QLabel):
    IMAGE_SIZE = QSize(224, 224)

    class WorkingMode(Enum):
        ACTIVATION = 'Activation'
        DECONV = 'Deconv'

    class BackpropViewOption(Enum):
        RAW = 'raw'
        GRAY = 'gray'
        NORM = 'norm'
        NORM_BLUR = 'blurred norm'

    class OverlayOptions(Enum):
        No_OVERLAY = 'No overlay'
        OVER_ACTIVE = 'Over active'
        OVER_INACTIVE = 'Over inactive'
        ONLY_ACTIVE = 'Only active'
        ONLY_INACTIVE = 'Only inactive'

    def __init__(self):
        super(QLabel, self).__init__()
        default_image = QPixmap(self.IMAGE_SIZE)
        default_image.fill(Qt.darkGreen)
        self.setPixmap(default_image)
        self.setAlignment(Qt.AlignCenter)
        self.setMargin(0)
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setFixedSize(QSize(240, 240))
        self.setStyleSheet("QWidget {background-color: blue}")
        self.overlay_mode = self.OverlayOptions.No_OVERLAY.value
        self.working_mode = self.WorkingMode.ACTIVATION.value
        self.backprop_view_mode = self.BackpropViewOption.RAW.value

    def display_activation(self, data, input_image_path):
        self.working_mode = self.WorkingMode.ACTIVATION.value
        self.activation = data
        self.input_image_path = input_image_path
        self.set_overlay_view(self.overlay_mode)

    def display_deconv(self, data):
        self.working_mode = self.WorkingMode.DECONV.value
        self.deconv = data
        self.set_backprop_view(self.backprop_view_mode)

    def mouseDoubleClickEvent(self, QMouseEvent):
        self.pixmap().save('./Temps/ForExternalProgram.png')
        Image.open("./Temps/ForExternalProgram.png").show()

    def set_backprop_view(self, view_mode=BackpropViewOption.RAW.value):
        self.backprop_view_mode = view_mode
        if hasattr(self, 'deconv') == False or self.working_mode != self.WorkingMode.DECONV.value:
            return

        data = self.deconv
        data = cv2.resize(data, (self.IMAGE_SIZE.width(), self.IMAGE_SIZE.height()))
        if view_mode == self.BackpropViewOption.RAW.value:
            data = norm0255c(data, 0)
        elif view_mode == self.BackpropViewOption.GRAY.value:
            data = norm0255c(data.mean(axis=2), 0)
            data = np.dstack([data] * 3)  # 244x224 => 224x224x3 (RGB!)
        elif view_mode == self.BackpropViewOption.NORM.value or view_mode == self.BackpropViewOption.NORM_BLUR.value:
            data = np.linalg.norm(data, axis=2)
            if view_mode == self.BackpropViewOption.NORM_BLUR.value:
                grad_norm_blur_radius = 0.02 * max([self.IMAGE_SIZE.width(), self.IMAGE_SIZE.height()])
                cv2.GaussianBlur(data, (0, 0), grad_norm_blur_radius, data)
            data = norm01(data)
            data = norm01(data)
            cmap = plt.get_cmap('jet')
            rgba_image = cmap(data)
            data = np.delete(rgba_image, 3, 2)
            data *= 255

        data = data.astype(np.uint8)
        image = QImage(data.tobytes(), data.shape[0], data.shape[1], data.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.setPixmap(pixmap)

    def set_overlay_view(self, mode=OverlayOptions.No_OVERLAY.value):
        self.overlay_mode = mode
        if not hasattr(self, 'activation') or self.working_mode != self.WorkingMode.ACTIVATION.value:
            return

        # prepare base image
        if mode == self.OverlayOptions.No_OVERLAY.value or mode == self.OverlayOptions.ONLY_ACTIVE.value or \
                mode == self.OverlayOptions.OVER_INACTIVE.value:
            pixmap = VGG16_Vis_Demo_View.get_pixmaps_from_data(np.array([self.activation.astype(np.uint8), ]))[0]
            pixmap = pixmap.scaledToWidth(self.IMAGE_SIZE.width())
        else:
            pixmap = QPixmap(self.IMAGE_SIZE)
            pixmap.fill(Qt.black)

        # Overlay
        if not mode == self.OverlayOptions.No_OVERLAY.value:
            data = self.activation

            # use sigmoid function to add non-linearity near border
            border = 32
            delta = 6.3  # delta=(255-border)/stretch_factor=(0-border)/stretch_factor
            stretch_factor = border / delta + (255 - 2 * border) / delta / 255 * data  # linear function
            data_norm = np.divide(data - border, stretch_factor)
            alpha = 255 / (1 + np.exp(-data_norm))

            if mode == self.OverlayOptions.OVER_INACTIVE.value or mode == self.OverlayOptions.ONLY_INACTIVE.value:
                alpha = 255 - alpha

            if len(self.activation.shape) > 1:
                alpha_channel = alpha
                alpha_channel = cv2.resize(alpha_channel, (self.IMAGE_SIZE.width(), self.IMAGE_SIZE.height()),
                                           interpolation=cv2.INTER_NEAREST)
            else:
                alpha_channel = np.full((self.IMAGE_SIZE.width(), self.IMAGE_SIZE.height()), alpha)
            alpha_channel = np.expand_dims(alpha_channel, 2)

            input_RGB = cv2.imread(self.input_image_path, 1)[:, :, (2, 1, 0)]
            input_RGB = cv2.resize(input_RGB, (self.IMAGE_SIZE.width(), self.IMAGE_SIZE.height()))
            input_ARGB = np.dstack((alpha_channel, input_RGB))[:, :, (3, 2, 1, 0)]  # dstack will reverse the order
            input_ARGB = input_ARGB.astype(np.uint8)
            input_image = QImage(input_ARGB.tobytes(), input_ARGB.shape[0], input_ARGB.shape[1],
                                 input_ARGB.shape[1] * 4, QImage.Format_ARGB32)

            painter = QPainter()
            painter.begin(pixmap)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.drawImage(0, 0, input_image)
            painter.end()

        self.setPixmap(pixmap)


# clickable QLabel in Layer View
class SmallUnitViewWidget(QLabel):
    clicked = pyqtSignal()

    def __init__(self):
        super(QLabel, self).__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setAlignment(Qt.AlignCenter)
        self.setMargin(0)
        self.setLineWidth(BORDER_WIDTH / 2)

    def mousePressEvent(self, QMouseEvent):
        self.clicked.emit()


# double-clickable QLabel
class DoubleClickableQLabel(QLabel):
    def __init__(self):
        super(QLabel, self).__init__()

    def mouseDoubleClickEvent(self, QMouseEvent):
        self.pixmap().save('./Temps/ForExternalProgram.png')
        Image.open("./Temps/ForExternalProgram.png").show()


class LayerViewWidget(QScrollArea):
    MIN_SCALE_FAKTOR = 0.2

    clicked_unit_index = 0

    n_w = 1  # number of units per row
    n_h = 1  # number of units per column

    clicked_unit_changed = pyqtSignal(int)

    def __init__(self, units):
        super(QScrollArea, self).__init__()
        self.grid = QGridLayout()
        self.units = units
        self.initUI(units)

    def initUI(self, units):
        self.grid = QGridLayout()
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.Box)
        self.setLayout(self.grid)
        self.grid.setVerticalSpacing(0)
        self.grid.setHorizontalSpacing(0)
        self.grid.setSpacing(0)
        self.grid.setContentsMargins(QMargins(0, 0, 0, 0))
        self.allocate_units(units)

    def allocate_units(self, units):
        N = len(units)
        oroginal_unit_width = units[0].width()
        H = self.height()
        W = self.width()

        # calculate the proper unit size, which makes full use of the space
        numerator = (np.sqrt(W * W + 4 * N * W * H) - W)
        denominator = 2 * N
        d = np.floor(np.divide(numerator, denominator))

        # if the resulting unit size is too small, display the unit s with the allowed minimum size
        allowed_min_width = np.ceil(np.multiply(oroginal_unit_width, self.MIN_SCALE_FAKTOR))
        self.displayed_unit_width = np.maximum(d - BORDER_WIDTH, allowed_min_width)

        self.n_w = np.floor(W / d)
        self.n_h = np.ceil(N / self.n_w)

        positions = [(i, j) for i in range(int(self.n_h)) for j in range(int(self.n_w))]
        for position, unit in itertools.izip(positions, units):
            unitView = SmallUnitViewWidget()
            unitView.clicked.connect(self.unit_clicked_action)
            unitView.index = position[0] * self.n_w + position[1]
            scaled_image = unit.scaledToWidth(self.displayed_unit_width)
            unitView.setPixmap(scaled_image)
            unitView.setFixedSize(QSize(d, d))
            self.grid.addWidget(unitView, *position)
        self._activate_last_clicked_unit()

    def _activate_last_clicked_unit(self):
        if self.clicked_unit_index >= len(self.units):
            self.clicked_unit_index = 0
        last_clicked_position = (self.clicked_unit_index // self.n_w, np.remainder(self.clicked_unit_index, self.n_w))
        last_clicked_unit = self.grid.itemAtPosition(last_clicked_position[0], last_clicked_position[1]).widget()
        last_clicked_unit.clicked.emit()

    def unit_clicked_action(self):
        # deactivate last one
        last_clicked_position = (self.clicked_unit_index // self.n_w, np.remainder(self.clicked_unit_index, self.n_w))
        last_clicked_unit = self.grid.itemAtPosition(last_clicked_position[0], last_clicked_position[1]).widget()
        last_clicked_unit.setStyleSheet("QWidget { background-color: %s }" % self.palette().color(10).name())
        # activate the clicked one
        clicked_unit = self.sender()
        clicked_unit.setStyleSheet("QWidget {  background-color: blue}")
        start = time.time()
        self.clicked_unit_changed.emit(clicked_unit.index)  # notify the main view to change the unit view
        self.clicked_unit_index = clicked_unit.index

    def resizeEvent(self, QResizeEvent):
        self.rearrange()

    def rearrange(self):
        self.clear_grid()
        self.allocate_units(self.units)

    def set_units(self, units):
        last_units_number = len(self.units)
        self.units = units
        if last_units_number == len(units):
            # if the arrangement will not be changed, simply update pixmaps to speed up loading
            self.update_data()
        else:
            self.rearrange()

    def update_data(self):
        for i in range(len(self.units)):
            unit_widget = self.grid.itemAt(i).widget()
            unit_widget.setPixmap(self.units[i].scaledToWidth(self.displayed_unit_width))
        self._activate_last_clicked_unit()

    def clear_grid(self):
        while self.grid.count():
            self.grid.itemAt(0).widget().deleteLater()
            self.grid.itemAt(0).widget().close()
            self.grid.removeItem(self.grid.itemAt(0))


class ProbsView(QGroupBox):
    def __init__(self):
        super(QGroupBox, self).__init__()
        self.setTitle("Results")
        vbox_probs = QVBoxLayout()
        self.lbl_probs = QLabel('#1 \n#2 \n#3 \n#4 \n#5 ')
        vbox_probs.addWidget(self.lbl_probs)
        self.setLayout(vbox_probs)

    def set_probs(self, probs, labels):
        num = min(5, len(probs))
        sorted_results_idx = sorted(range(len(probs)), reverse=True, key=lambda k: probs[k])
        txt = ''
        for i in range(num):
            if i != 0:
                txt += '\n'
            txt += '#%d %+16s    %4.2f%%' % (i, labels[sorted_results_idx[i]], probs[sorted_results_idx[i]] * 100)
        self.lbl_probs.setText(txt)


class VGG16_Vis_Demo_View(QMainWindow):
    _busy = 0

    def __init__(self, model, ctl):
        super(QMainWindow, self).__init__()
        self.model = model
        self.ctl = ctl
        self.model.dataChanged[int].connect(self.update_data)
        self.ctl.isBusy[bool].connect(self.set_busy)
        self.initUI()
        self.ctl.start(priority=QThread.LowestPriority)

    def initUI(self):
        # region vbox1
        vbox1 = QVBoxLayout()
        vbox1.setAlignment(Qt.AlignCenter)

        # model selection & input image
        grid_input = QGridLayout()
        font_bold = QFont()
        font_bold.setBold(True)
        lbl_model = QLabel('Model')
        lbl_model.setFont(font_bold)
        combo_model = QComboBox(self)
        combo_model.addItem('')
        model_names = self.model.get_data(VGG16_Vis_Demo_Model.data_idx_model_names)
        for model_name in model_names:
            combo_model.addItem(model_name)
        combo_model.activated[str].connect(self.ctl.set_model)
        lbl_input = QLabel('Input')
        lbl_input.setFont(font_bold)
        combo_input_source = QComboBox(self)
        combo_input_source.addItem('')  # null entry
        combo_input_source.addItem('Image')
        combo_input_source.addItem('Video')
        self.combo_input_image = QComboBox(self)
        self.combo_input_image.addItem('')  # null entry
        self.combo_input_image.activated[str].connect(self.ctl.set_input_image)
        ckb_input_image_background = QCheckBox('Background')
        ckb_input_image_background.setToolTip('The network loads PNG images with black background.')
        ckb_input_image_background.stateChanged.connect(self.toggle_input_background)
        grid_input.addWidget(lbl_model, 0, 1)
        grid_input.addWidget(combo_model, 0, 2)
        grid_input.addWidget(lbl_input, 1, 1)
        grid_input.addWidget(combo_input_source, 1, 2)
        grid_input.addWidget(self.combo_input_image, 2, 1, 1, 2)
        grid_input.addWidget(ckb_input_image_background, 3, 1, 1, 2)
        vbox1.addLayout(grid_input)

        pixm_input = QPixmap(QSize(224, 224))
        pixm_input.fill(Qt.black)
        self.lbl_input_image = DoubleClickableQLabel()
        self.lbl_input_image.setAlignment(Qt.AlignCenter)
        self.lbl_input_image.setPixmap(pixm_input)
        vbox1.addWidget(self.lbl_input_image)

        # Arrow
        lbl_arrow_input_to_vgg16 = QLabel('⬇️')
        lbl_arrow_input_to_vgg16.setFont(font_bold)
        lbl_arrow_input_to_vgg16.setAlignment(Qt.AlignCenter)
        vbox1.addWidget(lbl_arrow_input_to_vgg16)

        # Network overview
        gb_network = QGroupBox("VGG16")
        vbox_network = QVBoxLayout()
        vbox_network.setAlignment(Qt.AlignCenter)
        vgg16_layer_names = self.model.get_data(VGG16_Vis_Demo_Model.data_idx_layer_names)
        vgg16_layer_output_sizes = self.model.get_data(VGG16_Vis_Demo_Model.data_idx_layer_output_sizes)
        # todo: change the style of layers
        for layer_name in vgg16_layer_names:
            btn_layer = QRadioButton(layer_name)
            btn_layer.setFont(QFont('Times', 11, QFont.Bold))
            btn_layer.toggled.connect(self.select_layer_action)
            vbox_network.addWidget(btn_layer)
            size = vgg16_layer_output_sizes[layer_name]
            size_string = ''
            for value in size:
                size_string += str(value)
                size_string += '×'
            size_string = size_string[:len(size_string) - len('×')]
            lbl_arrow = QLabel(' ⬇️ ' + size_string)
            lbl_arrow.setFont(QFont("Helvetica", 8))
            vbox_network.addWidget(lbl_arrow)
        wrapper_vbox_network = QWidget()
        wrapper_vbox_network.setLayout(vbox_network)
        scroll_network = QScrollArea()
        scroll_network.setFrameShape(QFrame.Box)
        scroll_network.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_network.setWidget(wrapper_vbox_network)
        scroll_network.setAlignment(Qt.AlignCenter)
        layout_scroll_network = QVBoxLayout()
        layout_scroll_network.addWidget(scroll_network)
        gb_network.setLayout(layout_scroll_network)
        vbox1.addWidget(gb_network)

        # Arrow
        lbl_arrow_vgg16_to_probs = QLabel('⬇️')
        lbl_arrow_vgg16_to_probs.setFont(font_bold)
        lbl_arrow_vgg16_to_probs.setAlignment(Qt.AlignCenter)
        vbox1.addWidget(lbl_arrow_vgg16_to_probs)

        # Prob view
        self.probs_view = ProbsView()
        vbox1.addWidget(self.probs_view)
        # endregion

        # region vbox2
        vbox2 = QVBoxLayout()
        vbox2.setAlignment(Qt.AlignTop)

        # header
        combo_layer_view = QComboBox(self)
        combo_layer_view.addItem('Activations')
        combo_layer_view.addItem('Top 1 images')
        selected_layer_name = ' '
        self.lbl_layer_name = QLabel(
            "of layer <font color='blue'><b>%r</b></font>" % selected_layer_name)  # todo: delete default value
        ckb_group_units = QCheckBox('Group similar units')
        grid_layer_header = QGridLayout()
        grid_layer_header.addWidget(combo_layer_view, 0, 1)
        grid_layer_header.addWidget(self.lbl_layer_name, 0, 2)
        grid_layer_header.addWidget(ckb_group_units, 0, 4)
        vbox2.addLayout(grid_layer_header)

        # layer (units) view
        dummy_units = []
        for i in range(128):
            dummy_pxim_unit = QPixmap(QSize(112, 112))
            dummy_pxim_unit.fill(Qt.darkGreen)
            dummy_units.append(dummy_pxim_unit)
        self.layer_view = LayerViewWidget(dummy_units)
        self.layer_view.clicked_unit_changed.connect(self.select_unit_action)
        vbox2.addWidget(self.layer_view)
        # endregion

        # region vbox3
        vbox3 = QVBoxLayout()
        vbox3.setAlignment(Qt.AlignTop)

        # header
        self.combo_unit_view = QComboBox(self)
        for member in DetailedUnitViewWidget.WorkingMode:
            self.combo_unit_view.addItem(member.value)
        self.combo_unit_view.currentTextChanged.connect(self.load_detailed_unit_image)
        self.combo_unit_view.setCurrentText(DetailedUnitViewWidget.WorkingMode.ACTIVATION.value)
        selected_unit_name = ' '
        self.lbl_unit_name = QLabel(
            "of unit <font color='blue'><b>%r</b></font>" % selected_unit_name)
        hbox_unit_view_header = QHBoxLayout()
        hbox_unit_view_header.addWidget(self.combo_unit_view)
        hbox_unit_view_header.addWidget(self.lbl_unit_name)
        vbox3.addLayout(hbox_unit_view_header)

        # region settings of unit view

        # overlay setting
        hbox_overlay = QHBoxLayout()
        hbox_overlay.addWidget(QLabel("Overlay: "))
        self.combo_unit_overlay = QComboBox(self)
        for member in DetailedUnitViewWidget.OverlayOptions:
            self.combo_unit_overlay.addItem(member.value)
        self.combo_unit_overlay.activated[str].connect(self.overlay_action)
        hbox_overlay.addWidget(self.combo_unit_overlay)
        vbox3.addLayout(hbox_overlay)

        # Backprop Mode setting
        hbox_backprop_mode = QHBoxLayout()
        hbox_backprop_mode.addWidget(QLabel("Backprop mode: "))
        self.combo_unit_backprop_mode = QComboBox(self)
        for member in self.model.BackpropModeOption:
            self.combo_unit_backprop_mode.addItem(member.value)
        self.combo_unit_backprop_mode.currentTextChanged.connect(self.load_detailed_unit_image)
        self.combo_unit_backprop_mode.setEnabled(False)
        hbox_backprop_mode.addWidget(self.combo_unit_backprop_mode)
        vbox3.addLayout(hbox_backprop_mode)

        # Backprop view setting
        hbox_backprop_view = QHBoxLayout()
        hbox_backprop_view.addWidget(QLabel("Backprop view: "))
        self.combo_unit_backprop_view = QComboBox(self)
        for member in DetailedUnitViewWidget.BackpropViewOption:
            self.combo_unit_backprop_view.addItem(member.value)
        self.combo_unit_backprop_view.setCurrentText(DetailedUnitViewWidget.BackpropViewOption.RAW.value)
        self.combo_unit_backprop_view.currentTextChanged.connect(self.switch_backprop_view_action)
        self.combo_unit_backprop_view.setEnabled(False)
        hbox_backprop_view.addWidget(self.combo_unit_backprop_view)
        vbox3.addLayout(hbox_backprop_view)

        # endregion

        # unit image
        self.detailed_unit_view = DetailedUnitViewWidget()
        vbox3.addWidget(self.detailed_unit_view)

        # spacer
        vbox3.addSpacing(20)
        frm_line_unit_top9 = QFrame()
        frm_line_unit_top9.setFrameShape(QFrame.HLine)
        frm_line_unit_top9.setFrameShadow(QFrame.Sunken)
        frm_line_unit_top9.setLineWidth(1)
        vbox3.addWidget(frm_line_unit_top9)
        vbox3.addSpacing(20)

        # top 9 images
        vbox3.addWidget(QLabel("Top 9 images with heighest activations"))
        combo_top9_images_mode = QComboBox(self)
        combo_top9_images_mode.addItem("Input")
        combo_top9_images_mode.addItem("Deconv")
        vbox3.addWidget(combo_top9_images_mode)
        grid_top9 = QGridLayout()
        pixm_top_image = QPixmap(QSize(224, 224))
        pixm_top_image.fill(Qt.darkGray)
        _top_image_height = 64
        pixm_top_image_scaled = pixm_top_image.scaledToHeight(_top_image_height)
        positions_top9_images = [(i, j) for i in range(3) for j in range(3)]
        for position in positions_top9_images:
            lbl_top_image = QLabel()
            lbl_top_image.setPixmap(pixm_top_image_scaled)
            lbl_top_image.setFixedSize(QSize(_top_image_height + 4, _top_image_height + 4))
            grid_top9.addWidget(lbl_top_image, *position)
        vbox3.addLayout(grid_top9)

        # spacer
        vbox3.addSpacing(20)
        frm_line_top9_gd = QFrame()
        frm_line_top9_gd.setFrameShape(QFrame.HLine)
        frm_line_top9_gd.setFrameShadow(QFrame.Sunken)
        frm_line_top9_gd.setLineWidth(1)
        vbox3.addWidget(frm_line_top9_gd)
        vbox3.addSpacing(20)

        # gradient ascent
        btn_gradient_ascent = QPushButton("Find out what was learnt in this unit")
        vbox3.addWidget(btn_gradient_ascent)

        # endregion

        hbox = QHBoxLayout()
        widget_vbox1 = QFrame()
        widget_vbox1.setLayout(vbox1)
        widget_vbox1.setFixedWidth(widget_vbox1.sizeHint().width())
        widget_vbox2 = QFrame()
        widget_vbox2.setLayout(vbox2)
        widget_vbox3 = QFrame()
        widget_vbox3.setLayout(vbox3)
        widget_vbox3.setFixedWidth(widget_vbox3.sizeHint().width())

        frm_line_1_2 = QFrame()
        frm_line_1_2.setFrameShape(QFrame.VLine)
        frm_line_1_2.setFrameShadow(QFrame.Sunken)
        frm_line_1_2.setLineWidth(2)
        frm_line_2_3 = QFrame()
        frm_line_2_3.setFrameShape(QFrame.VLine)
        frm_line_2_3.setFrameShadow(QFrame.Sunken)
        frm_line_2_3.setLineWidth(2)

        hbox.addWidget(widget_vbox1)
        hbox.addWidget(frm_line_1_2)
        hbox.addWidget(widget_vbox2)
        hbox.addWidget(frm_line_2_3)
        hbox.addWidget(widget_vbox3)

        central_widget = QWidget()
        central_widget.setLayout(hbox)

        self.statusbar = self.statusBar()
        self.statusbar.showMessage('ready')
        self.setCentralWidget(central_widget)
        self.setWindowTitle('VGG16 Visualizer')
        self.center()
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def update_data(self, data_idx):
        if data_idx == VGG16_Vis_Demo_Model.data_idx_input_image_names:
            self.update_combobox_input_image()
        elif data_idx == VGG16_Vis_Demo_Model.data_idx_new_input:
            self.refresh()

    def update_combobox_input_image(self):
        self.combo_input_image.clear()
        self.combo_input_image.addItem('')  # null entry
        input_image_names = self.model.get_data(VGG16_Vis_Demo_Model.data_idx_input_image_names)
        for name in input_image_names:
            self.combo_input_image.addItem(name)

    def select_layer_action(self):
        btn = self.sender()
        if btn.isChecked():
            self.selected_layer_name = btn.text()
            self.load_layer_view()

    def load_layer_view(self):
        if hasattr(self, 'selected_layer_name'):

            self.set_busy(True)
            self.lbl_layer_name.setText("of layer <font color='blue'><b>%s</b></font>" % str(self.selected_layer_name))
            try:
                data = self.model.get_activations(self.selected_layer_name)
                data = self._prepare_data_for_display(data)
                start = time.time()
                pximaps = self.get_pixmaps_from_data(data)
                print("get pixmap from data time: " + str(time.time() - start))
                self.layer_view.set_units(pximaps)
            except AttributeError, Argument:
                pass
            self.set_busy(False)
            print("load layer time: " + str(time.time() - start))

    def select_unit_action(self, unit_index):
        if hasattr(self, 'selected_layer_name'):
            self.lbl_unit_name.setText(
                "of unit <font color='blue'><b>%s@%s</b></font>" % (str(unit_index), str(self.selected_layer_name)))
            self.selected_unit_index = unit_index
            self.load_detailed_unit_image()

    def load_detailed_unit_image(self):
        if hasattr(self, 'selected_layer_name') and hasattr(self, 'selected_unit_index'):
            self.set_busy(True)
            mode = self.combo_unit_view.currentText()
            if mode == DetailedUnitViewWidget.WorkingMode.ACTIVATION.value:
                self.combo_unit_overlay.setEnabled(True)
                self.combo_unit_backprop_view.setEnabled(False)
                self.combo_unit_backprop_mode.setEnabled(False)
                data = self.model.get_activations(self.selected_layer_name)
                try:
                    data = self._prepare_data_for_display(data)
                    self.detailed_unit_view.display_activation(data[self.selected_unit_index], self.model.get_data(
                        VGG16_Vis_Demo_Model.data_idx_input_image_path))
                except AttributeError, Argument:
                    pass

            elif mode == DetailedUnitViewWidget.WorkingMode.DECONV.value:
                self.combo_unit_overlay.setEnabled(False)
                self.combo_unit_backprop_view.setEnabled(True)
                self.combo_unit_backprop_mode.setEnabled(True)
                data = self.model.get_deconv(self.selected_layer_name, self.selected_unit_index,
                                             self.combo_unit_backprop_mode.currentText())
                try:
                    self.detailed_unit_view.display_deconv(data)
                except AttributeError, Argument:
                    pass
            self.set_busy(False)

    def _prepare_data_for_display(self, data):
        max = data.max()
        min = data.min()
        range = max - min
        if range == 0:
            data = data / data * 255  # prevent division by zero
        else:
            data = (data - min) / range * 255  # normalize data for display
        return data

    # if the data has colors, argument 'data' is a 4D (conv_layer) or 3D (fc_layer) array,
    # else the 'data' is a 3D (conv_layer) or 2D (fc_layer) array.
    # The first axis is the unit index
    @staticmethod
    def get_pixmaps_from_data(data, color=False):
        data = data.astype(np.uint8)  # convert to 8-bit unsigned integer
        pixmaps = []
        num = data.shape[0]
        for i in range(num):
            if color:
                if len(data[i].shape) < 3:
                    unit = QPixmap(QSize(8, 8))
                    unit.fill(QColor(data[i][0], data[i][1], data[i][2]))
                else:
                    unit = QImage(data[i][:], data[i].shape[2], data[i].shape[1], data[i].shape[1] * 3,
                                  QImage.Format_RGB888)
                    unit = QPixmap.fromImage(unit)
            else:
                if len(data[i].shape) < 2:
                    unit = QPixmap(QSize(8, 8))
                    unit.fill(QColor(data[i], data[i], data[i]))
                else:
                    unit = QImage(data[i][:], data[i].shape[1], data[i].shape[0], data[i].shape[0],
                                  QImage.Format_Grayscale8)
                    unit = QPixmap.fromImage(unit)
            pixmaps.append(unit)
        return pixmaps

    def refresh(self):
        # get input image
        self.lbl_input_image.setPixmap(QPixmap(self.model.get_data(VGG16_Vis_Demo_Model.data_idx_input_image_path)))

        # get probs
        results = self.model.get_data(VGG16_Vis_Demo_Model.data_idx_probs)
        labels = self.model.get_data(VGG16_Vis_Demo_Model.data_idx_labels)
        self.probs_view.set_probs(results, labels)

        # load layer view. This also triggers the last clicked unit to be loaded in unit view
        self.load_layer_view()

    def overlay_action(self, mode):
        self.detailed_unit_view.set_overlay_view(mode)

    def switch_backprop_view_action(self, mode):
        self.detailed_unit_view.set_backprop_view(mode)

    def toggle_input_background(self, state):
        if state == Qt.Checked:
            self.lbl_input_image.setStyleSheet("QWidget {background-color: black}")
        else:
            self.lbl_input_image.setStyleSheet("QWidget { background-color: %s }" % self.palette().color(10).name())

    def set_busy(self, isBusy):
        previous_busy_count = self._busy
        if isBusy:
            self._busy += 1
        else:
            self._busy -= 1
        if self._busy == 0:
            self.statusbar.showMessage('Ready')
        elif previous_busy_count == 0:
            self.statusbar.showMessage('Busy')
