# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (QAction, qApp, QTextEdit, QMainWindow, QMessageBox, QDesktopWidget, QLabel, QComboBox,
                             QPushButton, QWidget, QApplication, QMenu, QHBoxLayout, QVBoxLayout, QGridLayout,
                             QLCDNumber, QSlider, QLineEdit, QRadioButton, QGroupBox, QScrollArea, QCheckBox,
                             QInputDialog, QFrame, QColorDialog, QFileDialog, QProgressBar, QSplitter)
from PyQt5.QtGui import QFont, QIcon, QColor, QPainter, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QBasicTimer, QSize, QMargins
import numpy as np
import itertools

# todo: change to read from model
vgg16_layers = {'input': '224×224×3', 'conv1_1': '224×224×64', 'conv1_2': '224×224×64', 'pool1': '112×112×64',
                'conv2_1': '112×112×128', 'conv2_2': '112×112×128', 'pool2': '56×56×128', 'conv3_1': '56×56×256',
                'conv_3_2': '56×56×256', 'conv_3_3': '56×56×256', 'pool3': '28×28×256', 'conv4_1': '28×28×512',
                'conv4_2': '28×28×512', 'conv4_3': '28×28×512', 'pool4': '14×14×512', 'conv5_1': '14×14×512',
                'conv5_2': '14×14×512', 'conv5_3': '14×14×512', 'pool5': '7×7×512', 'fc6': '4096',
                'fc7': '4096', 'fc8': '16'}
vgg16_layers_sorted = ['input', 'conv1_1', 'conv1_2', 'pool1',
                       'conv2_1', 'conv2_2', 'pool2', 'conv3_1',
                       'conv_3_2', 'conv_3_3', 'pool3', 'conv4_1',
                       'conv4_2', 'conv4_3', 'pool4', 'conv5_1',
                       'conv5_2', 'conv5_3', 'pool5', 'fc6',
                       'fc7', 'fc8']

BORDER_WIDTH = 10


# clickable QLabel
class UnitViewWidget(QLabel):
    clicked = pyqtSignal()

    def __init__(self):
        super(QLabel, self).__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setAlignment(Qt.AlignCenter)
        self.setMargin(0)
        self.setLineWidth(BORDER_WIDTH / 2)

    def mousePressEvent(self, QMouseEvent):
        self.clicked.emit()


class LayerViewWidget(QScrollArea):
    MIN_SCALE_FAKTOR = 0.2

    clicked_unit_index = 0

    n_w = 1  # number of units per row
    n_h = 1  # number of units per column

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
        displayed_unit_width = np.maximum(d - BORDER_WIDTH, allowed_min_width)

        self.n_w = np.floor(W / d)
        self.n_h = np.ceil(N / self.n_w)

        positions = [(i, j) for i in range(int(self.n_h)) for j in range(int(self.n_w))]
        for position, unit in itertools.izip(positions, units):
            unitView = UnitViewWidget()
            unitView.clicked.connect(self.unit_clicked_action)
            unitView.index = position[0] * self.n_w + position[1]
            scaled_image = unit.scaledToWidth(displayed_unit_width)
            unitView.setPixmap(scaled_image)
            unitView.setFixedSize(QSize(d, d))
            self.grid.addWidget(unitView, *position)
        last_clicked_position = (self.clicked_unit_index // self.n_w, np.remainder(self.clicked_unit_index, self.n_w))
        lastClicked = self.grid.itemAtPosition(last_clicked_position[0], last_clicked_position[1]).widget()
        lastClicked.clicked.emit()

    def unit_clicked_action(self):
        # deactivate last one
        last_clicked_position = (self.clicked_unit_index // self.n_w, np.remainder(self.clicked_unit_index, self.n_w))
        lastClicked = self.grid.itemAtPosition(last_clicked_position[0], last_clicked_position[1]).widget()
        lastClicked.setStyleSheet(
            "QWidget { background-color: %s }" % self.palette().color(10).name())
        clicked_unit = self.sender()
        clicked_unit.setStyleSheet(
            "QWidget {  background-color: blue}")
        self.clicked_unit_index = clicked_unit.index

    def resizeEvent(self, QResizeEvent):
        self.clear_grid()
        self.allocate_units(self.units)
        self.show()
        pass

    def clear_grid(self):
        while self.grid.count():
            self.grid.itemAt(0).widget().deleteLater()
            self.grid.itemAt(0).widget().close()
            self.grid.removeItem(self.grid.itemAt(0))


class VGG16_Vis_Demo_View(QMainWindow):

    def __init__(self):
        super(QMainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        # region vbox1
        vbox1 = QVBoxLayout()
        vbox1.setAlignment(Qt.AlignCenter)

        # input image
        grid_input = QGridLayout()
        font_bold = QFont()
        font_bold.setBold(True)
        lbl_input = QLabel('Input', self)
        lbl_input.setFont(font_bold)
        combo_input_source = QComboBox(self)
        combo_input_source.addItem('Image')
        combo_input_source.addItem('Video')
        combo_input_image = QComboBox(self)
        grid_input.addWidget(lbl_input, 0, 1)
        grid_input.addWidget(combo_input_source, 0, 2)
        grid_input.addWidget(combo_input_image, 1, 1, 1, 2)
        vbox1.addLayout(grid_input)

        pixm_input = QPixmap(QSize(224, 224))
        pixm_input.fill(Qt.black)
        lbl_input_image = QLabel(self)
        lbl_input_image.setAlignment(Qt.AlignCenter)
        lbl_input_image.setPixmap(pixm_input)
        vbox1.addWidget(lbl_input_image)

        # Arrow
        lbl_arrow_input_to_vgg16 = QLabel('⬇️')
        lbl_arrow_input_to_vgg16.setFont(font_bold)
        lbl_arrow_input_to_vgg16.setAlignment(Qt.AlignCenter)
        vbox1.addWidget(lbl_arrow_input_to_vgg16)

        # Network overview
        gb_network = QGroupBox("VGG16")
        vbox_network = QVBoxLayout()
        vbox_network.setAlignment(Qt.AlignCenter)
        # todo: change the style of layers
        for layer_name in vgg16_layers_sorted:
            btn_layer = QRadioButton(layer_name)
            btn_layer.setFont(QFont('Times', 11, QFont.Bold))
            vbox_network.addWidget(btn_layer)
            lbl_arrow = QLabel(' ⬇️ ' + (vgg16_layers[layer_name]))
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
        gb_probs = QGroupBox("Results")
        vbox_probs = QVBoxLayout()
        lbl_probs = QLabel('#1 \n#2 \n#3 \n#4 \n#5 ')
        vbox_probs.addWidget(lbl_probs)
        gb_probs.setLayout(vbox_probs)
        vbox1.addWidget(gb_probs)
        # endregion

        # region vbox2
        vbox2 = QVBoxLayout()
        vbox2.setAlignment(Qt.AlignTop)

        # header
        combo_layer_view = QComboBox(self)
        combo_layer_view.addItem('Activations')
        combo_layer_view.addItem('Top 1 images')
        selected_layer_name = 'conv1_1'
        lbl_layer_name = QLabel(
            "of layer <font color='blue'><b>%r</b></font>" % selected_layer_name)  # todo: delete default value
        ckb_group_units = QCheckBox('Group similar units')
        grid_layer_header = QGridLayout()
        grid_layer_header.addWidget(combo_layer_view, 0, 1)
        grid_layer_header.addWidget(lbl_layer_name, 0, 2)
        grid_layer_header.addWidget(ckb_group_units, 0, 4)
        vbox2.addLayout(grid_layer_header)

        # layer (units) view
        dummy_units = []
        for i in range(128):
            dummy_pxim_unit = QPixmap(QSize(112, 112))
            dummy_pxim_unit.fill(Qt.darkGreen)
            dummy_units.append(dummy_pxim_unit)
        layer_view = LayerViewWidget(dummy_units)
        vbox2.addWidget(layer_view)
        # endregion

        # region vbox3
        vbox3 = QVBoxLayout()
        vbox3.setAlignment(Qt.AlignTop)

        # header
        combo_unit_view = QComboBox(self)
        combo_unit_view.addItem('Activations')
        combo_unit_view.addItem('Deconv')
        selected_unit_name = '0@conv1_1'
        lbl_unit_name = QLabel(
            "of unit <font color='blue'><b>%r</b></font>" % selected_unit_name)  # todo: delete default value
        hbox_unit_view_header = QHBoxLayout()
        hbox_unit_view_header.addWidget(combo_unit_view)
        hbox_unit_view_header.addWidget(lbl_unit_name)
        vbox3.addLayout(hbox_unit_view_header)

        # region settings of unit view

        # overlay setting
        hbox_overlay = QHBoxLayout()
        hbox_overlay.addWidget(QLabel("Overlay: "))
        combo_unit_overlay = QComboBox(self)
        combo_unit_overlay.addItem("No Overlay")
        combo_unit_overlay.addItem("Over active")
        combo_unit_overlay.addItem("Over inactive")
        hbox_overlay.addWidget(combo_unit_overlay)
        vbox3.addLayout(hbox_overlay)

        # Backprop Mode setting
        hbox_backprop_mode = QHBoxLayout()
        hbox_backprop_mode.addWidget(QLabel("Backprop mode: "))
        combo_unit_backprop_mode = QComboBox(self)
        combo_unit_backprop_mode.addItem("No backprop")
        combo_unit_backprop_mode.addItem("Gradient")
        combo_unit_backprop_mode.addItem("ZF deconv")
        combo_unit_backprop_mode.addItem("Guided backprop")
        hbox_backprop_mode.addWidget(combo_unit_backprop_mode)
        vbox3.addLayout(hbox_backprop_mode)

        # Backprop view setting
        hbox_backprop_view = QHBoxLayout()
        hbox_backprop_view.addWidget(QLabel("Backprop view: "))
        combo_unit_backprop_view = QComboBox(self)
        combo_unit_backprop_view.addItem("Raw")
        combo_unit_backprop_view.addItem("Gray")
        combo_unit_backprop_view.addItem("Norm")
        combo_unit_backprop_view.addItem("Blurred norm")
        hbox_backprop_view.addWidget(combo_unit_backprop_view)
        vbox3.addLayout(hbox_backprop_view)

        # endregion

        # unit image
        pixm_unit = QPixmap(QSize(224, 224))
        pixm_unit.fill(Qt.darkCyan)
        lbl_unit_image = QLabel()
        lbl_unit_image.setPixmap(pixm_unit)
        lbl_unit_image.setAlignment(Qt.AlignCenter)
        lbl_unit_image.setMargin(0)
        lbl_unit_image.setContentsMargins(QMargins(0, 0, 0, 0))
        lbl_unit_image.setFixedSize(QSize(240, 240))
        lbl_unit_image.setStyleSheet("QWidget {background-color: blue}")
        vbox3.addWidget(lbl_unit_image)

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
        self.setCentralWidget(central_widget)
        self.setWindowTitle('VGG16 Visualizer')
        self.center()
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
