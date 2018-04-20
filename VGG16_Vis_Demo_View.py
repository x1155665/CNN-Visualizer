# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (QAction, qApp, QTextEdit, QMainWindow, QMessageBox, QDesktopWidget, QLabel, QComboBox,
                             QPushButton, QWidget, QApplication, QMenu, QHBoxLayout, QVBoxLayout, QGridLayout,
                             QLCDNumber, QSlider, QLineEdit, QRadioButton, QGroupBox, QScrollArea,
                             QInputDialog, QFrame, QColorDialog, QFileDialog, QProgressBar, QSplitter)
from PyQt5.QtGui import QFont, QIcon, QColor, QPainter, QPen, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QBasicTimer, QSize

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


class VGG16_Vis_Demo_View(QMainWindow):

    def __init__(self):
        super(QWidget, self).__init__()
        self.initUI()

    def initUI(self):
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
            btn = QRadioButton(layer_name)
            btn.setFont(QFont('Times', 11, QFont.Bold))
            vbox_network.addWidget(btn)
            lbl_arrow = QLabel(' ⬇️ '+ (vgg16_layers[layer_name]))
            lbl_arrow.setFont(QFont("Helvetica", 8))
            vbox_network.addWidget(lbl_arrow)
        wrapper_vbox_network = QWidget()
        wrapper_vbox_network.setLayout(vbox_network)
        scroll_network = QScrollArea()
        scroll_network.setFrameShape(QFrame.Box)
        scroll_network.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
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

        hbox = QHBoxLayout()
        hbox.addLayout(vbox1)
        splitter1 = QSplitter(Qt.Horizontal)

        splitter2 = QSplitter(Qt.Vertical)

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
