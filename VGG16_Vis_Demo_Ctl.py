import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
import os
from VGG16_Vis_Demo_Model import VGG16_Vis_Demo_Model


class VGG16_Vis_Demo_Ctl(object):
    def __init__(self, model):
        self.model = model

    def set_model(self, model_name):
        if self.statusbar:
            self.statusbar.showMessage('busy')
        if model_name and model_name != '':
            self.model.set_model(model_name)
        if self.statusbar:
            self.statusbar.showMessage('ready')

    def set_input_image(self, image_name):
        if self.statusbar:
            self.statusbar.showMessage('busy')
        if image_name and image_name != '':
            self.model.set_input_and_forward(image_name)
        if self.statusbar:
            self.statusbar.showMessage('ready')

    def set_statusbar_instance(self, statusbar):
        self.statusbar = statusbar
