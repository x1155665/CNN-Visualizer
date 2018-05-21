import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QThread
import os
from VGG16_Vis_Demo_Model import VGG16_Vis_Demo_Model
import time
from multiprocessing import Process
import caffe


class VGG16_Vis_Demo_Ctl(QThread):
    isBusy = pyqtSignal(bool)

    def __init__(self, model):
        super(QThread, self).__init__()
        self.model = model
        self.FLAG_set_model = False
        self.FLAG_set_input = False
        self.model_name = ''
        self.input_image_name = ''
        self.start_time=0

    def run(self):
        caffe.set_mode_gpu()  # otherwise caffe will run with cpu in this thread !!!
        while True:
            if self.FLAG_set_model:
                self._set_model(self.model_name)
            if self.FLAG_set_input:
                self._set_input_image(self.input_image_name)
            time.sleep(0.1)

    def set_model(self, model_name):
        self.FLAG_set_model = True
        self.model_name = model_name

    def set_input_image(self, image_name):
        self.FLAG_set_input = True
        self.input_image_name = image_name

    def _set_model(self, model_name):
        self.FLAG_set_model = False
        self.isBusy.emit(True)
        if model_name and model_name != '':
            self.model.set_model(model_name)
        self.isBusy.emit(False)

    def _set_input_image(self, image_name):
        self.FLAG_set_input = False
        print(time.time() - self.start_time)
        self.start_time = time.time()
        self.isBusy.emit(True)
        if image_name and image_name != '':
            self.model.set_input_and_forward(image_name)
        self.isBusy.emit(False)

