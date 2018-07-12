from PyQt5.QtCore import pyqtSignal, QThread
import time
import sys, os


class CNN_Vis_Demo_Ctl(QThread):
    isBusy = pyqtSignal(bool)

    def __init__(self, model):
        super(QThread, self).__init__()
        self.model = model
        self.FLAG_set_model = False
        self.FLAG_set_input = False
        self.model_name = ''
        self.input_image_name = ''
        self.FLAG_video = False
        self.camera_as_source = False  # True for Video; False for Image

    def run(self):
        """ ops must be run here to avoid conflict """
        sys.path.insert(0, os.path.join(self.model.caffevis_caffe_root, 'python'))
        import caffe
        caffe.set_mode_gpu()  # otherwise caffe will run with cpu in this thread
        while True:
            if self.FLAG_set_model:
                self._set_model(self.model_name)
            self._switch_source(self.camera_as_source)
            if self.FLAG_video:
                self._set_input_image(None, True)
            else:
                if self.FLAG_set_input:
                    self._set_input_image(self.input_image_name)
            time.sleep(0.03)

    def set_model(self, model_name):
        self.FLAG_set_model = True
        self.model_name = model_name

    def set_input_image(self, image_name):
        self.FLAG_set_input = True
        self.input_image_name = image_name

    def switch_source(self, source):
        self.camera_as_source = source == 'Video'  # True for Video; False for Image

    def _switch_source(self, source):
        """
        Switch on the camera, then change the FLAG_video to start using the camera as input source.
        :param source:
        :return:
        """
        if self.FLAG_video != source:
            if source:
                self.model.switch_camera(True)
                self.FLAG_video = True
                self.isBusy.emit(True)  # always busy when using camera as source
            else:
                self.model.switch_camera(False)
                self.FLAG_video = False
                self.isBusy.emit(False)

    def _set_model(self, model_name):
        self.FLAG_set_model = False
        self.isBusy.emit(True)
        if model_name and model_name != '':
            self.model.set_model(model_name)
        self.isBusy.emit(False)

    def _set_input_image(self, image_name, video=False):
        self.FLAG_set_input = False
        self.isBusy.emit(True)
        self.model.set_input_and_forward(image_name, video)
        self.isBusy.emit(False)
