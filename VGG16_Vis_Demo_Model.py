import caffe
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage
import os

model_names = ['icons']
model_folders = {"icons": "./Models/icons"}
weights_file = {"icons": "./Models/icons/icons_vgg16.caffemodel"}
prototxts = {"icons": "./Models/icons/VGG_ICONS_16_layers_deploy.prototxt"}
label_files = {"icons": "./Models/icons/labels.txt"}
input_image_paths = {"icons": "./Models/icons/input_images"}

# todo: read layer info from net
vgg16_layer_names = ['input', 'conv1_1', 'conv1_2', 'pool1',
                     'conv2_1', 'conv2_2', 'pool2', 'conv3_1',
                     'conv_3_2', 'conv_3_3', 'pool3', 'conv4_1',
                     'conv4_2', 'conv4_3', 'pool4', 'conv5_1',
                     'conv5_2', 'conv5_3', 'pool5', 'fc6',
                     'fc7', 'fc8']
vgg16_layer_output_sizes = {'input': (224, 224, 3), 'conv1_1': (224, 224, 64), 'conv1_2': (224, 224, 64),
                            'pool1': (112, 112, 64),
                            'conv2_1': (112, 112, 128), 'conv2_2': (112, 112, 128), 'pool2': (56, 56, 128),
                            'conv3_1': (56, 56, 256), 'conv_3_2': (56, 56, 256), 'conv_3_3': (56, 56, 256),
                            'pool3': (28, 28, 256),
                            'conv4_1': (28, 28, 512), 'conv4_2': (28, 28, 512), 'conv4_3': (28, 28, 512),
                            'pool4': (14, 14, 512),
                            'conv5_1': (14, 14, 512), 'conv5_2': (14, 14, 512,), 'conv5_3': (14, 14, 512),
                            'pool5': (7, 7, 512),
                            'fc6': [4096], 'fc7': [4096], 'fc8': [16]}

mean = np.array([103.939, 116.779, 123.68])


class VGG16_Vis_Demo_Model(QObject):
    dataChanged = pyqtSignal(int)

    data_idx_model_names = 0
    data_idx_layer_names = 1
    data_idx_layer_output_sizes = 2
    data_idx_layer_activation = 3
    data_idx_probs = 4
    data_idx_input_image_names = 5
    data_idx_input_image = 6

    def __init__(self):
        super(QObject, self).__init__()
        caffe.set_mode_cpu()

    def set_model(self, model_name):
        if model_names.__contains__(model_name):
            self.load_net(model_name)

    def load_net(self, model_name):
        self.model_name = model_name
        self.model_def = prototxts[model_name]
        self.model_weights = weights_file[model_name]
        self.labels = np.loadtxt(label_files[model_name], str, delimiter='\n')
        self.net = caffe.Net(self.model_def, self.model_weights, caffe.TEST)
        self.input_image_names = [icon_name for icon_name in os.listdir(input_image_paths[self.model_name]) if
                           ".png" in icon_name]
        self.dataChanged.emit(5)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
        self.transformer.set_mean('data', mean)  # subtract the dataset-mean value in each channel
        self.transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]

    def set_input_and_forward(self, input_image_name):
        if self.input_image_names.__contains__(input_image_name):
            self.input_image_path = os.path.join(input_image_paths[self.model_name], input_image_name)
            image = caffe.io.load_image(self.input_image_path)
            image = caffe.io.resize(image, [224, 224], mode='constant', cval=0)
            transformed_image = self.transformer.preprocess('data', image)
            self.net.blobs['data'].data[...] = transformed_image
            self.dataChanged.emit(self.data_idx_input_image)
            self.net.forward()
            self.dataChanged.emit(self.data_idx_probs)

    def get_data(self, data_idx):
        if data_idx == self.data_idx_model_names:
            return model_names
        elif data_idx == self.data_idx_layer_names:
            return vgg16_layer_names
        elif data_idx == self.data_idx_layer_output_sizes:
            return vgg16_layer_output_sizes
        elif data_idx == self.data_idx_probs:
            return self.net.blobs['probs'].data.flatten()
        elif data_idx == self.data_idx_input_image_names:
            return self.input_image_names
        elif data_idx == self.data_idx_input_image:
            return QPixmap(self.input_image_path)
