import caffe
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage
import os
import cv2
import matplotlib.pyplot as plt
from enum import Enum
from multiprocessing import Process, Queue, Manager
import time

model_names = ['icons', 'Simpsons']
model_folders = {"icons": "./Models/icons", "Simpsons": "./Models/Simpsons"}
weights_file = {"icons": "./Models/icons/icons_vgg16.caffemodel",
                "Simpsons": "./Models/Simpsons/Simpsons_vgg16.caffemodel"}
prototxts = {"icons": "./Models/icons/VGG_ICONS_16_layers_deploy.prototxt",
             "Simpsons": "./Models/Simpsons/VGG_16_Simpsons_deploy.prototxt"}
label_files = {"icons": "./Models/icons/labels.txt", "Simpsons": "./Models/Simpsons/labels.txt"}
input_image_paths = {"icons": "./Models/icons/input_images", "Simpsons": "./Models/Simpsons/input_images"}
deepvis_outputs_paths = {"Simpsons": "../Simpsons_data/deepvis_outputs"}

caffevis_caffe_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(caffe.__file__))))

# todo: read layer info from net
vgg16_layer_names = ['conv1_1', 'conv1_2', 'pool1',
                     'conv2_1', 'conv2_2', 'pool2', 'conv3_1',
                     'conv3_2', 'conv3_3', 'pool3', 'conv4_1',
                     'conv4_2', 'conv4_3', 'pool4', 'conv5_1',
                     'conv5_2', 'conv5_3', 'pool5', 'fc6',
                     'fc7', 'fc8']
vgg16_layer_output_sizes = {'conv1_1': (224, 224, 64), 'conv1_2': (224, 224, 64),
                            'pool1': (112, 112, 64),
                            'conv2_1': (112, 112, 128), 'conv2_2': (112, 112, 128), 'pool2': (56, 56, 128),
                            'conv3_1': (56, 56, 256), 'conv3_2': (56, 56, 256), 'conv3_3': (56, 56, 256),
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
    data_idx_labels = 7
    data_idx_new_input = 128
    data_idx_input_image_path = 8

    class BackpropModeOption(Enum):
        GRADIENT = 'Gradient'
        ZF = 'ZF Deconv'
        GUIDED = 'Guided Backprop'

    def __init__(self):
        super(QObject, self).__init__()
        # caffe.set_device(0)
        caffe.set_mode_gpu()
        self.cap = cv2.VideoCapture(0)

        # setting frame size not working properly
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

        self.online = False

    def set_model(self, model_name):
        if model_names.__contains__(model_name):
            self.load_net(model_name)

    def load_net(self, model_name):
        self._model_name = model_name
        self._model_def = prototxts[model_name]
        self._model_weights = weights_file[model_name]
        self._labels = np.loadtxt(label_files[model_name], str, delimiter='\n')

        processed_prototxt = self._process_network_proto(self._model_def)
        self._net = caffe.Classifier(processed_prototxt, self._model_weights, mean=mean, raw_scale=255.0,
                                     channel_swap=(0, 1, 2))
        # self._net.transformer.set_mean(self._net.inputs[0], mean)
        current_input_shape = self._net.blobs[self._net.inputs[0]].shape
        current_input_shape[0] = 1
        self._net.blobs[self._net.inputs[0]].reshape(*current_input_shape)
        self._net.reshape()

        self._input_image_names = [icon_name for icon_name in os.listdir(input_image_paths[self._model_name])]
        self.dataChanged.emit(self.data_idx_input_image_names)
        self._transformer = caffe.io.Transformer({'data': self._net.blobs['data'].data.shape})
        self._transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
        self._transformer.set_mean('data', mean)  # subtract the dataset-mean value in each channel
        self._transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]

    def set_input_and_forward(self, input_image_name, video=False):
        """
        use static image file or camera as input to forward the network.
        If video is set, input_image_name will be ignored.
        View will be informed to resfresh the content.
        :param input_image_name: The file name of the local image file
        :param video: set True to use camera s input
        """

        def _forward_image(_image):
            input_image = caffe.io.resize(_image, [224, 224], mode='constant', cval=0)
            self._input_image = (input_image * 255).astype(np.uint8)
            transformed_image = self._transformer.preprocess('data', input_image)
            self._net.blobs['data'].data[...] = transformed_image
            start = time.time()
            self._net.forward()
            print('forward time: ' + str(time.time() - start))
            self.online = True
            self.dataChanged.emit(self.data_idx_new_input)

        def _square(_image):
            # adjust image dimensions so that the image will be expanded to the largest side
            # padding order: top, bottom, left, right
            [height, width, _] = _image.shape
            # icon portrait mode
            if width < height:
                pad_size = height - width
                if pad_size % 2 == 0:
                    icon_squared = cv2.copyMakeBorder(_image, 0, 0, pad_size / 2, pad_size / 2, cv2.BORDER_CONSTANT,
                                                      value=[0, 0, 0])
                else:
                    icon_squared = cv2.copyMakeBorder(_image, 0, 0, pad_size / 2 + 1, pad_size / 2, cv2.BORDER_CONSTANT,
                                                      value=[0, 0, 0])
                return icon_squared
            # icon landscape mode
            elif height < width:
                pad_size = width - height
                if pad_size % 2 == 0:
                    # top, bottom, left, right
                    icon_squared = cv2.copyMakeBorder(_image, pad_size / 2, pad_size / 2, 0, 0, cv2.BORDER_CONSTANT,
                                                      value=[0, 0, 0])
                else:
                    icon_squared = cv2.copyMakeBorder(_image, pad_size / 2 + 1, pad_size / 2, 0, 0, cv2.BORDER_CONSTANT,
                                                      value=[0, 0, 0])
                return icon_squared
            elif height == width:
                return _image

        def _crop_max_square(_image):
            # crop the biggest square from the center of a image
            h, w, c = _image.shape
            l = min(h, w)
            if (h + l) % 2 != 0:
                _image = _image[(h - l + 1) / 2:(h + l + 1) / 2, :, :]
            elif (w + l) % 2 != 0:
                _image = _image[:, (w - l + 1) / 2:(w + l + 1) / 2, :]
            else:
                _image = _image[(h - l) / 2:(h + l) / 2, (w - l) / 2:(w + l) / 2, :]
            return _image

        if video:
            ret, frame = self.cap.read()
            squared_image = _crop_max_square(frame)
            _forward_image(cv2.flip(squared_image[:, :, (2, 1, 0)], 1))
        else:
            if self._input_image_names.__contains__(input_image_name):
                self._input_image_path = os.path.join(input_image_paths[self._model_name], input_image_name)
                image = caffe.io.load_image(self._input_image_path)
                image = _square(image)
                _forward_image(image)

    def get_data(self, data_idx):
        if data_idx == self.data_idx_model_names:
            return model_names
        elif data_idx == self.data_idx_layer_names:
            return vgg16_layer_names
        elif data_idx == self.data_idx_layer_output_sizes:
            return vgg16_layer_output_sizes
        elif data_idx == self.data_idx_probs:
            return self._net.blobs['prob'].data.flatten()
        elif data_idx == self.data_idx_input_image_names:
            return self._input_image_names
        elif data_idx == self.data_idx_labels:
            return self._labels
        elif data_idx == self.data_idx_input_image_path:
            return self._input_image_path
        elif data_idx == self.data_idx_input_image:
            return self._input_image

    def get_activations(self, layer_name):
        if self.online and vgg16_layer_names.__contains__(layer_name):
            activations = self._net.blobs[layer_name].data[0]
            return activations

    def get_activation(self, layer_name, unit_index):
        if self.online and vgg16_layer_names.__contains__(layer_name) and unit_index < \
                vgg16_layer_output_sizes[layer_name][len(vgg16_layer_output_sizes[layer_name]) - 1]:
            activation = self._net.blobs[layer_name].data[0][unit_index]
            return activation

    def get_top_k_images_of_unit(self, layer_name, unit_index, k, get_deconv):
        if self.online and self._model_name in deepvis_outputs_paths and vgg16_layer_names.__contains__(layer_name) \
                and unit_index < vgg16_layer_output_sizes[layer_name][len(vgg16_layer_output_sizes[layer_name]) - 1]:
            unit_dir = os.path.join(deepvis_outputs_paths[self._model_name], layer_name, 'unit_%04d' % unit_index)
            assert k <= 9
            if get_deconv:
                type = 'deconvnorm'
            else:
                type = 'maxim'
            pixmaps = []
            for i in range(k):
                file_name = '%s_%03d.png' % (type, i)
                file_path = os.path.join(unit_dir, file_name)
                assert os.path.exists(file_path)
                pixmaps.append(QPixmap(file_path))
            return pixmaps

    def get_top_1_images_of_layer(self, layer_name):
        if self.online and self._model_name in deepvis_outputs_paths and vgg16_layer_names.__contains__(layer_name):
            channel_number = vgg16_layer_output_sizes[layer_name][len(vgg16_layer_output_sizes[layer_name])-1]
            pixmaps = []
            for unit_index in range(channel_number):
                unit_dir = os.path.join(deepvis_outputs_paths[self._model_name], layer_name, 'unit_%04d' % unit_index)
                file_name = 'maxim_000.png'
                file_path = os.path.join(unit_dir, file_name)
                pixmaps.append(QPixmap(file_path))
            return pixmaps

    def get_deconv(self, layer_name, unit_index, backprop_mode):
        diffs = self._net.blobs[layer_name].diff[0]
        diffs = diffs * 0
        data = self._net.blobs[layer_name].data[0]
        diffs[unit_index] = data[unit_index]
        diffs = np.expand_dims(diffs, 0)  # add batch dimension
        layer_name = str(layer_name)

        if backprop_mode == self.BackpropModeOption.GRADIENT.value:
            result = self._net.backward_from_layer(layer_name, diffs, zero_higher=True)
        elif backprop_mode == self.BackpropModeOption.ZF.value:
            result = self._net.deconv_from_layer(layer_name, diffs, zero_higher=True, deconv_type='Zeiler & Fergus')
        elif backprop_mode == self.BackpropModeOption.GUIDED.value:
            result = self._net.deconv_from_layer(layer_name, diffs, zero_higher=True, deconv_type='Guided Backprop')
        else:
            result = None
        if result is not None:
            result = np.transpose(result[self._net.inputs[0]][0], (1, 2, 0))
        return result

    def _get_sorted_probs(self):
        results = self._net.blobs['prob'].data.flatten()
        sorted_results_idx = sorted(range(len(results)), reverse=True, key=lambda k: results[k])
        evaluation = [{self._labels[sorted_results_idx[k]]: results[sorted_results_idx[k]]} for k in
                      range(len(results))]
        return evaluation

    def _process_network_proto(self, prototxt):
        processed_prototxt = prototxt + ".processed_by_deepvis"

        # check if force_backwards is missing
        found_force_backwards = False
        with open(prototxt, 'r') as proto_file:
            for line in proto_file:
                fields = line.strip().split()
                if len(fields) == 2 and fields[0] == 'force_backward:' and fields[1] == 'true':
                    found_force_backwards = True
                    break

        # write file, adding force_backward if needed
        with open(prototxt, 'r') as proto_file:
            with open(processed_prototxt, 'w') as new_proto_file:
                if not found_force_backwards:
                    new_proto_file.write('force_backward: true\n')
                for line in proto_file:
                    new_proto_file.write(line)

        # run upgrade tool on new file name (same output file)
        upgrade_tool_command_line = caffevis_caffe_root + '/build/tools/upgrade_net_proto_text.bin ' + processed_prototxt + ' ' + processed_prototxt
        os.system(upgrade_tool_command_line)

        return processed_prototxt

    def switch_camera(self, on):
        if on:
            self.cap.open(0)
        else:
            self.cap.release()
