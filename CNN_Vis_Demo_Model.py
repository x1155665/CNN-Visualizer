import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap
import os, sys
import cv2
from enum import Enum
import time
from Settings import Settings


# TODO: Improve the notification mechanism between model and view
class CNN_Vis_Demo_Model(QObject):
    # These indices are used to notify the view about the changes
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

    dataChanged = pyqtSignal(int)

    settings = None

    class BackpropModeOption(Enum):
        GRADIENT = 'Gradient'
        ZF = 'ZF Deconv'
        GUIDED = 'Guided Backprop'

    def __init__(self):
        super(QObject, self).__init__()
        self.settings = Settings()  # Read settings from files

        self.caffevis_caffe_root = self.settings.caffevis_caffe_root
        sys.path.insert(0, os.path.join(self.caffevis_caffe_root, 'python'))
        import caffe
        if self.settings.use_GPU:
            caffe.set_mode_gpu()
            caffe.set_device(self.settings.gpu_id)
            print('Loaded caffe in GPU mode, using device', self.settings.gpu_id)
        else:
            caffe.set_mode_cpu()
            print('Loaded caffe in CPU mode')
        self.camera_id = self.settings.camera_id
        self.cap = cv2.VideoCapture(self.camera_id)

        self._layer_list = []  # to be read from prototxt
        self._layer_output_sizes = {}  # to be read from prototxt

        self.online = False  # indicates if the network has finished classifying an image

    def set_model(self, model_name):
        """
        set the network model
        :param model_name:
        :return:
        """
        if self.settings.model_names.__contains__(model_name):
            self.settings.load_settings(model_name)
            self.online = False
            self.load_net(model_name)

    def load_net(self, model_name):
        sys.path.insert(0, os.path.join(self.caffevis_caffe_root, 'python'))
        import caffe

        self._model_name = model_name
        self._model_def = self.settings.prototxt
        self._model_weights = self.settings.network_weights
        self._labels = np.loadtxt(self.settings.label_file, str, delimiter='\n')

        processed_prototxt = self._process_network_proto(self._model_def)  # enable deconvolution
        self._net = caffe.Classifier(processed_prototxt, self._model_weights, mean=self.settings.mean, raw_scale=255.0,
                                     channel_swap=self.settings.channel_swap)
        current_input_shape = self._net.blobs[self._net.inputs[0]].shape
        current_input_shape[0] = 1
        self._net.blobs[self._net.inputs[0]].reshape(*current_input_shape)
        self._net.reshape()
        self._get_layers_info()
        self.dataChanged.emit(self.data_idx_layer_names)

        # get the names of demo-images
        self._input_image_names = [image_name for image_name in os.listdir(self.settings.input_image_path)]
        self.dataChanged.emit(self.data_idx_input_image_names)

        self._transformer = caffe.io.Transformer(
            {self._data_blob_name: self._net.blobs[self._data_blob_name].data.shape})
        self._transformer.set_transpose(self._data_blob_name, (2, 0, 1))  # move image channels to outermost dimension
        self._transformer.set_mean(self._data_blob_name,
                                   self.settings.mean)  # subtract the dataset-mean value in each channel
        self._transformer.set_raw_scale(self._data_blob_name, 255)  # rescale from [0, 1] to [0, 255]
        self._transformer.set_channel_swap(self._data_blob_name, self.settings.channel_swap)

    def set_input_and_forward(self, input_image_name, video=False):
        """
        use static image file or camera as input to forward the network.
        View will be informed to resfresh the content.
        If video is set, input_image_name will be ignored.
        :param input_image_name: The file name of the local image file
        :param video: set True to use camera as input
        """
        sys.path.insert(0, os.path.join(self.caffevis_caffe_root, 'python'))
        import caffe

        def _forward_image(_image):
            input_image = caffe.io.resize(_image, self._input_dims, mode='constant', cval=0)
            self._input_image = (input_image * 255).astype(np.uint8)
            transformed_image = self._transformer.preprocess(self._data_blob_name, input_image)
            self._net.blobs[self._data_blob_name].data[...] = transformed_image
            self._net.forward()
            self.online = True
            self.dataChanged.emit(self.data_idx_new_input)

        def _square(_image):
            """
            adjust image dimensions so that the image will be expanded to the largest side padding order: top, bottom, left, right
            :param _image: image to be processed
            :return: processed image
            """
            [height, width, _] = _image.shape
            # icon portrait mode
            if width < height:
                pad_size = height - width
                if pad_size % 2 == 0:
                    icon_squared = cv2.copyMakeBorder(_image, 0, 0, pad_size // 2, pad_size // 2, cv2.BORDER_CONSTANT,
                                                      value=[0, 0, 0])
                else:
                    icon_squared = cv2.copyMakeBorder(_image, 0, 0, pad_size // 2 + 1, pad_size // 2, cv2.BORDER_CONSTANT,
                                                      value=[0, 0, 0])
                return icon_squared
            # icon landscape mode
            elif height < width:
                pad_size = width - height
                if pad_size % 2 == 0:
                    # top, bottom, left, right
                    icon_squared = cv2.copyMakeBorder(_image, pad_size // 2, pad_size // 2, 0, 0, cv2.BORDER_CONSTANT,
                                                      value=[0, 0, 0])
                else:
                    icon_squared = cv2.copyMakeBorder(_image, pad_size // 2 + 1, pad_size // 2, 0, 0, cv2.BORDER_CONSTANT,
                                                      value=[0, 0, 0])
                return icon_squared
            elif height == width:
                return _image

        def _crop_max_square(_image):
            """
            crop the biggest square at the center of a image
            :param _image: image to be processed
            :return: processed image
            """
            h, w, c = _image.shape
            l = min(h, w)
            if (h + l) % 2 != 0:
                _image = _image[(h - l + 1) // 2:(h + l + 1) // 2, :, :]
            elif (w + l) % 2 != 0:
                _image = _image[:, (w - l + 1) // 2:(w + l + 1) // 2, :]
            else:
                _image = _image[(h - l) // 2:(h + l) // 2, (w - l) // 2:(w + l) // 2, :]
            return _image

        if video:
            ret, frame = self.cap.read()
            squared_image = _crop_max_square(frame)
            _forward_image(cv2.flip(squared_image[:, :, (2, 1, 0)], 1))  # RGB
        else:
            if self._input_image_names.__contains__(input_image_name):
                self._input_image_path = os.path.join(self.settings.input_image_path, input_image_name)
                image = caffe.io.load_image(self._input_image_path)  # RGB
                image = _square(image)
                _forward_image(image)

    def get_data(self, data_idx):
        """
        Use the data index to get the data.
        The intend was to add control logic in access. But, this seems to be useless.
        :param data_idx:
        :return: Desired data
        """
        if data_idx == self.data_idx_model_names:
            return self.settings.model_names
        elif data_idx == self.data_idx_layer_names:
            return self._layer_list
        elif data_idx == self.data_idx_layer_output_sizes:
            return self._layer_output_sizes
        elif data_idx == self.data_idx_probs:
            return self._net.blobs[self._props_blob_name].data.flatten()
        elif data_idx == self.data_idx_input_image_names:
            return self._input_image_names
        elif data_idx == self.data_idx_labels:
            return self._labels
        elif data_idx == self.data_idx_input_image_path:
            return self._input_image_path
        elif data_idx == self.data_idx_input_image:
            return self._input_image

    def get_activations(self, layer_name):
        """
        Get all the activations of one layer
        :param layer_name:
        :return: activations (N, H, W)
        """
        if self.online and self._layer_list.__contains__(layer_name):
            activations = self._net.blobs[layer_name].data[0]
            return activations

    def get_activation(self, layer_name, unit_index):
        """
        Get the activation of a neuron
        :param layer_name:
        :return: activations (H, W)
        """
        if self.online and self._layer_list.__contains__(layer_name) and unit_index < \
                self._layer_output_sizes[layer_name][0]:
            activation = self._net.blobs[layer_name].data[0][unit_index]
            return activation

    def get_top_k_images_of_unit(self, layer_name, unit_index, k, get_deconv):
        """
        Get k images with highest acivation to one certain neuron.
        :param layer_name:
        :param unit_index:
        :param k:
        :param get_deconv: Get the deconv results of the top images
        :return: Desired top k images
        """
        if self.online and self.settings.deepvis_outputs_path and self._layer_list.__contains__(layer_name) \
                and unit_index < self._layer_output_sizes[layer_name][0]:
            unit_dir = os.path.join(self.settings.deepvis_outputs_path, layer_name, 'unit_%04d' % unit_index)
            assert k <= 9
            if get_deconv:
                type = 'deconvnorm'
            else:
                type = 'maxim'
            pixmaps = []
            for i in range(k):
                file_name = '%s_%03d.png' % (type, i)
                file_path = os.path.join(unit_dir, file_name)
                if os.path.exists(file_path):
                    pixmaps.append(QPixmap(file_path))
                else:
                    print(file_path + " not exists.")
            return pixmaps

    def get_top_1_images_of_layer(self, layer_name):
        """
        Get the top 1 images of all units in one layers
        :param layer_name:
        :return: The images with highest activations to the units
        """
        if self.online and self.settings.deepvis_outputs_path and self._layer_list.__contains__(layer_name):
            channel_number = self._layer_output_sizes[layer_name][0]
            pixmaps = []
            for unit_index in range(channel_number):
                unit_dir = os.path.join(self.settings.deepvis_outputs_path, layer_name, 'unit_%04d' % unit_index)
                file_name = 'maxim_000.png'
                file_path = os.path.join(unit_dir, file_name)
                pixmaps.append(QPixmap(file_path))
            return pixmaps

    def get_deconv(self, layer_name, unit_index, backprop_mode):
        """
        Compute the backprop/deconv of one unit
        :param layer_name:
        :param unit_index:
        :param backprop_mode: Avaliable options: self.BackpropModeOption
        :return: result
        """
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
        upgrade_tool_command_line = self.caffevis_caffe_root + '/build/tools/upgrade_net_proto_text.bin ' + processed_prototxt + ' ' + processed_prototxt
        os.system(upgrade_tool_command_line)

        return processed_prototxt

    def switch_camera(self, on):
        if on:
            self.cap.open(self.camera_id)
        else:
            self.cap.release()

    def _get_layers_info(self):
        """
        Get the layer names / output sizes / 'data' blob name / 'prob' blob name / input dimension
        :return:
        """
        self._layer_list = []
        self._layer_output_sizes = {}
        # go over layers
        all_layer_list = list(self._net._layer_names)
        total_layer_number = len(all_layer_list)
        for idx in range(total_layer_number):
            layer_name = all_layer_list[idx]
            # skip input, output and inplace layers. eg. relu
            if idx == 0 or idx == total_layer_number - 1 or (
                    len(self._net.top_names[layer_name]) == 1 and len(self._net.bottom_names[layer_name]) == 1 and
                    self._net.top_names[layer_name][0] == self._net.bottom_names[layer_name][0]):
                continue

            self._layer_list.append(layer_name)

            # get layer output size
            top_shape = self._net.blobs[layer_name].data[0].shape

            self._layer_output_sizes.update({layer_name: top_shape})

        # get data blob name
        self._data_blob_name = self._net.top_names[all_layer_list[0]][0]
        # get input dims
        self._input_dims = self._net.blobs[self._data_blob_name].data.shape[2:4]
        # get prob blob name
        self._props_blob_name = self._net.top_names[all_layer_list[-1]][0]
