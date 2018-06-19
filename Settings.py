import yaml
import numpy as np
import os

main_setting_file = './main_settings.yaml'


class Settings:
    model_names = []
    main_settings = {}

    def __init__(self):
        with open(main_setting_file) as fp:
            self.main_settings.update(yaml.load(fp))
        self.use_GPU = self.main_settings['Use_GPU']
        self.gpu_id = self.main_settings['GPU_ID']
        self.caffevis_caffe_root = self.main_settings['caffevis_caffe_root']
        for key in self.main_settings['Model_config_path']:
            self.model_names.append(key)

    def load_settings(self, model_name):
        assert model_name in self.main_settings['Model_config_path']
        print "Loading settings of " + model_name + ' from ' + self.main_settings['Model_config_path'][model_name]
        with open(self.main_settings['Model_config_path'][model_name]) as fp:
            configs = yaml.load(fp)
        self.network_weights = configs['network_weights']
        self.prototxt = configs['prototxt']
        self.label_file = configs['label_file']
        self.input_image_path = configs['input_image_path'] if 'input_image_path' in configs else None
        self.deepvis_outputs_path = configs['deepvis_outputs_path'] if 'deepvis_outputs_path' in configs else None
        self.channel_swap = configs['channel_swap'] if 'channel_swap' in configs else [0, 1, 2]
        self.mean = np.array(configs['mean'] if 'mean' in configs else [103.939, 116.779, 123.68])
        self.data_dir = configs['data_dir'] if 'data_dir' in configs else None
        self.find_maxes_output_file = os.path.join(self.deepvis_outputs_path, 'find_max_acts_output.pickled') if self.deepvis_outputs_path else None
        self.N = configs['N'] if 'N' in configs else 9
        self.layers_to_output_in_offline_scripts = configs[
            'layers_to_output_in_offline_scripts'] if 'layers_to_output_in_offline_scripts' in configs else []
        self.search_min = configs['search_min'] if 'search_min' in configs else False
        self.max_tracker_batch_size = configs['max_tracker_batch_size'] if 'max_tracker_batch_size' in configs else 1
        self.max_tracker_do_maxes = configs['max_tracker_do_maxes'] if 'max_tracker_do_maxes' in configs else True
        self.max_tracker_do_deconv = configs['max_tracker_do_deconv'] if 'max_tracker_do_deconv' in configs else False
        self.max_tracker_do_deconv_norm = configs[
            'max_tracker_do_deconv_norm'] if 'max_tracker_do_deconv_norm' in configs else False
        self.max_tracker_do_backprop = configs[
            'max_tracker_do_backprop'] if 'max_tracker_do_backprop' in configs else False
        self.max_tracker_do_backprop_norm = configs[
            'max_tracker_do_backprop_norm'] if 'max_tracker_do_backprop_norm' in configs else False
