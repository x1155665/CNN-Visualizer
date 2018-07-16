# this import must comes first to make sure we use the non-display backend
import matplotlib
matplotlib.use('Agg')

import os
from jby_misc import WithTimer
from max_tracker import scan_images_for_maxes
from misc import mkdir_p
import pickle as pickle
from misc import load_network
import argparse

# add parent folder to search path, to enable import of core modules like settings
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import Settings


def main():
    parser = argparse.ArgumentParser(
        description='Finds images in a training set that cause max activation for a network; saves results in a pickled NetMaxTracker.')
    parser.add_argument('--model')
    args = parser.parse_args()
    settings = Settings.Settings()
    settings.load_settings(args.model)
    net = load_network(settings)

    # set network batch size
    current_input_shape = net.blobs[net.inputs[0]].shape
    current_input_shape[0] = settings.max_tracker_batch_size
    net.blobs[net.inputs[0]].reshape(*current_input_shape)
    net.reshape()

    with WithTimer('Scanning images'):
        net_max_tracker = scan_images_for_maxes(settings, net, settings.data_dir, settings.N, settings.deepvis_outputs_path, settings.search_min)

    save_max_tracker_to_file(settings.find_maxes_output_file, net_max_tracker)


def save_max_tracker_to_file(filename, net_max_tracker):

    dir_name = os.path.dirname(filename)
    mkdir_p(dir_name)

    with WithTimer('Saving maxes'):
        with open(filename, 'wb') as ff:
            pickle.dump(net_max_tracker, ff, -1)
        # save text version of pickle file for easier debugging
        pickle_to_text(filename)


def pickle_to_text(pickle_filename):
    with open(pickle_filename, 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    data_dict = data.__dict__.copy()

    with open(pickle_filename + ".txt", 'wt') as text_file:
        text_file.write(str(data_dict))

    return


def load_max_tracker_from_file(filename):
    with open(filename, 'rb') as tracker_file:
        net_max_tracker = pickle.load(tracker_file)

    return net_max_tracker


if __name__ == '__main__':
    main()