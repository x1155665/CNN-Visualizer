#! /usr/bin/env python

# this import must comes first to make sure we use the non-display backend
import matplotlib

matplotlib.use('Agg')
# add parent folder to search path, to enable import of core modules like settings

import argparse
from jby_misc import WithTimer
from max_tracker import output_max_patches
from find_max_act import load_max_tracker_from_file
from misc import load_network
# add parent folder to search path, to enable import of core modules like settings

import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import Settings


def main():
    parser = argparse.ArgumentParser(
        description='Loads a pickled NetMaxTracker and outputs one or more of {the patches of the image, a deconv patch, a backprop patch} associated with the maxes.')
    parser.add_argument('--model')
    parser.add_argument('--idx-begin', type=int, default=None, help='Start at this unit (default: all units).')
    parser.add_argument('--idx-end', type=int, default=None, help='End at this unit (default: all units).')
    args = parser.parse_args()
    settings = Settings.Settings()
    settings.load_settings(args.model)

    net = load_network(settings)

    # set network batch size
    current_input_shape = net.blobs[net.inputs[0]].shape
    current_input_shape[0] = settings.max_tracker_batch_size
    net.blobs[net.inputs[0]].reshape(*current_input_shape)
    net.reshape()

    assert settings.max_tracker_do_maxes or settings.max_tracker_do_deconv or settings.max_tracker_do_deconv_norm or settings.max_tracker_do_backprop or settings.max_tracker_do_backprop_norm, 'Specify at least one do_* option to output.'

    nmt = load_max_tracker_from_file(settings.find_maxes_output_file)

    for layer_name in settings.layers_to_output_in_offline_scripts:

        print('Started work on layer %s' % (layer_name))

        mt = nmt.max_trackers[layer_name]

        if args.idx_begin is None:
            idx_begin = 0
        if args.idx_end is None:
            idx_end = mt.max_vals.shape[0]

        with WithTimer('Saved %d images per unit for %s units %d:%d.' % (settings.N, layer_name, idx_begin, idx_end)):

            output_max_patches(settings, mt, net, layer_name, idx_begin, idx_end,
                               settings.N, settings.data_dir, settings.deepvis_outputs_path, False,
                               (settings.max_tracker_do_maxes, settings.max_tracker_do_deconv,
                                settings.max_tracker_do_deconv_norm, settings.max_tracker_do_backprop,
                                settings.max_tracker_do_backprop_norm, True))

            if settings.search_min:
                output_max_patches(settings, mt, net, layer_name, idx_begin, idx_end,
                                   settings.N, settings.data_dir, settings.deepvis_outputs_path, True,
                                   (settings.max_tracker_do_maxes, settings.max_tracker_do_deconv,
                                    settings.max_tracker_do_deconv_norm, settings.max_tracker_do_backprop,
                                    settings.max_tracker_do_backprop_norm, True))


if __name__ == '__main__':
    main()
