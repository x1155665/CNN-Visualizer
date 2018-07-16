import re
import os
import sys
from jby_misc import WithTimer
from misc import layer_name_to_top_name, get_files_list, resize_without_fit, mkdir_p, get_max_data_extent, \
    compute_data_layer_focus_area, extract_patch_from_image, save_caffe_image
from datetime import datetime
import numpy as np


class MaxTrackerCropBatchRecord(object):

    def __init__(self, cc=None, channel_idx=None, info_filename=None, maxim_filenames=None,
                 deconv_filenames=None, deconvnorm_filenames=None, backprop_filenames=None,
                 backpropnorm_filenames=None, info_file=None, max_idx_0=None, max_idx=None, im_idx=None,
                 selected_input_index=None, ii=None, jj=None, recorded_val=None,
                 out_ii_start=None, out_ii_end=None, out_jj_start=None, out_jj_end=None, data_ii_start=None,
                 data_ii_end=None, data_jj_start=None, data_jj_end=None, im=None,
                 denormalized_layer_name=None, denormalized_top_name=None, layer_format=None):
        self.cc = cc
        self.channel_idx = channel_idx
        self.info_filename = info_filename
        self.maxim_filenames = maxim_filenames
        self.deconv_filenames = deconv_filenames
        self.deconvnorm_filenames = deconvnorm_filenames
        self.backprop_filenames = backprop_filenames
        self.backpropnorm_filenames = backpropnorm_filenames
        self.info_file = info_file
        self.max_idx_0 = max_idx_0
        self.max_idx = max_idx
        self.im_idx = im_idx
        self.selected_input_index = selected_input_index
        self.ii = ii
        self.jj = jj
        self.recorded_val = recorded_val
        self.out_ii_start = out_ii_start
        self.out_ii_end = out_ii_end
        self.out_jj_start = out_jj_start
        self.out_jj_end = out_jj_end
        self.data_ii_start = data_ii_start
        self.data_ii_end = data_ii_end
        self.data_jj_start = data_jj_start
        self.data_jj_end = data_jj_end
        self.im = im
        self.denormalized_layer_name = denormalized_layer_name
        self.denormalized_top_name = denormalized_top_name
        self.layer_format = layer_format


class MaxTrackerBatchRecord(object):

    def __init__(self, image_idx=None, filename=None, im=None):
        self.image_idx = image_idx
        self.filename = filename
        self.im = im


def scan_images_for_maxes(settings, net, datadir, n_top, outdir, search_min):
    image_filenames, image_labels = get_files_list(datadir)
    print('Scanning %d files' % len(image_filenames))
    print('  First file', os.path.join(datadir, image_filenames[0]))

    sys.path.insert(0, os.path.join(settings.caffevis_caffe_root, 'python'))
    import caffe

    tracker = NetMaxTracker(settings, n_top=n_top, layers=settings.layers_to_output_in_offline_scripts,
                            search_min=search_min)

    net_input_dims = net.blobs['data'].data.shape[2:4]

    # prepare variables used for batches
    batch = [None] * settings.max_tracker_batch_size
    for i in range(0, settings.max_tracker_batch_size):
        batch[i] = MaxTrackerBatchRecord()

    batch_index = 0

    for image_idx in range(len(image_filenames)):

        batch[batch_index].image_idx = image_idx
        batch[batch_index].filename = image_filenames[image_idx]

        do_print = (batch[batch_index].image_idx % 100 == 0)
        if do_print:
            print('%s   Image %d/%d' % (datetime.now().ctime(), batch[batch_index].image_idx, len(image_filenames)))

        with WithTimer('Load image', quiet=not do_print):
            try:
                batch[batch_index].im = caffe.io.load_image(os.path.join(datadir, batch[batch_index].filename),
                                                            color=True)
                batch[batch_index].im = resize_without_fit(batch[batch_index].im, net_input_dims)
                batch[batch_index].im = batch[batch_index].im.astype(np.float32)
            except:
                # skip bad/missing inputs
                print("WARNING: skipping bad/missing input:", batch[batch_index].filename)
                continue

        batch_index += 1

        # if current batch is full
        if batch_index == settings.max_tracker_batch_size \
                or image_idx == len(image_filenames) - 1:  # or last iteration

            # batch predict
            with WithTimer('Predict on batch  ', quiet=not do_print):
                im_batch = [record.im for record in batch]
                net.predict(im_batch, oversample=False)  # Just take center crop

            # go over batch and update statistics
            for i in range(0, batch_index):
                with WithTimer('Update    ', quiet=not do_print):
                    tracker.update(net, batch[i].image_idx, net_unique_input_source=batch[i].filename, batch_index=i)

            batch_index = 0

    print('done!')
    return tracker


class NetMaxTracker(object):
    def __init__(self, settings, layers, n_top=10, initial_val=-1e99, dtype='float32', search_min=False):
        self.layers = layers
        self.init_done = False
        self.n_top = n_top
        self.search_min = search_min
        self.initial_val = initial_val
        self.settings = settings

    def _init_with_net(self, net):
        self.max_trackers = {}

        for layer_name in self.layers:

            print('init layer: ', layer_name)
            top_name = layer_name_to_top_name(net, layer_name)
            blob = net.blobs[top_name].data

            is_spatial = (len(blob.shape) == 4)

            # only add normalized layer once
            if layer_name not in self.max_trackers:
                self.max_trackers[layer_name] = MaxTracker(is_spatial, blob.shape[1], n_top=self.n_top,
                                                           initial_val=self.initial_val,
                                                           dtype=blob.dtype, search_min=self.search_min)

        self.init_done = True

    def update(self, net, image_idx, net_unique_input_source, batch_index):
        '''Updates the maxes found so far with the state of the given net. If a new max is found, it is stored together with the image_idx.'''

        if not self.init_done:
            self._init_with_net(net)

        for layer_name in self.layers:
            # print "processing layer %s" % layer_name

            top_name = layer_name_to_top_name(net, layer_name)
            blob = net.blobs[top_name].data

            self.max_trackers[layer_name].update(blob[batch_index], image_idx, -1,
                                                 net_unique_input_source, layer_name)

        pass

    def calculate_histograms(self, outdir):

        print("calculate_histograms on network")
        for layer_name in self.layers:
            print("calculate_histogram on layer %s" % layer_name)

            self.max_trackers[layer_name].calculate_histogram(layer_name, outdir)

        pass

    def calculate_correlation(self, outdir):

        print("calculate_correlation on network")
        for layer_name in self.layers:
            print("calculate_correlation on layer %s" % layer_name)

            self.max_trackers[layer_name].calculate_correlation(layer_name, outdir)

        pass

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['settings']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)

        self.settings = None


class MaxTracker(object):

    def __init__(self, is_spatial, n_channels, n_top=10, initial_val=-1e99, dtype='float32', search_min=False):
        self.is_spatial = is_spatial
        self.n_top = n_top
        self.search_min = search_min

        self.max_vals = np.ones((n_channels, n_top), dtype=dtype) * initial_val
        if is_spatial:
            self.max_locs = -np.ones((n_channels, n_top, 4), dtype='int')  # image_idx, selected_input_index, i, j
        else:
            self.max_locs = -np.ones((n_channels, n_top, 2), dtype='int')  # image_idx, selected_input_index

        if self.search_min:
            self.min_vals = np.ones((n_channels, n_top), dtype=dtype) * (-initial_val)
            if is_spatial:
                self.min_locs = -np.ones((n_channels, n_top, 4), dtype='int')  # image_idx, selected_input_index, i, j
            else:
                self.min_locs = -np.ones((n_channels, n_top, 2), dtype='int')  # image_idx, selected_input_index

        # set of seen inputs, used to avoid updating on the same input twice
        self.seen_inputs = set()

        # will hold a list of np array, each containing the max values of all the channels for one input
        self.all_max_vals = list()

        # keeps a map between channel index and histogram values
        self.channel_to_histogram = [None] * n_channels

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['seen_inputs']
        del state['all_max_vals']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)

        self.seen_inputs = None
        self.all_max_vals = None

    def __repr__(self):
        return str(self.__dict__.copy())

    def update(self, data, image_idx, selected_input_index, layer_unique_input_source, layer_name):

        # if unique_input_source already exist, we can skip the update since we've already seen it
        if layer_unique_input_source in self.seen_inputs:
            return

        # add input identifier to seen inputs set
        self.seen_inputs.add(layer_unique_input_source)

        n_channels = data.shape[0]
        data_unroll = data.reshape((n_channels, -1))  # Note: no copy eg (96,3025). Does nothing if not is_spatial

        max_indexes = data_unroll.argmax(1)  # maxes for each channel, eg. (96,)

        # add maxes for all channels to a list, bounded to avoid consuming too much memory
        maxes = data_unroll[range(n_channels), max_indexes]
        MAX_LIST_SIZE = 10000
        if len(self.all_max_vals) < MAX_LIST_SIZE:
            self.all_max_vals.append(maxes)

        produced_warning = False

        # insertion_idx = zeros((n_channels,))
        # pdb.set_trace()
        for ii in range(n_channels):

            max_value = data_unroll[ii, max_indexes[ii]]

            # skip nan
            if np.isnan(max_value):
                # only warn once
                if not produced_warning:
                    print('WARNING: got NAN activation on input', str(layer_unique_input_source))
                    produced_warning = True
                continue

            idx = np.searchsorted(self.max_vals[ii], max_value)
            # if not smaller than all elements
            if idx != 0:
                # Store new value in the proper order. Update both arrays:
                # self.max_vals:
                self.max_vals[ii, :idx - 1] = self.max_vals[ii, 1:idx]  # shift lower values
                self.max_vals[ii, idx - 1] = max_value  # store new max value
                # self.max_locs
                self.max_locs[ii, :idx - 1] = self.max_locs[ii, 1:idx]  # shift lower location data
                # store new location
                if self.is_spatial:
                    self.max_locs[ii, idx - 1] = (image_idx, selected_input_index) + np.unravel_index(max_indexes[ii],
                                                                                                      data.shape[1:])
                else:
                    self.max_locs[ii, idx - 1] = (image_idx, selected_input_index)

            if self.search_min:
                idx = np.searchsorted(self.min_vals[ii], max_value)
                # if not bigger than all elements
                if idx != self.n_top:
                    # Store new value in the proper order. Update both arrays:
                    # self.min_vals:
                    self.min_vals[ii, (idx + 1):(self.n_top)] = self.min_vals[ii,
                                                                idx:(self.n_top - 1)]  # shift upper values
                    self.min_vals[ii, idx] = max_value  # store new value
                    # self.min_locs
                    self.min_locs[ii, (idx + 1):(self.n_top)] = self.min_locs[ii,
                                                                idx:(self.n_top - 1)]  # shift upper location data
                    # store new location
                    if self.is_spatial:
                        self.min_locs[ii, idx] = (image_idx, selected_input_index) + np.unravel_index(max_indexes[ii],
                                                                                                      data.shape[1:])
                    else:
                        self.min_locs[ii, idx] = (image_idx, selected_input_index)

    def calculate_histogram(self, layer_name, outdir):

        # convert list of arrays to numpy array
        all_max_array = np.vstack(self.all_max_vals)

        def channel_to_histogram_values(channel_idx):
            # get values
            max_for_single_channel = all_max_array[:, channel_idx]

            # create histogram
            hist, bin_edges = np.histogram(max_for_single_channel, bins=50)

            # save histogram values
            self.channel_to_histogram[channel_idx] = (hist, bin_edges)

            return hist, bin_edges

        def process_channel_figure(channel_idx, fig):
            unit_dir = os.path.join(outdir, layer_name, 'unit_%04d' % channel_idx)
            mkdir_p(unit_dir)
            filename = os.path.join(unit_dir, 'max_histogram.png')
            fig.savefig(filename)
            pass

        def process_layer_figure(fig):
            filename = os.path.join(outdir, layer_name, 'layer_inactivity.png')
            fig.savefig(filename)
            pass

        n_channels = all_max_array.shape[1]
        prepare_max_histogram(layer_name, n_channels, channel_to_histogram_values, process_channel_figure,
                              process_layer_figure)

        pass

    def calculate_correlation(self, layer_name, outdir):

        # convert list of arrays to numpy array
        all_max_array = np.vstack(self.all_max_vals)

        # skip layers with only one channel
        if all_max_array.shape[1] == 1:
            return

        corr = np.corrcoef(all_max_array.transpose())

        # fix possible NANs
        corr = np.nan_to_num(corr)
        np.fill_diagonal(corr, 1)

        # sort correlation matrix
        # import cPickle as pickle
        #  with open('corr_%s.pickled' % layer_name, 'wb') as ff:
        #     pickle.dump(corr, ff, protocol=2)

        # alternative sorting
        # values = np.dot(corr, np.arange(corr.shape[0]))
        # indexes = np.argsort(values)

        indexes = np.lexsort(corr)
        sorted_corr = corr[indexes, :][:, indexes]

        # plot correlation matrix
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 10))
        plt.subplot(1, 1, 1)
        plt.imshow(sorted_corr, interpolation='nearest', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('channels activations correlation matrix for layer %s' % (layer_name))
        plt.tight_layout()

        # save correlation matrix
        layer_dir = os.path.join(outdir, layer_name)
        mkdir_p(layer_dir)
        filename = os.path.join(layer_dir, 'channels_correlation.png')
        fig.savefig(filename, bbox_inches='tight')

        plt.close()

        return


def prepare_max_histogram(layer_name, n_channels, channel_to_histogram_values, process_channel_figure,
                          process_layer_figure):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # for each channel
    percent_dead = np.zeros((n_channels), dtype=np.float32)
    for channel_idx in range(n_channels):

        if channel_idx % 100 == 0:
            print("calculating histogram for channel %d out of %d" % (channel_idx, n_channels))

        hist, bin_edges = channel_to_histogram_values(channel_idx)

        # generate histogram image file
        width = 0.7 * (bin_edges[1] - bin_edges[0])
        center = (bin_edges[:-1] + bin_edges[1:]) / 2

        barlist = ax.bar(center, hist, align='center', width=width, color='g')

        for i in range(len(hist)):
            if 0 >= bin_edges[i] and 0 < bin_edges[i + 1]:
                # mark dead bar in red
                barlist[i].set_color('r')

                # save percent dead
                percent_dead[channel_idx] = 100.0 * hist[i] / sum(hist)

                break

        fig.suptitle('max activations histgoram of layer %s channel %d' % (layer_name, channel_idx))
        ax.xaxis.label.set_text('max activation value')
        ax.yaxis.label.set_text('inputs count')

        process_channel_figure(channel_idx, fig)

        ax.cla()

    # generate histogram for layer
    num_bins = 20
    hist, bin_edges = np.histogram(100 - percent_dead, bins=num_bins, range=(0, 100))
    width = 0.7 * (bin_edges[1] - bin_edges[0])
    center = (bin_edges[:-1] + bin_edges[1:]) / 2

    bar_colors = [None] * num_bins
    begin_color = np.array([1.0, 0, 0])
    end_color = np.array([0, 1.0, 0])
    color_step = (end_color - begin_color) / (num_bins - 1)
    current_color = begin_color
    for i in range(num_bins):
        bar_colors[i] = tuple(current_color)
        current_color += color_step

    ax.bar(center, hist, align='center', width=width, color=bar_colors)

    fig.suptitle('activity of layer %s' % (layer_name))
    ax.xaxis.label.set_text('activity percent')
    ax.yaxis.label.set_text('channels count')

    process_layer_figure(fig)

    fig.clf()
    plt.close(fig)

    pass


def generate_output_names(unit_dir, num_top, do_info, do_maxes, do_deconv, do_deconv_norm, do_backprop,
                          do_backprop_norm, search_min):
    # init values
    info_filename = []
    maxim_filenames = []
    deconv_filenames = []
    deconvnorm_filenames = []
    backprop_filenames = []
    backpropnorm_filenames = []

    prefix = 'min_' if search_min else ''

    if do_info:
        info_filename = [os.path.join(unit_dir, prefix + 'info.txt')]

    for max_idx_0 in range(num_top):
        if do_maxes:
            maxim_filenames.append(os.path.join(unit_dir, prefix + 'maxim_%03d.png' % max_idx_0))

        if do_deconv:
            deconv_filenames.append(os.path.join(unit_dir, prefix + 'deconv_%03d.png' % max_idx_0))

        if do_deconv_norm:
            deconvnorm_filenames.append(os.path.join(unit_dir, prefix + 'deconvnorm_%03d.png' % max_idx_0))

        if do_backprop:
            backprop_filenames.append(os.path.join(unit_dir, prefix + 'backprop_%03d.png' % max_idx_0))

        if do_backprop_norm:
            backpropnorm_filenames.append(os.path.join(unit_dir, prefix + 'backpropnorm_%03d.png' % max_idx_0))

    return (
        info_filename, maxim_filenames, deconv_filenames, deconvnorm_filenames, backprop_filenames,
        backpropnorm_filenames)


class InfoFileMetadata(object):

    def __init__(self, info_file=None, ref_count=None):
        self.info_file = info_file
        self.ref_count = ref_count


def output_max_patches(settings, max_tracker, net, layer_name, idx_begin, idx_end, num_top, datadir, outdir,
                       search_min, do_which):
    '''

    :param settings:
    :param max_tracker:
    :param net:
    :param layer_name:
    :param idx_begin:
    :param idx_end:
    :param num_top:
    :param datadir:
    :param outdir:
    :param search_min:
    :param do_which: do_info must be True
    :return:
    '''
    do_maxes, do_deconv, do_deconv_norm, do_backprop, do_backprop_norm, do_info = do_which
    assert do_maxes or do_deconv or do_deconv_norm or do_backprop or do_backprop_norm or do_info, 'nothing to do'

    sys.path.insert(0, os.path.join(settings.caffevis_caffe_root, 'python'))
    import caffe

    mt = max_tracker

    locs = mt.min_locs if search_min else mt.max_locs
    vals = mt.min_vals if search_min else mt.max_vals

    image_filenames, image_labels = get_files_list(datadir)

    print('Loaded filenames and labels for %d files' % len(image_filenames))
    print('  First file', os.path.join(datadir, image_filenames[0]))

    num_top_in_mt = locs.shape[1]
    assert num_top <= num_top_in_mt, 'Requested %d top images but MaxTracker contains only %d' % (
        num_top, num_top_in_mt)
    assert idx_end >= idx_begin, 'Range error'

    # minor fix for backwards compatability
    if hasattr(mt, 'is_conv'):
        mt.is_spatial = mt.is_conv

    size_ii, size_jj = get_max_data_extent(net, settings, layer_name, mt.is_spatial)
    data_size_ii, data_size_jj = net.blobs['data'].data.shape[2:4]

    net_input_dims = net.blobs['data'].data.shape[2:4]

    # prepare variables used for batches
    batch = [None] * settings.max_tracker_batch_size
    for i in range(0, settings.max_tracker_batch_size):
        batch[i] = MaxTrackerCropBatchRecord()

    batch_index = 0

    channel_to_info_file = dict()

    n_total_images = (idx_end - idx_begin) * num_top
    for cc, channel_idx in enumerate(range(idx_begin, idx_end)):

        unit_dir = os.path.join(outdir, layer_name, 'unit_%04d' % channel_idx)
        mkdir_p(unit_dir)

        # check if all required outputs exist, in which case skip this iteration
        [info_filename,
         maxim_filenames,
         deconv_filenames,
         deconvnorm_filenames,
         backprop_filenames,
         backpropnorm_filenames] = generate_output_names(unit_dir, num_top, do_info, do_maxes, do_deconv,
                                                         do_deconv_norm, do_backprop, do_backprop_norm, search_min)

        relevant_outputs = info_filename + \
                           maxim_filenames + \
                           deconv_filenames + \
                           deconvnorm_filenames + \
                           backprop_filenames + \
                           backpropnorm_filenames

        # we skip generation if:
        # 1. all outputs exist, AND
        # 2.1.   (not last iteration OR
        # 2.2.    last iteration, but batch is empty)
        relevant_outputs_exist = [os.path.exists(file_name) for file_name in relevant_outputs]
        if all(relevant_outputs_exist) and \
                ((channel_idx != idx_end - 1) or ((channel_idx == idx_end - 1) and (batch_index == 0))):
            print("skipped generation of channel %d in layer %s since files already exist" % (channel_idx, layer_name))
            continue

        if do_info:
            channel_to_info_file[channel_idx] = InfoFileMetadata()
            channel_to_info_file[channel_idx].info_file = open(info_filename[0], 'w')
            channel_to_info_file[channel_idx].ref_count = num_top

            # print >> channel_to_info_file[channel_idx].info_file, '# is_spatial val image_idx selected_input_index i(if is_spatial) j(if is_spatial) filename'
            print('# is_spatial val image_idx selected_input_index i(if is_spatial) j(if is_spatial) filename', file=channel_to_info_file[channel_idx].info_file)

        # iterate through maxes from highest (at end) to lowest
        for max_idx_0 in range(num_top):
            batch[batch_index].cc = cc
            batch[batch_index].channel_idx = channel_idx
            batch[batch_index].info_filename = info_filename
            batch[batch_index].maxim_filenames = maxim_filenames
            batch[batch_index].deconv_filenames = deconv_filenames
            batch[batch_index].deconvnorm_filenames = deconvnorm_filenames
            batch[batch_index].backprop_filenames = backprop_filenames
            batch[batch_index].backpropnorm_filenames = backpropnorm_filenames
            batch[batch_index].info_file = channel_to_info_file[channel_idx].info_file

            batch[batch_index].max_idx_0 = max_idx_0
            batch[batch_index].max_idx = num_top_in_mt - 1 - batch[batch_index].max_idx_0

            if mt.is_spatial:

                # fix for backward compatability
                if locs.shape[2] == 5:
                    # remove second column
                    locs = np.delete(locs, 1, 2)

                batch[batch_index].im_idx, batch[batch_index].selected_input_index, batch[batch_index].ii, batch[
                    batch_index].jj = locs[batch[batch_index].channel_idx, batch[batch_index].max_idx]
            else:
                # fix for backward compatability
                if locs.shape[2] == 3:
                    # remove second column
                    locs = np.delete(locs, 1, 2)

                batch[batch_index].im_idx, batch[batch_index].selected_input_index = locs[
                    batch[batch_index].channel_idx, batch[batch_index].max_idx]
                batch[batch_index].ii, batch[batch_index].jj = 0, 0

            # if ii and jj are invalid then there is no data for this "top" image, so we can skip it
            if (batch[batch_index].ii, batch[batch_index].jj) == (-1, -1):
                continue

            batch[batch_index].recorded_val = vals[batch[batch_index].channel_idx, batch[batch_index].max_idx]
            batch[batch_index].filename = image_filenames[batch[batch_index].im_idx]
            do_print = (batch[batch_index].max_idx_0 == 0)
            if do_print:
                print('%s   Output file/image(s) %d/%d   layer %s channel %d' % (
                    datetime.now().ctime(), batch[batch_index].cc * num_top, n_total_images, layer_name,
                    batch[batch_index].channel_idx))

            # print "DEBUG: (mt.is_spatial, batch[batch_index].ii, batch[batch_index].jj, layer_name, size_ii, size_jj, data_size_ii, data_size_jj)", str((mt.is_spatial, batch[batch_index].ii, batch[batch_index].jj, rc, layer_name, size_ii, size_jj, data_size_ii, data_size_jj))

            [batch[batch_index].out_ii_start,
             batch[batch_index].out_ii_end,
             batch[batch_index].out_jj_start,
             batch[batch_index].out_jj_end,
             batch[batch_index].data_ii_start,
             batch[batch_index].data_ii_end,
             batch[batch_index].data_jj_start,
             batch[batch_index].data_jj_end] = \
                compute_data_layer_focus_area(mt.is_spatial, batch[batch_index].ii, batch[batch_index].jj, settings,
                                              layer_name,
                                              size_ii, size_jj, data_size_ii, data_size_jj)

            # print "DEBUG: channel:%d out_ii_start:%d out_ii_end:%d out_jj_start:%d out_jj_end:%d data_ii_start:%d data_ii_end:%d data_jj_start:%d data_jj_end:%d" % \
            #       (channel_idx,
            #        batch[batch_index].out_ii_start, batch[batch_index].out_ii_end,
            #        batch[batch_index].out_jj_start, batch[batch_index].out_jj_end,
            #        batch[batch_index].data_ii_start, batch[batch_index].data_ii_end,
            #        batch[batch_index].data_jj_start, batch[batch_index].data_jj_end)

            if do_info:
                print(1 if mt.is_spatial else 0, '%.6f' % vals[batch[batch_index].channel_idx, batch[batch_index].max_idx], file=batch[batch_index].info_file)
                if mt.is_spatial:
                    print('%d %d %d %d' % tuple(
                        locs[batch[batch_index].channel_idx, batch[batch_index].max_idx]),file=batch[batch_index].info_file)
                else:
                    print('%d %d' % tuple(
                        locs[batch[batch_index].channel_idx, batch[batch_index].max_idx]),file=batch[batch_index].info_file)
                print(batch[batch_index].filename, file=batch[batch_index].info_file)

            if not (do_maxes or do_deconv or do_deconv_norm or do_backprop or do_backprop_norm):
                continue

            with WithTimer('Load image', quiet=not do_print):
                # load image
                batch[batch_index].im = caffe.io.load_image(os.path.join(datadir, batch[batch_index].filename),
                                                            color=True)

                # resize images according to input dimension
                batch[batch_index].im = resize_without_fit(batch[batch_index].im, net_input_dims)

                # convert to float to avoid caffe destroying the image in the scaling phase
                batch[batch_index].im = batch[batch_index].im.astype(np.float32)

            batch_index += 1

            # if current batch is full
            if batch_index == settings.max_tracker_batch_size \
                    or ((channel_idx == idx_end - 1) and (max_idx_0 == num_top - 1)):  # or last iteration

                with WithTimer('Predict on batch  ', quiet=not do_print):
                    im_batch = [record.im for record in batch]
                    net.predict(im_batch, oversample=False)

                # go over batch and update statistics
                for i in range(0, batch_index):

                    batch[i].denormalized_layer_name = layer_name
                    batch[i].denormalized_top_name = layer_name_to_top_name(net, batch[i].denormalized_layer_name)
                    batch[i].layer_format = 'normal'  # non-siamese

                    if len(net.blobs[batch[i].denormalized_top_name].data.shape) == 4:
                        reproduced_val = net.blobs[batch[i].denormalized_top_name].data[
                                i, batch[i].channel_idx, batch[i].ii, batch[i].jj]

                    else:
                        reproduced_val = net.blobs[batch[i].denormalized_top_name].data[i, batch[i].channel_idx]

                    if abs(reproduced_val - batch[i].recorded_val) > .1:
                        print('Warning: recorded value %s is suspiciously different from reproduced value %s. Is the filelist the same?' % (
                            batch[i].recorded_val, reproduced_val))

                    if do_maxes:
                        # grab image from data layer, not from im (to ensure preprocessing / center crop details match between image and deconv/backprop)

                        out_arr = extract_patch_from_image(net.blobs['data'].data[i], net,
                                                           batch[i].selected_input_index, settings,
                                                           batch[i].data_ii_end, batch[i].data_ii_start,
                                                           batch[i].data_jj_end, batch[i].data_jj_start,
                                                           batch[i].out_ii_end, batch[i].out_ii_start,
                                                           batch[i].out_jj_end, batch[i].out_jj_start, size_ii, size_jj)

                        with WithTimer('Save img  ', quiet=not do_print):
                            save_caffe_image(out_arr, batch[i].maxim_filenames[batch[i].max_idx_0],
                                             autoscale=False, autoscale_center=0,
                                             channel_swap=settings.channel_swap)

                if do_deconv or do_deconv_norm:

                    # TODO: we can improve performance by doing batch of deconv_from_layer, but only if we group
                    # together instances which have the same selected_input_index, this can be done by holding two
                    # separate batches

                    for i in range(0, batch_index):
                        diffs = net.blobs[batch[i].denormalized_top_name].diff * 0

                        if len(diffs.shape) == 4:
                                diffs[i, batch[i].channel_idx, batch[i].ii, batch[i].jj] = 1.0
                        else:
                            diffs[i, batch[i].channel_idx] = 1.0

                        with WithTimer('Deconv    ', quiet=not do_print):
                            net.deconv_from_layer(batch[i].denormalized_layer_name, diffs, zero_higher=True,
                                                  deconv_type='Guided Backprop')

                        out_arr = extract_patch_from_image(net.blobs['data'].diff[i], net,
                                                           batch[i].selected_input_index, settings,
                                                           batch[i].data_ii_end, batch[i].data_ii_start,
                                                           batch[i].data_jj_end, batch[i].data_jj_start,
                                                           batch[i].out_ii_end, batch[i].out_ii_start,
                                                           batch[i].out_jj_end, batch[i].out_jj_start, size_ii, size_jj)

                        if out_arr.max() == 0:
                            print('Warning: Deconv out_arr in range', out_arr.min(), 'to', out_arr.max(), 'ensure force_backward: true in prototxt')

                        if do_deconv:
                            with WithTimer('Save img  ', quiet=not do_print):
                                save_caffe_image(out_arr, batch[i].deconv_filenames[batch[i].max_idx_0],
                                                 autoscale=False, autoscale_center=0,
                                                 channel_swap=settings.channel_swap)
                        if do_deconv_norm:
                            out_arr = np.linalg.norm(out_arr, axis=0)
                            with WithTimer('Save img  ', quiet=not do_print):
                                save_caffe_image(out_arr, batch[i].deconvnorm_filenames[batch[i].max_idx_0],
                                                 channel_swap=settings.channel_swap)

                if do_backprop or do_backprop_norm:

                    for i in range(0, batch_index):
                        diffs = net.blobs[batch[i].denormalized_top_name].diff * 0

                        if len(diffs.shape) == 4:
                            diffs[i, batch[i].channel_idx, batch[i].ii, batch[i].jj] = 1.0
                        else:
                            diffs[i, batch[i].channel_idx] = 1.0

                    with WithTimer('Backward batch  ', quiet=not do_print):
                        net.backward_from_layer(batch[i].denormalized_layer_name, diffs)

                    for i in range(0, batch_index):

                        out_arr = extract_patch_from_image(net.blobs['data'].diff[i], net,
                                                           batch[i].selected_input_index, settings,
                                                           batch[i].data_ii_end, batch[i].data_ii_start,
                                                           batch[i].data_jj_end, batch[i].data_jj_start,
                                                           batch[i].out_ii_end, batch[i].out_ii_start,
                                                           batch[i].out_jj_end, batch[i].out_jj_start, size_ii, size_jj)

                        if out_arr.max() == 0:
                            print('Warning: Deconv out_arr in range', out_arr.min(), 'to', out_arr.max(), 'ensure force_backward: true in prototxt')
                        if do_backprop:
                            with WithTimer('Save img  ', quiet=not do_print):
                                save_caffe_image(out_arr, batch[i].backprop_filenames[batch[i].max_idx_0],
                                                 autoscale=False, autoscale_center=0,
                                                 channel_swap=settings.channel_swap)
                        if do_backprop_norm:
                            out_arr = np.linalg.norm(out_arr, axis=0)
                            with WithTimer('Save img  ', quiet=not do_print):
                                save_caffe_image(out_arr, batch[i].backpropnorm_filenames[batch[i].max_idx_0],
                                                 channel_swap=settings.channel_swap)

                # close info files
                for i in range(0, batch_index):
                    channel_to_info_file[batch[i].channel_idx].ref_count -= 1
                    if channel_to_info_file[batch[i].channel_idx].ref_count == 0:
                        if do_info:
                            channel_to_info_file[batch[i].channel_idx].info_file.close()

                batch_index = 0
