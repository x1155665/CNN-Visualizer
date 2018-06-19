import cv2
import os
import sys
import re
import numpy as np
import errno
import skimage


def load_network(settings):
    sys.path.insert(0, os.path.join(settings.caffevis_caffe_root, 'python'))
    import caffe

    if settings.use_GPU:
        caffe.set_mode_gpu()
        caffe.set_device(settings.gpu_id)
        print 'Loaded caffe in GPU mode, using device', settings.gpu_id
    else:
        caffe.set_mode_cpu()
        print 'Loaded caffe in CPU mode'

    processed_prototxt = process_network_proto(settings.prototxt, settings.caffevis_caffe_root)

    net = caffe.Classifier(processed_prototxt, settings.network_weights, mean=settings.mean, raw_scale=255.0,
                           channel_swap=settings.channel_swap)

    return net


def process_network_proto(prototxt, caffevis_caffe_root):
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


def mkdir_p(path):
    # From https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_files_list(data_dir):
    print 'Getting image list...'
    # available_files - local list of files
    available_files = get_files_from_directory(data_dir)
    labels = None
    print 'Getting image list... Done.'
    return available_files, labels


def get_files_from_directory(data_dir):
    # returns list of files in requested directory

    available_files = []
    match_flags = re.IGNORECASE
    for filename in os.listdir(data_dir):
        if re.match('.*\.(jpg|jpeg|png)$', filename, match_flags):
            available_files.append(filename)
    return available_files


def layer_name_to_top_name(net, layer_name):
    if net.top_names.has_key(layer_name) and len(net.top_names[layer_name]) >= 1:
        return net.top_names[layer_name][0]

    else:
        return None


def resize_without_fit(img, out_max_shape,
                dtype_out = None,
                shrink_interpolation = cv2.INTER_LINEAR,
                grow_interpolation = cv2.INTER_NEAREST):
    '''Resizes (without fit) to out_max_shape.

    If one of the out_max_shape dimensions is None, then use only the other dimension to perform resizing.

    '''

    if dtype_out is not None and img.dtype != dtype_out:
        dtype_in_size = img.dtype.itemsize
        dtype_out_size = np.dtype(dtype_out).itemsize
        convert_early = (dtype_out_size < dtype_in_size)
        convert_late = not convert_early
    else:
        convert_early = False
        convert_late = False
    if out_max_shape[0] is None:
        scale_0 = float(out_max_shape[1]) / img.shape[1]
        scale_1 = float(out_max_shape[1]) / img.shape[1]
    elif out_max_shape[1] is None:
        scale_1 = float(out_max_shape[0]) / img.shape[0]
        scale_0 = float(out_max_shape[0]) / img.shape[0]
    else:

        scale_0 = float(out_max_shape[0]) / img.shape[0]
        scale_1 = float(out_max_shape[1]) / img.shape[1]

    if convert_early:
        img = np.array(img, dtype=dtype_out)

    if len(img.shape) == 3:
        out = np.stack([cv2.resize(img[:,:,i],  # 0,0), fx=scale_1, fy=scale_0,
                                   (int(round(img.shape[1] * scale_1)), int(round(img.shape[0] * scale_0))),  # in (c,r) order
                                   interpolation=grow_interpolation if min(scale_0, scale_1) > 1 else shrink_interpolation)
                        for i in range(img.shape[2])], axis=2)
    else:
        out = cv2.resize(img,  # 0,0), fx=scale_1, fy=scale_0,
                         (int(round(img.shape[1] * scale_1)), int(round(img.shape[0] * scale_0))),  # in (c,r) order
                         interpolation=grow_interpolation if min(scale_0, scale_1) > 1 else shrink_interpolation)

    if convert_late:
        out = np.array(out, dtype=dtype_out)

    # fix resize of grayscale images
    if len(img.shape) == 3 and img.shape[2] == 1 and len(out.shape) == 2:
        out = out[:, :, np.newaxis]

    return out


def get_max_data_extent(net, settings, layer_name, is_spatial):
    '''Gets the maximum size of the data layer that can influence a unit on layer.'''

    data_size = net.blobs['data'].data.shape[2:4]  # e.g. (227,227) for fc6,fc7,fc8,prop

    if is_spatial:
        top_name = layer_name_to_top_name(net, layer_name)
        conv_size = net.blobs[top_name].data.shape[2:4]        # e.g. (13,13) for conv5
        layer_slice_middle = (conv_size[0]/2,conv_size[0]/2+1, conv_size[1]/2,conv_size[1]/2+1)   # e.g. (6,7,6,7,), the single center unit
        data_slice = RegionComputer.convert_region_dag(settings, layer_name, 'input', layer_slice_middle)
        data_slice_size = data_slice[1]-data_slice[0], data_slice[3]-data_slice[2]   # e.g. (163, 163) for conv5
        # crop data slice size to data size
        data_slice_size = min(data_slice_size[0], data_size[0]), min(data_slice_size[1], data_size[1])
        return data_slice_size
    else:
        # Whole data region
        return data_size


class RegionComputer(object):
    '''Computes regions of possible influcence from higher layers to lower layers.'''

    @staticmethod
    def region_converter(top_slice, filter_width=(1, 1), stride=(1, 1), pad=(0, 0)):
        '''
        Works for conv or pool

        vector<int> ConvolutionLayer<Dtype>::JBY_region_of_influence(const vector<int>& slice) {
        +  CHECK_EQ(slice.size(), 4) << "slice must have length 4 (ii_start, ii_end, jj_start, jj_end)";
        +  // Crop region to output size
        +  vector<int> sl = vector<int>(slice);
        +  sl[0] = max(0, min(height_out_, slice[0]));
        +  sl[1] = max(0, min(height_out_, slice[1]));
        +  sl[2] = max(0, min(width_out_, slice[2]));
        +  sl[3] = max(0, min(width_out_, slice[3]));
        +  vector<int> roi;
        +  roi.resize(4);
        +  roi[0] = sl[0] * stride_h_ - pad_h_;
        +  roi[1] = (sl[1]-1) * stride_h_ + kernel_h_ - pad_h_;
        +  roi[2] = sl[2] * stride_w_ - pad_w_;
        +  roi[3] = (sl[3]-1) * stride_w_ + kernel_w_ - pad_w_;
        +  return roi;
        +}
        '''
        assert len(top_slice) == 4
        assert len(filter_width) == 2
        assert len(stride) == 2
        assert len(pad) == 2

        # Crop top slice to allowable region
        top_slice = [ss for ss in top_slice]  # Copy list or array -> list

        bot_slice = [-123] * 4

        bot_slice[0] = top_slice[0] * stride[0] - pad[0]
        bot_slice[1] = top_slice[1] * stride[0] - pad[0] + filter_width[0] - 1
        bot_slice[2] = top_slice[2] * stride[1] - pad[1]
        bot_slice[3] = top_slice[3] * stride[1] - pad[1] + filter_width[1] - 1

        return bot_slice


    @staticmethod
    def merge_regions(region1, region2):

        region1_x_start, region1_x_end, region1_y_start, region1_y_end = region1
        region2_x_start, region2_x_end, region2_y_start, region2_y_end = region2

        merged_x_start = min(region1_x_start, region2_x_start)
        merged_x_end = max(region1_x_end, region2_x_end)
        merged_y_start = min(region1_y_start, region2_y_start)
        merged_y_end = max(region1_y_end, region2_y_end)

        merged_region = (merged_x_start, merged_x_end, merged_y_start, merged_y_end)

        return merged_region


    @staticmethod
    def convert_region_dag(settings, from_layer, to_layer, region):

        step_region = None

        layer_def = settings._layer_name_to_record[from_layer] if from_layer in settings._layer_name_to_record else None

        # do single step to convert according to from_layer
        if not layer_def:
            # fallback to doing nothing
            step_region = region

        else:

            if layer_def.type in ['Convolution', 'Pooling']:
                step_region = RegionComputer.region_converter(region, layer_def.filter, layer_def.stride, layer_def.pad)

            else:
                # fallback to doing nothing
                step_region = region

        if from_layer == to_layer:
            return step_region

        # handle the rest
        total_region = None

        if layer_def is not None:
            for parent_layer in layer_def.parents:

                # skip inplace layers
                if len(parent_layer.tops) == 1 and len(parent_layer.bottoms) == 1 and parent_layer.tops[0] == parent_layer.bottoms[0]:
                    continue

                # calculate convert_region_dag on each one
                current_region = RegionComputer.convert_region_dag(settings, parent_layer.name, to_layer, step_region)

                # aggregate results
                if total_region is None:
                    total_region = current_region
                else:
                    total_region = RegionComputer.merge_regions(total_region, current_region)

        if total_region is None:
            return step_region

        return total_region


def compute_data_layer_focus_area(is_spatial, ii, jj, settings, layer_name, size_ii, size_jj, data_size_ii, data_size_jj):

    if is_spatial:

        # Compute the focus area of the data layer
        layer_indices = (ii, ii + 1, jj, jj + 1)

        data_indices = RegionComputer.convert_region_dag(settings, layer_name, 'input', layer_indices)
        data_ii_start, data_ii_end, data_jj_start, data_jj_end = data_indices

        # safe guard edges
        data_ii_start = max(data_ii_start, 0)
        data_jj_start = max(data_jj_start, 0)
        data_ii_end = min(data_ii_end, data_size_ii)
        data_jj_end = min(data_jj_end, data_size_jj)

        touching_imin = (data_ii_start == 0)
        touching_jmin = (data_jj_start == 0)

        # Compute how much of the data slice falls outside the actual data [0,max] range
        ii_outside = size_ii - (data_ii_end - data_ii_start)  # possibly 0
        jj_outside = size_jj - (data_jj_end - data_jj_start)  # possibly 0

        if touching_imin:
            out_ii_start = ii_outside
            out_ii_end = size_ii
        else:
            out_ii_start = 0
            out_ii_end = size_ii - ii_outside
        if touching_jmin:
            out_jj_start = jj_outside
            out_jj_end = size_jj
        else:
            out_jj_start = 0
            out_jj_end = size_jj - jj_outside

    else:
        data_ii_start, out_ii_start, data_jj_start, out_jj_start = 0, 0, 0, 0
        data_ii_end, out_ii_end, data_jj_end, out_jj_end = size_ii, size_ii, size_jj, size_jj

    return [out_ii_start, out_ii_end, out_jj_start, out_jj_end, data_ii_start, data_ii_end, data_jj_start, data_jj_end]


def extract_patch_from_image(data, net, selected_input_index, settings,
                             data_ii_end, data_ii_start, data_jj_end, data_jj_start,
                             out_ii_end, out_ii_start, out_jj_end, out_jj_start, size_ii, size_jj):
    out_arr = np.zeros((3, size_ii, size_jj), dtype='float32')
    out_arr[:, out_ii_start:out_ii_end, out_jj_start:out_jj_end] = data[:,
                                                                   data_ii_start:data_ii_end,
                                                                   data_jj_start:data_jj_end]
    return out_arr


def save_caffe_image(img, filename, autoscale = True, autoscale_center = None, channel_swap = (2,1,0)):
    '''Takes an image in caffe format (01) or (c01, 3_channels) and saves it to a file'''
    if len(img.shape) == 2:
        # upsample grayscale 01 -> 01c
        img = np.tile(img[:,:,np.newaxis], (1,1,3))
    else:
        img = img[channel_swap,:,:].transpose((1, 2, 0))
    if autoscale_center is not None:
        img = norm01c(img, autoscale_center)
    elif autoscale:
        img = img.copy()
        img -= img.min()
        img *= 1.0 / (img.max() + 1e-10)
    skimage.io.imsave(filename, img)


def norm01c(arr, center):
    '''Maps the input range to [0,1] such that the center value maps to .5'''
    arr = arr.copy()
    arr -= center
    arr /= max(2 * arr.max(), -2 * arr.min()) + 1e-10
    arr += .5
    assert arr.min() >= 0
    assert arr.max() <= 1
    return arr
