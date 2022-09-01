""" NOTE: this file is for testing purposes only and should not be used for forknet. Please see model.py for FORKNET """
from __future__ import print_function

import json
import os
import numpy as np
from os import listdir

import cv2
# from keras.layers import *
# from keras.models import *
from tiler import Tiler, Merger
from .deeplab import build_deep_lab_v3_model

import configs

EDGE_DIMS = (350, 350)
EDGE_OVERLAP = 0.25  # include 25% pixels of overlap for each window
SLIT_DIMS = (150, 150)
SLIT_OVERLAP = 0.3  # include 30% pixels of overlap
EDGE_BATCH_SIZE = 12


settings = configs.load_project_settings()
try:
    # dynamically loaded settings
    crop = settings['crop_image']
    model_dir = settings['model_dir']
    img_size_w = crop[0]  # 1024  # the width of the images
    img_size_h = crop[1]  # 1104  # the height of the images

    # CNN (resized) size
    merge_depth = True  # currently only turns rgb images into greyscale images
    img_size = tuple(settings['network_in'])  # (832, 832)
    img_depth = 3
    img_shape = (img_size[1], img_size[0], 1)
    img_depth_shape = img_shape[:2] + (img_depth,)
    img_aspect_size = (img_size[0], int(img_size[1] * float(img_size_h / img_size_w)))
    img_aspect_shape = (img_aspect_size[1], img_aspect_size[0], 1)
except (IndexError, KeyError) as err:
    print(f'The configs file is missing a key {str(err)}')
    exit(1)

# globally changed params
model = None  # the currently loaded model
layer_model = True  # does the model have multiple layers
batch_size = 1  # by default only batch 1 item on the GPU
props = {}  # the loaded dataprops (eventually loaded by dataprops.json) for image normalization
log = lambda to_log: print(to_log)   # by default logging is just printing


def run_prediction(image_list):
    """ Produces the segmentation maps using the defined model for the specified image list

    Args:
        image_list (list of paths or np.ndarray): a list of images to process (either strings or numpy arrays) to produce the segmentation maps

    Returns:
        tuple: a tuple of lists of np.ndarray (background image array (np.ndarray list), segmentation maps (np.ndarray list with 2 layers))
    """
    global model

    batch_len = len(image_list)
    if batch_len == 0:
        return [], []
    log('running prediction on %d images' % batch_len)

    # load the image and apply the normalization
    # UNCOMMENT FOR FORKNET
    # if isinstance(image_list[0], str):
    #     log('images ' + str(configs.strip_file_names(image_list)))
    #     image_list = [configs.apply_normalization(configs.load_image(image)) for image in image_list]
    # else:
    #     image_list = [configs.apply_normalization(image) for image in image_list]

    if isinstance(image_list[0], str):
        log('images ' + str(configs.strip_file_names(image_list)))
        image_list = [configs.load_image(image) for image in image_list]
    # else:
    #     image_list = [configs.apply_normalization(image) for image in image_list]

    # keep a copy of the original list
    original_list = []

    # populate the batch
    # batch = np.zeros((batch_len,) + img_shape, np.float32)
    # layers = np.zeros((batch_len, num_classes,) + img_aspect_shape, np.uint8)
    batch = []
    batch_input = []
    layers = []

    # first run on first mask
    for ind, image in enumerate(image_list):
        # resize the image
        # if image.shape[:2] != img_shape[:2]:
        #     image = cv2.resize(image, img_size, interpolation=cv2.INTER_LANCZOS4).reshape(img_shape)

        # make a copy of the original image
        # original_list.append(configs.rescale_out_image(image, high_quality=True))
        original_list.append(image)

        # increase the contrast
        image = image.astype(np.float32)

        # normalize the input
        image = (image - props['x_mean']) / props['x_std']
        # batch[ind] = image

        # run tiling
        first_mask = run_tile_prediction('first', image, EDGE_DIMS + (1,), EDGE_OVERLAP, n_out=1)
        
        # combine results
        bimage = np.zeros(first_mask.shape[:2] + (2,), np.float32)
        bimage[:, :, 0] = image.reshape(image.shape[:2])
        bimage[:, :, 1] = first_mask.reshape(first_mask.shape[:2]).astype(np.float32) / 255.0
        batch.append(bimage)
    
    # now run through second tiler
    for ind, b in enumerate(batch):
        # add original image and membrane
        second_masks = run_tile_prediction('second', b, SLIT_DIMS + (2,), SLIT_OVERLAP, n_out=2)

        # create output
        out = np.zeros((2,) + second_masks[0].shape, np.uint8)
        out[0] = second_masks[0]
        out[1] = second_masks[1]

        # print(out.shape)

        # add to the layers output for the batch
        layers.append(out)

    # run the prediction on the batch
    # predicted = model.predict(batch)

    log('done')
    return original_list, layers

    # for the layer model we have to process the two classes
    # this project has the segmentation edge layer and the slit layer
    # if layer_model:
    #     # reshape the output batch
    #     num_classes = 2
    #     layers = np.zeros((batch_len, num_classes,) + img_aspect_shape, np.uint8)

    #     for c in range(num_classes):
    #         for ind in range(batch_len):
    #             layers[ind][c] = configs.rescale_out_image((predicted[c][ind] * 255).astype(np.uint8))
    #     return original_list, layers
    # else:
    #     fixed_pred = np.zeros((batch_len,) + img_aspect_shape, np.uint8)
    #     for ind in range(batch_len):
    #         fixed_pred[ind] = configs.rescale_out_image((predicted[ind] * 255).astype(np.uint8))
    #     return original_list, fixed_pred




def make_tiler(image: np.ndarray, tile_shape: np.ndarray, overlap: np.ndarray, n_out=1):
    """ Tiles up an image to subdivide it into chunks """
    tiler = Tiler(
        data_shape=image.shape,
        tile_shape=tile_shape,
        overlap=overlap,
        mode='constant',
        constant_value=0,
        channel_dimension=2
    )
    tiler2 = Tiler(   # used for merging single masks
        data_shape=image.shape[:2],
        tile_shape=tile_shape[:2],
        overlap=overlap,
        mode='constant',
        constant_value=0
    )
    merger = []

    for i in range(n_out):
        merger.append(Merger(tiler=tiler2, window='triang',))
    return tiler, merger


def fix_mask(mask: np.ndarray):
    """ Fixes output mask to be properly scaled """
    return np.clip((mask * 255.0), a_min=0, a_max=255).astype(np.uint8)


def run_tile_prediction(model_type: str, image: np.ndarray, tile_shape: np.ndarray, overlap: np.ndarray, n_out: int=1):
    """ Runs the image through the membrane model through the tiles """
    global model

    # first step is to take original image and split it up into chunks for processing
    tiler, merger = make_tiler(image, tile_shape, overlap, n_out=n_out)

    # load model
    smodel = model[model_type]

    # let's pass
    # print('running on', image.shape)
    for tid, tile in tiler.iterate(image, batch_size=EDGE_BATCH_SIZE):
        # predict on the specified image
        if len(tile.shape) != 4:
            tile = tile.reshape(tile.shape + (1,))
        batch = smodel.predict(tile)
        fixed = fix_mask(batch)

        # print(merger)
        if len(merger) == 1:
            merger[0].add_batch(tid, EDGE_BATCH_SIZE, fixed.reshape(fixed.shape[:3]))
        else:
            # print(fixed.shape)
            for i, m in enumerate(merger):
                m.add_batch(tid, EDGE_BATCH_SIZE, fixed[:, :, :, i])

    # merge all of the tiled windows
    if len(merger) == 1:
        image_pred = merger[0].merge(dtype=np.uint8)
    else:
        image_pred = [m.merge(dtype=np.uint8) for m in merger]

    return image_pred


def load_model(model_path, weights_path, props_path, is_layer_model=True, force_reload=True, bs=4):
    """ Loads the keras (backend tensorflow) model from the disk

    Args:
        model_path (str): path to the model.json (must be supported by tf 1.15)
        weights_path (str): path to model weights.h5
        props_path (str): path to dataprops.json that contains the mean and std of the data
        is_layer_model (bool, optional): is this model 2 layers or a single layer. Defaults to True.
        force_reload (bool, optional): force reloading weights/model even if one has been currently loaded. Defaults to True.
        bs (int, optional): batch size to use when running predicitions. Defaults to 4.
    
    Note: this loads the model into this module and does not return a model object
    """
    global model, layer_model, props, batch_size, auto_batch

    # update global batch_size
    batch_size = int(bs)

    if model is not None and not force_reload:
        log('model already loaded!')
        return

    # use deeplab window approach
    log('creating models')
    model = {
        'first': build_deep_lab_v3_model(EDGE_DIMS + (1,), classes=1),
        'second': build_deep_lab_v3_model(SLIT_DIMS + (2,), classes=2)
    }
    log('done')

    log('loading model props')
    with open(props_path, 'r') as props_f:
        props = json.loads(props_f.read())
    log('done')

    # load the weights
    log('loading model weights')
    model['first'].load_weights(os.path.join(model_dir, '350x350-membrane-seg.h5'))
    model['second'].load_weights(os.path.join(model_dir, '150-x150-memb-slit-seg.h5'))
    log('done')

    """ Uncomment for ForkNET version
    from keras.initializers import glorot_uniform
    from keras.utils import CustomObjectScope

    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        with open(model_path, 'r') as json_f:
            model_json = json_f.read()

        log('loading model')
        model = model_from_json(model_json)
        layer_model = is_layer_model
        log('done')

        log('loading model props')
        with open(props_path, 'r') as props_f:
            props = json.loads(props_f.read())
        log('done')

        log('loading weights from %s' % str(weights_path))
        model.load_weights(weights_path)
        log('done')
    """

def set_log(logger):
    """ Updates the logging function from parent module

    Args:
        logger (logging.Logger): logger module
    """
    global log
    log = logger


def run_prediction_on_folder(folder, to_log=True):
    """ When specified a folder will automatically predict on all of the images in the folder

    Args:
        folder (str, list): path of folder to run predictions on, or a list of files (that are "in" that folder)
        to_log (bool, optional): log progress. Defaults to True.

    Yields (on every batch processed):
        tuple: (index of batch (int), total batch count (int), number of files in batch (int)) 
    """
    if isinstance(folder, list):
        files = folder
    else:
        files = [os.path.join(folder, f) for f in listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    ind = 0
    batches = len(files) // batch_size
    for batch in range(0, len(files), batch_size):
        if to_log:
            log('running batch %d/%d' % (ind, batches))
        batch_files = files[batch:batch + batch_size]
        yield (ind, batches, batch_files) + run_prediction(batch_files)
        ind += 1

