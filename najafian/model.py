from __future__ import print_function

import json
import os
from os import listdir

import cv2
from keras.layers import *
from keras.models import *

import configs

settings = configs.load_project_settings()
try:
    # dynamically loaded settings
    crop = settings['crop_image']
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
    if isinstance(image_list[0], str):
        log('images ' + str(configs.strip_file_names(image_list)))
        image_list = [configs.apply_normalization(configs.load_image(image)) for image in image_list]
    else:
        image_list = [configs.apply_normalization(image) for image in image_list]

    # keep a copy of the original list
    original_list = []

    # populate the batch
    batch = np.zeros((batch_len,) + img_shape, np.float32)

    for ind, image in enumerate(image_list):
        # resize the image
        if image.shape[:2] != img_shape[:2]:
            image = cv2.resize(image, img_size, interpolation=cv2.INTER_LANCZOS4).reshape(img_shape)

        # make a copy of the original image
        original_list.append(configs.rescale_out_image(image, high_quality=True))

        # increase the contrast
        image = image.astype(np.float32)

        # normalize the input
        image = (image - props['x_mean']) / props['x_std']
        batch[ind] = image

    # run the prediction on the batch
    predicted = model.predict(batch)

    log('done')

    # for the layer model we have to process the two classes
    # this project has the segmentation edge layer and the slit layer
    if layer_model:
        # reshape the output batch
        num_classes = 2
        layers = np.zeros((batch_len, num_classes,) + img_aspect_shape, np.uint8)

        for c in range(num_classes):
            for ind in range(batch_len):
                layers[ind][c] = configs.rescale_out_image((predicted[c][ind] * 255).astype(np.uint8))
        return original_list, layers
    else:
        fixed_pred = np.zeros((batch_len,) + img_aspect_shape, np.uint8)
        for ind in range(batch_len):
            fixed_pred[ind] = configs.rescale_out_image((predicted[ind] * 255).astype(np.uint8))
        return original_list, fixed_pred


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

