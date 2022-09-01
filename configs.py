import json
import os
from collections import OrderedDict
from itertools import filterfalse

import cv2
import numpy as np
from skimage.external.tifffile.tifffile import TiffWriter
import tinycss2.color3 as color
from skimage.external.tifffile import TiffFile

# global config
# remove files with a '.' dot in front of them (hidden files)
IGNORE_DOT_FILES = True
FILE_EXTENSION = '.tiff'  # the extension for predicted files

# load the configs.json from the current dir
file_dir = os.path.dirname(os.path.realpath(__file__))
project_file = os.path.join(file_dir, 'configs.json')
model_dir = os.path.join(file_dir, 'model')
project_settings = None  # this is a static so it's not loaded every time


def load_project_settings():
    """ Gets the model settings and other parameters from configs.json

    :return: A dictionary of the settings in configs.json (check that file for structure)
    """
    global project_settings

    if project_settings is None:
        with open(project_file, 'r') as configs_f:
            data = str(configs_f.read())

            if '{' not in data or '}' not in data:
                raise Exception('the configs ' + project_file +
                                ' is not a valid json object')

            # fix the json object location just in case of some bad formatting
            data = data[data.index('{'):data.rfind('}') + 1]
            project_settings = json.loads(data, object_pairs_hook=OrderedDict)

    project_settings.update({
        'model_dir': model_dir
    })

    return project_settings


try:
    settings = load_project_settings()
    crop = settings['crop_image']
    img_size_w = crop[0]  # 1024  # the width of the images
    img_size_h = crop[1]  # 1104  # the height of the images

    # CNN (resized) size
    merge_depth = True  # currently only turns rgb images into greyscale images
    img_size = tuple(settings['network_in'])  # (832, 832)
    img_depth = 3
    img_shape = (img_size[1], img_size[0], 1)
    img_depth_shape = img_shape[:2] + (img_depth,)
    img_aspect_size = (img_size[0], int(
        img_size[1] * float(img_size_h / img_size_w)))
    img_aspect_shape = (img_aspect_size[1], img_aspect_size[0], 1)
    # the minimum probability consider the classification as valid
    min_class_threshold = settings['min_threshold']
    min_class_depth_threshold = [min_class_threshold] * img_depth
except (IndexError, KeyError) as err:
    print(f'The configs file is missing a key {str(err)}')
    exit(1)


def unique_everseen(iterable, key=None):
    """ List unique elements, preserving order. Remember all elements ever seen

    Args:
        iterable: anything that can be iterated over 
        key (function, optional): A function (similar to sorted key arg) to capture value from object. Defaults to None.

    Yields:
        obj: items in the right order

    Examples:
        unique_everseen('AAAABBBCCDAABBB') --> A B C D
        unique_everseen('ABBCcAD', str.lower) --> A B C D
    """
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def base_split(f):
    """ Gets the filename without the extension """
    return os.path.splitext(os.path.basename(f))[0]


def remove_dotfiles(files, sort=True):
    """ Simple function to filter 

    Args:
        files (list of str): a list of filepaths or filenames
        sort (bool, optional): sort the result. Defaults to True.

    Returns:
        list of str: a (possibly sorted) list of files with dot files removed
    """

    fdata = []
    if IGNORE_DOT_FILES:
        fdata = [gf for gf in files if not os.path.basename(
            str(gf)).startswith('.')]
    else:
        fdata = files

    if sort:
        return sorted(fdata)
    return fdata


def save_mask(path, nparr):
    """ Saves the numpy mask to the specified output """
    tfile = TiffWriter(path)
    tfile.save(nparr, compress=5)


def is_valid_in_file(path):
    """ Determines if the file path is a valid input image file

    Args:
        path (str): path of possible image file

    Returns:
        tuple: (bool of valid or not, str of file extension)
    """
    based = os.path.splitext(path)[1].lower()[1:]

    if IGNORE_DOT_FILES and os.path.basename(str(path)).startswith('.'):
        return False, None

    for ext in ['tif', 'bmp', 'png', 'jpg', 'jpeg', 'npy', 'npz']:
        if ext in based:
            return True, ext

    return False, None


def strip_file_names(paths):
    """ Returns the file name without the extensions for all files in paths """
    return [os.path.splitext(os.path.basename(str(path)))[0] for path in paths]


def strip_extensions(paths):
    """Will separate the filepath from the extension and return tuples for each path in paths """
    return [os.path.splitext(str(path)) for path in paths]


def fix_image_channels(image):
    """ Some input images are dual channels and if so only the first layer is used/returned """
    channels = image.shape[2]
    if channels >= 2:
        image = image[:, :, 0]
    return image


def apply_normalization(image):
    """ Applies some preprocessing to normalize an image for the model.

    Note: some of this could definetely be removed and is unnecessary but to keep results consistent we've decided to keep the methods in here as described in the paper

    Some normalizations include:
      - histogram normalization
      - histogram peak correction/target and beta (shifting all values)
      - image sharpening

    Args:
        image (np.ndarray): input image to normalize

    Returns:
        np.ndarray: image to be normalized
    """
    # ease functions
    target_peak = 120
    avg_hist_peaks = 20

    def g_hist(im): return cv2.calcHist(
        [im], [0], None, [256], [0, 256]).flatten()
    def m_hist(hist): return np.mean(np.array(
        sorted(enumerate(hist), key=lambda x: x[1], reverse=True))[:avg_hist_peaks, 0])

    # gamma target peak correction
    hist = g_hist(image)
    hist_max = m_hist(hist)
    gamma = 1.0 + np.clip(float(hist_max - target_peak) /
                          np.std(image), -0.2, 0.4)
    look_up = np.empty((1, 256), np.uint8)
    for i in range(256):
        look_up[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    image = cv2.LUT(image, look_up)

    # simple post target beta correction
    hist = g_hist(image)
    hist_max = m_hist(hist)
    offset = np.clip(np.array([target_peak - hist_max], np.float32), -10, 10)
    image = np.clip(image.astype(np.float32) + offset, 0, 255).astype(np.uint8)

    # sharpen image
    blur = cv2.GaussianBlur(image, (5, 5), 1)
    image = cv2.addWeighted(image, 1.0, blur, -0.1, 0)
    return image


def load_image(path, do_fix=True, fix_depth=False):
    """ Simplified method to load images from disk of multiple formats

    Args:
        path (str): path of image
        do_fix (bool, optional): possibly simplify the channels and convert colorspaces. Defaults to True.
        fix_depth (bool, optional): fix output depth from gray to rgb. Defaults to False.

    Returns:
        np.ndarray: loaded and corrected (channelwise) image
    """
    data, fix = load_image_raw(path)

    # fix the data or channels if the file is not a saved numpy array or an output file
    if fix and do_fix:
        img = fix_image_channels(data[0: img_size_h, 0: img_size_w]).reshape(
            img_size_h, img_size_w, 1)

        if fix_depth:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img

    return data


# this fixes the network's square image back to the original aspect ratio
def rescale_out_image(image, high_quality=False):
    """ Fix the network image (warped) back into a processable image for output

    Args:
        image (np.ndarray): input mask image
        high_quality (bool, optional): use lanczos4 or just nearest (for most masks). Defaults to False.

    Returns:
        np.ndarray: resized image (original size)
    """
    return cv2.resize(image, img_aspect_size, interpolation=cv2.INTER_LANCZOS4 if high_quality else cv2.INTER_NEAREST).reshape(img_aspect_shape)


def load_image_raw(path):
    """ Load the input image and handle the different cases for different file types

    Args:
        path (str): path of image to load

    Raises:
        Exception: not valid input file (based on extension)

    Returns:
        np.ndarray: loaded input image (not resized)
    """
    valid, extension = is_valid_in_file(path)
    if not valid:
        raise Exception(path + ' is not a valid input file')

    if extension in ['bmp', 'png', 'jpg', 'jpeg']:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), True
    elif 'tif' in extension:
        tiff = TiffFile(path)

        # create a copy of the tiff image
        image = tiff.asarray().copy()
        tiff.close()

        # make sure we have a depth to the image
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)

        return image, True
    elif extension in ['npy', 'npz']:
        return np.load(path), False


def parse_color(clr):
    """ Parses a string color (such as hex) and returns a tuple of (r, g, b) """
    clr = color.parse_color(str(clr))
    return clr.red, clr.green, clr.blue


def mask_image(classes, img_stack, img_shape=None, background=None):
    """ Taking the mask image stack (layers), classes, and background image this will generate a single image with
    the colors specified in the classes list and create a visual representation of the results 

    Args:
        classes (list of dict): dictionaries that contain metadata about the classes (currently just color )
        img_stack (np.ndarray): a stack of segmentation masks
        img_shape (tuple, optional): the shape of the background image. Defaults to None.
        background (np.ndarray, optional): background image to draw on. Defaults to None.

    Returns:
        np.ndarray: a single compiled image with the segmentation masks drawn on it in order
    """
    if img_shape is None:
        img_shape = img_stack.shape[1:]

    base_buffer = np.zeros(img_shape[:2] + (3,), dtype=np.uint8)
    add_background = background is not None
    stack_height = len(classes)

    # loop through each layer in reverse and only update values that are zero
    for i in range(stack_height - 1, -2 if add_background else -1, -1):
        if i == -1:
            data = background
        else:
            data = img_stack[i]

        # only updates the elements that haven't been set yet
        update_elements = np.repeat(np.bitwise_and(data, np.all(base_buffer < min_class_depth_threshold, axis=-1)
                                                   .astype(np.uint8).reshape(img_shape) * 255), img_depth, axis=-1)

        if i == -1:
            # masked_layer = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
            masked_layer = update_elements
        else:
            # update_elements = np.repeat(data, 3, axis=-1)
            r, g, b = parse_color(classes[i]['color'])
            b_u, g_u, r_u = cv2.split(update_elements.astype(np.float32))
            masked_layer = cv2.merge(
                (b_u * b, g_u * g, r_u * r)).astype(np.uint8)
        base_buffer = np.bitwise_or(base_buffer, masked_layer)

    return base_buffer
