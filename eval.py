""" Simple script to evaluate the performance of the model and generate a spreadsheet of the dices """
import glob
import os

import cv2
import numpy as np
from openpyxl import Workbook
from skimage.external.tifffile import tifffile
from tqdm import tqdm

import configs

# specify the evaluation directories
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = 'dataset'
PREDICTED_MASKS = os.path.join(
    WORK_DIR, DATASET, 'eval', 'images', 'prediction', '*.tiff')
SEGMENTED_MASKS = os.path.join(
    WORK_DIR, DATASET, 'eval', 'masks', '*.tiff')
NETWORK_SIZE = tuple(configs.load_project_settings()[
                     'network_in'])  # get network image size


def sort_unique_files(files):
    """ Gets the unique and sorted list of files with the hidden files removed """
    if files is None:  # ignore this
        return None
    return sorted(configs.unique_everseen(configs.remove_dotfiles(files, sort=True), key=configs.base_split), key=configs.base_split)


def dice_coef(y_true, y_pred):
    """ Calculates intersection over union (basic DICE coefficient)

    Args:
        y_true (np.ndarray): ground truth mask
        y_pred (np.ndarray): predicted mask

    Returns:
        float: dice score between 0 and 1
    """
    # flatten, convert to float, and scale to 0-1
    y_true_f = y_true.flatten().astype(np.float32) / 255
    y_pred_f = y_pred.flatten().astype(np.float32) / 255

    # calc total intersect by mult both masks and summing overlapped pixels
    intersection = np.sum(y_true_f * y_pred_f)

    # smoothing param is 1 and assume perfect overlap is 1 by mult (that's why we scaled to 1)
    return (2. * intersection + 1) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1)


def compare():
    """ Main function """
    orig_files = sort_unique_files(glob.glob(SEGMENTED_MASKS))
    pred_files = sort_unique_files(glob.glob(PREDICTED_MASKS))

    # do some sanity checks
    if len(orig_files) != len(pred_files):
        raise Exception(
            f'Length of segmented mask list does not equal predicted mask list (SEG: {len(orig_files)}, PRED: {len(pred_files)})')

    # match the files
    for o, p, in zip(orig_files, pred_files):
        if configs.base_split(o) != configs.base_split(p):
            raise Exception(
                f'The sorted files of segmented {o} and predicted {p} do not have the same file name')

    # load the files
    print(f'Processing {len(orig_files)} files...')
    dice_groups = {}  # aggregate scores per biopsy

    # create workbook for spreadsheet
    wb = Workbook()
    ws = wb.active
    ws.title = 'All Scores'
    ws.append(['Name', 'Dice Membrane', 'Dice Slits', 'Dice Average'])

    # process each pair of files
    for i, (o, p) in tqdm(enumerate(zip(orig_files, pred_files)), desc='Evaluating', total=len(orig_files)):
        # load images
        # just append the edge and slits (as these masks include more data than necessary)
        orig = tifffile.imread(o)[2:4]
        pred = tifffile.imread(p)

        # calculate dice coef for each layer
        dice_layers = []
        for j in range(0, 2):
            # resize with nearest to network size
            orig_r = cv2.resize(orig[j], NETWORK_SIZE,
                                interpolation=cv2.INTER_NEAREST)
            pred_r = cv2.resize(pred[j], NETWORK_SIZE,
                                interpolation=cv2.INTER_NEAREST)

            # calc dice and append to list
            dice_v = float(dice_coef(orig_r, pred_r))
            dice_layers.append(dice_v)

            # add it to aggregate score
            biopsy_prefix = configs.base_split(
                orig_files[i]).replace('-', '_').split('_')[0]
            if biopsy_prefix not in dice_groups:
                dice_groups[biopsy_prefix] = {}

            # create layer group
            if j not in dice_groups[biopsy_prefix]:
                dice_groups[biopsy_prefix][j] = []

            # append to layer group
            dice_groups[biopsy_prefix][j].append(dice_v)

        # add values to spreadsheet
        ws.append([configs.base_split(orig_files[i]), dice_layers[0],
                  dice_layers[1], float(np.mean(dice_layers))])
    print('Done!')

    # let's calculate overall averages by their grouped scores
    ws = wb.create_sheet('Aggregate Scores')
    ws.append(['Biopsy Prefix', 'Membrane Average',
              'Membrane STD', 'Slit Average', 'Slit STD'])
    for agg in dice_groups.keys():
        membranes = dice_groups[agg][0]
        slits = dice_groups[agg][1]

        ws.append([agg, float(np.mean(membranes)), float(
            np.std(membranes)), float(np.mean(slits)), float(np.std(slits))])

    print('Exporting notebook....')
    wb.save('dices.xlsx')
    print('Done! Saved results to dices.xlsx')


if __name__ == '__main__':
    import traceback
    print('Starting evaluation script')
    try:
        compare()
    except Exception as err:
        print(f'Failed to evaluate. Error {str(err)}')
        traceback.print_exc()
        print('Exiting...')
        exit(1)
