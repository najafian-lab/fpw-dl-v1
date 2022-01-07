""" A module that helps generate figures that are similar to the ones in the paper """
import argparse
import random
import json

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing._private.utils import runstring
import seaborn as sns
from matplotlib import rc

random.seed(100)  # to make results consistent
SHAPES = [
    ('solid', 'solid'),  # Same as (0, ()) or '-'
    ('dotted', 'dotted'),  # Same as (0, (1, 1)) or '.'
    ('dashed', 'dashed'),  # Same as '--'
    ('dashdot', 'dashdot'),  # Same as '-.'
    ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
    ('loosely dotted',        (0, (1, 10))),
    ('dotted',                (0, (1, 1))),
    ('densely dotted',        (0, (1, 1))),

    ('loosely dashed',        (0, (5, 10))),
    ('dashed',                (0, (5, 5))),
    ('densely dashed',        (0, (5, 1))),

    ('loosely dashdotted',    (0, (3, 10, 1, 10))),
    ('dashdotted',            (0, (3, 5, 1, 5))),
    ('densely dashdotted',    (0, (3, 1, 1, 1)))
]


# ---- ARGUMENT PARSING ----
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Generates figures\nPlease note arguments do not have all options! Please look at figures.py code for more fine tuning')
parser.add_argument(
    '--running_average', help='show a window preview of segmentations and results', action='store_true')
parser.add_argument(
    '--running_average_file', type=str, default='dataset/fabry/prediction/running_average_individual.json', help='File with all of the file measurements')
parser.add_argument(
    '--running_average_num', type=int, default=20, help='Number of biopsies to show for running average')
parser.add_argument(
    '--running_average_offset', type=int, default=0, help='File offset (think of this of seed)')
parser.add_argument(
    '--running_average_title', type=str, default='Running average of fabry samples', help='Title of the figure to generate')
parser.add_argument(
    '--running_average_use_overall_average',  help='Use the overall average of normalized averages', action='store_true')
parser.add_argument(
    '--running_average_show_convergence',  help='Show the point where values closely converge to the mean', action='store_true')
parser.add_argument(
    '--running_average_hide_individual',  help='Hide the individual biopsy running averages', action='store_true')
args = parser.parse_args()
# --- END ARGUMENT PARSING ---


if args.running_average:
    print('Generating Running Average')
    sns.set_style('white')
    sns.set_palette('deep', 100)

    csfont = {'fontname': 'Arial'}
    hfont = {'fontname': 'Arial'}
    rc('font', family='serif', serif=['Arial'], weight='bold')
    rc('text', usetex=False)
    rc('axes', titlesize=25)
    rc('axes', labelsize=18)
    rc('xtick', labelsize=13)
    rc('ytick', labelsize=13)
    rc('legend', fontsize=9)

    with open(args.running_average_file, 'r') as fp:
        data = json.load(fp)

    data = data['data']
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    colors = sns.color_palette('viridis', len(data) + 2)  # [(0, 0, 0)] * (len(data) + 2)

    counts = []
    y_lim = [0.4, 1.6]
    data = data[args.running_average_offset:]
    max_length = max([len(values) for values in data[:args.running_average_num]])  # get longest biopsy 
    x_vals = list(range(max_length))
    overall_average = np.ones((max_length,), np.float32)
    overall_counts = np.zeros_like(overall_average, np.uint32)
    for runs in range(len(data[:args.running_average_num])):
        # random shuffling of data (to get glom distribution)
        random.shuffle(data[runs])

        average_list = []
        cumsum = 0
        # normsum = np.mean(data[runs])
        # normval = np.std(data[runs])  # np.max(data[runs] - normsum)  # normalize to max difference
        normsum = np.sum(data[runs])
        count = len(data[runs])
        counts.append(count)
        normv = count / normsum
        for i, v in enumerate(data[runs]):
            cumsum += v * normv
            average_list.append(cumsum / float(i + 1))
            overall_average[i] += abs((cumsum / float(i + 1)) - 1.0) # append normalized value (val / cumsum with counts will normalize to 1)
            overall_counts[i] += 1
            # average_list.append(np.mean(data[runs][:(i + 1)] - normsum) / normval)

        if not args.running_average_hide_individual:
            ax.plot(list(range(len(data[runs]))), average_list, color=colors[runs % len(
                colors)], label='Biopsy {}'.format(runs + 1), alpha=1.0)  # linestyle=SHAPES[runs % len(SHAPES)][1])

    if args.running_average_use_overall_average:
        norm_avg = overall_average / (np.max(overall_counts)).astype(np.float32)
        last_avg = norm_avg[-1]

        # move zero centered normalized value back to the location 1.0 on the graph (with the above 0.0 centered deltas)
        norm_avg_p = norm_avg + (1.0 - last_avg)  
        norm_avg_m = (1.0 - norm_avg) + last_avg

        # let's find the location where the normalized first converges to 0
        if args.running_average_show_convergence:
            for i, v in enumerate(norm_avg_p):
                if abs(v - 1.0) < 0.01:  # epsilon as second number
                    ax.axvline(x=(i + 1), color='green', alpha=0.5)
                    ax.text((i + 1) + 0.1, y_lim[0] + 0.1, ' Roughly converges at {}'.format(i + 1), rotation=0)
                    break

        ax.fill_between(x_vals, norm_avg_m, norm_avg_p, color='black', alpha=0.55, zorder=3, label='Overall\nNormalized\nAverage')
        # ax.plot(x_vals, norm_avg, color='black', linewidth=4, linestyle='dashed', label='Overall\nNormalized\nAverage')

    ax.axhline(y=1, color='gray', alpha=0.8)

    # add the min and max count lines
    ax.axvline(x=np.min(counts), color='gray', alpha=0.8)
    ax.text(np.min(counts) + 0.1, y_lim[0] + 0.1, ' Min Images {}'.format(np.min(counts)), rotation=0)
    ax.axvline(x=np.max(counts), color='gray', alpha=0.8,
               linestyle='-')
    ax.text(np.max(counts) - 0.1, y_lim[0] + 0.1, 'Max Images {} '.format(np.max(counts)), rotation=0, horizontalalignment='right')
    
    ax.set_ylabel('FPW Normalized Running Average', **csfont)
    ax.set_xlabel('# of Images', **csfont)
    ax.set_title(args.running_average_title, **csfont)
    ax.set_ylim(y_lim)
    ax.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig('running_average.png')

print('Exiting...')
