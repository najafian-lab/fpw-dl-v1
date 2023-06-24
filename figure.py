""" A module that helps generate figures that are similar to the ones in the paper """
import argparse
import random
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing._private.utils import runstring
import seaborn as sns
from scipy import stats
import math
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
# import george 
# from george import kernels
from scipy.optimize import curve_fit
from matplotlib import rc
from docker import OUTPUT_DIR, ensure_output, IN_DOCKER


# x and y limitations (REQUIRED)
Y_LIM = [0.4, 1.6]
X_LIM = [-5.0, 320]
MAX_BIOPSIES = 20
EXCLUDE_BIOPSIES = ['85-0852', '82-0118']  # two normals (not actually normals...) to be excluded
EPS = 0.01
STD = 1.96
ensure_output()  # ensure output folder exists

# it was noticed that np.random.seed was not set for paper figures. Convergence results should be within a couple images 0-4
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
sps = parser.add_subparsers(dest='command', help='Select figure to generate')

# RUNNING AVERAGE COMMAND
run_avg = sps.add_parser('running_average', help='show a window preview of segmentations and results')
run_avg.add_argument(
    '--file', type=str, default='output/fabry/running_average_individual.json', help='File with all of the file measurements')
run_avg.add_argument(
    '--num', type=int, default=20, help='Number of biopsies to show for running average')
run_avg.add_argument(
    '--offset', type=int, default=0, help='File offset (think of this of seed)')
run_avg.add_argument(
    '--title', type=str, default='Running average of fabry samples', help='Title of the figure to generate')
run_avg.add_argument(
    '--use_overall_average',  help='Use the overall average of normalized averages', action='store_true')
run_avg.add_argument(
    '--_show_convergence',  help='Show the point where values closely converge to the mean', action='store_true')
run_avg.add_argument(
    '--hide_individual',  help='Hide the individual biopsy running averages', action='store_true')
run_avg.add_argument(
    '--x_lim_max', type=int, default=X_LIM[1], help='Max number of images. Usually 140 max per biopsy')

# SAMPLED CONVERGENCE
run_avg = sps.add_parser('sampled_convergence', help='...')
run_avg.add_argument(
    '--folder', type=str, default='output', help='folder with outputs')

# SAMPLED CONVERGENCE
run_clt = sps.add_parser('sampled_clt', help='...')
run_clt.add_argument(
    '--folder', type=str, default='output', help='folder with outputs')

args = parser.parse_args()
# X_LIM = (X_LIM[0], args.x_lim_max)
# --- END ARGUMENT PARSING ---

sns.set_style('white')
sns.set_palette('deep', 100)

csfont = {'fontname': 'Arial'}
hfont = {'fontname': 'Arial'}
rc('font', family='serif', serif=['Arial'], weight='bold')
rc('text', usetex=False)
rc('axes', titlesize=31)
rc('axes', labelsize=25)
rc('xtick', labelsize=17)
rc('ytick', labelsize=17)
rc('legend', fontsize=11)


if args.command == 'running_average':
    print('Generating Running Average')
    with open(args.running_average_file, 'r') as fp:
        data = json.load(fp)

    data = data['data']
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    colors = sns.color_palette('viridis', len(data) + 2)  # [(0, 0, 0)] * (len(data) + 2)

    counts = []
    data = data[args.running_average_offset:]
    max_length = max([len(values) for values in data[:args.running_average_num]])  # get longest biopsy 
    x_vals = list(range(max_length))
    overall_average = np.ones((max_length,), np.float32)
    overall_counts = np.zeros_like(overall_average, np.uint32)
    bnum = 1
    for runs in range(len(data[:args.running_average_num])):
        # random shuffling of data (to get glom distribution)
        random.shuffle(data[runs])

        # skip too long segments
        if len(data[runs]) > X_LIM[1]:
            print('Skipping too long of a run', runs)
            continue

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
                colors)], label='Biopsy {}'.format(bnum), alpha=1.0)  # linestyle=SHAPES[runs % len(SHAPES)][1])
            bnum += 1

    if args.running_average_use_overall_average:
        norm_avg = overall_average / (np.max(overall_counts)).astype(np.float32)
        last_avg = norm_avg[-1]

        # move zero centered normalized value back to the location 1.0 on the graph (with the above 0.0 centered deltas)
        norm_avg_p = norm_avg + (1.0 - last_avg)  
        norm_avg_m = (1.0 - norm_avg) + last_avg

        # let's find the location where the normalized first converges to 0
        if args.running_average_show_convergence:
            for i, v in enumerate(norm_avg_p):
                if abs(v - 1.0) < EPS:  # epsilon as second number
                    ax.axvline(x=(i + 1), color='green', alpha=0.5)
                    ax.text((i + 1) + 1.0, Y_LIM[0] + 1.03, ' Roughly\n converges\n at {}'.format(i + 1), rotation=0)
                    break

        ax.fill_between(x_vals, norm_avg_m, norm_avg_p, color='black', alpha=0.55, zorder=3, label='Overall\nNormalized\nAverage')
        # ax.plot(x_vals, norm_avg, color='black', linewidth=4, linestyle='dashed', label='Overall\nNormalized\nAverage')

    ax.axhline(y=1, color='gray', alpha=0.8)

    # add the min and max count lines
    ax.axvline(x=np.min(counts), color='gray', alpha=0.8)
    ax.text(np.min(counts) - 1.7, Y_LIM[0] + 0.1, ' Min\n images {}'.format(np.min(counts)), rotation=0, horizontalalignment='right')
    ax.axvline(x=np.max(counts), color='gray', alpha=0.8,
               linestyle='-')
    ax.text(np.max(counts) + 1.7, Y_LIM[0] + 0.1, 'Max \nimages {} '.format(np.max(counts)), rotation=0)
    
    ax.set_ylabel('FPW Normalized Running Average', **csfont)
    ax.set_xlabel('# of Images', **csfont)
    ax.set_title(args.running_average_title, **csfont)
    ax.set_ylim(Y_LIM)
    ax.set_xlim(X_LIM)
    ax.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig(os.path.join(OUTPUT_DIR, 'running_average.png'))
elif args.command == 'sampled_convergence':
    listed = ['fabry-f', 'fabry', 'dkd', 'mcdfsgs', 'normal']  # os.listdir(args.folder)
    names = ['Fabry Female', 'Fabry Male', 'DKD', 'MCD', 'Normal']
    folders = [os.path.join(args.folder, f) for f in  listed if os.path.isdir(os.path.join(args.folder, f))]

    # get valid file names
    file_name = 'running_average_individual.json'
    vfiles = [(os.path.join(f, file_name), os.path.basename(f)) for f in folders if os.path.isfile(os.path.join(f, file_name))]
    
    # read all data
    all_data = []
    max_length = 0
    max_runs = 0
    for fpath, outtype in vfiles:
        print(f'Computing for {outtype}')
        # with open(fpath, 'r') as fp:
        #     data = json.load(fp)
        # data = data['data'][:20]
        with open(fpath, 'r') as fp:
            data_f = json.load(fp)
        data = []

        # filter data first
        for folder, d in zip(data_f['folders'], data_f['data']):
            passed = True
            for fn in EXCLUDE_BIOPSIES:
                if f'prediction/{fn}' in folder:
                    print('Excluding', fn)
                    passed = False
                    break
            if passed:
                data.append(d)
        
        data = data[:MAX_BIOPSIES]
        max_length = max([len(values) for values in data] + [max_length])  # get longest biopsy
        max_runs = max(len(data), max_runs)
        all_data.append(data)
    
    # max the num of biopsies
    max_length = min(max_length, X_LIM[1])

    def central_limit_convergence(data, ax, max_length, label, color):
        if len(data) > max_length:
            vec = np.array(data[:max_length], np.float32)
        else:
            vec = np.array(data, np.float32)  # exclude last sample since variance would be 0
        
        # create a shuffled matrix
        samples = 10000
        # running = vec.reshape(1, -1).repeat(samples, axis=0)
        running = np.zeros((samples, max_length), np.float32)
        rng = np.random.default_rng()
        for s in range(samples):
            # running[s, :] = rng.choice(running[s, :], size=len(vec), replace=True)
            running[s, :] = rng.choice(vec, size=max_length, replace=True)
            # rng.shuffle(running[s, :])  # shuffle each run

        # DP to calculate running average
        # for i in range(1, len(vec)):  # first is already average
        for i in range(1, max_length):
            # we multiply previous average by number of previous images, add the current image mean and divide by the total current images
            running[:, i] = ((i * running[:, i - 1]) + running[:, i]) / (i + 1)

        # create the running sampled average statistics
        range_runs = np.arange(1, max_length + 1)  # len(vec) + 1)

        # for all runs
        std_runs = np.std(running, axis=0)
        mean_runs = np.mean(running, axis=0)
        upper_all = (STD*std_runs).tolist()

        ax.fill_between(
            range_runs[:i + 1],
            np.clip(mean_runs - STD*std_runs, a_min=0, a_max=None),
            mean_runs + STD*std_runs,
            alpha=0.8,
            label=f"{label}",
            color=color
        )

        # assumed_mean = mean_runs[i]
        assumed_mean = np.mean(mean_runs)
        within_mean = assumed_mean * 0.15  # within 10% of estimated mean consider converged
        events = []
        for i, u_v in enumerate(upper_all):
            # print(u_v, within_mean)
            if u_v <= within_mean:  # within standard error (CI)
                # ax.axvline(x=i, color='gray', linewidth=2, alpha=0.8, linestyle='-')
                events.append(i + 1)
                break

        """
        # fit to assumption of central limit theorem
        # find true variance of averages
        def var_reduce(ind, sigma):
            return sigma / np.sqrt(ind + 1)  # we expect by CLT variance to follow (sigma of X) / sqrt(n)

        # fit to find the closest estimated true standard deviation of X 
        # this should be a better estimation than just doing std(FPW) by using multiple steps of CLT (could be wrong)        
        # params, _ = curve_fit(var_reduce, range_runs, std_runs[:i + 1])
        # est_sigma = np.std(data, ddof=1) # params[0]
        # assumed_mean = mean_runs[i]
        # lower = []
        # upper = []
        # # for j in range(i, max_length):
        # for j in range(1, max_length):
        #     new_sval = (est_sigma / np.sqrt(j + 1))

        #     lower.append(assumed_mean - STD*new_sval)
        #     upper.append(assumed_mean + STD*new_sval)
        #     upper_all.append(STD*new_sval)

        # within_mean = assumed_mean * 0.1  # within 10% of estimated mean consider converged
        # events = []  # np.zeros(max_length + 1, dtype=np.int32)
        # for i, u_v in enumerate(upper_all):
        #     # print(u_v, within_mean)
        #     if 0.5*u_v < within_mean:  # two tailed STD
        #         # ax.axvline(x=i, color='gray', linewidth=2, alpha=0.8, linestyle='-')
        #         events.append(i + 1)
        #         break

        # ax.fill_between(
        #     np.arange(len(lower)), # np.arange(len(vec) - 1, max_length),
        #     lower,
        #     upper,
        #     alpha=0.8,
        #     color=color
        # )
        """

        # determine convergence line
        # for i in range(max_length):
        #     if 
        return events

    all_run_biopsies = []
    max_biop_run = 0
    for d_num, ((fpath, outtype), data) in enumerate(zip(vfiles, all_data)):
        # for each run in max length 
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)

        all_biopsies = []
        colors = sns.color_palette('viridis', len(data) + 2)

        events = []
        for run_num in range(len(data)):
            all_biopsies.extend(data[run_num])
            events.extend(central_limit_convergence(data[run_num], ax, max_length, f'Biopsy {run_num + 1}', colors[run_num]))

        # include converged CDF
        ax2 = ax.twinx()
        # print(events)
        n, bins, patches = ax2.hist(events, bins=max_length, linewidth=2, density=True, cumulative=True, label='Convergence CDF', histtype='step', color='gray', alpha=0.8)
        # print(n)
        patches[0].set_xy(patches[0].get_xy()[:-1])  # remove right edge
        ax2.set_ylim(-0.02, 1.02)
        for i in range(len(n)):
            # print(i, n[i])
            if n[i] >= 0.85:
                ax2.axvline(int(bins[i]), ymin=-0.02, ymax=1.02, linestyle='dashed', linewidth=2, label=f'85% CDF (i = {int(bins[i])})')
                break

        ax2.set_ylabel('Biopsy Convergence CDF', **csfont)
        ax2.plot([[events[-1], 1.0], [max_length, 1.0]], alpha=0.8, color='gray')

        ax.set_ylabel('FPW Pixel Average', **csfont)
        ax.set_xlabel('# of Images', **csfont)
        ax.set_title(f'{names[d_num]}', **csfont)
        ax2.legend(frameon=True, loc='upper left')
        ax.legend(frameon=True, loc='upper right')  # , bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f'sampled_{outtype}.png'))

        # now do aggregate results
        all_run_biopsies.append((all_biopsies, outtype))
        max_biop_run = max(max_biop_run, len(all_biopsies))

    # restrict max biopsy run
    max_biop_run = min(max_biop_run, 800)

    # plot current runs max/min for the
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)

    # create stats for each run
    colors = sns.color_palette('viridis', len(all_run_biopsies) + 2)
    for ind, (biops, outtype) in enumerate(all_run_biopsies):
        running = np.array(biops, np.float32)
        central_limit_convergence(running, ax, max_biop_run, names[ind], colors[ind])

    # np.zeros((len(all_data), len(data), max_length, 3), np.float32)
    ax.set_ylabel('Aggerate FPW Pixel Average', **csfont)
    ax.set_xlabel('# of Images', **csfont)
    ax.set_title('Aggregated sampled FPW Pixel Image Convergence', **csfont)
    ax.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.savefig(os.path.join(OUTPUT_DIR, 'sampled_aggregate.png'))
elif args.command == 'sampled_clt':
    listed = ['normal']  # os.listdir(args.folder)
    names = ['Normal']
    folders = [os.path.join(args.folder, f) for f in  listed if os.path.isdir(os.path.join(args.folder, f))]

    # get valid file names
    file_name = 'running_average_individual.json'
    vfiles = [(os.path.join(f, file_name), os.path.basename(f)) for f in folders if os.path.isfile(os.path.join(f, file_name))]
    
    # read all data
    all_data = []
    max_length = 0
    max_runs = 0
    for fpath, outtype in vfiles:
        print(f'Computing for {outtype}')
        with open(fpath, 'r') as fp:
            data = json.load(fp)
        data = data['data']

        # # filter data first
        # for folder, data in zip(data_f['folders'], data_f['data']):
        #     print(folder)

        # exit(0)
        max_length = max([len(values) for values in data] + [max_length])  # get longest biopsy
        max_runs = max(len(data), max_runs)
        all_data.append(data)

    samples = 200
    run_num = 1
    for d_num, ((fpath, outtype), data) in enumerate(zip(vfiles, all_data)):
        # for each run in max length 
        fig = plt.figure(figsize=(15, 8))

        num_sample = [1, 5, 10, 15] # np.arange(1, len(data), 2)
        ax = fig.subplots(1, len(num_sample))

        all_biopsies = []
        colors = sns.color_palette('viridis', len(data) + 2)
        vec = np.array(data[run_num], dtype=np.float32)

        rng = np.random.default_rng()
        # x = []
        y = []
        for ind, n in enumerate(num_sample):
            fpw_avgs = []
            for i in range(samples):
                sampled = rng.choice(vec, size=n, replace=True)
                fpw_avgs.append(np.mean(sampled))
                # x.append(n)
                # y.append(np.mean(sampled))
            
            std = np.std(fpw_avgs)
            mean = np.mean(fpw_avgs)
            min_ = np.min(fpw_avgs)
            max_ = np.max(fpw_avgs)
            scaler = np.linspace(min_, max_, 100) # stats.norm.ppf(0.01), stats.norm.ppf(0.99), 100)
            sns.histplot(fpw_avgs, ax=ax[ind], bins=20, color='black', stat='density')
            sns.lineplot(scaler, stats.norm.pdf(scaler, loc=mean, scale=std), ax=ax[ind], linewidth=4, linestyle='dashed', label='Normal Dist')
            # ax[ind].hist(fpw_avgs, density=True)
            # ax[ind].plot(scaler, stats.norm.pdf(scaler, loc=mean, scale=std))
            ax[ind].set_xlabel(f'FPW Pixel Average')
            ax[ind].set_title(f'i = {n}')
            ax[ind].legend(loc='upper right')
            
            # ax[ind]
        # sns.kdeplot(
        #     x=x, y=y, fill=True, ax=ax
        # )
        ax[0].set_ylabel('Density')
        fig.suptitle('Histograms of FPW Pixel Average', fontsize=34)
        fig.tight_layout()
        fig.savefig('test.png')
else:
    print('Command not provided. Please use --help to see list of commands')

print('Exiting...')
