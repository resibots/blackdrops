#!/usr/bin/python
from numpy import mean, median, var, sqrt
from math import sqrt
from subprocess import Popen, PIPE
from sys import argv

import glob
from pylab import *
import scipy.stats
from scipy.interpolate import spline, interp1d
from scipy.optimize import curve_fit
import scipy.io
import numpy as np

import brewer2mpl
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors

params = {
    'axes.labelsize': 8,
    'font.size': 5,
    'legend.fontsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'text.usetex': False,
    'figure.figsize': [20.0, 10.0]
}
rcParams.update(params)

# Command to verify which experiments ended and return the number of them
# grep -rE "\#15|Dummy reward:.*" | grep -Pzo "\#15.*\n.*Dummy reward:.*" | sed -re "N;s/.*\n.*exp_([0-9]+).*/\1/g" | sort -V | wc -l

try:
    from subprocess import DEVNULL
except ImportError:
    import os
    DEVNULL = open(os.devnull, 'wb')

def beautify_boxplot(bp, ax):
    for i in range(len(bp['boxes'])):
        box = bp['boxes'][i]
        box.set_linewidth(0)
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
            boxCoords = zip(boxX,boxY)
            boxPolygon = Polygon(boxCoords, facecolor = colors[i % len(colors)], linewidth=0)
            ax.add_patch(boxPolygon)

    for i in range(0, len(bp['boxes'])):
        c_i = i%len(colors)
        bp['boxes'][i].set_color(colors[c_i])
        # we have two whiskers!
        bp['whiskers'][i*2].set_color(colors[c_i])
        bp['whiskers'][i*2 + 1].set_color(colors[c_i])
        bp['whiskers'][i*2].set_linewidth(2)
        bp['whiskers'][i*2 + 1].set_linewidth(2)
        # top and bottom fliers
        bp['fliers'][i].set(markerfacecolor=colors[c_i],
                        marker='o', alpha=0.75, markersize=6,
                        markeredgecolor='none')
        bp['medians'][i].set_color('black')
        bp['medians'][i].set_linewidth(3)
        # and 4 caps to remove
        for c in bp['caps']:
            c.set_linewidth(0)

def plot_box(data, name):
    fig = figure()

    ax1 = fig.gca()

    data = map(np.asarray, data)
    bp = ax1.boxplot(data, notch=0, sym='b+', vert=1, whis=[1.5, 1.5], positions=None, widths=0.3, manage_xticks=True)

    beautify_boxplot(bp, ax1)


    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax1.tick_params(axis='x', direction='out')
    ax1.tick_params(axis='y', length=0)
    ylim([0,32])

    ax1.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    ax1.set_axisbelow(True)

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    fig.savefig(name+'.png')
    close()

def stats(n):
    return (
        mean(n),
        median(n),
        sqrt(var(n)),
        min(n),
        max(n),
        len([r for r in n if r < 10]),
        len([r for r in n if 10 <= r and r < 15]),
        len([r for r in n if 15 <= r and r < 20]),
        len([r for r in n if 20 <= r and r < 25]),
        len([r for r in n if 25 <= r and r < 30]),
        len([r for r in n if 30 <= r and r < 35]),
        len(n)
    )

def get_statistics(data, th, ep, reducer=median):
    data = sorted(data, key=lambda e: (e[0], e[1]))
    iterations = []
    dummy_rewards = []
    rewards = []

    last_id = -1
    for e in data:
        if last_id != e[0]:
            if last_id != -1 and len(ei) >= ep:
                iterations.append(list(ei[:ep]))
                dummy_rewards.append(list(edr[:ep]))
                rewards.append(list(er[:ep]))
            ei = []
            edr = []
            er = []
            last_id = e[0]
        ei.append(e[2])
        edr.append(e[3])
        er.append(e[4])

    if len(ei) >= ep:
        iterations.append(list(ei[:ep]))
        dummy_rewards.append(list(edr[:ep]))
        rewards.append(list(er[:ep]))

    count_above_th = lambda vs: sum(map(lambda v: 1 if v >= th else 0, vs))
    count_below_th = lambda vs: sum(map(lambda v: 1 if v < th else 0, vs))
    def count_till_th(vs):
        pos = 0
        for v in vs:
            if v >= th:
                return pos
            pos += 1
        return len(vs)

    def count_success_ath(vs):
        if len(vs) == count_till_th(vs):
            return 0
        return 100*float(count_above_th(vs[count_till_th(vs)-1:])-1)/(len(vs)-count_till_th(vs))

    stats_rewards = lambda eps: (
                        min(map(count_till_th, eps)),
                        reducer(map(count_till_th, eps)),
                        max(map(count_till_th, eps))
                    )
    stats_success = lambda eps: reducer(map(count_success_ath, eps))

    def stats_iters_cols(eps):
        res = []
        for j in range(len(eps[0])):
            vs = []
            for i in range(len(eps)):
                vs.append(eps[i][j])
            res.append(int(reducer(vs)))
        return res
    def stats_rewards_cols(eps, red):
        res = []
        for j in range(len(eps[0])):
            vs = []
            for i in range(len(eps)):
                vs.append(eps[i][j])
            res.append(red(vs))
        return res

    stats_iters = lambda eps: map(lambda vs: int(mean(vs)), eps)

    return (
            len(rewards),
            stats_rewards(dummy_rewards),
            stats_rewards(rewards),
            stats_success(dummy_rewards),
            stats_success(rewards),
            stats_rewards_cols(dummy_rewards, reducer),
            stats_rewards_cols(rewards, reducer),
            stats_rewards_cols(dummy_rewards, lambda v: sqrt(var(v))),
            stats_rewards_cols(rewards, lambda v: sqrt(var(v))),
            int(mean(stats_iters(iterations))),
            stats_iters_cols(iterations),
            stats_rewards_cols(dummy_rewards, lambda v: v),
            stats_rewards_cols(rewards, lambda v: v),
    )

# Get the path from command line
path = "."
if len(argv) > 1:
    path = argv[1]

# Get the commands ready
grep_1 = """grep -rnE "Optimization iterations: [0-9]+.*|Dummy reward: [0-9]+\\.?.*|Reward: [0-9]+\\.?.*|Learning iteration #.*" """
grep_2 = """grep -Pzo ".*Learning iteration #.*\\n.*Optimization iterations: [0-9]+.*\\n.*Dummy reward: [0-9]+\\.?.*\\n.*Reward: [0-9]+\\.?.*" """
sed_1 = """sed -re 'N;N;N;s/.*exp\\_([0-9]+).*Learning iteration #([0-9]+).*\\n.*Optimization iterations: ([0-9]+).*\\n.*Dummy reward: ([0-9]+\\.[0-9e\\-]*)\\n.*Reward: ([0-9]+\\.[0-9e\-]*)/(\\1, \\2, \\3, \\4, \\5),/g' """

# Launch them
p = Popen([grep_1], stdout=PIPE, stderr=DEVNULL, shell=True, cwd=path)
p = Popen([grep_2], stdin=p.stdout, stdout=PIPE, stderr=DEVNULL, shell=True, cwd=path)
p = Popen([sed_1], stdin=p.stdout, stdout=PIPE, stderr=DEVNULL, shell=True, cwd=path)
out, err = p.communicate()
results = eval("["+out.replace("\n", " ")[:-2]+"]")

# Do statistics
th = 30
ep = 15
stats = get_statistics(results, th, ep, median)

# Description of get_statistics result:
#   Processed repetitions (to be processed the episodes should be complete)

#   Mean/Median of the first episode when the dummy reward was higher than TH
#   Lower value of the first episode when the dummy reward was higher than TH
#   Higer value of the first episode when the dummy reward was higher than TH

#   Mean/Median of the first episode when the reward was higher than TH
#   Lower value of the first episode when the reward was higher than TH
#   Higer value of the first episode when the reward was higher than TH

#   Mean/Median of the dummy reward of each iteration
#   Mean/Median of the reward of each iteration

#   Variance of the dummy reward of each iteration
#   Variance of the reward of each iteration

#   Mean/Median of the porcentage of dummy rewards above TH after the 1st one
#   Mean/Median of the porcentage of rewards above TH after the 1st one

#   Mean of all the iterations of cmaes
#   Mean of the iterations for each episode for cmaes (15 values)

#   Dummy rewards raw data
#   Rewards raw data

# Plot
print "\nExperiment data (th is " + str(th) + "): "
print "    Dummy th: ", stats[1]
print "    Real th: ", stats[2]
print "    Dummy success after th: ", stats[3]
print "    Real success after th: ", stats[4]
print "    Average of cmaes iterations: ", stats[9]

print ""
plot_box(stats[-2], "dummy_rewards_ep")
plot_box(stats[-1], "rewards_ep")
print "Box plots creatin finished."
