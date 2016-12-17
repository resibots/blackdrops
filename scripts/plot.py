import glob
from pylab import *
import scipy.stats
from scipy.interpolate import spline, interp1d
from scipy.optimize import curve_fit
import scipy.io

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

def load_exps(dir):
    f_list = glob.glob(dir+'/*/*/results.dat')
    reality = []
    estimate = []
    e = 0
    for fi in f_list:
        with open(fi) as f:
            content = f.readlines()
            n = len(content)
            if n < 30:
                continue
            reality.append([])
            estimate.append([])
            for i in range(1, n):
                s = content[i].split()
                s = [float(k) for k in s]
                if(len(s)<40):
                    continue
                if i%2 == 0:
                    reality[e].append(s)
                else:
                    estimate[e].append(s)
        e = e+1
    # reality[e_index][trial_index] -> sequence of rewards
    return reality,estimate

def load_pilco(dir):
    f_list = glob.glob(dir+'/*/cartPole_15_H40.mat')
    reality = []
    estimate = []
    e = 0
    for fi in f_list:
        mat = scipy.io.loadmat(fi)
        reals = mat['realCost'][0]
        estis = mat['fantasy']['mean'][0][0][0]
        reality.append([])
        estimate.append([])
        for k in range(1,len(reals)):
            reality[e].append([1-c for c in reals[k][0]])
            estimate[e].append([1-c for c in estis[k-1][0]])

        e = e+1
        # with open(fi) as f:
        #     content = f.readlines()
        #     n = len(content)
        #     reality.append([])
        #     estimate.append([])
        #     for i in range(1, n):
        #         s = content[i].split()
        #         s = [float(k) for k in s]
        #         if(len(s)<40):
        #             continue
        #         if i%2 == 0:
        #             reality[e].append(s)
        #         else:
        #             estimate[e].append(s)
        #     e = e+1
    # reality[e_index][trial_index] -> sequence of rewards
    return reality,estimate

def plot_fig(reality, estimates, name):
    reals = []
    reals_p_25 = []
    reals_p_75 = []
    estis = []
    estis_p_25 = []
    estis_p_75 = []
    for i in range(len(reality[0])):
        real = []
        real_p_25 = []
        real_p_75 = []
        esti = []
        esti_p_25 = []
        esti_p_75 = []
        for j in range(len(reality[0][0])):
            real.append(np.median([reality[k][i][j] for k in range(len(reality))]))
            real_p_25.append(np.percentile([reality[k][i][j] for k in range(len(reality))], 25))
            real_p_75.append(np.percentile([reality[k][i][j] for k in range(len(reality))], 75))
            esti.append(np.median([estimates[k][i][j] for k in range(len(estimates))]))
            esti_p_25.append(np.percentile([estimates[k][i][j] for k in range(len(estimates))], 25))
            esti_p_75.append(np.percentile([estimates[k][i][j] for k in range(len(estimates))], 75))
        reals.append(real)
        reals_p_25.append(real_p_25)
        reals_p_75.append(real_p_75)
        estis.append(esti)
        estis_p_25.append(esti_p_25)
        estis_p_75.append(esti_p_75)

    for i in range(len(reals)):
        fig = figure()

        ax1 = fig.gca()

        xaxis = [k for k in range(1,41)]

        ax1.plot(xaxis, reals[i], '-')
        ax1.fill_between(xaxis, reals_p_75[i], reals_p_25[i], alpha=0.3)
        # ax1.plot(xaxis, estis[i], 'r-')
        # ax1.fill_between(xaxis, estis_p_75[i], estis_p_25[i], color='r', alpha=0.3)


        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.get_xaxis().tick_bottom()
        ax1.get_yaxis().tick_left()
        ax1.tick_params(axis='x', direction='out')
        ax1.tick_params(axis='y', length=0)
        ylim([0,1])

        ax1.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
        ax1.set_axisbelow(True)

        fig.tight_layout()
        fig.subplots_adjust(top=0.95)

        fig.savefig(name+'_'+str(i+1)+'.png')
        close()

    # for i in range(len(reality)):
    #     for j in range(len(reality[i])):
    #         fig = figure()
    #
    #         ax1 = fig.gca()
    #
    #         xaxis = [k for k in range(1,41)]
    #
    #         ax1.plot(xaxis, reality[i][j], '-')
    #         ax1.plot(xaxis, estimates[i][j], 'r-')
    #
    #
    #         ax1.spines['top'].set_visible(False)
    #         ax1.spines['right'].set_visible(False)
    #         ax1.spines['left'].set_visible(False)
    #         ax1.get_xaxis().tick_bottom()
    #         ax1.get_yaxis().tick_left()
    #         ax1.tick_params(axis='x', direction='out')
    #         ax1.tick_params(axis='y', length=0)
    #
    #         ax1.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    #         ax1.set_axisbelow(True)
    #
    #         fig.tight_layout()
    #         fig.subplots_adjust(top=0.95)
    #
    #         fig.savefig(name+'_'+str(i)+'_'+str(j)+'.png')
    #         close()

def plot_comp(reality, estimates, name):
    reals = []
    reals_p_25 = []
    reals_p_75 = []
    estis = []
    estis_p_25 = []
    estis_p_75 = []
    for i in range(len(reality[0])):
        real = []
        real_p_25 = []
        real_p_75 = []
        esti = []
        esti_p_25 = []
        esti_p_75 = []
        for j in range(len(reality[0][0])):
            real.append(np.median([reality[k][i][j] for k in range(len(reality))]))
            real_p_25.append(np.percentile([reality[k][i][j] for k in range(len(reality))], 25))
            real_p_75.append(np.percentile([reality[k][i][j] for k in range(len(reality))], 75))
            esti.append(np.median([estimates[k][i][j] for k in range(len(estimates))]))
            esti_p_25.append(np.percentile([estimates[k][i][j] for k in range(len(estimates))], 25))
            esti_p_75.append(np.percentile([estimates[k][i][j] for k in range(len(estimates))], 75))
        reals.append(real)
        reals_p_25.append(real_p_25)
        reals_p_75.append(real_p_75)
        estis.append(esti)
        estis_p_25.append(esti_p_25)
        estis_p_75.append(esti_p_75)

    for i in range(len(reals)):
        fig = figure()

        ax1 = fig.gca()

        xaxis = [k for k in range(1,41)]

        ax1.plot(xaxis, reals[i], '-')
        ax1.fill_between(xaxis, reals_p_75[i], reals_p_25[i], alpha=0.3)
        ax1.plot(xaxis, estis[i], 'r-')
        ax1.fill_between(xaxis, estis_p_75[i], estis_p_25[i], color='r', alpha=0.3)


        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.get_xaxis().tick_bottom()
        ax1.get_yaxis().tick_left()
        ax1.tick_params(axis='x', direction='out')
        ax1.tick_params(axis='y', length=0)
        ylim([0,1])

        ax1.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
        ax1.set_axisbelow(True)

        fig.tight_layout()
        fig.subplots_adjust(top=0.95)

        fig.savefig(name+'_'+str(i+1)+'.png')
        close()

def plot_box(reality, estimates, name):
    reals = []
    estis = []
    for i in range(len(reality[0])):
        real = []
        esti = []
        for j in range(len(reality)):
            real.append(np.sum([reality[j][i][k] for k in range(len(reality[0][0]))]))
            esti.append(np.sum([estimates[j][i][k] for k in range(len(estimates[0][0]))]))
        reals.append(real)
        estis.append(esti)

    for i in range(len(reals)):
        fig = figure()

        ax1 = fig.gca()

        res = []
        res.append(reals[i])
        res.append(estis[i])

        bp = ax1.boxplot(res, notch=0, sym='b+', vert=1, whis=[1.5, 1.5],
                                 positions=None, widths=0.3, manage_xticks=True)

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

        fig.savefig(name+'_'+str(i+1)+'.png')
        close()

def plot_best(reality, estimates, name):
    reals = []
    estis = []
    for i in range(len(reality[0])):
        real = []
        esti = []
        for j in range(len(reality[0])):
            real.append(np.sum([reality[i][j][k] for k in range(len(reality[0][0]))]))
            esti.append(np.sum([estimates[i][j][k] for k in range(len(estimates[0][0]))]))
        reals.append(real)
        estis.append(esti)

    b_real = []
    b_esti = []
    for i in range(len(reals)):
        b_real.append(np.max(reals[i]))
        b_esti.append(np.max(estis[i]))
    fig = figure()

    ax1 = fig.gca()

    res = []
    res.append(b_real)
    res.append(b_esti)

    bp = ax1.boxplot(res, notch=0, sym='b+', vert=1, whis=[1.5, 1.5],
                             positions=None, widths=0.3, manage_xticks=True)

    beautify_boxplot(bp, ax1)


    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.get_xaxis().tick_bottom()
    ax1.get_yaxis().tick_left()
    ax1.tick_params(axis='x', direction='out')
    ax1.tick_params(axis='y', length=0)
    # ylim([0,12])

    ax1.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    ax1.set_axisbelow(True)

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

    fig.savefig(name+'.png')
    close()

dirs = ['ten', 'mean']
# pilco = ['../PILCO']

realities = {}
estimates = {}

for d in dirs:
    real, est = load_exps(d)
    print len(real)
    realities[d] = real
    estimates[d] = est
#     plot_fig(real, est, d)

#[real, esti] = load_pilco('../PILCO')
#print len(real)
# plot_fig(real, esti, 'pilco')

plot_comp(realities['ten'], realities['mean'], 'comp')
# plot_box(realities['ten'], estimates['ten'], 'box')
plot_best(realities['ten'], realities['mean'], 'best')
