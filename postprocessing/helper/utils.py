# Jonas Braun
# jonas.braun@tum.de
# 19.07.2019

import os
import time
import datetime
import numpy as np
import scipy.stats as sstats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#from PyPDF2 import PdfFileMerger

#import predictability.plotting.colors as clrs

INCH = 2.54


def cm2inch(cm):
    if isinstance(cm, tuple):
        return tuple(i/INCH for i in cm)
    else:
        return cm/INCH


def plot_mu_sem(mu, err, x=None, label="", alpha=0.3, color=None, ax=None):
    """
    plot mean and standard deviation, e.g. when plotting mean of predictability across neurons over lag
    :param mu: mean, shape [N_samples, N_lines] or [N_samples]
    :param err: error to be plotted, e.g. standard error of the mean, shape [N_samples, N_lines] or [N_samples]
    :param x: shape [N_samples]. If not specified will be np.arange(mu.shape[0])
    :param label: the label for each line either a string if only one line or list of strings if multiple lines
    :param alpha: transparency of the shaded area. default 0.3
    :param color: pre-specify colour. if None, use Python default colour cycle
    :param ax: axis to be plotted on, otherwise it will get the current axis with plt.gca()
    :return:
    """
    if ax is None:
        ax = plt.gca()
    if x is None:
        x = np.arange(mu.shape[0])
    p = ax.plot(x, mu, lw=1, label=label, color=color)
    if len(mu.shape) is 1:
        ax.fill_between(x, mu - err, mu + err, alpha=alpha, color=p[0].get_color())
    else:
        for i in np.arange(mu.shape[1]):
            ax.fill_between(x, mu[:, i] - err[:, i], mu[:, i] + err[:, i], alpha=alpha, color=p[i].get_color())


def beautify_plot(fig=None, ax=None, grid=False, legend=True, frame = True, ylabel="", xlabel="", title="",
              global_title=None, colorbar=None, colorbar_label="", tight=True, fontsize=16, axiswidth=1):
    """
    beautify plots with one command
    modified by Auguste Schulz 29.09.2019
    :param fontsize: fontsize of the text
    :param axiswidth: width of x and y axis
    :param tight: tight layout
    :param fig: figure to be applied on, otherwise plt.gcf()
    :param ax: axis to be plotted on, otherwise it will get the current axis with plt.gca()
    :param grid: True if grid to be switched on
    :param legend: True if legend is to be shown
    :param ylabel: y label
    :param xlabel: x label
    :param title: axis title
    :param global_title: figure title
    :param colorbar: should be a list of length 2 including beginning and end value of colourbar.
    :param colorbar_label: label of the colour bar
    :return:
    """
    # TODO: include function that recognises position of subplot in figure and plots xlabel and ylabel correspondingly
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    if grid:
        ax.grid()

    if legend:
        ax.legend(fontsize=fontsize, frameon = frame)

    if global_title is not None:
        fig.suptitle(global_title, fontsize=fontsize)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    # switch of top and right axis
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top', 'right']:
        ax.spines[axis].set_linewidth(0)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.locator_params(axis='both', nbins=4)
    if colorbar is not None:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        if isinstance(colorbar, list):
            assert len(colorbar) == 2, "len(colorbar) has to be 2. a beginning and a start value"
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=colorbar[0], vmax=colorbar[1]))
            sm._A = []
        else:
            raise NotImplementedError
            # TODO: either get colourmap from figure or include option to hand over
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(colorbar_label)
    elif tight:
        fig.tight_layout()



def plot_nice(fig=None, ax=None, grid=True, legend=True, ylabel="", xlabel="", title="", global_title=None,
              colorbar=None, colorbar_label="", tight=True, x0=True):
    """
    make a plot nicer in just one command
    :param fig: figure to be applied on, otherwise plt.gcf()
    :param ax: axis to be plotted on, otherwise it will get the current axis with plt.gca()
    :param grid: True if grid to be switched on
    :param legend: True if legend is to be shown
    :param ylabel: y label
    :param xlabel: x label
    :param title: axis title
    :param global_title: figure title
    :param colorbar: should be a list of length 2 including beginning and end value of colourbar.
    :param colorbar_label: label of the colour bar
    :return:
    """
    # TODO: include function that recognises position of subplot in figure and plots xlabel and ylabel correspondingly
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    if grid:
        ax.grid()

    if legend:
        ax.legend()

    if global_title is not None:
        fig.suptitle(global_title, fontsize=14)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if x0:
        ax.spines['left'].set_position('zero')

        # turn off the right spine/ticks
        ax.spines['right'].set_color('none')
        ax.yaxis.tick_left()

        # set the y-spine
        ax.spines['bottom'].set_position('zero')

        # turn off the top spine/ticks
        ax.spines['top'].set_color('none')
        ax.xaxis.tick_bottom()

    if colorbar is not None:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        if isinstance(colorbar, list):
            assert len(colorbar) == 2, "len(colorbar) has to be 2. a beginning and a start value"
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=colorbar[0], vmax=colorbar[1]))
            sm._A = []
        else:
            raise NotImplementedError
            # TODO: either get colourmap from figure or include option to hand over
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(colorbar_label)
    elif tight:
        fig.tight_layout()


def return_colors(sorted=True):
    """
    return a list of colours that look nicer than the python standard colours
    :type sorted: specify if list should be sorted i.e. different hues after oneanother otherwise first 4 shades of one
    colour
    """
    color = ["midnightblue", "lightskyblue", "royalblue", "lightsteelblue", "darkred", "darksalmon", "saddlebrown",
             "lightcoral", "darkgreen", "greenyellow", "darkolivegreen", "chartreuse", "darkmagenta", "thistle",
             "indigo", "mediumslateblue", "darkorange", "tan", "sienna", "orange"]
    if not sorted:
        return color
    else:
        return color[0::4] + color[1::4] + color[2::4] + color[3::4]


def save_fig(location="./", name="plot"):
    """
    save figure as .pdf and .png
    """
    plt.savefig(location + name + ".pdf")
    plt.savefig(location + name + ".png")


def sig_diff_plot(an, region, abbrv1, abbrv2, thres=[0.05, 0.01, 0.005],
                  plot=True, cbar=True, cshift=0.75, csize=0.5, fig=None, ax=None, title=None, global_title=None):
    stats = np.zeros((len(an.lags), len(an.sessions) + 1))
    signif = np.zeros_like(stats)
    for i_lag, lag in enumerate(an.lags):
        for i_s, session in enumerate(an.sessions):
            for i_test, (sign, string) in enumerate(zip([1, -1], ['greater', 'less'])):
                stats[i_lag, i_s], p = sstats.wilcoxon(
                    x=an.var_mean(abbrv=abbrv1, region=region, sessions=[session])[i_lag, :],
                    y=an.var_mean(abbrv=abbrv2, region=region, sessions=[session])[i_lag, :],
                    zero_method='wilcox', correction=False, alternative=string)  # 'two-sided')
                signif[i_lag, i_s] += sign * np.sum([p < t for t in thres])

        for i_test, (sign, string) in enumerate(zip([1, -1], ['greater', 'less'])):
            stats[i_lag, -1], p = sstats.wilcoxon(
                x=an.var_mean(abbrv=abbrv1, region=region, sessions=an.sessions)[i_lag, :],
                y=an.var_mean(abbrv=abbrv2, region=region, sessions=an.sessions)[i_lag, :],
                zero_method='wilcox', correction=False, alternative=string)  # 'two-sided')
            signif[i_lag, -1] += sign * np.sum([p < t for t in thres])

    if not plot:
        return signif

    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    plt.imshow(signif[an.lag_offset:, :].T, cmap="bwr", vmin=-len(thres), vmax=len(thres))  # "Greys")

    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(an.lags_sec[an.lag_offset:])
    ax.set_yticklabels(an.sessions + ["all"])
    ax.set_ylabel("sessions")
    ax.set_xlabel("lag (s)")

    ax.set_title(region + ": difference between " + abbrv1 + " and " + abbrv2 if title is None else title)
    if global_title is not None:
        fig.suptitle(global_title, fontsize=14)

    if not cbar:
        return signif

    fig.subplots_adjust(right=cshift)
    cbar_ax = fig.add_axes([cshift + 0.05, (1 - csize) / 2, 0.05, csize])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.bwr, norm=plt.Normalize(vmin=-len(thres), vmax=len(thres)))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cbar.ax.set_yticklabels(['p<0.001\n' + abbrv1 + '<' + abbrv2, 'p<0.01', 'p<0.05', 'n.s.',
                             'p<0.05', 'p<0.01', abbrv1 + '>' + abbrv2 + '\np<0.001'])

    return signif


def result_fig(myANs, names, abbrvs, size=(20.9, 29.7)):
    fig = plt.figure(figsize=cm2inch(size))
    fig.set_canvas(plt.gcf().canvas)
    fig.suptitle(' '.join(abbrvs), fontsize=14)
    i_max = len(myANs) * 2
    ax = plt.subplot(i_max, 3, 1)

    for i_r, region in enumerate(["CA1", "V1"]):
        for i, (an, name) in enumerate(zip(myANs, names)):
            row = 3 * i + 3 * len(myANs) * i_r

            # summary plot
            ax1 = plt.subplot(i_max, 3, 1 + row, sharex=ax, sharey=ax)
            for i_a, abbrv in enumerate(abbrvs):
                an.plot_mu_sem(abbrv=abbrv, region=region, sessions=an.sessions, color=clrs.get_color(abbrv))
            plot_nice(xlabel="lags (s)" if i_r == 1 and i == len(myANs) - 1 else None,
                      legend=True if i_r == 0 and i == 0 else False,
                      ylabel="$R^2$", title=region + " " + name, tight=False, grid=False)

            # each session plot
            ax2 = plt.subplot(i_max, 3, 2 + row, sharex=ax, sharey=ax)
            for i_a, abbrv in enumerate(abbrvs):
                for i_s, (session, alpha) in enumerate(zip(an.sessions, np.linspace(1, 0.25, len(an.sessions)))):
                    plt.plot(an.lags_sec[an.lag_offset:],
                             an.mu_n(abbrv=abbrv, region=region, sessions=[session])[an.lag_offset:],
                             color=clrs.get_color(abbrv), alpha=alpha, label=abbrv if i_s == 0 else "")
            plot_nice(xlabel="lags [s]" if i_r == 1 and i == len(myANs) - 1 else None,
                      tight=False, grid=False, legend=False,
                      title="sessions" if i_r == 0 and i == 0 else None)
            plt.setp(ax2.get_yticklabels(), visible=False)

            # significance plot
            ax3 = plt.subplot(i_max, 3, 3 + row)
            signif = sig_diff_plot(an, region, abbrvs[0], abbrvs[1], plot=False)
            plt.imshow(signif[an.lag_offset:, :].T, cmap="bwr", vmin=-3, vmax=3)
            ax3.set_xticks(np.arange(len(an.lags_sec) - an.lag_offset))
            ax3.set_yticks(np.arange(len(an.sessions) + 1))
            ax3.set_xticklabels(an.lags_sec[an.lag_offset:] if i_r == 1 and i == len(myANs) - 1
                                else [""] * (len(an.lags_sec) - an.lag_offset))
            ax3.set_yticklabels(an.sessions + ["all"])
            ax3.set_ylabel("sessions")
            ax3.set_xlabel("lag (s)" if i_r == 1 and i == len(myANs) - 1 else "")
            ax3.set_title("red: " + abbrvs[0] + ">" + abbrvs[1] if i_r == 0 and i == 0 else "")

    return fig


def save_results_inpdf(myANs, names, list_abbrvs, folder, filename="results", page_size=(30, 40)):
    # with PdfPages(os.path.join(folder,filename+'_'+\
    #                            datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M")+\
    #                            '.pdf')) as pdf:
    path = os.path.join(folder, filename+'_' +
                        datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M"))
    """
    print(path)
    pdf = PdfPages(path+'.pdf')
    for i, abbrvs in enumerate(list_abbrvs):
        fig = result_fig(myANs, names, abbrvs, page_size)
        pdf.savefig(fig)
        pdf.close()
    d = pdf.infodict()
    d['Title'] = 'Results of predictability analysis '+\
                 datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M")
    d['Author'] = 'Jonas Braun'
    d['Subject'] = 'plotting differences between different predictions' +\
                   'and different ways to calculate variance explained'
    d['CreationDate'] = datetime.datetime.today()
    d['ModDate'] = datetime.datetime.today()
    """
    merger = PdfFileMerger()

    for i, abbrvs in enumerate(list_abbrvs):
        fig = result_fig(myANs, names, abbrvs, page_size)
        plt.savefig(fname=path+"{}.pdf".format(i))
        plt.close(fig)
        merger.append(path+"{}.pdf".format(i))
    merger.write(path+".pdf")
    merger.close()
    [os.remove(path+"{}.pdf".format(i)) for i in range(len(list_abbrvs))]

