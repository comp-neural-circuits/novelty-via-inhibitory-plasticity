##############################################################################
#
# This code is part of the publication:
# https://www.biorxiv.org/content/10.1101/2020.11.30.403840v1
#
# The generation of cortical novelty responses through inhibitory plasticity
# Auguste Schulz*, Christoph Miehl*, Michael J. Berry II, Julijana Gjorgjieva
#
# * equal contribution
#
##############################################################################


from IPython.display import HTML, IFrame, Image

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os, fnmatch
import time
import h5py
from scipy.signal import find_peaks, peak_prominences
from scipy import stats
from scipy import optimize
import scipy.signal as ss
import scipy.ndimage.filters as filt

import matplotlib.colors as colors
#%matplotlib inline

# From Joe Kington
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
#norm=MidpointNormalize(midpoint=0.)

def save_fig(location = "./", name = "populationAverage"):
        """ save figure """
        plt.savefig(location + name + ".pdf")
        plt.savefig(location + name + ".png")

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
def count(list1, l, r):
    c = 0
    # traverse in the list1
    for x in list1:
        # condition check
        if x>= l and x<= r:
            c+= 1
    return c

def get_time_idx(time, onsets):
    idx = []
    for tt in onsets:
        #print(tt)
        #print(min(min(np.where(tt<=time))))
        idx.append(min(min(np.where(time>=tt))))

    return idx

def hp_filter(spikes, fc=0.0005, N=2):
    if fc == 0:
        return spikes
    else:
        fs = 60
        fn = fs/2
        Wn = fc/fn
        # b,a = ss.butter(N=N, Wn=Wn, btype="highpass")
        # return ss.filtfilt(b,a,spikes.astype(np.double))
        sos = ss.cheby2(N=N, rs=40, Wn=Wn, btype='highpass', output='sos')
        xtmp = np.flip(ss.sosfilt(sos, spikes.astype(np.double), axis=0), axis=0)
        return np.flip(ss.sosfilt(sos, xtmp, axis=0),  axis=0)


def lp_filter_gauss(spikes, sigma=600):
    if sigma > 0:
        return filt.gaussian_filter1d(spikes, sigma=sigma, axis=0, truncate=3.0)
    else:
        return spikes

# --------------------- finding peaks/ max val in time series -----------------
# determine peaks of novelty and peak at onset

def find_novelty_peak(time, trace, samplefraction = 0.8, height = 2, distance = 100, ifplot = True):
    # determine peaks of novelty
    nsamples = len(time)
    distance = nsamples
    lastsample = int(round(samplefraction*nsamples))
    peaks, _ = find_peaks(trace[lastsample:-1], height = height, distance = distance)
    peaks += lastsample
    if ifplot:
        plt.plot(time[0:-2], trace[0:-2])
        plt.plot(time[peaks], trace[peaks], "rx")
        #plt.plot(np.zeros_like(time) + height, "--", color="gray")
        plt.show()
    return time[peaks],trace[peaks], peaks

# determine peaks of novelty and peak at onset
def find_transient_peak(time, trace, samplefraction = 0.4, height = 2, distance = 100, ifplot = True):
    # find transient peak (at onset)
    nsamples = len(time)
    distance = nsamples
    lastsample = int(round(samplefraction*nsamples))
    peaks, _ = find_peaks(trace[0:lastsample], height = height, distance = distance)
    if ifplot:
        plt.plot(time[0:-2], trace[0:-2])
        plt.plot(time[peaks], trace[peaks], "rx")
        #plt.plot(np.zeros_like(time) + height, "--", color="gray")
        plt.show()
    return time[peaks],trace[peaks], peaks

def find_surround_max(idx, time, trace, margin = 2, ifplot = True):
    """ find the maximum of a trace within a margin around an index """
    idx_min = idx-int(margin)
    idx_max = idx+int(margin)
    # ensure the slices are within the length of the arrat
    if idx_min < 0:
        idx_min = 0
    if idx_max >= len(trace):
        idx_max = len(trace) - 1

    max_idx, max_val = maximum(trace[int(idx_min):int(idx_max)])
    peaks = int(idx_min + max_idx)
    if ifplot:
        plt.plot(time[0:-2], trace[0:-2])
        plt.plot(time[peaks], trace[peaks], "rx")
        #plt.plot(np.zeros_like(time) + height, "--", color="gray")
        plt.show()
    return time[peaks],trace[peaks], int(peaks)


def maximum(values):
    """ get maximum index and value """
    max_index = np.argmax(values)
    #max_value = values[max_index]
    return int(max_index) , values[max_index]

# -------------------------- plotting functions ------------------------------
def plot_popavg(time, data, legend = "E", iflegend = False, color = "blue", ifcolor = False, lw = 3,fontsize = 20, xlabel = "time [s]", ylabel ="$\\rho$ [Hz]", ifioff = True, axiswidth = 1):
        """ plot population average one figure a time"""
        if ifioff:
            plt.ioff()
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        if iflegend:
            if ifcolor:
                plt.plot(time[0:-2], data[0:-2], label = legend, color = color, lw = lw)
                plt.xlabel(xlabel, fontsize = fontsize)
                plt.ylabel(ylabel, fontsize = fontsize)
                plt.xticks(fontsize = fontsize)
                plt.yticks(fontsize = fontsize)
                plt.tight_layout()
                legend(fontsize = fontsize, frameon = False)
            else:
                plt.plot(time[0:-2], data[0:-2], label = legend, lw = lw)
                plt.xlabel(xlabel, fontsize = fontsize)
                plt.ylabel(ylabel, fontsize = fontsize)
                plt.legend(fontsize = fontsize, frameon = False)
                plt.xticks(fontsize = fontsize)
                plt.yticks(fontsize = fontsize)
                plt.tight_layout()

        else:
            if ifcolor:
                plt.plot(time[0:-2], data[0:-2], color = color, lw = lw)
                plt.xlabel(xlabel, fontsize = fontsize)
                plt.ylabel(ylabel, fontsize = fontsize)
                plt.xticks(fontsize = fontsize)
                plt.yticks(fontsize = fontsize)
                plt.tight_layout()
            else:
                plt.plot(time[0:-2], data[0:-2], lw = lw)
                plt.xlabel(xlabel, fontsize = fontsize)
                plt.ylabel(ylabel, fontsize = fontsize)
                plt.xticks(fontsize = fontsize)
                plt.yticks(fontsize = fontsize)
                plt.tight_layout()

def plot_popavg_mult(fig,time, data, legend = "E", iflegend = False, color = "blue", ifcolor = False, lw = 3,fontsize = 20, xlabel = "time [s]", ylabel ="$\\rho$ [Hz]", ifioff = True, axiswidth = 1, ncol = 1, alpha = 1):
        """ plot population average in one figure fig (def before)"""
        #fig = plt.figure(figsize=(20, 10)) #run first and then conse
        if ifioff:
            plt.ioff()

        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        if iflegend:
            if ifcolor:
                plt.plot(time[0:-2], data[0:-2], label = legend, color = color, lw = lw, alpha = alpha)
                plt.xlabel(xlabel, fontsize = fontsize)
                plt.ylabel(ylabel, fontsize = fontsize)
                plt.xticks(fontsize = fontsize)
                plt.yticks(fontsize = fontsize)
                plt.legend(fontsize = fontsize, frameon = False, ncol = ncol)
                plt.tight_layout()

            else:
                plt.plot(time[0:-2], data[0:-2], label = legend, lw = lw, alpha = alpha)
                plt.xlabel(xlabel, fontsize = fontsize)
                plt.ylabel(ylabel, fontsize = fontsize)
                plt.legend(fontsize = fontsize, frameon = False, ncol = ncol)
                plt.xticks(fontsize = fontsize)
                plt.yticks(fontsize = fontsize)
                plt.tight_layout()
        else:
            if ifcolor:
                plt.plot(time[0:-2], data[0:-2], color = color, lw = lw, alpha = alpha)
                plt.xlabel(xlabel, fontsize = fontsize)
                plt.ylabel(ylabel, fontsize = fontsize)
                plt.xticks(fontsize = fontsize)
                plt.yticks(fontsize = fontsize)
                plt.tight_layout()
            else:
                plt.plot(time[0:-2], data[0:-2], lw = lw, alpha = alpha)
                plt.xlabel(xlabel, fontsize = fontsize)
                plt.ylabel(ylabel, fontsize = fontsize)
                plt.xticks(fontsize = fontsize)
                plt.yticks(fontsize = fontsize)
                plt.tight_layout()

def plot_mean_with_errorband_mult(fig,time, data, error, noveltyonset = 22.5, legend = "E", iflegend = False, color = "darkblue", ifcolor = False,
                 lw = 3, xlabel = "time [s]", ylabel ="z-score",fontsize = 24, ifxpositions = False, ifaxvline = False, x_positions = [5,10,15,20], x_labels = ["5","10","15","20"],
                              ifioff = True, alpha=0.2, axiswidth = 1):
    """ plot the mean with +- std in one figure fig (def before)"""
    #fig = plt.figure(figsize=(20, 10)) #run first and then conse
    if ifioff:
        plt.ioff()

    ax = fig.add_subplot(111)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    ifcolor = True
    if iflegend:
        if ifcolor:
            plt.plot(time, data, label = legend, color = color, lw = lw)
            plt.fill_between(time, data-error, data+error,alpha=alpha, edgecolor=color, facecolor=color)
            plt.xlabel(xlabel, fontsize = fontsize)
            plt.ylabel(ylabel, fontsize = fontsize)
            plt.xticks(fontsize = fontsize)
            plt.yticks(fontsize = fontsize)
            plt.legend(fontsize = fontsize, frameon = False)
            plt.tight_layout()
        else:
            plt.plot(time, data, label = legend, color = color, lw = lw)
            plt.fill_between(time, data-error, data+error,alpha=alpha, edgecolor=color, facecolor=color)
            plt.xlabel(xlabel, fontsize = fontsize)
            plt.ylabel(ylabel, fontsize = fontsize)
            plt.legend(fontsize = fontsize, frameon = False)
            plt.xticks(fontsize = fontsize)
            plt.yticks(fontsize = fontsize)
            plt.tight_layout()
    else:
        if ifcolor:
            plt.plot(time, data, label = legend, color = color, lw = lw)
            plt.fill_between(time, data-error, data+error,alpha=alpha, edgecolor=color, facecolor=color)
            plt.xlabel(xlabel, fontsize = fontsize)
            plt.ylabel(ylabel, fontsize = fontsize)
            plt.xticks(fontsize = fontsize)
            plt.yticks(fontsize = fontsize)
            plt.tight_layout()
        else:
            plt.plot(time, data, label = legend, color = color, lw = lw)
            plt.fill_between(time, data-error, data+error,alpha=alpha, edgecolor=color, facecolor=color)
            plt.xlabel(xlabel, fontsize = fontsize)
            plt.ylabel(ylabel, fontsize = fontsize)
            plt.xticks(fontsize = fontsize)
            plt.yticks(fontsize = fontsize)
            plt.tight_layout()
    if ifxpositions:
        plt.xticks(x_positions, x_labels,fontsize = fontsize)
    if ifaxvline:
        ax.axvline(x=noveltyonset,color ="k")#,lw = 3, "k")
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)

def plot_array(xaxis, data, legend = "E", iflegend = False, color = "darkred", figsize=(10, 6),ifExcitatory = True,  xticks = [], ifxticks = False, ifcolor = False, lw = 3,fontsize = 20, xlabel = "cell rank", ylabel ="rate [Hz]", ifioff = True, axiswidth = 1):
        """ plot population average one figure a time"""
        if ifioff:
            plt.ioff()
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        if ifExcitatory:
            color = "darkblue"
        if iflegend:
            if ifcolor:
                plt.plot(xaxis, data, label = legend, color = color, lw = lw)
                plt.xlabel(xlabel, fontsize = fontsize)
                plt.ylabel(ylabel, fontsize = fontsize)

                if ifxticks:
                    plt.xticks(xticks, fontsize = fontsize)
                else:
                    plt.xticks(fontsize = fontsize)

                plt.yticks(fontsize = fontsize)
                plt.legend(fontsize = fontsize, frameon = False)
                plt.tight_layout()
            else:
                plt.plot(xaxis, data, label = legend, lw = lw)
                plt.xlabel(xlabel, fontsize = fontsize)
                plt.ylabel(ylabel, fontsize = fontsize)
                plt.legend(fontsize = fontsize, frameon = False)
                if ifxticks:
                    plt.xticks(xticks,fontsize = fontsize)
                else:
                    plt.xticks(fontsize = fontsize)
                plt.yticks(fontsize = fontsize)
                plt.tight_layout()
        else:
            if ifcolor:
                plt.plot(xaxis, data, color = color, lw = lw)
                plt.xlabel(xlabel, fontsize = fontsize)
                plt.ylabel(ylabel, fontsize = fontsize)
                if ifxticks:
                    plt.xticks(xticks,fontsize = fontsize)
                else:
                    plt.xticks(fontsize = fontsize)
                plt.yticks(fontsize = fontsize)
                plt.tight_layout()
            else:
                plt.plot(xaxis, data, lw = lw)
                plt.tight_layout()
                plt.xlabel(xlabel, fontsize = fontsize)
                plt.ylabel(ylabel, fontsize = fontsize)
                if ifxticks:
                    plt.xticks(xticks,fontsize = fontsize)
                else:
                    plt.xticks(fontsize = fontsize)
                plt.yticks(fontsize = fontsize)
                plt.tight_layout()


def plot_array_mult(fig, xaxis, data, legend = "E", iflegend = False, color = "darkred", figsize=(10, 6),ifExcitatory = True, ncol = 1, xticks = [], ifxticks = False, ifcolor = False, lw = 3,fontsize = 20, xlabel = "cell rank", ylabel ="rate [Hz]", ifioff = True, axiswidth = 1):
        """ plot population average one figure a time"""
        if ifioff:
            plt.ioff()
        #fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        if ifExcitatory:
            color = "darkblue"
        if iflegend:
            if ifcolor:
                plt.plot(xaxis, data, label = legend, color = color, lw = lw)
                plt.xlabel(xlabel, fontsize = fontsize)
                plt.ylabel(ylabel, fontsize = fontsize)

                if ifxticks:
                    plt.xticks(xticks, fontsize = fontsize)
                else:
                    plt.xticks(fontsize = fontsize)

                plt.yticks(fontsize = fontsize)
                plt.legend(fontsize = fontsize, frameon = False, ncol=ncol)
                plt.tight_layout()
            else:
                plt.plot(xaxis, data, label = legend, lw = lw)
                plt.xlabel(xlabel, fontsize = fontsize)
                plt.ylabel(ylabel, fontsize = fontsize)
                plt.legend(fontsize = fontsize, frameon = False,ncol=ncol)
                if ifxticks:
                    plt.xticks(xticks,fontsize = fontsize)
                else:
                    plt.xticks(fontsize = fontsize)
                plt.yticks(fontsize = fontsize)
                plt.tight_layout()
        else:
            if ifcolor:
                plt.plot(xaxis, data, color = color, lw = lw)
                plt.xlabel(xlabel, fontsize = fontsize)
                plt.ylabel(ylabel, fontsize = fontsize)
                if ifxticks:
                    plt.xticks(xticks,fontsize = fontsize)
                else:
                    plt.xticks(fontsize = fontsize)
                plt.yticks(fontsize = fontsize)
                plt.tight_layout()
            else:
                plt.plot(xaxis, data, lw = lw)
                plt.xlabel(xlabel, fontsize = fontsize)
                plt.ylabel(ylabel, fontsize = fontsize)
                if ifxticks:
                    plt.xticks(xticks,fontsize = fontsize)
                else:
                    plt.xticks(fontsize = fontsize)
                plt.yticks(fontsize = fontsize)
                plt.tight_layout()

def plot_peak_overlap_seq(x, y, Nseq = 5, iflegend = False, colorseq = ["r","g","b","k","c","m"],figsize=(15,12),
                     lw = 3, xlabel = "overlap with previous sequence [%]", ylabel ="novelty peak rate [Hz]",
                          fontsize = 24,ifioff = False, ifsavefig = True, savehandle = "E", Nimg = 4, figure_directory = "./",ifyticks = False, yticks = [3,4], axiswidth = 1):
    """ plot novelty/transient peakheight versus overlap with previous sequence"""
    if ifioff:
        plt.ioff()
#novelty_overlap[seq-1,:,0]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    for seq in range(1,Nseq + 1):
        plt.plot(x[seq-1][:]*100, y[seq-1][:], "o", label = str(seq), color = colorseq[seq-1], markersize = 20)
    if iflegend:
        plt.legend(fontsize = fontsize)
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    plt.tight_layout()

    if ifyticks:
        plt.yticks(yticks)

    savetitle = "Overlap_" + savehandle
    if ifsavefig:
        save_fig(figure_directory, savetitle)

# --------------------------------------------- weight evolution ---------------------

def plotweightmatrix(weights, Nmax = 4000,Nmin = 1, maxval = 22, fontsize = 24, ifcolorbar = True):
    plt.figure(figsize=(14, 10))
    plt.imshow(weights[Nmin:Nmax,Nmin:Nmax], origin="lower", cmap="bone_r", vmin = 0, vmax = maxval)
    plt.xticks([],fontsize = fontsize)
    plt.yticks([],fontsize = fontsize)
    plt.ylabel("presynaptic neuron",fontsize = fontsize)
    plt.xlabel("postsynaptic neuron",fontsize = fontsize)
    if ifcolorbar:
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=fontsize)
        cb.set_label("$w$ [pF]", fontsize = fontsize)
    plt.tight_layout()


def plotavgweightmatrix(avgweights, maxval = 22, fontsize = 24, ifcolorbar = True):
    plt.figure(figsize=(14, 10))
    plt.imshow(avgweights, origin="lower", cmap="bone_r", vmin = 0, vmax = maxval)
    plt.xticks([],fontsize = fontsize)
    plt.yticks([],fontsize = fontsize)
    plt.ylabel("presynaptic assembly",fontsize = fontsize)
    plt.xlabel("postsynaptic assembly",fontsize = fontsize)
    if ifcolorbar:
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=fontsize)
        cb.set_label(r"$ \bar{w}$ [pF]", fontsize = fontsize)
    plt.tight_layout()

# ------------------------------------ rate planes -------------------------------
def getaveragerateplane(matrix):
    # average across all blocks
    # input: maxtrix (sequences, blocks, Ncells, timepoints)
    # output: matrix (sequences, Ncells, timepoints)
    return np.mean(matrix,axis = 1)

def plotaveragerateplane(timevector,blockavg, idxstart = 0, figure_directory = "./", Ne = 4000, cmap = "YlGnBu_r", xlabel = "time [s]",
                         ylabel = "neuron index",Nseq = 5,
                         fontsize = 24, x_positions = [50,100,150,200], x_labels = ["5","10","15","20"], cbarlabel = r"$ \nu $ [Hz]",
                         savetitle = "firing_rates_averaged", ifExcitatory = True, origin="lower", ififnorm = False, midpoint=0.):
    # input: matrix (sequences, Ncells, timepoints)
    # output: planes for each sequence
    if ifExcitatory:
        savetitle = "Excitatory_" + savetitle
        ylabel = "excitatory " + ylabel
    else:
        if cmap == "YlGnBu_r":
            cmap = "YlOrRd_r"
        savetitle = "Inhibitory_" + savetitle
        ylabel = "inhibitory " + ylabel
    for seq in range(1,Nseq + 1):
        plt.figure(figsize=(15,12))
        #norm=MidpointNormalize(midpoint=midpoint),
        if ififnorm:
            if ifExcitatory:
                plt.imshow(blockavg[seq-1,0:Ne,idxstart:], aspect='auto', norm=MidpointNormalize(midpoint=midpoint),cmap = cmap, origin=origin)#np.linspace(0,24,240000)/10000,
            else:
                plt.imshow(blockavg[seq-1,Ne:,idxstart:], aspect='auto', norm=MidpointNormalize(midpoint=midpoint),cmap = cmap,origin=origin)#np.linspace(0,24,240000)/10000,
        else:
            if ifExcitatory:
                plt.imshow(blockavg[seq-1,0:Ne,idxstart:], aspect='auto', cmap = cmap, origin=origin)#np.linspace(0,24,240000)/10000,
            else:
                plt.imshow(blockavg[seq-1,Ne:,idxstart:], aspect='auto', cmap = cmap,origin=origin)#np.linspace(0,24,240000)/10000,

        plt.xlabel(xlabel, fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)
        #plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.xticks(x_positions, x_labels,fontsize = fontsize)
        #plt.xticks(fontsize = fontsize)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=fontsize)
        cb.set_label(cbarlabel, fontsize = fontsize)
        plt.tight_layout()

        save_fig(figure_directory, savetitle + "Seq%d"% (seq-1))
# f, (a0, a1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]}, sharex=True)
# a0.plot(x,y)
# a1.plot(y,x)

def plotaveragerateplanewithavg(timevector, blockavg, idxstart = 0, figure_directory = "./", Ne = 4000, cmap = "YlGnBu_r", xlabel = "time [s]", ylabel = "neuron index",
                         fontsize = 24, yminE = 2.5, yminI = 2.8, x_positions = [50,100,150,200], x_labels = ["5","10","15","20"], cbarlabel = r"$ \nu $ [Hz]",
                         avglabel = r"$ \rho $ [Hz]", savetitle = "firing_rates_averaged", ifExcitatory = True, origin="lower",Nseq = 5,ififnorm = False, midpoint=0.):
    # input: matrix (sequences, Ncells, timepoints)
    # output: planes for each sequence
    if ifExcitatory:
        savetitle = "Excitatory_" + savetitle
        ylabel = "excitatory " + ylabel
    else:
        if cmap == "YlGnBu_r":
            cmap = "YlOrRd_r"
        savetitle = "Inhibitory_" + savetitle
        ylabel = "inhibitory " + ylabel
    for seq in range(1,Nseq + 1):
        #plt.figure(figsize=(15,12))
        f, (a0, a1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]}, sharex=True, figsize=(16,15))
        f.subplots_adjust(hspace=0.1)
        if ifExcitatory:
            im = a0.imshow(blockavg[seq-1,0:Ne,idxstart:], aspect='auto', cmap = cmap, origin=origin)#np.linspace(0,24,240000)/10000,
            a1.plot(np.mean(blockavg[seq-1,0:Ne,idxstart:], axis=0),  color = "darkblue", lw = 3)
            yminE = min(np.mean(blockavg[seq-1,0:Ne,idxstart:], axis=0))-0.05
            a1.set_ylim(ymin=yminE)

        else:
            im = a0.imshow(blockavg[seq-1,Ne:,idxstart:], aspect='auto', cmap = cmap,origin=origin)#np.linspace(0,24,240000)/10000,
            a1.plot(np.mean(blockavg[seq-1,Ne:,idxstart:], axis=0),  color = "darkred", lw = 3)
            yminI = min(np.mean(blockavg[seq-1,Ne:,idxstart:], axis=0))-0.05
            a1.set_ylim(ymin=yminI)

        a0.set_xlabel(" ",fontsize = 1)
        a0.set_ylabel(ylabel,fontsize =fontsize)
        a1.set_xlabel(xlabel,fontsize =fontsize)
        a1.set_ylabel(avglabel,fontsize =fontsize)
        a0.tick_params(labelsize=fontsize)
        a1.tick_params(labelsize=fontsize)
        xlim = np.size(blockavg[seq-1,0:Ne,idxstart:],axis = 1)
        a1.set_xlim([0, xlim-1])
        #a1.set_ylim([2.5,5])
        a0.set_xlim([0, xlim-1])
        plt.yticks(fontsize = fontsize)
        plt.xticks(x_positions, x_labels,fontsize = fontsize)
        #plt.xticks(fontsize = fontsize)
        # get first y axis positions
        b = a0.get_position()
        points = b.get_points()
        #print(points)

        for axis in ['bottom','left']:
            a1.spines[axis].set_linewidth(1.5)
#             a0.spines[axis].set_linewidth(1.5)
#             cbar_ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            a1.spines[axis].set_linewidth(0)
#             a0.spines[axis].set_linewidth(1.5)
#             cbar_ax.spines[axis].set_linewidth(axiswidth)
        if len(x_positions) == 4:
            f.subplots_adjust(right=0.8)
            #[x0, y0, width, height]
            cbar_ax = f.add_axes([0.82, b.get_points()[0,1], 0.03, b.get_points()[1,1]])#b.get_points()[1,1]-b.get_points()[0,1]])
        else:
            f.subplots_adjust(right=0.9)
            cbar_ax = f.add_axes([0.91, b.get_points()[0,1], 0.025, b.get_points()[1,1]-b.get_points()[0,1]])

        f.colorbar(im, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=fontsize)
        cbar_ax.set_ylabel(cbarlabel, fontsize = fontsize)
        #cbar_ax.spines
         # adjust axis width
        #plt.tight_layout()

        save_fig(figure_directory, savetitle + "WithAvgRateSeq%d"% (seq))

def plotaveragerateplanewithavgnew(timevector,blockavg, idxstart = 0, figure_directory = "./", Ne = 4000, cmap = "YlGnBu_r", xlabel = "time [s]", ylabel = "neuron index",
                         fontsize = 24, yminE = 2.5, yminI = 2.8, Nseq = 5, x_positions = [50,100,150,200], x_labels = ["5","10","15","20"], cbarlabel = r"$ \nu $ [Hz]",
                         avglabel = r"$ \rho $ [Hz]", savetitle = "firing_rates_averaged", ifExcitatory = True, origin="lower",ififnorm = False, midpoint=0.):
    # input: matrix (sequences, Ncells, timepoints)
    # output: planes for each sequence
    if ifExcitatory:
        savetitle = "Excitatory_" + savetitle
        ylabel = "excitatory " + ylabel
    else:
        if cmap == "YlGnBu_r":
            cmap = "YlOrRd_r"
        savetitle = "Inhibitory_" + savetitle
        ylabel = "inhibitory " + ylabel
    for seq in range(1,Nseq + 1):
        #plt.figure(figsize=(15,12))
        f, (a0, a1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]}, sharex=True, figsize=(15,15))
        f.subplots_adjust(hspace=0.1)
        if ifExcitatory:
            im = a0.imshow(blockavg[seq-1,0:Ne,idxstart:], aspect='auto', cmap = cmap, origin=origin)#np.linspace(0,24,240000)/10000,
            a1.plot(np.mean(blockavg[seq-1,0:Ne,idxstart:], axis=0),  color = "darkblue", lw = 3)
            yminE = min(np.mean(blockavg[seq-1,0:Ne,idxstart:], axis=0))-0.05
            a1.set_ylim(ymin=yminE)

        else:
            im = a0.imshow(blockavg[seq-1,Ne:,idxstart:], aspect='auto', cmap = cmap,origin=origin)#np.linspace(0,24,240000)/10000,
            a1.plot(np.mean(blockavg[seq-1,Ne:,idxstart:], axis=0),  color = "darkred", lw = 3)
            yminI = min(np.mean(blockavg[seq-1,Ne:,idxstart:], axis=0))-0.05
            a1.set_ylim(ymin=yminI)

        a0.set_xlabel(" ",fontsize = 1)
        a0.set_ylabel(ylabel,fontsize =fontsize)
        a1.set_xlabel(xlabel,fontsize =fontsize)
        a1.set_ylabel(avglabel,fontsize =fontsize)
        a0.tick_params(labelsize=fontsize)
        a1.tick_params(labelsize=fontsize)
        xlim = np.size(blockavg[seq-1,0:Ne,idxstart:],axis = 1)
        a1.set_xlim([0, xlim-1])
        #a1.set_ylim([2.5,5])
        a0.set_xlim([0, xlim-1])
        plt.yticks(fontsize = fontsize)
        plt.xticks(x_positions, x_labels,fontsize = fontsize)
        #plt.xticks(fontsize = fontsize)

        # get first y axis positions
        b = a0.get_position()
        points = b.get_points()
        print(points)

        for axis in ['bottom','left']:
            a1.spines[axis].set_linewidth(1.5)
#             a0.spines[axis].set_linewidth(1.5)
#             cbar_ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            a1.spines[axis].set_linewidth(0)
#             a0.spines[axis].set_linewidth(1.5)
#             cbar_ax.spines[axis].set_linewidth(axiswidth)
        if len(x_positions) == 4:
            f.subplots_adjust(right=0.8)
            #[x0, y0, width, height]
            cbar_ax = f.add_axes([0.82, b.get_points()[0,1], 0.03, b.get_points()[1,1]])#b.get_points()[1,1]-b.get_points()[0,1]])
        else:
            f.subplots_adjust(right=1)
            cbar_ax = f.add_axes([1.02, b.get_points()[0,1], 0.03, b.get_points()[1,1]-b.get_points()[0,1]])

        f.colorbar(im, cax=cbar_ax)
        cbar_ax.tick_params(labelsize=fontsize)
        cbar_ax.set_ylabel(cbarlabel, fontsize = fontsize)
        #cbar_ax.spines
         # adjust axis width
        #plt.tight_layout()

        save_fig(figure_directory, savetitle + "WithAvgRateSeq%d"% (seq))


def plotrateplane(timevector, block, seq, bl, idxstart = 0, figure_directory = "./", Ne = 4000, cmap = "YlGnBu_r", xlabel = "time [s]", ylabel = "neuron index",
                         fontsize = 24, x_positions = [50,100,150,200], x_labels = ["5","10","15","20"], cbarlabel = r"$ \nu $ [Hz]",
                         savetitle = "firing_rates_", ifExcitatory = True, origin="lower", ififnorm = False,midpoint=0.,Nseq = 5):
    # input: matrix (sequences, blocks, Ncells, timepoints)
    # output: planes for selected sequence or block
    # also possible for zscore plots
    if ifExcitatory:
        savetitle = "Excitatory_" + savetitle
        ylabel = "excitatory " + ylabel
    else:
        if cmap == "YlGnBu_r":
            cmap = "YlOrRd_r"
        savetitle = "Inhibitory_" + savetitle
        ylabel = "inhibitory " + ylabel

    plt.figure(figsize=(15,12))
    if ififnorm:
        if ifExcitatory:
            plt.imshow(block[seq-1,bl-1,0:Ne,idxstart:], aspect='auto', norm=MidpointNormalize(midpoint=midpoint), cmap = cmap, origin=origin)#np.linspace(0,24,240000)/10000,
        else:
            plt.imshow(block[seq-1,bl-1,Ne:,idxstart:], aspect='auto', norm=MidpointNormalize(midpoint=midpoint), cmap = cmap,origin=origin)#np.linspace(0,24,240000)/10000,
    else:
        if ifExcitatory:
            plt.imshow(block[seq-1,bl-1,0:Ne,idxstart:], aspect='auto', cmap = cmap, origin=origin)#np.linspace(0,24,240000)/10000,
        else:
            plt.imshow(block[seq-1,bl-1,Ne:,idxstart:], aspect='auto', cmap = cmap,origin=origin)#np.linspace(0,24,240000)/10000,

    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    #plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.xticks(x_positions, x_labels,fontsize = fontsize)
    #plt.xticks(fontsize = fontsize)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label(cbarlabel, fontsize = fontsize)
    plt.tight_layout()

    save_fig(figure_directory, savetitle + "Seq%dBlock%d"% (seq,bl))



def plotrateplanewithavg(timevector,block, seq, bl, idxstart = 0, figure_directory = "./", Ne = 4000, cmap = "YlGnBu_r", xlabel = "time [s]", ylabel = "neuron index",
                         fontsize = 24, x_positions = [50,100,150,200],yminE=2.8,yminI=2.8, ylimbar = 400, ifylimbar = False,  x_labels = ["5","10","15","20"], cbarlabel = r"$ \nu $ [Hz]",
                         avglabel = r"$ \rho $ [Hz]", savetitle = "firing_rates_WithAvgRate", ifExcitatory = True, origin="lower", Nseq = 5,ififnorm = False, midpoint=0.):
    # input: matrix (sequences, Ncells, timepoints)
    # output: planes for each sequence
    if ifExcitatory:
        savetitle = "Excitatory_" + savetitle
        ylabel = "excitatory " + ylabel
    else:
        if cmap == "YlGnBu_r":
            cmap = "YlOrRd_r"
        savetitle = "Inhibitory_" + savetitle
        ylabel = "inhibitory " + ylabel
        #plt.figure(figsize=(15,12))
    f, (a0, a1) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]}, sharex=True, figsize=(16,15))
    f.subplots_adjust(hspace=0.1)
    # if ifExcitatory:
    #     im = a0.imshow(block[seq-1,bl-1,0:Ne,idxstart:], aspect='auto', cmap = cmap, origin=origin)#np.linspace(0,24,240000)/10000,
    #     a1.plot(np.mean(block[seq-1,bl-1,0:Ne,idxstart:], axis=0),  color = "darkblue", lw = 3)
    #     yminE = min(np.mean(block[seq-1,bl-1,0:Ne,idxstart:], axis=0))-0.05
    #     a1.set_ylim(ymin=yminE)
    #
    # else:
    #     im = a0.imshow(block[seq-1,bl-1,Ne:,idxstart:], aspect='auto', cmap = cmap,origin=origin)#np.linspace(0,24,240000)/10000,
    #     a1.plot(np.mean(block[seq-1,bl-1,Ne:,idxstart:], axis=0),  color = "darkred", lw = 3)
    #     yminI = min(np.mean(block[seq-1,bl-1,Ne:,idxstart:], axis=0))-0.05
    #     a1.set_ylim(ymin=yminI)
    if ififnorm:
        if ifExcitatory:
            im = a0.imshow(block[seq-1,bl-1,0:Ne,idxstart:], aspect='auto', cmap = cmap, norm=MidpointNormalize(midpoint=midpoint),origin=origin)#np.linspace(0,24,240000)/10000,
            a1.plot(np.mean(block[seq-1,bl-1,0:Ne,idxstart:], axis=0),  color = "darkblue", lw = 3)
            yminE = min(np.mean(block[seq-1,bl-1,0:Ne,idxstart:], axis=0))-0.05
            a1.set_ylim(ymin=yminE)

        else:
            im = a0.imshow(block[seq-1,bl-1,Ne:,idxstart:], aspect='auto', cmap = cmap,norm=MidpointNormalize(midpoint=midpoint),origin=origin)#np.linspace(0,24,240000)/10000,
            a1.plot(np.mean(block[seq-1,bl-1,Ne:,idxstart:], axis=0),  color = "darkred", lw = 3)
            yminI = min(np.mean(block[seq-1,bl-1,Ne:,idxstart:], axis=0))-0.05
            a1.set_ylim(ymin=yminI)
        # if ifExcitatory:
        #     plt.imshow(block[seq-1,bl-1,0:Ne,idxstart:], aspect='auto', norm=MidpointNormalize(midpoint=midpoint), cmap = cmap, origin=origin)#np.linspace(0,24,240000)/10000,
        # else:
        #     plt.imshow(block[seq-1,bl-1,Ne:,idxstart:], aspect='auto', norm=MidpointNormalize(midpoint=midpoint), cmap = cmap,origin=origin)#np.linspace(0,24,240000)/10000,
    else:
        if ifExcitatory:
            if ifylimbar:
                im = a0.imshow(block[seq-1,bl-1,0:Ne,idxstart:], aspect='auto', cmap = cmap, origin=origin, vmax= ylimbar)#np.linspace(0,24,240000)/10000,
            else:
                im = a0.imshow(block[seq-1,bl-1,0:Ne,idxstart:], aspect='auto', cmap = cmap, origin=origin)#np.linspace(0,24,240000)/10000,

            a1.plot(np.mean(block[seq-1,bl-1,0:Ne,idxstart:], axis=0),  color = "darkblue", lw = 3)
            yminE = min(np.mean(block[seq-1,bl-1,0:Ne,idxstart:], axis=0))-0.05
            a1.set_ylim(ymin=yminE)

        else:
            if ifylimbar:
                im = a0.imshow(block[seq-1,bl-1,Ne:,idxstart:], aspect='auto', cmap = cmap, origin=origin, vmax= ylimbar)#np.linspace(0,24,240000)/10000,
            else:
                im = a0.imshow(block[seq-1,bl-1,Ne:,idxstart:], aspect='auto', cmap = cmap, origin=origin)#np.linspace(0,24,240000)/10000,

            #im = a0.imshow(block[seq-1,bl-1,Ne:,idxstart:], aspect='auto', cmap = cmap,origin=origin)#np.linspace(0,24,240000)/10000,
            a1.plot(np.mean(block[seq-1,bl-1,Ne:,idxstart:], axis=0),  color = "darkred", lw = 3)
            yminI = min(np.mean(block[seq-1,bl-1,Ne:,idxstart:], axis=0))-0.05
            a1.set_ylim(ymin=yminI)
        # if ifExcitatory:
        #     plt.imshow(block[seq-1,bl-1,0:Ne,idxstart:], aspect='auto', cmap = cmap, origin=origin)#np.linspace(0,24,240000)/10000,
        # else:
        #     plt.imshow(block[seq-1,bl-1,Ne:,idxstart:], aspect='auto', cmap = cmap,origin=origin)#np.linspace(0,24,240000)/10000,



    a0.set_xlabel(" ",fontsize = 1)
    a0.set_ylabel(ylabel,fontsize =fontsize)
    a1.set_xlabel(xlabel,fontsize =fontsize)
    a1.set_ylabel(avglabel,fontsize =fontsize)
    a0.tick_params(labelsize=fontsize, )
    a1.tick_params(labelsize=fontsize)
    xlimmax = np.size(block[seq-1,bl-1,Ne:,idxstart:],axis = 1) - 1
    a1.set_xlim([0, xlimmax])
    #a1.set_ylim([2.5,5])
    a0.set_xlim([0, xlimmax])
    plt.yticks(fontsize = fontsize)
    plt.xticks(x_positions, x_labels,fontsize = fontsize)
    #plt.xticks(fontsize = fontsize)

    # get first y axis positions
    b = a0.get_position()
    points = b.get_points()
    if len(x_positions) == 4:
        f.subplots_adjust(right=0.8)
        #[x0, y0, width, height]
        cbar_ax = f.add_axes([0.82, b.get_points()[0,1], 0.03, b.get_points()[1,1]-b.get_points()[0,1]])#b.get_points()[1,1]-b.get_points()[0,1]])
    else:
        f.subplots_adjust(right=0.9)
        cbar_ax = f.add_axes([0.91, b.get_points()[0,1], 0.025, b.get_points()[1,1]-b.get_points()[0,1]])

    # f.subplots_adjust(right=0.8)
    # cbar_ax = f.add_axes([0.82, b.get_points()[0,1], 0.03, b.get_points()[1,1]-b.get_points()[0,1]])
    for axis in ['bottom','left']:
        a1.spines[axis].set_linewidth(1.5)
#             a0.spines[axis].set_linewidth(1.5)
#             cbar_ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top','right']:
        a1.spines[axis].set_linewidth(0)
#             a0.spines[axis].set_linewidth(1.5)
#             cbar_ax.spines[axis].set_linewidth(axiswidth)
    f.colorbar(im, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=fontsize)
    cbar_ax.set_ylabel(cbarlabel, fontsize = fontsize)
    #cbar_ax.spines
     # adjust axis width
    #plt.tight_layout()

    save_fig(figure_directory, savetitle + "Seq%dBlock%d"% (seq,bl))



def plotzscorecounts(timevector, count_zscore, seq, bl, noveltyonset = 22.5, figure_directory = "./", Ne = 4000, cmap = "YlGnBu_r", xlabel = "time [s]", ylabel = "counts",
                         fontsize = 24, x_positions = [5,10,15,20], x_labels = ["5","10","15","20"], cbarlabel = r"$ \nu $ [Hz]",
                         savetitle = "sparseness_counts", ifAvg = False, ifBlockAvg = False, ifExcitatory = True, origin="lower", axiswidth = 1,Nseq = 5):
    # plot the accumulated count of positive,negative and zero z-scores
    if ifExcitatory:
        savetitle = "Excitatory_" + savetitle
        ylabel = "excitatory " + ylabel
        count_zscore = count_zscore[0:Ne,:]
    else:
        savetitle = "Inhibitory_" + savetitle
        ylabel = "inhibitory " + ylabel
        count_zscore = count_zscore[Ne:,:]

    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    #sumpos = np.sum(count_zscore == 1,axis=0) # sum of cells firig above average
    plt.plot(timevector, np.sum(count_zscore == 1,axis=0), label = "z > 0", color  = "orangered", lw = 3)
    #sumneg = np.sum(count_zscore == -1,axis=0)# sum of cells firig below average
    plt.plot(timevector,np.sum(count_zscore == -1,axis=0), label = "z < 0", color  = "darkslateblue", lw = 3)
    #sumzero = np.sum(count_zscore == 0,axis=0)# sum of cells firig at average level
    plt.plot(timevector,np.sum(count_zscore == 0,axis=0), label = "z = 0", color  = "grey", lw = 3)
    #plt.xlim([8,232])
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    #plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.legend(fontsize = fontsize, frameon = True)
    #plt.xticks(x_positions, x_labels,fontsize = fontsize)
    plt.xticks(fontsize = fontsize)

    ax.axvline(x=noveltyonset,color ="k")#,lw = 3, "k")
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    plt.tight_layout()
    if ifAvg:
        save_fig(figure_directory, savetitle + "Average")
    elif ifBlockAvg:
        save_fig(figure_directory, savetitle + "AvgOverBlocksSeq%d" % seq)
    else:
        save_fig(figure_directory, savetitle + "Seq%dBlock%d" % (seq,bl))

# ------------------------------------- plot histograms -------------------------------
def plot_histograms_seq(data, bl, idx, Nseq =5, iflegend = False, bins = np.linspace(0,50,51), colorseq = ["r","g","b","k","c","m"],
                     lw = 3, xlabel = "rate [Hz]", ylabel ="counts",fontsize = 24,ifioff = False, alpha=0.2, axiswidth = 1, Nblocks = 10):
    """ plot population average in one figure fig (def before)"""
    if ifioff:
        plt.ioff()

    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(111)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    for seq in range(1,Nseq+1):
        plt.hist(data[seq-1,bl,idx],bins = bins, color = colorseq[seq-1], label="seq. "+str(seq),alpha = alpha)#,density=True)

    plt.legend(fontsize = fontsize)
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    plt.tight_layout()

def plot_histograms_block(data, seq, idx, Nblocks =10, iflegend = False, bins = np.linspace(0,50,51), colorseq = ["r","g","b","k","c","m","r","g","b","k","c","m","r","g","b","k","c","m"],
                     lw = 3, xlabel = "rate [Hz]", ylabel ="counts",fontsize = 24,ifioff = False, alpha=0.2, axiswidth = 1):
    """ plot population average in one figure fig (def before)"""
    if ifioff:
        plt.ioff()

    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(111)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    for bl in range(1,Nblocks+1):
        plt.hist(data[seq,bl-1,idx],bins = bins, color = colorseq[bl-1], label="block "+str(bl),alpha = alpha)#,density=True)

    plt.legend(fontsize = fontsize, ncol = 2)
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    plt.tight_layout()

def plot_histograms_seq_cut(data, bl, idx, cutlow = 200, cuthigh = 1500, Nseq =5, iflegend = False, bins = np.linspace(0,50,51), colorseq = ["r","g","b","k","c","m"],
                     lw = 3, xlabel = "rate [Hz]", ylabel ="counts",fontsize = 24,ifioff = False, alpha=0.2, axiswidth = 1, Nblocks = 10):
    """ plot population average in one figure fig (def before)"""
    if ifioff:
        plt.ioff()
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15,12))

#     fig = plt.figure(figsize=(15,12))
#     ax = fig.add_subplot(111)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(axiswidth)
        ax2.spines[axis].set_linewidth(1.5)

    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
        ax2.spines[axis].set_linewidth(0)

    for seq in range(1,Nseq+1):
        ax.hist(data[seq-1,bl,idx],bins = bins, color = colorseq[seq-1], label="seq. "+str(seq),alpha = alpha)#,density=True)
        ax2.hist(data[seq-1,bl,idx],bins = bins, color = colorseq[seq-1], label="seq. "+str(seq),alpha = alpha)#,density=True)

    ax.set_ylim(bottom = cuthigh)  # outliers only
    ax2.set_ylim(top = cutlow)  # most of the data
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')  # don't put tick labels at the top
    #ax2.xaxis.tick_bottom()

    d = .005  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, lw = 4)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    #ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    #ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
#     ax.legend(fontsize = fontsize)
    for label in (ax.get_yticklabels()):
#         label.set_fontname('Arial')
        label.set_fontsize(fontsize)
        #label.set_tick_params(width=3)
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=axiswidth)

    ax2.xaxis.set_tick_params(width=3)
    ax2.yaxis.set_tick_params(width=3)
    plt.legend(fontsize = fontsize,ncol=2)
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    plt.tight_layout()

def plot_histograms_block_cut(data, seq, idx, cutlow = 200, cuthigh = 1500, Nblocks =10, iflegend = False, bins = np.linspace(0,50,51), colorseq = ["r","g","b","k","c","m","r","g","b","k","c","m"],
                     lw = 3, xlabel = "rate [Hz]", ylabel ="counts",fontsize = 24,ifioff = False, alpha=0.2, axiswidth = 1):
    """ plot population average in one figure fig (def before)"""
    if ifioff:
        plt.ioff()
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15,12))

#     fig = plt.figure(figsize=(15,12))
#     ax = fig.add_subplot(111)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(axiswidth)
        ax2.spines[axis].set_linewidth(1.5)

    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
        ax2.spines[axis].set_linewidth(0)

    for bl in range(1,Nblocks+1):
        ax.hist(data[seq,bl-1,idx],bins = bins, color = colorseq[bl-1], label="block "+str(bl),alpha = alpha)#,density=True)
        ax2.hist(data[seq,bl-1,idx],bins = bins, color = colorseq[bl-1], label="block "+str(bl),alpha = alpha)#,density=True)

    ax.set_ylim(bottom = cuthigh)  # outliers only
    ax2.set_ylim(top = cutlow)  # most of the data
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')  # don't put tick labels at the top
    #ax2.xaxis.tick_bottom()

    d = .005  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, lw = 4)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    #ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    #ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    #ax.legend(fontsize = fontsize, ncol = 2)
    for label in (ax.get_yticklabels()):
#         label.set_fontname('Arial')
        label.set_fontsize(fontsize)
        #label.set_tick_params(width=3)
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=axiswidth)

    ax2.xaxis.set_tick_params(width=3)
    ax2.yaxis.set_tick_params(width=3)
    plt.legend(fontsize = fontsize,  ncol=2)
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    plt.tight_layout()

def plot_histogram_cut(data, cutlow = 200, cuthigh = 1500, Nblocks =10, iflegend = False, bins = np.linspace(0,50,51), colorseq = ["r","g","b","k","c","m","r","g","b","k","c","m"],
                     lw = 3, xlabel = "rate [Hz]", ylabel ="counts",fontsize = 24,color = "darkred",ifioff = False, alpha=0.2, ifExcitatory = True, axiswidth = 1,figsize=(15,12)):
    """ plot population average in one figure fig (def before)"""
    if ifioff:
        plt.ioff()
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True,figsize=figsize )
    if ifExcitatory:
        color = "darkblue"
#     fig = plt.figure(figsize=(15,12))
#     ax = fig.add_subplot(111)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(axiswidth)
        ax2.spines[axis].set_linewidth(1.5)

    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
        ax2.spines[axis].set_linewidth(0)

    # plot histograms
    ax.hist(data,bins = bins, color = color,alpha = alpha)#,density=True)
    ax2.hist(data,bins = bins, color = color, alpha = alpha)#,density=True)

    ax.set_ylim(bottom = cuthigh)  # outliers only
    ax2.set_ylim(top = cutlow)  # most of the data
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')  # don't put tick labels at the top
    #ax2.xaxis.tick_bottom()

    d = .005  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, lw = 4)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    #ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    #ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    #ax.legend(fontsize = fontsize, ncol = 2)
    for label in (ax.get_yticklabels()):
#         label.set_fontname('Arial')
        label.set_fontsize(fontsize)
        #label.set_tick_params(width=3)
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=axiswidth)

    ax2.xaxis.set_tick_params(width=3)
    ax2.yaxis.set_tick_params(width=3)
#     plt.legend(fontsize = fontsize,  ncol=2)
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    plt.tight_layout()

def plot_histogram(data,iflegend = False, bins = np.linspace(0,50,51),figsize=(15,12),
                     lw = 3, xlabel = "rate [Hz]", ylabel ="counts",fontsize = 24,color = "darkred",ifioff = False, alpha=0.2, ifExcitatory = True, axiswidth = 1, Nblocks = 10):
    """ plot population average in one figure fig (def before)"""
    if ifioff:
        plt.ioff(figsize=figsize)
    if ifExcitatory:
        color = "darkblue"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    plt.hist(data,bins = bins, color = color, alpha = alpha)#,density=True)


    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    plt.tight_layout()

def plot_histogram_mult(fig, data, bins = np.linspace(0,50,51),iflegend = False, legend = " ",
                     lw = 3, xlabel = "rate [Hz]", ylabel ="counts",fontsize = 24,ncol=2,color = "darkred",ifioff = False, alpha=0.2, ifExcitatory = True, axiswidth = 1, Nblocks = 10):
    """ plot population average in one figure fig (def before)"""
    if ifioff:
        plt.ioff()
    if ifExcitatory:
        color = "darkblue"
    #fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(111)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    if iflegend:
        plt.hist(data,bins = bins, color = color, alpha = alpha, label=legend)#,density=True)
        plt.legend(fontsize = fontsize, ncol=ncol)
    else:
        plt.hist(data,bins = bins, color = color, alpha = alpha)#,density=True)

    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    plt.tight_layout()

# ------------------------ fitting --------------------------------------------
def exp_with_offset(t, a, tau, aoff):
    """ return exponential with offset"""
    return a * np.exp(- t/tau) + aoff

def exp_no_offset(t, a, tau):
    """ return exponential with offset"""
    return a * np.exp(- t/tau)

def one_exp_no_offset(t, a, tau):
    """ return exponential with offset"""
    return a * (1-np.exp(- t/tau))
# use optimize curvefit function
# params, params_covariance = optimize.curve_fit(exp_with_offset, x_data, y_data, p0=[10, 6, 3])

def one_exp_with_offset(t, a, tau,aoff):
    """ return exponential with offset"""
    return a * (1-np.exp(- t/tau))+ aoff

def lin_with_offset(x, m, yo):
    """ return line with slope m and offset yo"""
    return m * x + yo

def lin_no_offset(x, m):
    """ return line with slope m and offset yo"""
    return m * x
# use optimize curvefit function
# params, params_covariance = optimize.curve_fit(exp_with_offset, x_data, y_data, p0=[10, 6, 3])



        #save_fig(figure_directory, "Fit_PopulationAverages_E" + str(Nimg[seq-1]))

# def plot_all_averages(timelist, datalist, Nreps,  startidx = 0, endidx = -1, figure_directory = "./", figsize=(20, 10), ifoffset = True,
#                               offset = 1, iflegend = False, ifcolor = True, color = ["r","g","b","k","c","m","r","g","b","k","c","m"],
#                             fontsize = 20, lw = 3, xlabel = "time [s]", ylabel ="$\\rho$ [Hz]",
#                               ifioff = False, ifyticks = True, yticks = [3,4], ifsavefig = True, savehandle = "E", axiswidth = 1):
#     """ plot population averages in one figure
#     input:  list with means
#             time vector
#             number of repetitions vector"""
#     fig = plt.figure(figsize=figsize) #run first and then conse
#     for seq in reversed(range(Nreponset,len(Nreps) + 1)):
#         if ifoffset:
#             plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][startidx:endidx] + offset*(seq-1),
#                              iflegend = iflegend, legend = "Nreps: " + str(Nreps[seq-1]),
#                              lw = lw, ifcolor = ifcolor, color = color[seq-1])
#         else:
#             plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][startidx:endidx],
#                                  iflegend = iflegend, legend = "Nreps: " + str(Nreps[seq-1]),
#                                  lw = lw, ifcolor = ifcolor, color = color[seq-1])
#     if ifyticks:
#         plt.yticks(yticks)
#
#     savetitle = "PopulationAveragesSeq_" + savehandle
#     if ifoffset:
#         savetitle = savetitle + "offset"
#     if ifsavefig:
#         save_fig(figure_directory, savetitle)
#
# def plot_all_traces_and_average(timelist, datalist, meandatalist,  Nreps, Nblocks = 10, startidx = 0, endidx = -1,figure_directory = "./", figsize=(20, 10), ifoffset = True,
#                               offset = 1, iflegend = False, ifcolor = True, color = ["r","g","b","k","c","m","r","g","b","k","c","m"],
#                             fontsize = 20, lw = 3, xlabel = "time [s]", ylabel ="$\\rho$ [Hz]",
#                               ifioff = False, ifyticks = True, yticks = [3,4], ifsavefig = True, savehandle = "E", axiswidth = 1):
#     """ plot population average traces per Nreps with average
#     input:  list with means
#             time vector
#             number of repetitions vector"""
#     for seq in reversed(range(Nreponset,len(Nreps) + 1)):
#         fig = plt.figure(figsize=figsize)
#         for bl in range(1, Nblocks + 1):
#             plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][bl-1,startidx:endidx],
#                              iflegend = iflegend,
#                              lw = 1,ifcolor = True, color = "lightgrey")
#         plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],meandatalist[seq-1][startidx:endidx],
#                              iflegend = iflegend,
#                              lw = lw, ifcolor = ifcolor, color = color[seq-1])
#
#         if ifyticks:
#             plt.yticks(yticks)
#
#         savetitle = "PopulationAverages_and_traces_" + savehandle + "Nreps"+ str(Nreps[seq-1])
#
#         if ifsavefig:
#             save_fig(figure_directory, savetitle)
#

def plot_all_averages(timelist, datalist, Nreps,  startidx = 0, endidx = -1, figure_directory = "./", figsize=(20, 10), ifoffset = True,
                              offset = 1, ifseqlen = False, iflegend = False, ifcolor = True, color = ["r","g","b","k","c","m","r","g","b","k","c","m"],
                            fontsize = 20, lw = 3, xlabel = "time [s]", legendhandle = "Nimg: ", Nreponset = 6, ylabel ="$\\rho$ [Hz]",
                              ifioff = False, ifyticks = True, yticks = [3,4], ifsavefig = True, savehandle = "E", axiswidth = 1):
    """ plot population averages in one figure
    input:  list with means
            time vector
            number of repetitions vector"""
    fig = plt.figure(figsize=figsize) #run first and then conse
    if ifseqlen:
        for seq in reversed(range(1,len(Nreps) + 1)):
            if ifoffset:
                plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][startidx:endidx] + offset*(seq-1),
                                 iflegend = iflegend, legend = legendhandle + str(Nreps[seq-1]),
                                 lw = lw, ifcolor = ifcolor, color = color[seq-1])
            else:
                plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][startidx:endidx],
                                     iflegend = iflegend, legend = legendhandle + str(Nreps[seq-1]),
                                     lw = lw, ifcolor = ifcolor, color = color[seq-1])

    else:
        for seq in reversed(range(Nreponset,len(Nreps) + 1)):
            if ifoffset:
                plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][startidx:endidx] + offset*(seq-1),
                                 iflegend = iflegend, legend = "Nreps: " + str(Nreps[seq-1]),
                                 lw = lw, ifcolor = ifcolor, color = color[seq-1])
            else:
                plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][startidx:endidx],
                                     iflegend = iflegend, legend = "Nreps: " + str(Nreps[seq-1]),
                                     lw = lw, ifcolor = ifcolor, color = color[seq-1])
    if ifyticks:
        plt.yticks(yticks)
    plt.tight_layout()
    savetitle = "PopulationAveragesSeq_" + savehandle
    if ifoffset:
        savetitle = savetitle + "offset"
    if ifsavefig:
        save_fig(figure_directory, savetitle)

def plot_all_averages_with_fits(timelist, datalist, Nreps,  params_blockavg, startidx = 0, endidx = -1, figure_directory = "./", figsize=(10, 20), ifoffset = True,
                              offset = 1, ifseqlen = False, iflegend = False, ifcolor = True, color = ["r","g","b","k","c","m","r","g","b","k","c","m"],
                            fontsize = 20, lw = 3, xlabel = "time [s]", legendhandle = "Nimg: ", Nreponset = 6, ylabel ="$\\rho$ [Hz]",
                              ifioff = False, ifyticks = True, yticks = [3,4], ifsavefig = True, savehandle = "E", axiswidth = 1):
    """ plot population averages in one figure
    input:  list with means
            time vector
            number of repetitions vector"""
    fig = plt.figure(figsize=figsize) #run first and then conse
    if ifseqlen:
        for seq in range(1,len(Nreps) + 1):
            if ifoffset:
                # plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][startidx:endidx] + offset*(seq-1),
                #                  iflegend = iflegend, legend = legendhandle + str(Nreps[seq-1]),
                #                  lw = lw, ifcolor = ifcolor, color = color[seq-1])
                plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][startidx:endidx] + offset*(seq-1),
                                 iflegend = iflegend, legend = legendhandle + str(Nreps[seq-1]),
                                 lw = lw, ifcolor = ifcolor, color = "midnightblue")
                plt.plot(timelist[seq-1][startidx:endidx], exp_with_offset(timelist[seq-1][startidx:endidx], params_blockavg[seq-1,0], params_blockavg[seq-1,1],params_blockavg[seq-1,2])+ offset*(seq-1), color = "red")
            else:
                # plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][startidx:endidx],
                #                      iflegend = iflegend, legend = legendhandle + str(Nreps[seq-1]),
                #                      lw = lw, ifcolor = ifcolor, color = color[seq-1])
                plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][startidx:endidx],
                                     iflegend = iflegend, legend = legendhandle + str(Nreps[seq-1]),
                                     lw = lw, ifcolor = ifcolor, color = "midnightblue")
                plt.plot(timelist[seq-1][startidx:endidx], exp_with_offset(timelist[seq-1][startidx:endidx], params_blockavg[seq-1,0], params_blockavg[seq-1,1],params_blockavg[seq-1,2])+ offset*(seq-1), color = "red")

    else:
        for seq in reversed(range(Nreponset,len(Nreps) + 1)):
            if ifoffset:
                plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][startidx:endidx] + offset*(seq-1),
                                 iflegend = iflegend, legend = "Nreps: " + str(Nreps[seq-1]),
                                 lw = lw, ifcolor = ifcolor, color = color[seq-1])
            else:
                plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][startidx:endidx],
                                     iflegend = iflegend, legend = "Nreps: " + str(Nreps[seq-1]),
                                     lw = lw, ifcolor = ifcolor, color = color[seq-1])
    if ifyticks:
        plt.yticks(yticks)
    plt.tight_layout()
    savetitle = "PopulationAveragesSeq_" + savehandle
    if ifoffset:
        savetitle = savetitle + "offset"
    if ifsavefig:
        save_fig(figure_directory, savetitle)

def plot_all_traces_and_average(timelist, datalist, meandatalist,  Nreps, Nblocks = 10, startidx = 0, endidx = -1,figure_directory = "./", figsize=(20, 10), ifoffset = True,
                              offset = 1, ifseqlen = False, iflegend = False, ifcolor = True, color = ["r","g","b","k","c","m","r","g","b","k","c","m"],
                            fontsize = 20, lw = 3, Nreponset = 6, xlabel = "time [s]", ylabel ="$\\rho$ [Hz]",
                              ifioff = False, ifyticks = True, yticks = [3,4], ifsavefig = True, savehandle = "E", axiswidth = 1):
    """ plot population average traces per Nreps with average
    input:  list with means
            time vector
            number of repetitions vector"""
    if ifseqlen:
        for seq in reversed(range(1,len(Nreps) + 1)):
            fig = plt.figure(figsize=figsize)
            for bl in range(1, Nblocks + 1):
                plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][bl-1,startidx:endidx],
                                 iflegend = iflegend,
                                 lw = 1,ifcolor = True, color = "lightgrey")
            plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],meandatalist[seq-1][startidx:endidx],
                                 iflegend = iflegend,
                                 lw = lw, ifcolor = ifcolor, color = color[seq-1])
            if ifyticks:
                plt.yticks(yticks)
            if ifseqlen:
                savetitle = "PopulationAverages_and_traces_" + savehandle + "Nimg"+ str(Nreps[seq-1])
            else:
                savetitle = "PopulationAverages_and_traces_" + savehandle + "Nreps"+ str(Nreps[seq-1])
            plt.tight_layout()
            if ifsavefig:
                save_fig(figure_directory, savetitle)
    else:
        for seq in reversed(range(Nreponset,len(Nreps) + 1)):
            fig = plt.figure(figsize=figsize)
            for bl in range(1, Nblocks + 1):
                plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][bl-1,startidx:endidx],
                                 iflegend = iflegend,
                                 lw = 1,ifcolor = True, color = "lightgrey")
            plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],meandatalist[seq-1][startidx:endidx],
                                 iflegend = iflegend,
                                 lw = lw, ifcolor = ifcolor, color = color[seq-1])
            if ifyticks:
                plt.yticks(yticks)
            if ifseqlen:
                savetitle = "PopulationAverages_and_traces_" + savehandle + "Nimg"+ str(Nreps[seq-1])
            else:
                savetitle = "PopulationAverages_and_traces_" + savehandle + "Nreps"+ str(Nreps[seq-1])
            plt.tight_layout()
            if ifsavefig:
                save_fig(figure_directory, savetitle)

def plot_all_traces_and_average_E_I(timelist, datalistE, datalistI,  Nreps, Nblocks = 10, startidx = 0, endidx = -1,figure_directory = "./", figsize=(7, 5), ifoffset = True,
                              offset = 1, ifseqlen = False, iflegend = False, ifcolor = True, color = ["r","g","b","k","c","m","r","g","b","k","c","m"],
                            fontsize = 20, lw = 3, Nreponset = 6, xlabel = "time [s]", ylabel ="$\\rho$ [Hz]",
                              ifioff = False, ifyticks = True, yticks = [3,4], ifsavefig = True, savehandle = "E", axiswidth = 1):
    """ plot population average traces per Nreps with average
    input:  list with means
            time vector
            number of repetitions vector"""
    E = []
    I = []
    fig = plt.figure(figsize=figsize)
    for seq in reversed(range(len(Nreps))):
        for bl in range(Nblocks):
            plot_popavg_mult(fig,  timelist[seq][startidx:endidx],datalistE[seq][bl,startidx:endidx],
                             iflegend = False,
                             lw = 1,ifcolor = True, color ="midnightblue",alpha=0.5)
            E.append(datalistE[seq][bl,startidx:endidx])
            I.append(datalistI[seq][bl,startidx:endidx])
            plot_popavg_mult(fig,  timelist[seq][startidx:endidx],datalistI[seq][bl,startidx:endidx],
                              iflegend = False,
                              lw = 1,ifcolor = True, color ="darkred",alpha=0.5)
    plot_popavg_mult(fig,  timelist[seq][startidx:endidx],np.mean(E,axis = 0),
                             iflegend = True,legend = "E",
                             lw = lw, ifcolor = ifcolor, color = "midnightblue")
    plot_popavg_mult(fig,  timelist[seq][startidx:endidx],np.mean(I,axis = 0),
                             iflegend = True,legend = "I",
                             lw = lw, ifcolor = ifcolor, color = "darkred")
    if ifyticks:
        plt.yticks(yticks)
    if ifseqlen:
        savetitle = "PopulationAverages_and_traces_EandI" + savehandle
    else:
        savetitle = "PopulationAverages_and_traces_EandI" + savehandle
    plt.tight_layout()
    if ifsavefig:
        save_fig(figure_directory, savetitle)



def fit_variable_repetitions_gen_arrays(edges,datalist, meandatalist, lenstim, lenpause, Nreps, Nimg, Nblocks, avgindices = 20, initialparams = [10, 6, 3], bounds=(0, [10., 140., 10]), ifplot = False):
    """ perform fitting of all traces included in datalist and meandatalist
        determine the baseline firing rate prior to the novelty stimulation
    input:  edges (timevector)
            data lists
            duration of stimulus and length of the pause
            Nrepetitions array
            number of images per sequence
            number of blocks
            initial paramteres for the fits
            number of samples in the baseline average"""
    Nseq = len(Nreps)
    params = np.zeros((Nseq,Nblocks,3)) # params: a, tau, a_offset
    params_covariance = np.zeros((Nseq,Nblocks,3,3)) # covariance matrices of fit params
    params_err = np.zeros((Nseq,Nblocks,3)) # errors of fit params
    # initialise parameter arrays for block averages
    params_blockavg = np.zeros((Nseq,3))
    params_covariance_blockavg = np.zeros((Nseq,3,3))
    params_err_blockavg = np.zeros((Nseq,3))
    t_before_nov = np.zeros(Nseq)
    valmin = np.zeros(Nseq)
    idxmin = np.zeros(Nseq).astype(int)
    baseline = np.zeros((Nseq,Nblocks))
    baseline_avg = np.zeros(Nseq)


    #print(avgindices)
    for rep in range(1,len(Nreps) + 1):
        #t_before_nov[rep-1] = edges[rep-1][-1] - 7. - (Nreps[rep-1]+1)*lenstim/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
        t_before_nov[rep-1] = ((Nreps[rep-1]-2)*Nimg*(lenstim+lenpause) + (Nimg-3)*(lenstim+lenpause))/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)

        #print(t_before_nov[rep-1])
        idxmin[rep-1],valmin[rep-1] = maximum(edges[rep-1]>t_before_nov[rep-1])
        params_blockavg[rep-1][:], params_covariance_blockavg[rep-1][:][:] = fit_traces(edges[rep-1][0:idxmin[rep-1]],
                                                                                        meandatalist[rep-1][0:idxmin[rep-1]], exp_with_offset, initialparams = initialparams, bounds=bounds, ifplot = ifplot)
        params_err_blockavg[rep-1][:] = np.sqrt(np.diag(params_covariance_blockavg[rep-1]))
        baseline_avg[rep-1] = np.mean(meandatalist[rep-1][max(0,idxmin[rep-1]-avgindices):idxmin[rep-1]])

        for bl in range(1, Nblocks + 1):
            params[rep-1][bl-1][:], params_covariance[rep-1][bl-1][:][:] = fit_traces(edges[rep-1][0:idxmin[rep-1]],
                                                                                      datalist[rep-1][bl-1,0:idxmin[rep-1]], exp_with_offset, initialparams = initialparams, bounds=bounds, ifplot = ifplot)
            params_err[rep-1][bl-1][:] = np.sqrt(np.diag(params_covariance[rep-1][bl-1]))
            baseline[rep-1][bl-1] = np.mean(datalist[rep-1][bl-1,max(0,idxmin[rep-1]-avgindices):idxmin[rep-1]])

    return t_before_nov, params_blockavg, params_covariance_blockavg, params_err_blockavg, baseline_avg, params, params_covariance, params_err, baseline


def fit_variable_repetitions(edges,datalist, meandatalist, lenstim, Nreps, Nblocks,  avgindices, idxmin, valmin, t_before_nov, params_blockavg,
                             params_covariance_blockavg, params_err_blockavg, baseline_avg, params, params_covariance, params_err, baseline):

    print(avgindices)
    for rep in range(1,len(Nreps) + 1):
        t_before_nov[rep-1] = edges[rep-1][-1] - 7. - (Nreps[rep-1]+1)*lenstim/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
        #print(t_before_nov[rep-1])
        idxmin[rep-1],valmin[rep-1] = maximum(edges[rep-1]>t_before_nov[rep-1])
        if rep < 2:
            print(rep)
            print("min idx rep " + str(rep) + " idx:" + str(idxmin[rep-1]))
            print(np.linspace(max(0,idxmin[rep-1]-avgindices),idxmin[rep-1], avgindices+1))
            # for block averaged traces
        #print(idxmin)
        params_blockavg[rep-1][:], params_covariance_blockavg[rep-1][:][:] = fit_traces(edges[rep-1][0:idxmin[rep-1]],
                                                                                        meandatalist[rep-1][0:idxmin[rep-1]], exp_with_offset, initialparams = [10, 6, 3])
        params_err_blockavg[rep-1][:] = np.sqrt(np.diag(params_covariance_blockavg[rep-1]))
        baseline_avg[rep-1] = np.mean(meandatalist[rep-1][max(0,idxmin[rep-1]-avgindices):idxmin[rep-1]])
        if rep < 2:
            print("bl avg  " + str(rep) + " bl:" + str(baseline_avg[rep-1]))
        for bl in range(1, Nblocks + 1):
            params[rep-1][bl-1][:], params_covariance[rep-1][bl-1][:][:] = fit_traces(edges[rep-1][0:idxmin[rep-1]],
                                                                                      datalist[rep-1][bl-1,0:idxmin[rep-1]], exp_with_offset, initialparams = [10, 6, 3])
            params_err[rep-1][bl-1][:] = np.sqrt(np.diag(params_covariance[rep-1][bl-1]))
            baseline[rep-1][bl-1] = np.mean(datalist[rep-1][bl-1,max(0,idxmin[rep-1]-avgindices):idxmin[rep-1]])
            if rep < 2:
                print("bl avg  " + str(rep) + " bl:" +str(bl)  + " bl val :" + str(baseline[rep-1][bl-1]))

    return t_before_nov, params_blockavg, params_covariance_blockavg, params_err_blockavg, baseline_avg, params, params_covariance, params_err, baseline

def fit_variable_repetitions_copy(Edges,Datalist, Meandatalist, lenstim, Nreps, Nblocks,  Avgindices, idxmin, valmin, t_before_nov, Params_blockavg,
                             Params_covariance_blockavg, Params_err_blockavg, Baseline_avg, Params, Params_covariance, Params_err, Baseline):

    print(Avgindices)
    for rep in range(1,len(Nreps) + 1):
        t_before_nov[rep-1] = Edges[rep-1][-1] - 7. - (Nreps[rep-1]+1)*lenstim/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
        #print(t_before_nov[rep-1])
        idxmin[rep-1],valmin[rep-1] = maximum(Edges[rep-1]>t_before_nov[rep-1])
        if rep < 2:
            print(rep)
            print("min idx rep " + str(rep) + " idx:" + str(idxmin[rep-1]))
            print(np.linspace(max(0,idxmin[rep-1]-Avgindices),idxmin[rep-1], Avgindices+1))
            # for block averaged traces
        #print(idxmin)
        Params_blockavg[rep-1][:], Params_covariance_blockavg[rep-1][:][:] = fit_traces(Edges[rep-1][0:idxmin[rep-1]],
                                                                                        Meandatalist[rep-1][0:idxmin[rep-1]], exp_with_offset, initialparams = [10, 6, 3])
        Params_err_blockavg[rep-1][:] = np.sqrt(np.diag(Params_covariance_blockavg[rep-1]))
        Baseline_avg[rep-1] = np.mean(Meandatalist[rep-1][max(0,idxmin[rep-1]-Avgindices):idxmin[rep-1]])
        if rep < 2:
            print("bl avg  " + str(rep) + " bl:" + str(Baseline_avg[rep-1]))
        for bl in range(1, Nblocks + 1):
            Params[rep-1][bl-1][:], Params_covariance[rep-1][bl-1][:][:] = fit_traces(Edges[rep-1][0:idxmin[rep-1]],
                                                                                      Datalist[rep-1][bl-1,0:idxmin[rep-1]], exp_with_offset, initialparams = [10, 6, 3])
            Params_err[rep-1][bl-1][:] = np.sqrt(np.diag(Params_covariance[rep-1][bl-1]))
            Baseline[rep-1][bl-1] = np.mean(Datalist[rep-1][bl-1,max(0,idxmin[rep-1]-Avgindices):idxmin[rep-1]])
            if rep < 2:
                print("bl avg  " + str(rep) + " bl:" +str(bl)  + " bl val :" + str(Baseline[rep-1][bl-1]))

    return t_before_nov, Params_blockavg, Params_covariance_blockavg, Params_err_blockavg, Baseline_avg, Params, Params_covariance, Params_err, Baseline

def fit_variable_repetitions_gen_arrays_startidx_oldwithBL(edges,datalist, meandatalist, lenstim, lenpause, Nreps, Nimg, Nblocks, fit_function = exp_with_offset, avgindices = 20, initialparams = [10, 6, 3], bounds=(0, [10., 140., 10]), ifplot = False, startimg = 4, idxconv = 4):
    """ perform fitting of all traces included in datalist and meandatalist
        determine the baseline firing rate prior to the novelty stimulation
    input:  edges (timevector)
            data lists
            duration of stimulus and length of the pause
            Nrepetitions array
            number of images per sequence
            number of blocks
            initial paramteres for the fits
            number of samples in the baseline average

            ensure only values above convolution idx are considered
            set startindex to index when N images were played once"""
    Nseq = len(Nreps)
    params = np.zeros((Nseq,Nblocks,3)) # params: a, tau, a_offset
    params_covariance = np.zeros((Nseq,Nblocks,3,3)) # covariance matrices of fit params
    params_err = np.zeros((Nseq,Nblocks,3)) # errors of fit params
    # initialise parameter arrays for block averages
    params_blockavg = np.zeros((Nseq,3))
    params_covariance_blockavg = np.zeros((Nseq,3,3))
    params_err_blockavg = np.zeros((Nseq,3))
    t_before_nov = np.zeros(Nseq)
    idxmin = np.zeros(Nseq).astype(int) # cut off index before novelty
    baseline = np.zeros((Nseq,Nblocks))
    baseline_avg = np.zeros(Nseq)
    t_start_img = startimg*(lenstim+lenpause)/1000.
    startidx = np.argmax(edges[0]>t_start_img)
    print(startidx)
    startidx = max([startidx, idxconv]) # select maximum value between convolution cutoff and minimum for fitting

    #print(avgindices)
    for rep in range(1,len(Nreps) + 1):
        #t_before_nov[rep-1] = edges[rep-1][-1] - 7. - (Nreps[rep-1]+1)*lenstim/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
        t_before_nov[rep-1] = ((Nreps[rep-1]-2)*Nimg*(lenstim+lenpause) + (Nimg-3)*(lenstim+lenpause))/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)

        #print(t_before_nov[rep-1])
        idxmin[rep-1] = np.argmax(edges[rep-1]>t_before_nov[rep-1])
        params_blockavg[rep-1][:], params_covariance_blockavg[rep-1][:][:] = fit_traces(edges[rep-1][startidx:idxmin[rep-1]],
                                                                                        meandatalist[rep-1][startidx:idxmin[rep-1]], fit_function, initialparams = initialparams, bounds=bounds, ifplot = ifplot)
        params_err_blockavg[rep-1][:] = np.sqrt(np.diag(params_covariance_blockavg[rep-1]))
        baseline_avg[rep-1] = np.mean(meandatalist[rep-1][max(idxconv,idxmin[rep-1]-avgindices):idxmin[rep-1]])

        for bl in range(1, Nblocks + 1):
            params[rep-1][bl-1][:], params_covariance[rep-1][bl-1][:][:] = fit_traces(edges[rep-1][startidx:idxmin[rep-1]],
                                                                                      datalist[rep-1][bl-1,startidx:idxmin[rep-1]], fit_function, initialparams = initialparams, bounds=bounds, ifplot = ifplot)
            params_err[rep-1][bl-1][:] = np.sqrt(np.diag(params_covariance[rep-1][bl-1]))
            baseline[rep-1][bl-1] = np.mean(datalist[rep-1][bl-1,max(idxconv,idxmin[rep-1]-avgindices):idxmin[rep-1]])

    return t_before_nov, params_blockavg, params_covariance_blockavg, params_err_blockavg, baseline_avg, params, params_covariance, params_err, baseline

def fit_variable_repetitions_gen_arrays_startidx(edges,datalist, meandatalist, lenstim, lenpause,
 Nreps, Nimg, Nblocks, fit_function = exp_with_offset, ifseqlen = False, avgindices = 20,
  initialparams = [10, 6, 3], bounds=(0, [10., 140., 10]), ifplot = False, startimg = 4, idxconv = 4):
    """ perform fitting of all traces included in datalist and meandatalist
        determine the baseline firing rate prior to the novelty stimulation
    input:  edges (timevector)
            data lists
            duration of stimulus and length of the pause
            Nrepetitions array
            number of images per sequence
            number of blocks
            initial paramteres for the fits
            number of samples in the baseline average

            ensure only values above convolution idx are considered
            set startindex to index when N images were played once"""
    if ifseqlen:
        Nseq = len(Nimg)
    else:
        Nseq = len(Nreps)
    params = np.zeros((Nseq,Nblocks,3)) # params: a, tau, a_offset
    params_covariance = np.zeros((Nseq,Nblocks,3,3)) # covariance matrices of fit params
    params_err = np.zeros((Nseq,Nblocks,3)) # errors of fit params
    # initialise parameter arrays for block averages
    params_blockavg = np.zeros((Nseq,3))
    params_covariance_blockavg = np.zeros((Nseq,3,3))
    params_err_blockavg = np.zeros((Nseq,3))
    # get the time before the novelty peak
    t_before_nov = np.zeros(Nseq)
    idxmin = np.zeros(Nseq).astype(int) # cut off index before novelty
    if ifseqlen:
        t_start_img = np.zeros(Nseq)
        #startidx = np.zeros(Nseq)

         # select maximum value between convolution cutoff and minimum for fitting

        #print(avgindices)
        for rep in range(1,Nseq + 1):
            t_start_img[rep-1] = Nimg[rep-1]*(lenstim+lenpause)/1000.
            startidx = np.argmax(edges[rep-1]>t_start_img[rep-1])
            #print(startidx)
            startidx = max([startidx, idxconv])
            print(startidx)
            #t_before_nov[rep-1] = edges[rep-1][-1] - 7. - (Nreps[rep-1]+1)*lenstim/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
            t_before_nov[rep-1] = ((Nreps-2)*Nimg[rep-1]*(lenstim+lenpause) + (Nimg[rep-1]-3)*(lenstim+lenpause))/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)

            #print(t_before_nov[rep-1])
            idxmin[rep-1] = np.argmax(edges[rep-1]>t_before_nov[rep-1])
            params_blockavg[rep-1][:], params_covariance_blockavg[rep-1][:][:] = fit_traces(edges[rep-1][startidx:idxmin[rep-1]],
                                                                                            meandatalist[rep-1][startidx:idxmin[rep-1]], fit_function, initialparams = initialparams, bounds=bounds, ifplot = ifplot)
            params_err_blockavg[rep-1][:] = np.sqrt(np.diag(params_covariance_blockavg[rep-1]))

            for bl in range(1, Nblocks + 1):
                params[rep-1][bl-1][:], params_covariance[rep-1][bl-1][:][:] = fit_traces(edges[rep-1][startidx:idxmin[rep-1]],
                                                                                          datalist[rep-1][bl-1,startidx:idxmin[rep-1]], fit_function, initialparams = initialparams, bounds=bounds, ifplot = False)
                params_err[rep-1][bl-1][:] = np.sqrt(np.diag(params_covariance[rep-1][bl-1]))
# for rep in range(1,Nseq + 1):
#     t_start_img[rep-1] = Nimg[rep-1]*(lenstim+lenpause)/1000.
#     startidx[rep-1] = np.argmax(edges[rep-1]>t_start_img[rep-1])
#     #print(startidx)
#     startidx[rep-1] = max([startidx[rep-1], idxconv])
#
#     #t_before_nov[rep-1] = edges[rep-1][-1] - 7. - (Nreps[rep-1]+1)*lenstim/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
#     t_before_nov[rep-1] = ((Nreps-2)*Nimg[rep-1]*(lenstim+lenpause) + (Nimg[rep-1]-3)*(lenstim+lenpause))/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
#
#     #print(t_before_nov[rep-1])
#     idxmin[rep-1] = np.argmax(edges[rep-1]>t_before_nov[rep-1])
#     params_blockavg[rep-1][:], params_covariance_blockavg[rep-1][:][:] = fit_traces(edges[rep-1][startidx[rep-1]:idxmin[rep-1]],
#                                                                                     meandatalist[rep-1][startidx[rep-1]:idxmin[rep-1]], fit_function, initialparams = initialparams, bounds=bounds, ifplot = ifplot)
#     params_err_blockavg[rep-1][:] = np.sqrt(np.diag(params_covariance_blockavg[rep-1]))
#
#     for bl in range(1, Nblocks + 1):
#         params[rep-1][bl-1][:], params_covariance[rep-1][bl-1][:][:] = fit_traces(edges[rep-1][startidx[rep-1]:idxmin[rep-1]],
#                                                                                   datalist[rep-1][bl-1,startidx[rep-1]:idxmin[rep-1]], fit_function, initialparams = initialparams, bounds=bounds, ifplot = ifplot)
#         params_err[rep-1][bl-1][:] = np.sqrt(np.diag(params_covariance[rep-1][bl-1]))

    else:
        t_start_img = startimg*(lenstim+lenpause)/1000.
        print(t_start_img)
        startidx = np.argmax(edges[0]>t_start_img)
        print(startidx)
        startidx = max([startidx, idxconv]) # select maximum value between convolution cutoff and minimum for fitting
        print("min startidx idxconv", startidx)
        #print(avgindices)
        for rep in range(1,Nseq + 1):
            #t_before_nov[rep-1] = edges[rep-1][-1] - 7. - (Nreps[rep-1]+1)*lenstim/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
            t_before_nov[rep-1] = ((Nreps[rep-1]-2)*Nimg*(lenstim+lenpause) + (Nimg-3)*(lenstim+lenpause))/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
            print(t_before_nov[rep-1])
            #print(t_before_nov[rep-1])
            idxmin[rep-1] = np.argmax(edges[rep-1]>t_before_nov[rep-1])
            print("idxmin rep-1", idxmin[rep-1])

            if (idxmin[rep-1] - startidx) > 3:
            # fit the trace before the novelty
                print("(idxmin[rep-1] - startidx) > 3:")
                params_blockavg[rep-1][:], params_covariance_blockavg[rep-1][:][:] = fit_traces(edges[rep-1][startidx:idxmin[rep-1]],
                                                                                                meandatalist[rep-1][startidx:idxmin[rep-1]], fit_function, initialparams = initialparams, bounds=bounds, ifplot = ifplot)
                params_err_blockavg[rep-1][:] = np.sqrt(np.diag(params_covariance_blockavg[rep-1]))

                for bl in range(1, Nblocks + 1):
                    params[rep-1][bl-1][:], params_covariance[rep-1][bl-1][:][:] = fit_traces(edges[rep-1][startidx:idxmin[rep-1]],
                                                                                              datalist[rep-1][bl-1,startidx:idxmin[rep-1]], fit_function, initialparams = initialparams, bounds=bounds, ifplot = False)
                    params_err[rep-1][bl-1][:] = np.sqrt(np.diag(params_covariance[rep-1][bl-1]))
            else:
                print("(idxmin[rep-1] - startidx) > 3: SKIP  this number of reps: ", rep)
    return t_before_nov, params_blockavg, params_covariance_blockavg, params_err_blockavg, params, params_covariance, params_err



def fit_gen_arrays_startidx(edges,datalist, meandatalist, lenstim, lenpause, Nreps, Nimg, Seqs, Nblocks, fit_function = exp_with_offset, ifseqlen = False, avgindices = 20, Nseq = 5, initialparams = [10, 6, 3], bounds=(0, [10., 140., 10]), ifplot = False, startimg = 4, idxconv = 4):
    """ perform fitting of all traces included in datalist and meandatalist
        determine the baseline firing rate prior to the novelty stimulation
    input:  edges (timevector)
            data lists
            duration of stimulus and length of the pause
            Nrepetitions
            number of images per sequence
            number of sequences
            number of blocks
            initial paramteres for the fits
            number of samples in the baseline average

            ensure only values above convolution idx are considered
            set startindex to index when N images were played once"""

    Nseq = len(Seqs)
    params = np.zeros((Nseq,Nblocks,3)) # params: a, tau, a_offset
    params_covariance = np.zeros((Nseq,Nblocks,3,3)) # covariance matrices of fit params
    params_err = np.zeros((Nseq,Nblocks,3)) # errors of fit params
    # initialise parameter arrays for block averages
    params_blockavg = np.zeros((Nseq,3))
    params_covariance_blockavg = np.zeros((Nseq,3,3))
    params_err_blockavg = np.zeros((Nseq,3))
    t_before_nov = np.zeros(Nseq)
    idxmin = np.zeros(Nseq).astype(int) # cut off index before novelty

    t_start_img = startimg*(lenstim+lenpause)/1000.
    startidx = np.argmax(edges[0]>t_start_img)
    print(startidx)
    startidx = max([startidx, idxconv]) # select maximum value between convolution cutoff and minimum for fitting
    t_before_nov = ((Nreps-2)*Nimg*(lenstim+lenpause) + (Nimg-3)*(lenstim+lenpause))/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)

    #print(avgindices)
    for rep in range(1,Nseq + 1):
        #t_before_nov[rep-1] = edges[rep-1][-1] - 7. - (Nreps[rep-1]+1)*lenstim/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
        #t_before_nov[rep-1] = ((Nreps-2)*Nimg*(lenstim+lenpause) + (Nimg-3)*(lenstim+lenpause))/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)

        #print(t_before_nov[rep-1])
        idxmin[rep-1] = np.argmax(edges[rep-1]>t_before_nov)
        params_blockavg[rep-1][:], params_covariance_blockavg[rep-1][:][:] = fit_traces(edges[rep-1][startidx:idxmin[rep-1]],
                                                                                        meandatalist[rep-1][startidx:idxmin[rep-1]], fit_function, initialparams = initialparams, bounds=bounds, ifplot = ifplot)
        params_err_blockavg[rep-1][:] = np.sqrt(np.diag(params_covariance_blockavg[rep-1]))

        for bl in range(1, Nblocks + 1):
            params[rep-1][bl-1][:], params_covariance[rep-1][bl-1][:][:] = fit_traces(edges[rep-1][startidx:idxmin[rep-1]],
                                                                                      datalist[rep-1][bl-1,startidx:idxmin[rep-1]], fit_function, initialparams = initialparams, bounds=bounds, ifplot = False)
            params_err[rep-1][bl-1][:] = np.sqrt(np.diag(params_covariance[rep-1][bl-1]))

    return t_before_nov, params_blockavg, params_covariance_blockavg, params_err_blockavg, params, params_covariance, params_err


# def fit_variable_repetitions_gen_arrays_startidx(edges,datalist, meandatalist, lenstim, lenpause, Nreps, Nimg, Nblocks, fit_function = exp_with_offset, avgindices = 20, initialparams = [10, 6, 3], bounds=(0, [10., 140., 10]), ifplot = False, startimg = 4, idxconv = 4):
#     """ perform fitting of all traces included in datalist and meandatalist
#         determine the baseline firing rate prior to the novelty stimulation
#     input:  edges (timevector)
#             data lists
#             duration of stimulus and length of the pause
#             Nrepetitions array
#             number of images per sequence
#             number of blocks
#             initial paramteres for the fits
#             number of samples in the baseline average
#
#             ensure only values above convolution idx are considered
#             set startindex to index when N images were played once"""
#     Nseq = len(Nreps)
#     params = np.zeros((Nseq,Nblocks,3)) # params: a, tau, a_offset
#     params_covariance = np.zeros((Nseq,Nblocks,3,3)) # covariance matrices of fit params
#     params_err = np.zeros((Nseq,Nblocks,3)) # errors of fit params
#     # initialise parameter arrays for block averages
#     params_blockavg = np.zeros((Nseq,3))
#     params_covariance_blockavg = np.zeros((Nseq,3,3))
#     params_err_blockavg = np.zeros((Nseq,3))
#     t_before_nov = np.zeros(Nseq)
#     idxmin = np.zeros(Nseq).astype(int) # cut off index before novelty
#     t_start_img = startimg*(lenstim+lenpause)/1000.
#     startidx = np.argmax(edges[0]>t_start_img)
#     print(startidx)
#     startidx = max([startidx, idxconv]) # select maximum value between convolution cutoff and minimum for fitting
#
#     #print(avgindices)
#     for rep in range(1,len(Nreps) + 1):
#         #t_before_nov[rep-1] = edges[rep-1][-1] - 7. - (Nreps[rep-1]+1)*lenstim/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
#         t_before_nov[rep-1] = ((Nreps[rep-1]-2)*Nimg*(lenstim+lenpause) + (Nimg-3)*(lenstim+lenpause))/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
#
#         #print(t_before_nov[rep-1])
#         idxmin[rep-1] = np.argmax(edges[rep-1]>t_before_nov[rep-1])
#         params_blockavg[rep-1][:], params_covariance_blockavg[rep-1][:][:] = fit_traces(edges[rep-1][startidx:idxmin[rep-1]],
#                                                                                         meandatalist[rep-1][startidx:idxmin[rep-1]], fit_function, initialparams = initialparams, bounds=bounds, ifplot = ifplot)
#         params_err_blockavg[rep-1][:] = np.sqrt(np.diag(params_covariance_blockavg[rep-1]))
#
#         for bl in range(1, Nblocks + 1):
#             params[rep-1][bl-1][:], params_covariance[rep-1][bl-1][:][:] = fit_traces(edges[rep-1][startidx:idxmin[rep-1]],
#                                                                                       datalist[rep-1][bl-1,startidx:idxmin[rep-1]], fit_function, initialparams = initialparams, bounds=bounds, ifplot = ifplot)
#             params_err[rep-1][bl-1][:] = np.sqrt(np.diag(params_covariance[rep-1][bl-1]))
#
#     return t_before_nov, params_blockavg, params_covariance_blockavg, params_err_blockavg, params, params_covariance, params_err

def get_baseline_firing_rate(edges,datalist, meandatalist, lenstim, lenpause, Nreps, Nimg, Nblocks, ifseqlen = False, ifrepseq = False, Nseq = 5, avgindices = 20, idxconv = 4):
    """ get the baseline firing rate by averaging the avgindices before the novelty onset
    get baseline of averaged traces and each trace individually
    determine the avg and std of the individual trace baseline values """
    if ifseqlen:
        Nseq = len(Nimg)
    elif ifrepseq:
        Nseq = Nseq
    else:
        Nseq = len(Nreps)

    baseline = np.zeros((Nseq,Nblocks))
    baseline_avg = np.zeros(Nseq)
    t_before_nov = np.zeros(Nseq)
    idxmin = np.zeros(Nseq).astype(int) # cut off index before novelty

    for rep in range(1,Nseq + 1):
        #t_before_nov[rep-1] = edges[rep-1][-1] - 7. - (Nreps[rep-1]+1)*lenstim/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
        if ifseqlen:
            t_before_nov[rep-1] = ((Nreps-2)*Nimg[rep-1]*(lenstim+lenpause) + (Nimg[rep-1]-3)*(lenstim+lenpause))/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
        elif ifrepseq:
            t_before_nov[rep-1] = ((Nreps-2)*Nimg*(lenstim+lenpause) + (Nimg-3)*(lenstim+lenpause))/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
        else:
            t_before_nov[rep-1] = ((Nreps[rep-1]-2)*Nimg*(lenstim+lenpause) + (Nimg-3)*(lenstim+lenpause))/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)

        idxmin[rep-1] = np.argmax(edges[rep-1]>t_before_nov[rep-1])
        baseline_avg[rep-1] = np.mean(meandatalist[rep-1][max(idxconv,idxmin[rep-1]-avgindices):idxmin[rep-1]])
        for bl in range(1, Nblocks + 1):
            baseline[rep-1][bl-1] = np.mean(datalist[rep-1][bl-1,max(idxconv,idxmin[rep-1]-avgindices):idxmin[rep-1]])

    mean_baseline = np.mean(baseline, axis = 1)
    std_baseline = np.std(baseline, axis = 1)

    return baseline_avg, baseline, mean_baseline, std_baseline

# def get_baseline_firing_rate(edges,datalist, meandatalist, lenstim, lenpause, Nreps, Nimg, Nblocks, avgindices = 20, startimg = 4, idxconv = 4):
#     """ get the baseline firing rate by averaging the avgindices before the novelty onset
#     get baseline of averaged traces and each trace individually
#     determine the avg and std of the individual trace baseline values """
#     Nseq = len(Nreps)
#     baseline = np.zeros((Nseq,Nblocks))
#     baseline_avg = np.zeros(Nseq)
#     t_before_nov = np.zeros(Nseq)
#     idxmin = np.zeros(Nseq).astype(int) # cut off index before novelty
#     t_start_img = startimg*(lenstim+lenpause)/1000.
#
#     for rep in range(1,Nseq + 1):
#         #t_before_nov[rep-1] = edges[rep-1][-1] - 7. - (Nreps[rep-1]+1)*lenstim/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
#         t_before_nov[rep-1] = ((Nreps[rep-1]-2)*Nimg*(lenstim+lenpause) + (Nimg-3)*(lenstim+lenpause))/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
#         idxmin[rep-1] = np.argmax(edges[rep-1]>t_before_nov[rep-1])
#         baseline_avg[rep-1] = np.mean(meandatalist[rep-1][max(idxconv,idxmin[rep-1]-avgindices):idxmin[rep-1]])
#         for bl in range(1, Nblocks + 1):
#             baseline[rep-1][bl-1] = np.mean(datalist[rep-1][bl-1,max(idxconv,idxmin[rep-1]-avgindices):idxmin[rep-1]])
#
#     mean_baseline = np.mean(baseline, axis = 1)
#     std_baseline = np.std(baseline, axis = 1)
#
#     return baseline_avg, baseline, mean_baseline, std_baseline

def fit_variable_repetitions_gen_arrays_average_postnovelty(edges,meandatalist,  lenstim, lenpause, Nreps, Nimg, Nblocks, fit_function = exp_with_offset, avgindices = 20, initialparams = [10, 6, 3], bounds=(0, [10., 140., 10]), ifplot = False, startimg = 4, idxconv = 4):
    """ perform fitting of all traces included in datalist and meandatalist
    input:  edges (timevector)
            data lists
            duration of stimulus and length of the pause
            Nrepetitions array
            number of images per sequence
            number of blocks
            initial paramteres for the fits
            number of samples in the baseline average

            ensure only values above convolution idx are considered
            set startindex to index when N images were played once"""
    Nseq = len(meandatalist)
    #params = np.zeros((Nseq,Nblocks,3)) # params: a, tau, a_offset
    #params_covariance = np.zeros((Nseq,Nblocks,3,3)) # covariance matrices of fit params
    #params_err = np.zeros((Nseq,Nblocks,3)) # errors of fit params
    # initialise parameter arrays for block averages
    params_blockavg = np.zeros((Nseq,3))
    params_covariance_blockavg = np.zeros((Nseq,3,3))
    params_err_blockavg = np.zeros((Nseq,3))
    t_before_trans = np.zeros(Nseq)
    idxstop = -idxconv#
    startidx = np.zeros(Nseq).astype(int) # cut off index before novelty
    #baseline = np.zeros((Nseq,Nblocks))
    #baseline_avg = np.zeros(Nseq)
    t_start_img = startimg*(lenstim+lenpause)/1000.

    #startidx = np.argmax(edges[0]>t_start_img)
    #print(startidx)
    #startidx = max([startidx, idxconv]) # select maximum value between convolution cutoff and minimum for fitting

    #print(avgindices)
    for rep in range(1,len(Nreps) + 1):
        #t_before_nov[rep-1] = edges[rep-1][-1] - 7. - (Nreps[rep-1]+1)*lenstim/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
        t_before_trans[rep-1] = ((Nreps[rep-1])*Nimg*(lenstim+lenpause))/1000. # Nimg*Nreps*(lenstim + lenpause)

        #print(t_before_nov[rep-1])
        startidx[rep-1] = np.argmax(edges>t_before_trans[rep-1] + t_start_img) # add start image time to end of first adaptation block
        params_blockavg[rep-1][:], params_covariance_blockavg[rep-1][:][:] = fit_traces(edges[startidx[rep-1]:idxstop],
                                                                                        meandatalist[rep-1][startidx[rep-1]:idxstop], fit_function, initialparams = initialparams, bounds=bounds, ifplot = ifplot)
        params_err_blockavg[rep-1][:] = np.sqrt(np.diag(params_covariance_blockavg[rep-1]))
        #baseline_avg[rep-1] = np.mean(meandatalist[rep-1][max(idxconv,idxmin[rep-1]-avgindices):idxmin[rep-1]])

    return t_before_trans, params_blockavg, params_err_blockavg

def fit_variable_repetitions_gen_arrays_postnovelty(edges,datalist, meandatalist, lenstim, lenpause, Nreps, Nimg, Nblocks, fit_function = exp_with_offset, avgindices = 20, initialparams = [10, 6, 3], bounds=(0, [10., 140., 10]), ifplot = False, startimg = 4, idxconv = 4):
    """ perform fitting of all traces included in datalist and meandatalist
    input:  edges (timevector)
            data lists
            duration of stimulus and length of the pause
            Nrepetitions array
            number of images per sequence
            number of blocks
            initial paramteres for the fits
            number of samples in the baseline average

            ensure only values above convolution idx are considered
            set startindex to index when N images were played once"""
    Nseq = len(Nreps)
    params = np.zeros((Nseq,Nblocks,3)) # params: a, tau, a_offset
    params_covariance = np.zeros((Nseq,Nblocks,3,3)) # covariance matrices of fit params
    params_err = np.zeros((Nseq,Nblocks,3)) # errors of fit params
    # initialise parameter arrays for block averages
    params_blockavg = np.zeros((Nseq,3))
    params_covariance_blockavg = np.zeros((Nseq,3,3))
    params_err_blockavg = np.zeros((Nseq,3))
    t_before_trans = np.zeros(Nseq)
    idxstop = -idxconv#
    startidx = np.zeros(Nseq).astype(int) # cut off index before novelty
    #baseline = np.zeros((Nseq,Nblocks))
    #baseline_avg = np.zeros(Nseq)
    t_start_img = startimg*(lenstim+lenpause)/1000.

    #startidx = np.argmax(edges[0]>t_start_img)
    #print(startidx)
    #startidx = max([startidx, idxconv]) # select maximum value between convolution cutoff and minimum for fitting

    #print(avgindices)
    for rep in range(1,len(Nreps) + 1):
        #t_before_nov[rep-1] = edges[rep-1][-1] - 7. - (Nreps[rep-1]+1)*lenstim/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
        t_before_trans[rep-1] = ((Nreps[rep-1])*Nimg*(lenstim+lenpause))/1000. # Nimg*Nreps*(lenstim + lenpause)

        #print(t_before_nov[rep-1])
        startidx[rep-1] = np.argmax(edges[rep-1]>t_before_trans[rep-1] + t_start_img) # add start image time to end of first adaptation block
        #samples = len(edges[rep-1])-startidx[rep-1] # samples of the extra 20 secs
        #idxstop = startidx[rep-1] + int(np.round(samples/2)) # restrict fitting to 10 secs
        params_blockavg[rep-1][:], params_covariance_blockavg[rep-1][:][:] = fit_traces(edges[rep-1][startidx[rep-1]:idxstop],
                                                                                        meandatalist[rep-1][startidx[rep-1]:idxstop], fit_function, initialparams = initialparams, bounds=bounds, ifplot = ifplot)
        params_err_blockavg[rep-1][:] = np.sqrt(np.diag(params_covariance_blockavg[rep-1]))
        #baseline_avg[rep-1] = np.mean(meandatalist[rep-1][max(idxconv,idxmin[rep-1]-avgindices):idxmin[rep-1]])

        for bl in range(1, Nblocks + 1):
            params[rep-1][bl-1][:], params_covariance[rep-1][bl-1][:][:] = fit_traces(edges[rep-1][startidx[rep-1]:idxstop],
                                                                                      datalist[rep-1][bl-1,startidx[rep-1]:idxstop], fit_function, initialparams = initialparams, bounds=bounds, ifplot = False)
            params_err[rep-1][bl-1][:] = np.sqrt(np.diag(params_covariance[rep-1][bl-1]))
            #baseline[rep-1][bl-1] = np.mean(datalist[rep-1][bl-1,max(idxconv,idxmin[rep-1]-avgindices):idxmin[rep-1]])

    return t_before_trans, params_blockavg, params_err_blockavg, params, params_err

def fit_traces(time, trace, fit_function, initialparams = [10, 6, 3], ifplot = False, saveplot = False, figure_directory = "./", title = "Fit_PopulationAverages_E", img = 1, bounds=(0, [10., 140., 10])):
    # use optimize curvefit function to fit the data
    #Constrain the optimization to the region of 0 <= a <= 10, 0 <= tau <= 20 and 0 <= offset <= 10:
    params, params_covariance = optimize.curve_fit(fit_function, time, trace, p0 = initialparams, bounds=bounds)
    #print(params)
    if ifplot:
        fig = plt.figure(figsize=(20, 10))
        #plt.plot(time, trace, label='Data')
        plot_popavg_mult(fig,time, trace, legend = "Data", iflegend = False, color = "darkblue", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [s]", ylabel ="$\\rho$ [Hz]", ifioff = False)
        plt.plot(time, fit_function(time, params[0], params[1], params[2]),
                 label='fit: a=%5.3f Hz, $\\tau$=%5.3f s, $a_{off}$=%5.3f Hz' % tuple(params), color = "red", lw = 3) #label='fit: a=%5.3f, tau=%5.3f, offset=%5.3f' % tuple(params))
        plt.legend(loc='best', frameon = False, fontsize = 20)
        #plt.show()
        #plt.ylim([2,4])
        if saveplot:
            save_fig(figure_directory, title + str(img))
        #plt.show()
    return params, params_covariance

def fit_traces_two(time, trace, fit_function, initialparams = [10, 6], ifplot = False, saveplot = False, figure_directory = "./", title = "Fit_PopulationAverages_E", img = 1, bounds=(0, [10., 140.])):
    # use optimize curvefit function to fit the data
    #Constrain the optimization to the region of 0 <= a <= 10, 0 <= tau <= 20 and 0 <= offset <= 10:
    params, params_covariance = optimize.curve_fit(fit_function, time, trace, p0 = initialparams, bounds=bounds)
    #print(params)
    if ifplot:
        fig = plt.figure(figsize=(20, 10))
        #plt.plot(time, trace, label='Data')
        plot_popavg_mult(fig,time, trace, legend = "Data", iflegend = False, color = "darkblue", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [s]", ylabel ="$\\rho$ [Hz]", ifioff = False)
        plt.plot(time, fit_function(time, params[0], params[1]),
                 label='fit params:' + str(tuple(params)), color = "red", lw = 3) #label='fit: a=%5.3f, tau=%5.3f, offset=%5.3f' % tuple(params))
        plt.legend(loc='best', frameon = False, fontsize = 20)
        #plt.show()
        #plt.ylim([2,4])
        if saveplot:
            save_fig(figure_directory, title + str(img))
        #plt.show()
    return params, params_covariance

def fit_traces_save(time, trace, fit_function, initialparams = [10, 6, 3], ifplot = False, saveplot = False, figure_directory = "./", title = "Fit_PopulationAverages_E", img = 1, bounds=(0, [10., 140., 10])):
    # use optimize curvefit function to fit the data
    #Constrain the optimization to the region of 0 <= a <= 10, 0 <= tau <= 20 and 0 <= offset <= 10:
    params, params_covariance = optimize.curve_fit(fit_function, time, trace, p0 = initialparams, bounds=bounds)
    #print(params)
    if ifplot:
        fig = plt.figure(figsize=(20, 10))
        #plt.plot(time, trace, label='Data')
        plot_popavg_mult(fig,time, trace, legend = "Data", iflegend = False, color = "darkblue", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [s]", ylabel ="$\\rho$ [Hz]", ifioff = False)
        plt.plot(time, fit_function(time, params[0], params[1], params[2]),
                 label='fit: a=%5.3f Hz, $\\tau$=%5.3f s, $a_{off}$=%5.3f Hz' % tuple(params), color = "red", lw = 3) #label='fit: a=%5.3f, tau=%5.3f, offset=%5.3f' % tuple(params))
        plt.legend(loc='best', frameon = False, fontsize = 20)
        #plt.show()
        #plt.ylim([2,4])
        if saveplot:
            save_fig(figure_directory, title + str(img))
        #plt.show()
    return params, params_covariance

def plot_Nreps_tau(Nreps, params, params_blockavg, Nblocks = 10, ifxlims = True,figure_directory = "./",
 figsize=(14, 10), xlims = [0,45], legend = "E", ifxticks = True, xtickstepsize = 3,
  iflegend = False, color = ["r","g","b","k","c","m","r","g","b","k","c","m"], lw = 3,fontsize = 24,
   xlabel = "number of repetitions", ylabel = "decay constant $\\tau$ [s]",
    ifioff = True, ifsavefig = True, savename = "NrepsTau", axiswidth = 1):
        """ plot population average one figure a time"""
        if ifioff:
            plt.ioff()
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        plt.plot(Nreps[0:],params_blockavg[0:,1], ":x", color = "red", label = "$\\tau$ block averages")
        for seq in range(1,len(Nreps)+1):
            plt.plot([Nreps[seq-1]]*Nblocks, params[seq-1,:,1], "o", color = "grey")#color[seq-1])
        plt.plot(max(Nreps)+10,0, "o", color = "grey", label = "$\\tau$ individual blocks")

        plt.xlabel(xlabel, fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)
        if ifxticks:
            plt.xticks(Nreps[0::xtickstepsize],fontsize = fontsize)
        else:
            plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.legend(fontsize = fontsize, frameon = True)
        if ifxlims:
            xlims = [min(Nreps)-1,max(Nreps)+1]
            plt.xlim(xlims)
        plt.tight_layout()
        if ifsavefig:
            save_fig(figure_directory, savename)

def convert_tau_avg(params_blockavg,params_blockavg_err):
    return params_blockavg[:,1], params_blockavg_err[:,1]

def convert_tau(params,params_err):
    return params[:,:,1], params_err[:,:,1]


def plot_Nreps_baseline(Nreps, params, params_blockavg, Nblocks = 10, ifxlims = True, figure_directory = "./",
 figsize=(14, 10), xlims = [0,45], legend = "E", ifxticks = True, xtickstepsize = 3,
  iflegend = False, color = ["r","g","b","k","c","m","r","g","b","k","c","m"], lw = 3,fontsize = 24,
   xlabel = "number of repetitions", ylabel ="baseline firing rate [Hz]",
    ifioff = True, ifsavefig = True, savename = "NrepsNaseline", axiswidth = 1):
        """ plot population average one figure a time"""
        if ifioff:
            plt.ioff()
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        plt.plot(Nreps[0:],params_blockavg[0:,2], ":x", color = "red",  label = "$\\rho$ block averages")
        for seq in range(1,len(Nreps)+1):
            plt.plot([Nreps[seq-1]]*Nblocks, params[seq-1,:,2], "o", color = "grey")#color[seq-1])
        plt.plot(max(Nreps)+10,0, "o", color = "grey", label = "$\\rho$ individual blocks")

        plt.xlabel(xlabel, fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)
        if ifxticks:
            plt.xticks(Nreps[0::xtickstepsize],fontsize = fontsize)
        else:
            plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.legend(fontsize = fontsize, frameon = True)
        if ifxlims:
            xlims = [min(Nreps)-1,max(Nreps)+1]
            plt.xlim(xlims)
        plt.tight_layout()

        if ifsavefig:
            save_fig(figure_directory, savename)

# def get_novelty_peak_height(edges,datalist, meandatalist, lenstim, lenpause, Nreps, Nimg, Nblocks, avgindices = 20, startimg = 4, idxconv = 4, search_margin = 20):
#     """ get the novelty firing rate by finding the maximum value around novelty time
#     in averaged traces and each block trace individually
#     determine the avg and std of the individual trace novelty values """
#     Nseq = len(Nreps)
#     novelty = np.zeros((Nseq,Nblocks))
#     novelty_avg = np.zeros(Nseq)
#     t_before_nov = np.zeros(Nseq)
#     noveltyidx = np.zeros((Nseq,Nblocks))
#     novelty_avgidx = np.zeros(Nseq)
#     idxmin = np.zeros(Nseq).astype(int) # cut off index before novelty
#     #t_start_img = startimg*(lenstim+lenpause)/1000.
#
#     for rep in range(1,Nseq + 1):
#         #t_before_nov[rep-1] = edges[rep-1][-1] - 7. - (Nreps[rep-1]+1)*lenstim/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
#         t_before_nov[rep-1] = ((Nreps[rep-1]-2)*Nimg*(lenstim+lenpause) + (Nimg-0.5)*(lenstim+lenpause))/1000. # final t - 7 sec offset - Nreps * Nreps * (lenstim + lenpause)
#         idxmin[rep-1] = np.argmax(edges[rep-1]>t_before_nov[rep-1])
#         novelty_avgidx[rep-1] = np.argmax(meandatalist[rep-1][max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)]) + max(0,idxmin[rep-1]-search_margin)
#         novelty_avg[rep-1] = max(meandatalist[rep-1][max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)])
#
#         #novelty_avg[rep-1] = np.mean(meandatalist[rep-1][max(idxconv,idxmin[rep-1]-avgindices):idxmin[rep-1]])
#         for bl in range(1, Nblocks + 1):
#             noveltyidx[rep-1][bl-1] = np.argmax(datalist[rep-1][bl-1,max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)]) + max(0,idxmin[rep-1]-search_margin)
#             novelty[rep-1][bl-1] = max(datalist[rep-1][bl-1,max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)])
#
#     mean_novelty = np.mean(novelty, axis = 1)
#     std_novelty = np.std(novelty, axis = 1)
#
#     # plot one example to check if it works fine
#     plt.figure()
#     plt.plot(meandatalist[Nseq-1])
#     plt.plot(novelty_avgidx[Nseq-1], novelty_avg[Nseq-1], "x")
#
#     return novelty_avg, novelty, mean_novelty, std_novelty, novelty_avgidx, noveltyidx

def get_peak_height(edges,datalist, meandatalist, lenstim, lenpause, Nreps, Nimg, Nblocks,
 iftransientpre = False, iftransientpost = False, ifseqlen=False, ifrepseq = False,
  avgindices = 20, Nseq = 5, startimg = 4, idxconv = 4, search_margin = 20):
    """ get the novelty firing rate by finding the maximum value around novelty time
    in averaged traces and each block trace individually
    determine the avg and std of the individual trace novelty values

    switch between novelty, pre novelty transient and post novelty transient peaks
    default = novelty peak
    iftransientpre pre novelty peak
    iftransientpost post novelty peak"""
    # TO DO INCLUE SEQLEN
    if ifseqlen:
        Nseq = len(Nimg)
    elif ifrepseq:
        Nseq = Nseq
    else:
        Nseq = len(Nreps)
    #Nseq = len(Nreps)
    novelty = np.zeros((Nseq,Nblocks)) # get the peak per block per Nreps/Nseqlen
    novelty_avg = np.zeros(Nseq) # get the peak of average over blocks per Nreps/Nseqlen

    noveltyidx = np.zeros((Nseq,Nblocks)) # index of respective peak
    novelty_avgidx = np.zeros(Nseq) # index of repective peak in average

    t_before_nov = np.zeros(Nseq) # time just before the peak (before novelty, before last img of first sequnece new/old)
    idxmin = np.zeros(Nseq).astype(int) # cut off index before novelty
    #t_start_img = startimg*(lenstim+lenpause)/1000.
    if ifseqlen:
        print("SEQLEN ON")
        for rep in range(1,Nseq + 1):
            # define region to search depending on condition pre, post, novelty
            if iftransientpre:
                t_before_nov[rep-1] = ((Nimg[rep-1]-0.5)*(lenstim+lenpause))/1000.
                print("iftransientpre")

            elif iftransientpost:
                t_before_nov[rep-1] = (Nreps*Nimg[rep-1]*(lenstim+lenpause) + (Nimg[rep-1]-0.5)*(lenstim+lenpause))/1000. # middle of last image in new sequence
                print("iftransientpost")

            else:
                #t_before_nov[rep-1] = ((Nreps[rep-1]-2)*Nimg*(lenstim+lenpause) + (Nimg-0.5)*(lenstim+lenpause))/1000. # middle of novelty image
                t_before_nov[rep-1] = ((Nreps-2)*Nimg[rep-1]*(lenstim+lenpause) + (Nimg[rep-1]-0.5)*(lenstim+lenpause))/1000. # middle of novelty image
                print("novelty")
            #print(t_before_nov[rep-1])

            #print(meandatalist[rep-1][max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)] + max(0,idxmin[rep-1]-search_margin))
            idxmin[rep-1] = np.argmax(edges[rep-1]>t_before_nov[rep-1])
            novelty_avgidx[rep-1] = np.argmax(meandatalist[rep-1][max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)]) + max(0,idxmin[rep-1]-search_margin)
            #print(novelty_avgidx[rep-1])

            novelty_avg[rep-1] = max(meandatalist[rep-1][max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)])
            #print(novelty_avg[rep-1])
            #novelty_avg[rep-1] = np.mean(meandatalist[rep-1][max(idxconv,idxmin[rep-1]-avgindices):idxmin[rep-1]])
            for bl in range(1, Nblocks + 1):
                noveltyidx[rep-1][bl-1] = np.argmax(datalist[rep-1][bl-1,max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)]) + max(0,idxmin[rep-1]-search_margin)
                novelty[rep-1][bl-1] = max(datalist[rep-1][bl-1,max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)])
    elif ifrepseq:
        print("Repeated sequences on")
        for rep in range(1,Nseq + 1):
            # define region to search depending on condition pre, post, novelty
            if iftransientpre:
                t_before_nov[rep-1] = ((Nimg-0.5)*(lenstim+lenpause))/1000.
                print("iftransientpre")

            elif iftransientpost:
                t_before_nov[rep-1] = (Nreps*Nimg*(lenstim+lenpause) + (Nimg-0.5)*(lenstim+lenpause))/1000. # middle of last image in new sequence
                print("iftransientpost")

            else:
                #t_before_nov[rep-1] = ((Nreps[rep-1]-2)*Nimg*(lenstim+lenpause) + (Nimg-0.5)*(lenstim+lenpause))/1000. # middle of novelty image
                t_before_nov[rep-1] = ((Nreps-2)*Nimg*(lenstim+lenpause) + (Nimg-0.5)*(lenstim+lenpause))/1000. # middle of novelty image
                print("novelty")
            #print(t_before_nov[rep-1])

            #print(meandatalist[rep-1][max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)] + max(0,idxmin[rep-1]-search_margin))
            idxmin[rep-1] = np.argmax(edges[rep-1]>t_before_nov[rep-1])
            novelty_avgidx[rep-1] = np.argmax(meandatalist[rep-1][max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)]) + max(0,idxmin[rep-1]-search_margin)
            #print(novelty_avgidx[rep-1])

            novelty_avg[rep-1] = max(meandatalist[rep-1][max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)])
            #print(novelty_avg[rep-1])
            #novelty_avg[rep-1] = np.mean(meandatalist[rep-1][max(idxconv,idxmin[rep-1]-avgindices):idxmin[rep-1]])
            for bl in range(1, Nblocks + 1):
                noveltyidx[rep-1][bl-1] = np.argmax(datalist[rep-1][bl-1,max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)]) + max(0,idxmin[rep-1]-search_margin)
                novelty[rep-1][bl-1] = max(datalist[rep-1][bl-1,max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)])

    else:
        for rep in range(1,Nseq + 1):
            # define region to search depending on condition pre, post, novelty
            if iftransientpre:
                t_before_nov[rep-1] = ((Nimg-0.5)*(lenstim+lenpause))/1000. # middle of last image in current sequence
                #search_margin = 1
            elif iftransientpost:
                t_before_nov[rep-1] = (Nreps[rep-1]*Nimg*(lenstim+lenpause) + (Nimg-0.5)*(lenstim+lenpause))/1000. # middle of last image in new sequence
                #search_margin = 1
            else:
                t_before_nov[rep-1] = ((Nreps[rep-1]-2)*Nimg*(lenstim+lenpause) + (Nimg-0.5)*(lenstim+lenpause))/1000. # middle of novelty image
                #print("novelty")
            #print(t_before_nov[rep-1])
            idxmin[rep-1] = np.argmax(edges[rep-1]>t_before_nov[rep-1])
            novelty_avgidx[rep-1] = np.argmax(meandatalist[rep-1][max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)]) + max(0,idxmin[rep-1]-search_margin)
            novelty_avg[rep-1] = max(meandatalist[rep-1][max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)])

            #novelty_avg[rep-1] = np.mean(meandatalist[rep-1][max(idxconv,idxmin[rep-1]-avgindices):idxmin[rep-1]])
            for bl in range(1, Nblocks + 1):
                noveltyidx[rep-1][bl-1] = np.argmax(datalist[rep-1][bl-1,max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)]) + max(0,idxmin[rep-1]-search_margin)
                novelty[rep-1][bl-1] = max(datalist[rep-1][bl-1,max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)])


    mean_novelty = np.mean(novelty, axis = 1)
    std_novelty = np.std(novelty, axis = 1)

    # plot one example to check if it works fine
    # plt.figure()
    # plt.plot(meandatalist[Nseq-1])
    # plt.plot(novelty_avgidx[Nseq-1], novelty_avg[Nseq-1], "x")

    return novelty_avg, novelty, mean_novelty, std_novelty, novelty_avgidx, noveltyidx




def plot_Nreps_array(Nreps, array, array_blockavg, Nblocks = 10, ifxlims = True, figure_directory = "./",
 figsize=(14, 10), xlims = [0,45], legend = "E", ifxticks = True, xtickstepsize = 3,
  iflegend = False, c1 = "grey",c3 = "darkgreen", c2 = "darkorange",lw = 3,fontsize = 24,
   xlabel = "number of repetitions", ylabel ="rate [Hz]",
    ifioff = True, ifsavefig = True, savename = "NrepsArray", axiswidth = 1):
        """ plot Narrays vs. array (Nreps,Nblocks) array_blockavg (Nreps)"""
        if ifioff:
            plt.ioff()
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        for seq in range(1,len(Nreps)+1):
            plt.plot([Nreps[seq-1]]*Nblocks, array[seq-1,:], "o", color = c1)
        plt.plot(Nreps,array_blockavg, ":o", color = c2,  label = "block average")
        plt.plot(Nreps,np.mean(array, axis = 1), ":o", color = c3,  label = "mean individual blocks")

        plt.xlabel(xlabel, fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)
        if ifxticks:
            plt.xticks(Nreps[0::xtickstepsize],fontsize = fontsize)
        else:
            plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.legend(fontsize = fontsize, frameon = True)
        if ifxlims:
            xlims = [min(Nreps)-1,max(Nreps)+1]
            plt.xlim(xlims)
        plt.tight_layout()
        if ifsavefig:
            save_fig(figure_directory, savename)

def plot_Nreps_array_errorband(Nreps, array, array_blockavg, Nblocks = 10, ifxlims = True, figure_directory = "./",
 figsize=(14, 10), xlims = [0,45], alpha = 0.2, legend = "E", ifxticks = True, xtickstepsize = 3,
  iflegend = False, c1 = "grey",c3 = "darkgreen", c2 = "darkorange",lw = 3,fontsize = 24,
   xlabel = "number of repetitions", ylabel ="rate [Hz]",
    ifioff = True, ifsavefig = True, savename = "NrepsArrayerrorband", axiswidth = 1):
        """ plot Narrays vs. array (Nreps,Nblocks) array_blockavg (Nreps)"""
        if ifioff:
            plt.ioff()
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        meanarr = np.mean(array, axis = 1)
        stdarr = np.std(array, axis = 1)

        plt.plot(Nreps,array_blockavg, ":o", color = c2,  label = "block average")
        plt.plot(Nreps,meanarr, ":o", color = c3,  label = "mean individual blocks")
        plt.fill_between(Nreps, meanarr-stdarr, meanarr+stdarr, alpha=alpha, edgecolor=c3, facecolor=c3)

        plt.xlabel(xlabel, fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)
        if ifxticks:
            plt.xticks(Nreps[0::xtickstepsize],fontsize = fontsize)
        else:
            plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.legend(fontsize = fontsize, frameon = True)
        if ifxlims:
            xlims = [min(Nreps)-1,max(Nreps)+1]
            plt.xlim(xlims)
        plt.tight_layout()
        if ifsavefig:
            save_fig(figure_directory, savename)

def plot_Nreps_array_errorbar(Nreps, array, array_blockavg, Nblocks = 10, ifxlims = True, figure_directory = "./",
 figsize=(14, 10), xlims = [0,45], alpha = 0.2, legend = "E", ifxticks = True, xtickstepsize = 3,
  iflegend = False, c1 = "grey",c3 = "darkgreen", c2 = "darkorange",lw = 3,fontsize = 24,
   xlabel = "number of repetitions", ylabel ="rate [Hz]",
    ifioff = True, ifsavefig = True, savename = "NrepsArrayerrorband", axiswidth = 1):
        """ plot Narrays vs. array (Nreps,Nblocks) array_blockavg (Nreps)"""
        if ifioff:
            plt.ioff()
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        meanarr = np.mean(array, axis = 1)
        stdarr = np.std(array, axis = 1)

        plt.plot(Nreps,array_blockavg, ":o", color = c2,  label = "block average")
        #plt.plot(Nreps,meanarr, ":o", color = c3,  label = "mean individual blocks")
        #plt.fill_between(Nreps, meanarr-stdarr, meanarr+stdarr, alpha=alpha edgecolor=c3, facecolor=c3)
        plt.errorbar(Nreps, meanarr, yerr=stdarr, fmt='o', color = c3,
             ecolor=c3, elinewidth=2, capsize=3, label = "mean individual blocks")
        plt.xlabel(xlabel, fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)
        if ifxticks:
            plt.xticks(Nreps[0::xtickstepsize],fontsize = fontsize)
        else:
            plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.legend(fontsize = fontsize, frameon = True)
        plt.tight_layout()
        if ifxlims:
            xlims = [min(Nreps)-1,max(Nreps)+1]
            plt.xlim(xlims)
        if ifsavefig:
            save_fig(figure_directory, savename)

def barplot_Nreps_array(Nreps, array, array_blockavg, Nblocks = 10, ifxlims = True, figure_directory = "./",
 figsize=(14, 10), xlims = [0,45], alpha = 0.2, legend = "E", ifxticks = True, xtickstepsize = 3,
  iflegend = False, c1 = "grey",c3 = "darkgreen", c2 = "darkorange",lw = 3,fontsize = 24,
   xlabel = "number of repetitions", ylabel ="rate [Hz]",
    ifioff = True, ifsavefig = True, savename = "NrepsArrayerrorband", axiswidth = 1):
        """ plot Narrays vs. array (Nreps,Nblocks) array_blockavg (Nreps)"""
        if ifioff:
            plt.ioff()
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        meanarr = np.mean(array, axis = 1)
        stdarr = np.std(array, axis = 1)
        plt.bar(Nreps,meanarr,color = c3, edgecolor=c3, yerr = stdarr, align='center', alpha=alpha)
        #
        # plt.plot(Nreps,array_blockavg, ":o", color = c2,  label = "block average")
        # #plt.plot(Nreps,meanarr, ":o", color = c3,  label = "mean individual blocks")
        # #plt.fill_between(Nreps, meanarr-stdarr, meanarr+stdarr, alpha=alpha edgecolor=c3, facecolor=c3)
        # plt.errorbar(Nreps, meanarr, yerr=stdarr, fmt='o', color = c3,
        #      ecolor=c3, elinewidth=2, capsize=3, label = "mean individual blocks")
        plt.xlabel(xlabel, fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)
        if ifxticks:
            plt.xticks(Nreps[0::xtickstepsize],fontsize = fontsize)
        else:
            plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.legend(fontsize = fontsize, frameon = True)
        plt.tight_layout()
        if ifxlims:
            xlims = [min(Nreps)-1,max(Nreps)+1]
            plt.xlim(xlims)
        if ifsavefig:
            save_fig(figure_directory, savename)

def plot_mean_with_errorband_mult(fig,time, data, error, noveltyonset = 22.5, legend = "E", iflegend = False, color = "darkblue", ifcolor = False,
                 lw = 3, xlabel = "time [s]", ylabel ="z-score",fontsize = 24, ifxpositions = False, ifaxvline = False, x_positions = [5,10,15,20], x_labels = ["5","10","15","20"],
                              ifioff = True, alpha=0.2, axiswidth = 1, Nseq = 5):
    """ plot the mean with +- std in one figure fig (def before)"""
    #fig = plt.figure(figsize=(20, 10)) #run first and then conse
    if ifioff:
        plt.ioff()

    ax = fig.add_subplot(111)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    ifcolor = True
    if iflegend:
        if ifcolor:
            plt.plot(time, data, label = legend, color = color, lw = lw)
            plt.fill_between(time, data-error, data+error,alpha=alpha, edgecolor=color, facecolor=color)
            plt.xlabel(xlabel, fontsize = fontsize)
            plt.ylabel(ylabel, fontsize = fontsize)
            plt.xticks(fontsize = fontsize)
            plt.yticks(fontsize = fontsize)
            plt.legend(fontsize = fontsize, frameon = False)
        else:
            plt.plot(time, data, label = legend, color = color, lw = lw)
            plt.fill_between(time, data-error, data+error,alpha=alpha, edgecolor=color, facecolor=color)
            plt.xlabel(xlabel, fontsize = fontsize)
            plt.ylabel(ylabel, fontsize = fontsize)
            plt.legend(fontsize = fontsize, frameon = False)
            plt.xticks(fontsize = fontsize)
            plt.yticks(fontsize = fontsize)
    else:
        if ifcolor:
            plt.plot(time, data, label = legend, color = color, lw = lw)
            plt.fill_between(time, data-error, data+error,alpha=alpha, edgecolor=color, facecolor=color)
            plt.xlabel(xlabel, fontsize = fontsize)
            plt.ylabel(ylabel, fontsize = fontsize)
            plt.xticks(fontsize = fontsize)
            plt.yticks(fontsize = fontsize)
        else:
            plt.plot(time, data, label = legend, color = color, lw = lw)
            plt.fill_between(time, data-error, data+error,alpha=alpha, edgecolor=color, facecolor=color)
            plt.xlabel(xlabel, fontsize = fontsize)
            plt.ylabel(ylabel, fontsize = fontsize)
            plt.xticks(fontsize = fontsize)
            plt.yticks(fontsize = fontsize)
    if ifxpositions:
        plt.xticks(x_positions, x_labels,fontsize = fontsize)
    if ifaxvline:
        ax.axvline(x=noveltyonset,color ="k")#,lw = 3, "k")
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)
    plt.tight_layout()


#     return transient_avg, transient, mean_transient, std_transient, transient_avgidx, transientidx
def plot_Nreps_array_comparison(Nreps, array, Nfiles = 5, ifxlims = True, figure_directory = "./",
 figsize=(14, 10), xlims = [0,45], alpha = 0.2, legend = "E", ifxticks = True, xtickstepsize = 3,
  iflegend = False, c1 = "grey",c3 = "darkgreen", c2 = "darkorange",lw = 3,fontsize = 24,
   xlabel = "number of repetitions", ylabel ="rate [Hz]",
    ifioff = True, ifsavefig = True, savename = "NrepsArrayerrorband", axiswidth = 1):
        """ plot Narrays vs. array (Nreps,Nblocks) array_blockavg (Nreps)"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        meanarr = np.mean(array, axis = 0)
        stdarr = np.std(array, axis = 0)

        plt.plot(Nreps,meanarr, ":o", color = c3,  label = "mean individual blocks")
        plt.fill_between(Nreps, meanarr-stdarr, meanarr+stdarr, alpha=alpha, edgecolor=c3, facecolor=c3)

        plt.xlabel(xlabel, fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)
        if ifxticks:
            plt.xticks(Nreps[0::xtickstepsize],fontsize = fontsize)
        else:
            plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.tight_layout()
        #plt.show()
        #plt.legend(fontsize = fontsize, frameon = True)
        if ifxlims:
            xlims = [min(Nreps)-1,max(Nreps)+1]
            plt.xlim(xlims)
        if ifsavefig:
            for fn in range(0,len(figure_directory)):
                save_fig(figure_directory[fn], savename)

def barplot_Nreps_array_comparison(Nreps, array, Nfiles = 5, ifxlims = True, figure_directory = "./",
 figsize=(14, 10), xlims = [0,45], alpha = 0.2, legend = "E", ifxticks = True, xtickstepsize = 3,
  iflegend = False, c1 = "grey",c3 = "darkgreen", c2 = "darkorange",lw = 3,fontsize = 24,
   xlabel = "number of repetitions", ylabel ="rate [Hz]",
    ifioff = True, ifsavefig = True, savename = "NrepsArrayBar", axiswidth = 1):
        """ plot Narrays vs. array (Nreps,Nblocks) array_blockavg (Nreps)"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        meanarr = np.mean(array, axis = 0)
        stdarr = np.std(array, axis = 0)

        plt.bar(Nreps,meanarr,color = c3, edgecolor=c3, yerr = stdarr, align='center', alpha=alpha)
        #plt.fill_between(Nreps, meanarr-stdarr, meanarr+stdarr, alpha=alpha, edgecolor=c3, facecolor=c3)

        plt.xlabel(xlabel, fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)
        if ifxticks:
            plt.xticks(Nreps[0::xtickstepsize],fontsize = fontsize)
        else:
            plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.tight_layout()
        #plt.show()
        #plt.legend(fontsize = fontsize, frameon = True)
        if ifxlims:
            xlims = [min(Nreps)-1,max(Nreps)+1]
            plt.xlim(xlims)
        if ifsavefig:
            for fn in range(0,len(figure_directory)):
                save_fig(figure_directory[fn], savename)

def barplot_peak_comparison_EI(noveltyE, noveltyI, transientE, transientI, baselineE, baselineI, ifxlims = False, figure_directory = "./",
 figsize=(7, 5), xlims = [0,45], alpha = 0.8, legend = "E", ifxticks = True, xtickstepsize = 3,
  iflegend = False, c1 = "grey",c3 = "midnightblue", c2 = "darkred",lw = 3,fontsize = 24,
   xlabel = "number of repetitions", ylabel ="rate [Hz]",
    ifioff = True, ifsavefig = True, savename = "ComparisonPeakHeightsBarPlot", axiswidth = 1):
        """ plot Narrays vs. array (Nreps,Nblocks) array_blockavg (Nreps)"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)


        plt.bar(1, np.mean(baselineE),color = c3, edgecolor=c3, yerr = np.std(baselineE), align='center', alpha=alpha)
        plt.bar(4, np.mean(noveltyE),color = c3, edgecolor=c3, yerr = np.std(noveltyE), align='center', alpha=alpha)
        plt.bar(7, np.mean(transientE),color = c3, edgecolor=c3, yerr = np.std(transientE), label = "E",align='center', alpha=alpha)

        plt.bar(2, np.mean(baselineI),color = c2, edgecolor=c2, yerr = np.std(baselineI), label = "I",align='center', alpha=alpha)
        plt.bar(5, np.mean(noveltyI),color = c2, edgecolor=c2, yerr = np.std(noveltyI), align='center', alpha=alpha)
        plt.bar(8, np.mean(transientI),color = c2, edgecolor=c2, yerr = np.std(transientI), align='center', alpha=alpha)


        #plt.fill_between(Nreps, meanarr-stdarr, meanarr+stdarr, alpha=alpha, edgecolor=c3, facecolor=c3)

        plt.xlabel(xlabel, fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)

        plt.xticks([1.5,4.5,7.5],fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.tight_layout()
        if iflegend:
            plt.legend(fontsize = fontsize, frameon = False)
        if ifsavefig:
            save_fig(figure_directory, savename)

def barplot_peak_comparison_general(noveltyE, noveltyI, transientE, transientI, baselineE, baselineI, ifxlims = False, figure_directory = "./",
 figsize=(7, 5), xlims = [0,45], alpha = 0.8, legend = "E", ifxticks = True, xtickstepsize = 3,
  iflegend = False, c1 = "grey",c3 = "midnightblue", c2 = "darkred",lw = 3,fontsize = 24,
   xlabel = "number of repetitions", ylabel ="rate [Hz]", labelleft = "novelty", labelright = "shuffled",
    ifioff = True, ifsavefig = True, savename = "ComparisonPeakHeightsBarPlotShuffle", axiswidth = 1):
        """ plot peak heights """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)


        plt.bar(1, np.mean(baselineE),color = c3, edgecolor=c3, yerr = np.std(baselineE), align='center', alpha=alpha)
        plt.bar(4, np.mean(noveltyE),color = c3, edgecolor=c3, yerr = np.std(noveltyE), align='center', alpha=alpha)
        plt.bar(7, np.mean(transientE),color = c3, edgecolor=c3, yerr = np.std(transientE), label = labelleft,align='center', alpha=alpha)

        plt.bar(2, np.mean(baselineI),color = c2, edgecolor=c2, yerr = np.std(baselineI), label = labelright,align='center', alpha=alpha)
        plt.bar(5, np.mean(noveltyI),color = c2, edgecolor=c2, yerr = np.std(noveltyI), align='center', alpha=alpha)
        plt.bar(8, np.mean(transientI),color = c2, edgecolor=c2, yerr = np.std(transientI), align='center', alpha=alpha)


        #plt.fill_between(Nreps, meanarr-stdarr, meanarr+stdarr, alpha=alpha, edgecolor=c3, facecolor=c3)

        plt.xlabel(xlabel, fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)

        plt.xticks([1.5,4.5,7.5],fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.tight_layout()
        if iflegend:
            plt.legend(fontsize = fontsize, frameon = False)
        if ifsavefig:
            save_fig(figure_directory, savename)

def barplot_peak_comparison_ax(ax, noveltyE, noveltyI, transientE, transientI, baselineE, baselineI, ifxlims = False, figure_directory = "./",
 figsize=(7, 5), xlims = [0,45], alpha = 0.8, legend = "E", ifxticks = True, xtickstepsize = 3,
  iflegend = False, c1 = "grey",c3 = "midnightblue", c2 = "darkred",lw = 3,fontsize = 24,
   xlabel = "number of repetitions", ylabel ="rate [Hz]",
    ifioff = True, ifsavefig = True, savename = "ComparisonPeakHeightsBarPlot", axiswidth = 1):
        """ plot barplots on axes instead of new figure"""


        ax.bar(1, np.mean(baselineE),color = c3, edgecolor=c3, yerr = np.std(baselineE), align='center', alpha=alpha)
        ax.bar(4, np.mean(noveltyE),color = c3, edgecolor=c3, yerr = np.std(noveltyE), align='center', alpha=alpha)
        ax.bar(7, np.mean(transientE),color = c3, edgecolor=c3, yerr = np.std(transientE), label = "E",align='center', alpha=alpha)

        ax.bar(2, np.mean(baselineI),color = c2, edgecolor=c2, yerr = np.std(baselineI), label = "I",align='center', alpha=alpha)
        ax.bar(5, np.mean(noveltyI),color = c2, edgecolor=c2, yerr = np.std(noveltyI), align='center', alpha=alpha)
        ax.bar(8, np.mean(transientI),color = c2, edgecolor=c2, yerr = np.std(transientI), align='center', alpha=alpha)


        #plt.fill_between(Nreps, meanarr-stdarr, meanarr+stdarr, alpha=alpha, edgecolor=c3, facecolor=c3)


        if iflegend:
            ax.legend(frameon = False)



def plot_Nreps_array_comparison_mult(fig,Nreps, array, Nfiles = 5, ifxlims = True, figure_directory = "./",
 figsize=(14, 10), xlims = [0,45], alpha = 0.2, legend = "E", ifxticks = True, xtickstepsize = 3,
  iflegend = False, c1 = "grey",c3 = "darkgreen", c2 = "darkorange",lw = 3,fontsize = 24,
   xlabel = "number of repetitions", ylabel ="rate [Hz]",
    ifioff = True, ifsavefig = True, savename = "NrepsArrayerrorband", axiswidth = 1):
        """ plot Narrays vs. array (Nreps,Nblocks) array_blockavg (Nreps)"""

        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        meanarr = np.mean(array, axis = 0)
        stdarr = np.std(array, axis = 0)

        plt.plot(Nreps,meanarr, ":o", color = c3,  label = legend)
        plt.fill_between(Nreps, meanarr-stdarr, meanarr+stdarr, alpha=alpha, edgecolor=c3, facecolor=c3)

        plt.xlabel(xlabel, fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)
        if ifxticks:
            plt.xticks(Nreps[0::xtickstepsize],fontsize = fontsize)
        else:
            plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        #plt.show()
        if iflegend:
            plt.legend(fontsize = fontsize, frameon = True)
        plt.tight_layout()
        if ifxlims:
            xlims = [min(Nreps)-1,max(Nreps)+1]
            plt.xlim(xlims)
        if ifsavefig:
            for fn in range(0,len(figure_directory)):
                save_fig(figure_directory[fn], savename)

def plot_Nreps_array_comparison_errorbar(Nreps, array, Nfiles = 5, ifxlims = True, figure_directory = "./",
 figsize=(14, 10), xlims = [0,45], alpha = 0.2, legend = "E", ifxticks = True, xtickstepsize = 3,
  iflegend = False, c1 = "grey",c3 = "k", c2 = "darkgrey",lw = 3,fontsize = 24,
   xlabel = "number of repetitions", ylabel ="rate [Hz]",
    ifioff = True, ifsavefig = True, savename = "NrepsArrayerrorband", axiswidth = 1):
        """ plot Narrays vs. array (Nreps,Nblocks) array_blockavg (Nreps)"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        meanarr = np.mean(array, axis = 0)
        stdarr = np.std(array, axis = 0)

        # plt.plot(Nreps,meanarr, ":o", color = c3,  label = "mean individual blocks")
        # plt.fill_between(Nreps, meanarr-stdarr, meanarr+stdarr, alpha=alpha, edgecolor=c3, facecolor=c3)
        plt.errorbar(Nreps, meanarr, yerr=stdarr, fmt='o', color = c3,
        ecolor=c3, elinewidth=2, capsize=3, label = legend)
        plt.xlabel(xlabel, fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)
        if ifxticks:
            plt.xticks(Nreps[0::xtickstepsize],fontsize = fontsize)
        else:
            plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        plt.tight_layout()
        #plt.show()
        #plt.legend(fontsize = fontsize, frameon = True)
        if ifxlims:
            xlims = [min(Nreps)-1,max(Nreps)+1]
            plt.xlim(xlims)
        if ifsavefig:
            for fn in range(0,len(figure_directory)):
                save_fig(figure_directory[fn], savename)


def plot_Nreps_array_comparison_errorbar_mult(fig,Nreps, array, Nfiles = 5, ifxlims = True, figure_directory = "./",
 figsize=(14, 10), xlims = [0,45], alpha = 0.2, legend = "E", ifxticks = True, xtickstepsize = 3,
  iflegend = False, c1 = "grey",c3 = "darkgreen", c2 = "darkorange",lw = 3,fontsize = 24,
   xlabel = "number of repetitions", ylabel ="rate [Hz]",
    ifioff = True, ifsavefig = True, savename = "NrepsArrayerrorband", axiswidth = 1, ifnanignore = False):
        """ plot Narrays vs. array (Nreps,Nblocks) array_blockavg (Nreps)"""

        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        if ifnanignore:
            meanarr = np.nanmean(array, axis = 0)
            stdarr = np.nanstd(array, axis = 0)
        else:
            meanarr = np.mean(array, axis = 0)
            stdarr = np.std(array, axis = 0)

        plt.errorbar(Nreps, meanarr, yerr=stdarr, fmt='o', color = c3,
             ecolor=c3, elinewidth=2, capsize=3, label = legend)

        plt.xlabel(xlabel, fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)
        if ifxticks:
            plt.xticks(Nreps[0::xtickstepsize],fontsize = fontsize)
        else:
            plt.xticks(fontsize = fontsize)
        plt.yticks(fontsize = fontsize)
        #plt.show()
        if iflegend:
            plt.legend(fontsize = fontsize, frameon = True)
        if ifxlims:
            xlims = [min(Nreps)-1,max(Nreps)+1]
            plt.xlim(xlims)
        plt.tight_layout()
        if ifsavefig:
            for fn in range(0,len(figure_directory)):
                save_fig(figure_directory[fn], savename)

def plot_all_averages_new(timelist, datalist, Nreps, endindices = [], startidx = 0, endidx = -1,figure_directory = "./", figsize=(20, 10), ifoffset = True,
                              offset = 1, ifseqlen = False, iflegend = False, ifcolor = True, color = ["r","g","b","k","c","m","r","g","b","k","c","m"],
                            fontsize = 20, lw = 1, xlabel = "time [s]", legendhandle = "Nimg: ", Nreponset = 6, ylabel ="$\\rho$ [Hz]",
                              ifioff = False, ifyticks = True, yticks = [3,4], ifsavefig = True, savehandle = "E", axiswidth = 1):
    """ plot population averages in one figure
    input:  list with means
            time vector
            number of repetitions vector"""
    fig = plt.figure(figsize=figsize) #run first and then conse
    # check if specific end indices were specified
    # else
    if len(endindices) != len(Nreps):
        endindices = [endidx]*len(Nreps)

    if ifseqlen:
        print("Sequences Length")

        for seq in reversed(range(1,len(Nreps) + 1)):
            if ifoffset:
                plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][startidx:endidx] + offset*(seq-1),
                                 iflegend = iflegend, legend = legendhandle + str(Nreps[seq-1]),
                                 lw = lw, ifcolor = ifcolor, color = color[seq-1])
            else:
                plot_popavg_mult(fig,  timelist[seq-1][startidx:endidx],datalist[seq-1][startidx:endidx],
                                     iflegend = iflegend, legend = legendhandle + str(Nreps[seq-1]),
                                     lw = lw, ifcolor = ifcolor, color = "midnightblue")#color = color[seq-1])

    else:
        print("Repeated Sequences")
        for seq in reversed(range(Nreponset,len(Nreps) + 1)):
            if ifoffset:
                plot_popavg_mult(fig,  timelist[seq-1][startidx:endindices[seq-1]],datalist[seq-1][startidx:endindices[seq-1]] + offset*(seq-1),
                                 iflegend = iflegend, legend = "Nreps: " + str(Nreps[seq-1]),
                                 lw = lw, ifcolor = ifcolor, color = "midnightblue")#color = color[seq-1])
            else:
                plot_popavg_mult(fig,  timelist[seq-1][startidx:endindices[seq-1]],datalist[seq-1][startidx:endindices[seq-1]],
                                     iflegend = iflegend, legend = "Nreps: " + str(Nreps[seq-1]),
                                     lw = lw, ifcolor = ifcolor, color = "midnightblue")#color[seq-1])
    if ifyticks:
        plt.yticks(yticks)
    plt.tight_layout()
    savetitle = "PopulationAveragesSeq_" + savehandle
    if ifoffset:
        savetitle = savetitle + "offset"
    if ifsavefig:
        save_fig(figure_directory, savetitle)

def fit_traces_Nreps(time, trace, fit_function, initialparams = [10, 6, 3],
                     ifplot = False, saveplot = False, figure_directory = "./",
                     title = "Fit_PopulationAverages_E", img = 1, bounds=(0, [10., 140., 10]), sigma=[]):
    # use optimize curvefit function to fit the data
    #Constrain the optimization to the region of 0 <= a <= 10, 0 <= tau <= 20 and 0 <= offset <= 10:
    # specify weights given to parameters
    if len(sigma)==0:
        sigma = np.ones_like(trace)
    params, params_covariance = optimize.curve_fit(fit_function, time, trace, p0 = initialparams, sigma=sigma)#, bounds=bounds)
    #print(params)

    if ifplot:
        fig = plt.figure(figsize=(20, 10))
        #plt.plot(time, trace, label='Data')
        plot_popavg_mult(fig,time, trace, legend = "Data", iflegend = True, color = "darkblue", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [s]", ylabel ="$\\rho$ [Hz]", ifioff = False)
        plt.plot(time, fit_function(time, params[0], params[1], params[2]), color = "red", lw = 3, label='fit: a=%5.3f, tau=%5.3f, offset=%5.3f' % tuple(params))
        plt.legend(loc='best', frameon = False, fontsize = 20)
        #plt.show()
        #plt.ylim([2,4])
        if saveplot:
            save_fig(figure_directory, title + str(img))
        #plt.show()
    return params, params_covariance

def fit_traces_Nreps_2params(time, trace, fit_function, initialparams = [10, 6], ifplot = False,
                             saveplot = False, figure_directory = "./", title = "Fit_PopulationAverages_E",
                             img = 1, bounds=(0, [10., 10]), sigma=[]):
    # use optimize curvefit function to fit the data
    if len(sigma)==0:
        sigma = np.ones_like(trace)
    #Constrain the optimization to the region of 0 <= a <= 10, 0 <= tau <= 20 and 0 <= offset <= 10:
    params, params_covariance = optimize.curve_fit(fit_function, time, trace, p0 = initialparams, sigma=sigma)#, bounds=bounds)
    #print(params)

    if ifplot:
        fig = plt.figure(figsize=(20, 10))
        #plt.plot(time, trace, label='Data')
        plot_popavg_mult(fig,time, trace, legend = "Data", iflegend = True, color = "darkblue", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [s]", ylabel ="$\\rho$ [Hz]", ifioff = False)
        plt.plot(time, fit_function(time, params[0], params[1]), color = "red", lw = 3, label='fit: a=%5.3f, tau=%5.3f' % tuple(params))
        plt.legend(loc='best', frameon = False, fontsize = 20)
        #plt.show()
        #plt.ylim([2,4])
        if saveplot:
            save_fig(figure_directory, title + str(img))
        #plt.show()
    return params, params_covariance
