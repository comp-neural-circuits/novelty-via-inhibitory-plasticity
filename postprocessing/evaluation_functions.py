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
from scipy.signal import decimate
from scipy import signal
import gc
import time
import matplotlib.colors as colors
#%matplotlib inline


from matplotlib import rcParams, cm
rcParams['grid.linewidth'] = 0
rcParams['pdf.fonttype'] = 42
#%matplotlib inline
from helper_functions import *
import helper_functions

# list of all evaluation codes
# 1. run_single_neuron_eval

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#################################################################
#                                                               #
#                                                               #
#              weightdistribution                               #
#                                                               #
#                                                               #
#################################################################

def weightdistribution(file_name, ifcontin,  indivassembly = True, figsize=(20,10),ncol = 1, Ne = 4000, Ncells =5000, RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR
    #run_folder = "/gpfs/gjor/personal/schulza/data/main/sequences/"
    # folder with analysed results from spiketime analysis in julia & where to results are stored
    #results_folder = "/home/schulza/Documents/results/main/sequences/"
    #results_folder = "/gpfs/gjor/personal/schulza/results/sequences/"

    # define folder where figues should be stored
    figure_directory = results_folder + file_name + "/" + "weightdistribution/"
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    # read in run parameters
    file_name_run = run_folder + file_name
    # open file
    frun = h5py.File(file_name_run, "r")

    # read in stimulus parameters
    Nimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength  = frun["initial"]["stimparams"].value
    seqnumber  = frun["initial"]["seqnumber"].value
    stimulus  = frun["initial"]["stimulus"].value
    idxblockonset  = frun["initial"]["idxblockonset"].value

    assemblymembers = frun["initial"]["assemblymembers"].value.transpose()
    weights = frun["postsim"]["weights"].value.transpose()
    weightsdursim = frun["dursim"]["weights"].value.transpose()
    Jeemin = 1.78 #minimum ee weight  pF
    Jeemax = 21.4 #maximum ee weight pF

    Jeimin = 48.7 #minimum ei weight pF
    Jeimax = 243 #maximum ei weight pF
    # get submatrix with assembly 1
    Nass = np.size(assemblymembers,axis= 0)
    if Nimg == 4:
        color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange", "black", "silver","dimgrey","gainsboro", "fuchsia", "orchid","plum", "mediumvioletred", "lightseagreen", "lightcyan", "darkslategray", "paleturquoise", "goldenrod","gold", "wheat","darkgoldenrod", "forestgreen", "aquamarine", "palegreen", "lime", ]
    elif Nimg == 3:
        color = ["midnightblue","lightskyblue","royalblue","darkred","darksalmon", "saddlebrown","darkgreen","greenyellow","darkolivegreen","darkmagenta","thistle","indigo","darkorange","tan","sienna", "black", "silver","dimgrey", "fuchsia", "orchid","plum",  "lightseagreen", "lightcyan", "darkslategray",  "goldenrod","gold", "wheat","forestgreen", "aquamarine", "palegreen"]
    elif Nimg == 5:
        color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","indigo","darkred","darksalmon", "saddlebrown","lightcoral","rosybrown","darkgreen","greenyellow","darkolivegreen","chartreuse","turquoise","darkmagenta","thistle","indigo","mediumslateblue","blueviolet","darkorange","tan","sienna","orange", "brown", "black", "silver","dimgrey","gainsboro", "grey","fuchsia", "orchid","plum", "mediumvioletred","purple", "lightseagreen", "lightcyan", "darkslategray", "paleturquoise","teal", "goldenrod","gold", "wheat","darkgoldenrod", "darkkhaki","forestgreen", "aquamarine", "palegreen", "lime", "darkseagreen"]
    sumweights = np.zeros(Ne)
    sumdursimweights = np.zeros(Ne)
    sumweightsI = np.zeros(Ne)
    sumdursimweightsI = np.zeros(Ne)

    Nee = np.zeros(Ne)
    Nei = np.zeros(Ne)

    for cc in range(Ne): # post
        for dd in range(Ne): # pre
            sumweights[cc] += weights[dd,cc]
            sumdursimweights[cc] += weightsdursim[dd,cc]
            if weights[dd,cc] > 0:
                Nee[cc] += 1
        for dd in range(Ne,Ncells): # pre
            sumweightsI[cc] += weights[dd,cc]
            sumdursimweightsI[cc] += weightsdursim[dd,cc]
            if weights[dd,cc] > 0:
                Nei[cc] += 1
    print(sumweights)
    print(sumweightsI)
    plot_histogram(sumweights,ifExcitatory=False,figsize = (7,5), color = "midnightblue", alpha=1, xlabel = "summed wee [pF]", bins = np.linspace(2000,2800,51))
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    save_fig(figure_directory, "HistogramSummedWeights")
    plot_histogram(sumdursimweights,ifExcitatory=False, figsize = (7,5),color = "midnightblue", alpha=1, xlabel = "pretrain summed wee [pF]", bins = np.linspace(2000,2800,51))
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    save_fig(figure_directory, "HistogramSummedDursimWeights")
    plot_histogram(sumweightsI,ifExcitatory=False, figsize = (7,5),color = "darkred", alpha=1, xlabel = "summed wei [pF]", bins = np.linspace(8000,40000,50))
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    save_fig(figure_directory, "HistogramSummedWeightsI")
    plot_histogram(sumdursimweightsI,ifExcitatory=False,figsize = (7,5),color = "darkred", alpha=1, xlabel = "pretrain summed wei [pF]", bins = np.linspace(8000,40000,50))
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    save_fig(figure_directory, "HistogramSummedDursimWeightsI")
    #fig = plt.figure(figsize = (7,5))
    plot_histogram(Nee,ifExcitatory=False,color = "midnightblue", alpha=1, xlabel = "number of E inputs", bins = np.linspace(700,900,20))
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    save_fig(figure_directory, "NumberOfExcitatoryInputstoANeuron")
    plot_histogram(Nei,ifExcitatory=False,color = "darkred", alpha=1, xlabel ="number of I inputs", bins = np.linspace(100,300,20))
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    save_fig(figure_directory, "NumberOfInhibitoryInputstoANeuron")


    for ass in range(3):#Nass:
        members1 = assemblymembers[ass,:]
        members1 = np.unique(members1[members1>0])- 1 # convert to pyhton smallest index 0

        Npre	= np.size(members1,axis=0) # pre
        Npost = np.size(members1,axis=0) # post
        submatEE = []
        submatEEdursim = []

        precount = -1
        for pre in members1:
            for post in members1:
                if weights[pre,post] !=0:
                    submatEE.append(weights[pre,post])
                if weightsdursim[pre,post] !=0:
                    submatEEdursim.append(weightsdursim[pre,post])
        print(len(submatEE))
        #plot_histogram(submatrix_weights11)
        plot_histogram_cut(submatEE,ifExcitatory=False, figsize = (7,5),cutlow = 200, cuthigh = 201, color = "midnightblue", alpha=1, xlabel = "w [pF]", bins = np.linspace(Jeemin,Jeemax,50))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramEtoEAssembly%d"%ass)
        plot_histogram_cut(submatEEdursim,ifExcitatory=False, cutlow = 500, cuthigh = 501,figsize = (7,5),color ="midnightblue", alpha=1, xlabel = "w [pF]", bins = np.linspace(Jeemin,Jeemax,50))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "DursimHistogramEtoEAssembly%d"%ass)
        #norm = sum(submat .!= 0)
        # inhibioty
        Npre	= np.size(members1,axis=0) # pre
        Npost = 1000 # post
        submatEI = []
        submatEIdursim = []

        precount = -1
        for pre in range(Ne,Ncells):
            for post in members1:
                if weights[pre,post] !=0:
                    submatEI.append(weights[pre,post])
                if weightsdursim[pre,post] !=0:
                    submatEIdursim.append(weightsdursim[pre,post])
        print(len(submatEI))
        print(len(submatEIdursim))

        #plot_histogram(submatrix_weights11)
        plot_histogram_cut(submatEI,ifExcitatory=False, figsize = (7,5),cutlow=4000,cuthigh = 4001,color = "darkred", alpha=1, xlabel = "w i [pF]", bins = np.linspace(Jeimin,Jeimax,50))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramInhibitorytoEAssembly%d"%ass)
        plot_histogram_cut(submatEIdursim,ifExcitatory=False, cutlow=4000,cuthigh = 4001, figsize = (7,5),color = "darkred", alpha=1, xlabel = "w i [pF]", bins = np.linspace(Jeimin,Jeimax,50))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "DursimHistogramInhibitorytoEAssembly%d"%ass)
        #norm = sum(submat .!= 0)


        plot_histogram(sumweights[members1],ifExcitatory=False, figsize = (7,5),color = "midnightblue", alpha=1, xlabel = "summed wee [pF]", bins = np.linspace(2000,2800,51))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramSummedWeightsAssembly%d"%ass)
        plot_histogram(sumdursimweights[members1],ifExcitatory=False, figsize = (7,5),color = "midnightblue", alpha=1, xlabel = "pretrain summed wee [pF]", bins = np.linspace(2000,2800,51))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramSummedDursimWeightsAssembly%d"%ass)
        plot_histogram(sumweightsI[members1],ifExcitatory=False, figsize = (7,5),color = "darkred", alpha=1, xlabel = "summed wei [pF]", bins = np.linspace(8000,40000,51))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramSummedWeightsInhibitorytoEAssembly%d"%ass)
        plot_histogram(sumdursimweightsI[members1],ifExcitatory=False, figsize = (7,5),color = "darkred", alpha=1, xlabel = "pretrain summed wei [pF]", bins = np.linspace(8000,40000,51))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)

        save_fig(figure_directory, "HistogramSummedDursimWeightsInhibitorytoEAssembly%d"%ass)


#################################################################
#                                                               #
#                                                               #
#              run_variable_repetitions                         #
#                                                               #
#                                                               #
#################################################################


def run_variable_repetitions_short(file_name, avgwindow = 8, timestr = "_now",RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR

    #run_folder = "/gpfs/gjor/personal/schulza/data/main/sequences/"
    # folder with analysed results from spiketime analysis in julia & where to results are stored
    #results_folder = "/gpfs/gjor/personal/schulza/results/sequences/"
    #results_folder = "/gpfs/gjor/personal/schulza/results/varrep/"


    # define folder where figues should be stored
    figure_directory = results_folder + file_name + "/" + "figures_window%d/"%avgwindow
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    # read in run parameters
    file_name_run = run_folder + file_name
    # open file
    frun = h5py.File(file_name_run, "r")

    # read in stimulus parameters
    Nimg, lenNreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength  = frun["initial"]["stimparams"].value
    repetitions  = frun["initial"]["repetitions"].value
    Nreps  = frun["initial"]["Nreps"].value
    ifSTDP, ifwadapt = frun["params"]["STDPwadapt"].value
    if ifwadapt == 1:
        print("ADAPT ON")
    else:
        print("NON ADAPTIVE")
    print(Nreps)
    assemblymembers = frun["initial"]["assemblymembers"].value.transpose()
    # close file

    frun.close()
    Nblocks = Nblocks -1 # remove last block as we no longer have the last 20 seconds of following sequence
    # read in population averages
    listOfFiles = os.listdir(results_folder + file_name)
    pattern = "spiketime*.h5"
    sub_folder = []

    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
                sub_folder.append(entry)

    # get name of subfolder spiketime + date.h5
    file_name_spikes = results_folder + file_name + "/" + sub_folder[0]
    print(file_name_spikes)
    f = h5py.File(file_name_spikes, "r")

    Nseq = len(Nreps) # to avoid changing all variables select Nseq as the number of differnet repetitions
    print(Nseq)
    novelty_overlap = np.zeros((Nseq,Nblocks,3))
    binsize = f["params"]["binsize"].value

    # filter values
    cutoff = 0.05 # times the Nyquist frequency
    if binsize == 80:
        cutoff = 0.3
    if binsize == 10:
        cutoff = 0.1
    # make filter arrays
    b, a = signal.butter(4, cutoff, analog=False)

    # -------------------------- copied from single neuron response ---------------------------
    #
    winlength = avgwindow*binsize

    # initialise lists every trace has a different length
    hist_E  = []
    mean_hist_E = []
    hist_I  = []
    mean_hist_I = []
    hist_E_nomem  = []
    mean_hist_E_nomem = []
    hist_E_nomemnonov  = []
    mean_hist_E_nomemnonov = []
    hist_E_nov  = []
    mean_hist_E_nov = []
    hist_E_boxcar  = []
    mean_hist_E_boxcar = []
    edges = []

    for seq in range(1,Nseq + 1):
        edges.append(f["E%dmsedges" % binsize]["seq"+ str(seq) + "block"+ str(1)].value)
        hist_E.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_I.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_nomem.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_nomemnonov.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_boxcar.append(np.zeros((Nblocks,len(edges[seq-1]))))

        for bl in range(1, Nblocks + 1):
            #vars()['hist_E_all' + str(seq-1)][bl-1][:] = f["E%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E[seq-1][bl-1,:] = f["E%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_boxcar[seq-1][bl-1,:] = np.convolve(hist_E[seq-1][bl-1,:], np.ones((avgwindow,))/avgwindow, mode='same')

        # get averages over blocks
        mean_hist_E.append(np.mean(hist_E[seq-1][:,:],axis = 0))
        mean_hist_E_boxcar.append(np.mean(hist_E_boxcar[seq-1][:,:],axis = 0))

    # plotting
    color = ["midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon"]

    idxconv = np.floor_divide(avgwindow,2)+1
    t_before_nov = np.zeros(Nseq) # time just before the peak (before novelty, before last img of first sequnece new/old)
    idxmin = np.zeros(Nseq).astype(int) # c
    for rep in range(1,Nseq + 1):
        t_before_nov[rep-1] = (Nreps[rep-1]*Nimg*(lenstim+lenpause) + (Nimg-0.5)*(lenstim+lenpause))/1000.+5 # middle of last image in new sequence
        idxmin[rep-1] = np.argmax(edges[rep-1]>t_before_nov[rep-1])
    print(t_before_nov)
    print(idxmin)
        #t_before_nov[rep-1] = ((Nreps-2)*Nimg[rep-1]*(lenstim+lenpause) + (Nimg[rep-1]-0.5)*(lenstim+lenpause))/1000. # middle of novelty image

#print(t_before_nov[rep-1])
# TODO
    #print(meandatalist[rep-1][max(0,idxmin[rep-1]-search_margin):min(len(meandatalist[rep-1]),idxmin[rep-1]+search_margin)] + max(0,idxmin[rep-1]-search_margin))
    print(Nreps)
    figsize_cm = (6.33,10)
    figsize_inch = cm2inch(figsize_cm)
    # plot_all_averages_new(edges, mean_hist_E, Nreps, savehandle = "E",
    #                       figure_directory = figure_directory, Nreponset = 1, color = color,
    #                       ifoffset=True, iflegend=False, ifyticks=True,  offset = 2)
    plot_all_averages_new(edges[1:], mean_hist_E[1:], Nreps[1:], savehandle = "E",
                          figure_directory = figure_directory, Nreponset = 1, color = color,
                          ifoffset=True, iflegend=False, ifyticks=True,  offset = 2.3, endindices =idxmin[1:],
                         figsize=figsize_inch, yticks = [3,5])
    return edges[1:], mean_hist_E[1:], Nreps[1:], idxmin[1:]


def evaluate_multiple_variable_repetitions(file_names, avgwindow):
    run_folder = "/gpfs/gjor/personal/schulza/data/main/sequences/"
    run_folder = "../data/"

    # folder with analysed results from spiketime analysis in julia & where to results are stored
    results_folder = "/gpfs/gjor/personal/schulza/results/sequences/"
    results_folder = "../results/"

    figure_directory = [] # initialise everything as list
    file_name_run = [] # initialise everything as list
    file_name_results = []
    stimparams = [] # initialise everything as list
    repetitions = [] # initialise everything as list
    Nreps = [] # initialise everything as list
    assemblymembers = [] # initialise everything as list
    #Nimg, lenNreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength = [np.ones(len(file_names))] * 8
    Nimg = np.ones(len(file_names))
    lenNreps = np.ones(len(file_names))
    Nseq = np.ones(len(file_names))
    Nblocks = np.ones(len(file_names))
    stimstart = np.ones(len(file_names))
    lenstim = np.ones(len(file_names))
    lenpause = np.ones(len(file_names))
    strength = np.ones(len(file_names))

    baseline_avg = []
    height_novelty_avg = []
    height_trans_post_avg = []
    tau_transientpost_avg = []
    tau_transientpre_avg = []


    mean_E_hist6 = []
    mean_E_hist11 = []
    mean_E_hist16 = []
    mean_E_hist21 = []
    mean_E_hist26 = []
    mean_E_hist31 = []
    mean_E_hist36 = []
    mean_E_hist41 = []
    edges = []


    for fn in range(0,len(file_names)):
        figure_directory.append(results_folder + file_names[fn] + "/" + "comparison%d/"%avgwindow)
        if not os.path.exists(figure_directory[fn]):
                os.makedirs(figure_directory[fn])
            # read in run parameters
        file_name_run.append(run_folder + file_names[fn])
        # open file
        frun = h5py.File(file_name_run[fn], "r")

        # read in stimulus parameters
        Nimg[fn], lenNreps[fn], Nseq[fn], Nblocks[fn], stimstart[fn], lenstim[fn], lenpause[fn], strength[fn]  = frun["initial"]["stimparams"].value
        repetitions.append(frun["initial"]["repetitions"].value)
        Nreps.append(frun["initial"]["Nreps"].value)

        #print(Nreps)
        assemblymembers.append(frun["initial"]["assemblymembers"].value.transpose())
        # close file

        frun.close()

        # read in population averages
        listOfFiles = os.listdir(results_folder + file_names[fn])
        pattern = "results*.h5"
        sub_folder = []
        # print(listOfFiles)
        for entry in listOfFiles:
            if fnmatch.fnmatch(entry, pattern):
                    sub_folder.append(entry)
        # print(sub_folder)
        # get name of subfolder spiketime + date.h5
        file_name_results.append(results_folder + file_names[fn] + "/" + sub_folder[0])

        # print(file_name_results)
        f = h5py.File(file_name_results[fn], "r")

        #'avgwindow%d'%avgwindow, data=avgwindow
        #hist_E[seq-1][bl-1,:] = f["E%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
        #datasetnames=f.keys()
        datasetnames = [key for key in f.keys()]

        avgwindow = 4

        booldata = np.zeros(len(datasetnames), dtype=bool)
    #            print(avgwindow)
        for dt in range(0,len(datasetnames)):
            booldata[dt] = fnmatch.fnmatch(datasetnames[dt], "Avgwindow%d"%avgwindow)
        # print(any(booldata))
        if not any(booldata):
            avgwindow = 8
            print(avgwindow)
            booldata = np.zeros(len(datasetnames), dtype=bool)
            print(any(booldata))
            for dt in range(0,len(datasetnames)):
                booldata[dt] = fnmatch.fnmatch(datasetnames[dt], "Avgwindow%d"%avgwindow)
            print(any(booldata))
        if not any(booldata):
            avgwindow = 1
            print(avgwindow)
            booldata = np.zeros(len(datasetnames), dtype=bool)
            for dt in range(0,len(datasetnames)):
                booldata[dt] = fnmatch.fnmatch(datasetnames[dt], "Avgwindow%d"%avgwindow)

        baseline_avg.append(f["Avgwindow%d"%avgwindow]["baseline_avg"].value)
        height_novelty_avg.append(f["Avgwindow%d"%avgwindow]["height_novelty_avg"].value)
        height_trans_post_avg.append(f["Avgwindow%d"%avgwindow]["height_trans_post_avg"].value)
        tau_transientpost_avg.append(f["Avgwindow%d"%avgwindow]["tau_transientpost_avg"].value)
        tau_transientpre_avg.append(f["Avgwindow%d"%avgwindow]["tau_transientpre_avg"].value)
        f.close()

    return Nreps[0], baseline_avg, height_novelty_avg, height_trans_post_avg# [ht-bl for ht,bl in zip(height_novelty_avg,baseline_avg)], [ht-bl for ht,bl in zip(height_trans_post_avg,baseline_avg)]

#################################################################
#                                                               #
#                                                               #
#              run_sequence_length                              #
#                                                               #
#                                                               #
#################################################################

def sequence_length_single(file_name, avgwindow = 8, timestr = "_now",RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    """ evalueate one sequnence length experiment including fits of the decay of pop activity"""
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR


    # define folder where figues should be stored
    figure_directory = results_folder + file_name + "/" + "figures_window%d/"%avgwindow
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    # read in run parameters
    file_name_run = run_folder + file_name
    # open file
    frun = h5py.File(file_name_run, "r")

    # now Nimg is the varying parameter Nreps is fixed
    lenNimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength  = frun["initial"]["stimparams"].value
    seqlen  = frun["initial"]["seqlen"].value
    Nimg  = frun["initial"]["Nimg"].value
    ifSTDP, ifwadapt = frun["params"]["STDPwadapt"].value
    if ifwadapt == 1:
        print("ADAPT ON")
    else:
        print("NON ADAPTIVE")

    assemblymembers = frun["initial"]["assemblymembers"].value.transpose()
    # close file

    frun.close()

    # read in population averages
    listOfFiles = os.listdir(results_folder + file_name)
    pattern = "spiketime*.h5"
    sub_folder = []

    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
                sub_folder.append(entry)

    # get name of subfolder spiketime + date.h5
    file_name_spikes = results_folder + file_name + "/" + sub_folder[0]
    print(file_name_spikes)
    f = h5py.File(file_name_spikes, "r")

    Nseq = len(Nimg) # to avoid changing all variables select Nseq as the number of differnet repetitions
    print(Nseq)
    novelty_overlap = np.zeros((Nseq,Nblocks,3))
    binsize = f["params"]["binsize"].value

    # filter values
    cutoff = 0.05 # times the Nyquist frequency
    if binsize == 80:
        cutoff = 0.3
    if binsize == 10:
        cutoff = 0.1
    # make filter arrays
    b, a = signal.butter(4, cutoff, analog=False)

    # -------------------------- copied from single neuron response ---------------------------
    #
    winlength = avgwindow*binsize

    # initialise lists every trace has a different length
    hist_E  = []
    mean_hist_E = []
    hist_I  = []
    mean_hist_I = []
    hist_E_nomem  = []
    mean_hist_E_nomem = []
    hist_E_nomemnonov  = []
    mean_hist_E_nomemnonov = []
    hist_E_nov  = []
    mean_hist_E_nov = []
    hist_E_boxcar  = []
    mean_hist_E_boxcar = []
    edges = []

    for seq in range(1,Nseq + 1):
        edges.append(f["E%dmsedges" % binsize]["seq"+ str(seq) + "block"+ str(1)].value)
        hist_E.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_I.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_nomem.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_nomemnonov.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_boxcar.append(np.zeros((Nblocks,len(edges[seq-1]))))

        for bl in range(1, Nblocks + 1):
            #vars()['hist_E_all' + str(seq-1)][bl-1][:] = f["E%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E[seq-1][bl-1,:] = f["E%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_I[seq-1][bl-1,:] = f["I%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_nomem[seq-1][bl-1,:] = f["ENonMem%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            #hist_E_nomemnonov[seq-1][bl-1,:] = f["ENonMemNoNov%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_boxcar[seq-1][bl-1,:] = np.convolve(hist_E[seq-1][bl-1,:], np.ones((avgwindow,))/avgwindow, mode='same')

        # get averages over blocks
        mean_hist_E.append(np.mean(hist_E[seq-1][:,:],axis = 0))
        mean_hist_I.append(np.mean(hist_I[seq-1][:,:],axis = 0))
        mean_hist_E_nomem.append(np.mean(hist_E_nomem[seq-1][:,:],axis = 0))
        mean_hist_E_nomemnonov.append(np.mean(hist_E_nomemnonov[seq-1][:,:],axis = 0))
        mean_hist_E_boxcar.append(np.mean(hist_E_boxcar[seq-1][:,:],axis = 0))

    # plotting
    color = ["midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon"]

    idxconv = np.floor_divide(avgwindow,2)+1
    ifplotting = False
    if ifplotting:
        plot_all_averages(edges, mean_hist_E, Nimg, savehandle = "E", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
        plot_all_averages(edges, mean_hist_I, Nimg, savehandle = "I", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
        plot_all_averages(edges, mean_hist_E_boxcar, Nimg, savehandle = "E_boxcar", ifseqlen=True,startidx = idxconv, endidx = -idxconv, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
        plot_all_averages(edges, mean_hist_E_nomem, Nimg, savehandle = "E_nomem", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
        plot_all_averages(edges, mean_hist_E_nomemnonov, Nimg, savehandle = "E_nomemnonov", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)

        plot_all_traces_and_average(edges, hist_E, mean_hist_E, Nimg, savehandle = "E", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
        plot_all_traces_and_average(edges, hist_I, mean_hist_I, Nimg, savehandle = "I", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
        plot_all_traces_and_average(edges, hist_E_boxcar, mean_hist_E_boxcar, Nimg, ifseqlen=True, savehandle = "E_boxcar_avg%d" % int(avgwindow) , startidx = idxconv, endidx = -idxconv, Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
        plot_all_traces_and_average(edges, hist_E_nomem, mean_hist_E_nomem, Nimg, ifseqlen=True, savehandle = "E", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
        plot_all_traces_and_average(edges, hist_E_nomemnonov, mean_hist_E_nomemnonov, Nimg, ifseqlen=True, savehandle = "E_nomemnonov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)


    # ------------------------------------ FITTING ---------------------------------------------
    # """fit_variable_repetitions_gen_arrays(args):
    #     perform fitting of all traces included in datalist and meandatalist
    #         determine the baseline firing rate prior to the novelty stimulation

    # set initial parameters for fitting of the exponential curve
    # fit a * exp(-t/tau) + a_0
    initial_params = [2, 20, 3]
    #                [a, tau,a_0]
    fit_bounds = (0, [10., 60., 10])
    avgindices = 30
    startimg = Nimg # after which image should fit start at block onset update for Seqlen in function always last img
    # fitting of initial transient
    t_before_nov, params_blockavg, params_covariance_blockavg, params_err_blockavg, params, params_covariance, params_err = fit_variable_repetitions_gen_arrays_startidx(
        edges,hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        ifseqlen=True, avgindices = avgindices, initialparams=initial_params, bounds=fit_bounds, ifplot = False,
        startimg = startimg, idxconv = idxconv)

    #get_baseline_firing_rate
    baseline_avg, baseline, mean_baseline, std_baseline = get_baseline_firing_rate(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        ifseqlen=True, avgindices = avgindices, idxconv = idxconv)

    if ifplotting:
        plot_all_averages_with_fits(edges, mean_hist_E, Nimg, params_blockavg, savehandle = "E_withfits", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=True, iflegend=False, ifyticks=False)
        plot_all_averages_with_fits(edges, mean_hist_E, Nimg, params_blockavg, savehandle = "E_boxcar_withfits", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=True, iflegend=False, ifyticks=False)

    # collect garbage
    gc.collect()

    tau_transientpre, tau_transientpre_err = convert_tau(params,params_err)
#     tau_transientpost, tau_transientpost_err = convert_tau(params_trans, params_err_trans)
    tau_transientpre_avg, tau_transientpre_err_avg = convert_tau_avg(params_blockavg, params_err_blockavg)
#     tau_transientpost_avg, tau_transientpost_err_avg = convert_tau_avg(params_blockavg,params_err_blockavg)

    # ----------------------------------------- get peaks -----------------------------------------

    samples_img = int(round(lenstim/binsize))
    height_novelty_avg, height_novelty, mean_novelty, std_novelty, novelty_avgidx, noveltyidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = False, iftransientpost = False, ifseqlen = True,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = 8*samples_img)

    height_trans_pre_avg, height_trans_pre, mean_trans_pre, std_trans_pre, trans_pre_avgidx, trans_preidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = True, iftransientpost = False, ifseqlen=True,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = 8*samples_img)


    # ---------------------------------------- plotting --------------------------------------------------------
    if ifplotting:

        # plot pre transient decay constant vs. number of repetitions
        plot_Nreps_tau(Nimg, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=False, xlabel="sequence length", xtickstepsize = 1, savename = "NimgTau")
        # plot baseline determined from fit vs. number of repetitions
        plot_Nreps_baseline(Nimg, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=False, xlabel="sequence length", xtickstepsize = 1,savename = "NimgBaseline")
        # saving and reloading for comparing instantiations

        # plot unsubtracted data transients, novelty and baseline
        plot_Nreps_array(Nimg, height_trans_pre, height_trans_pre_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient peak rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre_grey_dots", xtickstepsize = 1)
        plot_Nreps_array(Nimg, baseline, baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_BL_grey_dots", xtickstepsize = 1)
        plot_Nreps_array(Nimg, height_novelty, height_novelty_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty peak rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty_grey_dots", xtickstepsize = 1)
        #plot_Nreps_array(Nimg, height_trans_post, height_trans_post_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient peak rate [Hz]", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre_grey_dots")

        # plot data transients, novelty subtracted baseline
        #plot_Nreps_array(Nimg, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPost-BL_grey_dots", xtickstepsize = 1)
        plot_Nreps_array(Nimg, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty-BL_grey_dots", xtickstepsize = 1)
        plot_Nreps_array(Nimg, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre-BL_grey_dots", xtickstepsize = 1)

        # plot data transients, novelty subtracted baseline with errorbars
        #plot_Nreps_array_errorbar(Nimg, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPost-BL_grey_errorbar", xtickstepsize = 1)
        plot_Nreps_array_errorbar(Nimg, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty-BL_grey_errorbar", xtickstepsize = 1)
        plot_Nreps_array_errorbar(Nimg, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre-BL_grey_errorbar", xtickstepsize = 1)

        plot_Nreps_array_errorbar(Nimg, height_novelty, height_novelty_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty_grey_errorbar", xtickstepsize = 1)
        plot_Nreps_array_errorbar(Nimg, height_trans_pre, height_trans_pre_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre_grey_errorbar", xtickstepsize = 1)

        # plot data transients, novelty subtracted baseline with errorbands
        #plot_Nreps_array_errorband(Nimg, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPost-BL_grey_errorband", xtickstepsize = 1)
        plot_Nreps_array_errorband(Nimg, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty-BL_grey_errorband", xtickstepsize = 1)
        plot_Nreps_array_errorband(Nimg, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre-BL_grey_errorband", xtickstepsize = 1)


    return edges, mean_hist_E, Nimg, params_blockavg


def evaluate_multiple_sequence_length_short(file_names, avgwindow):
    run_folder = "../data/"#"/gpfs/gjor/personal/schulza/data/main/sequences/"
    # folder with analysed results from spiketime analysis in julia & where to results are stored
    results_folder = "../results/"#"/gpfs/gjor/personal/schulza/results/sequences/"
    figure_directory = [] # initialise everything as list
    file_name_run = [] # initialise everything as list
    file_name_results = []
    stimparams = [] # initialise everything as list
    seqlens = [] # initialise everything as list
    Nimg = [] # initialise everything as list
    assemblymembers = [] # initialise everything as list
    lenNimg = np.ones(len(file_names))
    Nreps = np.ones(len(file_names))
    Nseq = np.ones(len(file_names))
    Nblocks = np.ones(len(file_names))
    stimstart = np.ones(len(file_names))
    lenstim = np.ones(len(file_names))
    lenpause = np.ones(len(file_names))
    strength = np.ones(len(file_names))

    baseline_avg = []
    height_novelty_avg = []
    height_trans_pre_avg = []
    tau_transientpost_avg = []
    tau_transientpre_avg = []

    for fn in range(0,len(file_names)):
        figure_directory.append(results_folder + file_names[fn] + "/" + "figures_window%d/"%avgwindow)
        if not os.path.exists(figure_directory[fn]):
                os.makedirs(figure_directory[fn])
            # read in run parameters
        file_name_run.append(run_folder + file_names[fn])
        # open file
        frun = h5py.File(file_name_run[fn], "r")

        # read in stimulus parameters
        #lenNimg[fn], Nreps[fn], Nseq[fn], Nblocks[fn], stimstart[fn], lenstim[fn], lenpause[fn], strength[fn]  = frun["initial"]["stimparams"].value
        stimparams = frun["initial"]["stimparams"].value
        lenNimg[fn], Nreps[fn], Nseq[fn], Nblocks[fn], stimstart[fn], lenstim[fn], lenpause[fn], strength[fn] = frun["initial"]["stimparams"].value
#         lenNimg[fn] = stimparams[0]
#         Nreps[fn] = stimparams[1]
#         Nseq[fn] = stimparams[2]
#         Nblocks[fn] = stimparams[3]
#         strength[fn] = stimparams[7]

        print(lenNimg[fn])
        print(Nreps[fn])
        print(Nseq[fn])

        seqlens.append(frun["initial"]["seqlen"].value)
        Nimg.append(frun["initial"]["Nimg"].value)

        #print(Nreps)
        assemblymembers.append(frun["initial"]["assemblymembers"].value.transpose())
        # close file

        frun.close()

        # read in population averages
        listOfFiles = os.listdir(results_folder + file_names[fn])
        pattern = "results*.h5"
        sub_folder = []

        for entry in listOfFiles:
            if fnmatch.fnmatch(entry, pattern):
                    sub_folder.append(entry)

        # get name of subfolder spiketime + date.h5
        file_name_results.append(results_folder + file_names[fn] + "/" + sub_folder[0])
        f = h5py.File(file_name_results[fn], "r")

        #'avgwindow%d'%avgwindow, data=avgwindow
        #hist_E[seq-1][bl-1,:] = f["E%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
        #datasetnames=f.keys()
        datasetnames = [key for key in f.keys()]
        avgwindow = 4
        print(datasetnames)
        booldata = np.zeros(len(datasetnames), dtype=bool)
    #            print(avgwindow)
        for dt in range(0,len(datasetnames)):
            booldata[dt] = fnmatch.fnmatch(datasetnames[dt], "Avgwindow%d"%avgwindow)
        print(any(booldata))
        if not any(booldata):
            avgwindow = 8
            print(avgwindow)
            booldata = np.zeros(len(datasetnames), dtype=bool)
            print(any(booldata))
            for dt in range(0,len(datasetnames)):
                booldata[dt] = fnmatch.fnmatch(datasetnames[dt], "Avgwindow%d"%avgwindow)
            print(any(booldata))
        if not any(booldata):
            avgwindow = 1
            print(avgwindow)
            booldata = np.zeros(len(datasetnames), dtype=bool)
            for dt in range(0,len(datasetnames)):
                booldata[dt] = fnmatch.fnmatch(datasetnames[dt], "Avgwindow%d"%avgwindow)

        baseline_avg.append(f["Avgwindow%d"%avgwindow]["baseline_avg"].value)
        height_novelty_avg.append(f["Avgwindow%d"%avgwindow]["height_novelty_avg"].value)
        height_trans_pre_avg.append(f["Avgwindow%d"%avgwindow]["height_trans_pre_avg"].value)
        #tau_transientpost_avg.append(f["Avgwindow%d"%avgwindow]["tau_transientpost_avg"].value)
        tau_transientpre_avg.append(f["Avgwindow%d"%avgwindow]["tau_transientpre_avg"].value)

        f.close()
    return Nimg[0], tau_transientpre_avg



def analyse_filename(file_name, avgwindow = 8, timestr = "_now", RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    """ analysis file that reads in simulation data from data and results
        to allow for plotting the PSTHs
    """


    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR

    figure_directory = results_folder + file_name + "/" + "figures/"
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    # read in run parameters
    file_name_run = run_folder + file_name
    # open file
    frun = h5py.File(file_name_run, "r")

    # read in stimulus parameters
    Nimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength  = frun["initial"]["stimparams"].value
    Nblocks = min([10,Nblocks])
    seqnumber  = frun["initial"]["seqnumber"].value
    assemblymembers = frun["initial"]["assemblymembers"].value.transpose()
    Seqs = np.arange(1,Nseq+1)
    ifSTDP, ifwadapt = frun["params"]["STDPwadapt"].value
    if ifwadapt == 1:
        print("ADAPT ON")
    else:
        print("NON ADAPTIVE")
    # close file
    frun.close()

    # read in population averages
    listOfFiles = os.listdir(results_folder + file_name)
    pattern = "spiketime*.h5"
    sub_folder = []

    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
                sub_folder.append(entry)

    # get name of subfolder spiketime + date.h5
    file_name_spikes = results_folder + file_name + "/" + sub_folder[0]
    f = h5py.File(file_name_spikes, "r")
    # ------------------------- set binsize and boxcal sliding window --------------------------------------
    binsize = f["params"]["binsize"].value
    winlength = avgwindow*binsize

    novelty_overlap = np.zeros((Nseq,Nblocks,3))
    novelty_indices = np.zeros((Nseq,Nblocks,200))

    hist_E  = []
    mean_hist_E = []
    hist_I  = []
    mean_hist_I = []
    hist_E_nomem  = []
    mean_hist_E_nomem = []
    hist_E_nomemnonov  = []
    mean_hist_E_nomemnonov = []
    hist_E_nov  = []
    mean_hist_E_nov = []
    hist_E_boxcar  = []
    mean_hist_E_boxcar = []
    hist_I_boxcar  = []
    mean_hist_I_boxcar = []
    edges = []
    novelty_overlap = []
    #novelty_indices = []

    for seq in range(1,Nseq + 1):
        edges.append(f["E%dmsedges" % binsize]["seq"+ str(seq) + "block"+ str(1)].value)
        hist_E.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_I.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_nomem.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_nov.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_nomemnonov.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_boxcar.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_I_boxcar.append(np.zeros((Nblocks,len(edges[seq-1]))))
        novelty_overlap.append(np.zeros(Nblocks))

        for bl in range(1, Nblocks + 1):
            #vars()['hist_E_all' + str(seq-1)][bl-1][:] = f["E%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E[seq-1][bl-1,:] = f["E%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_I[seq-1][bl-1,:] = f["I%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_nomem[seq-1][bl-1,:] = f["ENonMem%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_nov[seq-1][bl-1,:] = f["Nov%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_nomemnonov[seq-1][bl-1,:] = f["ENonMemNoNov%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_boxcar[seq-1][bl-1,:] = np.convolve(hist_E[seq-1][bl-1,:], np.ones((avgwindow,))/avgwindow, mode='same')
            hist_I_boxcar[seq-1][bl-1,:] = np.convolve(hist_I[seq-1][bl-1,:], np.ones((avgwindow,))/avgwindow, mode='same')
            novelty_overlap[seq-1][bl-1] = f["noveltyoverlap"]["seq"+ str(seq) + "block"+ str(bl)].value[0]
        # get averages over blocks
        mean_hist_E.append(np.mean(hist_E[seq-1][:,:],axis = 0))
        mean_hist_I.append(np.mean(hist_I[seq-1][:,:],axis = 0))
        mean_hist_E_nomem.append(np.mean(hist_E_nomem[seq-1][:,:],axis = 0))
        mean_hist_E_nomemnonov.append(np.mean(hist_E_nomemnonov[seq-1][:,:],axis = 0))
        mean_hist_E_boxcar.append(np.mean(hist_E_boxcar[seq-1][:,:],axis = 0))
        mean_hist_I_boxcar.append(np.mean(hist_I_boxcar[seq-1][:,:],axis = 0))
        mean_hist_E_nov.append(np.mean(hist_E_nov[seq-1][:,:],axis = 0))


    # close file
    f.close()

    return mean_hist_E,mean_hist_I, edges[0],hist_E, hist_E_boxcar, figure_directory, hist_E_nov


def analyse_mechanism(file_name, avgwindow = 8, timestr = "_now", RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR

    # define folder where figues should be stored
    figure_directory = results_folder + file_name + "/" + "figures/"
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    # read in run parameters
    file_name_run = run_folder + file_name
    # open file
    frun = h5py.File(file_name_run, "r")

    # read in stimulus parameters
    Nimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength  = frun["initial"]["stimparams"].value
    Nblocks = min([10,Nblocks])
    seqnumber  = frun["initial"]["seqnumber"].value
    assemblymembers = frun["initial"]["assemblymembers"].value.transpose()
    Seqs = np.arange(1,Nseq+1)
    ifSTDP, ifwadapt = frun["params"]["STDPwadapt"].value
    if ifwadapt == 1:
        print("ADAPT ON")
    else:
        print("NON ADAPTIVE")
    # close file
    frun.close()

    # read in population averages
    listOfFiles = os.listdir(results_folder + file_name)
    pattern = "spiketime*.h5"
    sub_folder = []

    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
                sub_folder.append(entry)

    # get name of subfolder spiketime + date.h5
    file_name_spikes = results_folder + file_name + "/" + sub_folder[0]
    f = h5py.File(file_name_spikes, "r")
    # ------------------------- set binsize and boxcal sliding window --------------------------------------
    binsize = f["params"]["binsize"].value
    winlength = avgwindow*binsize

    novelty_overlap = np.zeros((Nseq,Nblocks,3))
    novelty_indices = np.zeros((Nseq,Nblocks,200))

    hist_E  = []
    mean_hist_E = []
    hist_I  = []
    mean_hist_I = []
    hist_E_nomem  = []
    mean_hist_E_nomem = []
    hist_E_nomemnonov  = []
    mean_hist_E_nomemnonov = []
    hist_E_nov  = []
    mean_hist_E_nov = []
    hist_E_boxcar  = []
    mean_hist_E_boxcar = []
    hist_I_boxcar  = []
    mean_hist_I_boxcar = []
    edges = []
    novelty_overlap = []
    #novelty_indices = []

    for seq in range(1,Nseq + 1):
        edges.append(f["E%dmsedges" % binsize]["seq"+ str(seq) + "block"+ str(1)].value)
        hist_E.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_I.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_nomem.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_nov.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_nomemnonov.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_boxcar.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_I_boxcar.append(np.zeros((Nblocks,len(edges[seq-1]))))
        novelty_overlap.append(np.zeros(Nblocks))

        for bl in range(1, Nblocks + 1):
            #vars()['hist_E_all' + str(seq-1)][bl-1][:] = f["E%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E[seq-1][bl-1,:] = f["E%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_I[seq-1][bl-1,:] = f["I%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_nomem[seq-1][bl-1,:] = f["ENonMem%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_nov[seq-1][bl-1,:] = f["Nov%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_nomemnonov[seq-1][bl-1,:] = f["ENonMemNoNov%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_boxcar[seq-1][bl-1,:] = np.convolve(hist_E[seq-1][bl-1,:], np.ones((avgwindow,))/avgwindow, mode='same')
            hist_I_boxcar[seq-1][bl-1,:] = np.convolve(hist_I[seq-1][bl-1,:], np.ones((avgwindow,))/avgwindow, mode='same')
            novelty_overlap[seq-1][bl-1] = f["noveltyoverlap"]["seq"+ str(seq) + "block"+ str(bl)].value[0]
        # get averages over blocks
        mean_hist_E.append(np.mean(hist_E[seq-1][:,:],axis = 0))
        mean_hist_I.append(np.mean(hist_I[seq-1][:,:],axis = 0))
        mean_hist_E_nomem.append(np.mean(hist_E_nomem[seq-1][:,:],axis = 0))
        mean_hist_E_nomemnonov.append(np.mean(hist_E_nomemnonov[seq-1][:,:],axis = 0))
        mean_hist_E_boxcar.append(np.mean(hist_E_boxcar[seq-1][:,:],axis = 0))
        mean_hist_I_boxcar.append(np.mean(hist_I_boxcar[seq-1][:,:],axis = 0))
        mean_hist_E_nov.append(np.mean(hist_E_nov[seq-1][:,:],axis = 0))


    # close file
    f.close()
    # plotting
    if Nblocks == 1:
        colorE = ["midnightblue"]*150
        colorI = ["darkred"]*150
    else:
        colorE = ["midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon"]
        colorI = ["midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon"]
    color = ["midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon"]
    colorI = ["darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred","darkred"]
    idxconv = np.floor_divide(avgwindow,2)+1

    return mean_hist_E,mean_hist_I, edges[0],hist_E, hist_E_boxcar, figure_directory




def analyse_weights(file_name, ifcontin,  indivassembly = True, figsize=(20,10),ncol = 1,RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR

    # define folder where figues should be stored
    figure_directory = results_folder + file_name + "/" + "weightfigures/"
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    # read in run parameters
    file_name_run = run_folder + file_name
    # open file
    frun = h5py.File(file_name_run, "r")

    # read in stimulus parameters
    Nimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength  = frun["initial"]["stimparams"].value
    seqnumber  = frun["initial"]["seqnumber"].value
    stimulus  = frun["initial"]["stimulus"].value
    idxblockonset  = frun["initial"]["idxblockonset"].value

    assemblymembers = frun["initial"]["assemblymembers"].value.transpose()
        # close file
    if Nseq > 5:
        ncol = 2

    # read in weights
    # get length of
    avgassemblytononmemscount = 0
    avgXassemblycount = 0
    avgItoassemblycount = 0
    avgnoveltytoassemblycount = 0
    keysavgweights = frun["dursimavg"].keys()
    for text in keysavgweights:
        if "avgnoveltytoassembly" in text:
            avgnoveltytoassemblycount += 1
        if "avgItoassembly" in text:
            avgItoassemblycount += 1
        if "avgXassembly" in text:
            avgXassemblycount += 1
        if "avgassemblytononmems" in text:
            avgassemblytononmemscount += 1


    dtsaveweights = frun["params"]["dtsaveweights"].value * 10 # convert in 0.1 ms # how often are the weights stored
    modwstore = frun["params"]["modwstore"].value # convert in 0.1 ms
    minwstore = frun["params"]["minwstore"].value # convert in 0.1 ms
    if ifcontin:
        tts = range(1,avgXassemblycount+1)
    else:
        tts = range(1,minwstore+1)
        tts.extend(range(minwstore + modwstore, minwstore + modwstore + (avgXassemblycount-minwstore)*modwstore, modwstore))

    Ni=1000
    Nass = np.size(assemblymembers,axis= 0)


    Xweight = np.zeros((Nass,Nass,avgXassemblycount))
    InhibXweight = np.zeros((Nass,Nass,avgXassemblycount))

    ItoAweight = np.zeros((Nass,avgXassemblycount))
    Etononmensweight = np.zeros((Nass,avgXassemblycount))
    noveltytoAweight = np.zeros((Nass,avgXassemblycount))
    nonmenstoEweight = np.zeros((Nass,avgXassemblycount))
    Atonoveltyweight = np.zeros((Nass,avgXassemblycount))
    Itoneuron1 = []
    Itoneuron2 = []

    timevector = np.zeros(len(tts))
    timecounter = 0
    for tt in tts:
        #print(tt)
        timett = tt*dtsaveweights/0.1
        timevector[timecounter] = timett/6000000 # in min
        #weightname = "avgXassembly%d_%d" % (tt, (tt*dtsaveweights))
        Xweight[:,:,timecounter] = frun["dursimavg"]["avgXassembly%d_%d" % (tt, (tt*dtsaveweights))].value.transpose()
        InhibXweight[:,:,timecounter] = frun["dursimavg"]["avgInhibXassembly%d_%d" % (tt, (tt*dtsaveweights))].value.transpose()

        ItoAweight[:,timecounter] = frun["dursimavg"]["avgItoassembly%d_%d" % (tt, (tt*dtsaveweights))].value
        Etononmensweight[:,timecounter] = frun["dursimavg"]["avgnonmemstoassembly%d_%d" % (tt, (tt*dtsaveweights))].value
        nonmenstoEweight[:,timecounter] = frun["dursimavg"]["avgassemblytononmems%d_%d" % (tt, (tt*dtsaveweights))].value
        noveltytoAweight[:,timecounter] = frun["dursimavg"]["avgnoveltytoassembly%d_%d" % (tt, (tt*dtsaveweights))].value
        Atonoveltyweight[:,timecounter] = frun["dursimavg"]["avgassemblytonovelty%d_%d" % (tt, (tt*dtsaveweights))].value

        Itoneuron1.append(frun["dursimavg"]["Itoneuron1%d_%d" % (tt, (tt*dtsaveweights))].value)
        Itoneuron2.append(frun["dursimavg"]["Itoneuron2%d_%d" % (tt, (tt*dtsaveweights))].value)
        #plotavgweightmatrix(Xweight[:,:,timecounter], maxval= 14)
        timecounter += 1
    # plotavgweightmatrix(Xweight[:,:,-1], maxval = 14)
    # save_fig(figure_directory, "Final_avgweightmatrix")
    #
    # plotavgweightmatrix(InhibXweight[:,:,-1], maxval= 255)
    # save_fig(figure_directory, "Final_Inhibavgweightmatrix")

    frun.close()
    if Nimg == 4:
        color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange", "black", "silver","dimgrey","gainsboro", "fuchsia", "orchid","plum", "mediumvioletred", "lightseagreen", "lightcyan", "darkslategray", "paleturquoise", "goldenrod","gold", "wheat","darkgoldenrod", "forestgreen", "aquamarine", "palegreen", "lime", ]
    elif Nimg == 3:
        color = ["midnightblue","lightskyblue","royalblue","darkred","darksalmon", "saddlebrown","darkgreen","greenyellow","darkolivegreen","darkmagenta","thistle","indigo","darkorange","tan","sienna", "black", "silver","dimgrey", "fuchsia", "orchid","plum",  "lightseagreen", "lightcyan", "darkslategray",  "goldenrod","gold", "wheat","forestgreen", "aquamarine", "palegreen"]
    elif Nimg == 5:
        color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","indigo","darkred","darksalmon", "saddlebrown","lightcoral","rosybrown","darkgreen","greenyellow","darkolivegreen","chartreuse","turquoise","darkmagenta","thistle","indigo","mediumslateblue","blueviolet","darkorange","tan","sienna","orange", "brown", "black", "silver","dimgrey","gainsboro", "grey","fuchsia", "orchid","plum", "mediumvioletred","purple", "lightseagreen", "lightcyan", "darkslategray", "paleturquoise","teal", "goldenrod","gold", "wheat","darkgoldenrod", "darkkhaki","forestgreen", "aquamarine", "palegreen", "lime", "darkseagreen"]
    else:
        color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","indigo","darkred","darksalmon", "saddlebrown","lightcoral","rosybrown","darkgreen","greenyellow","darkolivegreen","chartreuse","turquoise","darkmagenta","thistle","indigo","mediumslateblue","blueviolet","darkorange","tan","sienna","orange", "brown", "black", "silver","dimgrey","gainsboro", "grey","fuchsia", "orchid","plum", "mediumvioletred","purple", "lightseagreen", "lightcyan", "darkslategray", "paleturquoise","teal", "goldenrod","gold", "wheat","darkgoldenrod", "darkkhaki","forestgreen", "aquamarine", "palegreen", "lime", "darkseagreen"]

    colormain = np.copy(color[0:-1:Nimg])

    startidx = np.arange(0,Nseq*Nimg,Nimg)
    seqnum = Nseq + 1

    #fig = plt.figure(figsize=figsize)
    avgweightEass = np.zeros((Nass,len(timevector)))

    # plot avg weight development check if that causes spikes
    for i in reversed(range(2)):
        avgweightEass[i,:] += Xweight[i,i,:]

    avgweightInhibEass = np.zeros((Nass,len(timevector)))

    # plot avg weight development check if that causes spikes
    for i in reversed(range(2)):
        avgweightInhibEass[i,:] += InhibXweight[i,i,:]


    avgweightE = np.mean(avgweightEass, axis = 0)
    avgweightEmem = np.mean(avgweightEass[0:Nimg*Nseq,:], axis = 0)
    avgweightEnov = np.mean(avgweightEass[Nimg*Nseq:,:], axis = 0)
    stdweightE = np.std(avgweightEass, axis = 0)
    stdweightEmem = np.std(avgweightEass[0:Nimg*Nseq,:], axis = 0)
    stdweightEnov = np.std(avgweightEass[Nimg*Nseq:,:], axis = 0)

    InhibavgweightE = np.mean(avgweightInhibEass, axis = 0)
    InhibavgweightEmem = np.mean(avgweightInhibEass[0:Nimg*Nseq,:], axis = 0)
    InhibavgweightEnov = np.mean(avgweightInhibEass[Nimg*Nseq:,:], axis = 0)
    InhibstdweightE = np.std(avgweightInhibEass, axis = 0)
    InhibstdweightEmem = np.std(avgweightInhibEass[0:Nimg*Nseq,:], axis = 0)
    InhibstdweightEnov = np.std(avgweightInhibEass[Nimg*Nseq:,:], axis = 0)

    avgweightI = np.mean(ItoAweight, axis = 0)
    avgweightImem = np.mean(ItoAweight[0:Nimg*Nseq,:], axis = 0)
    avgweightInov = np.mean(ItoAweight[Nimg*Nseq:,:], axis = 0)
    stdweightI = np.std(ItoAweight, axis = 0)
    stdweightImem = np.std(ItoAweight[0:Nimg*Nseq,:], axis = 0)
    stdweightInov = np.std(ItoAweight[Nimg*Nseq:,:], axis = 0)

    return Xweight, ItoAweight, timevector, avgweightEmem, avgweightImem, avgweightEnov, avgweightImem, Itoneuron1, Itoneuron2, InhibXweight, seqnumber, stimulus, colormain, idxblockonset
