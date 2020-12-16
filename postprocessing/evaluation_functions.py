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
#              run_single_neuron_eval                           #
#                                                               #
#                                                               #
#################################################################

def run_single_neuron_eval(file_name, binwidth = 50, avgwindow = 5, timestr = "_now", RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR
    # input: file to be evaluated
    # smoothing parameters rebinning binwidth in ms, smoothing window N N*binwidth in ms

    #output: figures of whole plane after moving avg, zscore, sparseness analysis
    #    file_name = "dur1.624e6mslenstim300lenpause0Nreps20strength8wadaptfalseiSTDPtrueTime2019-04-09-21-45-25repeatedsequences.h5"

    file_name_results = results_folder + file_name + "/results.h5"
    #f_results = h5py.File(file_name_results, "w")

    # define folder where figues should be stored
    figure_directory = results_folder + file_name + "/" + "singlefigures%d/"% (avgwindow*binwidth)
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    # read in run parameters
    file_name_run = run_folder + file_name
    # open file
    frun = h5py.File(file_name_run, "r")

    # read in stimulus parameters
    Nimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength  = frun["initial"]["stimparams"].value
    Ni = frun["params"]["Ni"].value
    Ne = frun["params"]["Ne"].value
    Ncells = Ni + Ne
    seqnumber  = frun["initial"]["seqnumber"].value
    assemblymembers = frun["initial"]["assemblymembers"].value.transpose()
    color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange", "midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange"]
    Nblocks = min([Nblocks, 10])
    # get indices of all assembly members and novelty members as well as untargeted neuron indices
    Nass = Nimg*Nseq
    members = assemblymembers[0:Nass,:]
    novelty = assemblymembers[Nass:,:]

    membersidx = np.unique(members[members>0])
    noveltyidx = np.unique(novelty[novelty>0])

    untargetedidx = np.ones(Ncells, dtype=bool)
    untargetedidx[membersidx-1] = False
    untargetedidx[noveltyidx-1] = False
    untargetedidx[Ne:] = False
    inhibitoryidx = np.linspace(Ne,Ncells-1,Ncells-Ne, dtype=int)

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
    # bin width
    lenblocktt  = f["params"]["lenblocktt"].value
    blockbegintt  = f["params"]["blockbegintt"].value
    dt = 0.1
    nbins = int(np.round(lenblocktt*dt/binwidth+1))
    bins = np.linspace(0,lenblocktt, nbins)
    bin_counts = np.zeros((Ncells,nbins-1))
    conv = np.zeros((Ncells,nbins-1), dtype=float)

    timevector = np.linspace((binwidth/2)/1000.,lenblocktt*dt/1000. - (binwidth/2)/1000.,nbins-1)

    all_bincounts = np.zeros((Nseq,Nblocks,Ncells, nbins-1))
    all_bincounts_conv = np.zeros((Nseq,Nblocks,Ncells, nbins-1))
    all_zscore_firing = np.zeros((Nseq,Nblocks,Ncells, nbins-1))

    noveltyonset = (Nimg*(Nreps-1)*(lenstim+lenpause)-(lenstim+lenpause))/1000. #convert to seconds
    idxstart = int((noveltyonset - 2)*10)
    idxend = int((noveltyonset + 3)*10)
    winlength = avgwindow*binwidth

    idxconv = np.floor_divide(avgwindow,2)+1 # account for convolution in mode same ignore first len(kernel)/2 samples
    for seq in range(1,Nseq + 1):
        for bl in range(1,Nblocks + 1):
            # read in spiketimes
            spiketimes = f["spiketimeblocks"]["seq"+ str(seq) + "block"+ str(bl)].value.transpose()

            # rebin spiketimes in binsize of 100 ms
            for cc in range(0,Ncells):
                 all_bincounts[seq-1,bl-1,cc,:], bin_edges = np.histogram(spiketimes[cc,:], bins=bins)#lenblocktt/1000)
            spiketimes = 0
            gc.collect()
            # get firing rates by dividing by the length of the binwidth in seconds
            bin_edges = bin_edges*dt/1000 # seconds
            all_bincounts[seq-1,bl-1,:,:] = all_bincounts[seq-1,bl-1,:,:]*(1000/binwidth) # divide by binwidth in seconds to get rate
            # apply moving average across 8 bins
            for cc in range(0,Ncells):
                all_bincounts_conv[seq-1,bl-1,cc,1:] = np.convolve(all_bincounts[seq-1,bl-1,cc,1:], np.ones((avgwindow,))/avgwindow, mode='same')# pay attnetion to first and last 800 ms altered due to zero padding

            #---------------------------- zscore -----------------------------------------------------
            # evaluate std and avg of all neurons
            # eval zscore and plot for novelty region
            # order according to value

            # store all variables
    #         f_results = h5py.File(file_name_results, "w")
    #         f_results.create_dataset('single_neurons/timevectorSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=timevector)
    #         f_results.create_dataset('single_neurons/firing_rates_smoothedSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=conv)
    #         f_results.create_dataset('single_neurons/zscore_firingSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=zscore_firing)
    #         f_results.create_dataset('single_neurons/binwidthSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=binwidth)
    #         f_results.create_dataset('single_neurons/bin_edgesSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=bin_edges)
    #         f_results.create_dataset('single_neurons/bin_countsSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=bin_counts)
    #         f_results.create_dataset('single_neurons/mean_firingSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=mean_firing)
    #         f_results.create_dataset('single_neurons/std_firingSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=std_firing)
    #         f_results.create_dataset('single_neurons/noveltyonsetSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=noveltyonset)
    #         f_results.close()
    # f_results = h5py.File(file_name_results, "w")
    # f_results.create_dataset('single_neurons/all_bincountsBinsize%d' % (avgwindow*binwidth), data=all_bincounts)
    # f_results.create_dataset('single_neurons/all_bincounts_convBinsize%d' % (avgwindow*binwidth), data=all_bincounts_conv)
    # f_results.close()
                # -------------------------- evaluate denseness ------------------------------------


    mean_firing = np.mean(all_bincounts_conv[:,:,:,idxconv:-idxconv], axis=3)
    std_firing = np.std(all_bincounts_conv[:,:,:,idxconv:-idxconv], axis=3)

    # z-score for whole dataset
    zscore_firing = stats.zscore(all_bincounts_conv[:,:,:,idxconv:-idxconv], axis=3) # get zscore for all neurons fr all blcosk and all sequneces ignoring the invalid convolution part
    zscore_firing[np.isnan(zscore_firing)] = 0
    # -------------------------- evaluate denseness ------------------------------------

    count_zscore = np.copy(zscore_firing)
    count_zscore[count_zscore < 0] = -1
    count_zscore[count_zscore > 0] = 1


    # take average of firing rates across blocks
    avg_bincounts_conv_blocks = np.mean(all_bincounts_conv[:,:,:,idxconv:-idxconv],axis = 1)
    #determine zscore of this "double" average
    zscore_over_all_blocks = stats.zscore(avg_bincounts_conv_blocks, axis = 2) # zscore over 240 time
    zscore_over_all_blocks[np.isnan(zscore_over_all_blocks)] = 0
    count_zscore_avg_blocks = np.copy(zscore_over_all_blocks)
    count_zscore_avg_blocks[zscore_over_all_blocks < 0] = -1
    count_zscore_avg_blocks[zscore_over_all_blocks > 0] = 1



    # take average of firing rates across both blocks and sequences
    avg_bincounts_conv_blocks_seq = np.mean(np.mean(all_bincounts_conv[:,:,:,idxconv:-idxconv],axis = 0),axis = 0)
    #determine zscore of this "double" average
    zscore_over_all_blocks_seq = stats.zscore(avg_bincounts_conv_blocks_seq, axis = 1)
    zscore_over_all_blocks_seq[np.isnan(zscore_over_all_blocks_seq)] = 0

    # count of this "double" average
    count_zscore_avg_blocks_seq = np.copy(zscore_over_all_blocks_seq)
    count_zscore_avg_blocks_seq[zscore_over_all_blocks_seq < 0] = -1
    count_zscore_avg_blocks_seq[zscore_over_all_blocks_seq > 0] = 1



    # # -------------------------- evaluate denseness ------------------------------------
    # # --------------- zscore mean ------------------
    zscore_meanE = np.mean(zscore_firing[:,:,0:Ne,:], axis=2)
    zscore_meanI = np.mean(zscore_firing[:,:,Ne:,:], axis=2)
    zscore_stdE = np.std(zscore_firing[:,:,0:Ne,:], axis=2)
    zscore_stdI = np.std(zscore_firing[:,:,Ne:,:], axis=2)


    # -------------------------- evaluate denseness ------------------------------------
    # determine 4 relevant x positions in the array
    # not nice but works for now
    maxtemp = max(timevector)
    if maxtemp > 20:
        xpos = np.array([50,100,150,200])*100/binwidth-idxconv
        xlab = ["5","10","15","20"]
    elif maxtemp > 15:
        xpos = np.array([50,100,150])*100/binwidth-idxconv
        xlab = ["5","10","15"]
    elif maxtemp > 10:
        xpos = np.array([50,100])*100/binwidth-idxconv
        xlab = ["5","10"]
    elif maxtemp > 7.5:
        xpos = np.array([25,50,75])*100/binwidth-idxconv
        xlab = ["2.5","5","7.5"]
    elif maxtemp > 5:
        xpos = np.array([25,50])*100/binwidth-idxconv
        xlab = ["2.5","5"]
    else:
        xpos = np.array([0,10])*100/binwidth-idxconv
        xlab = ["0","1"]
        # --------------- plotting ------------------
    #plt.ioff()

    ifplotting = True
    if ifplotting:
        for seq in range(1,Nseq+1):
            # plot the zscore counts for each individual sequence  average out Nblocks
            plotzscorecounts(timevector[idxconv:-idxconv], count_zscore_avg_blocks[seq-1], seq, 0, ifBlockAvg=True, figure_directory=figure_directory, noveltyonset=noveltyonset, ifExcitatory = True, Nseq = Nseq)#
            plotzscorecounts(timevector[idxconv:-idxconv], count_zscore_avg_blocks[seq-1], seq, 0, ifBlockAvg=True, figure_directory=figure_directory, noveltyonset=noveltyonset, ifExcitatory = False, Nseq = Nseq)#
            for bl in range(1,Nblocks+1):
                # fig = plt.figure(figsize=(15,15))
                # plot_mean_with_errorband_mult(fig,timevector[idxconv:-idxconv], zscore_meanE[seq-1,bl-1,:], zscore_stdE[seq-1,bl-1,:],  legend="E", color = "darkblue", iflegend=True, ifioff=True, Nseq = Nseq)#
                # plot_mean_with_errorband_mult(fig,timevector[idxconv:-idxconv], zscore_meanI[seq-1,bl-1,:], zscore_stdI[seq-1,bl-1,:],  legend="I", color = "darkred", iflegend=True, ifioff=True, Nseq = Nseq)#
                # save_fig(figure_directory, "IdxMeanZscoreIandESeq%dBlock%d" % (seq,bl))
                # plotzscorecounts(timevector[idxconv:-idxconv],count_zscore[seq-1,bl-1,:,:], seq, bl, figure_directory=figure_directory, noveltyonset=noveltyonset,ifExcitatory = True, Nseq = Nseq)#
                # plotzscorecounts(timevector[idxconv:-idxconv],count_zscore[seq-1,bl-1,:,:], seq, bl, figure_directory=figure_directory, noveltyonset=noveltyonset,ifExcitatory = False, Nseq = Nseq)#
                plotrateplanewithavg(timevector[idxconv:-idxconv], all_bincounts_conv[:,:,0:100,idxconv:-idxconv], seq, bl, x_positions = xpos,x_labels = xlab,cmap = "inferno", figure_directory=figure_directory, idxstart = 0, ifExcitatory = True,savetitle = "firing_rates_first100Neurons", Nseq = Nseq,ififnorm = False,midpoint=3, ylimbar = 350, ifylimbar = True)#
                #plotrateplanewithavg(timevector[idxconv:-idxconv],all_bincounts_conv[:,:,Ne:(Ne+100),idxconv:-idxconv], seq, bl, x_positions = xpos,x_labels = xlab,figure_directory=figure_directory, idxstart = 0, ifExcitatory = False,savetitle = "firing_rates_first100Neurons", Nseq = Nseq)#
                # plotrateplane(timevector[idxconv:-idxconv],all_zscore_firing, seq, bl, x_positions = xpos,x_labels = xlab, cmap = "coolwarm",figure_directory=figure_directory, idxstart = 0, ifExcitatory = True, savetitle = "zscore", cbarlabel="z-score", ififnorm = True, Nseq = Nseq)#
                # plotrateplane(timevector[idxconv:-idxconv],all_zscore_firing, seq, bl, x_positions = xpos,x_labels = xlab, cmap = "coolwarm",figure_directory=figure_directory, idxstart = 0, ifExcitatory = False, savetitle = "zscore", cbarlabel="z-score", ififnorm = True, Nseq = Nseq)#

        plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(zscore_firing,axis = 1), x_positions = xpos,x_labels = xlab,cmap = "coolwarm",figure_directory=figure_directory, idxstart = 0, ifExcitatory = False, savetitle = "zscore", cbarlabel="z-score", ififnorm = True, Nseq = Nseq)#
        plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(zscore_firing,axis = 1), x_positions = xpos,x_labels = xlab,cmap = "coolwarm",figure_directory=figure_directory, idxstart = 0, ifExcitatory = True, savetitle = "zscore", cbarlabel="z-score", ififnorm = True, Nseq = Nseq)#


        plotzscorecounts(timevector[idxconv:-idxconv],count_zscore_avg_blocks_seq, 0, 0, ifAvg=True, figure_directory=figure_directory, noveltyonset=noveltyonset, ifExcitatory = True, Nseq = Nseq)#
        plotzscorecounts(timevector[idxconv:-idxconv],count_zscore_avg_blocks_seq, 0, 0, ifAvg=True, figure_directory=figure_directory, noveltyonset=noveltyonset, ifExcitatory = False, Nseq = Nseq)#

        plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,:,idxconv:-idxconv],axis = 1), cmap = "inferno",idxstart = 0, x_positions = xpos,x_labels = xlab, figure_directory = figure_directory, ifExcitatory = True, Nseq = Nseq,ififnorm = False)#
        plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,:,idxconv:-idxconv],axis = 1), cmap = "inferno",idxstart = 0, x_positions = xpos,x_labels = xlab, figure_directory = figure_directory, ifExcitatory = False, Nseq = Nseq,ififnorm = False)#
        plotaveragerateplanewithavg(timevector[idxconv:-idxconv],avg_bincounts_conv_blocks, idxstart = 0, x_positions = xpos,cmap = "inferno",x_labels = xlab,figure_directory = figure_directory, ifExcitatory = False, Nseq = Nseq,ififnorm = False, savetitle = "firing_rates_allNeurons")#
        plotaveragerateplanewithavg(timevector[idxconv:-idxconv],avg_bincounts_conv_blocks, idxstart = 0, x_positions = xpos,cmap = "inferno",x_labels = xlab,figure_directory = figure_directory, ifExcitatory = True, Nseq = Nseq,ififnorm = False,savetitle = "firing_rates_allNeurons")#
        #%plotaveragerateplanewithavg(timevector[idxconv:-idxconv],avg_bincounts_conv_blocks, idxstart = 0, x_positions = xpos,cmap = "inferno",x_labels = xlab,figure_directory = figure_directory, ifExcitatory = False, Nseq = Nseq,ififnorm = False, savetitle = "firing_rates_allNeurons")#
        plotaveragerateplanewithavg(timevector[idxconv:-idxconv],avg_bincounts_conv_blocks[:,0:100,idxconv:-idxconv], idxstart = 0, x_positions = xpos,cmap = "inferno",x_labels = xlab,figure_directory = figure_directory, ifExcitatory = True, Nseq = Nseq,ififnorm = False,savetitle = "firing_rates_first100Neurons_withavg")#
        plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,0:100,idxconv:-idxconv],axis = 1), cmap = "inferno",idxstart = 0, x_positions = xpos,x_labels = xlab, figure_directory = figure_directory, ifExcitatory = True, savetitle = "firing_rates_first100Neurons", Nseq = Nseq,ififnorm = False)#, cmap = "parula")
# plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,:,idxconv:-idxconv],axis = 1), cmap = "bone_r",idxstart = 0, x_positions = xpos,x_labels = xlab, figure_directory = figure_directory, ifExcitatory = True, Nseq = Nseq)#
# plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,:,idxconv:-idxconv],axis = 1), cmap = "bone_r",idxstart = 0, x_positions = xpos,x_labels = xlab, figure_directory = figure_directory, ifExcitatory = False, Nseq = Nseq)#
# plotaveragerateplanewithavg(timevector[idxconv:-idxconv],avg_bincounts_conv_blocks, idxstart = 0, x_positions = xpos,cmap = "bone_r",x_labels = xlab,figure_directory = figure_directory, ifExcitatory = False, Nseq = Nseq)#
# plotaveragerateplanewithavg(timevector[idxconv:-idxconv],avg_bincounts_conv_blocks, idxstart = 0, x_positions = xpos,cmap = "bone_r",x_labels = xlab,figure_directory = figure_directory, ifExcitatory = True, Nseq = Nseq)#
# plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,0:100,idxconv:-idxconv],axis = 1), cmap = "bone_r",idxstart = 0, x_positions = xpos,x_labels = xlab, figure_directory = figure_directory, ifExcitatory = True, savetitle = "firing_rates_first100Neurons", Nseq = Nseq)#, cmap = "parula")

        #plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,Ne:(Ne+100),idxconv:-idxconv],axis = 1), cmap = "bone_r", idxstart = 0, x_positions = xpos,x_labels = xlab, figure_directory = figure_directory, ifExcitatory = False, savetitle = "firing_rates_first100Neurons", Nseq = Nseq)##, cmap = "parula")
        # plotaveragerateplanewithavg(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,0:100,idxconv:-idxconv],axis = 1), cmap = "bone_r", idxstart = 0, x_positions = xpos,x_labels = xlab,figure_directory = figure_directory, ifExcitatory = False,savetitle = "firing_rates_first100Neurons_with_avg", Nseq = Nseq)#
        # plotaveragerateplanewithavg(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,Ne:(Ne+100),idxconv:-idxconv],axis = 1), cmap = "bone_r", idxstart = 0, x_positions = xpos,x_labels = xlab,figure_directory = figure_directory, ifExcitatory = True,savetitle = "firing_rates_first100Neurons_with_avg", Nseq = Nseq)#
# plotaveragerateplanewithavg(timevector[idxconv:-idxconv],avg_bincounts_conv_blocks[0:100,idxconv:-idxconv], cmap = "bone_r", idxstart = 0, x_positions = xpos,x_labels = xlab,figure_directory = figure_directory, ifExcitatory = False,savetitle = "firing_rates_first100Neurons_with_avg", Nseq = Nseq)#
# plotaveragerateplanewithavg(timevector[idxconv:-idxconv],avg_bincounts_conv_blocks[0:100,idxconv:-idxconv], cmap = "bone_r", idxstart = 0, x_positions = xpos,x_labels = xlab,figure_directory = figure_directory, ifExcitatory = True,savetitle = "firing_rates_first100Neurons_with_avg", Nseq = Nseq)#

    ifhistograms = True
    colorseq = ["midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan"]
    colorblocks = ["lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan"]
    if Nimg == 4:
        color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange", "black", "silver","dimgrey","gainsboro", "fuchsia", "orchid","plum", "mediumvioletred", "lightseagreen", "lightcyan", "darkslategray", "paleturquoise", "goldenrod","gold", "wheat","darkgoldenrod", "forestgreen", "aquamarine", "palegreen", "lime", ]
    elif Nimg == 3:
        color = ["midnightblue","lightskyblue","royalblue","darkred","darksalmon", "saddlebrown","darkgreen","greenyellow","darkolivegreen","darkmagenta","thistle","indigo","darkorange","tan","sienna", "black", "silver","dimgrey", "fuchsia", "orchid","plum",  "lightseagreen", "lightcyan", "darkslategray",  "goldenrod","gold", "wheat","forestgreen", "aquamarine", "palegreen"]
    elif Nimg == 5:
        color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","indigo","darkred","darksalmon", "saddlebrown","lightcoral","rosybrown","darkgreen","greenyellow","darkolivegreen","chartreuse","turquoise","darkmagenta","thistle","indigo","mediumslateblue","blueviolet","darkorange","tan","sienna","orange", "brown", "black", "silver","dimgrey","gainsboro", "grey","fuchsia", "orchid","plum", "mediumvioletred","purple", "lightseagreen", "lightcyan", "darkslategray", "paleturquoise","teal", "goldenrod","gold", "wheat","darkgoldenrod", "darkkhaki","forestgreen", "aquamarine", "palegreen", "lime", "darkseagreen"]
    elif Nimg == 1:
        color = ["midnightblue","saddlebrown","darkorange"]
    else:
        color = ["midnightblue","saddlebrown","darkorange"]
    colormain = np.copy(color[0:-1:Nimg])
    #color = ["midnightblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange"]
    ifhistograms = False
    if ifhistograms:
        avg_bincounts_conv_time = np.mean(all_bincounts_conv[:,:,:,idxconv:-idxconv],axis = 3)
        print(Nblocks)
        # plot histograms of the average firing rates (average over whole block length)
        # make individual histograms per sequence showing all distributions per blocks
        # make individual histograms per block showing all distributions for all sequences
        for bl in [1,5,9]:#[1,10,19]:#range(1,Nblocks+1):
            plot_histograms_seq_cut(avg_bincounts_conv_time,bl-1,membersidx-1,colorseq=colorseq,bins = np.linspace(0,70,51),alpha = 0.6, Nblocks =Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            save_fig(figure_directory, "HistogramMembersBlock%d" % bl)

        for seq in [1,3,5]:#range(1,Nseq+1):
            plot_histograms_block_cut(avg_bincounts_conv_time,seq-1,membersidx-1,colorseq=colorblocks,bins = np.linspace(0,70,51),alpha = 0.6, Nblocks =Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            save_fig(figure_directory, "HistogramMembersSeq%d" % seq)

        for bl in [1,5,9]:#[1,10,19]:#range(1,Nblocks+1):
            plot_histograms_seq_cut(avg_bincounts_conv_time,bl-1,noveltyidx-1,cutlow=400,cuthigh=800,bins = np.linspace(0,70,51),colorseq=colorseq,alpha = 0.6, Nblocks =Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            save_fig(figure_directory, "HistogramNoveltyBlock%d" % bl)

        for seq in [1,3,5]:#range(1,Nseq+1):
            plot_histograms_block_cut(avg_bincounts_conv_time,seq-1,noveltyidx-1,cutlow=400,cuthigh=800,bins = np.linspace(0,70,51),colorseq=colorblocks,alpha = 0.6, Nblocks =Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            save_fig(figure_directory, "HistogramNoveltySeq%d" % seq)

        for bl in [1,5,9]:#[1,10,19]:#range(1,Nblocks+1):
            plot_histograms_seq(avg_bincounts_conv_time,bl-1,untargetedidx,colorseq=colorseq,bins = np.linspace(0,30,31),alpha = 0.6, Nblocks =Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            save_fig(figure_directory, "HistogramUntargetedBlock%d" % bl)

        for seq in [1,3,5]:#range(1,Nseq+1):
            plot_histograms_block(avg_bincounts_conv_time,seq-1,untargetedidx,colorseq=colorblocks,bins = np.linspace(0,30,31),alpha = 0.6, Nblocks = Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            save_fig(figure_directory, "HistogramUntargetedSeq%d" % seq)

        for bl in [1,5,9]:#[1,10,19]:#range(1,Nblocks+1):
            plot_histograms_seq(avg_bincounts_conv_time,bl-1,inhibitoryidx,colorseq=colorseq,bins = np.linspace(0,45,91),alpha = 0.6, Nblocks = Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            save_fig(figure_directory, "HistogramInhibitoryBlock%d" % bl)

        for seq in [1,3,5]:#range(1,Nseq+1):
            plot_histograms_block(avg_bincounts_conv_time,seq-1,inhibitoryidx,bins = np.linspace(0,45,91),colorseq=colorblocks,alpha = 0.6, Nblocks = Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            save_fig(figure_directory, "HistogramInhibitorySeq%d" % seq)

    ifpooledresponses = False
    if ifpooledresponses:

        # replicate figure 6D sustained activity vs. cell rank
        starttime = 10 # sec
        endtime = 15.5 # sec
        startidx = np.argmax(timevector>starttime)
        endidx = np.argmax(timevector>endtime)
        starttimeT = 0.5 # sec
        endtimeT = 3.5 # sec
        startidxT = np.argmax(timevector>starttimeT)
        endidxT = np.argmax(timevector>endtimeT)
        starttimeN = 16.7 # sec
        endtimeN = 17.2 # sec
        startidxN = np.argmax(timevector>starttimeN)
        endidxN = np.argmax(timevector>endtimeN)
        print(startidxN)
        print(endidxN)
        endidxN = min(endidxN, np.size(all_bincounts_conv,axis = 3))
        #get the average firing rates for each sequence individually
        avg_rates_time_seqE = np.mean(np.mean(all_bincounts_conv[:,:,:Ne,startidx:endidx],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(np.mean(all_bincounts_conv[:,:,Ne:,startidx:endidx],axis = 3),axis=1) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks
        # plot pooled histograms
        plot_histogram_cut(avg_rates_time_seq_pooledE, cutlow=1500, cuthigh=1600,ifExcitatory=True, alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledSustainedResponsesE")
        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledSustainedResponsesI")

        # get ranked sustained repsonses
        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = True, figsize=(5,4),xticks = [0,5000,10000,15000,20000])
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedStustainedResponsesE")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedStustainedResponsesI")

        # get the average firing rates for each sequence individually for whole time
        avg_rates_time_seqE = np.mean(np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(np.mean(all_bincounts_conv[:,:,Ne:,idxconv:-idxconv],axis = 3),axis=1) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks

        plot_histogram_cut(avg_rates_time_seq_pooledE, cutlow=1500, cuthigh=1600,ifExcitatory=True, alpha=1, Nblocks = Nblocks,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramAllSustainedResponsesE")

        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramAllSustainedResponsesI")

        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = True, figsize=(5,4),xticks = [0,5000,10000,15000,20000])
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedAllResponsesE")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedAllResponsesI")

        # Transient firing rates

        # get the average firing rates for each sequence individually
        avg_rates_time_seqE = np.mean(np.mean(all_bincounts_conv[:,:,:Ne,startidxT:endidxT],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(np.mean(all_bincounts_conv[:,:,Ne:,startidxT:endidxT],axis = 3),axis=1) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks
        # plot pooled histograms
        plot_histogram_cut(avg_rates_time_seq_pooledE, cutlow=1000, cuthigh=1100,ifExcitatory=True, alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledTransientResponsesE")
        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledTransientResponsesI")

        # get ranked sustained repsonses
        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = True, xticks = [0,5000,10000,15000,20000],figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedTransientResponsesE")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedTransientResponsesI")

        # Novelty firing rates
        # get the average firing rates for each sequence individually
        avg_rates_time_seqE = np.mean(np.mean(all_bincounts_conv[:,:,:Ne,startidxN:endidxN],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(np.mean(all_bincounts_conv[:,:,Ne:,startidxN:endidxN],axis = 3),axis=1) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks
        # plot pooled histograms
        plot_histogram_cut(avg_rates_time_seq_pooledE, cutlow=3000, cuthigh=3500,ifExcitatory=True, alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledNoveltyResponsesE")
        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledNoveltyResponsesI")

        # get ranked sustained repsonses
        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = True, xticks = [0,5000,10000,15000,20000],figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedNoveltyResponsesE")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedNoveltyResponsesI")

         # novelty unaveraged over blocks
        avg_rates_time_seqE = np.mean(all_bincounts_conv[:,:,:Ne,startidxN:endidxN],axis = 3) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(all_bincounts_conv[:,:,Ne:,startidxN:endidxN],axis = 3) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks
        # plot pooled histograms
        plot_histogram_cut(avg_rates_time_seq_pooledE, cutlow=3000, cuthigh=3500,ifExcitatory=True,bins = np.linspace(0,300,51), alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledNoveltyResponsesENotBlockAvg")
        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledNoveltyResponsesINotBlockAvg")

        # get ranked sustained repsonses
        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedNoveltyResponsesENotBlockAvg")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedNoveltyResponsesINotBlockAvg")

    ifpooledresponsesmax = False
    if ifpooledresponsesmax:

        # replicate figure 6D sustained activity vs. cell rank
        starttime = 10 # sec
        endtime = 15.5 # sec
        startidx = np.argmax(timevector>starttime)
        endidx = np.argmax(timevector>endtime)
        starttimeT = 0.5 # sec
        endtimeT = 3.5 # sec
        startidxT = np.argmax(timevector>starttimeT)
        endidxT = np.argmax(timevector>endtimeT)
        starttimeN = 16.7 # sec
        endtimeN = 17.2 # sec
        startidxN = np.argmax(timevector>starttimeN)
        endidxN = np.argmax(timevector>endtimeN)
        print(startidxN)
        print(endidxN)
        endidxN = min(endidxN, np.size(all_bincounts_conv,axis = 3))
        #get the average firing rates for each sequence individually
        avg_rates_time_seqE = np.mean(np.amax(all_bincounts_conv[:,:,:Ne,startidx:endidx],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(np.amax(all_bincounts_conv[:,:,Ne:,startidx:endidx],axis = 3),axis=1) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks
        # plot pooled histograms
        plot_histogram(avg_rates_time_seq_pooledE, ifExcitatory=True, alpha=1,bins = np.linspace(0,150,51),figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledSustainedResponsesEMaximum")
        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1,bins = np.linspace(0,60,51),figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledSustainedResponsesIMaximum")

        # get ranked sustained repsonses
        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = False, xticks = [0,5000,10000,15000,20000],figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedStustainedResponsesEMaximum")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedStustainedResponsesIMaximum")

        # get the average firing rates for each sequence individually for whole time
        avg_rates_time_seqE = np.mean(np.amax(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(np.amax(all_bincounts_conv[:,:,Ne:,idxconv:-idxconv],axis = 3),axis=1) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks

        plot_histogram(avg_rates_time_seq_pooledE, ifExcitatory=True, alpha=1, Nblocks = Nblocks,bins = np.linspace(0,150,51),figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramAllSustainedResponsesEMaximum")

        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1,bins = np.linspace(0,50,51),figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramAllSustainedResponsesIMaximum")

        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = True, xticks = [0,5000,10000,15000,20000],figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedAllResponsesEMaximum")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedAllResponsesIMaximum")

        # Transient firing rates

        # get the average firing rates for each sequence individually
        avg_rates_time_seqE = np.mean(np.amax(all_bincounts_conv[:,:,:Ne,startidxT:endidxT],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(np.amax(all_bincounts_conv[:,:,Ne:,startidxT:endidxT],axis = 3),axis=1) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks
        # plot pooled histograms
        plot_histogram_cut(avg_rates_time_seq_pooledE, cutlow = 1000, cuthigh = 1200,ifExcitatory=True, alpha=1,bins = np.linspace(0,180,51),figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledTransientResponsesEMaximum")
        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1, bins = np.linspace(0,70,51),figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledTransientResponsesIMaximum")

        # get ranked sustained repsonses
        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = True, xticks = [0,5000,10000,15000,20000],figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedTransientResponsesEMaximum")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedTransientResponsesIMaximum")

        # Novelty firing rates
        # get the average firing rates for each sequence individually
        avg_rates_time_seqE = np.mean(np.amax(all_bincounts_conv[:,:,:Ne,startidxN:endidxN],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(np.amax(all_bincounts_conv[:,:,Ne:,startidxN:endidxN],axis = 3),axis=1) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks
        # plot pooled histograms
        plot_histogram_cut(avg_rates_time_seq_pooledE, cutlow=1000, cuthigh=1500,ifExcitatory=True, alpha=1, bins = np.linspace(0,150,51),figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledNoveltyResponsesEMaximum")
        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1, bins = np.linspace(0,70,51), figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledNoveltyResponsesIMaximum")

        # get ranked sustained repsonses
        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = True, xticks = [0,5000,10000,15000,20000],figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedNoveltyResponsesEMaximum")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedNoveltyResponsesIMaximum")

        # Novelty firing rates maximum not averaged across blocks
        # get the average firing rates for each sequence individually
        avg_rates_time_seqE = np.amax(all_bincounts_conv[:,:,:Ne,startidxN:endidxN],axis = 3)# first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.amax(all_bincounts_conv[:,:,Ne:,startidxN:endidxN],axis = 3) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks
        # plot pooled histograms
        plot_histogram_cut(avg_rates_time_seq_pooledE, cutlow=1000, cuthigh=510,ifExcitatory=True, alpha=1, bins = np.linspace(0,400,401),figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledNoveltyResponsesEMaximumNotAvgOverBlocks")
        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1, bins = np.linspace(0,70,51), figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledNoveltyResponsesIMaximumNotAvgOverBlocks")

        # get ranked sustained repsonses
        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = False ,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedNoveltyResponsesEMaximumNotAvgOverBlocks")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedNoveltyResponsesIMaximumNotAvgOverBlocks")

    ifassemblyhist = False
    if ifassemblyhist:
        # make histogram of peak firing rates of one assembly being stimulated when sequence is playing
        # average firing rates over blocks then get average and peak firnig rates plot one histo for each assembly in sequence

        #seq 1 ass 0 - ass 3
        # seq 2 ass 4 -ass 7

        avg_rates_time_seqE = np.mean(np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis = 1),axis=2) # first avg over blocks then over time (specified window)
        max_rates_time_seqE = np.amax(np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis = 1),axis=2) # first avg over blocks then take maximum over time (specified window)


        #plot_histogram(avg_rates_time_seqE[0,assemblymembers[0,assemblymembers[0,:]>0]], ifExcitatory=False,color = color[0],bins = np.linspace(0,70,51),alpha = 1)
        #save_fig(figure_directory, "HistogramAllSustainedResponsesE")
        figsize=(10,8)
        images = [range(0,Nimg)]
        for seq in [1,3]:#range(1,Nseq+1):
            fig = plt.figure(figsize=figsize)
            for ass in range((seq-1)*Nimg,seq*Nimg):
                plot_histogram_mult(fig, max_rates_time_seqE[seq-1,assemblymembers[ass,assemblymembers[ass,:]>0]-1],ifExcitatory=False,iflegend = True, legend = "assembly "+str(ass+1), color = color[ass],bins = np.linspace(0,300,51),alpha = 0.8, Nblocks = Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=6)
            save_fig(figure_directory, "HistogramAssemblyMaxRatesSeq%d"%seq)

        #avg_rates_time_seqE = np.mean(np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        for seq in [1,3]:#range(1,Nseq+1):
            fig = plt.figure(figsize=figsize)
            for ass in range((seq-1)*Nimg,seq*Nimg):
                plot_histogram_mult(fig, avg_rates_time_seqE[seq-1,assemblymembers[ass,assemblymembers[ass,:]>0]-1],ifExcitatory=False,iflegend = True, legend = "assembly "+str(ass+1), color = color[ass],bins = np.linspace(0,60,51),alpha = 0.8, Nblocks = Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=6)
            save_fig(figure_directory, "HistogramAssemblyMeanRatesSeq%d"%seq)

    ifassemblyhistblock = True
    if ifassemblyhistblock:
        # make histogram of peak firing rates of one assembly being stimulated when sequence is playing
        # average firing rates over blocks then get average and peak firnig rates plot one histo for each assembly in sequence

        #seq 1 ass 0 - ass 3
        # seq 2 ass 4 -ass 7

        avg_rates_time_seqE = np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis=3) # first avg over blocks then over time (specified window)
        max_rates_time_seqE = np.amax(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis=3) # first avg over blocks then take maximum over time (specified window)
        # avg_rates_time_seqE = np.mean(np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis = 1),axis=2) # first avg over blocks then over time (specified window)
        # max_rates_time_seqE = np.amax(np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis = 1),axis=2) # first avg over blocks then take maximum over time (specified window)


        #plot_histogram(avg_rates_time_seqE[0,assemblymembers[0,assemblymembers[0,:]>0]], ifExcitatory=False,color = color[0],bins = np.linspace(0,70,51),alpha = 1)
        #save_fig(figure_directory, "HistogramAllSustainedResponsesE")
        images = [range(0,Nimg)]
        figsize=(10,8)
        for seq in [1,3]:#Nseq+1):
            # fig = plt.figure(figsize=(15,12))

            for bl in [0,5,9]:#[0,10,19]:#range(Nblocks):#[0,1,2,3,4,5,6,7,8,9,10,19]:#range(0,2):#(Nblocks)
                fig = plt.figure(figsize=figsize)
                for ass in range((seq-1)*Nimg,seq*Nimg):
                    plot_histogram_mult(fig, max_rates_time_seqE[seq-1,bl,assemblymembers[ass,assemblymembers[ass,:]>0]-1],ifExcitatory=False,iflegend = True, legend = "assembly "+str(ass+1), color = color[ass],bins = np.linspace(0,300,51),alpha = 0.8, Nblocks = Nblocks)
                plt.locator_params(axis='x', nbins=4)
                plt.locator_params(axis='y', nbins=6)
                save_fig(figure_directory, "HistogramAssemblyMaxRatesSeq%dBlock%d"%(seq,bl))
                fig = plt.figure(figsize=figsize)
                for ass in range((seq)*Nimg,(seq+1)*Nimg):
                    plot_histogram_mult(fig, max_rates_time_seqE[seq-1,bl,assemblymembers[ass,assemblymembers[ass,:]>0]-1],ifExcitatory=False,iflegend = True, legend = "assembly "+str(ass+1), color = color[ass],bins = np.linspace(0,300,51),alpha = 0.8, Nblocks = Nblocks)
                plt.locator_params(axis='x', nbins=4)
                plt.locator_params(axis='y', nbins=6)
                save_fig(figure_directory, "HistogramAssemblyMaxRatesNextSetofAssembliesSeq%dBlock%d"%(seq,bl))
        for seq in [1,3]:#Nseq+1):
            for bl in [0,5,9]:#[0,10,19]:#range(Nblocks):
                fig = plt.figure(figsize=figsize)
                for ass in range((seq-1)*Nimg,seq*Nimg):
                    plot_histogram_mult(fig, avg_rates_time_seqE[seq-1,bl,assemblymembers[ass,assemblymembers[ass,:]>0]-1],ifExcitatory=False,iflegend = True, legend = "assembly "+str(ass+1), color = color[ass],bins = np.linspace(0,60,51),alpha = 0.8, Nblocks = Nblocks)
                plt.locator_params(axis='x', nbins=4)
                plt.locator_params(axis='y', nbins=6)
                save_fig(figure_directory, "HistogramAssemblyMeanRatesSeq%dBlock%d"%(seq,bl))
                fig = plt.figure(figsize=figsize)
                for ass in range((seq)*Nimg,(seq+1)*Nimg):
                    plot_histogram_mult(fig, avg_rates_time_seqE[seq-1,bl,assemblymembers[ass,assemblymembers[ass,:]>0]-1],ifExcitatory=False,iflegend = True, legend = "assembly "+str(ass+1), color = color[ass],bins = np.linspace(0,60,51),alpha = 0.8, Nblocks = Nblocks)
                plt.locator_params(axis='x', nbins=4)
                plt.locator_params(axis='y', nbins=6)
                save_fig(figure_directory, "HistogramAssemblyMeanRatesNextSetofAssembliesSeq%dBlock%d"%(seq,bl))

        for seq in [1,3]:##Nseq+1):
            for bl in [0,5,9]:#[0,10,19]:#range(Nblocks):
                fig = plt.figure(figsize=figsize)
                #fig = plt.figure(figsize=(15,12))
                for ass in range(1,Nseq*Nimg+1):
                    plot_histogram_mult(fig, avg_rates_time_seqE[seq-1,bl,assemblymembers[ass,assemblymembers[ass,:]>0]-1],ifExcitatory=False,iflegend = True, legend = "assembly "+str(ass), color = color[ass-1],bins = np.linspace(0,60,51),alpha = 0.8, Nblocks = Nblocks)
                plt.locator_params(axis='x', nbins=4)
                plt.locator_params(axis='y', nbins=6)
                save_fig(figure_directory, "HistogramAssemblyMeanRates_AllAssembliesSeq%dBlock%d"%(seq,bl))
        # #avg_rates_time_seqE = np.mean(np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        # for seq in range(1,Nseq+1):
        #     fig = plt.figure(figsize=(15,12))
        #     for ass in range((seq-1)*Nimg,seq*Nimg):
        #         plot_histogram_mult(fig, avg_rates_time_seqE[seq-1,assemblymembers[ass,assemblymembers[ass,:]>0]-1],ifExcitatory=False,iflegend = True, legend = "assembly "+str(ass+1), color = color[ass],bins = np.linspace(0,100,51),alpha = 0.8, Nblocks = Nblocks)
        #     save_fig(figure_directory, "HistogramAssemblyMeanRatesSeq%d"%seq)

    ifsingleneurontraces = False
    if ifsingleneurontraces:
        # plot traces of individual neurons

        avg_rates_blocks_seqE = np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis=1) # first avg over time (specified window) then over blocks
        avg_rates_blocks_seqI = np.mean(all_bincounts_conv[:,:,Ne:,idxconv:-idxconv],axis=1) # first avg over time (specified window) then over blocks


        plot_array(timevector[idxconv:-idxconv],avg_rates_blocks_seqE[0,0,:],xlabel="time [s]",figsize=(5, 3))
        plt.xlim(0,5)
        plot_array(timevector[idxconv:-idxconv],avg_rates_blocks_seqE[0,31,:],xlabel="time [s]",figsize=(5, 3))
        plt.xlim(0,5)
        plot_array(timevector[idxconv:-idxconv],avg_rates_blocks_seqE[0,30,:],xlabel="time [s]",figsize=(5, 3))
        plt.xlim(0,5)

        for seq in range(1,Nseq+1):
            for ass in range((seq-1)*Nimg,seq*Nimg):
                fig = plt.figure(figsize=figsize)
                count = 0
                for cc in assemblymembers[ass,assemblymembers[ass,:]>0]:
                    if count <= 20:
                        count = count + 1
                        plot_array_mult(fig,timevector[idxconv:-idxconv],avg_rates_blocks_seqE[seq-1,cc-1,:],xlabel="time [s]",figsize=(5, 3), ncol = 4,legend = str(cc),iflegend=True, lw = 2)
                save_fig(figure_directory, "IndividualRateTracesSeq%dAss%d"%(seq,ass))
            fig = plt.figure(figsize=figsize)
            count = 0
            for cc in noveltyidx:
                if count <= 20:
                    count = count + 1
                    plot_array_mult(fig,timevector[idxconv:-idxconv],avg_rates_blocks_seqE[seq-1,cc-1,:],xlabel="time [s]",figsize=(5, 3), ncol = 4,legend = str(cc),iflegend=True, lw = 2)
            save_fig(figure_directory, "IndividualRateTracesNoveltySeq%d"%seq)
            fig = plt.figure(figsize=figsize)
            count = 0
            for cc in np.where(untargetedidx)[0]:
                if count <= 3:
                    count = count + 1
                    plot_array_mult(fig,timevector[idxconv:-idxconv],avg_rates_blocks_seqE[seq-1,cc,:],xlabel="time [s]",figsize=(5, 3), ncol = 4,legend = str(cc),iflegend=True, lw = 2)
            save_fig(figure_directory, "IndividualRateTracesUntergetedSeq%d"%seq)


    pattern = ["timevector*","avg_rates*","*bincounts_conv*", "zscore*"]
    antipattern = ["*hist*","edges"] # specify lists with different length -> different treatment

    # create results file
    ifsaveresults = False
    if ifsaveresults:
        file_name_results = results_folder + file_name + "/singleneuronresults%s.h5"%timestr
        f_results = h5py.File(file_name_results, "a")


        f_results.create_dataset('avgwindow%d'%avgwindow, data=avgwindow)
        #f_results.create_dataset('Avgwindow%d/Nreps'%avgwindow, data=Nreps)

        for key in dir():
            if fnmatch.fnmatch(key, pattern[0]):
                if not fnmatch.fnmatch(key, antipattern[0]):
                    f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
                else:
                    listlen = len(vars()[key])
                    for i in range(0,listlen):
                        f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])
            elif fnmatch.fnmatch(key, pattern[1]):
                f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
            elif fnmatch.fnmatch(key, pattern[2]):
                f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
            elif fnmatch.fnmatch(key, pattern[3]):
                f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
            elif fnmatch.fnmatch(key, antipattern[1]):
                listlen = len(vars()[key])
                for i in range(0,listlen):
                    f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])

        f_results.close()
    gc.collect()
#################################################################
#                                                               #
#                                                               #
#                run_whole_analysis                             #
#                                                               #
#                                                               #
#################################################################
#specify file to be analysed
#specify file to be analysed
def run_whole_analysis(file_name, avgwindow = 8, timestr = "_now", RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR
    # folder with stored data from the run
    #run_folder = "/gpfs/gjor/personal/schulza/data/main/sequences/"
    # folder with analysed results from spiketime analysis in julia & where to results are stored
    #results_folder = "/home/schulza/Documents/results/main/sequences/"
    #results_folder = "/gpfs/gjor/personal/schulza/results/sequences/"
    #run_folder = "../data/"
    #results_folder = "../results/"
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

    idxconv = np.floor_divide(avgwindow,2)+1
    plot_all_averages(edges, mean_hist_E, Seqs, savehandle = "E", ifseqlen=True, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=True,legendhandle = "Seq. ", ifyticks=False)
    plot_all_averages(edges, mean_hist_I, Seqs, savehandle = "I", ifseqlen=True, figure_directory = figure_directory, color = colorI, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    plot_all_averages(edges, mean_hist_E_boxcar, Seqs, savehandle = "E_boxcar", ifseqlen=True,startidx = idxconv, endidx = -idxconv, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=False, legendhandle = "Seq. : ", ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomem, Seqs, savehandle = "E_nomem", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomemnonov, Seqs, savehandle = "E_nomemnonov", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nov, Seqs, savehandle = "E_nov", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)

    plot_all_traces_and_average(edges, hist_E, mean_hist_E, Seqs, savehandle = "E", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_I, mean_hist_I, Seqs, savehandle = "I", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_boxcar, mean_hist_E_boxcar, Seqs, ifseqlen=True, savehandle = "E_boxcar_avg%d" % int(avgwindow) , startidx = idxconv, endidx = -idxconv, Nblocks = Nblocks, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomem, mean_hist_E_nomem, Seqs, ifseqlen=True, savehandle = "E_nomem", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomemnonov, mean_hist_E_nomemnonov, Seqs, ifseqlen=True, savehandle = "E_nomemnonov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nov, mean_hist_E_nov, Seqs, ifseqlen=True, savehandle = "E_nov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)

    plot_all_traces_and_average_E_I(edges, hist_E, hist_I, Seqs, savehandle = "E_Itest", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=True, ifyticks=False)
    plot_all_traces_and_average_E_I(edges, hist_E_boxcar, hist_I_boxcar, Seqs, savehandle = "E_Itestboxcar", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, startidx = idxconv, endidx = -idxconv, color = colorE, ifoffset=False, iflegend=True, ifyticks=False)

    nsamples = len(edges[0])
    sample_fraction = 0.8
    # change sample_fraction for Nreps smaller than 5
    if Nreps <= 5:
        sample_fraction = 0.6
    # get the last sample to be considered in fit (discarded novelty response)
    lastsample = int(round(sample_fraction*nsamples))

    # plot_all_averages(edges, mean_hist_E, Seqs, savehandle = "E_cutoff", ifseqlen=True, endidx = lastsample, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=True,legendhandle = "Seq. ", ifyticks=False)
    # plot_all_averages(edges, mean_hist_I, Seqs, savehandle = "I_cutoff", ifseqlen=True, endidx = lastsample, figure_directory = figure_directory, color = colorI, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    # plot_all_averages(edges, mean_hist_E_boxcar, Seqs, savehandle = "E_boxcar_cutoff", ifseqlen=True, endidx = lastsample,startidx = idxconv, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, legendhandle = "Seq. : ", ifyticks=False)
    # plot_all_averages(edges, mean_hist_E_nomem, Seqs, savehandle = "E_nomem_cutoff", endidx = lastsample, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    # plot_all_averages(edges, mean_hist_E_nomemnonov, Seqs, savehandle = "E_nomemnonov_cutoff", endidx = lastsample, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    #
    # plot_all_traces_and_average(edges, hist_E, mean_hist_E, Seqs, savehandle = "E_cutoff", endidx = lastsample, Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=False, ifyticks=False)
    # plot_all_traces_and_average(edges, hist_I, mean_hist_I, Seqs, savehandle = "I_cutoff", endidx = lastsample, Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = colorI, ifoffset=False, iflegend=False, ifyticks=False)
    # plot_all_traces_and_average(edges, hist_E_boxcar, mean_hist_E_boxcar, Seqs, ifseqlen=True, endidx = lastsample, savehandle = "E_boxcar_avg%d_cutoff" % int(avgwindow) , startidx = idxconv, Nblocks = Nblocks, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=False, ifyticks=False)
    # plot_all_traces_and_average(edges, hist_E_nomem, mean_hist_E_nomem, Seqs, ifseqlen=True, endidx = lastsample, savehandle = "E_nomem_cutoff", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    # plot_all_traces_and_average(edges, hist_E_nomemnonov, mean_hist_E_nomemnonov, Seqs, ifseqlen=True, endidx = lastsample, savehandle = "E_nomemnonov_cutoff", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    #

    # ------------------------------------ FITTING ---------------------------------------------
    # """fit_variable_repetitions_gen_arrays(args):
    #     perform fitting of all traces included in datalist and meandatalist
    #         determine the baseline firing rate prior to the novelty stimulation

    # set initial parameters for fitting of the exponential curve
    # fit a * exp(-t/tau) + a_0
    initial_params = [2, 20, 3]
    #                [a, tau,a_0]
    fit_bounds = (0, [10., 140., 10])
    avgindices = 30
    startimg = Nimg # after which image should fit start at block onset update for Seqlen in function always last img
    # fitting of initial transient
    t_before_nov, params_blockavg, params_covariance_blockavg, params_err_blockavg, params, params_covariance, params_err = fit_gen_arrays_startidx(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Seqs, Nblocks,
        ifseqlen=False, avgindices = avgindices, Nseq = Nseq, initialparams=initial_params, bounds=fit_bounds, ifplot = False,
        startimg = startimg, idxconv = idxconv)

    #get_baseline_firing_rate
    baseline_avg, baseline, mean_baseline, std_baseline = get_baseline_firing_rate(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        ifseqlen=False, ifrepseq = True, Nseq = Nseq,  avgindices = avgindices, idxconv = idxconv)

    gc.collect()

    tau_transientpre, tau_transientpre_err = convert_tau(params,params_err)
#     tau_transientpost, tau_transientpost_err = convert_tau(params_trans, params_err_trans)
    tau_transientpre_avg, tau_transientpre_err_avg = convert_tau_avg(params_blockavg, params_err_blockavg)
#     tau_transientpost_avg, tau_transientpost_err_avg = convert_tau_avg(params_blockavg,params_err_blockavg)

    # ----------------------------------------- get peaks -----------------------------------------

    samples_img = int(round(lenstim/binsize))
    height_novelty_avg, height_novelty, mean_novelty, std_novelty, novelty_avgidx, noveltyidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = False, iftransientpost = False, ifseqlen=False, ifrepseq = True, Nseq = Nseq,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = 8*samples_img)

    height_trans_pre_avg, height_trans_pre, mean_trans_pre, std_trans_pre, trans_pre_avgidx, trans_preidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = True, iftransientpost = False, ifseqlen=False, ifrepseq = True, Nseq = Nseq,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = 8*samples_img)
# -------------------------------- inhibitory ----------------------------------------------------------
        #get_baseline_firing_rate
    baseline_avgI, baselineI, mean_baselineI, std_baselineI = get_baseline_firing_rate(
        edges, hist_I_boxcar, mean_hist_I_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        ifseqlen=False, ifrepseq = True, Nseq = Nseq,  avgindices = avgindices, idxconv = idxconv)

    gc.collect()


    # ----------------------------------------- get peaks -----------------------------------------

    samples_img = int(round(lenstim/binsize))
    height_novelty_avgI, height_noveltyI, mean_noveltyI, std_noveltyI, novelty_avgidxI, noveltyidxI = get_peak_height(
        edges, hist_I_boxcar, mean_hist_I_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = False, iftransientpost = False, ifseqlen=False, ifrepseq = True, Nseq = Nseq,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = 8*samples_img)

    height_trans_pre_avgI, height_trans_preI, mean_trans_preI, std_trans_preI, trans_pre_avgidxI, trans_preidxI = get_peak_height(
        edges, hist_I_boxcar, mean_hist_I_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = True, iftransientpost = False, ifseqlen=False, ifrepseq = True, Nseq = Nseq,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = 8*samples_img)


    # ----------------------------------------- plotting --------------------------------------------
    plot_Nreps_tau(Seqs, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xlabel="sequence number", xtickstepsize = 1, savename = "NimgTau")
    plot_Nreps_baseline(Seqs, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xlabel="sequence number", xtickstepsize = 1,savename = "NimgBaseline")



    # plot unsubtracted data transients, novelty and baseline
    plot_Nreps_array(Seqs, height_trans_pre, height_trans_pre_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient peak rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre_grey_dots", xtickstepsize = 1)
    plot_Nreps_array(Seqs, baseline, baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_BL_grey_dots", xtickstepsize = 1)
    plot_Nreps_array(Seqs, height_novelty, height_novelty_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty peak rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty_grey_dots", xtickstepsize = 1)
    #plot_Nreps_array(Nimg, height_trans_post, height_trans_post_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient peak rate [Hz]", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre_grey_dots")

    # plot data transients, novelty subtracted baseline
    #plot_Nreps_array(Nimg, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPost-BL_grey_dots", xtickstepsize = 1)
    plot_Nreps_array(Seqs, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty-BL_grey_dots", xtickstepsize = 1)
    plot_Nreps_array(Seqs, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre-BL_grey_dots", xtickstepsize = 1)

    # plot data transients, novelty subtracted baseline with errorbars
    #plot_Nreps_array_errorbar(Nimg, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPost-BL_grey_errorbar", xtickstepsize = 1)
    plot_Nreps_array_errorbar(Seqs, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty-BL_grey_errorbar", xtickstepsize = 1)
    plot_Nreps_array_errorbar(Seqs, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]",xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre-BL_grey_errorbar", xtickstepsize = 1)

    # plot data transients, novelty subtracted baseline with errorbands
    #plot_Nreps_array_errorband(Nimg, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPost-BL_grey_errorband", xtickstepsize = 1)
    plot_Nreps_array_errorband(Seqs, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty-BL_grey_errorband", xtickstepsize = 1)
    plot_Nreps_array_errorband(Seqs, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre-BL_grey_errorband", xtickstepsize = 1)

    # overlap with previous sequence
    #plot_Nreps_array_barplot(Seqs, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty-BL_grey_errorbar", xtickstepsize = 1)
    barplot_Nreps_array(Seqs, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty-BL_grey_barplot", xtickstepsize = 1,alpha = 1)
    barplot_Nreps_array(Seqs, height_novelty, height_novelty_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty_barplot", xtickstepsize = 1,alpha = 1)

    plot_peak_overlap_seq(novelty_overlap, height_novelty-baseline, Nseq = Nseq, iflegend = False, colorseq = color ,figsize=(15,12),
                     lw = 3, xlabel = "overlap with previous sequence [%]", ylabel ="novelty peak rate [Hz]",
                          fontsize = 24,ifioff = False, ifsavefig = True, Nimg = Nimg, savehandle = "NoveltyPeak", figure_directory = figure_directory, ifyticks = False, yticks = [3,4])

    plot_peak_overlap_seq(novelty_overlap, height_trans_pre-baseline, Nseq = Nseq, iflegend = False, colorseq = color ,figsize=(15,12),
                     lw = 3, xlabel = "overlap with previous sequence [%]", ylabel ="transient peak rate [Hz]",
                          fontsize = 24, ifioff = False, ifsavefig = True, Nimg = Nimg, savehandle = "TransientPeak", figure_directory = figure_directory,ifyticks = False, yticks = [3,4])

    # ----------------------------------------------------------------- bar plot ------------------------------------
    barplot_peak_comparison_EI(height_novelty, height_noveltyI, height_trans_pre, height_trans_preI, baseline, baselineI, iflegend=True, figure_directory=figure_directory, ifsavefig = True, xlabel=" ", alpha = 1)
    barplot_peak_comparison_EI(height_novelty_avg, height_novelty_avgI, height_trans_pre_avg, height_trans_pre_avgI, baseline_avg, baseline_avgI, iflegend=True, figure_directory=figure_directory, ifsavefig = True, xlabel=" ", alpha = 1, savename = "ComparisonPeakHeightsBarPlot_averages")


    # ---------------------------------- store variables -----------------------------------------
    pattern = ["mean*","params*","height*", "tau*", "baseline*"]
    antipattern = ["*hist*","edges"] # specify lists with different length -> different treatment

    # create results file
    file_name_results = results_folder + file_name + "/results%s.h5"%timestr
    f_results = h5py.File(file_name_results, "a")


    f_results.create_dataset('avgwindow%d'%avgwindow, data=avgwindow)
    f_results.create_dataset('Avgwindow%d/Nreps'%avgwindow, data=Nreps)

    for key in dir():
        if fnmatch.fnmatch(key, pattern[0]):
            if not fnmatch.fnmatch(key, antipattern[0]):
                f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
            else:
                listlen = len(vars()[key])
                for i in range(0,listlen):
                    f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])
        elif fnmatch.fnmatch(key, pattern[1]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[2]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[3]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[4]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, antipattern[1]):
            listlen = len(vars()[key])
            for i in range(0,listlen):
                f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])

    f_results.close()
    return mean_hist_E,mean_hist_I

#################################################################
#                                                               #
#                                                               #
#                run_whole_analysis                             #
#                                                               #
#                                                               #
#################################################################
#specify file to be analysed
#specify file to be analysed
def run_whole_analysis_member(file_name, avgwindow = 8, timestr = "_now", RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR

    #"/gpfs/gjor/personal/schulza/data/main/sequences/"
    # folder with analysed results from spiketime analysis in julia & where to results are stored
    #results_folder = "/home/schulza/Documents/results/main/sequences/"
    #results_folder = "/gpfs/gjor/personal/schulza/results/sequences/"
    #run_folder = "../data/"
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
    #Nblocks = min([Nblocks, 10])
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
    hist_E_mem  = []
    mean_hist_E_mem = []
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

    idxconv = np.floor_divide(avgwindow,2)+1
    plot_all_averages(edges, mean_hist_E, Seqs, savehandle = "E", ifseqlen=True, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=True,legendhandle = "Seq. ", ifyticks=False)
    plot_all_averages(edges, mean_hist_I, Seqs, savehandle = "I", ifseqlen=True, figure_directory = figure_directory, color = colorI, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    plot_all_averages(edges, mean_hist_E_boxcar, Seqs, savehandle = "E_boxcar", ifseqlen=True,startidx = idxconv, endidx = -idxconv, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=False, legendhandle = "Seq. : ", ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomem, Seqs, savehandle = "E_nomem", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomemnonov, Seqs, savehandle = "E_nomemnonov", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nov, Seqs, savehandle = "E_nov", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)

    plot_all_traces_and_average(edges, hist_E, mean_hist_E, Seqs, savehandle = "E", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_I, mean_hist_I, Seqs, savehandle = "I", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_boxcar, mean_hist_E_boxcar, Seqs, ifseqlen=True, savehandle = "E_boxcar_avg%d" % int(avgwindow) , startidx = idxconv, endidx = -idxconv, Nblocks = Nblocks, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomem, mean_hist_E_nomem, Seqs, ifseqlen=True, savehandle = "E_nomem", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomemnonov, mean_hist_E_nomemnonov, Seqs, ifseqlen=True, savehandle = "E_nomemnonov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nov, mean_hist_E_nov, Seqs, ifseqlen=True, savehandle = "E_nov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)

    plot_all_traces_and_average_E_I(edges, hist_E, hist_I, Seqs, savehandle = "E_Itest", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=True, ifyticks=False)
    plot_all_traces_and_average_E_I(edges, hist_E_boxcar, hist_I_boxcar, Seqs, savehandle = "E_Itestboxcar", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, startidx = idxconv, endidx = -idxconv, color = colorE, ifoffset=False, iflegend=True, ifyticks=False)

    nsamples = len(edges[0])
    sample_fraction = 0.8
    # change sample_fraction for Nreps smaller than 5
    if Nreps <= 5:
        sample_fraction = 0.6
    # get the last sample to be considered in fit (discarded novelty response)
    lastsample = int(round(sample_fraction*nsamples))

    # plot_all_averages(edges, mean_hist_E, Seqs, savehandle = "E_cutoff", ifseqlen=True, endidx = lastsample, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=True,legendhandle = "Seq. ", ifyticks=False)
    # plot_all_averages(edges, mean_hist_I, Seqs, savehandle = "I_cutoff", ifseqlen=True, endidx = lastsample, figure_directory = figure_directory, color = colorI, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    # plot_all_averages(edges, mean_hist_E_boxcar, Seqs, savehandle = "E_boxcar_cutoff", ifseqlen=True, endidx = lastsample,startidx = idxconv, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, legendhandle = "Seq. : ", ifyticks=False)
    # plot_all_averages(edges, mean_hist_E_nomem, Seqs, savehandle = "E_nomem_cutoff", endidx = lastsample, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    # plot_all_averages(edges, mean_hist_E_nomemnonov, Seqs, savehandle = "E_nomemnonov_cutoff", endidx = lastsample, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    #
    # plot_all_traces_and_average(edges, hist_E, mean_hist_E, Seqs, savehandle = "E_cutoff", endidx = lastsample, Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=False, ifyticks=False)
    # plot_all_traces_and_average(edges, hist_I, mean_hist_I, Seqs, savehandle = "I_cutoff", endidx = lastsample, Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = colorI, ifoffset=False, iflegend=False, ifyticks=False)
    # plot_all_traces_and_average(edges, hist_E_boxcar, mean_hist_E_boxcar, Seqs, ifseqlen=True, endidx = lastsample, savehandle = "E_boxcar_avg%d_cutoff" % int(avgwindow) , startidx = idxconv, Nblocks = Nblocks, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=False, ifyticks=False)
    # plot_all_traces_and_average(edges, hist_E_nomem, mean_hist_E_nomem, Seqs, ifseqlen=True, endidx = lastsample, savehandle = "E_nomem_cutoff", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    # plot_all_traces_and_average(edges, hist_E_nomemnonov, mean_hist_E_nomemnonov, Seqs, ifseqlen=True, endidx = lastsample, savehandle = "E_nomemnonov_cutoff", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    #

    # ------------------------------------ FITTING ---------------------------------------------
    # """fit_variable_repetitions_gen_arrays(args):
    #     perform fitting of all traces included in datalist and meandatalist
    #         determine the baseline firing rate prior to the novelty stimulation

    # set initial parameters for fitting of the exponential curve
    # fit a * exp(-t/tau) + a_0
    initial_params = [2, 20, 3]
    #                [a, tau,a_0]
    fit_bounds = (0, [10., 140., 10])
    avgindices = 30
    startimg = Nimg # after which image should fit start at block onset update for Seqlen in function always last img
    # fitting of initial transient
    t_before_nov, params_blockavg, params_covariance_blockavg, params_err_blockavg, params, params_covariance, params_err = fit_gen_arrays_startidx(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Seqs, Nblocks,
        ifseqlen=False, avgindices = avgindices, Nseq = Nseq, initialparams=initial_params, bounds=fit_bounds, ifplot = False,
        startimg = startimg, idxconv = idxconv)

    #get_baseline_firing_rate
    baseline_avg, baseline, mean_baseline, std_baseline = get_baseline_firing_rate(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        ifseqlen=False, ifrepseq = True, Nseq = Nseq,  avgindices = avgindices, idxconv = idxconv)

    gc.collect()

    tau_transientpre, tau_transientpre_err = convert_tau(params,params_err)
#     tau_transientpost, tau_transientpost_err = convert_tau(params_trans, params_err_trans)
    tau_transientpre_avg, tau_transientpre_err_avg = convert_tau_avg(params_blockavg, params_err_blockavg)
#     tau_transientpost_avg, tau_transientpost_err_avg = convert_tau_avg(params_blockavg,params_err_blockavg)

    # ----------------------------------------- get peaks -----------------------------------------

    samples_img = int(round(lenstim/binsize))
    height_novelty_avg, height_novelty, mean_novelty, std_novelty, novelty_avgidx, noveltyidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = False, iftransientpost = False, ifseqlen=False, ifrepseq = True, Nseq = Nseq,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = 8*samples_img)

    height_trans_pre_avg, height_trans_pre, mean_trans_pre, std_trans_pre, trans_pre_avgidx, trans_preidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = True, iftransientpost = False, ifseqlen=False, ifrepseq = True, Nseq = Nseq,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = 8*samples_img)
# -------------------------------- inhibitory ----------------------------------------------------------
        #get_baseline_firing_rate
    baseline_avgI, baselineI, mean_baselineI, std_baselineI = get_baseline_firing_rate(
        edges, hist_I_boxcar, mean_hist_I_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        ifseqlen=False, ifrepseq = True, Nseq = Nseq,  avgindices = avgindices, idxconv = idxconv)

    gc.collect()


    # ----------------------------------------- get peaks -----------------------------------------

    samples_img = int(round(lenstim/binsize))
    height_novelty_avgI, height_noveltyI, mean_noveltyI, std_noveltyI, novelty_avgidxI, noveltyidxI = get_peak_height(
        edges, hist_I_boxcar, mean_hist_I_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = False, iftransientpost = False, ifseqlen=False, ifrepseq = True, Nseq = Nseq,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = 8*samples_img)

    height_trans_pre_avgI, height_trans_preI, mean_trans_preI, std_trans_preI, trans_pre_avgidxI, trans_preidxI = get_peak_height(
        edges, hist_I_boxcar, mean_hist_I_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = True, iftransientpost = False, ifseqlen=False, ifrepseq = True, Nseq = Nseq,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = 8*samples_img)


    # ----------------------------------------- plotting --------------------------------------------
    plot_Nreps_tau(Seqs, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xlabel="sequence number", xtickstepsize = 1, savename = "NimgTau")
    plot_Nreps_baseline(Seqs, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xlabel="sequence number", xtickstepsize = 1,savename = "NimgBaseline")



    # plot unsubtracted data transients, novelty and baseline
    plot_Nreps_array(Seqs, height_trans_pre, height_trans_pre_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient peak rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre_grey_dots", xtickstepsize = 1)
    plot_Nreps_array(Seqs, baseline, baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_BL_grey_dots", xtickstepsize = 1)
    plot_Nreps_array(Seqs, height_novelty, height_novelty_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty peak rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty_grey_dots", xtickstepsize = 1)
    #plot_Nreps_array(Nimg, height_trans_post, height_trans_post_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient peak rate [Hz]", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre_grey_dots")

    # plot data transients, novelty subtracted baseline
    #plot_Nreps_array(Nimg, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPost-BL_grey_dots", xtickstepsize = 1)
    plot_Nreps_array(Seqs, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty-BL_grey_dots", xtickstepsize = 1)
    plot_Nreps_array(Seqs, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre-BL_grey_dots", xtickstepsize = 1)

    # plot data transients, novelty subtracted baseline with errorbars
    #plot_Nreps_array_errorbar(Nimg, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPost-BL_grey_errorbar", xtickstepsize = 1)
    plot_Nreps_array_errorbar(Seqs, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty-BL_grey_errorbar", xtickstepsize = 1)
    plot_Nreps_array_errorbar(Seqs, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]",xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre-BL_grey_errorbar", xtickstepsize = 1)

    # plot data transients, novelty subtracted baseline with errorbands
    #plot_Nreps_array_errorband(Nimg, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPost-BL_grey_errorband", xtickstepsize = 1)
    plot_Nreps_array_errorband(Seqs, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty-BL_grey_errorband", xtickstepsize = 1)
    plot_Nreps_array_errorband(Seqs, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre-BL_grey_errorband", xtickstepsize = 1)

    # overlap with previous sequence
    #plot_Nreps_array_barplot(Seqs, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty-BL_grey_errorbar", xtickstepsize = 1)
    barplot_Nreps_array(Seqs, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty-BL_grey_barplot", xtickstepsize = 1,alpha = 1)
    barplot_Nreps_array(Seqs, height_novelty, height_novelty_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty_barplot", xtickstepsize = 1,alpha = 1)

    plot_peak_overlap_seq(novelty_overlap, height_novelty-baseline, Nseq = Nseq, iflegend = False, colorseq = color ,figsize=(15,12),
                     lw = 3, xlabel = "overlap with previous sequence [%]", ylabel ="novelty peak rate [Hz]",
                          fontsize = 24,ifioff = False, ifsavefig = True, Nimg = Nimg, savehandle = "NoveltyPeak", figure_directory = figure_directory, ifyticks = False, yticks = [3,4])

    plot_peak_overlap_seq(novelty_overlap, height_trans_pre-baseline, Nseq = Nseq, iflegend = False, colorseq = color ,figsize=(15,12),
                     lw = 3, xlabel = "overlap with previous sequence [%]", ylabel ="transient peak rate [Hz]",
                          fontsize = 24, ifioff = False, ifsavefig = True, Nimg = Nimg, savehandle = "TransientPeak", figure_directory = figure_directory,ifyticks = False, yticks = [3,4])

    # ----------------------------------------------------------------- bar plot ------------------------------------
    barplot_peak_comparison_EI(height_novelty, height_noveltyI, height_trans_pre, height_trans_preI, baseline, baselineI, iflegend=True, figure_directory=figure_directory, ifsavefig = True, xlabel=" ", alpha = 1)
    barplot_peak_comparison_EI(height_novelty_avg, height_novelty_avgI, height_trans_pre_avg, height_trans_pre_avgI, baseline_avg, baseline_avgI, iflegend=True, figure_directory=figure_directory, ifsavefig = True, xlabel=" ", alpha = 1, savename = "ComparisonPeakHeightsBarPlot_averages")


    # ---------------------------------- store variables -----------------------------------------
    pattern = ["mean*","params*","height*", "tau*", "baseline*"]
    antipattern = ["*hist*","edges"] # specify lists with different length -> different treatment

    # create results file
    file_name_results = results_folder + file_name + "/results%s.h5"%timestr
    f_results = h5py.File(file_name_results, "a")


    f_results.create_dataset('avgwindow%d'%avgwindow, data=avgwindow)
    f_results.create_dataset('Avgwindow%d/Nreps'%avgwindow, data=Nreps)

    for key in dir():
        if fnmatch.fnmatch(key, pattern[0]):
            if not fnmatch.fnmatch(key, antipattern[0]):
                f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
            else:
                listlen = len(vars()[key])
                for i in range(0,listlen):
                    f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])
        elif fnmatch.fnmatch(key, pattern[1]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[2]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[3]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[4]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, antipattern[1]):
            listlen = len(vars()[key])
            for i in range(0,listlen):
                f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])

    f_results.close()
    return mean_hist_E,mean_hist_I

# def run_whole_analysis(file_name, avgwindow = 8, timestr = "_now"):
#     # folder with stored data from the run
#     run_folder = "/gpfs/gjor/personal/schulza/data/main/sequences/"
#     # folder with analysed results from spiketime analysis in julia & where to results are stored
#     #results_folder = "/home/schulza/Documents/results/main/sequences/"
#     results_folder = "/gpfs/gjor/personal/schulza/results/sequences/"
#
#     # define folder where figues should be stored
#     figure_directory = results_folder + file_name + "/" + "figures/"
#     if not os.path.exists(figure_directory):
#         os.makedirs(figure_directory)
#
#     # read in run parameters
#     file_name_run = run_folder + file_name
#     # open file
#     frun = h5py.File(file_name_run, "r")
#
#     # read in stimulus parameters
#     Nimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength  = frun["initial"]["stimparams"].value
#     #Nblocks = 9
#     seqnumber  = frun["initial"]["seqnumber"].value
#
#
#     assemblymembers = frun["initial"]["assemblymembers"].value.transpose()
#     Seqs = np.arange(1,Nseq+1)
#     ifSTDP, ifwadapt = frun["params"]["STDPwadapt"].value
#     if ifwadapt == 1:
#         print("ADAPT ON")
#     else:
#         print("NON ADAPTIVE")
#     # close file
#     frun.close()
#
#     # read in population averages
#     listOfFiles = os.listdir(results_folder + file_name)
#     pattern = "spiketime*.h5"
#     sub_folder = []
#
#     for entry in listOfFiles:
#         if fnmatch.fnmatch(entry, pattern):
#                 sub_folder.append(entry)
#
#     # get name of subfolder spiketime + date.h5
#     file_name_spikes = results_folder + file_name + "/" + sub_folder[0]
#     f = h5py.File(file_name_spikes, "r")
#     # ------------------------- set binsize and boxcal sliding window --------------------------------------
#     binsize = f["params"]["binsize"].value
#     winlength = avgwindow*binsize
#
#     novelty_overlap = np.zeros((Nseq,Nblocks,3))
#     novelty_indices = np.zeros((Nseq,Nblocks,200))
#
#     hist_E  = []
#     mean_hist_E = []
#     hist_I  = []
#     mean_hist_I = []
#     hist_E_nomem  = []
#     mean_hist_E_nomem = []
#     hist_E_nomemnonov  = []
#     mean_hist_E_nomemnonov = []
#     hist_E_nov  = []
#     mean_hist_E_nov = []
#     hist_E_boxcar  = []
#     mean_hist_E_boxcar = []
#     edges = []
#     novelty_overlap = []
#     #novelty_indices = []
#
#     for seq in range(1,Nseq + 1):
#         edges.append(f["E%dmsedges" % binsize]["seq"+ str(seq) + "block"+ str(1)].value)
#         hist_E.append(np.zeros((Nblocks,len(edges[seq-1]))))
#         hist_I.append(np.zeros((Nblocks,len(edges[seq-1]))))
#         hist_E_nomem.append(np.zeros((Nblocks,len(edges[seq-1]))))
#         hist_E_nov.append(np.zeros((Nblocks,len(edges[seq-1]))))
#         hist_E_nomemnonov.append(np.zeros((Nblocks,len(edges[seq-1]))))
#         hist_E_boxcar.append(np.zeros((Nblocks,len(edges[seq-1]))))
#         novelty_overlap.append(np.zeros(Nblocks))
#
#         for bl in range(1, Nblocks + 1):
#             #vars()['hist_E_all' + str(seq-1)][bl-1][:] = f["E%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
#             hist_E[seq-1][bl-1,:] = f["E%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
#             hist_I[seq-1][bl-1,:] = f["I%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
#             hist_E_nomem[seq-1][bl-1,:] = f["ENonMem%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
#             hist_E_nov[seq-1][bl-1,:] = f["Nov%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
#             hist_E_nomemnonov[seq-1][bl-1,:] = f["ENonMemNoNov%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
#             hist_E_boxcar[seq-1][bl-1,:] = np.convolve(hist_E[seq-1][bl-1,:], np.ones((avgwindow,))/avgwindow, mode='same')
#             novelty_overlap[seq-1][bl-1] = f["noveltyoverlap"]["seq"+ str(seq) + "block"+ str(bl)].value[0]
#         # get averages over blocks
#         mean_hist_E.append(np.mean(hist_E[seq-1][:,:],axis = 0))
#         mean_hist_I.append(np.mean(hist_I[seq-1][:,:],axis = 0))
#         mean_hist_E_nomem.append(np.mean(hist_E_nomem[seq-1][:,:],axis = 0))
#         mean_hist_E_nomemnonov.append(np.mean(hist_E_nomemnonov[seq-1][:,:],axis = 0))
#         mean_hist_E_boxcar.append(np.mean(hist_E_boxcar[seq-1][:,:],axis = 0))
#         mean_hist_E_nov.append(np.mean(hist_E_nov[seq-1][:,:],axis = 0))
#
#
#     # close file
#     f.close()
#     # plotting
#     color = ["midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon"]
#
#     idxconv = np.floor_divide(avgwindow,2)+1
#     plot_all_averages(edges, mean_hist_E, Seqs, savehandle = "E", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True,legendhandle = "Seq. ", ifyticks=False)
#     plot_all_averages(edges, mean_hist_I, Seqs, savehandle = "I", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
#     plot_all_averages(edges, mean_hist_E_boxcar, Seqs, savehandle = "E_boxcar", ifseqlen=True,startidx = idxconv, endidx = -idxconv, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, legendhandle = "Seq. : ", ifyticks=False)
#     plot_all_averages(edges, mean_hist_E_nomem, Seqs, savehandle = "E_nomem", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
#     plot_all_averages(edges, mean_hist_E_nomemnonov, Seqs, savehandle = "E_nomemnonov", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
#     plot_all_averages(edges, mean_hist_E_nov, Seqs, savehandle = "E_nov", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
#
#     plot_all_traces_and_average(edges, hist_E, mean_hist_E, Seqs, savehandle = "E", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
#     plot_all_traces_and_average(edges, hist_I, mean_hist_I, Seqs, savehandle = "I", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
#     plot_all_traces_and_average(edges, hist_E_boxcar, mean_hist_E_boxcar, Seqs, ifseqlen=True, savehandle = "E_boxcar_avg%d" % int(avgwindow) , startidx = idxconv, endidx = -idxconv, Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
#     plot_all_traces_and_average(edges, hist_E_nomem, mean_hist_E_nomem, Seqs, ifseqlen=True, savehandle = "E_nomem", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
#     plot_all_traces_and_average(edges, hist_E_nomemnonov, mean_hist_E_nomemnonov, Seqs, ifseqlen=True, savehandle = "E_nomemnonov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
#     plot_all_traces_and_average(edges, hist_E_nov, mean_hist_E_nov, Seqs, ifseqlen=True, savehandle = "E_nov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
#
#     plot_all_traces_and_average_E_I(edges, hist_E, hist_I, Seqs, savehandle = "E_Itest", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=True, ifyticks=False)
#
#     nsamples = len(edges[0])
#     sample_fraction = 0.8
#     # change sample_fraction for Nreps smaller than 5
#     if Nreps <= 5:
#         sample_fraction = 0.6
#     # get the last sample to be considered in fit (discarded novelty response)
#     lastsample = int(round(sample_fraction*nsamples))
#
#     plot_all_averages(edges, mean_hist_E, Seqs, savehandle = "E_cutoff", ifseqlen=True, endidx = lastsample, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True,legendhandle = "Seq. ", ifyticks=False)
#     plot_all_averages(edges, mean_hist_I, Seqs, savehandle = "I_cutoff", ifseqlen=True, endidx = lastsample, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
#     plot_all_averages(edges, mean_hist_E_boxcar, Seqs, savehandle = "E_boxcar_cutoff", ifseqlen=True, endidx = lastsample,startidx = idxconv, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, legendhandle = "Seq. : ", ifyticks=False)
#     plot_all_averages(edges, mean_hist_E_nomem, Seqs, savehandle = "E_nomem_cutoff", endidx = lastsample, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
#     plot_all_averages(edges, mean_hist_E_nomemnonov, Seqs, savehandle = "E_nomemnonov_cutoff", endidx = lastsample, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
#
#     plot_all_traces_and_average(edges, hist_E, mean_hist_E, Seqs, savehandle = "E_cutoff", endidx = lastsample, Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
#     plot_all_traces_and_average(edges, hist_I, mean_hist_I, Seqs, savehandle = "I_cutoff", endidx = lastsample, Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
#     plot_all_traces_and_average(edges, hist_E_boxcar, mean_hist_E_boxcar, Seqs, ifseqlen=True, endidx = lastsample, savehandle = "E_boxcar_avg%d_cutoff" % int(avgwindow) , startidx = idxconv, Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
#     plot_all_traces_and_average(edges, hist_E_nomem, mean_hist_E_nomem, Seqs, ifseqlen=True, endidx = lastsample, savehandle = "E_nomem_cutoff", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
#     plot_all_traces_and_average(edges, hist_E_nomemnonov, mean_hist_E_nomemnonov, Seqs, ifseqlen=True, endidx = lastsample, savehandle = "E_nomemnonov_cutoff", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
#
#
#
#     # ------------------------------------ FITTING ---------------------------------------------
#     # """fit_variable_repetitions_gen_arrays(args):
#     #     perform fitting of all traces included in datalist and meandatalist
#     #         determine the baseline firing rate prior to the novelty stimulation
#
#     # set initial parameters for fitting of the exponential curve
#     # fit a * exp(-t/tau) + a_0
#     initial_params = [2, 20, 3]
#     #                [a, tau,a_0]
#     fit_bounds = (0, [10., 140., 10])
#     avgindices = 30
#     startimg = Nimg # after which image should fit start at block onset update for Seqlen in function always last img
#     # fitting of initial transient
#     t_before_nov, params_blockavg, params_covariance_blockavg, params_err_blockavg, params, params_covariance, params_err = fit_gen_arrays_startidx(
#         edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Seqs, Nblocks,
#         ifseqlen=False, avgindices = avgindices, Nseq = Nseq, initialparams=initial_params, bounds=fit_bounds, ifplot = False,
#         startimg = startimg, idxconv = idxconv)
#
#     #get_baseline_firing_rate
#     baseline_avg, baseline, mean_baseline, std_baseline = get_baseline_firing_rate(
#         edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
#         ifseqlen=False, ifrepseq = True, Nseq = Nseq,  avgindices = avgindices, idxconv = idxconv)
#
#     gc.collect()
#
#     tau_transientpre, tau_transientpre_err = convert_tau(params,params_err)
# #     tau_transientpost, tau_transientpost_err = convert_tau(params_trans, params_err_trans)
#     tau_transientpre_avg, tau_transientpre_err_avg = convert_tau_avg(params_blockavg, params_err_blockavg)
# #     tau_transientpost_avg, tau_transientpost_err_avg = convert_tau_avg(params_blockavg,params_err_blockavg)
#
#     # ----------------------------------------- get peaks -----------------------------------------
#
#     samples_img = int(round(lenstim/binsize))
#     height_novelty_avg, height_novelty, mean_novelty, std_novelty, novelty_avgidx, noveltyidx = get_peak_height(
#         edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
#         iftransientpre = False, iftransientpost = False, ifseqlen=False, ifrepseq = True, Nseq = Nseq,
#         avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = 8*samples_img)
#
#     height_trans_pre_avg, height_trans_pre, mean_trans_pre, std_trans_pre, trans_pre_avgidx, trans_preidx = get_peak_height(
#         edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
#         iftransientpre = True, iftransientpost = False, ifseqlen=False, ifrepseq = True, Nseq = Nseq,
#         avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = 8*samples_img)
#
#     # ----------------------------------------- plotting --------------------------------------------
#     plot_Nreps_tau(Seqs, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xlabel="sequence number", xtickstepsize = 1, savename = "NimgTau")
#     plot_Nreps_baseline(Seqs, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xlabel="sequence number", xtickstepsize = 1,savename = "NimgBaseline")
#
#
#
#     # plot unsubtracted data transients, novelty and baseline
#     plot_Nreps_array(Seqs, height_trans_pre, height_trans_pre_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient peak rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre_grey_dots", xtickstepsize = 1)
#     plot_Nreps_array(Seqs, baseline, baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_BL_grey_dots", xtickstepsize = 1)
#     plot_Nreps_array(Seqs, height_novelty, height_novelty_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty peak rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty_grey_dots", xtickstepsize = 1)
#     #plot_Nreps_array(Nimg, height_trans_post, height_trans_post_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient peak rate [Hz]", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre_grey_dots")
#
#     # plot data transients, novelty subtracted baseline
#     #plot_Nreps_array(Nimg, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPost-BL_grey_dots", xtickstepsize = 1)
#     plot_Nreps_array(Seqs, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty-BL_grey_dots", xtickstepsize = 1)
#     plot_Nreps_array(Seqs, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre-BL_grey_dots", xtickstepsize = 1)
#
#     # plot data transients, novelty subtracted baseline with errorbars
#     #plot_Nreps_array_errorbar(Nimg, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPost-BL_grey_errorbar", xtickstepsize = 1)
#     plot_Nreps_array_errorbar(Seqs, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty-BL_grey_errorbar", xtickstepsize = 1)
#     plot_Nreps_array_errorbar(Seqs, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]",xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre-BL_grey_errorbar", xtickstepsize = 1)
#
#     # plot data transients, novelty subtracted baseline with errorbands
#     #plot_Nreps_array_errorband(Nimg, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence length", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPost-BL_grey_errorband", xtickstepsize = 1)
#     plot_Nreps_array_errorband(Seqs, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_Novelty-BL_grey_errorband", xtickstepsize = 1)
#     plot_Nreps_array_errorband(Seqs, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xlabel="sequence number", figure_directory = figure_directory, ifsavefig = True, savename="Nimg_TransientPre-BL_grey_errorband", xtickstepsize = 1)
#
#     # overlap with previous sequence
#
#
#     plot_peak_overlap_seq(novelty_overlap, height_novelty-baseline, Nseq = Nseq, iflegend = False, colorseq = color ,figsize=(15,12),
#                      lw = 3, xlabel = "overlap with previous sequence [%]", ylabel ="novelty peak rate [Hz]",
#                           fontsize = 24,ifioff = False, ifsavefig = True, Nimg = Nimg, savehandle = "NoveltyPeak", figure_directory = figure_directory, ifyticks = False, yticks = [3,4])
#
#     plot_peak_overlap_seq(novelty_overlap, height_trans_pre-baseline, Nseq = Nseq, iflegend = False, colorseq = color ,figsize=(15,12),
#                      lw = 3, xlabel = "overlap with previous sequence [%]", ylabel ="transient peak rate [Hz]",
#                           fontsize = 24, ifioff = False, ifsavefig = True, Nimg = Nimg, savehandle = "TransientPeak", figure_directory = figure_directory,ifyticks = False, yticks = [3,4])
#
#     # ---------------------------------- store variables -----------------------------------------
#     pattern = ["mean*","params*","height*", "tau*", "baseline*"]
#     antipattern = ["*hist*","edges"] # specify lists with different length -> different treatment
#
#     # create results file
#     file_name_results = results_folder + file_name + "/results%s.h5"%timestr
#     f_results = h5py.File(file_name_results, "a")
#
#
#     f_results.create_dataset('avgwindow%d'%avgwindow, data=avgwindow)
#     f_results.create_dataset('Avgwindow%d/Nreps'%avgwindow, data=Nreps)
#
#     for key in dir():
#         if fnmatch.fnmatch(key, pattern[0]):
#             if not fnmatch.fnmatch(key, antipattern[0]):
#                 f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
#             else:
#                 listlen = len(vars()[key])
#                 for i in range(0,listlen):
#                     f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])
#         elif fnmatch.fnmatch(key, pattern[1]):
#             f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
#         elif fnmatch.fnmatch(key, pattern[2]):
#             f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
#         elif fnmatch.fnmatch(key, pattern[3]):
#             f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
#         elif fnmatch.fnmatch(key, pattern[4]):
#             f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
#         elif fnmatch.fnmatch(key, antipattern[1]):
#             listlen = len(vars()[key])
#             for i in range(0,listlen):
#                 f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])
#
#     f_results.close()
#     return mean_hist_E,mean_hist_I
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

    #norm = sum(submat .!= 0)
#norm = sum(submat .!= 0)

        # print(membersidx)
        # submatrix_weights11 = weights[membersidx,membersidx] # EE
        # submatrix_weightsI1 = weights[membersidx,Ne:] # inhibitory to E
        # submatrix_weights1I = weights[Ne:,membersidx] # E to inhibitory
        #print(submatEE)
        #plot_histogram(submatrix_weights11)
        #plot_histogram(submatEE.flatten(),ifExcitatory=True, alpha=1, bins = np.linspace(0,14,29))
        #save_fig(figure_directory, "HistogramEtoEAssembly%d"%ass)
        # plot_histogram(submatrix_weightsI1.flatten(), ifExcitatory=False, alpha=1, bins = np.linspace(0,200,201))
        # save_fig(figure_directory, "HistogramEtoIAssembly%d"%ass)
        # plot_histogram(submatrix_weightsI1.flatten(), ifExcitatory=False, alpha=1, bins = np.linspace(0,200,201))
        # save_fig(figure_directory, "HistogramItoEAssembly%d"%ass)

# function getXassemblyweight(memberspre::Array{Int32,1},memberspost::Array{Int32,1}, weights::Array{Float64,2})
# 	Npre	= size(memberspre,1) # pre
# 	Npost = size(memberspost,1) # post
# 	submat = zeros(Npre,Npost)
# 	precount = 0
# 	for pre in memberspre
# 		precount += 1
# 		postcount = 0
# 		for post in memberspost
# 			postcount += 1
# 			submat[precount, postcount] = weights[pre,post]
# 		end
# 	end
# 	norm = sum(submat .!= 0)
# 	avgweight = sum(submat)/norm
# 	return avgweight
# end
#
#
#
# function getItoassemblyweight(memberspre::Array{Int64,1},memberspost::Array{Int64,1}, weights::Array{Float64,2})
# 	Npre	= size(memberspre,1) # pre
# 	Npost = size(memberspost,1) # post
# 	submat = zeros(Npre,Npost)
# 	precount = 0
# 	for pre in memberspre
# 		precount += 1
# 		postcount = 0
# 		for post in memberspost
# 			postcount += 1
# 			submat[precount, postcount] = weights[pre,post]
# 		end
# 	end
# 	norm = sum(submat .!= 0)
# 	avgweight = sum(submat)/norm
# 	return avgweight
# end

#################################################################
#                                                               #
#                                                               #
#              weightevolution                                  #
#                                                               #
#                                                               #
#################################################################

def weightevolution(file_name, ifcontin,  indivassembly = True, figsize=(20,10),ncol = 1,RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR
    #run_folder = "/gpfs/gjor/personal/schulza/data/main/sequences/"
    # folder with analysed results from spiketime analysis in julia & where to results are stored
    #results_folder = "/home/schulza/Documents/results/main/sequences/"
    #results_folder = "/gpfs/gjor/personal/schulza/results/sequences/"

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
    # weights = frun["postsim"]["weights"].value.transpose()
    #
    # # get submatrix with assembly 1
    # members1 = assemblymembers[0,:]
    #
    # membersidx = np.unique(members1[members1>0])
    # submatrix_weights11 = weights[membersidx,membersidx] # EE
    # submatrix_weightsI1 = weights[membersidx,Ne:] # inhibitory
    # submatrix_weights1I = weights[Ne:,membersidx] # inhibitory

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


    #print(keysavgweights)#
    #print(frun["dursimavg"].keys())
    dtsaveweights = frun["params"]["dtsaveweights"].value * 10 # convert in 0.1 ms # how often are the weights stored
    modwstore = frun["params"]["modwstore"].value # convert in 0.1 ms
    minwstore = frun["params"]["minwstore"].value # convert in 0.1 ms
    if ifcontin:
        tts = range(1,avgXassemblycount+1)
    else:
        tts = range(1,minwstore+1)
        tts.extend(range(minwstore + modwstore, minwstore + modwstore + (avgXassemblycount-minwstore)*modwstore, modwstore))
# print(tts)
    # print(len(tts))

    Nass = np.size(assemblymembers,axis= 0)
    Xweight = np.zeros((Nass,Nass,avgXassemblycount))
    ItoAweight = np.zeros((Nass,avgXassemblycount))
    Etononmensweight = np.zeros((Nass,avgXassemblycount))
    noveltytoAweight = np.zeros((Nass,avgXassemblycount))
    nonmenstoEweight = np.zeros((Nass,avgXassemblycount))
    Atonoveltyweight = np.zeros((Nass,avgXassemblycount))

    #f[groups[0]].keys()
    #Nass = len(assemblymembers[0,;])f
    timevector = np.zeros(len(tts))
    timecounter = 0
    #for tt in range(1,avgXassemblycount+1):#if no minwstore
    for tt in tts:
        #print(tt)
        timett = tt*dtsaveweights/0.1
        timevector[timecounter] = timett/6000000 # in min
        #weightname = "avgXassembly%d_%d" % (tt, (tt*dtsaveweights))
        Xweight[:,:,timecounter] = frun["dursimavg"]["avgXassembly%d_%d" % (tt, (tt*dtsaveweights))].value.transpose()
        ItoAweight[:,timecounter] = frun["dursimavg"]["avgItoassembly%d_%d" % (tt, (tt*dtsaveweights))].value
        Etononmensweight[:,timecounter] = frun["dursimavg"]["avgnonmemstoassembly%d_%d" % (tt, (tt*dtsaveweights))].value
        nonmenstoEweight[:,timecounter] = frun["dursimavg"]["avgassemblytononmems%d_%d" % (tt, (tt*dtsaveweights))].value
        noveltytoAweight[:,timecounter] = frun["dursimavg"]["avgnoveltytoassembly%d_%d" % (tt, (tt*dtsaveweights))].value
        Atonoveltyweight[:,timecounter] = frun["dursimavg"]["avgassemblytonovelty%d_%d" % (tt, (tt*dtsaveweights))].value
        #plotavgweightmatrix(Xweight[:,:,timecounter], maxval= 14)
        timecounter += 1
    plotavgweightmatrix(Xweight[:,:,-1], maxval= 14)
    save_fig(figure_directory, "Final_avgweightmatrix")

    frun.close()
    if Nimg == 4:
        color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange", "black", "silver","dimgrey","gainsboro", "fuchsia", "orchid","plum", "mediumvioletred", "lightseagreen", "lightcyan", "darkslategray", "paleturquoise", "goldenrod","gold", "wheat","darkgoldenrod", "forestgreen", "aquamarine", "palegreen", "lime", ]
    elif Nimg == 3:
        color = ["midnightblue","lightskyblue","royalblue","darkred","darksalmon", "saddlebrown","darkgreen","greenyellow","darkolivegreen","darkmagenta","thistle","indigo","darkorange","tan","sienna", "black", "silver","dimgrey", "fuchsia", "orchid","plum",  "lightseagreen", "lightcyan", "darkslategray",  "goldenrod","gold", "wheat","forestgreen", "aquamarine", "palegreen"]
    elif Nimg == 5:
        color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","indigo","darkred","darksalmon", "saddlebrown","lightcoral","rosybrown","darkgreen","greenyellow","darkolivegreen","chartreuse","turquoise","darkmagenta","thistle","indigo","mediumslateblue","blueviolet","darkorange","tan","sienna","orange", "brown", "black", "silver","dimgrey","gainsboro", "grey","fuchsia", "orchid","plum", "mediumvioletred","purple", "lightseagreen", "lightcyan", "darkslategray", "paleturquoise","teal", "goldenrod","gold", "wheat","darkgoldenrod", "darkkhaki","forestgreen", "aquamarine", "palegreen", "lime", "darkseagreen"]
    elif Nimg == 1:
        color = ["midnightblue","saddlebrown","darkorange"]
    else:
        color = ["midnightblue","saddlebrown","darkorange"]
    colormain = np.copy(color[0:-1:Nimg])
    # plot sequence order
    axiswidth  = 1.5
    fig = plt.figure(figsize=(figsize[0],2))
    ax = fig.add_subplot(111)
    for axis in ['bottom']:
        ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top','right', 'left']:
        ax.spines[axis].set_linewidth(0)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=0)

    lengthblock = (stimulus[1,int(idxblockonset[1]-1)]-stimulus[1,int(idxblockonset[0]-1)])
    for i in range(len(seqnumber)):
        #print(stimulus[1,int(idxblockonset[i]-1)]/60000)
        x = np.linspace(stimulus[1,int(idxblockonset[i]-1)],stimulus[1,int(idxblockonset[i]-1)]+lengthblock, 2)
        y = np.ones(len(x))
        plt.plot(x/60000,y, color = colormain[int(seqnumber[i]-1)],lw = 20)
    plt.plot(0,1, color = "lightgrey",lw = 10)
    plt.xlabel("time [min]", fontsize = 20)
    plt.ylabel("sequence number", fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks([1.0],["1.0"],fontsize = 20)
    plt.tight_layout()
    save_fig(figure_directory, "Sequence_visualisation")
    plt.xlim([20,22])
    plt.legend().remove()
    save_fig(figure_directory, "Sequence_visualisation_range_20_22")
    #axiswidth  = 1.5

    fig = plt.figure(figsize=(figsize[0],3))
    ax = fig.add_subplot(111)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)

    lengthblock = (stimulus[1,int(idxblockonset[1]-1)]-stimulus[1,int(idxblockonset[0]-1)])
    for i in range(len(seqnumber)):
        #print(stimulus[1,int(idxblockonset[i]-1)]/60000)
        x = np.linspace(stimulus[1,int(idxblockonset[i]-1)],stimulus[1,int(idxblockonset[i]-1)]+lengthblock, 2)
        y = np.ones(len(x))*int(seqnumber[i])
        plt.plot(x/60000,y, color = colormain[int(seqnumber[i]-1)],lw = 2)
    plt.plot(0,1, color = "lightgrey",lw = 2)
    plt.xlabel("time [min]", fontsize = 20)
    plt.ylabel("sequence number", fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.tight_layout()
    save_fig(figure_directory, "Sequence_visualisation_bar")
    plt.xlim([20,22])
    plt.legend().remove()
    save_fig(figure_directory, "Sequence_visualisation_bar_range_20_22")

    fig = plt.figure(figsize=(figsize[0],2))
    ax = fig.add_subplot(111)
    for axis in ['bottom']:
        ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top','right', 'left']:
        ax.spines[axis].set_linewidth(0)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=0)

    lengthblock = (stimulus[1,int(idxblockonset[1]-1)]-stimulus[1,int(idxblockonset[0]-1)])
    for i in range(len(seqnumber)):
        #print(stimulus[1,int(idxblockonset[i]-1)]/60000)
        #x = np.linspace(stimulus[1,int(idxblockonset[i]-1)],stimulus[1,int(idxblockonset[i]-1)]+lengthblock, 2)
        #y = np.ones(len(x))
        plt.plot(stimulus[1,int(idxblockonset[i]-1)]/60000,1, "s",color = colormain[int(seqnumber[i]-1)])
    plt.plot(0,1, color = "lightgrey",lw = 10)
    plt.xlabel("time [min]", fontsize = 20)
    plt.ylabel("sequence number", fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks([1.0],["1.0"],fontsize = 20)
    plt.tight_layout()
    save_fig(figure_directory, "Sequence_visualisation_square")
    plt.xlim([20,22])
    plt.legend().remove()
    save_fig(figure_directory, "Sequence_visualisation_square_range_20_22")
    #axiswidth  = 1.5
    fig = plt.figure(figsize=(figsize[0],3))
    ax = fig.add_subplot(111)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)

    lengthblock = (stimulus[1,int(idxblockonset[1]-1)]-stimulus[1,int(idxblockonset[0]-1)])
    for i in range(len(seqnumber)):
        plt.plot(stimulus[1,int(idxblockonset[i]-1)]/60000,int(seqnumber[i]), "_",color = colormain[int(seqnumber[i]-1)])
    plt.plot(0,1, color = "lightgrey",lw = 10)
    plt.xlabel("time [min]", fontsize = 20)
    plt.ylabel("sequence number", fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.tight_layout()
    save_fig(figure_directory, "Sequence_visualisation_bar_steps")
    plt.xlim([20,22])
    plt.legend().remove()
    save_fig(figure_directory, "Sequence_visualisation_bar_steps_range_20_22")

    fig = plt.figure(figsize=(figsize[0],3))
    ax = fig.add_subplot(111)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=axiswidth)

    lengthblock = (stimulus[1,int(idxblockonset[1]-1)]-stimulus[1,int(idxblockonset[0]-1)])
    for i in range(len(seqnumber)):
        plt.plot(stimulus[1,int(idxblockonset[i]-1)]/60000,int(seqnumber[i]), "s",color = colormain[int(seqnumber[i]-1)])
    plt.plot(0,1, color = "lightgrey",lw = 10)
    plt.xlabel("time [min]", fontsize = 20)
    plt.ylabel("sequence number", fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.tight_layout()
    save_fig(figure_directory, "Sequence_visualisation_square_steps")
    plt.xlim([20,22])
    plt.legend().remove()
    save_fig(figure_directory, "Sequence_visualisation_square_steps_range_20_22")

    fig = plt.figure(figsize=(figsize[0],2))
    ax = fig.add_subplot(111)
    for axis in ['bottom']:
        ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top','right', 'left']:
        ax.spines[axis].set_linewidth(0)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=0)

    lengthblock = (stimulus[1,int(idxblockonset[1]-1)]-stimulus[1,int(idxblockonset[0]-1)])
    for i in range(len(seqnumber)):
        #print(stimulus[1,int(idxblockonset[i]-1)]/60000)
        #x = np.linspace(stimulus[1,int(idxblockonset[i]-1)],stimulus[1,int(idxblockonset[i]-1)]+lengthblock, 2)
        #y = np.ones(len(x))
        plt.plot(stimulus[1,int(idxblockonset[i]-1)]/60000,1, "|",color = colormain[int(seqnumber[i]-1)])
    plt.plot(0,1, color = "lightgrey",lw = 10)
    plt.xlabel("time [min]", fontsize = 20)
    plt.ylabel("sequence number", fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks([1.0],["1.0"],fontsize = 20)
    plt.tight_layout()
    save_fig(figure_directory, "Sequence_visualisation_vline")
    plt.xlim([20,22])
    plt.legend().remove()
    save_fig(figure_directory, "Sequence_visualisation_vline_range_20_22")

    fig = plt.figure(figsize=(figsize[0],2))
    ax = fig.add_subplot(111)
    for axis in ['bottom']:
        ax.spines[axis].set_linewidth(axiswidth)
    for axis in ['top','right', 'left']:
        ax.spines[axis].set_linewidth(0)
    ax.xaxis.set_tick_params(width=axiswidth)
    ax.yaxis.set_tick_params(width=0)

    lengthblock = (stimulus[1,int(idxblockonset[1]-1)]-stimulus[1,int(idxblockonset[0]-1)])
    for i in range(len(seqnumber)):
        #print(stimulus[1,int(idxblockonset[i]-1)]/60000)
        #x = np.linspace(stimulus[1,int(idxblockonset[i]-1)],stimulus[1,int(idxblockonset[i]-1)]+lengthblock, 2)
        #y = np.ones(len(x))
        plt.plot(stimulus[1,int(idxblockonset[i]-1)]/60000,1, ",", color = colormain[int(seqnumber[i]-1)])
    plt.plot(0,1, color = "lightgrey",lw = 10)
    plt.xlabel("time [min]", fontsize = 20)
    plt.ylabel("sequence number", fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks([1.0],["1.0"],fontsize = 20)
    plt.tight_layout()
    save_fig(figure_directory, "Sequence_visualisation_dot")
    #color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange", "midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange"]
    #startidx = 0
    startidx = np.arange(0,Nseq*Nimg,Nimg)
    seqnum = Nseq + 1
    fig = plt.figure(figsize=figsize)
    # for i in range(0,20):
    #     plot_popavg_mult(fig,timevector, Xweight[i,i,:], legend = "E", iflegend = False, color = color[i], ifcolor = True, lw = 3,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True)
    for i in reversed(range(Nass)):
        if i in startidx:#i == startidx or i == startidx+Nimg or i == startidx+2*Nimg or i == startidx+3*Nimg or i == startidx+4*Nimg or i == startidx+5*Nimg or i == startidx+6*Nimg or i == startidx+7*Nimg or i == startidx+8*Nimg or i == startidx+9*Nimg:# or i == startidx+10*Nimg or i == startidx+11*Nimg or i == startidx+12*Nimg or i == startidx+13*Nimg or i == startidx+14*Nimg or i == startidx+15*Nimg:
            seqnum -= 1
            plot_popavg_mult(fig,timevector, Xweight[i,i,:], legend = "assemblies seq. %d" % (seqnum), iflegend = True, color = color[i], ifcolor = True, lw = 3,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        elif i < Nimg*Nseq:
            plot_popavg_mult(fig,timevector, Xweight[i,i,:], legend = "E", iflegend = False, color = color[i], ifcolor = True, lw = 3,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        elif i == Nimg*Nseq:
            plot_popavg_mult(fig,timevector, Xweight[i,i,:], legend = "assemblies novelty", iflegend = True, color = "lightgrey", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        else:
            plot_popavg_mult(fig,timevector, Xweight[i,i,:], legend = "E", iflegend = False, color = "lightgrey", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
    save_fig(figure_directory, "XassemblyWeightT")
    plt.xlim([20,22])
    plt.legend().remove()
    save_fig(figure_directory, "XassemblyWeightT_range_20_22")
    plt.xlim([0,0.35])
    plt.ylim([2.5,6])
    save_fig(figure_directory, "Pretraining_first_weight_increase")


    fig = plt.figure(figsize=figsize)
    # for i in range(0,20):
    #     plot_popavg_mult(fig,timevector, Xweight[i,i,:], legend = "E", iflegend = False, color = color[i], ifcolor = True, lw = 3,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True)
    avgweightEass = np.zeros((Nass,len(timevector)))
    #weightENov = np.zeros((Nass,len(timevector)))
    #avgweightIass = np.zeros((Nass,len(timevector)))
    #avgweightINov = np.zeros((Nass,len(timevector)))

    # plot avg weight development check if that causes spikes
    for i in reversed(range(Nass)):
        #plot_popavg_mult(fig,timevector, Xweight[i,i,:], legend = "E", iflegend = False, color = "lightgrey", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True)
        avgweightEass[i,:] += Xweight[i,i,:]

    avgweightE = np.mean(avgweightEass, axis = 0)
    avgweightEmem = np.mean(avgweightEass[0:Nimg*Nseq,:], axis = 0)
    avgweightEnov = np.mean(avgweightEass[Nimg*Nseq:,:], axis = 0)
    stdweightE = np.std(avgweightEass, axis = 0)
    stdweightEmem = np.std(avgweightEass[0:Nimg*Nseq,:], axis = 0)
    stdweightEnov = np.std(avgweightEass[Nimg*Nseq:,:], axis = 0)

    #plot_popavg_mult(fig,timevector, Xweight[1,1,:], legend = "weight traces", iflegend = True, color = "lightgrey", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True)
    #plot_popavg_mult(fig,timevector, avgweightE, legend = "avg. all", iflegend = True, color = "black", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True)#, ncol = ncol)
    plot_popavg_mult(fig,timevector, avgweightEnov, legend = "avg. nov", iflegend = True, color = "darkorange", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True)#, ncol = ncol)
    plot_popavg_mult(fig,timevector, avgweightEmem, legend = "avg. mem", iflegend = True, color = "darkblue", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True)#, ncol = ncol)
    #plt.fill_between(timevector, avgweightE-stdweightE, avgweightE+stdweightE,alpha=0.2, edgecolor='black', facecolor='black')
    plt.fill_between(timevector, avgweightEnov-stdweightEnov, avgweightEnov+stdweightEnov,alpha=0.2, edgecolor='darkorange', facecolor='darkorange')
    plt.fill_between(timevector, avgweightEmem-stdweightEmem, avgweightEmem+stdweightEmem,alpha=0.2, edgecolor='darkblue', facecolor='darkblue')

#     plot_popavg_mult(fig,timevector, Xweight[i,i,:], legend = "avg. all", iflegend = False, color = "black", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True)
#     plot_popavg_mult(fig,timevector, Xweight[i,i,:], legend = "avg. all", iflegend = False, color = "black", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True)
#     plot_popavg_mult(fig,timevector, Xweight[i,i,:], legend = "avg. all", iflegend = False, color = "black", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True)

    save_fig(figure_directory, "XassemblyWeightTWithaverages")


    seqnum = Nseq + 1
    fig = plt.figure(figsize=figsize)
    for i in reversed(range(0,Nass)):
        if i in startidx:#if i == startidx or i == startidx+Nimg or i == startidx+2*Nimg or i == startidx+3*Nimg or i == startidx+4*Nimg or i == startidx+5*Nimg or i == startidx+6*Nimg or i == startidx+7*Nimg or i == startidx+8*Nimg or i == startidx+9*Nimg or i == startidx+10*Nimg or i == startidx+11*Nimg or i == startidx+12*Nimg or i == startidx+13*Nimg or i == startidx+14*Nimg or i == startidx+15*Nimg:
            seqnum -= 1
            plot_popavg_mult(fig,timevector, ItoAweight[i,:], legend = "assemblies seq. %d" % (seqnum), iflegend = True, color = color[i], ifcolor = True, lw = 3,fontsize = 20, xlabel = "time [min]", ylabel ="winhib [pF]", ifioff = True, ncol = ncol)
        elif i < Nimg*Nseq:
            plot_popavg_mult(fig,timevector, ItoAweight[i,:], legend = "E", iflegend = False, color = color[i], ifcolor = True, lw = 3,fontsize = 20, xlabel = "time [min]", ylabel ="winhib [pF]", ifioff = True)
        elif i == Nimg*Nseq:
            plot_popavg_mult(fig,timevector, ItoAweight[i,:], legend = "assemblies novelty", iflegend = True, color = "lightgrey", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="winhib [pF]", ifioff = True, ncol = ncol)
        else:
            plot_popavg_mult(fig,timevector, ItoAweight[i,:], legend = "assemblies novelty", iflegend = False, color = "lightgrey", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="winhib [pF]", ifioff = True, ncol = ncol)
    save_fig(figure_directory, "ItoassemblyWeightT")
    plt.xlim([20,22])
    plt.legend().remove()
    save_fig(figure_directory, "ItoassemblyWeightT_range_20_22")
    avgweightI = np.mean(ItoAweight, axis = 0)
    avgweightImem = np.mean(ItoAweight[0:Nimg*Nseq,:], axis = 0)
    avgweightInov = np.mean(ItoAweight[Nimg*Nseq:,:], axis = 0)
    stdweightI = np.std(ItoAweight, axis = 0)
    stdweightImem = np.std(ItoAweight[0:Nimg*Nseq,:], axis = 0)
    stdweightInov = np.std(ItoAweight[Nimg*Nseq:,:], axis = 0)

    fig = plt.figure(figsize=figsize)
    #plot_popavg_mult(fig,timevector, avgweightI, legend = "avg. all", iflegend = True, color = "black", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="winhib [pF]", ifioff = True)#, ncol = ncol)
    plot_popavg_mult(fig,timevector, avgweightInov, legend = "avg. nov", iflegend = True, color = "darkorange", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="winhib [pF]", ifioff = True)#,, ncol = ncol)
    plot_popavg_mult(fig,timevector, avgweightImem, legend = "avg. mem", iflegend = True, color = "darkblue", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="winhib [pF]", ifioff = True)#,, ncol = ncol)
    #plt.fill_between(timevector, avgweightI-stdweightI, avgweightI+stdweightI,alpha=0.2, edgecolor='black', facecolor='black')
    plt.fill_between(timevector, avgweightInov-stdweightInov, avgweightInov+stdweightInov,alpha=0.2, edgecolor='darkorange', facecolor='darkorange')
    plt.fill_between(timevector, avgweightImem-stdweightImem, avgweightImem+stdweightImem,alpha=0.2, edgecolor='darkblue', facecolor='darkblue')
    save_fig(figure_directory, "ItoassemblyWeightTWithaverages")



    seqnum = Nseq + 1
    fig = plt.figure(figsize=figsize)
    for i in reversed(range(0,Nass)):
        if i in startidx:#if i == startidx or i == startidx+Nimg or i == startidx+2*Nimg or i == startidx+3*Nimg or i == startidx+4*Nimg or i == startidx+5*Nimg or i == startidx+6*Nimg or i == startidx+7*Nimg or i == startidx+8*Nimg or i == startidx+9*Nimg or i == startidx+10*Nimg or i == startidx+11*Nimg or i == startidx+12*Nimg or i == startidx+13*Nimg or i == startidx+14*Nimg or i == startidx+15*Nimg:
            seqnum -= 1
            plot_popavg_mult(fig,timevector, Etononmensweight[i,:], legend = "assemblies seq. %d" % (seqnum), iflegend = True, color = color[i], ifcolor = True, lw = 3,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        elif i < Nimg*Nseq:
            plot_popavg_mult(fig,timevector, Etononmensweight[i,:], legend = "E", iflegend = False, color = color[i], ifcolor = True, lw = 3,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        elif i == Nimg*Nseq:
            plot_popavg_mult(fig,timevector, Etononmensweight[i,:], legend = "assemblies novelty", iflegend = True, color = "lightgrey", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        else:
            plot_popavg_mult(fig,timevector, Etononmensweight[i,:], legend = "assemblies novelty", iflegend = False, color = "lightgrey", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
    save_fig(figure_directory, "EtononmensweightT")
    plt.xlim([20,22])
    plt.legend().remove()
    save_fig(figure_directory, "EtononmensweightT_range_20_22")

    seqnum = Nseq + 1
    fig = plt.figure(figsize=figsize)
    for i in reversed(range(0,Nass)):
        if i in startidx:#if i == startidx or i == startidx+Nimg or i == startidx+2*Nimg or i == startidx+3*Nimg or i == startidx+4*Nimg or i == startidx+5*Nimg or i == startidx+6*Nimg or i == startidx+7*Nimg or i == startidx+8*Nimg or i == startidx+9*Nimg or i == startidx+10*Nimg or i == startidx+11*Nimg or i == startidx+12*Nimg or i == startidx+13*Nimg or i == startidx+14*Nimg or i == startidx+15*Nimg:
            seqnum -= 1
            plot_popavg_mult(fig,timevector, nonmenstoEweight[i,:], legend = "assemblies seq. %d" % (seqnum), iflegend = True, color = color[i], ifcolor = True, lw = 3,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        elif i < Nimg*Nseq:
            plot_popavg_mult(fig,timevector, nonmenstoEweight[i,:], legend = "E", iflegend = False, color = color[i], ifcolor = True, lw = 3,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        elif i == Nimg*Nseq:
            plot_popavg_mult(fig,timevector, nonmenstoEweight[i,:], legend = "assemblies novelty", iflegend = True, color = "lightgrey", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        else:
            plot_popavg_mult(fig,timevector, nonmenstoEweight[i,:], legend = "assemblies novelty", iflegend = False, color = "lightgrey", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
    save_fig(figure_directory, "nonmenstoEweightT")
    plt.xlim([20,22])
    plt.legend().remove()
    save_fig(figure_directory, "nonmenstoEweightT_range_20_22")


    seqnum = Nseq + 1
    fig = plt.figure(figsize=figsize)
    for i in reversed(range(0,Nass)):
        if i in startidx:#if i == startidx or i == startidx+Nimg or i == startidx+2*Nimg or i == startidx+3*Nimg or i == startidx+4*Nimg or i == startidx+5*Nimg or i == startidx+6*Nimg or i == startidx+7*Nimg or i == startidx+8*Nimg or i == startidx+9*Nimg or i == startidx+10*Nimg or i == startidx+11*Nimg or i == startidx+12*Nimg or i == startidx+13*Nimg or i == startidx+14*Nimg or i == startidx+15*Nimg:
            seqnum -= 1
            plot_popavg_mult(fig,timevector, noveltytoAweight[i,:], legend = "assemblies seq. %d" % (seqnum), iflegend = True, color = color[i], ifcolor = True, lw = 3,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        elif i < Nimg*Nseq:
            plot_popavg_mult(fig,timevector, noveltytoAweight[i,:], legend = "E", iflegend = False, color = color[i], ifcolor = True, lw = 3,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        elif i == Nimg*Nseq:
            plot_popavg_mult(fig,timevector, noveltytoAweight[i,:], legend = "assemblies novelty", iflegend = True, color = "lightgrey", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        else:
            plot_popavg_mult(fig,timevector, noveltytoAweight[i,:], legend = "assemblies novelty", iflegend = False, color = "lightgrey", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
    save_fig(figure_directory, "noveltytoAweightT")
    plt.xlim([20,22])
    plt.legend().remove()
    save_fig(figure_directory, "noveltytoAweightT_range_20_22")


    seqnum = Nseq + 1
    fig = plt.figure(figsize=figsize)
    for i in reversed(range(0,Nass)):
        if i in startidx:#if i == startidx or i == startidx+Nimg or i == startidx+2*Nimg or i == startidx+3*Nimg or i == startidx+4*Nimg or i == startidx+5*Nimg or i == startidx+6*Nimg or i == startidx+7*Nimg or i == startidx+8*Nimg or i == startidx+9*Nimg or i == startidx+10*Nimg or i == startidx+11*Nimg or i == startidx+12*Nimg or i == startidx+13*Nimg or i == startidx+14*Nimg or i == startidx+15*Nimg:
            seqnum -= 1
            plot_popavg_mult(fig,timevector, Atonoveltyweight[i,:], legend = "assemblies seq. %d" % (seqnum), iflegend = True, color = color[i], ifcolor = True, lw = 3,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        elif i < Nimg*Nseq:
            plot_popavg_mult(fig,timevector, Atonoveltyweight[i,:], legend = "E", iflegend = False, color = color[i], ifcolor = True, lw = 3,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        elif i == Nimg*Nseq:
            plot_popavg_mult(fig,timevector, Atonoveltyweight[i,:], legend = "assemblies novelty", iflegend = True, color = "lightgrey", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        else:
            plot_popavg_mult(fig,timevector, Atonoveltyweight[i,:], legend = "assemblies novelty", iflegend = False, color = "lightgrey", ifcolor = True, lw = 2,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
    save_fig(figure_directory, "AtonoveltyweightT")
    plt.xlim([20,22])
    plt.legend().remove()
    save_fig(figure_directory, "AtonoveltyweightT_range_20_22")

    axiswidth = 1
    if indivassembly:
    # plot trace of individual assembly on background of all other assemblies to highlight one weight evolution with respect to all others
        specialassembly = 59
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        for i in reversed(range(Nass)):
            plt.plot(timevector,Xweight[i,i,:], label = str(i), color = "lightgrey",lw = 2)
        plt.plot(timevector,Xweight[specialassembly,specialassembly,:], label = str(specialassembly), color = "green",lw = 3)
        plt.xlabel("time [min]",fontsize = 24)
        plt.ylabel("w [pF]",fontsize = 24)
        plt.xticks(fontsize = 24)
        plt.yticks(fontsize = 24)
        save_fig(figure_directory, "IndividualHighlightAss%dEweights" %(specialassembly))

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        for i in reversed(range(Nass)):
            plt.plot(timevector,ItoAweight[i,:], label = str(i), color = "lightgrey",lw = 2)
        plt.plot(timevector,ItoAweight[specialassembly,:], label = str(specialassembly), color = "green",lw = 3)
        plt.xlabel("time [min]",fontsize = 24)
        plt.ylabel("winhib [pF]",fontsize = 24)
        plt.xticks(fontsize = 24)
        plt.yticks(fontsize = 24)
        save_fig(figure_directory, "IndividualHighlightAss%dIweights" %(specialassembly))
        plt.show()
        plt.close(fig)
    return Xweight, ItoAweight, timevector, avgweightEmem, avgweightImem, avgweightEnov, avgweightImem



#################################################################
#                                                               #
#                                                               #
#              run_variable_repetitions                         #
#                                                               #
#                                                               #
#################################################################

def run_variable_repetitions(file_name, avgwindow = 8, timestr = "_now",RUN_DIR="../data/", RESULTS_DIR ="../results/"):
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
            hist_I[seq-1][bl-1,:] = f["I%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_nomem[seq-1][bl-1,:] = f["ENonMem%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_nomemnonov[seq-1][bl-1,:] = f["ENonMemNoNov%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
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
    print(Nreps)
    plot_all_averages(edges, mean_hist_E, Nreps, savehandle = "E", figure_directory = figure_directory, Nreponset = 1, color = color, ifoffset=True, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_I, Nreps, savehandle = "I", figure_directory = figure_directory, Nreponset = 1, color = color, ifoffset=True, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_boxcar, Nreps, savehandle = "E_boxcar", Nreponset = 1, startidx = idxconv, endidx = -idxconv, figure_directory = figure_directory, color = color, ifoffset=True, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomem, Nreps, savehandle = "E_nomem", Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=True, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomemnonov, Nreps, savehandle = "E_nomemnonov", Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=True, iflegend=False, ifyticks=False)

    plot_all_traces_and_average(edges, hist_E, mean_hist_E, Nreps, savehandle = "E", Nblocks = Nblocks, Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_I, mean_hist_I, Nreps, savehandle = "I", Nblocks = Nblocks, Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_boxcar, mean_hist_E_boxcar, Nreps, Nreponset = 1, savehandle = "E_boxcar_avg%d" % int(avgwindow) , startidx = idxconv, endidx = -idxconv, Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomem, mean_hist_E_nomem, Nreps, Nreponset = 1, savehandle = "E_nonmem", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomemnonov, mean_hist_E_nomemnonov, Nreps, Nreponset = 1, savehandle = "E_nomemnonov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)

    plot_all_averages(edges, mean_hist_E, Nreps, savehandle = "Enooffset", figure_directory = figure_directory, Nreponset = 1, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_I, Nreps, savehandle = "Inooffset", figure_directory = figure_directory, Nreponset = 1, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_boxcar, Nreps, savehandle = "E_boxcarnooffset", Nreponset = 1, startidx = idxconv, endidx = -idxconv, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomem, Nreps, savehandle = "E_nomemnooffset", Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomemnonov, Nreps, savehandle = "E_nomemnonovnooffset", Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)

    plot_all_traces_and_average(edges, hist_E, mean_hist_E, Nreps, savehandle = "Enooffset", Nblocks = Nblocks, Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_I, mean_hist_I, Nreps, savehandle = "Inooffset", Nblocks = Nblocks, Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_boxcar, mean_hist_E_boxcar, Nreps, Nreponset = 1, savehandle = "Enooffset_boxcar_avg%d" % int(avgwindow) , startidx = idxconv, endidx = -idxconv, Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomem, mean_hist_E_nomem, Nreps, Nreponset = 1, savehandle = "Enooffsetnonmem", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomemnonov, mean_hist_E_nomemnonov, Nreps, Nreponset = 1, savehandle = "Enooffset_nomemnonov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)


    # ------------------------------------ FITTING ---------------------------------------------
    # """fit_variable_repetitions_gen_arrays(args):
    #     perform fitting of all traces included in datalist and meandatalist
    #         determine the baseline firing rate prior to the novelty stimulation

    # set initial parameters for fitting of the exponential curve
    # fit a * exp(-t/tau) + a_0
    initial_params = [2, 20, 3]
    #                [a, tau,a_0]
    fit_bounds = (0, [10., 20., 10])
    print("FIT_Bounds")
    print(fit_bounds)
    avgindices = 30
    startimg = Nimg # after which image should fit start at block onset
    print(startimg)
    print(idxconv)
    print("Issue here: ")

    # fitting of initial transient
    t_before_nov, params_blockavg, params_covariance_blockavg, params_err_blockavg, params, params_covariance, params_err = fit_variable_repetitions_gen_arrays_startidx(
        edges,hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        avgindices = avgindices, initialparams=initial_params, bounds=fit_bounds, ifplot = True,
        startimg = startimg, idxconv = idxconv)

    #get_baseline_firing_rate
    baseline_avg, baseline, mean_baseline, std_baseline = get_baseline_firing_rate(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        avgindices = avgindices, idxconv = idxconv)



    # fitting of post novelty transient
    t_before_trans, params_blockavg_trans, params_err_blockavg_trans, params_trans, params_err_trans = fit_variable_repetitions_gen_arrays_postnovelty(
        edges,hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        avgindices = avgindices, initialparams=initial_params, bounds=fit_bounds, ifplot = True,
        startimg = startimg, idxconv = idxconv)

    # collect garbage
    gc.collect()

    tau_transientpre, tau_transientpre_err = convert_tau(params,params_err)
    tau_transientpost, tau_transientpost_err = convert_tau(params_trans, params_err_trans)
    tau_transientpre_avg, tau_transientpre_err_avg = convert_tau_avg(params_blockavg, params_err_blockavg)
    tau_transientpost_avg, tau_transientpost_err_avg = convert_tau_avg(params_blockavg_trans, params_err_blockavg_trans)

    # -----------------------------------------

    samples_img = int(round(lenstim/binsize))
    height_novelty_avg, height_novelty, mean_novelty, std_novelty, novelty_avgidx, noveltyidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = False, iftransientpost = False,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = Nimg*samples_img)

    height_trans_pre_avg, height_trans_pre, mean_trans_pre, std_trans_pre, trans_pre_avgidx, trans_preidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = True, iftransientpost = False,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = Nimg*samples_img)

    height_trans_post_avg, height_trans_post, mean_trans_post, std_trans_post, trans_post_avgidx, trans_postidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = False, iftransientpost = True,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = Nimg*samples_img)

    # ---------------------------------------- plotting --------------------------------------------------------
    # plot pre transient decay constant vs. number of repetitions
    plot_Nreps_tau(Nreps, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xtickstepsize=1,savename="Nreps_Decay_Const_From_Fit_Pre_grey_dots")
    # plot baseline determined from fit vs. number of repetitions
    plot_Nreps_baseline(Nreps, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xtickstepsize=1,savename="Nreps_Baseline_From_Fit_Pre_grey_dots")
    # saving and reloading for comparing instantiations

    # plot post novelty transient decay constant vs. number of repetitions
    plot_Nreps_tau(Nreps, params_trans, params_blockavg_trans, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xtickstepsize=1,savename="Nreps_Decay_Const_From_Fit_Post_grey_dots")
    # plot baseline determined from fit vs. number of repetitions
    plot_Nreps_baseline(Nreps, params_trans, params_blockavg_trans, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xtickstepsize=1,savename="Nreps_Baseline_From_Fit_Post_grey_dots")



    # plot unsubtracted data transients, novelty and baseline
    plot_Nreps_array(Nreps, height_trans_pre, height_trans_pre_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient peak rate [Hz]", figure_directory = figure_directory, xtickstepsize=1, ifsavefig = True, savename="Nreps_TransientPre_grey_dots")
    plot_Nreps_array(Nreps, baseline, baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="baseline rate [Hz]", figure_directory = figure_directory, ifsavefig = True, xtickstepsize=1, savename="Nreps_BL_grey_dots")
    plot_Nreps_array(Nreps, height_novelty, height_novelty_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty peak rate [Hz]", figure_directory = figure_directory, xtickstepsize=1, ifsavefig = True, savename="Nreps_Novelty_grey_dots")
    plot_Nreps_array(Nreps, height_trans_post, height_trans_post_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient peak rate [Hz]", figure_directory = figure_directory, xtickstepsize=1, ifsavefig = True, savename="Nreps_TransientPre_grey_dots")

    # plot data transients, novelty subtracted baseline
    plot_Nreps_array(Nreps, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPost-BL_grey_dots")
    plot_Nreps_array(Nreps, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_Novelty-BL_grey_dots")
    plot_Nreps_array(Nreps, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPre-BL_grey_dots")

    # plot data transients, novelty subtracted baseline with errorbars
    plot_Nreps_array_errorbar(Nreps, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPost-BL_grey_errorbar")
    plot_Nreps_array_errorbar(Nreps, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_Novelty-BL_grey_errorbar")
    plot_Nreps_array_errorbar(Nreps, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPre-BL_grey_errorbar")

    # plot data transients, novelty subtracted baseline with errorbands
    plot_Nreps_array_errorband(Nreps, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPost-BL_grey_errorband")
    plot_Nreps_array_errorband(Nreps, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_Novelty-BL_grey_errorband")
    plot_Nreps_array_errorband(Nreps, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPre-BL_grey_errorband")

    # declare variable name patterns to be stored in hdf5 file
    # lists with different lengths cannot be stored in hdf5 -> split up into indiv arrays with dataset name string(index)

    # -------------------------------------- saving ----------------------------------------------------------------
    pattern = ["mean*","params*","height*", "tau*", "baseline*"]
    antipattern = ["*hist*","edges"] # specify lists with different length -> different treatment

    # create results file
    file_name_results = results_folder + file_name + "/results%s.h5"%timestr
    f_results = h5py.File(file_name_results, "a")
    print(f_results)

    f_results.create_dataset('avgwindow%d'%avgwindow, data=avgwindow)
    f_results.create_dataset('Avgwindow%d/Nreps'%avgwindow, data=Nreps)

    for key in dir():
        if fnmatch.fnmatch(key, pattern[0]):
            if not fnmatch.fnmatch(key, antipattern[0]):
                f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
            else:
                listlen = len(vars()[key])
                for i in range(0,listlen):
                    f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])
        elif fnmatch.fnmatch(key, pattern[1]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[2]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[3]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[4]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, antipattern[1]):
            listlen = len(vars()[key])
            for i in range(0,listlen):
                f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])

    f_results.close()

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
#################################################################
#                                                               #
#                                                               #
#              run_sequence_length                              #
#                                                               #
#                                                               #
#################################################################

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

    # figsize = (10, 6)
    # plot_Nreps_array_comparison(Nreps[0], baseline_avg, Nfiles = len(file_names), ifxlims = True, figsize= figsize, xtickstepsize = 1, fontsize = 20, ylabel ="baseline rate [Hz]", figure_directory = figure_directory, ifsavefig = True, savename="All_baseline_rate_band")
    # plot_Nreps_array_comparison(Nreps[0], height_novelty_avg, Nfiles = len(file_names), c3 = "darkorange",ifxlims = True, figsize= figsize, xtickstepsize = 1, fontsize = 20, ylabel ="novelty rate [Hz]", figure_directory = figure_directory, ifsavefig = True, savename="All_novelty_rate_band")
    # plot_Nreps_array_comparison(Nreps[0], height_trans_post_avg, Nfiles = len(file_names), c3 = "darkgreen", ifxlims = True, figsize= figsize, xtickstepsize = 1, fontsize = 20, ylabel ="transient rate [Hz]", figure_directory = figure_directory, ifsavefig = True, savename="All_transient_rate_band")
    #
    # plot_Nreps_array_comparison(Nreps[0], [ht-bl for ht,bl in zip(height_novelty_avg,baseline_avg)], Nfiles = len(file_names), figsize= figsize, xtickstepsize = 1, fontsize = 20, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", figure_directory = figure_directory, ifsavefig = True, savename="All_novelty_-_baseline_rate_band")
    # plot_Nreps_array_comparison(Nreps[0], [ht-bl for ht,bl in zip(height_trans_post_avg,baseline_avg)], Nfiles = len(file_names), figsize= figsize, xtickstepsize = 1, fontsize = 20, ifxlims = True, ylabel ="transient - baseline rate [Hz]", figure_directory = figure_directory, ifsavefig = True, savename="All_transient_-_baseline_rate_band")
    #
    # fig = plt.figure(figsize= figsize)
    # plot_Nreps_array_comparison_mult(fig,Nreps[0], [ht-bl for ht,bl in zip(height_trans_post_avg,baseline_avg)], c3 = "darkblue", Nfiles = len(file_names), ifxlims = True,figsize= figsize, xtickstepsize = 1, fontsize = 20,  ylabel ="peak - baseline rate [Hz]", legend="transient",iflegend=True,figure_directory = figure_directory, ifsavefig = True, savename="N-BL_T-BL_band")
    # plot_Nreps_array_comparison_mult(fig,Nreps[0], [ht-bl for ht,bl in zip(height_novelty_avg,baseline_avg)], c3 = "darkorange", Nfiles = len(file_names), ifxlims = True, figsize= figsize, xtickstepsize = 1, fontsize = 20, ylabel ="peak - baseline rate [Hz]", legend="novelty", iflegend=True, figure_directory = figure_directory, ifsavefig = True, savename="N-BL_T-BL_band")
    #
    # # plot decay constants
    # plot_Nreps_array_comparison(Nreps[0], tau_transientpost_avg, figsize= figsize, xtickstepsize = 1, fontsize = 20, Nfiles = len(file_names), c3 = "darkgreen", ifxlims = True, ylabel ="decay constant [s]", figure_directory = figure_directory, ifsavefig = True, savename="All_Tau_transientpost_avg_band")
    # plot_Nreps_array_comparison(Nreps[0], tau_transientpre_avg, figsize= figsize, xtickstepsize = 1, fontsize = 20,Nfiles = len(file_names), c3 = "darksalmon", ifxlims = True, ylabel ="decay constant [s]", figure_directory = figure_directory, ifsavefig = True, savename="All_Tau_transientpre_avg_band")
    #
    # fig = plt.figure(figsize= figsize)
    # plot_Nreps_array_comparison_mult(fig,Nreps[0], tau_transientpost_avg, c3 = "darkgreen", Nfiles = len(file_names), ifxlims = True, figsize= figsize, xtickstepsize = 1, fontsize = 20, ylabel ="decay constant [s]", legend="post novelty",iflegend=True,figure_directory = figure_directory, ifsavefig = True, savename="All_tau_transient_pre_and_post_band")
    # plot_Nreps_array_comparison_mult(fig,Nreps[0], tau_transientpre_avg, c3 = "darksalmon", Nfiles = len(file_names), ifxlims = True, figsize= figsize, xtickstepsize = 1, fontsize = 20, ylabel ="decay constant [s]", legend="pre novelty", iflegend=True, figure_directory = figure_directory, ifsavefig = True, savename="All_tau_transient_pre_and_post_band")
    #
    #
    # # plot with error bars
    # plot_Nreps_array_comparison_errorbar(Nreps[0], baseline_avg, Nfiles = len(file_names), ifxlims = True, figsize= figsize, xtickstepsize = 1, fontsize = 20, ylabel ="baseline rate [Hz]", figure_directory = figure_directory, ifsavefig = True, savename="All_baseline_rate_errorbar")
    # plot_Nreps_array_comparison_errorbar(Nreps[0], height_novelty_avg, Nfiles = len(file_names), c3 = "darkorange",ifxlims = True, figsize= figsize, xtickstepsize = 1, fontsize = 20, ylabel ="novelty rate [Hz]", figure_directory = figure_directory, ifsavefig = True, savename="All_novelty_rate_errorbar")
    # plot_Nreps_array_comparison_errorbar(Nreps[0], height_trans_post_avg, Nfiles = len(file_names), c3 = "darkgreen", ifxlims = True, figsize= figsize, xtickstepsize = 1, fontsize = 20, ylabel ="transient rate [Hz]", figure_directory = figure_directory, ifsavefig = True, savename="All_transient_rate_errorbar")
    #
    # plot_Nreps_array_comparison_errorbar(Nreps[0], [ht-bl for ht,bl in zip(height_novelty_avg,baseline_avg)], Nfiles = len(file_names), figsize= figsize, xtickstepsize = 1, fontsize = 20, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", figure_directory = figure_directory, ifsavefig = True, savename="All_novelty_-_baseline_rate_errorbar")
    # plot_Nreps_array_comparison_errorbar(Nreps[0], [ht-bl for ht,bl in zip(height_trans_post_avg,baseline_avg)], Nfiles = len(file_names), figsize= figsize, xtickstepsize = 1, fontsize = 20, ifxlims = True, ylabel ="transient - baseline rate [Hz]", figure_directory = figure_directory, ifsavefig = True, savename="All_transient_-_baseline_rate_errorbar")
    #
    # fig = plt.figure(figsize= figsize)
    # plot_Nreps_array_comparison_errorbar_mult(fig,Nreps[0], [ht-bl for ht,bl in zip(height_trans_post_avg,baseline_avg)], c3 = "black", Nfiles = len(file_names), ifxlims = True,figsize= figsize, xtickstepsize = 1, fontsize = 20,  ylabel ="peak - baseline rate [Hz]", legend="transient",iflegend=True,figure_directory = figure_directory, ifsavefig = True, savename="N-BL_T-BL_errorbar")
    # plot_Nreps_array_comparison_errorbar_mult(fig,Nreps[0], [ht-bl for ht,bl in zip(height_novelty_avg,baseline_avg)], c3 = "grey", Nfiles = len(file_names), ifxlims = True, figsize= figsize, xtickstepsize = 1, fontsize = 20, ylabel ="peak - baseline rate [Hz]", legend="novelty", iflegend=True, figure_directory = figure_directory, ifsavefig = True, savename="N-BL_T-BL_errorbar")
    #
    #
    # # plot decay constants
    # plot_Nreps_array_comparison_errorbar(Nreps[0], tau_transientpost_avg, figsize= figsize, xtickstepsize = 1, fontsize = 20, Nfiles = len(file_names), c3 = "darkgreen", ifxlims = True, ylabel ="decay constant [s]", figure_directory = figure_directory, ifsavefig = True, savename="All_Tau_transientpost_avg_errorbar")
    # plot_Nreps_array_comparison_errorbar(Nreps[0], tau_transientpre_avg, figsize= figsize, xtickstepsize = 1, fontsize = 20,Nfiles = len(file_names), c3 = "darksalmon", ifxlims = True, ylabel ="decay constant [s]", figure_directory = figure_directory, ifsavefig = True, savename="All_Tau_transientpre_avg_errorbar")
    #
    # fig = plt.figure(figsize= figsize)
    # plot_Nreps_array_comparison_errorbar_mult(fig,Nreps[0], tau_transientpost_avg, c3 = "darkgreen", Nfiles = len(file_names), ifxlims = True, figsize= figsize, xtickstepsize = 1, fontsize = 20, ylabel ="decay constant [s]", legend="post novelty",iflegend=True,figure_directory = figure_directory, ifsavefig = True, savename="All_tau_transient_pre_and_post_errorbar")
    # plot_Nreps_array_comparison_errorbar_mult(fig,Nreps[0], tau_transientpre_avg, c3 = "darksalmon", Nfiles = len(file_names), ifxlims = True, figsize= figsize, xtickstepsize = 1, fontsize = 20, ylabel ="decay constant [s]", legend="pre novelty", iflegend=True, figure_directory = figure_directory, ifsavefig = True, savename="All_tau_transient_pre_and_post_errorbar")

    return Nreps[0], baseline_avg, height_novelty_avg, height_trans_post_avg# [ht-bl for ht,bl in zip(height_novelty_avg,baseline_avg)], [ht-bl for ht,bl in zip(height_trans_post_avg,baseline_avg)]

def run_sequence_length(file_name, avgwindow = 8, timestr = "_now",RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR
    #run_folder = "/gpfs/gjor/personal/schulza/data/main/sequences/"
    # folder with analysed results from spiketime analysis in julia & where to results are stored
    #results_folder = "/gpfs/gjor/personal/schulza/results/sequences/"


    # define folder where figues should be stored
    figure_directory = results_folder + file_name + "/" + "figures_window%d/"%avgwindow
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    # read in run parameters
    file_name_run = run_folder + file_name
    # open file
    frun = h5py.File(file_name_run, "r")

    # read in stimulus parameters
#     Nimg, lenNreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength  = frun["initial"]["stimparams"].value
#     repetitions  = frun["initial"]["repetitions"].value
#     Nreps  = frun["initial"]["Nreps"].value

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
    ifplotting = True
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
        ifseqlen=True, avgindices = avgindices, initialparams=initial_params, bounds=fit_bounds, ifplot = True,
        startimg = startimg, idxconv = idxconv)

    #get_baseline_firing_rate
    baseline_avg, baseline, mean_baseline, std_baseline = get_baseline_firing_rate(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        ifseqlen=True, avgindices = avgindices, idxconv = idxconv)


    plot_all_averages_with_fits(edges, mean_hist_E, Nimg, params_blockavg, savehandle = "E_withfits", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=True, iflegend=False, ifyticks=False)
    plot_all_averages_with_fits(edges, mean_hist_E, Nimg, params_blockavg, savehandle = "E_boxcar_withfits", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=True, iflegend=False, ifyticks=False)

#     # fitting of post novelty transient
#     t_before_trans, params_blockavg_trans, params_err_blockavg_trans, params_trans, params_err_trans = fit_variable_repetitions_gen_arrays_postnovelty(
#         edges,hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
#         ifseqlen=True, avgindices = avgindices, initialparams=initial_params, bounds=fit_bounds, ifplot = False,
#         startimg = startimg, idxconv = idxconv)

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


#     height_trans_post_avg, height_trans_post, mean_trans_post, std_trans_post, trans_post_avgidx, trans_postidx = get_peak_height(
#         edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
#         iftransientpre = False, iftransientpost = True,
#         avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = Nimg*samples_img)

    # ---------------------------------------- plotting --------------------------------------------------------
    if ifplotting:

        # plot pre transient decay constant vs. number of repetitions
        plot_Nreps_tau(Nimg, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=False, xlabel="sequence length", xtickstepsize = 1, savename = "NimgTau")
        # plot baseline determined from fit vs. number of repetitions
        plot_Nreps_baseline(Nimg, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=False, xlabel="sequence length", xtickstepsize = 1,savename = "NimgBaseline")
        # saving and reloading for comparing instantiations

#     # plot post novelty transient decay constant vs. number of repetitions
#     plot_Nreps_tau(Nreps, params_trans, params_blockavg_trans, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=False)
#     # plot baseline determined from fit vs. number of repetitions
#     plot_Nreps_baseline(Nreps, params_trans, params_blockavg_trans, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=False)


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

    # declare variable name patterns to be stored in hdf5 file
    # lists with different lengths cannot be stored in hdf5 -> split up into indiv arrays with dataset name string(index)

    # -------------------------------------- saving ----------------------------------------------------------------
    pattern = ["mean*","params*","height*", "tau*", "baseline*"]
    antipattern = ["*hist*","edges"] # specify lists with different length -> different treatment

    # create results file
    file_name_results = results_folder + file_name + "/results%s.h5"%timestr
    f_results = h5py.File(file_name_results, "a")


    f_results.create_dataset('avgwindow%d'%avgwindow, data=avgwindow)
    f_results.create_dataset('Avgwindow%d/Nreps'%avgwindow, data=Nreps)

    for key in dir():
        if fnmatch.fnmatch(key, pattern[0]):
            if not fnmatch.fnmatch(key, antipattern[0]):
                f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
            else:
                listlen = len(vars()[key])
                for i in range(0,listlen):
                    f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])
        elif fnmatch.fnmatch(key, pattern[1]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[2]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[3]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[4]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, antipattern[1]):
            listlen = len(vars()[key])
            for i in range(0,listlen):
                f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])

    f_results.close()
    gc.collect()



def run_variable_repetitions_member(file_name, avgwindow = 8, timestr = "_now",RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR
# same as other run_variable_repetitions function but includes memebers
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
    hist_E_mem  = []
    mean_hist_E_mem = []
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
        hist_E_mem.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_I.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_nomem.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_nomemnonov.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_boxcar.append(np.zeros((Nblocks,len(edges[seq-1]))))

        for bl in range(1, Nblocks + 1):
            #vars()['hist_E_all' + str(seq-1)][bl-1][:] = f["E%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E[seq-1][bl-1,:] = f["E%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_mem[seq-1][bl-1,:] = f["EMem%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_I[seq-1][bl-1,:] = f["I%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_nomem[seq-1][bl-1,:] = f["ENonMem%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_nomemnonov[seq-1][bl-1,:] = f["ENonMemNoNov%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_boxcar[seq-1][bl-1,:] = np.convolve(hist_E[seq-1][bl-1,:], np.ones((avgwindow,))/avgwindow, mode='same')

        # get averages over blocks
        mean_hist_E.append(np.mean(hist_E[seq-1][:,:],axis = 0))
        mean_hist_E_mem.append(np.mean(hist_E_mem[seq-1][:,:],axis = 0))
        mean_hist_I.append(np.mean(hist_I[seq-1][:,:],axis = 0))
        mean_hist_E_nomem.append(np.mean(hist_E_nomem[seq-1][:,:],axis = 0))
        mean_hist_E_nomemnonov.append(np.mean(hist_E_nomemnonov[seq-1][:,:],axis = 0))
        mean_hist_E_boxcar.append(np.mean(hist_E_boxcar[seq-1][:,:],axis = 0))

    # plotting
    color = ["midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon"]

    idxconv = np.floor_divide(avgwindow,2)+1
    print(Nreps)
    plot_all_averages(edges, mean_hist_E, Nreps, savehandle = "E", figure_directory = figure_directory, Nreponset = 1, color = color, ifoffset=True, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_mem, Nreps, savehandle = "E_mem", figure_directory = figure_directory, Nreponset = 1, color = color, ifoffset=True, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_I, Nreps, savehandle = "I", figure_directory = figure_directory, Nreponset = 1, color = color, ifoffset=True, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_boxcar, Nreps, savehandle = "E_boxcar", Nreponset = 1, startidx = idxconv, endidx = -idxconv, figure_directory = figure_directory, color = color, ifoffset=True, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomem, Nreps, savehandle = "E_nomem", Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=True, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomemnonov, Nreps, savehandle = "E_nomemnonov", Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=True, iflegend=False, ifyticks=False)

    plot_all_traces_and_average(edges, hist_E, mean_hist_E, Nreps, savehandle = "E", Nblocks = Nblocks, Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_mem, mean_hist_E_mem, Nreps, savehandle = "E_mem", Nblocks = Nblocks, Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_I, mean_hist_I, Nreps, savehandle = "I", Nblocks = Nblocks, Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_boxcar, mean_hist_E_boxcar, Nreps, Nreponset = 1, savehandle = "E_boxcar_avg%d" % int(avgwindow) , startidx = idxconv, endidx = -idxconv, Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomem, mean_hist_E_nomem, Nreps, Nreponset = 1, savehandle = "E_nonmem", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomemnonov, mean_hist_E_nomemnonov, Nreps, Nreponset = 1, savehandle = "E_nomemnonov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)

    plot_all_averages(edges, mean_hist_E, Nreps, savehandle = "Enooffset", figure_directory = figure_directory, Nreponset = 1, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_mem, Nreps, savehandle = "E_mem_nooffset", figure_directory = figure_directory, Nreponset = 1, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_I, Nreps, savehandle = "Inooffset", figure_directory = figure_directory, Nreponset = 1, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_boxcar, Nreps, savehandle = "E_boxcarnooffset", Nreponset = 1, startidx = idxconv, endidx = -idxconv, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomem, Nreps, savehandle = "E_nomemnooffset", Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomemnonov, Nreps, savehandle = "E_nomemnonovnooffset", Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)

    plot_all_traces_and_average(edges, hist_E, mean_hist_E, Nreps, savehandle = "Enooffset", Nblocks = Nblocks, Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_mem, mean_hist_E_mem, Nreps, savehandle = "E_mem_nooffset", Nblocks = Nblocks, Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_I, mean_hist_I, Nreps, savehandle = "Inooffset", Nblocks = Nblocks, Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_boxcar, mean_hist_E_boxcar, Nreps, Nreponset = 1, savehandle = "Enooffset_boxcar_avg%d" % int(avgwindow) , startidx = idxconv, endidx = -idxconv, Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomem, mean_hist_E_nomem, Nreps, Nreponset = 1, savehandle = "Enooffsetnonmem", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomemnonov, mean_hist_E_nomemnonov, Nreps, Nreponset = 1, savehandle = "Enooffset_nomemnonov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)


    # ------------------------------------ FITTING ---------------------------------------------
    # """fit_variable_repetitions_gen_arrays(args):
    #     perform fitting of all traces included in datalist and meandatalist
    #         determine the baseline firing rate prior to the novelty stimulation

    # set initial parameters for fitting of the exponential curve
    # fit a * exp(-t/tau) + a_0
    initial_params = [2, 20, 3]
    #                [a, tau,a_0]
    fit_bounds = (0, [10., 20., 10])
    print("FIT_Bounds")
    print(fit_bounds)
    avgindices = 30
    startimg = Nimg # after which image should fit start at block onset
    print(startimg)
    print(idxconv)
    print(edges)
    print(hist_E_boxcar)
    print(mean_hist_E_boxcar)
    print("Issue here: ")

    # fitting of initial transient
    t_before_nov, params_blockavg, params_covariance_blockavg, params_err_blockavg, params, params_covariance, params_err = fit_variable_repetitions_gen_arrays_startidx(
        edges,hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        avgindices = avgindices, initialparams=initial_params, bounds=fit_bounds, ifplot = True,
        startimg = startimg, idxconv = idxconv)

    #get_baseline_firing_rate
    baseline_avg, baseline, mean_baseline, std_baseline = get_baseline_firing_rate(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        avgindices = avgindices, idxconv = idxconv)



    # fitting of post novelty transient
    t_before_trans, params_blockavg_trans, params_err_blockavg_trans, params_trans, params_err_trans = fit_variable_repetitions_gen_arrays_postnovelty(
        edges,hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        avgindices = avgindices, initialparams=initial_params, bounds=fit_bounds, ifplot = True,
        startimg = startimg, idxconv = idxconv)

    # collect garbage
    gc.collect()

    tau_transientpre, tau_transientpre_err = convert_tau(params,params_err)
    tau_transientpost, tau_transientpost_err = convert_tau(params_trans, params_err_trans)
    tau_transientpre_avg, tau_transientpre_err_avg = convert_tau_avg(params_blockavg, params_err_blockavg)
    tau_transientpost_avg, tau_transientpost_err_avg = convert_tau_avg(params_blockavg_trans, params_err_blockavg_trans)

    # -----------------------------------------

    samples_img = int(round(lenstim/binsize))
    height_novelty_avg, height_novelty, mean_novelty, std_novelty, novelty_avgidx, noveltyidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = False, iftransientpost = False,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = Nimg*samples_img)

    height_trans_pre_avg, height_trans_pre, mean_trans_pre, std_trans_pre, trans_pre_avgidx, trans_preidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = True, iftransientpost = False,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = Nimg*samples_img)

    height_trans_post_avg, height_trans_post, mean_trans_post, std_trans_post, trans_post_avgidx, trans_postidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = False, iftransientpost = True,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = Nimg*samples_img)

    # ---------------------------------------- plotting --------------------------------------------------------
    # plot pre transient decay constant vs. number of repetitions
    plot_Nreps_tau(Nreps, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xtickstepsize=1,savename="Nreps_Decay_Const_From_Fit_Pre_grey_dots")
    # plot baseline determined from fit vs. number of repetitions
    plot_Nreps_baseline(Nreps, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xtickstepsize=1,savename="Nreps_Baseline_From_Fit_Pre_grey_dots")
    # saving and reloading for comparing instantiations

    # plot post novelty transient decay constant vs. number of repetitions
    plot_Nreps_tau(Nreps, params_trans, params_blockavg_trans, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xtickstepsize=1,savename="Nreps_Decay_Const_From_Fit_Post_grey_dots")
    # plot baseline determined from fit vs. number of repetitions
    plot_Nreps_baseline(Nreps, params_trans, params_blockavg_trans, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xtickstepsize=1,savename="Nreps_Baseline_From_Fit_Post_grey_dots")



    # plot unsubtracted data transients, novelty and baseline
    plot_Nreps_array(Nreps, height_trans_pre, height_trans_pre_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient peak rate [Hz]", figure_directory = figure_directory, xtickstepsize=1, ifsavefig = True, savename="Nreps_TransientPre_grey_dots")
    plot_Nreps_array(Nreps, baseline, baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="baseline rate [Hz]", figure_directory = figure_directory, ifsavefig = True, xtickstepsize=1, savename="Nreps_BL_grey_dots")
    plot_Nreps_array(Nreps, height_novelty, height_novelty_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty peak rate [Hz]", figure_directory = figure_directory, xtickstepsize=1, ifsavefig = True, savename="Nreps_Novelty_grey_dots")
    plot_Nreps_array(Nreps, height_trans_post, height_trans_post_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient peak rate [Hz]", figure_directory = figure_directory, xtickstepsize=1, ifsavefig = True, savename="Nreps_TransientPre_grey_dots")

    # plot data transients, novelty subtracted baseline
    plot_Nreps_array(Nreps, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPost-BL_grey_dots")
    plot_Nreps_array(Nreps, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_Novelty-BL_grey_dots")
    plot_Nreps_array(Nreps, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPre-BL_grey_dots")

    # plot data transients, novelty subtracted baseline with errorbars
    plot_Nreps_array_errorbar(Nreps, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPost-BL_grey_errorbar")
    plot_Nreps_array_errorbar(Nreps, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_Novelty-BL_grey_errorbar")
    plot_Nreps_array_errorbar(Nreps, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPre-BL_grey_errorbar")

    # plot data transients, novelty subtracted baseline with errorbands
    plot_Nreps_array_errorband(Nreps, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPost-BL_grey_errorband")
    plot_Nreps_array_errorband(Nreps, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_Novelty-BL_grey_errorband")
    plot_Nreps_array_errorband(Nreps, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPre-BL_grey_errorband")

    # declare variable name patterns to be stored in hdf5 file
    # lists with different lengths cannot be stored in hdf5 -> split up into indiv arrays with dataset name string(index)

    # -------------------------------------- saving ----------------------------------------------------------------
    pattern = ["mean*","params*","height*", "tau*", "baseline*"]
    antipattern = ["*hist*","edges"] # specify lists with different length -> different treatment

    # create results file
    file_name_results = results_folder + file_name + "/results%s.h5"%timestr
    f_results = h5py.File(file_name_results, "a")


    f_results.create_dataset('avgwindow%d'%avgwindow, data=avgwindow)
    f_results.create_dataset('Avgwindow%d/Nreps'%avgwindow, data=Nreps)

    for key in dir():
        if fnmatch.fnmatch(key, pattern[0]):
            if not fnmatch.fnmatch(key, antipattern[0]):
                f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
            else:
                listlen = len(vars()[key])
                for i in range(0,listlen):
                    f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])
        elif fnmatch.fnmatch(key, pattern[1]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[2]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[3]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[4]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, antipattern[1]):
            listlen = len(vars()[key])
            for i in range(0,listlen):
                f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])

    f_results.close()


#################################################################
#                                                               #
#                                                               #
#              run_sequence_length                              #
#                                                               #
#                                                               #
#################################################################



def run_sequence_length_member(file_name, avgwindow = 8, timestr = "_now", RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR
    #run_folder = "/gpfs/gjor/personal/schulza/data/main/sequences/"
    # folder with analysed results from spiketime analysis in julia & where to results are stored
    #results_folder = "/gpfs/gjor/personal/schulza/results/sequences/"


    # define folder where figues should be stored
    figure_directory = results_folder + file_name + "/" + "figures_window%d/"%avgwindow
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    # read in run parameters
    file_name_run = run_folder + file_name
    # open file
    frun = h5py.File(file_name_run, "r")

    # read in stimulus parameters
#     Nimg, lenNreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength  = frun["initial"]["stimparams"].value
#     repetitions  = frun["initial"]["repetitions"].value
#     Nreps  = frun["initial"]["Nreps"].value

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
    hist_E_mem  = []
    mean_hist_E_mem = []
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
        hist_E_mem.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_I.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_nomem.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_nomemnonov.append(np.zeros((Nblocks,len(edges[seq-1]))))
        hist_E_boxcar.append(np.zeros((Nblocks,len(edges[seq-1]))))

        for bl in range(1, Nblocks + 1):
            #vars()['hist_E_all' + str(seq-1)][bl-1][:] = f["E%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E[seq-1][bl-1,:] = f["E%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_mem[seq-1][bl-1,:] = f["EMem%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_I[seq-1][bl-1,:] = f["I%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_nomem[seq-1][bl-1,:] = f["ENonMem%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            #hist_E_nomemnonov[seq-1][bl-1,:] = f["ENonMemNoNov%dmscounts" % binsize]["seq"+ str(seq) + "block"+ str(bl)].value
            hist_E_boxcar[seq-1][bl-1,:] = np.convolve(hist_E[seq-1][bl-1,:], np.ones((avgwindow,))/avgwindow, mode='same')

        # get averages over blocks
        mean_hist_E.append(np.mean(hist_E[seq-1][:,:],axis = 0))
        mean_hist_E_mem.append(np.mean(hist_E_mem[seq-1][:,:],axis = 0))
        mean_hist_I.append(np.mean(hist_I[seq-1][:,:],axis = 0))
        mean_hist_E_nomem.append(np.mean(hist_E_nomem[seq-1][:,:],axis = 0))
        mean_hist_E_nomemnonov.append(np.mean(hist_E_nomemnonov[seq-1][:,:],axis = 0))
        mean_hist_E_boxcar.append(np.mean(hist_E_boxcar[seq-1][:,:],axis = 0))

    # plotting
    color = ["midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon", "midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan", "brown", "slategrey", "darksalmon","grey", "green", "salmon"]

    idxconv = np.floor_divide(avgwindow,2)+1
    ifplotting = True
    if ifplotting:
        plot_all_averages(edges, mean_hist_E, Nimg, savehandle = "E", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
        plot_all_averages(edges, mean_hist_E_mem, Nimg, savehandle = "E_mem", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
        plot_all_averages(edges, mean_hist_I, Nimg, savehandle = "I", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
        plot_all_averages(edges, mean_hist_E_boxcar, Nimg, savehandle = "E_boxcar", ifseqlen=True,startidx = idxconv, endidx = -idxconv, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
        plot_all_averages(edges, mean_hist_E_nomem, Nimg, savehandle = "E_nomem", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
        plot_all_averages(edges, mean_hist_E_nomemnonov, Nimg, savehandle = "E_nomemnonov", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)

        plot_all_traces_and_average(edges, hist_E, mean_hist_E, Nimg, savehandle = "E", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
        plot_all_traces_and_average(edges, hist_E_mem, mean_hist_E_mem, Nimg, savehandle = "E_mem", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
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
        ifseqlen=True, avgindices = avgindices, initialparams=initial_params, bounds=fit_bounds, ifplot = True,
        startimg = startimg, idxconv = idxconv)

    #get_baseline_firing_rate
    baseline_avg, baseline, mean_baseline, std_baseline = get_baseline_firing_rate(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        ifseqlen=True, avgindices = avgindices, idxconv = idxconv)


    plot_all_averages_with_fits(edges, mean_hist_E, Nimg, params_blockavg, savehandle = "E_withfits", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=True, iflegend=False, ifyticks=False)
    plot_all_averages_with_fits(edges, mean_hist_E, Nimg, params_blockavg, savehandle = "E_boxcar_withfits", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=True, iflegend=False, ifyticks=False)

#     # fitting of post novelty transient
#     t_before_trans, params_blockavg_trans, params_err_blockavg_trans, params_trans, params_err_trans = fit_variable_repetitions_gen_arrays_postnovelty(
#         edges,hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
#         ifseqlen=True, avgindices = avgindices, initialparams=initial_params, bounds=fit_bounds, ifplot = False,
#         startimg = startimg, idxconv = idxconv)

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


#     height_trans_post_avg, height_trans_post, mean_trans_post, std_trans_post, trans_post_avgidx, trans_postidx = get_peak_height(
#         edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
#         iftransientpre = False, iftransientpost = True,
#         avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = Nimg*samples_img)

    # ---------------------------------------- plotting --------------------------------------------------------
    if ifplotting:

        # plot pre transient decay constant vs. number of repetitions
        plot_Nreps_tau(Nimg, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=False, xlabel="sequence length", xtickstepsize = 1, savename = "NimgTau")
        # plot baseline determined from fit vs. number of repetitions
        plot_Nreps_baseline(Nimg, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=False, xlabel="sequence length", xtickstepsize = 1,savename = "NimgBaseline")
        # saving and reloading for comparing instantiations

#     # plot post novelty transient decay constant vs. number of repetitions
#     plot_Nreps_tau(Nreps, params_trans, params_blockavg_trans, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=False)
#     # plot baseline determined from fit vs. number of repetitions
#     plot_Nreps_baseline(Nreps, params_trans, params_blockavg_trans, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=False)


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

    # declare variable name patterns to be stored in hdf5 file
    # lists with different lengths cannot be stored in hdf5 -> split up into indiv arrays with dataset name string(index)

    # -------------------------------------- saving ----------------------------------------------------------------
    pattern = ["mean*","params*","height*", "tau*", "baseline*"]
    antipattern = ["*hist*","edges"] # specify lists with different length -> different treatment

    # create results file
    file_name_results = results_folder + file_name + "/results%s.h5"%timestr
    f_results = h5py.File(file_name_results, "a")


    f_results.create_dataset('avgwindow%d'%avgwindow, data=avgwindow)
    f_results.create_dataset('Avgwindow%d/Nreps'%avgwindow, data=Nreps)

    for key in dir():
        if fnmatch.fnmatch(key, pattern[0]):
            if not fnmatch.fnmatch(key, antipattern[0]):
                f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
            else:
                listlen = len(vars()[key])
                for i in range(0,listlen):
                    f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])
        elif fnmatch.fnmatch(key, pattern[1]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[2]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[3]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[4]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, antipattern[1]):
            listlen = len(vars()[key])
            for i in range(0,listlen):
                f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])

    f_results.close()
    gc.collect()


def weightevolution_SSA(file_name, ifcontin,  indivassembly = True, figsize=(20,10),ncol = 1,RUN_DIR="../data/", RESULTS_DIR ="../results/"):
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


    #print(keysavgweights)#
    #print(frun["dursimavg"].keys())
    dtsaveweights = frun["params"]["dtsaveweights"].value * 10 # convert in 0.1 ms # how often are the weights stored
    modwstore = frun["params"]["modwstore"].value # convert in 0.1 ms
    minwstore = frun["params"]["minwstore"].value # convert in 0.1 ms
    if ifcontin:
        tts = range(1,avgXassemblycount+1)
    else:
        tts = range(1,minwstore+1)
        tts.extend(range(minwstore + modwstore, minwstore + modwstore + (avgXassemblycount-minwstore)*modwstore, modwstore))
# print(tts)
    # print(len(tts))
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
    #f[groups[0]].keys()
    #Nass = len(assemblymembers[0,;])f
    timevector = np.zeros(len(tts))
    timecounter = 0
    #for tt in range(1,avgXassemblycount+1):#if no minwstore
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
    plotavgweightmatrix(Xweight[:,:,-1], maxval = 14)
    save_fig(figure_directory, "Final_avgweightmatrix")

    plotavgweightmatrix(InhibXweight[:,:,-1], maxval= 255)
    save_fig(figure_directory, "Final_Inhibavgweightmatrix")

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
    fig = plt.figure(figsize=figsize)
    # for i in range(0,20):
    #     plot_popavg_mult(fig,timevector, Xweight[i,i,:], legend = "E", iflegend = False, color = color[i], ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True)
    for i in reversed(range(Nass)):
        if i in startidx:#i == startidx or i == startidx+Nimg or i == startidx+2*Nimg or i == startidx+3*Nimg or i == startidx+4*Nimg or i == startidx+5*Nimg or i == startidx+6*Nimg or i == startidx+7*Nimg or i == startidx+8*Nimg or i == startidx+9*Nimg:# or i == startidx+10*Nimg or i == startidx+11*Nimg or i == startidx+12*Nimg or i == startidx+13*Nimg or i == startidx+14*Nimg or i == startidx+15*Nimg:
            seqnum -= 1
            plot_popavg_mult(fig,timevector, InhibXweight[i,i,:], legend = "assemblies seq. %d" % (seqnum), iflegend = True, color = color[i], ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        elif i < Nimg*Nseq:
            plot_popavg_mult(fig,timevector, InhibXweight[i,i,:], legend = "E", iflegend = False, color = color[i], ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        elif i == Nimg*Nseq:
            plot_popavg_mult(fig,timevector, InhibXweight[i,i,:], legend = "assemblies novelty", iflegend = True, color = "lightgrey", ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        else:
            plot_popavg_mult(fig,timevector, InhibXweight[i,i,:], legend = "E", iflegend = False, color = "lightgrey", ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
    save_fig(figure_directory, "InhibXweightXassemblyWeightT")
    #plt.xlim([20,22])
    plt.legend().remove()

    seqnum = Nseq + 1
    fig = plt.figure(figsize=figsize)
    for i in reversed(range(Nass)):
        if i in startidx:#i == startidx or i == startidx+Nimg or i == startidx+2*Nimg or i == startidx+3*Nimg or i == startidx+4*Nimg or i == startidx+5*Nimg or i == startidx+6*Nimg or i == startidx+7*Nimg or i == startidx+8*Nimg or i == startidx+9*Nimg:# or i == startidx+10*Nimg or i == startidx+11*Nimg or i == startidx+12*Nimg or i == startidx+13*Nimg or i == startidx+14*Nimg or i == startidx+15*Nimg:
            seqnum -= 1
            plot_popavg_mult(fig,timevector, Xweight[i,i,:], legend = "assemblies seq. %d" % (seqnum), iflegend = True, color = color[i], ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        elif i < Nimg*Nseq:
            plot_popavg_mult(fig,timevector, Xweight[i,i,:], legend = "E", iflegend = False, color = color[i], ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        elif i == Nimg*Nseq:
            plot_popavg_mult(fig,timevector, Xweight[i,i,:], legend = "assemblies novelty", iflegend = True, color = "lightgrey", ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
        else:
            plot_popavg_mult(fig,timevector, Xweight[i,i,:], legend = "E", iflegend = False, color = "lightgrey", ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True, ncol = ncol)
    save_fig(figure_directory, "XassemblyWeightT")
    #plt.xlim([20,22])
    plt.legend().remove()
    save_fig(figure_directory, "XassemblyWeightT_range_20_22")
    #     plt.xlim([0,0.35])
    #     plt.ylim([2.5,6])
    save_fig(figure_directory, "Pretraining_first_weight_increase")


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

    fig = plt.figure(figsize=figsize)

    plot_popavg_mult(fig,timevector, avgweightEnov, legend = "avg. nov", iflegend = True, color = "darkorange", ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True)#, ncol = ncol)
    plot_popavg_mult(fig,timevector, avgweightEmem, legend = "avg. mem", iflegend = True, color = "darkblue", ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True)#, ncol = ncol)
    plt.fill_between(timevector, avgweightEnov-stdweightEnov, avgweightEnov+stdweightEnov,alpha=0.2, edgecolor='darkorange', facecolor='darkorange')
    plt.fill_between(timevector, avgweightEmem-stdweightEmem, avgweightEmem+stdweightEmem,alpha=0.2, edgecolor='darkblue', facecolor='darkblue')
    save_fig(figure_directory, "XassemblyWeightTWithaverages")
    fig = plt.figure(figsize=figsize)

    plot_popavg_mult(fig,timevector, InhibavgweightEnov, legend = "avg. nov", iflegend = True, color = "darkorange", ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True)#, ncol = ncol)
    plot_popavg_mult(fig,timevector, InhibavgweightEmem, legend = "avg. mem", iflegend = True, color = "darkblue", ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="w [pF]", ifioff = True)#, ncol = ncol)
    plt.fill_between(timevector, InhibavgweightEnov-InhibstdweightEnov, InhibavgweightEnov+InhibstdweightEnov,alpha=0.2, edgecolor='darkorange', facecolor='darkorange')
    plt.fill_between(timevector, InhibavgweightEmem-InhibstdweightEmem, InhibavgweightEmem+InhibstdweightEmem,alpha=0.2, edgecolor='darkblue', facecolor='darkblue')
    save_fig(figure_directory, "InhibXassemblyWeightTWithaverages")

    seqnum = Nseq + 1
    fig = plt.figure(figsize=figsize)
    for i in reversed(range(0,Nass)):
        if i in startidx:#if i == startidx or i == startidx+Nimg or i == startidx+2*Nimg or i == startidx+3*Nimg or i == startidx+4*Nimg or i == startidx+5*Nimg or i == startidx+6*Nimg or i == startidx+7*Nimg or i == startidx+8*Nimg or i == startidx+9*Nimg or i == startidx+10*Nimg or i == startidx+11*Nimg or i == startidx+12*Nimg or i == startidx+13*Nimg or i == startidx+14*Nimg or i == startidx+15*Nimg:
            seqnum -= 1
            plot_popavg_mult(fig,timevector, ItoAweight[i,:], legend = "assemblies seq. %d" % (seqnum), iflegend = True, color = color[i], ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="winhib [pF]", ifioff = True, ncol = ncol)
        elif i < Nimg*Nseq:
            plot_popavg_mult(fig,timevector, ItoAweight[i,:], legend = "E", iflegend = False, color = color[i], ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="winhib [pF]", ifioff = True)
        elif i == Nimg*Nseq:
            plot_popavg_mult(fig,timevector, ItoAweight[i,:], legend = "assemblies novelty", iflegend = True, color = "lightgrey", ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="winhib [pF]", ifioff = True, ncol = ncol)
        else:
            plot_popavg_mult(fig,timevector, ItoAweight[i,:], legend = "assemblies novelty", iflegend = False, color = "lightgrey", ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="winhib [pF]", ifioff = True, ncol = ncol)
    save_fig(figure_directory, "ItoassemblyWeightT")
    #plt.xlim([20,22])
    plt.legend().remove()
    save_fig(figure_directory, "ItoassemblyWeightT_range_20_22")
    avgweightI = np.mean(ItoAweight, axis = 0)
    avgweightImem = np.mean(ItoAweight[0:Nimg*Nseq,:], axis = 0)
    avgweightInov = np.mean(ItoAweight[Nimg*Nseq:,:], axis = 0)
    stdweightI = np.std(ItoAweight, axis = 0)
    stdweightImem = np.std(ItoAweight[0:Nimg*Nseq,:], axis = 0)
    stdweightInov = np.std(ItoAweight[Nimg*Nseq:,:], axis = 0)

    fig = plt.figure(figsize=figsize)
    #plot_popavg_mult(fig,timevector, avgweightI, legend = "avg. all", iflegend = True, color = "black", ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="winhib [pF]", ifioff = True)#, ncol = ncol)
    plot_popavg_mult(fig,timevector, avgweightInov, legend = "avg. nov", iflegend = True, color = "darkorange", ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="winhib [pF]", ifioff = True)#,, ncol = ncol)
    plot_popavg_mult(fig,timevector, avgweightImem, legend = "avg. mem", iflegend = True, color = "darkblue", ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]", ylabel ="winhib [pF]", ifioff = True)#,, ncol = ncol)
    #plt.fill_between(timevector, avgweightI-stdweightI, avgweightI+stdweightI,alpha=0.2, edgecolor='black', facecolor='black')
    plt.fill_between(timevector, avgweightInov-stdweightInov, avgweightInov+stdweightInov,alpha=0.2, edgecolor='darkorange', facecolor='darkorange')
    plt.fill_between(timevector, avgweightImem-stdweightImem, avgweightImem+stdweightImem,alpha=0.2, edgecolor='darkblue', facecolor='darkblue')
    save_fig(figure_directory, "ItoassemblyWeightTWithaverages")



    seqnum = Nseq + 1
    fig = plt.figure(figsize=figsize)
    for i in reversed(range(0,Nass)):
        if i in startidx:#if i == startidx or i == startidx+Nimg or i == startidx+2*Nimg or i == startidx+3*Nimg or i == startidx+4*Nimg or i == startidx+5*Nimg or i == startidx+6*Nimg or i == startidx+7*Nimg or i == startidx+8*Nimg or i == startidx+9*Nimg or i == startidx+10*Nimg or i == startidx+11*Nimg or i == startidx+12*Nimg or i == startidx+13*Nimg or i == startidx+14*Nimg or i == startidx+15*Nimg:
            seqnum -= 1
            plot_popavg_mult(fig,timevector, Etononmensweight[i,:], legend = "assemblies seq. %d" % (seqnum),
                             iflegend = True, color = color[i], ifcolor = True, lw = 1,fontsize = 20,
                             xlabel = "time [min]", ylabel ="w Etononmens [pF]", ifioff = True, ncol = ncol)
        elif i < Nimg*Nseq:
            plot_popavg_mult(fig,timevector, Etononmensweight[i,:], legend = "E", iflegend = False,
                             color = color[i], ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]",
                             ylabel ="w Etononmens [pF]", ifioff = True, ncol = ncol)
        elif i == Nimg*Nseq:
            plot_popavg_mult(fig,timevector, Etononmensweight[i,:], legend = "assemblies novelty",
                             iflegend = True, color = "lightgrey", ifcolor = True, lw = 1,fontsize = 20,
                             xlabel = "time [min]", ylabel ="w Etononmens [pF]", ifioff = True, ncol = ncol)
        else:
            plot_popavg_mult(fig,timevector, Etononmensweight[i,:], legend = "assemblies novelty",
                             iflegend = False, color = "lightgrey", ifcolor = True, lw = 1,fontsize = 20,
                             xlabel = "time [min]", ylabel ="w Etononmens [pF]", ifioff = True, ncol = ncol)
    save_fig(figure_directory, "EtononmensweightT")

    seqnum = Nseq + 1
    fig = plt.figure(figsize=figsize)
    for i in reversed(range(0,Nass)):
        if i in startidx:#if i == startidx or i == startidx+Nimg or i == startidx+2*Nimg or i == startidx+3*Nimg or i == startidx+4*Nimg or i == startidx+5*Nimg or i == startidx+6*Nimg or i == startidx+7*Nimg or i == startidx+8*Nimg or i == startidx+9*Nimg or i == startidx+10*Nimg or i == startidx+11*Nimg or i == startidx+12*Nimg or i == startidx+13*Nimg or i == startidx+14*Nimg or i == startidx+15*Nimg:
            seqnum -= 1
            plot_popavg_mult(fig,timevector, nonmenstoEweight[i,:], legend = "assemblies seq. %d" % (seqnum),
                             iflegend = True, color = color[i], ifcolor = True, lw = 1,fontsize = 20,
                             xlabel = "time [min]", ylabel ="w nonmenstoE [pF]", ifioff = True, ncol = ncol)
        elif i < Nimg*Nseq:
            plot_popavg_mult(fig,timevector, nonmenstoEweight[i,:], legend = "E", iflegend = False,
                             color = color[i], ifcolor = True, lw = 1,fontsize = 20, xlabel = "time [min]",
                             ylabel ="w nonmenstoE [pF]", ifioff = True, ncol = ncol)
        elif i == Nimg*Nseq:
            plot_popavg_mult(fig,timevector, nonmenstoEweight[i,:], legend = "assemblies novelty",
                             iflegend = True, color = "lightgrey", ifcolor = True, lw = 1,
                             fontsize = 20, xlabel = "time [min]", ylabel ="w nonmenstoE [pF]", ifioff = True, ncol = ncol)
        else:
            plot_popavg_mult(fig,timevector, nonmenstoEweight[i,:], legend = "assemblies novelty",
                             iflegend = False, color = "lightgrey", ifcolor = True, lw = 1,fontsize = 20,
                             xlabel = "time [min]", ylabel ="w nonmenstoE [pF]", ifioff = True, ncol = ncol)
    save_fig(figure_directory, "nonmenstoEweightT")
    #plt.xlim([20,22])

    axiswidth = 1
    if indivassembly:
    # plot trace of individual assembly on background of all other assemblies to highlight one weight evolution with respect to all others
        specialassembly = 0
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        for i in reversed(range(Nass)):
            plt.plot(timevector,Xweight[i,i,:], label = str(i), color = "lightgrey",lw = 1)
        plt.plot(timevector,Xweight[specialassembly,specialassembly,:], label = str(specialassembly), color = "green",lw = 1)
        plt.xlabel("time [min]",fontsize = 24)
        plt.ylabel("w [pF]",fontsize = 24)
        plt.xticks(fontsize = 24)
        plt.yticks(fontsize = 24)
        save_fig(figure_directory, "IndividualHighlightAss%dEweights" %(specialassembly))

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        for i in reversed(range(Nass)):
            plt.plot(timevector,ItoAweight[i,:], label = str(i), color = "lightgrey",lw = 1)
        plt.plot(timevector,ItoAweight[specialassembly,:], label = str(specialassembly), color = "green",lw = 1)
        plt.xlabel("time [min]",fontsize = 24)
        plt.ylabel("winhib [pF]",fontsize = 24)
        plt.xticks(fontsize = 24)
        plt.yticks(fontsize = 24)
        save_fig(figure_directory, "IndividualHighlightAss%dIweights" %(specialassembly))
        plt.show()
# =============================== inhib ===============================================================
        specialassembly = 0
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(axiswidth)
        for axis in ['top','right']:
            ax.spines[axis].set_linewidth(0)
        ax.xaxis.set_tick_params(width=axiswidth)
        ax.yaxis.set_tick_params(width=axiswidth)
        for i in reversed(range(Nass)):
            plt.plot(timevector,InhibXweight[i,i,:], label = str(i), color = "lightgrey",lw = 1)
        plt.plot(timevector,InhibXweight[specialassembly,specialassembly,:], label = str(specialassembly), color = "green",lw = 1)
        plt.xlabel("time [min]",fontsize = 24)
        plt.ylabel("w [pF]",fontsize = 24)
        plt.xticks(fontsize = 24)
        plt.yticks(fontsize = 24)
        save_fig(figure_directory, "InhibXweightIndividualHighlightAss%dEweights" %(specialassembly))


    plt.figure(figsize=figsize)
    plt.plot(timevector,Itoneuron1)
    plt.xlabel('time [min]',fontsize = 24)
    plt.ylabel('winhib individual [pF]',fontsize = 24)
    plt.xticks(fontsize = 24)
    plt.yticks(fontsize = 24)
    plt.tight_layout()

    save_fig(figure_directory, "IndividuaInhibWeightsToNeuron1")

    plt.figure(figsize=figsize)
    plt.plot(timevector,Itoneuron2)
    plt.xlabel('time [min]',fontsize = 24)
    plt.ylabel('winhib individual [pF]',fontsize = 24)
    plt.xticks(fontsize = 24)
    plt.yticks(fontsize = 24)
    plt.tight_layout()
    save_fig(figure_directory, "IndividuaInhibWeightsToNeuron1")

    plt.figure(figsize=figsize)
    for i in range(len(seqnumber)):
        #print(stimulus[1,int(idxblockonset[i]-1)]/60000)
        #x = np.linspace(stimulus[1,int(idxblockonset[i]-1)],stimulus[1,int(idxblockonset[i]-1)]+lengthblock, 2)
        #y = np.ones(len(x))
        plt.plot(stimulus[1,int(idxblockonset[i]-1)]/60000,1, ",", color = colormain[int(seqnumber[i]-1)])
    plt.xlabel('time [min]',fontsize = 24)
    plt.ylabel('winhib individual [pF]',fontsize = 24)
    plt.xticks(fontsize = 24)
    plt.yticks(fontsize = 24)
    plt.tight_layout()
    save_fig(figure_directory, "Sequence_Visualisation")
    return Xweight, ItoAweight, timevector, avgweightEmem, avgweightImem, avgweightEnov, avgweightImem, Itoneuron1, Itoneuron2, InhibXweight, seqnumber, stimulus, colormain, idxblockonset


def run_single_neuron_eval_SSA(file_name, binwidth = 50, avgwindow = 5, timestr = "_now", RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR
    # input: file to be evaluated
    # smoothing parameters rebinning binwidth in ms, smoothing window N N*binwidth in ms

    #output: figures of whole plane after moving avg, zscore, sparseness analysis
    #    file_name = "dur1.624e6mslenstim300lenpause0Nreps20strength8wadaptfalseiSTDPtrueTime2019-04-09-21-45-25repeatedsequences.h5"

    #def spiketimeanalysis(file_name, ifcontin,  indivassembly = True):
    #run_folder = "/gpfs/gjor/personal/schulza/data/main/sequences/"
    # folder with analysed results from spiketime analysis in julia & where to results are stored
    #results_folder = "/home/schulza/Documents/results/main/sequences/"
    #results_folder = "/gpfs/gjor/personal/schulza/results/sequences/"

    #run_folder = "../data/"
    #results_folder = "../results/"

    file_name_results = results_folder + file_name + "/results.h5"
    #f_results = h5py.File(file_name_results, "w")

    # define folder where figues should be stored
    figure_directory = results_folder + file_name + "/" + "singlefigures%d/"% (avgwindow*binwidth)
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    # read in run parameters
    file_name_run = run_folder + file_name
    # open file
    frun = h5py.File(file_name_run, "r")

    # read in stimulus parameters
    Nimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength  = frun["initial"]["stimparams"].value
    Ni = frun["params"]["Ni"].value
    Ne = frun["params"]["Ne"].value
    Ncells = Ni + Ne
    seqnumber  = frun["initial"]["seqnumber"].value
    assemblymembers = frun["initial"]["assemblymembers"].value.transpose()
    color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange", "midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange"]
    Nblocks = min([Nblocks, 10])
    # get indices of all assembly members and novelty members as well as untargeted neuron indices
    Nass = Nimg*Nseq
    members = assemblymembers[0:Nass,:]
    novelty = assemblymembers[Nass:,:]

    membersidx = np.unique(members[members>0])
    noveltyidx = np.unique(novelty[novelty>0])

    untargetedidx = np.ones(Ncells, dtype=bool)
    untargetedidx[membersidx-1] = False
    untargetedidx[noveltyidx-1] = False
    untargetedidx[Ne:] = False
    inhibitoryidx = np.linspace(Ne,Ncells-1,Ncells-Ne, dtype=int)

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
    # bin width
    lenblocktt  = f["params"]["lenblocktt"].value
    blockbegintt  = f["params"]["blockbegintt"].value
    dt = 0.1
    nbins = int(np.round(lenblocktt*dt/binwidth+1))
    bins = np.linspace(0,lenblocktt, nbins)
    bin_counts = np.zeros((Ncells,nbins-1))
    conv = np.zeros((Ncells,nbins-1), dtype=float)

    timevector = np.linspace((binwidth/2)/1000.,lenblocktt*dt/1000. - (binwidth/2)/1000.,nbins-1)

    all_bincounts = np.zeros((Nseq,Nblocks,Ncells, nbins-1))
    all_bincounts_conv = np.zeros((Nseq,Nblocks,Ncells, nbins-1))
    all_zscore_firing = np.zeros((Nseq,Nblocks,Ncells, nbins-1))

    noveltyonset = (Nimg*(Nreps-1)*(lenstim+lenpause)-(lenstim+lenpause))/1000. #convert to seconds
    idxstart = int((noveltyonset - 2)*10)
    idxend = int((noveltyonset + 3)*10)
    winlength = avgwindow*binwidth

    idxconv = np.floor_divide(avgwindow,2)+1 # account for convolution in mode same ignore first len(kernel)/2 samples
    for seq in range(1,Nseq + 1):
        for bl in range(1,Nblocks + 1):
            # read in spiketimes
            spiketimes = f["spiketimeblocks"]["seq"+ str(seq) + "block"+ str(bl)].value.transpose()

            # rebin spiketimes in binsize of 100 ms
            for cc in range(0,Ncells):
                 all_bincounts[seq-1,bl-1,cc,:], bin_edges = np.histogram(spiketimes[cc,:], bins=bins)#lenblocktt/1000)
            spiketimes = 0
            gc.collect()
            # get firing rates by dividing by the length of the binwidth in seconds
            bin_edges = bin_edges*dt/1000 # seconds
            all_bincounts[seq-1,bl-1,:,:] = all_bincounts[seq-1,bl-1,:,:]*(1000/binwidth) # divide by binwidth in seconds to get rate
            # apply moving average across 8 bins
            for cc in range(0,Ncells):
                all_bincounts_conv[seq-1,bl-1,cc,1:] = np.convolve(all_bincounts[seq-1,bl-1,cc,1:], np.ones((avgwindow,))/avgwindow, mode='same')# pay attnetion to first and last 800 ms altered due to zero padding

            #---------------------------- zscore -----------------------------------------------------
            # evaluate std and avg of all neurons
            # eval zscore and plot for novelty region
            # order according to value

            # store all variables
    #         f_results = h5py.File(file_name_results, "w")
    #         f_results.create_dataset('single_neurons/timevectorSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=timevector)
    #         f_results.create_dataset('single_neurons/firing_rates_smoothedSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=conv)
    #         f_results.create_dataset('single_neurons/zscore_firingSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=zscore_firing)
    #         f_results.create_dataset('single_neurons/binwidthSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=binwidth)
    #         f_results.create_dataset('single_neurons/bin_edgesSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=bin_edges)
    #         f_results.create_dataset('single_neurons/bin_countsSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=bin_counts)
    #         f_results.create_dataset('single_neurons/mean_firingSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=mean_firing)
    #         f_results.create_dataset('single_neurons/std_firingSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=std_firing)
    #         f_results.create_dataset('single_neurons/noveltyonsetSeq%dBlock%dBinsize%d' % (seq,bl,avgwindow*binwidth), data=noveltyonset)
    #         f_results.close()
    # f_results = h5py.File(file_name_results, "w")
    # f_results.create_dataset('single_neurons/all_bincountsBinsize%d' % (avgwindow*binwidth), data=all_bincounts)
    # f_results.create_dataset('single_neurons/all_bincounts_convBinsize%d' % (avgwindow*binwidth), data=all_bincounts_conv)
    # f_results.close()
                # -------------------------- evaluate denseness ------------------------------------


    mean_firing = np.mean(all_bincounts_conv[:,:,:,idxconv:-idxconv], axis=3)
    std_firing = np.std(all_bincounts_conv[:,:,:,idxconv:-idxconv], axis=3)

    # z-score for whole dataset
    zscore_firing = stats.zscore(all_bincounts_conv[:,:,:,idxconv:-idxconv], axis=3) # get zscore for all neurons fr all blcosk and all sequneces ignoring the invalid convolution part
    zscore_firing[np.isnan(zscore_firing)] = 0
    # -------------------------- evaluate denseness ------------------------------------

    count_zscore = np.copy(zscore_firing)
    count_zscore[count_zscore < 0] = -1
    count_zscore[count_zscore > 0] = 1


    # take average of firing rates across blocks
    avg_bincounts_conv_blocks = np.mean(all_bincounts_conv[:,:,:,idxconv:-idxconv],axis = 1)
    #determine zscore of this "double" average
    zscore_over_all_blocks = stats.zscore(avg_bincounts_conv_blocks, axis = 2) # zscore over 240 time
    zscore_over_all_blocks[np.isnan(zscore_over_all_blocks)] = 0
    count_zscore_avg_blocks = np.copy(zscore_over_all_blocks)
    count_zscore_avg_blocks[zscore_over_all_blocks < 0] = -1
    count_zscore_avg_blocks[zscore_over_all_blocks > 0] = 1



    # take average of firing rates across both blocks and sequences
    avg_bincounts_conv_blocks_seq = np.mean(np.mean(all_bincounts_conv[:,:,:,idxconv:-idxconv],axis = 0),axis = 0)
    #determine zscore of this "double" average
    zscore_over_all_blocks_seq = stats.zscore(avg_bincounts_conv_blocks_seq, axis = 1)
    zscore_over_all_blocks_seq[np.isnan(zscore_over_all_blocks_seq)] = 0

    # count of this "double" average
    count_zscore_avg_blocks_seq = np.copy(zscore_over_all_blocks_seq)
    count_zscore_avg_blocks_seq[zscore_over_all_blocks_seq < 0] = -1
    count_zscore_avg_blocks_seq[zscore_over_all_blocks_seq > 0] = 1



    # # -------------------------- evaluate denseness ------------------------------------
    # # --------------- zscore mean ------------------
    zscore_meanE = np.mean(zscore_firing[:,:,0:Ne,:], axis=2)
    zscore_meanI = np.mean(zscore_firing[:,:,Ne:,:], axis=2)
    zscore_stdE = np.std(zscore_firing[:,:,0:Ne,:], axis=2)
    zscore_stdI = np.std(zscore_firing[:,:,Ne:,:], axis=2)


    # -------------------------- evaluate denseness ------------------------------------
    # determine 4 relevant x positions in the array
    # not nice but works for now
    maxtemp = max(timevector)
    if maxtemp > 20:
        xpos = np.array([50,100,150,200])*100/binwidth-idxconv
        xlab = ["5","10","15","20"]
    elif maxtemp > 15:
        xpos = np.array([50,100,150])*100/binwidth-idxconv
        xlab = ["5","10","15"]
    elif maxtemp > 10:
        xpos = np.array([50,100])*100/binwidth-idxconv
        xlab = ["5","10"]
    elif maxtemp > 7.5:
        xpos = np.array([25,50,75])*100/binwidth-idxconv
        xlab = ["2.5","5","7.5"]
    elif maxtemp > 5:
        xpos = np.array([25,50])*100/binwidth-idxconv
        xlab = ["2.5","5"]
    else:
        xpos = np.array([0,10])*100/binwidth-idxconv
        xlab = ["0","1"]
        # --------------- plotting ------------------
    #plt.ioff()

    ifplotting = True
    if ifplotting:
        for seq in range(1,Nseq+1):
            # plot the zscore counts for each individual sequence  average out Nblocks
            plotzscorecounts(timevector[idxconv:-idxconv], count_zscore_avg_blocks[seq-1], seq, 0, ifBlockAvg=True, figure_directory=figure_directory, noveltyonset=noveltyonset, ifExcitatory = True, Nseq = Nseq)#
            plotzscorecounts(timevector[idxconv:-idxconv], count_zscore_avg_blocks[seq-1], seq, 0, ifBlockAvg=True, figure_directory=figure_directory, noveltyonset=noveltyonset, ifExcitatory = False, Nseq = Nseq)#
            for bl in range(1,Nblocks+1):
                # fig = plt.figure(figsize=(15,15))
                # plot_mean_with_errorband_mult(fig,timevector[idxconv:-idxconv], zscore_meanE[seq-1,bl-1,:], zscore_stdE[seq-1,bl-1,:],  legend="E", color = "darkblue", iflegend=True, ifioff=True, Nseq = Nseq)#
                # plot_mean_with_errorband_mult(fig,timevector[idxconv:-idxconv], zscore_meanI[seq-1,bl-1,:], zscore_stdI[seq-1,bl-1,:],  legend="I", color = "darkred", iflegend=True, ifioff=True, Nseq = Nseq)#
                # save_fig(figure_directory, "IdxMeanZscoreIandESeq%dBlock%d" % (seq,bl))
                # plotzscorecounts(timevector[idxconv:-idxconv],count_zscore[seq-1,bl-1,:,:], seq, bl, figure_directory=figure_directory, noveltyonset=noveltyonset,ifExcitatory = True, Nseq = Nseq)#
                # plotzscorecounts(timevector[idxconv:-idxconv],count_zscore[seq-1,bl-1,:,:], seq, bl, figure_directory=figure_directory, noveltyonset=noveltyonset,ifExcitatory = False, Nseq = Nseq)#
                plotrateplanewithavg(timevector[idxconv:-idxconv], all_bincounts_conv[:,:,0:100,idxconv:-idxconv], seq, bl, x_positions = xpos,x_labels = xlab,cmap = "Greys", figure_directory=figure_directory, idxstart = 0, ifExcitatory = True,savetitle = "firing_rates_first100Neurons", Nseq = Nseq,ififnorm = False,midpoint=3, ylimbar = 350, ifylimbar = True)#
                #plotrateplanewithavg(timevector[idxconv:-idxconv],all_bincounts_conv[:,:,Ne:(Ne+100),idxconv:-idxconv], seq, bl, x_positions = xpos,x_labels = xlab,figure_directory=figure_directory, idxstart = 0, ifExcitatory = False,savetitle = "firing_rates_first100Neurons", Nseq = Nseq)#
                # plotrateplane(timevector[idxconv:-idxconv],all_zscore_firing, seq, bl, x_positions = xpos,x_labels = xlab, cmap = "coolwarm",figure_directory=figure_directory, idxstart = 0, ifExcitatory = True, savetitle = "zscore", cbarlabel="z-score", ififnorm = True, Nseq = Nseq)#
                # plotrateplane(timevector[idxconv:-idxconv],all_zscore_firing, seq, bl, x_positions = xpos,x_labels = xlab, cmap = "coolwarm",figure_directory=figure_directory, idxstart = 0, ifExcitatory = False, savetitle = "zscore", cbarlabel="z-score", ififnorm = True, Nseq = Nseq)#

        plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(zscore_firing,axis = 1), x_positions = xpos,x_labels = xlab,cmap = "coolwarm",figure_directory=figure_directory, idxstart = 0, ifExcitatory = False, savetitle = "zscore", cbarlabel="z-score", ififnorm = True, Nseq = Nseq)#
        plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(zscore_firing,axis = 1), x_positions = xpos,x_labels = xlab,cmap = "coolwarm",figure_directory=figure_directory, idxstart = 0, ifExcitatory = True, savetitle = "zscore", cbarlabel="z-score", ififnorm = True, Nseq = Nseq)#


        plotzscorecounts(timevector[idxconv:-idxconv],count_zscore_avg_blocks_seq, 0, 0, ifAvg=True, figure_directory=figure_directory, noveltyonset=noveltyonset, ifExcitatory = True, Nseq = Nseq)#
        plotzscorecounts(timevector[idxconv:-idxconv],count_zscore_avg_blocks_seq, 0, 0, ifAvg=True, figure_directory=figure_directory, noveltyonset=noveltyonset, ifExcitatory = False, Nseq = Nseq)#

        plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,:,idxconv:-idxconv],axis = 1), cmap = "Greys",idxstart = 0, x_positions = xpos,x_labels = xlab, figure_directory = figure_directory, ifExcitatory = True, Nseq = Nseq,ififnorm = False)#
        plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,:,idxconv:-idxconv],axis = 1), cmap = "Greys",idxstart = 0, x_positions = xpos,x_labels = xlab, figure_directory = figure_directory, ifExcitatory = False, Nseq = Nseq,ififnorm = False)#
        plotaveragerateplanewithavg(timevector[idxconv:-idxconv],avg_bincounts_conv_blocks, idxstart = 0, x_positions = xpos,cmap = "Greys",x_labels = xlab,figure_directory = figure_directory, ifExcitatory = False, Nseq = Nseq,ififnorm = False, savetitle = "firing_rates_allNeurons")#
        plotaveragerateplanewithavg(timevector[idxconv:-idxconv],avg_bincounts_conv_blocks, idxstart = 0, x_positions = xpos,cmap = "Greys",x_labels = xlab,figure_directory = figure_directory, ifExcitatory = True, Nseq = Nseq,ififnorm = False,savetitle = "firing_rates_allNeurons")#
        #%plotaveragerateplanewithavg(timevector[idxconv:-idxconv],avg_bincounts_conv_blocks, idxstart = 0, x_positions = xpos,cmap = "Greys",x_labels = xlab,figure_directory = figure_directory, ifExcitatory = False, Nseq = Nseq,ififnorm = False, savetitle = "firing_rates_allNeurons")#
        plotaveragerateplanewithavg(timevector[idxconv:-idxconv],avg_bincounts_conv_blocks[:,0:100,idxconv:-idxconv], idxstart = 0, x_positions = xpos,cmap = "Greys",x_labels = xlab,figure_directory = figure_directory, ifExcitatory = True, Nseq = Nseq,ififnorm = False,savetitle = "firing_rates_first100Neurons_withavg")#
        plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,0:100,idxconv:-idxconv],axis = 1), cmap = "Greys",idxstart = 0, x_positions = xpos,x_labels = xlab, figure_directory = figure_directory, ifExcitatory = True, savetitle = "firing_rates_first100Neurons", Nseq = Nseq,ififnorm = False)#, cmap = "parula")
# plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,:,idxconv:-idxconv],axis = 1), cmap = "bone_r",idxstart = 0, x_positions = xpos,x_labels = xlab, figure_directory = figure_directory, ifExcitatory = True, Nseq = Nseq)#
# plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,:,idxconv:-idxconv],axis = 1), cmap = "bone_r",idxstart = 0, x_positions = xpos,x_labels = xlab, figure_directory = figure_directory, ifExcitatory = False, Nseq = Nseq)#
# plotaveragerateplanewithavg(timevector[idxconv:-idxconv],avg_bincounts_conv_blocks, idxstart = 0, x_positions = xpos,cmap = "bone_r",x_labels = xlab,figure_directory = figure_directory, ifExcitatory = False, Nseq = Nseq)#
# plotaveragerateplanewithavg(timevector[idxconv:-idxconv],avg_bincounts_conv_blocks, idxstart = 0, x_positions = xpos,cmap = "bone_r",x_labels = xlab,figure_directory = figure_directory, ifExcitatory = True, Nseq = Nseq)#
# plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,0:100,idxconv:-idxconv],axis = 1), cmap = "bone_r",idxstart = 0, x_positions = xpos,x_labels = xlab, figure_directory = figure_directory, ifExcitatory = True, savetitle = "firing_rates_first100Neurons", Nseq = Nseq)#, cmap = "parula")

        #plotaveragerateplane(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,Ne:(Ne+100),idxconv:-idxconv],axis = 1), cmap = "bone_r", idxstart = 0, x_positions = xpos,x_labels = xlab, figure_directory = figure_directory, ifExcitatory = False, savetitle = "firing_rates_first100Neurons", Nseq = Nseq)##, cmap = "parula")
        # plotaveragerateplanewithavg(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,0:100,idxconv:-idxconv],axis = 1), cmap = "bone_r", idxstart = 0, x_positions = xpos,x_labels = xlab,figure_directory = figure_directory, ifExcitatory = False,savetitle = "firing_rates_first100Neurons_with_avg", Nseq = Nseq)#
        # plotaveragerateplanewithavg(timevector[idxconv:-idxconv],np.mean(all_bincounts_conv[:,:,Ne:(Ne+100),idxconv:-idxconv],axis = 1), cmap = "bone_r", idxstart = 0, x_positions = xpos,x_labels = xlab,figure_directory = figure_directory, ifExcitatory = True,savetitle = "firing_rates_first100Neurons_with_avg", Nseq = Nseq)#
# plotaveragerateplanewithavg(timevector[idxconv:-idxconv],avg_bincounts_conv_blocks[0:100,idxconv:-idxconv], cmap = "bone_r", idxstart = 0, x_positions = xpos,x_labels = xlab,figure_directory = figure_directory, ifExcitatory = False,savetitle = "firing_rates_first100Neurons_with_avg", Nseq = Nseq)#
# plotaveragerateplanewithavg(timevector[idxconv:-idxconv],avg_bincounts_conv_blocks[0:100,idxconv:-idxconv], cmap = "bone_r", idxstart = 0, x_positions = xpos,x_labels = xlab,figure_directory = figure_directory, ifExcitatory = True,savetitle = "firing_rates_first100Neurons_with_avg", Nseq = Nseq)#

    ifhistograms = True
    colorseq = ["midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan","midnightblue","darkred","darkgreen","darkmagenta","darkorange","darkcyan"]
    colorblocks = ["lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan","lightskyblue","sienna","chartreuse","darksalmon","orange","darkolivegreen","indigo","lightsteelblue","thistle","lightcoral","tan"]
    if Nimg == 4:
        color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange", "black", "silver","dimgrey","gainsboro", "fuchsia", "orchid","plum", "mediumvioletred", "lightseagreen", "lightcyan", "darkslategray", "paleturquoise", "goldenrod","gold", "wheat","darkgoldenrod", "forestgreen", "aquamarine", "palegreen", "lime", ]
    elif Nimg == 3:
        color = ["midnightblue","lightskyblue","royalblue","darkred","darksalmon", "saddlebrown","darkgreen","greenyellow","darkolivegreen","darkmagenta","thistle","indigo","darkorange","tan","sienna", "black", "silver","dimgrey", "fuchsia", "orchid","plum",  "lightseagreen", "lightcyan", "darkslategray",  "goldenrod","gold", "wheat","forestgreen", "aquamarine", "palegreen"]
    elif Nimg == 5:
        color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","indigo","darkred","darksalmon", "saddlebrown","lightcoral","rosybrown","darkgreen","greenyellow","darkolivegreen","chartreuse","turquoise","darkmagenta","thistle","indigo","mediumslateblue","blueviolet","darkorange","tan","sienna","orange", "brown", "black", "silver","dimgrey","gainsboro", "grey","fuchsia", "orchid","plum", "mediumvioletred","purple", "lightseagreen", "lightcyan", "darkslategray", "paleturquoise","teal", "goldenrod","gold", "wheat","darkgoldenrod", "darkkhaki","forestgreen", "aquamarine", "palegreen", "lime", "darkseagreen"]
    elif Nimg == 1:
        color = ["midnightblue","saddlebrown","darkorange"]
    else:
        color = ["midnightblue","saddlebrown","darkorange"]
    colormain = np.copy(color[0:-1:Nimg])
    #color = ["midnightblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange"]
    ifhistograms = False
    if ifhistograms:
        avg_bincounts_conv_time = np.mean(all_bincounts_conv[:,:,:,idxconv:-idxconv],axis = 3)
        print(Nblocks)
        # plot histograms of the average firing rates (average over whole block length)
        # make individual histograms per sequence showing all distributions per blocks
        # make individual histograms per block showing all distributions for all sequences
        for bl in range(1,Nblocks+1):#[1,10,19]:#range(1,Nblocks+1):
            plot_histograms_seq_cut(avg_bincounts_conv_time,bl-1,membersidx-1,colorseq=colorseq,bins = np.linspace(0,70,51),alpha = 0.6, Nblocks =Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            save_fig(figure_directory, "HistogramMembersBlock%d" % bl)

        for seq in range(1,Nseq+1):
            plot_histograms_block_cut(avg_bincounts_conv_time,seq-1,membersidx-1,colorseq=colorblocks,bins = np.linspace(0,70,51),alpha = 0.6, Nblocks =Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            save_fig(figure_directory, "HistogramMembersSeq%d" % seq)

        for bl in range(1,Nblocks+1):#[1,10,19]:#range(1,Nblocks+1):
            plot_histograms_seq_cut(avg_bincounts_conv_time,bl-1,noveltyidx-1,cutlow=400,cuthigh=800,bins = np.linspace(0,70,51),colorseq=colorseq,alpha = 0.6, Nblocks =Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            save_fig(figure_directory, "HistogramNoveltyBlock%d" % bl)

        for seq in range(1,Nseq+1):
            plot_histograms_block_cut(avg_bincounts_conv_time,seq-1,noveltyidx-1,cutlow=400,cuthigh=800,bins = np.linspace(0,70,51),colorseq=colorblocks,alpha = 0.6, Nblocks =Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            save_fig(figure_directory, "HistogramNoveltySeq%d" % seq)

        for bl in range(1,Nblocks+1):#[1,10,19]:#range(1,Nblocks+1):
            plot_histograms_seq(avg_bincounts_conv_time,bl-1,untargetedidx,colorseq=colorseq,bins = np.linspace(0,30,31),alpha = 0.6, Nblocks =Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            save_fig(figure_directory, "HistogramUntargetedBlock%d" % bl)

        for seq in range(1,Nseq+1):
            plot_histograms_block(avg_bincounts_conv_time,seq-1,untargetedidx,colorseq=colorblocks,bins = np.linspace(0,30,31),alpha = 0.6, Nblocks = Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            save_fig(figure_directory, "HistogramUntargetedSeq%d" % seq)

        for bl in range(1,Nblocks+1):#[1,10,19]:#range(1,Nblocks+1):
            plot_histograms_seq(avg_bincounts_conv_time,bl-1,inhibitoryidx,colorseq=colorseq,bins = np.linspace(0,45,91),alpha = 0.6, Nblocks = Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            save_fig(figure_directory, "HistogramInhibitoryBlock%d" % bl)

        for seq in range(1,Nseq+1):
            plot_histograms_block(avg_bincounts_conv_time,seq-1,inhibitoryidx,bins = np.linspace(0,45,91),colorseq=colorblocks,alpha = 0.6, Nblocks = Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=4)
            save_fig(figure_directory, "HistogramInhibitorySeq%d" % seq)

    ifpooledresponses = False
    if ifpooledresponses:

        # replicate figure 6D sustained activity vs. cell rank
        starttime = 10 # sec
        endtime = 15.5 # sec
        startidx = np.argmax(timevector>starttime)
        endidx = np.argmax(timevector>endtime)
        starttimeT = 0.5 # sec
        endtimeT = 3.5 # sec
        startidxT = np.argmax(timevector>starttimeT)
        endidxT = np.argmax(timevector>endtimeT)
        starttimeN = 16.7 # sec
        endtimeN = 17.2 # sec
        startidxN = np.argmax(timevector>starttimeN)
        endidxN = np.argmax(timevector>endtimeN)
        print(startidxN)
        print(endidxN)
        endidxN = min(endidxN, np.size(all_bincounts_conv,axis = 3))
        #get the average firing rates for each sequence individually
        avg_rates_time_seqE = np.mean(np.mean(all_bincounts_conv[:,:,:Ne,startidx:endidx],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(np.mean(all_bincounts_conv[:,:,Ne:,startidx:endidx],axis = 3),axis=1) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks
        # plot pooled histograms
        plot_histogram_cut(avg_rates_time_seq_pooledE, cutlow=1500, cuthigh=1600,ifExcitatory=True, alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledSustainedResponsesE")
        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledSustainedResponsesI")

        # get ranked sustained repsonses
        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = True, figsize=(5,4),xticks = [0,5000,10000,15000,20000])
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedStustainedResponsesE")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedStustainedResponsesI")

        # get the average firing rates for each sequence individually for whole time
        avg_rates_time_seqE = np.mean(np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(np.mean(all_bincounts_conv[:,:,Ne:,idxconv:-idxconv],axis = 3),axis=1) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks

        plot_histogram_cut(avg_rates_time_seq_pooledE, cutlow=1500, cuthigh=1600,ifExcitatory=True, alpha=1, Nblocks = Nblocks,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramAllSustainedResponsesE")

        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramAllSustainedResponsesI")

        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = True, figsize=(5,4),xticks = [0,5000,10000,15000,20000])
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedAllResponsesE")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedAllResponsesI")

        # Transient firing rates

        # get the average firing rates for each sequence individually
        avg_rates_time_seqE = np.mean(np.mean(all_bincounts_conv[:,:,:Ne,startidxT:endidxT],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(np.mean(all_bincounts_conv[:,:,Ne:,startidxT:endidxT],axis = 3),axis=1) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks
        # plot pooled histograms
        plot_histogram_cut(avg_rates_time_seq_pooledE, cutlow=1000, cuthigh=1100,ifExcitatory=True, alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledTransientResponsesE")
        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledTransientResponsesI")

        # get ranked sustained repsonses
        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = True, xticks = [0,5000,10000,15000,20000],figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedTransientResponsesE")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedTransientResponsesI")

        # Novelty firing rates
        # get the average firing rates for each sequence individually
        avg_rates_time_seqE = np.mean(np.mean(all_bincounts_conv[:,:,:Ne,startidxN:endidxN],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(np.mean(all_bincounts_conv[:,:,Ne:,startidxN:endidxN],axis = 3),axis=1) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks
        # plot pooled histograms
        plot_histogram_cut(avg_rates_time_seq_pooledE, cutlow=3000, cuthigh=3500,ifExcitatory=True, alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledNoveltyResponsesE")
        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledNoveltyResponsesI")

        # get ranked sustained repsonses
        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = True, xticks = [0,5000,10000,15000,20000],figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedNoveltyResponsesE")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedNoveltyResponsesI")

         # novelty unaveraged over blocks
        avg_rates_time_seqE = np.mean(all_bincounts_conv[:,:,:Ne,startidxN:endidxN],axis = 3) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(all_bincounts_conv[:,:,Ne:,startidxN:endidxN],axis = 3) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks
        # plot pooled histograms
        plot_histogram_cut(avg_rates_time_seq_pooledE, cutlow=3000, cuthigh=3500,ifExcitatory=True,bins = np.linspace(0,300,51), alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledNoveltyResponsesENotBlockAvg")
        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledNoveltyResponsesINotBlockAvg")

        # get ranked sustained repsonses
        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedNoveltyResponsesENotBlockAvg")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedNoveltyResponsesINotBlockAvg")

    ifpooledresponsesmax = False
    if ifpooledresponsesmax:

        # replicate figure 6D sustained activity vs. cell rank
        starttime = 10 # sec
        endtime = 15.5 # sec
        startidx = np.argmax(timevector>starttime)
        endidx = np.argmax(timevector>endtime)
        starttimeT = 0.5 # sec
        endtimeT = 3.5 # sec
        startidxT = np.argmax(timevector>starttimeT)
        endidxT = np.argmax(timevector>endtimeT)
        starttimeN = 16.7 # sec
        endtimeN = 17.2 # sec
        startidxN = np.argmax(timevector>starttimeN)
        endidxN = np.argmax(timevector>endtimeN)
        print(startidxN)
        print(endidxN)
        endidxN = min(endidxN, np.size(all_bincounts_conv,axis = 3))
        #get the average firing rates for each sequence individually
        avg_rates_time_seqE = np.mean(np.amax(all_bincounts_conv[:,:,:Ne,startidx:endidx],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(np.amax(all_bincounts_conv[:,:,Ne:,startidx:endidx],axis = 3),axis=1) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks
        # plot pooled histograms
        plot_histogram(avg_rates_time_seq_pooledE, ifExcitatory=True, alpha=1,bins = np.linspace(0,150,51),figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledSustainedResponsesEMaximum")
        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1,bins = np.linspace(0,60,51),figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledSustainedResponsesIMaximum")

        # get ranked sustained repsonses
        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = False, xticks = [0,5000,10000,15000,20000],figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedStustainedResponsesEMaximum")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedStustainedResponsesIMaximum")

        # get the average firing rates for each sequence individually for whole time
        avg_rates_time_seqE = np.mean(np.amax(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(np.amax(all_bincounts_conv[:,:,Ne:,idxconv:-idxconv],axis = 3),axis=1) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks

        plot_histogram(avg_rates_time_seq_pooledE, ifExcitatory=True, alpha=1, Nblocks = Nblocks,bins = np.linspace(0,150,51),figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramAllSustainedResponsesEMaximum")

        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1,bins = np.linspace(0,50,51),figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramAllSustainedResponsesIMaximum")

        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = True, xticks = [0,5000,10000,15000,20000],figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedAllResponsesEMaximum")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedAllResponsesIMaximum")

        # Transient firing rates

        # get the average firing rates for each sequence individually
        avg_rates_time_seqE = np.mean(np.amax(all_bincounts_conv[:,:,:Ne,startidxT:endidxT],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(np.amax(all_bincounts_conv[:,:,Ne:,startidxT:endidxT],axis = 3),axis=1) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks
        # plot pooled histograms
        plot_histogram_cut(avg_rates_time_seq_pooledE, cutlow = 1000, cuthigh = 1200,ifExcitatory=True, alpha=1,bins = np.linspace(0,180,51),figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledTransientResponsesEMaximum")
        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1, bins = np.linspace(0,70,51),figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledTransientResponsesIMaximum")

        # get ranked sustained repsonses
        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = True, xticks = [0,5000,10000,15000,20000],figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedTransientResponsesEMaximum")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedTransientResponsesIMaximum")

        # Novelty firing rates
        # get the average firing rates for each sequence individually
        avg_rates_time_seqE = np.mean(np.amax(all_bincounts_conv[:,:,:Ne,startidxN:endidxN],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.mean(np.amax(all_bincounts_conv[:,:,Ne:,startidxN:endidxN],axis = 3),axis=1) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks
        # plot pooled histograms
        plot_histogram_cut(avg_rates_time_seq_pooledE, cutlow=1000, cuthigh=1500,ifExcitatory=True, alpha=1, bins = np.linspace(0,150,51),figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledNoveltyResponsesEMaximum")
        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1, bins = np.linspace(0,70,51), figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledNoveltyResponsesIMaximum")

        # get ranked sustained repsonses
        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = True, xticks = [0,5000,10000,15000,20000],figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedNoveltyResponsesEMaximum")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedNoveltyResponsesIMaximum")

        # Novelty firing rates maximum not averaged across blocks
        # get the average firing rates for each sequence individually
        avg_rates_time_seqE = np.amax(all_bincounts_conv[:,:,:Ne,startidxN:endidxN],axis = 3)# first avg over time (specified window) then over blocks
        avg_rates_time_seqI = np.amax(all_bincounts_conv[:,:,Ne:,startidxN:endidxN],axis = 3) # first avg over time (specified window) then over blocks

        avg_rates_time_seq_pooledE = avg_rates_time_seqE.flatten() # first avg over time (specified window) then over blocks
        avg_rates_time_seq_pooledI = avg_rates_time_seqI.flatten() # first avg over time (specified window) then over blocks
        # plot pooled histograms
        plot_histogram_cut(avg_rates_time_seq_pooledE, cutlow=1000, cuthigh=510,ifExcitatory=True, alpha=1, bins = np.linspace(0,400,401),figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledNoveltyResponsesEMaximumNotAvgOverBlocks")
        plot_histogram(avg_rates_time_seq_pooledI, ifExcitatory=False, alpha=1, bins = np.linspace(0,70,51), figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "HistogramPooledNoveltyResponsesIMaximumNotAvgOverBlocks")

        # get ranked sustained repsonses
        avg_rates_time_seq_pooledE.sort()
        rankE = np.linspace(len(avg_rates_time_seq_pooledE)-1,0,len(avg_rates_time_seq_pooledE))

        avg_rates_time_seq_pooledI.sort()
        rankI = np.linspace(len(avg_rates_time_seq_pooledI)-1,0,len(avg_rates_time_seq_pooledI))

        plot_array(rankE,avg_rates_time_seq_pooledE, ifcolor= True,ifExcitatory=True, ifxticks = False ,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedNoveltyResponsesEMaximumNotAvgOverBlocks")
        plot_array(rankI,avg_rates_time_seq_pooledI, ifcolor= True,ifExcitatory=False,figsize=(5,4))
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        save_fig(figure_directory, "SortedNoveltyResponsesIMaximumNotAvgOverBlocks")

    ifassemblyhist = False
    if ifassemblyhist:
        # make histogram of peak firing rates of one assembly being stimulated when sequence is playing
        # average firing rates over blocks then get average and peak firnig rates plot one histo for each assembly in sequence

        #seq 1 ass 0 - ass 3
        # seq 2 ass 4 -ass 7

        avg_rates_time_seqE = np.mean(np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis = 1),axis=2) # first avg over blocks then over time (specified window)
        max_rates_time_seqE = np.amax(np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis = 1),axis=2) # first avg over blocks then take maximum over time (specified window)


        #plot_histogram(avg_rates_time_seqE[0,assemblymembers[0,assemblymembers[0,:]>0]], ifExcitatory=False,color = color[0],bins = np.linspace(0,70,51),alpha = 1)
        #save_fig(figure_directory, "HistogramAllSustainedResponsesE")
        figsize=(10,8)
        images = [range(0,Nimg)]
        for seq in range(1,Nseq+1):
            fig = plt.figure(figsize=figsize)
            for ass in range((seq-1)*Nimg,seq*Nimg):
                plot_histogram_mult(fig, max_rates_time_seqE[seq-1,assemblymembers[ass,assemblymembers[ass,:]>0]-1],ifExcitatory=False,iflegend = True, legend = "assembly "+str(ass+1), color = color[ass],bins = np.linspace(0,300,51),alpha = 0.8, Nblocks = Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=6)
            save_fig(figure_directory, "HistogramAssemblyMaxRatesSeq%d"%seq)

        #avg_rates_time_seqE = np.mean(np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        for seq in range(1,Nseq+1):
            fig = plt.figure(figsize=figsize)
            for ass in range((seq-1)*Nimg,seq*Nimg):
                plot_histogram_mult(fig, avg_rates_time_seqE[seq-1,assemblymembers[ass,assemblymembers[ass,:]>0]-1],ifExcitatory=False,iflegend = True, legend = "assembly "+str(ass+1), color = color[ass],bins = np.linspace(0,60,51),alpha = 0.8, Nblocks = Nblocks)
            plt.locator_params(axis='x', nbins=4)
            plt.locator_params(axis='y', nbins=6)
            save_fig(figure_directory, "HistogramAssemblyMeanRatesSeq%d"%seq)

    ifassemblyhistblock = True
    if ifassemblyhistblock:
        # make histogram of peak firing rates of one assembly being stimulated when sequence is playing
        # average firing rates over blocks then get average and peak firnig rates plot one histo for each assembly in sequence

        #seq 1 ass 0 - ass 3
        # seq 2 ass 4 -ass 7

        avg_rates_time_seqE = np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis=3) # first avg over blocks then over time (specified window)
        max_rates_time_seqE = np.amax(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis=3) # first avg over blocks then take maximum over time (specified window)
        # avg_rates_time_seqE = np.mean(np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis = 1),axis=2) # first avg over blocks then over time (specified window)
        # max_rates_time_seqE = np.amax(np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis = 1),axis=2) # first avg over blocks then take maximum over time (specified window)


        #plot_histogram(avg_rates_time_seqE[0,assemblymembers[0,assemblymembers[0,:]>0]], ifExcitatory=False,color = color[0],bins = np.linspace(0,70,51),alpha = 1)
        #save_fig(figure_directory, "HistogramAllSustainedResponsesE")
        images = [range(0,Nimg)]
        figsize=(10,8)
        for seq in range(Nseq+1):
            # fig = plt.figure(figsize=(15,12))

            for bl in range(Nblocks):#[0,1,2,3,4,5,6,7,8,9,10,19]:#range(0,2):#(Nblocks)
                fig = plt.figure(figsize=figsize)
                for ass in range((seq-1)*Nimg,seq*Nimg):
                    plot_histogram_mult(fig, max_rates_time_seqE[seq-1,bl,assemblymembers[ass,assemblymembers[ass,:]>0]-1],ifExcitatory=False,iflegend = True, legend = "assembly "+str(ass+1), color = color[ass],bins = np.linspace(0,300,51),alpha = 0.8, Nblocks = Nblocks)
                plt.locator_params(axis='x', nbins=4)
                plt.locator_params(axis='y', nbins=6)
                save_fig(figure_directory, "HistogramAssemblyMaxRatesSeq%dBlock%d"%(seq,bl))
                fig = plt.figure(figsize=figsize)
                for ass in range((seq)*Nimg,(seq+1)*Nimg):
                    plot_histogram_mult(fig, max_rates_time_seqE[seq-1,bl,assemblymembers[ass,assemblymembers[ass,:]>0]-1],ifExcitatory=False,iflegend = True, legend = "assembly "+str(ass+1), color = color[ass],bins = np.linspace(0,300,51),alpha = 0.8, Nblocks = Nblocks)
                plt.locator_params(axis='x', nbins=4)
                plt.locator_params(axis='y', nbins=6)
                save_fig(figure_directory, "HistogramAssemblyMaxRatesNextSetofAssembliesSeq%dBlock%d"%(seq,bl))
        for seq in range(Nseq+1):
            for bl in range(Nblocks):#[0,5,9]:#[0,10,19]:#range(Nblocks):#
                fig = plt.figure(figsize=figsize)
                for ass in range((seq-1)*Nimg,seq*Nimg):
                    plot_histogram_mult(fig, avg_rates_time_seqE[seq-1,bl,assemblymembers[ass,assemblymembers[ass,:]>0]-1],ifExcitatory=False,iflegend = True, legend = "assembly "+str(ass+1), color = color[ass],bins = np.linspace(0,60,51),alpha = 0.8, Nblocks = Nblocks)
                plt.locator_params(axis='x', nbins=4)
                plt.locator_params(axis='y', nbins=6)
                save_fig(figure_directory, "HistogramAssemblyMeanRatesSeq%dBlock%d"%(seq,bl))
                fig = plt.figure(figsize=figsize)
                for ass in range((seq)*Nimg,(seq+1)*Nimg):
                    plot_histogram_mult(fig, avg_rates_time_seqE[seq-1,bl,assemblymembers[ass,assemblymembers[ass,:]>0]-1],ifExcitatory=False,iflegend = True, legend = "assembly "+str(ass+1), color = color[ass],bins = np.linspace(0,60,51),alpha = 0.8, Nblocks = Nblocks)
                plt.locator_params(axis='x', nbins=4)
                plt.locator_params(axis='y', nbins=6)
                save_fig(figure_directory, "HistogramAssemblyMeanRatesNextSetofAssembliesSeq%dBlock%d"%(seq,bl))

        for seq in range(Nseq+1):#[1,3]:##Nseq+1):
            for bl in range(Nblocks):#[0,5,9]:#[0,10,19]:#range(Nblocks):#
                fig = plt.figure(figsize=figsize)
                #fig = plt.figure(figsize=(15,12))
                for ass in range(1,Nseq*Nimg+1):
                    plot_histogram_mult(fig, avg_rates_time_seqE[seq-1,bl,assemblymembers[ass,assemblymembers[ass,:]>0]-1],ifExcitatory=False,iflegend = True, legend = "assembly "+str(ass), color = color[ass-1],bins = np.linspace(0,60,51),alpha = 0.8, Nblocks = Nblocks)
                plt.locator_params(axis='x', nbins=4)
                plt.locator_params(axis='y', nbins=6)
                save_fig(figure_directory, "HistogramAssemblyMeanRates_AllAssembliesSeq%dBlock%d"%(seq,bl))
        # #avg_rates_time_seqE = np.mean(np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis = 3),axis=1) # first avg over time (specified window) then over blocks
        # for seq in range(1,Nseq+1):
        #     fig = plt.figure(figsize=(15,12))
        #     for ass in range((seq-1)*Nimg,seq*Nimg):
        #         plot_histogram_mult(fig, avg_rates_time_seqE[seq-1,assemblymembers[ass,assemblymembers[ass,:]>0]-1],ifExcitatory=False,iflegend = True, legend = "assembly "+str(ass+1), color = color[ass],bins = np.linspace(0,100,51),alpha = 0.8, Nblocks = Nblocks)
        #     save_fig(figure_directory, "HistogramAssemblyMeanRatesSeq%d"%seq)

    ifsingleneurontraces = False
    if ifsingleneurontraces:
        # plot traces of individual neurons

        avg_rates_blocks_seqE = np.mean(all_bincounts_conv[:,:,:Ne,idxconv:-idxconv],axis=1) # first avg over time (specified window) then over blocks
        avg_rates_blocks_seqI = np.mean(all_bincounts_conv[:,:,Ne:,idxconv:-idxconv],axis=1) # first avg over time (specified window) then over blocks


        plot_array(timevector[idxconv:-idxconv],avg_rates_blocks_seqE[0,0,:],xlabel="time [s]",figsize=(5, 3))
        plt.xlim(0,5)
        plot_array(timevector[idxconv:-idxconv],avg_rates_blocks_seqE[0,31,:],xlabel="time [s]",figsize=(5, 3))
        plt.xlim(0,5)
        plot_array(timevector[idxconv:-idxconv],avg_rates_blocks_seqE[0,30,:],xlabel="time [s]",figsize=(5, 3))
        plt.xlim(0,5)

        for seq in range(1,Nseq+1):
            for ass in range((seq-1)*Nimg,seq*Nimg):
                fig = plt.figure(figsize=figsize)
                count = 0
                for cc in assemblymembers[ass,assemblymembers[ass,:]>0]:
                    if count <= 20:
                        count = count + 1
                        plot_array_mult(fig,timevector[idxconv:-idxconv],avg_rates_blocks_seqE[seq-1,cc-1,:],xlabel="time [s]",figsize=(5, 3), ncol = 4,legend = str(cc),iflegend=True, lw = 2)
                save_fig(figure_directory, "IndividualRateTracesSeq%dAss%d"%(seq,ass))
            fig = plt.figure(figsize=figsize)
            count = 0
            for cc in noveltyidx:
                if count <= 20:
                    count = count + 1
                    plot_array_mult(fig,timevector[idxconv:-idxconv],avg_rates_blocks_seqE[seq-1,cc-1,:],xlabel="time [s]",figsize=(5, 3), ncol = 4,legend = str(cc),iflegend=True, lw = 2)
            save_fig(figure_directory, "IndividualRateTracesNoveltySeq%d"%seq)
            fig = plt.figure(figsize=figsize)
            count = 0
            for cc in np.where(untargetedidx)[0]:
                if count <= 3:
                    count = count + 1
                    plot_array_mult(fig,timevector[idxconv:-idxconv],avg_rates_blocks_seqE[seq-1,cc,:],xlabel="time [s]",figsize=(5, 3), ncol = 4,legend = str(cc),iflegend=True, lw = 2)
            save_fig(figure_directory, "IndividualRateTracesUntergetedSeq%d"%seq)


    pattern = ["timevector*","avg_rates*","*bincounts_conv*", "zscore*"]
    antipattern = ["*hist*","edges"] # specify lists with different length -> different treatment

    # create results file
    ifsaveresults = False
    if ifsaveresults:
        file_name_results = results_folder + file_name + "/singleneuronresults%s.h5"%timestr
        f_results = h5py.File(file_name_results, "a")


        f_results.create_dataset('avgwindow%d'%avgwindow, data=avgwindow)
        #f_results.create_dataset('Avgwindow%d/Nreps'%avgwindow, data=Nreps)

        for key in dir():
            if fnmatch.fnmatch(key, pattern[0]):
                if not fnmatch.fnmatch(key, antipattern[0]):
                    f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
                else:
                    listlen = len(vars()[key])
                    for i in range(0,listlen):
                        f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])
            elif fnmatch.fnmatch(key, pattern[1]):
                f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
            elif fnmatch.fnmatch(key, pattern[2]):
                f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
            elif fnmatch.fnmatch(key, pattern[3]):
                f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
            elif fnmatch.fnmatch(key, antipattern[1]):
                listlen = len(vars()[key])
                for i in range(0,listlen):
                    f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])

        f_results.close()
    gc.collect()

def run_whole_analysis_mechanism(file_name, avgwindow = 8, timestr = "_now", RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR
    # folder with stored data from the run
    #run_folder = "/gpfs/gjor/personal/schulza/data/main/sequences/"
    # folder with analysed results from spiketime analysis in julia & where to results are stored
    #results_folder = "/home/schulza/Documents/results/main/sequences/"
    #results_folder = "/gpfs/gjor/personal/schulza/results/sequences/"
    #run_folder = "../data/"
    #results_folder = "../results/"
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
    plot_all_averages(edges, mean_hist_E, Seqs, savehandle = "E", ifseqlen=True, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=True,legendhandle = "Seq. ", ifyticks=False)
    plot_all_averages(edges, mean_hist_I, Seqs, savehandle = "I", ifseqlen=True, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    #plot_all_averages(edges, mean_hist_E_boxcar, Seqs, savehandle = "E_boxcar", ifseqlen=True,startidx = idxconv, endidx = -idxconv, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=False, legendhandle = "Seq. : ", ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomem, Seqs, savehandle = "E_nomem", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomemnonov, Seqs, savehandle = "E_nomemnonov", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nov, Seqs, savehandle = "E_nov", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)

    plot_all_traces_and_average(edges, hist_E, mean_hist_E, Seqs, savehandle = "E", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_I, mean_hist_I, Seqs, savehandle = "I", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    #plot_all_traces_and_average(edges, hist_E_boxcar, mean_hist_E_boxcar, Seqs, ifseqlen=True, savehandle = "E_boxcar_avg%d" % int(avgwindow) , startidx = idxconv, endidx = -idxconv, Nblocks = Nblocks, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomem, mean_hist_E_nomem, Seqs, ifseqlen=True, savehandle = "E_nomem", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomemnonov, mean_hist_E_nomemnonov, Seqs, ifseqlen=True, savehandle = "E_nomemnonov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nov, mean_hist_E_nov, Seqs, ifseqlen=True, savehandle = "E_nov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)

    plot_all_traces_and_average_E_I(edges, hist_E, hist_I, Seqs, savehandle = "E_Itest", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=True, ifyticks=False)
    #plot_all_traces_and_average_E_I(edges, hist_E_boxcar, hist_I_boxcar, Seqs, savehandle = "E_Itestboxcar", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, startidx = idxconv, endidx = -idxconv, color = colorE, ifoffset=False, iflegend=True, ifyticks=False)

    plt.figure()
    for seq in range(1,Nseq + 1):
        for bl in range(1, Nblocks + 1):
            plt.plot(edges[0][:-5], hist_E[seq-1][bl-1,:-5], label = f'seq. {seq} block {bl}')
    plt.legend()
    return mean_hist_E,mean_hist_I, edges[0],hist_E, hist_E_boxcar, figure_directory


def run_whole_analysis_SSA(file_name, avgwindow = 8, timestr = "_now", RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR
    # folder with stored data from the run
    #run_folder = "/gpfs/gjor/personal/schulza/data/main/sequences/"
    # folder with analysed results from spiketime analysis in julia & where to results are stored
    #results_folder = "/home/schulza/Documents/results/main/sequences/"
    #results_folder = "/gpfs/gjor/personal/schulza/results/sequences/"
    #run_folder = "../data/"
    #results_folder = "../results/"
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
    plot_all_averages(edges, mean_hist_E, Seqs, savehandle = "E", ifseqlen=True, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=True,legendhandle = "Seq. ", ifyticks=False)
    plot_all_averages(edges, mean_hist_I, Seqs, savehandle = "I", ifseqlen=True, figure_directory = figure_directory, color = colorI, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    #plot_all_averages(edges, mean_hist_E_boxcar, Seqs, savehandle = "E_boxcar", ifseqlen=True,startidx = idxconv, endidx = -idxconv, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=False, legendhandle = "Seq. : ", ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomem, Seqs, savehandle = "E_nomem", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomemnonov, Seqs, savehandle = "E_nomemnonov", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nov, Seqs, savehandle = "E_nov", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=True, legendhandle = "Seq. ", ifyticks=False)

    plot_all_traces_and_average(edges, hist_E, mean_hist_E, Seqs, savehandle = "E", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_I, mean_hist_I, Seqs, savehandle = "I", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    #plot_all_traces_and_average(edges, hist_E_boxcar, mean_hist_E_boxcar, Seqs, ifseqlen=True, savehandle = "E_boxcar_avg%d" % int(avgwindow) , startidx = idxconv, endidx = -idxconv, Nblocks = Nblocks, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomem, mean_hist_E_nomem, Seqs, ifseqlen=True, savehandle = "E_nomem", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomemnonov, mean_hist_E_nomemnonov, Seqs, ifseqlen=True, savehandle = "E_nomemnonov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nov, mean_hist_E_nov, Seqs, ifseqlen=True, savehandle = "E_nov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)

    plot_all_traces_and_average_E_I(edges, hist_E, hist_I, Seqs, savehandle = "E_Itest", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, color = colorE, ifoffset=False, iflegend=True, ifyticks=False)
    #plot_all_traces_and_average_E_I(edges, hist_E_boxcar, hist_I_boxcar, Seqs, savehandle = "E_Itestboxcar", Nblocks = Nblocks, ifseqlen=True, figure_directory = figure_directory, startidx = idxconv, endidx = -idxconv, color = colorE, ifoffset=False, iflegend=True, ifyticks=False)

    plt.figure()
    for seq in range(1,Nseq + 1):
        for bl in range(1, Nblocks + 1):
            plt.plot(edges[0][:-5], hist_E[seq-1][bl-1,:-5], label = f'seq. {seq} block {bl}')
    plt.legend()
    return mean_hist_E,mean_hist_I, edges[0],hist_E, hist_E_boxcar, figure_directory, hist_E_nov


def run_variable_repetitions_short_old(file_name, avgwindow = 8, timestr = "_now",RUN_DIR="../data/", RESULTS_DIR ="../results/"):
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
    print(Nreps)
    plot_all_averages(edges, mean_hist_E, Nreps, savehandle = "E", figure_directory = figure_directory, Nreponset = 1, color = color, ifoffset=True, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_boxcar, Nreps, savehandle = "E_boxcar", Nreponset = 1, startidx = idxconv, endidx = -idxconv, figure_directory = figure_directory, color = color, ifoffset=True, iflegend=False, ifyticks=False)
    #
    # plot_all_traces_and_average(edges, hist_E, mean_hist_E, Nreps, savehandle = "E", Nblocks = Nblocks, Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    # plot_all_traces_and_average(edges, hist_I, mean_hist_I, Nreps, savehandle = "I", Nblocks = Nblocks, Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    # plot_all_traces_and_average(edges, hist_E_boxcar, mean_hist_E_boxcar, Nreps, Nreponset = 1, savehandle = "E_boxcar_avg%d" % int(avgwindow) , startidx = idxconv, endidx = -idxconv, Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    # plot_all_traces_and_average(edges, hist_E_nomem, mean_hist_E_nomem, Nreps, Nreponset = 1, savehandle = "E_nonmem", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    # plot_all_traces_and_average(edges, hist_E_nomemnonov, mean_hist_E_nomemnonov, Nreps, Nreponset = 1, savehandle = "E_nomemnonov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)

    plot_all_averages(edges, mean_hist_E, Nreps, savehandle = "Enooffset", figure_directory = figure_directory, Nreponset = 1, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_I, Nreps, savehandle = "Inooffset", figure_directory = figure_directory, Nreponset = 1, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_boxcar, Nreps, savehandle = "E_boxcarnooffset", Nreponset = 1, startidx = idxconv, endidx = -idxconv, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomem, Nreps, savehandle = "E_nomemnooffset", Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_averages(edges, mean_hist_E_nomemnonov, Nreps, savehandle = "E_nomemnonovnooffset", Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)

    plot_all_traces_and_average(edges, hist_E, mean_hist_E, Nreps, savehandle = "Enooffset", Nblocks = Nblocks, Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_I, mean_hist_I, Nreps, savehandle = "Inooffset", Nblocks = Nblocks, Nreponset = 1, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_boxcar, mean_hist_E_boxcar, Nreps, Nreponset = 1, savehandle = "Enooffset_boxcar_avg%d" % int(avgwindow) , startidx = idxconv, endidx = -idxconv, Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomem, mean_hist_E_nomem, Nreps, Nreponset = 1, savehandle = "Enooffsetnonmem", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)
    plot_all_traces_and_average(edges, hist_E_nomemnonov, mean_hist_E_nomemnonov, Nreps, Nreponset = 1, savehandle = "Enooffset_nomemnonov", Nblocks = Nblocks, figure_directory = figure_directory, color = color, ifoffset=False, iflegend=False, ifyticks=False)


    # ------------------------------------ FITTING ---------------------------------------------
    # """fit_variable_repetitions_gen_arrays(args):
    #     perform fitting of all traces included in datalist and meandatalist
    #         determine the baseline firing rate prior to the novelty stimulation

    # set initial parameters for fitting of the exponential curve
    # fit a * exp(-t/tau) + a_0
    initial_params = [2, 20, 3]
    #                [a, tau,a_0]
    fit_bounds = (0, [10., 20., 10])
    print("FIT_Bounds")
    print(fit_bounds)
    avgindices = 30
    startimg = Nimg # after which image should fit start at block onset
    print(startimg)
    print(idxconv)
    print("Issue here: ")

    # fitting of initial transient
    t_before_nov, params_blockavg, params_covariance_blockavg, params_err_blockavg, params, params_covariance, params_err = fit_variable_repetitions_gen_arrays_startidx(
        edges,hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        avgindices = avgindices, initialparams=initial_params, bounds=fit_bounds, ifplot = True,
        startimg = startimg, idxconv = idxconv)

    #get_baseline_firing_rate
    baseline_avg, baseline, mean_baseline, std_baseline = get_baseline_firing_rate(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        avgindices = avgindices, idxconv = idxconv)



    # fitting of post novelty transient
    t_before_trans, params_blockavg_trans, params_err_blockavg_trans, params_trans, params_err_trans = fit_variable_repetitions_gen_arrays_postnovelty(
        edges,hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        avgindices = avgindices, initialparams=initial_params, bounds=fit_bounds, ifplot = True,
        startimg = startimg, idxconv = idxconv)

    # collect garbage
    gc.collect()

    tau_transientpre, tau_transientpre_err = convert_tau(params,params_err)
    tau_transientpost, tau_transientpost_err = convert_tau(params_trans, params_err_trans)
    tau_transientpre_avg, tau_transientpre_err_avg = convert_tau_avg(params_blockavg, params_err_blockavg)
    tau_transientpost_avg, tau_transientpost_err_avg = convert_tau_avg(params_blockavg_trans, params_err_blockavg_trans)

    # -----------------------------------------

    samples_img = int(round(lenstim/binsize))
    height_novelty_avg, height_novelty, mean_novelty, std_novelty, novelty_avgidx, noveltyidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = False, iftransientpost = False,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = Nimg*samples_img)

    height_trans_pre_avg, height_trans_pre, mean_trans_pre, std_trans_pre, trans_pre_avgidx, trans_preidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = True, iftransientpost = False,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = Nimg*samples_img)

    height_trans_post_avg, height_trans_post, mean_trans_post, std_trans_post, trans_post_avgidx, trans_postidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = False, iftransientpost = True,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = Nimg*samples_img)

    # ---------------------------------------- plotting --------------------------------------------------------
    # plot pre transient decay constant vs. number of repetitions
    plot_Nreps_tau(Nreps, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xtickstepsize=1,savename="Nreps_Decay_Const_From_Fit_Pre_grey_dots")
    # plot baseline determined from fit vs. number of repetitions
    plot_Nreps_baseline(Nreps, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xtickstepsize=1,savename="Nreps_Baseline_From_Fit_Pre_grey_dots")
    # saving and reloading for comparing instantiations

    # plot post novelty transient decay constant vs. number of repetitions
    plot_Nreps_tau(Nreps, params_trans, params_blockavg_trans, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xtickstepsize=1,savename="Nreps_Decay_Const_From_Fit_Post_grey_dots")
    # plot baseline determined from fit vs. number of repetitions
    plot_Nreps_baseline(Nreps, params_trans, params_blockavg_trans, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=True, xtickstepsize=1,savename="Nreps_Baseline_From_Fit_Post_grey_dots")



    # plot unsubtracted data transients, novelty and baseline
    plot_Nreps_array(Nreps, height_trans_pre, height_trans_pre_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient peak rate [Hz]", figure_directory = figure_directory, xtickstepsize=1, ifsavefig = True, savename="Nreps_TransientPre_grey_dots")
    plot_Nreps_array(Nreps, baseline, baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="baseline rate [Hz]", figure_directory = figure_directory, ifsavefig = True, xtickstepsize=1, savename="Nreps_BL_grey_dots")
    plot_Nreps_array(Nreps, height_novelty, height_novelty_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty peak rate [Hz]", figure_directory = figure_directory, xtickstepsize=1, ifsavefig = True, savename="Nreps_Novelty_grey_dots")
    plot_Nreps_array(Nreps, height_trans_post, height_trans_post_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient peak rate [Hz]", figure_directory = figure_directory, xtickstepsize=1, ifsavefig = True, savename="Nreps_TransientPre_grey_dots")

    # plot data transients, novelty subtracted baseline
    plot_Nreps_array(Nreps, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPost-BL_grey_dots")
    plot_Nreps_array(Nreps, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_Novelty-BL_grey_dots")
    plot_Nreps_array(Nreps, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPre-BL_grey_dots")

    # plot data transients, novelty subtracted baseline with errorbars
    plot_Nreps_array_errorbar(Nreps, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPost-BL_grey_errorbar")
    plot_Nreps_array_errorbar(Nreps, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_Novelty-BL_grey_errorbar")
    plot_Nreps_array_errorbar(Nreps, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPre-BL_grey_errorbar")

    # plot data transients, novelty subtracted baseline with errorbands
    plot_Nreps_array_errorband(Nreps, height_trans_post-baseline, height_trans_post_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPost-BL_grey_errorband")
    plot_Nreps_array_errorband(Nreps, height_novelty-baseline, height_novelty_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="novelty - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_Novelty-BL_grey_errorband")
    plot_Nreps_array_errorband(Nreps, height_trans_pre-baseline, height_trans_pre_avg-baseline_avg, Nblocks = Nblocks, ifxlims = True, ylabel ="transient - baseline rate [Hz]", xtickstepsize=1, figure_directory = figure_directory, ifsavefig = True, savename="Nreps_TransientPre-BL_grey_errorband")

    # declare variable name patterns to be stored in hdf5 file
    # lists with different lengths cannot be stored in hdf5 -> split up into indiv arrays with dataset name string(index)

    # -------------------------------------- saving ----------------------------------------------------------------
    pattern = ["mean*","params*","height*", "tau*", "baseline*"]
    antipattern = ["*hist*","edges"] # specify lists with different length -> different treatment

    # create results file
    file_name_results = results_folder + file_name + "/results%s.h5"%timestr
    f_results = h5py.File(file_name_results, "a")
    print(f_results)

    f_results.create_dataset('avgwindow%d'%avgwindow, data=avgwindow)
    f_results.create_dataset('Avgwindow%d/Nreps'%avgwindow, data=Nreps)

    for key in dir():
        if fnmatch.fnmatch(key, pattern[0]):
            if not fnmatch.fnmatch(key, antipattern[0]):
                f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
            else:
                listlen = len(vars()[key])
                for i in range(0,listlen):
                    f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])
        elif fnmatch.fnmatch(key, pattern[1]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[2]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[3]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, pattern[4]):
            f_results.create_dataset('Avgwindow%d/'%avgwindow + key, data=vars()[key])
        elif fnmatch.fnmatch(key, antipattern[1]):
            listlen = len(vars()[key])
            for i in range(0,listlen):
                f_results.create_dataset('%s_window%d/'%(key,avgwindow) + str(i), data=vars()[key][i])

    f_results.close()



def repeated_sequence_analysis(file_name, avgwindow = 8, timestr = "_now", RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR
    # folder with stored data from the run
    #run_folder = "/gpfs/gjor/personal/schulza/data/main/sequences/"
    # folder with analysed results from spiketime analysis in julia & where to results are stored
    #results_folder = "/home/schulza/Documents/results/main/sequences/"
    #results_folder = "/gpfs/gjor/personal/schulza/results/sequences/"
    #run_folder = "../data/"
    #results_folder = "../results/"
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
    idxconv = np.floor_divide(avgwindow,2)+1

    nsamples = len(edges[0])
    sample_fraction = 0.8
    # change sample_fraction for Nreps smaller than 5
    if Nreps <= 5:
        sample_fraction = 0.6
    # get the last sample to be considered in fit (discarded novelty response)
    lastsample = int(round(sample_fraction*nsamples))
    # ------------------------------------ FITTING ---------------------------------------------
    # """fit_variable_repetitions_gen_arrays(args):
    #     perform fitting of all traces included in datalist and meandatalist
    #         determine the baseline firing rate prior to the novelty stimulation

    # set initial parameters for fitting of the exponential curve
    # fit a * exp(-t/tau) + a_0
    initial_params = [2, 20, 3]
    #                [a, tau,a_0]
    fit_bounds = (0, [10., 140., 10])
    avgindices = 30
    startimg = Nimg # after which image should fit start at block onset update for Seqlen in function always last img
    # fitting of initial transient
    t_before_nov, params_blockavg, params_covariance_blockavg, params_err_blockavg, params, params_covariance, params_err = fit_gen_arrays_startidx(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Seqs, Nblocks,
        ifseqlen=False, avgindices = avgindices, Nseq = Nseq, initialparams=initial_params, bounds=fit_bounds, ifplot = False,
        startimg = startimg, idxconv = idxconv)

    #get_baseline_firing_rate
    baseline_avg, baseline, mean_baseline, std_baseline = get_baseline_firing_rate(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        ifseqlen=False, ifrepseq = True, Nseq = Nseq,  avgindices = avgindices, idxconv = idxconv)

    gc.collect()

    tau_transientpre, tau_transientpre_err = convert_tau(params,params_err)
#     tau_transientpost, tau_transientpost_err = convert_tau(params_trans, params_err_trans)
    tau_transientpre_avg, tau_transientpre_err_avg = convert_tau_avg(params_blockavg, params_err_blockavg)
#     tau_transientpost_avg, tau_transientpost_err_avg = convert_tau_avg(params_blockavg,params_err_blockavg)

    # ----------------------------------------- get peaks -----------------------------------------
    # Excitatory
    samples_img = int(round(lenstim/binsize))
    height_novelty_avg, height_novelty, mean_novelty, std_novelty, novelty_avgidx, noveltyidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = False, iftransientpost = False, ifseqlen=False, ifrepseq = True, Nseq = Nseq,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = 8*samples_img)

    height_trans_pre_avg, height_trans_pre, mean_trans_pre, std_trans_pre, trans_pre_avgidx, trans_preidx = get_peak_height(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = True, iftransientpost = False, ifseqlen=False, ifrepseq = True, Nseq = Nseq,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = 8*samples_img)
# -------------------------------- inhibitory ----------------------------------------------------------
        #get_baseline_firing_rate
    baseline_avgI, baselineI, mean_baselineI, std_baselineI = get_baseline_firing_rate(
        edges, hist_I_boxcar, mean_hist_I_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        ifseqlen=False, ifrepseq = True, Nseq = Nseq,  avgindices = avgindices, idxconv = idxconv)

    gc.collect()


    # ----------------------------------------- get peaks -----------------------------------------
    # INhibitory

    samples_img = int(round(lenstim/binsize))
    height_novelty_avgI, height_noveltyI, mean_noveltyI, std_noveltyI, novelty_avgidxI, noveltyidxI = get_peak_height(
        edges, hist_I_boxcar, mean_hist_I_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = False, iftransientpost = False, ifseqlen=False, ifrepseq = True, Nseq = Nseq,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = 8*samples_img)

    height_trans_pre_avgI, height_trans_preI, mean_trans_preI, std_trans_preI, trans_pre_avgidxI, trans_preidxI = get_peak_height(
        edges, hist_I_boxcar, mean_hist_I_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        iftransientpre = True, iftransientpost = False, ifseqlen=False, ifrepseq = True, Nseq = Nseq,
        avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = 8*samples_img)


    # ----------------------------------------- plotting --------------------------------------------

    # ----------------------------------------------------------------- bar plot ------------------------------------
    barplot_peak_comparison_EI(height_novelty, height_noveltyI, height_trans_pre, height_trans_preI, baseline, baselineI, iflegend=True, figure_directory=figure_directory, ifsavefig = True, xlabel=" ", alpha = 1)
    barplot_peak_comparison_EI(height_novelty_avg, height_novelty_avgI, height_trans_pre_avg, height_trans_pre_avgI, baseline_avg, baseline_avgI, iflegend=True, figure_directory=figure_directory, ifsavefig = True, xlabel=" ", alpha = 1, savename = "ComparisonPeakHeightsBarPlot_averages")

    barplot_peak = [height_novelty, height_noveltyI, height_trans_pre, height_trans_preI, baseline, baselineI]
    barplot_peak_avg = [height_novelty_avg, height_novelty_avgI, height_trans_pre_avg, height_trans_pre_avgI, baseline_avg, baseline_avgI]
    return mean_hist_E,mean_hist_I, edges[0],hist_E, hist_E_boxcar, figure_directory, barplot_peak, barplot_peak_avg



def sequence_length_single(file_name, avgwindow = 8, timestr = "_now",RUN_DIR="../data/", RESULTS_DIR ="../results/"):
    # folder with stored data from the run
    run_folder = RUN_DIR
    results_folder = RESULTS_DIR
    #run_folder = "/gpfs/gjor/personal/schulza/data/main/sequences/"
    # folder with analysed results from spiketime analysis in julia & where to results are stored
    #results_folder = "/gpfs/gjor/personal/schulza/results/sequences/"


    # define folder where figues should be stored
    figure_directory = results_folder + file_name + "/" + "figures_window%d/"%avgwindow
    if not os.path.exists(figure_directory):
        os.makedirs(figure_directory)

    # read in run parameters
    file_name_run = run_folder + file_name
    # open file
    frun = h5py.File(file_name_run, "r")

    # read in stimulus parameters
#     Nimg, lenNreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength  = frun["initial"]["stimparams"].value
#     repetitions  = frun["initial"]["repetitions"].value
#     Nreps  = frun["initial"]["Nreps"].value

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
        ifseqlen=True, avgindices = avgindices, initialparams=initial_params, bounds=fit_bounds, ifplot = True,
        startimg = startimg, idxconv = idxconv)

    #get_baseline_firing_rate
    baseline_avg, baseline, mean_baseline, std_baseline = get_baseline_firing_rate(
        edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
        ifseqlen=True, avgindices = avgindices, idxconv = idxconv)


    plot_all_averages_with_fits(edges, mean_hist_E, Nimg, params_blockavg, savehandle = "E_withfits", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=True, iflegend=False, ifyticks=False)
    plot_all_averages_with_fits(edges, mean_hist_E, Nimg, params_blockavg, savehandle = "E_boxcar_withfits", ifseqlen=True, figure_directory = figure_directory, color = color, ifoffset=True, iflegend=False, ifyticks=False)

#     # fitting of post novelty transient
#     t_before_trans, params_blockavg_trans, params_err_blockavg_trans, params_trans, params_err_trans = fit_variable_repetitions_gen_arrays_postnovelty(
#         edges,hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
#         ifseqlen=True, avgindices = avgindices, initialparams=initial_params, bounds=fit_bounds, ifplot = False,
#         startimg = startimg, idxconv = idxconv)

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


#     height_trans_post_avg, height_trans_post, mean_trans_post, std_trans_post, trans_post_avgidx, trans_postidx = get_peak_height(
#         edges, hist_E_boxcar, mean_hist_E_boxcar, lenstim, lenpause, Nreps, Nimg, Nblocks,
#         iftransientpre = False, iftransientpost = True,
#         avgindices = avgindices, startimg = startimg, idxconv = idxconv, search_margin = Nimg*samples_img)

    # ---------------------------------------- plotting --------------------------------------------------------
    if ifplotting:

        # plot pre transient decay constant vs. number of repetitions
        plot_Nreps_tau(Nimg, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=False, xlabel="sequence length", xtickstepsize = 1, savename = "NimgTau")
        # plot baseline determined from fit vs. number of repetitions
        plot_Nreps_baseline(Nimg, params, params_blockavg, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=False, xlabel="sequence length", xtickstepsize = 1,savename = "NimgBaseline")
        # saving and reloading for comparing instantiations

#     # plot post novelty transient decay constant vs. number of repetitions
#     plot_Nreps_tau(Nreps, params_trans, params_blockavg_trans, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=False)
#     # plot baseline determined from fit vs. number of repetitions
#     plot_Nreps_baseline(Nreps, params_trans, params_blockavg_trans, color = color, Nblocks = Nblocks, figure_directory = figure_directory, ifsavefig=False)


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

    # declare variable name patterns to be stored in hdf5 file
    # lists with different lengths cannot be stored in hdf5 -> split up into indiv arrays with dataset name string(index)


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
