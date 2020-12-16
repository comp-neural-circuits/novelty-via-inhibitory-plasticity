# this file is part of the V1 2Ch Predictive Code project
# ----------------------------------------------------------------------------------------
# initialise the evaluation right after the simulation for easier post processing
# input: file name, weight or spiketimeevaluation
# output:population averages, weight analyses stored in hdf5 files
# ----------------------------------------------------------------------------------------

using PyPlot
using HDF5
using Distributions
using Dates
using Profile
using LinearAlgebra
using StatsBase

# import all functions relevant for evaluation
include("../evaluation/evaluationfunctions.jl")
include("../evaluation/spiketimeevaluationfunction.jl")

Nt = 25#140 # number of weights matrices to be analysed nd plotted
Ne = 4000
Ni = 1000
Ncells = Ne + Ni

println(pwd())

filelocation = "../data/"

fname = filesavename * datetime * tag
binsize = 400
println(fname)
flags = "sequence"

spiketimeevaluationfunction(filelocation, fname, flags, storedweights = true,
	 storedspiketimes = true, sporderedt = true, Nt = Nt, savefigures = true,
	  showfigures = false, savematrices = true, binsize = binsize)
