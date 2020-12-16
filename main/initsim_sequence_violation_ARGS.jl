# this file is part of the V1 2Ch Predictive Code project
# ----------------------------------------------------------------------------------------
# initialise the simulation parameters and start the simulation
# input: stim params
# output: hdf5 file with stimulus parameters, run results (weight matrices, spiketimes, total number of spikes)
# ----------------------------------------------------------------------------------------


# if using atom and open folder function go into main folder
#cd("./main/")


# include inbuilt modules
using PyCall
using PyPlot
PyCall.PyDict(matplotlib["rcParams"])["pdf.fonttype"] = 42
using HDF5
using Distributions
using Dates
using LinearAlgebra
using Random
using Distributed
#using Revise
#using Profile

println(pwd())
# enable or disable certain functions
plotting            = false  # make plots

# --------------------- include functions  ------------------------------------------------
# include runs the respective julia code, i.e. defined functions are then in the workspace

include("../simulation/runsimulation_inhibtuning.jl")
include("../simulation/sequencefunctions.jl")
include("../simulation/helperfunctions.jl")
include("../evaluation/evaluationfunctions.jl")

# --------------------- initialise simulation --------------------------------------------

# Define number of excitatory and inhibitory neurons
const Ne = 4000
const Ni = 1000

Ncells = Ne + Ni

# Set integration timestep
dt 	= 0.1 #integration timestep in ms

ifiSTDP = true 		#  include inhibitory plasticity
ifwadapt = false	#  consider AdEx or Ex IF neurons

# --------------------- generate the stimulus --------------------------------------------

# stimulus parameters
Nimg = parse(Int64, ARGS[1])		# number of stimuli per sequence
Nreps = parse(Int64, ARGS[2])		# number of repetitions per sequence cycle
Nseq = parse(Int64, ARGS[3])		# number of sequences
Nblocks = parse(Int64, ARGS[4])		# number of sequence block repetitions
stimstart = 4000	# start time of stimulation in ms
lenstimpre = parse(Int64, ARGS[5])		# duration of assembly stimulation per image in ms
lenpausepre = 0		# duration of no stimulation between two images in ms
strengthpre = parse(Int64, ARGS[6])	# strength of the stimulation in kHz added to baseline input
lenstim = lenstimpre		# duration of assembly stimulation per image in ms
lenpause = lenpausepre		# duration of no stimulation between two images in ms
strength = strengthpre		# strength of the stimulation in kHz added to baseline input
Ntrain = parse(Int64, ARGS[7])
Nass = (Nimg) * Nseq +  Nseq * Nblocks    # total number of assemblies Nimg * Nseq + all novelty assemblies

stimparams = [Nimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength] # store stimulus param array in hdf5 file
stimparams_prestim = [Nimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength, lenstimpre, lenpausepre, strengthpre, Ntrain] # store stimulus param array in hdf5 file

# adjustment of learning rates
adjustfactor = parse(Int64, ARGS[8])/10
adjustfactorinhib = adjustfactor

if parse(Int64, ARGS[9]) == 1
	ifwadapt = true
end

inhibfactor = parse(Int64, ARGS[10])/100 # 100 more refined steering possibility to just switch it off
stimulus = zeros(1, 4)	# initialisation of the stimulus
# nohup julia initsim_sequence_violation_ARGS.jl 3 20 5 10 300 12 5 10 0 10 &> ../tmp/Pre5_Final_Run_Block_nonadapt_10_inhibtuning_nohetero_tunefactor_10_shortpretrain.txt &




tag = "repseq.h5"

# ---------------------- generate the stimulus --------------------------------
withnovelty = true
shuffledstim = false
pretrainig = true
if withnovelty
	if pretrainig
		stimulus = genstimparadigmpretraining(stimulus, Nass = Nass, Ntrain = Ntrain, stimstart = stimstart, lenstim = lenstimpre, lenpause = lenpausepre, strength = strengthpre)
		lenpretrain = size(stimulus,1)
	end

	if shuffledstim
		if pretrainig
			stimulus, blockonset = genstimparadigmnovelshuffledpretrain(stimulus, lenpretrain, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
			tag = "repeatedsequencesshuffled.h5"

		else
			stimulus, blockonset = genstimparadigmnovelshuffled(stimulus, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength )
			tag = "repeatedsequencesshuffled.h5"
		end
	else
		stimulus, blockonset = genstimparadigmnovelcont(stimulus, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength )
		tag = "repeatedsequences.h5"
	end
else
	# even if no novelty assemblies are shown still train with all novelty assemblies
	if pretrainig
		#println(stimulus)
		stimulus = genstimparadigmpretraining(stimulus, Nass = Nass, Ntrain = Ntrain, stimstart = stimstart, lenstim = lenstimpre, lenpause = lenpausepre, strength = strengthpre)
		#println(stimulus)
		lenpretrain = size(stimulus,1)
		tag = "repeatedsequences_imprinting.h5"

	end
	tag = "repeatedsequencesnoNovelty.h5"
	stimulus, blockonset = genstimparadigmNonovel(stimulus, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength )
end


# set length of pretraining to 0 if no pretraining
if Ntrain == 0
	lenpretrain = 0
end

# get sequence order
blockidx = collect(lenpretrain + 1:Nreps*Nimg:size(stimulus,1)) # get all stimulus indices when a new sequence starts
seqnumber = stimulus[blockidx,1]
for n = 2:Nseq
    seqnumber[seqnumber .== (1 + (n-1)*Nimg)] .= n
end



# Switch last two images instead of novel image
lastshuffled = false
if lastshuffled
	if pretrainig

		idx = findall(stimulus[lenpretrain + 1:end,1] .> Nseq*Nimg)
		vals = stimulus[idx.+(lenpretrain),1]
		stimulus[idx.+(lenpretrain),1] .= stimulus[idx.+(lenpretrain-1),1]
		stimulus[idx.+(lenpretrain-1),1] .= stimulus[idx.+(lenpretrain-1),1] .+ 1
		tag = "shufflenovelty" * tag
	else
		idx = findall(stimulus[:,1] .> Nseq*Nimg)
		vals = stimulus[idx,1]
		stimulus[idx,1] = stimulus[idx.-1,1]
		stimulus[idx.-1,1] = stimulus[idx,1] .+ 1
		tag = "shufflenovelty" * tag


	end

end


# reduce novelty input by a certain factor a different one for each sequence
# to infer the relevance of the novelty stimulus strength
reducednovelty = false
lastimages = collect(Nimg:Nimg:Nimg*Nseq)
reducefactor =  collect(0:(1/(Nseq-1)):1)
secondtolastimage = lastimages.-1

if reducednovelty
	if pretrainig

		idx = findall(stimulus[lenpretrain + 1:end,1] .> Nseq*Nimg)
		vals = stimulus[idx.+(lenpretrain),1]
		for i in idx
			for im = 1:length(secondtolastimage)
				if stimulus[i+(lenpretrain-1),1] == secondtolastimage[im]# if previous image is certain second to last image of sequence reduce by corresponding factor
					stimulus[i+(lenpretrain),4] = stimulus[i+(lenpretrain),4]*reducefactor[im]
				end
			end
		end
		tag = "reducednovelty$(reducefactor)" * tag
	else
		idx = findall(stimulus[:,1] .> Nseq*Nimg)
		vals = stimulus[idx,1]
		for i in idx
			for im = 1:length(secondtolastimage)
				if stimulus[i-1,1] == secondtolastimage[im]# if previous image is certain second to last image of sequence reduce by corresponding factor
					stimulus[i,4] = stimulus[i,4]*reducefactor[im]
				end
			end
		end
		tag = "reducednovelty$(reducefactor)" * tag
	end

end

# simulation run time
T = stimulus[end,3]+lenpause # last stimulation time + pause duration
println(T)

println("Simulation run time: $T")

# --------------------- initialise savefile ---------------------------------------------------------

# initialise savefile and avoid overwriting when identical parameters are used
datetime = Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS")

filesavename = "Seq_Violation_$(inhibfactor)_dur$(T)msNblocks$(Nblocks)Ntrain$(Ntrain)lenstim$(lenstim)lenpause$(lenpause)Nreps$(Nreps)strength$(strength)wadapt$(ifwadapt)iSTDP$(ifiSTDP)RateAdjust$(adjustfactor)Time"

savefile = "../data/"*filesavename * datetime * tag
println(savefile)

# --------------------- initialise weights and assemblymembers -----------------------------------------

weights = initweights(weightpars(Ne = Ne, Ni = Ni))
# excitatory tuning
assemblymembers = initassemblymembers(Nassemblies = Nass,Ne = Ne)
# inhibitory tuning
inhibassemblies = initinhibassemblymembers(Nassemblies = Nass, Ne = Ne, Ni = Ni)
winit = copy(weights) # make a copy of initial weights as weights are updated by simulation

# --------------------- store relevant initialisation parameters -------------------------------------------

h5write(savefile, "initial/stimulus", stimulus)
h5write(savefile, "initial/lengthpretrain", lenpretrain)#
h5write(savefile, "initial/stimparams", stimparams)
h5write(savefile, "initial/stimparams_prestim", stimparams_prestim)
h5write(savefile, "initial/seqnumber", seqnumber)
h5write(savefile, "initial/idxblockonset", blockonset)
h5write(savefile, "initial/weights", weights)
h5write(savefile, "initial/assemblymembers", assemblymembers)
h5write(savefile, "initial/inhibassemblies", inhibassemblies)

h5write(savefile, "params/T", T)
h5write(savefile, "params/Ne", Ne)
h5write(savefile, "params/Ni", Ni)




# Define storage decisions
Ndec = 1
storagedec = zeros(Bool,Ndec)
storagetimes = ones(Int,Ndec)*1000

# Stroage decisions
storagedec[1] = true  # store times when spikes occured

# --------------------- initialise spiketime matrix ID - t  -----------------------------------------

Tstore = 1000 # average duration of one spiketime matrix
avgrate = 10 # Hz
Nspikesmax = Ncells*Tstore*avgrate/1000 # Nr. neurons x run time in seconds x avg. firing rate
spiket = zeros(Int32,Int(Nspikesmax),2)


if Ntrain == 0 # switch back to one as it is used in run simulation
	lenpretrain = 1
end
# --------------------- run simulation ---------------------------------------------------------------
# precompile simulation function (speed up)
@time runsimulation_inhibtuning(ifiSTDP,ifwadapt,stimparams,stimulus, weights,
assemblymembers, spiket, storagedec, storagetimes,savefile, lenpretrain, inhibassemblies,
 adjustfactor = adjustfactor, adjustfactorinhib = adjustfactorinhib,
   inhibfactor = inhibfactor, T =1)
# run main simulation
@time totalspikes = runsimulation_inhibtuning(ifiSTDP,ifwadapt,stimparams,stimulus,
 weights, assemblymembers, spiket, storagedec, storagetimes, savefile,lenpretrain,
 inhibassemblies, adjustfactor = adjustfactor, adjustfactorinhib = adjustfactorinhib,
  inhibfactor = inhibfactor, T = T)

# ---------------------- store final params ---------------------------------------------------------
h5write(savefile, "postsim/weights", weights)
h5write(savefile, "postsim/spiketimes", spiket)
h5write(savefile, "postsim/totalspikes", totalspikes)

# start evaluation
include("initevalimmediate.jl")
