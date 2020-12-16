# this file is part of the V1 2Ch Predictive Code project
# ----------------------------------------------------------------------------------------
# initialise the simulation parameters and start the simulation
# input: stim params
# output: hdf5 file with stimulus parameters, run results (weight matrices, spiketimes, total number of spikes)
# ----------------------------------------------------------------------------------------

# if using atom and open folder function go into main folder
#cd("./main/")

# set module load path
# include("loadpath.jl")

# include inbuilt modules
using PyPlot
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
usetrained          = false  # use previously trained network

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
#Threads.@threads for reps = 1:10 # repeat meas several times
Ncells = Ne + Ni
# Set integration timestep
dt 	= 0.1 #integration timestep in ms

ifiSTDP = true 		#  include inhibitory plasticity
ifwadapt = false	#  consider AdEx or Ex IF neurons
# --------------------- generate the stimulus --------------------------------------------

# stimulus parameters
Nimg = 1		# number of images per sequence
Nreps = parse(Int64, ARGS[1])		# number of repetitions per sequence cycle
Nseq = 1		# number of sequences
Nblocks = parse(Int64, ARGS[2])	# number of sequence block repetitions
stimstart = 4000	# start time of stimulation in ms
lenstimpre = parse(Int64, ARGS[3])	# duration of assembly stimulation per image in ms
lenpausepre = parse(Int64, ARGS[4])	# duration of no stimulation between two images in ms
strengthpre = 12		# strength of the stimulation in kHz added to baseline input
lenstim = lenstimpre		# duration of assembly stimulation per image in ms
lenpause = lenpausepre		# duration of no stimulation between two images in ms
strength = strengthpre		# strength of the stimulation in kHz added to baseline input
Ntrain = 0
Nass = (Nimg) * Nseq +  Nseq * Nblocks    # total number of assemblies Nimg * Nseq + all novelty assemblies

n_neurons = parse(Int64, ARGS[5])
SSA = false
pretrainig = false

if parse(Int64, ARGS[6]) == 1
	pretrainig = true
end

inhibfactor = parse(Int64, ARGS[7])/100 # higher resolution

# increase in inhibitory assembly heterosynaptic decrease in the input
inhibhetero = false

# 20 reps 1 block 300 lenstim 900 lenpause 200 neurons AB assembly 0 no pretraining inhibfactor 0.1
# nohup julia initsim_oddball_ARGS.jl 20 1 300 900 200 0 10 &> ../tmp/SSA_standard_condition.txt &



stimparams = [Nimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength] # store stimulus param array in hdf5 file
stimparams_prestim = [Nimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength, lenstimpre, lenpausepre, strengthpre, Ntrain] # store stimulus param array in hdf5 file

# adjustment of learning rates
adjustfactor = 1
adjustfactorinhib = adjustfactor


stimulus = zeros(1, 4)	# initialisation of the stimulus
initstim = true # if inital stimulus
if initstim
	stimulus = [1.0 1000.0 1300.0 12.0; 2.0 2000.0 2300.0 12.0;1.0 3100.0 3100.0 0] # show that initially firing rates equal
end # add dummy to ensure stimulation start at 4000
tag = "repseq.h5"
stimulus = [1.0 1000.0 1300.0 0; 2.0 2000.0 2300.0 0;1.0 4000.0-(lenpause) 4000.0-(lenpause) 0]
println(stimulus)
#stimulus = genstimparadigmpretraining(stimulus, Nass = Nass, Ntrain = Ntrain, stimstart = stimstart, lenstim = lenstimpre, lenpause = lenpausepre, strength = strengthpre)
#println(stimulus)
lenpretrain = size(stimulus,1) # returns one if empty
println("SSA")
#include("../simulation/sequencefunctions.jl")



if pretrainig
	#println(stimulus)
	stimulus = genstimparadigmpretraining(stimulus, Nass = Nass, Ntrain = Ntrain, stimstart = stimstart, lenstim = lenstimpre, lenpause = lenpausepre, strength = strengthpre)
	#println(stimulus)
	lenpretrain = size(stimulus,1)
end
if SSA
	println("SSA")
	stimulus, blockonset = genstimparadigmssanew(stimulus, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength )
	tag = "SSA.h5"
else
	println("gen via novelcont")
	stimulus, blockonset = genstimparadigmnovelcont(stimulus, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength )
	tag = "repeatedsequences.h5"
end

# TODO: fix issue with lenpretrain setting it to zero and back
# set length of pretraining to 0 if no pretraining all rest of indexing depends on it to be zero
if Ntrain == 0 && !initstim
	lenpretrain = 0
end
# get sequence order
blockidx = collect(lenpretrain + 1:Nreps*Nimg:size(stimulus,1)) # get all stimulus indices when a new sequence starts
seqnumber = stimulus[blockidx,1]
for n = 2:Nseq
    seqnumber[seqnumber .== (1 + (n-1)*Nimg)] .= n
end
# replace all novel images by 2
stimulus[stimulus[:,1].>1,1] .= 2
swap_stim = blockidx[floor(Int8,length(blockidx)./2)+1]

#println(stimulus)
ifswap = false
if ifswap
	for i in range(swap_stim,size(stimulus)[1])
		if stimulus[i,1]==1 # swap 1 and 2
			stimulus[i,1]=2
		else
			stimulus[i,1]=1
		end
	end
end
addelay = false
if addelay # ensure waiting time of 10000 before new blocks start
	for bb = 2:length(blockidx)
		stimulus[blockidx[bb]:end,2] = stimulus[blockidx[bb]:end,2] .+ 10000
		stimulus[blockidx[bb]:end,3] = stimulus[blockidx[bb]:end,3] .+ 10000
	end
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
#reducefactor = 0.5
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
		#stimulus[idx.+(lenpretrain-1),1] .= stimulus[idx.+(lenpretrain-1),1] .+ 1
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
		#stimulus[idx.+(lenpretrain-1),1] .= stimulus[idx.+(lenpretrain-1),1] .+ 1
		tag = "reducednovelty$(reducefactor)" * tag
	end

end

# simulation run time
T = stimulus[end,3]+lenpause # last stimulation time + pause duration
#T = 1.5*T
println(T)

#T = 1000
println("Simulation run time: $T")

# --------------------- initialise savefile ---------------------------------------------------------

# initialise savefile and avoid overwriting when identical parameters are used
datetime = Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS")

if usetrained
	filesavename = "SSA_$(n_neurons)_inhibtunning_$(inhibfactor)_ifinhibhetero_$(inhibhetero)_Traineddur$(T)msNblocks$(Nblocks)Ntrain$(Ntrain)lenstim$(lenstim)lenpause$(lenpause)Nreps$(Nreps)strength$(strength)wadapt$(ifwadapt)iSTDP$(ifiSTDP)RateAdjust$(adjustfactor)Time"
else
	filesavename = "SSA_$(n_neurons)_inhibtunning_$(inhibfactor)_ifinhibhetero_$(inhibhetero)__dur$(T)msNblocks$(Nblocks)Ntrain$(Ntrain)lenstim$(lenstim)lenpause$(lenpause)Nreps$(Nreps)strength$(strength)wadapt$(ifwadapt)iSTDP$(ifiSTDP)RateAdjust$(adjustfactor)Time"
end

savefile = "/gpfs/gjor/personal/schulza/data/main/sequences/"*filesavename * datetime * tag
savefile = "../data/"*filesavename * datetime * tag
println(savefile)

print(stimulus)
# --------------------- select run mode pretrained vs. untrained  -----------------------------------------

# Select previous run or initialise new simulation
if usetrained
	# Open the h5 file and read in the stored variables
	filelocation = "/gpfs/gjor/personal/schulza/data/main/sequences/"
	fname = "dur1.624e6mslenstim300lenpause0Nreps20strength8wadaptfalseiSTDPtrueTime2019-04-08-16-11-30repeatedsequences.h5"
	fid = h5open(filelocation*fname,"r")
	assemblymembers = read(fid["initial"]["assemblymembers"])
	inhibassemblies = read(fid["initial"]["inhibassemblies"])

	weights = read(fid["postsim"]["weights"])
	durweights = read(fid["dursim"]["weights"])

	close(fid)
else
	weights = initweights(weightpars(Ne = Ne, Ni = Ni))
	assemblymembers = initassemblymembers(Nassemblies = Nass,Ne = Ne)
	# inhib tuning
	inhibassemblies = initinhibassemblymembers(Nassemblies = Nass, Ne = Ne, Ni = Ni)
end
winit = copy(weights) # make a copy of initial weights as weights are updated by simulation

# Ensure SSA neuron 1 and 2 are both in assembly one and two
if n_neurons == 200 # ensure really same amount of neurons
	assemblymembers[1,:] .= -1
	assemblymembers[2,:] .= -1
end
# ensure that for two assemblies n_neurons are equal
assemblymembers[1,1:n_neurons] = collect(range(1,n_neurons))
assemblymembers[2,1:n_neurons] = collect(range(1,n_neurons))
assemblymembers = assemblymembers[1:2,:]
# correct to the true amount of assemblymembers
Nass = size(assemblymembers)[1]


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
Ndec = 5
storagedec = zeros(Bool,Ndec)
storagetimes = ones(Int,Ndec)*1000

# Stroage decisions
storagedec[1] = true  # store times when spikes occured
storagedec[2] = false # store weights
storagedec[3] = false # store average of weights
storagedec[4] = false # store membrane potential
storagedec[5] = false # store weights


# --------------------- initialise spiketime matrix ID - t  -----------------------------------------

Tstore = 1000 # average duration of one spiketime matrix
avgrate = 10 # Hz
Nspikesmax = Ncells*Tstore*avgrate/1000 # Nr. neurons x run time in seconds x avg. firing rate
spiket = zeros(Int32,Int(Nspikesmax),2)

if Ntrain == 0 && !initstim # switch lenpretrain to 1 again as rest of simulation depends on it
	lenpretrain = 1
end
# --------------------- run simulation ---------------------------------------------------------------
# precompile simulation function (speed up)
@time runsimulation_inhibtuning(ifiSTDP,ifwadapt,stimparams,stimulus, weights, assemblymembers, spiket, storagedec,
 storagetimes,savefile, lenpretrain, inhibassemblies, adjustfactor = adjustfactor,
  adjustfactorinhib = adjustfactorinhib, inhibfactor = inhibfactor, inhibhetero = inhibhetero,T =1)
# run main simulation
@time totalspikes = runsimulation_inhibtuning(ifiSTDP,ifwadapt,stimparams,stimulus, weights,
 assemblymembers, spiket, storagedec, storagetimes, savefile,lenpretrain,inhibassemblies,
  adjustfactor = adjustfactor, adjustfactorinhib = adjustfactorinhib,inhibfactor = inhibfactor,
  inhibhetero = inhibhetero, T = T)

# ---------------------- store final params ---------------------------------------------------------
h5write(savefile, "postsim/weights", weights)
h5write(savefile, "postsim/spiketimes", spiket)
h5write(savefile, "postsim/totalspikes", totalspikes)

# run evaluation
include("initevalimmediate.jl")
