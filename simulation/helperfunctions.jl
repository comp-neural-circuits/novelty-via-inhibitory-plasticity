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

""" helper functions
	- weight initialisation
	- excitatory and inhibitory stimulus tuning (referred to assembly initialisation)
	TODO:  further clean up and commenting
	TODO: rename rather unfortunate abbreviation for assembly
"""




function initweights((Ne, Ni, p, Jee0, Jei0, Jie, Jii))
	""" initialise the connectivity weight matrix based on initial
	E-to-E Jee0 initial weight e to e plastic
	I-to-E Jei0 initial weight i to e plastic
	E-to-I Jie0 constant weight e to i not plastic
	I-to-I Jii constant weight i to i not plastic"""
	Ncells = Ne+Ni

	# initialise weight matrix
	# w[i,j] is the weight from pre- i to postsynaptic j neuron
	weights = zeros(Float64,Ncells,Ncells)
	weights[1:Ne,1:Ne] .= Jee0
	weights[1:Ne,(1+Ne):Ncells] .= Jie
	weights[(1+Ne):Ncells,1:Ne] .= Jei0
	weights[(1+Ne):Ncells,(1+Ne):Ncells] .= Jii
	# set diagonal elements to 0
	# for cc = 1:Ncells
	# 	weights[cc,cc] = 0
	# end
	weights[diagind(weights)] .= 0.0
	# ensure that the connection probability is only p
	weights = weights.*(rand(Ncells,Ncells) .< p)
	return weights
end

function weightpars(;Ne = 4000, Ni = 1000, p = 0.2 )
	"""Ne, Ni number of excitatory, inhibitory neurons
	p initial connection probability"""
	Jee0 = 2.86 #initial weight e to e plastic
	Jei0 = 48.7 #initial weight i to e plastic
	Jie = 1.27 #constant weight e to i not plastic
	Jii = 16.2 #constant weight i to i not plastic
	return Ne,Ni,p,Jee0,Jei0,Jie,Jii
end

function initassemblymembers(;Nassemblies = 20, pmember = .05, Nmembersmax = 300, Ne = 4000)
	"""Nassemblies number of assemblies
	pmember probability of belonging to any assembly
	Nmembersmax maximum number of neurons in a population (to set size of matrix)

	set up excitatory assemblies"""
	#seed = 1
	#Random.seed!(seed) # ensure same Assemblies when repeating
	assemblymembers = ones(Int,Nassemblies,Nmembersmax)*(-1)
	for pop = 1:Nassemblies
		members = findall(rand(Ne) .< pmember)
		assemblymembers[pop,1:length(members)] = members
	end
	#println(assemblymembers)
	return assemblymembers
end

function initinhibassemblymembers(;Nassemblies = 20, pmember = .15, Nmembersmax = 200, Ne = 4000, Ni = 1000)
	"""Nassemblies number of assemblies
	pmember probability of belonging to any assembly
	Nmembersmax maximum number of neurons in a population (to set size of matrix)

	set up inhibitory assemblies
	higher connection probability"""
	#seed = 1
	#Random.seed!(seed) # ensure same Assemblies when repeating
	inhibmembers = ones(Int,Nassemblies,Nmembersmax)*(-1)
	for pop = 1:Nassemblies
		members = findall(rand(Ni) .< pmember) .+ Ne
		inhibmembers[pop,1:length(members)] = members
	end
	#println(assemblymembers)
	return inhibmembers
end


function orientationmembers(; Nrf = 400, Ne = 4000, Nuntuned = 0, Noverlap = 0)
	"""
	calculate orientation members
	Nrf: Neurons with same receptive field
	Nuntuned: Neurons not tuned
	Noverlap: Overlap with previous assembly
	"""
	# Select numbers of orientation assemblz neurons
	# Nrf Number of neurons with same receptive field
	# TODO: Make sure cyclic orientation tuing i.e. last one has overlap of fiorst assembly

	Norient = floor(Integer,(Ne-Nuntuned)/Nrf) # number of orientation tunings

	orientationmembers = zeros(Int,Norient,Nrf)
	for o = 1:Norient
		#members = findall(rand(Ne) .< pmember)
		orientationmembers[o,:] = collect(1:1:Nrf) .+ (o-1)*Nrf .-(o-1)*Noverlap
	end
		return orientationmembers
end


function genstim!(stim::Array{Float64,2})
	"""generate a stimulus sequence
	"""
    stim[1,1] = 1
    stim[1,2] = 1000
    stim[1,3] = 2000
    stim[1,4] = 8

    for i = 2:size(stim,1)
        stim[i,1] = i % 20
        if stim[i,1] == 0
            stim[i,1] = 20
        end
        # 1 second stimulation and 3 seconds wait time
        stim[i,2] = stim[i-1,2] + 4000
        stim[i,3] = stim[i,2]+1000
        stim[i,4] = 8
    end

    return stim
end # function genstim


function genstimargs!(stim::Array{Float64,2}; stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
	"""generate a stimulus sequence"""
    stim[1,1] = 1
    stim[1,2] = stimstart
    stim[1,3] = stimstart + lenstim
    stim[1,4] = strength

    for i = 2:size(stim,1)
        stim[i,1] = i % 20
        if stim[i,1] == 0
            stim[i,1] = 20
        end
        # lenstim ms stimulation and lenpause ms wait time
        stim[i,2] = stim[i-1,2] + lenstim + lenpause
        stim[i,3] = stim[i,2]+lenstim
        stim[i,4] = strength
    end

    return stim
end # function genstim

function getassembliessequencelength(assmembers; Nblocks = 5, Nimg = [4,8,12], Ne = 4000, Nmembersmax = 300)
""" get the assemblies for the sequence length experiment"""
	counter = 0

	for i = 1:Nblocks
		for j = 1:length(Nimg)
			counter += 1
			assmembers[counter,:,:] = initassemblymembers(Nassemblies = maximum(Nimg)+1, Ne = Ne, Nmembersmax = Nmembersmax) # also consider
		end
	end
	return assmembers
end

function getassembliesvariablerepetition(assmembers; Nblocks = 5, Nreps = [15,20,25], Nimg = 4, Ne = 4000, Nmembersmax = 300)
	""" get the assemblies for the variable repetition experiment"""

	counter = 0
	for i = 1:Nblocks
		for j = 1:length(Nreps)
			counter += 1
			assmembers[counter,:,:] = initassemblymembers(Nassemblies = Nimg+1, Ne = Ne, Nmembersmax = Nmembersmax)# also consider
		end
	end
	return assmembers
end


function getinhibassembliessequencelength(assmembers; Nblocks = 5, Nimg = [4,8,12], Ne = 4000, Ni = 1000, Nmembersmax = 200)
	"""return several assembly member matrices one for each sequence block no matter how long
	+ 1 novelty assembly always select the N sequence length assemblies for each respsective block
	here for inhibitory assemblies"""
	counter = 0

	for i = 1:Nblocks
		for j = 1:length(Nimg)
			counter += 1
			assmembers[counter,:,:] = initinhibassemblymembers(Nassemblies = maximum(Nimg)+1, Ne = Ne, Ni = Ni, Nmembersmax = Nmembersmax) # also consider
		end
	end
	return assmembers
end

# function initinhibassemblymembers(;Nassemblies = 20, pmember = .15, Nmembersmax = 200, Ne = 4000, Ni = 1000)

function getinhibassembliesvariablerepetition(assmembers; Nblocks = 5, Nreps = [15,20,25], Nimg = 4, Ne = 4000, Ni = 1000, Nmembersmax = 200)
	"""return assembly member matrices for each block
	"""
	counter = 0
	for i = 1:Nblocks
		for j = 1:length(Nreps)
			counter += 1
			assmembers[counter,:,:] = initinhibassemblymembers(Nassemblies = Nimg+1, Ne = Ne, Ni = Ni, Nmembersmax = Nmembersmax)# also consider
		end
	end
	return assmembers
end
