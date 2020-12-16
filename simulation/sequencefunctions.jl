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

"""seqeunce generation functions
	- generate sequences each stimulus gets a number which corresponds to stimulated assemblies
	TODO:  further clean up and doc strings
"""


function genfirstsequence!(stim::Array{Float64,2}; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
	"""generate a stimulus sequence
	where each assembly is
	Nimg Number of images per sequence
	Nreps Number of seqeunce repititions in one block
	Nseq Number of sequences
	Nblocks Number of block repititions
"""
    stim[1,1] = 1
    stim[1,2] = stimstart
    stim[1,3] = stimstart + lenstim
    stim[1,4] = strength

    for i = 2:Nimg*Nreps #number in image times
        stim[i,1] = i % Nimg
        if stim[i,1] == 0
            stim[i,1] = Nimg
        end
        # lenstim ms stimulation and lenpause ms wait time
        stim[i,2] = stim[i-1,2] + lenstim + lenpause
        stim[i,3] = stim[i,2]+lenstim
        stim[i,4] = strength
    end

    return stim
end # function genstim


function gennextsequence!(stim::Array{Float64,2}, firstimg; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
	"""generate a stimulus sequence
	where each assembly is
	Nimg Number of images per sequence
	Nreps Number of seqeunce repititions in one block
	Nseq Number of sequences
	Nblocks Number of block repititions"""
	Nstim = Nimg*Nreps
	tempstim = zeros(Nstim, 4)
    tempstim[1,1] = firstimg
	if stim[end,3] == 0
		tempstim[1,2] = stimstart
	else
    	tempstim[1,2] = stim[end,3]+lenpause
	end
    tempstim[1,3] = tempstim[1,2] + lenstim
    tempstim[1,4] = strength

    for i = 2:Nimg*Nreps #number in image times
        tempstim[i,1] = (i % Nimg)
        if tempstim[i,1] == 0
            tempstim[i,1] = Nimg + firstimg - 1
		else
			tempstim[i,1] += firstimg - 1
        end
        # lenstim ms stimulation and lenpause ms wait time
        tempstim[i,2] = tempstim[i-1,2] + lenstim + lenpause
        tempstim[i,3] = tempstim[i,2]+lenstim
        tempstim[i,4] = strength
    end
	if stim[end,3] == 0
		stim = tempstim
	else
		stim = vcat(stim, tempstim)
	end
	#tempstim = nothing
    return stim
end # function genstim




function genstimparadigm(stimulus; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
	""" generate the stimulation paradigm """
	firstimages = collect(1:Nimg:Nimg*Nseq)



	# for img in firstimages
	# 	stimulus = gennextsequence!(stimulus, img)
	# end
	for b = 1:Nblocks
		if b == 1
			for img in firstimages
				stimulus = gennextsequence!(stimulus, img, Nreps = Nreps)
			end
		else
			for img in shuffle(firstimages)
				stimulus = gennextsequence!(stimulus, img, Nreps = Nreps)
			end
		end
	end

	return stimulus

end


function genstimparadigmnovel(stimulus; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
	""" generate the stimulation paradigm with a novel stimulus"""

	firstimages = collect(1:Nimg:Nimg*Nseq)
	novelimg = firstimages .+ Nimg*Nseq
	novelrep = collect(Nimg*Nseq:Nimg*Nseq+Nimg).+1

	# for img in firstimages
	# 	stimulus = gennextsequence!(stimulus, img)
	# end
	storefirstimages = copy(firstimages)
#	println("end $(storefirstimages[end])")
	blockonset = []

	for b = 1:Nblocks
		#println("block $(b) -------------------")
		if b == 1
			for img in firstimages
				append!(blockonset, size(stimulus,1) + 1)
				stimulus = gennextsequencenovel(stimulus, img, img + Nseq * Nimg, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
			end
		else
			#println("end $(storefirstimages[end])")
			shuffleimg = shuffle(firstimages)
			#println("begin $(shuffleimg[1])")
			while shuffleimg[1] == storefirstimages[end]
				println("shuffle again")
				shuffleimg = shuffle(firstimages)
				#println("new begin $(shuffleimg[1])")
			end
			storefirstimages = vcat(storefirstimages, shuffleimg)
			#println(storefirstimages)
			for img in shuffleimg  # since we shuffle ensure that it is still the correct novel image
				append!(blockonset, size(stimulus,1) + 1)
				stimulus = gennextsequencenovel(stimulus, img, img + Nseq * Nimg,Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
			end
		end
	end
	# ensure mapping from novel img to right value despite shuffling
	for i = 1:length(novelimg)
		stimulus[stimulus[:,1] .== novelimg[i],1] .= novelrep[i]
	end

	# ensure that blockonset 1 when we start with an emptz array
	if blockonset[1] == 2
		blockonset[1] = 1
	end
	return stimulus, convert(Array{Int64,1}, blockonset)

end

function gennextsequencenovel(stim::Array{Float64,2}, firstimg, novelimg; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
	"""generate a stimulus sequence with novel stim
	Nimg Number of images per sequence
	Nreps Number of seqeunce repititions in one block
	Nseq Number of sequences
	Nblocks Number of block repititions"""
	Nstim = Nimg*Nreps
	tempstim = zeros(Nstim, 4)
    #tempstim[1,1] = firstimg
	# repeat sequence several times, i.e. assemblie numbers 123412341234...
	tempstim[:,1] = repeat(firstimg:(firstimg+Nimg-1), outer = Nreps)
	if stim[end,3] == 0
		tempstim[1,2] = stimstart
	else
    	tempstim[1,2] = stim[end,3]+lenpause
	end
    tempstim[1,3] = tempstim[1,2] + lenstim
    tempstim[1,4] = strength

    for i = 2:Nimg*Nreps #number in image times
        # # lenstim ms stimulation and lenpause ms wait time
        tempstim[i,2] = tempstim[i-1,2] + lenstim + lenpause
        tempstim[i,3] = tempstim[i,2]+lenstim
        tempstim[i,4] = strength
    end
	tempstim[end-Nimg,1] = novelimg
	if stim[end,3] == 0
		stim = tempstim
	else
		stim = vcat(stim, tempstim)
	end
	#tempstim = nothing
    return stim
end # function genstim



function gennextshuffledsequencenovel(stim::Array{Float64,2}, firstimg, novelimg; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
	"""generate a stimulus sequence that is shuffled
	Nimg Number of images per sequence
	Nreps Number of seqeunce repititions in one block
	Nseq Number of sequences
	Nblocks Number of block repititions"""
	Nstim = Nimg*Nreps
	tempstim = zeros(Nstim, 4)
	#tempstim[1,1] = firstimg
	# repeat sequence several times, i.e. assemblie numbers 123412341234...
	#tempstim[:,1] = repeat(firstimg:(firstimg+Nimg-1), outer = Nreps)
	# ensure that images are shuffled but never the same image after the other
	images = firstimg:(firstimg+Nimg-1)
	assemblysequence = copy(images)
	for rep = 2:Nreps
		shuffleimg = shuffle(images)
		#println("begin $(shuffleimg[1])")
		while shuffleimg[1] == assemblysequence[end]
			println("shuffle again")
			shuffleimg = shuffle(images)
			#println("new begin $(shuffleimg[1])")
		end
	assemblysequence = vcat(assemblysequence, shuffleimg)
	end
	tempstim[:,1] .= assemblysequence
	if stim[end,3] == 0
		tempstim[1,2] = stimstart
	else
		tempstim[1,2] = stim[end,3]+lenpause
	end
	tempstim[1,3] = tempstim[1,2] + lenstim
	tempstim[1,4] = strength

	for i = 2:Nimg*Nreps #number in image times
		# # lenstim ms stimulation and lenpause ms wait time
		tempstim[i,2] = tempstim[i-1,2] + lenstim + lenpause
		tempstim[i,3] = tempstim[i,2]+lenstim
		tempstim[i,4] = strength
	end
	tempstim[end-Nimg,1] = novelimg
	if stim[end,3] == 0
		stim = tempstim
	else
		stim = vcat(stim, tempstim)
	end
	#tempstim = nothing
	return stim
end # function genstim



function genstimparadigmnovelshuffled(stimulus; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)

	firstimages = collect(1:Nimg:Nimg*Nseq)
	novelimg = firstimages .+ Nimg*Nseq
	novelrep = collect(Nimg*Nseq:Nimg*Nseq+Nimg).+1

	# for img in firstimages
	# 	stimulus = gennextsequence!(stimulus, img)
	# end
	storefirstimages = copy(firstimages)
#	println("end $(storefirstimages[end])")
	blockonset = []
	for b = 1:Nblocks
		#println("block $(b) -------------------")
		if b == 1
			for img in firstimages
				append!(blockonset, size(stimulus,1) + 1)
				stimulus = gennextshuffledsequencenovel(stimulus, img, img + Nseq * Nimg, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
			end
		else
			#println("end $(storefirstimages[end])")
			shuffleimg = shuffle(firstimages)
			#println("begin $(shuffleimg[1])")
			while shuffleimg[1] == storefirstimages[end]
				println("shuffle again")
				shuffleimg = shuffle(firstimages)
				#println("new begin $(shuffleimg[1])")
			end
			storefirstimages = vcat(storefirstimages, shuffleimg)
			#println(storefirstimages)
			for img in shuffleimg  # since we shuffle ensure that it is still the correct novel image
				append!(blockonset, size(stimulus,1) + 1)
				stimulus = gennextshuffledsequencenovel(stimulus, img, img + Nseq * Nimg,Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
			end
		end
	end
	# ensure mapping from novel img to right value despite shuffling

	for i = 1:length(novelimg)
		stimulus[stimulus[:,1] .== novelimg[i],1] .= novelrep[i]
	end

	if blockonset[1] == 2
		blockonset[1] = 1
	end
	return stimulus, convert(Array{Int64,1}, blockonset)

end
function genstimparadigmnovelshuffledpretrain(stimulus, lenpretrain; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)


	firstimages = collect(1:Nimg:Nimg*Nseq)
	novelimg = firstimages .+ Nimg*Nseq
	novelrep = collect(Nimg*Nseq:Nimg*Nseq+Nimg).+1
	println(novelimg)
	println(novelrep)
	# for img in firstimages
	# 	stimulus = gennextsequence!(stimulus, img)
	# end
	storefirstimages = copy(firstimages)
#	println("end $(storefirstimages[end])")
	blockonset = []
	for b = 1:Nblocks
		#println("block $(b) -------------------")
		if b == 1
			for img in firstimages
				append!(blockonset, size(stimulus,1) + 1)
				stimulus = gennextshuffledsequencenovel(stimulus, img, img + Nseq * Nimg, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
			end
		else
			#println("end $(storefirstimages[end])")
			shuffleimg = shuffle(firstimages)
			#println("begin $(shuffleimg[1])")
			while shuffleimg[1] == storefirstimages[end]
				println("shuffle again")
				shuffleimg = shuffle(firstimages)
				#println("new begin $(shuffleimg[1])")
			end
			storefirstimages = vcat(storefirstimages, shuffleimg)
			#println(storefirstimages)
			for img in shuffleimg  # since we shuffle ensure that it is still the correct novel image
				append!(blockonset, size(stimulus,1) + 1)
				stimulus = gennextshuffledsequencenovel(stimulus, img, img + Nseq * Nimg,Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
			end
		end
	end
	# ensure mapping from novel img to right value despite shuffling
	# for i = 1:length(novelimg)
	# 	println(novelimg[i])
	# 	println(sum([zeros(Bool,lenpretrain-1);(stimulus[lenpretrain:end,1] .== novelimg[i])]))
	# 	print(novelrep[i])
	# 	stimulus[[zeros(Bool,lenpretrain-1);(stimulus[lenpretrain:end,1] .== novelimg[i])],1] .= novelrep[i]
	# end

	if blockonset[1] == 2
		blockonset[1] = 1
	end
	asscounter = Nseq*Nimg + 1
	for i in collect(lenpretrain+1:size(stimulus,1))
		if stimulus[i,1] > Nseq*Nimg
			stimulus[i,1] = asscounter
			asscounter += 1
		end
	end
	return stimulus, convert(Array{Int64,1}, blockonset)

end


function gennextsequenceNonovel(stim::Array{Float64,2}, firstimg, novelimg; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
	# generate a stimulus sequence
	# where each assembly is
	# Nimg Number of images per sequence
	# Nreps Number of seqeunce repititions in one block
	# Nseq Number of sequences
	# Nblocks Number of block repititions
	Nstim = Nimg*Nreps
	tempstim = zeros(Nstim, 4)
    #tempstim[1,1] = firstimg
	tempstim[:,1] = repeat(firstimg:(firstimg+Nimg-1), outer = Nreps)

	if stim[end,3] == 0
		tempstim[1,2] = stimstart
	else
    	tempstim[1,2] = stim[end,3]+lenpause
	end
    tempstim[1,3] = tempstim[1,2] + lenstim
    tempstim[1,4] = strength

    for i = 2:Nimg*Nreps #number in image times
        # lenstim ms stimulation and lenpause ms wait time
        tempstim[i,2] = tempstim[i-1,2] + lenstim + lenpause
        tempstim[i,3] = tempstim[i,2]+lenstim
        tempstim[i,4] = strength
    end
	#tempstim[end-Nimg,1] = novelimg
	if stim[end,3] == 0
		stim = tempstim
	else
		stim = vcat(stim, tempstim)
	end
	#tempstim = nothing
    return stim
end # function genstim


function genstimparadigmNonovel(stimulus; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)

	firstimages = collect(1:Nimg:Nimg*Nseq)
	novelimg = firstimages .+ Nimg*Nseq
	novelrep = collect(Nimg*Nseq:Nimg*Nseq+Nimg).+1

	# for img in firstimages
	# 	stimulus = gennextsequence!(stimulus, img)
	# end
	blockonset = []
	storefirstimages = copy(firstimages)
	for b = 1:Nblocks
		if b == 1
			for img in firstimages
				append!(blockonset, size(stimulus,1) + 1)
				stimulus = gennextsequenceNonovel(stimulus, img, img + Nseq * Nimg, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
			end
		else
			shuffleimg = shuffle(firstimages)
			while shuffleimg[1] == storefirstimages[end]
				shuffleimg = shuffle(firstimages)
			end
			storefirstimages = vcat(storefirstimages, shuffleimg)
			for img in shuffleimg # since we shuffle ensure that it is still the correct novel image
				append!(blockonset, size(stimulus,1) + 1)
				stimulus = gennextsequenceNonovel(stimulus, img, img + Nseq * Nimg,Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
			end
		end
	end
	# ensure mapping from novel img to right value despite shuffling
	for i = 1:length(novelimg)
		stimulus[stimulus[:,1] .== novelimg[i],1] .= novelrep[i]
	end

	# ensure that blockonset 1 when we start with an emptz array
	if blockonset[1] == 2
		blockonset[1] = 1
	end
	return stimulus, convert(Array{Int64,1}, blockonset)

end


function variablerepetitions(; Nimg = 4, Nreps = [3,4,6,11,21,41], Nblocks = 10, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
	# generate repetition sequence
	# Nblocks with Nimg * varying Nreps where the Nreps are shuffled
	reps = copy(Nreps)
	repetition = shuffle(reps)
	# shuffle number of repetitions in each block
	for b = 1:Nblocks-1
			shufflereps = shuffle(reps)
			# if the same previous cycle length shuffle again (avoid eg 3-3 after another)
			while shufflereps[1] == repetition[end]
				println("shuffle again")
				shufflereps = shuffle(shufflereps)
				#println("new begin $(shuffleimg[1])")
			end
			repetition = vcat(repetition, shufflereps)
	end

	println(repetition)
	return repetition
end


function variablesequencelength(; Nimg = [3,6,9,12,24], Nreps = 20, Nblocks = 10, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
	# generate sequence length sequence
	# Nblocks with varying Nimg * Nreps where the Nimg are shuffled
	imgs = copy(Nimg)
	seqlength = shuffle(Nimg)
	# shuffle number of repetitions in each block
	for b = 1:Nblocks-1
			shufflelens = shuffle(imgs)
			# if the same previous cycle length shuffle again (avoid eg 3-3 after another)
			while shufflelens[1] == seqlength[end]
				println("shuffle again")
				shufflelens = shuffle(shufflelens)
				#println("new begin $(shuffleimg[1])")
			end
			seqlength = vcat(seqlength, shufflelens)
	end

	println(seqlength)
	return seqlength
end



function genstimvariablerepetitions(stimulus; Nimg = 3, Nreps = [3,4,6,11,21,41], Nblocks = 20, stimstart = 4000, lenstim = 300, lenpause = 0, strength = 8)
	# the stimulation paradigm must be different from previous stimulation style since the sequences are not
	# repeated but each time new assemblies are selected
	# 123123123123...125124123123123125124123123123.... # make novel and last image differernt
	#  1  2  3  4     10 11 1  2  3  4  5  1  2  3 # cycle repetitions
	firstimg = 1 # always set to 1
	novelimg = 1000 # set it to a number that will not be reached by other assemblies
	# if 10000 save novelty image
	lastimg = 2000 # if
	blockonset = []
	repetitions = variablerepetitions(Nimg = Nimg, Nreps = Nreps, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
		# generate repetitions
	for rep = 1:length(repetitions)
		#if rep > 1
		append!(blockonset, size(stimulus,1) + 1)
		#end
		stimulus = gennextsequencenovel(stimulus, firstimg, novelimg, Nimg = Nimg, Nreps = repetitions[rep], stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
		stimulus[end,1] = lastimg
	end
	if blockonset[1] == 2
		blockonset[1] = 1
	end
	return stimulus, repetitions, convert(Array{Int64,1}, blockonset)

end


function genstimvariablesequencelength(stimulus; Nimg = [3,6,9,12,24], Nreps = 20, Nblocks = 20, stimstart = 4000, lenstim = 300, lenpause = 0, strength = 8)
	# the stimulation paradigm must be different from previous stimulation style since the sequences are not
	# repeated but each time new assemblies are selected
	# 123123123123...123122000123567891235678912356789...1235678912356782000 # make last image differernt no novelty
	#  1  2  3  4 ... 19 20      1       2            ...   19      20     cycle repetitions
	firstimg = 1 # always set to 1
	novelimg = 1000 # set it to a number that will not be reached by other assemblies
	# if 10000 save novelty image
	lastimg = 2000 # if
	blockonset = []
	seqlen = variablesequencelength(Nimg = Nimg, Nreps = Nreps, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)

	for rep = 1:length(seqlen)
		#if rep > 1
			append!(blockonset, size(stimulus,1) + 1)
		#end
		# set novelimage to Nimg
		stimulus = gennextsequencenovel(stimulus, firstimg, novelimg, Nimg = seqlen[rep], Nreps = Nreps, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
		stimulus[end,1] = lastimg
	end
	if blockonset[1] == 2
		blockonset[1] = 1
	end
	return stimulus, seqlen, convert(Array{Int64,1}, blockonset)

end

function genstimvariablesequencelengthnonovel(stimulus; Nimg = [3,6,9,12,24], Nreps = 20, Nblocks = 20, stimstart = 4000, lenstim = 300, lenpause = 0, strength = 8)
	# the stimulation paradigm must be different from previous stimulation style since the sequences are not
	# repeated but each time new assemblies are selected
	# 123123123123...123122000123567891235678912356789...1235678912356782000 # make last image differernt no novelty
	#  1  2  3  4 ... 19 20      1       2            ...   19      20     cycle repetitions
	firstimg = 1 # always set to 1
	novelimg = 1000 # set it to a number that will not be reached by other assemblies
	# if 10000 save novelty image
	lastimg = 2000 # if
	#blockonset = [1]
	blockonset = []
	seqlen = variablesequencelength(Nimg = Nimg, Nreps = Nreps, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)

	for rep = 1:length(seqlen)
		append!(blockonset, size(stimulus,1) + 1)
		# set novelimage to Nimg
		stimulus = gennextsequencenovel(stimulus, firstimg, seqlen[rep], Nimg = seqlen[rep], Nreps = Nreps, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
		stimulus[end,1] = lastimg
	end

	if blockonset[1] == 2
		blockonset[1] = 1
	end
	return stimulus, seqlen, convert(Array{Int64,1}, blockonset)

end




function genstimvariablerepetitions(stimulus; Nimg = 3, Nreps = [3,4,6,11,21,41], Nblocks = 20, stimstart = 4000, lenstim = 300, lenpause = 0, strength = 8)
	# the stimulation paradigm must be different from previous stimulation style since the sequences are not
	# repeated but each time new assemblies are selected
	# 123123123123...125124123123123125124123123123.... # make novel and last image differernt
	#  1  2  3  4     10 11 1  2  3  4  5  1  2  3 # cycle repetitions
	firstimg = 1 # always set to 1
	novelimg = 1000 # set it to a number that will not be reached by other assemblies
	# if 10000 save novelty image
	lastimg = 2000 # if
	blockonset = []
	repetitions = variablerepetitions(Nimg = Nimg, Nreps = Nreps, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
		# generate repetitions
	for rep = 1:length(repetitions)

		append!(blockonset, size(stimulus,1) + 1)

		stimulus = gennextsequencenovel(stimulus, firstimg, novelimg, Nimg = Nimg, Nreps = repetitions[rep], stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
		stimulus[end,1] = lastimg
	end
	if blockonset[1] == 2
		blockonset[1] = 1
	end
	return stimulus, repetitions, convert(Array{Int64,1}, blockonset)

end


#function gennextsequencepretraining(stim::Array{Float64,2}, firstimg, novelimg; Nimg = 4, Ntrain = 20, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
function genstimparadigmpretraining(stim::Array{Float64,2}; Nimg = 4, Nass = 20, Ntrain = 20, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
	# generate a stimulus sequence
	# where each assembly is
	# Nimg Number of images per sequence
	# Nreps Number of seqeunce repititions in one block
	# Nseq Number of sequences
	# Nblocks Number of block repititions

	if Ntrain != 0
		Nstim = Nass * Ntrain
		tempstim = zeros(Nstim, 4)

		#tempstim[1,1] = firstimg
		# repeat sequence several times, i.e. assemblie numbers 123412341234...
		#tempstim[:,1] = repeat(firstimg:(firstimg+Nimg-1), outer = Nreps)
		# ensure that images are shuffled but never the same image after the other
		images = 1:Nass
		assemblysequence = shuffle(copy(images))
		for rep = 2:Ntrain
			shuffleimg = shuffle(images)
			#println("begin $(shuffleimg[1])")
			while shuffleimg[1] == assemblysequence[end]
				println("shuffle again")
				shuffleimg = shuffle(images)
				#println("new begin $(shuffleimg[1])")
			end
		assemblysequence = vcat(assemblysequence, shuffleimg)
		end
		tempstim[:,1] .= assemblysequence
		if stim[end,3] == 0
			tempstim[1,2] = stimstart
		else
			tempstim[1,2] = stim[end,3]+lenpause
		end
		tempstim[1,3] = tempstim[1,2] + lenstim
		tempstim[1,4] = strength

		for i = 2:Nstim#Nimg*Nreps #number in image times
			# # lenstim ms stimulation and lenpause ms wait time
			tempstim[i,2] = tempstim[i-1,2] + lenstim + lenpause
			tempstim[i,3] = tempstim[i,2]+lenstim
			tempstim[i,4] = strength
		end
		if stim[end,3] == 0
			stim = tempstim
		else
			stim = vcat(stim, tempstim)
		end
		#tempstim = nothing
	end
	return stim
end # function genstim


function genstimparadigmnovelcont(stimulus; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
	# ensure that for each block there is a different novel image
	firstimages = collect(1:Nimg:Nimg*Nseq)
	novelimg = Nimg*Nseq + 1 # start counting from the first assembly after the core assemblies
	novelrep = collect(Nimg*Nseq:Nimg*Nseq+Nimg).+1

	# for img in firstimages
	# 	stimulus = gennextsequence!(stimulus, img)
	# end
	storefirstimages = copy(firstimages)
#	println("end $(storefirstimages[end])")
	blockonset = []

	for b = 1:Nblocks
		#println("block $(b) -------------------")
		if b == 1
			for img in firstimages
				append!(blockonset, size(stimulus,1) + 1)
				stimulus = gennextsequencenovel(stimulus, img, novelimg, Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
				novelimg += 1
			end
		else
			#println("end $(storefirstimages[end])")
			shuffleimg = shuffle(firstimages)
			#println("begin $(shuffleimg[1])")
			while shuffleimg[1] == storefirstimages[end] && Nseq > 1 # added Nseq larger 1 cause otgerwise always the same
				println("shuffle again")
				shuffleimg = shuffle(firstimages)
				#println("new begin $(shuffleimg[1])")
			end
			storefirstimages = vcat(storefirstimages, shuffleimg)
			#println(storefirstimages)
			for img in shuffleimg  # since we shuffle ensure that it is still the correct novel image
				append!(blockonset, size(stimulus,1) + 1)
				stimulus = gennextsequencenovel(stimulus, img, novelimg ,Nimg = Nimg, Nreps = Nreps, Nseq = Nseq, Nblocks = Nblocks, stimstart = stimstart, lenstim = lenstim, lenpause = lenpause, strength = strength)
				novelimg += 1
			end
		end
	end

	# ensure that blockonset 1 when we start with an emptz array
	if blockonset[1] == 2
		blockonset[1] = 1
	end
	return stimulus, convert(Array{Int64,1}, blockonset)

end



function genstimparadigmssa(stimulus; Nimg = 4, Nreps = 20, Nseq = 5, Nblocks = 1, stimstart = 1000, lenstim = 1000, lenpause = 3000, strength = 8)
	"""
	generate SSA stimulation paradigm
	A A A A A A A B A A A ... Block 1
	B B B B B 				 Block 2 ...
	Nreps 20 times
	Blocks
	"""
	Nimg = 2
	Nreps = 5
	Nseq = 5
	Nblocks = 10
	stimulus, blockonset = genstimparadigmnovelcont(stimulus, Nimg = 2, Nreps = 5, Nseq = 5, Nblocks = 10, stimstart = stimstart, lenstim = 100, lenpause = 300, strength = strength )

	#last_onset = blockonset[1]
	freq = 1
	dev = 2
	for bb =1:length(blockonset)
		#idxfreq = collect(blockonset[bb]:(blockonset[bb]+Nimg*Nreps))
		freq_start = blockonset[bb]
		freq_end = blockonset[bb]+Nimg*Nreps
		dev_end = blockonset[bb]+Nimg*Nreps
		dev_start = blockonset[bb]+Nreps
		if bb == length(blockonset)
			stimulus[freq_start:end,1] .= freq
			#println(idxfreq)
			#idxdev = collect(blockonset[bb]+Nreps:Nreps:(blockonset[bb]+Nimg*Nreps))
			stimulus[dev_start:Nreps:end,1] .= dev
			#printl
		else
			stimulus[freq_start:freq_end,1] .= freq
			#println(idxfreq)
			#idxdev = collect(blockonset[bb]+Nreps:Nreps:(blockonset[bb]+Nimg*Nreps))
			stimulus[dev_start:Nreps:dev_end,1] .= dev
			#println(idxdev)
			# for the nect block switch deviant and frequent
		end
		temp =freq
		freq = dev
		dev = temp
	end

	return stimulus, blockonset
end
