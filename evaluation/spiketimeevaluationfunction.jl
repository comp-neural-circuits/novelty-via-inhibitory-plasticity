function spiketimeevaluationfunction(filelocation,fname, flags::String; pst = 0.1, storedweights = false, storedspiketimes = true, sporderedt = true, Nt = 25, savefigures = true, showfigures = false, savematrices = true, binsize = 800)
	"""
	evaluate stored spike times and calculate the population averages
	read in spiketimes from filename:  fname (hdf5 format)
	get population averages and raster plots if selected and store them in a folder in ../results/fname

	TODO: include type defs here
	"""
	print(filelocation*fname)
    if !showfigures
        ioff()
    end

    # open h5 file - remember to close after use
    fid = h5open(filelocation*fname,"r")
	# -------------------- read in relevant parameters ---------------------------------------------------
    @time assemblymembers = read(fid["initial"]["assemblymembers"])
	Nass = size(assemblymembers,1)
    @time T = read(fid["params"]["T"])
    @time Ne = read(fid["params"]["Ne"])
	print(T," ",Ne)
	#
	if ish5dataset("initial","seqnumber",fid)
		seqnumber = read(fid["initial"]["seqnumber"])
	else
		println("Not opened")
	end

	if ish5dataset("initial","stimparams",fid)
		Nimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength  = read(fid["initial"]["stimparams"])
	else
		println("Not opened")
	end



	if ish5dataset("initial","stimparams",fid)
		stimparams  = read(fid["initial"]["stimparams"])
	else
		println("Not opened")
	end


	if ish5dataset("initial","idxblockonset",fid)
		blockonset  = read(fid["initial"]["idxblockonset"])
	else
		println("idxblockonset does not exist")
	end

	if ish5dataset("initial","lengthpretrain",fid)
		lengthpretrain  = read(fid["initial"]["lengthpretrain"])
		iflenpre = true
	else
		println("idxblockonset does not exist")
		iflenpre = true
		lengthpretrain = 1400
	end


	stimulus = read(fid["initial"]["stimulus"])


    Ncells = Ne + Ni

	identifier = flags * "lenstim$(lenstim)lenpause$(lenpause)"


    close(fid)
    # -------------------- initialise savepath --------------------------------------
    datetime = Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS")

	resultlocation = "../results/"
	# replace . with dots in identifier string
	identifier = replace(identifier, "." => "dot")
    pngend = identifier*".png"
    svgend = identifier*".pdf"
    # check if folder already exists
    # if not create this folder
    if !isdir(resultlocation*fname)
        mkdir(resultlocation*fname)
        mkdir(resultlocation*fname*"/figuresspiketime/")
    else # if it exists add datetime to figure filenames
		if !isdir(resultlocation*fname*"/figuresspiketime/")
			mkdir(resultlocation*fname*"/figuresspiketime/")
		end
        pngend = "Time"*datetime*pngend
        svgend = "Time"*datetime*svgend
    end
    savefile = resultlocation*fname*"/spiketime"*datetime*".h5"
    figurepath = resultlocation *fname* "/figuresspiketime/"

	# ------------ get spiketimes during simulation -------------------
	# determine corners where new sequences start
	dt = 0.1
	#binsize = 800
	h5write(savefile, "params/binsize", round(Int,binsize*0.1))

	# REPS EXCHANGE BY READ IN Block index begin
	lenblocktt = round(Int,Nimg*Nreps*(lenstim+lenpause)/dt) # length of one block in samples (0.1 ms)
	blockend = round(Int,T/dt-lenpause/dt) # maximum value in samples when stimulation finishes
	# beginn after training
	if iflenpre
		blockbegintt = collect(Int, round(Int,(stimulus[lengthpretrain+1,2])/dt):lenblocktt:blockend-1) # samples when new blocks start - stimulation start in 0.1 ms until end of stimulation
	else
		blockbegintt = collect(Int, round(Int,stimstart/dt):lenblocktt:blockend-1) # samples when new blocks start - stimulation start in 0.1 ms until end of stimulation
	# beginn after training
	end
	append!(blockbegintt,blockend) # block end is the last entry
	println("blockbegintt")
	println("len $(length(blockbegintt))")
	println(blockbegintt)


	# open file
	fid = h5open(filelocation*fname,"r")
	spobj = fid["dursimspikes"] # object containing all spiketimes recorded during stimulation
	spobjnames = names(spobj) # get names of groups
	Nspobj = length(spobjnames) # get nuber of groups

	# initialise size of spiketime array
	sizearr = 50000
	# read in first object and correct for actual size
	temp = read(spobj[spobjnames[1]])
	sizearr = size(temp,1)
	println("Initialise spiketime array: ")
	@time spiketimes = zeros(Int32, sizearr*Nspobj,2) # all arrays read in in this way have the same size



	println("read in spiketimes from file...")
		@time for tt = 1:Nspobj
			# loop over all objects
			spiketimename = "spiketimes$(tt)"
			startind = (tt - 1)*sizearr + 1
		  	endind = startind + sizearr - 1
		  	spiketimes[startind:endind,:] .= read(spobj[spiketimename])#vcat(spiketimes, temp)
		end

	# include final spikes
	# read in final spikes
	finalspikes = read(fid["postsim"]["spiketimes"])
	# cut vector at point where it gets repetitive from previous stored spiketime
	maxval = maximum(finalspikes[:,1])
	maxidx = maximum(findall(x->x==maxval, finalspikes[:,1]))
	finalspikes = finalspikes[1:maxidx,:]
	#concat arrays
	spiketimes = vcat(spiketimes,finalspikes)
	# get maximum spikecount
	totalspikes = read(fid["postsim"]["totalspikes"])
	maxsp, cellidmax = findmax(totalspikes) # get cell with maximal number of spikes

	# split spiketimes in adequate chunks according to sequence and block
	#Nblocks = 10
	block = zeros(Int32,Ncells,maxsp)
	totspcheck = zeros(Nseq,Nblocks,Ncells) # total spikecount
	#if length(blockbegintt) > length(seqnumber) + 2
	totspcheck_poststim = zeros(length(blockbegintt)-length(seqnumber),Ncells) # total spikecount for the blocks where stimulus was shut off
	#end
	minidx = zeros(length(blockbegintt)) # lowest index that is part of a certain block
	minidx[1] = minimum(findall(x->x>=blockbegintt[1], spiketimes[:,1])) # initialise smallest spiketime index when stimulation starts
	blockNum = 1
	seqNum = 1
	println("sequence")
	println("len $(length(seqnumber))")
	println(seqnumber)

	seqcounter::Int32 = 0

# ---------------------- loop over whole blocks -------------------------------------
	# determine minimum indinces
	@time for b = 2:length(blockbegintt)
		if b == length(blockbegintt)
			minidx[b] = size(spiketimes[:,1],1)
		else
			minidx[b] = minimum(findall(x->x>=blockbegintt[b], spiketimes[:,1]))
		end
	end

	println("minidx")
	println(minidx)
	h5write(savefile, "params/minidx", minidx)
	#h5write(savefile, "params/minidxend", minidxend)
	h5write(savefile, "params/lenblocktt", lenblocktt)
	h5write(savefile, "params/blockbegintt", blockbegintt)
	h5write(savefile, "params/seqnumber", seqnumber)
	h5write(savefile, "params/stimparams", stimparams)
	h5write(savefile, "params/assemblymembers", assemblymembers)

	for b = 2:length(blockbegintt) # for now start with 52 which is the first non
		if b <= (length(seqnumber) + 1)
			seqNum = Int(seqnumber[b-1])
		else # continue counting to avoid same name when storing - set fixed block number
			println("unstim region")
			seqcounter += 1
			seqNum = seqcounter
			blockNum = 1000
		end
		# read in novelty
		if !ish5dataset("novelty","indices$(b-1)",fid) # no novelty stored for unstimulated times
			idxnovelty = []
		else
			idxnovelty = read(fid["novelty"]["indices$(b-1)"])
		end

		# temporary spiketime array of required time window
		temp = copy(spiketimes[Int(minidx[b-1]):Int(minidx[b]),:])
		ids = zeros(Bool,size(temp,1))

		println("reset times get max spikes per neuron in seq $(seqNum) and block $(blockNum)")

		if b <= (length(seqnumber) + 1) # Change back after spont activity evaluation
			@time for cc = 1:Ncells
				ids .= temp[:,2].==cc
				#println(size(ids))
				block[cc,1:sum(ids)] .= temp[ids,1] .- blockbegintt[b-1]
				totspcheck[seqNum,blockNum,cc] = sum(ids) # total number of spikes per neuron per block per sequence
				#@time ids = nothing
				GC.gc()
				end
			println("get pop avg $(seqNum) and block $(blockNum)")
			@time getpopulationaverages(seqNum,blockNum, true,true, assemblymembers, block[:,1:Int(maximum(totspcheck[seqNum,blockNum,:]))],idxnovelty, totspcheck, savefile, figurepath, pngend,svgend,Nseq = Nseq,Nreps = Nreps, Nass = Nass,Nimg = Nimg, Ni = Ni, Ne = Ne, Ncells = Ncells, lenblocktt = lenblocktt, binsize = binsize, fontsize = 24, blockbegintt = blockbegintt[b-1])

			GC.gc()
			#
			println(b)
		else
			@time for cc = 1:Ncells
				ids .= temp[:,2].==cc
				#println(size(ids))
				block[cc,1:sum(ids)] .= temp[ids,1] .- blockbegintt[b-1]
				totspcheck_poststim[seqNum,cc] = sum(ids) # total number of spikes per neuron per block per sequence
				#@time ids = nothing
				GC.gc()
				end
			println("get pop avg postsim $(seqNum) and block $(blockNum)")
			# get population averages only without plots
			@time getpopulationaveragespoststim(seqNum,blockNum, true,true, assemblymembers, block[:,1:Int(maximum(totspcheck_poststim[seqNum,:]))],idxnovelty, totspcheck_poststim, savefile, figurepath, pngend,svgend,Nseq = Nseq,Nreps = Nreps, Nass = Nass, Ni = Ni, Nimg = Nimg, Ne = Ne, Ncells = Ncells, lenblocktt = lenblocktt, binsize = binsize, fontsize = 24, blockbegintt = blockbegintt[b-1])
		end
		println("reset block to zero")
		@time block .= 0#zeros(Int32,Ncells,maxsp)
		GC.gc()

		# iterate over blocks
		if mod(b-1,Nseq) == 0
			blockNum += 1
			println(blockNum)
			if blockNum > Nblocks # set  large blocknumber to be easily read in without confusion later
				blockNum = 1000
				#break
			end
		end
		temp = nothing
		GC.gc()
	end # loop over minidx

	#
end # function
