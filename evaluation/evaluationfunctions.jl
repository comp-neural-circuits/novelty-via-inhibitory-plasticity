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
"""

collection of evaluation helper functions
- sorting synaptic weights according to stimulus-specific assembly membership
- useful for calculating the populaion averages
- calculating assembly and cross assembly E-to-E weights
- calculating stimulus-specific I-to-E weights
- plotting of sorted rasterplots

"""

function sortweights(members::Array{Int64,1}, nonmembers::Array{Int64,1}, weights::Array{Float64,2})
    weights = weights[[members;nonmembers], [members;nonmembers]]
    return weights
end

function getnonmembers(members::Array{Int64,1}; Ne = 4000)
	restcells = ones(Bool,Ne)
	for nn = 1:Ne
			if any(x->x==nn, members)
			restcells[nn] = false
			end
	end
	return collect(1:Ne)[restcells]
end


function getnumberofmembers!(nummems::Array{Int32,1}, asseblymembers::Array{Int64,2} ; Nass = 20)
	#asseblymembers = copy(asseblymembers)
	for mm = 1:Nass # loop over all assemblies
		#check when first -1 comes
		nummems[mm] = sum(assemblymembers[mm,:].!=-1)
	end
	return nummems
end

function getXassemblyweight(memberspre::Array{Int64,1},memberspost::Array{Int64,1}, weights::Array{Float64,2})
	Npre	= size(memberspre,1) # pre
	Npost = size(memberspost,1) # post
	submat = zeros(Npre,Npost)
	precount = 0
	for pre in memberspre
		precount += 1
		postcount = 0
		for post in memberspost
			postcount += 1
			submat[precount, postcount] = weights[pre,post]
		end
	end
	norm = sum(submat .!= 0)
	avgweight = sum(submat)/norm
	return avgweight
end


function getXassemblyweight(memberspre,memberspost, weights::Array{Float64,2})
	Npre	= size(memberspre,1) # pre
	Npost = size(memberspost,1) # post
	submat = zeros(Npre,Npost)
	precount = 0
	for pre in memberspre
		precount += 1
		postcount = 0
		for post in memberspost
			postcount += 1
			submat[precount, postcount] = weights[pre,post]
		end
	end
	norm = sum(submat .!= 0)
	avgweight = sum(submat)/norm
	return avgweight
end

function getXassemblyweight(memberspre::Array{Int32,1},memberspost::Array{Int32,1}, weights::Array{Float64,2})
	Npre	= size(memberspre,1) # pre
	Npost = size(memberspost,1) # post
	submat = zeros(Npre,Npost)
	precount = 0
	for pre in memberspre
		precount += 1
		postcount = 0
		for post in memberspost
			postcount += 1
			submat[precount, postcount] = weights[pre,post]
		end
	end
	norm = sum(submat .!= 0)
	avgweight = sum(submat)/norm
	return avgweight
end



function getItoassemblyweight(memberspre::Array{Int64,1},memberspost::Array{Int64,1}, weights::Array{Float64,2})
	Npre	= size(memberspre,1) # pre
	Npost = size(memberspost,1) # post
	submat = zeros(Npre,Npost)
	precount = 0
	for pre in memberspre
		precount += 1
		postcount = 0
		for post in memberspost
			postcount += 1
			submat[precount, postcount] = weights[pre,post]
		end
	end
	norm = sum(submat .!= 0)
	avgweight = sum(submat)/norm
	return avgweight
end

# plotting functions

function plotweightmatrix(weights::Array{Float64,2}; Nmax = 4000, Nmin = 1, maxval = 22, fontsize = 14)#, assemblymembers::Array{Int64,2},
    figure() # ensure never too many open at same time
	fig,ax = subplots()
	imshow(weights[Nmin:Nmax,Nmin:Nmax], origin="lower", cmap="bone_r", vmin = 0, vmax = maxval)
	a = collect(0:5:maxval)
	cb = colorbar(ticks=a)
	cb[:set_label](L"$ w  \; [\mathrm{pF}]$", fontsize = fontsize) #ylabel(L"$ \bar W_{inhib}  \; [\mathrm{pF}]$", fontsize = fontsize)
	cb[:ax][:set_yticklabels](string.(a),fontsize = fontsize)#ylabel(L"$ \bar W_{inhib}  \; [\mathrm{pF}]$", fontsize = fontsize)
	xlabel("time [min]", fontsize = fontsize)
    ylabel("presynaptic neuron",fontsize = fontsize)
    xlabel("postsynaptic neuron",fontsize = fontsize)
	# xticks([])
	# yticks([])
	xticks([1,250,500],fontsize = fontsize)
	yticks([1,250,500],fontsize = fontsize)
    #title("weights")
end


function plotavgweightmatrix(avgweights::Array{Float64,2}; titlestr = " ", maxval = 20, fontsize = 14)#, assemblymembers::Array{Int64,2},
    figure() # ensure never too many open at same time
	fig,ax = subplots()
	imshow(avgweights, origin="lower", cmap="bone_r",vmin = 0, vmax = maxval)
	a = collect(0:5:maxval)
	cb = colorbar(ticks=a)
	cb[:set_label](L"$ \bar w  \; [\mathrm{pF}]$", fontsize = fontsize) #ylabel(L"$ \bar W_{inhib}  \; [\mathrm{pF}]$", fontsize = fontsize)
	cb[:ax][:set_yticklabels](string.(a),fontsize = fontsize) #ylabel(L"$ \bar W_{inhib}  \; [\mathrm{pF}]$", fontsize = fontsize)
	xlabel("time [min]", fontsize = fontsize)
    ylabel("presynaptic assembly",fontsize = fontsize)
    xlabel("postsynaptic assembly",fontsize = fontsize)
	xticks([],fontsize = fontsize)
	yticks([],fontsize = fontsize)
    #title("weights")
end

function getallspiketimes(spiketimes, spobj, sizearr)
  objcount::Int64 = 0; # total numberof spikes
  @time for obj in spobj
    objcount += 1

    #temp = read(obj)
    startind = (objcount - 1)*sizearr + 1
    endind = startind + sizearr - 1
    spiketimes[startind:endind,:] .= read(obj)#vcat(spiketimes, temp)
    #println(obj)
  end
  return spiketimes
end

# functions that check if a group or dataset exists and throws error if not
# in case a second h5 file is read in one can use fid2

function ish5dataset(group, dataset,fid)
       try
       obj = fid[group][dataset]
       return true
       catch
       return false
       end
end

function ish5group(group,fid)
       try
       obj = fid[group]
       return true
       catch
       return false
       end
end


function makehist(data; Nbins = 100)
	# make histogram statistics from data
	# return edges and counts
	# requires StatsBase
	data = data[data .> 0]
	his = fit(Histogram, data, nbins = Nbins)
	edgevector = collect(his.edges[1][2:end])
	counts = his.weights
	his = nothing
	GC.gc()
	return edgevector, counts
end




function makehistbins(data; lenblocktt = 240000,binsize = 500)
	# make histogram statistics from data
	# return edges and counts
	# requires StatsBase
	# ensure that all zero elements are removed
	# artefact from the way how the data is stored not actually spikes at t = 0
	data = data[data .> 0]
	bins = 0:binsize:lenblocktt+binsize
	his = fit(Histogram, data,bins,closed=:left)
	edgevector = collect(his.edges[1][1:end-1])
	counts = his.weights
	his = nothing
	GC.gc()
	return edgevector, counts
end

function checkAssemblyOverlap(assemblymembers::Array{Int64,2}; Nass = 20)
	# get overall average assembly overlap
	# input: assemblymembers
	#		 Number of assemblies
	# only consider cross assembly overlap as in one assembly it will be 100%
	countvar = 0
	NOverlap = zeros(Nass*Nass - Nass) # consider only off diagonal
	for i = 1:Nass #targetassembly
		for n = 1:Nass
			if i == n
				continue
			end
			countvar += 1
			# get the percentage of overlapping neurons in assembly n with assembly i (ignore -1 values) amd divide by total number of assemblies in i
			NOverlap[countvar] = (length(intersect(assemblymembers[n,:], assemblymembers[i,:])) - 1)/sum(assemblymembers[i,:].>0)
			println("target $(i) origin $(n) overlap $(NOverlap[countvar])")
		end
	end
	avgOverlap = mean(NOverlap)
	stdOverlap = std(NOverlap)
	return avgOverlap, stdOverlap, NOverlap
end


function checkAssemblyTotalOverlap(assemblymembers::Array{Int64,2}; Nass = 20)
	# get overall average assembly overlap
	# input: assemblymembers
	#		 Number of assemblies
	# only consider cross assembly overlap as in one assembly it will be 100%
	NOverlap = zeros(Nass) # consider only off diagonal
	for i = 1:Nass #targetassembly
			# get the percentage of overlapping neurons with all 10 other assemblies
			NOverlap[i] = (length(intersect(assemblymembers[i,:], union(assemblymembers[setdiff(collect(1:Nass),i),:]))) - 1)/sum(assemblymembers[i,:].>0)
			println("target $(i) overlap $(NOverlap[i])")
	end
	avgOverlap = mean(NOverlap)
	stdOverlap = std(NOverlap)
	return avgOverlap, stdOverlap, NOverlap
end

function plotBlockAssRasterIndiv(seqnumber,blocknumber,rasterplot::Bool, indivsequences::Bool, assemblymembers::Array{Int64,2},block::Array{Int32,2}, idxnovelty, totspcheck::Array{Float64,3}, savefile, figurepath, pngend,svgend;Nseq = 1,Nreps = 20,Nblocks = 1, Nass = 20,Ni = 1000, Ne = 4000, Ncells = 5000,binsize = 1, fontsize = 24,Nimg = 4 , lenblocktt = 240000)
	#println("plotRastwePSTH")
	Npop = size(assemblymembers,1)
	Nmaxmembers = size(assemblymembers,2)
	if Nimg == 4
		color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange"]
	elseif Nimg == 3
		color = ["midnightblue","lightskyblue","royalblue","darkred","darksalmon", "saddlebrown","darkgreen","greenyellow","darkolivegreen","darkmagenta","thistle","indigo","darkorange","tan","sienna"]
	elseif Nimg == 5
		color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","indigo","darkred","darksalmon", "saddlebrown","lightcoral","rosybrown","darkgreen","greenyellow","darkolivegreen","chartreuse","turquoise","darkmagenta","thistle","indigo","mediumslateblue","blueviolet","darkorange","tan","sienna","orange", "brown"]
	end#color = color[1:Nimg]
	colornoass = "lightslategrey"
	# b = Int(copy(blocknumber))
	# seq = Int(copy(seqnumber))
	Slimit = maximum(totspcheck[seqnumber,blocknumber,:])#10000
	#println("seq $(seqnumber) block $(blocknumber) Slimit $Slimit")
	#block = copy(spiketimesseperate[seq,b,:,1:Int(Slimit)]) # cut it to make it more handable
	totspcheck[seqnumber,blocknumber,totspcheck[seqnumber,blocknumber,:] .> Slimit] .= Int(Slimit)
	#println("$(totspcheck[seqnumber,blocknumber,1:10])")
	GC.gc()
	if rasterplot
		h5write(savefile, "spiketimeblocks/seq$(seqnumber)block$(blocknumber)", block)
	end


	# get non members
	nonmembers = collect(1:Ne)
	members = sort(unique(assemblymembers[assemblymembers .> 0]))
	deleteat!(nonmembers, members)

	# get overlap of novelty assembly with
	# 1. previously played sequence
	Onoveltysequence = (length(intersect(idxnovelty, union(assemblymembers[1+(seqnumber-1)*Nimg:(seqnumber-1)*Nimg+Nimg,:]))))/length(idxnovelty)
	# 2. nonmembers
	Onoveltynonmembers = (length(intersect(idxnovelty, nonmembers)))/length(idxnovelty)
	# 3. all assemblymembers
	Onoveltymembers = (length(intersect(idxnovelty, members)))/length(idxnovelty)

	h5write(savefile, "noveltyoverlap/seq$(seqnumber)block$(blocknumber)", [Onoveltysequence,Onoveltynonmembers,Onoveltymembers])


	figure(figsize=(12,10))
	#println("plotting exh")
	@time eNe,cNe = makehistbins(block[1:Ne,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	ax = PyPlot.subplot(111)
	plot(eNe[1:end-1]/10000,cNe[1:end-1]/(0.0001*Ne)/binsize,label = "E", lw = 2, c = "midnightblue")
	ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
	ax[:spines]["right"][:set_color]("none")
	##println(c[100:200])
	xlim([1/10000,(lenblocktt-1)/10000])
	ylim([2,10])
	xlabel("time [s]", fontsize = fontsize)
	ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
	yticks(fontsize = fontsize);
	xticks(fontsize = fontsize);
	savefig(figurepath*"PSTHHzExcNcells$(Ncells)TimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)"*pngend)
	savefig(figurepath*"PSTHHzExcNcells$(Ncells)TimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)"*svgend)

	h5write(savefile, "ExhhistBinsec$(round(Int,binsize*0.1))edges/seq$(seqnumber)block$(blocknumber)", eNe[1:end]/10000)
	h5write(savefile, "ExhhistBinsec$(round(Int,binsize*0.1))counts/seq$(seqnumber)block$(blocknumber)", cNe[1:end]/(0.0001*Ne)/binsize)


	# plot histogram for excitatory non members

	figure(figsize=(12,10))
	#println("plotting exh")
	@time eNenm,cNenm = makehistbins(block[nonmembers,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	ax = PyPlot.subplot(111)
	plot(eNenm[1:end-1]/10000,cNenm[1:end-1]/(0.0001*length(nonmembers))/binsize,label = "E", lw = 2, c = "midnightblue")
	ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
	ax[:spines]["right"][:set_color]("none")
	##println(c[100:200])
	xlim([1/10000,(lenblocktt-1)/10000])
	#ylim([2,10])
	xlabel("time [s]", fontsize = fontsize)
	ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
	yticks(fontsize = fontsize);
	xticks(fontsize = fontsize);
	savefig(figurepath*"PSTHHzExcNonmembersTimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)NovONonMems$(Onoveltynonmembers)NovOSeq$(Onoveltysequence)"*pngend)
	savefig(figurepath*"PSTHHzExcNonmembersTimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)NovONonMems$(Onoveltynonmembers)NovOSeq$(Onoveltysequence)"*svgend)

	h5write(savefile, "NonmembershistBinsec$(round(Int,binsize*0.1))edges/seq$(seqnumber)block$(blocknumber)", eNenm[1:end]/10000)
	h5write(savefile, "NonmembershistBinsec$(round(Int,binsize*0.1))counts/seq$(seqnumber)block$(blocknumber)", cNenm[1:end]/(0.0001*length(nonmembers))/binsize)


	# plot histogram for non members excluding novelty ones

	nonovnonmems = setdiff(nonmembers, idxnovelty)

	figure(figsize=(12,10))
	#println("plotting exh")
	@time eNenmnoN,cNenmnoN = makehistbins(block[nonovnonmems,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	ax = PyPlot.subplot(111)
	plot(eNenmnoN[1:end-1]/10000,cNenmnoN[1:end-1]/(0.0001*length(nonovnonmems))/binsize,label = "E", lw = 2, c = "midnightblue")
	ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
	ax[:spines]["right"][:set_color]("none")
	##println(c[100:200])
	xlim([1/10000,(lenblocktt-1)/10000])
	#ylim([2,10])
	xlabel("time [s]", fontsize = fontsize)
	ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
	yticks(fontsize = fontsize);
	xticks(fontsize = fontsize);
	savefig(figurepath*"PSTHHzExcNonmembersnoNovTimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)NovONonMems$(Onoveltynonmembers)NovOSeq$(Onoveltysequence)"*pngend)
	savefig(figurepath*"PSTHHzExcNonmembersnoNovTimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)NovONonMems$(Onoveltynonmembers)NovOSeq$(Onoveltysequence)"*svgend)

	h5write(savefile, "NonmembershistnoNovBinsec$(round(Int,binsize*0.1))edges/seq$(seqnumber)block$(blocknumber)", eNenmnoN[1:end]/10000)
	h5write(savefile, "NonmembershistnoNovBinsec$(round(Int,binsize*0.1))counts/seq$(seqnumber)block$(blocknumber)", cNenmnoN[1:end]/(0.0001*length(nonovnonmems))/binsize)


	# plot histogram for inhibitory neurons
	figure(figsize=(12,10))
	#println("plotting inhib")
	#@time e,c = makehist(spiketimesseperate[seq,b,Ne+1:Ncells,1:Int(Slimit)][:],Nbins = lenblocktt)
	#plot(e[2:end]/10000,c[2:end]/(0.0001*Ni),"k")

	@time eNi,cNi = makehistbins(block[Ne+1:Ncells,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	ax = PyPlot.subplot(111)
	plot(eNi[1:end-1]/10000,cNi[1:end-1]/(0.0001*Ni)/binsize,label = "I", lw = 2, c = "darkred")
	ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
	ax[:spines]["right"][:set_color]("none")
	xlim([1/10000,(lenblocktt-1)/10000])
	#ylim([0,20])
	xlabel("time [s]", fontsize = fontsize)
	ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
	yticks(fontsize = fontsize);
	xticks(fontsize = fontsize);
	savefig(figurepath*"PSTHInhibNcells$(Ncells)TimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)NovONonMems$(Onoveltynonmembers)NovOSeq$(Onoveltysequence)"*pngend)
	savefig(figurepath*"PSTHInhibNcells$(Ncells)TimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)NovONonMems$(Onoveltynonmembers)NovOSeq$(Onoveltysequence)"*svgend)

	#savefig(figurepath*"PSTHExcNcells$(Nc)TimeSeq$(seq)Block$(b)binsize$(binsize)"*svgend)
	h5write(savefile, "InhibhistBinsec$(round(Int,binsize*0.1))edges/seq$(seqnumber)block$(blocknumber)", eNi[1:end]/10000)
	h5write(savefile, "InhibhistBinsec$(round(Int,binsize*0.1))counts/seq$(seqnumber)block$(blocknumber)", cNi[1:end]/(0.0001*Ni)/binsize)


	figure(figsize=(12,10))
	#println("plotting exh")
	@time eNn,cNn = makehistbins(block[idxnovelty,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	ax = PyPlot.subplot(111)
	plot(eNn[1:end-1]/10000,cNn[1:end-1]/(0.0001*length(idxnovelty))/binsize,label = "E", lw = 2, c = "slategrey")
	ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
	ax[:spines]["right"][:set_color]("none")
	##println(c[100:200])
	xlim([1/10000,(lenblocktt-1)/10000])
	#ylim([2,10])
	xlabel("time [s]", fontsize = fontsize)
	ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
	yticks(fontsize = fontsize);
	xticks(fontsize = fontsize);
	savefig(figurepath*"PSTHHzNoveltyNcells$(Ncells)TimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)NovONonMems$(Onoveltynonmembers)NovOSeq$(Onoveltysequence)"*pngend)
	savefig(figurepath*"PSTHHzNoveltyNcells$(Ncells)TimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)NovONonMems$(Onoveltynonmembers)NovOSeq$(Onoveltysequence)"*svgend)

	h5write(savefile, "NoveltyBinsec$(round(Int,binsize*0.1))edges/seq$(seqnumber)block$(blocknumber)", eNn[1:end]/10000)
	h5write(savefile, "NoveltyBinsec$(round(Int,binsize*0.1))counts/seq$(seqnumber)block$(blocknumber)", cNn[1:end]/(0.0001*length(idxnovelty))/binsize)

	indivsequences = false
	if indivsequences
		for ii  = 1:Nseq

				figure(figsize=(12,10))
				#println("plotting exh")
				neur = assemblymembers[1+(ii-1)*Nimg:ii*Nimg,:][:]
				neur = unique(neur[neur .> 0])
				#println(length(neur))
				@time e,c = makehistbins(block[neur,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
				ax = PyPlot.subplot(111)
				plot(e[1:end-1]/10000,c[1:end-1]/(0.0001*length(neur))/binsize,label = "E", lw = 2, c = "midnightblue")
				ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
				ax[:spines]["right"][:set_color]("none")
				##println(c[100:200])
				xlim([1/10000,(lenblocktt-1)/10000])
				#ylim([0,20])
				xlabel("time [s]", fontsize = fontsize)
				ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
				yticks(fontsize = fontsize);
				xticks(fontsize = fontsize);
				savefig(figurepath*"PSTHHzExcSeq$(ii)TimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)"*pngend)
				savefig(figurepath*"PSTHHzExcSeq$(ii)TimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)"*svgend)

				#h5write(savefile, "Seq$(ii)histBinsec$(round(Int,binsize*0.1))edges/seq$(seqnumber)block$(blocknumber)", e[1:end]/10000)
				#h5write(savefile, "Seq$(ii)histBinsec$(round(Int,binsize*0.1))counts/seq$(seqnumber)block$(blocknumber)", c[1:end]/(0.0001*length(neur))/binsize)
				neur = nothing
		end
	end

	indivassemblies = false
		if indivassemblies
			for ij  = 1:8#round(Int, Nseq *Nimg) #enough to show it for first 8 assemblies

					figure(figsize=(12,10))
					#println("plotting exh")
					neura = assemblymembers[ij,:]
					##println(neura)
					neura = neura[neura .> 0]
					@time e,c = makehistbins(block[neura,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
					ax = PyPlot.subplot(111)
					plot(e[1:end-1]/10000,c[1:end-1]/(0.0001*length(neura))/binsize,label = "E", lw = 2, c = "midnightblue")
					ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
					ax[:spines]["right"][:set_color]("none")
					##println(c[100:200])
					xlim([1/10000,(lenblocktt-1)/10000])
					#ylim([0,20])
					xlabel("time [s]", fontsize = fontsize)
					ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
					yticks(fontsize = fontsize);
					xticks(fontsize = fontsize);
					#title("10 ass members $(neura[1:20])")
					savefig(figurepath*"PSTHHzExcAss$(ij)TimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)"*pngend)
					savefig(figurepath*"PSTHHzExcAss$(ij)TimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)"*svgend)

					# h5write(savefile, "Ass$(ij)histBinsec$(round(Int,binsize*0.1))edges/seq$(seqnumber)block$(blocknumber)", e[1:end]/10000)
					# h5write(savefile, "Ass$(ij)histBinsec$(round(Int,binsize*0.1))counts/seq$(seqnumber)block$(blocknumber)", c[1:end]/(0.0001*length(neura))/binsize)
					neura = nothing
			end
		end



# ----------------------------------- plot raster ---------------------------------------------------------------------
rasterplot = true
	if rasterplot
		fontsize += 8
	#println("creating plot")
	figure(figsize=(20,20))
	#T = 240000
	xlim(0,(lenblocktt-1)/10000)
	ylim(0,sum(assemblymembers[1:Nass,:].>0)+Ni+length(idxnovelty))
	xlabel("time [s]", fontsize = fontsize)
	ylabel("sorted neurons", fontsize = fontsize)
	yticks([],fontsize = fontsize);
	xticks(fontsize = fontsize);



	#plot raster with the order of rows determined by population membership
	rowcount = 0


		@time for pp = 1:Int(Nass)
			print("\rpopulation ",pp)
			for cc = 1:Int(Nmaxmembers)
				if assemblymembers[pp,cc] < 1
					break
				end
				rowcount+=1
				ind = assemblymembers[pp,cc]
				vals = block[ind,1:round(Int,totspcheck[seqnumber,blocknumber,ind])]/10000
				y = rowcount*ones(length(vals))
				if pp <= Nimg*Nseq
					scatter(vals,y,s=.3,marker="o",c = color[pp],linewidths=0)

				else
					scatter(vals,y,s=.3,marker="o",c = "darkblue",linewidths=0)
				end

			end
		end

		for cc = 1:length(idxnovelty) # inhibitory cells
			rowcount+=1
			vals = block[idxnovelty[cc],1:round(Int,totspcheck[seqnumber,blocknumber,idxnovelty[cc]])]/10000
			y = rowcount*ones(length(vals))
			scatter(vals,y,s=.3,marker="o",c = colornoass,linewidths=0)
		end

		#println("inhib")
		for cc = Ne+1:Ncells # inhibitory cells
			rowcount+=1
			vals = block[cc,1:round(Int,totspcheck[seqnumber,blocknumber,cc])]/10000
			y = rowcount*ones(length(vals))
			scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
		end
		savefig(figurepath*"RasterSeq$(seqnumber)Block$(blocknumber)"*pngend)
		savefig(figurepath*"RasterSeq$(seqnumber)Block$(blocknumber)"*svgend)


		GC.gc()

	end #if rasterplot


	rasterplot = true
		if rasterplot

		#println("creating plot")
		figure(figsize=(20,20))
		#T = 240000
		xlim(0,(lenblocktt-1)/10000)
		ylim(0,sum(assemblymembers[1:Nass,:].>0)+Ni+length(idxnovelty) + length(nonmembers))
		xlabel("time [s]", fontsize = fontsize)
		ylabel("sorted neurons", fontsize = fontsize)
		yticks([],fontsize = fontsize);
		xticks(fontsize = fontsize);



		#rowcount::Integer = 0
		rowcount = 0

			@time for pp = 1:Int(Nass)
				print("\rpopulation ",pp)
				for cc = 1:Int(Nmaxmembers)
					if assemblymembers[pp,cc] < 1
						break
					end
					rowcount+=1
					ind = assemblymembers[pp,cc]
					vals = block[ind,1:round(Int,totspcheck[seqnumber,blocknumber,ind])]/10000
					y = rowcount*ones(length(vals))
					if pp <= Nimg*Nseq
						scatter(vals,y,s=.3,marker="o",c = color[pp],linewidths=0)

						#scatter(vals,y,s=.3,marker="o",c = color[mod(pp,Nimg)+1],linewidths=0)
					else
						scatter(vals,y,s=.3,marker="o",c = "darkblue",linewidths=0)
					end

				end
			end
			for cc = 1:length(idxnovelty) # inhibitory cells
				rowcount+=1
				vals = block[idxnovelty[cc],1:round(Int,totspcheck[seqnumber,blocknumber,idxnovelty[cc]])]/10000
				y = rowcount*ones(length(vals))
				scatter(vals,y,s=.3,marker="o",c = colornoass,linewidths=0)
			end

			for cc = 1:length(nonmembers) # inhibitory cells
				rowcount+=1
				vals = block[nonmembers[cc],1:round(Int,totspcheck[seqnumber,blocknumber,nonmembers[cc]])]/10000
				y = rowcount*ones(length(vals))
				scatter(vals,y,s=.3,marker="o",c = colornoass,linewidths=0)
			end

			for cc = Ne+1:Ncells # inhibitory cells
				rowcount+=1
				vals = block[cc,1:round(Int,totspcheck[seqnumber,blocknumber,cc])]/10000
				y = rowcount*ones(length(vals))
				scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
			end
			savefig(figurepath*"NonMemsRasterSeq$(seqnumber)Block$(blocknumber)"*pngend)
			savefig(figurepath*"NonMemsRasterSeq$(seqnumber)Block$(blocknumber)"*svgend)


			GC.gc()

		end #if rasterplot


			rasterplot = true
				if rasterplot

				nonovnonmems = setdiff(nonmembers, idxnovelty)
				#println("creating plot")
				figure(figsize=(20,20))
				#T = 240000
				xlim(0,(lenblocktt-1)/10000)
				ylim(0,sum(assemblymembers[1:Nass,:].>0)+Ni+length(idxnovelty) + length(nonovnonmems))
				xlabel("time [s]", fontsize = fontsize)
				ylabel("sorted neurons", fontsize = fontsize)
				yticks([],fontsize = fontsize);
				xticks(fontsize = fontsize);



				rowcount = 0


					@time for pp = 1:Int(Nass)
						print("\rpopulation ",pp)
						for cc = 1:Int(Nmaxmembers)
							if assemblymembers[pp,cc] < 1
								break
							end
							rowcount+=1
							ind = assemblymembers[pp,cc]
							#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
							vals = block[ind,1:round(Int,totspcheck[seqnumber,blocknumber,ind])]/10000
							y = rowcount*ones(length(vals))
							#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
							if pp <= Nimg*Nseq
								scatter(vals,y,s=.3,marker="o",c = color[pp],linewidths=0)

								#scatter(vals,y,s=.3,marker="o",c = color[mod(pp,Nimg)+1],linewidths=0)
							else
								scatter(vals,y,s=.3,marker="o",c = "darkblue",linewidths=0)
							end

						end
					end
					for cc = 1:length(idxnovelty) # inhibitory cells
						rowcount+=1
						#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
						vals = block[idxnovelty[cc],1:round(Int,totspcheck[seqnumber,blocknumber,idxnovelty[cc]])]/10000
						y = rowcount*ones(length(vals))
						#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
						scatter(vals,y,s=.3,marker="o",c = colornoass,linewidths=0)
					end

					for cc = 1:length(nonovnonmems) # inhibitory cells
						rowcount+=1
						#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
						vals = block[nonovnonmems[cc],1:round(Int,totspcheck[seqnumber,blocknumber,nonovnonmems[cc]])]/10000
						y = rowcount*ones(length(vals))
						#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
						scatter(vals,y,s=.3,marker="o",c = colornoass,linewidths=0)
					end

					#println("inhib")
					for cc = Ne+1:Ncells # inhibitory cells
						rowcount+=1
						#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
						vals = block[cc,1:round(Int,totspcheck[seqnumber,blocknumber,cc])]/10000
						y = rowcount*ones(length(vals))
						#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
						scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
					end
					#println(figurepath*"RasterSeq$(seq)Block$(b)"*pngend)
					savefig(figurepath*"NonMemsNoveltySepRasterSeq$(seqnumber)Block$(blocknumber)"*pngend)
					savefig(figurepath*"NonMemsNoveltySepRasterSeq$(seqnumber)Block$(blocknumber)"*svgend)

					#savefig(figurepath*"RasterSeq$(seq)Block$(b)"*svgend)

					GC.gc()

				end #if rasterplot



	if rasterplot
		#fontsize += 4
	#println("creating plot")
	figure(figsize=(20,20))
	#T = 240000
	xlim(0,(lenblocktt-1)/10000)
	ylim(0,Ncells)
	xlabel("time [s]", fontsize = fontsize)
	ylabel("unsorted neurons", fontsize = fontsize)
	yticks([],fontsize = fontsize);
	xticks(fontsize = fontsize);


	rowcount = 0


		for cc = 1:Ne # inhibitory cells
			rowcount+=1
			#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
			vals = block[cc,1:round(Int,totspcheck[seqnumber,blocknumber,cc])]/10000
			y = rowcount*ones(length(vals))
			#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
			scatter(vals,y,s=.3,marker="o",c = "darkblue",linewidths=0)
		end
		for cc = Ne+1:Ncells # inhibitory cells
			rowcount+=1
			#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
			vals = block[cc,1:round(Int,totspcheck[seqnumber,blocknumber,cc])]/10000
			y = rowcount*ones(length(vals))
			#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
			scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
		end
		savefig(figurepath*"UnsortedRasterSeq$(seqnumber)Block$(blocknumber)"*pngend)
		savefig(figurepath*"UnsortedRasterSeq$(seqnumber)Block$(blocknumber)"*svgend)


		GC.gc()

	end #if rasterplot

	return eNe[1:end]/10000,eNi[1:end]/10000, cNe[1:end]/(0.0001*Ne)/binsize, cNi[1:end]/(0.0001*Ni)/binsize
GC.gc()
end # end function plotBlockAssRasterIndiv



function plotBlockAssRasterIndivOld(seqnumber,blocknumber,rasterplot::Bool, indivsequences::Bool, assemblymembers::Array{Int64,2},block::Array{Int32,2}, totspcheck::Array{Float64,3}, savefile, figurepath, pngend,svgend;Nseq = 1,Nreps = 20,Nblocks = 1, Nass = 20,Ni = 1000, Ne = 4000, Ncells = 5000,binsize = 1,fontsize = 24,Nimg = 4 , lenblocktt = 240000)
	println("plotRastwePSTH")
	Npop = size(assemblymembers,1)
	Nmaxmembers = size(assemblymembers,2)
	if Nimg == 4
		color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange"]
	elseif Nimg == 3
		color = ["midnightblue","lightskyblue","royalblue","darkred","darksalmon", "saddlebrown","darkgreen","greenyellow","darkolivegreen","darkmagenta","thistle","indigo","darkorange","tan","sienna"]
	elseif Nimg == 5
		color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","indigo","darkred","darksalmon", "saddlebrown","lightcoral","rosybrown","darkgreen","greenyellow","darkolivegreen","chartreuse","turquoise","darkmagenta","thistle","indigo","mediumslateblue","blueviolet","darkorange","tan","sienna","orange", "brown"]
	end#color = color[1:Nimg]
	colornoass = "lightslategrey"
	# b = Int(copy(blocknumber))
	# seq = Int(copy(seqnumber))
	Slimit = maximum(totspcheck[seqnumber,blocknumber,:])#10000
	println("seq $(seqnumber) block $(blocknumber) Slimit $Slimit")
	#block = copy(spiketimesseperate[seq,b,:,1:Int(Slimit)]) # cut it to make it more handable
	totspcheck[seqnumber,blocknumber,totspcheck[seqnumber,blocknumber,:] .> Slimit] .= Int(Slimit)
	println("$(totspcheck[seqnumber,blocknumber,1:10])")
	GC.gc()
	if rasterplot
		h5write(savefile, "spiketimeblocks/seq$(seqnumber)block$(blocknumber)", block)
	end

	figure(figsize=(12,10))
	println("plotting exh")
	@time eNe,cNe = makehist(block[1:Ne,1:Int(Slimit)][:],Nbins = round(Int,lenblocktt/binsize))
	ax = PyPlot.subplot(111)
	plot(eNe[2:end]/10000,cNe[2:end]/(0.0001*Ne)/binsize,"k")
	ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
	ax[:spines]["right"][:set_color]("none")
	#println(c[100:200])
	xlim([1/10000,lenblocktt/10000])
	#ylim([0,20])
	xlabel("time [s]", fontsize = fontsize)
	ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
	yticks(fontsize = fontsize);
	xticks(fontsize = fontsize);
	savefig(figurepath*"PSTHHzExcNcells$(Ncells)TimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)"*pngend)
	h5write(savefile, "ExhhistBinsec$(round(Int,binsize*0.1))edges/seq$(seqnumber)block$(blocknumber)", eNe[1:end]/10000)
	h5write(savefile, "ExhhistBinsec$(round(Int,binsize*0.1))counts/seq$(seqnumber)block$(blocknumber)", cNe[1:end]/(0.0001*Ne)/binsize)


	figure(figsize=(12,10))
	println("plotting novelty")
	@time eNe,cNe = makehist(block[1:Ne,1:Int(Slimit)][:],Nbins = round(Int,lenblocktt/binsize))
	ax = PyPlot.subplot(111)
	plot(eNe[2:end]/10000,cNe[2:end]/(0.0001*Ne)/binsize,"k")
	ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
	ax[:spines]["right"][:set_color]("none")
	#println(c[100:200])
	xlim([1/10000,lenblocktt/10000])
	#ylim([0,20])
	xlabel("time [s]", fontsize = fontsize)
	ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
	yticks(fontsize = fontsize);
	xticks(fontsize = fontsize);
	savefig(figurepath*"PSTHHzExcNcells$(Ncells)TimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)"*pngend)
	h5write(savefile, "ExhhistBinsec$(round(Int,binsize*0.1))edges/seq$(seqnumber)block$(blocknumber)", eNe[1:end]/10000)
	h5write(savefile, "ExhhistBinsec$(round(Int,binsize*0.1))counts/seq$(seqnumber)block$(blocknumber)", cNe[1:end]/(0.0001*Ne)/binsize)

	indivsequences = true
	if indivsequences
		for ii  = 1:Nseq

				figure(figsize=(12,10))
				println("plotting exh")
				neur = assemblymembers[1+(ii-1)*Nimg:ii*Nimg,:][:]
				neur = unique(neur[neur .> 0])
				println(length(neur))
				@time e,c = makehist(block[neur,1:Int(Slimit)][:],Nbins = round(Int,lenblocktt/binsize))
				ax = PyPlot.subplot(111)
				plot(e[2:end]/10000,c[2:end]/(0.0001*length(neur))/binsize,"k")
				ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
				ax[:spines]["right"][:set_color]("none")
				#println(c[100:200])
				xlim([1/10000,lenblocktt/10000])
				#ylim([0,20])
				xlabel("time [s]", fontsize = fontsize)
				ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
				yticks(fontsize = fontsize);
				xticks(fontsize = fontsize);
				savefig(figurepath*"PSTHHzExcSeq$(ii)TimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)"*pngend)
				#h5write(savefile, "Seq$(ii)histBinsec$(round(Int,binsize*0.1))edges/seq$(seqnumber)block$(blocknumber)", e[1:end]/10000)
				#h5write(savefile, "Seq$(ii)histBinsec$(round(Int,binsize*0.1))counts/seq$(seqnumber)block$(blocknumber)", c[1:end]/(0.0001*length(neur))/binsize)
				neur = nothing
		end
	end

	indivassemblies = true
		if indivassemblies
			for ij  = 1:8#round(Int, Nseq *Nimg) #enough to show it for first 8 assemblies

					figure(figsize=(12,10))
					println("plotting exh")
					neura = assemblymembers[ij,:]
					#println(neura)
					neura = neura[neura .> 0]
					@time e,c = makehist(block[neura,1:Int(Slimit)][:],Nbins = round(Int,lenblocktt/binsize))
					ax = PyPlot.subplot(111)
					plot(e[2:end]/10000,c[2:end]/(0.0001*length(neura))/binsize,"k")
					ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
					ax[:spines]["right"][:set_color]("none")
					#println(c[100:200])
					xlim([1/10000,lenblocktt/10000])
					#ylim([0,20])
					xlabel("time [s]", fontsize = fontsize)
					ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
					yticks(fontsize = fontsize);
					xticks(fontsize = fontsize);
					#title("10 ass members $(neura[1:20])")
					savefig(figurepath*"PSTHHzExcAss$(ij)TimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)"*pngend)
					# h5write(savefile, "Ass$(ij)histBinsec$(round(Int,binsize*0.1))edges/seq$(seqnumber)block$(blocknumber)", e[1:end]/10000)
					# h5write(savefile, "Ass$(ij)histBinsec$(round(Int,binsize*0.1))counts/seq$(seqnumber)block$(blocknumber)", c[1:end]/(0.0001*length(neura))/binsize)
					neura = nothing
			end
		end



	figure(figsize=(12,10))
	println("plotting inhib")
	#@time e,c = makehist(spiketimesseperate[seq,b,Ne+1:Ncells,1:Int(Slimit)][:],Nbins = lenblocktt)
	#plot(e[2:end]/10000,c[2:end]/(0.0001*Ni),"k")

	@time eNi,cNi = makehist(block[Ne+1:Ncells,1:Int(Slimit)][:],Nbins = round(Int,lenblocktt/binsize))
	ax = PyPlot.subplot(111)
	plot(eNi[2:end]/10000,cNi[2:end]/(0.0001*Ni)/binsize,"k")
	ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
	ax[:spines]["right"][:set_color]("none")
	xlim([1/10000,lenblocktt/10000])
	#ylim([0,20])
	xlabel("time [s]", fontsize = fontsize)
	ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
	yticks(fontsize = fontsize);
	xticks(fontsize = fontsize);
	savefig(figurepath*"PSTHInhibNcells$(Ncells)TimeSeq$(seqnumber)Block$(blocknumber)binsize$(binsize)"*pngend)
	#savefig(figurepath*"PSTHExcNcells$(Nc)TimeSeq$(seq)Block$(b)binsize$(binsize)"*svgend)
	h5write(savefile, "InhibhistBinsec$(round(Int,binsize*0.1))edges/seq$(seqnumber)block$(blocknumber)", eNi[1:end]/10000)
	h5write(savefile, "InhibhistBinsec$(round(Int,binsize*0.1))counts/seq$(seqnumber)block$(blocknumber)", cNi[1:end]/(0.0001*Ni)/binsize)

# ----------------------------------- plot raster ---------------------------------------------------------------------
rasterplot = true
	if rasterplot
		fontsize += 4
	println("creating plot")
	figure(figsize=(20,20))
	#T = 240000
	xlim(0,lenblocktt/10000)
	ylim(0,sum(assemblymembers.>0)+1000)
	xlabel("time [s]", fontsize = fontsize)
	ylabel("neuron index", fontsize = fontsize)
	yticks(fontsize = fontsize);
	xticks(fontsize = fontsize);


	rowcount = 0

		@time for pp = 1:Int(Nass)
			print("\rpopulation ",pp)
			for cc = 1:Int(Nmaxmembers)
				if assemblymembers[pp,cc] < 1
					break
				end
				rowcount+=1
				ind = assemblymembers[pp,cc]
				vals = block[ind,1:round(Int,totspcheck[seqnumber,blocknumber,ind])]/10000
				y = rowcount*ones(length(vals))
				if pp <= Nimg*Nseq
					scatter(vals,y,s=.3,marker="o",c = color[pp],linewidths=0)

					#scatter(vals,y,s=.3,marker="o",c = color[mod(pp,Nimg)+1],linewidths=0)
				else
					scatter(vals,y,s=.3,marker="o",c = colornoass,linewidths=0)
				end

			end
		end

		println("inhib")
		for cc = Ne+1:Ncells # inhibitory cells
			rowcount+=1
			vals = block[cc,1:round(Int,totspcheck[seqnumber,blocknumber,cc])]/10000
			y = rowcount*ones(length(vals))
			scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
		end
		savefig(figurepath*"RasterSeq$(seqnumber)Block$(blocknumber)"*pngend)

		GC.gc()

	end #if rasterplot


	if rasterplot
		fontsize += 4
	println("creating plot")
	figure(figsize=(20,20))
	#T = 240000
	xlim(0,lenblocktt/10000)
	ylim(0,Ncells)
	xlabel("time [s]", fontsize = fontsize)
	ylabel("unsorted neurons", fontsize = fontsize)
	yticks([],fontsize = fontsize);
	xticks(fontsize = fontsize);


	rowcount = 0

		for cc = 1:Ne # inhibitory cells
			rowcount+=1
			vals = block[cc,1:round(Int,totspcheck[seqnumber,blocknumber,cc])]/10000
			y = rowcount*ones(length(vals))
			scatter(vals,y,s=.3,marker="o",c = "darkblue",linewidths=0)
		end
		for cc = Ne+1:Ncells # inhibitory cells
			rowcount+=1
			vals = block[cc,1:round(Int,totspcheck[seqnumber,blocknumber,cc])]/10000
			y = rowcount*ones(length(vals))
			scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
		end
		savefig(figurepath*"UnsortedRasterSeq$(seqnumber)Block$(blocknumber)"*pngend)

		GC.gc()

	end #if rasterplot

	return eNe[1:end]/10000,eNi[1:end]/10000, cNe[1:end]/(0.0001*Ne)/binsize, cNi[1:end]/(0.0001*Ni)/binsize

end # end function plotBlockAssRasterIndiv



function getpopulationaverages(seqnumber,blocknumber,rasterplot::Bool, indivsequences::Bool, assemblymembers::Array{Int64,2},block::Array{Int32,2}, idxnovelty, totspcheck::Array{Float64,3}, savefile, figurepath, pngend,svgend;Nseq = 1,Nreps = 20,Nblocks = 1, Nass = 20,Ni = 1000, Ne = 4000, Ncells = 5000,binsize = 1, fontsize = 24,Nimg = 4 , lenblocktt = 240000, blockbegintt = 1)
	# get the population averages of individual blocks
	# input: 	sequence and block number
	# 			assemblymembers
	# 			block: array with spiketimes and cell ID
	# 			indices of neurons stimulated during novel image
	# 			total number of spikes per neuron in this block
	# 			file name and figurepath to save data
	# output:	stored histogram arrays (bin edges and counts) with population averages of
	#				- all excitatory neurons
	#				- untargeted excitatory neurons
	#				- untargeted excitatory neurons excluding novelty neurons
	#				- novelty neurons
	#				- all inhibitory neurons


	Npop = size(assemblymembers,1)
	Nmaxmembers = size(assemblymembers,2)
	if Nimg == 4
		color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange"]
	elseif Nimg == 3
		color = ["midnightblue","lightskyblue","royalblue","darkred","darksalmon", "saddlebrown","darkgreen","greenyellow","darkolivegreen","darkmagenta","thistle","indigo","darkorange","tan","sienna"]
	elseif Nimg == 5
		color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","indigo","darkred","darksalmon", "saddlebrown","lightcoral","rosybrown","darkgreen","greenyellow","darkolivegreen","chartreuse","turquoise","darkmagenta","thistle","indigo","mediumslateblue","blueviolet","darkorange","tan","sienna","orange", "brown"]
	end
	colornoass = "lightslategrey"
	Slimit = maximum(totspcheck[seqnumber,blocknumber,:])#10000
	totspcheck[seqnumber,blocknumber,totspcheck[seqnumber,blocknumber,:] .> Slimit] .= Int(Slimit)

	GC.gc()
	println(" write block to file ")
	@time if rasterplot
		h5write(savefile, "spiketimeblocks/seq$(seqnumber)block$(blocknumber)", block)
	end


	nonmembers = collect(1:Ne)
	membersarr = assemblymembers[1:Nseq*Nimg,:]
	members = sort(unique(membersarr[membersarr .> 0]))
	#membersarr = nothing

	deleteat!(nonmembers, members)

	# get overlap of novelty assembly with
	# 1. previously played sequence
	Onoveltysequence = (length(intersect(idxnovelty, union(assemblymembers[1+(seqnumber-1)*Nimg:(seqnumber-1)*Nimg+Nimg,:]))))/length(idxnovelty)
	# 2. nonmembers
	Onoveltynonmembers = (length(intersect(idxnovelty, nonmembers)))/length(idxnovelty)
	# 3. all assemblymembers
	Onoveltymembers = (length(intersect(idxnovelty, members)))/length(idxnovelty)

	h5write(savefile, "noveltyoverlap/seq$(seqnumber)block$(blocknumber)", [Onoveltysequence,Onoveltynonmembers,Onoveltymembers])


	@time eNe,cNe = makehistbins(block[1:Ne,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "E$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNe[1:end]/10000)
	h5write(savefile, "E$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNe[1:end]/(0.0001*Ne)/binsize)

	@time eNemem,cNemem = makehistbins(block[members,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "EMem$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNemem[1:end]/10000)
	h5write(savefile, "EMem$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNemem[1:end]/(0.0001*length(members))/binsize)

	# histogram for excitatory non members

	@time eNenm,cNenm = makehistbins(block[nonmembers,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "ENonMem$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNenm[1:end]/10000)
	h5write(savefile, "ENonMem$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNenm[1:end]/(0.0001*length(nonmembers))/binsize)


	# plot histogram for non members excluding novelty ones

	nonovnonmems = setdiff(nonmembers, idxnovelty)
	@time eNenmnoN,cNenmnoN = makehistbins(block[nonovnonmems,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "ENonMemNoNov$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNenmnoN[1:end]/10000)
	h5write(savefile, "ENonMemNoNov$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNenmnoN[1:end]/(0.0001*length(nonovnonmems))/binsize)

	# plot histogram for inhibitory neurons

	@time eNi,cNi = makehistbins(block[Ne+1:Ncells,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile,  "I$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNi[1:end]/10000)
	h5write(savefile,  "I$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNi[1:end]/(0.0001*Ni)/binsize)

	# plot histogram for novelty neurons

	@time eNn,cNn = makehistbins(block[idxnovelty,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "Nov$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNn[1:end]/10000)
	h5write(savefile, "Nov$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNn[1:end]/(0.0001*length(idxnovelty))/binsize)

	GC.gc() # garbage collector
end # end function


function getpopulationaveragesRaster(seqnumber,blocknumber,rasterplot::Bool, indivsequences::Bool, assemblymembers::Array{Int64,2},block::Array{Int32,2}, idxnovelty, totspcheck::Array{Float64,3}, savefile, figurepath, pngend,svgend;Nseq = 5,Nreps = 20,Nblocks = 1, Nass = 20,Ni = 1000, Ne = 4000, Ncells = 5000,binsize = 1, fontsize = 24,Nimg = 4 , lenblocktt = 240000, blockbegintt = 1)
	# get the population averages of individual blocks
	# input: 	sequence and block number
	# 			assemblymembers
	# 			block: array with spiketimes and cell ID
	# 			indices of neurons stimulated during novel image
	# 			total number of spikes per neuron in this block
	# 			file name and figurepath to save data
	# output:	stored histogram arrays (bin edges and counts) with population averages of
	#				- all excitatory neurons
	#				- untargeted excitatory neurons
	#				- untargeted excitatory neurons excluding novelty neurons
	#				- novelty neurons
	#				- all inhibitory neurons


	Npop = size(assemblymembers,1)
	Nmaxmembers = size(assemblymembers,2)
	if Nimg == 4
		color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange","midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange"]
	elseif Nimg == 3
		color = ["midnightblue","lightskyblue","royalblue","darkred","darksalmon", "saddlebrown","darkgreen","greenyellow","darkolivegreen","darkmagenta","thistle","indigo","darkorange","tan","sienna","midnightblue","lightskyblue","royalblue","darkred","darksalmon", "saddlebrown","darkgreen","greenyellow","darkolivegreen","darkmagenta","thistle","indigo","darkorange","tan","sienna"]
	elseif Nimg == 5
		color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","indigo","darkred","darksalmon", "saddlebrown","lightcoral","rosybrown","darkgreen","greenyellow","darkolivegreen","chartreuse","turquoise","darkmagenta","thistle","indigo","mediumslateblue","blueviolet","darkorange","tan","sienna","orange", "brown","midnightblue","lightskyblue","royalblue","lightsteelblue","indigo","darkred","darksalmon", "saddlebrown","lightcoral","rosybrown","darkgreen","greenyellow","darkolivegreen","chartreuse","turquoise","darkmagenta","thistle","indigo","mediumslateblue","blueviolet","darkorange","tan","sienna","orange", "brown"]
	end
	colornoass = "lightslategrey"
	Slimit = maximum(totspcheck[seqnumber,blocknumber,:])#10000
	totspcheck[seqnumber,blocknumber,totspcheck[seqnumber,blocknumber,:] .> Slimit] .= Int(Slimit)

	GC.gc()
	println(" write block to file ")
	@time if rasterplot
		h5write(savefile, "spiketimeblocks/seq$(seqnumber)block$(blocknumber)", block)
	end


	# get non members
	nonmembers = collect(1:Ne)
	#members = sort(unique(assemblymembers[assemblymembers .> 0]))
	membersarr = assemblymembers[1:Nseq*Nimg,:]
	members = sort(unique(membersarr[membersarr .> 0]))
	deleteat!(nonmembers, members)

	# get overlap of novelty assembly with
	# 1. previously played sequence
	Onoveltysequence = (length(intersect(idxnovelty, union(assemblymembers[1+(seqnumber-1)*Nimg:(seqnumber-1)*Nimg+Nimg,:]))))/length(idxnovelty)
	# 2. nonmembers
	Onoveltynonmembers = (length(intersect(idxnovelty, nonmembers)))/length(idxnovelty)
	# 3. all assemblymembers
	Onoveltymembers = (length(intersect(idxnovelty, members)))/length(idxnovelty)

	h5write(savefile, "noveltyoverlap/seq$(seqnumber)block$(blocknumber)", [Onoveltysequence,Onoveltynonmembers,Onoveltymembers])


	@time eNe,cNe = makehistbins(block[1:Ne,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "E$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNe[1:end]/10000)
	h5write(savefile, "E$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNe[1:end]/(0.0001*Ne)/binsize)


	# histogram for excitatory non members

	@time eNenm,cNenm = makehistbins(block[nonmembers,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "ENonMem$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNenm[1:end]/10000)
	h5write(savefile, "ENonMem$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNenm[1:end]/(0.0001*length(nonmembers))/binsize)

	@time eNemem,cNemem = makehistbins(block[members,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "EMem$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNemem[1:end]/10000)
	h5write(savefile, "EMem$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNemem[1:end]/(0.0001*length(members))/binsize)

	# plot histogram for non members excluding novelty ones

	nonovnonmems = setdiff(nonmembers, idxnovelty)
	@time eNenmnoN,cNenmnoN = makehistbins(block[nonovnonmems,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "ENonMemNoNov$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNenmnoN[1:end]/10000)
	h5write(savefile, "ENonMemNoNov$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNenmnoN[1:end]/(0.0001*length(nonovnonmems))/binsize)

	# plot histogram for inhibitory neurons

	@time eNi,cNi = makehistbins(block[Ne+1:Ncells,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile,  "I$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNi[1:end]/10000)
	h5write(savefile,  "I$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNi[1:end]/(0.0001*Ni)/binsize)

	# plot histogram for novelty neurons

	@time eNn,cNn = makehistbins(block[idxnovelty,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "Nov$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNn[1:end]/10000)
	h5write(savefile, "Nov$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNn[1:end]/(0.0001*length(idxnovelty))/binsize)

	GC.gc() # garbage collector


	rasterplot = true
		if rasterplot

		nonovnonmems = setdiff(nonmembers, idxnovelty)
		#println("creating plot")
		figure(figsize=(20,20))
		#T = 240000
		xlim(0,(lenblocktt-1)/10000)
		ylim(0,sum(assemblymembers[1:Nimg*Nseq,:].>0)+Ni+length(idxnovelty) + length(nonovnonmems))
		xlabel("time [s]", fontsize = fontsize)
		ylabel("sorted neurons", fontsize = fontsize)
		yticks([],fontsize = fontsize);
		xticks(fontsize = fontsize);



		#plot raster with the order of rows determined by population membership
		#rowcount::Integer = 0
		rowcount = 0
		#color = ["k","b","g","r","k","b","g","r","k","b","g","r", "k","b","g","r", "k","b","g","r","k","b","g","r","k","b","g","r"]

		#color = #["midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue"]

			@time for pp = 1:Int(Nimg*Nseq)
				print("\rpopulation ",pp)
				for cc = 1:Int(Nmaxmembers)
					if assemblymembers[pp,cc] < 1
						break
					end
					rowcount+=1
					ind = assemblymembers[pp,cc]
					#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
					vals = block[ind,1:round(Int,totspcheck[seqnumber,blocknumber,ind])]/10000
					y = rowcount*ones(length(vals))
					#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
					if pp <= Nimg*Nseq
						scatter(vals,y,s=.3,marker="o",c = color[pp],linewidths=0)

						#scatter(vals,y,s=.3,marker="o",c = color[mod(pp,Nimg)+1],linewidths=0)
					# else
					# 	scatter(vals,y,s=.3,marker="o",c = "darkblue",linewidths=0)
					end

				end
			end
			#savefig(figurepath*"NoInhibRasterSeq$(seq)Block$(b)"*pngend)
			#savefig(figurepath*"NoInhibRasterSeq$(seq)Block$(b)"*svgend)
			for cc = 1:length(idxnovelty) # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[idxnovelty[cc],1:round(Int,totspcheck[seqnumber,blocknumber,idxnovelty[cc]])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = colornoass,linewidths=0)
			end

			for cc = 1:length(nonovnonmems) # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[nonovnonmems[cc],1:round(Int,totspcheck[seqnumber,blocknumber,nonovnonmems[cc]])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = colornoass,linewidths=0)
			end

			#println("inhib")
			for cc = Ne+1:Ncells # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[cc,1:round(Int,totspcheck[seqnumber,blocknumber,cc])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
			end
			#println(figurepath*"RasterSeq$(seq)Block$(b)"*pngend)
			savefig(figurepath*"NonMemsNoveltySepRasterSeq$(seqnumber)Block$(blocknumber)"*pngend)
			savefig(figurepath*"NonMemsNoveltySepRasterSeq$(seqnumber)Block$(blocknumber)"*svgend)

			#savefig(figurepath*"RasterSeq$(seq)Block$(b)"*svgend)

			GC.gc()

		end #if rasterplot
        if rasterplot
    		#fontsize += 4
    	#println("creating plot")
    	figure(figsize=(20,20))
    	#T = 240000
    	xlim(0,(lenblocktt-1)/10000)
    	ylim(0,Ncells)
    	xlabel("time [s]", fontsize = fontsize)
    	ylabel("unsorted neurons", fontsize = fontsize)
    	yticks([],fontsize = fontsize);
    	xticks(fontsize = fontsize);


    	#plot raster with the order of rows determined by population membership
    	#rowcount::Integer = 0
    	rowcount = 0
    	#color = ["k","b","g","r","k","b","g","r","k","b","g","r", "k","b","g","r", "k","b","g","r","k","b","g","r","k","b","g","r"]

    	#color = #["midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue"]

    		for cc = 1:Ne # inhibitory cells
    			rowcount+=1
    			#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
    			vals = block[cc,1:round(Int,totspcheck[seqnumber,blocknumber,cc])]/10000
    			y = rowcount*ones(length(vals))
    			#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
    			scatter(vals,y,s=.3,marker="o",c = "darkblue",linewidths=0)
    		end
    		for cc = Ne+1:Ncells # inhibitory cells
    			rowcount+=1
    			#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
    			vals = block[cc,1:round(Int,totspcheck[seqnumber,blocknumber,cc])]/10000
    			y = rowcount*ones(length(vals))
    			#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
    			scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
    		end
    		#println(figurepath*"RasterSeq$(seq)Block$(b)"*pngend)
    		savefig(figurepath*"UnsortedRasterSeq$(seqnumber)Block$(blocknumber)"*pngend)
    		savefig(figurepath*"UnsortedRasterSeq$(seqnumber)Block$(blocknumber)"*svgend)

    		#savefig(figurepath*"RasterSeq$(seq)Block$(b)"*svgend)

    		GC.gc()

    	end #if rasterplot
end # end function

function getpopulationaveragespoststim(seqnumber,blocknumber,rasterplot::Bool, indivsequences::Bool, assemblymembers::Array{Int64,2},block::Array{Int32,2}, idxnovelty, totspcheck::Array{Float64,2}, savefile, figurepath, pngend,svgend;Nseq = 1,Nreps = 20,Nblocks = 1, Nass = 20,Ni = 1000, Ne = 4000, Ncells = 5000,binsize = 1, fontsize = 24,Nimg = 4 , lenblocktt = 240000, blockbegintt = 1)
	# get the population averages of individual blocks
	# input: 	sequence and block number
	# 			assemblymembers
	# 			block: array with spiketimes and cell ID
	# 			indices of neurons stimulated during novel image
	# 			total number of spikes per neuron in this block
	# 			file name and figurepath to save data
	# output:	stored histogram arrays (bin edges and counts) with population averages of
	#				- all excitatory neurons
	#				- untargeted excitatory neurons
	#				- untargeted excitatory neurons excluding novelty neurons
	#				- novelty neurons
	#				- all inhibitory neurons

	#assemblymembers = asseblymembers[1:Nimg,:]
	Npop = size(assemblymembers,1)
	#Nass = Nimg
	Nmaxmembers = size(assemblymembers,2)
	if Nimg == 4
		color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange"]
	elseif Nimg == 3
		color = ["midnightblue","lightskyblue","royalblue","darkred","darksalmon", "saddlebrown","darkgreen","greenyellow","darkolivegreen","darkmagenta","thistle","indigo","darkorange","tan","sienna"]
	elseif Nimg == 5
		color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","indigo","darkred","darksalmon", "saddlebrown","lightcoral","rosybrown","darkgreen","greenyellow","darkolivegreen","chartreuse","turquoise","darkmagenta","thistle","indigo","mediumslateblue","blueviolet","darkorange","tan","sienna","orange", "brown"]
	end
		colornoass = "lightslategrey"
	Slimit = maximum(totspcheck[seqnumber,:])# reduce dimension of totspeck
	totspcheck[seqnumber,totspcheck[seqnumber,:] .> Slimit] .= Int(Slimit)

	GC.gc()
	println(" write block to file ")
	@time if rasterplot
		h5write(savefile, "poststimblocks/block$(seqnumber)", block)
		h5write(savefile, "poststimtime/block$(seqnumber)", blockbegintt*0.1/60000) # save block onset time in min

	end

	# get non members
	nonmembers = collect(1:Ne)
	members = sort(unique(assemblymembers[assemblymembers .> 0]))
	deleteat!(nonmembers, members)

	@time eNe,cNe = makehistbins(block[1:Ne,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "postE$(round(Int,binsize*0.1))msedges/block$(seqnumber)", eNe[1:end]/10000)
	h5write(savefile, "postE$(round(Int,binsize*0.1))mscounts/block$(seqnumber)", cNe[1:end]/(0.0001*Ne)/binsize)

	# histogram for excitatory members
	@time eNem,cNem = makehistbins(block[members,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "postEMem$(round(Int,binsize*0.1))msedges/block$(seqnumber)", eNem[1:end]/10000)
	h5write(savefile, "postEMem$(round(Int,binsize*0.1))mscounts/block$(seqnumber)", cNem[1:end]/(0.0001*length(members))/binsize)

	# histogram for excitatory non members
	@time eNenm,cNenm = makehistbins(block[nonmembers,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "postENonMem$(round(Int,binsize*0.1))msedges/block$(seqnumber)", eNenm[1:end]/10000)
	h5write(savefile, "postENonMem$(round(Int,binsize*0.1))mscounts/block$(seqnumber)", cNenm[1:end]/(0.0001*length(nonmembers))/binsize)

	# plot histogram for inhibitory neurons
	@time eNi,cNi = makehistbins(block[Ne+1:Ncells,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "postI$(round(Int,binsize*0.1))msedges/block$(seqnumber)", eNi[1:end]/10000)
	h5write(savefile, "postI$(round(Int,binsize*0.1))mscounts/block$(seqnumber)", cNi[1:end]/(0.0001*Ni)/binsize)

	GC.gc() # garbage collector
end # end function



function getRasterpoststim(seqnumber,blocknumber,rasterplot::Bool, indivsequences::Bool, assemblymembers::Array{Int64,2},block::Array{Int32,2}, idxnovelty, totspcheck::Array{Float64,2}, savefile, figurepath, pngend,svgend;Nseq = 1,Nreps = 20,Nblocks = 1, Nass = 20,Ni = 1000, Ne = 4000, Ncells = 5000,binsize = 1, fontsize = 24,Nimg = 4 , lenblocktt = 240000,blockbegintt = 1)
	# get the population averages of individual blocks
	#println("plotRastwePSTH")

	Npop = size(assemblymembers,1)
	Nmaxmembers = size(assemblymembers,2)
	if Nimg == 4
		color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange","midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange"]
	elseif Nimg == 3
		color = ["midnightblue","lightskyblue","royalblue","darkred","darksalmon", "saddlebrown","darkgreen","greenyellow","darkolivegreen","darkmagenta","thistle","indigo","darkorange","tan","sienna","midnightblue","lightskyblue","royalblue","darkred","darksalmon", "saddlebrown","darkgreen","greenyellow","darkolivegreen","darkmagenta","thistle","indigo","darkorange","tan","sienna"]
	elseif Nimg == 5
		color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","indigo","darkred","darksalmon", "saddlebrown","lightcoral","rosybrown","darkgreen","greenyellow","darkolivegreen","chartreuse","turquoise","darkmagenta","thistle","indigo","mediumslateblue","blueviolet","darkorange","tan","sienna","orange", "brown","midnightblue","lightskyblue","royalblue","lightsteelblue","indigo","darkred","darksalmon", "saddlebrown","lightcoral","rosybrown","darkgreen","greenyellow","darkolivegreen","chartreuse","turquoise","darkmagenta","thistle","indigo","mediumslateblue","blueviolet","darkorange","tan","sienna","orange", "brown"]
	end
	colornoass = "lightslategrey"
	Slimit = maximum(totspcheck[seqnumber,:])# reduce dimension of totspeck
	totspcheck[seqnumber,totspcheck[seqnumber,:] .> Slimit] .= Int(Slimit)

	GC.gc()
	println(" write block to file ")
	@time if rasterplot
		h5write(savefile, "poststimblocks/block$(seqnumber)", block)
		h5write(savefile, "poststimtime/block$(seqnumber)", blockbegintt*0.1/60000) # save block onset time in min

	end

	# get non members
	nonmembers = collect(1:Ne)
	members = sort(unique(assemblymembers[assemblymembers .> 0]))
	deleteat!(nonmembers, members)

	@time eNe,cNe = makehistbins(block[1:Ne,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "postE$(round(Int,binsize*0.1))msedges/block$(seqnumber)", eNe[1:end]/10000)
	h5write(savefile, "postE$(round(Int,binsize*0.1))mscounts/block$(seqnumber)", cNe[1:end]/(0.0001*Ne)/binsize)

	# histogram for excitatory members
	@time eNem,cNem = makehistbins(block[members,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "postEMem$(round(Int,binsize*0.1))msedges/block$(seqnumber)", eNem[1:end]/10000)
	h5write(savefile, "postEMem$(round(Int,binsize*0.1))mscounts/block$(seqnumber)", cNem[1:end]/(0.0001*length(members))/binsize)

	# histogram for excitatory non members
	@time eNenm,cNenm = makehistbins(block[nonmembers,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "postENonMem$(round(Int,binsize*0.1))msedges/block$(seqnumber)", eNenm[1:end]/10000)
	h5write(savefile, "postENonMem$(round(Int,binsize*0.1))mscounts/block$(seqnumber)", cNenm[1:end]/(0.0001*length(nonmembers))/binsize)

	# plot histogram for inhibitory neurons
	@time eNi,cNi = makehistbins(block[Ne+1:Ncells,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "postI$(round(Int,binsize*0.1))msedges/block$(seqnumber)", eNi[1:end]/10000)
	h5write(savefile, "postI$(round(Int,binsize*0.1))mscounts/block$(seqnumber)", cNi[1:end]/(0.0001*Ni)/binsize)



# ----------------------------------- plot raster ---------------------------------------------------------------------
	Nass = Nimg * Nseq
	rasterplot = true
		if rasterplot
			fontsize += 8
		#println("creating plot")
		figure(figsize=(20,20))
		#T = 240000
		xlim(0,(lenblocktt-1)/10000)
		ylim(0,sum(assemblymembers[1:Nass,:].>0)+ Ni + length(nonmembers))
		xlabel("time [s]", fontsize = fontsize)
		ylabel("sorted neurons", fontsize = fontsize)
		yticks([],fontsize = fontsize);
		xticks(fontsize = fontsize);

		rowcount = 0
			@time for pp = 1:Int(Nass)
				print("\rpopulation ",pp)
				for cc = 1:Int(Nmaxmembers)
					if assemblymembers[pp,cc] < 1
						break
					end
					rowcount+=1
					ind = assemblymembers[pp,cc]
					#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
					vals = block[ind,1:round(Int,totspcheck[seqnumber,ind])]/10000
					y = rowcount*ones(length(vals))
					#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
					if pp <= Nimg*Nseq
						scatter(vals,y,s=.3,marker="o",c = color[pp],linewidths=0)

						#scatter(vals,y,s=.3,marker="o",c = color[mod(pp,Nimg)+1],linewidths=0)
					else
						scatter(vals,y,s=.3,marker="o",c = "darkblue",linewidths=0)
					end

				end
			end

			for cc = 1:length(nonmembers) # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[nonmembers[cc],1:round(Int,totspcheck[seqnumber,nonmembers[cc]])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = colornoass,linewidths=0)
			end

			#println("inhib")
			for cc = Ne+1:Ncells # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[cc,1:round(Int,totspcheck[seqnumber,cc])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
			end
			#println(figurepath*"RasterSeq$(seq)Block$(b)"*pngend)
			savefig(figurepath*"PostStimRasterSeq$(seqnumber)"*pngend)
			savefig(figurepath*"PostStimRasterSeq$(seqnumber)"*svgend)

			#savefig(figurepath*"RasterSeq$(seq)Block$(b)"*svgend)

			GC.gc()

		end #if rasterplot

	if rasterplot
		#fontsize += 4
	#println("creating plot")
	figure(figsize=(20,20))
	#T = 240000
	xlim(0,(lenblocktt-1)/10000)
	ylim(0,Ncells)
	xlabel("time [s]", fontsize = fontsize)
	ylabel("unsorted neurons", fontsize = fontsize)
	yticks([],fontsize = fontsize);
	xticks(fontsize = fontsize);

	rowcount = 0

		for cc = 1:Ne # inhibitory cells
			rowcount+=1
			#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
			vals = block[cc,1:round(Int,totspcheck[seqnumber,cc])]/10000
			y = rowcount*ones(length(vals))
			#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
			scatter(vals,y,s=.3,marker="o",c = "darkblue",linewidths=0)
		end
		for cc = Ne+1:Ncells # inhibitory cells
			rowcount+=1
			#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
			vals = block[cc,1:round(Int,totspcheck[seqnumber,cc])]/10000
			y = rowcount*ones(length(vals))
			#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
			scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
		end
		#println(figurepath*"RasterSeq$(seq)Block$(b)"*pngend)
		savefig(figurepath*"PostStimUnsortedRasterSeq$(seqnumber)"*pngend)
		savefig(figurepath*"PostStimUnsortedRasterSeq$(seqnumber)"*svgend)

		#savefig(figurepath*"RasterSeq$(seq)Block$(b)"*svgend)

		GC.gc()

	end #if rasterplot

GC.gc()

end # end function getRasterpoststim

function plotRastervariablerepetitions(repnumber,blocknumber,rasterplot::Bool, indivsequences::Bool, assemblymembers::Array{Int64,2},block::Array{Int32,2}, idxnovelty, totspcheck::Array{Float64,3}, savefile, figurepath, pngend,svgend;Nseq = 1,Nreps = 20,Nblocks = 1, Nass = 20,Ni = 1000, Ne = 4000, Ncells = 5000,binsize = 1, fontsize = 24,Nimg = 4 , lenblocktt = 240000)
	#println("plotRastwePSTH")
	assemblymembers = assemblymembers[1:Nimg,:] # here always only the currently stimulated assemblies are considered as members
	#println(assemblymembers)
	Npop = size(assemblymembers,1)
	Nass = Nimg
	Nmaxmembers = size(assemblymembers,2)

	repnumberidx = findall(Nreps .== repnumber)[1] # get the index of the variable for totspcheck
	if Nimg == 4
		color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange"]
	elseif Nimg == 3
		color = ["midnightblue","lightskyblue","royalblue","darkred","darksalmon", "saddlebrown","darkgreen","greenyellow","darkolivegreen","darkmagenta","thistle","indigo","darkorange","tan","sienna"]
	elseif Nimg == 5
		color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","indigo","darkred","darksalmon", "saddlebrown","lightcoral","rosybrown","darkgreen","greenyellow","darkolivegreen","chartreuse","turquoise","darkmagenta","thistle","indigo","mediumslateblue","blueviolet","darkorange","tan","sienna","orange", "brown"]
	end#color = color[1:Nimg]
	colornoass = "lightslategrey"
	# b = Int(copy(blocknumber))
	# seq = Int(copy(seqnumber))
	Slimit = maximum(totspcheck[repnumberidx,blocknumber,:])#10000
	#println("seq $(seqnumber) block $(blocknumber) Slimit $Slimit")
	#block = copy(spiketimesseperate[seq,b,:,1:Int(Slimit)]) # cut it to make it more handable
	totspcheck[repnumberidx,blocknumber,totspcheck[repnumberidx,blocknumber,:] .> Slimit] .= Int(Slimit)
	#println("$(totspcheck[seqnumber,blocknumber,1:10])")
	GC.gc()
	if rasterplot
		h5write(savefile, "spiketimeblocks/reps$(repnumber)block$(blocknumber)", block)
	end



	# get non members
	nonmembers = collect(1:Ne)
	members = sort(unique(assemblymembers[assemblymembers .> 0]))
	deleteat!(nonmembers, members)

	# get overlap of novelty assembly with
	# 1. previously played sequence
	Onoveltysequence = (length(intersect(idxnovelty, union(assemblymembers))))/length(idxnovelty)
	# 2. nonmembers
	Onoveltynonmembers = (length(intersect(idxnovelty, nonmembers)))/length(idxnovelty)
	# 3. all assemblymembers
	Onoveltymembers = (length(intersect(idxnovelty, members)))/length(idxnovelty)

	h5write(savefile, "noveltyoverlap/reps$(repnumber)block$(blocknumber)", [Onoveltysequence,Onoveltynonmembers,Onoveltymembers])
	#
	# println("BLOCK in function")
	# println(block[1:2,:])

	figure(figsize=(12,10))
	#println("plotting exh")
	@time eNe,cNe = makehistbins(block[1:Ne,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	ax = PyPlot.subplot(111)
	plot(eNe[1:end-1]/10000,cNe[1:end-1]/(0.0001*Ne)/binsize,label = "E", lw = 2, c = "midnightblue")
	ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
	ax[:spines]["right"][:set_color]("none")
	##println(c[100:200])
	xlim([1/10000,(lenblocktt-1)/10000])
	ylim([2,10])
	xlabel("time [s]", fontsize = fontsize)
	ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
	yticks(fontsize = fontsize);
	xticks(fontsize = fontsize);
	savefig(figurepath*"PSTHHzExcNcells$(Ncells)TimeRep$(repnumber)Block$(blocknumber)binsize$(binsize)"*pngend)
	savefig(figurepath*"PSTHHzExcNcells$(Ncells)TimeRep$(repnumber)Block$(blocknumber)binsize$(binsize)"*svgend)

	h5write(savefile, "ExhhistBinsec$(round(Int,binsize*0.1))edges/rep$(repnumber)block$(blocknumber)", eNe[1:end]/10000)
	h5write(savefile, "ExhhistBinsec$(round(Int,binsize*0.1))counts/rep$(repnumber)block$(blocknumber)", cNe[1:end]/(0.0001*Ne)/binsize)

	#h5write(savefile, "params/binsize", round(Int,binsize*0.1))
	# plot histogram for excitatory non members

	figure(figsize=(12,10))
	#println("plotting exh")
	@time eNenm,cNenm = makehistbins(block[nonmembers,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	ax = PyPlot.subplot(111)
	plot(eNenm[1:end-1]/10000,cNenm[1:end-1]/(0.0001*length(nonmembers))/binsize,label = "E", lw = 2, c = "midnightblue")
	ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
	ax[:spines]["right"][:set_color]("none")
	##println(c[100:200])
	xlim([1/10000,(lenblocktt-1)/10000])
	#ylim([2,10])
	xlabel("time [s]", fontsize = fontsize)
	ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
	yticks(fontsize = fontsize);
	xticks(fontsize = fontsize);
	savefig(figurepath*"PSTHHzExcNonmembersTimeRep$(repnumber)Block$(blocknumber)binsize$(binsize)NovONonMems$(Onoveltynonmembers)NovOSeq$(Onoveltysequence)"*pngend)
	savefig(figurepath*"PSTHHzExcNonmembersTimeRep$(repnumber)Block$(blocknumber)binsize$(binsize)NovONonMems$(Onoveltynonmembers)NovOSeq$(Onoveltysequence)"*svgend)

	h5write(savefile, "NonmembershistBinsec$(round(Int,binsize*0.1))edges/rep$(repnumber)block$(blocknumber)", eNenm[1:end]/10000)
	h5write(savefile, "NonmembershistBinsec$(round(Int,binsize*0.1))counts/rep$(repnumber)block$(blocknumber)", cNenm[1:end]/(0.0001*length(nonmembers))/binsize)


	# plot histogram for non members excluding novelty ones

	nonovnonmems = setdiff(nonmembers, idxnovelty)

	figure(figsize=(12,10))
	#println("plotting exh")
	@time eNenmnoN,cNenmnoN = makehistbins(block[nonovnonmems,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	ax = PyPlot.subplot(111)
	plot(eNenmnoN[1:end-1]/10000,cNenmnoN[1:end-1]/(0.0001*length(nonovnonmems))/binsize,label = "E", lw = 2, c = "midnightblue")
	ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
	ax[:spines]["right"][:set_color]("none")
	##println(c[100:200])
	xlim([1/10000,(lenblocktt-1)/10000])
	#ylim([2,10])
	xlabel("time [s]", fontsize = fontsize)
	ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
	yticks(fontsize = fontsize);
	xticks(fontsize = fontsize);
	savefig(figurepath*"PSTHHzExcNonmembersnoNovTimeRep$(repnumber)Block$(blocknumber)binsize$(binsize)NovONonMems$(Onoveltynonmembers)NovOSeq$(Onoveltysequence)"*pngend)
	savefig(figurepath*"PSTHHzExcNonmembersnoNovTimeRep$(repnumber)Block$(blocknumber)binsize$(binsize)NovONonMems$(Onoveltynonmembers)NovOSeq$(Onoveltysequence)"*svgend)

	h5write(savefile, "NonmembershistnoNovBinsec$(round(Int,binsize*0.1))edges/rep$(repnumber)block$(blocknumber)", eNenmnoN[1:end]/10000)
	h5write(savefile, "NonmembershistnoNovBinsec$(round(Int,binsize*0.1))counts/rep$(repnumber)block$(blocknumber)", cNenmnoN[1:end]/(0.0001*length(nonovnonmems))/binsize)


	# plot histogram for inhibitory neurons
	figure(figsize=(12,10))
	#println("plotting inhib")
	#@time e,c = makehist(spiketimesseperate[seq,b,Ne+1:Ncells,1:Int(Slimit)][:],Nbins = lenblocktt)
	#plot(e[2:end]/10000,c[2:end]/(0.0001*Ni),"k")

	@time eNi,cNi = makehistbins(block[Ne+1:Ncells,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	ax = PyPlot.subplot(111)
	plot(eNi[1:end-1]/10000,cNi[1:end-1]/(0.0001*Ni)/binsize,label = "I", lw = 2, c = "darkred")
	ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
	ax[:spines]["right"][:set_color]("none")
	xlim([1/10000,(lenblocktt-1)/10000])
	#ylim([0,20])
	xlabel("time [s]", fontsize = fontsize)
	ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
	yticks(fontsize = fontsize);
	xticks(fontsize = fontsize);
	savefig(figurepath*"PSTHInhibNcells$(Ncells)TimeRep$(repnumber)Block$(blocknumber)binsize$(binsize)NovONonMems$(Onoveltynonmembers)NovOSeq$(Onoveltysequence)"*pngend)
	savefig(figurepath*"PSTHInhibNcells$(Ncells)TimeRep$(repnumber)Block$(blocknumber)binsize$(binsize)NovONonMems$(Onoveltynonmembers)NovOSeq$(Onoveltysequence)"*svgend)

	#savefig(figurepath*"PSTHExcNcells$(Nc)TimeSeq$(seq)Block$(b)binsize$(binsize)"*svgend)
	h5write(savefile, "InhibhistBinsec$(round(Int,binsize*0.1))edges/rep$(repnumber)block$(blocknumber)", eNi[1:end]/10000)
	h5write(savefile, "InhibhistBinsec$(round(Int,binsize*0.1))counts/rep$(repnumber)block$(blocknumber)", cNi[1:end]/(0.0001*Ni)/binsize)


	figure(figsize=(12,10))
	#println("plotting exh")
	@time eNn,cNn = makehistbins(block[idxnovelty,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	ax = PyPlot.subplot(111)
	plot(eNn[1:end-1]/10000,cNn[1:end-1]/(0.0001*length(idxnovelty))/binsize,label = "E", lw = 2, c = "slategrey")
	ax[:spines]["top"][:set_color]("none") # Remove the top axis boundary
	ax[:spines]["right"][:set_color]("none")
	##println(c[100:200])
	xlim([1/10000,(lenblocktt-1)/10000])
	#ylim([2,10])
	xlabel("time [s]", fontsize = fontsize)
	ylabel(L"\rho \; \; [Hz]", fontsize = fontsize)
	yticks(fontsize = fontsize);
	xticks(fontsize = fontsize);
	savefig(figurepath*"PSTHHzNoveltyNcells$(Ncells)TimeRep$(repnumber)Block$(blocknumber)binsize$(binsize)NovONonMems$(Onoveltynonmembers)NovOSeq$(Onoveltysequence)"*pngend)
	savefig(figurepath*"PSTHHzNoveltyNcells$(Ncells)TimeRep$(repnumber)Block$(blocknumber)binsize$(binsize)NovONonMems$(Onoveltynonmembers)NovOSeq$(Onoveltysequence)"*svgend)

	h5write(savefile, "NoveltyBinsec$(round(Int,binsize*0.1))edges/rep$(repnumber)block$(blocknumber)", eNn[1:end]/10000)
	h5write(savefile, "NoveltyBinsec$(round(Int,binsize*0.1))counts/rep$(repnumber)block$(blocknumber)", cNn[1:end]/(0.0001*length(idxnovelty))/binsize)

	# ----------------------------------- plot raster ---------------------------------------------------------------------
	rasterplot = false
		if rasterplot
			fontsize += 8
		#println("creating plot")
		figure(figsize=(20,20))
		#T = 240000
		xlim(0,(lenblocktt-1)/10000)
		ylim(0,sum(assemblymembers[1:Nass,:].>0)+Ni+length(idxnovelty))
		xlabel("time [s]", fontsize = fontsize)
		ylabel("sorted neurons", fontsize = fontsize)
		yticks([],fontsize = fontsize);
		xticks(fontsize = fontsize);



		#plot raster with the order of rows determined by population membership
		#rowcount::Integer = 0
		rowcount = 0
		#color = ["k","b","g","r","k","b","g","r","k","b","g","r", "k","b","g","r", "k","b","g","r","k","b","g","r","k","b","g","r"]

		#color = #["midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue"]

			@time for pp = 1:Int(Nass)
				print("\rpopulation ",pp)
				for cc = 1:Int(Nmaxmembers)
					if assemblymembers[pp,cc] < 1
						break
					end
					rowcount+=1
					ind = assemblymembers[pp,cc]
					#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
					vals = block[ind,1:round(Int,totspcheck[repnumberidx,blocknumber,ind])]/10000
					y = rowcount*ones(length(vals))
					#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
					if pp <= Nimg*Nseq
						scatter(vals,y,s=.3,marker="o",c = color[pp],linewidths=0)

						#scatter(vals,y,s=.3,marker="o",c = color[mod(pp,Nimg)+1],linewidths=0)
					else
						scatter(vals,y,s=.3,marker="o",c = "darkblue",linewidths=0)
					end

				end
			end
			#savefig(figurepath*"NoInhibRasterSeq$(seq)Block$(b)"*pngend)
			#savefig(figurepath*"NoInhibRasterSeq$(seq)Block$(b)"*svgend)
			for cc = 1:length(idxnovelty) # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[idxnovelty[cc],1:round(Int,totspcheck[repnumberidx,blocknumber,idxnovelty[cc]])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = colornoass,linewidths=0)
			end

			#println("inhib")
			for cc = Ne+1:Ncells # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[cc,1:round(Int,totspcheck[repnumberidx,blocknumber,cc])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
			end
			#println(figurepath*"RasterSeq$(seq)Block$(b)"*pngend)
			savefig(figurepath*"RasterRep$(repnumber)Block$(blocknumber)"*pngend)
			savefig(figurepath*"RasterRep$(repnumber)Block$(blocknumber)"*svgend)

			#savefig(figurepath*"RasterSeq$(seq)Block$(b)"*svgend)

			GC.gc()

		end #if rasterplot


				rasterplot = true
					if rasterplot
					fontsize += 8

					nonovnonmems = setdiff(nonmembers, idxnovelty)
					#println("creating plot")
					figure(figsize=(20,20))
					#T = 240000
					xlim(0,(lenblocktt-1)/10000)
					ylim(0,sum(assemblymembers[1:Nass,:].>0)+Ni+length(idxnovelty) + length(nonovnonmems))
					xlabel("time [s]", fontsize = fontsize)
					ylabel("sorted neurons", fontsize = fontsize)
					yticks([],fontsize = fontsize);
					xticks(fontsize = fontsize);



					#plot raster with the order of rows determined by population membership
					#rowcount::Integer = 0
					rowcount = 0
					#color = ["k","b","g","r","k","b","g","r","k","b","g","r", "k","b","g","r", "k","b","g","r","k","b","g","r","k","b","g","r"]

					#color = #["midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue"]

						@time for pp = 1:Int(Nass)
							print("\rpopulation ",pp)
							for cc = 1:Int(Nmaxmembers)
								if assemblymembers[pp,cc] < 1
									break
								end
								rowcount+=1
								ind = assemblymembers[pp,cc]
								#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
								vals = block[ind,1:round(Int,totspcheck[repnumberidx,blocknumber,ind])]/10000
								y = rowcount*ones(length(vals))
								#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
								if pp <= Nimg*Nseq
									scatter(vals,y,s=.3,marker="o",c = "midnightblue",linewidths=0)

									#scatter(vals,y,s=.3,marker="o",c = color[mod(pp,Nimg)+1],linewidths=0)
								else
									scatter(vals,y,s=.3,marker="o",c = "darkblue",linewidths=0)
								end

							end
						end
						#savefig(figurepath*"NoInhibRasterSeq$(seq)Block$(b)"*pngend)
						#savefig(figurepath*"NoInhibRasterSeq$(seq)Block$(b)"*svgend)
						for cc = 1:length(idxnovelty) # inhibitory cells
							rowcount+=1
							#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
							vals = block[idxnovelty[cc],1:round(Int,totspcheck[repnumberidx,blocknumber,idxnovelty[cc]])]/10000
							y = rowcount*ones(length(vals))
							#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
							scatter(vals,y,s=.3,marker="o",c = "darkorange",linewidths=0)
						end

						for cc = 1:length(nonovnonmems) # inhibitory cells
							rowcount+=1
							#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
							vals = block[nonovnonmems[cc],1:round(Int,totspcheck[repnumberidx,blocknumber,nonovnonmems[cc]])]/10000
							y = rowcount*ones(length(vals))
							#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
							scatter(vals,y,s=.3,marker="o",c = "midnightblue",linewidths=0)
						end

						#println("inhib")
						for cc = Ne+1:Ncells # inhibitory cells
							rowcount+=1
							#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
							vals = block[cc,1:round(Int,totspcheck[repnumberidx,blocknumber,cc])]/10000
							y = rowcount*ones(length(vals))
							#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
							scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
						end
						#println(figurepath*"RasterSeq$(seq)Block$(b)"*pngend)
						savefig(figurepath*"NonMemsNoveltySepRasterRep$(repnumber)Block$(blocknumber)"*pngend)
						savefig(figurepath*"NonMemsNoveltySepRasterRep$(repnumber)Block$(blocknumber)"*svgend)

						#savefig(figurepath*"RasterSeq$(seq)Block$(b)"*svgend)

						GC.gc()

					end #if rasterplot



		if rasterplot
			#fontsize += 4
		#println("creating plot")
		figure(figsize=(20,20))
		#T = 240000
		xlim(0,(lenblocktt-1)/10000)
		ylim(0,Ncells)
		xlabel("time [s]", fontsize = fontsize)
		ylabel("unsorted neurons", fontsize = fontsize)
		yticks([],fontsize = fontsize);
		xticks(fontsize = fontsize);


		#plot raster with the order of rows determined by population membership
		#rowcount::Integer = 0
		rowcount = 0
		#color = ["k","b","g","r","k","b","g","r","k","b","g","r", "k","b","g","r", "k","b","g","r","k","b","g","r","k","b","g","r"]

		#color = #["midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue"]

			for cc = 1:Ne # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[cc,1:round(Int,totspcheck[repnumberidx,blocknumber,cc])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = "midnightblue",linewidths=0)
			end
			for cc = Ne+1:Ncells # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[cc,1:round(Int,totspcheck[repnumberidx,blocknumber,cc])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
			end
			#println(figurepath*"RasterSeq$(seq)Block$(b)"*pngend)
			savefig(figurepath*"UnsortedRasterRep$(repnumber)Block$(blocknumber)"*pngend)
			savefig(figurepath*"UnsortedRasterRep$(repnumber)Block$(blocknumber)"*svgend)

			#savefig(figurepath*"RasterSeq$(seq)Block$(b)"*svgend)

			GC.gc()

		end #if rasterplot


	#return eNe[1:end]/10000,eNi[1:end]/10000, cNe[1:end]/(0.0001*Ne)/binsize, cNi[1:end]/(0.0001*Ni)/binsize
	# eNi[1:end]/10000)
	# h5write(savefile, "InhibhistBinsec$(round(Int,binsize*0.1))counts/seq$(seqnumber)block$(blocknumber)", cNi[1:end]/(0.0001*Ni)/binsize)
GC.gc()
end # end function plotBlockAssRasterIndiv



function getpopulationaveragesvariablerepetitions(repnumber,blocknumber,rasterplot::Bool, indivsequences::Bool, assemblymembers::Array{Int64,2},block::Array{Int32,2}, idxnovelty, totspcheck::Array{Float64,3}, savefile, figurepath, pngend,svgend;Nseq = 1,Nreps = 20,Nblocks = 1, Nass = 20,Ni = 1000, Ne = 4000, Ncells = 5000,binsize = 1, fontsize = 24,Nimg = 4 , lenblocktt = 240000)
	#println("plotRastwePSTH")
	# input: 	sequence and block number
	# 			assemblymembers
	# 			block: array with spiketimes and cell ID
	# 			indices of neurons stimulated during novel image
	# 			total number of spikes per neuron in this block
	# 			file name and figurepath to save data
	# output:	stored histogram arrays (bin edges and counts) with population averages of
	#				- all excitatory neurons
	#				- untargeted excitatory neurons
	#				- untargeted excitatory neurons excluding novelty neurons
	#				- novelty neurons
	#				- all inhibitory neurons

	assemblymembers = assemblymembers[1:Nimg,:]
	Npop = size(assemblymembers,1)
	Nass = Nimg
	Nmaxmembers = size(assemblymembers,2)

	repnumberidx = findall(Nreps .== repnumber)[1] # get the index of the variable for totspcheck
	seqnumber = copy(repnumberidx) # avoid adjusting code to much keep seq instead of changing it to rep everywhere
	if Nimg == 4
		color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange"]
	elseif Nimg == 3
		color = ["midnightblue","lightskyblue","royalblue","darkred","darksalmon", "saddlebrown","darkgreen","greenyellow","darkolivegreen","darkmagenta","thistle","indigo","darkorange","tan","sienna"]
	elseif Nimg == 5
		color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","indigo","darkred","darksalmon", "saddlebrown","lightcoral","rosybrown","darkgreen","greenyellow","darkolivegreen","chartreuse","turquoise","darkmagenta","thistle","indigo","mediumslateblue","blueviolet","darkorange","tan","sienna","orange", "brown"]
	end#color = color[1:Nimg]
	colornoass = "lightslategrey"
	# b = Int(copy(blocknumber))
	# seq = Int(copy(seqnumber))
	Slimit = maximum(totspcheck[repnumberidx,blocknumber,:])#10000
	#println("seq $(seqnumber) block $(blocknumber) Slimit $Slimit")
	#block = copy(spiketimesseperate[seq,b,:,1:Int(Slimit)]) # cut it to make it more handable
	totspcheck[repnumberidx,blocknumber,totspcheck[repnumberidx,blocknumber,:] .> Slimit] .= Int(Slimit)
	#println("$(totspcheck[seqnumber,blocknumber,1:10])")
	GC.gc()

	if rasterplot
		h5write(savefile, "spiketimeblocks/seq$(seqnumber)block$(blocknumber)", block)
	end


	# get non members
	nonmembers = collect(1:Ne)
	members = sort(unique(assemblymembers[assemblymembers .> 0]))
	deleteat!(nonmembers, members)

	# get overlap of novelty assembly with
	# 1. previously played sequence
	Onoveltysequence = (length(intersect(idxnovelty, union(assemblymembers))))/length(idxnovelty)
	# 2. nonmembers
	Onoveltynonmembers = (length(intersect(idxnovelty, nonmembers)))/length(idxnovelty)
	# 3. all assemblymembers
	Onoveltymembers = (length(intersect(idxnovelty, members)))/length(idxnovelty)

	h5write(savefile, "noveltyoverlap/seq$(seqnumber)block$(blocknumber)", [Onoveltysequence,Onoveltynonmembers,Onoveltymembers])

	@time eNe,cNe = makehistbins(block[1:Ne,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "E$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNe[1:end]/10000)
	h5write(savefile, "E$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNe[1:end]/(0.0001*Ne)/binsize)

	eNe = nothing
	cNe = nothing
	# histogram for excitatory non members

	@time eNenm,cNenm = makehistbins(block[nonmembers,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "ENonMem$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNenm[1:end]/10000)
	h5write(savefile, "ENonMem$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNenm[1:end]/(0.0001*length(nonmembers))/binsize)
	eNenm = nothing
	cNenm = nothing

	@time eNemem,cNemem = makehistbins(block[members,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "EMem$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNemem[1:end]/10000)
	h5write(savefile, "EMem$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNemem[1:end]/(0.0001*length(members))/binsize)

	# plot histogram for non members excluding novelty ones

	nonovnonmems = setdiff(nonmembers, idxnovelty)
	@time eNenmnoN,cNenmnoN = makehistbins(block[nonovnonmems,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "ENonMemNoNov$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNenmnoN[1:end]/10000)
	h5write(savefile, "ENonMemNoNov$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNenmnoN[1:end]/(0.0001*length(nonovnonmems))/binsize)
	eNenmnoN = nothing
	cNenmnoN = nothing
	# plot histogram for inhibitory neurons

	@time eNi,cNi = makehistbins(block[Ne+1:Ncells,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile,  "I$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNi[1:end]/10000)
	h5write(savefile,  "I$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNi[1:end]/(0.0001*Ni)/binsize)
	eNi = nothing
	cNi = nothing
	# plot histogram for novelty neurons

	@time eNn,cNn = makehistbins(block[idxnovelty,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "Nov$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNn[1:end]/10000)
	h5write(savefile, "Nov$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNn[1:end]/(0.0001*length(idxnovelty))/binsize)
	eNn = nothing
	cNn = nothing
	GC.gc()
	# ----------------------------------- plot raster ---------------------------------------------------------------------
	rasterplot = false
		if rasterplot
			fontsize += 8
		#println("creating plot")
		figure(figsize=(20,20))
		#T = 240000
		xlim(0,(lenblocktt-1)/10000)
		ylim(0,sum(assemblymembers[1:Nass,:].>0)+Ni+length(idxnovelty))
		xlabel("time [s]", fontsize = fontsize)
		ylabel("sorted neurons", fontsize = fontsize)
		yticks([],fontsize = fontsize);
		xticks(fontsize = fontsize);



		#plot raster with the order of rows determined by population membership
		#rowcount::Integer = 0
		rowcount = 0
		#color = ["k","b","g","r","k","b","g","r","k","b","g","r", "k","b","g","r", "k","b","g","r","k","b","g","r","k","b","g","r"]

		#color = #["midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue"]

			@time for pp = 1:Int(Nass)
				print("\rpopulation ",pp)
				for cc = 1:Int(Nmaxmembers)
					if assemblymembers[pp,cc] < 1
						break
					end
					rowcount+=1
					ind = assemblymembers[pp,cc]
					#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
					vals = block[ind,1:round(Int,totspcheck[repnumberidx,blocknumber,ind])]/10000
					y = rowcount*ones(length(vals))
					#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
					if pp <= Nimg*Nseq
						scatter(vals,y,s=.3,marker="o",c = color[pp],linewidths=0)

						#scatter(vals,y,s=.3,marker="o",c = color[mod(pp,Nimg)+1],linewidths=0)
					else
						scatter(vals,y,s=.3,marker="o",c = "darkblue",linewidths=0)
					end

				end
			end
			#savefig(figurepath*"NoInhibRasterSeq$(seq)Block$(b)"*pngend)
			#savefig(figurepath*"NoInhibRasterSeq$(seq)Block$(b)"*svgend)
			for cc = 1:length(idxnovelty) # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[idxnovelty[cc],1:round(Int,totspcheck[repnumberidx,blocknumber,idxnovelty[cc]])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = colornoass,linewidths=0)
			end

			#println("inhib")
			for cc = Ne+1:Ncells # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[cc,1:round(Int,totspcheck[repnumberidx,blocknumber,cc])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
			end
			#println(figurepath*"RasterSeq$(seq)Block$(b)"*pngend)
			savefig(figurepath*"RasterRep$(repnumber)Block$(blocknumber)"*pngend)
			savefig(figurepath*"RasterRep$(repnumber)Block$(blocknumber)"*svgend)

			#savefig(figurepath*"RasterSeq$(seq)Block$(b)"*svgend)

			GC.gc()

		end #if rasterplot


				#rasterplot = false
					if rasterplot
					#fontsize += 8

					nonovnonmems = setdiff(nonmembers, idxnovelty)
					#println("creating plot")
					figure(figsize=(20,20))
					#T = 240000
					xlim(0,(lenblocktt-1)/10000)
					ylim(0,sum(assemblymembers[1:Nass,:].>0)+Ni+length(idxnovelty) + length(nonovnonmems))
					xlabel("time [s]", fontsize = fontsize)
					ylabel("sorted neurons", fontsize = fontsize)
					yticks([],fontsize = fontsize);
					xticks(fontsize = fontsize);



					#plot raster with the order of rows determined by population membership
					#rowcount::Integer = 0
					rowcount = 0
					#color = ["k","b","g","r","k","b","g","r","k","b","g","r", "k","b","g","r", "k","b","g","r","k","b","g","r","k","b","g","r"]

					#color = #["midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue"]

						@time for pp = 1:Int(Nass)
							print("\rpopulation ",pp)
							for cc = 1:Int(Nmaxmembers)
								if assemblymembers[pp,cc] < 1
									break
								end
								rowcount+=1
								ind = assemblymembers[pp,cc]
								#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
								vals = block[ind,1:round(Int,totspcheck[repnumberidx,blocknumber,ind])]/10000
								y = rowcount*ones(length(vals))
								#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
								if pp <= Nimg*Nseq
									scatter(vals,y,s=.3,marker="o",c = "midnightblue",linewidths=0)

									#scatter(vals,y,s=.3,marker="o",c = color[mod(pp,Nimg)+1],linewidths=0)
								else
									scatter(vals,y,s=.3,marker="o",c = "darkblue",linewidths=0)
								end

							end
						end
						#savefig(figurepath*"NoInhibRasterSeq$(seq)Block$(b)"*pngend)
						#savefig(figurepath*"NoInhibRasterSeq$(seq)Block$(b)"*svgend)
						for cc = 1:length(idxnovelty) # inhibitory cells
							rowcount+=1
							#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
							vals = block[idxnovelty[cc],1:round(Int,totspcheck[repnumberidx,blocknumber,idxnovelty[cc]])]/10000
							y = rowcount*ones(length(vals))
							#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
							scatter(vals,y,s=.3,marker="o",c = "darkorange",linewidths=0)
						end

						for cc = 1:length(nonovnonmems) # inhibitory cells
							rowcount+=1
							#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
							vals = block[nonovnonmems[cc],1:round(Int,totspcheck[repnumberidx,blocknumber,nonovnonmems[cc]])]/10000
							y = rowcount*ones(length(vals))
							#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
							scatter(vals,y,s=.3,marker="o",c = "midnightblue",linewidths=0)
						end

						#println("inhib")
						for cc = Ne+1:Ncells # inhibitory cells
							rowcount+=1
							#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
							vals = block[cc,1:round(Int,totspcheck[repnumberidx,blocknumber,cc])]/10000
							y = rowcount*ones(length(vals))
							#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
							scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
						end
						#println(figurepath*"RasterSeq$(seq)Block$(b)"*pngend)
						savefig(figurepath*"NonMemsNoveltySepRasterRep$(repnumber)Block$(blocknumber)"*pngend)
						savefig(figurepath*"NonMemsNoveltySepRasterRep$(repnumber)Block$(blocknumber)"*svgend)

						#savefig(figurepath*"RasterSeq$(seq)Block$(b)"*svgend)

						GC.gc()

					end #if rasterplot


		#rasterplot = false
		if rasterplot
			#fontsize += 4
		#println("creating plot")
		figure(figsize=(20,20))
		#T = 240000
		xlim(0,(lenblocktt-1)/10000)
		ylim(0,Ncells)
		xlabel("time [s]", fontsize = fontsize)
		ylabel("unsorted neurons", fontsize = fontsize)
		yticks([],fontsize = fontsize);
		xticks(fontsize = fontsize);


		#plot raster with the order of rows determined by population membership
		#rowcount::Integer = 0
		rowcount = 0
		#color = ["k","b","g","r","k","b","g","r","k","b","g","r", "k","b","g","r", "k","b","g","r","k","b","g","r","k","b","g","r"]

		#color = #["midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue"]

			for cc = 1:Ne # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[cc,1:round(Int,totspcheck[repnumberidx,blocknumber,cc])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = "midnightblue",linewidths=0)
			end
			for cc = Ne+1:Ncells # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[cc,1:round(Int,totspcheck[repnumberidx,blocknumber,cc])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
			end
			#println(figurepath*"RasterSeq$(seq)Block$(b)"*pngend)
			savefig(figurepath*"UnsortedRasterRep$(repnumber)Block$(blocknumber)"*pngend)
			savefig(figurepath*"UnsortedRasterRep$(repnumber)Block$(blocknumber)"*svgend)

			#savefig(figurepath*"RasterSeq$(seq)Block$(b)"*svgend)

			GC.gc()

		end #if rasterplot


	#return eNe[1:end]/10000,eNi[1:end]/10000, cNe[1:end]/(0.0001*Ne)/binsize, cNi[1:end]/(0.0001*Ni)/binsize
	# eNi[1:end]/10000)
	# h5write(savefile, "InhibhistBinsec$(round(Int,binsize*0.1))counts/seq$(seqnumber)block$(blocknumber)", cNi[1:end]/(0.0001*Ni)/binsize)
	GC.gc()
end # end function plotBlockAssRasterIndiv


function getpopulationaveragessequencelength(repnumber,blocknumber,rasterplot::Bool, indivsequences::Bool, assemblymembers::Array{Int64,2},block::Array{Int32,2}, idxnovelty, totspcheck::Array{Float64,3}, savefile, figurepath, pngend,svgend;Nseq = 1,Nreps = 20,Nblocks = 1, Nass = 20,Ni = 1000, Ne = 4000, Ncells = 5000,binsize = 1, fontsize = 24,Nimg = 4 , lenblocktt = 240000)
	#println("plotRastwePSTH")
	# input: 	sequence and block number
	# 			assemblymembers
	# 			block: array with spiketimes and cell ID
	# 			indices of neurons stimulated during novel image
	# 			total number of spikes per neuron in this block
	# 			file name and figurepath to save data
	# output:	stored histogram arrays (bin edges and counts) with population averages of
	#				- all excitatory neurons
	#				- untargeted excitatory neurons
	#				- untargeted excitatory neurons excluding novelty neurons
	#				- novelty neurons
	#				- all inhibitory neurons

	assemblymembers = assemblymembers[1:repnumber,:]
	Npop = size(assemblymembers,1)
	Nass = repnumber
	println(Nass)
	Nmaxmembers = size(assemblymembers,2)

	repnumberidx = findall(Nimg .== repnumber)[1] # get the index of the variable for totspcheck
	seqnumber = copy(repnumberidx) # avoid adjusting code to much keep seq instead of changing it to rep everywhere
	color = ["midnightblue","lightskyblue","royalblue","lightsteelblue","darkred","darksalmon", "saddlebrown","lightcoral","darkgreen","greenyellow","darkolivegreen","chartreuse","darkmagenta","thistle","indigo","mediumslateblue","darkorange","tan","sienna","orange"]
	#color = color[1:Nimg]
	colornoass = "lightslategrey"
	# b = Int(copy(blocknumber))
	# seq = Int(copy(seqnumber))
	Slimit = maximum(totspcheck[repnumberidx,blocknumber,:])#10000
	#println("seq $(seqnumber) block $(blocknumber) Slimit $Slimit")
	#block = copy(spiketimesseperate[seq,b,:,1:Int(Slimit)]) # cut it to make it more handable
	totspcheck[repnumberidx,blocknumber,totspcheck[repnumberidx,blocknumber,:] .> Slimit] .= Int(Slimit)
	#println("$(totspcheck[seqnumber,blocknumber,1:10])")
	GC.gc()

	if rasterplot
		h5write(savefile, "spiketimeblocks/seq$(seqnumber)block$(blocknumber)", block)
	end


	# get non members
	nonmembers = collect(1:Ne)
	members = sort(unique(assemblymembers[assemblymembers .> 0]))
	deleteat!(nonmembers, members)

	# get overlap of novelty assembly with
	# 1. previously played sequence
	# Onoveltysequence = (length(intersect(idxnovelty, union(assemblymembers))))/length(idxnovelty)
	# # 2. nonmembers
	# Onoveltynonmembers = (length(intersect(idxnovelty, nonmembers)))/length(idxnovelty)
	# # 3. all assemblymembers
	# Onoveltymembers = (length(intersect(idxnovelty, members)))/length(idxnovelty)
	#
	# h5write(savefile, "noveltyoverlap/seq$(seqnumber)block$(blocknumber)", [Onoveltysequence,Onoveltynonmembers,Onoveltymembers])

	@time eNe,cNe = makehistbins(block[1:Ne,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "E$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNe[1:end]/10000)
	h5write(savefile, "E$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNe[1:end]/(0.0001*Ne)/binsize)


	# histogram for excitatory non members

	@time eNemem,cNemem = makehistbins(block[members,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "EMem$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNemem[1:end]/10000)
	h5write(savefile, "EMem$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNemem[1:end]/(0.0001*length(members))/binsize)

	@time eNenm,cNenm = makehistbins(block[nonmembers,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile, "ENonMem$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNenm[1:end]/10000)
	h5write(savefile, "ENonMem$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNenm[1:end]/(0.0001*length(nonmembers))/binsize)

	# plot histogram for non members excluding novelty ones

	# nonovnonmems = setdiff(nonmembers, idxnovelty)
	# @time eNenmnoN,cNenmnoN = makehistbins(block[nonovnonmems,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	# h5write(savefile, "ENonMemNoNov$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNenmnoN[1:end]/10000)
	# h5write(savefile, "ENonMemNoNov$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNenmnoN[1:end]/(0.0001*length(nonovnonmems))/binsize)
	#
	# # plot histogram for inhibitory neurons

	@time eNi,cNi = makehistbins(block[Ne+1:Ncells,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	h5write(savefile,  "I$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNi[1:end]/10000)
	h5write(savefile,  "I$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNi[1:end]/(0.0001*Ni)/binsize)

	# plot histogram for novelty neurons
	#
	# @time eNn,cNn = makehistbins(block[idxnovelty,1:Int(Slimit)][:],lenblocktt = lenblocktt, binsize = binsize)
	# h5write(savefile, "Nov$(round(Int,binsize*0.1))msedges/seq$(seqnumber)block$(blocknumber)", eNn[1:end]/10000)
	# h5write(savefile, "Nov$(round(Int,binsize*0.1))mscounts/seq$(seqnumber)block$(blocknumber)", cNn[1:end]/(0.0001*length(idxnovelty))/binsize)
	eNe = nothing
	cNe = nothing
	eNenm = nothing
	cNenm = nothing

	# eNenmnoN = nothing
	# cNenmnoN = nothing
	eNi = nothing
	cNi = nothing
	# eNn = nothing
	# cNn = nothing
	GC.gc()
	# ----------------------------------- plot raster ---------------------------------------------------------------------
	rasterplot = false
		if rasterplot
			fontsize += 8
		#println("creating plot")
		figure(figsize=(20,20))
		#T = 240000
		xlim(0,(lenblocktt-1)/10000)
		ylim(0,sum(assemblymembers[1:Nass,:].>0)+Ni+length(idxnovelty))
		xlabel("time [s]", fontsize = fontsize)
		ylabel("sorted neurons", fontsize = fontsize)
		yticks([],fontsize = fontsize);
		xticks(fontsize = fontsize);



		#plot raster with the order of rows determined by population membership
		#rowcount::Integer = 0
		rowcount = 0
		#color = ["k","b","g","r","k","b","g","r","k","b","g","r", "k","b","g","r", "k","b","g","r","k","b","g","r","k","b","g","r"]

		#color = #["midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue"]

			@time for pp = 1:Int(Nass)
				print("\rpopulation ",pp)
				for cc = 1:Int(Nmaxmembers)
					if assemblymembers[pp,cc] < 1
						break
					end
					rowcount+=1
					ind = assemblymembers[pp,cc]
					#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
					vals = block[ind,1:round(Int,totspcheck[repnumberidx,blocknumber,ind])]/10000
					y = rowcount*ones(length(vals))
					#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
					if pp <= repnumber
						scatter(vals,y,s=.3,marker="o",c = "midnightblue",linewidths=0)

						#scatter(vals,y,s=.3,marker="o",c = color[mod(pp,Nimg)+1],linewidths=0)
					else
						scatter(vals,y,s=.3,marker="o",c = "darkblue",linewidths=0)
					end

				end
			end
			#savefig(figurepath*"NoInhibRasterSeq$(seq)Block$(b)"*pngend)
			#savefig(figurepath*"NoInhibRasterSeq$(seq)Block$(b)"*svgend)
			for cc = 1:length(idxnovelty) # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[idxnovelty[cc],1:round(Int,totspcheck[repnumberidx,blocknumber,idxnovelty[cc]])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = colornoass,linewidths=0)
			end

			#println("inhib")
			for cc = Ne+1:Ncells # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[cc,1:round(Int,totspcheck[repnumberidx,blocknumber,cc])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
			end
			#println(figurepath*"RasterSeq$(seq)Block$(b)"*pngend)
			savefig(figurepath*"RasterRep$(repnumber)Block$(blocknumber)"*pngend)
			savefig(figurepath*"RasterRep$(repnumber)Block$(blocknumber)"*svgend)

			#savefig(figurepath*"RasterSeq$(seq)Block$(b)"*svgend)

			GC.gc()

		end #if rasterplot


				#rasterplot = true
					if rasterplot
					#fontsize += 8

					nonovnonmems = setdiff(nonmembers, idxnovelty)
					#println("creating plot")
					figure(figsize=(20,20))
					#T = 240000
					xlim(0,(lenblocktt-1)/10000)
					ylim(0,sum(assemblymembers[1:Nass,:].>0)+Ni+length(idxnovelty) + length(nonovnonmems))
					xlabel("time [s]", fontsize = fontsize)
					ylabel("sorted neurons", fontsize = fontsize)
					yticks([],fontsize = fontsize);
					xticks(fontsize = fontsize);



					#plot raster with the order of rows determined by population membership
					#rowcount::Integer = 0
					rowcount = 0
					#color = ["k","b","g","r","k","b","g","r","k","b","g","r", "k","b","g","r", "k","b","g","r","k","b","g","r","k","b","g","r"]

					#color = #["midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue"]

						@time for pp = 1:Int(Nass)
							print("\rpopulation ",pp)
							for cc = 1:Int(Nmaxmembers)
								if assemblymembers[pp,cc] < 1
									break
								end
								rowcount+=1
								ind = assemblymembers[pp,cc]
								#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
								vals = block[ind,1:round(Int,totspcheck[repnumberidx,blocknumber,ind])]/10000
								y = rowcount*ones(length(vals))
								#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
								if pp <= repnumber*Nseq
									scatter(vals,y,s=.3,marker="o",c = "midnightblue",linewidths=0)

									#scatter(vals,y,s=.3,marker="o",c = color[mod(pp,Nimg)+1],linewidths=0)
								else
									scatter(vals,y,s=.3,marker="o",c = "darkblue",linewidths=0)
								end

							end
						end
						#savefig(figurepath*"NoInhibRasterSeq$(seq)Block$(b)"*pngend)
						#savefig(figurepath*"NoInhibRasterSeq$(seq)Block$(b)"*svgend)
						for cc = 1:length(idxnovelty) # inhibitory cells
							rowcount+=1
							#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
							vals = block[idxnovelty[cc],1:round(Int,totspcheck[repnumberidx,blocknumber,idxnovelty[cc]])]/10000
							y = rowcount*ones(length(vals))
							#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
							scatter(vals,y,s=.3,marker="o",c = "darkorange",linewidths=0)
						end

						for cc = 1:length(nonovnonmems) # inhibitory cells
							rowcount+=1
							#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
							vals = block[nonovnonmems[cc],1:round(Int,totspcheck[repnumberidx,blocknumber,nonovnonmems[cc]])]/10000
							y = rowcount*ones(length(vals))
							#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
							scatter(vals,y,s=.3,marker="o",c = "midnightblue",linewidths=0)
						end

						#println("inhib")
						for cc = Ne+1:Ncells # inhibitory cells
							rowcount+=1
							#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
							vals = block[cc,1:round(Int,totspcheck[repnumberidx,blocknumber,cc])]/10000
							y = rowcount*ones(length(vals))
							#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
							scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
						end
						#println(figurepath*"RasterSeq$(seq)Block$(b)"*pngend)
						savefig(figurepath*"NonMemsNoveltySepRasterRep$(repnumber)Block$(blocknumber)"*pngend)
						savefig(figurepath*"NonMemsNoveltySepRasterRep$(repnumber)Block$(blocknumber)"*svgend)

						#savefig(figurepath*"RasterSeq$(seq)Block$(b)"*svgend)

						GC.gc()

					end #if rasterplot


		#rasterplot = false
		if rasterplot
			#fontsize += 4
		#println("creating plot")
		figure(figsize=(20,20))
		#T = 240000
		xlim(0,(lenblocktt-1)/10000)
		ylim(0,Ncells)
		xlabel("time [s]", fontsize = fontsize)
		ylabel("unsorted neurons", fontsize = fontsize)
		yticks([],fontsize = fontsize);
		xticks(fontsize = fontsize);


		#plot raster with the order of rows determined by population membership
		#rowcount::Integer = 0
		rowcount = 0
		#color = ["k","b","g","r","k","b","g","r","k","b","g","r", "k","b","g","r", "k","b","g","r","k","b","g","r","k","b","g","r"]

		#color = #["midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue","royalblue","midnightblue"]

			for cc = 1:Ne # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[cc,1:round(Int,totspcheck[repnumberidx,blocknumber,cc])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = "midnightblue",linewidths=0)
			end
			for cc = Ne+1:Ncells # inhibitory cells
				rowcount+=1
				#vals = sptimes[ind,1:Rtotalspikes[ind]]/1000 # 1:maxspcheck
				vals = block[cc,1:round(Int,totspcheck[repnumberidx,blocknumber,cc])]/10000
				y = rowcount*ones(length(vals))
				#scatter(vals,y,s=.3,c="k",marker="o",linewidths=0)
				scatter(vals,y,s=.3,marker="o",c = "r",linewidths=0)
			end
			#println(figurepath*"RasterSeq$(seq)Block$(b)"*pngend)
			savefig(figurepath*"UnsortedRasterRep$(repnumber)Block$(blocknumber)"*pngend)
			savefig(figurepath*"UnsortedRasterRep$(repnumber)Block$(blocknumber)"*svgend)

			#savefig(figurepath*"RasterSeq$(seq)Block$(b)"*svgend)

			GC.gc()

		end #if rasterplot

		GC.gc()
end # end function plotBlockAssRasterIndiv




function savespiketimes(seqnumber,blocknumber,rasterplot::Bool, indivsequences::Bool, assemblymembers::Array{Int64,2},block::Array{Int32,2}, idxnovelty, totspcheck::Array{Float64,3}, savefile, figurepath, pngend,svgend;Nseq = 1,Nreps = 20,Nblocks = 1, Nass = 20,Ni = 1000, Ne = 4000, Ncells = 5000,binsize = 1, fontsize = 24,Nimg = 4 , lenblocktt = 240000, blockbegintt = 1)
	"""save the spike times
	input: 	sequence and block number
				assemblymembers
				block: array with spiketimes and cell ID
				indices of neurons stimulated during novel image
				total number of spikes per neuron in this block
				file name and figurepath to save data
	output:	stored histogram arrays (bin edges and counts) with population averages of
					- all excitatory neurons
					- untargeted excitatory neurons
					- untargeted excitatory neurons excluding novelty neurons
					- novelty neurons
					- all inhibitory neurons"""


	Npop = size(assemblymembers,1)
	Nmaxmembers = size(assemblymembers,2)

	Slimit = maximum(totspcheck[seqnumber,blocknumber,:])#10000
	totspcheck[seqnumber,blocknumber,totspcheck[seqnumber,blocknumber,:] .> Slimit] .= Int(Slimit)

	GC.gc()
	println(" write block to file ")

	h5write(savefile, "spiketimeblocks/seq$(seqnumber)block$(blocknumber)", block)



	GC.gc() # garbage collector
end # end function
