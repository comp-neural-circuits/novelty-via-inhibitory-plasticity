# this file is part of the V1 2Ch Predictive Code project
# Additional information and instruction in README


function runsimulation_inhibtuning(ifiSTDP,ifwadapt,stimparams,stimulus::Array{Float64,2},
	weights::Array{Float64,2}, assemblymembers::Array{Int64,2},
	spiketimes::Array{Int32,2},storagedec::Array{Bool,1},
	storagetimes::Array{Int64,1}, savefile::String, lenpretrain,
	inhibassemblies::Array{Int64,2}; dt = 0.1, T = 2000,
	adjustfactor = 1, adjustfactorinhib=1, inhibtuing = true, inhibfactor = 0.1,
	bwfactor=100, tauw_adapt=150)

"""	Runs a new simulation of a plastic E-I spiking neural network model where both E and I
	neurons receive tuned input

	Inputs:
		ifiSTDP boolean: if inhibitory spike timing plasticity is active
		ifwadapt boolean: if an intrisic adaptive current is included for E neurons
		stimparams array: with stimulus parameters
		stimulus array: stimulus constisting of [stimulated assembly, start time, stop time, rate increase]
		weights array: initial weight matrix
		assemblymembers array: members of an excitatory assembly (group of E neurons tuned to the same stimulus)
		spiketimes array: spiketimes of all neuron [spike time, neuron id] (in julia speed is increased by prior initialisation)
		storagedec array of booleans: storage decisions what should be stored
		storagetimes array: how often each of the items should be stored
		savefile string: name of the file to store to
		lenpretrain int: duration of the pretraining phase prior to starting the real simulation in stimulus indices
		inhibassemblies array: members of an inhibitory "assembly" (group of I neurons tuned to the same stimulus)
		dt float: integration timestep (optional)
		T float: total duration of the run (optional)
		adjustfactor float: adjust factor of the E-to-E plasticity after pretraining to allow for switching off plasticity
		adjustfactorinhib float: adjust factor of the I-to-E plasticity after pretraining to allow for switching off plasticity
		inhibtuing boolean: if inhibitory neurons are tuned to stimuli as well
		inhibfactor float: the scaling factor by how much the inhibitory neurons are driven by an external stimulus compared to the excitatory tuned neurons
		bwfactor float: scaling factor of the adaptive currents (only required for adaptive experiments)
		tauw_adapt float: timescale of the adaptive currents (only required for adaptive experiments)

	Output:

		totalspikes array: set of last spikes
		several run parameters are stored in hdf5 files including all spiketimes of all neurons across the entire simulation
		"""

		# Naming convention
		# e corresponds to excitatory
		# i corresponds to inhibitory
		# x corresponds to external

		#membrane dynamics
		taue = 20 #e membrane time constant ms
		taui = 20 #i membrane time constant ms
		vreste = -70 #e resting potential mV
		vresti = -62 #i resting potential mV
		vpeak = 20 #cutoff for voltage.  when crossed, record a spike and reset mV
		eifslope = 2 #eif slope parameter mV
		C = 300 #capacitance pF
		erev = 0 #e synapse reversal potential mV
		irev = -75 #i synapse reversal potntial mV
		vth0 = -52 #initial spike voltage threshold mV
		thrchange = false # can be switched off to have vth constant at vth0
		ath = 10 #increase in threshold post spike mV
		tauth = 30 #threshold decay timescale ms
		vreset = -60 #reset potential mV
		taurefrac = 1 #absolute refractory period ms
		aw_adapt = 4 #adaptation parameter a nS conductance
		bw_adapt = bwfactor*0.805 #adaptation parameter b pA current

		if T > 1 # this should not be saved for the precompile run
			h5write(savefile, "params/MembraneDynamics", [taue,taui, vreste,vresti , vpeak , eifslope, C, erev, irev, vth0, thrchange, ath, tauth, vreset, taurefrac, aw_adapt,bw_adapt, tauw_adapt ])
			h5write(savefile, "params/STDPwadapt", [Int(ifiSTDP),Int(ifwadapt)])
			h5write(savefile, "params/bwfactor", bwfactor)
			h5write(savefile, "params/adjustfactor", adjustfactor)
			h5write(savefile, "params/adjustfactorinhib", adjustfactorinhib)

		end

		# total number of neurons
		Ncells = Ne+Ni

		# synaptic kernel
		tauerise = 1 #e synapse rise time
		tauedecay = 6 #e synapse decay time
		tauirise = .5 #i synapse rise time
		tauidecay = 2 #i synapse decay time

		# external input
		rex = 4.5 #external input rate to e (khz) since the timestep is one ms an input of 4.5 corresp to 4.5kHz
		rix = 2.5#2.25 #external input rate to i (khz) # on average this is reduced to 2.25
		println("Larger inhibitory input")
		# Ensure the overall inhibitory input remains the same when inhibiory tuning is included

		# ---------------- inhibitory tuning ------------------
		# reduce the overall inhibitory input by the amount added during each stimulus presentation
		memsinhib1 = inhibassemblies[1,inhibassemblies[1,:] .!= -1]
		# reduce the total inhibition based on the total added inhibition to stimulated inhibitory neurons
		added_inhib = length(memsinhib1)*stimulus[end,4]*inhibfactor
		println("initial rix ", rix)
		rix -= added_inhib/Ni
		println("reduced rix ", rix)


		# initial connectivity
		Jeemin = 1.78 #minimum ee weight  pF
		Jeemax = 21.4 #maximum ee weight pF

		Jeimin = 48.7 #minimum ei weight pF
		Jeimax = 243 #maximum ei weight pF


		Jex = 1.78 #external to e weight pF
		Jix = 1.27 #external to i weight pF

		if T > 1
			h5write(savefile, "params/Connectivity", [tauerise, tauedecay, tauirise, tauidecay, rex, rix, Jeemin, Jeemax, Jeimin, Jeimax, Jex, Jix])
		end

		#voltage based stdp (for alternative testing not used here)
		altd = .0008 #ltd strength pA/mV pairwise STDP LTD
		altp = .0014 #ltp strength pA/mV^2 triplet STDP LTP
		thetaltd = -70 #ltd voltage threshold mV
		thetaltp = -49 #ltp voltage threshold mV
		tauu = 10 #timescale for u variable ms
		tauv = 7 #timescale for v variable ms
		taux = 15 #timescale for x variable ms
		if T > 1
			h5write(savefile, "params/voltageSTDP", [altd, altp, thetaltd, thetaltp, tauu, tauv, taux])
		end


		#inhibitory stdp
		tauy = 20 #width of istdp curve ms
		eta = 1 #istdp learning rate pA
		r0 = .003 #target rate (khz)
		if T > 1
			h5write(savefile, "params/iSTDP", [tauy, eta, r0])
		end


		# triplet parameters
		tripletrule = true
		o1 = zeros(Float64,Ne);
		o2 = zeros(Float64,Ne);
		r1 = zeros(Float64,Ne);
		r2 = zeros(Float64,Ne);
		if T > 1
			h5write(savefile, "params/Triplet", Int(tripletrule))
		end

		tau_p = 16.8;        # in ms
		tau_m = 33.7;        # in s
		tau_x = 101.0;        # in s
		tau_y = 125.0;        # in s
		# init LTP and LTD variables
		A_2p = 7.5*10^(-10); # pairwise LTP disabled
		A_2m = 7.0*10^(-3);
		A_2m_eff = A_2m;       #effective A_2m, includes the sliding threshold
		A_3p = 9.3*10^(-3)
		A_3m = 2.3*10^(-4); # triplet LTP disabled

		if T > 1
			h5write(savefile, "params/TripletTausAs", [tau_p,tau_m, tau_x,tau_y , A_2p , A_2m, A_3p, A_3m])
		end

		# simulation parameters
		taurefrac = 1 #ms refractory preriod clamped for 1 ms
		dtnormalize = 20 #how often to normalize rows of ee weights ms heterosynaptic plasticity
		stdpdelay = 1000 #time before stdp is activated, to allow transients to die out ms
		dtsaveweights = 2000  # save weights  every 2000 ms
		# minimum and maximum of storing the weight matrices
		minwstore = 80
		modwstore = 10

		if T > 1
			h5write(savefile, "params/Normdt", dtnormalize)
			h5write(savefile, "params/dtsaveweights", dtsaveweights)
			h5write(savefile, "params/minwstore", minwstore)
			h5write(savefile, "params/modwstore", modwstore)
		end
		Nassemblies = size(assemblymembers,1) #number of assemblies
		Nmembersmax = size(assemblymembers,2) #maximum number of neurons in a population

		ttstimbegin = round(Integer,stimulus[lenpretrain,2]/dt) # is set to zero

		# stimulus parameters
		Nimg, Nreps, Nseq, Nblocks, stimstart, lenstim, lenpause, strength = stimparams


		# Initialisation of spike arrays ------------------------------

		# spike count and spike times

		if storagedec[1]
			totalspikes = zeros(Int,Ncells)
		else
			spiketimes = nothing
			totalspikes = nothing
			totsp = nothing
		end
		totsp::Int64 = 0; # total numberof spikes
		spmax::Int64 = size(spiketimes,1); # maximum recorded spikes

		# further arrays for storing inputs and synaptic integration
		forwardInputsE = zeros(Float64,Ncells) #sum of all incoming weights from excitatpry inputs both external and EE and IE
		forwardInputsI = zeros(Float64,Ncells) #sum of all incoming weights from inhibitory inputs both II and EI
		forwardInputsEPrev = zeros(Float64,Ncells) #as above, for previous timestep
		forwardInputsIPrev = zeros(Float64,Ncells)

		xerise = zeros(Float64,Ncells) #auxiliary variables for E/I currents (difference of exponentials)
		xedecay = zeros(Float64,Ncells)
		xirise = zeros(Float64,Ncells)
		xidecay = zeros(Float64,Ncells)

		expdist = Exponential()

		v = zeros(Float64,Ncells) #membrane voltage
		nextx = zeros(Float64,Ncells) #time of next external excitatory input
		sumwee0 = zeros(Float64,Ne) #initial summed e weight, for normalization
		Nee = zeros(Int,Ne) #number of e->e inputs, for normalization
		rx = zeros(Float64,Ncells) #rate of external input


		# initialisation of membrane potentials and poisson inputs
		for cc = 1:Ncells
			v[cc] = vreset + (vth0-vreset)*rand()
			if cc <= Ne # excitatory neurons
				rx[cc] = rex
				nextx[cc] = rand(expdist)/rx[cc]
				for dd = 1:Ne
					sumwee0[cc] += weights[dd,cc]
					if weights[dd,cc] > 0
						Nee[cc] += 1
					end
				end
			else # inhibtory neurons
				rx[cc] = rix
				nextx[cc] = rand(expdist)/rx[cc]
			end
		end


		vth = vth0*ones(Float64,Ncells) #adaptive threshold
		wadapt = aw_adapt*(vreset-vreste)*ones(Float64,Ne) #adaptation current
		lastSpike = -100*ones(Ncells) #last time the neuron spiked
		trace_istdp = zeros(Float64,Ncells) #low-pass filtered spike train for istdp
		u_vstdp = vreset*zeros(Float64,Ne) # for voltage rule (not used here)
		v_vstdp = vreset*zeros(Float64,Ne) # for voltage rule (not used here)
		x_vstdp = zeros(Float64,Ne) # for voltage rule (not used here)


		# ---------------------------------- set up storing avg weights -----------------

		idxnovelty = zeros(Int32, 200)

		Nass = Nassemblies # number of distinct stimuli
		nm = zeros(Int32,Nass) # Number of memebers in this assembly
		inhibnm = zeros(Int32,Nass) # Number of memebers in this inhibitory "assembly"
		for mm = 1:Nass # loop over all assemblies
			#check when first -1 comes to determine number of neurons
			nm[mm] = sum(assemblymembers[mm,:].!=-1)
			inhibnm[mm] = sum(inhibassemblies[mm,:].!=-1)
		end

		# initialise avergae cross and inhbitiory assembly weights
		# Calculate them here instead of storing whole matrices (too large files)
		avgXassembly = zeros(Float64,Nass,Nass) # average cross assembly weights
		avgInhibXassembly = zeros(Float64,Nass,Nass) # average inhibitory to excitatory cross assembly weights
		avgItoassembly = zeros(Float64,Nass) # avergate I to E assembly weights
		avgnonmemstoassembly = zeros(Float64,Nass) # avergate weights from neurons not stimulus driven to exc. assemblies
		avgassemblytononmems = zeros(Float64,Nass) # avergate weights from exc. assemblies to neurons not stimulus driven
		avgassemblytonovelty = zeros(Float64,Nass) # avergate weights from exc. assemblies to novelty assemblies
		avgnoveltytoassembly = zeros(Float64,Nass) # avergate weights from exc. novelty assemblies to exc. assemblies

		# determine neurons not part of any excitatory assembly
		nonmems = collect(1:Ne)
		members = sort(unique(assemblymembers[assemblymembers .> 0]))
		deleteat!(nonmems, members)

		# total number of simulation steps, steps when to normalise, and save
		Nsteps = round(Int,T/dt)
		inormalize = round(Int,dtnormalize/dt)
		saveweights = round(Int,dtsaveweights/dt)

		# true time
		t::Float64 = 0.0
		tprev::Float64 = 0.0

		# counters how often a variable was saved
		weightstore::Integer = 0
		spikestore::Integer = 0
		novelstore::Integer = 0

		# bool counter if a neuron has just had a spike
		spiked = zeros(Bool,Ncells)


	#   ------------------------------------------------------------------------
	#
	#   				begin actual simulation
	#
	#   ------------------------------------------------------------------------

	# evaluate the run via @time
	@time	for tt = 1:Nsteps

				if mod(tt,Nsteps/100) == 1  #print percent complete
					print("\r",round(Int,100*tt/Nsteps))
				end

				forwardInputsE[:] .= 0.
				forwardInputsI[:] .= 0.
				t = dt*tt
				tprev = dt*(tt-1)

				# iterate over all stimuli in the passed stimulus array
				for ss = 1:size(stimulus)[1]

					if (tprev<stimulus[ss,2]) && (t>=stimulus[ss,2])  #just entered stimulation period
						ass = round(Int,stimulus[ss,1]) # TODO: refactor somewhat unfortunate naming of assembly

						# if assembly is a novelty assembly
						if ass > Nimg*Nseq && (t>stimulus[lenpretrain,2])
							# name idxnovelty still stems from time before defining novelty assembly directly just novelty assembly
							idxnovelty = assemblymembers[ass,assemblymembers[ass,:] .!= -1]
							novelstore += 1
							println("Ass :$(ass) time $(t) > $(stimulus[lenpretrain,2])")
							println("novelstore :$(novelstore)")

							h5write(savefile, "novelty/indices$(novelstore)", assemblymembers[ass,assemblymembers[ass,:] .!= -1])
							h5write(savefile, "novelty/assembly$(novelstore)", ass)
						end

						# stimulus tuning: increase the external stimulus to rx of the corresponding E assembly memebers
						mems = assemblymembers[ass,assemblymembers[ass,:] .!= -1]
						rx[mems] .+= stimulus[ss,4]

						# ---------------------- inhibtuning --------------------
						if inhibtuing
							memsinhib = inhibassemblies[ass,inhibassemblies[ass,:] .!= -1]
							rx[memsinhib] .+= stimulus[ss,4]*inhibfactor
						end

					end # if just entered stim period



					if (tprev<stimulus[ss,3]) && (t>=stimulus[ss,3]) #just exited stimulation period
						ass = round(Int,stimulus[ss,1])

						mems = assemblymembers[ass,assemblymembers[ass,:] .!= -1]
						rx[mems] .-= stimulus[ss,4]


						if inhibtuing
							memsinhib = inhibassemblies[ass,inhibassemblies[ass,:] .!= -1]
							rx[memsinhib] .-= stimulus[ss,4]*inhibfactor
						end

					end # if just left stim period

				end #end loop over stimuli

				if mod(tt,inormalize) == 0 #excitatory synaptic normalization
					for cc = 1:Ne
						sumwee = 0.
						for dd = 1:Ne
							sumwee += weights[dd,cc]
						end # dd

						for dd = 1:Ne
							if weights[dd,cc] > 0.
								weights[dd,cc] -= (sumwee-sumwee0[cc])/Nee[cc]
								if weights[dd,cc] < Jeemin
									weights[dd,cc] = Jeemin
								elseif weights[dd,cc] > Jeemax
									weights[dd,cc] = Jeemax
								end # if
							end # if
						end # dd for
					end # cc for
					#println("Normalised...")
				end #end normalization

				if mod(tt,saveweights) == 0 #&& (weightstore < 20 || mod(weightstore,100) == 0)#excitatory synaptic normalization
					weightstore += 1
					#if (weightstore < minwstore || mod(weightstore,modwstore) == 0)
						#h5write(savefile, "dursim/weights$(weightstore)_$(tt)", weights)
						for pre = 1:Nass # use maximum to be sure to capture it both for sequencelength and variablerepetitions
							# for inhib to assembly get avg. weights here pre is actually post
							avgnonmemstoassembly[pre] = getXassemblyweight(nonmems, assemblymembers[Int(pre),1:nm[Int(pre)]], weights)
							avgassemblytononmems[pre] = getXassemblyweight(assemblymembers[Int(pre),1:nm[Int(pre)]],nonmems, weights)
							if sum(idxnovelty) == 0 # before first novelty arose still zero
								avgnoveltytoassembly[pre] = 0
								avgassemblytonovelty[pre] = 0
							else
								avgnoveltytoassembly[pre] = getXassemblyweight(idxnovelty[idxnovelty.>0], assemblymembers[Int(pre),1:nm[Int(pre)]], weights)
								avgassemblytonovelty[pre] = getXassemblyweight(assemblymembers[Int(pre),1:nm[Int(pre)]],idxnovelty[idxnovelty.>0], weights)
							end
							avgItoassembly[pre] = getXassemblyweight(collect(Ne+1:Ncells), assemblymembers[Int(pre),1:nm[Int(pre)]], weights)
							for post = 1:Nass
								avgXassembly[pre,post] = getXassemblyweight(assemblymembers[Int(pre),1:nm[Int(pre)]], assemblymembers[Int(post),1:nm[Int(post)]], weights)
								# determine the average inhibitory assembly to excitatory assembly weight  define inhibnm
								avgInhibXassembly[pre,post] = getXassemblyweight(inhibassemblies[Int(pre),1:inhibnm[Int(pre)]], assemblymembers[Int(post),1:nm[Int(post)]], weights)

							end # post assemblies
						end # pre assemblies



						h5write(savefile, "dursimavg/avgXassembly$(weightstore)_$(tt)", avgXassembly)
						h5write(savefile, "dursimavg/avgInhibXassembly$(weightstore)_$(tt)", avgInhibXassembly)

						h5write(savefile, "dursimavg/avgItoassembly$(weightstore)_$(tt)", avgItoassembly)
						h5write(savefile, "dursimavg/avgnonmemstoassembly$(weightstore)_$(tt)", avgnonmemstoassembly)
						h5write(savefile, "dursimavg/avgassemblytononmems$(weightstore)_$(tt)", avgassemblytononmems)
						h5write(savefile, "dursimavg/avgassemblytonovelty$(weightstore)_$(tt)", avgassemblytonovelty)
						h5write(savefile, "dursimavg/avgnoveltytoassembly$(weightstore)_$(tt)", avgnoveltytoassembly)

						h5write(savefile, "dursimavg/Itoneuron1$(weightstore)_$(tt)", weights[Ne+1:Ncells,1][weights[Ne+1:Ncells,1].>0])# store only non zero inhibitory to neuron 1/2 weights
						h5write(savefile, "dursimavg/Itoneuron2$(weightstore)_$(tt)", weights[Ne+1:Ncells,2][weights[Ne+1:Ncells,2].>0]) # store only non zero inhibitory to neuron 1/2 weights

				end # mod(tt,saveweights) == 0

				fill!(spiked,zero(Bool)) # reset spike bool without new memory allocation

				for cc = 1:Ncells
					trace_istdp[cc] -= dt*trace_istdp[cc]/tauy

					while(t > nextx[cc]) #external input
						nextx[cc] += rand(expdist)/rx[cc]
						if cc < Ne
							forwardInputsEPrev[cc] += Jex
						else
							forwardInputsEPrev[cc] += Jix
						end
					end

					xerise[cc] += -dt*xerise[cc]/tauerise + forwardInputsEPrev[cc]
					xedecay[cc] += -dt*xedecay[cc]/tauedecay + forwardInputsEPrev[cc]
					xirise[cc] += -dt*xirise[cc]/tauirise + forwardInputsIPrev[cc]
					xidecay[cc] += -dt*xidecay[cc]/tauidecay + forwardInputsIPrev[cc]

					if cc < Ne # excitatory
						if thrchange
						vth[cc] += dt*(vth0 - vth[cc])/tauth;
						end
						wadapt[cc] += dt*(aw_adapt*(v[cc]-vreste) - wadapt[cc])/tauw_adapt;
						u_vstdp[cc] += dt*(v[cc] - u_vstdp[cc])/tauu;
						v_vstdp[cc] += dt*(v[cc] - v_vstdp[cc])/tauv;
						x_vstdp[cc] -= dt*x_vstdp[cc]/taux;


						# triplet accumulators
						r1[cc] += -dt*r1[cc]/tau_p # exponential decay of all
			            r2[cc] += -dt*r2[cc]/tau_x
						o1[cc] += -dt*o1[cc]/tau_m
			            o2[cc] += -dt*o2[cc]/tau_y
					end

					if t > (lastSpike[cc] + taurefrac) #not in refractory period
						# update membrane voltage

						ge = (xedecay[cc] - xerise[cc])/(tauedecay - tauerise);
						gi = (xidecay[cc] - xirise[cc])/(tauidecay - tauirise);

						if cc < Ne #excitatory neuron (eif), has adaptation
							if ifwadapt
								dv = (vreste - v[cc] + eifslope*exp((v[cc]-vth[cc])/eifslope))/taue + ge*(erev-v[cc])/C + gi*(irev-v[cc])/C - wadapt[cc]/C;
							else
								dv = (vreste - v[cc] + eifslope*exp((v[cc]-vth[cc])/eifslope))/taue + ge*(erev-v[cc])/C + gi*(irev-v[cc])/C;
							end
							v[cc] += dt*dv;
							if v[cc] > vpeak
								spiked[cc] = true
								wadapt[cc] += bw_adapt
							end
						else
							dv = (vresti - v[cc])/taui + ge*(erev-v[cc])/C + gi*(irev-v[cc])/C;
							v[cc] += dt*dv;
							if v[cc] > vth0
								spiked[cc] = true
							end
						end

						if spiked[cc] #spike occurred
							spiked[cc] = true;
							v[cc] = vreset;
							lastSpike[cc] = t;
							totalspikes[cc] += 1;
							totsp += 1;
							if totsp < spmax
								spiketimes[totsp,1] = tt; # time index as a sparse way to save spiketimes
								spiketimes[totsp,2] = cc; # cell id
							elseif totsp == spmax
								spiketimes[totsp,1] = tt; # time index
								spiketimes[totsp,2] = cc; # cell id

								totsp = 0 # reset counter total number of spikes
								# store spiketimes
								spikestore += 1
								h5write(savefile, "dursimspikes/spiketimes$(spikestore)", spiketimes)
							end


							trace_istdp[cc] += 1.;
							if cc<Ne
								x_vstdp[cc] += 1. / taux;
							end

							if cc < Ne && thrchange # only change for excitatory cells and when thrchange == true
								vth[cc] = vth0 + ath;
							end

							#loop over synaptic projections
							for dd = 1:Ncells # postsynaptic cells dd  - cc presynaptic cells
								if cc <= Ne #excitatory synapse
									forwardInputsE[dd] += weights[cc,dd];
								else #inhibitory synapse
									forwardInputsI[dd] += weights[cc,dd];
								end
							end

						end #end if(spiked)
					end #end if(not refractory)

					if ifiSTDP # select if iSTDP
						#istdp
						if spiked[cc] && (t > stdpdelay)
							if cc < Ne #excitatory neuron fired, potentiate i inputs
								for dd = (Ne+1):Ncells
									if weights[dd,cc] == 0.
										continue
									end
									weights[dd,cc] += eta*trace_istdp[dd]

									if weights[dd,cc] > Jeimax
										weights[dd,cc] = Jeimax
									end
								end
							else #inhibitory neuron fired, modify outputs to e neurons
								for dd = 1:Ne
									if weights[cc,dd] == 0.
										continue
									end

									weights[cc,dd] += eta*(trace_istdp[dd] - 2*r0*tauy)
									if weights[cc,dd] > Jeimax
										weights[cc,dd] = Jeimax
									elseif weights[cc,dd] < Jeimin
										weights[cc,dd] = Jeimin
									end
								end
							end
						end #end istdp
					end # ifiSTDP

					if tripletrule

					#triplet, ltd component
					if spiked[cc] && (t > stdpdelay) && (cc < Ne)
						r1[cc] = r1[cc] + 1 # incrememt r1 before weight update
						for dd = 1:Ne #depress weights from cc to all its postsyn cells
							# cc = pre dd = post
							if weights[cc,dd] == 0. # ignore connections that were not establishe in the beginning
								continue
							end

			                weights[cc,dd] -= o1[dd]*(A_2m + A_3m*r2[cc])

							if weights[cc,dd] < Jeemin
								weights[cc,dd] = Jeemin
							end

						end # for loop over Ne
						 r2[cc] = r2[cc] + 1 # increment after weight update
					end # ltd

					#triplet, ltp component
					if spiked[cc] && (t > stdpdelay) && (cc < Ne)
						o1[cc] = o1[cc] + 1 # incrememt r1 before weight update
						# cc = post dd = pre
						for dd = 1:Ne #increase weights from cc to all its presyn cells dd
							if weights[dd,cc] == 0.
								continue
							end

							weights[dd,cc] += r1[dd]*(A_2p + A_3p*o2[cc]) #A_2p = 0

							if weights[dd,cc] > Jeemax
								weights[dd,cc] = Jeemax
							end

						end # loop over cells presynaptic
						o2[cc] = o2[cc] + 1 # increment after weight update

					end #ltp

				else # not triplet but voltage rule

				#vstdp, ltd component
					if spiked[cc] && (t > stdpdelay) && (cc < Ne)
						for dd = 1:Ne #depress weights from cc to cj
							if weights[cc,dd] == 0.
								continue
							end

							if u_vstdp[dd] > thetaltd
								weights[cc,dd] -= altd*(u_vstdp[dd]-thetaltd)

								if weights[cc,dd] < Jeemin
									weights[cc,dd] = Jeemin

								end
							end
						end
					end #end ltd

					#vstdp, ltp component
					if (t > stdpdelay) && (cc < Ne) && (v[cc] > thetaltp) && (v_vstdp[cc] > thetaltd)
						for dd = 1:Ne
							if weights[dd,cc] == 0.
								continue
							end

							weights[dd,cc] += dt*altp*x_vstdp[dd]*(v[cc] - thetaltp)*(v_vstdp[cc] - thetaltd);
							if weights[dd,cc] > Jeemax
								weights[dd,cc] = Jeemax
							end
						end
					end #end ltp
				end # if triplet rule
			end #end loop over cells
			forwardInputsEPrev = copy(forwardInputsE)
			forwardInputsIPrev = copy(forwardInputsI)

			# once the actual stimulation begins, eventually readjust the learning rate
			if tt == ttstimbegin

						tau_p = 16.8;        # in ms
						tau_m = 33.7;        # in s
						tau_x = 101.0;        # in s
						tau_y = 125.0;        # in s
						# init LTP and LTD variables
						A_2p = adjustfactor*7.5*10^(-10); # pairwise LTP disabled
						A_2m = adjustfactor*7.0*10^(-3);  #small learning rate
						A_2m_eff = A_2m;       #effective A_2m, includes the sliding threshold
						A_3p = adjustfactor*9.3*10^(-3)
						A_3m = adjustfactor*2.3*10^(-4); # triplet LTP disabled

						eta_old = eta
						eta = adjustfactorinhib*eta_old
						println("new eta iSTDP",eta)
						if T > 1
							h5write(savefile, "params/TripletTausAs_stim", [tau_p,tau_m, tau_x,tau_y , A_2p , A_2m, A_3p, A_3m])
							h5write(savefile, "params/eta_stim", eta)

							h5write(savefile, "params/adjustfactor2", adjustfactor)
							h5write(savefile, "params/adjustfactorinhib2", adjustfactorinhib)

							h5write(savefile, "dursim/weights", weights)

						end # if T > 1

			end # if tt == ttstimbegin
		end # tt loop over time
	print("\r")

	println("simulation finished")

	return totalspikes
end # function simulation
