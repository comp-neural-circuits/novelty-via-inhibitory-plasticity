# novelty-via-inhibitory-plasticity

novelty-via-inhibitory-plasticity project code accompanying the manuscript:  
##### The generation of cortical novelty responses through inhibitory plasticity  
Auguste Schulz\*, Christoph Miehl\*, Michael J. Berry II, Julijana Gjorgjieva   

\* equal contribution  

https://www.biorxiv.org/content/10.1101/2020.11.30.403840v1

## A spiking neural network model to investigate cortical novelty responses
This project is concerned with the underlying mechanisms that give rise to novelty responses in sensory cortices. Stimulation paradigms were inspired by experiments performed in the primary visual cortex as reported in Homann et al., BioRxiv 2017 and in primary auditory cortex in Natan et al., eLife 2015.

We set up a spiking neural network model of the mouse primary sensory cortex (80% excitatory, 20% inhibitory) using various plasticity mechanisms.  

Some of the simulation code follows code provided by Litwin-Kumar and Doiron [1].

The simulation code is written in Julia (https://julialang.org/, https://github.com/JuliaLang/julia) (Version 1.0.1).

Required packages:

1. PyPlot
2. HDF5
3. Distributions
4. Dates
5. LinearAlgebra
6. Random
7. Profile
8. StatsBase
9. PyCall

Packages can be installed via `using Pkg` `Pkg.add("PackageName")`.  

### Project structure  
* `main`  - contains initialisation files that start the simulation
 * `simulation`  - contains the main simulation and simulation helper files
 * `evaluation`    - contains the main evaluation (weight, spiketime) and evaluation helper files
  * `postprocessing`    - contains the postprocessing notebooks (python unsing standard packages)

The simulation data (parameters, weight evolution, and spiketimes & cell id) are stored using `hdf5` file format in  the `data` folder.
Evaluation of the stored data, i.e. generating histograms of the spiketimes, starts as soon as the main simulation finished and is stored in `results` in folders that have the same name as the corresponding data .h5 file.


### Simulations
Here we demonstrate
1. a sequence violation paradigm ABCABCABC...ABNABC
2. oddball paradigm AAA...ABA

Further running instructions:

Run the following commands from a Ubuntu command line (tested on Ubuntu 18.04).
The arguments set the sequence and stimulation parameters and enable flexible testing of different parameters.

Step into the `main` folder to run the initialisation files. Ensure that the correct version of Julia including the required packages are installed and that the Julia path is specified.

#### For 1. the sequence violation paradigm,  
run the following command
- for the unique sequence paradigm (compare Figure 1)  
Nimg 3, Nreps 20, Nseq 10, Nblocks 1, lenstim 300, strength 12, Ntrain 5, adjustfactor 1.0, adaptive neurons false, inhibfactor 0.1  
> nohup julia initsim_sequence_violation_ARGS.jl 3 20 10 1 300 12 5 10 0 10 &> ../tmp/standard_run_unique_sequence.txt &

- for the repeated sequence paradigm (compare Figure 3,4, Suppl. Figure 1)  
Nimg 3, Nreps 20, Nseq 5, Nblocks 10, lenstim 300, strength 12, Ntrain 5, adjustfactor 1.0, adaptive neurons false, inhibfactor 0.1  
> nohup julia initsim_sequence_violation_ARGS.jl 3 20 5 10 300 12 5 10 0 10 &> ../tmp/standard_run_repeated_sequence.txt &


#### For 2. the oddball paradigm,  
run the following command
- for the standard SSA oddball experiment without disinhibition (compare Figure 5)  
Nreps 20, Nblocks 1, lenstim 300, lenpause 900, Nneurons 200, pretrain false, inhibfactor 0.1 , disinhibstrength 0
> nohup julia initsim_oddball_disinhib_ARGS.jl 20 1 300 900 200 0 10 0 &> ../tmp/SSA_disinhibition_strength_0.txt &

- for the standard SSA oddball experiment with disinhibition  (compare Figure 6)  
Nreps 20, Nblocks 1, lenstim 300, lenpause 900, Nneurons 200, pretrain false, inhibfactor 0.1, disinhibstrength 1.5  
> nohup julia initsim_oddball_disinhib_ARGS.jl 20 1 300 900 200 0 10 150 &> ../tmp/SSA_disinhibition_strength_150.txt &

In order to produce Fig. 6 D and E the code above was run 5 times for 8 varying disinhibstrength values ranging from (0 to 1.5).  

#### Postprocessing  

Visualisation and further processing steps are performed in jupyter notebooks.
Specify the name of the file to be analysed in the respective notebooks.

To test if the simulations were successful we provide two example notebooks

- for the unique sequence paradigm:   
Figure_sequence_violation.ipynb
- oddball experiment:  
Figure_oddball_paradigm.ipynb

#### Figure generating notebooks

Note that at the moment, some of the post processing steps performed in the notebooks might only be compatible with certain stimulation parameters and or various other stimulation paradigms not specified here.

Upon publication, we will provide all data accompanying the figures.  

In case you would like to already generate all the figures shown in the paper, please send an email to ga84zah@mytum.de, we are happy to provide you with the corresponding datafiles. They are too large to include in this repository. The notebooks generating the figures can be found under `postprocessing/Figure_x_.ipynb`

Further commenting and clean up will be made prior to publication to ensure reusability and reproducibility.



[1] A. Litwin-Kumar & B. Doiron.  Formation and maintenance of neuronal assemblies through synaptic plasticity.  Nature Communications (2014).  
Copyright notice:  
http://lk.zuckermaninstitute.columbia.edu/  
litwin-kumar_doiron_formation_2014  
Copyright (C) 2014 Ashok Litwin-Kumar
