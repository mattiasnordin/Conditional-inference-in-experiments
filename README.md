# Conditional inference in experiments
The code is somewhat computationally demanding which means that the scripts will take quite som time to run for all combinations of parameters. The results in the paper came from running code in parallel to speed up the process, hence the exact results are not replicable as it was not possible to set specific random seeds. However, since the number of replications is generally quite large, the results should be very similar to the ones in the paper, and all qualitative results should hold, if simulations are run again.

The code is either Julia or R code. Below, a short description of each of these different files and their purposes are given:
- `sim_exhaustive.jl`: This code performs the simulations shown in Figure 1 in the paper and Figure A1 in the supplmentary material. It should take one to two hours to run on a standard desktop computer.
- `sim_algorithm.jl` This code runs simulations for Figure 2 and Tables 1 to 4 in the paper, as well as Figure A2 in the supplementary material. It may take several days to run.
- `sim_tibs.R` This code runs the simulation for the cross-estimation estimator for Figure 2 and Tables 1 to 4 in the paper, as well as Figure A2 in the supplementary material. It may take a couple of days to run.
- `figures_tables.jl` This code uses the `csv`-files created by the previous three scripts and creates all figures and tables in the paper and supplementary material.
