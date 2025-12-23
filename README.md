# ***ligscreen***
Ligand screening toolkit that combines OpenMM with MOPAC to perform QM/MM-style binding-energy predictions for accelerated lead discovery.

### First of all, you can download the CASF data in order to run tests
save it in the "data" directory
The Comparative Assessment of Scoring Functions (CASF) benchmark suite was employed to evaluate the performance of the scoring function under study. The CASF package (version 2016) was obtained from the official PDBbind-CN web portal (Wang, et al., J. Chem. Inf. Model., 2021) at the following public repository:
URL: https://www.pdbbind-plus.org.cn/casf

### then download gnina file
and then save that in the "gnina" directory

---
# **TO_DO**
- [x] downloading the CASF data 
- [ ] vina:
    1. [x] script
    2. [x] run
    3. [ ] results
- [ ] gnina results script, running and results
- [ ] gnina results with optimizing the ligand
- [ ] MM openmm code, script, calculating binding energies, results
- [ ] openmm binding energy calculation with optimizing
- [ ] parmed pocket extracting 
- [ ] QM mopac code, script, calculating binding energies, results
- [ ] calculating other properties for our SF(scoring function):
- [ ] QM/MM code, script, do the optimization and then calculatig binding energies, results
- [ ] download other datasets (other proteins in PDBbind specially metalloproteins)
- [ ] compare the results of diffrente scoring functions 

