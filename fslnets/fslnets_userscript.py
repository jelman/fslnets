import os, sys
from glob import glob
import numpy as np
sys.path.insert(0, '/home/jagust/jelman/CODE/fslnets/fslnets')
import simple_fslnets as fslnets

"""
Preliminary wrapper script for simple_fslnets.py. For all subjects takes 
a set of timeseries from multiple nodes and calculates temporal correlation 
between nodes. Applies an r to z transform using an AR(1) correction.

Parameters:
-----------
datadir : path to directory containing timeseries (usually dr stage 1 output)
subfile : file containing list of subject codes
goodics : list of good components. any components not listed here will 
            be regressed out as noise. 
infile : string of filename containing subject's timeseries data

"""

datadir = '/home/jagust/rsfmri_ica/data/OldYoung_All_6mm_IC40.gica/dual_regress'
subfile = '/home/jagust/rsfmri_ica/Spreadsheets/Filelists/allsubs.txt'
goodics = [0, 1, 2, 7, 9, 10, 11, 13, 15, 16, 17, 19, 20, 23, 26, 28] # Start at 0
nois = [1, 2, 7, 9, 10, 11, 15, 16, 20, 23, 28] # Nodes of interest. Determines number of comparisons
                                                # to correct for.

with open(subfile, 'r') as f:
    sublist = f.read().splitlines()
group_ts = {}   # Dictionary to hold timeseries of all subjects
group_corr = []    # List to hold correlation matrices of all subjects
group_pcorr = []    # List to hold partial correlation matrices of all subjects
for subj in sublist:
    infile = '_'.join(['dr_stage1', subj, 'and_confound_regressors_6mm'])
    data = np.loadtxt(os.path.join(datadir,infile), dtype='float')
    # Load data and normalize
    norm_data = fslnets.normalise_data(data)
    # Regresses out components not listed in goodics
    clean_data = fslnets.remove_regress_bad_components(norm_data, goodics)
    nnodes = clean_data.shape[1]
    # Calculate correlation and partial correlation matrices of all good components
    corr_stat = fslnets.corrcoef(clean_data)
    pcorr_stat = fslnets.partial_corr(clean_data)
    # Reshape matrices to 1d shape (1 x number of nodes * number of nodes)
    reshaped_corr = corr_stat.reshape((1, nnodes * nnodes))
    reshaped_pcorr = pcorr_stat.reshape((1, nnodes * nnodes))    
    # Append subject data to group data
    group_ts[subj] = clean_data
    group_corr.append(reshaped_corr)
    group_pcorr.append(reshaped_pcorr)

# Create concatenated matrix of subjects' timeseries data
concat_ts = fslnets.concat_subjects([group_ts[key] for key in sorted(group_ts.keys())])
nsubs, ntimepts, nnodes = concat_ts.shape
# Create concatenated matrix of subjects' correlation matrices
concat_corr = np.array(group_corr)
reshaped_corr = concat_corr.reshape((nsubs, nnodes*nnodes))
concat_pcorr = np.array(group_pcorr)
reshaped_pcorr = concat_pcorr.reshape((nsubs, nnodes*nnodes))
# Convert to z-scores with AR(1) correction
z_corr = fslnets.r_to_z(reshaped_corr, concat_ts)
z_pcorr = fslnets.r_to_z(reshaped_pcorr, concat_ts)
