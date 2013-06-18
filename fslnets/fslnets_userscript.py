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

datadir = '/home/jagust/rsfmri_ica/data/OldICA_IC0_ecat_2mm_6fwhm_125.gica/dual_regress'
subfile = '/home/jagust/rsfmri_ica/Spreadsheets/Filelists/OLDICA_5mm_125_orig_sublist.txt'
goodics = [0,1,4,5,7,8,9,10,11,12,14,16,17,20,22,25,30,33]
with open(subfile, 'r') as f:
    sublist = f.read().splitlines()
group_ts = {}   # Dictionary to hold timeseries of all subjects
group_stats = []    # List to hold correlation matrices of all subjects
for subj in sublist:
    infile = '_'.join(['dr_stage1', subj, 'and_confound_regressors_6mm'])
    data = np.loadtxt(os.path.join(datadir,infile), dtype='float')
    # Load data and normalize
    norm_data = fslnets.normalise_data(data)
    # Regresses out components not listed in goodics
    clean_data = fslnets.remove_regress_bad_components(norm_data, goodics)
    nnodes = clean_data.shape[1]
    # Calculate correlation matrix of all good components
    node_stat = fslnets.corrcoef(clean_data)
    reshaped_stat = node_stat.reshape((1, nnodes * nnodes))
    # Append subject data to group data
    group_ts[subj] = clean_data
    group_stats.append(reshaped_stat)

# Create concatenated matrix of subjects' timeseries data
concat_ts = fslnets.concat_subjects(group_ts.values())
nsubs, ntimepts, nnodes = concat_ts.shape
# Create concatenated matrix of subjects' corerlation matrices
concat_stats = np.array(group_stats)
reshaped_stats = concat_stats.reshape((nsubs, nnodes*nnodes))
# Convert to z-scores with AR(1) correction
zdat = fslnets.r_to_z(reshaped_stats, concat_ts)
