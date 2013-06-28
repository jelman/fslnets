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
des_file = os.path.join('/home/jagust/rsfmri_ica/data/OldYoung_All_6mm_IC40.gica/models',
              'OneSamp_PIBIndex_Age_GM.mat')
con_file = os.path.join('/home/jagust/rsfmri_ica/data/OldYoung_All_6mm_IC40.gica/models',
              'OneSamp_PIBIndex_Age_GM.con')      

with open(subfile, 'r') as f:
    sublist = f.read().splitlines()
group_ts = {}   # Dictionary to hold timeseries of all subjects
group_corrcoef = {}    # Dictionary to hold correlation matrices of all subjects
group_partialcorr = {}    # Dictionary to hold partial correlation matrices of all subjects
for subj in sublist:
    infile = '_'.join(['dr_stage1', subj, 'and_confound_regressors_6mm'])
    data = np.loadtxt(os.path.join(datadir,infile), dtype='float')
    # Load data and normalize
    norm_data = fslnets.normalise_data(data)
    # Regresses out components not listed in goodics
    clean_data = fslnets.remove_regress_bad_components(norm_data, goodics)
    nnodes = clean_data.shape[1]
    # Calculate correlation and partial correlation matrices of all good components
    corrcoef_data = fslnets.corrcoef(clean_data)
    partialcorr_data = fslnets.partial_corr(clean_data)
    # Reshape matrices to 1d shape (1 x number of nodes * number of nodes)
    reshaped_corrcoef = corrcoef_data.reshape((1, nnodes * nnodes))
    reshaped_partialcorr = partialcorr_data.reshape((1, nnodes * nnodes))    
    # Append subject data to group data
    group_ts[subj] = clean_data
    group_corrcoef[subj] = reshaped_corrcoef
    group_partialcorr[subj] = reshaped_partialcorr

# Create concatenated matrix of subjects' timeseries data
concat_ts = fslnets.concat_subjects([group_ts[key] for key in sorted(group_ts.keys())])
nsubs, ntimepts, nnodes = concat_ts.shape
# Create concatenated matrix of subjects' correlation matrices
concat_corrcoef = np.array([group_corrcoef[key] for key in sorted(group_corrcoef.keys())])
reshaped_corrcoef = concat_corrcoef.reshape((nsubs, nnodes*nnodes))
concat_partialcorr = np.array([group_partialcorr[key] for key in sorted(group_partialcorr.keys())])
reshaped_partialcorr = concat_partialcorr.reshape((nsubs, nnodes*nnodes))
# Convert to z-scores with AR(1) correction
z_corrcoef = fslnets.r_to_z(reshaped_corrcoef, concat_ts)
z_partialcorr = fslnets.r_to_z(reshaped_partialcorr, concat_ts)

# Save matrices of z transformed data to nifti images for input to randomise
corrcoef_img = fslnets.save_img(z_corrcoef, 
                                os.path.join(datadir, 
                                'fslnets_corrcoef4D.nii.gz'))
partialcorr_img = fslnets.save_img(z_partialcorr, 
                                os.path.join(datadir, 
                                'fslnets_partialcorr4D.nii.gz'))
# Run randomise with specified design and contrast files
corrcoef_uncorr, corrcoef_corr = fslnets.randomise(corrcoef_img, 
                                            os.path.join(datadir, 'fslnets_corrcoef'), 
                                            des_file, 
                                            con_file)
partialcorr_uncorr, partialcorr_corr = fslnets.randomise(corrcoef_img, 
                                            os.path.join(datadir, 'fslnets_corrcoef'), 
                                            des_file, 
                                            con_file)
# Load randomise output for each contrast

# Run correction for multiple comparisons
