import os, sys
from glob import glob
import numpy as np
sys.path.insert(0, '/home/jagust/jelman/CODE/fslnets/fslnets')
import simple_fslnets as fslnets

"""
Preliminary user script, not for actual use!
"""
datadir = '/home/jagust/rsfmri_ica/data/OldICA_IC0_ecat_2mm_6fwhm_125.gica/dual_regress'
subfile = '/home/jagust/rsfmri_ica/Spreadsheets/Filelists/OLDICA_5mm_125_orig_sublist.txt'
goodics = [0,1,4,5,7,8,9,10,11,12,14,16,17,20,22,25,30,33]
with open(subfile, 'r') as f:
    sublist = f.read().splitlines()
group_ts = {}
group_stats = {}
for i in arange(len(sublist)):
    infile = '_'.join(['dr_stage1', subj, 'and_confound_regressors_6mm'])
    data = np.loadtxt(os.path.join(datadir,infile), dtype='float')
    norm_data = fslnets.normalise_data(data)
    clean_data = fslnets.remove_regress_bad_components(norm_data, goodics)
    nnodes = clean_data.shape[1]
    node_stat = fslnets.corrcoef(clean_data)
    reshaped_data = corr_data.reshape((1, nnodes * nnodes))
    group_ts[subj] = clean_data
    group_stats[subj] = corr_data
    
concat_ts = fslnets.concat_subjects(group_ts.values())
concat_stats = fslnets.concat_subjects(group_stats.values())
nsubs, ntimepts, nnodes = concat_ts.shape
reshaped_stats = concat_stats.reshape((len(sublist), nnodes * nnodes))
zdat = fslnets.r_to_z(reshaped_stats, concat_ts)
