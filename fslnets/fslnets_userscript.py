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
group_dat = {}
for subj in sublist:
    infile = '_'.join(['dr_stage1', subj, 'and_confound_regressors_6mm'])
    data = np.loadtxt(os.path.join(datadir,infile), dtype='float')
    norm_data = fslnets.normalise_data(data)
    clean_data = fslnets.remove_regress_bad_components(norm_data, goodics)
    group_dat[subj] = clean_data
concat_data = fslnets.concat_subjects(group_dat.values())
n_subs, n_tps, n_nodes = concat_data.shape
reshaped_data = concat_data.reshape((n_subs * n_tps, n_nodes))
corr_data = fslnets.corrcoef(reshaped_data)
zdat = fslnets.r_to_z(corr_data, concat_data)
