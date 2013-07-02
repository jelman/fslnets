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
datadir : str
    path to directory containing timeseries (usually dr stage 1 output)
subfile : str
    file containing list of subject codes
goodics : list
    column indices of good components (starting at 0). 
    any nodes not listed here will be regressed out as noise. 
nois : list 
    column indices of nodes of interest. any nodes listed in goodics
    but not nois will be accounted for in partial correlations but
    NOT included in correction for multiple comparisons
des_file : str
    path to fsl design (.mat) file. input to randomise for group glm
con_file : str
    path to fsl contrast (.con) file. input to randomise for group glm
net_measures : list
    list of network correlation measures to obtain
        `corrcoef` : full correlation
        `partialcorr` : partial correlation
infile : str
    filename containing subject's timeseries data.
    set this variable in subject loop
"""

datadir = '/home/jagust/rsfmri_ica/data/OldYoung_All_6mm_IC40.gica/dual_regress'
subfile = '/home/jagust/rsfmri_ica/Spreadsheets/Filelists/allsubs.txt'
goodics = [0, 1, 2, 7, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 23, 26, 28] # Start at 0
nois = [1, 2, 7, 9, 10, 11, 15, 16, 20, 23, 28]
des_file = os.path.join('/home/jagust/rsfmri_ica/data/OldYoung_All_6mm_IC40.gica/models',
              'OneSamp_PIBIndex_Age.mat')
con_file = os.path.join('/home/jagust/rsfmri_ica/data/OldYoung_All_6mm_IC40.gica/models',
              'OneSamp_PIBIndex_Age.con')      
net_measures = ['corrcoef', 'partialcorr']
# Define variables based on parameters set above
with open(subfile, 'r') as f:
    sublist = f.read().splitlines()
nsubs = len(sublist)    # get  number of subjects
nnodes = len(goodics)   # get nuber of nodes
noi_idx = [goodics.index(i) for i in nois]  # get indices of nodes to correct for
exists, resultsdir = fslnets.make_dir(datadir,'fslnets_results') # make results directory
all_ts = [] # Create empty list to hold timeseries data
           
for i in range(len(sublist)):
    subj = sublist[i]
    infile = '_'.join(['dr_stage1', subj, 'and_confound_regressors_6mm'])
    data = np.loadtxt(os.path.join(datadir,infile), dtype='float')
    # Load data and normalize
    norm_data = fslnets.normalise_data(data)
    # Regresses out components not listed in goodics
    clean_data = fslnets.remove_regress_bad_components(norm_data, goodics)
    # Append subject data to group array
    all_ts.append(clean_data)   
# Create concatenated matrix of subjects' timeseries data
group_ts = fslnets.concat_subjects(all_ts)
    
for stat in net_measures:
    all_netmat = np.zeros((nsubs, nnodes*nnodes))     
    for i in range(len(sublist)):
        subj = sublist[i]  
        # Calculate correlation and partial correlation matrices of all good components
        if stat == 'corrcoef':
            subj_netmat = fslnets.corrcoef(group_ts[i,:,:])
        elif stat == 'partialcorr':
            subj_netmat = fslnets.partial_corr(group_ts[i,:,:])
        # Reshape matrices to 1d shape and append to group array
        all_netmat[i,:] = subj_netmat.reshape((1, nnodes * nnodes))

    # Convert to z-scores with AR(1) correction
    z_netmat = fslnets.r_to_z(all_netmat, group_ts)   
    # Save matrices of z transformed data to nifti images for input to randomise
    img_fname = os.path.join(resultsdir, '_'.join(['fslnets',stat,'4D.nii.gz']))
    netmat_img = fslnets.save_img(z_netmat, img_fname)

    # Run randomise with specified design and contrast files
    rand_basename = os.path.join(resultsdir, '_'.join(['fslnets',stat]))
    netmat_uncorr, netmat_corr = fslnets.randomise(netmat_img, 
                                                rand_basename, 
                                                des_file, 
                                                con_file)
    # Load randomise output for each contrast
    uncorr_results = fslnets.get_results(netmat_uncorr)
    corr_results = fslnets.get_results(netmat_corr)

    # Run correction for multiple comparisons
    fdr_results = {}
    for i in range(len(uncorr_results.keys())):
        conname = sorted(uncorr_results.keys())[i]
        fdr_results[conname] = fslnets.multi_correct(uncorr_results[conname], noi_idx)
        outfile = os.path.join(resultsdir, 
                            ''.join(['fslnets_',stat,'_fdr_corrp_','tstat',str(i+1),'.txt']))
        np.savetxt(outfile, 
                    fdr_results[conname], 
                    fmt='%1.3f', 
                    delimiter='\t')  

    
    
