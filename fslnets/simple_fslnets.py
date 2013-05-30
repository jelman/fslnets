

"""
first basic pass

(nets_load)
grab all .txt files (timepoint X component) 
 for each subject
 1. demean by full mean , divide by whole std
 2. clean (removes bad components and regresses them out

put onto large matrix 
(nSub X timepoints) by ncomponents(only the good)

calc corrcoef, and zero out diagonal >0

ncomponents X ncomponents matrix
(just_diag == 0)
reshape 1 X ncomponents x ncomponents

## TODO
once data is cleaned, calc'd corrcoef, we can concatenate subjects

option to create a subset of good components to run through randomise


quick estimate of median AR(1) coeff
use this to do a r2z transform
(compare to Fischer Transform)

save the result to a nifti images (requires reshape, and set aff etc)


run randomise (consdier ways to parallelize this??)

fdr correction and possibly give user summary of pvalues for each component
(sorted)



 DD are wanted components, will regress out aanything not in this list
"""

import os
from glob import glob
import numpy as np


def normalise_data(dat):
    """ demans and divides by std
    data is timepoints X components"""
    # demean each column
    tmp = (dat - dat.mean(0)) 
    tmp = tmp / tmp.std(ddof=1)
    # std clced using ddof=1; 1/(n-1) * sum( (xi- mean(x))**2 )
    return tmp

def parse_idx(good, n):
    """ given a list of good indicies (good), out of
    n indicies, return an array of
    good_idx
    bad_idx"""
    all_idx = np.arange(n)
    bad_idx = np.array([x for x in all_idx if not x in good])
    good_idx = np.array([ x for x in all_idx if x in good])
    test_idx = sorted(np.concatenate((good_idx, bad_idx)))
    if not all([x in good_idx for x in good]):
        raise IOError('index issue %s not in range %s'%(good, all_idx))
    return good_idx, bad_idx

def simple_regress(good, bad):
    """ simple linear regression to regress bad components
    from good components"""
    result = good - np.dot(bad,  np.dot(np.linalg.pinv(bad),good))
    return result

def remove_regress_bad_components(dat, good):
    """ returns timepoints by good components for this subject"""
    ntimepoints, ncomponents = dat.shape
    good_ids, bad_ids = parse_idx(good, ncomponents)
    good_dat = dat[:,good_ids]
    bad_dat = dat[:,bad_ids]
    return simple_regress(good_dat, bad_dat)

def concat_subjects(subjects):
    """turn a list of subject data arrays into one array"""
    nsubs = len(subjects)
    ntimepts, ncomp = subjects[0].shape
    concat = np.array(subjects)
    concat.shape = (nsubs * ntimepts, ncomp)
    return concat

def corrcoef(data):
    """calcs the corrcoef for data with structure
    (ntimepoints * nsub ) X ncomponents
    Returns array : ncomponents X ncomponents"""
    return np.corrcoef(data.T)


