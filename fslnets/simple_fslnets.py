

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
    tmp = tmp / tmp.std()
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


def remove_regress_bad_components(dat, good_idx):
    good_idx = np.array(good_idx)

    good = dat[:,good_idx]







