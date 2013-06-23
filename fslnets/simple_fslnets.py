

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
import itertools
import scipy.linalg as linalg

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
    concat.shape = (nsubs , ntimepts, ncomp)
    return concat

def corrcoef(data):
    """calcs the corrcoef for data with structure
    (ntimepoints * nsub ) X ncomponents
    Returns array : ncomponents X ncomponents
    zeros diagonal"""
    res =  np.corrcoef(data.T)
    np.fill_diagonal(res, 0)
    return res

def partial_corr(data):
    """ calcs partial correlation for data with structure
    (ntimepoints X ncomponents)
    zeros diagonal"""
    timepts, ncomp = data.shape
    all_pcorr = np.zeros((ncomp, ncomp))
    allcond = set(np.arange(ncomp))
    for a, b in itertools.combinations(allcond, 2):
        xy = data[:,np.array([a,b])]
        rest = allcond - set([a,b])
        confounds = data[:, np.array([x for x in rest])]
        part_corr = _cond_partial_cor(xy, confounds)
        all_pcorr[a,b] = part_corr
        all_pcorr[b,a] = part_corr
    return all_pcorr


def _cond_partial_cor(xy,  confounds=[]):
    """ Returns the partial correlation of y and x, conditioning on
    confounds.
    
    Parameters
    -----------
    xy : numpy array
        num_timeponts X 2
    confounds : numpy array
        numtimepoints X nconfounds

    Returns
    -------
    pcorr : float
        partial correlation of x, y condioned on conf

    """
    if len(confounds):
        res = linalg.lstsq(confounds, xy)
        xy = xy - np.dot(confounds, res[0])

    return np.dot(xy[:,0], xy[:,1]) / np.sqrt(np.dot(xy[:,1], xy[:,1]) \
            *np.dot(xy[:,0], xy[:,0]))

def calc_arone(sdata):
    """quick estimate of median AR(1) coefficient
    across subjects concatenated data
    nsub, ntimepoints X ncomponents"""
    arone = np.sum(sdata[:,:,:-1] * sdata[:,:,1:], 2) / np.sum(sdata * sdata,2)
    return np.median(arone)

def _calc_r2z_correction(sdata, arone):
    """ use the pre computed median auto regressive AR(1)
    coefficinet to z-transform subjects data

    Parameters
    ----------
    sdata : array
        array of data (nsub X ntimepoints X nnodes)
    arone : float
        median AR(1) computed from subject data

    Returns
    -------

    r_to_z_correct : float
        value used to z-transfom sdata
        
    """
    
    nsub, ntimepts, nnodes = sdata.shape
    null = np.zeros(sdata.shape)
    null[:,0,:]  = np.random.randn(nsub , nnodes)
    for i in range(ntimepts -1):
        null[:,i+1,:] = null[:,i,:] * arone
    null[:,1:,:] = null[:,1:,:] + np.random.randn(nsub, ntimepts-1, nnodes)
    non_diag = np.empty((nsub, nnodes * nnodes - nnodes))
    for sub, slice in enumerate(null):
        tmpr = np.corrcoef(slice)
        non_diag[sub] = tmpr[np.eye(nnodes) < 1]
    tmpz =  0.5 * np.log( (1 + non_diag) / (1 - non_diag))
    r_to_z_correct = 1.0 / tmpz.std()
    return r_to_z_correct

def r_to_z(subs_node_stat, sdata):
    """ calc and return ztransformed data

    Parameters
    ----------
    subs_node_stat : array
        subject summary stat data (eg correlation)
        
    
    sdata : array
        subject data (nsub X ntimepoints X nnodes)
        used to calculate AR91) for z transform
    """


    arone = calc_arone(sdata)
    r_to_z_val = _calc_r2z_correction(sdata, arone)
    zdat = 0.5 * np.log(( 1 + subs_node_stat) / (1 - subs_node_stat)) \
            * r_to_z_val
    return zdat

