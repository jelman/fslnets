

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


