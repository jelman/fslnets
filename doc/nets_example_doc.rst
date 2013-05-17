nets_example
============

Requirements
++++++++++++

* FSL 5.0  to 5.0.2.2

Libraries (matlab)
++++++++++++++++++

* L1precision toolbox
* pairwise causality tools ( pwling) 
  (less important at this point)

Inputs
++++++

* your-group-ICA.ica/melodic_IC

  4D file of group components (output fo group ICA)

* directory of Dual_regression outputs

* list of **good** nodes, anything not listed in this will be regressed
  out  (**NOTE:** counting starts at 0)

* How important is TR in this analysis? Is it worth looking at or combining
  the TR 189's ?


Main Tools
++++++++++

slices_summary tool
-------------------
    (Low priority, omly FSL 5.0+)
    This is a wrapper for a FSL tool that takes in the melodic_IC and 
    creates summary images (one for each component) for display of components
    (specifically for heirarchical clustering display of components)
  
    eg. nets_hierarchy(meanCORR,meanPCORR,ts.DD,sprintf('%s.sum',group_maps));


net_load
--------

    Loads timeseries data

    loads/ stores the TR

    MOST IMPT: control variance normalization

        0=none, 
        
        1=normalise whole subject stddev, 
          normalize all components at once for one subject 
          (high priority and default)
        
        2=normalise each separate timeseries from each subject
         (timeseries normalized individually)

    output is used in basically every other algorithm, preprocessing


nets_spectra
------------

    diagnostic to view spectra before and after regressing out bad components

nets_tsclean
------------

    regress bad ICs out of good and removes timeseries



nets_pics
---------

    another diagnostic to quickly view good and bad components



nets_make_mats
--------------

    generate network matricies

    covariance vs amplitude vs full correlation vs regularized partial
    vs unregularized partial, vs Hyvariens pairwise causality measure


nets_r2z
---------

    implements z-transform 


nets_consistency
----------------

    what does this do? seems more diagnostic

nets_hierarchy
--------------

    calc average correlation or partial correlation across subjects

    run heirarchical clustering, arg1 lower triangular, arg2 upper triangular

    not highest priority


nets_glm
--------

    glm on all connections, **OR** glm on connections above threshold (eg strong)
    masked by raw t-stat 

    check how interacts with randomise

    can this be edited to do a single group with covariates?

    dichotomous and continuous

    tool to generate design.mat, design.con  files (later feature)
    esp importtant when working with groups non-contiguous with Subject ids

nets_lda
--------

    linear discriminat analysis (cross subject multivariate)

    just for two group cases

    outputs from Jeremy listserve


nets_boxplots
-------------

    create boxplots for the two groups for a network-matrix-element 
    of interest (e.g., selected from GLM output)

    only for **two-group** test

        
