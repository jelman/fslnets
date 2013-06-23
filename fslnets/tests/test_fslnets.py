from unittest import TestCase, skipIf, skipUnless
import numpy as np
from numpy.random import RandomState
from numpy.testing import (assert_raises, assert_equal, assert_almost_equal)
from os.path import (exists, join, split, abspath)
import os
from .. import simple_fslnets as fsln

class TestFSLNets(TestCase):

    def setUp(self):
        """ create small example data """
        prng = RandomState(42) # consistent pseudo random number generator
        self.data = prng.randn(10, 5)
        self.good = [0, 1, 2] 


    def test_normalize_data(self):
        cleaned = fsln.normalise_data(self.data)
        assert_equal(cleaned.shape, self.data.shape)
        expected = np.array([ 0.67899281, -0.19832348,  0.58056727,
                            2.43518385,  0.40408347])
        assert_almost_equal(cleaned[0], expected)

    def test_parse_idx(self):
        good = self.good
        ntimepts, ncomp = self.data.shape
        good_idx, bad_idx = fsln.parse_idx(good, ncomp)
        assert_equal(good_idx, np.array([0, 1, 2]))
        assert_equal(bad_idx, np.array([3, 4]))
        ## raise error if good "ids" not in range of n
        assert_raises(IOError, fsln.parse_idx, good, 2)

    def test_regress(self):
        ntimepts, ncomp = self.data.shape
        cleaned = fsln.normalise_data(self.data)
        gids, bids = fsln.parse_idx(self.good, ncomp)
        good = cleaned[:,gids]
        bad = cleaned[:, bids]
        assert_almost_equal(good[0], 
                            np.array([0.67899281, -0.19832348,  0.58056727]))
        assert_almost_equal(bad[0],
                            np.array([ 2.43518385,  0.40408347]))
        res = fsln.simple_regress(good, bad)
        assert_almost_equal(res[0], 
                            np.array([0.35004692, -0.38582794, -0.27400644 ]))

    def test_remove_regress_bad_components(self):
        cleaned = fsln.normalise_data(self.data)
        res = fsln.remove_regress_bad_components(cleaned, self.good)
        assert_almost_equal(res[0],
                            np.array([0.35004692, -0.38582794, -0.27400644 ]))

    def test_concat_subjects(self):
        tmp = [self.data, self.data]
        ntimepts, ncomp = self.data.shape
        concat = fsln.concat_subjects(tmp)
        assert_equal(concat.shape, (2, ntimepts, ncomp))
        assert_equal(concat[0,:,:], self.data)

    def test_corrcoef(self):
        res = fsln.corrcoef(self.data)
        _, ncomp = self.data.shape
        assert_equal(res.shape, (ncomp, ncomp))
        assert_equal(np.diag(res), np.zeros(ncomp))
        expected = np.corrcoef(self.data.T)
        np.fill_diagonal(expected,0)
        assert_equal(res, expected)

    def test_calc_arone(self):
        sdata = fsln.concat_subjects([self.data, self.data])
        arone = fsln.calc_arone(sdata)
        assert_almost_equal(0.11384589, arone)
        ## should raise error if data not 3d nsub X ntp X ncomponents
        assert_raises(IndexError, fsln.calc_arone,  self.data)

    def test_calc_r2z_correction(self):
        sdata = fsln.concat_subjects([self.data, self.data])
        arone = fsln.calc_arone(sdata)
        r2z_correction = fsln._calc_r2z_correction(sdata, arone)
        assert_equal(r2z_correction > 0, True)

    def test_cond_partial_cor(self):
        # make correlated data
        newdata = self.data.copy()
        newdata[:,1] = newdata[:,0] + 0.1 * newdata[:,2]
        pcorr = fsln._cond_partial_cor(newdata[:,:2], newdata[:,2:])
        assert_equal(pcorr, 1.0)
        pcorr = fsln._cond_partial_cor(newdata[:,:2])
        assert_almost_equal(pcorr, 0.99531066,decimal=7 )

    def test_partial_corr(self):
        # make correlated data
        newdata = self.data.copy()
        newdata[:,1] = newdata[:,0] + 0.1 * newdata[:,2]
        res = fsln.partial_corr(newdata) 
        assert_equal(res.shape, (5,5))
        assert_equal(res[0,0], 0) # diag is zeros
        assert_equal(res[0,1] , 1.0) # perfect correl once conf regressed
        assert_almost_equal(res[0,2], -1.000,decimal = 2) 
        # neg correl once regressor removed






if __name__ == '__main__':
    unittest.main()
