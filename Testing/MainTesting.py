# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../")
from AppWHAM.Code import WeightedHistogram, UtilWHAM
from AppWHAM.Lib.SimulationFEC import Test
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d


def check_losses(expected,predicted,atol,max_rel_loss=0.0137,rtol=2e-2,
                 assert_doesnt_match=False):
    """
    :param expected: "actual" landscape
    :param predicted: WHAM prediction
    :param atol: absolute energy tol (J)
    :param max_rel_loss: maximum loss (relative to sum of all eergy)
    :param rtol: relative tolerance
    :param assert_doesnt_match: if true, makes the the loss is *worse* than
    expected by at least a factor of 10
    :return:  nothing, throws error if something goes wrong
    """
    loss = np.abs(predicted-expected)
    loss_rel = np.sum(loss)/np.sum(np.mean([predicted,expected],axis=0))
    if assert_doesnt_match:
        assert not np.isfinite(loss_rel) or (loss_rel > max_rel_loss * 10)
    else:
        assert loss_rel < max_rel_loss
        np.testing.assert_allclose(predicted,expected,atol=atol,rtol=rtol)

def expected_bidirectional(data_base,q_predicted):
    """
    :param data_base: where the data live
    :param q_predicted: the extenesions we want the landscpe at
    :return: expected G0 in J
    """
    bidir = np.loadtxt(data_base + "data_bidir.csv",delimiter=",")
    expected = bidir
    ext_fwd_m, G_fwd_J = expected[:,0] * 1e-9, expected[:,1]*4.1e-21
    # interpolate the expected landscape onto the actual one
    interp_expected = interp1d(x=ext_fwd_m, y=G_fwd_J, kind='linear',
                               fill_value='extrapolate',bounds_error=False)
    G0_expected = interp_expected(q_predicted)
    G0_expected -= np.min(G0_expected)
    return G0_expected

def tst_landscapes(fwd_wham,rev_wham):
    wham_landcape = WeightedHistogram.wham(fwd_input=fwd_wham,
                                           rev_input=rev_wham)
    q = wham_landcape.q
    data_base = "../data/"
    G0_expected = expected_bidirectional(data_base, q)
    G0_WHAM = wham_landcape.G0
    G0_WHAM -= min(G0_WHAM)
    kT = 1 / wham_landcape.beta
    kw_err = dict(atol=1.25 * kT, max_rel_loss=0.0162, rtol=3e-2)
    check_losses(expected=G0_expected, predicted=G0_WHAM, **kw_err)
    # check the forward is close
    wham_landcape_fwd = WeightedHistogram.wham(fwd_input=fwd_wham)
    G0_fwd = wham_landcape_fwd.G0
    G0_fwd -= min(G0_fwd)
    check_losses(expected=G0_expected, predicted=G0_fwd, **kw_err)
    # check the 'forward as reverse' ist close
    wham_landcape_fwd_2 = WeightedHistogram.wham(rev_input=fwd_wham)
    G0_fwd_2 = wham_landcape_fwd_2.G0
    G0_fwd_2 -= min(G0_fwd_2)
    check_losses(expected=G0_expected, predicted=G0_fwd_2,
                 assert_doesnt_match=True, **kw_err)
    # check  the 'reverse as fwd' isnt close
    wham_landcape_rev_2 = WeightedHistogram.wham(fwd_input=rev_wham)
    G0_rev_2 = wham_landcape_rev_2.G0
    G0_rev_2 -= min(G0_rev_2)
    check_losses(expected=G0_expected, predicted=G0_rev_2,
                 assert_doesnt_match=True, **kw_err)
    # check that the reverse is close
    wham_landcape_rev = WeightedHistogram.wham(rev_input=rev_wham)
    G0_rev = wham_landcape_rev.G0
    G0_rev -= min(G0_rev)
    check_losses(expected=G0_expected, predicted=G0_rev, **kw_err)

def _check_whitebox(expected_terms,actual_terms):
    error_kw = dict(atol=1e-30, rtol=1e-6)
    fs = [lambda x: x.z_array,
          lambda x: x.extension_array,
          lambda x: x.with_rightmost_q,
          lambda x: x.with_rightmost_z,
          lambda x: x.beta,
          lambda x: x.W_offset,
          lambda x: x.work_subtracted,
          lambda x: x.boltz_array,
          lambda x: x.V_i_j_offset]
    for i,f in enumerate(fs):
        expected = f(expected_terms)
        actual = f(actual_terms)
        np.testing.assert_allclose(expected,actual,**error_kw)

def tst_whitebox(fwd_input,rev_input):
    # get the terms with fwd and reverse
    rev_terms, fwd_terms, key_terms, delta_A, n_f, n_r, beta = \
        WeightedHistogram._term_helper(fwd_input=fwd_input, rev_input=rev_input)
    h_fwd, h_rev, boltz_fwd, boltz_rev = \
        WeightedHistogram._h_and_boltz_helper(fwd_terms, rev_terms, delta_A,
                                              beta, n_f, n_r)
    # get the arguments when the forward or reverse are missing; make sure they
    # are the same
    dict_vals = [ dict(fwd_input=fwd_input, rev_input=rev_input),
                  dict(fwd_input=fwd_input, rev_input=None),
                  dict(fwd_input=None, rev_input=rev_input)]
    for kw in dict_vals:
        check_fwd = kw['fwd_input'] is not None
        check_rev = kw['rev_input'] is not None
        n_f_tmp = n_f if check_fwd else 0
        n_r_tmp = n_r if check_rev else 0
        delta_A_tmp = delta_A if check_fwd and check_rev else 0
        dict_final = dict(delta_A=delta_A_tmp,beta=beta, n_f=n_f_tmp,
                          n_r=n_r_tmp,**kw)
        rev_terms_tmp, fwd_terms_tmp, key_terms_tmp, delta_A, n_f, n_r, beta = \
            WeightedHistogram._term_helper(**kw)
        if (check_fwd):
            _check_whitebox(fwd_terms_tmp, fwd_terms)



def tst_hummer():
    """
    :return: Nothing; tests humer data
    """
    fwd,rev = Test.HummerData(n=100)
    fwd_wham = UtilWHAM.to_wham_input(fwd)
    rev_wham = UtilWHAM.to_wham_input(rev)
    # do some 'whitebox' testing, to make sure things are OK...
    #tst_whitebox(fwd_wham, rev_wham)
    # make sure the landscapes are ok
    tst_landscapes(fwd_wham,rev_wham)




def run():
    tst_hummer()

if __name__ == "__main__":
    run()
