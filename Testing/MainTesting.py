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
    denom = np.mean([predicted, expected], axis=0)
    ok_idx = np.where(np.isfinite(predicted) & (denom > 0))
    predicted = predicted[ok_idx]
    expected = expected[ok_idx]
    denom = denom[ok_idx]
    loss = np.abs(predicted-expected)
    loss_rel = np.sum(loss)/np.sum(denom)
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

def _check_whitebox(expected_terms,actual_terms,max_median_loss = 0.151):
    error_kw = dict(atol=1e-30, rtol=1e-2)
    fs = [ [lambda x: x.with_rightmost_q,error_kw],
           [lambda x: x.with_rightmost_z,error_kw],
           [lambda x: x.beta,error_kw]]
    for i,(f,error_kw_tmp) in enumerate(fs):
        expected = f(expected_terms)
        actual = f(actual_terms)
        np.testing.assert_allclose(expected,actual,**error_kw_tmp)
    # check the V a little differently
    V_exp = expected_terms.V_i_j_offset
    V_actual = actual_terms.V_i_j_offset
    V_loss = np.abs(V_exp - V_actual)
    V_mean = np.mean([V_exp, V_actual], axis=0)
    V_rel_loss = (V_loss / V_mean)
    median_loss = np.median(V_rel_loss)
    assert median_loss < max_median_loss

def tst_whitebox(fwd_input,rev_input):
    # get the terms with fwd and reverse
    rev_terms, fwd_terms, key_terms, delta_A, n_f, n_r, beta = \
        WeightedHistogram._term_helper(fwd_input=fwd_input, rev_input=rev_input)
    h_fwd, h_rev, boltz_fwd, boltz_rev = \
        WeightedHistogram._h_and_boltz_helper(fwd_terms, rev_terms, delta_A,
                                              beta, n_f, n_r)
    # try when we just have reverse
    rev_terms_only, _, _, delta_A_rev, n_f_rev, n_r_rev, beta_rev= \
        WeightedHistogram._term_helper(fwd_input=None, rev_input=rev_input)
    _, h_rev_only, _, boltz_rev_only = \
        WeightedHistogram._h_and_boltz_helper(None, rev_terms_only, delta_A_rev,
                                              beta_rev, n_f_rev, n_r_rev)
    # make sure the fwd and reverse terms match OK
    _check_whitebox(key_terms, rev_terms)
    # key_terms, boltz_fwd, boltz_rev, h_fwd, h_rev
    term_dicts = [ [rev_terms, 0, boltz_rev, 0, h_rev],
                   [key_terms, boltz_fwd, boltz_rev, h_fwd, h_rev]]
    numers, denoms = [],[]
    for terms in term_dicts:
        numer,denom = WeightedHistogram._numer_and_denom(*terms)
        numers.append(numer)
        denoms.append(denom)
    to_plot = [-np.log(n/d) for n,d in zip(numers,denoms)]
    colors = ['r','b','g']
    key = to_plot[-1]
    for p,c in zip(to_plot,colors):
        where_finite = np.where(np.isfinite(p))
        # XXX assert finite everywhere?
        # all units are kbT
        np.testing.assert_allclose(p[where_finite],key[where_finite],
                                   atol=3,rtol=1e-2)
    # POST: 'just reverse' is pretty blose to bidirectional
    pass

def tst_hummer():
    """
    :return: Nothing; tests humer data
    """
    fwd,rev = Test.HummerData(n=100)
    fwd_wham = UtilWHAM.to_wham_input(fwd)
    rev_wham = UtilWHAM.to_wham_input(rev)
    # do some 'whitebox' testing, to make sure things are OK...
    tst_whitebox(fwd_wham, rev_wham)
    # make sure the landscapes are ok
    tst_landscapes(fwd_wham,rev_wham)




def run():
    tst_hummer()

if __name__ == "__main__":
    run()
