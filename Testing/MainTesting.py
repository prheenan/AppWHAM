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

def tst_hummer():
    """
    :return: Nothing; tests humer data
    """
    fwd,rev = Test.HummerData(n=100)
    fwd_wham = UtilWHAM.to_wham_input(fwd)
    rev_wham = UtilWHAM.to_wham_input(rev)
    wham_landcape = WeightedHistogram.wham(fwd_input=fwd_wham,
                                           rev_input=rev_wham)
    q = wham_landcape.q
    data_base = "../data/"
    G0_expected = expected_bidirectional(data_base,q)
    G0_WHAM = wham_landcape.G0
    G0_WHAM -= min(G0_WHAM)
    kT = 1/wham_landcape.beta
    kw_err = dict(atol=1.25*kT,max_rel_loss=0.0162, rtol=3e-2)
    check_losses(expected=G0_expected, predicted=G0_WHAM,**kw_err)
    # check the forward is close
    wham_landcape_fwd = WeightedHistogram.wham(fwd_input=fwd_wham)
    G0_fwd = wham_landcape_fwd.G0
    G0_fwd -= min(G0_fwd)
    check_losses(expected=G0_expected, predicted=G0_fwd,**kw_err)
    # check the 'forward as reverse' ist close
    wham_landcape_fwd_2 = WeightedHistogram.wham(rev_input=fwd_wham)
    G0_fwd_2 = wham_landcape_fwd_2.G0
    G0_fwd_2 -= min(G0_fwd_2)
    check_losses(expected=G0_expected, predicted=G0_fwd_2,
                 assert_doesnt_match=True,**kw_err)
    # check  the 'reverse as fwd' isnt close
    wham_landcape_rev_2 = WeightedHistogram.wham(fwd_input=rev_wham)
    G0_rev_2 = wham_landcape_rev_2.G0
    G0_rev_2 -= min(G0_rev_2)
    check_losses(expected=G0_expected, predicted=G0_rev_2,
                 assert_doesnt_match=True,**kw_err)
    # check that the reverse is close
    wham_landcape_rev = WeightedHistogram.wham(rev_input=rev_wham)
    G0_rev = wham_landcape_rev.G0
    G0_rev -= min(G0_rev)
    check_losses(expected=G0_expected, predicted=G0_rev,**kw_err)



def run():
    tst_hummer()

if __name__ == "__main__":
    run()
