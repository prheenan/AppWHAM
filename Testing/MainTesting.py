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

sys.path.append("..")
from Code import WeightedHistogram, UtilWHAM
from Lib.SimulationFEC import Test
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d


def check_losses(expected,predicted,atol,max_rel_loss=0.0137,rtol=2e-2):
    loss = np.abs(predicted-expected)
    loss_rel = np.sum(loss)/np.sum(np.mean([predicted,expected],axis=0))
    assert loss_rel < max_rel_loss
    np.testing.assert_allclose(predicted,expected,atol=atol,rtol=rtol)

def expected_bidirectional(data_base,q_predicted):
    bidir = np.loadtxt(data_base + "data_bidir.csv",delimiter=",")
    expected = bidir
    ext_fwd_m, G_fwd_J = expected[:,0] * 1e-9, expected[:,1]*4.1e-21
    # interpolate the expected landscape onto the actual one
    interp_expected = interp1d(x=ext_fwd_m, y=G_fwd_J, kind='linear',
                               fill_value='extrapolate',bounds_error=False)
    G0_expected = interp_expected(q_predicted)
    G0_expected -= np.min(G0_expected)
    return G0_expected

def run():
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
    check_losses(expected=G0_expected, predicted=G0_WHAM, atol=1.25*kT,
                 max_rel_loss=0.0137, rtol=2e-2)
    plt.plot(q,G0_WHAM,'r')
    plt.plot(q,G0_expected,color='g')
    plt.show()
    pass

if __name__ == "__main__":
    run()
