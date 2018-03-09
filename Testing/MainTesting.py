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

sys.path.append("../")
from Code import WeightedHistogram
from Lib.SimulationFEC import Test
from scipy.integrate import cumtrapz

def run():
    fwd,rev = Test.HummerData(n=100)
    key = fwd[0]
    z = key.Offset + key.Velocity * (key.Time - min(key.Time))
    extensions = [f.Extension for f in fwd]
    forces = [f.Force for f in fwd]
    works = [cumtrapz(y=f,x=z,initial=0) for f in forces]
    wham_landcape = WeightedHistogram.wham(extensions=extensions,
                                           z=z,
                                           works=works,
                                           kbT=key.kT,k=key.SpringConstant,
                                           n_ext_bins=50)
    data_base = "../data/"
    fwd = np.loadtxt(data_base + "data_fwd.csv",delimiter=",")
    bidir = np.loadtxt(data_base + "data_bidir.csv",delimiter=",")
    expected = bidir
    ext_fwd_m, G_fwd_kT = expected[:,0] * 1e-9, expected[:,1]
    offset = wham_landcape._offset_G0_of_q
    q = wham_landcape.q
    offset_G_0 = (wham_landcape.G0-offset)
    offset_G_0 -= min(offset_G_0)
    plt.plot(q,(offset_G_0-14e-12*q)/4.1e-21)
    plt.plot(ext_fwd_m,(G_fwd_kT *4.1e-21 - 14e-12 * ext_fwd_m)/4.1e-21)
    plt.show()
    pass

if __name__ == "__main__":
    run()
