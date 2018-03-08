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
    plt.plot(wham_landcape.q,
             (wham_landcape.G0-wham_landcape._offset_G0_of_q) / 4.1e-21)
    plt.show()
    pass

if __name__ == "__main__":
    run()
