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

def to_wham(objs):
    key = objs[0]
    z = key.Offset + key.Velocity * (key.Time - min(key.Time))
    extensions = [f.Extension for f in objs]
    forces = [f.Force for f in objs]
    works = [cumtrapz(y=f,x=z,initial=0) for f in forces]
    dict_obj = dict(extensions=extensions,
                    z=z,
                    works=works,
                    kbT=key.kT,
                    k=key.SpringConstant,
                    n_ext_bins=200)
    to_ret = WeightedHistogram.InputWHAM(**dict_obj)
    return to_ret

def run():
    fwd,rev = Test.HummerData(n=100)
    fwd_wham = to_wham(fwd)
    rev_wham = to_wham(rev)
    wham_landcape = WeightedHistogram.wham(fwd_input=fwd_wham,
                                           rev_input=rev_wham)
    data_base = "../data/"
    fwd = np.loadtxt(data_base + "data_fwd.csv",delimiter=",")
    bidir = np.loadtxt(data_base + "data_bidir.csv",delimiter=",")
    expected = bidir
    ext_fwd_m, G_fwd_kT = expected[:,0] * 1e-9, expected[:,1]
    offset = wham_landcape._offset_G0_of_q
    q = wham_landcape.q
    offset_G_0 = (wham_landcape.G0-offset)
    offset_G_0 -= min(offset_G_0)
    plt.plot(q,offset_G_0,'r')
    plt.plot(ext_fwd_m,G_fwd_kT*4.1e-21 - min(G_fwd_kT*4.1e-21),color='g')
    plt.show()
    pass

if __name__ == "__main__":
    run()
