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

from scipy.stats import binned_statistic_2d, binned_statistic
from ..Lib.UtilLandscape import BidirectionalUtil

from scipy.integrate import cumtrapz
from . import WeightedHistogram

def to_wham_input(objs):
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