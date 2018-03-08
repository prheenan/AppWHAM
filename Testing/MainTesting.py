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

def run():
    WeightedHistogram.wham(extensions=[[1,2,3,4],[1,2,3,4]],
                           z=[1,2,3,4],
                           works=[[0,2,10,20],[0,3,12,30]],
                           kbT=4.1e-21,k=1e-3,
                           n_ext_bins=3)

if __name__ == "__main__":
    run()
