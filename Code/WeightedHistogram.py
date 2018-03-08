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


from scipy.sparse import csr_matrix

from scipy.stats import binned_statistic_2d

def wham(extensions,works,kbT,n_ext_bins):
    beta = kbT
    # get the offsets for the works
    ext_bins = np.linspace(np.min(extensions),np.max(extensions),endpoint=True,
                           num=n_ext_bins)
    dq = np.median(np.diff(ext_bins))
    work_array = np.array(works,dtype=np.float64)
    n_z = work_array.shape[1]
    n_q = n_ext_bins
    z_bins = np.arange(n_z)
    extension_array = np.array(extensions,dtype=np.float64)
    boltz_array = np.exp(-work_array * beta)
    partition = np.mean(boltz_array,axis=0)
    matrix_z = np.array([z_bins for w in work_array])
    hist = binned_statistic_2d(x=extension_array.flatten(),
                               y=matrix_z.flatten(),
                               values=boltz_array.flatten(),
                               statistic='sum',
                               bins=(n_q,n_z))
    stat, x, y, binnumber = hist
    q_centered = x + np.median(np.diff(x))/2
    # XXX why doesn't this work?...
    assert stat.shape == (n_q,n_z)
    # POST: have histogram[i][j]
    offset = np.mean(works,axis=0)
    work_array -= offset


