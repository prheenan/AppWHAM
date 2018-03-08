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

def _hist_mean()

def _histogram(extension_array,value_array,ext_bins):
    n_fec = value_array.shape[0]
    n_z = value_array.shape[1]
    n_q = ext_bins.size
    matrix_q_idx = np.digitize(x=extension_array,bins=ext_bins)-1
    histogram = [ [ [] for _i in range(n_z)] for _j in range(n_q) ]
    for i,b in enumerate(ext_bins):
        # XXX inefficient
        # determine where any fecs
        idx_this_b = np.where(matrix_q_idx == i)
        elements = value_array[idx_this_b]
        idx_z = idx_this_b[1]
        tmp = []
        for idx_z,e in zip(idx_z,elements):
            to_append = e
            histogram[i][idx_z].append(to_append)
    assert len(histogram) == n_q
    assert len(histogram[0]) == n_z
    # loop throught and get the mean
    to_ret_mean = np.zeros((n_q,n_z))
    for j_q, list_v in enumerate(histogram):
        for i_z, list_mean in enumerate(list_v):
            if len(list_mean) > 0:
                to_set = np.sum(list_mean)/n_fec
            else:
                to_set = np.nan
            to_ret_mean[j_q, i_z] = to_set
    return to_ret_mean

def wham(extensions,works,kbT,n_ext_bins):
    beta = kbT
    # get the offsets for the works
    ext_bins = np.linspace(np.min(extensions),np.max(extensions),endpoint=True,
                           num=n_ext_bins)
    dq = np.median(np.diff(ext_bins))
    ext_centered = ext_bins - dq/2
    work_array = np.array(works,dtype=np.float64)
    extension_array = np.array(extensions,dtype=np.float64)
    boltz_array = np.exp(-work_array * beta)
    boltz = np.mean(boltz_array,axis=0)
    # histogram [i][j] is list with extension i, z position j. Elements of
    # list are values with that extension and position
    hist = _histogram(extension_array=extension_array, value_array=work_array,
                      ext_bins=ext_centered)
    # POST: have histogram[i][j]
    offset = np.mean(works,axis=0)
    work_array -= offset


