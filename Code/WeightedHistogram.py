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

def work_histogram(ext_bins,extensions,values):
    # W[i][j] is fec i, z [j]
    matrix_values = np.array(values)
    # switch so that z is along the first axis
    matrix_values_T = matrix_values.T
    matrix_q_T = np.array(extensions).T
    flat_vals = np.concatenate([w for w in matrix_values_T])
    ext_idx = [np.digitize(x=w,bins=ext_bins)-1 for w in matrix_q_T]
    indptr = [0] + list(np.cumsum([len(list_v) for list_v in ext_idx]))
    to_ret = csr_matrix((flat_vals,np.concatenate(ext_idx),indptr))
    return to_ret

def wham(extensions,works,kbT,n_ext_bins):
    beta = kbT
    # get the offsets for the works
    ext_bins = np.linspace(np.min(extensions),np.max(extensions),endpoint=True,
                           num=n_ext_bins)
    dq = np.median(np.diff(ext_bins))
    ext_centered = ext_bins - dq/2
    work_array = np.array(works,dtype=np.float64)
    extension_array = np.array(extensions,dtype=np.float64)
    offset = np.mean(works,axis=0)
    work_array -= offset
    ones = np.ones(work_array.shape)
    indicator = work_histogram(ext_centered, extensions,ones)
    n_z = work_array.shape[1]
    n_q = n_ext_bins
    assert indicator.shape == (n_z,n_q) , "Matrix not (N_z,N_q)"
    hist_work = work_histogram(ext_centered, extensions,work_array)
    hist_boltz = hist_work * beta
    boltzmann_weight = np.expm1(hist_boltz) + indicator
    # get the sum along the
    for i,q_tmp in enumerate(ext_centered):
        pass
    pass

