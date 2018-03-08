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

def work_histogram(ext_bins,extensions,works):
    # W[i][j] is fec i, z [j]
    matrix_works = np.array(works)
    # switch so that z is along the first axis
    matrix_works_T = matrix_works.T
    matrix_q_T = np.array(extensions).T
    histogram = [ [ [] for _i in ext_bins]
                  for _ in range(matrix_works_T.shape[0])]
    offset = np.mean(matrix_works_T,axis=1)
    for i,(q,w) in enumerate(zip(matrix_works_T,matrix_q_T)):
        idx_q = np.digitize(x=q,bins=w,right=False)
        for j,idx_q_tmp in enumerate(idx_q):
            histogram[i][idx_q_tmp].append(w[j]-offset[i])
    return histogram

def wham(extensions,works,kbT,n_ext_bins):
    beta = kbT
    # get the offsets for the works
    ext_bins = np.linspace(np.min(extensions),np.max(extensions),endpoint=True,
                           num=n_ext_bins)
    dq = np.median(np.diff(ext_bins))
    ext_centered = ext_bins - dq/2
    histogram = work_histogram(ext_centered, extensions, works)
    pass

