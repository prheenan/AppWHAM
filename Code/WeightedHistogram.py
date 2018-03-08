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

class LandscapeWHAM(object):
    def __init__(self,q,G0,offset_G0_of_q):
        self._q = q
        self._G0 = G0
        self._offset_G0_of_q = offset_G0_of_q
    @property
    def energy(self):
        return self._G0
    @property
    def G0(self):
        return self.energy
    @property
    def q(self):
        return self._q

def _harmonic_V(q,z,k):
    """
    :param q: extension
    :param z: spring position
    :param k: spring constant
    :return: simple harmonic potential.
    """
    return k * ((q-z)**2)/2

def _bin_with_rightmost(array,n,extra=0):
    """
    :param array: what array to bin
    :param n: how many bins to have between the min (left edge) and max
    (right edge)
    :param extra: if the max should be increased at all.
    :return: list of bins of size n+1 spanning the range of array
    """
    with_rightmost = np.linspace(np.min(array),np.max(array)+extra,
                                 endpoint=True,num=n+1)
    return with_rightmost

def wham(extensions,works,z,kbT,n_ext_bins,k):
    """
    :param extensions: list; each element is array like of extension with
    length N. The entire list has length M (number of FEC). Units of m.
    :param works: list of length M;
    each element is an array-like of work with length N. Units of J
    :param z: either a single list of length N, or an MxN list. Units of m.
    :param kbT: boltzmann energy in J
    :param n_ext_bins: number of extension bins to use
    :param k: spring constant in N/m
    :return:
    """
    beta = 1/kbT
    work_array = np.array(works,dtype=np.float64)
    extension_array = np.array(extensions,dtype=np.float64)
    z_array = np.array(z,dtype=np.float64)
    assert work_array.shape == extension_array.shape , \
        "Work and extension must have the same number"
    assert len(work_array.shape) > 1 and work_array.shape[1] > 0 , \
        "Must have at least one z point to use "
    assert work_array.shape[0] > 1 , "Must have at least 2 FEC"
    # get the extension bins, including one for the rightmost edge (last point)
    with_rightmost_q = _bin_with_rightmost(extensions,n_ext_bins,extra=0)
    with_rightmost_z = _bin_with_rightmost(z,n=z_array.shape[0],extra=1)
    bins_q = with_rightmost_q[:-1]
    bins_z = with_rightmost_z[:-1]
    # make the z matrix; allow for just passing in a single one...
    if (len(z_array.shape) == 1 or z_array.shape[1] == 0):
        z_array = np.array([z for _ in works])
    else:
        z_array = np.array(z)
    # get the potential, using the bins
    zz, qq = np.meshgrid(bins_z,bins_q)
    V_i_j = _harmonic_V(qq,zz,k)
    # determine the energy offset at each Z.
    work_offset = np.mean(work_array,axis=0)
    # offset the work and potential to avoid overflows
    W_offset = work_array - work_offset
    V_i_j_offset = (V_i_j - work_offset)
    # POST: work_array and V_i_j are now offset how we like..
    n_fec_M = W_offset.shape[0]
    n_z = W_offset.shape[1]
    n_q = n_ext_bins
    boltz_array = np.exp(-W_offset * beta)
    # get h_i_j, unnormalized
    q_flat = extension_array.flatten()
    z_flat = z_array.flatten()
    val_flat = boltz_array.flatten()
    hist = binned_statistic_2d(x=q_flat,
                               y=z_flat,
                               values=val_flat,
                               statistic='sum',
                               bins=(with_rightmost_q,with_rightmost_z))
    stat, bins_q, bins_z, binnumber = hist
    dq_hist = np.median(np.diff(bins_q))/2
    q_centered = bins_q + dq_hist
    # XXX check bins are correct
    assert stat.shape == (n_q,n_z)
    # make h_i_j --  i runs over extension q, j runs over control z --
    # by dividing by the number of curves
    h_i_j = stat/n_fec_M
    eta_i = np.mean(np.exp(-beta * W_offset),axis=0)
    assert h_i_j.shape == (n_q,n_z)
    assert eta_i.shape == (n_z,)
    assert V_i_j_offset.shape == (n_q,n_z)
    boltzmann_V_i_j = np.exp(-beta * V_i_j_offset)
    numer_j = np.sum(h_i_j/eta_i,axis=1)
    denom_j = np.sum(boltzmann_V_i_j/eta_i,axis=1)
    # make sure the shapes match and are the same
    assert numer_j.shape == denom_j.shape
    assert numer_j.shape == (n_q,)
    assert (numer_j > 0).all() , \
        "Invalid <W>_z; mean work > true work"
    assert (denom_j > 0).all() , \
        "Invalid <W>_z or V(q,z). Mean work > potential"
    G0_rel = -1/beta * (np.log(numer_j) - np.log(denom_j))
    # determine the mean work at each extension
    mean_w_q, _, _ = binned_statistic(x=q_flat,values=work_array.flatten(),
                                      bins=with_rightmost_q,statistic='mean')
    # add back in the offset to go into real units
    q = q_centered[:-1]
    offset_G0_of_q = mean_w_q
    G0 = G0_rel + offset_G0_of_q
    return LandscapeWHAM(q,G0,offset_G0_of_q)



