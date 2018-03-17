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
import warnings

from scipy.stats import binned_statistic_2d, binned_statistic
from .UtilLandscape import BidirectionalUtil, Conversions

class LandscapeWHAM(object):
    def __init__(self,q,G0,offset_G0_of_q,beta):
        """
        :param q: extension in meters, size N
        :param G0: energy in J, size N
        :param offset_G0_of_q: offset used for energy, size N
        :param beta:  1/kbT, units of J
        """
        self._q = q
        self._G0 = G0
        self._offset_G0_of_q = offset_G0_of_q
        self.beta = beta
    @property
    def energy(self):
        return self._G0
    @property
    def G0(self):
        return self.energy
    @property
    def q(self):
        return self._q
    @property
    def q_nm(self):
        return self.q * 1e9
    @property
    def G0_kT(self):
        return self.G0 * self.beta
    @property
    def G0_kcal_per_mol(self):
        # J/mol -> kcal/mol (note that the 1/mol are implicit)
        return self.G0 * Conversions.kcal_per_mol_per_J()

class InputWHAM(object):
    def __init__(self,extensions,works,z,kbT,n_ext_bins,k):
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
        self.extensions = extensions
        self.works = works
        self.z = z
        self.kbT = kbT
        self.n_ext_bins = n_ext_bins
        self.k = k
    @property
    def n(self):
        return np.array(self.works).shape[0]

class _HistogramTerms(object):
    def __init__(self, boltz_array, V_i_j_offset, extension_array,z_array,
                 q_hist,z_hist,W_offset,beta):
        self.boltz_array = boltz_array
        self.V_i_j_offset = V_i_j_offset
        self.z_array = z_array
        self.extension_array = extension_array
        self.with_rightmost_q = q_hist
        self.with_rightmost_z = z_hist
        self.W_offset = W_offset
        self.beta = beta
    @property
    def n_fec_M(self):
        return self.boltz_array.shape[0]
    @property
    def bins_z(self):
        return self.with_rightmost_z[:-1]
    @property
    def bins_q(self):
        return self.with_rightmost_q[:-1]
    @property
    def n_z(self):
        return self.bins_z.size
    @property
    def n_q(self):
        return self.bins_q.size

def _harmonic_V(q,z,k):
    """
    :param q: extension
    :param z: spring position
    :param k: spring constant
    :return: simple harmonic potential.
    """
    return k * ((q-z)**2)/2

def _bin_with_rightmost(array,n):
    """
    :param array: what array to bin
    :param n: how many bins to have between the min (left edge) and max
    (right edge)
    :return: list of bins of size n+1 spanning the range of array
    """
    with_rightmost = np.linspace(np.min(array),np.max(array),
                                 endpoint=True,num=n+1)
    return with_rightmost

def _histogram_terms(z,extensions,works,n_ext_bins,work_offset,k,beta):
    """
    :param z: array of size M; each element are the spring positions (i.e. z,
    e.g. the stage position in AFM) of size N. Units of m
    :param extensions: see z, except for molecular extension. Units of m
    :param works: see z, except the work in J
    :param n_ext_bins: number of extension bins to use
    :param work_offset: how to offset the work, in J. Size MxN
    :param k: spring constant, in N/m
    :param beta: the inverse boltzmann energy (1/kbT) in J
    :return: _HistogramTerms object
    """
    work_array = np.array(works,dtype=np.float64)
    extension_array = np.array(extensions,dtype=np.float64)
    z_array = np.array(z,dtype=np.float64)
    assert work_array.shape == extension_array.shape , \
        "Work and extension must have the same number"
    assert len(work_array.shape) > 1 and work_array.shape[1] > 0 , \
        "Must have at least one z point to use "
    assert work_array.shape[0] > 1 , "Must have at least 2 FEC"
    # get the extension bins, including one for the rightmost edge (last point)
    with_rightmost_q = _bin_with_rightmost(extensions,n_ext_bins)
    with_rightmost_z = _bin_with_rightmost(z,n=z_array.shape[0])
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
    assert work_offset.size == work_array.shape[1]
    # offset the work and potential to avoid overflows
    W_offset = work_array - work_offset
    V_i_j_offset = (V_i_j - work_offset)
    # POST: work_array and V_i_j are now offset how we like..
    boltz_array = np.exp(-W_offset * beta)
    to_ret = _HistogramTerms(boltz_array, V_i_j_offset, extension_array,z_array,
                             with_rightmost_q,with_rightmost_z,W_offset,beta)
    return to_ret

def _wham_sum_hij_times_M(fwd,value_array):
    """
    :param fwd: InputWHAM object. If none,  return 0
    :return: h_ij * M, where h_ij is defined in Hummer, PNAS, 2010, SI, eq S3
    """
    if fwd is None:
        return 0
    # get h_i_j, unnormalized
    q_flat = fwd.extension_array.flatten()
    z_flat = fwd.z_array.flatten()
    val_flat = value_array.flatten()
    hist = binned_statistic_2d(x=q_flat,
                               y=z_flat,
                               values=val_flat,
                               statistic='sum',
                               bins=(fwd.with_rightmost_q,fwd.with_rightmost_z))
    stat, bins_q, bins_z, binnumber = hist
    # XXX check bins?
    return stat

def get_terms(fwd,work_offset,beta):
    """
    ease-of-use funciton for _histogram_terms

    :param fwd: InputWHAM object (not necessarily forward)
    :param work_offset:  how to offset fwd
    :param beta: 1/kbT, in J
    :return: see _histogram_terms
    """
    if (fwd is None):
        return None
    fwd = _histogram_terms(fwd.z, fwd.extensions, fwd.works, fwd.n_ext_bins,
                           work_offset, fwd.k, beta)
    return fwd

def _weighted_value(terms,f,**kw):
    """
    :param terms: see get_terms
    :param f: function to use for weighting
    (e.g. BidirectionalUtil.ForwardWeighted)
    :param kw: passed to f
    :return: see f
    """
    if (terms is None):
        return 0
    Wn = np.array([w[-1]*np.ones(w.size) for w in terms.W_offset])
    fwd_value = f(v=1,W=terms.W_offset,Wn=Wn,**kw)
    return fwd_value

def h_ij_bidirectional(terms,**kw):
    """
    :param terms:see  see get_terms
    :param kw: passed to _weighted_value
    :return: see _wham_sum_hij_times_M
    """
    if (terms is None):
        return 0
    fwd_value = _weighted_value(terms,**kw)
    fwd_h = _wham_sum_hij_times_M(terms,value_array=fwd_value)
    return fwd_h/terms.n_fec_M

def _G0_from_parition(boltz_fwd,h_fwd,boltz_rev,h_rev,key_terms):
    """
    :param boltz_fwd: eta_i in the forward direction, or 0 if no forward
    :param h_fwd: see h_ij_bidirectional, for the forward direction
    :param boltz_rev: see boltz_fwd
    :param h_rev: see h_fwd
    :param key_terms: see get_terms; used for (e.g.) getting bin sizes, beta
    :return:
    """
    # XXX check bins are correct
    n_q = key_terms.n_q
    n_z = key_terms.n_z
    n_fec_M = key_terms.n_fec_M
    beta = key_terms.beta
    # POST: things are OK.
    combined_weighted = boltz_fwd + boltz_rev
    h_combined = h_fwd + h_rev
    eta_bidir = np.sum(combined_weighted, axis=0) / n_fec_M
    # make h_i_j --  i runs over extension q, j runs over control z --
    # by dividing by the number of curves
    h_i_j = h_combined
    eta_i = eta_bidir
    assert h_i_j.shape == (n_q, n_z)
    assert eta_i.shape == (n_z,)
    assert key_terms.V_i_j_offset.shape == (n_q, n_z)
    boltzmann_arg_ij = -beta * key_terms.V_i_j_offset
    boltzmann_arg_ij = np.maximum(boltzmann_arg_ij,-700)
    boltzmann_arg_ij = np.minimum(boltzmann_arg_ij,700)
    boltzmann_V_i_j = BidirectionalUtil.Exp(boltzmann_arg_ij)
    numer_j = np.sum(h_i_j / eta_i, axis=1)
    denom_j = np.sum(boltzmann_V_i_j / eta_i, axis=1)
    # make sure the shapes match and are the same
    assert numer_j.shape == denom_j.shape
    assert numer_j.shape == (n_q,)
    if not (numer_j > 0).all():
        warnings.warn("Invalid <W>_z; mean work > true work",RuntimeWarning)
    if not (denom_j > 0).all():
        warnings.warn("Invalid <W>_z or V(q,z). Mean work > potential",
                      RuntimeWarning)
    G0_rel = -1 / beta * (np.log(numer_j) - np.log(denom_j))
    return G0_rel

def wham(fwd_input=None,rev_input=None):
    """
    :param fwd: InputWHAM object
    :return: LandscapeWHAM
    """
    n_f = fwd_input.n if fwd_input is not None else 0
    n_r = rev_input.n if rev_input is not None else 0
    have_fwd = n_f > 0
    have_rev = n_r > 0
    assert have_fwd or have_rev , "No forward or reverse data; can't do anything"
    # get the key (for getting beta and such)
    key_input = fwd_input if have_fwd else rev_input
    beta = 1/key_input.kbT
    if (have_fwd and have_rev):
        delta_A = BidirectionalUtil._solve_DeltaA(fwd_input.works,
                                                 rev_input.works,
                                                 offset_fwd=0,
                                                 beta=beta)
    else:
        delta_A = 0
    work_offset = np.mean(key_input.works,axis=0)
    # get the forward
    fwd_terms = get_terms(fwd_input, work_offset, beta)
    rev_terms = get_terms(rev_input, work_offset, beta)
    if (have_fwd and not have_rev):
        # use forward
        key_terms = fwd_terms
    elif (have_rev and not have_fwd):
        # use reverse
        key_terms = rev_terms
    else:
        # use both; key_terms will be forward (arbitrary)
        key_terms = fwd_terms
    kw_bidir = dict(delta_A=delta_A,beta=beta,nf=n_f,nr=n_r)
    fwd_weight = BidirectionalUtil.ForwardWeighted
    rev_weight = BidirectionalUtil.ReverseWeighted
    kw_fwd = dict(terms=fwd_terms,f=fwd_weight,**kw_bidir)
    kw_rev = dict(terms=rev_terms,f=rev_weight,**kw_bidir)
    h_fwd = h_ij_bidirectional(**kw_fwd)
    h_rev = h_ij_bidirectional(**kw_rev)
    # determine which will be the key_terms
    kw_key = kw_fwd if have_fwd else kw_rev
    dq_hist = np.median(np.diff(key_terms.bins_q))/2
    q_centered = key_terms.bins_q + dq_hist
    # get the bidirectional estimators
    boltz_fwd = _weighted_value(**kw_fwd)
    boltz_rev = _weighted_value(**kw_rev)
    # make sure if things aren't directional, nothing is changing.
    if not have_rev:
        assert boltz_rev == 0
        assert h_rev == 0
    if not have_fwd:
        assert boltz_fwd == 0
        assert h_fwd == 0
    G0_rel = _G0_from_parition(boltz_fwd,h_fwd,boltz_rev,h_rev,key_terms)
    # add back in the offset to go into real units
    q = q_centered
    offset_G0_of_q = 0
    G0 = G0_rel + offset_G0_of_q
    return LandscapeWHAM(q,G0,offset_G0_of_q,beta=beta)



