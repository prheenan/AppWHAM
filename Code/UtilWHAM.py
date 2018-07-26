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

from scipy.integrate import cumtrapz
from . import WeightedHistogram

def _wham_input_from_z(objs,z,n_ext_bins):
    """
    :param objs: n_ext_bins: see _wham_input_from_z
    :param n_ext_bins: see _wham_input_from_z
    :param z: the z values to use when calculating work. z[i] should correspond
    to (roughly) f[i].
    :return:  see _wham_input_from_z
    :return:
    """
    assert len(objs) > 0
    key = objs[0]
    extensions = [f.Extension for f in objs]
    forces = [f.Force for f in objs]
    works = [cumtrapz(y=f,x=z,initial=0) for f in forces]
    dict_obj = dict(extensions=extensions,
                    z=z,
                    works=works,
                    kbT=key.kT,
                    k=key.SpringConstant,
                    n_ext_bins=n_ext_bins)
    to_ret = WeightedHistogram.InputWHAM(**dict_obj)
    return to_ret

def to_wham_input(objs,n_ext_bins=200):
    """
    :param objs: n_ext_bins: see _wham_input_from_z
    :param n_ext_bins: see _wham_input_from_z
    :return:  see _wham_input_from_z
    """
    if len(objs) == 0:
        return []
    # POST: actually have something to return.
    key = objs[0]
    offset =  key.Offset
    z = offset + key.Velocity * (key.Time - min(key.Time))
    to_ret = _wham_input_from_z(objs,z,n_ext_bins)
    return to_ret

def _debug_run(fwd_input,rev_input):
    rev_terms, fwd_terms, key_terms, delta_A, n_f, n_r, beta = \
        WeightedHistogram._term_helper(fwd_input=fwd_input, rev_input=rev_input)
    h_fwd, h_rev, boltz_fwd, boltz_rev = \
        WeightedHistogram._h_and_boltz_helper(fwd_terms, rev_terms, delta_A,
                                              beta, n_f, n_r)
    t_fwd = WeightedHistogram._energy_terms(key_terms, boltz_fwd, 0, h_fwd, 0)
    t_rev = WeightedHistogram._energy_terms(key_terms, 0, boltz_rev, 0, h_rev)
    t_both = WeightedHistogram._energy_terms(key_terms, boltz_fwd, boltz_rev,
                                             h_fwd, h_rev)
    # get the final numerator and denominator
    numers, denoms = [], []
    for t in [t_fwd,t_rev,t_both]:
        numer_tmp, denom_tmp = WeightedHistogram._fraction_terms(*t)
        numers.append(numer_tmp)
        denoms.append(denom_tmp)
    # make several plots to see how the calculation is proceeding
    style_common = dict(alpha=0.3)
    plot_args = [[t_fwd, dict(linestyle=':',label='fwd',**style_common)],
                 [t_rev, dict(linestyle='--',label='rev',**style_common)],
                 [t_both, dict(linestyle='-',label='both',**style_common)]]
    # make a plot of how we obtain h_i_j
    fwd_rev = [fwd_input, rev_input]
    V_i_j = WeightedHistogram._V_i_j_harmonic(key_input=fwd_input)
    for input_tmp, (_,style) in zip(fwd_rev,plot_args):
        mean_work = np.mean(input_tmp.works,axis=0)
        plt.plot(mean_work,**style)
    plt.plot(np.mean(V_i_j,axis=0))
    plt.show()
    # make a plot of the various
    for t,style in plot_args:
        boltzmann_V_i_j, h_i_j, eta_i = t
        sanit = lambda x: np.log(np.sum(x,axis=1))
        plt.subplot(1,3,1)
        plt.plot(sanit(boltzmann_V_i_j),**style)
        plt.ylabel("log($\Sigma V_{i,j}$)")
        plt.legend()
        plt.subplot(1,3,2)
        plt.plot(sanit(h_i_j),**style)
        plt.legend()
        plt.ylabel("log($\Sigma h_{i,j}$)")
        plt.subplot(1,3,3)
        plt.plot(np.log(eta_i),**style)
        plt.ylabel("log($\eta_i$)")
        plt.legend()
    plt.show()
    for n, d, (_,style) in zip(numers, denoms, plot_args):
        plt.subplot(1, 3, 1)
        plt.plot(np.log(n),**style)
        plt.ylabel("Fraction term")
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.plot(np.log(d),**style)
        plt.subplot(1, 3, 3)
        plt.plot(-np.log(n/d),**style)
        plt.xlabel("x (au)")
        plt.legend()
    plt.show()
    pass


