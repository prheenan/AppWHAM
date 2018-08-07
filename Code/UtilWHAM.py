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
    to (roughly) f[i]. XXX should make this
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
    plt.close()
    plot_x = lambda _x: _x * 1e9
    for i, list_v in enumerate([fwd_input, rev_input]):
        plt.subplot(2, 1, (i + 1))
        plt.plot(plot_x(list_v.z), plot_x(list_v.z),
                 label="q=z")
        for j, ext in enumerate(list_v.extensions):
            plt.plot(plot_x(list_v.z), plot_x(ext), ',')
            if i > 0:
                plt.xlabel("z (nm)")
            if i == 0:
                plt.title("fwd")
            else:
                plt.title('rev')
            plt.ylabel("q (nm)")
        plt.legend()
    plt.show()
    # make several plots to see how the calculation is proceeding
    style_common = dict(alpha=0.3)
    plot_args = [[t_fwd, dict(linestyle=':',label='fwd',color='r',
                              **style_common)],
                 [t_rev, dict(linestyle='--',label='rev',color='b',
                              **style_common)],
                 [t_both, dict(linestyle='-',label='both',color='k',
                               **style_common)]]
    # make a plot of how we obtain h_i_j
    fwd_rev = [fwd_input, rev_input]
    V_i_j = WeightedHistogram._V_i_j_harmonic(key_input=fwd_input)
    mean_V_ij = np.mean(V_i_j,axis=0)
    x = np.arange(0,mean_V_ij.size,1)
    stdevs = []
    plt.subplot(2,1,1)
    for input_tmp, (_,style) in zip(fwd_rev,plot_args):
        mean_work = np.mean(input_tmp.works,axis=0)
        std_work = np.std(input_tmp.works,axis=0)
        plt.plot(x,mean_work,**style)
        plt.fill_between(x,mean_work-std_work,mean_work+std_work,
                         color=style['color'],alpha=0.3)
        stdevs.append(std_work)
    plt.plot(x,mean_V_ij)
    plt.legend()
    plt.ylabel("Work (J), $\mu \pm \sigma$")
    plt.subplot(2,1,2)
    for s, (_,style) in zip(stdevs,plot_args):
        plt.plot(s/4.1e-21,**style)
    plt.legend()
    plt.ylabel("Stdev of Work (kT)")
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


