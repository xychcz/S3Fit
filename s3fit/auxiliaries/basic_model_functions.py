# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: s3fit@xychen.me

import numpy as np
np.set_printoptions(linewidth=10000)
from copy import deepcopy as copy
# import astropy.units as u
# import astropy.constants as const

from .auxiliary_functions import casefold, convolve_fix_width_fft

#####################################################################
########################## line functions ###########################

def single_line(obs_wave_w, lamb_c_rest, voff, fwhm, flux, v0_redshift=0, R_inst_rw=1e8, profile='Gaussian'):
    if fwhm <= 0: raise ValueError((f"Non-positive line fwhm: {fwhm}"))
    if flux < 0: raise ValueError((f"Negative line flux: {flux}"))

    lamb_c_obs = lamb_c_rest * (1 + v0_redshift)
    mu =   (1 + voff/299792.458) * lamb_c_obs
    fwhm_line = fwhm/299792.458  * lamb_c_obs

    if np.isscalar(R_inst_rw):
        local_R_inst = copy(R_inst_rw)
    else:
        local_R_inst = np.interp(lamb_c_obs, R_inst_rw[0], R_inst_rw[1])
    fwhm_inst = 1 / local_R_inst * lamb_c_obs

    if casefold(profile) in ['gaussian', 'gauss']:
        fwhm_tot = np.sqrt(fwhm_line**2 + fwhm_inst**2)
        sigma_tot = fwhm_tot / np.sqrt(np.log(256))
        model = np.exp(-0.5 * ((obs_wave_w-mu) / sigma_tot)**2) / (sigma_tot * np.sqrt(2*np.pi)) 
    elif casefold(profile) in ['lorentzian', 'lorentz']:
        gamma_line = fwhm_line / 2
        model = 1 / (1 + ((obs_wave_w-mu) / gamma_line)**2) / (gamma_line * np.pi)
        model = convolve_fix_width_fft(obs_wave_w, model, dw_fwhm=fwhm_inst, reset_edge=False)
    elif casefold(profile) in ['exponential', 'exp', 'laplace']:
        b_line = fwhm_line / np.log(4)
        model = np.exp(-np.abs(obs_wave_w-mu) / b_line) / (b_line * 2)
        model = convolve_fix_width_fft(obs_wave_w, model, dw_fwhm=fwhm_inst, reset_edge=False)
    else:
        raise ValueError((f"Please specify one of the line profiles: Gaussian, Lorentzian, or Exponential."))

    return model * flux

#####################################################################
####################### continuum functions #########################

def simple_powerlaw(wavelength, wave_norm=None, flux_norm=1.0, alpha_lambda=None):
    pl = flux_norm * (wavelength/wave_norm)**alpha_lambda
    return pl

def bending_powerlaw(wavelength, wave_turn=None, flux_trun=1.0, alpha_lambda1=None, alpha_lambda2=None, curvature=None, bending=False):
    # alpha_lambda1, alpha_lambda2: index with wavelength <= wave_turn and wavelength > wave_turn
    # curvature <= 0: broken two-side powerlaw
    # curvature > 0: smoothed bending powerlaw. larger curvature --> smoother break (5: very smooth; 0.1: very sharp)

    if isinstance(wavelength, (int,float)): wavelength = np.array([wavelength])
    if curvature is None: curvature = 0
    if alpha_lambda2 is not None:
        if alpha_lambda1 > alpha_lambda2: curvature = 0 # smoothing does not work in this case

    pl = simple_powerlaw(wavelength, wave_turn, flux_trun, alpha_lambda1)

    if bending:
        if curvature <= 0:
            # sharp, continuous broken power law
            mask_w = wavelength > wave_turn
            pl[mask_w] = simple_powerlaw(wavelength[mask_w], wave_turn, flux_trun, alpha_lambda2)
        else:
            pl_2 = simple_powerlaw(wavelength, wave_turn, 1, (alpha_lambda2-alpha_lambda1)/curvature)
            pl *= ((1+pl_2)/2.0)**curvature

    return pl

def powerlaw_func(wavelength, wave_norm=None, alpha_lambda1=None, alpha_lambda2=None, curvature=None, bending=False):
    # normalized to given flux density (e.g.,the same unit of obs) at rest wave_norm before extinct

    if isinstance(wavelength, (int,float)): wavelength = np.array([wavelength])
    pl = bending_powerlaw(wavelength, wave_norm, 1, alpha_lambda1, alpha_lambda2, curvature, bending)

    # set cutting index in longer and shorter wavelength ranges
    # https://sites.google.com/site/skirtorus/sed-library, Primary source: accretion disk
    alpha_long = -3-1; wave_long = 5e4
    alpha_short1 = 0-1; wave_short1 = 0.1e4
    alpha_short2 = 1.2-1; wave_short2 = 0.01e4

    mask_w = wavelength > wave_long
    if mask_w.sum() > 0: 
        pl_long = bending_powerlaw(wave_long, wave_norm, 1, alpha_lambda1, alpha_lambda2, curvature, bending)
        pl[mask_w] = simple_powerlaw(wavelength[mask_w], wave_long, pl_long, alpha_long)

    mask_w = wavelength < wave_short1
    if mask_w.sum() > 0: 
        pl_short1 = bending_powerlaw(wave_short1, wave_norm, 1, alpha_lambda1, alpha_lambda2, curvature, bending)
        pl[mask_w] = simple_powerlaw(wavelength[mask_w], wave_short1, pl_short1, alpha_short1)

    mask_w = wavelength < wave_short2
    if mask_w.sum() > 0: 
        pl_short1 = bending_powerlaw(wave_short1, wave_norm, 1, alpha_lambda1, alpha_lambda2, curvature, bending)
        pl_short2 = simple_powerlaw(wave_short2, wave_short1, pl_short1, alpha_short1)
        pl[mask_w] = simple_powerlaw(wavelength[mask_w], wave_short2, pl_short2, alpha_short2)

    return pl

def blackbody_func(wavelength, log_tem=None, if_norm=True, wave_norm=None):
    # parameters: temperature (K)

    def get_bb(wavelength):
        # Planck function for the given temperature
        C1 = 1.1910429723971884e27 # 2 * const.h.value * const.c.value**2 * 1e40 * 1e3
        C2 = 1.4387768775039336e8  # const.h.value * const.c.value / const.k_B.value * 1e10
        tmp = C2 / (wavelength * 10.0**log_tem)
        tmp = np.minimum(tmp, 700) # avoid overflow warning in np.exp()
        return C1 / wavelength**5 / (np.exp(tmp) - 1) # in erg/s/cm2/AA/sr
    
    ret_bb_w = get_bb(wavelength) 
    if if_norm: ret_bb_w /= get_bb(wave_norm)

    return ret_bb_w

# def bac_func(wavelength, log_e_tem=None, log_tau_be=None):
#     # parameters: electron temperature (K), optical depth at balmer edge (3646)
#     # normalize at rest 3000 AA (default)
#     wave_norm = 3000
#     balmer_edge = 3646.0

#     def get_bac(wavelength):
#         planck_flux_w = blackbody_func(wavelength, log_tem=log_e_tem, if_norm=False)
#         # calculate the optical depth at each wavelength
#         # τ_λ = τ_BE * (λ_BE / λ)^3  (as in Grandi 1982)
#         # the exponent can vary depending on the specific model
#         optical_depth = 10.0**log_tau_be * (balmer_edge / wavelength)**3
#         # return the Balmer continuum flux
#         return planck_flux_w * (1 - np.exp(-optical_depth))

#     bac_flux_w = get_bac(wavelength) / get_bac(wave_norm)
#     bac_flux_w[wavelength >= balmer_edge] = 0

#     return bac_flux_w

def recombination_func(wavelength, log_e_tem=None, log_tau_be=None, H_series=None, wave_norm=3000):
    # Hydrogen Radiative Recombination Continuum (free-bound)
    # parameters: electron temperature (K), optical depth at balmer edge (3646)
    # temperature range: ~3000--30000 K. lower: neutral H dominated; higher: free-free dominated

    def get_rec(wavelength):
        if isinstance(wavelength, (int,float)): wavelength = np.array([wavelength])
        rec_w = np.zeros_like(wavelength, dtype=float)
        bb_w = blackbody_func(wavelength, log_tem=log_e_tem, if_norm=False)
        for lv_n in H_series:
            wave_edge = lv_n**2 / 1.0973731568160e-3 # n**2 / const.Ryd.value * 1e10, in AA
            if min(wavelength) > wave_edge: continue
            # assume the bound-free cross section at threshold scales prop to n**(-5)
            tau_edge = 10.0**log_tau_be * (2.0/lv_n)**5
            # calculate the optical depth at each wavelength, τ_λ = τ_BE * (λ_BE / λ)^3 (Grandi 1982)
            tau_w = tau_edge * (wave_edge / wavelength)**3
            tmp_rec_w = bb_w * (1 - np.exp(-tau_w))
            tmp_rec_w[wavelength > wave_edge] = 0.0
            rec_w += tmp_rec_w
        return rec_w

    # normalize at rest 3000 AA (default)
    ret_rec_w = get_rec(wavelength) / get_rec(wave_norm)

    return ret_rec_w

#####################################################################
#####################################################################

