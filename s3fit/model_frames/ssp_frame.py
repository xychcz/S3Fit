# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: s3fit@xychen.me

import numpy as np
np.set_printoptions(linewidth=10000)
from copy import deepcopy as copy
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import WMAP9 as cosmo
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from ..auxiliary_func import print_log, convolve_fix_width_fft, convolve_var_width_fft
from ..extinct_law import ExtLaw

class SSPFrame(object):
    def __init__(self, cframe=None, v0_redshift=None, R_inst_rw=None, 
                 filename=None, w_min=None, w_max=None, w_norm=5500, dw_norm=25, 
                 R_mod_ratio=None, ds_fwhm_wave=None,
                 verbose=True, log_message=[]):

        self.cframe = cframe
        self.v0_redshift = v0_redshift
        self.R_inst_rw = R_inst_rw
        self.filename = filename
        self.w_min = w_min
        self.w_max = w_max
        self.w_norm = w_norm
        self.dw_norm = dw_norm
        self.R_mod_ratio = R_mod_ratio # resolution ratio of model / instrument
        self.ds_fwhm_wave = ds_fwhm_wave # convolving width for downsampling
        self.verbose = verbose
        self.log_message = log_message

        # load popstar library
        self.read_ssp_library()

        # read SFH setup from input config file
        self.sfh_names = np.array([d['sfh_name'] for d in self.cframe.info_c])
        self.num_comps = self.cframe.num_comps
        if self.num_comps == 1:
            if self.sfh_names[0] == 'nonparametric':
                self.num_coeffs = self.num_mets * self.num_ages
            else:
                self.num_coeffs = self.num_mets
        else:
            if np.sum(self.sfh_names == 'nonparametric') == 0:
                self.num_coeffs = self.num_mets * self.num_comps
            else:
                raise ValueError((f"Nonparametric SFH can only be used with a single component."))

        # currently do not consider negative spectra 
        self.mask_absorption_e = np.zeros((self.num_coeffs), dtype='bool')

        for i_comp in range(self.num_comps):
            if 10.0**self.cframe.max_cp[i_comp][3] > cosmo.age(self.v0_redshift).value:
                self.cframe.max_cp[i_comp][3] = np.log10(cosmo.age(self.v0_redshift).value)
                print_log(f"[WARNING]: Upper bound of CSP_Age of the component '{self.cframe.comp_c[i_comp]}' "
                    +f" is reset to the universe age {cosmo.age(self.v0_redshift).value:.3f} Gyr at z = {self.v0_redshift}.", self.log_message)
            if 10.0**self.cframe.min_cp[i_comp][3] > cosmo.age(self.v0_redshift).value:
                self.cframe.min_cp[i_comp][3] = np.log10(self.age_e[self.mask_ssp_allowed()].min()*1.0001) # take a factor of 1.0001 to avoid (csp_age-ssp_age) < 0
                print_log(f"[WARNING]: Lower bound of CSP_Age of the component '{self.cframe.comp_c[i_comp]}' "
                    +f" exceeds the universe age {cosmo.age(self.v0_redshift).value:.3f} Gyr at z = {self.v0_redshift}, "
                    +f" is reset to the available minimum SSP age {self.age_e[self.mask_ssp_allowed()].min():.3f} Gyr.", self.log_message)
            if 10.0**self.cframe.min_cp[i_comp][3] < self.age_e[self.mask_ssp_allowed()].min():
                self.cframe.min_cp[i_comp][3] = np.log10(self.age_e[self.mask_ssp_allowed()].min()*1.0001)
                print_log(f"[WARNING]: Lower bound of CSP_Age of the component '{self.cframe.comp_c[i_comp]}' "
                    +f" is reset to the available minimum SSP age {self.age_e[self.mask_ssp_allowed()].min():.3f} Gyr.", self.log_message)                
        
        if self.verbose:
            print_log(f'SSP models normalization wavelength: {w_norm} +- {dw_norm}', self.log_message)
            print_log(f'SSP models number: {self.mask_ssp_allowed().sum()} used in a total of {self.num_models}', self.log_message)
            print_log(f'SSP models age range (Gyr): from {self.age_e[self.mask_ssp_allowed()].min():.3f} to {self.age_e[self.mask_ssp_allowed()].max():.3f}', 
                      self.log_message)
            print_log(f'SSP models metallicity (Z/H): {np.unique(self.met_e[self.mask_ssp_allowed()])}', self.log_message) 
            print_log(f'SFH functions: {self.sfh_names} for {self.cframe.comp_c} components, respectively.', self.log_message)

    def read_ssp_library(self):
        ##############################################################
        ###### Modify this section to use a different SSP model ######
        ssp_lib = fits.open(self.filename)
        # template resolution step of 0.1 AA, from https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.4781M/abstract
        self.init_fwhm_wave = 0.1 

        self.header = ssp_lib[0].header
        # load models
        self.init_flux_ew = ssp_lib[0].data
        # wave_axis = 1
        # crval = self.header[f'CRVAL{wave_axis}']
        # cdelt = self.header[f'CDELT{wave_axis}']
        # naxis = self.header[f'NAXIS{wave_axis}']
        # crpix = self.header[f'CRPIX{wave_axis}']
        # if not cdelt: cdelt = 1
        # self.init_wave_w = crval + cdelt*(np.arange(naxis) + 1 - crpix)
        # use wavelength in data instead of one from header
        self.init_wave_w = ssp_lib[1].data
        # load remaining mass fraction
        self.remainmassfrac_e = ssp_lib[2].data
        # leave remainmassfrac_e = 1 if not provided. 

        self.num_models = self.header['NAXIS2']
        self.age_e = np.zeros(self.num_models, dtype='float')
        self.met_e = np.zeros(self.num_models, dtype='float')
        for i in range(self.num_models):
            met, age = self.header[f'NAME{i}'].split('.dat')[0].split('_')[1:3]
            self.age_e[i] = 10**float(age.replace('logt',''))/1e9
            self.met_e[i] = float(met.replace('Z',''))
        ##############################################################
        ##############################################################

        self.num_ages = np.unique(self.age_e).shape[0]
        self.num_mets = np.unique(self.met_e).shape[0]
        # obtain the duration (i.e., bin width) of each ssp if considering a continous SFH
        duration_a = np.gradient(np.unique(self.age_e))
        self.duration_e = np.tile(duration_a, (self.num_mets,1)).flatten()

        # select model spectra in given wavelength range
        orig_wave_w = self.init_wave_w # to save memory
        orig_flux_ew = self.init_flux_ew # to save memory
        mask_select_w = (orig_wave_w >= self.w_min) & (orig_wave_w <= self.w_max)
        orig_wave_w = orig_wave_w[mask_select_w]
        orig_flux_ew = orig_flux_ew[:, mask_select_w]

        # calculate the mean flux at both 5500 AA and the user given wavelength
        mask_5500_w = np.abs(orig_wave_w - 5500) < 25
        flux_5500_e = np.mean(orig_flux_ew[:, mask_5500_w], axis=1)
        mask_norm_w = np.abs(orig_wave_w - self.w_norm) < self.dw_norm
        flux_norm_e = np.mean(orig_flux_ew[:, mask_norm_w], axis=1)
        self.flux_norm_ratio_e = flux_norm_e / flux_5500_e

        # normalize models at 5500+/-25 AA
        orig_flux_ew /= flux_5500_e[:, None]
        # The original spectra of SSP models are normalized by 1 Msun in unit Lsun/AA.
        # The spectra used here is re-normalized to 1 Lsun/AA at rest 5500AA,
        # i.e., the norm-factor is norm=L5500 before re-normalization.  
        # The re-normalized spectra corresponds to mass of (1/norm) Msun, 
        # i.e., mass-to-lum(5500) ratio is (1/norm) Msun / (1 Lsun/AA) = (1/norm) Msun/(Lsun/AA),
        # i.e., mtol = 1/norm = 1/L5500
        self.mtol_e = 1 / flux_5500_e # i.e., (1 Msun) / (flux_5500_e Lsun/AA)
        # The corresponding SFR of normalized spectra is 
        # mtol_e Msun / (duration_e Gyr) = mtol_e/duration_e Msun/Gyr, duration_e in unit of Gyr (as age)
        # Name sfrtol_e = mtol_e/duration_e * 1e-9, and 
        # self.orig_flux_ew / sfrtol_e return models renormalized to unit SFR, i.e., 1 Mun/yr.
        self.sfrtol_e = self.mtol_e / (self.duration_e * 1e9)

        # determine the required model resolution and bin size (in AA) to downsample the model
        if self.R_mod_ratio is not None:
            ds_R_mod_w = np.interp(orig_wave_w*(1+self.v0_redshift), self.R_inst_rw[0], self.R_inst_rw[1] * self.R_mod_ratio) # R_inst_rw in observed frame
            self.ds_fwhm_wave_w = orig_wave_w / ds_R_mod_w # required resolving width in rest frame
        else:
            if self.ds_fwhm_wave is not None:
                self.ds_fwhm_wave_w = np.full(len(orig_wave_w), self.ds_fwhm_wave)
            else:
                self.ds_fwhm_wave_w = None
        if self.ds_fwhm_wave_w is not None:
            if (self.ds_fwhm_wave_w > self.init_fwhm_wave).all(): 
                preconvolving = True
            else:
                preconvolving = False
                self.ds_fwhm_wave_w = np.full(len(orig_wave_w), self.init_fwhm_wave)
            self.ds_bin_wave = self.ds_fwhm_wave_w.min() * 0.5 # required min bin wavelength following Nyquist–Shannon sampling
            if self.R_inst_rw is not None:
                self.ds_bin_wave = min(self.ds_bin_wave, np.median(np.diff(self.R_inst_rw[0]))/(1+self.v0_redshift) * 0.5) # also require model bin wavelength <= 0.5 of data bin width
            self.ds_bin_pix  = int(self.ds_bin_wave / np.median(np.diff(orig_wave_w))) # required min bin number of pixels
            self.ds_bin_wave = self.ds_bin_pix * np.median(np.diff(orig_wave_w)) # update value
            if self.ds_bin_pix > 1:
                if preconvolving:
                    if self.verbose: 
                        print_log(f'Downsample preconvolved SSP models with bin width of {self.ds_bin_wave:.3f} AA in a min resolution of {self.ds_fwhm_wave_w.min():.3f} AA', 
                                  self.log_message)
                    # before downsampling, smooth the model to avoid aliasing (like in ADC or digital signal reduction)
                    # here assume the internal dispersion in the original model (e.g., in stellar atmosphere) is indepent from the measured dispersion (i.e., stellar motion) in the fitting
                    orig_flux_ew = convolve_fix_width_fft(orig_wave_w, orig_flux_ew, self.ds_fwhm_wave_w.min())
                else:
                    if self.verbose: 
                        print_log(f'Downsample original SSP models with bin width of {self.ds_bin_wave:.3f} AA in a min resolution of {self.ds_fwhm_wave_w.min():.3f} AA', 
                                  self.log_message)  
                orig_wave_w = orig_wave_w[::self.ds_bin_pix]
                orig_flux_ew = orig_flux_ew[:,::self.ds_bin_pix]
                self.ds_fwhm_wave_w = self.ds_fwhm_wave_w[::self.ds_bin_pix]

        # save the smoothed models
        self.orig_wave_w = orig_wave_w
        self.orig_flux_ew = orig_flux_ew

        # extend to longer wavelength in NIR-MIR (e.g., > 3 micron)
        # please comment these lines if moving to another SSP library that initially covers the NIR-MIR range. 
        if  (orig_wave_w.max() < self.w_max) & (self.w_max > 2.28e4):
            mask_ref_w = (orig_wave_w > 2.1e4) & (orig_wave_w <= 2.28e4) # avoid edge for which fft convolution does not work well
            index = -4 # i.e., blackbody
            ratio_e = np.mean(orig_flux_ew[:,mask_ref_w] / orig_wave_w[None,mask_ref_w]**index, axis=1)

            ext_wave_logbin = 0.05
            ext_wave_num = int(np.round(np.log10(self.w_max/orig_wave_w[mask_ref_w][-1]) / ext_wave_logbin))
            ext_wave_w = np.logspace(np.log10(orig_wave_w[mask_ref_w][-1]+1), np.log10(self.w_max), ext_wave_num)
            ext_flux_ew = ext_wave_w[None,:]**index * ratio_e[:,None]

            mask_keep_w = orig_wave_w < ext_wave_w[0]
            self.orig_flux_ew = np.hstack((orig_flux_ew[:,mask_keep_w], ext_flux_ew))
            self.orig_wave_w = np.hstack((orig_wave_w[mask_keep_w], ext_wave_w))

    def sfh_factor(self, i_comp, sfh_pars):
        # For a given SFH, i.e., SFR(t) = SFR(csp_age-ssp_age_e), in unit of Msun/yr, 
        # the model of a given ssp (_e) is ssp_spec_ew = SFR(csp_age-ssp_age_e) * (self.orig_flux_ew/sfrtol_e), 
        # Name sfh_factor_e = SFR(csp_age-ssp_age_e) / sfrtol_e = SFR(csp_age-ssp_age_e) * ltosfr_e, 
        # here ltosfr_e can be considered as the lum(rest5500) per unit SFR;
        # full expression of ltosfr_e = (self.duration_e * 1e9) / self.mtol_e.
        # Following this way, sfh_factor_e is the lum(rest5500)_e to achieve a given SFR(csp_age-ssp_age_e), i.e., sfh_func_e. 
        # The returned models are ssp_spec_ew = self.orig_flux_ew * sfh_factor_e.
        # The corresponding lum(rest5500) is 1 * sfh_factor_e, in unit of Lsun/AA,
        # the corresponding mass is mtol_e * sfh_factor_e = SFR(csp_age-ssp_age_e) * (duration_e*1e9), in unit of Msun.
        csp_age = 10.0**sfh_pars[0]
        ssp_age_e = self.age_e
        evo_time_e = csp_age - ssp_age_e

        if self.sfh_names[i_comp] == 'exponential': 
            tau = 10.0**sfh_pars[1]
            sfh_func_e = np.exp(-(evo_time_e) / tau)
        if self.sfh_names[i_comp] == 'delayed': 
            tau = 10.0**sfh_pars[1]
            sfh_func_e = np.exp(-(evo_time_e) / tau) * evo_time_e
        if self.sfh_names[i_comp] == 'constant': 
            sfh_func_e = np.ones_like(evo_time_e)
        if self.sfh_names[i_comp] == 'user': 
            sfh_func_e = self.cframe.info_c[i_comp]['sfh_func'](evo_time_e, sfh_pars)
        ############################
        # Add new SFH function here. 
        ############################

        sfh_func_e[~self.mask_ssp_allowed(i_comp)] = 0 # do not use ssp out of allowed range
        sfh_func_e[evo_time_e < 0] = 0 # do not allow ssp older than csp_age 
        sfh_func_e /= sfh_func_e.max()
        sfh_factor_e = sfh_func_e / self.sfrtol_e
        return sfh_factor_e
        # The total csp model is csp_spec_w = ssp_spec_ew.sum(axis=0) = (self.orig_flux_ew * sfh_factor_e).sum(axis=0)
        # The corresponding lum(at 5500) is (1 * sfh_factor_e).sum(axis=0), in unit of Lsun/AA.
        # The corresponding mass is (mtol_e * sfh_factor_e * remain_e).sum(axis=0), in unit of Msun, 
        # remain_e is the remaining mass fraction. 

        # If the best-fit csp has csp_coeff, 
        # the corresponding lum(at 5500) of csp is csp_coeff * (1 * sfh_factor_e).sum(axis=0) Lsun/AA;
        # the corresponding lum(at 5500) of ssp_e is csp_coeff * sfh_factor_e Lsun/AA, 
        # which equals to ssp_coeff_e = (csp_coeff * sfh_factor_e) for direct usage of self.orig_flux_ew (i.e., nonparametic SFH).
        # Here the meanning of csp_coeff is the value of SFR in SFH peak epoch (due to the above normalization) in Mun/yr; 
        # the meanning of converted ssp_coeff_e is still lum(at 5500) of each unit-normalized ssp model template.  
        # The total mass can be calculated as (csp_coeff * sfh_factor_e * mtol_e * remain_e).sum(axis=0).
        # Note that in all above (_e).sum(axis=0) is indeed (_e[mask_e]).sum(axis=0), 
        # mask_e is used to mask the allowed ssp model elements (age and met ranges). 

    def models_unitnorm_obsframe(self, obs_wave_w, input_pars, if_pars_flat=True, mask_lite_e=None, conv_nbin=None):
        # The input model is spectra per unit Lsun/AA at rest 5500AA before dust reddening and redshift, 
        # corresponds to mass of 1/L5500 Msun (L5500 is the lum-value in unit of Lsun/AA from original models normalized per unit Msun).
        # In the fitting for the observed spectra in unit of in erg/s/AA/cm2, 
        # the output model can be considered to be re-normlized to 1 erg/s/AA/cm2 at rest 5500AA before dust reddening and redshift.
        # The corresponding lum is (1/3.826e33*Area) Lsun/AA, where Area is the lum-area in unit of cm2, 
        # and the corresponding mass is (1/3.826e33*Area) * (1/L5500) Msun, 
        # i.e., the real mtol is (1/3.826e33*Area) * (1/L5500) = (1/3.826e33*Area) * self.mtol
        # The values will be used to calculate the stellar mass from the best-fit results. 
        if if_pars_flat: 
            par_cp = self.cframe.flat_to_arr(input_pars)
        else:
            par_cp = copy(input_pars)
        if mask_lite_e is not None:
            mask_lite_ce = self.cframe.flat_to_arr(mask_lite_e)

        for i_comp in range(par_cp.shape[0]):
            if self.sfh_names[i_comp] == 'nonparametric':
                orig_flux_int_ew = copy(self.orig_flux_ew) # copy intrinsic models
            else:
                sfh_factor_e = self.sfh_factor(i_comp, par_cp[i_comp,3:])
                tmp_mask_e = sfh_factor_e > 0
                tmp_ew = np.zeros_like(self.orig_flux_ew)
                tmp_ew[tmp_mask_e,:] = self.orig_flux_ew[tmp_mask_e,:] * sfh_factor_e[tmp_mask_e,None] # scaled with sfh_factor_e
                # tmp_ew = self.orig_flux_ew * sfh_factor_e[:,None] # scaled with sfh_factor_e
                orig_flux_int_ew = tmp_ew.reshape(self.num_mets, self.num_ages, len(self.orig_wave_w)).sum(axis=1)
                # sum in ages to create csp 
            if mask_lite_e is not None:
                orig_flux_int_ew = orig_flux_int_ew[mask_lite_ce[i_comp,:],:] # limit element number for accelarate calculation

            # dust extinction
            orig_flux_d_ew = orig_flux_int_ew * 10.0**(-0.4 * par_cp[i_comp,2] * ExtLaw(self.orig_wave_w))
            # redshift models
            z_ratio = (1 + self.v0_redshift) * (1 + par_cp[i_comp,0]/299792.458) # (1+z) = (1+zv0) * (1+v/c)
            orig_wave_z_w = self.orig_wave_w * z_ratio
            orig_flux_dz_ew = orig_flux_d_ew / z_ratio
            # convolve with intrinsic and instrumental dispersion if self.R_inst_rw is not None
            if (self.R_inst_rw is not None) & (conv_nbin is not None):
                R_inst_w = np.interp(orig_wave_z_w, self.R_inst_rw[0], self.R_inst_rw[1])
                orig_flux_dzc_ew = convolve_var_width_fft(orig_wave_z_w, orig_flux_dz_ew, fwhm_vel_kin=par_cp[i_comp,1], fwhm_wave_ref=self.ds_fwhm_wave_w, R_inst_w=R_inst_w, num_bins=conv_nbin)
                # convolution in redshifted- or rest-wavelength does not change result
            else:
                orig_flux_dzc_ew = orig_flux_dz_ew 
                # just copy if convlution not required, e.g., for broad-band sed fitting
            # project to observed wavelength
            interp_func = interp1d(orig_wave_z_w, orig_flux_dzc_ew, axis=1, kind='linear', fill_value="extrapolate")
            obs_flux_scomp_ew = interp_func(obs_wave_w)
            if i_comp == 0: 
                obs_flux_mcomp_ew = obs_flux_scomp_ew
            else:
                obs_flux_mcomp_ew = np.vstack((obs_flux_mcomp_ew, obs_flux_scomp_ew))
        return obs_flux_mcomp_ew
    
    def mask_ssp_allowed(self, i_comp=0, csp=False):
        if not csp: # i.e. for all ssp, depends on i_comp
            age_min, age_max = self.cframe.info_c[i_comp]['age_min'], self.cframe.info_c[i_comp]['age_max']
            age_min = self.age_e.min() if age_min is None else 10.0**age_min
            age_max = cosmo.age(self.v0_redshift).value if age_max == 'universe' else 10.0**age_max
            mask_ssp_allowed_e = (self.age_e >= age_min) & (self.age_e <= age_max)
            met_sel = self.cframe.info_c[i_comp]['met_sel']
            if met_sel != 'all':
                if met_sel == 'solar':
                    mask_ssp_allowed_e &= self.met_e == 0.02
                else:
                    mask_ssp_allowed_e &= np.isin(self.met_e, met_sel)
        else: # loop for all comp
            mask_ssp_allowed_e = np.array([], dtype='bool')
            for i_comp in range(self.num_comps):
                tmp_mask_e = np.ones(self.num_mets, dtype='bool') 
                met_sel = self.cframe.info_c[i_comp]['met_sel']
                if met_sel != 'all':
                    if met_sel == 'solar':
                        tmp_mask_e &= np.unique(self.met_e) == 0.02
                    else:
                        tmp_mask_e &= np.isin(np.unique(self.met_e), met_sel)
                mask_ssp_allowed_e = np.hstack((mask_ssp_allowed_e, tmp_mask_e))
        return mask_ssp_allowed_e

    def mask_ssp_lite_with_num_mods(self, num_ages_lite=8, num_mets_lite=1, verbose=True):
        if self.sfh_names[0] == 'nonparametric':
            # only used in nonparametic, single component
            ages_full, num_ages_full = np.unique(self.age_e), len(np.unique(self.age_e))
            ages_allowed = np.unique(self.age_e[self.mask_ssp_allowed()])
            ages_lite = np.logspace(np.log10(ages_allowed.min()), np.log10(ages_allowed.max()), num=num_ages_lite)
            ages_lite *= 10.0**((np.random.rand(num_ages_lite)-0.5)*np.log10(ages_lite[1]/ages_lite[0]))
            # request log-even ages with random shift
            ind_ages_lite = [np.where(np.abs(ages_full-a)==np.min(np.abs(ages_full-a)))[0][0] for a in ages_lite]
            # np.round(np.linspace(0, num_ages_full-1, num_ages_lite)).astype(int)
            ind_mets_lite = [2,1,3,0][:num_mets_lite] # Z = 0.02 (solar), 0.008, 0.05, 0.004, select with this order
            ind_ssp_lite = np.array([ind_met*num_ages_full+np.arange(num_ages_full)[ind_age] 
                                     for ind_met in ind_mets_lite for ind_age in ind_ages_lite])
            mask_ssp_lite_e = np.zeros_like(self.age_e, dtype='bool')
            mask_ssp_lite_e[ind_ssp_lite] = True
            mask_ssp_lite_e &= self.mask_ssp_allowed()
            if verbose: print_log(f'Number of used SSP models: {mask_ssp_lite_e.sum()}', self.log_message) 
        else:
            mask_ssp_lite_e = self.mask_ssp_allowed(csp=True)
            if verbose: print_log(f'Number of used CSP models: {mask_ssp_lite_e.sum()}', self.log_message) 
        return mask_ssp_lite_e

    def mask_ssp_lite_with_coeffs(self, coeffs=None, mask=None, num_mods_min=32, verbose=True):
        if self.sfh_names[0] == 'nonparametric':
            # only used in nonparametic, single component
            coeffs_full = np.zeros(self.num_models)
            coeffs_full[mask if mask is not None else self.mask_ssp_allowed()] = coeffs
            coeffs_sort = np.sort(coeffs_full)
            # coeffs_min = coeffs_sort[np.cumsum(coeffs_sort)/np.sum(coeffs_sort) < 0.01].max() 
            # # i.e., keep coeffs with sum > 99%
            # mask_ssp_lite = coeffs_full >= np.minimum(coeffs_min, coeffs_sort[-num_mods_min]) 
            # # keep minimum num of models
            # mask_ssp_lite &= self.mask_ssp_allowed()
            # print('Number of used SSP models:', mask_ssp_lite.sum()) #, np.unique(self.age_e[mask_ssp_lite]))
            # print('Ages with coeffs.sum > 99%:', np.unique(self.age_e[coeffs_full >= coeffs_min]))
            mask_ssp_lite_e = coeffs_full >= coeffs_sort[-num_mods_min]
            mask_ssp_lite_e &= self.mask_ssp_allowed()
            if verbose: 
                print_log(f'Number of used SSP models: {mask_ssp_lite_e.sum()}', self.log_message) 
                print_log(f'Coeffs.sum of used SSP models: {1-np.cumsum(coeffs_sort)[-num_mods_min]/np.sum(coeffs_sort)}', self.log_message) 
                print_log(f'Ages of dominant SSP models: {np.unique(self.age_e[coeffs_full >= coeffs_sort[-5]])}', self.log_message) 
        else:
            mask_ssp_lite_e = self.mask_ssp_allowed(csp=True)
            if verbose: print_log(f'Number of used CSP models: {mask_ssp_lite_e.sum()}', self.log_message)             
        return mask_ssp_lite_e

    ##########################################################################
    ########################## Output functions ##############################

    def extract_results(self, ff=None, step=None, print_results=True, show_average=False, return_results=False):
        if (step is None) | (step == 'best') | (step == 'final'):
            step = 'joint_fit_3' if ff.have_phot else 'joint_fit_2'
        if (step == 'spec+SED'):  step = 'joint_fit_3'
        if (step == 'spec') | (step == 'pure-spec'): step = 'joint_fit_2'
        
        best_chi_sq_l = copy(ff.output_s[step]['chi_sq_l'])
        best_par_lp   = copy(ff.output_s[step]['par_lp'])
        best_coeff_le = copy(ff.output_s[step]['coeff_le'])

        mod = 'ssp'
        fp0, fp1, fe0, fe1 = ff.search_model_index(mod, ff.full_model_type)
        spec_wave_w = ff.spec['wave_w']
        spec_flux_scale = ff.spec_flux_scale
        num_loops = ff.num_loops
        comp_c = self.cframe.comp_c
        num_comps = self.cframe.num_comps
        num_pars_per_comp = self.cframe.num_pars_per_comp
        num_coeffs_per_comp = int(self.num_coeffs / num_comps)

        # list the properties to be output
        val_names  = ['voff', 'fwhm', 'AV'] # basic fitting parameters
        val_names += ['sfh_par'+str(i) for i in range(num_pars_per_comp-3)] # SFH related fitting parameters
        val_names += ['redshift', 
                      'flux_5500', 'loglambLum_5500', 
                      'flux_wavenorm', 'loglambLum_wavenorm', 
                      'logMass_formed', 'logMass_remaining', 'logMtoL',
                      'logAge_Lweight', 'logAge_Mweight', 'logZ_Lweight', 'logZ_Mweight']

        # format of results
        # output_c['comp']['par_lp'][i_l,i_p]: parameters
        # output_c['comp']['coeff_le'][i_l,i_e]: coefficients
        # output_c['comp']['values']['name_l'][i_l]: calculated values
        output_c = {}
        for i_comp in range(num_comps): 
            output_c[comp_c[i_comp]] = {} # init results for each comp
            output_c[comp_c[i_comp]]['par_lp']   = best_par_lp[:, fp0:fp1].reshape(num_loops, num_comps, num_pars_per_comp)[:, i_comp, :]
            output_c[comp_c[i_comp]]['coeff_le'] = best_coeff_le[:, fe0:fe1].reshape(num_loops, num_comps, num_coeffs_per_comp)[:, i_comp, :]
            output_c[comp_c[i_comp]]['values'] = {}
            for val_name in val_names:
                output_c[comp_c[i_comp]]['values'][val_name] = np.zeros(num_loops, dtype='float')
        output_c['sum'] = {}
        output_c['sum']['values'] = {} # only init values for sum of all comp
        for val_name in val_names:
            output_c['sum']['values'][val_name] = np.zeros(num_loops, dtype='float')

        for i_comp in range(num_comps): 
            for i_par in range(num_pars_per_comp):
                output_c[comp_c[i_comp]]['values'][val_names[i_par]] = output_c[comp_c[i_comp]]['par_lp'][:, i_par]
            for i_loop in range(num_loops):
                par_p   = output_c[comp_c[i_comp]]['par_lp'][i_loop]
                coeff_e = output_c[comp_c[i_comp]]['coeff_le'][i_loop]
                rev_redshift = (1+par_p[0]/299792.458)*(1+self.v0_redshift)-1
                output_c[comp_c[i_comp]]['values']['redshift'][i_loop] = copy(rev_redshift)

                tmp_spec_w = ff.output_mc[mod][comp_c[i_comp]]['spec_lw'][i_loop, :]
                mask_norm_w = np.abs(spec_wave_w/(1+rev_redshift) - 5500) < 25 # for observed flux at rest 5500 AA
                if mask_norm_w.sum() > 0:
                    output_c[comp_c[i_comp]]['values']['flux_5500'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                    output_c['sum']['values']['flux_5500'][i_loop] += tmp_spec_w[mask_norm_w].mean()

                mask_norm_w = np.abs(spec_wave_w/(1+rev_redshift) - self.w_norm) < self.dw_norm # for observed flux at user given wavenorm
                output_c[comp_c[i_comp]]['values']['flux_wavenorm'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                output_c['sum']['values']['flux_wavenorm'][i_loop] += tmp_spec_w[mask_norm_w].mean()

                dist_lum = cosmo.luminosity_distance(rev_redshift).to('cm').value
                unitconv = 4*np.pi*dist_lum**2 / const.L_sun.to('erg/s').value * spec_flux_scale # convert intrinsic flux5500(rest) to L5500 
                if self.sfh_names[i_comp] != 'nonparametric':
                    sfh_factor_e = self.sfh_factor(i_comp, par_p[3:]) 
                    # sfh_factor_e means lum(5500) of each ssp model element per unit SFR (of the peak SFH epoch in this case)
                    # if use csp with a sfh, coeff_e*unitconv means the value of SFH of this csp element in Mun/yr at the peak SFH epoch
                    # coeff_e*unitconv*sfh_factor_e gives the correspoinding the best-fit lum(5500) of each ssp model element
                    coeff_e = np.tile(coeff_e, (self.num_ages,1)).T.flatten() * sfh_factor_e 
                Lum_e = coeff_e * unitconv # intrinsic L5500, in Lsun/AA
                lambLum_e = Lum_e * 5500 # intrinsic λL5500, in Lsun
                Mass_formed_e = Lum_e * self.mtol_e
                Mass_remaining_e = Mass_formed_e * self.remainmassfrac_e

                output_c[comp_c[i_comp]]['values']['loglambLum_5500'][i_loop]   = np.log10(lambLum_e.sum())
                output_c[comp_c[i_comp]]['values']['loglambLum_wavenorm'][i_loop]   = np.log10((lambLum_e * self.flux_norm_ratio_e).sum())
                output_c[comp_c[i_comp]]['values']['logMass_formed'][i_loop]    = np.log10(Mass_formed_e.sum())
                output_c[comp_c[i_comp]]['values']['logMass_remaining'][i_loop] = np.log10(Mass_remaining_e.sum())
                output_c[comp_c[i_comp]]['values']['logMtoL'][i_loop]   = np.log10(Mass_remaining_e.sum() / lambLum_e.sum())
                output_c[comp_c[i_comp]]['values']['logAge_Lweight'][i_loop] = (lambLum_e * np.log10(self.age_e)).sum() / lambLum_e.sum()
                output_c[comp_c[i_comp]]['values']['logAge_Mweight'][i_loop] = (Mass_remaining_e * np.log10(self.age_e)).sum() / Mass_remaining_e.sum()
                output_c[comp_c[i_comp]]['values']['logZ_Lweight'][i_loop] = (lambLum_e * np.log10(self.met_e)).sum() / lambLum_e.sum()
                output_c[comp_c[i_comp]]['values']['logZ_Mweight'][i_loop] = (Mass_remaining_e * np.log10(self.met_e)).sum() / Mass_remaining_e.sum()

                output_c['sum']['values']['loglambLum_5500'][i_loop]   += lambLum_e.sum() # keep in linear for sum
                output_c['sum']['values']['loglambLum_wavenorm'][i_loop]   += (lambLum_e * self.flux_norm_ratio_e).sum() # keep in linear for sum
                output_c['sum']['values']['logMass_formed'][i_loop]    += Mass_formed_e.sum() # keep in linear for sum
                output_c['sum']['values']['logMass_remaining'][i_loop] += Mass_remaining_e.sum() # keep in linear for sum
                output_c['sum']['values']['logAge_Lweight'][i_loop] += (lambLum_e * np.log10(self.age_e)).sum()
                output_c['sum']['values']['logAge_Mweight'][i_loop] += (Mass_remaining_e * np.log10(self.age_e)).sum()
                output_c['sum']['values']['logZ_Lweight'][i_loop] += (lambLum_e * np.log10(self.met_e)).sum()
                output_c['sum']['values']['logZ_Mweight'][i_loop] += (Mass_remaining_e * np.log10(self.met_e)).sum()

        output_c['sum']['values']['logMtoL'] = np.log10(output_c['sum']['values']['logMass_remaining'] / output_c['sum']['values']['loglambLum_wavenorm'])
        output_c['sum']['values']['logAge_Lweight'] = output_c['sum']['values']['logAge_Lweight'] / output_c['sum']['values']['loglambLum_wavenorm']
        output_c['sum']['values']['logAge_Mweight'] = output_c['sum']['values']['logAge_Mweight'] / output_c['sum']['values']['logMass_remaining']
        output_c['sum']['values']['logZ_Lweight'] = output_c['sum']['values']['logZ_Lweight'] / output_c['sum']['values']['loglambLum_wavenorm']
        output_c['sum']['values']['logZ_Mweight'] = output_c['sum']['values']['logZ_Mweight'] / output_c['sum']['values']['logMass_remaining']
        output_c['sum']['values']['loglambLum_5500']       = np.log10(output_c['sum']['values']['loglambLum_5500'])
        output_c['sum']['values']['loglambLum_wavenorm']   = np.log10(output_c['sum']['values']['loglambLum_wavenorm'])
        output_c['sum']['values']['logMass_formed']    = np.log10(output_c['sum']['values']['logMass_formed'])
        output_c['sum']['values']['logMass_remaining'] = np.log10(output_c['sum']['values']['logMass_remaining'])

        if self.sfh_names[0] == 'nonparametric':
            coeff_le = output_c[comp_c[0]]['coeff_le']
            output_c[comp_c[0]]['coeff_norm_le'] = coeff_le / coeff_le.sum(axis=1)[:,None]

        self.output_c = output_c # save to model frame
        self.num_loops = num_loops # for reconstruct_sfh and print_results
        self.spec_flux_scale = spec_flux_scale # for reconstruct_sfh and print_results

        if print_results: self.print_results(log=ff.log_message, show_average=show_average)
        if return_results: return output_c

    def print_results(self, log=[], show_average=False):
        mask_l = np.ones(self.num_loops, dtype='bool')
        if not show_average: mask_l[1:] = False

        if self.cframe.num_comps > 1:
            num_comps = len([*self.output_c])
        else:
            num_comps = 1

        for i_comp in range(num_comps):
            tmp_values_vl = self.output_c[[*self.output_c][i_comp]]['values']
            if i_comp < self.cframe.num_comps:
                if self.sfh_names[i_comp] == 'nonparametric':
                    print_log('', log)
                    print_log('Best-fit single stellar populations (SSP) of nonparametric SFH', log)
                    cols = 'ID,Age (Gyr),Metallicity,Coeff.mean,Coeff.rms,log(M/L5500)'
                    fmt_cols = '| {0:^4} | {1:^10} | {2:^6} | {3:^6} | {4:^9} | {5:^8} |'
                    fmt_numbers = '| {:=04d} |   {:=6.4f}   |    {:=6.4f}   |   {:=6.4f}   |   {:=6.4f}  |    {:=6.4f}    |'
                    cols_split = cols.split(',')
                    tbl_title = fmt_cols.format(*cols_split)
                    tbl_border = len(tbl_title)*'-'
                    print_log(tbl_border, log)
                    print_log(tbl_title, log)
                    print_log(tbl_border, log)
                    for i_e in range(self.num_models):
                        coeff_norm_mn_e  = self.output_c[[*self.output_c][0]]['coeff_norm_le'][mask_l].mean(axis=0)
                        coeff_norm_std_e = self.output_c[[*self.output_c][0]]['coeff_norm_le'].std(axis=0)
                        if coeff_norm_mn_e[i_e] < 0.05: continue
                        tbl_row = []
                        tbl_row.append(i_e)
                        tbl_row.append(self.age_e[i_e])
                        tbl_row.append(self.met_e[i_e])
                        tbl_row.append(coeff_norm_mn_e[i_e]) 
                        tbl_row.append(coeff_norm_std_e[i_e])
                        tbl_row.append(np.log10(self.mtol_e[i_e]))
                        print_log(fmt_numbers.format(*tbl_row), log)
                    print_log(tbl_border, log)
                    print_log(f'[Note] Coeff is the normalized fraction of the intrinsic flux at rest 5500 AA.', log)
                    print_log(f'[Note] only SSPs with flux fraction over 5% are listed.', log)

            print_log('', log)
            msg = ''
            if i_comp < self.cframe.num_comps:
                print_log(f'Best-fit stellar properties of the <{self.cframe.comp_c[i_comp]}> component with {self.sfh_names[i_comp]} SFH.', log)
                msg += f'| Redshift                                  = {tmp_values_vl["redshift"][mask_l].mean():10.4f}'
                msg += f' +/- {tmp_values_vl["redshift"].std():<8.4f}|\n'
                msg += f'| Velocity dispersion (σ,km/s)              = {tmp_values_vl["fwhm"][mask_l].mean()/2.355:10.4f}'
                msg += f' +/- {tmp_values_vl["fwhm"].std()/2.355:<8.4f}|\n'
                msg += f'| Extinction (AV)                           = {tmp_values_vl["AV"][mask_l].mean():10.4f}'
                msg += f' +/- {tmp_values_vl["AV"].std():<8.4f}|\n'
                if np.isin(self.sfh_names[i_comp], ['exponential', 'delayed', 'constant', 'user']):
                    msg += f'| Max age of composite star.pop. (log Gyr)  = {tmp_values_vl["sfh_par0"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["sfh_par0"].std():<8.4f}|\n'
                if np.isin(self.sfh_names[i_comp], ['exponential', 'delayed']):
                    msg += f'| Declining timescale of SFH (log Gyr)      = {tmp_values_vl["sfh_par1"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["sfh_par1"].std():<8.4f}|\n'
                if np.isin(self.sfh_names[i_comp], ['user']):
                    for par_name in [*tmp_values_vl]:
                        if par_name[:3] != 'sfh': continue
                        if par_name == 'sfh_par0': continue
                        msg += f'| SFH parameter {par_name[-1]}            = {tmp_values_vl[par_name][mask_l].mean():10.4f}'
                        msg += f' +/- {tmp_values_vl[par_name].std():<8.4f}|\n'
            else:
                print_log(f'Best-fit stellar properties of the sum of all components.', log)
            msg += f'| F5500 (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/AA) = {tmp_values_vl["flux_5500"][mask_l].mean():10.4f}'
            msg += f' +/- {tmp_values_vl["flux_5500"].std():<8.4f}|\n'
            if self.w_norm != 5500:
                msg += f'| F{self.w_norm} (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/AA) = {tmp_values_vl["flux_wavenorm"][mask_l].mean():10.4f}'
                msg += f' +/- {tmp_values_vl["flux_wavenorm"].std():<8.4f}|\n'
            msg += f'| λL5500 (rest,intrinsic) (log Lsun)        = {tmp_values_vl["loglambLum_5500"][mask_l].mean():10.4f}'
            msg += f' +/- {tmp_values_vl["loglambLum_5500"].std():<8.4f}|\n'
            if self.w_norm != 5500:
                msg += f'| λL{self.w_norm} (rest,intrinsic) (log Lsun)        = {tmp_values_vl["loglambLum_wavenorm"][mask_l].mean():10.4f}'
                msg += f' +/- {tmp_values_vl["loglambLum_wavenorm"].std():<8.4f}|\n'
            msg += f'| Mass (all formed) (log Msun)              = {tmp_values_vl["logMass_formed"][mask_l].mean():10.4f}'
            msg += f' +/- {tmp_values_vl["logMass_formed"].std():<8.4f}|\n'
            msg += f'| Mass (remaining) (log Msun)               = {tmp_values_vl["logMass_remaining"][mask_l].mean():10.4f}'
            msg += f' +/- {tmp_values_vl["logMass_remaining"].std():<8.4f}|\n'
            msg += f'| Mass/λL5500 (log Msun/Lsun)               = {tmp_values_vl["logMtoL"][mask_l].mean():10.4f}'
            msg += f' +/- {tmp_values_vl["logMtoL"].std():<8.4f}|\n'
            msg += f'| λL5500-weight age (log Gyr)               = {tmp_values_vl["logAge_Lweight"][mask_l].mean():10.4f}'
            msg += f' +/- {tmp_values_vl["logAge_Lweight"].std():<8.4f}|\n'
            msg += f'| Mass-weight age (log Gyr)                 = {tmp_values_vl["logAge_Mweight"][mask_l].mean():10.4f}'
            msg += f' +/- {tmp_values_vl["logAge_Mweight"].std():<8.4f}|\n'
            msg += f'| λL5500-weight metallicity (log Z)         = {tmp_values_vl["logZ_Lweight"][mask_l].mean():10.4f}'
            msg += f' +/- {tmp_values_vl["logZ_Lweight"].std():<8.4f}|\n'
            msg += f'| Mass-weight metallicity (log Z)           = {tmp_values_vl["logZ_Mweight"][mask_l].mean():10.4f}'
            msg += f' +/- {tmp_values_vl["logZ_Mweight"].std():<8.4f}|'
            bar = '=' * len(msg.split('\n')[-1])
            print_log(bar, log)
            print_log(msg, log)
            print_log(bar, log)

    def reconstruct_sfh(self, output_c=None, num_bins=None, plot=True, return_sfh=False, show_average=True):
        mask_l = np.ones(self.num_loops, dtype='bool')
        if not show_average: mask_l[1:] = False

        if output_c is None: output_c = self.output_c
        comp_c = self.cframe.comp_c
        num_comps = self.cframe.num_comps
        # num_pars_per_comp = self.cframe.num_pars_per_comp
        num_coeffs_per_comp = int(self.num_coeffs / num_comps)

        # par_p   = output_c[comp_c[i_comp]]['par_lp'][i_loop]
        # coeff_e = output_c[comp_c[i_comp]]['coeff_le'][i_loop]

        age_a = self.age_e[:self.num_ages]
        output_sfh_lcza = np.zeros((self.num_loops, num_comps, self.num_mets, self.num_ages))

        for i_comp in range(num_comps):
            for i_loop in range(self.num_loops):
                par_p   = output_c[comp_c[i_comp]]['par_lp'][i_loop]
                coeff_e = output_c[comp_c[i_comp]]['coeff_le'][i_loop]
                if self.sfh_names[i_comp] != 'nonparametric':
                    sfh_factor_e = self.sfh_factor(i_comp, par_p[3:]) 
                    # sfh_factor_e means lum(5500) of each ssp model element per unit SFR (of the peak SFH epoch in this case)
                    # if use csp with a sfh, coeff_e*unitconv means the value of SFH of this csp element in Mun/yr at the peak SFH epoch
                    # coeff_e*unitconv*sfh_factor_e gives the correspoinding the best-fit lum(5500) of each ssp model element
                    coeff_e = np.tile(coeff_e, (self.num_ages,1)).T.flatten() * sfh_factor_e 
                rev_redshift = (1+par_p[0]/299792.458)*(1+self.v0_redshift)-1
                dist_lum = cosmo.luminosity_distance(rev_redshift).to('cm').value
                unitconv = 4*np.pi*dist_lum**2 / const.L_sun.to('erg/s').value * self.spec_flux_scale # convert intrinsic flux5500(rest) to L5500 
                output_sfh_lcza[i_loop,i_comp,:,:] = (coeff_e * unitconv * self.sfrtol_e).reshape(self.num_mets, self.num_ages)

        if num_bins is not None:
            output_sfh_lczb = np.zeros((self.num_loops, num_comps, self.num_mets, num_bins))
            age_b = np.zeros((num_bins))
            logage_a = np.log10(age_a)
            bwidth = (logage_a[-1] - logage_a[0]) / num_bins
            for i_bin in range(num_bins):
                mask_a = (logage_a > (logage_a[0]+i_bin*bwidth)) & (logage_a <= (logage_a[0]+(i_bin+1)*bwidth))
                duration_b = 10.0**(logage_a[0]+(i_bin+1)*bwidth) - 10.0**(logage_a[0]+i_bin*bwidth)
                output_sfh_lczb[:,:,:,i_bin] = (output_sfh_lcza[:,:,:,mask_a] * self.duration_e[:self.num_ages][mask_a]).sum(axis=3) / duration_b
                age_b[i_bin] = 10.0**(logage_a[0]+(i_bin+1/2)*bwidth)
                
        if plot:
            plt.figure(figsize=(9,3))
            plt.subplots_adjust(bottom=0.15, top=0.9, left=0.08, right=0.98, hspace=0, wspace=0.2)
            ax = plt.subplot(1, 2, 1)
            for i_comp in range(output_sfh_lcza.shape[1]):  
                for i_loop in range(output_sfh_lcza.shape[0]):
                    plt.plot(np.log10(age_a), output_sfh_lcza[i_loop,i_comp,2,:], '--')
                plt.plot(np.log10(age_a), output_sfh_lcza[:,i_comp,2,:][mask_l].mean(axis=0), linewidth=4, alpha=0.5, label=f'Mean {self.cframe.comp_c[i_comp]}')
            plt.xlim(1.5,-3); plt.ylim(1,1e4); plt.yscale('log')
            plt.xlabel('Log looking back time (Gyr)'); plt.ylabel('SFR (Msun/yr)'); plt.legend()
            plt.title('Before binning in log time')

            if num_bins is not None:
                ax = plt.subplot(1, 2, 2)
                for i_comp in range(output_sfh_lczb.shape[1]):  
                    for i_loop in range(output_sfh_lczb.shape[0]):
                        plt.bar(np.log10(age_b), output_sfh_lczb[i_loop,i_comp,2,:], bottom=0, width=(np.log10(age_b)[1]-np.log10(age_b)[0])*0.8, 
                        alpha=0.5/output_sfh_lczb.shape[0])
                    plt.bar(np.log10(age_b), output_sfh_lczb[:,i_comp,2,:][mask_l].mean(axis=0), bottom=0, width=(np.log10(age_b)[1]-np.log10(age_b)[0])*0.8,
                           alpha=0.3, hatch='///', ec='C7', linewidth=4, label=f'Mean {self.cframe.comp_c[i_comp]}')
                plt.xlim(1.5,-3); plt.ylim(1,1e4); plt.yscale('log')
                plt.xlabel('Log looking back time (Gyr)'); plt.ylabel('SFR (Msun/yr)'); plt.legend()
                plt.title('After binning in log time')
                
        if return_sfh:
            if num_bins is None:
                return output_sfh_lcza, age_a
            else:
                return output_sfh_lczb, age_b
