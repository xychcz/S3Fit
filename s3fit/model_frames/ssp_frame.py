# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Contact: xiaoyang.chen.cz@gmail.com

import numpy as np
from copy import deepcopy as copy
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import WMAP9 as cosmo
import matplotlib.pyplot as plt

from ..auxiliary_func import convert_linw_to_logw, convolve_spec_logw
from ..extinct_law import ExtLaw

class SSPFrame(object):
    def __init__(self, filename=None, w_min=None, w_max=None, w_norm=5500, dw_norm=25, 
                 cframe=None, v0_redshift=None, spec_R_inst=None, spec_R_init=None, spec_R_rsmp=None, verbose=True):
        self.filename = filename
        self.w_min = w_min
        self.w_max = w_max
        self.w_norm = w_norm
        self.dw_norm = dw_norm
        self.cframe = cframe
        self.v0_redshift = v0_redshift
        self.spec_R_inst = spec_R_inst # instrumental resolution, use to convolve spec to fit dispersion velocity
        self.spec_R_init = spec_R_init # initial resolution of models, use to create low-R spec for multi-band sed fitting (and extraploting)
        self.spec_R_rsmp = spec_R_rsmp # use to set resampling rate of models
        if spec_R_rsmp is None: 
            if spec_R_inst is not None:
                self.spec_R_rsmp = self.spec_R_inst * 5 # set a high resampling density for spectral fit
            else: 
                if spec_R_init is not None:
                    self.spec_R_rsmp = self.spec_R_init * 5

        # load popstar library
        self.read_ssp_library()
        # resample (wave,spec) to logw grid for following spectral convolution
        self.to_logw_grid(w_min=self.w_min, w_max=self.w_max, w_norm=self.w_norm, dw_norm=self.dw_norm, 
                          spec_R_init=self.spec_R_init, spec_R_rsmp=self.spec_R_rsmp)

        # read SFH setup from input config file
        self.sfh_names = np.array([d['sfh_name'] for d in self.cframe.info_c])
        self.num_comps = self.cframe.num_comps
        if self.num_comps == 1:
            if self.sfh_names[0] == 'nonparametric':
                self.num_coeffs = self.num_mets * self.num_ages
            else:
                self.num_coeffs = self.num_mets
        else:
            if np.sum(self.sfh_names == 'nonparametric')  == 0:
                self.num_coeffs = self.num_mets * self.num_comps
            else:
                raise ValueError((f"Nonparametric SFH can only be used with a single component."))

        for i_comp in range(self.num_comps):
            if 10.0**self.cframe.max_cp[i_comp][3] > cosmo.age(self.v0_redshift).value:
                raise ValueError((f"Upper bound of CSP_Age of the component '{self.cframe.comp_c[i_comp]}', "
                    +f" exceeds the universe age {cosmo.age(self.v0_redshift).value} Gyr."))
        
        if verbose:
            print('SSP models normalization wavelength:', w_norm, '+-', dw_norm)
            print('SSP models number:', self.mask_ssp_allowed().sum(), 'used in', self.num_models)
            print('SSP models age range (Gyr):', self.age_m[self.mask_ssp_allowed()].min(),self.age_m[self.mask_ssp_allowed()].max())
            print('SSP models metallicity (Z/H):', np.unique(self.met_m[self.mask_ssp_allowed()])) 
            print('SFH pattern:', self.sfh_names)

    def read_ssp_library(self):
        ##############################################################
        ###### Modify this section to use a different SSP model ######
        self.header = fits.open(self.filename)[0].header
        # wave_axis = 1
        # crval = self.header[f'CRVAL{wave_axis}']
        # cdelt = self.header[f'CDELT{wave_axis}']
        # naxis = self.header[f'NAXIS{wave_axis}']
        # crpix = self.header[f'CRPIX{wave_axis}']
        # if not cdelt: cdelt = 1
        # self.orig_wave_w = crval + cdelt*(np.arange(naxis) + 1 - crpix)
        # use wavelength in data instead of one from header
        self.orig_wave_w = fits.open(self.filename)[1].data
        # load models
        self.orig_flux_mw = fits.open(self.filename)[0].data
        # load remaining mass fraction
        self.remainmassfrac_m = fits.open(self.filename)[2].data
        # leave remainmassfrac_m = 1 if not provided. 

        self.num_models = self.header['NAXIS2']
        self.age_m = np.zeros(self.num_models, dtype='float')
        self.met_m = self.age_m.copy()
        for i in range(self.num_models):
            met, age = self.header[f'NAME{i}'].split('.dat')[0].split('_')[1:3]
            self.age_m[i] = 10**float(age.replace('logt',''))/1e9
            self.met_m[i] = float(met.replace('Z',''))
            # self.mtol_m[i] = 1/float(self.header[f'NORM{i}'])
        ##############################################################
        ##############################################################
        self.num_ages = np.unique(self.age_m).shape[0]
        self.num_mets = np.unique(self.met_m).shape[0]

        # obtain the duration (i.e., bin width) of each ssp if considering a continous SFH
        age_a = np.unique(self.age_m) # self.age_m[:self.num_ages]
        mn_a = (age_a[1:] + age_a[:-1]) / 2
        duration_a = mn_a[1:] - mn_a[:-1]
        duration_a = np.hstack((duration_a[0],duration_a,duration_a[-1]))
        self.duration_m = np.tile(duration_a, (self.num_mets,1)).flatten()

    def to_logw_grid(self, w_min=None, w_max=None, w_norm=None, dw_norm=None, spec_R_init=None, spec_R_rsmp=None):
        # re-project models to log-wavelength grid to following convolution,
        # and normalize models at given wavelength
        # convolve models if spec_R_init is not None
        mask_valid_w = (self.orig_wave_w >= w_min) & (self.orig_wave_w <= w_max)
        linw_wave_w = self.orig_wave_w[mask_valid_w]
        logw_flux_mw = []
        mtol_m = np.zeros(self.num_models, dtype='float')
        for i_mod in range(self.num_models):
            linw_flux_w = self.orig_flux_mw[i_mod, mask_valid_w]
            logw_wave_w, logw_flux_w = convert_linw_to_logw(linw_wave_w, linw_flux_w, resolution=spec_R_rsmp)
            # The original spectra of SSP models are normalized by 1 Msun in unit Lsun/AA.
            # The spectra used here is re-normalized to 1 Lsun/AA at rest 5500AA,
            # i.e., the norm-factor is norm=L5500 before re-normalization.  
            # The re-normalized spectra corresponds to mass of 1 Msun / norm, 
            # i.e., mass-to-lum(5500) ratio is (1 Msun / norm) / (1 Lsun/AA) = 1/norm Msun/(Lsun/AA),
            # i.e., mtol = 1/norm = 1/L5500
            if spec_R_init is not None:
                sigma_init = 299792.458 / spec_R_init / np.sqrt(np.log(256))
                logw_flux_w = convolve_spec_logw(logw_wave_w, logw_flux_w, sigma_init, axis=0)
            mask_norm_w = np.abs(logw_wave_w - w_norm) < dw_norm
            logw_flux_norm = np.mean(logw_flux_w[mask_norm_w])
            logw_flux_mw.append(logw_flux_w / logw_flux_norm)
            mtol_m[i_mod] = 1 / logw_flux_norm # i.e., 1 Msun / logw_flux_norm Lsun/AA
        logw_flux_mw = np.array(logw_flux_mw)
        self.logw_flux_mw = logw_flux_mw
        self.logw_wave_w = logw_wave_w
        self.mtol_m = mtol_m

        # self.logw_flux_mw is normalized to 1 Lsun/AA at rest 5500AA.
        # The corresponding mass is 1 Lsun/AA * mtol_m Msun/(Lsun/AA) = mtol_m Msun = 1/L5500 Msun.
        # The corresponding SFR is mtol_m Msun / (duration_m Gyr) = mtol_m/duration_m Msun/Gyr, duration_m in unit of Gyr (as age)
        # Name sfrtol_m = mtol_m/duration_m * 1e-9, and 
        # self.logw_flux_mw / sfrtol_m return models renormalized to unit SFR, i.e., 1 Mun/yr.
        self.sfrtol_m = self.mtol_m / (self.duration_m * 1e9)

        # extend to longer wavelength in NIR-MIR (e.g., > 3 micron)
        # please comment these lines if moving to another SSP library that initially covers the NIR-MIR range. 
        if  (self.orig_wave_w.max() < 3e4) & (w_max > 2.3e4):
            tmp_log_w = np.log10(logw_wave_w)
            tmp_dlog = tmp_log_w[-1] - tmp_log_w[-2]
            tmp_addn = 1+int((np.log10(w_max) - tmp_log_w[-1]) / tmp_dlog)
            ext_wave_w = 10.0**np.hstack((tmp_log_w, tmp_log_w[-1] + tmp_dlog * (np.arange(tmp_addn)+1))) 
            mask0 = (ext_wave_w > 2.1e4) & (ext_wave_w <= 2.3e4)
            mask1 = (ext_wave_w > 2.3e4)
            index = -4
            ext_flux_mw = []
            for i_mod in range(self.num_models):
                    ext_flux_w = np.interp(ext_wave_w, logw_wave_w, logw_flux_mw[i_mod])
                    tmp_r = np.mean(ext_flux_w[mask0]/ext_wave_w[mask0]**index)
                    ext_flux_w[mask1] = ext_wave_w[mask1]**index * tmp_r
                    ext_flux_mw.append(ext_flux_w)
            ext_flux_mw = np.array(ext_flux_mw)
            self.logw_flux_mw = ext_flux_mw
            self.logw_wave_w = ext_wave_w

    def sfh_factor(self, i_comp, sfh_name, sfh_pars):
        # For a given SFH, i.e., SFR(t) = SFR(csp_age-ssp_age_m), in unit of Msun/yr, 
        # the model of a given ssp (_m) is ssp_spec_mw = SFR(csp_age-ssp_age_m) * (self.logw_flux_mw/sfrtol_m), 
        # Name sfh_factor_m = SFR(csp_age-ssp_age_m) / sfrtol_m = SFR(csp_age-ssp_age_m) * ltosfr_m, 
        # here ltosfr_m can be considered as the lum(rest5500) per unit SFR;
        # full expression of ltosfr_m = (self.duration_m * 1e9) / self.mtol_m.
        # Following this way, sfh_factor_m is the lum(rest5500)_m to achieve a given SFR(csp_age-ssp_age_m), i.e., sfh_func_m. 
        # The returned models are ssp_spec_mw = self.logw_flux_mw * sfh_factor_m.
        # The corresponding lum(rest5500) is 1 * sfh_factor_m, in unit of Lsun/AA,
        # the corresponding mass is mtol_m * sfh_factor_m = SFR(csp_age-ssp_age_m) * (duration_m*1e9), in unit of Msun.
        csp_age = 10.0**sfh_pars[0]
        if csp_age > cosmo.age(self.v0_redshift).value:
            raise ValueError((f"CSP_Age of the {i_comp}-th componnet exceeds the universe age {cosmo.age(self.v0_redshift).value} Gyr."))
        ssp_age_m = self.age_m
        evo_time_m = csp_age - ssp_age_m

        if sfh_name == 'exponential': 
            tau = 10.0**sfh_pars[1]
            sfh_func_m = np.exp(-(evo_time_m) / tau)
        if sfh_name == 'delayed': 
            tau = 10.0**sfh_pars[1]
            sfh_func_m = np.exp(-(evo_time_m) / tau) * evo_time_m
        if sfh_name == 'constant': 
            sfh_func_m = np.ones_like(evo_time_m)
        if sfh_name == 'user': 
            sfh_func_m = self.cframe.info_c[i_comp]['sfh_func'](evo_time_m, sfh_pars)
        ############################
        # Add new SFH function here. 
        ############################

        sfh_func_m[~self.mask_ssp_allowed(i_comp)] = 0 # do not use ssp out of allowed range
        sfh_func_m[evo_time_m < 0] = 0 # do not allow ssp older than csp_age 
        sfh_func_m /= sfh_func_m.max()
        sfh_factor_m = sfh_func_m / self.sfrtol_m
        return sfh_factor_m
        # The total csp model is csp_spec_w = ssp_spec_mw.sum(axis=0) = (self.logw_flux_mw * sfh_factor_m).sum(axis=0)
        # The corresponding lum(at 5500) is (1 * sfh_factor_m).sum(axis=0), in unit of Lsun/AA.
        # The corresponding mass is (mtol_m * sfh_factor_m * remain_m).sum(axis=0), in unit of Msun, 
        # remain_m is the remaining mass fraction. 

        # If the best-fit csp has csp_coeff, 
        # the corresponding lum(at 5500) of csp is csp_coeff * (1 * sfh_factor_m).sum(axis=0) Lsun/AA;
        # the corresponding lum(at 5500) of ssp_m is csp_coeff * sfh_factor_m Lsun/AA, 
        # which equals to coeff_m = (csp_coeff * sfh_factor_m) for direct usage of self.logw_flux_mw (i.e., nonparametic SFH).
        # The total mass can be calculated as (csp_coeff * sfh_factor_m * mtol_m * remain_m).sum(axis=0).
        # Note that in all above (_m).sum(axis=0) is indeed (_m[mask_m]).sum(axis=0), 
        # mask_m is used to mask the allowed ssp models (age and met ranges). 

    def models_unitnorm_obsframe(self, obs_wave_w, input_pars, if_pars_flat=True, spec_R_inst=None):
        # The input model is spectra per unit Lsun/AA at rest 5500AA before dust reddening and redshift, 
        # corresponds to mass of 1/L5500 Msun (L5500 is the lum-value in unit of Lsun/AA from original models normalized per unit Msun).
        # In the fitting for the observed spectra in unit of in erg/s/AA/cm2, 
        # the output model can be considered to be re-normlized to 1 erg/s/AA/cm2 at rest 5500AA before dust reddening and redshift.
        # The corresponding lum is (1/3.826e33*Area) Lsun/AA, where Area is the lum-area in unit of cm2, 
        # and the corresponding mass is (1/3.826e33*Area) * (1/L5500) Msun, 
        # i.e., the real mtol is (1/3.826e33*Area) * (1/L5500) = (1/3.826e33*Area) * self.mtol
        # The values will be used to calculate the stellar mass from the best-fit results. 
        if if_pars_flat: 
            pars = self.cframe.flat_to_arr(input_pars)
        else:
            pars = copy(input_pars)
        if spec_R_inst is None: spec_R_inst = self.spec_R_inst

        for i_comp in range(pars.shape[0]):
            if self.sfh_names[i_comp] == 'nonparametric':
                logw_flux_int_mw = self.logw_flux_mw # copy intrinsic models
            else:
                sfh_factor_m = self.sfh_factor(i_comp, self.sfh_names[i_comp], pars[i_comp,3:])
                tmp_mw = self.logw_flux_mw * sfh_factor_m[:,None] # scaled with sfh_factor_m
                logw_flux_int_mw = tmp_mw.reshape(self.num_mets, self.num_ages, self.logw_wave_w.shape[0]).sum(axis=1)
                # sum in ages to create csp 
            # dust extinction, use logw_spectra for correct convolution (i.e., in logw grid)
            logw_flux_e_mw = logw_flux_int_mw * 10.0**(-0.4 * pars[i_comp,2] * ExtLaw(self.logw_wave_w))
            # redshift models
            z_ratio = (1 + self.v0_redshift) * (1 + pars[i_comp,0]/299792.458) # (1+z) = (1+zv0) * (1+v/c)
            logw_wave_z_w = self.logw_wave_w * z_ratio
            logw_flux_ez_mw = logw_flux_e_mw / z_ratio
            # convolve with intrinsic and instrumental dispersion if spec_R_inst is not None
            if spec_R_inst is not None:
                sigma_disp = pars[i_comp,1] / np.sqrt(np.log(256))
                sigma_inst = 299792.458 / spec_R_inst / np.sqrt(np.log(256))
                sigma_conv = np.sqrt(sigma_disp**2+sigma_inst**2)
                logw_flux_ezc_mw = convolve_spec_logw(logw_wave_z_w, logw_flux_ez_mw, sigma_conv, axis=1)
                # convolution in redshifted- or rest-wavelength does not change result
            else:
                logw_flux_ezc_mw = logw_flux_ez_mw 
                # just copy if convlution not required, e.g., for broad-band sed fitting
            # project to observed wavelength
            obs_flux_scomp_mw = []
            for i_model in range(logw_flux_ezc_mw.shape[0]):
                obs_flux_scomp_mw.append(np.interp(obs_wave_w, logw_wave_z_w, logw_flux_ezc_mw[i_model,:]))
            obs_flux_scomp_mw = np.array(obs_flux_scomp_mw)
            if i_comp == 0: 
                obs_flux_mcomp_mw = obs_flux_scomp_mw
            else:
                obs_flux_mcomp_mw = np.vstack((obs_flux_mcomp_mw, obs_flux_scomp_mw))
        return obs_flux_mcomp_mw
    
    def mask_ssp_allowed(self, i_comp=0, csp=False):
        if not csp: # i.e. for all ssp, depends on i_comp
            age_min, age_max = self.cframe.info_c[i_comp]['age_min'], self.cframe.info_c[i_comp]['age_max']
            age_min = self.age_m.min() if age_min is None else 10.0**age_min
            age_max = cosmo.age(self.v0_redshift).value if age_max == 'universe' else 10.0**age_max
            mask_ssp_allowed_m = (self.age_m >= age_min) & (self.age_m <= age_max)
            met_sel = self.cframe.info_c[i_comp]['met_sel']
            if met_sel != 'all':
                if met_sel == 'solar':
                    mask_ssp_allowed_m &= self.met_m == 0.02
                else:
                    mask_ssp_allowed_m &= np.isin(self.met_m, met_sel)
        else: # loop for all comp
            mask_ssp_allowed_m = np.array([], dtype='bool')
            for i_comp in range(self.num_comps):
                tmp_mask_m = np.ones(self.num_mets, dtype='bool') 
                met_sel = self.cframe.info_c[i_comp]['met_sel']
                if met_sel != 'all':
                    if met_sel == 'solar':
                        tmp_mask_m &= np.unique(self.met_m) == 0.02
                    else:
                        tmp_mask_m &= np.isin(np.unique(self.met_m), met_sel)
                mask_ssp_allowed_m = np.hstack((mask_ssp_allowed_m, tmp_mask_m))
        return mask_ssp_allowed_m

    def mask_ssp_lite_with_num_mods(self, num_ages_lite=8, num_mets_lite=1, verbose=True):
        if self.sfh_names[0] == 'nonparametric':
            # only used in nonparametic, single component
            ages_full, num_ages_full = np.unique(self.age_m), len(np.unique(self.age_m))
            ages_allowed = np.unique(self.age_m[self.mask_ssp_allowed()])
            ages_lite = np.logspace(np.log10(ages_allowed.min()), np.log10(ages_allowed.max()), num=num_ages_lite)
            ages_lite *= 10.0**((np.random.rand(num_ages_lite)-0.5)*np.log10(ages_lite[1]/ages_lite[0]))
            # request log-even ages with random shift
            ind_ages_lite = [np.where(np.abs(ages_full-a)==np.min(np.abs(ages_full-a)))[0][0] for a in ages_lite]
            # np.round(np.linspace(0, num_ages_full-1, num_ages_lite)).astype(int)
            ind_mets_lite = [2,1,3,0][:num_mets_lite] # Z = 0.02 (solar), 0.008, 0.05, 0.004, select with this order
            ind_ssp_lite = np.array([ind_met*num_ages_full+np.arange(num_ages_full)[ind_age] 
                                     for ind_met in ind_mets_lite for ind_age in ind_ages_lite])
            mask_ssp_lite_m = np.zeros_like(self.age_m, dtype='bool')
            mask_ssp_lite_m[ind_ssp_lite] = True
            mask_ssp_lite_m &= self.mask_ssp_allowed()
            if verbose: print('Number of used SSP models:', mask_ssp_lite_m.sum()) 
        else:
            mask_ssp_lite_m = self.mask_ssp_allowed(csp=True)
            if verbose: print('Number of used CSP models:', mask_ssp_lite_m.sum()) 
        return mask_ssp_lite_m

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
            # print('Number of used SSP models:', mask_ssp_lite.sum()) #, np.unique(self.age_m[mask_ssp_lite]))
            # print('Ages with coeffs.sum > 99%:', np.unique(self.age_m[coeffs_full >= coeffs_min]))
            mask_ssp_lite_m = coeffs_full >= coeffs_sort[-num_mods_min]
            mask_ssp_lite_m &= self.mask_ssp_allowed()
            if verbose: 
                print('Number of used SSP models:', mask_ssp_lite_m.sum()) 
                print('Coeffs.sum of used SSP models:', 1-np.cumsum(coeffs_sort)[-num_mods_min]/np.sum(coeffs_sort) ) 
                print('Ages of dominant SSP models:', np.unique(self.age_m[coeffs_full >= coeffs_sort[-5]])) 
        else:
            mask_ssp_lite_m = self.mask_ssp_allowed(csp=True)
            if verbose: print('Number of used CSP models:', mask_ssp_lite_m.sum())             
        return mask_ssp_lite_m

    ##########################################################################
    ########################## Output functions ##############################

    def output_results(self, ff=None):
        num_ssp_comps = self.cframe.num_comps
        num_ssp_pars = self.cframe.num_pars
        num_ssp_coeffs = int(self.num_coeffs / num_ssp_comps)

        spec_wave_w = ff.spec['wave_w']
        spec_flux_scale = ff.spec_flux_scale
        num_mock_loops = ff.num_mock_loops
        best_chi_sq_l = ff.best_chi_sq
        best_x_lp = ff.best_fits_x
        best_coeff_lm = ff.best_coeffs
        fx0, fx1, fc0, fc1 = ff.model_index('ssp', ff.full_model_type)

        name_outvals  = ['chi_sq', 'ssp_voff', 'ssp_fwhm', 'ssp_AV'] # basic fitting parameters
        name_outvals += ['sfh_par'+str(i) for i in range(num_ssp_pars-3)] # SFH related fitting parameters
        name_outvals += ['redshift', 'flux_wavenorm',
                         'loglum_wavenorm', 'logmass_formed', 'logmass_remaining', 'logmtol',
                         'logage_lw', 'logage_mw', 'logmet_lw', 'logmet_mw']
        ind_outvals = {}
        for ind, name in enumerate(name_outvals): ind_outvals[name] = ind

        output_ssp_lcp = np.zeros((num_mock_loops, num_ssp_comps, len(ind_outvals)+num_ssp_coeffs ))
        # p: chi_sq, output_values, ssp_coeffs
        output_ssp_lcp[:, :, 0]                = best_chi_sq_l[:, None]
        output_ssp_lcp[:,:,1:(1+num_ssp_pars)] = best_x_lp[:, fx0:fx1].reshape(num_mock_loops, num_ssp_comps, num_ssp_pars)
        output_ssp_lcp[:,:,-num_ssp_coeffs:]   = best_coeff_lm[:, fc0:fc1].reshape(num_mock_loops, num_ssp_comps, num_ssp_coeffs)
        output_ssp_lcp[:,:,ind_outvals['redshift']] = (1+output_ssp_lcp[:,:,ind_outvals['ssp_voff']]/299792.458)*(1+self.v0_redshift)-1

        ssp_x_lcp = output_ssp_lcp[:,:,1:(1+num_ssp_pars)] # short name
        ssp_coeff_lcm = output_ssp_lcp[:,:,-num_ssp_coeffs:]
        for i_loop in range(num_mock_loops):
            for i_comp in range(num_ssp_comps):
                rev_redshift = output_ssp_lcp[i_loop,i_comp,ind_outvals['redshift']]
                mask_norm_w = np.abs(spec_wave_w/(1+rev_redshift) - self.w_norm) < self.dw_norm # for observed flux at wavenorm=5500AA(rest)
                tmp_spec_mw = self.models_unitnorm_obsframe(spec_wave_w, best_x_lp[i_loop, fx0:fx1])
                tmp_spec_cmw = tmp_spec_mw.reshape(num_ssp_comps, num_ssp_coeffs, len(mask_norm_w))
                tmp_spec_w = np.dot(ssp_coeff_lcm[i_loop,i_comp], tmp_spec_cmw[i_comp])
                output_ssp_lcp[i_loop,i_comp,ind_outvals['flux_wavenorm']] = tmp_spec_w[mask_norm_w].mean()

                dist_lum = cosmo.luminosity_distance(rev_redshift).to('cm').value
                unitconv = 4*np.pi*dist_lum**2 / const.L_sun.to('erg/s').value * spec_flux_scale # convert intrinsic flux5500(rest) to L5500 
                if self.sfh_names[i_comp] == 'nonparametric':
                    coeff_m = ssp_coeff_lcm[i_loop,i_comp] # _m represents all ssp models
                else:
                    sfh_factor_m = self.sfh_factor(i_comp, self.sfh_names[i_comp], ssp_x_lcp[i_loop,i_comp,3:]) 
                    coeff_m = np.tile(ssp_coeff_lcm[i_loop,i_comp], (self.num_ages,1)).T.flatten() * sfh_factor_m 
                    # recover ssp_coeff_lcm where _m represents all csp, to coeff_m where _m represents all ssp
                lum_m = coeff_m * unitconv # intrinsic L5500, in Lsun/AA
                mass_formed_m = lum_m * self.mtol_m
                mass_remaining_m = mass_formed_m * self.remainmassfrac_m
                output_ssp_lcp[i_loop,i_comp,ind_outvals['loglum_wavenorm']] = np.log10((lum_m).sum())
                output_ssp_lcp[i_loop,i_comp,ind_outvals['logmass_formed']] = np.log10((mass_formed_m).sum())
                output_ssp_lcp[i_loop,i_comp,ind_outvals['logmass_remaining']] = np.log10((mass_remaining_m).sum())
                output_ssp_lcp[i_loop,i_comp,ind_outvals['logmtol']] = np.log10(mass_remaining_m.sum() / lum_m.sum())

                output_ssp_lcp[i_loop,i_comp,ind_outvals['logage_lw']] = (lum_m * np.log10(self.age_m)).sum() / lum_m.sum()
                output_ssp_lcp[i_loop,i_comp,ind_outvals['logage_mw']] = (mass_remaining_m * np.log10(self.age_m)).sum() / mass_remaining_m.sum()
                output_ssp_lcp[i_loop,i_comp,ind_outvals['logmet_lw']] = (lum_m * np.log10(self.met_m)).sum() / lum_m.sum()
                output_ssp_lcp[i_loop,i_comp,ind_outvals['logmet_mw']] = (mass_remaining_m * np.log10(self.met_m)).sum() / mass_remaining_m.sum()

        self.output_ssp_lcp = output_ssp_lcp # save to model frame
        self.num_mock_loops = num_mock_loops # for reconstruct_sfh
        self.spec_flux_scale = spec_flux_scale # for reconstruct_sfh and print_results

        # to print in screen
        output_ssp_vals = {}
        for i_comp in range(num_ssp_comps):
            comp_name = self.cframe.comp_c[i_comp]
            output_ssp_vals[comp_name] = { 'mean': ind_outvals.copy(), 'rms': ind_outvals.copy() }
            for name in ind_outvals.keys():
                output_ssp_vals[comp_name]['mean'][name] = np.average(output_ssp_lcp[:,i_comp,ind_outvals[name]], weights=1/best_chi_sq_l)
                output_ssp_vals[comp_name]['rms'][name] = np.std(output_ssp_lcp[:,i_comp,ind_outvals[name]], ddof=1) 
            if self.sfh_names[i_comp] == 'nonparametric':
                ssp_coeff_best_m = np.average(ssp_coeff_lcm[:,0,:], weights=1/best_chi_sq_l, axis=0)
                output_ssp_vals[comp_name]['ssp_normcoeff_mean_m'] = ssp_coeff_best_m / ssp_coeff_best_m.sum()
                output_ssp_vals[comp_name]['ssp_normcoeff_rms_m'] = np.std(ssp_coeff_lcm, axis=(0,1), ddof=1) / ssp_coeff_best_m.sum()
        if num_ssp_comps > 1:
            output_ssp_vals['total'] = { 'mean': ind_outvals.copy(), 'rms': ind_outvals.copy() }
            for val in ind_outvals.keys():
                output_ssp_vals['total']['mean'][val] = 0
                output_ssp_vals['total']['rms'][val] = 0
            lum_l = (10.0**output_ssp_lcp[:,:,ind_outvals['loglum_wavenorm']]).sum(axis=1)
            mass_formed_l = (10.0**output_ssp_lcp[:,:,ind_outvals['logmass_formed']]).sum(axis=1)
            mass_remaining_l = (10.0**output_ssp_lcp[:,:,ind_outvals['logmass_remaining']]).sum(axis=1)
            for (name, val) in zip(['loglum_wavenorm', 'logmass_formed', 'logmass_remaining', 'logmtol'], 
                                   [lum_l, mass_formed_l, mass_remaining_l, mass_remaining_l/lum_l]):
                output_ssp_vals['total']['mean'][name] = np.average(np.log10(val), weights=1/best_chi_sq_l)
                output_ssp_vals['total']['rms'][name] = np.std(np.log10(val), ddof=1)

        self.output_ssp_vals = output_ssp_vals # save to model frame
        self.print_results()

    def print_results(self, output_ssp_vals=None):
        if output_ssp_vals is None: output_ssp_vals = self.output_ssp_vals
        for i_comp in range(len(output_ssp_vals)):
            tmp_vals = output_ssp_vals[[*output_ssp_vals][i_comp]]
            if i_comp < self.cframe.num_comps:
                if self.sfh_names[i_comp] == 'nonparametric':
                    print('')
                    print('Best-fit single stellar populations of nonparametric SFH')
                    cols = 'ID,Age (Gyr),Metallicity,Coeff.mean,Coeff.rms,log(M/L)'
                    fmt_cols = '| {0:^4} | {1:^10} | {2:^6} | {3:^6} | {4:^9} | {5:^8} |'
                    fmt_numbers = '| {:=04d} |   {:=6.4f}   |    {:=6.4f}   |   {:=6.4f}   |   {:=6.4f}  |  {:=6.4f}  |'
                    cols_split = cols.split(',')
                    tbl_title = fmt_cols.format(*cols_split)
                    tbl_border = len(tbl_title)*'-'
                    print(tbl_border)
                    print(tbl_title)
                    print(tbl_border)
                    for i in range(self.num_models):
                        min_ncoeffs = tmp_vals['ssp_normcoeff_mean_m'].max()/10
                        if tmp_vals['ssp_normcoeff_mean_m'][i] < min_ncoeffs: continue
                        tbl_row = []
                        tbl_row.append(i)
                        tbl_row.append(self.age_m[i])
                        tbl_row.append(self.met_m[i])
                        tbl_row.append(tmp_vals['ssp_normcoeff_mean_m'][i]) 
                        tbl_row.append(tmp_vals['ssp_normcoeff_rms_m'][i])
                        tbl_row.append(np.log10(self.mtol_m[i]))
                        print(fmt_numbers.format(*tbl_row))
                    print(tbl_border)
                    print(f'Coeff is the intrinsic flux at rest {self.w_norm}AA in unit of {self.spec_flux_scale} erg/s/cm2/AA.')

            print('')
            msg = ''
            if i_comp < self.cframe.num_comps:
                print(f'Best-fit stellar properties of the <{self.cframe.comp_c[i_comp]}> component with {self.sfh_names[i_comp]} SFH.')
                msg += f'| Chi^2 of best-fit                = {tmp_vals["mean"]["chi_sq"]:10.4f}\n'
                msg += f'| Redshift                         = {tmp_vals["mean"]["redshift"]:10.4f}'
                msg += f' +/- {tmp_vals["rms"]["redshift"]:0.4f}\n'
                msg += f'| Velocity dispersion (km/s)       = {tmp_vals["mean"]["ssp_fwhm"]/2.355:10.4f}'
                msg += f' +/- {tmp_vals["rms"]["ssp_fwhm"]/2.355:0.4f}\n'
                msg += f'| Extinction (AV)                  = {tmp_vals["mean"]["ssp_AV"]:10.4f}'
                msg += f' +/- {tmp_vals["rms"]["ssp_AV"]:0.4f}\n'
                if np.isin(self.sfh_names[i_comp], ['exponential', 'delayed', 'constant', 'user']):
                    msg += f'| log Age_max of CSP (Gyr)         = {tmp_vals["mean"]["sfh_par0"]:10.4f}'
                    msg += f' +/- {tmp_vals["rms"]["sfh_par0"]:0.4f}\n'
                if np.isin(self.sfh_names[i_comp], ['exponential', 'delayed']):
                    msg += f'| log Tau of SFH (Gyr)             = {tmp_vals["mean"]["sfh_par1"]:10.4f}'
                    msg += f' +/- {tmp_vals["rms"]["sfh_par1"]:0.4f}\n'
                if np.isin(self.sfh_names[i_comp], ['user']):
                    for par_name in [*tmp_vals["mean"]]:
                        if par_name[:3] != 'sfh': continue
                        if par_name == 'sfh_par0': continue
                        msg += f'| SFH parameter {par_name[-1]}            = {tmp_vals["mean"][par_name]:10.4f}'
                        msg += f' +/- {tmp_vals["rms"][par_name]:0.4f}\n'
                msg += f'| F{self.w_norm}(rest) ({self.spec_flux_scale} erg/s/cm2/AA) = {tmp_vals["mean"]["flux_wavenorm"]:10.4f}'
                msg += f' +/- {tmp_vals["rms"]["flux_wavenorm"]:0.4f}\n'
            else:
                print(f'Best-fit stellar properties of the total components.')
            msg += f'| log L{self.w_norm}(rest) (Lsun/AA)        = {tmp_vals["mean"]["loglum_wavenorm"]:10.4f}'
            msg += f' +/- {tmp_vals["rms"]["loglum_wavenorm"]:0.4f}\n'
            msg += f'| log Mass (all formed) (Msun)     = {tmp_vals["mean"]["logmass_formed"]:10.4f}'
            msg += f' +/- {tmp_vals["rms"]["logmass_formed"]:0.4f}\n'
            msg += f'| log Mass (remainning) (Msun)     = {tmp_vals["mean"]["logmass_remaining"]:10.4f}'
            msg += f' +/- {tmp_vals["rms"]["logmass_remaining"]:0.4f}\n'
            msg += f'| log M/L{self.w_norm} (Msun/(Lsun/AA))     = {tmp_vals["mean"]["logmtol"]:10.4f}'
            msg += f' +/- {tmp_vals["rms"]["logmtol"]:0.4f}'
            if i_comp < self.cframe.num_comps:
                msg += '\n'
                msg += f'| Lum-weight log Age (Gyr)         = {tmp_vals["mean"]["logage_lw"]:10.4f}'
                msg += f' +/- {tmp_vals["rms"]["logage_lw"]:0.4f}\n'
                msg += f'| Mass-weight log Age (Gyr)        = {tmp_vals["mean"]["logage_mw"]:10.4f}'
                msg += f' +/- {tmp_vals["rms"]["logage_mw"]:0.4f}\n'
                msg += f'| Lum-weight log Metallicity (Z)   = {tmp_vals["mean"]["logmet_lw"]:10.4f}'
                msg += f' +/- {tmp_vals["rms"]["logmet_lw"]:0.4f}\n'
                msg += f'| Mass-weight log Metallicity (Z)  = {tmp_vals["mean"]["logmet_mw"]:10.4f}'
                msg += f' +/- {tmp_vals["rms"]["logmet_mw"]:0.4f}'
            bar = '='*60
            print(bar)
            print(msg)
            print(bar)

    def reconstruct_sfh(self, output_ssp_lcp=None, num_bins=None, plot=True, return_sfh=False):
        num_ssp_comps = self.cframe.num_comps
        num_ssp_pars = self.cframe.num_pars
        num_ssp_coeffs = int(self.num_coeffs / num_ssp_comps)

        if output_ssp_lcp is None: output_ssp_lcp = self.output_ssp_lcp
        ssp_x_lcp = output_ssp_lcp[:,:,1:(1+num_ssp_pars)]
        ssp_coeff_lcm = output_ssp_lcp[:,:,-num_ssp_coeffs:]

        age_a = self.age_m[:self.num_ages]
        output_sfh_lcza = np.zeros((self.num_mock_loops, num_ssp_comps, self.num_mets, self.num_ages))

        for i_loop in range(self.num_mock_loops):
            for i_comp in range(num_ssp_comps):
                if self.sfh_names[i_comp] == 'nonparametric':
                    coeff_m = ssp_coeff_lcm[i_loop,i_comp] # _m represents all ssp models
                else:
                    sfh_factor_m = self.sfh_factor(i_comp, self.sfh_names[i_comp], ssp_x_lcp[i_loop,i_comp,3:]) 
                    coeff_m = np.tile(ssp_coeff_lcm[i_loop,i_comp], (self.num_ages,1)).T.flatten() * sfh_factor_m 
                    # recover ssp_coeff_lcm where _m represents all csp, to coeff_m where _m represents all ssp
                rev_redshift = (1+output_ssp_lcp[i_loop,i_comp,1]/299792.458)*(1+self.v0_redshift)-1
                dist_lum = cosmo.luminosity_distance(rev_redshift).to('cm').value
                unitconv = 4*np.pi*dist_lum**2 / const.L_sun.to('erg/s').value * self.spec_flux_scale # convert intrinsic flux5500(rest) to L5500 
                output_sfh_lcza[i_loop,i_comp,:,:] = (coeff_m * unitconv * self.sfrtol_m).reshape(self.num_mets, self.num_ages)

        if num_bins is not None:
            output_sfh_lczb = np.zeros((self.num_mock_loops, num_ssp_comps, self.num_mets, num_bins))
            age_b = np.zeros((num_bins))
            logage_a = np.log10(age_a)
            bwidth = (logage_a[-1] - logage_a[0]) / num_bins
            for i_bin in range(num_bins):
                mask_a = (logage_a > (logage_a[0]+i_bin*bwidth)) & (logage_a <= (logage_a[0]+(i_bin+1)*bwidth))
                duration_b = 10.0**(logage_a[0]+(i_bin+1)*bwidth) - 10.0**(logage_a[0]+i_bin*bwidth)
                output_sfh_lczb[:,:,:,i_bin] = (output_sfh_lcza[:,:,:,mask_a] * self.duration_m[:self.num_ages][mask_a]).sum(axis=3) / duration_b
                age_b[i_bin] = 10.0**(logage_a[0]+(i_bin+1/2)*bwidth)
                
        if plot:
            plt.figure(figsize=(9,3))
            ax = plt.subplot(1, 2, 1)
            for i_comp in range(output_sfh_lcza.shape[1]):  
                for i_loop in range(output_sfh_lcza.shape[0]):
                    plt.plot(np.log10(age_a), output_sfh_lcza[i_loop,i_comp,2,:], '--')
                plt.plot(np.log10(age_a), output_sfh_lcza[:,i_comp,2,:].mean(axis=0), linewidth=4, alpha=0.5, label=f'Mean {self.cframe.comp_c[i_comp]}')
            plt.xlim(1.5,-3); plt.ylim(1,1e4); plt.yscale('log')
            plt.xlabel('Log looking back time (Gyr)'); plt.ylabel('SFR (Msun/yr)'); plt.legend()
            plt.title('Before binning in log time')

            if num_bins is not None:
                ax = plt.subplot(1, 2, 2)
                for i_comp in range(output_sfh_lczb.shape[1]):  
                    for i_loop in range(output_sfh_lczb.shape[0]):
                        plt.bar(np.log10(age_b), output_sfh_lczb[i_loop,i_comp,2,:], bottom=0, width=(np.log10(age_b)[1]-np.log10(age_b)[0])*0.8, 
                        alpha=0.5/output_sfh_lczb.shape[0])
                    plt.bar(np.log10(age_b), output_sfh_lczb[:,i_comp,2,:].mean(axis=0), bottom=0, width=(np.log10(age_b)[1]-np.log10(age_b)[0])*0.8,
                           alpha=0.3, hatch='///', ec='C7', linewidth=4, label=f'Mean {self.cframe.comp_c[i_comp]}')
                plt.xlim(1.5,-3); plt.ylim(1,1e4); plt.yscale('log')
                plt.xlabel('Log looking back time (Gyr)'); plt.ylabel('SFR (Msun/yr)'); plt.legend()
                plt.title('After binning in log time')
                
        if return_sfh:
            if num_bins is None:
                return output_sfh_lcza, age_a
            else:
                return output_sfh_lczb, age_b
