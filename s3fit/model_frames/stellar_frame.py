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
from astropy.cosmology import Planck18 as cosmo
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from ..auxiliaries.auxiliary_frames import ConfigFrame
from ..auxiliaries.auxiliary_functions import print_log, casefold, color_list_dict, convolve_fix_width_fft, convolve_var_width_fft
from ..auxiliaries.extinct_laws import ExtLaw

class StellarFrame(object):
    def __init__(self, mod_name=None, fframe=None, 
                 config=None, file_path=None, 
                 v0_redshift=None, R_inst_rw=None, 
                 w_min=None, w_max=None, w_norm=5500, dw_norm=25, 
                 Rratio_mod=None, dw_fwhm_dsp=None, dw_pix_inst=None, 
                 verbose=True, log_message=[]):

        self.mod_name = mod_name
        self.fframe = fframe
        self.config = config
        self.file_path = file_path
        self.v0_redshift = v0_redshift
        self.R_inst_rw = R_inst_rw
        self.w_min = w_min
        self.w_max = w_max
        self.w_norm = w_norm
        self.dw_norm = dw_norm
        self.Rratio_mod = Rratio_mod # resolution ratio of model / instrument
        self.dw_fwhm_dsp = dw_fwhm_dsp # model convolving width for downsampling (rest frame)
        self.dw_pix_inst = dw_pix_inst # data sampling width (obs frame)
        self.verbose = verbose
        self.log_message = log_message

        self.cframe=ConfigFrame(self.config)
        self.comp_name_c = self.cframe.comp_name_c
        self.num_comps = self.cframe.num_comps

        ############################################################
        # to be compatible with old version <= 2.2.4
        if len(self.cframe.par_index_cP[0]) == 0:
            self.cframe.par_name_cp = np.array([['voff', 'fwhm', 'Av', 'log_csp_age', 'log_csp_tau'] for i_comp in range(self.num_comps)])
            self.cframe.par_index_cP = [{'voff': 0, 'fwhm': 1, 'Av': 2, 'log_csp_age': 3, 'log_csp_tau': 4} for i_comp in range(self.num_comps)]
        for i_comp in range(self.num_comps):
            if 'age_min' in [*self.cframe.info_c[i_comp]]: self.cframe.info_c[i_comp]['log_ssp_age_min'] = self.cframe.info_c[i_comp]['age_min']
            if 'age_max' in [*self.cframe.info_c[i_comp]]: self.cframe.info_c[i_comp]['log_ssp_age_max'] = self.cframe.info_c[i_comp]['age_max']
            if 'met_sel' in [*self.cframe.info_c[i_comp]]: self.cframe.info_c[i_comp]['ssp_metallicity'] = self.cframe.info_c[i_comp]['met_sel']
        ############################################################

        # read SFH setup from input config file; check alternative names
        for i_comp in range(self.num_comps):
            if casefold(self.cframe.info_c[i_comp]['sfh_name']) in ['exponential']: 
                self.cframe.info_c[i_comp]['sfh_name'] = 'exponential'
            if casefold(self.cframe.info_c[i_comp]['sfh_name']) in ['delayed']: 
                self.cframe.info_c[i_comp]['sfh_name'] = 'delayed'
            if casefold(self.cframe.info_c[i_comp]['sfh_name']) in ['constant', 'burst', 'starburst']: 
                self.cframe.info_c[i_comp]['sfh_name'] = 'constant'
            if casefold(self.cframe.info_c[i_comp]['sfh_name']) in ['nonparametric', 'non-parametric', 'non_parametric']: 
                self.cframe.info_c[i_comp]['sfh_name'] = 'nonparametric'
            if casefold(self.cframe.info_c[i_comp]['sfh_name']) in ['user', 'custom', 'customized']: 
                self.cframe.info_c[i_comp]['sfh_name'] = 'user'
        self.sfh_name_c = np.array([info['sfh_name'] for info in self.cframe.info_c])
        if self.num_comps > 1:
            if np.sum(self.sfh_name_c == 'nonparametric') >= 1:
                raise ValueError((f"Nonparametric SFH can only be used with a single component."))

        # load ssp library
        self.read_ssp_library()

        # count the number of independent model elements
        self.num_coeffs_c = np.zeros(self.num_comps, dtype='int')
        for i_comp in range(self.num_comps):
            if self.sfh_name_c[i_comp] == 'nonparametric':
                self.num_coeffs_c[i_comp] = self.num_mets * self.num_ages
            else:
                self.num_coeffs_c[i_comp] = self.num_mets 
        self.num_coeffs = self.num_coeffs_c.sum()

        # currently do not consider negative spectra 
        self.mask_absorption_e = np.zeros((self.num_coeffs), dtype='bool')

        # check boundaries of stellar ages
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            i_par_log_csp_age = self.cframe.par_index_cP[i_comp]['log_csp_age']
            if self.cframe.par_max_cp[i_comp, i_par_log_csp_age] > np.log10(cosmo.age(self.v0_redshift).value):
                self.cframe.par_max_cp[i_comp, i_par_log_csp_age] = np.log10(cosmo.age(self.v0_redshift).value)
                print_log(f"[WARNING]: Upper bound of log_csp_age of the component '{comp_name}' "
                    +f" is reset to the universe age {cosmo.age(self.v0_redshift).value:.3f} Gyr at z = {self.v0_redshift}.", self.log_message)
            if self.cframe.par_min_cp[i_comp, i_par_log_csp_age] > np.log10(cosmo.age(self.v0_redshift).value):
                self.cframe.par_min_cp[i_comp, i_par_log_csp_age] = np.log10(self.age_e[self.mask_lite_allowed()].min()*1.0001) # take a factor of 1.0001 to avoid (csp_age-ssp_age) < 0
                print_log(f"[WARNING]: Lower bound of log_csp_age of the component '{comp_name}' "
                    +f" exceeds the universe age {cosmo.age(self.v0_redshift).value:.3f} Gyr at z = {self.v0_redshift}, "
                    +f" is reset to the available minimum SSP age {self.age_e[self.mask_lite_allowed()].min():.3f} Gyr.", self.log_message)
            if self.cframe.par_min_cp[i_comp, i_par_log_csp_age] < np.log10(self.age_e[self.mask_lite_allowed()].min()):
                self.cframe.par_min_cp[i_comp, i_par_log_csp_age] = np.log10(self.age_e[self.mask_lite_allowed()].min()*1.0001)
                print_log(f"[WARNING]: Lower bound of log_csp_age of the component '{comp_name}' "
                    +f" is reset to the available minimum SSP age {self.age_e[self.mask_lite_allowed()].min():.3f} Gyr.", self.log_message) 

        # set plot styles
        self.plot_style_C = {}
        self.plot_style_C['sum'] = {'color': 'C0', 'alpha': 0.75, 'linestyle': '-', 'linewidth': 1.5}
        i_red, i_green, i_blue = 0, 0, 0
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            self.plot_style_C[str(comp_name)] = {'color': 'None', 'alpha': 0.5, 'linestyle': '-', 'linewidth': 1}
            i_par_log_csp_age = self.cframe.par_index_cP[i_comp]['log_csp_age']
            log_csp_age_mid = 0.5 * (self.cframe.par_min_cp[i_comp, i_par_log_csp_age] + self.cframe.par_max_cp[i_comp, i_par_log_csp_age])
            if log_csp_age_mid > 0: # > 1 Gyr
                self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['red'], i_red, mode="wrap"))
                i_red += 1
            elif log_csp_age_mid > -1: # 100 Myr - 1 Gyr
                self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['green'], i_green, mode="wrap"))
                i_green += 1
            else: # < 100 Myr
                self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['blue'], i_blue, mode="wrap"))
                i_blue += 1

        if self.verbose:
            print_log(f"SSP models normalization wavelength: {w_norm} +- {dw_norm}", self.log_message)
            print_log(f"SSP models number: {self.mask_lite_allowed().sum()} used in a total of {self.num_models}", self.log_message)
            print_log(f"SSP models age range (Gyr): from {self.age_e[self.mask_lite_allowed()].min():.3f} to {self.age_e[self.mask_lite_allowed()].max():.3f}", self.log_message)
            print_log(f"SSP models metallicity (Z/H): {np.unique(self.met_e[self.mask_lite_allowed()])}", self.log_message) 
            print_log(f"SFH functions: {self.sfh_name_c} for {self.cframe.comp_name_c} components, respectively.", self.log_message)

    ##########################################################################

    def read_ssp_library(self):
        ##############################################################
        ###### Modify this section to use a different SSP model ######
        ssp_lib = fits.open(self.file_path)
        # template resolution step of 0.1 AA, from https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.4781M/abstract
        self.init_dw_fwhm = 0.1 # assume init_dw_fwhm = init_dw_pix

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
            self.age_e[i] = 10.0**float(age.replace('logt',''))/1e9
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
        if self.Rratio_mod is not None:
            ds_R_mod_w = np.interp(orig_wave_w*(1+self.v0_redshift), self.R_inst_rw[0], self.R_inst_rw[1] * self.Rratio_mod) # R_inst_rw in observed frame
            self.dw_fwhm_dsp_w = orig_wave_w / ds_R_mod_w # required resolving width in rest frame
        else:
            if self.dw_fwhm_dsp is not None:
                self.dw_fwhm_dsp_w = np.full(len(orig_wave_w), self.dw_fwhm_dsp)
            else:
                self.dw_fwhm_dsp_w = None
        if self.dw_fwhm_dsp_w is not None:
            if (self.dw_fwhm_dsp_w > self.init_dw_fwhm).all(): 
                preconvolving = True
            else:
                preconvolving = False
                self.dw_fwhm_dsp_w = np.full(len(orig_wave_w), self.init_dw_fwhm)
            self.dw_dsp = self.dw_fwhm_dsp_w.min() * 0.5 # required min bin wavelength following Nyquist–Shannon sampling
            if self.dw_pix_inst is not None:
                self.dw_dsp = min(self.dw_dsp, self.dw_pix_inst/(1+self.v0_redshift) * 0.5) # also require model bin wavelength <= 0.5 of data bin width (convert to rest frame)
            self.dpix_dsp = int(self.dw_dsp / np.median(np.diff(orig_wave_w))) # required min bin number of pixels
            self.dw_dsp = self.dpix_dsp * np.median(np.diff(orig_wave_w)) # update value
            if self.dpix_dsp > 1:
                if preconvolving:
                    if self.verbose: 
                        print_log(f'Downsample preconvolved SSP models with bin width of {self.dw_dsp:.3f} AA in a min resolution of {self.dw_fwhm_dsp_w.min():.3f} AA', 
                                  self.log_message)
                    # before downsampling, smooth the model to avoid aliasing (like in ADC or digital signal reduction)
                    # here assume the internal dispersion in the original model (e.g., in stellar atmosphere) is indepent from the measured dispersion (i.e., stellar motion) in the fitting
                    orig_flux_ew = convolve_fix_width_fft(orig_wave_w, orig_flux_ew, dw_fwhm=self.dw_fwhm_dsp_w.min())
                else:
                    if self.verbose: 
                        print_log(f'Downsample original SSP models with bin width of {self.dw_dsp:.3f} AA in a min resolution of {self.dw_fwhm_dsp_w.min():.3f} AA', 
                                  self.log_message)  
                orig_wave_w = orig_wave_w[::self.dpix_dsp]
                orig_flux_ew = orig_flux_ew[:,::self.dpix_dsp]
                self.dw_fwhm_dsp_w = self.dw_fwhm_dsp_w[::self.dpix_dsp]

        # save the smoothed models
        self.orig_wave_w = orig_wave_w
        self.orig_flux_ew = orig_flux_ew

        # extend to longer wavelength in NIR-MIR (e.g., > 3 micron)
        # please comment these lines if moving to another SSP library that initially covers the NIR-MIR range. 
        if  (orig_wave_w.max() < self.w_max) & (self.w_max > 2.28e4):
            mask_ref_w = (orig_wave_w > 2.1e4) & (orig_wave_w <= 2.28e4) # avoid edge for which fft convolution does not work well
            index = -4 # i.e., blackbody
            ratio_e = np.mean(orig_flux_ew[:,mask_ref_w] / orig_wave_w[None,mask_ref_w]**index, axis=1)

            ext_wave_logbin = 0.02
            ext_wave_num = int(np.round(np.log10(self.w_max/orig_wave_w[mask_ref_w][-1]) / ext_wave_logbin))
            ext_wave_w = np.logspace(np.log10(orig_wave_w[mask_ref_w][-1]+1), np.log10(self.w_max), ext_wave_num)
            ext_flux_ew = ext_wave_w[None,:]**index * ratio_e[:,None]

            mask_keep_w = orig_wave_w < ext_wave_w[0]
            self.orig_flux_ew = np.hstack((orig_flux_ew[:,mask_keep_w], ext_flux_ew))
            self.orig_wave_w = np.hstack((orig_wave_w[mask_keep_w], ext_wave_w))

    ##########################################################################

    def sfh_factor(self, i_comp, par_p):
        # For a given SFH, i.e., SFR(t) = SFR(csp_age-ssp_age_e), in unit of Msun/yr, 
        # the model of a given ssp (_e) is ssp_spec_ew = SFR(csp_age-ssp_age_e) * (self.orig_flux_ew/sfrtol_e), 
        # Name sfh_factor_e = SFR(csp_age-ssp_age_e) / sfrtol_e = SFR(csp_age-ssp_age_e) * ltosfr_e, 
        # here ltosfr_e can be considered as the lum(rest5500) per unit SFR;
        # full expression of ltosfr_e = (self.duration_e * 1e9) / self.mtol_e.
        # Following this way, sfh_factor_e is the lum(rest5500)_e to achieve a given SFR(csp_age-ssp_age_e), i.e., sfh_func_e. 
        # The returned models are ssp_spec_ew = self.orig_flux_ew * sfh_factor_e.
        # The corresponding lum(rest5500) is 1 * sfh_factor_e, in unit of Lsun/AA,
        # the corresponding mass is mtol_e * sfh_factor_e = SFR(csp_age-ssp_age_e) * (duration_e*1e9), in unit of Msun.
        csp_age = 10.0**par_p[self.cframe.par_index_cP[i_comp]['log_csp_age']]
        ssp_age_e = self.age_e
        evo_time_e = csp_age - ssp_age_e

        if self.sfh_name_c[i_comp] == 'exponential': 
            csp_tau = 10.0**par_p[self.cframe.par_index_cP[i_comp]['log_csp_tau']]
            sfh_func_e = np.exp(-(evo_time_e) / csp_tau)
        if self.sfh_name_c[i_comp] == 'delayed': 
            csp_tau = 10.0**par_p[self.cframe.par_index_cP[i_comp]['log_csp_tau']]
            sfh_func_e = np.exp(-(evo_time_e) / csp_tau) * evo_time_e
        if self.sfh_name_c[i_comp] == 'constant': 
            sfh_func_e = np.ones_like(evo_time_e)
        if self.sfh_name_c[i_comp] == 'user': 
            sfh_func_e = self.cframe.info_c[i_comp]['sfh_func'](evo_time_e, par_p, i_comp, self)
        ##########################################################################
        # a user defined sfh function (input via config) has the following format
        # def sfh_user(*args):
        #     # please do not touch the following two lines
        #     time, parameters, i_component, StellarFrame = args
        #     def get_par(par_name): return parameters[StellarFrame.cframe.par_index_cP[i_component][par_name]]
        #     # you can modify the sfh function and parameter names (should be the same as those in input config)
        #     log_t_peak = get_par('log_t_peak')
        #     log_tau = get_par('log_tau')
        #     t_peak = 10.0**log_t_peak
        #     tau = 10.0**log_tau
        #     sfh = np.exp(-(time-t_peak)**2 / tau**2/2)
        #     return sfh
        # or directly add new SFH function here. 
        ##########################################################################

        sfh_func_e[~self.mask_lite_allowed(i_comp)] = 0 # do not use ssp out of allowed range
        sfh_func_e[evo_time_e < 0] = 0 # do not allow ssp older than csp_age 
        if sfh_func_e.max() > 0: sfh_func_e /= sfh_func_e.max()
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

    def models_unitnorm_obsframe(self, obs_wave_w, par_p, mask_lite_e=None, components=None, 
                                 if_dust_ext=True, if_ism_abs=False, if_igm_abs=False, if_redshift=True, if_convolve=True, conv_nbin=None):
        # The input model is spectra per unit Lsun/AA at rest 5500AA before dust reddening and redshift, 
        # corresponds to mass of 1/L5500 Msun (L5500 is the lum-value in unit of Lsun/AA from original models normalized per unit Msun).
        # In the fitting for the observed spectra in unit of in erg/s/AA/cm2, 
        # the output model can be considered to be re-normlized to 1 erg/s/AA/cm2 at rest 5500AA before dust reddening and redshift.
        # The corresponding lum is (1/3.826e33*Area) Lsun/AA, where Area is the lum-area in unit of cm2, 
        # and the corresponding mass is (1/3.826e33*Area) * (1/L5500) Msun, 
        # i.e., the real mtol is (1/3.826e33*Area) * (1/L5500) = (1/3.826e33*Area) * self.mtol
        # The values will be used to calculate the stellar mass from the best-fit results. 
        par_cp = self.cframe.reshape_by_comp(par_p)
        if mask_lite_e is not None: mask_lite_ce = self.cframe.reshape_by_comp(mask_lite_e) # components share the same num_coeffs

        obs_flux_mcomp_ew = None
        for i_comp in range(self.num_comps):
            # build models with SFH function
            if self.sfh_name_c[i_comp] == 'nonparametric':
                if mask_lite_e is not None:
                    orig_flux_int_ew = self.orig_flux_ew[mask_lite_ce[i_comp,:],:] # limit element number for accelarate calculation
                else:
                    orig_flux_int_ew = self.orig_flux_ew
            else:
                sfh_factor_e = self.sfh_factor(i_comp, par_cp[i_comp])
                tmp_mask_e = sfh_factor_e > 0
                tmp_ew = np.zeros_like(self.orig_flux_ew)
                tmp_ew[tmp_mask_e,:] = self.orig_flux_ew[tmp_mask_e,:] * sfh_factor_e[tmp_mask_e,None] # scaled with sfh_factor_e
                orig_flux_int_ew = tmp_ew.reshape(self.num_mets, self.num_ages, len(self.orig_wave_w)).sum(axis=1)
                # sum in ages to create csp 
                if mask_lite_e is not None:
                    orig_flux_int_ew = orig_flux_int_ew[mask_lite_ce[i_comp,:],:]

            # dust extinction
            Av = par_cp[i_comp, self.cframe.par_index_cP[i_comp]['Av']]
            orig_flux_d_ew = orig_flux_int_ew * 10.0**(-0.4 * Av * ExtLaw(self.orig_wave_w))

            # redshift models
            voff = par_cp[i_comp, self.cframe.par_index_cP[i_comp]['voff']]
            z_ratio = (1 + self.v0_redshift) * (1 + voff/299792.458) # (1+z) = (1+zv0) * (1+v/c)
            orig_wave_z_w = self.orig_wave_w * z_ratio
            orig_flux_dz_ew = orig_flux_d_ew / z_ratio

            # convolve with intrinsic and instrumental dispersion if self.R_inst_rw is not None
            if (self.R_inst_rw is not None) & (conv_nbin is not None):
                fwhm = par_cp[i_comp, self.cframe.par_index_cP[i_comp]['fwhm']]
                R_inst_w = np.interp(orig_wave_z_w, self.R_inst_rw[0], self.R_inst_rw[1])
                orig_flux_dzc_ew = convolve_var_width_fft(orig_wave_z_w, orig_flux_dz_ew, dv_fwhm_obj=fwhm, 
                                                          dw_fwhm_ref=self.dw_fwhm_dsp_w*z_ratio, R_inst_w=R_inst_w, num_bins=conv_nbin)
            else:
                orig_flux_dzc_ew = orig_flux_dz_ew # just copy if convlution not required, e.g., for broad-band sed fitting

            # project to observed wavelength
            interp_func = interp1d(orig_wave_z_w, orig_flux_dzc_ew, axis=1, kind='linear', fill_value='extrapolate')
            obs_flux_scomp_ew = interp_func(obs_wave_w)

            if obs_flux_mcomp_ew is None: 
                obs_flux_mcomp_ew = obs_flux_scomp_ew
            else:
                obs_flux_mcomp_ew = np.vstack((obs_flux_mcomp_ew, obs_flux_scomp_ew))

        return obs_flux_mcomp_ew
    
    def mask_lite_allowed(self, i_comp=0, csp=False):
        if not csp: 
            # mask for all SSP elements, for an individual comp
            log_age_min, log_age_max = self.cframe.info_c[i_comp]['log_ssp_age_min'], self.cframe.info_c[i_comp]['log_ssp_age_max']
            age_min = self.age_e.min() if log_age_min is None else 10.0**log_age_min
            age_max = cosmo.age(self.v0_redshift).value if log_age_max in ['universe', 'Universe'] else 10.0**log_age_max
            mask_lite_ssp_e = (self.age_e >= age_min) & (self.age_e <= age_max)
            met_sel = self.cframe.info_c[i_comp]['ssp_metallicity']
            if met_sel != 'all':
                if met_sel in ['solar', 'Solar']:
                    mask_lite_ssp_e &= self.met_e == 0.02
                else:
                    mask_lite_ssp_e &= np.isin(self.met_e, met_sel)
            return mask_lite_ssp_e

        else: 
            # mask for all CSP elements, loop for all comps
            mask_lite_csp_e = np.array([], dtype='bool')
            for i_comp in range(self.num_comps):
                tmp_mask_e = np.ones(self.num_mets, dtype='bool') 
                met_sel = self.cframe.info_c[i_comp]['ssp_metallicity']
                if met_sel != 'all':
                    if met_sel == 'solar':
                        tmp_mask_e &= np.unique(self.met_e) == 0.02
                    else:
                        tmp_mask_e &= np.isin(np.unique(self.met_e), met_sel)
                mask_lite_csp_e = np.hstack((mask_lite_csp_e, tmp_mask_e))
            return mask_lite_csp_e

    def mask_lite_with_num_mods(self, num_ages_lite=8, num_mets_lite=1, verbose=True):
        if self.sfh_name_c[0] == 'nonparametric':
            # only used in nonparametic, single component
            ages_full, num_ages_full = np.unique(self.age_e), len(np.unique(self.age_e))
            ages_allowed = np.unique(self.age_e[self.mask_lite_allowed()])
            ages_lite = np.logspace(np.log10(ages_allowed.min()), np.log10(ages_allowed.max()), num=num_ages_lite)
            ages_lite *= 10.0**((np.random.rand(num_ages_lite)-0.5)*np.log10(ages_lite[1]/ages_lite[0]))
            # request log-even ages with random shift
            ind_ages_lite = [np.where(np.abs(ages_full-a)==np.min(np.abs(ages_full-a)))[0][0] for a in ages_lite]
            # np.round(np.linspace(0, num_ages_full-1, num_ages_lite)).astype(int)
            ind_mets_lite = [2,1,3,0][:num_mets_lite] # Z = 0.02 (solar), 0.008, 0.05, 0.004, select with this order
            ind_ssp_lite = np.array([ind_met*num_ages_full+np.arange(num_ages_full)[ind_age] 
                                     for ind_met in ind_mets_lite for ind_age in ind_ages_lite])
            mask_lite_ssp_e = np.zeros_like(self.age_e, dtype='bool')
            mask_lite_ssp_e[ind_ssp_lite] = True
            mask_lite_ssp_e &= self.mask_lite_allowed()
            if verbose: print_log(f'Number of used SSP models: {mask_lite_ssp_e.sum()}', self.log_message) 
            return mask_lite_ssp_e

        else:
            mask_lite_csp_e = self.mask_lite_allowed(csp=True)
            if verbose: print_log(f'Number of used CSP models: {mask_lite_csp_e.sum()}', self.log_message) 
            return mask_lite_csp_e

    def mask_lite_with_coeffs(self, coeffs=None, mask=None, num_mods_min=32, verbose=True):
        if self.sfh_name_c[0] == 'nonparametric':
            # only used in nonparametic, single component
            coeffs_full = np.zeros(self.num_models)
            coeffs_full[mask if mask is not None else self.mask_lite_allowed()] = coeffs
            coeffs_sort = np.sort(coeffs_full)
            # coeffs_min = coeffs_sort[np.cumsum(coeffs_sort)/np.sum(coeffs_sort) < 0.01].max() 
            # # i.e., keep coeffs with sum > 99%
            # mask_ssp_lite = coeffs_full >= np.minimum(coeffs_min, coeffs_sort[-num_mods_min]) 
            # # keep minimum num of models
            # mask_ssp_lite &= self.mask_lite_allowed()
            # print('Number of used SSP models:', mask_ssp_lite.sum()) #, np.unique(self.age_e[mask_ssp_lite]))
            # print('Ages with coeffs.sum > 99%:', np.unique(self.age_e[coeffs_full >= coeffs_min]))
            mask_lite_ssp_e = coeffs_full >= coeffs_sort[-num_mods_min]
            mask_lite_ssp_e &= self.mask_lite_allowed()
            if verbose: 
                print_log(f'Number of used SSP models: {mask_lite_ssp_e.sum()}', self.log_message) 
                print_log(f'Coeffs.sum of used SSP models: {1-np.cumsum(coeffs_sort)[-num_mods_min]/np.sum(coeffs_sort)}', self.log_message) 
                print_log(f'Ages of dominant SSP models: {np.unique(self.age_e[coeffs_full >= coeffs_sort[-5]])}', self.log_message) 
            return mask_lite_ssp_e

        else:
            mask_lite_csp_e = self.mask_lite_allowed(csp=True)
            if verbose: print_log(f'Number of used CSP models: {mask_lite_csp_e.sum()}', self.log_message)             
            return mask_lite_csp_e

    ##########################################################################
    ########################## Output functions ##############################

    def extract_results(self, step=None, if_print_results=True, if_return_results=False, if_rev_v0_redshift=False, if_show_average=False, lum_unit='Lsun', **kwargs):

        ############################################################
        # check and replace the args to be compatible with old version <= 2.2.4
        if 'print_results'  in [*kwargs]: if_print_results = kwargs['print_results']
        if 'return_results' in [*kwargs]: if_return_results = kwargs['return_results']
        if 'show_average'   in [*kwargs]: if_show_average = kwargs['show_average']
        ############################################################

        if (step is None) | (step in ['best', 'final']): step = 'joint_fit_3' if self.fframe.have_phot else 'joint_fit_2'
        if  step in ['spec+SED', 'spectrum+SED']:  step = 'joint_fit_3'
        if  step in ['spec', 'pure-spec', 'spectrum', 'pure-spectrum']:  step = 'joint_fit_2'
        
        best_chi_sq_l = copy(self.fframe.output_S[step]['chi_sq_l'])
        best_par_lp   = copy(self.fframe.output_S[step]['par_lp'])
        best_coeff_le = copy(self.fframe.output_S[step]['coeff_le'])

        # update best-fit voff and fwhm if systemic redshift is updated
        if if_rev_v0_redshift & (self.fframe.rev_v0_redshift is not None):
            best_par_lp[:, self.fframe.par_name_p == 'voff'] -= self.fframe.ref_voff_l[0]
            best_par_lp[:, self.fframe.par_name_p == 'fwhm'] *= (1+self.fframe.v0_redshift) / (1+self.fframe.rev_v0_redshift)

        self.num_loops = self.fframe.num_loops # for reconstruct_sfh and print_results
        self.spec_flux_scale = self.fframe.spec_flux_scale # for reconstruct_sfh and print_results
        comp_name_c = self.cframe.comp_name_c
        num_comps = self.cframe.num_comps
        par_name_cp = self.cframe.par_name_cp
        num_pars_per_comp = self.cframe.num_pars_c_max
        num_coeffs_per_comp = self.num_coeffs_c[0] # components share the same num_coeffs

        # list the properties to be output; the print will follow this order
        value_names_additive = ['flux_5500', 'flux_wavenorm', 
                                'log_lambLum_5500', 'log_lambLum_wavenorm', 
                                'log_Mass_formed', 'log_Mass_remaining', 'log_MtoL',
                                'log_Age_Lweight', 'log_Age_Mweight', 'log_Z_Lweight', 'log_Z_Mweight']
        value_names_C = {}
        for comp_name in comp_name_c: 
            value_names_C[comp_name] = ['redshift', 'sigma'] + value_names_additive

        # format of results
        # output_C['comp']['par_lp'][i_l,i_p]: parameters
        # output_C['comp']['coeff_le'][i_l,i_e]: coefficients
        # output_C['comp']['value_Vl']['name_l'][i_l]: calculated values
        output_C = {}
        for (i_comp, comp_name) in enumerate(comp_name_c):
            output_C[str(comp_name)] = {} # init results for each comp
            output_C[comp_name]['value_Vl']   = {}
            for val_name in par_name_cp[i_comp].tolist() + value_names_C[comp_name]:
                output_C[comp_name]['value_Vl'][val_name] = np.zeros(self.num_loops, dtype='float')
        output_C['sum'] = {}
        output_C['sum']['value_Vl'] = {} # only init values for sum of all comp
        for val_name in value_names_additive:
            output_C['sum']['value_Vl'][val_name] = np.zeros(self.num_loops, dtype='float')

        i_pars_0_of_mod, i_pars_1_of_mod, i_coeffs_0_of_mod, i_coeffs_1_of_mod = self.fframe.search_mod_index(self.mod_name, self.fframe.full_model_type)
        for (i_comp, comp_name) in enumerate(comp_name_c):
            output_C[comp_name]['par_lp']   = best_par_lp[:, i_pars_0_of_mod:i_pars_1_of_mod].reshape(self.num_loops, num_comps, num_pars_per_comp)[:, i_comp, :]
            output_C[comp_name]['coeff_le'] = best_coeff_le[:, i_coeffs_0_of_mod:i_coeffs_1_of_mod].reshape(self.num_loops, num_comps, num_coeffs_per_comp)[:, i_comp, :]

            for i_par in range(num_pars_per_comp):
                output_C[comp_name]['value_Vl'][par_name_cp[i_comp,i_par]] = output_C[comp_name]['par_lp'][:, i_par]
            for i_loop in range(self.num_loops):
                par_p   = output_C[comp_name]['par_lp'][i_loop]
                coeff_e = output_C[comp_name]['coeff_le'][i_loop]

                voff = par_p[self.cframe.par_index_cP[i_comp]['voff']]
                rev_redshift = (1+self.v0_redshift) * (1+voff/299792.458) - 1
                output_C[comp_name]['value_Vl']['redshift'][i_loop] = copy(rev_redshift)
                fwhm = par_p[self.cframe.par_index_cP[i_comp]['fwhm']]
                output_C[comp_name]['value_Vl']['sigma'][i_loop] = fwhm/np.sqrt(np.log(256))

                tmp_spec_w = self.fframe.output_MC[self.mod_name][comp_name]['spec_lw'][i_loop, :]
                mask_norm_w = np.abs(self.fframe.spec['wave_w']/(1+rev_redshift) - 5500) < 25 # for observed flux at rest 5500 AA
                if mask_norm_w.sum() > 0:
                    output_C[comp_name]['value_Vl']['flux_5500'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                    output_C['sum']['value_Vl']['flux_5500'][i_loop] += tmp_spec_w[mask_norm_w].mean()

                mask_norm_w = np.abs(self.fframe.spec['wave_w']/(1+rev_redshift) - self.w_norm) < self.dw_norm # for observed flux at user given wavenorm
                output_C[comp_name]['value_Vl']['flux_wavenorm'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                output_C['sum']['value_Vl']['flux_wavenorm'][i_loop] += tmp_spec_w[mask_norm_w].mean()

                dist_lum = cosmo.luminosity_distance(rev_redshift).to('cm').value
                unitconv = 4*np.pi*dist_lum**2 * self.spec_flux_scale # convert intrinsic flux to Lum, in erg/s
                if lum_unit == 'Lsun': unitconv /= const.L_sun.to('erg/s').value

                if self.sfh_name_c[i_comp] != 'nonparametric':
                    sfh_factor_e = self.sfh_factor(i_comp, par_p) 
                    # sfh_factor_e means lum(5500) of each ssp model element per unit SFR (of the peak SFH epoch in this case)
                    # if use csp with a sfh, coeff_e*unitconv means the value of SFH of this csp element in Mun/yr at the peak SFH epoch
                    # coeff_e*unitconv*sfh_factor_e gives the correspoinding the best-fit lum(5500) of each ssp model element
                    coeff_e = np.tile(coeff_e, (self.num_ages,1)).T.flatten() * sfh_factor_e 
                Lum_5500_e = coeff_e * unitconv # intrinsic L5500, in Lsun/AA
                lambLum_5500_e = Lum_5500_e * 5500
                lambLum_wavenorm_e = Lum_5500_e * self.flux_norm_ratio_e * self.w_norm
                if lum_unit == 'erg/s': Lum_5500_e /= const.L_sun.to('erg/s').value
                Mass_formed_e = Lum_5500_e * self.mtol_e # mtol_e is in unit of Msun/(Lsun/AA)
                Mass_remaining_e = Mass_formed_e * self.remainmassfrac_e

                output_C[comp_name]['value_Vl']['log_lambLum_5500'][i_loop]   = np.log10(lambLum_5500_e.sum())
                output_C[comp_name]['value_Vl']['log_lambLum_wavenorm'][i_loop]   = np.log10(lambLum_wavenorm_e.sum())
                output_C[comp_name]['value_Vl']['log_Mass_formed'][i_loop]    = np.log10(Mass_formed_e.sum())
                output_C[comp_name]['value_Vl']['log_Mass_remaining'][i_loop] = np.log10(Mass_remaining_e.sum())
                output_C[comp_name]['value_Vl']['log_MtoL'][i_loop]   = np.log10(Mass_remaining_e.sum() / lambLum_5500_e.sum())
                output_C[comp_name]['value_Vl']['log_Age_Lweight'][i_loop] = (lambLum_5500_e * np.log10(self.age_e)).sum() / lambLum_5500_e.sum()
                output_C[comp_name]['value_Vl']['log_Age_Mweight'][i_loop] = (Mass_remaining_e * np.log10(self.age_e)).sum() / Mass_remaining_e.sum()
                output_C[comp_name]['value_Vl']['log_Z_Lweight'][i_loop] = (lambLum_5500_e * np.log10(self.met_e)).sum() / lambLum_5500_e.sum()
                output_C[comp_name]['value_Vl']['log_Z_Mweight'][i_loop] = (Mass_remaining_e * np.log10(self.met_e)).sum() / Mass_remaining_e.sum()

                output_C['sum']['value_Vl']['log_lambLum_5500'][i_loop]   += lambLum_5500_e.sum() # keep in linear for sum
                output_C['sum']['value_Vl']['log_lambLum_wavenorm'][i_loop]   += lambLum_wavenorm_e.sum() # keep in linear for sum
                output_C['sum']['value_Vl']['log_Mass_formed'][i_loop]    += Mass_formed_e.sum() # keep in linear for sum
                output_C['sum']['value_Vl']['log_Mass_remaining'][i_loop] += Mass_remaining_e.sum() # keep in linear for sum
                output_C['sum']['value_Vl']['log_Age_Lweight'][i_loop] += (lambLum_5500_e * np.log10(self.age_e)).sum()
                output_C['sum']['value_Vl']['log_Age_Mweight'][i_loop] += (Mass_remaining_e * np.log10(self.age_e)).sum()
                output_C['sum']['value_Vl']['log_Z_Lweight'][i_loop] += (lambLum_5500_e * np.log10(self.met_e)).sum()
                output_C['sum']['value_Vl']['log_Z_Mweight'][i_loop] += (Mass_remaining_e * np.log10(self.met_e)).sum()

        output_C['sum']['value_Vl']['log_MtoL'] = np.log10(output_C['sum']['value_Vl']['log_Mass_remaining'] / output_C['sum']['value_Vl']['log_lambLum_5500'])
        output_C['sum']['value_Vl']['log_Age_Lweight'] = output_C['sum']['value_Vl']['log_Age_Lweight'] / output_C['sum']['value_Vl']['log_lambLum_5500']
        output_C['sum']['value_Vl']['log_Age_Mweight'] = output_C['sum']['value_Vl']['log_Age_Mweight'] / output_C['sum']['value_Vl']['log_Mass_remaining']
        output_C['sum']['value_Vl']['log_Z_Lweight'] = output_C['sum']['value_Vl']['log_Z_Lweight'] / output_C['sum']['value_Vl']['log_lambLum_5500']
        output_C['sum']['value_Vl']['log_Z_Mweight'] = output_C['sum']['value_Vl']['log_Z_Mweight'] / output_C['sum']['value_Vl']['log_Mass_remaining']
        output_C['sum']['value_Vl']['log_lambLum_5500']       = np.log10(output_C['sum']['value_Vl']['log_lambLum_5500'])
        output_C['sum']['value_Vl']['log_lambLum_wavenorm']   = np.log10(output_C['sum']['value_Vl']['log_lambLum_wavenorm'])
        output_C['sum']['value_Vl']['log_Mass_formed']    = np.log10(output_C['sum']['value_Vl']['log_Mass_formed'])
        output_C['sum']['value_Vl']['log_Mass_remaining'] = np.log10(output_C['sum']['value_Vl']['log_Mass_remaining'])

        ############################################################
        # keep aliases for output in old version <= 2.2.4
        for (i_comp, comp_name) in enumerate(comp_name_c):
            output_C[comp_name]['values'] = output_C[comp_name]['value_Vl']
        output_C['sum']['values'] = output_C['sum']['value_Vl']
        ############################################################

        i_comp = 0 # only enable one comp if nonparametric SFH is used
        if self.sfh_name_c[i_comp] == 'nonparametric':
            coeff_le = output_C[comp_name_c[i_comp]]['coeff_le']
            output_C[comp_name_c[i_comp]]['coeff_norm_le'] = coeff_le / coeff_le.sum(axis=1)[:,None]

        self.output_C = output_C # save to model frame
 
        if if_print_results: self.print_results(log=self.fframe.log_message, if_show_average=if_show_average, lum_unit=lum_unit)
        if if_return_results: return output_C

    def print_results(self, log=[], if_show_average=False, lum_unit='Lsun'):
        print_log(f"#### Best-fit stellar properties ####", log)

        mask_l = np.ones(self.num_loops, dtype='bool')
        if not if_show_average: mask_l[1:] = False
        lum_unit_str = '(log Lsun) ' if lum_unit == 'Lsun' else '(log erg/s)'

        # set the print name for each value
        value_names = [value_name for comp_name in self.output_C for value_name in [*self.output_C[comp_name]['value_Vl']]]
        value_names = list(dict.fromkeys(value_names)) # remove duplicates
        print_names = {}
        for value_name in value_names: print_names[value_name] = value_name
        print_names['voff'] = 'Velocity shift in relative to z_sys (km/s)'
        print_names['fwhm'] = 'Velocity FWHM (km/s)'
        print_names['sigma'] = 'Velocity dispersion (σ) (km/s)'
        print_names['Av'] = 'Extinction (Av)'
        print_names['log_csp_age'] = 'Max age of composite star.pop. (log Gyr)'
        print_names['log_csp_tau'] = 'Declining timescale of SFH (log Gyr)'
        print_names['redshift'] = 'Redshift (from continuum absorptions)'
        print_names['flux_5500'] = f"F5500 (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)"
        print_names['log_lambLum_5500'] = f"λL5500 (rest,intrinsic) "+lum_unit_str
        print_names['flux_wavenorm'] = f"F{self.w_norm:.0f} (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)"
        print_names['log_lambLum_wavenorm'] = f"λL{self.w_norm:.0f} (rest,intrinsic) "+lum_unit_str
        print_names['log_Mass_formed'] = 'Mass (all formed) (log Msun)'
        print_names['log_Mass_remaining'] = 'Mass (remaining) (log Msun)'
        print_names['log_MtoL'] = 'Mass/λL5500 (log Msun/Lsun)'
        print_names['log_Age_Lweight'] = 'λL5500-weight age (log Gyr)'
        print_names['log_Age_Mweight'] = 'Mass-weight age (log Gyr)'
        print_names['log_Z_Lweight'] = 'λL5500-weight metallicity (log Z)'
        print_names['log_Z_Mweight'] = 'Mass-weight metallicity (log Z)'
        print_length = max([len(print_names[value_name]) for value_name in print_names] + [40]) # set min length
        for value_name in print_names:
            print_names[value_name] += ' '*(print_length-len(print_names[value_name]))

        for i_comp in range(len(self.output_C)):
            values_vl = self.output_C[[*self.output_C][i_comp]]['value_Vl']
            value_names = [*values_vl]
            msg = ''
            if i_comp < self.cframe.num_comps: # print best-fit pars for each comp
                print_log(f"# Stellar component <{self.cframe.comp_name_c[i_comp]}> with {self.sfh_name_c[i_comp]} SFH:", log)
                value_names = [value_name for value_name in value_names if value_name[:6] != 'Empty_'] # remove unused pars
                value_names.remove('redshift'); value_names = ['redshift'] + value_names # move redshift to the begining
                value_names.remove('sigma'); value_names = ['sigma' if value_name == 'fwhm' else value_name for value_name in value_names] # print sigma instead of fwhm
            elif self.cframe.num_comps >= 2: # print sum only if using >= 2 comps
                print_log(f"# Best-fit properties of the sum of all stellar components.", log)
            else: 
                continue
            if self.w_norm == 5500:
                value_names.remove('flux_wavenorm')
                value_names.remove('log_lambLum_wavenorm')
            for value_name in value_names:
                msg += '| ' + print_names[value_name] + f" = {values_vl[value_name][mask_l].mean():10.4f}" + f" +/- {values_vl[value_name].std():<10.4f}|\n"
            msg = msg[:-1] # remove the last \n
            bar = '=' * len(msg.split('\n')[-1])
            print_log(bar, log)
            print_log(msg, log)
            print_log(bar, log)
            print_log('', log)

        i_comp = 0 # only enable one comp if nonparametric SFH is used
        if self.sfh_name_c[i_comp] == 'nonparametric':
            print_log('# Best-fit single stellar populations (SSP) with nonparametric SFH', log)
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
                coeff_norm_mn_e  = self.output_C[self.cframe.comp_name_c[i_comp]]['coeff_norm_le'][mask_l].mean(axis=0)
                coeff_norm_std_e = self.output_C[self.cframe.comp_name_c[i_comp]]['coeff_norm_le'].std(axis=0)
                if coeff_norm_mn_e[i_e] < 0.01: continue
                tbl_row = []
                tbl_row.append(i_e)
                tbl_row.append(self.age_e[i_e])
                tbl_row.append(self.met_e[i_e])
                tbl_row.append(coeff_norm_mn_e[i_e]) 
                tbl_row.append(coeff_norm_std_e[i_e])
                tbl_row.append(np.log10(self.mtol_e[i_e]))
                print_log(fmt_numbers.format(*tbl_row), log)
            print_log(tbl_border, log)
            print_log(f"[Note] Coeff is the normalized fraction of the intrinsic flux at rest 5500 AA.", log)
            print_log(f"[Note] only SSPs with Coeff over 1% are listed.", log)
            print_log('', log)

    def reconstruct_sfh(self, output_C=None, num_bins=None, if_plot_sfh=True, if_return_sfh=False, if_show_average=True, **kwargs):

        ############################################################
        # check and replace the args to be compatible with old version <= 2.2.4
        if 'plot'         in [*kwargs]: if_plot_sfh = kwargs['plot']
        if 'return_sfh'   in [*kwargs]: if_return_sfh = kwargs['return_sfh']
        if 'show_average' in [*kwargs]: if_show_average = kwargs['show_average']
        ############################################################

        mask_l = np.ones(self.num_loops, dtype='bool')
        if not if_show_average: mask_l[1:] = False

        if output_C is None: output_C = self.output_C
        comp_name_c = self.cframe.comp_name_c
        num_comps = self.cframe.num_comps

        age_a = self.age_e[:self.num_ages]
        output_sfh_lcza = np.zeros((self.num_loops, num_comps, self.num_mets, self.num_ages))

        for (i_comp, comp_name) in enumerate(comp_name_c):
            for i_loop in range(self.num_loops):
                par_p   = output_C[comp_name]['par_lp'][i_loop]
                coeff_e = output_C[comp_name]['coeff_le'][i_loop]
                if self.sfh_name_c[i_comp] != 'nonparametric':
                    sfh_factor_e = self.sfh_factor(i_comp, par_p) 
                    # sfh_factor_e means lum(5500) of each ssp model element per unit SFR (of the peak SFH epoch in this case)
                    # if use csp with a sfh, coeff_e*unitconv means the value of SFH of this csp element in Mun/yr at the peak SFH epoch
                    # coeff_e*unitconv*sfh_factor_e gives the correspoinding the best-fit lum(5500) of each ssp model element
                    coeff_e = np.tile(coeff_e, (self.num_ages,1)).T.flatten() * sfh_factor_e
                voff = par_p[self.cframe.par_index_cP[i_comp]['voff']]
                rev_redshift = (1+self.v0_redshift) * (1+voff/299792.458) - 1
                dist_lum = cosmo.luminosity_distance(rev_redshift).to('cm').value
                unitconv = 4*np.pi*dist_lum**2 / const.L_sun.to('erg/s').value * self.spec_flux_scale # convert intrinsic flux5500(rest) to L5500 
                output_sfh_lcza[i_loop,i_comp,:,:] = (coeff_e * unitconv * self.sfrtol_e).reshape(self.num_mets, self.num_ages)

        if num_bins is not None:
            output_sfh_lczb = np.zeros((self.num_loops, num_comps, self.num_mets, num_bins))
            age_b = np.zeros((num_bins))
            log_age_a = np.log10(age_a)
            bwidth = (log_age_a[-1] - log_age_a[0]) / num_bins
            for i_bin in range(num_bins):
                mask_a = (log_age_a > (log_age_a[0]+i_bin*bwidth)) & (log_age_a <= (log_age_a[0]+(i_bin+1)*bwidth))
                duration_b = 10.0**(log_age_a[0]+(i_bin+1)*bwidth) - 10.0**(log_age_a[0]+i_bin*bwidth)
                output_sfh_lczb[:,:,:,i_bin] = (output_sfh_lcza[:,:,:,mask_a] * self.duration_e[:self.num_ages][mask_a]).sum(axis=3) / duration_b
                age_b[i_bin] = 10.0**(log_age_a[0]+(i_bin+1/2)*bwidth)
                
        if if_plot_sfh:
            plt.figure(figsize=(9,3))
            plt.subplots_adjust(bottom=0.15, top=0.9, left=0.08, right=0.98, hspace=0, wspace=0.2)
            ax = plt.subplot(1, 2, 1)
            for i_comp in range(output_sfh_lcza.shape[1]):  
                for i_loop in range(output_sfh_lcza.shape[0]):
                    plt.plot(np.log10(age_a), output_sfh_lcza[i_loop,i_comp,2,:], '--')
                plt.plot(np.log10(age_a), output_sfh_lcza[:,i_comp,2,:][mask_l].mean(axis=0), linewidth=4, alpha=0.5, label=f'Mean {self.cframe.comp_name_c[i_comp]}')
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
                           alpha=0.3, hatch='///', ec='C7', linewidth=4, label=f'Mean {self.cframe.comp_name_c[i_comp]}')
                plt.xlim(1.5,-3); plt.ylim(1,1e4); plt.yscale('log')
                plt.xlabel('Log looking back time (Gyr)'); plt.ylabel('SFR (Msun/yr)'); plt.legend()
                plt.title('After binning in log time')
                
        if if_return_sfh:
            if num_bins is None:
                return output_sfh_lcza, age_a
            else:
                return output_sfh_lczb, age_b
