# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Contact: xiaoyang.chen.cz@gmail.com

import os, time, traceback
import numpy as np
from copy import deepcopy as copy
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import WMAP9 as cosmo
from astropy.convolution import Gaussian1DKernel
from scipy.signal import fftconvolve
from scipy.optimize import least_squares, lsq_linear
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

#############################################################################################################
################################## Auxiliary Functions ######################################################

def print_time(message, time_last=time.time(), time_init=time.time()):
    print(f'**** {message}, {time.time()-time_last:.1f} s; totally spent, {time.time()-time_init:.1f} s ****')
    return time.time()

# https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
def lamb_air_to_vac(lamb_air):
    s = 1e4 / lamb_air
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + \
            0.0001599740894897 / (38.92568793293 - s**2)
    lamb_vac = lamb_air * n
    return lamb_vac

def convert_linw_to_logw(linw_wave, linw_flux, linw_error=None, resolution=None):
    # adopt grid resampling to keep (1/0.8) times original density at the long wavelength end
    if resolution is None: resolution = linw_wave.max()/(0.8*(linw_wave[1]-linw_wave[0]))
    logw_logwidth = np.log(1/resolution + 1)
    logw_wave = np.logspace(np.log(linw_wave.min()), np.log(linw_wave.max()), base=np.e, 
                            num=int(np.log(linw_wave.max()/linw_wave.min()) / logw_logwidth))
    logw_flux = np.interp(logw_wave, linw_wave, linw_flux)
    if linw_error is not None: 
        logw_error = np.interp(logw_wave, linw_wave, linw_error)
        return logw_wave, logw_flux, logw_error
    else:
        return logw_wave, logw_flux

def convolve_spec_logw(logw_wave, logw_flux, conv_sigma, axis=0):
    # logw_wave, logw_flux need to be uniform with log_e wavelength
    logw_width = np.log(logw_wave[1])-np.log(logw_wave[0])
    kernel = Gaussian1DKernel(stddev=conv_sigma/299792.458/logw_width).array
    kernel /= kernel.sum()
    if len(logw_flux.shape) == 2:
        if axis == 0: kernel = kernel[:, None]
        if axis == 1: kernel = kernel[None, :]
    logw_fcon = fftconvolve(logw_flux, kernel, mode='same', axes=axis)
    return logw_fcon

#############################################################################################################
################################## Extinction Functions #####################################################

# https://articles.adsabs.harvard.edu/pdf/1989ApJ...345..245C
def ExtLaw_CCM89(wave, RV=3.1):
    # wave in Angstorm
    x = 1e4 / wave # in um-1
    a_out, b_out = np.zeros_like(x), np.zeros_like(x)
    # IR: 0.3 <= x <= 1.1 um-1
    mask_x = (x >= 0.3) & (x <= 1.1)
    a =  0.574*x**1.61
    b = -0.527*x**1.61
    a_out[mask_x], b_out[mask_x] = a[mask_x], b[mask_x]
    # Optical/NIR: 1.1 < x <= 3.3
    mask_x = (x > 1.1) & (x <= 3.3)
    y = x - 1.82    
    a = 1.+ 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b =     1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    a_out[mask_x], b_out[mask_x] = a[mask_x], b[mask_x]
    # UV: 3.3 < x <= 8
    mask_x = (x >= 5.9) & (x <= 8.0)
    F_a = -0.04473*(x - 5.9)**2 - 0.009779*(x - 5.9)**3
    F_b =  0.21300*(x - 5.9)**2 + 0.120700*(x - 5.9)**3
    F_a[~mask_x], F_b[~mask_x] = 0, 0
    mask_x = (x > 3.3) & (x <= 8.0)
    a =  1.752 - 0.316*x - 0.104/((x - 4.67)**2 + 0.341) + F_a
    b = -3.090 + 1.825*x + 1.206/((x - 4.62)**2 + 0.263) + F_b
    a_out[mask_x], b_out[mask_x] = a[mask_x], b[mask_x]
    # FUV: 8 < x <= 10
    mask_x = (x > 8.0) & (x <= 10.0)
    a = -1.073 - 0.628*(x - 8.0) + 0.137*(x - 8.0)**2 - 0.070*(x - 8.0)**3
    b = 13.670 + 4.257*(x - 8.0) - 0.420*(x - 8.0)**2 + 0.374*(x - 8.0)**3
    a_out[mask_x], b_out[mask_x] = a[mask_x], b[mask_x]
    return a_out + b_out / RV # A_lambda / AV

# http://www.bo.astro.it/~micol/Hyperz/old_public_v1/hyperz_manual1/node10.html
def ExtLaw_Calzetti00(wave, RV=4.05):
    # wave in Angstorm
    x = 1e4 / wave # in um-1
    k_out = np.zeros_like(x)
    # extend short-wave edge from 1200 to 90
    mask_w = (wave >= 90) & (wave <= 6300)
    k = 2.659*(-2.156 + 1.509*x - 0.198*x**2 + 0.011*x**3) + RV
    k_out[mask_w] = k[mask_w]
    # 6300 -> 22000 AA
    mask_w = (wave > 6300) & (wave <= 22000)
    k = 2.659*(-1.857 + 1.040*x) + RV
    k_out[mask_w] = k[mask_w]
    if np.sum(wave > 22000) > 0:
        mask_w0 = wave <= 22000
        index, r = np.polyfit(np.log10(wave[mask_w0][-2:]), np.log10(k_out[mask_w0][-2:]), 1)
        mask_w1 = wave > 22000
        k_out[mask_w1] = wave[mask_w1]**index*10.0**r
    return k_out / RV # A_lambda / AV

#################################################
# define the extinction law used in the fit
ExtLaw = ExtLaw_Calzetti00
#################################################

#############################################################################################################
#############################################################################################################

class FitFrame(object):
    def __init__(self, 
                 spec_wave_w=None, spec_flux_w=None, spec_ferr_w=None, 
                 spec_valid_range=None, spec_R_inst=None, spec_flux_scale=None, 
                 phot_name_b=None, phot_flux_b=None, phot_ferr_b=None, phot_fluxunit='mJy', 
                 phot_trans_dir=None, sed_wave_w=None, sed_waveunit='angstrom', 
                 v0_redshift=None, model_config=None, 
                 num_mock_loops=0, fit_raw=True, fit_grid='linear', 
                 plot_step=False, print_step=True, verbose=False): 

        # print('v2, 240306: (1) [NII] broad 2; (2) add option of fit of raw flux (non-mocked)')
        # print('v3, 241029: (1) [NI]5200; (2) tie Balmer lines with AV; (3) limit [SII] ratio')
        # print('v4, 241116: (1) AGN PL; (2) rebuild')
        # print('v4.1, 241119: (1) AGN PL; (2) rebuild; (3) components examine')
        # print('v5, 241211: (1) fit weight cor; (2) add flux_scale; ')
        # print('v5.1, 241217: (1) PhotFrame (2) Rename')
        # print('v5.2, 250120: (1) Joint fit (2) Add torus')
        # print('v1, 250121: (1) S3Fit initialized')
        # print('v1.2, 250125: (1) Update ConfigFrame')
        # print('v1.4, 250126: (1) Support flexible SFH (2) Add SFH output function'
        print('v1.5, 250128: (1) First release of S3Fit')
        
        # read spec data
        self.spec = {'wave_w': spec_wave_w, 'flux_w': spec_flux_w/spec_flux_scale, 'ferr_w': spec_ferr_w/spec_flux_scale}
        self.num_spec_wave = len(self.spec['wave_w'])
        self.spec_valid_range = spec_valid_range
        self.spec_R_inst = spec_R_inst
        self.spec_flux_scale = spec_flux_scale # flux_scale is used to avoid too small values
        
        # read phot-SED data
        self.have_phot = True if phot_name_b is not None else False
        if self.have_phot:
            print('Photometric data in bands:', phot_name_b)
            self.have_phot = True
            self.pframe = PhotFrame(name_b=phot_name_b, flux_b=phot_flux_b, ferr_b=phot_ferr_b, fluxunit=phot_fluxunit,
                                    trans_dir=phot_trans_dir, wave_w=sed_wave_w, waveunit=sed_waveunit)
            self.phot = {'wave_b': self.pframe.wave_b, 'flux_b': self.pframe.flux_b/spec_flux_scale, 'ferr_b': self.pframe.ferr_b/spec_flux_scale, 
                         'trans_bw': self.pframe.read_transmission(trans_dir=self.pframe.trans_dir, name_b=self.pframe.name_b, wave_w=self.pframe.wave_w)[1]}
            self.num_phot_band = len(self.phot['wave_b'])
            self.sed = {'wave_w': self.pframe.wave_w } 
            self.num_sed_wave = len(self.sed['wave_w'])
        
        # load default redshift
        self.v0_redshift = v0_redshift
        # set fitting wavelength range (rest frame), set tolerance of [-1000,1000] km/s
        self.spec_wmin = spec_wave_w.min() / (1+self.v0_redshift) / (1+1000/299792.458) - 100
        self.spec_wmax = spec_wave_w.max() / (1+self.v0_redshift) / (1-1000/299792.458) + 100
        self.spec_wmin = np.maximum(self.spec_wmin, 91) # set lower limit of wavelength to 91A
        print('Spec models wavelength range (rest):', self.spec_wmin, self.spec_wmax)
        if self.have_phot:
            self.sed_wmin = self.sed['wave_w'].min() / (1+self.v0_redshift)
            self.sed_wmax = self.sed['wave_w'].max() / (1+self.v0_redshift)
            print('SED models wavelength range (rest):', self.sed_wmin, self.sed_wmax)        

        # import models
        self.load_models(model_config)

        # set fitting boundaries 
        self.set_parameter_constraints()

        # set mask_valid_w and mask_noeline_w
        self.set_masks(eline_waves=self.model_dict['el']['specmod'].line_rest_n, eline_vwin=[-4000,2000]) 

        # fitting grid, can be 'linear' or 'log'; for pure-el fit only 'linear' is used
        self.fit_grid = fit_grid

        # format to save fitting results
        self.fit_raw = fit_raw
        self.num_mock_loops = num_mock_loops
        if self.fit_raw: self.num_mock_loops += 1
        self.best_fits_x = np.zeros((self.num_mock_loops, self.num_tot_xs), dtype='float')
        self.best_coeffs = np.zeros((self.num_mock_loops, self.num_tot_coeffs),dtype='float')
        self.best_chi_sq = np.zeros(self.num_mock_loops, dtype='float')
        self.fit_quality = np.zeros(self.num_mock_loops, dtype='int')
        self.spec_fmock_lw = np.tile(self.spec['flux_w'], (self.num_mock_loops, 1))
        if self.have_phot:
            self.phot_fmock_lb = np.tile(self.phot['flux_b'], (self.num_mock_loops, 1))
        
        # whether to plot and print intermediate results
        self.plot_step = plot_step
        self.print_step = print_step
        self.verbose = verbose 

    def load_models(self, model_config):
        # models init setup
        self.full_model_type = ''
        self.model_dict = {}

        mod = 'ssp'
        if np.isin(mod, [*model_config]):
            if model_config[mod]['enable']: 
                self.full_model_type += mod + '+'
                self.model_dict[mod] = {'cf': ConfigFrame(model_config[mod]['config'])}
                self.model_dict[mod]['specmod'] = SSPModels(filename=model_config[mod]['file'], w_min=self.spec_wmin, w_max=self.spec_wmax, 
                                                            cframe=self.model_dict[mod]['cf'], v0_redshift=self.v0_redshift, spec_R_inst=self.spec_R_inst) 
                if self.have_phot:
                    self.model_dict[mod]['sedmod'] = SSPModels(filename=model_config[mod]['file'], w_min=self.sed_wmin, w_max=self.sed_wmax, 
                                                               cframe=self.model_dict[mod]['cf'], v0_redshift=self.v0_redshift, 
                                                               spec_R_inst=None, spec_R_init=300, verbose=False)
        mod = 'el'
        if np.isin(mod, [*model_config]):
            if model_config[mod]['enable']:             
                self.full_model_type += mod + '+'
                self.model_dict[mod] = {'cf': ConfigFrame(model_config[mod]['config'])}
                self.model_dict[mod]['specmod'] = ELineModels(w_min=self.spec_wmin, w_max=self.spec_wmax, 
                                                              cframe=self.model_dict[mod]['cf'], v0_redshift=self.v0_redshift, spec_R_inst=self.spec_R_inst) 
                self.model_dict[mod]['specfunc'] = self.model_dict[mod]['specmod'].models_unitnorm_obsframe
                if self.have_phot:
                    self.model_dict[mod]['sedmod'] = ELineModels(w_min=self.sed_wmin, w_max=self.sed_wmax, 
                                                                 cframe=self.model_dict[mod]['cf'], v0_redshift=self.v0_redshift, spec_R_inst=self.spec_R_inst, 
                                                                 verbose=False)
        mod = 'agn'
        if np.isin(mod, [*model_config]):
            if model_config[mod]['enable']: 
                self.full_model_type += mod + '+'
                self.model_dict[mod] = {'cf': ConfigFrame(model_config[mod]['config'])}
                self.model_dict[mod]['specmod'] = AGNModels(w_min=self.spec_wmin, w_max=self.spec_wmax, 
                                                            cframe=self.model_dict[mod]['cf'], v0_redshift=self.v0_redshift, spec_R_inst=self.spec_R_inst) 
                if self.have_phot:
                    self.model_dict[mod]['sedmod'] = AGNModels(w_min=self.sed_wmin, w_max=self.sed_wmax, 
                                                               cframe=self.model_dict[mod]['cf'], v0_redshift=self.v0_redshift, spec_R_inst=self.spec_R_inst) 
        mod = 'torus'
        if np.isin(mod, [*model_config]):
            if model_config[mod]['enable']: 
                self.full_model_type += mod + '+'
                self.model_dict[mod] = {'cf': ConfigFrame(model_config[mod]['config'])}
                self.model_dict[mod]['specmod'] = TorusModels(filename=model_config[mod]['file'], 
                                                              cframe=self.model_dict[mod]['cf'], v0_redshift=self.v0_redshift, flux_scale=self.spec_flux_scale) 
                if self.have_phot:
                    self.model_dict[mod]['sedmod'] = self.model_dict[mod]['specmod']                    
        
        if self.full_model_type[-1] == '+': self.full_model_type = self.full_model_type[:-1]
        print('#### Models used in the fitting:', self.full_model_type, '####')
        for mod in self.full_model_type.split('+'):
            self.model_dict[mod]['specfunc'] = self.model_dict[mod]['specmod'].models_unitnorm_obsframe
            if self.have_phot:
                self.model_dict[mod]['sedfunc'] = self.model_dict[mod]['sedmod'].models_unitnorm_obsframe

    def set_parameter_constraints(self):
        self.num_tot_xs = 0
        self.num_tot_coeffs = 0
        self.tie_x = np.array([])
        self.bound_min_x = np.array([])
        self.bound_max_x = np.array([])
        for mod in self.full_model_type.split('+'):
            self.model_dict[mod]['num_xs'] = self.model_dict[mod]['cf'].num_compxpars
            self.model_dict[mod]['num_coeffs'] = self.model_dict[mod]['specmod'].num_coeffs
            self.num_tot_xs += self.model_dict[mod]['num_xs']
            self.num_tot_coeffs += self.model_dict[mod]['num_coeffs']
            self.tie_x = np.hstack((self.tie_x, self.model_dict[mod]['cf'].tie_cp.flatten())) 
            self.bound_min_x = np.hstack((self.bound_min_x, self.model_dict[mod]['cf'].min_cp.flatten())) 
            self.bound_max_x = np.hstack((self.bound_max_x, self.model_dict[mod]['cf'].max_cp.flatten()))

        # update bounds to match requirement of fitting fucntion, but these values will not be indeed used 
        for i_x in range(len(self.tie_x)):
            if self.tie_x[i_x] == 'free': 
                continue
            else:
                if self.tie_x[i_x] == 'fix': 
                    self.bound_max_x[i_x] = self.bound_min_x[i_x] + 1e-8
                else:
                    for single_tie in self.tie_x[i_x].split(';'):
                        ref_mod, ref_comp, ref_i_par = single_tie.split(':')
                        if np.isin(ref_mod, self.full_model_type.split('+')):
                            ref_num_comps = self.model_dict[ref_mod]['cf'].num_comps
                            ref_i_comp = np.where(np.array([self.model_dict[ref_mod]['cf'].comp_c]) == ref_comp)[0]
                            if len(ref_i_comp) == 1:
                                ref_i_comp = ref_i_comp[0]
                            else:
                                raise ValueError((f"The reference component: {ref_comp} is not available in {self.model_dict[ref_mod]['cf'].comp_c}"))
                            ref_i_x = ref_num_comps*ref_i_comp + int(ref_i_par)
                            fx0, fx1 = self.model_index(ref_mod, self.full_model_type)[0:2]
                            if np.isnan(self.bound_min_x[i_x]):
                                self.bound_min_x[i_x] = self.bound_min_x[fx0:fx1][ref_i_x] 
                                self.bound_max_x[i_x] = self.bound_max_x[fx0:fx1][ref_i_x] 
                            else: 
                                self.bound_min_x[i_x] = np.minimum(self.bound_min_x[i_x], self.bound_min_x[fx0:fx1][ref_i_x])
                                self.bound_max_x[i_x] = np.maximum(self.bound_max_x[i_x], self.bound_max_x[fx0:fx1][ref_i_x])
                        else:
                            raise ValueError((f"The reference model {ref_mod} is not provided."))
        self.bound_width_x = self.bound_max_x - self.bound_min_x


    def set_masks(self, eline_waves=None, eline_vwin=None):
        # mask in obs.frame
        spec_wave_w = self.spec['wave_w']
        mask_valid_w = np.zeros_like(spec_wave_w, dtype='bool')
        for i_waveslot in range(len(self.spec_valid_range)):
            waveslot = self.spec_valid_range[i_waveslot]
            mask_valid_w += (spec_wave_w >= waveslot[0]) & (spec_wave_w <= waveslot[1])
        self.spec['mask_valid_w'] = mask_valid_w
        # print('Valid data wavelength range:', self.spec_valid_range)
        
        # eline mask from rest frame line position
        if eline_waves is not None:
            mask_eline_w = np.zeros_like(spec_wave_w, dtype='bool')
            for i_eline in range(len(eline_waves)):
                eline_wave = eline_waves[i_eline] * (1 + self.v0_redshift) * (1+np.array(eline_vwin)/299792.458)
                mask_eline_w += (spec_wave_w >= eline_wave[0]) & (spec_wave_w <= eline_wave[1])
            self.spec['mask_noeline_w'] = mask_valid_w & (~mask_eline_w)
        else:
            self.spec['mask_noeline_w'] = mask_valid_w

    ###############################################################################
    ############################## Fitting Functions ##############################

    def model_index(self, selcomp, model_type, mask_ssp_lite=None, mask_el_lite=None):
        rev_model_type = ''
        for mod in self.full_model_type.split('+'):
            if np.isin(mod, model_type.split('+')): rev_model_type += mod+'+'
        rev_model_type = rev_model_type[:-1] # re-sort the input model_type to fit the order in self.full_model_type
        if rev_model_type.split(selcomp)[0] == rev_model_type: raise ValueError((f"No such model combination: {selcomp} in {rev_model_type}"))

        model_nums = {}
        for mod in self.full_model_type.split('+'):
            model_nums[mod] = {'par': self.model_dict[mod]['num_xs'], 'coeff': self.model_dict[mod]['num_coeffs']}
            if (mod == 'ssp') & (mask_ssp_lite is not None): model_nums[mod]['coeff'] = mask_ssp_lite.sum()
            if (mod == 'el')  & (mask_el_lite  is not None): model_nums[mod]['coeff'] = mask_el_lite.sum()
            
        index_start_x = 0; index_start_coeff = 0
        for precomp in rev_model_type.split(selcomp)[0].split('+'):
            if precomp == '': continue
            index_start_x += model_nums[precomp]['par']
            index_start_coeff += model_nums[precomp]['coeff']
        index_end_x = index_start_x + 0
        index_end_coeff = index_start_coeff + 0
        if len(selcomp.split('+')) == 1:
            index_end_x += model_nums[selcomp]['par']
            index_end_coeff += model_nums[selcomp]['coeff']
        else:
            for singlecomp in selcomp.split('+'):
                index_end_x += model_nums[singlecomp]['par']
                index_end_coeff += model_nums[singlecomp]['coeff']            
        return index_start_x, index_end_x, index_start_coeff, index_end_coeff
    
    def examine_model_SN(self, model_w, noise_w, accept_SN=1.5):
        mask_valid_w = model_w > (np.nanmax(model_w)*0.05) # only consider emission line range with non-zero values
        peak_SN = np.nanpercentile(model_w[mask_valid_w] / noise_w[mask_valid_w], 90)
        return peak_SN, peak_SN >= accept_SN

    def lin_lsq_func(self, flux_w, ferr_w, model_mw, freedom_w, fit_grid='linear', verbose=False):
        # solve linear least-square functions to obtain the normlization values (i.e., coeffs) of each models
        
        weight_w = np.divide(1, ferr_w, where=ferr_w>0, out=np.zeros_like(ferr_w))
        n_models = model_mw.shape[0]
        solution = lsq_linear(model_mw.T * weight_w[:,None], flux_w * weight_w, 
                              bounds=np.array([(0.,np.inf) for i in range(n_models)]).T, 
                              verbose=verbose) # max_iter=200, lsmr_tol='auto', tol=1e-12, 
        coeff_m = solution.x
        ret_model_w = np.dot(coeff_m, model_mw)
        if fit_grid == 'linear': chi_w = np.divide(ret_model_w-flux_w, ferr_w, where=ferr_w>0, out=np.zeros_like(ferr_w))
        if fit_grid == 'log':    chi_w = np.divide(np.log(ret_model_w/flux_w)*flux_w, ferr_w, where=ferr_w>0, out=np.zeros_like(ferr_w))
        # linear: (model/flux-1)*flux/ferr, log: ln(model/flux)*flux/ferr
        n_free = np.sum(freedom_w[ferr_w>0]) - (n_models + 1)
        chi_w *= np.sqrt(freedom_w/np.maximum(1, n_free)) # reduced
        chi_sq = np.sum(chi_w**2)
        return coeff_m, chi_sq, ret_model_w, chi_w*np.sqrt(2)
        # select chi_w*np.sqrt(2) to let the cost function (0.5*ret**2) to return reduced chi_sq

    def residual_func(self, x, wave_w, flux_w, ferr_w,
                      model_type, mask_ssp_lite=None, mask_el_lite=None, 
                      fit_phot=False, fit_grid='linear', ret_coeffs=False):
        # for a give set of parameters, return models and residuals
        # the residuals are used to solve non-linear least-square fit
        
        rev_model_type = ''
        for mod in self.full_model_type.split('+'):
            if np.isin(mod, model_type.split('+')): rev_model_type += mod+'+'
        rev_model_type = rev_model_type[:-1] # re-sort the input model_type to fit the order in self.full_model_type
        
        if fit_phot: # spec+band
            spec_wave_w = wave_w[:-self.num_phot_band]
            sed_wave_w = self.sed['wave_w']
            rev_ferr_w = ferr_w + 0.1*flux_w # add 10% flux to ferr to account for cross-instrumental errors
            freedom_w = np.ones(len(wave_w))
            freedom_w[:self.num_spec_wave] = (spec_wave_w[1]-spec_wave_w[0])/(spec_wave_w.mean()/self.spec_R_inst)
            # account for effective spectral sampling in fitting (all bands are considered as independent)
            freedom_w[freedom_w > 1] = 1
        else: # only spec 
            spec_wave_w = wave_w
            rev_ferr_w = ferr_w
            freedom_w = np.ones(len(wave_w))

        # tie parameters following setup in input _config 
        bmin = np.array([])
        ties = np.array([])
        for mod in rev_model_type.split('+'):
            bmin = np.hstack((bmin, self.model_dict[mod]['cf'].min_cp.flatten()))
            ties = np.hstack((ties, self.model_dict[mod]['cf'].tie_cp.flatten()))
        for i_x in range(len(x)):
            if ties[i_x] == 'free': 
                continue
            else:
                if ties[i_x] == 'fix': 
                    x[i_x] = bmin[i_x]
                else:
                    for single_tie in ties[i_x].split(';'):
                        ref_mod, ref_comp, ref_i_par = single_tie.split(':')
                        if np.isin(ref_mod, rev_model_type.split('+')):
                            ref_num_comps = self.model_dict[ref_mod]['cf'].num_comps
                            ref_i_comp = np.where(np.array([self.model_dict[ref_mod]['cf'].comp_c]) == ref_comp)[0]
                            if len(ref_i_comp) == 1:
                                ref_i_comp = ref_i_comp[0]
                            else:
                                raise ValueError((f"The reference component: {ref_comp} is not available in {self.model_dict[ref_mod]['cf'].comp_c}"))
                            ref_i_x = ref_num_comps*ref_i_comp + int(ref_i_par)
                            mx0, mx1 = self.model_index(ref_mod, rev_model_type)[0:2]
                            x[i_x] = x[mx0:mx1][ref_i_x]                   
                            break
        
        fit_model_mw = None
        for mod in rev_model_type.split('+'):
            mx0, mx1 = self.model_index(mod, rev_model_type)[0:2]
            spec_fmod_mw = self.model_dict[mod]['specfunc'](spec_wave_w, x[mx0:mx1])
            if fit_phot:
                sed_fmod_mw = self.model_dict[mod]['sedfunc'](sed_wave_w, x[mx0:mx1])
                sed_fmod_mb = self.pframe.spec2phot(sed_wave_w, sed_fmod_mw, self.phot['trans_bw'])
                spec_fmod_mw = np.hstack((spec_fmod_mw, sed_fmod_mb))
            if (mod == 'ssp') & (mask_ssp_lite is not None): spec_fmod_mw = spec_fmod_mw[mask_ssp_lite, :]
            if (mod == 'el')  & (mask_el_lite  is not None): spec_fmod_mw = spec_fmod_mw[mask_el_lite, :]
            fit_model_mw = spec_fmod_mw if (fit_model_mw is None) else np.vstack((fit_model_mw, spec_fmod_mw))

        coeff_m, chi_sq, model_w, chi_w = self.lin_lsq_func(flux_w, rev_ferr_w, fit_model_mw, freedom_w, fit_grid, verbose=self.verbose)
        if np.sum(coeff_m <0) + np.sum(fit_model_mw.sum(axis=1) < 0) > 0: 
            raise ValueError((f"Negative model coeff: {np.where(coeff_m <0), np.where(fit_model_mw.sum(axis=1) < 0)}"))
        # coeff_m[fit_model_mw.sum(axis=1) <= 0] = 0 # to remove emission lines not well covered
        
        if ret_coeffs:
            return coeff_m, chi_sq, model_w
        else:
            # for least_squares fitting, compute the residual weighted by the flux standard deviation, i.e., chi
            return chi_w
        
    def nl_lsq_func(self, x0, wave_w, flux_w, ferr_w, 
                    model_type, mask_ssp_lite=None, mask_el_lite=None, fit_phot=False, fit_grid='linear', 
                    refit_rand_x0=True, max_fit_ntry=3, accept_chi_sq=5, verbose=False, plot_title=None): 
        # core fitting function to obtain solution of non-linear least-square problems
        
        bound_min_x, bound_max_x, bound_width_x = self.bound_min_x.copy(), self.bound_max_x.copy(), self.bound_width_x.copy()
        mask_x = np.zeros_like(x0, dtype='bool') 
        for mod in model_type.split('+'):
            fx0, fx1 = self.model_index(mod, self.full_model_type)[0:2]
            mask_x[fx0:fx1] = True
            
        fit_success, fit_ntry, chi_sq, tmp_chi_sq = False, 0, 1e4, 1e4
        while (fit_success == False) & (fit_ntry < max_fit_ntry): 
            try:
                best_fit = least_squares(fun=self.residual_func, 
                                         args=(wave_w, flux_w, ferr_w, model_type, mask_ssp_lite, mask_el_lite, fit_phot, fit_grid),
                                         x0=x0[mask_x], bounds=(bound_min_x[mask_x], bound_max_x[mask_x]), 
                                         diff_step=3, # real x_step is x * diff_step 
                                         x_scale='jac', jac='3-point', ftol=1e-4, max_nfev=10000, 
                                         verbose=verbose) # ftol=0.5*len(ferr_w)*np.nanpercentile(ferr_w,10)**2,
                fit_ntry += 1
            except Exception as ex: 
                # if self.verbose: 
                print('Exception:', ex); traceback.print_exc()
            else:
                coeff_m, chi_sq, model_w = self.residual_func(best_fit.x, wave_w, flux_w, ferr_w, 
                                                              model_type, mask_ssp_lite, mask_el_lite, fit_phot, fit_grid, ret_coeffs=True)
                if chi_sq < accept_chi_sq: 
                    fit_success = best_fit.success # i.e., accept this fit 
                else:
                    if chi_sq < tmp_chi_sq: 
                        tmp_best_fit = copy(best_fit); tmp_chi_sq = copy(chi_sq) # save for available min chi_sq
                    if self.print_step:
                        print(f'fit_ntry={fit_ntry}, '+
                              f'poor fit with chi_sq={chi_sq:.3f} > {accept_chi_sq:.3f} (accepted); '+
                              f'available min chi_sq={tmp_chi_sq:.3f}')
                    if refit_rand_x0: # re-generate random tmp_x0 for refit
                        tmp_x0 = bound_min_x[mask_x] + np.random.rand(mask_x.sum()) * bound_width_x[mask_x]
                    else: # slightly shift tmp_x0 for refit
                        tmp_x0 = x0[mask_x] + np.random.randn(mask_x.sum()) * bound_width_x[mask_x]*0.01 # 1% scaled 
                        tmp_x0 = np.maximum(tmp_x0, bound_min_x[mask_x])
                        tmp_x0 = np.minimum(tmp_x0, bound_max_x[mask_x])
                    x0[mask_x] = tmp_x0 # fill into input x0
        if (fit_success == False): 
            best_fit = tmp_best_fit # back to fit with available min chi_sq
            coeff_m, chi_sq, model_w = self.residual_func(best_fit.x, wave_w, flux_w, ferr_w, 
                                                          model_type, mask_ssp_lite, mask_el_lite, fit_phot, fit_grid, ret_coeffs=True)
        if self.print_step: 
            print(f'fit_ntry={fit_ntry}, chi_sq={chi_sq:.3f}')
            self.time_last = print_time(plot_title, self.time_last, self.time_init)
        if self.plot_step:
            plt.figure()
            if fit_phot:
                plt.plot(wave_w[:-self.num_phot_band], flux_w[:-self.num_phot_band], c='C0', label='Data')
                plt.plot(wave_w[:-self.num_phot_band], model_w[:-self.num_phot_band], c='C1', label='Best-fit')
                plt.plot(wave_w[:-self.num_phot_band], (flux_w-model_w)[:-self.num_phot_band], c='C2', label='Resoduals')
                plt.plot(wave_w[:-self.num_phot_band], (ferr_w+0.1*flux_w)[:-self.num_phot_band], c='C7', label='1$\sigma$ error')
                plt.plot(wave_w[:-self.num_phot_band], -(ferr_w+0.1*flux_w)[:-self.num_phot_band], c='C7')
                ind_b = np.argsort(wave_w[-self.num_phot_band:])
                plt.plot(wave_w[-self.num_phot_band:][ind_b], flux_w[-self.num_phot_band:][ind_b], '--o', c='C0')
                plt.plot(wave_w[-self.num_phot_band:][ind_b], model_w[-self.num_phot_band:][ind_b], '--o', c='C1')
                plt.plot(wave_w[-self.num_phot_band:][ind_b], (flux_w-model_w)[-self.num_phot_band:][ind_b], '--o', c='C2')
                plt.plot(wave_w[-self.num_phot_band:][ind_b], (ferr_w+0.1*flux_w)[-self.num_phot_band:][ind_b], '--o', c='C7')
                plt.plot(wave_w[-self.num_phot_band:][ind_b], -(ferr_w+0.1*flux_w)[-self.num_phot_band:][ind_b], '--o', c='C7')
                plt.xscale('log')
            else:
                plt.plot(wave_w, flux_w, c='C0', label='Data')
                plt.plot(wave_w, model_w, c='C1', label='Best-fit')
                plt.plot(wave_w, flux_w-model_w, c='C2', label='Resoduals')
                plt.plot(wave_w, ferr_w, c='C7'); plt.plot(wave_w, -ferr_w, c='C7', label='1$\sigma$ error')
            plt.legend()
            plt.xlabel('Wavelength ($\AA$)')
            plt.ylabel('Flux ('+str(self.spec_flux_scale)+' $erg/s/cm2/\AA$)')
            plt.title(plot_title)
        return best_fit, coeff_m, chi_sq

    def main_fit(self):
        spec_wave_w, spec_flux_w, spec_ferr_w = self.spec['wave_w'], self.spec['flux_w'], self.spec['ferr_w']
        mask_valid_w, mask_noeline_w = self.spec['mask_valid_w'], self.spec['mask_noeline_w']
        if self.have_phot:
            phot_wave_b, phot_flux_b, phot_ferr_b = self.phot['wave_b'], self.phot['flux_b'], self.phot['ferr_b']
            sed_wave_w = self.sed['wave_w']
        
        nl_lsq_func = self.nl_lsq_func
        examine_model_SN = self.examine_model_SN
        bound_min_x, bound_max_x, bound_width_x = self.bound_min_x, self.bound_max_x, self.bound_width_x
        ssp_mod = self.model_dict['ssp']['specmod']
        el_mod = self.model_dict['el']['specmod']
        # ssp and el are always enabled
        
        n_loops = self.num_mock_loops
        success_count, total_count = 0, 0
        while success_count < n_loops:
            i_loop_now = np.where(self.fit_quality == 0)[0][0] 
            # i_loop_now (to save results) is the 1st loop index of non-good-fits
            print(f'#### loop {i_loop_now}/{n_loops} start: ####')
            self.time_init = time.time(); self.time_last = self.time_init
            
            # use the raw flux for the 1st loop if fit_raw is True
            # otherwise randomly draw a mocked spectrum assuming a Gaussian distribution of the errors
            if self.fit_raw & (i_loop_now == 0): 
                print('#### fit the raw spectrum (non-mocked) ####')
                spec_fmock_w = spec_flux_w.copy()
                if self.have_phot: phot_fmock_b = phot_flux_b.copy()
            else:
                print('#### fit the mocked spectrum ####')
                spec_fmock_w = spec_flux_w + np.random.randn(spec_flux_w.shape[0]) * spec_ferr_w
                self.spec_fmock_lw[i_loop_now] = spec_fmock_w # save for output
                if self.have_phot: 
                    phot_fmock_b = phot_flux_b + np.random.randn(phot_flux_b.shape[0]) * phot_ferr_b
                    self.phot_fmock_lb[i_loop_now] = phot_fmock_b
            
            # create random initial parameters
            x0mock = bound_min_x + np.random.rand(self.num_tot_xs) * bound_width_x
            
            ##############################
            # # for test
            # model_type = 'ssp'
            # mask_ssp_lite = ssp_mod.mask_ssp_lite_with_num_mods(num_ages_lite=8, num_mets_lite=1)
            # cont_fit, cont_coeff_m, cont_chi_sq = nl_lsq_func(x0mock, spec_wave_w[mask_noeline_w], 
            #                                                   spec_fmock_w[mask_noeline_w], spec_ferr_w[mask_noeline_w], 
            #                                                   model_type, mask_ssp_lite, 
            #                                                   plot_title='Spec Fit, init continua (cont_fit_init)')
            # # model_type = 'ssp+torus'
            # # cont_fit, cont_coeff_m, cont_chi_sq = nl_lsq_func(x0mock, 
            # #                                                np.hstack((spec_wave_w[mask_noeline_w], phot_wave_b)), 
            # #                                                np.hstack((spec_fmock_w[mask_noeline_w], phot_fmock_b)), 
            # #                                                np.hstack((spec_ferr_w[mask_noeline_w], phot_ferr_b)), 
            # #                                                model_type, mask_ssp_lite, fit_phot=True, verbose=2)
            # print(cont_fit, cont_coeff_m, cont_chi_sq)
            # self.output_ssp()
            # break
            ##############################

            ########################################
            ############# init fit cycle ###########
            cont_type = ''
            for mod in self.full_model_type.split('+'):
                if mod == 'el': continue
                if (mod == 'torus') & (spec_wave_w.max()/(1+self.v0_redshift) < 1e4): continue
                cont_type += mod + '+'
            cont_type = cont_type[:-1]; 
            if self.print_step: print('Continuum models:', cont_type)
            model_type = cont_type+'' # copy
            # obtain a rough fit of continuum with emission line ranges masked out
            mask_ssp_lite = ssp_mod.mask_ssp_lite_with_num_mods(num_ages_lite=8, num_mets_lite=1, verbose=self.print_step)
            cont_fit, cont_coeff_m, cont_chi_sq = nl_lsq_func(x0mock, spec_wave_w[mask_noeline_w], 
                                                              spec_fmock_w[mask_noeline_w], spec_ferr_w[mask_noeline_w], 
                                                              model_type, mask_ssp_lite, fit_grid=self.fit_grid,
                                                              plot_title='Spec Fit, init continua (cont_fit_init)')
            cont_fmod_w = spec_wave_w * 0
            for mod in model_type.split('+'):
                mx0, mx1, mc0, mc1 = self.model_index(mod, model_type, mask_ssp_lite)
                spec_fmod_mw = self.model_dict[mod]['specfunc'](spec_wave_w, cont_fit.x[mx0:mx1])
                if (mod == 'ssp') & (mask_ssp_lite is not None): spec_fmod_mw = spec_fmod_mw[mask_ssp_lite, :]
                cont_fmod_w += np.dot(cont_coeff_m[mc0:mc1], spec_fmod_mw) 
            ########################################
            model_type = 'el'
            # obtain a rough fit of emission lines with continuum of ssp_fit_init subtracted
            el_fit, el_coeff_m, el_chi_sq = nl_lsq_func(x0mock, spec_wave_w[mask_valid_w], 
                                                        (spec_fmock_w - cont_fmod_w)[mask_valid_w], spec_ferr_w[mask_valid_w], model_type,
                                                        plot_title='Spec Fit, init emission lines (el_fit_init)')
            el_fmod_w = np.dot(el_coeff_m, el_mod.models_unitnorm_obsframe(spec_wave_w, el_fit.x))
            ########################################
            ########################################
            
            ########################################
            ############## 1st fit cycle ###########
            model_type = cont_type+''
            # obtain a better fit of stellar continuum with emission lines of el_fit_init subtracted
            mask_ssp_lite = ssp_mod.mask_ssp_lite_with_num_mods(num_ages_lite=16, num_mets_lite=1, verbose=self.print_step)
            cont_fit, cont_coeff_m, cont_chi_sq = nl_lsq_func(x0mock, spec_wave_w[mask_valid_w], 
                                                              (spec_fmock_w - el_fmod_w)[mask_valid_w], spec_ferr_w[mask_valid_w], 
                                                              model_type, mask_ssp_lite, fit_grid=self.fit_grid,
                                                              plot_title='Spec Fit, update continua (cont_fit_1)')
            cont_fmod_w = spec_wave_w * 0
            for mod in model_type.split('+'):
                mx0, mx1, mc0, mc1 = self.model_index(mod, model_type, mask_ssp_lite)
                spec_fmod_mw = self.model_dict[mod]['specfunc'](spec_wave_w, cont_fit.x[mx0:mx1])
                if (mod == 'ssp') & (mask_ssp_lite is not None): spec_fmod_mw = spec_fmod_mw[mask_ssp_lite, :]
                cont_fmod_w += np.dot(cont_coeff_m[mc0:mc1], spec_fmod_mw) 
                fx0, fx1 = self.model_index(mod, self.full_model_type)[0:2]
                x0mock[fx0:fx1] = cont_fit.x[mx0:mx1] # save the best fit pars for later step 
            ########################################
            model_type = 'el'
            # obtain a better fit of emission lines with continuum of ssp_fit_1 subtracted
            el_fit, el_coeff_m, el_chi_sq = nl_lsq_func(x0mock, spec_wave_w[mask_valid_w], 
                                                        (spec_fmock_w - cont_fmod_w)[mask_valid_w], spec_ferr_w[mask_valid_w], model_type,
                                                        plot_title='Spec Fit, update emission lines (el_fit_1)')
            fx0, fx1 = self.model_index('el', self.full_model_type)[0:2]
            x0mock[fx0:fx1] = el_fit.x # save the best fit pars for later step  
            ########################################
            model_type = cont_type+'+el'
            # joint fit of continuum and emission lines with initial values from best-fit of ssp_fit_1 and el_fit_1
            joint_fit, joint_coeff_m, joint_chi_sq = nl_lsq_func(x0mock, spec_wave_w[mask_valid_w], 
                                                                 spec_fmock_w[mask_valid_w], spec_ferr_w[mask_valid_w], 
                                                                 model_type, mask_ssp_lite, fit_grid=self.fit_grid,
                                                                 refit_rand_x0=False, accept_chi_sq=np.maximum(cont_chi_sq, el_chi_sq),
                                                                 plot_title='Spec Fit, all models (joint_fit_1)') 
            for mod in model_type.split('+'):
                fx0, fx1 = self.model_index(mod, self.full_model_type)[0:2]
                mx0, mx1 = self.model_index(mod, model_type)[0:2]
                x0mock[fx0:fx1] = joint_fit.x[mx0:mx1] # save the best fit pars for later step  
            ########################################
            ########################################
            
            ########################################
            ########### Examine models #############
            # examine whether each continuum component is indeed required
            cont_type = '' # reset
            for mod in model_type.split('+'):
                if mod == 'el': continue
                mx0, mx1, mc0, mc1 = self.model_index(mod, model_type, mask_ssp_lite)
                spec_fmod_mw = self.model_dict[mod]['specfunc'](spec_wave_w, joint_fit.x[mx0:mx1])
                if (mod == 'ssp') & (mask_ssp_lite is not None): spec_fmod_mw = spec_fmod_mw[mask_ssp_lite, :]
                spec_fmod_w = np.dot(joint_coeff_m[mc0:mc1], spec_fmod_mw) 
                mod_peak_SN, mod_examine = examine_model_SN(spec_fmod_w[mask_valid_w], spec_ferr_w[mask_valid_w], accept_SN=2)
                if mod_examine: cont_type += mod + '+'
                if self.print_step: print(f'{mod} continuum peak SN={mod_peak_SN},', 'enabled' if mod_examine else 'disabled')
            if cont_type[-1] == '+': cont_type = cont_type[:-1]
            if cont_type == '':
                cont_type = 'ssp'
                if self.print_step: print(f'#### faint continuum, still enable stellar continuum ####')
            if self.print_step: print('#### continuum models after examination:', cont_type, '####')         
            ########################################
            # examine whether each emission line component is indeed required
            el_comps = [] # reset; ['NLR','outflow_2']
            for i_comp in range(el_mod.num_comps):
                comp = el_mod.cframe.info_c[i_comp]['comp_name']
                mask_el_lite = el_mod.mask_el_lite(enabled_comps=[comp])
                mx0, mx1, mc0, mc1 = self.model_index('el', model_type, mask_ssp_lite)
                el_fmod_w = np.dot(joint_coeff_m[mc0:mc1][mask_el_lite], el_mod.models_unitnorm_obsframe(spec_wave_w, 
                                   joint_fit.x[mx0:mx1])[mask_el_lite,:])
                el_peak_SN, el_examine = examine_model_SN(el_fmod_w[mask_valid_w], spec_ferr_w[mask_valid_w], accept_SN=2)
                if el_examine: el_comps.append(comp)
                if self.print_step: print(f'{comp} pean SN={el_peak_SN},', 'enabled' if el_examine else 'disabled')
            if len(el_comps) == 0:
                el_comps = ['NLR']
                if self.print_step: print(f'#### faint emission lines, still enable NLR ####')
            if self.print_step: print('#### emission line components after examination:', el_comps, '####')
            mask_el_lite = el_mod.mask_el_lite(enabled_comps=el_comps) # only keep enabled line components
            ########################################
            ########################################
            
            ########################################
            ############# 2nd fit cycle ############
            mx0, mx1, mc0, mc1 = self.model_index('el', model_type, mask_ssp_lite)
            el_fmod_w = np.dot(joint_coeff_m[mc0:mc1], el_mod.models_unitnorm_obsframe(spec_wave_w, joint_fit.x[mx0:mx1]))            
            ########################################
            model_type = cont_type+''
            # repeat use initial best-fit values and subtract emission lines from joint_fit_1 (update with mask_el_lite later)
            cont_fit, cont_coeff_m, cont_chi_sq = nl_lsq_func(x0mock, spec_wave_w[mask_valid_w], 
                                                             (spec_fmock_w - el_fmod_w)[mask_valid_w], spec_ferr_w[mask_valid_w], 
                                                              model_type, mask_ssp_lite, fit_grid=self.fit_grid,
                                                              plot_title='Spec Fit, update continua (cont_fit_2.1)') 
            for mod in model_type.split('+'):
                fx0, fx1 = self.model_index(mod, self.full_model_type)[0:2]
                mx0, mx1 = self.model_index(mod, model_type)[0:2]
                x0mock[fx0:fx1] = cont_fit.x[mx0:mx1] # save the best fit pars for later step  
            ########################################
            model_type = cont_type+''
            # in steps above, ssp models in a sparse grid of ages (and metalicities) are used, now update continuum fit with all allowed ssp models
            mask_ssp_lite = ssp_mod.mask_ssp_allowed(csp=(ssp_mod.sfh_names[0] != 'nonparametric'))
            # use initial best-fit values from cont_fit_2.1 and subtract emission lines from joint_fit_1
            cont_fit, cont_coeff_m, cont_chi_sq = nl_lsq_func(x0mock, spec_wave_w[mask_valid_w], 
                                                             (spec_fmock_w - el_fmod_w)[mask_valid_w], spec_ferr_w[mask_valid_w], 
                                                              model_type, mask_ssp_lite, fit_grid=self.fit_grid,
                                                              plot_title='Spec Fit, update continua (cont_fit_2.2)')
            cont_fmod_w = spec_wave_w * 0
            for mod in model_type.split('+'):
                mx0, mx1, mc0, mc1 = self.model_index(mod, model_type, mask_ssp_lite)
                spec_fmod_mw = self.model_dict[mod]['specfunc'](spec_wave_w, cont_fit.x[mx0:mx1])
                if (mod == 'ssp') & (mask_ssp_lite is not None): spec_fmod_mw = spec_fmod_mw[mask_ssp_lite, :]
                cont_fmod_w += np.dot(cont_coeff_m[mc0:mc1], spec_fmod_mw) 
                fx0, fx1 = self.model_index(mod, self.full_model_type)[0:2]
                x0mock[fx0:fx1] = cont_fit.x[mx0:mx1] # save the best fit pars for later step 
            # (do not move up mask_ssp_lite updating) create new mask_ssp_lite with new ssp_coeffs and 
            # the weight of integrated flux in the fitting wavelength range (ssp_coeffs itself is weight at 5500A);
            # do not use full allowed ssp models to save time
            if np.isin('ssp', model_type.split('+')): 
                mx0, mx1, mc0, mc1 = self.model_index('ssp', model_type, mask_ssp_lite=mask_ssp_lite)
                ssp_coeff_m = cont_coeff_m[mc0:mc1]
                ssp_coeff_m *= ssp_mod.models_unitnorm_obsframe(spec_wave_w[mask_valid_w], cont_fit.x[mx0:mx1]).sum(axis=1)[mask_ssp_lite]
                mask_ssp_lite = ssp_mod.mask_ssp_lite_with_coeffs(ssp_coeff_m, num_mods_min=24, verbose=self.print_step)
            ########################################
            model_type = 'el'
            # update emission line with mask_el_lite
            # use initial values from best-fit of joint_fit_1 and subtract continuum from cont_fit_2 
            el_fit, el_coeff_m, el_chi_sq = nl_lsq_func(x0mock, spec_wave_w[mask_valid_w], 
                                                       (spec_fmock_w - cont_fmod_w)[mask_valid_w], spec_ferr_w[mask_valid_w], 
                                                        model_type, mask_el_lite=mask_el_lite,
                                                        plot_title='Spec Fit, update emission lines (el_fit_2)')
            fx0, fx1 = self.model_index('el', self.full_model_type)[0:2]
            x0mock[fx0:fx1] = el_fit.x # save the best fit pars for later step 
            ########################################
            model_type = cont_type+'+el'
            # joint fit of continuum and emission lines with initial values from best-fit of cont_fit_2 and el_fit_2
            joint_fit, joint_coeff_m, joint_chi_sq = nl_lsq_func(x0mock, spec_wave_w[mask_valid_w], 
                                                                 spec_fmock_w[mask_valid_w], spec_ferr_w[mask_valid_w], 
                                                                 model_type, mask_ssp_lite, mask_el_lite, fit_grid=self.fit_grid, 
                                                                 refit_rand_x0=False, accept_chi_sq=np.maximum(cont_chi_sq, el_chi_sq),
                                                                 plot_title='Spec Fit, all models (joint_fit_2)')
            for mod in model_type.split('+'):
                fx0, fx1 = self.model_index(mod, self.full_model_type)[0:2]
                mx0, mx1 = self.model_index(mod, model_type)[0:2]
                x0mock[fx0:fx1] = joint_fit.x[mx0:mx1] # save the best fit pars for later step  
            ########################################
            ########################################
            
            ########################################
            ############ 3rd fit cycle #############
            if self.have_phot:
                if self.print_step: print('#### Perform Simultaneous Spec+SED Fit ####')
                
                mx0, mx1, mc0, mc1 = self.model_index('el', model_type, mask_ssp_lite, mask_el_lite)
                el_fmod_w = np.dot(joint_coeff_m[mc0:mc1], el_mod.models_unitnorm_obsframe(spec_wave_w, joint_fit.x[mx0:mx1]))            
                el_fsed_w = np.dot(joint_coeff_m[mc0:mc1], el_mod.models_unitnorm_obsframe(sed_wave_w,  joint_fit.x[mx0:mx1]))
                el_fsed_b = self.pframe.spec2phot(sed_wave_w, el_fsed_w, self.phot['trans_bw'])    
                ########################################
                if np.isin('torus', self.full_model_type.split('+')): 
                    if sed_wave_w.max()/(1+self.v0_redshift) > 1e4: cont_type += '+torus'
                if self.print_step: print('#### Continuum models used in Spec+SED fit:', cont_type, '####')
                model_type = cont_type+''
                # spec+sed cont_fit ; use initial best-fit values; subtract emission lines from joint_fit_2
                cont_fit, cont_coeff_m, cont_chi_sq = nl_lsq_func(x0mock, 
                                                                  np.hstack((spec_wave_w[mask_valid_w], phot_wave_b)),
                                                                  np.hstack(((spec_fmock_w - el_fmod_w)[mask_valid_w], phot_fmock_b-el_fsed_b)),
                                                                  np.hstack((spec_ferr_w[mask_valid_w], phot_ferr_b)),
                                                                  model_type, mask_ssp_lite, fit_phot=True, fit_grid=self.fit_grid,
                                                                  plot_title='Spec+SED Fit, update continua (cont_fit_3.1)') 
                for mod in model_type.split('+'):
                    fx0, fx1 = self.model_index(mod, self.full_model_type)[0:2]
                    mx0, mx1 = self.model_index(mod, model_type)[0:2]
                    x0mock[fx0:fx1] = cont_fit.x[mx0:mx1] # save the best fit pars for later step  
                ########################################
                model_type = cont_type+''
                # update mask_ssp_lite for spec+sed cont_fit
                mask_ssp_lite = ssp_mod.mask_ssp_allowed(csp=(ssp_mod.sfh_names[0] != 'nonparametric'))
                # use initial best-fit values from cont_fit_3.1 and subtract emission lines from joint_fit_2
                cont_fit, cont_coeff_m, cont_chi_sq = nl_lsq_func(x0mock, 
                                                                  np.hstack((spec_wave_w[mask_valid_w], phot_wave_b)),
                                                                  np.hstack(((spec_fmock_w - el_fmod_w)[mask_valid_w], phot_fmock_b-el_fsed_b)),
                                                                  np.hstack((spec_ferr_w[mask_valid_w], phot_ferr_b)),
                                                                  model_type, mask_ssp_lite, fit_phot=True, fit_grid=self.fit_grid,
                                                                  plot_title='Spec+SED Fit, update continua (cont_fit_3.2)')
                cont_fmod_w = spec_wave_w * 0
                for mod in model_type.split('+'):
                    mx0, mx1, mc0, mc1 = self.model_index(mod, model_type, mask_ssp_lite)
                    spec_fmod_mw = self.model_dict[mod]['specfunc'](spec_wave_w, cont_fit.x[mx0:mx1])
                    if (mod == 'ssp') & (mask_ssp_lite is not None): spec_fmod_mw = spec_fmod_mw[mask_ssp_lite, :]
                    cont_fmod_w += np.dot(cont_coeff_m[mc0:mc1], spec_fmod_mw) 
                    fx0, fx1 = self.model_index(mod, self.full_model_type)[0:2]
                    x0mock[fx0:fx1] = cont_fit.x[mx0:mx1] # save the best fit pars for later step 
                # (do not move up mask_ssp_lite updating) create new mask_ssp_lite with new ssp_coeffs and 
                # the weight of integrated flux in the fitting wavelength range (ssp_coeffs itself is weight at 5500A);
                # do not use full allowed ssp models to save time
                if np.isin('ssp', model_type.split('+')): 
                    mx0, mx1, mc0, mc1 = self.model_index('ssp', model_type, mask_ssp_lite=mask_ssp_lite)
                    ssp_coeff_m = cont_coeff_m[mc0:mc1]
                    ssp_coeff_m *= ssp_mod.models_unitnorm_obsframe(spec_wave_w[mask_valid_w], cont_fit.x[mx0:mx1]).sum(axis=1)[mask_ssp_lite]
                    mask_ssp_lite = ssp_mod.mask_ssp_lite_with_coeffs(ssp_coeff_m, num_mods_min=24, verbose=self.print_step) # 12
                ########################################
                model_type = 'el'
                # update emission line, use initial values from best-fit of joint_fit_1 and subtract continuum from cont_fit_3.2
                el_fit, el_coeff_m, el_chi_sq = nl_lsq_func(x0mock, spec_wave_w[mask_valid_w], 
                                                           (spec_fmock_w - cont_fmod_w)[mask_valid_w], spec_ferr_w[mask_valid_w], 
                                                            model_type, mask_el_lite=mask_el_lite,
                                                            plot_title='Spec+SED Fit, update emission lines (el_fit_3)')
                fx0, fx1 = self.model_index('el', self.full_model_type)[0:2]
                x0mock[fx0:fx1] = el_fit.x # save the best fit pars for later step 
                ########################################
                model_type = cont_type+'+el'
                # joint fit of continuum and emission lines with initial values from best-fit of cont_fit_2 and el_fit_2            
                joint_fit, joint_coeff_m, joint_chi_sq = nl_lsq_func(x0mock, 
                                                                     np.hstack((spec_wave_w[mask_valid_w], phot_wave_b)), 
                                                                     np.hstack((spec_fmock_w[mask_valid_w], phot_fmock_b)), 
                                                                     np.hstack((spec_ferr_w[mask_valid_w], phot_ferr_b)), 
                                                                     model_type, mask_ssp_lite, mask_el_lite, fit_phot=True, fit_grid=self.fit_grid,
                                                                     refit_rand_x0=False, accept_chi_sq=cont_chi_sq,
                                                                     plot_title='Spec+SED Fit, use all models (joint_fit_3)')
                # here set accept_chi_sq=cont_chi_sq, since el_chi_sq is much larger due to non-enlarged error in el_fit_3
            ########################################
            ########################################
            
            if joint_fit.success: 
                success_count += 1; total_count += 1
                self.fit_quality[i_loop_now] = 1 # set as good fits temporarily
                # record _x, _coeffs and _chi_sq in this loop
                mask_lite_dict = {'ssp':mask_ssp_lite, 
                                  'agn':np.ones((1), dtype='bool'), # temp
                                  'el':mask_el_lite, 
                                  'torus':np.ones((1), dtype='bool') } # temp
                mask_lite_list = []
                for mod in model_type.split('+'): mask_lite_list.append(mask_lite_dict[mod])                
                for (mod, mask_lite) in zip(model_type.split('+'), mask_lite_list):
                    fx0, fx1, fc0, fc1 = self.model_index(mod, self.full_model_type, mask_ssp_lite=None, mask_el_lite=None)
                    mx0, mx1, mc0, mc1 = self.model_index(mod, model_type, mask_ssp_lite=mask_ssp_lite, mask_el_lite=mask_el_lite)
                    self.best_fits_x[i_loop_now, fx0:fx1] = joint_fit.x[mx0:mx1]
                    self.best_coeffs[i_loop_now, fc0:fc1][mask_lite] = joint_coeff_m[mc0:mc1]
                self.best_chi_sq[i_loop_now] = joint_chi_sq
                
            # check fitting quality after all loops finished
            # allow additional loops to remove outlier fit; exit if additional loops > 3
            if (success_count == n_loops) & (total_count <= (n_loops+3)):
                if self.fit_raw & (self.best_chi_sq.shape[0] > 1):
                    accept_chi_sq = self.best_chi_sq[1:].min() * 1.5
                else:
                    accept_chi_sq = self.best_chi_sq.min() * 1.5
                mask_good_fits = self.best_chi_sq < accept_chi_sq
                self.fit_quality[~mask_good_fits] = 0 # set bad fits to refit
                success_count = np.sum(mask_good_fits) # reduce counts to start refit
                print(str(success_count)+' loops have good fit, chi_sq:', self.best_chi_sq[mask_good_fits])
                print(str(n_loops-success_count)+' loops need refit, chi_sq:', self.best_chi_sq[~mask_good_fits])
                # mask out bad fit results
                self.best_fits_x[~mask_good_fits,:] = 0
                self.best_coeffs[~mask_good_fits,:] = 0
                self.best_chi_sq[~mask_good_fits]   = 0
            print(f'#### loop {i_loop_now}/{n_loops} end, {time.time()-self.time_init:.1f} s ####')
        print(f'######## {success_count} successful loops in total {total_count} loops ########')

        # create outputs
        self.output_spec()
        if np.isin('ssp', self.full_model_type.split('+')): self.output_ssp()
        if np.isin('el', self.full_model_type.split('+')): self.output_el()
        if np.isin('agn', self.full_model_type.split('+')): self.output_agn()
        if np.isin('torus', self.full_model_type.split('+')): self.output_torus()

        print('######## S3Fit finish ########')

    ##########################################################################
    ################## Output best-fit spectra and SED #######################

    def output_spec(self):
        chi_sq_l = self.best_chi_sq
        n_loops = self.num_mock_loops
        n_mods = len(self.full_model_type.split('+'))
        
        spec_wave_w = self.spec['wave_w']
        self.output_spec_ltw = np.zeros((n_loops, n_mods+3, len(spec_wave_w)))
        self.output_spec_ltw[:, -1, :] = self.spec_fmock_lw # self.spec['flux_w']
        if self.have_phot:
            sed_wave_w = self.sed['wave_w']
            self.output_sed_ltw = np.zeros((n_loops, n_mods+1, len(sed_wave_w)))
        
        model_dict = self.model_dict
        i_mod = 0
        for mod in self.full_model_type.split('+'): 
            fx0, fx1, fc0, fc1 = self.model_index(mod, self.full_model_type)
            for i_loop in range(n_loops): 
                fmod_w = np.dot(self.best_coeffs[i_loop, fc0:fc1], 
                                model_dict[mod]['specfunc'](spec_wave_w, self.best_fits_x[i_loop, fx0:fx1]))
                self.output_spec_ltw[i_loop, i_mod, :] = fmod_w
                if self.have_phot:
                    fmod_w = np.dot(self.best_coeffs[i_loop, fc0:fc1], 
                                    model_dict[mod]['sedfunc'](sed_wave_w, self.best_fits_x[i_loop, fx0:fx1]))
                    self.output_sed_ltw[i_loop, i_mod, :] = fmod_w
            self.spec['fmod_'+mod+'_w'] = np.average(self.output_spec_ltw[:, i_mod, :], weights=1/chi_sq_l, axis=0)
            if self.have_phot:
                self.sed['fmod_'+mod+'_w'] = np.average(self.output_sed_ltw[:, i_mod, :], weights=1/chi_sq_l, axis=0)                
            i_mod += 1
        # total model:
        self.output_spec_ltw[:, n_mods, :] = np.sum(self.output_spec_ltw[:, :n_mods, :], axis=1)
        self.spec['fmod_tot_w'] = np.average(self.output_spec_ltw[:, n_mods, :], weights=1/chi_sq_l, axis=0)
        # residuals:
        self.output_spec_ltw[:, -2, :] = self.output_spec_ltw[:, -1, :] - self.output_spec_ltw[:, n_mods, :]
        self.spec['fres_w'] = self.spec['flux_w'] - self.spec['fmod_tot_w']                
        if self.have_phot:
            self.output_sed_ltw[:, n_mods, :] = np.sum(self.output_sed_ltw[:, :n_mods, :], axis=1)
            self.sed['fmod_tot_w'] = np.average(self.output_sed_ltw[:, n_mods, :], weights=1/chi_sq_l, axis=0)
            self.phot['fmod_b'] = self.pframe.spec2phot(sed_wave_w, self.sed['fmod_tot_w'], self.phot['trans_bw'])
            self.phot['fres_b'] = self.phot['flux_b'] - self.phot['fmod_b']

    ##########################################################################
    ########################## Output stellar fit ############################

    def output_ssp(self):
        n_loops = self.num_mock_loops
        chi_sq_l = self.best_chi_sq

        ssp_mod, ssp_cf = self.model_dict['ssp']['specmod'], self.model_dict['ssp']['cf']
        num_ssp_comps = ssp_cf.num_comps
        num_ssp_pars = ssp_cf.num_pars
        num_ssp_coeffs = int(ssp_mod.num_coeffs / num_ssp_comps)
        fx0, fx1, fc0, fc1 = self.model_index('ssp', self.full_model_type)

        name_outvals = ['chi_sq',
                        'ssp_voff', 'ssp_fwhm', 'ssp_AV', # basic fitting parameters
                        'logcsp_age', 'logsfh_tau', # SFH related fitting parameters
                        'redshift', 'flux_wavenorm',
                        'loglum_wavenorm', 'logmass_formed', 'logmass_remaining', 'logmtol',
                        'logage_lw', 'logage_mw', 'logmet_lw', 'logmet_mw']
        ind_outvals = {}
        for ind, name in enumerate(name_outvals): ind_outvals[name] = ind

        self.output_ssp_lcp = np.zeros((n_loops, num_ssp_comps, len(ind_outvals)+num_ssp_coeffs ))
        # p: chi_sq, ssp_x, output_values, ssp_coeffs
        self.output_ssp_lcp[:, :, 0] = self.best_chi_sq[:, None]
        # for i_loop in range(n_loops):
        #     self.output_ssp_lcp[i_loop,:,1:(1+num_ssp_pars)] = ssp_cf.flat_to_arr(self.best_fits_x[i_loop, fx0:fx1])
        #     self.output_ssp_lcp[i_loop,:,-num_ssp_coeffs:]   = ssp_cf.flat_to_arr(self.best_coeffs[i_loop, fc0:fc1])
        self.output_ssp_lcp[:,:,1:(1+num_ssp_pars)] = self.best_fits_x[:, fx0:fx1].reshape(n_loops, num_ssp_comps, num_ssp_pars)
        self.output_ssp_lcp[:,:,-num_ssp_coeffs:]   = self.best_coeffs[:, fc0:fc1].reshape(n_loops, num_ssp_comps, num_ssp_coeffs)
        self.output_ssp_lcp[:,:,ind_outvals['redshift']] = (1+self.output_ssp_lcp[:,:,ind_outvals['ssp_voff']]/299792.458)*(1+self.v0_redshift)-1

        ssp_x_lcp = self.output_ssp_lcp[:,:,1:(1+num_ssp_pars)]
        ssp_coeff_lcm = self.output_ssp_lcp[:,:,-num_ssp_coeffs:]
        for i_loop in range(n_loops):
            for i_comp in range(num_ssp_comps):
                rev_redshift = self.output_ssp_lcp[i_loop,i_comp,ind_outvals['redshift']]
                mask_norm_w = np.abs(self.spec['wave_w']/(1+rev_redshift) - ssp_mod.w_norm) < ssp_mod.dw_norm # for observed flux at wavenorm=5500AA(rest)
                tmp_spec_mw = self.model_dict['ssp']['specfunc'](self.spec['wave_w'], self.best_fits_x[i_loop, fx0:fx1])
                tmp_spec_cmw = tmp_spec_mw.reshape(num_ssp_comps, num_ssp_coeffs, len(mask_norm_w))
                tmp_spec_w = np.dot(ssp_coeff_lcm[i_loop,i_comp], tmp_spec_cmw[i_comp])
                self.output_ssp_lcp[i_loop,i_comp,ind_outvals['flux_wavenorm']] = tmp_spec_w[mask_norm_w].mean()

                dist_lum = cosmo.luminosity_distance(rev_redshift).to('cm').value
                unitconv = 4*np.pi*dist_lum**2 / const.L_sun.to('erg/s').value * self.spec_flux_scale # convert intrinsic flux5500(rest) to L5500 
                if ssp_mod.sfh_names[i_comp] == 'nonparametric':
                    coeff_m = ssp_coeff_lcm[i_loop,i_comp] # _m represents all ssp models
                else:
                    sfh_factor_m = ssp_mod.sfh_factor(i_comp, ssp_mod.sfh_names[i_comp], ssp_x_lcp[i_loop,i_comp,3:]) 
                    coeff_m = np.tile(ssp_coeff_lcm[i_loop,i_comp], (ssp_mod.num_ages,1)).T.flatten() * sfh_factor_m 
                    # recover ssp_coeff_lcm where _m represents all csp, to coeff_m where _m represents all ssp
                lum_m = coeff_m * unitconv # intrinsic L5500, in Lsun/AA
                mass_formed_m = lum_m * ssp_mod.mtol_m
                mass_remaining_m = mass_formed_m * ssp_mod.remainmassfrac_m
                self.output_ssp_lcp[i_loop,i_comp,ind_outvals['loglum_wavenorm']] = np.log10((lum_m).sum())
                self.output_ssp_lcp[i_loop,i_comp,ind_outvals['logmass_formed']] = np.log10((mass_formed_m).sum())
                self.output_ssp_lcp[i_loop,i_comp,ind_outvals['logmass_remaining']] = np.log10((mass_remaining_m).sum())
                self.output_ssp_lcp[i_loop,i_comp,ind_outvals['logmtol']] = np.log10(mass_remaining_m.sum() / lum_m.sum())

                self.output_ssp_lcp[i_loop,i_comp,ind_outvals['logage_lw']] = (lum_m * np.log10(ssp_mod.age_m)).sum() / lum_m.sum()
                self.output_ssp_lcp[i_loop,i_comp,ind_outvals['logage_mw']] = (mass_remaining_m * np.log10(ssp_mod.age_m)).sum() / mass_remaining_m.sum()
                self.output_ssp_lcp[i_loop,i_comp,ind_outvals['logmet_lw']] = (lum_m * np.log10(ssp_mod.met_m)).sum() / lum_m.sum()
                self.output_ssp_lcp[i_loop,i_comp,ind_outvals['logmet_mw']] = (mass_remaining_m * np.log10(ssp_mod.met_m)).sum() / mass_remaining_m.sum()

        # output to screen
        self.output_ssp_vals = {}
        for i_comp in range(ssp_cf.num_comps):
            self.output_ssp_vals[ssp_cf.comp_c[i_comp]] = { 'mean': ind_outvals.copy(), 'rms': ind_outvals.copy() }
            for name in ind_outvals.keys():
                self.output_ssp_vals[ssp_cf.comp_c[i_comp]]['mean'][name] = np.average(self.output_ssp_lcp[:,i_comp,ind_outvals[name]], weights=1/chi_sq_l)
                self.output_ssp_vals[ssp_cf.comp_c[i_comp]]['rms'][name] = np.std(self.output_ssp_lcp[:,i_comp,ind_outvals[name]], ddof=1) 
            if ssp_mod.sfh_names[i_comp] == 'nonparametric':
                ssp_coeff_best_m = np.average(ssp_coeff_lcm[:,0,:], weights=1/chi_sq_l, axis=0)
                self.output_ssp_vals[ssp_cf.comp_c[i_comp]]['ssp_normcoeff_mean_m'] = ssp_coeff_best_m / ssp_coeff_best_m.sum()
                self.output_ssp_vals[ssp_cf.comp_c[i_comp]]['ssp_normcoeff_rms_m'] = np.std(ssp_coeff_lcm, axis=(0,1), ddof=1) / ssp_coeff_best_m.sum()
        if ssp_cf.num_comps > 1:
            self.output_ssp_vals['total'] = { 'mean': ind_outvals.copy(), 'rms': ind_outvals.copy() }
            for val in ind_outvals.keys():
                self.output_ssp_vals['total']['mean'][val] = 0
                self.output_ssp_vals['total']['rms'][val] = 0
            lum_l = (10.0**self.output_ssp_lcp[:,:,ind_outvals['loglum_wavenorm']]).sum(axis=1)
            mass_formed_l = (10.0**self.output_ssp_lcp[:,:,ind_outvals['logmass_formed']]).sum(axis=1)
            mass_remaining_l = (10.0**self.output_ssp_lcp[:,:,ind_outvals['logmass_remaining']]).sum(axis=1)
            for (name, val) in zip(['loglum_wavenorm', 'logmass_formed', 'logmass_remaining', 'logmtol'], 
                                   [lum_l, mass_formed_l, mass_remaining_l, mass_remaining_l/lum_l]):
                self.output_ssp_vals['total']['mean'][name] = np.average(np.log10(val), weights=1/chi_sq_l)
                self.output_ssp_vals['total']['rms'][name] = np.std(np.log10(val), ddof=1)

        self.output_ssp_to_screen()

    def output_ssp_to_screen(self):
        ssp_mod, ssp_cf = self.model_dict['ssp']['specmod'], self.model_dict['ssp']['cf']

        for i_comp in range(len(self.output_ssp_vals)):
            tmp_vals = self.output_ssp_vals[[*self.output_ssp_vals][i_comp]]
            if i_comp < ssp_cf.num_comps:
                if ssp_mod.sfh_names[i_comp] == 'nonparametric':
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
                    for i in range(ssp_mod.num_models):
                        min_ncoeffs = tmp_vals['ssp_normcoeff_mean_m'].max()/10
                        if tmp_vals['ssp_normcoeff_mean_m'][i] < min_ncoeffs: continue
                        tbl_row = []
                        tbl_row.append(i)
                        tbl_row.append(ssp_mod.age_m[i])
                        tbl_row.append(ssp_mod.met_m[i])
                        tbl_row.append(tmp_vals['ssp_normcoeff_mean_m'][i]) 
                        tbl_row.append(tmp_vals['ssp_normcoeff_rms_m'][i])
                        tbl_row.append(np.log10(ssp_mod.mtol_m[i]))
                        print(fmt_numbers.format(*tbl_row))
                    print(tbl_border)
                    print(f'Coeff is the intrinsic flux at rest {ssp_mod.w_norm}AA in unit of {self.spec_flux_scale} erg/s/cm2/AA.')

            print('')
            msg = ''
            if i_comp < ssp_cf.num_comps:
                print(f'Best-fit stellar properties of the <{ssp_cf.comp_c[i_comp]}> component with {ssp_mod.sfh_names[i_comp]} SFH.')
                msg += f'| Chi^2 of best-fit                = {self.best_chi_sq[0]:10.4f}\n'
                msg += f'| Redshift                         = {tmp_vals["mean"]["redshift"]:10.4f}'
                msg += f' +/- {tmp_vals["rms"]["redshift"]:0.4f}\n'
                msg += f'| Velocity dispersion (km/s)       = {tmp_vals["mean"]["ssp_fwhm"]/2.355:10.4f}'
                msg += f' +/- {tmp_vals["rms"]["ssp_fwhm"]/2.355:0.4f}\n'
                msg += f'| Extinction (AV)                  = {tmp_vals["mean"]["ssp_AV"]:10.4f}'
                msg += f' +/- {tmp_vals["rms"]["ssp_AV"]:0.4f}\n'
                msg += f'| log Age_max of CSP (Gyr)         = {tmp_vals["mean"]["logcsp_age"]:10.4f}'
                msg += f' +/- {tmp_vals["rms"]["logcsp_age"]:0.4f}\n'
                msg += f'| log Tau of SFH (Gyr)             = {tmp_vals["mean"]["logsfh_tau"]:10.4f}'
                msg += f' +/- {tmp_vals["rms"]["logsfh_tau"]:0.4f}\n'
                msg += f'| F{ssp_mod.w_norm}(rest) ({self.spec_flux_scale} erg/s/cm2/AA) = {tmp_vals["mean"]["flux_wavenorm"]:10.4f}'
                msg += f' +/- {tmp_vals["rms"]["flux_wavenorm"]:0.4f}\n'
            else:
                print(f'Best-fit stellar properties of the total components.')
            msg += f'| log L{ssp_mod.w_norm}(rest) (Lsun/AA)        = {tmp_vals["mean"]["loglum_wavenorm"]:10.4f}'
            msg += f' +/- {tmp_vals["rms"]["loglum_wavenorm"]:0.4f}\n'
            msg += f'| log Mass (all formed) (Msun)     = {tmp_vals["mean"]["logmass_formed"]:10.4f}'
            msg += f' +/- {tmp_vals["rms"]["logmass_formed"]:0.4f}\n'
            msg += f'| log Mass (remainning) (Msun)     = {tmp_vals["mean"]["logmass_remaining"]:10.4f}'
            msg += f' +/- {tmp_vals["rms"]["logmass_remaining"]:0.4f}\n'
            msg += f'| log M/L{ssp_mod.w_norm} (Msun/(Lsun/AA))     = {tmp_vals["mean"]["logmtol"]:10.4f}'
            msg += f' +/- {tmp_vals["rms"]["logmtol"]:0.4f}'
            if i_comp < ssp_cf.num_comps:
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
        ssp_mod, ssp_cf = self.model_dict['ssp']['specmod'], self.model_dict['ssp']['cf']
        num_ssp_comps = ssp_cf.num_comps
        num_ssp_pars = ssp_cf.num_pars
        num_ssp_coeffs = int(self.model_dict['ssp']['num_coeffs'] / num_ssp_comps)

        if output_ssp_lcp is None: output_ssp_lcp = self.output_ssp_lcp
        ssp_x_lcp = output_ssp_lcp[:,:,1:(1+num_ssp_pars)]
        ssp_coeff_lcm = output_ssp_lcp[:,:,-num_ssp_coeffs:]

        age_a = ssp_mod.age_m[:ssp_mod.num_ages]
        output_sfh_lcza = np.zeros((self.num_mock_loops, num_ssp_comps, ssp_mod.num_mets, ssp_mod.num_ages))

        # i_loop = 0
        for i_loop in range(self.num_mock_loops):
            for i_comp in range(num_ssp_comps):
                if ssp_mod.sfh_names[i_comp] == 'nonparametric':
                    coeff_m = ssp_coeff_lcm[i_loop,i_comp] # _m represents all ssp models
                else:
                    sfh_factor_m = ssp_mod.sfh_factor(i_comp, ssp_mod.sfh_names[i_comp], ssp_x_lcp[i_loop,i_comp,3:]) 
                    coeff_m = np.tile(ssp_coeff_lcm[i_loop,i_comp], (ssp_mod.num_ages,1)).T.flatten() * sfh_factor_m 
                    # recover ssp_coeff_lcm where _m represents all csp, to coeff_m where _m represents all ssp
                rev_redshift = (1+self.output_ssp_lcp[i_loop,i_comp,1]/299792.458)*(1+self.v0_redshift)-1
                dist_lum = cosmo.luminosity_distance(rev_redshift).to('cm').value
                unitconv = 4*np.pi*dist_lum**2 / const.L_sun.to('erg/s').value * self.spec_flux_scale # convert intrinsic flux5500(rest) to L5500 
                output_sfh_lcza[i_loop,i_comp,:,:] = (coeff_m * unitconv * ssp_mod.sfrtol_m).reshape(ssp_mod.num_mets, ssp_mod.num_ages)

        if num_bins is not None:
            output_sfh_lczb = np.zeros((self.num_mock_loops, num_ssp_comps, ssp_mod.num_mets, num_bins))
            age_b = np.zeros((num_bins))
            logage_a = np.log10(age_a)
            bwidth = (logage_a[-1] - logage_a[0]) / num_bins
            for i_bin in range(num_bins):
                mask_a = (logage_a > (logage_a[0]+i_bin*bwidth)) & (logage_a <= (logage_a[0]+(i_bin+1)*bwidth))
                duration_b = 10.0**(logage_a[0]+(i_bin+1)*bwidth) - 10.0**(logage_a[0]+i_bin*bwidth)
                output_sfh_lczb[:,:,:,i_bin] = (output_sfh_lcza[:,:,:,mask_a] * ssp_mod.duration_m[:ssp_mod.num_ages][mask_a]).sum(axis=3) / duration_b
                age_b[i_bin] = 10.0**(logage_a[0]+(i_bin+1/2)*bwidth)
                
        if plot:
            plt.figure(figsize=(9,3))
            ax = plt.subplot(1, 2, 1)
            for i_comp in range(output_sfh_lcza.shape[1]):  
                for i_loop in range(output_sfh_lcza.shape[0]):
                    plt.plot(np.log10(age_a), output_sfh_lcza[i_loop,i_comp,2,:], '--')
                plt.plot(np.log10(age_a), output_sfh_lcza[:,i_comp,2,:].mean(axis=0), linewidth=4, alpha=0.5, label=f'Mean {ssp_cf.comp_c[i_comp]}')
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
                           alpha=0.3, hatch='///', ec='C7', linewidth=4, label=f'Mean {ssp_cf.comp_c[i_comp]}')
                plt.xlim(1.5,-3); plt.ylim(1,1e4); plt.yscale('log')
                plt.xlabel('Log looking back time (Gyr)'); plt.ylabel('SFR (Msun/yr)'); plt.legend()
                plt.title('After binning in log time')
                
        if return_sfh:
            if num_bins is None:
                return output_sfh_lcza, age_a
            else:
                return output_sfh_lczb, age_b

    ##########################################################################
    ###################### Output emission line fit ##########################

    def output_el(self):
        n_loops = self.num_mock_loops
        chi_sq_l = self.best_chi_sq
        
        el_mod, el_cf = self.model_dict['el']['specmod'], self.model_dict['el']['cf']
        fx0, fx1, fc0, fc1 = self.model_index('el', self.full_model_type)
        
        self.output_el_lcp = np.zeros((n_loops, el_mod.num_comps, 1 + el_cf.num_pars + el_mod.num_lines))
        # use el_mod.num_lines instead of self.num_el_coeffs to cover non-free lines
        self.output_el_lcp[:, :, 0] = self.best_chi_sq[:, None]

        for i_loop in range(n_loops):
            self.output_el_lcp[i_loop, :, 1:(1+el_cf.num_pars)] = el_cf.flat_to_arr( self.best_fits_x[i_loop, fx0:fx1] )
            self.output_el_lcp[i_loop, :, -el_mod.num_lines:][el_mod.mask_free_cn] = self.best_coeffs[i_loop, fc0:fc1]
        
        # update linked_ratio_n
        new_linked_ratio_lcn = np.tile(el_mod.linked_ratio_n, (self.num_mock_loops, el_mod.num_comps, 1))
        mask_Balmer = el_mod.linked_to_n == 6564.632
        # determine extHa_base_n of extHa_base_BLR_n
        # add later
        new_linked_ratio_lcn[:,:,mask_Balmer] *= el_mod.extHa_base_n[None,None,mask_Balmer]**self.output_el_lcp[:,:,3:4] # keep:4
        i_SIIa = np.where(el_mod.line_rest_n == 6718.295)[0][0]
        new_linked_ratio_lcn[:,:,i_SIIa] = self.output_el_lcp[:,:,4] # SII_ratio = SIIa/SIIb
        self.new_linked_ratio_lcn = new_linked_ratio_lcn # for test
        
        for i_loop in range(n_loops):
            list_linked = np.where(el_mod.linked_to_n != -1)[0]
            for i_line in list_linked:
                i_main = 1 + el_cf.num_pars + np.where(el_mod.line_rest_n == el_mod.linked_to_n[i_line])[0]
                self.output_el_lcp[i_loop, :, 1 + el_cf.num_pars + i_line] = self.output_el_lcp[i_loop, :, i_main] \
                                                                         * new_linked_ratio_lcn[i_loop, :, i_line]
            self.output_el_lcp[i_loop, :, -el_mod.num_lines:][~el_mod.mask_valid_cn] = 0
            
        # output to screen
        self.output_el_vals = {
            'mean': np.average(self.output_el_lcp, weights=1/chi_sq_l, axis=0), 
            'rms' : np.std(self.output_el_lcp, axis=0, ddof=1) }
        self.output_el_to_screen()

    def output_el_to_screen(self):
        el_mod, el_cf = self.model_dict['el']['specmod'], self.model_dict['el']['cf']

        print('')
        print('Best-fit emission line components')

        cols = 'Par/Line Name'
        fmt_cols = '| {:^18} |'
        fmt_numbers = '| {:^18} |' #fmt_numbers = '| {:=13.4f} |'
        for i_kin in range(el_mod.num_comps): 
            cols += ','+el_cf.comp_c[i_kin]
            fmt_cols += ' {:^18} |'
            fmt_numbers += ' {:=8.2f} +- {:=6.2f} |'
        cols_split = cols.split(',')
        tbl_title = fmt_cols.format(*cols_split)
        tbl_border = len(tbl_title)*'-'
        print(tbl_border)
        print(tbl_title)
        print(tbl_border)

        names = ['Chi^2', 'Voff (km/s)', 'FWHM (km/s)', 'AV (Balmer decre.)', '[SII]6716/6731']
        for i_line in range(el_mod.num_lines): 
            names.append('{} {:=7.2f}'.format(el_mod.line_name_n[i_line], el_mod.line_rest_n[i_line]))
        for i_par in range(len(names)): 
            tbl_row = []
            tbl_row.append(names[i_par])
            for i_comp in range(el_mod.num_comps):
                tbl_row.append(self.output_el_vals['mean'][i_comp, i_par])
                tbl_row.append(self.output_el_vals['rms'][i_comp, i_par])
            print(fmt_numbers.format(*tbl_row))
        print(tbl_border)  

    ##########################################################################
    ############################ Output agn fit ##############################

    def output_agn(self):
        n_loops = self.num_mock_loops
        chi_sq_l = self.best_chi_sq

        num_agn_pars = self.model_dict['agn']['cf'].num_pars
        num_agn_coeffs = self.model_dict['agn']['num_coeffs']
        fx0, fx1, fc0, fc1 = self.model_index('agn', self.full_model_type)

        self.output_agn_lp = np.zeros((n_loops, 1 + num_agn_pars + num_agn_coeffs ))
        self.output_agn_lp[:, 0] = self.best_chi_sq
        self.output_agn_lp[:, 1:(1+num_agn_pars)] = self.best_fits_x[:, fx0:fx1]
        self.output_agn_lp[:, -num_agn_coeffs:] = self.best_coeffs[:, fc0:fc1]
        # output to screen
        self.output_agn_vals = {
            'mean': np.average(self.output_agn_lp, weights=1/chi_sq_l, axis=0), 
            'rms' : np.std(self.output_agn_lp, axis=0, ddof=1) }
        self.output_agn_to_screen()

    def output_agn_to_screen(self):
        agn_mod = self.model_dict['agn']['specmod']
        dist_lum = cosmo.luminosity_distance(self.v0_redshift).to('cm').value
        unitconv = 4*np.pi*dist_lum**2 / const.L_sun.to('erg/s').value * self.spec_flux_scale # convert intrinsic flux5500(rest) to L5500

        print('')
        print('Best-fit AGN component')
        msg  = f'| Voff (km/s)                      = {self.output_agn_vals["mean"][1]:10.4f}'
        msg += f' +/- {self.output_agn_vals["rms"][1]:0.4f}\n'
        msg += f'| FWHM (km/s)                      = {self.output_agn_vals["mean"][2]:10.4f}'
        msg += f' +/- {self.output_agn_vals["rms"][2]:0.4f}\n'
        msg += f'| Extinction (AV)                  = {self.output_agn_vals["mean"][3]:10.4f}'
        msg += f' +/- {self.output_agn_vals["rms"][3]:0.4f}\n'
        msg += f'| Powerlaw α_λ                     = {self.output_agn_vals["mean"][4]:10.4f}'
        msg += f' +/- {self.output_agn_vals["rms"][4]:0.4f}\n'
        msg += f'| F{agn_mod.w_norm}(rest) ({self.spec_flux_scale} erg/s/cm2/AA) = {self.output_agn_vals["mean"][5]:10.4f}'
        msg += f' +/- {self.output_agn_vals["rms"][5]:0.4f}\n'
        msg += f'| L{agn_mod.w_norm}(rest) (1e8 Lsun/AA)        = {self.output_agn_vals["mean"][5]*unitconv/1e8:10.4f}'
        msg += f' +/- {self.output_agn_vals["rms"][5]*unitconv/1e8:0.4f}'
        bar = '='*60
        print(bar)
        print(msg)
        print(bar)

    ##########################################################################
    ########################### Output torus fit #############################

    def output_torus(self):
        n_loops = self.num_mock_loops
        chi_sq_l = self.best_chi_sq
        
        num_torus_pars = self.model_dict['torus']['cf'].num_pars
        num_torus_coeffs = self.model_dict['torus']['num_coeffs']
        fx0, fx1, fc0, fc1 = self.model_index('torus', self.full_model_type)
        
        self.output_torus_lp = np.zeros((n_loops, 1 + num_torus_pars + num_torus_coeffs ))
        self.output_torus_lp[:, 0] = self.best_chi_sq
        self.output_torus_lp[:, 1:(1+num_torus_pars)] = self.best_fits_x[:, fx0:fx1]
        self.output_torus_lp[:, -num_torus_coeffs:] = self.best_coeffs[:, fc0:fc1]
        # output to screen
        self.output_torus_vals = {
            'mean': np.average(self.output_torus_lp, weights=1/chi_sq_l, axis=0), 
            'rms' : np.std(self.output_torus_lp, axis=0, ddof=1) }
        self.output_torus_to_screen()
        
    def output_torus_to_screen(self):
        print('')
        print('Best-fit torus components')
        # msg  = f'| Voff (km/s)           = {self.output_torus_vals["mean"][1]:10.4f}'
        # msg += f' +/- {self.output_torus_vals["rms"][1]:0.4f}\n'
        msg  = f'| Opacity               = {self.output_torus_vals["mean"][2]:10.4f}'
        msg += f' +/- {self.output_torus_vals["rms"][2]:0.4f}\n'
        msg += f'| Out/in radii ratio    = {self.output_torus_vals["mean"][4]:10.4f}'
        msg += f' +/- {self.output_torus_vals["rms"][4]:0.4f}\n'
        msg += f'| Half OpenAng (degree) = {self.output_torus_vals["mean"][3]:10.4f}'
        msg += f' +/- {self.output_torus_vals["rms"][3]:0.4f}\n'
        msg += f'| Inclination (degree)  = {self.output_torus_vals["mean"][5]:10.4f}'
        msg += f' +/- {self.output_torus_vals["rms"][5]:0.4f}\n'
        msg += f'| Torus Lum (1e12 Lsun) = {self.output_torus_vals["mean"][6]:10.4f}'
        msg += f' +/- {self.output_torus_vals["rms"][6]:0.4f}'
        bar = '='*50
        print(bar)
        print(msg)
        print(bar)

#############################################################################################################
#############################################################################################################

class PhotFrame(object):
    def __init__(self, 
                 name_b=None, flux_b=None, ferr_b=None, fluxunit='mJy', # on input data
                 trans_dir=None, wave_w=None, waveunit='angstrom'): # on transmission curves
        # add file_bac, file_iron later
        
        self.name_b = copy(name_b)
        self.flux_b = copy(flux_b)
        self.ferr_b = copy(ferr_b) 
        self.fluxunit = fluxunit
        
        self.trans_dir = trans_dir
        self.wave_w = copy(wave_w)
        self.waveunit = waveunit
        if (self.wave_w is not None) & (self.waveunit == 'micron'): self.wave_w *= 1e4 # convert to AA
                
        self.trans_dict, self.trans_bw, self.wave_w = self.read_transmission(self.trans_dir, name_b=self.name_b, wave_w=self.wave_w)
        self.rFnuFlam_w = self.rFnuFlam_func(self.wave_w)
        self.rFnuFlam_b = self.rFnuFlam_func(self.wave_w, self.trans_bw)
        self.wave_b = self.spec2phot(self.wave_w, self.wave_w, self.trans_bw)
        
        if self.fluxunit == 'mJy': 
            self.flux_b /= self.rFnuFlam_b # convert to erg/s/cm2/AA
            self.ferr_b /= self.rFnuFlam_b # convert to erg/s/cm2/AA
        
    def read_transmission(self, trans_dir, name_b=None, wave_w=None):        
        if name_b is None:
            file_list = np.array(os.listdir(trans_dir))
            file_list = file_list[np.array([f[-4:] for f in file_list]) == '.dat']
            name_b = np.array([f.split('.dat')[0] for f in file_list])
            trans_bw = None
        else:
            trans_bw = 1

        if wave_w is None:
            w_min, w_max = 1e16, 0
            for name in name_b: 
                filterdata = np.loadtxt(trans_dir+name+'.dat')
                wave_ini, trans_ini = filterdata[:,0], filterdata[:,1]
                w_min = np.minimum(w_min, wave_ini.min())
                w_max = np.maximum(w_max, wave_ini.max())
            wave_w = np.logspace(np.log10(w_min)-0.1, np.log10(w_max)+0.1, num=5000)

        trans_dict = {}
        for name in name_b:
            filterdata = np.loadtxt(trans_dir+name+'.dat')
            wave_ini, trans_ini = filterdata[:,0], filterdata[:,1]
            trans_ini /= np.trapz(trans_ini, x=wave_ini)
            # trans_ini /= np.trapz(trans_ini * const.c.to('micron Hz').value / wave_ini**2, x=wave_ini) # in Hz-1
            tmp_center = np.trapz(trans_ini*wave_ini, x=wave_ini)
            tmp_trans = np.interp(wave_w, wave_ini, trans_ini, left=0, right=0)
            # tmp_trans = np.interp(wave_w, np.array([wave_w[0]]+list(wave_ini)+[wave_w[-1]]), np.array([0]+list(trans_ini)+[0]), left=0, right=0)
            # tmp_trans[wave_w < wave_ini.min()] = 0; tmp_trans[wave_w > wave_ini.max()] = 0
            tmp_dict = {'center': tmp_center, 'trans': tmp_trans}
            trans_dict[name] = tmp_dict

        if trans_bw is not None:
            trans_bw = []
            for name in name_b:
                trans_bw.append(trans_dict[name]['trans'])
            trans_bw = np.array(trans_bw)

        return trans_dict, trans_bw, wave_w
        # Flam (mean) = INT(Flam * Tlam, x=lam) / INT(Tlam, x=lam), where INT(Tlam, x=lam) = 1
        # Fnu (mean) = INT(Flam * Tlam, x=lam) / INT(Tlam * dnu/dlam, x=lam)
        
    def rFnuFlam_func(self, wave_w, trans_bw=None):
        unitflam = (1 * u.erg/u.s/u.cm**2/u.angstrom)
        rDnuDlam = const.c.to('angstrom Hz') / (wave_w * u.angstrom)**2
        if trans_bw is None:
            # return the ratio of spectrum between Fnu (mJy) and Flam (erg/s/cm2/AA); wave in AA
            return (unitflam / rDnuDlam).to('mJy').value
        else:
            # return the ratio of band flux between Fnu (mJy) and Flam (erg/s/cm2/AA); wave in AA
            unitfint = unitflam * (1 * u.angstrom)
            # here (1 * u.angstrom) = np.trapz(trans, x=wave * u.angstrom, axis=axis), since trans is normalized to int=1
            width_nu = np.trapz(trans_bw * rDnuDlam, x=wave_w * u.angstrom, axis=trans_bw.ndim-1)
            return (unitfint / width_nu).to('mJy').value
        
    def spec2phot(self, wave_w, spec_mw, trans_bw):
        # convert spectrum in flam (erg/s/cm2/A) to mean flam in band (erg/s/cm2/A)
        if (spec_mw.ndim == 1) & (trans_bw.ndim == 1):
            return np.trapz(trans_bw * spec_mw, x=wave_w, axis=0) # return flux, 1-model, 1-band
        if (spec_mw.ndim == 1) & (trans_bw.ndim == 2):
            return np.trapz(trans_bw * spec_mw[None,:], x=wave_w, axis=1) # return flux_b, 1-model, multi-band
        if (spec_mw.ndim == 2) & (trans_bw.ndim == 1):
            return np.trapz(trans_bw[None,:] * spec_mw, x=wave_w, axis=1) # return flux_m, multi-model, 1-band
        if (spec_mw.ndim == 2) & (trans_bw.ndim == 2):
            return np.trapz(trans_bw[None,:,:] * spec_mw[:,None,:], x=wave_w, axis=2) # return flux_mb
        # short for np.trapz(trans * spec, x=wave, axis=axis) / np.trapz(trans, x=wave, axis=axis), trans is normalized to int=1


#############################################################################################################
#############################################################################################################

# class ConfigFrame(object):
#     def __init__(self, pmmc):
#         self.pmmc = pmmc
#         self.num_comps = len(pmmc)
#         self.num_pars = len(pmmc[0])-1
#         self.pars = np.zeros((self.num_comps, self.num_pars), dtype='float')
#         self.mins, self.maxs = self.pars.copy(), self.pars.copy()
#         self.comps = [] # do not initialize string array for unknown string length, e.g., dtype='<U256'
#         for i_comp in range(self.num_comps):
#             for i_par in range(self.num_pars):
#                 self.pars[i_comp,i_par] = pmmc[i_comp][i_par][0]
#                 self.mins[i_comp,i_par] = pmmc[i_comp][i_par][1]
#                 self.maxs[i_comp,i_par] = pmmc[i_comp][i_par][2]
#             self.comps.append(pmmc[i_comp][-1])
#         self.comps = np.array(self.comps)

#     def flat_to_arr(self, pars_flat):
#         return pars_flat.reshape(self.num_comps, self.num_pars)

class ConfigFrame(object):
    def __init__(self, config):
        self.config = config
        self.comp_c = [*config.keys()]
        self.num_comps = len([*config])
        self.num_pars = len(config[[*config][0]]['pars'])
        self.num_compxpars = self.num_comps * self.num_pars # used for flattened input parameters of fitting
        self.min_cp = np.zeros((self.num_comps, self.num_pars), dtype='float')
        self.max_cp = np.zeros((self.num_comps, self.num_pars), dtype='float')
        self.tie_cp = np.zeros((self.num_comps, self.num_pars), dtype='<U64')
        self.info_c = [] # do not initialize string array for unknown string length, e.g., dtype='<U256'
        for i_comp in range(self.num_comps):
            for i_par in range(self.num_pars):
                self.min_cp[i_comp,i_par] = config[[*config][i_comp]]['pars'][i_par][0]
                self.max_cp[i_comp,i_par] = config[[*config][i_comp]]['pars'][i_par][1]
                self.tie_cp[i_comp,i_par] = config[[*config][i_comp]]['pars'][i_par][2]
            self.info_c.append(config[[*config][i_comp]]['info'])
            self.info_c[i_comp]['comp_name'] = [*config.keys()][i_comp]
        # self.info_c = np.array(self.info_c)

    # def flat_to_arr(self, pars_flat):
    #     return pars_flat.reshape(self.num_comps, self.num_pars)
    def flat_to_arr(self, vals_flat):
        return vals_flat.reshape(self.num_comps, int(len(vals_flat)/self.num_comps))

#############################################################################################################
#############################################################################################################

class SSPModels(object):
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
        self.sfh_names = np.array([d['sfh'] for d in self.cframe.info_c])
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
        self.num_models = self.num_ages * self.num_mets # self.header['NAXIS2']

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
        evo_time = csp_age - ssp_age_m

        if sfh_name == 'exponential': 
            tau = 10.0**sfh_pars[1]
            sfh_func_m = np.exp(-(evo_time) / tau)
        if sfh_name == 'delayed': 
            tau = 10.0**sfh_pars[1]
            sfh_func_m = np.exp(-(evo_time) / tau) * evo_time
        if sfh_name == 'constant': 
            sfh_func_m = np.ones_like(evo_time)
        ############################
        # Add new SFH function here. 
        ############################

        sfh_func_m[~self.mask_ssp_allowed(i_comp)] = 0 # do not use ssp out of allowed range
        sfh_func_m[evo_time < 0] = 0 # do not allow ssp older than csp_age 
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

#############################################################################################################
#############################################################################################################

class ELineModels(object):
    def __init__(self, w_min=None, w_max=None, cframe=None, v0_redshift=0, spec_R_inst=1e8, verbose=True):
        self.w_min = w_min
        self.w_max = w_max
        self.cframe = cframe
        self.v0_redshift = v0_redshift
        self.spec_R_inst = spec_R_inst

        self.set_linelist()

        self.num_comps = len(cframe.info_c)
        self.num_lines = len(self.line_rest_n)
        self.mask_valid_cn = np.zeros((self.num_comps, self.num_lines), dtype='bool')
        self.mask_free_cn  = np.zeros((self.num_comps, self.num_lines), dtype='bool')
        for i_kin in range(self.num_comps):
            mask_valid_n  = self.line_rest_n > (self.w_min-50)
            mask_valid_n &= self.line_rest_n < (self.w_max+50)
            mask_free_n   = mask_valid_n & (self.linked_to_n == -1)
            if cframe.info_c[i_kin]['line_used'][0] != 'all': 
                mask_select_n = np.isin(self.line_name_n, cframe.info_c[i_kin]['line_used'])
                mask_valid_n &= mask_select_n
                mask_free_n  &= mask_select_n
            self.mask_valid_cn[i_kin, mask_valid_n] = True
            self.mask_free_cn[i_kin,  mask_free_n ] = True
        self.num_coeffs = self.mask_free_cn.sum()
        
        # set component name and enable mask for each free line; _f for free or coeffs
        self.component_f = [] # np.zeros((self.num_coeffs), dtype='<U16')
        for i_kin in range(self.num_comps):
            for i_line in range(self.num_lines):
                if self.mask_free_cn[i_kin, i_line]:
                    self.component_f.append(cframe.info_c[i_kin]['comp_name'])
        self.component_f = np.array(self.component_f)

        mask_Balmer = self.linked_to_n == 6564.632
        self.extHa_base_n = np.zeros_like(self.linked_ratio_n)
        tmp = ExtLaw(self.line_rest_n[mask_Balmer]) - ExtLaw(np.array([6564.632]))
        self.extHa_base_n[mask_Balmer] = 10.0**(-0.4 * tmp) # the base of extinction (ref to Halpha)
        # repeat for self.extHa_base_BLR_n
        
        if verbose:
            for i_kin in range(self.num_comps):
                print('Emission line complex', i_kin, cframe.info_c[i_kin]['comp_name'], 
                      ', total number:', self.mask_valid_cn[i_kin].sum(), ', free lines:', self.line_name_n[self.mask_free_cn[i_kin]])

    def set_linelist(self):
        self.line_rest_n, self.linked_to_n, self.linked_ratio_n, self.line_name_n = [],[],[],[]
        # _n denote line; _l already used for loop id
        # [v2] vacuum wavelength calculated with lamb_air_to_vac and air values in 
        # http://astronomy.nmsu.edu/drewski/tableofemissionlines.html
        # also check https://classic.sdss.org/dr6/algorithms/linestable.php
        # [v3] update with https://physics.nist.gov/PhysRefData/ASD/lines_form.html (H ref: T8637)
        # ratio of doublets calculated with pyneb under ne=100, Te=1e4
        self.line_rest_n.append(6564.632); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('Ha')
        self.line_rest_n.append(4862.691); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.349)  ; self.line_name_n.append('Hb')
        self.line_rest_n.append(4341.691); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.164)  ; self.line_name_n.append('Hg')
        self.line_rest_n.append(4102.899); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.0904) ; self.line_name_n.append('Hd')
        self.line_rest_n.append(3971.202); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.0555) ; self.line_name_n.append('H7')
        self.line_rest_n.append(3890.158); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.0367) ; self.line_name_n.append('H8')
        self.line_rest_n.append(3836.479); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.0255) ; self.line_name_n.append('H9')
        self.line_rest_n.append(3798.983); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.0185) ; self.line_name_n.append('H10')
        self.line_rest_n.append(3771.708); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.0139) ; self.line_name_n.append('H11')
        self.line_rest_n.append(3751.224); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.0107) ; self.line_name_n.append('H12')
        self.line_rest_n.append(3735.437); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.00838); self.line_name_n.append('H13')
        self.line_rest_n.append(3723.004); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.00671); self.line_name_n.append('H14')
        self.line_rest_n.append(5877.249); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('HeI')

        self.line_rest_n.append(6732.674); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[SII]b')
        self.line_rest_n.append(6718.295); self.linked_to_n.append(6732.674); self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[SII]a')
        self.line_rest_n.append(6585.270); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[NII]b')
        self.line_rest_n.append(6549.860); self.linked_to_n.append(6585.270); self.linked_ratio_n.append(0.340)  ; self.line_name_n.append('[NII]a')
        self.line_rest_n.append(6365.536); self.linked_to_n.append(6302.046); self.linked_ratio_n.append(0.319)  ; self.line_name_n.append('[OI]b')
        self.line_rest_n.append(6302.046); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[OI]a')
        self.line_rest_n.append(5201.705); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[NI]b')
        self.line_rest_n.append(5199.349); self.linked_to_n.append(5201.705); self.linked_ratio_n.append(0.783)  ; self.line_name_n.append('[NI]a')
        self.line_rest_n.append(5099.230); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[FeVI]')
        self.line_rest_n.append(5008.240); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[OIII]b')
        self.line_rest_n.append(4960.295); self.linked_to_n.append(5008.240); self.linked_ratio_n.append(0.335)  ; self.line_name_n.append('[OIII]a')
        self.line_rest_n.append(3968.590); self.linked_to_n.append(3869.860); self.linked_ratio_n.append(0.301)  ; self.line_name_n.append('[NeIII]b')
        self.line_rest_n.append(3869.860); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[NeIII]a')
        self.line_rest_n.append(3729.875); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[OII]b')
        self.line_rest_n.append(3727.092); self.linked_to_n.append(3729.875); self.linked_ratio_n.append(0.741)  ; self.line_name_n.append('[OII]a')
        self.line_rest_n.append(3426.864); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[NeV]b')
        self.line_rest_n.append(3346.783); self.linked_to_n.append(3426.864); self.linked_ratio_n.append(0.366)  ; self.line_name_n.append('[NeV]a')
        self.line_rest_n = np.array(self.line_rest_n)
        self.linked_to_n = np.array(self.linked_to_n)
        self.linked_ratio_n = np.array(self.linked_ratio_n)
        self.line_name_n = np.array(self.line_name_n)
        
        # recalculate Hydrogen ratios for BLR condition with ne=, Te=
        # add later
        # self.linked_ratio_BLR_n = self.linked_ratio_n.copy() 
        # self.linked_ratio_BLR_n[self.line_name_n=='Hb'] = 
            
    def mask_el_lite(self, enabled_comps='all'):
        self.enabled_f = np.zeros((self.num_coeffs), dtype='bool')
        if enabled_comps == 'all':
            self.enabled_f[:] = True
        else:
            for comp in enabled_comps:
                self.enabled_f[self.component_f == comp] = True
        return self.enabled_f

    def single_gaussian(self, wavelength, lamb_c_rest, voff, fwhm, flux, v0_redshift=0, spec_R_inst=1e8):
        if fwhm <= 0: raise ValueError((f"Non-positive eline fwhm: {fwhm}"))
        if flux < 0: raise ValueError((f"Negative eline flux: {flux}"))
        lamb_c_obs = lamb_c_rest * (1 + v0_redshift)
        mu =    (1 + voff/299792.458) * lamb_c_obs
        sigma_line = fwhm/299792.458  * lamb_c_obs / np.sqrt(np.log(256))
        sigma_inst = 1/spec_R_inst * lamb_c_obs / np.sqrt(np.log(256))
        sigma = np.sqrt(sigma_line**2 + sigma_inst**2)
        model = np.exp(-0.5 * ((wavelength-mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi)) 
        dw = (wavelength[1:]-wavelength[:-1]).min()
        if (model * dw).sum() < 0.10: flux = 0 # disable not well covered emission line
        return model * flux

    def models_single_comp(self, wavelength, voff, fwhm, el_AV, SII_ratio, list_valid, list_free):
        # update linked_ratio_n
        mask_Balmer = self.linked_to_n == 6564.632
        new_linked_ratio_n = self.linked_ratio_n.copy() 
        new_linked_ratio_n[mask_Balmer] *= self.extHa_base_n[mask_Balmer] ** el_AV
        # repeat for BLR:
        # new_linked_ratio_BLR_n = self.linked_ratio_BLR_n.copy() 
        # new_linked_ratio_BLR_n[mask_Balmer] *= self.extHa_base_BLR_n[mask_Balmer] ** el_AV
        i_SIIa = np.where(self.line_rest_n == 6718.295)[0][0]
        new_linked_ratio_n[i_SIIa] = SII_ratio # SII_ratio = SIIa/SIIb
        
        models_skin = []
        for i_free in list_free:
            model_scomp = self.single_gaussian(wavelength, self.line_rest_n[i_free], voff, fwhm, 
                                               1, self.v0_redshift, self.spec_R_inst) # flux=1
            list_linked = np.where(self.linked_to_n == self.line_rest_n[i_free])[0]
            list_linked = list_linked[np.isin(list_linked, list_valid)]
            for i_linked in list_linked:
                # determine new_linked_ratio_n or _BLR:
                # add later
                model_scomp += self.single_gaussian(wavelength, self.line_rest_n[i_linked], voff, fwhm, 
                                                    new_linked_ratio_n[i_linked], self.v0_redshift, self.spec_R_inst)
            models_skin.append(model_scomp)
        return models_skin
    
    def models_unitnorm_obsframe(self, wavelength, input_pars, if_pars_flat=True):
        if if_pars_flat: 
            kins_pars = self.cframe.flat_to_arr(input_pars)
        else:
            kins_pars = copy(input_pars)

        for i_kin in range(self.num_comps):
            voff, fwhm, el_AV, SII_ratio = kins_pars[i_kin]
            list_valid = np.arange(len(self.line_rest_n))[self.mask_valid_cn[i_kin,:]]
            list_free  = np.arange(len(self.line_rest_n))[self.mask_free_cn[i_kin,:]]
            models_skin = self.models_single_comp(wavelength, voff, fwhm, el_AV, SII_ratio, list_valid, list_free)
            if i_kin == 0: 
                models_mkin = models_skin
            else:
                models_mkin += models_skin
        return np.array(models_mkin)
    
#############################################################################################################
#############################################################################################################

class AGNModels(object):
    def __init__(self, filename=None, w_min=None, w_max=None, w_norm=5500, dw_norm=25, 
                 cframe=None, v0_redshift=None, spec_R_inst=None):
        # add file_bac, file_iron later
        
        self.w_min = w_min
        self.w_max = w_max
        self.w_norm = w_norm
        self.dw_norm = dw_norm
        self.cframe = cframe 
        self.v0_redshift = v0_redshift
        self.spec_R_inst = spec_R_inst
        
        # create log grid wavelength (rest) to project intrinsic template and then perform convolution
        linw_wave = np.linspace(w_min, w_max, num=10)
        resolution = spec_R_inst * 5 # select a high resampling density
        self.logw_wave = convert_linw_to_logw(linw_wave, linw_wave, resolution=resolution)[0]
        
        self.num_coeffs = 1
        
#     def read_bac(self):
#     def read_iron(self):
        # read template and project to self.logw_wave

    def powerlaw_unitnorm(self, wavelength, alpha_lambda, wave_norm=5500):
        # normalized to unit flux density (e.g.,the same unit of obs) at rest 5500AA before extinct
        # pl = (wavelength/5500)**alpha_lambda
        pl = (wavelength/wave_norm)**alpha_lambda
        
        # https://sites.google.com/site/skirtorus/sed-library, Primary source: accretion disk
        alpha_long = -3-1; wave_long = 5e4
        alpha_short1 = 0-1; wave_short1 = 0.1e4
        alpha_short2 = 1.2-1; wave_short2 = 0.01e4

        mask_w = wavelength > wave_long
        if mask_w.sum() > 0: 
            pl[mask_w] = (wavelength[mask_w]/wave_long)**alpha_long*(wave_long/wave_norm)**alpha_lambda
        mask_w = wavelength < wave_short1
        if mask_w.sum() > 0: 
            pl[mask_w] = (wavelength[mask_w]/wave_short1)**alpha_short1*(wave_short1/wave_norm)**alpha_lambda
        mask_w = wavelength < wave_short2
        if mask_w.sum() > 0: 
            pl[mask_w] = (wavelength[mask_w]/wave_short2)**alpha_short2*(wave_short2/wave_short1)**alpha_short1*(wave_short1/wave_norm)**alpha_lambda

        return pl

    def models_unitnorm_obsframe(self, wavelength, input_pars, if_pars_flat=True):
        # pars: voff, fwhm, AV; alpha_lambda (pl); add other models later 
        # comps: 'powerlaw', 'all'
        if if_pars_flat: 
            pars = self.cframe.flat_to_arr(input_pars)
        else:
            pars = copy(input_pars)

        for i_comp in range(pars.shape[0]):
            # read and append intrinsic templates in self.logw (rest)
            # powerlaw
            alpha_lambda = pars[i_comp,3]
            pl = self.powerlaw_unitnorm(self.logw_wave, alpha_lambda, wave_norm=self.w_norm)
            # Balmer continuum and high-order Balmer lines
            # iorn pseudo continuum
            # combine intrinsic agn templates in logw_wave; _mw
            models_scomp = np.vstack((pl))
            models_scomp = models_scomp.T # if only pl
            
            # dust extinction for models_scomp
            models_scomp *= 10.0**(-0.4 * pars[i_comp,2] * ExtLaw(self.logw_wave))
            # convolve with intrinsic and instrumental dispersion 
            if np.isin('iron', self.cframe.info_c[i_comp]['mod_used']): # != 'powerlaw':
                # only convolve if iron template is used (supported later)
                sigma_ssp = pars[i_comp,1] / np.sqrt(np.log(256))
                sigma_inst = 299792.458 / self.spec_R_inst / np.sqrt(np.log(256))
                models_scomp = convolve_spec_logw(self.logw_wave, models_scomp, np.sqrt(sigma_ssp**2+sigma_inst**2), axis=1)
            # redshift- or rest-wavelength does not change convolution result
            # redshift and project to observed wavelength
            z_ratio = (1 + self.v0_redshift) * (1 + pars[i_comp,0]/299792.458) # (1+z) = (1+zv0) * (1+v/c)
            models_scomp_obsframe = []
            for i_model in range(models_scomp.shape[0]):
                models_scomp_obsframe.append(np.interp(wavelength, self.logw_wave*z_ratio, models_scomp[i_model,:]/z_ratio))
            if i_comp == 0: 
                models_mcomp_obsframe = models_scomp_obsframe
            else:
                models_mcomp_obsframe += models_scomp_obsframe # not tested
        return np.array(models_mcomp_obsframe)            

#############################################################################################################
#############################################################################################################

class TorusModels(object): 
    # SKIRTor, https://sites.google.com/site/skirtorus/sed-library
    def __init__(self, filename=None, w_min=None, w_max=None, 
                 cframe=None, v0_redshift=None, flux_scale=None): # , spec_R_inst=None
        
        self.filename = filename        
        self.w_min = w_min # currently not used
        self.w_max = w_max # currently not used
        self.cframe = cframe 
        self.v0_redshift = v0_redshift
        self.flux_scale = flux_scale
        # self.spec_R_inst = spec_R_inst
        
        self.num_coeffs = 1 # only have one norm-free model in a single fitting component (since disc and torus are tied)
        self.read_skirtor()
        
    def read_skirtor(self): 
        # skirtor_disc = np.loadtxt(self.file_disc) # [n_wave_ini+6, n_tau*n_oa*n_rrat*n_incl+1]
        # skirtor_torus = np.loadtxt(self.file_dust) # [n_wave_ini+6, n_tau*n_oa*n_rrat*n_incl+1]
        skirtor_disc = fits.open(self.filename)[0].data[0]
        skirtor_torus = fits.open(self.filename)[0].data[1]

        wave = skirtor_disc[6:-1,0] # 1e-3 to 1e3 um; omit the last one with zero-value SED
        n_tau = 5; tau = np.array([ 3, 5, 7, 9, 11 ])
        n_oa = 8; oa = np.array([ 10, 20, 30, 40, 50, 60, 70, 80 ])
        n_rrat = 3; rrat = np.array([ 10, 20, 30 ])
        n_incl = 10; incl = np.array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 ])
                
        disc  = np.zeros([n_tau, n_oa, n_rrat, n_incl, len(wave)]) 
        torus = np.zeros([n_tau, n_oa, n_rrat, n_incl, len(wave)]) 
        mass  = np.zeros([n_tau, n_oa, n_rrat]) # torus dust mass
        eb    = np.zeros([n_tau, n_oa, n_rrat]) 
        # All spectra are given in erg/s/um, normalized to disc lum of 1 Lsun
        # Not that the spectra in lum unit should be considered as flux * 4pi * dl2, 
        # where flux is depended on viewing angle and 
        # the 1 Lsun normalization is integrated with anisotropic flux function.
        # Dust mass in Msun
        # eb is energy balance ratio of torus, i.e., inclination integrated Lum_torus/Lum_AGN(intrinsic)

        for i_tau in range(n_tau):
            for i_oa in range(n_oa):
                for i_rrat in range(n_rrat):
                    for i_incl in range(n_incl):
                        mask = skirtor_disc[0,:] == tau[i_tau] 
                        mask &= skirtor_disc[1,:] == oa[i_oa] 
                        mask &= skirtor_disc[2,:] == rrat[i_rrat] 
                        mask &= skirtor_disc[3,:] == incl[i_incl] 
                        mass[i_tau, i_oa, i_rrat] = skirtor_torus[4,mask][0]
                        eb[i_tau, i_oa, i_rrat] = skirtor_torus[5,mask][0]
                        disc[i_tau, i_oa, i_rrat, i_incl, :] = skirtor_disc[6:-1,mask][:,0]
                        torus[i_tau, i_oa, i_rrat, i_incl, :] = skirtor_torus[6:-1,mask][:,0]
                        # in the original library the torus sed and mass is normalized to Lum_AGN of 1 Lsun, 
                        # here renormlized them to Lum_Torus of 1e12 Lsun (i.e., Lum_AGN = 1e12 Lsun / EB_Torus) 
                        disc[i_tau, i_oa, i_rrat, i_incl, :]  *= 1e12 / eb[i_tau, i_oa, i_rrat]
                        torus[i_tau, i_oa, i_rrat, i_incl, :] *= 1e12 / eb[i_tau, i_oa, i_rrat]
                        mass[i_tau, i_oa, i_rrat] *= 1e12 / eb[i_tau, i_oa, i_rrat]
        
        # convert unit: 1 erg/s/um -> flux_scale * erg/s/AA/cm2
        wave *= 1e4
        lum_dist = cosmo.luminosity_distance(self.v0_redshift).to('cm').value
        lum_area = 4*np.pi * lum_dist**2 # in cm2
        disc *= 1e-4 / lum_area / self.flux_scale
        torus *= 1e-4 / lum_area / self.flux_scale
        disc[disc <= 0]   = disc[disc>0].min()
        torus[torus <= 0] = torus[torus>0].min()
        
        # for interpolation
        ini_pars = (tau, oa, rrat, incl, np.log10(wave))    
        fun_logdisc = RegularGridInterpolator(ini_pars, np.log10(disc), method='linear', bounds_error=False)
        fun_logtorus = RegularGridInterpolator(ini_pars, np.log10(torus), method='linear', bounds_error=False)
        # set bounds_error=False to avoid error by slight exceeding of x-val generated by least_square func
        # but do not use pars outside of initial range
        ini_pars = (tau, oa, rrat)    
        fun_mass = RegularGridInterpolator(ini_pars, mass, method='linear', bounds_error=False)
        fun_eb = RegularGridInterpolator(ini_pars, eb, method='linear', bounds_error=False)

        self.skirtor = {'tau':tau, 'oa':oa, 'rratio':rrat, 'incl':incl, 
                        'wave':wave, 'logwave':np.log10(wave), 
                        'disc':disc, 'fun_logdisc':fun_logdisc, 
                        'torus':torus, 'fun_logtorus':fun_logtorus, 
                        'mass':mass, 'fun_mass':fun_mass, 
                        'eb':eb, 'fun_eb':fun_eb } 
        
    def get_info(self, tau, oa, rratio):
        fun_mass = self.skirtor['fun_mass']
        fun_eb = self.skirtor['fun_eb']
        gen_pars = np.array([tau, oa, rratio])
        gen_mass = fun_mass(gen_pars)
        gen_eb   = fun_eb(gen_pars)
        return gen_mass, gen_eb

    def models_unitnorm_obsframe(self, wavelength, input_pars, if_pars_flat=True):
        # pars: voff (to adjust redshift), tau, oa, rratio, incl
        # comps: 'disc', 'torus', 'disc+torus'
        if if_pars_flat: 
            pars = self.cframe.flat_to_arr(input_pars)
        else:
            pars = copy(input_pars)

        for i_comp in range(pars.shape[0]):
            tau, oa, rratio, incl = pars[i_comp,:][1:5]
            
            # interpolate model for given pars in initial wavelength (rest)
            ini_logwave = self.skirtor['logwave'].copy()
            fun_logdisc = self.skirtor['fun_logdisc']
            fun_logtorus = self.skirtor['fun_logtorus']
            gen_pars = np.array([[tau, oa, rratio, incl, w] for w in ini_logwave]) # gen: generated
            if np.isin('disc', self.cframe.info_c[i_comp]['mod_used']):
                gen_logdisc = fun_logdisc(gen_pars)
            if np.isin('dust', self.cframe.info_c[i_comp]['mod_used']):
                gen_logtorus = fun_logtorus(gen_pars)    

            # redshifted to obs-frame
            ret_logwave = np.log10(wavelength) # in AA
            z_ratio = (1 + self.v0_redshift) * (1 + pars[i_comp,0]/299792.458) # (1+z) = (1+zv0) * (1+v/c)            
            ini_logwave += np.log10(z_ratio)
            if np.isin('disc', self.cframe.info_c[i_comp]['mod_used']):
                gen_logdisc -= np.log10(z_ratio)
                ret_logdisc  = np.interp(ret_logwave, ini_logwave, gen_logdisc, 
                                         left=np.minimum(gen_logdisc.min(),-100), right=np.minimum(gen_logdisc.min(),-100))
            if np.isin('dust', self.cframe.info_c[i_comp]['mod_used']):
                gen_logtorus -= np.log10(z_ratio)
                ret_logtorus = np.interp(ret_logwave, ini_logwave, gen_logtorus, 
                                         left=np.minimum(gen_logtorus.min(),-100), right=np.minimum(gen_logtorus.min(),-100))
                
            # extended to longer wavelength
            mask_w = ret_logwave > ini_logwave[-1]
            if np.sum(mask_w) > 0:
                if np.isin('disc', self.cframe.info_c[i_comp]['mod_used']):
                    index = (gen_logdisc[-2]-gen_logdisc[-1]) / (ini_logwave[-2]-ini_logwave[-1])
                    ret_logdisc[mask_w] = gen_logdisc[-1] + index * (ret_logwave[mask_w]-ini_logwave[-1])
                if np.isin('dust', self.cframe.info_c[i_comp]['mod_used']):
                    index = (gen_logtorus[-2]-gen_logtorus[-1]) / (ini_logwave[-2]-ini_logwave[-1])
                    ret_logtorus[mask_w] = gen_logtorus[-1] + index * (ret_logwave[mask_w]-ini_logwave[-1])
                    
            if np.isin('disc', self.cframe.info_c[i_comp]['mod_used']):
                ret_disc = 10.0**ret_logdisc
                ret_disc[ret_logdisc <= -100] = 0
            if np.isin('dust', self.cframe.info_c[i_comp]['mod_used']):
                ret_torus = 10.0**ret_logtorus
                ret_torus[ret_logtorus <= -100] = 0
                
            models_scomp_obsframe = np.zeros_like(ret_logwave)
            if np.isin('disc', self.cframe.info_c[i_comp]['mod_used']): models_scomp_obsframe += ret_disc
            if np.isin('dust', self.cframe.info_c[i_comp]['mod_used']): models_scomp_obsframe += ret_torus
                
            models_scomp_obsframe = np.vstack((models_scomp_obsframe))
            models_scomp_obsframe = models_scomp_obsframe.T # add .T for a uniform format with other models with n_coeffs > 1
            
            if i_comp == 0: 
                models_mcomp_obsframe = models_scomp_obsframe
            else:
                models_mcomp_obsframe += models_scomp_obsframe # not tested
        return np.array(models_mcomp_obsframe)           
#############################################################################################################
