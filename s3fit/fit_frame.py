# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Contact: xiaoyang.chen.cz@gmail.com

import time, traceback
import numpy as np
from copy import deepcopy as copy
from scipy.optimize import least_squares, lsq_linear
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from .config_frame import ConfigFrame
from .phot_frame import PhotFrame
# from .model_frames import *
from .auxiliary_func import print_time

class FitFrame(object):
    def __init__(self, 
                 spec_wave_w=None, spec_flux_w=None, spec_ferr_w=None, 
                 spec_valid_range=None, spec_R_inst=None, spec_flux_scale=None, 
                 phot_name_b=None, phot_flux_b=None, phot_ferr_b=None, phot_fluxunit='mJy', 
                 phot_trans_dir=None, sed_wave_w=None, sed_waveunit='angstrom', 
                 v0_redshift=None, model_config=None, 
                 num_mock_loops=0, fit_raw=True, fit_grid='linear', 
                 plot_step=False, print_step=True, verbose=False): 
        
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

        # set mask_valid_w, which is used to select covered emission lines
        self.set_masks() 

        # import all available models
        self.load_models(model_config)

        # set mask_noeline_w based on line list of emission line models
        self.set_masks(eline_waves=self.model_dict['el']['specmod'].linerest_n, eline_vwin=[-4000,2000]) 

        # set fitting boundaries 
        self.set_parameter_constraints()

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
                from .model_frames.ssp_frame import SSPFrame
                self.model_dict[mod]['specmod'] = SSPFrame(filename=model_config[mod]['file'], w_min=self.spec_wmin, w_max=self.spec_wmax, 
                                                            cframe=self.model_dict[mod]['cf'], v0_redshift=self.v0_redshift, spec_R_inst=self.spec_R_inst) 
                if self.have_phot:
                    self.model_dict[mod]['sedmod'] = SSPFrame(filename=model_config[mod]['file'], w_min=self.sed_wmin, w_max=self.sed_wmax, 
                                                               cframe=self.model_dict[mod]['cf'], v0_redshift=self.v0_redshift, 
                                                               spec_R_inst=None, spec_R_init=300, verbose=False)
        mod = 'el'
        if np.isin(mod, [*model_config]):
            if model_config[mod]['enable']:             
                self.full_model_type += mod + '+'
                self.model_dict[mod] = {'cf': ConfigFrame(model_config[mod]['config'])}
                from .model_frames.eline_frame import ELineFrame
                self.model_dict[mod]['specmod'] = ELineFrame(rest_wave_w=self.spec['wave_w']/(1+self.v0_redshift), mask_valid_w=self.spec['mask_valid_w'],
                                                              cframe=self.model_dict[mod]['cf'], v0_redshift=self.v0_redshift, spec_R_inst=self.spec_R_inst, 
                                                              use_pyneb=model_config[mod]['use_pyneb']) 
                if self.have_phot:
                    self.model_dict[mod]['sedmod'] = ELineFrame(rest_wave_w=self.sed['wave_w']/(1+self.v0_redshift), mask_valid_w=None, 
                                                                 cframe=self.model_dict[mod]['cf'], v0_redshift=self.v0_redshift, spec_R_inst=self.spec_R_inst, 
                                                                 use_pyneb=model_config[mod]['use_pyneb'], verbose=False)
        mod = 'agn'
        if np.isin(mod, [*model_config]):
            if model_config[mod]['enable']: 
                self.full_model_type += mod + '+'
                self.model_dict[mod] = {'cf': ConfigFrame(model_config[mod]['config'])}
                from .model_frames.agn_frame import AGNFrame
                self.model_dict[mod]['specmod'] = AGNFrame(w_min=self.spec_wmin, w_max=self.spec_wmax, 
                                                            cframe=self.model_dict[mod]['cf'], v0_redshift=self.v0_redshift, spec_R_inst=self.spec_R_inst) 
                if self.have_phot:
                    self.model_dict[mod]['sedmod'] = AGNFrame(w_min=self.sed_wmin, w_max=self.sed_wmax, 
                                                               cframe=self.model_dict[mod]['cf'], v0_redshift=self.v0_redshift, spec_R_inst=self.spec_R_inst) 
        mod = 'torus'
        if np.isin(mod, [*model_config]):
            if model_config[mod]['enable']: 
                self.full_model_type += mod + '+'
                self.model_dict[mod] = {'cf': ConfigFrame(model_config[mod]['config'])}
                from .model_frames.torus_frame import TorusFrame
                self.model_dict[mod]['specmod'] = TorusFrame(filename=model_config[mod]['file'], 
                                                              cframe=self.model_dict[mod]['cf'], v0_redshift=self.v0_redshift, flux_scale=self.spec_flux_scale) 
                if self.have_phot:
                    self.model_dict[mod]['sedmod'] = self.model_dict[mod]['specmod'] # just copy                
        
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
            mask_valid_w |= (spec_wave_w >= waveslot[0]) & (spec_wave_w <= waveslot[1])
        self.spec['mask_valid_w'] = mask_valid_w
        # print('Valid data wavelength range:', self.spec_valid_range)
        
        # eline mask from rest frame line position
        if eline_waves is not None:
            mask_eline_w = np.zeros_like(spec_wave_w, dtype='bool')
            for i_eline in range(len(eline_waves)):
                eline_wave = eline_waves[i_eline] * (1 + self.v0_redshift) * (1+np.array(eline_vwin)/299792.458)
                mask_eline_w |= (spec_wave_w >= eline_wave[0]) & (spec_wave_w <= eline_wave[1])
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
        for mod in self.full_model_type.split('+'):
            self.model_dict[mod]['specmod'].output_results(ff=self)

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
