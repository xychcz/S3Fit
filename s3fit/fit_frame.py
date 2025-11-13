# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: s3fit@xychen.me

import sys, time, traceback, inspect, pickle, gzip
import numpy as np
np.set_printoptions(linewidth=10000)
from copy import deepcopy as copy
from scipy.optimize import lsq_linear, least_squares, dual_annealing
from scipy.signal import savgol_filter
# from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from .config_frame import ConfigFrame
from .phot_frame import PhotFrame
# from .model_frames import *
from .auxiliary_func import print_log, center_string, convolve_var_width_fft

class FitFrame(object):
    def __init__(self, 
                 spec_wave_w=None, spec_flux_w=None, spec_ferr_w=None, 
                 spec_R_inst_w=None, spec_valid_range=None, spec_flux_scale=None, 
                 phot_name_b=None, phot_flux_b=None, phot_ferr_b=None, 
                 phot_calib_b=None, phot_flux_unit='mJy', 
                 phot_trans_dir=None, phot_trans_rsmp=10, 
                 sed_wave_w=None, sed_wave_unit='angstrom', sed_wave_num=None, 
                 v0_redshift=None, model_config=None, norm_wave=5500, norm_width=25, 
                 num_mocks=0, 
                 inst_calib_ratio=0.1, inst_calib_ratio_rev=True, inst_calib_smooth=1e4, 
                 examine_result=True, accept_model_SN=2, 
                 accept_chi_sq=3, nlfit_ntry_max=3, 
                 init_annealing=True, da_niter_max=10, perturb_scale=0.02, nllsq_ftol_ratio=0.01, 
                 fit_grid='linear', conv_nbin_max=5, R_mod_ratio=2, 
                 print_step=True, plot_step=False, canvas=None, 
                 save_per_loop=False, output_filename=None, 
                 save_test=False, verbose=False): 

        input_args = {k: v for k, v in locals().items() if k != "self"}
        if not np.array([isinstance(input_args[k], np.ndarray) for i, k in enumerate(input_args)]).any():
            if np.array([input_args[k] == self.__init__.__defaults__[i] for i, k in enumerate(input_args)]).all():
                print('[Note] Please input arguments or use FitFrame.reload() to initialize FitFrame.')
                return

        # use a list to save message in stdout
        self.log_message = []
        print_log(center_string('S3Fit starts', 80), self.log_message)
        print_log(f"You are now using S3Fit v2.3.", self.log_message)

        print_log(center_string('Initialize FitFrame', 80), self.log_message)
        # save spectral data and related properties
        self.spec_wave_w = np.array(spec_wave_w) if spec_wave_w is not None else None
        self.spec_flux_w = np.array(spec_flux_w) if spec_flux_w is not None else None
        self.spec_ferr_w = np.array(spec_ferr_w) if spec_ferr_w is not None else None
        self.spec_R_inst_w = np.array(spec_R_inst_w) if spec_R_inst_w is not None else None
        self.spec_valid_range = spec_valid_range
        self.spec_flux_scale = spec_flux_scale # flux_scale is used to avoid too small values

        # save photometric-SED data and related properties
        self.phot_name_b = np.array(phot_name_b) if phot_name_b is not None else None
        self.phot_flux_b = np.array(phot_flux_b) if phot_flux_b is not None else None
        self.phot_ferr_b = np.array(phot_ferr_b) if phot_ferr_b is not None else None
        self.phot_calib_b = np.array(phot_calib_b) if phot_calib_b is not None else None
        self.phot_flux_unit = phot_flux_unit
        self.phot_trans_dir = phot_trans_dir
        self.phot_trans_rsmp = phot_trans_rsmp
        self.sed_wave_w = np.array(sed_wave_w) if sed_wave_w is not None else None
        self.sed_wave_unit = sed_wave_unit
        self.sed_wave_num = sed_wave_num

        # set initial guess of systemic redshift, all velocity shifts are in relative to v0_redshift
        self.v0_redshift = v0_redshift
        # load model configuration
        self.model_config = copy(model_config) # use copy to avoid changing the input config
        # set wavelength and width (in AA) used to normalize model spectra
        self.norm_wave = norm_wave
        self.norm_width = norm_width

        # number of mock data
        self.num_mocks = num_mocks
        print_log(f"Perform fitting for the original data and {self.num_mocks} mock data.", self.log_message)

        # initial ratio to estimate calibration error
        self.inst_calib_ratio = inst_calib_ratio
        self.inst_calib_ratio_rev = inst_calib_ratio_rev
        # smoothing width when creating modified error
        self.inst_calib_smooth = inst_calib_smooth

        # control on fitting quality: fitting steps
        self.examine_result = examine_result 
        self.accept_model_SN = accept_model_SN
        if self.examine_result: 
            print_log(f"All continuum models and line components with peak S/N < {self.accept_model_SN} (set with 'accept_model_SN') will be automatically disabled in examination.", self.log_message)
        else:
            print_log(f"[Note] The examination of S/N of models and the updating of fitting will be skipped since 'examine_result' is set to False.", self.log_message)
        # control on fitting quality: nonlinear process, general
        self.accept_chi_sq = accept_chi_sq
        self.nlfit_ntry_max = nlfit_ntry_max 
        # control on fitting quality: nonlinear process, dual annealing
        self.init_annealing = init_annealing
        self.da_niter_max = da_niter_max
        # control on fitting quality: nonlinear process, parameter pertrubation
        self.perturb_scale = perturb_scale 
        # control on fitting quality: nonlinear process, nonlinear least-square
        self.nllsq_ftol_ratio = nllsq_ftol_ratio
        # control on fitting quality: linear process, grid
        self.fit_grid = fit_grid
        print_log(f"Perform fitting in {self.fit_grid} space.", self.log_message)
        if self.fit_grid == 'log':
            print_log(f"[Note] Pure line fitting (i.e., after subtracting continuum), if enabled, is always in linear space.", self.log_message)
        # control on fitting quality: linear process, maximum of bins to perform fft convolution with variable width/resolution
        self.conv_nbin_max = conv_nbin_max
        # control on fitting quality: the value equals the ratio of resolution of model (downsampled) / instrument
        self.R_mod_ratio = R_mod_ratio

        # whether to output intermediate results
        self.print_step = print_step # if display in stdout
        self.plot_step = plot_step # if display in matplotlib window
        self.canvas = canvas # canvas=(fig,ax), to display plots dynamically
        self.save_per_loop = save_per_loop
        self.output_filename = output_filename
        self.save_test = save_test # if save the iteration tracing for test
        self.verbose = verbose 

        arg_list = list(inspect.signature(self.__init__).parameters.values())
        self.input_args = {arg.name: getattr(self,arg.name,None) for arg in arg_list if arg.name != 'self'}
        # FitFrame class can be copied as FF_1 = FitFrame(**FF_0.input_args)

        # initialize input formats
        self.init_input_data()
        # import all available models; should be after set_masks to select covered lines
        self.load_models()
        # set constraints of fitting parameters 
        self.set_par_constraints()
        # initialize output formats; should be after set_par_constraints
        self.init_output_results()

        print_log(center_string('Initialization finishes', 80), self.log_message)

    def init_input_data(self, verbose=True):
        print_log(center_string('Read spectral data', 80), self.log_message, verbose)

        # mask out invalid data
        mask_valid_w  = np.isfinite(self.spec_wave_w)
        mask_valid_w &= np.isfinite(self.spec_flux_w)
        mask_valid_w &= np.isfinite(self.spec_ferr_w)
        if len(self.spec_R_inst_w) == len(self.spec_wave_w): mask_valid_w &= np.isfinite(self.spec_R_inst_w)
        if (~mask_valid_w).sum() > 0:
            print_log(f"Mask out {(~mask_valid_w).sum()} from {len(mask_valid_w)} data points with NaN or Inf values in the input spec_wave_w, spec_flux_w, spec_ferr_w, or spec_R_inst_w.", 
                    self.log_message, verbose)
            # only keep valid points to shrink stored array length
            self.spec_wave_w = self.spec_wave_w[mask_valid_w]
            self.spec_flux_w = self.spec_flux_w[mask_valid_w]
            self.spec_ferr_w = self.spec_ferr_w[mask_valid_w]
            if len(self.spec_R_inst_w) == len(self.spec_wave_w): self.spec_R_inst_w = self.spec_R_inst_w[mask_valid_w]
            mask_valid_w = mask_valid_w[mask_valid_w]

        # set valid wavelength range
        if self.spec_valid_range is not None:
            mask_specified_w = np.zeros_like(self.spec_wave_w, dtype='bool')
            for i_waveslot in range(len(self.spec_valid_range)):
                waveslot = self.spec_valid_range[i_waveslot]
                mask_specified_w |= (self.spec_wave_w >= waveslot[0]) & (self.spec_wave_w <= waveslot[1])
            print_log(f"Mask out {(~mask_specified_w).sum()} data points with the input spec_valid_range.", self.log_message, verbose)
            mask_valid_w &= mask_specified_w

        num_invalid_ferr = (mask_valid_w & (self.spec_ferr_w <= 0)).sum()
        if num_invalid_ferr > 0: 
            print_log(f"Mask out additional {num_invalid_ferr} wavelengths with non-positive spec_ferr_w.", self.log_message, verbose)
            mask_valid_w &= (self.spec_ferr_w > 0)


        if self.spec_flux_scale is None: 
            self.spec_flux_scale = 10.0**np.round(np.log10(np.median(self.spec_flux_w)))
        # create a dictionary for spectral data
        self.spec = {}
        self.spec['wave_w'] = self.spec_wave_w
        self.spec['flux_w'] = self.spec_flux_w / self.spec_flux_scale
        self.spec['ferr_w'] = self.spec_ferr_w / self.spec_flux_scale
        self.spec['mask_valid_w'] = mask_valid_w
        self.num_spec_wave = len(self.spec_wave_w)

        # check spectral resoltuion
        if len(self.spec_R_inst_w) == len(self.spec_wave_w):
            self.spec['R_inst_rw'] = np.vstack((self.spec_wave_w, self.spec_R_inst_w))
        else:
            print_log(f'[Note] A single value of spectral resolution {self.spec_R_inst_w[1]:.3f} is given at {self.spec_R_inst_w[0]:.3f}AA.', 
                      self.log_message, verbose)
            print_log(f'[Note] Assume a linear wavelength-dependency of spectral resolution in the fitting.', self.log_message, verbose)
            lin_R_inst_w = self.spec_wave_w / self.spec_R_inst_w[0] * self.spec_R_inst_w[1]
            self.spec['R_inst_rw'] = np.vstack((self.spec_wave_w, lin_R_inst_w))

        # account for effective spectral sampling in fitting (all bands are considered as independent)
        self.spec['significance_w'] = np.gradient(self.spec_wave_w) / (self.spec_wave_w/self.spec['R_inst_rw'][1,:]) # i.e., dw_data / dw_resolution
        self.spec['significance_w'][self.spec['significance_w'] > 1] = 1

        # set fitting wavelength range (rest frame) with tolerance of [-1000,1000] km/s
        vel_tol = 1000
        self.spec_wmin = self.spec_wave_w.min() / (1+self.v0_redshift) / (1+vel_tol/299792.458) - 100
        self.spec_wmax = self.spec_wave_w.max() / (1+self.v0_redshift) / (1-vel_tol/299792.458) + 100
        self.spec_wmin = np.maximum(self.spec_wmin, 912) # set lower limit of wavelength to 912A
        print_log(f'Spectral fitting will be performed in wavelength range (rest frame, AA): from {self.spec_wmin:.3f} to {self.spec_wmax:.3f}', self.log_message, verbose)
        print_log(f'[Note] The wavelength range is extended for a tolerance of redshift of {self.v0_redshift}+-{vel_tol/299792.458:.4f} (+-{vel_tol} km/s).', self.log_message, verbose)

        # check if norm_wave is coverd in input wavelength range
        if (self.norm_wave < self.spec_wmin) | (self.norm_wave > self.spec_wmax):
            med_wave = np.median(self.spec_wave_w[mask_valid_w]) / (1+self.v0_redshift)
            med_wave = round(med_wave/100)*100
            print_log(f'[WARNING] The input normalization wavelength (rest frame, AA) {self.norm_wave} is out of the valid range, which is forced to the median valid wavelength {med_wave}.', 
                      self.log_message, verbose)
            self.norm_wave = med_wave

        # create a dictionary for photometric-SED data
        self.have_phot = True if self.phot_name_b is not None else False
        if self.have_phot:
            print_log(center_string('Read photometric data', 80), self.log_message, verbose)
            print_log(f'Data available in bands: {self.phot_name_b}', self.log_message, verbose)
            # convert input photometric data
            self.pframe = PhotFrame(name_b=self.phot_name_b, flux_b=self.phot_flux_b, ferr_b=self.phot_ferr_b, flux_unit=self.phot_flux_unit,
                                    trans_dir=self.phot_trans_dir, trans_rsmp=self.phot_trans_rsmp, 
                                    wave_w=self.sed_wave_w, wave_unit=self.sed_wave_unit, wave_num=self.sed_wave_num)
            # create a dictionary for converted photometric data
            self.phot = {}
            self.phot['wave_b'] = self.pframe.wave_b
            self.phot['flux_b'] = self.pframe.flux_b / self.spec_flux_scale
            self.phot['ferr_b'] = self.pframe.ferr_b / self.spec_flux_scale
            self.phot['trans_bw'] = self.pframe.trans_bw # transmission curve matrix
            self.num_phot_band = len(self.pframe.wave_b)

            # set valid band range
            self.phot['mask_valid_b'] = self.pframe.ferr_b > 0
            # account for effective sampling in fitting, all bands are considered as independent
            self.phot['significance_b'] = np.ones(self.num_phot_band, dtype='float')

            # create self.sed to save the full SED covering all bands
            self.sed = {'wave_w': self.pframe.wave_w} 
            self.num_sed_wave = len(self.pframe.wave_w)
            # set fitting wavelength range (rest frame)
            self.sed_wmin = self.pframe.wave_w.min() / (1+self.v0_redshift)
            self.sed_wmax = self.pframe.wave_w.max() / (1+self.v0_redshift)
            print_log(f'SED fitting is performed in wavelength range (rest frame, AA): from {self.sed_wmin:.3f} to {self.sed_wmax:.3f}', self.log_message, verbose) 

            if self.phot_calib_b is not None:
                # corrent spectrum based on selected photometeic points
                # select fluxes and transmission curves in calibration bands in the order of phot_calib_b
                calib_flux_b = [self.phot['flux_b'][np.where(self.phot_name_b == name_b)[0][0]] for name_b in self.phot_calib_b]
                calib_trans_bw = self.pframe.read_transmission(name_b=self.phot_calib_b, trans_dir=self.phot_trans_dir, wave_w=self.spec['wave_w'])[1]
                # interplote spectrum by masking out invalid range
                spec_flux_interp_w = np.interp(self.spec['wave_w'], self.spec['wave_w'][self.spec['mask_valid_w']], 
                                                                    self.spec['flux_w'][self.spec['mask_valid_w']])
                spec_calib_ratio_b = calib_flux_b / self.pframe.spec2phot(self.spec['wave_w'], spec_flux_interp_w, calib_trans_bw)
                self.spec_calib_ratio = spec_calib_ratio_b.mean()
                self.spec['flux_w'] *= self.spec_calib_ratio
                self.spec['ferr_w'] *= self.spec_calib_ratio
                print_log(f'[Note] The input spectrum is calibrated with photometric fluxes in the bands: {self.phot_calib_b}.', self.log_message, verbose)
                print_log(f'[Note] The calibration ratio for spectrum is {self.spec_calib_ratio}.', self.log_message, verbose)

        self.input_initialized = True # used to mark if input data is modified in following steps

    def load_models(self):
        # models init setup
        self.full_model_type = ''
        self.model_dict = {}

        ###############################
        mod = 'ssp'
        if np.isin(mod, [*self.model_config]):
            if self.model_config[mod]['enable']: 
                print_log(center_string('Initialize stellar continuum models', 80), self.log_message)
                self.full_model_type += mod + '+'
                self.model_dict[mod] = {'cframe': ConfigFrame(self.model_config[mod]['config'])}
                from .model_frames.ssp_frame import SSPFrame
                self.model_dict[mod]['spec_mod'] = SSPFrame(cframe=self.model_dict[mod]['cframe'], v0_redshift=self.v0_redshift, R_inst_rw=self.spec['R_inst_rw'], 
                                                            filename=self.model_config[mod]['file'], w_min=self.spec_wmin, w_max=self.spec_wmax, w_norm=self.norm_wave, dw_norm=self.norm_width, 
                                                            R_mod_ratio=self.R_mod_ratio, log_message=self.log_message) 
                self.model_dict[mod]['spec_enable'] = (self.spec_wmax > 912) & (self.spec_wmin < 1e5)
                if self.have_phot:
                    self.model_dict[mod]['sed_mod'] = SSPFrame(cframe=self.model_dict[mod]['cframe'], v0_redshift=self.v0_redshift, R_inst_rw=None, 
                                                               filename=self.model_config[mod]['file'], w_min=self.sed_wmin, w_max=self.sed_wmax, w_norm=self.norm_wave, dw_norm=self.norm_width, 
                                                               ds_fwhm_wave=4000/100, verbose=False) # convolving with R=100 at rest 4000AA
                    self.model_dict[mod]['sed_enable'] = (self.sed_wmax > 912) & (self.sed_wmin < 1e5)
        ###############################
        mod = 'line'
        if np.isin([mod, 'el'], [*self.model_config]).any(): # also allow 'el' in model_config to be compatible with old version 
            if self.model_config[mod]['enable']:
                print_log(center_string('Initialize line models', 80), self.log_message)
                self.full_model_type += mod + '+'
                self.model_dict[mod] = {'cframe': ConfigFrame(self.model_config[mod]['config'])}
                from .model_frames.line_frame import LineFrame
                self.model_dict[mod]['spec_mod'] = LineFrame(cframe=self.model_dict[mod]['cframe'], v0_redshift=self.v0_redshift, R_inst_rw=self.spec['R_inst_rw'], 
                                                             rest_wave_w=self.spec['wave_w']/(1+self.v0_redshift), mask_valid_w=self.spec['mask_valid_w'], 
                                                             use_pyneb=self.model_config[mod]['use_pyneb'], log_message=self.log_message) 
                self.model_dict[mod]['spec_enable'] = (self.spec_wmax > 912) & (self.spec_wmin < 1e7)
                if self.have_phot:
                    self.model_dict[mod]['sed_mod'] = self.model_dict[mod]['spec_mod'] # just copy, only fit lines in spectral wavelength range
                    self.model_dict[mod]['sed_enable'] = (self.sed_wmax > 912) & (self.sed_wmin < 1e7)
                self.model_dict['el'] = self.model_dict[mod] # also allow 'el' in model_config to be compatible with old version 
        ###############################
        mod = 'agn'
        if np.isin(mod, [*self.model_config]):
            if self.model_config[mod]['enable']: 
                print_log(center_string('Initialize AGN continuum models', 80), self.log_message)
                self.full_model_type += mod + '+'
                self.model_dict[mod] = {'cframe': ConfigFrame(self.model_config[mod]['config'])}
                from .model_frames.agn_frame import AGNFrame
                self.model_dict[mod]['spec_mod'] = AGNFrame(cframe=self.model_dict[mod]['cframe'], v0_redshift=self.v0_redshift, R_inst_rw=self.spec['R_inst_rw'], 
                                                            filename=self.model_config[mod]['file'], w_min=self.spec_wmin, w_max=self.spec_wmax, w_norm=self.norm_wave, dw_norm=self.norm_width, 
                                                            R_mod_ratio=self.R_mod_ratio, log_message=self.log_message) 
                self.model_dict[mod]['spec_enable'] = (self.spec_wmax > 912) & (self.spec_wmin < 1e5)
                if self.have_phot:
                    self.model_dict[mod]['sed_mod'] = AGNFrame(cframe=self.model_dict[mod]['cframe'], v0_redshift=self.v0_redshift, R_inst_rw=None, 
                                                               filename=self.model_config[mod]['file'], w_min=self.sed_wmin, w_max=self.sed_wmax, w_norm=self.norm_wave, dw_norm=self.norm_width, 
                                                               ds_fwhm_wave=4000/100, verbose=False) # convolving with R=100 at rest 4000AA
                    self.model_dict[mod]['sed_enable'] = (self.sed_wmax > 912) & (self.sed_wmin < 1e5)
        ###############################
        mod = 'torus'
        if np.isin(mod, [*self.model_config]):
            if self.model_config[mod]['enable']: 
                print_log(center_string('Initialize AGN torus models', 80), self.log_message)
                self.full_model_type += mod + '+'
                self.model_dict[mod] = {'cframe': ConfigFrame(self.model_config[mod]['config'])}
                from .model_frames.torus_frame import TorusFrame
                self.model_dict[mod]['spec_mod'] = TorusFrame(cframe=self.model_dict[mod]['cframe'], v0_redshift=self.v0_redshift, 
                                                              filename=self.model_config[mod]['file'], flux_scale=self.spec_flux_scale, log_message=self.log_message) 
                self.model_dict[mod]['spec_enable'] = (self.spec_wmax > 1e4) & (self.spec_wmin < 1e6)
                if self.have_phot:
                    self.model_dict[mod]['sed_mod'] = self.model_dict[mod]['spec_mod'] # just copy
                    self.model_dict[mod]['sed_enable'] = (self.sed_wmax > 1e4) & (self.sed_wmin < 1e6)
        ###############################
        
        if self.full_model_type[-1] == '+': self.full_model_type = self.full_model_type[:-1]
        print_log(center_string('Model summary', 80), self.log_message)
        print_log(f'The fitting will be performed with these models: {self.full_model_type}.', self.log_message)
        for mod in self.full_model_type.split('+'):
            if self.have_phot:
                if not (self.model_dict[mod]['spec_enable'] | self.model_dict[mod]['sed_enable']): 
                    print_log(f"'{mod}' model will not be enabled in the fitting since the defined wavelength range "+
                              f"is not covered by both of the input spectrum and photometric-SED.", self.log_message)
            else:
                if not self.model_dict[mod]['spec_enable']: 
                    print_log(f"'{mod}' model will not be enabled in the spectral fitting since the defined wavelength range "+
                              f"is not covered by the input spectrum.", self.log_message)

        # add short name for returning function of model spectra
        for mod in self.full_model_type.split('+'):
            self.model_dict[mod]['spec_func'] = self.model_dict[mod]['spec_mod'].models_unitnorm_obsframe
            if self.have_phot:
                self.model_dict[mod]['sed_func'] = self.model_dict[mod]['sed_mod'].models_unitnorm_obsframe

        # create non-line mask if line is enabled
        if np.isin('line', self.full_model_type.split('+')): 
            line_center_n = self.model_dict['line']['spec_mod'].linerest_default * (1 + self.v0_redshift)
            vel_win = np.array([-3000, 3000])
            mask_line_w = np.zeros_like(self.spec['wave_w'], dtype='bool')
            for i_line in range(len(line_center_n)):
                line_bounds = line_center_n[i_line] * (1 + vel_win/299792.458)
                mask_line_w |= (self.spec['wave_w'] >= line_bounds[0]) & (self.spec['wave_w'] <= line_bounds[1])
            self.spec['mask_noline_w'] = self.spec['mask_valid_w'] & (~mask_line_w)
        else:
            self.spec['mask_noline_w'] = copy(self.spec['mask_valid_w'])

    def set_par_constraints(self):
        self.num_tot_pars = 0
        self.num_tot_coeffs = 0
        self.tie_p = np.array([])
        self.bound_min_p = np.array([])
        self.bound_max_p = np.array([])
        for mod in self.full_model_type.split('+'):
            self.model_dict[mod]['num_pars'] = self.model_dict[mod]['cframe'].num_pars
            self.model_dict[mod]['num_coeffs'] = self.model_dict[mod]['spec_mod'].num_coeffs
            self.num_tot_pars += self.model_dict[mod]['num_pars']
            self.num_tot_coeffs += self.model_dict[mod]['num_coeffs']
            self.tie_p = np.hstack((self.tie_p, self.model_dict[mod]['cframe'].tie_cp.flatten())) 
            self.bound_min_p = np.hstack((self.bound_min_p, self.model_dict[mod]['cframe'].min_cp.flatten())) 
            self.bound_max_p = np.hstack((self.bound_max_p, self.model_dict[mod]['cframe'].max_cp.flatten()))

        # update bounds to match requirement of fitting fucntion, but these values will not be indeed used 
        for i_p in range(len(self.tie_p)):
            if self.tie_p[i_p] == 'free': 
                continue
            else:
                if self.tie_p[i_p] == 'fix': 
                    self.bound_max_p[i_p] = self.bound_min_p[i_p] + 1.0 # actually not used, the value will be forced to bound_min_p; 1e-8
                else:
                    for single_tie in self.tie_p[i_p].split(';'):
                        ref_mod, ref_comp, ref_i_par = single_tie.split(':')
                        if np.isin(ref_mod, self.full_model_type.split('+')):
                            ref_num_pars_per_comp = self.model_dict[ref_mod]['cframe'].num_pars_per_comp
                            ref_i_comp = np.where(np.array(self.model_dict[ref_mod]['cframe'].comp_c) == ref_comp)[0]
                            if len(ref_i_comp) == 1:
                                ref_i_comp = ref_i_comp[0]
                            else:
                                raise ValueError((f"The reference component: {ref_comp} is not available in {self.model_dict[ref_mod]['cframe'].comp_c}"))
                            ref_i_x = ref_num_pars_per_comp*ref_i_comp + int(ref_i_par)
                            fp0, fp1 = self.search_model_index(ref_mod, self.full_model_type)[0:2]
                            if np.isnan(self.bound_min_p[i_p]):
                                self.bound_min_p[i_p] = self.bound_min_p[fp0:fp1][ref_i_x] 
                                self.bound_max_p[i_p] = self.bound_max_p[fp0:fp1][ref_i_x] 
                            else: 
                                self.bound_min_p[i_p] = np.minimum(self.bound_min_p[i_p], self.bound_min_p[fp0:fp1][ref_i_x])
                                self.bound_max_p[i_p] = np.maximum(self.bound_max_p[i_p], self.bound_max_p[fp0:fp1][ref_i_x])
                        else:
                            raise ValueError((f"The reference model {ref_mod} is not provided."))
        self.bound_width_p = self.bound_max_p - self.bound_min_p

    def init_output_results(self):
        self.num_loops = self.num_mocks+1
        # format to save fitting quality, chi_sq, parameters, and coefficients (normalization factors) of final best-fits    
        self.output_s = {}
        self.output_s['empty_step'] = {}
        self.output_s['empty_step']['chi_sq_l'] = np.zeros(self.num_loops, dtype='float')
        self.output_s['empty_step']['par_lp']   = np.zeros((self.num_loops, self.num_tot_pars), dtype='float')
        self.output_s['empty_step']['coeff_le'] = np.zeros((self.num_loops, self.num_tot_coeffs),dtype='float')
        self.output_s['empty_step']['ret_dict_l'] = [{} for i in range(self.num_loops)]

    def save_to_file(self, file):
        with gzip.open(file, "wb") as f: 
            pickle.dump([self.input_args, self.output_s, self.log_message], f)
        print(f'The input arguments, best-fit results, and running messages are saved to {file} (a python pickle compressed with gzip).')

    def reload_from_file(self, file):
        print(f'FitFrame is reloaded from {file} with the input arguments, best-fit results, and running messages.')
        with gzip.open(file, 'rb') as f: reloaded = pickle.load(f)
        self.__init__(**reloaded[0]) # re-initialize with the arguments
        self.output_s = reloaded[1] # copy the best-fit results
        self.log_message = reloaded[2] # copy the running messages

    ###############################################################################
    ######################### Model Auxiliary Functions ###########################

    def update_mask_lite_dict(self, model_name=None, mask_lite=None, dict=None):
        if dict is None:
            ret_dict = {} # create a new dict
            for mod in self.full_model_type.split('+'):
                ret_dict[mod] = np.ones((self.model_dict[mod]['num_coeffs']), dtype='bool')
        else:
            ret_dict = dict # update an existing dict

        if model_name is not None: # update a given model
            ret_dict[model_name] = mask_lite
        return ret_dict

    def search_model_index(self, sel_mods, model_type, mask_lite_dict=None):
        rev_model_type = ''
        for mod in self.full_model_type.split('+'):
            if np.isin(mod, model_type.split('+')): rev_model_type += mod+'+'
        rev_model_type = rev_model_type[:-1] # re-sort the input model_type to fit the order in self.full_model_type
        if rev_model_type.split(sel_mods)[0] == rev_model_type: raise ValueError((f"No such model combination: {sel_mods} in {rev_model_type}"))

        model_nums = {}
        for mod in self.full_model_type.split('+'): model_nums[mod] = {'par': self.model_dict[mod]['num_pars']}
        if mask_lite_dict is None: 
            for mod in self.full_model_type.split('+'): model_nums[mod]['coeff'] = self.model_dict[mod]['num_coeffs']
        else:
            for mod in self.full_model_type.split('+'): model_nums[mod]['coeff'] = mask_lite_dict[mod].sum()

        index_start_x = 0; index_start_coeff = 0
        for precomp in rev_model_type.split(sel_mods)[0].split('+'):
            if precomp == '': continue
            index_start_x += model_nums[precomp]['par']
            index_start_coeff += model_nums[precomp]['coeff']
        index_end_x = index_start_x + 0
        index_end_coeff = index_start_coeff + 0
        if len(sel_mods.split('+')) == 1:
            index_end_x += model_nums[sel_mods]['par']
            index_end_coeff += model_nums[sel_mods]['coeff']
        else:
            for singlecomp in sel_mods.split('+'):
                index_end_x += model_nums[singlecomp]['par']
                index_end_coeff += model_nums[singlecomp]['coeff']            
        return index_start_x, index_end_x, index_start_coeff, index_end_coeff
    
    def examine_model_SN(self, model_w, noise_w, accept_SN=2):
        if len(model_w) > 0:
            mask_valid_w = model_w > (np.nanmax(model_w)*0.05) # only consider line wavelength range with significant non-zero values
            if mask_valid_w.sum() > 0:
                peak_SN = np.nanpercentile(model_w[mask_valid_w] / noise_w[mask_valid_w], 90)
                return peak_SN, peak_SN >= accept_SN
            else:
                return 0, False
        else:
            return 0, False

    def examine_fit_quality(self, step=None):
        if step is None: step = 'joint_fit_3' if self.have_phot else 'joint_fit_2'

        best_chi_sq_l = self.output_s[step]['chi_sq_l'] 
        accept_chi_sq = copy(self.accept_chi_sq)
        if self.num_loops > 1:
            if (best_chi_sq_l[1:] > 0).sum() > 0:
                accept_chi_sq = best_chi_sq_l[1:][best_chi_sq_l[1:] > 0].min() * 1.5 # update using the finished fitting of mock data
        self.fit_quality_l = (best_chi_sq_l > 0) & (best_chi_sq_l < accept_chi_sq)
        success_count = self.fit_quality_l.sum()

        if success_count > 0:
            if self.fit_quality_l[0]: 
                tmp_msg = f'original data and {success_count - 1} mock data'
            else:
                tmp_msg = f'{success_count} mock data'
            print_log(f'{success_count} fitting loops ({tmp_msg}) have good quality, with chi_sq = {np.round(best_chi_sq_l[self.fit_quality_l],3)}.', self.log_message)
        else:
            print_log(f'No fitting loop has good quality.', self.log_message)

        if (self.num_loops-success_count) > 0:
            print_log(f'{self.num_loops-success_count} loops need refitting, with current chi_sq = {np.round(best_chi_sq_l[~self.fit_quality_l],3)}.', self.log_message)
        else:
            print_log(f'No fitting loop needs refitting. Run with main_fit(refit=True) to force refitting.', self.log_message)

        return success_count

    def create_mock_data(self, i_loop=None, ret_phot=False, chi_sq=None):
        spec_wave_w, spec_flux_w, spec_ferr_w = self.spec['wave_w'], self.spec['flux_w'], self.spec['ferr_w']
        mask_valid_w = self.spec['mask_valid_w']
        if ret_phot:
            phot_wave_b, phot_flux_b, phot_ferr_b = self.phot['wave_b'], self.phot['flux_b'], self.phot['ferr_b']
            mask_valid_b = self.phot['mask_valid_b']
            # specphot_wave_w = np.hstack((spec_wave_w, phot_wave_b))
            specphot_flux_w = np.hstack((spec_flux_w, phot_flux_b))
            specphot_ferr_w = np.hstack((spec_ferr_w, phot_ferr_b))
            specphot_mask_w = np.hstack((mask_valid_w, mask_valid_b))

        ####################
        # create modified errors if both spec and phot data are fit
        if ret_phot:        
            # revise calibration ratio in the 0th fitting loop for raw data
            # chi_sq is the reduced chi-squared
            # consider the case when calibration dominates the error, i.e., could be overestimated
            # chi_sq_0 = sum((residuals/error_0)**2)/N, a good-fit has chi_sq ~ 1, 
            # i.e., revise error_1 = error_0 * sqrt(chi_sq_0) to get sum((residuals/error_1)**2)/N ~ 1 in the next fitting
            if (self.inst_calib_ratio_rev == True) & (i_loop == 0) & (chi_sq is not None):
                k0_sq = self.inst_calib_ratio**2
                k1_sq = k0_sq * chi_sq - np.median((1 - chi_sq) * specphot_ferr_w[specphot_mask_w]**2 / specphot_flux_w[specphot_mask_w]**2)
                if k1_sq < 0: k1_sq = 0
                self.inst_calib_ratio = min(np.sqrt(k1_sq), 0.2) # set an upperlimit of 0.2
                print_log(f"The ratio of calibration error over flux is updated from {np.sqrt(k0_sq):.3f} to {np.sqrt(k1_sq):.3f}.", 
                          self.log_message, self.print_step)

            # create modified error
            if self.inst_calib_smooth < 1e-4: # use 1e-4 instead of 0 to avoid possible equal==0 error
                # use the original flux 
                specphot_reverr_w = np.sqrt(specphot_ferr_w**2 + specphot_flux_w**2 * self.inst_calib_ratio**2)
            else:
                # create smoothed spectrum
                if ~np.isin('joint_fit_1', [*self.output_s]):
                    # if not fit yet
                    spec_flux_smoothed_w = convolve_var_width_fft(spec_wave_w[mask_valid_w], spec_flux_w[mask_valid_w], fwhm_vel_kin=self.inst_calib_smooth, num_bins=self.conv_nbin_max, reset_edge=False)
                    spec_flux_smoothed_w = np.interp(spec_wave_w, spec_wave_w[mask_valid_w], spec_flux_smoothed_w)
                else:
                    # use joint_fit_1 fitting result to avoid cutting of continuum at edges
                    spec_flux_smoothed_w = spec_flux_w * 0
                    cont_spec_fmod_w = self.output_s['joint_fit_1']['ret_dict_l'][i_loop]['cont_spec_fmod_w']
                    spec_flux_smoothed_w += convolve_var_width_fft(spec_wave_w, cont_spec_fmod_w, fwhm_vel_kin=self.inst_calib_smooth, num_bins=self.conv_nbin_max, reset_edge=True)
                    line_spec_fmod_w = self.output_s['joint_fit_1']['ret_dict_l'][i_loop]['line_spec_fmod_w']
                    spec_flux_smoothed_w += convolve_var_width_fft(spec_wave_w, line_spec_fmod_w, fwhm_vel_kin=self.inst_calib_smooth, num_bins=self.conv_nbin_max, reset_edge=False)
                specphot_flux_smoothed_w = np.hstack((spec_flux_smoothed_w, phot_flux_b))
                specphot_reverr_w = np.sqrt(specphot_ferr_w**2 + specphot_flux_smoothed_w**2 * self.inst_calib_ratio**2)
        ####################

        ####################
        # if i_loop = 0, return raw data and skip the following steps
        if i_loop == 0:
            if not ret_phot:
                return spec_flux_w
            else:
                return specphot_flux_w, specphot_reverr_w
        else:
            # if i_loop > 0, create mock data and rescale error (if ret_phot)
            # extract the best-fit model and residuals from the first loop (i.e., raw data in default); 
            # modified from the last step (i_loop-1) to avoid developing offset from the raw data
            step = 'joint_fit_3' if ret_phot else 'joint_fit_2'
            fmod_w = self.output_s[step]['ret_dict_l'][0]['fmod_w']
            fres_w = self.output_s[step]['ret_dict_l'][0]['fres_w']
            spec_fmod_w = fmod_w[:self.num_spec_wave]
            spec_fres_w = fres_w[:self.num_spec_wave]
            if ret_phot: 
                phot_fmod_b = fmod_w[-self.num_phot_band:]
                phot_fres_b = fres_w[-self.num_phot_band:]

            # perfrom low-pass filter to residuals 
            window_length = int(50*(1+self.v0_redshift)/np.median(spec_wave_w[1:]-spec_wave_w[:-1])) # smooth high-frequency noise within rest 50AA
            spec_fres_lowpass_w = savgol_filter(spec_fres_w[mask_valid_w], window_length=min(window_length, len(spec_fres_w[mask_valid_w])), polyorder=2)
            spec_fres_lowpass_w = np.interp(spec_wave_w, spec_wave_w[mask_valid_w], spec_fres_lowpass_w)
            spec_flux_intrinsic_w = spec_fmod_w + spec_fres_lowpass_w
            if ret_phot: 
                # if len(phot_fres_b[mask_valid_b]) > 1: 
                #     phot_fres_lowpass_b = savgol_filter(phot_fres_b[mask_valid_b], window_length=min(2, len(phot_fres_b[mask_valid_b])), polyorder=1)
                #     phot_fres_lowpass_b = np.interp(phot_wave_b, phot_wave_b[mask_valid_b], phot_fres_lowpass_b)
                # else:
                #     phot_fres_lowpass_b = phot_fres_b
                # phot_flux_intrinsic_b = phot_fmod_b + phot_fres_lowpass_b
                # do not smooth photometric data since low-frequency trend is hard to obtain
                # also, when phot-data is considered, the appending errors are always enlarged by including calibration errors, 
                # i.e., no requirement to subtract intrinsic high-frequency noise from original data
                phot_flux_intrinsic_b = phot_flux_b
                specphot_flux_intrinsic_w = np.hstack((spec_flux_intrinsic_w, phot_flux_intrinsic_b))

            # create mock data
            if not ret_phot:
                # spec_fmock_w = fmod_w + spec_fres_lowpass_w + np.random.randn(len(spec_ferr_w)) * spec_ferr_w
                spec_fmock_w = spec_flux_intrinsic_w + np.random.normal(scale=spec_ferr_w) 
                return spec_fmock_w
            else:
                # specphot_fmock_w = fmod_w + specphot_fres_lowpass_w + np.random.randn(len(specphot_reverr_w)) * specphot_reverr_w
                specphot_fmock_w = specphot_flux_intrinsic_w + np.random.normal(scale=specphot_reverr_w)
                return specphot_fmock_w, specphot_reverr_w
        ####################

    ###############################################################################
    ############################## Fitting Functions ##############################

    def log_to_linear_lsq(self, A, b, x0, alpha=1e-3, epsilon=1e-8):
        # Try to solve Ax = b by minimizing |ln(Ax) - ln(b)|^2 .
        # Use the first-order Taylor expansion: ln(Ax) ~= ln(Ax0) + A(x-x0)/Ax0
        # and then non-linear ln(Ax) = ln(b)
        # is converted to linear Jx = k = (Jx0 - ln(Ax0) + ln(b)), where J = A/Ax0.

        Ax0 = A @ x0 # = np.dot(A, x0)
        Ax0[Ax0 < 0] = b.min() * 0.01 # force positive model values; b (flux) only contains positive values
        J = A / Ax0[:, None]
        k = J @ x0 - np.log(Ax0) + np.log(b)

        # # Use regularization with positivity constraints to avoid too small Ax0 or b in ln(),
        # # i.e., the function actually minimize |ln(Ax + eps) - ln(b + eps)|^2 . 
        # minimal = max(epsilon, alpha*np.median(Ax0), alpha*np.median(b)) 
        # J = A / (Ax0 + minimal)[:, None]
        # k = J @ x0 - np.log(Ax0 + minimal) + np.log(b + minimal)

        return J, k

    def linear_lsq_solver(self, flux_w, model_ew, weight_w, fit_grid='linear', verbose=False):
        # Solve linear least-square functions to obtain the normlization values (i.e., coeffs) of each models

        mask_valid_w = weight_w > 0 # i.e., ferr_w > 0; if fit_grid='log' also with flux_w > 0
        # Define Jacobian matrix and constants in the linear equations: A_wm @ x_m = b_w
        # Only use data and model in valid wavelengths. 
        A_wm = (model_ew.T * weight_w[:,None])[mask_valid_w,:]
        b_w = (flux_w * weight_w)[mask_valid_w]

        solution = lsq_linear(A_wm, b_w, bounds=(0., np.inf), verbose=verbose) 
        # max_iter=200, lsmr_tol='auto', tol=1e-12, 

        if fit_grid == 'log':
            # convolve Ax = b by minimizing |ln(Ax) - ln(b)|^2
            # to Jx = k by minimizing |Jx - k|^2
            # Taylor expansion surrounding linear lsq solution.x
            J_wm, k_w = self.log_to_linear_lsq(A_wm, b_w, solution.x)
            # solve Jx = k by minimizing |Jx - k|^2
            solution = lsq_linear(J_wm, k_w, bounds=(0., np.inf), verbose=verbose)

        coeff_e = solution.x
        ret_model_w = coeff_e @ model_ew # Returns best model in the full wavelength (not limited by mask_valid_w)
        chi_w = np.zeros_like(ret_model_w)
        # linear: (model/flux-1)*(flux*weight), log: ln(model/flux)*(flux*weight)
        if fit_grid == 'linear': 
            chi_w[mask_valid_w] = (ret_model_w - flux_w)[mask_valid_w] * weight_w[mask_valid_w] # reduced with weight_w
        if fit_grid == 'log':
            ret_model_w[ret_model_w <= 0] = flux_w[mask_valid_w].min() * 0.01 # force positive model values
            chi_w[mask_valid_w] = np.log(ret_model_w[mask_valid_w] / flux_w[mask_valid_w]) * (flux_w * weight_w)[mask_valid_w]

        if (coeff_e < 0).any(): 
            self.error = {'flux_w':flux_w, 'ret_model_w':ret_model_w, 'coeff_e':coeff_e, 'model_ew':model_ew, 'weight_w':weight_w, 'mask_valid_w':mask_valid_w}
            raise ValueError((f"Negative model coeff: {np.where(coeff_e <0)}-th in {coeff_e}."))
        if (~np.isfinite(coeff_e)).any():
            self.error = {'flux_w':flux_w, 'ret_model_w':ret_model_w, 'coeff_e':coeff_e, 'model_ew':model_ew, 'weight_w':weight_w, 'mask_valid_w':mask_valid_w}
            raise ValueError((f"NaN or Inf detected in model coeff: {np.where(~np.isfinite(coeff_e))}-th in {coeff_e}."))
        if (~np.isfinite(chi_w)).any():
            self.error = {'flux_w':flux_w, 'ret_model_w':ret_model_w, 'coeff_e':coeff_e, 'model_ew':model_ew, 'weight_w':weight_w, 'mask_valid_w':mask_valid_w}
            print('flux_w at chi_w=nan:', flux_w[~np.isfinite(chi_w)])
            print('model_w at chi_w=nan:', ret_model_w[~np.isfinite(chi_w)])
            raise ValueError((f"NaN or Inf detected in residuals at chi_w."))

        return coeff_e, ret_model_w, chi_w

    def linear_process(self, x, flux_w, ferr_w, mask_w, model_type, mask_lite_dict, 
                       fit_phot=False, fit_grid='linear', conv_nbin=None, ret_coeffs=False):
        # for a give set of parameters, return models and residuals
        # the residuals are used to solve non-linear least-square fit

        # re-sort the input model_type to fit the order in self.full_model_type
        rev_model_type = ''
        for mod in self.full_model_type.split('+'):
            if np.isin(mod, model_type.split('+')): rev_model_type += mod+'+'
        rev_model_type = rev_model_type[:-1] 

        # tie parameters following setup in input _config 
        bound_min_p = np.array([])
        bound_max_p = np.array([])
        tie_p = np.array([])
        for mod in rev_model_type.split('+'):
            bound_min_p = np.hstack((bound_min_p, self.model_dict[mod]['cframe'].min_cp.flatten()))
            bound_max_p = np.hstack((bound_max_p, self.model_dict[mod]['cframe'].max_cp.flatten()))
            tie_p = np.hstack((tie_p, self.model_dict[mod]['cframe'].tie_cp.flatten()))
        n_freepars = (tie_p == 'free').sum()
        for i_p in range(len(x)):
            if tie_p[i_p] == 'free': 
                if x[i_p] < bound_min_p[i_p]: x[i_p] = bound_min_p[i_p] # re-check if x matches bounds
                if x[i_p] > bound_max_p[i_p]: x[i_p] = bound_max_p[i_p]
            else:
                if tie_p[i_p] == 'fix': 
                    x[i_p] = bound_min_p[i_p]
                else:
                    for single_tie in tie_p[i_p].split(';'):
                        ref_mod, ref_comp, ref_i_par = single_tie.split(':')
                        if np.isin(ref_mod, rev_model_type.split('+')):
                            ref_num_pars_per_comp = self.model_dict[ref_mod]['cframe'].num_pars_per_comp
                            ref_i_comp = np.where(np.array(self.model_dict[ref_mod]['cframe'].comp_c) == ref_comp)[0]
                            if len(ref_i_comp) == 1:
                                ref_i_comp = ref_i_comp[0]
                            else:
                                raise ValueError((f"The reference component: {ref_comp} is not available in {self.model_dict[ref_mod]['cframe'].comp_c}"))
                            ref_i_x = ref_num_pars_per_comp*ref_i_comp + int(ref_i_par)
                            mp0, mp1 = self.search_model_index(ref_mod, rev_model_type)[0:2]
                            x[i_p] = x[mp0:mp1][ref_i_x]                   
                            break # only select the 1st effective tie relation

        spec_wave_w = self.spec['wave_w']
        if fit_phot: sed_wave_w = self.sed['wave_w']
        fit_model_ew = None
        for mod in rev_model_type.split('+'):
            mp0, mp1 = self.search_model_index(mod, rev_model_type)[0:2]
            if (~np.isfinite(x[mp0:mp1])).any():
                raise ValueError((f"NaN or Inf detected in x = {x[mp0:mp1]} of '{mod}' model."))
            ####
            spec_fmod_ew = self.model_dict[mod]['spec_func'](spec_wave_w, x[mp0:mp1], mask_lite_e=mask_lite_dict[mod], conv_nbin=conv_nbin)
            if fit_phot:
                sed_fmod_ew = self.model_dict[mod]['sed_func'](sed_wave_w, x[mp0:mp1], mask_lite_e=mask_lite_dict[mod], conv_nbin=None) #convolution not required
                sed_fmod_eb = self.pframe.spec2phot(sed_wave_w, sed_fmod_ew, self.phot['trans_bw'])
                spec_fmod_ew = np.hstack((spec_fmod_ew, sed_fmod_eb))
            ####
            if (~np.isfinite(spec_fmod_ew)).any():
                self.error = {'spec':spec_fmod_ew, 'wave':spec_wave_w, 'x':x[mp0:mp1], 'mask':mask_lite_dict[mod], 'conv_nbin':conv_nbin} # output for check
                raise ValueError((f"NaN or Inf detected in returned spectra of '{mod}' model with x = {x[mp0:mp1]}"
                                 +f", position (m,w) = {np.where(~np.isfinite(spec_fmod_ew))}."))
            spec_fmod_positive_ew = spec_fmod_ew * 1.0 # copy
            spec_fmod_positive_ew[self.model_dict[mod]['spec_mod'].mask_absorption_e[mask_lite_dict[mod]],:] *= -1 # convert nagative values for the following positive check
            if (spec_fmod_positive_ew < 0).any(): 
                self.error = {'spec':spec_fmod_ew, 'wave':spec_wave_w, 'x':x[mp0:mp1], 'mask':mask_lite_dict[mod], 'conv_nbin':conv_nbin} # output for check
                raise ValueError((f"Negative value detected in returned spectra of '{mod}' model with x = {x[mp0:mp1]}"
                                 +f", position (m,w) = {np.where(spec_fmod_positive_ew < 0)}."))
            # if (spec_fmod_positive_ew.sum(axis=1) <= 0).any(): 
            #     self.error = {'spec':spec_fmod_ew, 'wave':spec_wave_w, 'x':x[mp0:mp1], 'mask':mask_lite_dict[mod], 'conv_nbin':conv_nbin} # output for check
            #     raise ValueError((f"Zero or negative integrated flux detected in returned spectra of '{mod}' model with x = {x[mp0:mp1]}"
            #                      +f", position (m) = {np.where(spec_fmod_positive_ew.sum(axis=1) <= 0)}."))
            ####
            fit_model_ew = spec_fmod_ew if (fit_model_ew is None) else np.vstack((fit_model_ew, spec_fmod_ew))
        n_elements = fit_model_ew.shape[0]

        mask_valid_w = mask_w & (ferr_w > 0)
        if fit_grid == 'log': 
            if (mask_valid_w & (flux_w > 0)).sum() / mask_valid_w.sum() > 0.8:
                mask_valid_w &= flux_w > 0
            else:
                fit_grid == 'linear'
                print_log(f"[WARNING] Over 20% of the input data has non-positive values, reset fit_grid = linear.", self.log_message)

        significance_w = self.spec['significance_w']
        if fit_phot: significance_w = np.hstack((self.spec['significance_w'], self.phot['significance_b']))
        degree_of_freedom = (significance_w[mask_valid_w]).sum() - (n_elements + n_freepars) # only sum valid measurements with mask_valid_w
        if degree_of_freedom < 1: 
            raise ValueError((f"The effective significance of valid measurements, {(significance_w[mask_valid_w]).sum()}, "
                             +f"(original number of measurements: {(mask_valid_w).sum()}), "
                             +f"is less than the total number of free models ({n_elements}) and free parameters({n_freepars})."))
        weight_w = np.divide(np.sqrt(significance_w/degree_of_freedom), ferr_w, where=mask_valid_w, out=np.zeros_like(ferr_w))

        coeff_e = np.zeros(n_elements, dtype='float')
        mask_valid_e = fit_model_ew[:,mask_valid_w].sum(axis=1) != 0
        coeff_e[mask_valid_e], model_w, chi_w = self.linear_lsq_solver(flux_w, fit_model_ew[mask_valid_e,:], weight_w, fit_grid, verbose=self.verbose)
        chi_sq = (chi_w**2).sum() # already reduced
        
        if not ret_coeffs:
            # for callback of optimization solvers
            return chi_w*np.sqrt(2)
            # return np.sqrt(2 * chi_sq)
            # then the cost function of least_squares, i.e., 0.5*sum(ret**2), is the chi_sq itself (for check)
            # should match format of jac_matrix: Jac_1x if sqrt(2 * chi_sq); Jac_wx if sqrt(2)*chi_w
        else:
            if not fit_phot:
                print_log(f"Fit with {n_elements} free elements and {n_freepars} free parameters of {len(rev_model_type.split('+'))} models, "
                         +f"reduced chi-squared = {chi_sq:.3f}.", self.log_message, self.print_step)
            else:                
                print_log(f"Fit with {n_elements} free elements and {n_freepars} free parameters of {len(rev_model_type.split('+'))} models: ", 
                          self.log_message, self.print_step)
                print_log(f"Reduced chi-squared with scaled errors = {chi_sq:.3f} for spectrum+SED;", self.log_message, self.print_step)
                spec_chi_sq = (chi_w[:self.num_spec_wave]**2).sum()
                print_log(f"Reduced chi-squared with scaled errors = {spec_chi_sq:.3f} for pure spectrum;", self.log_message, self.print_step)
                phot_chi_sq = (chi_w[-self.num_phot_band:]**2).sum()
                print_log(f"Reduced chi-squared with scaled errors = {phot_chi_sq:.3f} for pure phot-SED;", self.log_message, self.print_step)

                orig_ferr_w = np.hstack((self.spec['ferr_w'], self.phot['ferr_b']))
                tmp_chi_w = np.divide(chi_w*ferr_w, orig_ferr_w, where=mask_valid_w, out=np.zeros_like(ferr_w))
                spec_chi_sq = (tmp_chi_w[:self.num_spec_wave]**2).sum()
                print_log(f"Reduced chi-squared with original errors = {spec_chi_sq:.3f} for pure spectrum;", self.log_message, self.print_step)
                phot_chi_sq = (tmp_chi_w[-self.num_phot_band:]**2).sum()
                print_log(f"Reduced chi-squared with original errors = {phot_chi_sq:.3f} for pure phot-SED.", self.log_message, self.print_step)

            return coeff_e, model_w, chi_sq

    def jac_matrix(self, function, x, args=(), alpha=0.01, epsilon=1e-4):
        # use custom 3-point jacobian functions to avoid bugs with the scipy internal one (jac='3-point')
        # over small (1e-10) or large (1e-2) h-value could lead to worse fitting
        num_xs = len(x) # number of x
        num_waves = len(args[0]) # if function returns sqrt(2)*chi_w

        Jac_wx = np.zeros((num_waves, num_xs))  # Jacobian matrix
        for i_x in range(num_xs):  # loop over x
            x_forward = x.copy(); x_backward = x.copy()
            x_step = max(alpha * abs(x[i_x]), epsilon)
            x_forward[i_x] += x_step
            x_backward[i_x] -= x_step
            f_forward_w = function(x_forward, *args)
            f_backward_w = function(x_backward, *args)
            # Compute central difference
            Jac_wx[:, i_x] = (f_forward_w - f_backward_w) / (2 * x_step)
            # if self.save_test: 
            #     self.log_test_jacs.append((i_x, x, x_forward, x_backward, f_forward_w, f_backward_w, Jac_wx))
            if (~np.isfinite(Jac_wx[:, i_x])).any():
                raise ValueError((f"NaN or Inf detected in Jacobian column {i_x} at x = {x}"))

        # def jac_per_x(i_x):
        #     x_forward = x.copy(); x_backward = x.copy()
        #     x_step = max(alpha * abs(x[i_x]), epsilon)
        #     x_forward[i_x] += x_step
        #     x_backward[i_x] -= x_step
        #     f_forward_w = function(x_forward, *args)
        #     f_backward_w = function(x_backward, *args)
        #     jac_ret_w = (f_forward_w - f_backward_w) / (2 * x_step)
        #     if (~np.isfinite(jac_ret_w)).any():
        #         raise ValueError((f"NaN or Inf detected in Jacobian column {i_x} at x = {x}"))
        #     return jac_ret_w
        # Jac_wx = np.column_stack( Parallel(n_jobs=-1)( delayed(jac_per_x)(i_x) for i_x in range(num_xs) ) )

        return Jac_wx

    def nonlinear_process(self, x0_input, flux_w, ferr_w, mask_w, 
                          model_type, mask_lite_dict, fit_phot=False, fit_grid='linear', conv_nbin=None, 
                          accept_chi_sq=None, nlfit_ntry_max=None, 
                          annealing=False, da_niter_max=None, perturb_scale=None, nllsq_ftol_ratio=None, 
                          fit_message=None, i_loop=None, save_best_fit=True, verbose=False): 
        # core fitting function to obtain solution of non-linear least-square problems

        print_log('#### <'+fit_message.split(':')[0]+'> start:'+fit_message.split(':')[1]+'.', 
                  self.log_message, self.print_step)
        self.time_step = time.time()

        if x0_input is None: x0_input = np.random.uniform(self.bound_min_p, self.bound_max_p) # create random parameters
        if accept_chi_sq is None: accept_chi_sq = self.accept_chi_sq
        if nlfit_ntry_max is None: nlfit_ntry_max = self.nlfit_ntry_max
        if da_niter_max is None: da_niter_max = self.da_niter_max
        if nllsq_ftol_ratio is None: nllsq_ftol_ratio = self.nllsq_ftol_ratio

        # create the dictonary to return; copy all input
        frame = inspect.currentframe()
        arg_list = list(self.nonlinear_process.__code__.co_varnames)
        ret_dict = {arg: copy(frame.f_locals[arg]) for arg in arg_list if arg != 'self' and arg != 'frame' and arg in frame.f_locals}
 
        x0 = copy(x0_input) # avoid modify the input x0_input
        # for input of called functions:
        args=(flux_w, ferr_w, mask_w, model_type, mask_lite_dict, fit_phot, fit_grid, conv_nbin)
        if self.save_test: 
            self.log_test_args = (x0, args)
            self.log_test_jacs = [] # to trace steps in jacobian
            self.log_test_ret_dict = ret_dict

        mask_x = np.zeros_like(self.bound_min_p, dtype='bool') 
        for mod in model_type.split('+'):
            fp0, fp1 = self.search_model_index(mod, self.full_model_type)[0:2]
            mask_x[fp0:fp1] = True
        
        ##############################################################
        # main fitting cycles
        accept_condition = False
        achieved_chi_sq, achieved_ls_solution = 1e4, None
        for i_fit in range(nlfit_ntry_max):
            try:
                if annealing: 
                    print_log(f'Perform Dual Annealing optimazation for a rough global search.', self.log_message, self.print_step)
                    # create randomly initialized parameters used in this step
                    # although x0 is not mandatory for dual_annealing, random x0 can enable it to explore from different starting
                    x0_new = np.random.uniform(self.bound_min_p, self.bound_max_p)
                    x0[mask_x] = x0_new[mask_x] # update x0 used in this step
                    # use dual_annealing with a few iteration for a rough solution of global minima
                    # calling func of dual_annealing should return a scalar; 
                    da_solution = dual_annealing(lambda x, *args: 0.5*(self.linear_process(x, *args)**2).sum(),
                                                 list(zip(self.bound_min_p[mask_x], self.bound_max_p[mask_x])), 
                                                 args=args, x0=x0[mask_x], no_local_search=True, initial_temp=1e4, visit=1.5, maxiter=da_niter_max)
                    x0[mask_x] = da_solution.x # update x0 used in this step
                    if self.plot_step: 
                        coeff_e, model_w, chi_sq = self.linear_process(da_solution.x, *args, ret_coeffs=True)
                        self.plot_canvas(flux_w, model_w, ferr_w, mask_w, fit_phot, '[DA] '+fit_message, chi_sq, i_loop)
                    else:
                        print_log(f'Non-linear fitting cycle {i_fit+1}/{nlfit_ntry_max}, Dual Annealing returns chi_sq = {da_solution.fun:.3f}.', 
                                  self.log_message, self.print_step)
                else:
                    if perturb_scale > 0:
                        print_log(f'Perturb transferred parameters with scatters of {perturb_scale*100}% of parameter ranges.', 
                                  self.log_message, self.print_step)
                        # only pertrub transferred parameters used in this step by appending scatters from scaled bound range
                        x0_new = x0 + np.random.normal(scale=self.bound_width_p * perturb_scale) # 5% scaled 
                        x0_new = np.maximum(x0_new, self.bound_min_p)
                        x0_new = np.minimum(x0_new, self.bound_max_p)
                        x0[mask_x] = x0_new[mask_x] # update x0 used in this step 
                    else:
                        print_log(f'Do not perturb transferred parameters from the former step.', self.log_message, self.print_step)
                ret_dict['x0_actual'] = copy(x0) # save the updated x0

                print_log(f'Perform Non-linear Least-square optimazation for fine tuning.', self.log_message, self.print_step)
                ls_solution = least_squares(fun=self.linear_process, args=args,
                                            x0=x0[mask_x], bounds=(self.bound_min_p[mask_x], self.bound_max_p[mask_x]), 
                                            x_scale='jac', jac=lambda x, *args: self.jac_matrix(self.linear_process, x, args=args), 
                                            ftol=nllsq_ftol_ratio/np.sqrt(len(flux_w)), max_nfev=10000, verbose=verbose) 
                                            # A chi-square distribution has mean of N and standard deviation of sqrt(2N).
                                            # Here linear_process returns reduced chi_sq, and the stopping condition is
                                            # (chi_sq_i1-chi_sq_i0) / chi_sq_i0 < ftol.
                                            # Since the standard deviation of reduced chi_sq is sqrt(2N)/N = sqrt(2/N), 
                                            # a proper choice of ftol is ~1/sqrt(N). Note that a more precise expression is 
                                            # sqrt(significance_w.mean())/sqrt(sum(significance_w)-n_mods-n_pars),
                                            # here simplify with significance_w=1 and n_mods=n_pars=0. 
                                            # Adopt a scaling ratio nllsq_ftol_ratio with default of 0.01 to guarantee a better accuracy (from test). 
            except Exception as ex: 
                print('Exception:', ex); traceback.print_exc(); # sys.exit()
            else:
                if ls_solution.success:
                    # coeff_e, model_w, chi_sq = self.linear_process(ls_solution.x, *args, ret_coeffs=True)
                    chi_sq = ls_solution.cost
                    accept_condition  =  chi_sq <= (accept_chi_sq * 1.1)
                    accept_condition |= (chi_sq <= (accept_chi_sq * 1.5)) & (achieved_chi_sq <= (accept_chi_sq * 1.5))
                    if accept_condition: 
                        if (chi_sq > accept_chi_sq) & (chi_sq <= (accept_chi_sq * 1.1)):
                            print_log(f'Non-linear fitting cycle {i_fit+1}/{nlfit_ntry_max}, '+
                                      f'accept this fitting with chi_sq = {chi_sq:.3f} / {accept_chi_sq:.3f} (goal) < 110%.', 
                                      self.log_message, self.print_step)
                        if (chi_sq > (accept_chi_sq * 1.1)) & (chi_sq <= (accept_chi_sq * 1.5)):
                            print_log(f'Non-linear fitting cycle {i_fit+1}/{nlfit_ntry_max}, '+
                                      f'accept this fitting with chi_sq = {chi_sq:.3f} / {accept_chi_sq:.3f} (goal) < 150%, achieved twice.', 
                                      self.log_message, self.print_step) 
                        break # exit nlfit cycle
                    else:
                        if chi_sq < achieved_chi_sq: 
                            achieved_chi_sq = copy(chi_sq) # save the achieved min chi_sq
                            achieved_ls_solution = copy(ls_solution)
                        print_log(f'Non-linear fitting cycle {i_fit+1}/{nlfit_ntry_max}, '+
                                  f'poor-fit with chi_sq = {chi_sq:.3f} > {accept_chi_sq:.3f} (goal) --> try refitting; '+
                                  f'achieved min_chi_sq = {achieved_chi_sq:.3f}', self.log_message, self.print_step)
                else:
                    print_log(f'Non-linear fitting cycle {i_fit+1}/{nlfit_ntry_max} failed --> try refitting; '+
                              f'achieved min_chi_sq = {achieved_chi_sq:.3f}', self.log_message, self.print_step) 
        if accept_condition: 
            best_fit = ls_solution # accept the solution in the final try
        else:
            if achieved_ls_solution is not None:
                best_fit = achieved_ls_solution # back to solution with achieved min_chi_sq
            else:
                best_fit = ls_solution # use the solution in the final try if all tries failed
        coeff_e, model_w, chi_sq = self.linear_process(best_fit.x, *args, ret_coeffs=True)
        # save fit_grid; it may be forced to 'linear' by linear_process if with too many non-positive fluxes
        ret_dict['fit_grid_actual'] = copy(fit_grid)
        if self.plot_step: self.plot_canvas(flux_w, model_w, ferr_w, mask_w, fit_phot, '[LS] '+fit_message, chi_sq, i_loop)
        ##############################################################

        ##############################################################
        # return the best-fit results
        ret_dict['best_fit'] = best_fit
        ret_dict['par_p'] = best_fit.x
        ret_dict['coeff_e'] = coeff_e
        ret_dict['chi_sq'] = chi_sq
        ret_dict['fmod_w'] = model_w
        ret_dict['fres_w'] = flux_w - model_w

        # create best-fit continuum and emission line models for subtracting them in following steps
        ret_dict['cont_spec_fmod_w'] = self.spec['wave_w'] * 0
        ret_dict['line_spec_fmod_w'] = self.spec['wave_w'] * 0
        if self.have_phot:
            ret_dict['cont_specphot_fmod_w'] = np.hstack((self.spec['wave_w'], self.phot['wave_b'])) * 0
            ret_dict['line_specphot_fmod_w'] = np.hstack((self.spec['wave_w'], self.phot['wave_b'])) * 0
        for mod in model_type.split('+'):
            mp0, mp1, me0, me1 = self.search_model_index(mod, model_type, mask_lite_dict)
            spec_fmod_ew = self.model_dict[mod]['spec_func'](self.spec['wave_w'], best_fit.x[mp0:mp1], 
                                                             mask_lite_e=mask_lite_dict[mod], conv_nbin=conv_nbin)
            spec_fmod_w = coeff_e[me0:me1] @ spec_fmod_ew
            if mod == 'line': 
                ret_dict['line_spec_fmod_w'] += spec_fmod_w
            else:
                ret_dict['cont_spec_fmod_w'] += spec_fmod_w
            if self.have_phot:
                sed_fmod_ew = self.model_dict[mod]['sed_func'](self.sed['wave_w'], best_fit.x[mp0:mp1], 
                                                               mask_lite_e=mask_lite_dict[mod], conv_nbin=None) # convolution no required
                sed_fmod_w = coeff_e[me0:me1] @ sed_fmod_ew
                phot_fmod_b = self.pframe.spec2phot(self.sed['wave_w'], sed_fmod_w, self.phot['trans_bw'])
                if mod == 'line': 
                    ret_dict['line_specphot_fmod_w'] += np.hstack((spec_fmod_w, phot_fmod_b))
                else:
                    ret_dict['cont_specphot_fmod_w'] += np.hstack((spec_fmod_w, phot_fmod_b))

        # save the best-fit parameters to the full x0 list (i.e., with all models in self.full_model_type) to guide following fitting
        ret_dict['x0_final'] = copy(ret_dict['x0_input'])
        for mod in model_type.split('+'):
            fp0, fp1 = self.search_model_index(mod, self.full_model_type)[0:2]
            mp0, mp1 = self.search_model_index(mod, model_type)[0:2]
            ret_dict['x0_final'][fp0:fp1] = best_fit.x[mp0:mp1]
        ##############################################################

        ##############################################################
        # save the best-fit results to the FitFrame class
        if save_best_fit:
            step_id = fit_message.split(':')[0]
            if ~np.isin(step_id, [*self.output_s]):
                # copy the format template
                self.output_s[step_id] = copy(self.output_s['empty_step'])

            if (self.output_s[step_id]['chi_sq_l'][i_loop] == 0) | (self.output_s[step_id]['chi_sq_l'][i_loop] > chi_sq): 
                # save results only if the loop is the 1st run or the refitting gets smaller chi_sq
                self.output_s[step_id]['chi_sq_l'][i_loop] = chi_sq
                for mod in model_type.split('+'):
                    fp0, fp1, fe0, fe1 = self.search_model_index(mod, self.full_model_type)
                    mp0, mp1, me0, me1 = self.search_model_index(mod, model_type, mask_lite_dict)
                    self.output_s[step_id]['par_lp'][i_loop, fp0:fp1] = best_fit.x[mp0:mp1]
                    self.output_s[step_id]['coeff_le'][i_loop, fe0:fe1][mask_lite_dict[mod]] = coeff_e[me0:me1]
                self.output_s[step_id]['ret_dict_l'][i_loop] = ret_dict
        ##############################################################

        print_log('#### <'+fit_message.split(':')[0]+'> finish: '+
                  f'{time.time()-self.time_step:.1f}s/'+
                  f'{time.time()-self.time_loop:.1f}s/'+
                  f'{time.time()-self.time_init:.1f}s '+
                  'spent in this step/loop/total.', 
                  self.log_message, self.print_step)            

        return ret_dict

    def main_fit(self, refit=False):
        self.time_init = time.time()

        if not self.input_initialized: 
            print_log(f'\n', self.log_message, display=False)
            print_log(f'[Note] Re-initialize FitFrame since the input data is modified.', self.log_message)
            output_s = self.output_s # save the current fitting results
            log_message = self.log_message # save the current running message
            current_canvas = self.canvas # save the current canvas
            self.__init__(**self.input_args) # reset FitFrame if data is modified (e.g., in extract_results)
            self.output_s = output_s # copy the current fitting results
            self.log_message = log_message + ['\n'] + self.log_message + ['\n'] # copy the current running message
            self.canvas = current_canvas # transfer the current canvas

        spec_wave_w, spec_flux_w, spec_ferr_w = self.spec['wave_w'], self.spec['flux_w'], self.spec['ferr_w']
        mask_valid_w, mask_noline_w = self.spec['mask_valid_w'], self.spec['mask_noline_w']
        if self.have_phot:
            phot_wave_b, phot_flux_b, phot_ferr_b = self.phot['wave_b'], self.phot['flux_b'], self.phot['ferr_b']
            mask_valid_b = self.phot['mask_valid_b']
            sed_wave_w = self.sed['wave_w']

        # restore the fitting status if it is reloaded
        step = 'joint_fit_3' if self.have_phot else 'joint_fit_2'
        if np.isin(step, [*self.output_s]): 
            if not refit:
                print_log(center_string(f'Reload the results from the finished fitting loops', 80), self.log_message)
                success_count = self.examine_fit_quality() # self.fit_quality_l updated
                total_count = copy(success_count)
            else:
                print_log(center_string(f'Current fitting results are erased; refitting starts', 80), self.log_message)
                self.fit_quality_l[:] = False
                success_count, total_count = 0, 0
        else:
            self.fit_quality_l = np.zeros(self.num_loops, dtype='bool')
            success_count, total_count = 0, 0

        while success_count < self.num_loops:
            i_loop = np.where(~self.fit_quality_l)[0][0] 
            # i_loop is the 0th loop index without good fits, which used to save the results of current fitting loop
            print_log(center_string(f'Loop {i_loop+1}/{self.num_loops} starts', 80), self.log_message)
            self.time_loop = time.time()

            if i_loop == 0: 
                print_log(center_string(f'Fit the original spectrum', 80), self.log_message, self.print_step)
            else:
                print_log(center_string(f'Fit the mock spectrum', 80), self.log_message, self.print_step)
            spec_fmock_w = self.create_mock_data(i_loop)
            
            ####################################################
            # # for test
            # sys.exit()
            ####################################################

            ####################################################
            ################## init fit cycle ##################
            # determine model types for continuum fitting
            cont_type = ''
            for mod in self.full_model_type.split('+'):
                if (mod != 'line') & self.model_dict[mod]['spec_enable']: cont_type += mod + '+'
            cont_type = cont_type[:-1] # remove the last '+' 
            print_log(f'Continuum models used in spectral fitting: {cont_type}', self.log_message, self.print_step)
            ########################################
            # obtain a rough fit of continuum with emission line wavelength ranges masked out
            if np.isin('ssp', cont_type.split('+')): 
                mask_ssp_lite = self.model_dict['ssp']['spec_mod'].mask_ssp_lite_with_num_mods(num_ages_lite=8, num_mets_lite=1, verbose=self.print_step)
                mask_lite_dict = self.update_mask_lite_dict('ssp', mask_ssp_lite)
            else:
                mask_lite_dict = self.update_mask_lite_dict()
            cont_fit_init = self.nonlinear_process(None, spec_fmock_w, spec_ferr_w, mask_noline_w, 
                                                   cont_type, mask_lite_dict, fit_phot=False, fit_grid=self.fit_grid, conv_nbin=1, 
                                                   annealing=self.init_annealing, perturb_scale=self.perturb_scale, 
                                                   fit_message='cont_fit_init: spectral fitting, initialize continuum models', i_loop=i_loop)
            ########################################
            if np.isin('line', self.full_model_type.split('+')): 
                # obtain a rough fit of emission lines with continuum of cont_fit_init subtracted
                line_fit_init = self.nonlinear_process(None, (spec_fmock_w - cont_fit_init['cont_spec_fmod_w']), spec_ferr_w, mask_valid_w, 
                                                       'line', mask_lite_dict, fit_phot=False, fit_grid='linear', conv_nbin=None,
                                                       annealing=self.init_annealing, perturb_scale=self.perturb_scale, 
                                                       fit_message='line_fit_init: spectral fitting, initialize emission lines', i_loop=i_loop)
            else:
                # just copy the cont_fit to line_fit. note that line_fit['line_spec_fmod_w'] = 0, but line_fit['fmod_w'] = cont_fit['fmod_w']
                line_fit_init = cont_fit_init
                self.output_s['line_fit_init'] = self.output_s['cont_fit_init']
            ####################################################
            ####################################################
            
            ####################################################
            ################## 1st fit cycle ###################
            # obtain a better fit of stellar continuum after subtracting emission lines of line_fit_init
            if np.isin('ssp', cont_type.split('+')): 
                mask_ssp_lite = self.model_dict['ssp']['spec_mod'].mask_ssp_lite_with_num_mods(num_ages_lite=16, num_mets_lite=1, verbose=self.print_step)
                mask_lite_dict = self.update_mask_lite_dict('ssp', mask_ssp_lite)
            else:
                mask_lite_dict = self.update_mask_lite_dict()
            cont_fit_1 = self.nonlinear_process(None, (spec_fmock_w - line_fit_init['line_spec_fmod_w']), spec_ferr_w, mask_valid_w, 
                                                cont_type, mask_lite_dict, fit_phot=False, fit_grid=self.fit_grid, conv_nbin=2, 
                                                annealing=self.init_annealing, perturb_scale=self.perturb_scale, 
                                                fit_message='cont_fit_1: spectral fitting, update continuum models', i_loop=i_loop)
            ########################################
            if np.isin('line', self.full_model_type.split('+')): 
                # examine if absorption line components are necessary
                line_mod = self.model_dict['line']['spec_mod']
                line_disabled_comps = [line_mod.cframe.comp_c[i_comp] for i_comp in range(line_mod.num_comps) if line_mod.cframe.info_c[i_comp]['sign'] == 'absorption']
                if len(line_disabled_comps) > 0:
                    mask_abs_w = mask_valid_w & (line_fit_init['line_spec_fmod_w'] < 0)
                    line_abs_peak_SN, line_abs_examine = self.examine_model_SN(-line_fit_init['line_spec_fmod_w'][mask_abs_w], spec_ferr_w[mask_abs_w], accept_SN=self.accept_model_SN)
                    if not line_abs_examine:
                        print_log(f'Absorption components {line_disabled_comps} are disabled due to low peak S/N = {line_abs_peak_SN:.3f} (abs) < {self.accept_model_SN} (set by accept_model_SN).', 
                                  self.log_message, self.print_step)                 
                        # fix the parameters of disabled components (to reduce number of free parameters)
                        for i_comp in range(line_mod.num_comps):
                            if line_mod.cframe.info_c[i_comp]['sign'] == 'absorption': self.model_dict['line']['cframe'].tie_cp[i_comp,:] = 'fix'
                        # update mask_lite_dict with emission line examination results, i.e., only keep enabled line components
                        mask_lite_dict = self.update_mask_lite_dict('line', line_mod.mask_line_lite(disabled_comps=line_disabled_comps), dict=mask_lite_dict)
                # obtain a better fit of emission lines after subtracting continuum models of cont_fit_1
                # here use cont_fit_1['x0_final'] to transfer the best-fit parameters from cont_fit_1
                line_fit_1 = self.nonlinear_process(cont_fit_1['x0_final'], (spec_fmock_w - cont_fit_1['cont_spec_fmod_w']), spec_ferr_w, mask_valid_w, 
                                                    'line', mask_lite_dict, fit_phot=False, fit_grid='linear', conv_nbin=None, 
                                                    annealing=self.init_annealing, perturb_scale=self.perturb_scale, 
                                                    fit_message='line_fit_1: spectral fitting, update emission lines', i_loop=i_loop)
            else:
                # just copy the cont_fit to line_fit, i.e., with zero line flux
                line_fit_1 = cont_fit_1
                self.output_s['line_fit_1'] = self.output_s['cont_fit_1']
            ########################################
            # joint fit of continuum models and emission lines with best-fit parameters of cont_fit_1 and line_fit_1
            model_type = cont_type+'+line' if np.isin('line', self.full_model_type.split('+')) else cont_type
            joint_fit_1 = self.nonlinear_process(line_fit_1['x0_final'], spec_fmock_w, spec_ferr_w, mask_valid_w, 
                                                 model_type, mask_lite_dict, fit_phot=False, fit_grid=self.fit_grid, conv_nbin=self.conv_nbin_max, 
                                                 annealing=False, perturb_scale=self.perturb_scale, accept_chi_sq=max(cont_fit_1['chi_sq'], line_fit_1['chi_sq']),
                                                 fit_message='joint_fit_1: spectral fitting, fit with all models', i_loop=i_loop) 
            ################################################################
            ################################################################

            ################################################################
            ############### Examine models and 2nd fit cycle ###############
            if self.examine_result: 
                ########################################
                ########### Examine models #############
                print_log(center_string(f'Examine if each continuum model is indeed required, i.e., with peak S/N >= {self.accept_model_SN} (set by accept_model_SN).', 80), 
                          self.log_message, self.print_step)
                cont_type = '' # reset
                for mod in joint_fit_1['model_type'].split('+'):
                    if mod == 'line': continue
                    mp0, mp1, me0, me1 = self.search_model_index(mod, joint_fit_1['model_type'], joint_fit_1['mask_lite_dict'])
                    spec_fmod_ew = self.model_dict[mod]['spec_func'](spec_wave_w, joint_fit_1['par_p'][mp0:mp1], mask_lite_e=joint_fit_1['mask_lite_dict'][mod], conv_nbin=1)
                    spec_fmod_w = joint_fit_1['coeff_e'][me0:me1] @ spec_fmod_ew
                    mod_peak_SN, mod_examine = self.examine_model_SN(spec_fmod_w[mask_valid_w], spec_ferr_w[mask_valid_w], accept_SN=self.accept_model_SN)
                    if mod_examine: 
                        cont_type += mod + '+'
                        print_log(f'{mod} continuum peak S/N = {mod_peak_SN:.3f} --> remaining', self.log_message, self.print_step)
                    else:
                        print_log(f'{mod} continuum peak S/N = {mod_peak_SN:.3f} --> disabled', self.log_message, self.print_step)
                if cont_type != '':
                    cont_type = cont_type[:-1] # remove the last '+'
                    print_log(f'#### Continuum models after examination: {cont_type}', self.log_message, self.print_step)
                else:
                    cont_type = 'ssp'
                    print_log(f'#### Continuum is very faint, only stellar continuum model is enabled.', self.log_message, self.print_step)
                # fix the parameters of disabled models (to reduce number of free parameters)
                for mod in joint_fit_1['model_type'].split('+'):
                    if mod == 'line': continue
                    if ~np.isin(mod, cont_type.split('+')): self.model_dict[mod]['cframe'].tie_cp[:,:] = 'fix'
                ########################################
                print_log(center_string(f'Examine if each emission line component is indeed required, i.e., with peak S/N >= {self.accept_model_SN} (set by accept_model_SN).', 80), 
                          self.log_message, self.print_step)
                if np.isin('line', joint_fit_1['model_type'].split('+')): 
                    line_mod = self.model_dict['line']['spec_mod']
                    mp0, mp1, me0, me1 = self.search_model_index('line', joint_fit_1['model_type'], joint_fit_1['mask_lite_dict'])
                    line_spec_fmod_ew = self.model_dict['line']['spec_func'](spec_wave_w, joint_fit_1['par_p'][mp0:mp1], mask_lite_e=joint_fit_1['mask_lite_dict']['line'])
                    line_comps = [] 
                    for i_comp in range(line_mod.num_comps):
                        line_comp = line_mod.cframe.comp_c[i_comp]
                        mask_line_lite = line_mod.mask_line_lite(enabled_comps=[line_comp])[joint_fit_1['mask_lite_dict']['line']]
                        line_comp_spec_fmod_w  = joint_fit_1['coeff_e'][me0:me1][mask_line_lite] @ line_spec_fmod_ew[mask_line_lite, :]
                        line_comp_peak_SN, line_comp_examine = self.examine_model_SN(line_comp_spec_fmod_w[mask_valid_w], spec_ferr_w[mask_valid_w], accept_SN=self.accept_model_SN)
                        if line_comp_examine: 
                            line_comps.append(line_comp)
                            print_log(f'{line_comp} peak S/N = {line_comp_peak_SN:.3f} --> remaining', self.log_message, self.print_step)
                        else:
                            print_log(f'{line_comp} peak S/N = {line_comp_peak_SN:.3f} --> disabled', self.log_message, self.print_step)
                    if len(line_comps) > 0:
                        print_log(f'#### Emission line components after examination: {line_comps}', self.log_message, self.print_step)
                    else:
                        line_comps.append(line_mod.cframe.comp_c[0]) # only use the 1st component if emission lines are too faint
                        print_log(f'#### Emission lines are too faint, only {line_comps} is enabled.', self.log_message, self.print_step)                    
                    # fix the parameters of disabled components (to reduce number of free parameters)
                    for i_comp in range(line_mod.num_comps):
                        if ~np.isin(line_mod.cframe.comp_c[i_comp], line_comps): self.model_dict['line']['cframe'].tie_cp[i_comp,:] = 'fix'                
                    # update mask_lite_dict with emission line examination results, i.e., only keep enabled line components
                    mask_lite_dict = self.update_mask_lite_dict('line', line_mod.mask_line_lite(enabled_comps=line_comps), dict=mask_lite_dict)
                ########################################
                # update parameter_constraints since parameters of disabled model components are fixed
                self.set_par_constraints()
                ########################################
                ########################################

                ########################################
                ############# 2nd fit cycle ############
                # update continuum models after model examination
                # initialize parameters using best-fit of joint_fit_1
                cont_fit_2a = self.nonlinear_process(joint_fit_1['x0_final'], (spec_fmock_w - joint_fit_1['line_spec_fmod_w']), spec_ferr_w, mask_valid_w, 
                                                     cont_type, mask_lite_dict, fit_phot=False, fit_grid=self.fit_grid, conv_nbin=2, 
                                                     annealing=False, perturb_scale=0, 
                                                     fit_message='cont_fit_2a: spectral fitting, update continuum models', i_loop=i_loop) 
                ########################################
                # in steps above, ssp models in a sparse grid of ages (and metalicities) are used, now update continuum fitting with all allowed ssp models
                # initialize parameters using best-fit of cont_fit_2a
                if np.isin('ssp', cont_type.split('+')): 
                    sfh_name = self.model_dict['ssp']['spec_mod'].sfh_names[0]
                    mask_ssp_lite = self.model_dict['ssp']['spec_mod'].mask_ssp_allowed(csp=(sfh_name!='nonparametric'))
                    mask_lite_dict = self.update_mask_lite_dict('ssp', mask_ssp_lite, dict=mask_lite_dict)
                    cont_fit_2b = self.nonlinear_process(cont_fit_2a['x0_final'], (spec_fmock_w - joint_fit_1['line_spec_fmod_w']), spec_ferr_w, mask_valid_w, 
                                                         cont_type, mask_lite_dict, fit_phot=False, fit_grid=self.fit_grid, conv_nbin=2, 
                                                         annealing=False, perturb_scale=self.perturb_scale, 
                                                         fit_message='cont_fit_2b: spectral fitting, update continuum models', i_loop=i_loop)
                    # create new mask_ssp_lite with new ssp_coeffs; do not use full allowed ssp models to save time
                    mp0, mp1, me0, me1 = self.search_model_index('ssp', cont_fit_2b['model_type'], cont_fit_2b['mask_lite_dict'])
                    ssp_coeff_e = cont_fit_2b['coeff_e'][me0:me1]
                    mask_ssp_lite = self.model_dict['ssp']['spec_mod'].mask_ssp_lite_with_coeffs(ssp_coeff_e, num_mods_min=24, verbose=self.print_step)
                    mask_lite_dict = self.update_mask_lite_dict('ssp', mask_ssp_lite, dict=mask_lite_dict)
                else:
                    cont_fit_2b = cont_fit_2a
                ########################################
                if np.isin('line', self.full_model_type.split('+')): 
                    # update emission line with the latest mask_lite_dict for el
                    # initialize parameters from best-fit of joint_fit_1 and subtract continuum models from cont_fit_2b
                    line_fit_2 = self.nonlinear_process(cont_fit_2b['x0_final'], (spec_fmock_w - cont_fit_2b['cont_spec_fmod_w']), spec_ferr_w, mask_valid_w, 
                                                        'line', mask_lite_dict, fit_phot=False, fit_grid='linear', conv_nbin=None, 
                                                        annealing=False, perturb_scale=self.perturb_scale, 
                                                        fit_message='line_fit_2: spectral fitting, update emission lines', i_loop=i_loop)
                else:
                    # just copy the cont_fit to line_fit, i.e., with zero line flux
                    line_fit_2 = cont_fit_2b
                    self.output_s['line_fit_2'] = self.output_s['cont_fit_2b']
                ########################################
                # joint fit of continuum and emission lines with initial values from best-fit of cont_fit_2b and line_fit_2
                model_type = cont_type+'+line' if np.isin('line', self.full_model_type.split('+')) else cont_type
                joint_fit_2 = self.nonlinear_process(line_fit_2['x0_final'], spec_fmock_w, spec_ferr_w, mask_valid_w, 
                                                     model_type, mask_lite_dict, fit_phot=False, fit_grid=self.fit_grid, conv_nbin=self.conv_nbin_max, 
                                                     annealing=False, perturb_scale=0, accept_chi_sq=max(cont_fit_2b['chi_sq'], line_fit_2['chi_sq']),
                                                     fit_message='joint_fit_2: spectral fitting, update all models', i_loop=i_loop)
                # set perturb_scale=0 since this is the final step for pure-spectral fitting
                ########################################
                ########################################
            else:
                # just copy the results of joint_fit_1 to joint_fit_2
                joint_fit_2 = joint_fit_1
                self.output_s['joint_fit_2'] = self.output_s['joint_fit_1']
            ################################################################
            ################################################################

            ################################################################
            ######################## 3rd fit cycle #########################
            # simultaneous spectrum+SED fitting
            if self.have_phot:
                specphot_flux_w = np.hstack((spec_flux_w, phot_flux_b))
                specphot_ferr_w = np.hstack((spec_ferr_w, phot_ferr_b))
                specphot_mask_w = np.hstack((mask_valid_w, mask_valid_b))
                # re-create mock spectrum and SED 
                if i_loop == 0: 
                    print_log(center_string(f'Perform simultaneous spectrum+SED fitting with original data', 80), self.log_message, self.print_step)
                else:
                    print_log(center_string(f'Perform simultaneous spectrum+SED fitting with mock data', 80), self.log_message, self.print_step)
                specphot_fmock_w, specphot_reverr_w = self.create_mock_data(i_loop, ret_phot=True)
                ########################################
                # update model types for coninuum fitting
                cont_type = ''
                for mod in self.full_model_type.split('+'):
                    if (mod != 'line') & self.model_dict[mod]['sed_enable']: cont_type += mod + '+'
                cont_type = cont_type[:-1] # remove the last '+' 
                print_log(f'Continuum models used in spectrum+SED fitting: {cont_type}', self.log_message, self.print_step)
                ########################################
                # spectrum+SED fitting for continuum
                # initialize parameters using best-fit of joint_fit_2; subtract emission lines from joint_fit_2
                cont_fit_3a = self.nonlinear_process(joint_fit_2['x0_final'], specphot_fmock_w - joint_fit_2['line_specphot_fmod_w'],
                                                     specphot_reverr_w, specphot_mask_w, 
                                                     cont_type, mask_lite_dict, fit_phot=True, fit_grid=self.fit_grid, conv_nbin=1, 
                                                     annealing=False, perturb_scale=0, 
                                                     fit_message='cont_fit_3a: spectrum+SED fitting, update continuum models', i_loop=i_loop) 
                # set conv_nbin=1 since this step may not be sensitive to convolved spectral features with scaled errors
                ########################################
                # update mask_lite_dict for ssp for spectrum+SED continuum fitting
                if np.isin('ssp', cont_type.split('+')): 
                    sfh_name = self.model_dict['ssp']['spec_mod'].sfh_names[0]
                    mask_ssp_lite = self.model_dict['ssp']['spec_mod'].mask_ssp_allowed(csp=(sfh_name!='nonparametric'))
                    mask_lite_dict = self.update_mask_lite_dict('ssp', mask_ssp_lite, dict=mask_lite_dict)
                # update scaled error based on chi_sq of cont_fit_3a and re-create mock data
                specphot_fmock_w, specphot_reverr_w = self.create_mock_data(i_loop, ret_phot=True, chi_sq=cont_fit_3a['chi_sq'])
                # use initial best-fit values from cont_fit_3a and subtract emission lines from joint_fit_2
                cont_fit_3b = self.nonlinear_process(cont_fit_3a['x0_final'], specphot_fmock_w - joint_fit_2['line_specphot_fmod_w'],
                                                     specphot_reverr_w, specphot_mask_w, 
                                                     cont_type, mask_lite_dict, fit_phot=True, fit_grid=self.fit_grid, conv_nbin=1, 
                                                     annealing=False, perturb_scale=self.perturb_scale, 
                                                     fit_message='cont_fit_3b: spectrum+SED fitting, update continuum models', i_loop=i_loop)
                # set conv_nbin=1 since this step may not be sensitive to convolved spectral features with scaled errors
                if np.isin('ssp', cont_type.split('+')): 
                    # create new mask_ssp_lite with new ssp_coeffs; do not use full allowed ssp models to save time
                    mp0, mp1, me0, me1 = self.search_model_index('ssp', cont_fit_3b['model_type'], cont_fit_3b['mask_lite_dict'])
                    ssp_coeff_e = cont_fit_3b['coeff_e'][me0:me1]
                    mask_ssp_lite = self.model_dict['ssp']['spec_mod'].mask_ssp_lite_with_coeffs(ssp_coeff_e, num_mods_min=24, verbose=self.print_step)
                    mask_lite_dict = self.update_mask_lite_dict('ssp', mask_ssp_lite, dict=mask_lite_dict)
                ########################################
                if np.isin('line', self.full_model_type.split('+')): 
                    # update emission line after subtracting the lastest continuum models from cont_fit_3b
                    # initialize parameters from best-fit of joint_fit_2 (transfer via cont_fit_3b)
                    # update scaled error based on chi_sq of cont_fit_3b and re-create mock data
                    specphot_fmock_w, specphot_reverr_w = self.create_mock_data(i_loop, ret_phot=True, chi_sq=cont_fit_3b['chi_sq'])
                    line_fit_3 = self.nonlinear_process(cont_fit_3b['x0_final'], specphot_fmock_w - cont_fit_3b['cont_specphot_fmod_w'], 
                                                        specphot_reverr_w, specphot_mask_w, 
                                                        'line', mask_lite_dict, fit_phot=True, fit_grid='linear', conv_nbin=None, 
                                                        annealing=False, perturb_scale=0, 
                                                        fit_message='line_fit_3: spectral fitting, update emission lines', i_loop=i_loop)
                else:
                    # just copy the cont_fit to line_fit, i.e., with zero line flux
                    line_fit_3 = cont_fit_3b
                    self.output_s['line_fit_3'] = self.output_s['cont_fit_3b']
                ########################################
                # joint fit of continuum models and emission lines with initial values from best-fit of cont_fit_3b and line_fit_3
                # update scaled error based on chi_sq of line_fit_3 and re-create mock data
                specphot_fmock_w, specphot_reverr_w = self.create_mock_data(i_loop, ret_phot=True, chi_sq=line_fit_3['chi_sq'])
                model_type = cont_type+'+line' if np.isin('line', self.full_model_type.split('+')) else cont_type
                joint_fit_3 = self.nonlinear_process(line_fit_3['x0_final'], specphot_fmock_w, specphot_reverr_w, specphot_mask_w, 
                                                     model_type, mask_lite_dict, fit_phot=True, fit_grid=self.fit_grid, conv_nbin=2, 
                                                     annealing=False, perturb_scale=0, accept_chi_sq=max(cont_fit_3b['chi_sq'], line_fit_3['chi_sq']),
                                                     fit_message='joint_fit_3: spectrum+SED fitting, update all models', i_loop=i_loop)
                # set conv_nbin=2 instead of self.conv_nbin_max since this step may not be sensitive to convolved spectral features with scaled errors
                # set perturb_scale=0 since this is the final step for spectrum+SED fitting
            ########################################
            ########################################
            
            success_count += 1; total_count += 1
            self.fit_quality_l[i_loop] = True # set as good fits temporarily
            last_chi_sq = joint_fit_2['chi_sq'] if not self.have_phot else joint_fit_3['chi_sq']
            print_log(center_string(f'Loop {i_loop+1}/{self.num_loops} ends, chi_sq = {last_chi_sq:.3f} '+
                                    f'{time.time()-self.time_loop:.1f}s', 80), self.log_message)
                
            # check fitting quality after all loops finished
            # allow additional loops to remove outlier fit; exit if additional loops > 3
            if (success_count == self.num_loops) & (total_count <= (self.num_loops+3)):
                success_count = self.examine_fit_quality() # self.fit_quality_l updated

            if self.save_per_loop:
                self.save_to_file(self.output_filename)

        print_log(center_string(f'{success_count} successful loops in total {total_count} loops, '+
                                f'{time.time()-self.time_init:.1f}s', 80), self.log_message)

        # self.output_s is the core results of the fitting
        # delete the format template with empty values
        if np.isin('empty_step', [*self.output_s]): hide_return = self.output_s.pop('empty_step') 

        # create outputs
        self.extract_results(step='final', print_results=True, return_results=False, num_sed_wave=5000)

        print_log(center_string(f'S3Fit all processes finish', 80), self.log_message)

    ##########################################################################
    ################# Output best-fit spectra and valurs #####################

    def extract_results(self, step=None, print_results=False, return_results=False, num_sed_wave=5000, flux_type='Flam'):
        if (step is None) | (step == 'best') | (step == 'final'):
            step = 'joint_fit_3' if self.have_phot else 'joint_fit_2'
        if (step == 'spec+SED'):  step = 'joint_fit_3'
        if (step == 'spec') | (step == 'pure-spec'): step = 'joint_fit_2'

        best_chi_sq_l   = self.output_s[step]['chi_sq_l']
        best_par_lp     = self.output_s[step]['par_lp']
        best_coeff_le   = self.output_s[step]['coeff_le']
        best_ret_dict_l = self.output_s[step]['ret_dict_l']

        if not self.input_initialized: 
            print_log(f'\n[Note] Re-initialize input data before extracting results.', self.log_message, display=False)
            self.init_input_data(verbose=False) # reset input if data is modified (e.g., in extract_results)
        spec_wave_w = self.spec['wave_w']
        if self.have_phot: 
            if num_sed_wave is not None:
                self.input_initialized = False # set to re-initialize input data in next call of main_fit or extract_results
                self.sed['wave_running_w'] = copy(self.sed['wave_w'])
                self.sed['wave_w'] = np.logspace(np.log10(self.sed['wave_w'].min()), 
                                                 np.log10(self.sed['wave_w'].max()), num=num_sed_wave)
            sed_wave_w = self.sed['wave_w']
            phot_trans_bw = self.pframe.read_transmission(name_b=self.phot_name_b, trans_dir=self.phot_trans_dir, wave_w=sed_wave_w)[1]

        rev_model_type = ''
        for mod in self.full_model_type.split('+'):
            for i_loop in range(self.num_loops): 
                if np.isin(mod, best_ret_dict_l[i_loop]['model_type'].split('+')):
                    if ~np.isin(mod, rev_model_type.split('+')):
                        rev_model_type += mod+'+'
        rev_model_type = rev_model_type[:-1] # remove the last '+'
        self.rev_model_type = rev_model_type # save for indexing output_mc
        print_log(f'The best-fit properties are extracted for the models: {rev_model_type}', self.log_message, self.print_step)

        # format of results
        # output_mc['mod']['comp']['spec_lw'][i_l,i_w]: spectra in observed spectral wavelength
        # output_mc['mod']['comp']['sed_lw'][i_l,i_w]: spectra in full SED wavelength
        # output_mc['mod']['comp']['values']['name_l'][i_l]: calculated values
        # output_mc['mod']['comp']['par_lp'][i_l,i_p]: copied parameters, sorted in the order in the input model_config
        # output_mc['mod']['comp']['coeff_le'][i_l,i_e]: copied coefficients
        # mod: mod0, mod1, ..., tot
        # comp: comp0, comp1, ..., sum
        # i_l: results in the i_l-th loop
        # when comp=sum:
        # output_mc['mod']['sum']['spec_lw'][i_l,i_w]: spectra in observed spectral wavelength
        # output_mc['mod']['sum']['sed_lw'][i_l,i_w]: spectra in full SED wavelength
        # output_mc['mod']['sum']['values']['name_l'][i_l]: calculated values
        # when mod=tot:
        # output_mc['tot']['comp']['spec_lw'][i_l,i_w]: spectra in observed spectral wavelength
        # output_mc['tot']['comp']['sed_lw'][i_l,i_w]: spectra in full SED wavelength
        # output_mc['tot']['comp']['phot_lb'][i_l,i_b]: photometric points in each band
        # comp can be fmod (model), flux (data), fres (residuals), ferr (errors)

        # init the dictionary of the results
        output_mc = {} 
        output_mc['tot'] = {} 
        # write the flux and ferr in each mock loop
        output_mc['tot']['flux'] = {}
        output_mc['tot']['flux']['spec_lw'] = np.array([best_ret_dict_l[i_loop]['flux_w'][:self.num_spec_wave] for i_loop in range(self.num_loops)])
        output_mc['tot']['ferr'] = {}
        output_mc['tot']['ferr']['spec_lw'] = np.array([best_ret_dict_l[i_loop]['ferr_w'][:self.num_spec_wave] for i_loop in range(self.num_loops)])
        if self.have_phot:
            # always use 'joint_fit_3' step since mock phot only created in this step
            output_mc['tot']['flux']['phot_lb'] = np.array([self.output_s['joint_fit_3']['ret_dict_l'][i_loop]['flux_w'][-self.num_phot_band:] for i_loop in range(self.num_loops)])
            output_mc['tot']['ferr']['phot_lb'] = np.array([self.output_s['joint_fit_3']['ret_dict_l'][i_loop]['ferr_w'][-self.num_phot_band:] for i_loop in range(self.num_loops)])

        # init zero formats for models        
        for mod in rev_model_type.split('+'): 
            output_mc[mod] = {} # init results for mod
            comp_c = self.model_dict[mod]['cframe'].comp_c
            num_comps = self.model_dict[mod]['cframe'].num_comps            
            for i_comp in range(num_comps): # init results for each comp of mod
                output_mc[mod][comp_c[i_comp]] = {}
                output_mc[mod][comp_c[i_comp]]['spec_lw'] = np.zeros((self.num_loops, len(spec_wave_w)))
                if self.have_phot:
                    output_mc[mod][comp_c[i_comp]]['sed_lw'] = np.zeros((self.num_loops, len(sed_wave_w)))
            output_mc[mod]['sum'] = {} # init results for the comp's sum for each mod
            output_mc[mod]['sum']['spec_lw'] = np.zeros((self.num_loops, len(spec_wave_w)))
            if self.have_phot:
                output_mc[mod]['sum']['sed_lw'] = np.zeros((self.num_loops, len(sed_wave_w)))
        output_mc['tot']['fmod'] = {} # init results for the total model
        output_mc['tot']['fmod']['spec_lw'] = np.zeros((self.num_loops, len(spec_wave_w)))
        if self.have_phot:
            output_mc['tot']['fmod']['sed_lw'] = np.zeros((self.num_loops, len(sed_wave_w)))

        # extract the best-fit models in spec and spec+SED fitting
        for mod in rev_model_type.split('+'): 
            comp_c = self.model_dict[mod]['cframe'].comp_c
            num_comps = self.model_dict[mod]['cframe'].num_comps
            num_coeffs = self.model_dict[mod]['spec_mod'].num_coeffs
            fp0, fp1, fe0, fe1 = self.search_model_index(mod, self.full_model_type)
            i_e0 = 0; i_e1 = 0
            for i_comp in range(num_comps):
                if mod == 'line':
                    i_e0 += 0 if i_comp == 0 else self.model_dict[mod]['spec_mod'].mask_free_cn[i_comp-1].sum()
                    i_e1 += self.model_dict[mod]['spec_mod'].mask_free_cn[i_comp].sum()
                else:
                    i_e0 += 0 if i_comp == 0 else int(num_coeffs / num_comps)
                    i_e1 += int(num_coeffs / num_comps)

                for i_loop in range(self.num_loops): 
                    spec_fmod_ew = self.model_dict[mod]['spec_func'](spec_wave_w, best_par_lp[i_loop, fp0:fp1], conv_nbin=self.conv_nbin_max)
                    spec_fmod_w  = best_coeff_le[i_loop, fe0:fe1][i_e0:i_e1] @ spec_fmod_ew[i_e0:i_e1]
                    output_mc[mod][comp_c[i_comp]]['spec_lw'][i_loop, :] = spec_fmod_w
                    output_mc[mod]['sum']['spec_lw'][i_loop, :] += spec_fmod_w
                    output_mc['tot']['fmod']['spec_lw'][i_loop, :] += spec_fmod_w
                    if self.have_phot:
                        sed_fmod_ew = self.model_dict[mod]['sed_func'](sed_wave_w, best_par_lp[i_loop, fp0:fp1], conv_nbin=None)
                        sed_fmod_w  = best_coeff_le[i_loop, fe0:fe1][i_e0:i_e1] @ sed_fmod_ew[i_e0:i_e1]
                        output_mc[mod][comp_c[i_comp]]['sed_lw'][i_loop, :] = sed_fmod_w
                        output_mc[mod]['sum']['sed_lw'][i_loop, :] += sed_fmod_w
                        output_mc['tot']['fmod']['sed_lw'][i_loop, :] += sed_fmod_w
        # convert best-fit model SED to phot
        if self.have_phot:
            output_mc['tot']['fmod']['phot_lb'] = self.pframe.spec2phot(sed_wave_w, output_mc['tot']['fmod']['sed_lw'], phot_trans_bw)

        # save fitting residuals
        output_mc['tot']['fres'] = {}
        output_mc['tot']['fres']['spec_lw'] = output_mc['tot']['flux']['spec_lw'] - output_mc['tot']['fmod']['spec_lw']
        if self.have_phot:
            output_mc['tot']['fres']['phot_lb'] = output_mc['tot']['flux']['phot_lb'] - output_mc['tot']['fmod']['phot_lb']

        # save model spectra in flambda to calculate observed flux in later .extract_results()
        self.output_mc = output_mc

        # convert to flux in mJy if required
        if flux_type == 'Fnu':
            for mod in [*output_mc]:
                for comp in [*output_mc[mod]]:
                    if np.isin('spec_lw', [*output_mc[mod][comp]]): 
                        output_mc[mod][comp]['spec_lw'] *= self.spec_flux_scale * PhotFrame.rFnuFlam_func(None,spec_wave_w) # None is for 'self' in PhotFrame definition
                    if np.isin('sed_lw', [*output_mc[mod][comp]]): 
                        output_mc[mod][comp]['sed_lw']  *= self.spec_flux_scale * PhotFrame.rFnuFlam_func(None,sed_wave_w)
                    if np.isin('phot_lb', [*output_mc[mod][comp]]): 
                        output_mc[mod][comp]['phot_lb'] *= self.spec_flux_scale * self.pframe.rFnuFlam_b
            self.input_initialized = False # set to re-initialize input data in next call of main_fit or extract_results
            self.spec['flux_w'] *= self.spec_flux_scale * PhotFrame.rFnuFlam_func(None,spec_wave_w)
            self.spec['ferr_w'] *= self.spec_flux_scale * PhotFrame.rFnuFlam_func(None,spec_wave_w)
            if self.have_phot:
                self.phot['flux_b'] *= self.spec_flux_scale * self.pframe.rFnuFlam_b
                self.phot['ferr_b'] *= self.spec_flux_scale * self.pframe.rFnuFlam_b

        # calculate average spectra
        for mod in rev_model_type.split('+'): 
            self.spec['fmod_'+mod+'_w'] = np.average(output_mc[mod]['sum']['spec_lw'], weights=1/best_chi_sq_l, axis=0)
        self.spec['fmod_tot_w'] = np.average(output_mc['tot']['fmod']['spec_lw'], weights=1/best_chi_sq_l, axis=0)
        self.spec['fres_w'] = self.spec['flux_w'] - self.spec['fmod_tot_w']
        if self.have_phot:
            for mod in rev_model_type.split('+'): 
                self.sed['fmod_'+mod+'_w'] = np.average(output_mc[mod]['sum']['sed_lw'], weights=1/best_chi_sq_l, axis=0)   
            self.sed['fmod_tot_w'] = np.average(output_mc['tot']['fmod']['sed_lw'], weights=1/best_chi_sq_l, axis=0)
            self.phot['fmod_b'] = self.pframe.spec2phot(sed_wave_w, self.sed['fmod_tot_w'], phot_trans_bw)
            self.phot['fres_b'] = self.phot['flux_b'] - self.phot['fmod_b']

        # save best-fit parameters and coefficients of each model, and calculate properties
        for mod in rev_model_type.split('+'):
            if self.have_phot:
                if not (self.model_dict[mod]['spec_enable'] | self.model_dict[mod]['sed_enable']): continue
            else:
                if not self.model_dict[mod]['spec_enable']: continue
            tmp_output_c = self.model_dict[mod]['spec_mod'].extract_results(self, step, print_results, return_results=True, show_average=False)
            comp_c = self.model_dict[mod]['cframe'].comp_c
            num_comps = self.model_dict[mod]['cframe'].num_comps
            for i_comp in range(num_comps): 
                output_mc[mod][comp_c[i_comp]]['par_lp'] = tmp_output_c[comp_c[i_comp]]['par_lp']
                output_mc[mod][comp_c[i_comp]]['coeff_le'] = tmp_output_c[comp_c[i_comp]]['coeff_le']
                output_mc[mod][comp_c[i_comp]]['values'] = tmp_output_c[comp_c[i_comp]]['values']
            output_mc[mod]['sum']['values'] = tmp_output_c['sum']['values']

        # also allow 'el' in output_mc to be compatible with old version 
        if np.isin('line', self.full_model_type.split('+')): self.output_mc['el'] = output_mc['line']

        self.output_mc = output_mc
        if return_results: return output_mc

    def plot_canvas(self, flux_w, model_w, ferr_w, mask_w, fit_phot, fit_message, chi_sq, i_loop):
        if self.canvas is None:
            fig, axs = plt.subplots(2, 1, figsize=(9, 3), dpi=100, gridspec_kw={'height_ratios':[2,1]})
            plt.subplots_adjust(bottom=0.15, top=0.9, hspace=0, wspace=0)
        else:
            fig, axs = self.canvas
        ax1, ax2 = axs; ax1.clear(); ax2.clear()

        tmp_z = (1+self.v0_redshift)
        rest_wave_w = self.spec['wave_w']/tmp_z
        mask_spec_w = mask_w[:self.num_spec_wave]
        ax1.plot(rest_wave_w, self.spec['flux_w'], c='C7', lw=0.3, alpha=0.75, label='Original spectrum')
        ax1.plot(rest_wave_w[mask_spec_w], flux_w[:self.num_spec_wave][mask_spec_w], c='C0', label='Data used for fitting (spec)')
        ax1.plot(rest_wave_w, model_w[:self.num_spec_wave], c='C1', label='Best-fit model (spec)')
        ax2.fill_between(rest_wave_w, -ferr_w[:self.num_spec_wave], ferr_w[:self.num_spec_wave], color='C5', alpha=0.2, label=r'1$\sigma$ error')
        ax2.plot(rest_wave_w[mask_spec_w], (flux_w-model_w)[:self.num_spec_wave][mask_spec_w], c='C2', alpha=0.6, label='Residuals (spec)')
        ax1.fill_between(rest_wave_w, -self.spec['flux_w'].max()*~mask_w[:self.num_spec_wave], self.spec['flux_w'].max()*~mask_w[:self.num_spec_wave], 
                         hatch='////', fc='None', ec='C5', alpha=0.25)
        ax2.fill_between(rest_wave_w, -self.spec['flux_w'].max()*~mask_w[:self.num_spec_wave], self.spec['flux_w'].max()*~mask_w[:self.num_spec_wave], 
                         hatch='////', fc='None', ec='C5', alpha=0.25)
        ax1.set_xlim(rest_wave_w.min()-50, rest_wave_w.max()+50)
        ax2.set_xlim(rest_wave_w.min()-50, rest_wave_w.max()+50)
        if fit_phot:
            rest_wave_b = self.phot['wave_b']/tmp_z
            mask_phot_b = mask_w[-self.num_phot_band:]
            ind_o_b = np.argsort(rest_wave_b)
            ind_m_b = np.argsort(rest_wave_b[mask_phot_b])
            ax1.plot(rest_wave_b[ind_o_b], self.phot['flux_b'][ind_o_b], '--o', c='C7', label='Original phot-SED')
            ax1.plot(rest_wave_b[mask_phot_b][ind_m_b], flux_w[-self.num_phot_band:][mask_phot_b][ind_m_b],'--o',c='C9',label='Data used for fitting (phot)')
            ax1.plot(rest_wave_b[ind_o_b], model_w[-self.num_phot_band:][ind_o_b], '--s', c='C3', label='Best-fit model (phot)')
            ax2.plot(rest_wave_b[ind_o_b],  ferr_w[-self.num_phot_band:][ind_o_b], '--o', c='C5')
            ax2.plot(rest_wave_b[ind_o_b], -ferr_w[-self.num_phot_band:][ind_o_b], '--o', c='C5')
            ax2.plot(rest_wave_b[mask_phot_b][ind_m_b], (flux_w-model_w)[-self.num_phot_band:][mask_phot_b][ind_m_b], '--s', c='C6',label='Residuals (phot)')
            ax1.set_xscale('log'); ax2.set_xscale('log')
            ax1.set_xlim(np.hstack((rest_wave_w,rest_wave_b)).min()*0.9, np.hstack((rest_wave_w,rest_wave_b)).max()*1.1)
            ax2.set_xlim(np.hstack((rest_wave_w,rest_wave_b)).min()*0.9, np.hstack((rest_wave_w,rest_wave_b)).max()*1.1)

        ax1.legend(ncol=2); ax2.legend(ncol=3, loc="lower right")
        ax1.set_ylim(flux_w[mask_w].min()-0.05*(flux_w[mask_w].max()-flux_w[mask_w].min()), flux_w[mask_w].max()*1.05)
        # ax2.set_ylim(-np.percentile(ferr_w[mask_w], 95)*1.1, np.percentile(ferr_w[mask_w], 95)*1.1)
        tmp_ylim = np.percentile(np.abs(flux_w-model_w)[mask_w], 90) * 1.5
        ax2.set_ylim(-tmp_ylim, tmp_ylim)
        ax1.set_xticks([]); ax2.set_xlabel(r'Wavelength ($\AA$)')
        ax1.set_ylabel('Flux ('+str(self.spec_flux_scale)+r' $erg/s/cm2/\AA$)'); ax2.set_ylabel('Res.')
        title = fit_message + r' ($\chi^2_{\nu}$ = ' + f'{chi_sq:.3f}, '
        title += f'loop {i_loop+1}/{self.num_loops}, ' + ('original data)' if i_loop == 0 else 'mock data)')
        ax1.set_title(title)
        if self.canvas is not None:
            fig.canvas.draw(); fig.canvas.flush_events() # refresh plot in the given window
        plt.pause(0.1)  # forces immediate update
