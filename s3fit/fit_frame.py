# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: s3fit@xychen.me

import os, sys, time, traceback, inspect, pickle, gzip
import numpy as np
np.set_printoptions(linewidth=10000)
from copy import deepcopy as copy
from scipy.optimize import lsq_linear, least_squares, dual_annealing
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from pathlib import Path
from importlib.metadata import version as pkg_version, PackageNotFoundError
# check if the fit_frame.py file is in the pip installed directory
if any(m in Path(__file__).resolve().parts for m in ['site-packages', 'dist-packages']):
    try:
        __version__ = pkg_version('s3fit') # use the installed distribution's version
    except PackageNotFoundError:
        __version__ = '0.0.0+unknown' # installed in a weird way or metadata missing
else:
    ##################################
    # manually specified local version
    __version__ = '2.3.0+local'
    ##################################

from .auxiliaries.auxiliary_frames import PhotFrame
from .auxiliaries.auxiliary_functions import print_log, center_string, casefold, convolve_var_width_fft

class FitFrame(object):
    def __init__(self, 
                 # spectral data
                 spec_wave_w=None, spec_flux_w=None, spec_ferr_w=None, 
                 spec_R_inst_w=None, spec_valid_range=None, spec_flux_scale=None, 
                 # photometirc data
                 phot_name_b=None, phot_flux_b=None, phot_ferr_b=None, phot_flux_unit='mJy', phot_trans_dir=None, 
                 sed_wave_w=None, sed_wave_unit='Angstrom', sed_wave_num=None, phot_trans_rsmp=10, 
                 # connection between spectral and photometric data
                 phot_calib_b=None, inst_calib_ratio=0.1, if_rev_inst_calib_ratio=True, inst_calib_smooth=1e4, 
                 if_keep_invalid=False, 
                 # model setup
                 model_config_M=None, norm_wave=5500, norm_width=25, model_R_ratio=2, 
                 v0_redshift=None, if_rev_v0_redshift=False, rev_v0_reference=None, 
                 # mock setup
                 num_mocks=0, if_use_multi_thread=False, num_multi_thread=-1, 
                 # basic fitting control
                 fit_grid='linear', if_examine_result=True, accept_model_SN=2, accept_absorption_SN=None, 
                 # detailed fitting quality control
                 accept_chi_sq=3, nlfit_ntry_max=3, nllsq_ftol_ratio=0.01, conv_nbin_max=5, 
                 if_run_init_annealing=True, da_niter_max=10, perturb_scale=0.02, 
                 # auxiliary
                 if_print_steps=True, if_plot_steps=False, canvas=None, 
                 if_plot_results=True, if_plot_icon=True, 
                 if_save_per_loop=False, output_filename=None, 
                 if_save_test=False, verbose=False, **kwargs): 

        ############################################################
        # check and replace the args to be compatible with old version <= 2.2.4
        if 'inst_calib_ratio_rev' in kwargs: if_rev_inst_calib_ratio = kwargs['inst_calib_ratio_rev']
        if 'keep_invalid'         in kwargs: if_keep_invalid = kwargs['keep_invalid']
        if 'model_config'         in kwargs: model_config_M = kwargs['model_config']
        if 'use_multi_thread'     in kwargs: if_use_multi_thread = kwargs['use_multi_thread']
        if 'examine_result'       in kwargs: if_examine_result = kwargs['examine_result']
        if 'init_annealing'       in kwargs: if_run_init_annealing = kwargs['init_annealing']
        if 'print_step'           in kwargs: if_print_steps = kwargs['print_step']
        if 'plot_step'            in kwargs: if_plot_steps = kwargs['plot_step']
        if 'save_per_loop'        in kwargs: if_save_per_loop = kwargs['save_per_loop']
        if 'save_test'            in kwargs: if_save_test = kwargs['save_test']
        ############################################################

        ############################################################
        # copy and save all input arguments
        self.input_args = {name: copy(value) for name, value in locals().items() if (name != 'self') & (name != 'kwargs')}
        # FitFrame class can be reloaded as self = FitFrame(**FF.input_args)

        # check status of input arguments
        mask_islist_a = [isinstance(self.input_args[name], (list, np.ndarray)) for name in self.input_args] # if inputs are list or array type
        mask_isdefault_a = []
        for i_arg, name in enumerate(self.input_args): 
            if mask_islist_a[i_arg]:
                mask_isdefault_a.append(False)
            elif self.input_args[name] == self.__init__.__defaults__[i_arg]: # if inputs have the default values
                mask_isdefault_a.append(True)
            else:
                mask_isdefault_a.append(False)
        if all(mask_isdefault_a):
            print('[Note] Please input arguments or use FitFrame.reload() to initialize FitFrame.')
            return
        # if not any( isinstance(self.input_args[name], (list, np.ndarray)) for name in self.input_args ): # not any inputs are list or array type
        #     if all( self.input_args[name] == self.__init__.__defaults__[i_arg] for i_arg, name in enumerate(self.input_args) ): # all inputs have the default values

        # check necessary spectrum arguments
        spec_arg_names = ['spec_wave_w', 'spec_flux_w', 'spec_ferr_w', 'spec_R_inst_w']
        for i_arg, name in enumerate(self.input_args):
            if name in spec_arg_names:
                if mask_isdefault_a[i_arg]:
                    raise ValueError((f"The input argument '{name}' is necessary for the fitting."))
                elif not mask_islist_a[i_arg]:
                    raise ValueError((f"The format of the input '{name}' should be list or np.ndarray."))
        if len(set([ len(self.input_args[name]) for i_arg, name in enumerate(self.input_args) if (name in spec_arg_names) & (name != 'spec_R_inst_w') ])) > 1:
            raise ValueError((f"The input {spec_arg_names} should have the same length."))

        # check necessary photometric-SED arguments
        phot_arg_names = ['phot_name_b', 'phot_flux_b', 'phot_ferr_b', 'phot_trans_dir']
        # enable spec+phot fitting if any of the phot_arg_names is input
        if any( not mask_isdefault_a[i_arg] for i_arg, name in enumerate(self.input_args) if (name in phot_arg_names) & (name != 'phot_trans_dir') ):
            self.have_phot = True 
            for i_arg, name in enumerate(self.input_args):
                if name in phot_arg_names:
                    if mask_isdefault_a[i_arg]:
                        raise ValueError((f"The input argument '{name}' is necessary for the fitting with photometric-SED."))
                    elif (not mask_islist_a[i_arg]) & (name != 'phot_trans_dir'):
                        raise ValueError((f"The format of the input '{name}' should be list or np.ndarray."))
            if len(set([ len(self.input_args[name]) for i_arg, name in enumerate(self.input_args) if (name in phot_arg_names) & (name != 'phot_trans_dir') ])) > 1:
                raise ValueError((f"The input {spec_arg_names} should have the same length."))
        else:
            self.have_phot = False

        # check necessary model arguments
        model_arg_names = ['model_config_M', 'v0_redshift']
        for i_arg, name in enumerate(self.input_args):
            if name in model_arg_names:
                if mask_isdefault_a[i_arg]:
                    raise ValueError((f"The input argument '{name}' is necessary for the fitting.")) 
        ############################################################

        # use a list to save message in stdout
        self.log_message = []
        print_log(center_string('S3Fit starts', 80), self.log_message)
        print_log(f"You are now using S3Fit v{__version__}.", self.log_message)

        print_log(center_string('Initialize FitFrame', 80), self.log_message)
        # save spectral data and related properties
        self.spec_wave_w = np.array(spec_wave_w)
        self.spec_flux_w = np.array(spec_flux_w)
        self.spec_ferr_w = np.array(spec_ferr_w)
        self.spec_R_inst_w = np.array(spec_R_inst_w)
        self.spec_valid_range = spec_valid_range
        self.spec_flux_scale = spec_flux_scale # flux_scale is used to avoid too small values

        if self.have_phot:
            # save photometric-SED data and related properties
            self.phot_name_b = np.array(phot_name_b)
            self.phot_flux_b = np.array(phot_flux_b)
            self.phot_ferr_b = np.array(phot_ferr_b)
            self.phot_flux_unit = phot_flux_unit
            self.phot_trans_dir = phot_trans_dir
            # generate wavelength grid to convolve spectra within each filter
            self.sed_wave_w = np.array(sed_wave_w) if sed_wave_w is not None else None
            self.sed_wave_unit = sed_wave_unit
            self.sed_wave_num = sed_wave_num
            self.phot_trans_rsmp = phot_trans_rsmp

            # connection between spectral and photometric data
            self.phot_calib_b = np.array(phot_calib_b) if phot_calib_b is not None else None
            # initial ratio to estimate calibration error
            self.inst_calib_ratio = inst_calib_ratio
            self.if_rev_inst_calib_ratio = if_rev_inst_calib_ratio
            # smoothing width when creating modified error
            self.inst_calib_smooth = inst_calib_smooth

        # whether to keep invalid wavelength range; if true, mock data and models will be created in invalid range
        self.if_keep_invalid = if_keep_invalid

        # load model configuration
        self.model_config_M = copy(model_config_M) # use copy to avoid changing the input config
        # set wavelength and width (in AA) used to normalize model spectra
        self.norm_wave = norm_wave
        self.norm_width = norm_width
        # control on fitting quality: the value equals the ratio of resolution of model (downsampled) / instrument
        self.model_R_ratio = model_R_ratio
        # set initial guess of systemic redshift, all velocity shifts are in relative to v0_redshift
        self.v0_redshift = v0_redshift
        self.if_rev_v0_redshift = if_rev_v0_redshift
        self.rev_v0_reference = rev_v0_reference

        # number of mock data
        self.num_mocks = num_mocks
        print_log(f"Perform fitting for the original data and {self.num_mocks} mock data.", self.log_message)
        # control parallel mock data fitting 
        self.if_use_multi_thread = if_use_multi_thread # if use joblib parallelism with backend='threading'
        self.num_multi_thread = num_multi_thread # number of threads 
        if self.if_use_multi_thread & (self.num_mocks > 1): 
            print_log(f"Perform fitting for the {self.num_mocks} mock data in multithreading with {num_multi_thread if num_multi_thread !=-1 else 'system available'} threads.", self.log_message)

        # basic fitting control
        # fitting grid in linear process
        self.fit_grid = casefold(fit_grid)
        print_log(f"Perform fitting in {self.fit_grid} space.", self.log_message)
        if self.fit_grid == 'log':
            print_log(f"[Note] Pure line fitting (i.e., after subtracting continuum), if enabled, is always in linear space.", self.log_message)
        # fitting steps
        self.if_examine_result = if_examine_result 
        self.accept_model_SN = accept_model_SN
        self.accept_absorption_SN = accept_absorption_SN if accept_absorption_SN is not None else accept_model_SN
        if self.if_examine_result: 
            print_log(f"All continuum models and line components with peak S/N < {self.accept_model_SN} (set with 'accept_model_SN') will be automatically disabled in examination.", self.log_message)
        else:
            print_log(f"[Note] The examination of S/N of models and the updating of fitting will be skipped since 'if_examine_result' is set to False.", self.log_message)

        # detailed fitting quality control
        # 1) fitting accuracy
        # control on fitting quality: nonlinear process, general
        self.accept_chi_sq = accept_chi_sq
        self.nlfit_ntry_max = nlfit_ntry_max 
        # control on fitting quality: nonlinear process, nonlinear least-square
        self.nllsq_ftol_ratio = nllsq_ftol_ratio
        # control on fitting quality: linear process, maximum of bins to perform fft convolution with variable width/resolution
        self.conv_nbin_max = conv_nbin_max
        # 2) searching global minima
        # control on fitting quality: nonlinear process, dual annealing
        self.if_run_init_annealing = if_run_init_annealing
        self.da_niter_max = da_niter_max
        # control on fitting quality: nonlinear process, parameter pertrubation
        self.perturb_scale = perturb_scale 

        # whether to output intermediate results
        self.if_print_steps = if_print_steps # if display in stdout
        self.if_plot_steps = if_plot_steps # if display in matplotlib window
        self.if_plot_results = if_plot_results # if plot the final results
        self.if_plot_icon = if_plot_icon # if display icon in step plots
        self.canvas = canvas # canvas=(fig,ax), to display plots dynamically
        self.if_save_per_loop = if_save_per_loop
        self.output_filename = output_filename
        self.if_save_test = if_save_test # if save the iteration tracing for test
        self.verbose = verbose 

        # alternative input_args
        # arg_list = list(inspect.signature(self.__init__).parameters.values())
        # self.input_args = {arg.name: copy(getattr(self,arg.name,None)) for arg in arg_list if arg.name != 'self'}

        # initialize input formats
        self.init_input_data()
        # import all available models; should be after set_masks to select covered lines
        self.load_models()
        # set constraints of fitting parameters 
        self.init_par_constraints()
        # initialize output formats; should be after init_par_constraints
        self.init_output_results()

        print_log(center_string('Initialization finishes', 80), self.log_message)

    def init_input_data(self, verbose=True):
        print_log(center_string('Read spectral data', 80), self.log_message, verbose)

        ##############################
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

        # save mask
        self.mask_valid_w = mask_valid_w
        ##############################

        mask_keep_w = copy(self.mask_valid_w)
        if self.if_keep_invalid: mask_keep_w[:] = True # mock data and models will be created in invalid range

        if self.spec_flux_scale is None: self.spec_flux_scale = 10.0**np.round(np.log10(np.median(self.spec_flux_w[mask_keep_w])))
        self.spec_flux_w = self.input_args['spec_flux_w'] / self.spec_flux_scale # use input_args to avoid iterative changing of values
        self.spec_ferr_w = self.input_args['spec_ferr_w'] / self.spec_flux_scale # use input_args to avoid iterative changing of values

        # create a dictionary for spectral data
        self.spec = {}
        self.spec['wave_w'] = self.spec_wave_w[mask_keep_w]
        self.spec['flux_w'] = self.spec_flux_w[mask_keep_w]
        self.spec['ferr_w'] = self.spec_ferr_w[mask_keep_w]
        self.spec['mask_valid_w'] = self.mask_valid_w[mask_keep_w]
        self.num_spec_wave = len(self.spec['wave_w'])

        # check spectral resoltuion
        if len(self.spec_R_inst_w) == len(self.spec_wave_w):
            self.spec['R_inst_rw'] = np.vstack((self.spec['wave_w'], self.spec_R_inst_w[mask_keep_w]))
        else:
            print_log(f"[Note] A single value of spectral resolution {self.spec_R_inst_w[1]:.3f} is given at {self.spec_R_inst_w[0]:.3f}AA.", 
                      self.log_message, verbose)
            print_log(f"[Note] Assume a linear wavelength-dependency of spectral resolution in the fitting.", self.log_message, verbose)
            lin_R_inst_w = self.spec['wave_w'] / self.spec_R_inst_w[0] * self.spec_R_inst_w[1]
            self.spec['R_inst_rw'] = np.vstack((self.spec['wave_w'], lin_R_inst_w))

        # account for effective spectral sampling in fitting (all bands are considered as independent)
        self.spec['significance_w'] = np.gradient(self.spec['wave_w']) / (self.spec['wave_w']/self.spec['R_inst_rw'][1,:]) # i.e., dw_pix_inst_w / dw_fwhm_inst_w
        self.spec['significance_w'][self.spec['significance_w'] > 1] = 1

        # set fitting wavelength range (rest frame) with tolerance: voff=[-1000,1000] km/s, fwhm max = 1000 km/s
        voff_tol = 1500; fwhm_tol = 1500
        dw_pad_per_w = 4 * fwhm_tol/np.sqrt(np.log(256))/299792.458 # convolving kernel pad per wavelength (+/-4sigma)
        self.spec_wmin = self.spec['wave_w'].min() / (1+self.v0_redshift) / (1+voff_tol/299792.458) * (1-dw_pad_per_w) #- 100
        self.spec_wmax = self.spec['wave_w'].max() / (1+self.v0_redshift) / (1-voff_tol/299792.458) * (1+dw_pad_per_w) #+ 100
        self.spec_wmin = np.maximum(self.spec_wmin, 912) # set lower limit of wavelength to 912A
        print_log(f"Spectral fitting will be performed in wavelength range (rest frame, AA): from {self.spec_wmin:.3f} to {self.spec_wmax:.3f}", self.log_message, verbose)
        print_log(f"[Note] The wavelength range is extended for tolerances of redshift of {self.v0_redshift}+-{voff_tol/299792.458:.4f} (+-{voff_tol} km/s) "+
                  f"and convolution/dispersion FWHM of max {fwhm_tol} km/s.", self.log_message, verbose)

        # check if norm_wave is coverd in input wavelength range
        if (self.norm_wave < self.spec_wmin) | (self.norm_wave > self.spec_wmax):
            med_wave = np.median(self.spec['wave_w'][self.spec['mask_valid_w']]) / (1+self.v0_redshift)
            med_wave = round(med_wave/100)*100
            print_log(f"[WARNING] The input normalization wavelength (rest frame, AA) {self.norm_wave} is out of the valid range, which is forced to the median valid wavelength {med_wave}.", 
                      self.log_message, verbose)
            self.norm_wave = med_wave

        # create a dictionary for photometric-SED data
        # self.have_phot = True if self.phot_name_b is not None else False
        if self.have_phot:
            print_log(center_string('Read photometric data', 80), self.log_message, verbose)
            print_log(f"Data available in bands: {self.phot_name_b}", self.log_message, verbose)
            # convert input photometric data
            self.pframe = PhotFrame(name_b=self.phot_name_b, flux_b=self.phot_flux_b, ferr_b=self.phot_ferr_b, flux_unit=self.phot_flux_unit,
                                    trans_dir=self.phot_trans_dir, trans_rsmp=self.phot_trans_rsmp, 
                                    wave_w=self.sed_wave_w, wave_unit=self.sed_wave_unit, wave_num=self.sed_wave_num)
            self.pframe.flux_b /= self.spec_flux_scale
            self.pframe.ferr_b /= self.spec_flux_scale

            # set valid band range
            self.mask_valid_b = self.pframe.ferr_b > 0
            mask_keep_b = copy(self.mask_valid_b)
            if self.if_keep_invalid: mask_keep_b[:] = True # mock data and models will be created in invalid range

            # create a dictionary for converted photometric data
            self.phot = {}
            self.phot['wave_b'] = self.pframe.wave_b[mask_keep_b]
            self.phot['flux_b'] = self.pframe.flux_b[mask_keep_b]
            self.phot['ferr_b'] = self.pframe.ferr_b[mask_keep_b]
            self.phot['trans_bw'] = self.pframe.trans_bw[mask_keep_b,:] # transmission curve matrix
            self.phot['mask_valid_b'] = self.mask_valid_b[mask_keep_b]
            self.num_phot_band = len(self.phot['wave_b'])

            # account for effective sampling in fitting, all bands are considered as independent
            self.phot['significance_b'] = np.ones(self.num_phot_band, dtype='float')

            # create self.sed to save the full SED covering all bands
            self.sed = {'wave_w': self.pframe.wave_w} 
            self.num_sed_wave = len(self.pframe.wave_w)
            # set fitting wavelength range (rest frame)
            self.sed_wmin = self.pframe.wave_w.min() / (1+self.v0_redshift)
            self.sed_wmax = self.pframe.wave_w.max() / (1+self.v0_redshift)
            print_log(f"SED fitting is performed in wavelength range (rest frame, AA): from {self.sed_wmin:.3f} to {self.sed_wmax:.3f}", self.log_message, verbose) 

            if self.phot_calib_b is not None:
                # corrent spectrum based on selected photometeic points
                # select fluxes and transmission curves in calibration bands in the order of phot_calib_b
                calib_flux_b = [self.phot['flux_b'][np.where(self.phot_name_b == name_b)[0][0]] for name_b in self.phot_calib_b]
                calib_trans_bw = self.pframe.read_transmission(name_b=self.phot_calib_b, trans_dir=self.phot_trans_dir, wave_w=self.spec['wave_w'])[1]
                # interplote spectrum by masking out invalid range
                spec_flux_interp_w = np.interp(self.spec['wave_w'], self.spec['wave_w'][self.spec['mask_valid_w']], self.spec['flux_w'][self.spec['mask_valid_w']])
                spec_calib_ratio_b = calib_flux_b / self.pframe.spec2phot(self.spec['wave_w'], spec_flux_interp_w, calib_trans_bw)
                self.spec_calib_ratio = spec_calib_ratio_b.mean()
                self.spec['flux_w'] *= self.spec_calib_ratio
                self.spec['ferr_w'] *= self.spec_calib_ratio
                self.spec_flux_w *= self.spec_calib_ratio
                self.spec_ferr_w *= self.spec_calib_ratio
                print_log(f"[Note] The input spectrum is calibrated with photometric fluxes in the bands: {self.phot_calib_b}.", self.log_message, verbose)
                print_log(f"[Note] The calibration ratio for spectrum is {self.spec_calib_ratio}.", self.log_message, verbose)

        self.if_input_modified = False # used to mark if input data is modified in following steps
        self.archived_input = {'spec': copy(self.spec)} # archive the original input data
        if self.have_phot: 
            self.archived_input['phot'] = copy(self.phot)
            self.archived_input['sed']  = copy(self.sed)

    def load_models(self):
        # models init setup
        self.model_dict_M = {}

        ############################################################
        # update old mod_name, 'ssp' and 'el', in model_config_M to be compatible with old version <= 2.2.4
        for mod_name in self.model_config_M:
            if mod_name == 'ssp': self.model_config_M['stellar'] = self.model_config_M.pop('ssp')
            if mod_name == 'el' : self.model_config_M['line']    = self.model_config_M.pop('el')
        ############################################################

        ############################################################
        mod_name = 'stellar'
        if mod_name in self.model_config_M:
            if self.model_config_M[mod_name]['enable']: 
                print_log(center_string('Initialize stellar continuum models', 80), self.log_message)
                from .model_frames.stellar_frame import StellarFrame
                self.model_dict_M[mod_name] = {}
                self.model_dict_M[mod_name]['spec_mod'] = StellarFrame(mod_name=mod_name, fframe=self, 
                                                                       config=self.model_config_M[mod_name]['config'], 
                                                                       file_path=self.model_config_M[mod_name]['file'] if 'file' in self.model_config_M[mod_name] else None, 
                                                                       v0_redshift=self.v0_redshift, R_inst_rw=self.spec['R_inst_rw'], 
                                                                       w_min=self.spec_wmin, w_max=self.spec_wmax, w_norm=self.norm_wave, dw_norm=self.norm_width, 
                                                                       Rratio_mod=self.model_R_ratio, dw_pix_inst=np.median(np.diff(self.spec['wave_w'])), 
                                                                       log_message=self.log_message) 
                self.model_dict_M[mod_name]['spec_enable'] = (self.spec_wmax > 912) & (self.spec_wmin < 1e5)
                if self.have_phot:
                    self.model_dict_M[mod_name]['sed_mod'] = StellarFrame(mod_name=mod_name, fframe=self, 
                                                                          config=self.model_config_M[mod_name]['config'], 
                                                                          file_path=self.model_config_M[mod_name]['file'] if 'file' in self.model_config_M[mod_name] else None, 
                                                                          v0_redshift=self.v0_redshift, R_inst_rw=None, 
                                                                          w_min=self.sed_wmin, w_max=self.sed_wmax, w_norm=self.norm_wave, dw_norm=self.norm_width, 
                                                                          dw_fwhm_dsp=4000/100, dw_pix_inst=None, # convolving with R=100 at rest 4000AA
                                                                          verbose=False) 
                    self.model_dict_M[mod_name]['sed_enable'] = (self.sed_wmax > 912) & (self.sed_wmin < 1e5)
        ############################################################
        mod_name = 'agn'
        if mod_name in self.model_config_M:
            if self.model_config_M[mod_name]['enable']: 
                print_log(center_string('Initialize AGN UV/optical continuum models', 80), self.log_message)
                from .model_frames.agn_frame import AGNFrame
                self.model_dict_M[mod_name] = {}
                self.model_dict_M[mod_name]['spec_mod'] = AGNFrame(mod_name=mod_name, fframe=self, 
                                                                   config=self.model_config_M[mod_name]['config'], 
                                                                   file_path=self.model_config_M[mod_name]['file'] if 'file' in self.model_config_M[mod_name] else None, 
                                                                   v0_redshift=self.v0_redshift, R_inst_rw=self.spec['R_inst_rw'],
                                                                   w_min=self.spec_wmin, w_max=self.spec_wmax, w_norm=self.norm_wave, dw_norm=self.norm_width, 
                                                                   Rratio_mod=self.model_R_ratio, dw_pix_inst=np.median(np.diff(self.spec['wave_w'])), 
                                                                   log_message=self.log_message) 
                self.model_dict_M[mod_name]['spec_enable'] = (self.spec_wmax > 912) & (self.spec_wmin < 1e5)
                if self.have_phot:
                    self.model_dict_M[mod_name]['sed_mod'] = AGNFrame(mod_name=mod_name, fframe=self, 
                                                                      config=self.model_config_M[mod_name]['config'], 
                                                                      file_path=self.model_config_M[mod_name]['file'] if 'file' in self.model_config_M[mod_name] else None, 
                                                                      v0_redshift=self.v0_redshift, R_inst_rw=None, 
                                                                      w_min=self.sed_wmin, w_max=self.sed_wmax, w_norm=self.norm_wave, dw_norm=self.norm_width, 
                                                                      dw_fwhm_dsp=4000/100, dw_pix_inst=None, # convolving with R=100 at rest 4000AA
                                                                      verbose=False) 
                    self.model_dict_M[mod_name]['sed_enable'] = (self.sed_wmax > 912) & (self.sed_wmin < 1e5)
        ############################################################
        mod_name = 'torus'
        if mod_name in self.model_config_M:
            if self.model_config_M[mod_name]['enable']: 
                print_log(center_string('Initialize AGN torus models', 80), self.log_message)
                from .model_frames.torus_frame import TorusFrame
                self.model_dict_M[mod_name] = {}
                self.model_dict_M[mod_name]['spec_mod'] = TorusFrame(mod_name=mod_name, fframe=self, 
                                                                     config=self.model_config_M[mod_name]['config'], 
                                                                     file_path=self.model_config_M[mod_name]['file'] if 'file' in self.model_config_M[mod_name] else None, 
                                                                     v0_redshift=self.v0_redshift, 
                                                                     flux_scale=self.spec_flux_scale, 
                                                                     log_message=self.log_message) 
                self.model_dict_M[mod_name]['spec_enable'] = (self.spec_wmax > 1e4) & (self.spec_wmin < 1e6)
                if self.have_phot:
                    self.model_dict_M[mod_name]['sed_mod'] = self.model_dict_M[mod_name]['spec_mod'] # just copy
                    self.model_dict_M[mod_name]['sed_enable'] = (self.sed_wmax > 1e4) & (self.sed_wmin < 1e6)
        ############################################################
        mod_name = 'line'
        if mod_name in self.model_config_M:
            if self.model_config_M[mod_name]['enable']:
                print_log(center_string('Initialize line models', 80), self.log_message)
                from .model_frames.line_frame import LineFrame
                self.model_dict_M[mod_name] = {}
                self.model_dict_M[mod_name]['spec_mod'] = LineFrame(mod_name=mod_name, fframe=self, 
                                                                    config=self.model_config_M[mod_name]['config'], 
                                                                    use_pyneb=self.model_config_M[mod_name]['use_pyneb'] if 'use_pyneb' in self.model_config_M[mod_name] else False, 
                                                                    v0_redshift=self.v0_redshift, R_inst_rw=self.spec['R_inst_rw'], 
                                                                    w_min=self.spec_wmin, w_max=self.spec_wmax, mask_valid_rw=[self.spec['wave_w'], self.spec['mask_valid_w']], 
                                                                    log_message=self.log_message) 
                self.model_dict_M[mod_name]['spec_enable'] = (self.spec_wmax > 912) & (self.spec_wmin < 1e5) # set range from Lyman break to 10 micron
                if self.have_phot:
                    self.model_dict_M[mod_name]['sed_mod'] = self.model_dict_M[mod_name]['spec_mod'] # just copy, only fit lines in spectral wavelength range
                    self.model_dict_M[mod_name]['sed_enable'] = (self.sed_wmax > 912) & (self.sed_wmin < 1e5)
        ############################################################

        print_log(center_string('Model summary', 80), self.log_message)
        print_log(f"S3Fit imports these models: {[*self.model_dict_M]}.", self.log_message)
        for mod_name in self.model_dict_M:
            if self.have_phot:
                if not (self.model_dict_M[mod_name]['spec_enable'] | self.model_dict_M[mod_name]['sed_enable']): 
                    print_log(f"'{mod_name}' model will not be enabled in the fitting since the defined wavelength range "+
                              f"is not covered by both of the input spectrum and photometric-SED.", self.log_message)
            else:
                if not self.model_dict_M[mod_name]['spec_enable']: 
                    print_log(f"'{mod_name}' model will not be enabled in the spectral fitting since the defined wavelength range "+
                              f"is not covered by the input spectrum.", self.log_message)

        # add short names to simplify the callback
        self.full_model_type = '+'.join([*self.model_dict_M])
        for mod_name in self.model_dict_M:
            setattr(self, mod_name, self.model_dict_M[mod_name]['spec_mod']) 
            self.model_dict_M[mod_name]['cframe'] = self.model_dict_M[mod_name]['spec_mod'].cframe
            self.model_dict_M[mod_name]['num_pars'] = self.model_dict_M[mod_name]['spec_mod'].cframe.num_pars
            self.model_dict_M[mod_name]['num_coeffs'] = self.model_dict_M[mod_name]['spec_mod'].num_coeffs
            self.model_dict_M[mod_name]['spec_func'] = self.model_dict_M[mod_name]['spec_mod'].models_unitnorm_obsframe
            if self.have_phot:
                self.model_dict_M[mod_name]['sed_func'] = self.model_dict_M[mod_name]['sed_mod'].models_unitnorm_obsframe

        # create non-line mask if line is enabled
        if 'line' in self.model_dict_M: 
            linerest_default = self.line.linerest_n[np.isin(self.line.linename_n, self.line.linelist_default)] 
            line_center_n = linerest_default * (1 + self.v0_redshift)
            vel_win = np.array([-3000, 3000])
            mask_line_w = np.zeros_like(self.spec['wave_w'], dtype='bool')
            for i_line in range(len(line_center_n)):
                line_bounds = line_center_n[i_line] * (1 + vel_win/299792.458)
                mask_line_w |= (self.spec['wave_w'] >= line_bounds[0]) & (self.spec['wave_w'] <= line_bounds[1])
            self.spec['mask_noline_w'] = self.spec['mask_valid_w'] & (~mask_line_w)
        else:
            self.spec['mask_noline_w'] = copy(self.spec['mask_valid_w'])
        self.archived_input['spec']['mask_noline_w'] = copy(self.spec['mask_noline_w'])

        ############################################################
        # check old mod_name, 'ssp' and 'el', in tying relations to be compatible with old version <= 2.2.4
        for mod_name in self.model_dict_M:
            mod_cframe = self.model_dict_M[mod_name]['cframe']
            for i_comp in range(mod_cframe.num_comps):
                for i_par in range(mod_cframe.num_pars_c[i_comp]):
                    tmp_tie = mod_cframe.par_tie_cp[i_comp][i_par]
                    if tmp_tie.split(':')[0] == 'ssp': tmp_tie = 'stellar' + tmp_tie[3:]
                    tmp_tie = tmp_tie.replace(';ssp:', ';stellar:')
                    if tmp_tie.split(':')[0] == 'el' : tmp_tie = 'line' + tmp_tie[2:]
                    tmp_tie = tmp_tie.replace(';el:', ';line:')
                    mod_cframe.par_tie_cp[i_comp][i_par] = tmp_tie
        # allow old mod_name, 'ssp' and 'el', in model_dict_M to be compatible with old version <= 2.2.4
        for mod_name in self.model_dict_M:
            if mod_name == 'stellar': self.model_dict_M['ssp'] = self.model_dict_M['stellar']
            if mod_name == 'line'   : self.model_dict_M['el']  = self.model_dict_M['line']
        self.model_dict = self.model_dict_M
        ############################################################

        if self.if_rev_v0_redshift:
            print_log(f"The systemic redshift (v0_redshift) will be updated using the model and component: '{self.rev_v0_reference}'.", self.log_message)
            if self.rev_v0_reference is None:
                raise ValueError((f"Please input the reference of systemic redshift, e.g., rev_v0_reference='model:component'; otherwise set if_rev_v0_redshift=False."))
            elif len(self.rev_v0_reference.split(':')) != 2:
                raise ValueError((f"Please correct for the reference of systemic redshift with the format, rev_v0_reference='model:component'."))
            elif not (self.rev_v0_reference.split(':')[0] in self.model_dict_M):
                raise ValueError((f"The reference model of systemic redshift, '{self.rev_v0_reference.split(':')[0]}', is not available in the imported models: {[*self.model_dict_M]}."))
            elif not (self.rev_v0_reference.split(':')[1] in self.model_dict_M[self.rev_v0_reference.split(':')[0]]['cframe'].comp_name_c):
                raise ValueError((f"The reference component of systemic redshift, '{self.rev_v0_reference.split(':')[1]}', is not available in the model: '{self.rev_v0_reference.split(':')[0]}'."))

    def init_par_constraints(self):
        self.mod_name_p  = np.array([])
        self.comp_name_p = np.array([])
        self.par_name_p  = np.array([])
        self.num_tot_pars   = 0
        self.num_tot_coeffs = 0
        for mod_name in self.full_model_type.split('+'):
            mod_cframe = self.model_dict_M[mod_name]['cframe']
            self.mod_name_p  = np.hstack(( self.mod_name_p , mod_cframe.num_pars * [mod_name] ))
            self.comp_name_p = np.hstack(( self.comp_name_p, mod_cframe.comp_name_p ))
            self.par_name_p  = np.hstack(( self.par_name_p , mod_cframe.par_name_p ))
            self.num_tot_pars   += self.model_dict_M[mod_name]['num_pars']
            self.num_tot_coeffs += self.model_dict_M[mod_name]['num_coeffs']

        # update bounds to match requirement of fitting fucntion, to avoid making conflict in non linear process
        self.par_min_p, self.par_max_p, self.par_tie_p = self.update_tied_pars(model_type=self.full_model_type)

    def init_output_results(self):
        self.num_loops = self.num_mocks+1
        # format to save fitting quality, chi_sq, parameters, and coefficients (normalization factors) of final best-fits    
        self.output_S = {}
        self.output_S['empty_step'] = {}
        self.output_S['empty_step']['chi_sq_l']   = np.zeros( self.num_loops, dtype='float')
        self.output_S['empty_step']['par_lp']     = np.zeros((self.num_loops, self.num_tot_pars  ), dtype='float')
        self.output_S['empty_step']['coeff_le']   = np.zeros((self.num_loops, self.num_tot_coeffs), dtype='float')
        self.output_S['empty_step']['ret_dict_l'] = [{} for i in range(self.num_loops)]

    def save_to_file(self, file):
        with gzip.open(file, 'wb') as f: 
            pickle.dump([self.input_args, self.output_S, self.log_message], f)
        print(f"The input arguments, best-fit results, and running messages are saved to {file} (a python pickle compressed with gzip).")

    def reload_from_file(self, file):
        print(f"FitFrame is reloaded from {file} with the input arguments, best-fit results, and running messages.")
        with gzip.open(file, 'rb') as f: reloaded = pickle.load(f)
        self.__init__(**reloaded[0]) # re-initialize with the arguments
        self.output_S = reloaded[1] # copy the best-fit results
        self.log_message = reloaded[2] # copy the running messages

    ###############################################################################
    ######################### Model Auxiliary Functions ###########################

    def update_mask_lite_Me(self, input_mod_name=None, input_mask_lite_e=None, input_mask_lite_Me=None):
        if input_mask_lite_Me is None:
            ret_mask_lite_Me = {} # create a new mask dict
            for mod_name in self.full_model_type.split('+'):
                ret_mask_lite_Me[mod_name] = np.ones((self.model_dict_M[mod_name]['num_coeffs']), dtype='bool')
        else:
            ret_mask_lite_Me = input_mask_lite_Me # update the input_mask_lite_Me

        if input_mod_name is not None: # update a given mod_name
            ret_mask_lite_Me[input_mod_name] = input_mask_lite_e
        return ret_mask_lite_Me

    def search_mod_index(self, input_mods, model_type, mask_lite_Me=None):
        # return the start and end indexes of par and coeff of input_mods in given model_type

        rev_model_type = ''
        for mod_name in self.full_model_type.split('+'):
            if mod_name in model_type.split('+'): rev_model_type += mod_name + '+'
        rev_model_type = rev_model_type[:-1] # re-sort the given model_type to fit the order in self.full_model_type
        if rev_model_type.split(input_mods)[0] == rev_model_type: 
            raise ValueError((f"No such model combination: {input_mods} in {rev_model_type}"))

        # count num of mod before input_mods in rev_model_type
        i_pars_0   = 0
        i_coeffs_0 = 0
        for mod_name in rev_model_type.split(input_mods)[0].split('+'):
            if mod_name == '': continue
            i_pars_0   += self.model_dict_M[mod_name]['num_pars']
            i_coeffs_0 += self.model_dict_M[mod_name]['num_coeffs'] if mask_lite_Me is None else mask_lite_Me[mod_name].sum()

        # count num of input_mods
        i_pars_1   = i_pars_0   + 0
        i_coeffs_1 = i_coeffs_0 + 0
        for mod_name in input_mods.split('+'): 
            if mod_name == '': continue
            i_pars_1   += self.model_dict_M[mod_name]['num_pars']
            i_coeffs_1 += self.model_dict_M[mod_name]['num_coeffs'] if mask_lite_Me is None else mask_lite_Me[mod_name].sum()

        return i_pars_0, i_pars_1, i_coeffs_0, i_coeffs_1

    def search_comp_index(self, input_comps, mod_name, mask_lite_Ce=None):
        # return the start and end indexes of par and coeff of input_comps among all comps in given mod

        comp_name_c = self.model_dict_M[mod_name]['cframe'].comp_name_c
        num_pars_c  = self.model_dict_M[mod_name]['cframe'].num_pars_c
        num_coeffs_c = self.model_dict_M[mod_name]['spec_mod'].num_coeffs_c

        num_pars_C   = {comp_name:num_pars   for (comp_name,num_pars)   in zip(comp_name_c, num_pars_c)}
        num_coeffs_C = {comp_name:num_coeffs for (comp_name,num_coeffs) in zip(comp_name_c, num_coeffs_c)}

        input_comps_str = '+'.join(input_comps) if isinstance(input_comps, list) else input_comps
        all_comps_str = '+'.join(comp_name_c)
        if all_comps_str.split(input_comps_str)[0] == all_comps_str: 
            raise ValueError((f"No such component combination: {input_comps_str} in the components {all_comps_str} of {mod_name}."))

        # count num of comp before input_comps_str in all_comps_str
        i_pars_0   = 0
        i_coeffs_0 = 0
        for comp_name in all_comps_str.split(input_comps_str)[0].split('+'):
            if comp_name == '': continue
            i_pars_0   += num_pars_C[comp_name]
            i_coeffs_0 += num_coeffs_C[comp_name] if mask_lite_Ce is None else mask_lite_Ce[comp_name].sum()

        # count num of input_comps_str
        i_pars_1   = i_pars_0   + 0
        i_coeffs_1 = i_coeffs_0 + 0
        for comp_name in input_comps_str.split('+'): 
            if comp_name == '': continue
            i_pars_1   += num_pars_C[comp_name]
            i_coeffs_1 += num_coeffs_C[comp_name] if mask_lite_Ce is None else mask_lite_Ce[comp_name].sum()

        return i_pars_0, i_pars_1, i_coeffs_0, i_coeffs_1

    def update_tied_pars(self, model_type=None, input_par_p=None):
        par_p = copy(input_par_p) # avoid changing input_par_p to make confilicts with the outer non-linear process

        min_p = np.array([])
        max_p = np.array([])
        tie_p = np.array([])
        for mod_name in model_type.split('+'):
            mod_cframe = self.model_dict_M[mod_name]['cframe']
            min_p = np.hstack(( min_p, mod_cframe.par_min_p ))
            max_p = np.hstack(( max_p, mod_cframe.par_max_p ))
            tie_p = np.hstack(( tie_p, mod_cframe.par_tie_p ))
        min_p = min_p.astype('float')
        max_p = max_p.astype('float')

        n_freepars = 0
        for i_par in range(len(tie_p)):
            if tie_p[i_par] == 'None': continue
            if tie_p[i_par] == 'free': 
                if par_p is not None:
                    par_p[i_par] = max(par_p[i_par], min_p[i_par]) # re-check if x matches bounds
                    par_p[i_par] = min(par_p[i_par], max_p[i_par])
                    n_freepars += 1
            elif tie_p[i_par] == 'fix': 
                if par_p is not None:
                    par_p[i_par] = min_p[i_par] * 1.0 # avoid changing min_p
                else:
                    max_p[i_par] = min_p[i_par] + 0.01 # avoid conflict of bounds in least_squares; actually not used
            else:
                for single_tie in tie_p[i_par].split(';'):
                    if len(single_tie.split(':')) == 3:
                        ref_mod_name, ref_comp_name, ref_par = single_tie.split(':')
                        tie_sign, tie_fix = None, False
                    elif len(single_tie.split(':')) == 4:
                        ref_mod_name, ref_comp_name, ref_par, tie_sign = single_tie.split(':')
                        tie_fix = False
                    elif len(single_tie.split(':')) == 5:
                        ref_mod_name, ref_comp_name, ref_par, tie_sign, tie_fix = single_tie.split(':')
                        tie_fix = True if tie_fix == 'fix' else False
                    else:
                        raise ValueError((f"The format of the tying relation {single_tie} is not supported. Please follow the format: 'mod_name:comp_name:par_name(:+/x:fix)'."))

                    if not (ref_mod_name in self.full_model_type.split('+')):
                        raise ValueError((f"The reference model '{ref_mod_name}' in tying relation {single_tie} is not provided in {self.full_model_type.split('+')}."))
                    if not (ref_mod_name in model_type.split('+')): continue # skip if the tied mod is not used in this fitting step
                    ref_mod_cframe = self.model_dict_M[ref_mod_name]['cframe']
                    ref_i_pars_0_of_mod, ref_i_pars_1_of_mod = self.search_mod_index(ref_mod_name, model_type)[0:2]

                    if not (ref_comp_name in ref_mod_cframe.comp_name_c):
                        raise ValueError((f"The reference component '{ref_comp_name}' in tying relation {single_tie} is not available in {ref_mod_cframe.comp_name_c}"))
                    ref_i_pars_0_of_comp_in_mod, ref_i_pars_1_of_comp_in_mod = self.search_comp_index(ref_comp_name, ref_mod_name)[0:2]

                    if ref_par.isascii() and ref_par.isdigit(): 
                        ref_i_par_in_comp = int(ref_par) # use par index in tying relation
                    else: 
                        ref_par_name = ref_par
                        ref_par_index_P = ref_mod_cframe.par_index_CP[ref_comp_name]
                        if ref_par_name in ref_par_index_P:
                            ref_i_par_in_comp = ref_par_index_P[ref_par_name]
                        else:
                            raise ValueError((f"The reference par '{ref_par_name}' in tying relation {single_tie} is not available in the parameter list, {[*ref_par_index_P]}."))

                    if par_p is not None:
                        ref_value = par_p[ref_i_pars_0_of_mod:ref_i_pars_1_of_mod][ref_i_pars_0_of_comp_in_mod:ref_i_pars_1_of_comp_in_mod][ref_i_par_in_comp]
                        if tie_sign is None:
                            par_p[i_par] = ref_value
                        elif tie_sign == '+':
                            if tie_fix: 
                                par_p[i_par] = min_p[i_par] + ref_value
                            else:
                                par_p[i_par] += ref_value
                                n_freepars += 1
                        elif tie_sign in ['x', '*']:
                            if tie_fix: 
                                par_p[i_par] = min_p[i_par] * ref_value
                            else:
                                par_p[i_par] *= ref_value
                                n_freepars += 1
                        else:
                            raise ValueError((f"The sign '{tie_sign}' of the tying relation {single_tie} is not supported. Please use the one in ['+','x','*']."))

                        break # only select the 1st effective tie relation
                    else:
                        ref_par_min = min_p[ref_i_pars_0_of_mod:ref_i_pars_1_of_mod][ref_i_pars_0_of_comp_in_mod:ref_i_pars_1_of_comp_in_mod][ref_i_par_in_comp]
                        ref_par_max = max_p[ref_i_pars_0_of_mod:ref_i_pars_1_of_mod][ref_i_pars_0_of_comp_in_mod:ref_i_pars_1_of_comp_in_mod][ref_i_par_in_comp]
                        if tie_sign is None:
                            # copy bounds to replace the input None bounds. 
                            # non-linear process use these values to generate new par_p but the par_p transfered to linear process will be replaced above
                            # --> deprecated since fully fixed pars are not generated in non-linear process
                            min_p[i_par] = ref_par_min
                            max_p[i_par] = ref_par_max
                        elif tie_fix:
                            # copy bounds to replace the input None par_max. 
                            max_p[i_par] = min_p[i_par] + 0.01 # avoid conflict of bounds in least_squares; actually not used        

        if par_p is not None: 
            return par_p, n_freepars
        else:
            return min_p, max_p, tie_p
    
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

        best_chi_sq_l = self.output_S[step]['chi_sq_l'] 
        accept_chi_sq = copy(self.accept_chi_sq)
        if self.num_loops > 1:
            if (best_chi_sq_l[1:] > 0).sum() > 0:
                accept_chi_sq = min(accept_chi_sq, best_chi_sq_l[1:][best_chi_sq_l[1:] > 0].min() * 1.5) # update using the finished fitting of mock data
        self.fit_quality_l = (best_chi_sq_l > 0) & (best_chi_sq_l < accept_chi_sq)
        success_count = self.fit_quality_l.sum()

        if success_count > 0:
            if self.fit_quality_l[0]: 
                tmp_msg = f'original data and {success_count - 1} mock data'
            else:
                tmp_msg = f'{success_count} mock data'
            print_log(f"{success_count} fitting loops ({tmp_msg}) have good quality, with chi_sq = {np.round(best_chi_sq_l[self.fit_quality_l],3)}.", self.log_message)
        else:
            print_log(f"No fitting loop has good quality.", self.log_message)

        if (self.num_loops-success_count) > 0:
            print_log(f"{self.num_loops-success_count} loops need refitting, with current chi_sq = {np.round(best_chi_sq_l[~self.fit_quality_l],3)}.", self.log_message)
        else:
            print_log(f"No fitting loop needs refitting. Run with FitFrame.main_fit(refit=True) to force refitting.", self.log_message)

        return success_count

    def create_mock_data(self, i_loop=None, ret_phot=False, chi_sq=None):
        # copy data
        spec_wave_w, spec_flux_w, spec_ferr_w = self.spec['wave_w'], self.spec['flux_w'], self.spec['ferr_w']
        mask_valid_w = self.spec['mask_valid_w']
        if ret_phot:
            phot_wave_b, phot_flux_b, phot_ferr_b = self.phot['wave_b'], self.phot['flux_b'], self.phot['ferr_b']
            mask_valid_b = self.phot['mask_valid_b']
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
            if (self.if_rev_inst_calib_ratio == True) & (i_loop == 0) & (chi_sq is not None):
                k0_sq = self.inst_calib_ratio**2
                k1_sq = k0_sq * chi_sq - np.median((1 - chi_sq) * specphot_ferr_w[specphot_mask_w]**2 / specphot_flux_w[specphot_mask_w]**2)
                if k1_sq < 0: k1_sq = 0
                self.inst_calib_ratio = min(np.sqrt(k1_sq), 0.2) # set an upperlimit of 0.2
                print_log(f"The ratio of calibration error over flux is updated from {np.sqrt(k0_sq):.3f} to {np.sqrt(k1_sq):.3f}.", 
                          self.log_message, self.if_print_steps)

            # create modified error
            if self.inst_calib_smooth < 1e-4: # use 1e-4 instead of 0 to avoid possible equal==0 error
                # use the original flux 
                specphot_reverr_w = np.sqrt(specphot_ferr_w**2 + specphot_flux_w**2 * self.inst_calib_ratio**2)
            else:
                # create smoothed spectrum
                if not ('joint_fit_1' in self.output_S):
                    # if not fit yet
                    spec_flux_smoothed_w = convolve_var_width_fft(spec_wave_w[mask_valid_w], spec_flux_w[mask_valid_w], dv_fwhm_obj=self.inst_calib_smooth, num_bins=self.conv_nbin_max, reset_edge=False)
                    spec_flux_smoothed_w = np.interp(spec_wave_w, spec_wave_w[mask_valid_w], spec_flux_smoothed_w)
                else:
                    # use joint_fit_1 fitting result to avoid cutting of continuum at edges
                    spec_flux_smoothed_w = spec_flux_w * 0
                    cont_spec_fmod_w = self.output_S['joint_fit_1']['ret_dict_l'][i_loop]['cont_spec_fmod_w']
                    spec_flux_smoothed_w += convolve_var_width_fft(spec_wave_w, cont_spec_fmod_w, dv_fwhm_obj=self.inst_calib_smooth, num_bins=self.conv_nbin_max, reset_edge=True)
                    line_spec_fmod_w = self.output_S['joint_fit_1']['ret_dict_l'][i_loop]['line_spec_fmod_w']
                    spec_flux_smoothed_w += convolve_var_width_fft(spec_wave_w, line_spec_fmod_w, dv_fwhm_obj=self.inst_calib_smooth, num_bins=self.conv_nbin_max, reset_edge=False)
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
            fmod_w = self.output_S[step]['ret_dict_l'][0]['fmod_w']
            fres_w = self.output_S[step]['ret_dict_l'][0]['fres_w']
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

    def linear_process(self, input_par_p, flux_w, ferr_w, mask_w, model_type, mask_lite_Me, 
                       fit_phot=False, fit_grid='linear', conv_nbin=None, ret_par_coeff=False):
        # for a give set of parameters, return models and residuals
        # the residuals are used to solve non-linear least-square fit

        # re-sort the input model_type to fit the order in self.full_model_type
        rev_model_type = ''
        for mod_name in self.full_model_type.split('+'):
            if mod_name in model_type.split('+'): rev_model_type += mod_name + '+'
        rev_model_type = rev_model_type[:-1] 

        par_p, n_freepars = self.update_tied_pars(model_type=rev_model_type, input_par_p=input_par_p)

        spec_wave_w = self.spec['wave_w']
        if fit_phot: sed_wave_w = self.sed['wave_w']
        fit_model_ew = None
        for mod_name in rev_model_type.split('+'):
            i_pars_0, i_pars_1 = self.search_mod_index(mod_name, rev_model_type)[0:2]
            if (~np.isfinite(par_p[i_pars_0:i_pars_1])).any():
                raise ValueError((f"NaN or Inf detected in x = {par_p[i_pars_0:i_pars_1]} of '{mod_name}' model."))
            ####
            spec_fmod_ew = self.model_dict_M[mod_name]['spec_func'](spec_wave_w, par_p[i_pars_0:i_pars_1], mask_lite_e=mask_lite_Me[mod_name], conv_nbin=conv_nbin)
            if fit_phot:
                sed_fmod_ew = self.model_dict_M[mod_name]['sed_func'](sed_wave_w, par_p[i_pars_0:i_pars_1], mask_lite_e=mask_lite_Me[mod_name], conv_nbin=None) #convolution not required
                sed_fmod_eb = self.pframe.spec2phot(sed_wave_w, sed_fmod_ew, self.phot['trans_bw'])
                spec_fmod_ew = np.hstack((spec_fmod_ew, sed_fmod_eb))
            ####
            if (~np.isfinite(spec_fmod_ew)).any():
                self.error = {'spec':spec_fmod_ew, 'wave':spec_wave_w, 'x':par_p[i_pars_0:i_pars_1], 'mask':mask_lite_Me[mod_name], 'conv_nbin':conv_nbin} # output for check
                raise ValueError((f"NaN or Inf detected in returned spectra of '{mod_name}' model with x = {par_p[i_pars_0:i_pars_1]}"
                                 +f", position (m,w) = {np.where(~np.isfinite(spec_fmod_ew))}."))
            spec_fmod_positive_ew = spec_fmod_ew * 1.0 # copy
            spec_fmod_positive_ew[self.model_dict_M[mod_name]['spec_mod'].mask_absorption_e[mask_lite_Me[mod_name]],:] *= -1 # convert nagative values for the following positive check
            if (spec_fmod_positive_ew < 0).any(): 
                self.error = {'spec':spec_fmod_ew, 'wave':spec_wave_w, 'x':par_p[i_pars_0:i_pars_1], 'mask':mask_lite_Me[mod_name], 'conv_nbin':conv_nbin} # output for check
                raise ValueError((f"Negative value detected in returned spectra of '{mod_name}' model with x = {par_p[i_pars_0:i_pars_1]}"
                                 +f", position (m,w) = {np.where(spec_fmod_positive_ew < 0)}."))
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
        
        if not ret_par_coeff:
            # for callback of optimization solvers
            return chi_w*np.sqrt(2)
            # return np.sqrt(2 * chi_sq)
            # then the cost function of least_squares, i.e., 0.5*sum(ret**2), is the chi_sq itself (for check)
            # should match format of jac_matrix: Jac_1x if sqrt(2 * chi_sq); Jac_wx if sqrt(2)*chi_w
        else:
            if not fit_phot:
                print_log(f"Fit with {n_elements} free elements and {n_freepars} free parameters of {len(rev_model_type.split('+'))} models, "
                         +f"reduced chi-squared = {chi_sq:.3f}.", self.log_message, self.if_print_steps)
            else:                
                print_log(f"Fit with {n_elements} free elements and {n_freepars} free parameters of {len(rev_model_type.split('+'))} models: ", 
                          self.log_message, self.if_print_steps)
                print_log(f"Reduced chi-squared with scaled errors = {chi_sq:.3f} for spectrum+SED;", self.log_message, self.if_print_steps)
                spec_chi_sq = (chi_w[:self.num_spec_wave]**2).sum()
                print_log(f"Reduced chi-squared with scaled errors = {spec_chi_sq:.3f} for pure spectrum;", self.log_message, self.if_print_steps)
                phot_chi_sq = (chi_w[-self.num_phot_band:]**2).sum()
                print_log(f"Reduced chi-squared with scaled errors = {phot_chi_sq:.3f} for pure phot-SED;", self.log_message, self.if_print_steps)

                orig_ferr_w = np.hstack((self.spec['ferr_w'], self.phot['ferr_b']))
                tmp_chi_w = np.divide(chi_w*ferr_w, orig_ferr_w, where=mask_valid_w, out=np.zeros_like(ferr_w))
                spec_chi_sq = (tmp_chi_w[:self.num_spec_wave]**2).sum()
                print_log(f"Reduced chi-squared with original errors = {spec_chi_sq:.3f} for pure spectrum;", self.log_message, self.if_print_steps)
                phot_chi_sq = (tmp_chi_w[-self.num_phot_band:]**2).sum()
                print_log(f"Reduced chi-squared with original errors = {phot_chi_sq:.3f} for pure phot-SED.", self.log_message, self.if_print_steps)

            return par_p, coeff_e, model_w, chi_sq

    def jac_matrix(self, function, x, args=(), alpha=0.01, epsilon=1e-4):
        # use custom 3-point jacobian functions to avoid bugs with the scipy internal one (jac='3-point')
        # over small (1e-10) or large (1e-2) h-value could lead to worse fitting
        num_xs = len(x) # number of x
        num_waves = len(args[0]) # if function returns sqrt(2)*chi_w

        Jac_wx = np.zeros((num_waves, num_xs))  # Jacobian matrix
        for i_x in range(num_xs):  # loop over x
            x_forward = x.copy(); x_backward = x.copy()
            x_step = max(alpha * abs(x[i_x]), epsilon)
            x_forward[i_x]  += x_step
            x_backward[i_x] -= x_step
            f_forward_w  = function(x_forward , *args)
            f_backward_w = function(x_backward, *args)
            # compute central difference
            Jac_wx[:, i_x] = (f_forward_w - f_backward_w) / (2 * x_step)
            # if self.if_save_test: 
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

    def nonlinear_process(self, input_par_p, flux_w, ferr_w, mask_w, 
                          model_type, mask_lite_Me, fit_phot=False, fit_grid='linear', conv_nbin=None, 
                          accept_chi_sq=None, nlfit_ntry_max=None, 
                          annealing=False, da_niter_max=None, perturb_scale=None, nllsq_ftol_ratio=None, 
                          fit_message=None, i_loop=None, save_best_fit=True, verbose=False): 
        # core fitting function to obtain solution of non-linear least-square problems

        print_log('#### <'+fit_message.split(':')[0]+'> start:'+fit_message.split(':')[1]+'.', 
                  self.log_message, self.if_print_steps)
        self.time_step = time.time()

        if input_par_p is None: input_par_p = np.random.uniform(self.par_min_p, self.par_max_p) # create random parameters
        if accept_chi_sq is None: accept_chi_sq = self.accept_chi_sq
        if nlfit_ntry_max is None: nlfit_ntry_max = self.nlfit_ntry_max
        if da_niter_max is None: da_niter_max = self.da_niter_max
        if nllsq_ftol_ratio is None: nllsq_ftol_ratio = self.nllsq_ftol_ratio

        # for input of called functions:
        args=(flux_w, ferr_w, mask_w, model_type, mask_lite_Me, fit_phot, fit_grid, conv_nbin)
        
        # create the dictonary to return; copy all input
        frame = inspect.currentframe()
        arg_list = list(self.nonlinear_process.__code__.co_varnames)
        ret_dict = {arg: copy(frame.f_locals[arg]) for arg in arg_list if arg != 'self' and arg != 'frame' and arg in frame.f_locals}
        # save fit_grid; the value transfered in arg may be forced to 'linear' by linear_process if with too many non-positive fluxes
        ret_dict['fit_grid_actual'] = copy(fit_grid)

        if self.if_save_test: 
            self.log_test_args = (input_par_p, args)
            self.log_test_jacs = [] # to trace steps in jacobian
            self.log_test_ret_dict = ret_dict

        mask_p = np.zeros_like(self.par_min_p, dtype='bool') 
        for mod_name in model_type.split('+'):
            i_pars_0_in_allmods, i_pars_1_in_allmods = self.search_mod_index(mod_name, self.full_model_type)[0:2]
            mask_p[i_pars_0_in_allmods:i_pars_1_in_allmods] = True
        
        ##############################################################
        # main fitting cycles
        accept_condition = False
        achieved_chi_sq, achieved_ls_solution = 1e4, None
        for i_fit in range(nlfit_ntry_max):
            try:
                init_par_p = copy(input_par_p) # avoid modify the transferred input_par_p
                if annealing: 
                    print_log(f"Perform Dual Annealing optimazation for a rough global search.", self.log_message, self.if_print_steps)
                    # create randomly initialized parameters used in this step
                    # although x0=init_par_p is not mandatory for dual_annealing, random x0 can enable it to explore from different starting
                    rand_par_p = np.random.uniform(self.par_min_p, self.par_max_p)
                    init_par_p[mask_p] = rand_par_p[mask_p] # update init_par_p used in this step

                    # use dual_annealing with a few iteration for a rough solution of global minima
                    # calling func of dual_annealing should return a scalar; 
                    da_solution = dual_annealing(lambda x, *args: 0.5*(self.linear_process(x, *args)**2).sum(),
                                                 list(zip(self.par_min_p[mask_p], self.par_max_p[mask_p])), 
                                                 args=args, x0=init_par_p[mask_p], no_local_search=True, initial_temp=1e4, visit=1.5, maxiter=da_niter_max)
                    init_par_p[mask_p] = da_solution.x # update init_par_p used in this step

                    if self.if_plot_steps: 
                        par_p, coeff_e, model_w, chi_sq = self.linear_process(da_solution.x, *args, ret_par_coeff=True)
                        self.plot_step(flux_w, model_w, ferr_w, mask_w, fit_phot, '[DA] '+fit_message, chi_sq, i_loop)
                    else:
                        print_log(f"Non-linear fitting cycle {i_fit+1}/{nlfit_ntry_max}, Dual Annealing returns chi_sq = {da_solution.fun:.3f}.", 
                                  self.log_message, self.if_print_steps)
                else:
                    if (perturb_scale > 0) & ((i_fit+1) < nlfit_ntry_max):
                        # do not perturb in the last try of nlfit
                        # pertrub transferred parameters by appending scatters from scaled bound range
                        rand_par_p = input_par_p + np.random.normal(scale=(self.par_max_p - self.par_min_p) * perturb_scale)
                        rand_par_p = np.maximum(rand_par_p, self.par_min_p)
                        rand_par_p = np.minimum(rand_par_p, self.par_max_p)
                        init_par_p[mask_p] = rand_par_p[mask_p] # update init_par_p used in this step 
                        print_log(f"Perturb transferred parameters with scatters of {perturb_scale*100}% of parameter ranges.", 
                                  self.log_message, self.if_print_steps)
                    else:
                        print_log(f"Do not perturb transferred parameters from the former step.", self.log_message, self.if_print_steps)
                ret_dict['init_par_p'] = copy(init_par_p) # save the updated init_par_p

                print_log(f"Perform Non-linear Least-square optimazation for fine tuning.", self.log_message, self.if_print_steps)
                ls_solution = least_squares(fun=self.linear_process, args=args,
                                            x0=init_par_p[mask_p], bounds=(self.par_min_p[mask_p], self.par_max_p[mask_p]), 
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
                    chi_sq = ls_solution.cost
                    accept_condition  =  chi_sq <= (accept_chi_sq * 1.1)
                    accept_condition |= (chi_sq <= (accept_chi_sq * 1.5)) & (achieved_chi_sq <= (accept_chi_sq * 1.5))
                    if accept_condition: 
                        if (chi_sq > accept_chi_sq) & (chi_sq <= (accept_chi_sq * 1.1)):
                            print_log(f"Non-linear fitting cycle {i_fit+1}/{nlfit_ntry_max}, "+
                                      f'accept this fitting with chi_sq = {chi_sq:.3f} / {accept_chi_sq:.3f} (goal) < 110%.', 
                                      self.log_message, self.if_print_steps)
                        if (chi_sq > (accept_chi_sq * 1.1)) & (chi_sq <= (accept_chi_sq * 1.5)):
                            print_log(f"Non-linear fitting cycle {i_fit+1}/{nlfit_ntry_max}, "+
                                      f'accept this fitting with chi_sq = {chi_sq:.3f} / {accept_chi_sq:.3f} (goal) < 150%, achieved twice.', 
                                      self.log_message, self.if_print_steps) 
                        break # exit nlfit cycle
                    else:
                        if chi_sq < achieved_chi_sq: 
                            achieved_chi_sq = copy(chi_sq) # save the achieved min chi_sq
                            achieved_ls_solution = copy(ls_solution)
                        if (i_fit+1) < nlfit_ntry_max:
                            print_log(f"Non-linear fitting cycle {i_fit+1}/{nlfit_ntry_max}, "+
                                      f"poor-fit with chi_sq = {chi_sq:.3f} > {accept_chi_sq:.3f} (goal) --> try refitting; "+
                                      f"achieved min_chi_sq = {achieved_chi_sq:.3f}", self.log_message, self.if_print_steps)
                        else:
                            print_log(f"Non-linear fitting cycle {i_fit+1}/{nlfit_ntry_max}, "+
                                      f"poor-fit with chi_sq = {chi_sq:.3f} > {accept_chi_sq:.3f} (goal); accept the solution with the "+
                                      f"achieved min_chi_sq = {achieved_chi_sq:.3f}", self.log_message, self.if_print_steps)                            
                else:
                    if (i_fit+1) < nlfit_ntry_max:
                        print_log(f"Non-linear fitting cycle {i_fit+1}/{nlfit_ntry_max} failed --> try refitting; "+
                                  f"achieved min_chi_sq = {achieved_chi_sq:.3f}", self.log_message, self.if_print_steps) 
                    else:
                        print_log(f"Non-linear fitting cycle {i_fit+1}/{nlfit_ntry_max} failed; accept the solution with the "+
                                  f"achieved min_chi_sq = {achieved_chi_sq:.3f}", self.log_message, self.if_print_steps) 
        
        # save the best solution
        if accept_condition: 
            best_fit = ls_solution # accept the solution in the final try
        else:
            if achieved_ls_solution is not None:
                best_fit = achieved_ls_solution # back to solution with achieved min_chi_sq
            else:
                best_fit = ls_solution # use the solution in the final try if all tries failed

        par_p, coeff_e, model_w, chi_sq = self.linear_process(best_fit.x, *args, ret_par_coeff=True)
        if self.if_plot_steps: self.plot_step(flux_w, model_w, ferr_w, mask_w, fit_phot, '[LS] '+fit_message, chi_sq, i_loop)
        ##############################################################

        ##############################################################
        # return the best-fit results
        ret_dict['best_fit'] = best_fit
        ret_dict['par_p']    = par_p
        ret_dict['coeff_e']  = coeff_e
        ret_dict['chi_sq']   = chi_sq
        ret_dict['fmod_w']   = model_w
        ret_dict['fres_w']   = flux_w - model_w

        # create best-fit continuum and emission line models for subtracting them in following steps
        ret_dict['cont_spec_fmod_w'] = self.spec['wave_w'] * 0
        ret_dict['line_spec_fmod_w'] = self.spec['wave_w'] * 0
        if self.have_phot:
            ret_dict['cont_specphot_fmod_w'] = np.hstack((self.spec['wave_w'], self.phot['wave_b'])) * 0
            ret_dict['line_specphot_fmod_w'] = np.hstack((self.spec['wave_w'], self.phot['wave_b'])) * 0
        for mod_name in model_type.split('+'):
            i_pars_0, i_pars_1, i_coeffs_0, i_coeffs_1 = self.search_mod_index(mod_name, model_type, mask_lite_Me)
            spec_fmod_ew = self.model_dict_M[mod_name]['spec_func'](self.spec['wave_w'], par_p[i_pars_0:i_pars_1], mask_lite_e=mask_lite_Me[mod_name], conv_nbin=conv_nbin)
            spec_fmod_w = coeff_e[i_coeffs_0:i_coeffs_1] @ spec_fmod_ew
            if mod_name == 'line': 
                ret_dict['line_spec_fmod_w'] += spec_fmod_w
            else:
                ret_dict['cont_spec_fmod_w'] += spec_fmod_w
            if self.have_phot:
                sed_fmod_ew = self.model_dict_M[mod_name]['sed_func'](self.sed['wave_w'], par_p[i_pars_0:i_pars_1], mask_lite_e=mask_lite_Me[mod_name], conv_nbin=None) # convolution no required
                sed_fmod_w = coeff_e[i_coeffs_0:i_coeffs_1] @ sed_fmod_ew
                phot_fmod_b = self.pframe.spec2phot(self.sed['wave_w'], sed_fmod_w, self.phot['trans_bw'])
                if mod_name == 'line': 
                    ret_dict['line_specphot_fmod_w'] += np.hstack((spec_fmod_w, phot_fmod_b))
                else:
                    ret_dict['cont_specphot_fmod_w'] += np.hstack((spec_fmod_w, phot_fmod_b))

        # update the best-fit parameters to the full par_p list (i.e., with all models in self.full_model_type) to guide following fitting
        ret_dict['final_par_p'] = copy(input_par_p)
        for mod_name in model_type.split('+'):
            i_pars_0_in_allmods, i_pars_1_in_allmods = self.search_mod_index(mod_name, self.full_model_type)[0:2]
            i_pars_0_in_selmods, i_pars_1_in_selmods = self.search_mod_index(mod_name, model_type)[0:2]
            ret_dict['final_par_p'][i_pars_0_in_allmods:i_pars_1_in_allmods] = best_fit.x[i_pars_0_in_selmods:i_pars_1_in_selmods] 
            # use best_fit.x instead of the converted par_p here to avoid bounds conflict in later non-linear process
        ##############################################################

        ##############################################################
        # save the best-fit results to the FitFrame class
        if save_best_fit:
            step_id = fit_message.split(':')[0]
            if not (step_id in self.output_S):
                # copy the format template
                self.output_S[step_id] = copy(self.output_S['empty_step'])

            if (self.output_S[step_id]['chi_sq_l'][i_loop] < 0.01) | (self.output_S[step_id]['chi_sq_l'][i_loop] > chi_sq): 
                # save results only if the loop is the 1st run (use < 0.01, not == 0, to make it robust and overwrite overfit), or the refitting gets smaller chi_sq
                self.output_S[step_id]['chi_sq_l'][i_loop] = chi_sq
                for mod_name in model_type.split('+'):
                    i_pars_0_in_allmods, i_pars_1_in_allmods, i_coeffs_0_in_allmods, i_coeffs_1_in_allmods = self.search_mod_index(mod_name, self.full_model_type)
                    i_pars_0_in_selmods, i_pars_1_in_selmods, i_coeffs_0_in_selmods, i_coeffs_1_in_selmods = self.search_mod_index(mod_name, model_type, mask_lite_Me)
                    self.output_S[step_id]['par_lp'  ][i_loop, i_pars_0_in_allmods:i_pars_1_in_allmods]                             = par_p[i_pars_0_in_selmods:i_pars_1_in_selmods]
                    self.output_S[step_id]['coeff_le'][i_loop, i_coeffs_0_in_allmods:i_coeffs_1_in_allmods][mask_lite_Me[mod_name]] = coeff_e[i_coeffs_0_in_selmods:i_coeffs_1_in_selmods]
                self.output_S[step_id]['ret_dict_l'][i_loop] = ret_dict
        ##############################################################

        print_log('#### <'+fit_message.split(':')[0]+'> finish: '+
                  f'{time.time()-self.time_step:.1f}s/'+
                  f'{time.time()-self.time_loop:.1f}s/'+
                  f'{time.time()-self.time_init:.1f}s '+
                  'spent in this step/loop/total.', 
                  self.log_message, self.if_print_steps)            

        return ret_dict

    def single_loop_fit(self, i_loop):

        # copy data
        spec_wave_w = self.spec['wave_w']
        # spec_flux_w = self.spec['flux_w']
        spec_ferr_w = self.spec['ferr_w']
        mask_valid_w = self.spec['mask_valid_w']
        mask_noline_w = self.spec['mask_noline_w']
        if self.have_phot:
            # phot_wave_b = self.phot['wave_b']
            # phot_flux_b = self.phot['flux_b']
            # phot_ferr_b = self.phot['ferr_b']
            mask_valid_b = self.phot['mask_valid_b']
            # specphot_flux_w = np.hstack((spec_flux_w, phot_flux_b))
            # specphot_ferr_w = np.hstack((spec_ferr_w, phot_ferr_b))
            specphot_mask_w = np.hstack((mask_valid_w, mask_valid_b))

        print_log(center_string(f"Loop {i_loop+1}/{self.num_loops} starts " + ('(original data)' if i_loop == 0 else '(mock data)'), 80), self.log_message)
        self.time_loop = time.time()

        if i_loop == 0: 
            print_log(center_string(f"Fit the original spectrum", 80), self.log_message, self.if_print_steps)
        else:
            print_log(center_string(f"Fit the mock spectrum", 80), self.log_message, self.if_print_steps)
        spec_fmock_w = self.create_mock_data(i_loop)
        
        ####################################################
        # # for test
        # sys.exit()
        ####################################################

        ####################################################
        ################## init fit cycle ##################
        # determine model types for continuum fitting
        cont_type = ''
        for mod_name in self.full_model_type.split('+'):
            if (mod_name != 'line') & self.model_dict_M[mod_name]['spec_enable']: cont_type += mod_name + '+'
        cont_type = cont_type[:-1] # remove the last '+' 
        print_log(f"Continuum models used in spectral fitting: {cont_type}", self.log_message, self.if_print_steps)
        ########################################
        # obtain a rough fit of continuum with emission line wavelength ranges masked out
        if 'stellar' in cont_type.split('+'): 
            mask_lite_stellar_e = self.stellar.mask_lite_with_num_mods(num_ages_lite=8, num_mets_lite=1, verbose=self.if_print_steps)
            mask_lite_Me = self.update_mask_lite_Me('stellar', mask_lite_stellar_e)
        else:
            mask_lite_Me = self.update_mask_lite_Me()
        cont_fit_init = self.nonlinear_process(None, spec_fmock_w, spec_ferr_w, mask_noline_w, 
                                               cont_type, mask_lite_Me, fit_phot=False, fit_grid=self.fit_grid, conv_nbin=1, 
                                               annealing=self.if_run_init_annealing, perturb_scale=self.perturb_scale, 
                                               fit_message='cont_fit_init: spectral fitting, initialize continuum models', i_loop=i_loop)
        ########################################
        if 'line' in self.full_model_type.split('+'): 
            # obtain a rough fit of emission lines with continuum of cont_fit_init subtracted
            line_fit_init = self.nonlinear_process(None, (spec_fmock_w - cont_fit_init['cont_spec_fmod_w']), spec_ferr_w, mask_valid_w, 
                                                   'line', mask_lite_Me, fit_phot=False, fit_grid='linear', conv_nbin=None,
                                                   annealing=self.if_run_init_annealing, perturb_scale=self.perturb_scale, 
                                                   fit_message='line_fit_init: spectral fitting, initialize emission lines', i_loop=i_loop)
        else:
            # just copy the cont_fit to line_fit. note that line_fit['line_spec_fmod_w'] = 0, but line_fit['fmod_w'] = cont_fit['fmod_w']
            line_fit_init = cont_fit_init
            self.output_S['line_fit_init'] = self.output_S['cont_fit_init']
        ####################################################
        ####################################################
        
        ####################################################
        ################## 1st fit cycle ###################
        # obtain a better fit of stellar continuum after subtracting emission lines of line_fit_init
        if 'stellar' in cont_type.split('+'): 
            mask_lite_stellar_e = self.stellar.mask_lite_with_num_mods(num_ages_lite=16, num_mets_lite=1, verbose=self.if_print_steps)
            mask_lite_Me = self.update_mask_lite_Me('stellar', mask_lite_stellar_e)
        else:
            mask_lite_Me = self.update_mask_lite_Me()
        cont_fit_1 = self.nonlinear_process(None, (spec_fmock_w - line_fit_init['line_spec_fmod_w']), spec_ferr_w, mask_valid_w, 
                                            cont_type, mask_lite_Me, fit_phot=False, fit_grid=self.fit_grid, conv_nbin=2, 
                                            annealing=self.if_run_init_annealing, perturb_scale=self.perturb_scale, 
                                            fit_message='cont_fit_1: spectral fitting, update continuum models', i_loop=i_loop)
        ########################################
        if 'line' in self.full_model_type.split('+'): 
            # examine if absorption line components are necessary
            line_disabled_comps = [self.line.cframe.comp_name_c[i_comp] for i_comp in range(self.line.num_comps) if self.line.cframe.info_c[i_comp]['sign'] == 'absorption']
            if len(line_disabled_comps) > 0:
                mask_abs_w = mask_valid_w & (line_fit_init['line_spec_fmod_w'] < 0)
                line_abs_peak_SN, line_abs_examine = self.examine_model_SN(-line_fit_init['line_spec_fmod_w'][mask_abs_w], spec_ferr_w[mask_abs_w], accept_SN=self.accept_absorption_SN)
                if not line_abs_examine:
                    print_log(f"Absorption components {line_disabled_comps} are disabled due to low peak S/N = {line_abs_peak_SN:.3f} (abs) < {self.accept_absorption_SN} (set by accept_absorption_SN).",
                              self.log_message, self.if_print_steps)                 
                    # fix the parameters of disabled components (to reduce number of free parameters)
                    for i_comp in range(self.line.num_comps):
                        if self.line.cframe.info_c[i_comp]['sign'] == 'absorption':  self.line.cframe.par_tie_cp[i_comp] = ['fix'] * self.line.cframe.num_pars_c[i_comp]
                    # update mask_lite_Me with emission line examination results, i.e., only keep enabled line components
                    mask_lite_Me = self.update_mask_lite_Me('line', self.line.mask_lite_with_comps(disabled_comps=line_disabled_comps), input_mask_lite_Me=mask_lite_Me)
            # obtain a better fit of emission lines after subtracting continuum models of cont_fit_1
            # here use cont_fit_1['final_par_p'] to transfer the best-fit parameters from cont_fit_1
            line_fit_1 = self.nonlinear_process(cont_fit_1['final_par_p'], (spec_fmock_w - cont_fit_1['cont_spec_fmod_w']), spec_ferr_w, mask_valid_w, 
                                                'line', mask_lite_Me, fit_phot=False, fit_grid='linear', conv_nbin=None, 
                                                annealing=self.if_run_init_annealing, perturb_scale=self.perturb_scale, 
                                                fit_message='line_fit_1: spectral fitting, update emission lines', i_loop=i_loop)
        else:
            # just copy the cont_fit to line_fit, i.e., with zero line flux
            line_fit_1 = cont_fit_1
            self.output_S['line_fit_1'] = self.output_S['cont_fit_1']
        ########################################
        # joint fit of continuum models and emission lines with best-fit parameters of cont_fit_1 and line_fit_1
        model_type = cont_type+'+line' if 'line' in self.full_model_type.split('+') else cont_type
        joint_fit_1 = self.nonlinear_process(line_fit_1['final_par_p'], spec_fmock_w, spec_ferr_w, mask_valid_w, 
                                             model_type, mask_lite_Me, fit_phot=False, fit_grid=self.fit_grid, conv_nbin=self.conv_nbin_max, 
                                             annealing=False, perturb_scale=self.perturb_scale, accept_chi_sq=max(cont_fit_1['chi_sq'], line_fit_1['chi_sq']),
                                             fit_message='joint_fit_1: spectral fitting, fit with all models', i_loop=i_loop) 
        ################################################################
        ################################################################

        ################################################################
        ############### Examine models and 2nd fit cycle ###############
        if self.if_examine_result: 
            ########################################
            ########### Examine models #############
            print_log(center_string(f"Examine if each continuum model is indeed required, i.e., with peak S/N >= {self.accept_model_SN} (set by accept_model_SN).", 80), 
                      self.log_message, self.if_print_steps)
            cont_type = '' # reset
            for mod_name in joint_fit_1['model_type'].split('+'):
                if mod_name == 'line': continue
                i_pars_0, i_pars_1, i_coeffs_0, i_coeffs_1 = self.search_mod_index(mod_name, joint_fit_1['model_type'], joint_fit_1['mask_lite_Me'])
                spec_fmod_ew = self.model_dict_M[mod_name]['spec_func'](spec_wave_w, joint_fit_1['par_p'][i_pars_0:i_pars_1], mask_lite_e=joint_fit_1['mask_lite_Me'][mod_name], conv_nbin=1)
                spec_fmod_w = joint_fit_1['coeff_e'][i_coeffs_0:i_coeffs_1] @ spec_fmod_ew
                mod_peak_SN, mod_examine = self.examine_model_SN(spec_fmod_w[mask_valid_w], spec_ferr_w[mask_valid_w], accept_SN=self.accept_model_SN)
                if mod_examine: 
                    cont_type += mod_name + '+'
                    print_log(f"{mod_name} continuum peak S/N = {mod_peak_SN:.3f} --> remaining", self.log_message, self.if_print_steps)
                else:
                    print_log(f"{mod_name} continuum peak S/N = {mod_peak_SN:.3f} --> disabled ", self.log_message, self.if_print_steps)
            if cont_type != '':
                cont_type = cont_type[:-1] # remove the last '+'
                print_log(f"#### Continuum models after examination: {cont_type}", self.log_message, self.if_print_steps)
            else:
                cont_type = 'stellar'
                print_log(f"#### Continuum is very faint, only stellar continuum model is enabled.", self.log_message, self.if_print_steps)
            # fix the parameters of disabled models (to reduce number of free parameters)
            for mod_name in joint_fit_1['model_type'].split('+'):
                if mod_name == 'line': continue
                if mod_name in cont_type.split('+'): continue
                mod_cframe = self.model_dict_M[mod_name]['cframe']
                for i_comp in range(mod_cframe.num_comps): mod_cframe.par_tie_cp[i_comp] = ['fix'] * mod_cframe.num_pars_c[i_comp]
            ########################################
            print_log(center_string(f"Examine if each emission line component is indeed required, i.e., with peak S/N >= {self.accept_model_SN} (set by accept_model_SN).", 80), 
                      self.log_message, self.if_print_steps)
            if 'line' in joint_fit_1['model_type'].split('+'): 
                i_pars_0, i_pars_1, i_coeffs_0, i_coeffs_1 = self.search_mod_index('line', joint_fit_1['model_type'], joint_fit_1['mask_lite_Me'])
                line_spec_fmod_ew = self.model_dict_M['line']['spec_func'](spec_wave_w, joint_fit_1['par_p'][i_pars_0:i_pars_1], mask_lite_e=joint_fit_1['mask_lite_Me']['line'])
                line_comps = [] 
                for i_comp in range(self.line.num_comps):
                    line_comp = self.line.cframe.comp_name_c[i_comp]
                    line_comp_mask_lite_e = self.line.mask_lite_with_comps(enabled_comps=[line_comp])[joint_fit_1['mask_lite_Me']['line']]
                    line_comp_spec_fmod_w  = joint_fit_1['coeff_e'][i_coeffs_0:i_coeffs_1][line_comp_mask_lite_e] @ line_spec_fmod_ew[line_comp_mask_lite_e, :]
                    line_comp_peak_SN, line_comp_examine = self.examine_model_SN(line_comp_spec_fmod_w[mask_valid_w], spec_ferr_w[mask_valid_w], accept_SN=self.accept_model_SN)
                    if line_comp_examine: 
                        line_comps.append(line_comp)
                        print_log(f"{line_comp} peak S/N = {line_comp_peak_SN:.3f} --> remaining", self.log_message, self.if_print_steps)
                    else:
                        print_log(f"{line_comp} peak S/N = {line_comp_peak_SN:.3f} --> disabled ", self.log_message, self.if_print_steps)
                if len(line_comps) > 0:
                    print_log(f"#### Emission line components after examination: {line_comps}", self.log_message, self.if_print_steps)
                else:
                    line_comps.append(self.line.cframe.comp_name_c[0]) # only use the 1st component if emission lines are too faint
                    print_log(f"#### Emission lines are too faint, only {line_comps} is enabled.", self.log_message, self.if_print_steps)                    
                # fix the parameters of disabled components (to reduce number of free parameters)
                for i_comp in range(self.line.num_comps):
                    if not (self.line.cframe.comp_name_c[i_comp] in line_comps):  self.line.cframe.par_tie_cp[i_comp] = ['fix'] * self.line.cframe.num_pars_c[i_comp]
                # update mask_lite_Me with emission line examination results, i.e., only keep enabled line components
                mask_lite_Me = self.update_mask_lite_Me('line', self.line.mask_lite_with_comps(enabled_comps=line_comps), input_mask_lite_Me=mask_lite_Me)
            ########################################
            # update parameter and constraints since parameters of disabled model components moved to 'fix', to avoid making conflict in non linear process
            self.par_min_p, self.par_max_p, self.par_tie_p = self.update_tied_pars(model_type=self.full_model_type)
            joint_fit_1['update_par_p'] = np.maximum(joint_fit_1['final_par_p'], self.par_min_p)
            joint_fit_1['update_par_p'] = np.minimum(joint_fit_1['final_par_p'], self.par_max_p)
            ########################################
            ########################################

            ########################################
            ############# 2nd fit cycle ############
            # update continuum models after model examination
            # initialize parameters using best-fit of joint_fit_1
            cont_fit_2a = self.nonlinear_process(joint_fit_1['update_par_p'], (spec_fmock_w - joint_fit_1['line_spec_fmod_w']), spec_ferr_w, mask_valid_w, 
                                                 cont_type, mask_lite_Me, fit_phot=False, fit_grid=self.fit_grid, conv_nbin=2, 
                                                 annealing=False, perturb_scale=0, 
                                                 fit_message='cont_fit_2a: spectral fitting, update continuum models', i_loop=i_loop) 
            ########################################
            # in steps above, stellar models in a sparse grid of ages (and metalicities) are used, now update continuum fitting with all allowed stellar models
            # initialize parameters using best-fit of cont_fit_2a
            if 'stellar' in cont_type.split('+'): 
                mask_lite_stellar_e = self.stellar.mask_lite_allowed(csp=(self.stellar.sfh_name_c[0]!='nonparametric'))
                mask_lite_Me = self.update_mask_lite_Me('stellar', mask_lite_stellar_e, input_mask_lite_Me=mask_lite_Me)
                cont_fit_2b = self.nonlinear_process(cont_fit_2a['final_par_p'], (spec_fmock_w - joint_fit_1['line_spec_fmod_w']), spec_ferr_w, mask_valid_w, 
                                                     cont_type, mask_lite_Me, fit_phot=False, fit_grid=self.fit_grid, conv_nbin=2, 
                                                     annealing=False, perturb_scale=self.perturb_scale, 
                                                     fit_message='cont_fit_2b: spectral fitting, update continuum models', i_loop=i_loop)
                # create new mask_lite_stellar_e with new coeffs; do not use full allowed stellar model elements to save time
                i_coeffs_0, i_coeffs_1 = self.search_mod_index('stellar', cont_fit_2b['model_type'], cont_fit_2b['mask_lite_Me'])[2:4]
                mask_lite_stellar_e = self.stellar.mask_lite_with_coeffs(cont_fit_2b['coeff_e'][i_coeffs_0:i_coeffs_1], num_mods_min=24, verbose=self.if_print_steps)
                mask_lite_Me = self.update_mask_lite_Me('stellar', mask_lite_stellar_e, input_mask_lite_Me=mask_lite_Me)
            else:
                cont_fit_2b = cont_fit_2a
            ########################################
            if 'line' in self.full_model_type.split('+'): 
                # update emission line with the latest mask_lite_Me for el
                # initialize parameters from best-fit of joint_fit_1 and subtract continuum models from cont_fit_2b
                line_fit_2 = self.nonlinear_process(cont_fit_2b['final_par_p'], (spec_fmock_w - cont_fit_2b['cont_spec_fmod_w']), spec_ferr_w, mask_valid_w, 
                                                    'line', mask_lite_Me, fit_phot=False, fit_grid='linear', conv_nbin=None, 
                                                    annealing=False, perturb_scale=self.perturb_scale, 
                                                    fit_message='line_fit_2: spectral fitting, update emission lines', i_loop=i_loop)
            else:
                # just copy the cont_fit to line_fit, i.e., with zero line flux
                line_fit_2 = cont_fit_2b
                self.output_S['line_fit_2'] = self.output_S['cont_fit_2b']
            ########################################
            # joint fit of continuum and emission lines with initial values from best-fit of cont_fit_2b and line_fit_2
            model_type = cont_type+'+line' if 'line' in self.full_model_type.split('+') else cont_type
            joint_fit_2 = self.nonlinear_process(line_fit_2['final_par_p'], spec_fmock_w, spec_ferr_w, mask_valid_w, 
                                                 model_type, mask_lite_Me, fit_phot=False, fit_grid=self.fit_grid, conv_nbin=self.conv_nbin_max, 
                                                 annealing=False, perturb_scale=0, accept_chi_sq=max(cont_fit_2b['chi_sq'], line_fit_2['chi_sq']),
                                                 fit_message='joint_fit_2: spectral fitting, update all models', i_loop=i_loop)
            # set perturb_scale=0 since this is the final step for pure-spectral fitting
            ########################################
            ########################################
        else:
            # just copy the results of joint_fit_1 to joint_fit_2
            joint_fit_2 = joint_fit_1
            self.output_S['joint_fit_2'] = self.output_S['joint_fit_1']
        ################################################################
        ################################################################

        ################################################################
        ######################## 3rd fit cycle #########################
        # simultaneous spectrum+SED fitting
        if self.have_phot:
            # re-create mock spectrum and SED 
            if i_loop == 0: 
                print_log(center_string(f"Perform simultaneous spectrum+SED fitting with original data", 80), self.log_message, self.if_print_steps)
            else:
                print_log(center_string(f"Perform simultaneous spectrum+SED fitting with mock data"    , 80), self.log_message, self.if_print_steps)
            specphot_fmock_w, specphot_reverr_w = self.create_mock_data(i_loop, ret_phot=True)
            ########################################
            # update model types for coninuum fitting
            cont_type = ''
            for mod_name in self.full_model_type.split('+'):
                if (mod_name != 'line') & self.model_dict_M[mod_name]['sed_enable']: cont_type += mod_name + '+'
            cont_type = cont_type[:-1] # remove the last '+' 
            print_log(f"Continuum models used in spectrum+SED fitting: {cont_type}", self.log_message, self.if_print_steps)
            ########################################
            # spectrum+SED fitting for continuum
            # initialize parameters using best-fit of joint_fit_2; subtract emission lines from joint_fit_2
            cont_fit_3a = self.nonlinear_process(joint_fit_2['final_par_p'], specphot_fmock_w - joint_fit_2['line_specphot_fmod_w'],
                                                 specphot_reverr_w, specphot_mask_w, 
                                                 cont_type, mask_lite_Me, fit_phot=True, fit_grid=self.fit_grid, conv_nbin=1, 
                                                 annealing=False, perturb_scale=0, 
                                                 fit_message='cont_fit_3a: spectrum+SED fitting, update continuum models', i_loop=i_loop) 
            # set conv_nbin=1 since this step may not be sensitive to convolved spectral features with scaled errors
            ########################################
            # update mask_lite_Me for stellar models for spectrum+SED continuum fitting
            if 'stellar' in cont_type.split('+'): 
                mask_lite_stellar_e = self.stellar.mask_lite_allowed(csp=(self.stellar.sfh_name_c[0]!='nonparametric'))
                mask_lite_Me = self.update_mask_lite_Me('stellar', mask_lite_stellar_e, input_mask_lite_Me=mask_lite_Me)
            # update scaled error based on chi_sq of cont_fit_3a and re-create mock data
            specphot_fmock_w, specphot_reverr_w = self.create_mock_data(i_loop, ret_phot=True, chi_sq=cont_fit_3a['chi_sq'])
            # use initial best-fit values from cont_fit_3a and subtract emission lines from joint_fit_2
            cont_fit_3b = self.nonlinear_process(cont_fit_3a['final_par_p'], specphot_fmock_w - joint_fit_2['line_specphot_fmod_w'],
                                                 specphot_reverr_w, specphot_mask_w, 
                                                 cont_type, mask_lite_Me, fit_phot=True, fit_grid=self.fit_grid, conv_nbin=1, 
                                                 annealing=False, perturb_scale=self.perturb_scale, 
                                                 fit_message='cont_fit_3b: spectrum+SED fitting, update continuum models', i_loop=i_loop)
            # set conv_nbin=1 since this step may not be sensitive to convolved spectral features with scaled errors
            if 'stellar' in cont_type.split('+'): 
                # create new mask_lite_stellar_e with new coeffs; do not use full allowed stellar model elements to save time
                i_coeffs_0, i_coeffs_1 = self.search_mod_index('stellar', cont_fit_3b['model_type'], cont_fit_3b['mask_lite_Me'])[2:4]
                mask_lite_stellar_e = self.stellar.mask_lite_with_coeffs(cont_fit_3b['coeff_e'][i_coeffs_0:i_coeffs_1], num_mods_min=24, verbose=self.if_print_steps)
                mask_lite_Me = self.update_mask_lite_Me('stellar', mask_lite_stellar_e, input_mask_lite_Me=mask_lite_Me)
            ########################################
            if 'line' in self.full_model_type.split('+'): 
                # update emission line after subtracting the lastest continuum models from cont_fit_3b
                # initialize parameters from best-fit of joint_fit_2 (transfer via cont_fit_3b)
                # update scaled error based on chi_sq of cont_fit_3b and re-create mock data
                specphot_fmock_w, specphot_reverr_w = self.create_mock_data(i_loop, ret_phot=True, chi_sq=cont_fit_3b['chi_sq'])
                line_fit_3 = self.nonlinear_process(cont_fit_3b['final_par_p'], specphot_fmock_w - cont_fit_3b['cont_specphot_fmod_w'], 
                                                    specphot_reverr_w, specphot_mask_w, 
                                                    'line', mask_lite_Me, fit_phot=True, fit_grid='linear', conv_nbin=None, 
                                                    annealing=False, perturb_scale=0, 
                                                    fit_message='line_fit_3: spectral fitting, update emission lines', i_loop=i_loop)
            else:
                # just copy the cont_fit to line_fit, i.e., with zero line flux
                line_fit_3 = cont_fit_3b
                self.output_S['line_fit_3'] = self.output_S['cont_fit_3b']
            ########################################
            # joint fit of continuum models and emission lines with initial values from best-fit of cont_fit_3b and line_fit_3
            # update scaled error based on chi_sq of line_fit_3 and re-create mock data
            specphot_fmock_w, specphot_reverr_w = self.create_mock_data(i_loop, ret_phot=True, chi_sq=line_fit_3['chi_sq'])
            model_type = cont_type+'+line' if 'line' in self.full_model_type.split('+') else cont_type
            joint_fit_3 = self.nonlinear_process(line_fit_3['final_par_p'], specphot_fmock_w, specphot_reverr_w, specphot_mask_w, 
                                                 model_type, mask_lite_Me, fit_phot=True, fit_grid=self.fit_grid, conv_nbin=2, 
                                                 annealing=False, perturb_scale=0, accept_chi_sq=max(cont_fit_3b['chi_sq'], line_fit_3['chi_sq']),
                                                 fit_message='joint_fit_3: spectrum+SED fitting, update all models', i_loop=i_loop)
            # set conv_nbin=2 instead of self.conv_nbin_max since this step may not be sensitive to convolved spectral features with scaled errors
            # set perturb_scale=0 since this is the final step for spectrum+SED fitting
        ########################################
        ########################################

        self.fit_quality_l[i_loop] = True # set as good fits temporarily
        last_chi_sq = joint_fit_2['chi_sq'] if not self.have_phot else joint_fit_3['chi_sq']
        print_log(center_string(f"Loop {i_loop+1}/{self.num_loops} ends, chi_sq = {last_chi_sq:.3f} "+
                                f"{time.time()-self.time_loop:.1f}s", 80), self.log_message)

    def main_fit(self, refit=False):
        self.time_init = time.time()

        if self.if_input_modified: 
            # restore the original input data
            self.spec = copy(self.archived_input['spec'])
            if self.have_phot: 
                self.phot = copy(self.archived_input['phot'])
                self.sed  = copy(self.archived_input['sed'])
            self.if_input_modified = False

        # restore the fitting status if it is reloaded
        step = 'joint_fit_3' if self.have_phot else 'joint_fit_2'
        if step in self.output_S: 
            if not refit:
                print_log(center_string(f"Reload the results from the finished fitting loops", 80), self.log_message)
                success_count = self.examine_fit_quality() # self.fit_quality_l updated
            else:
                print_log(center_string(f"Current fitting results are erased; refitting starts", 80), self.log_message)
                self.fit_quality_l[:] = False
                success_count = 0
        else:
            self.fit_quality_l = np.zeros(self.num_loops, dtype='bool')
            success_count = 0

        recycle_count = 0
        while (recycle_count < 3) & (success_count < self.num_loops):
            # pick up loops without good fits
            index_loops = np.where(~self.fit_quality_l)[0]

            if not self.if_use_multi_thread: 
                for i_loop in index_loops:
                    self.single_loop_fit(i_loop) # self.fit_quality_l[i_loop] updated
                    if self.if_save_per_loop: self.save_to_file(self.output_filename)
            else:
                if ~self.fit_quality_l[0]: 
                    # run original data fitting individually
                    self.single_loop_fit(0) 
                    index_loops = index_loops[1:] # remove 0th loop if it is included
                # run mock data fitting in parallel
                _ = Parallel(n_jobs=self.num_multi_thread, backend='threading')(delayed(self.single_loop_fit)(i_loop) for i_loop in index_loops)
                if self.if_save_per_loop: self.save_to_file(self.output_filename)

            # check fitting quality after all loops finished
            # allow additional loops to remove outlier fit; exit if additional loops > 3
            success_count = self.examine_fit_quality() # self.fit_quality_l updated
            recycle_count += 1

        print_log(center_string(f"{success_count} successful loops in {recycle_count} recyles, {time.time()-self.time_init:.1f}s", 80), self.log_message)
        print_log('', self.log_message)
        print_log(center_string(f"Best-fit results", 80), self.log_message)

        # self.output_S is the core results of the fitting
        # delete the format template with empty values
        if 'empty_step' in self.output_S: not_used = self.output_S.pop('empty_step') 
        # extract results
        self.extract_results(step='final', if_print_results=True, if_rev_v0_redshift=self.if_rev_v0_redshift)

        if self.if_plot_results:
            self.plot_results(step='spec', if_plot_phot=False, if_plot_comp=True, 
                              plot_range='spec', wave_type='rest', wave_unit='Angstrom', 
                              flux_type='Flam', res_type='residual', ferr_num=3, xyscale=('linear','linear'), 
                              title='Best-fit models of pure-spectral fitting with all components ' + r'($\chi^2_{\nu}$ = ' + f"{self.output_S['joint_fit_2']['chi_sq_l'][0]:.3f})")

        if self.if_plot_results & self.have_phot:
            self.plot_results(step='spec+SED', if_plot_phot=True, if_plot_comp=True, 
                              plot_range='SED', wave_type='rest', wave_unit='micron', 
                              flux_type='Fnu', res_type='residual/data', ferr_num=3, xyscale=('log','log'), 
                              title='Best-fit models of spectrum+SED fitting with all components ' + r'($\chi^2_{\nu}$ = ' + f"{self.output_S['joint_fit_3']['chi_sq_l'][0]:.3f})")

        print_log(center_string(f"S3Fit all processes finish", 80), self.log_message)

    ##########################################################################
    ################# Extract best-fit spectra and values ####################

    def extract_results(self, step=None, if_print_results=False, if_return_results=False, if_rev_v0_redshift=None, if_show_average=False, num_sed_wave=5000, flux_type='Flam', **kwargs):

        ############################################################
        # check and replace the args to be compatible with old version <= 2.2.4
        if 'print_results'  in kwargs: if_print_results = kwargs['print_results']
        if 'return_results' in kwargs: if_return_results = kwargs['return_results']
        ############################################################

        if (step is None) | (step in ['best', 'final']): step = 'joint_fit_3' if self.have_phot else 'joint_fit_2'
        if  step in ['spec+SED', 'spectrum+SED']:  step = 'joint_fit_3'
        if  step in ['spec', 'pure-spec', 'spectrum', 'pure-spectrum']:  step = 'joint_fit_2'

        best_chi_sq_l   = copy(self.output_S[step]['chi_sq_l'])
        best_par_lp     = copy(self.output_S[step]['par_lp'])
        best_coeff_le   = copy(self.output_S[step]['coeff_le'])
        best_ret_dict_l = self.output_S[step]['ret_dict_l']

        if self.if_input_modified: 
            # restore the original input data
            self.spec = copy(self.archived_input['spec'])
            if self.have_phot: 
                self.phot = copy(self.archived_input['phot'])
                self.sed  = copy(self.archived_input['sed'])
            self.if_input_modified = False

        spec_wave_w = self.spec['wave_w']
        if self.have_phot: 
            if num_sed_wave is not None:
                # re-generate wavelength grid to output better sed
                self.sed['wave_w'] = np.logspace(np.log10(self.sed['wave_w'].min()), np.log10(self.sed['wave_w'].max()), num=num_sed_wave)
                self.if_input_modified = True
            sed_wave_w = self.sed['wave_w']
            phot_trans_bw = self.pframe.read_transmission(name_b=self.phot_name_b, trans_dir=self.phot_trans_dir, wave_w=sed_wave_w)[1]

        rev_model_type = ''
        for mod_name in self.full_model_type.split('+'):
            for i_loop in range(self.num_loops): 
                if mod_name in best_ret_dict_l[i_loop]['model_type'].split('+'):
                    if not (mod_name in rev_model_type.split('+')):
                        rev_model_type += mod_name + '+'
        rev_model_type = rev_model_type[:-1] # remove the last '+'
        self.rev_model_type = rev_model_type # save for indexing output_MC
        print_log(f"The best-fit properties are extracted for the models: {rev_model_type}", self.log_message, self.if_print_steps)

        if if_rev_v0_redshift is None: if_rev_v0_redshift = self.if_rev_v0_redshift
        if if_rev_v0_redshift:
            mask_v0_p  = self.par_name_p  == 'voff'
            mask_v0_p &= self.mod_name_p  == self.rev_v0_reference.split(':')[0]
            mask_v0_p &= self.comp_name_p == self.rev_v0_reference.split(':')[1]
            if sum(mask_v0_p) == 1:
                # get the updated systemic redshift
                self.ref_voff_l = best_par_lp[:, mask_v0_p][:,0] # [:,0] is required to keep _l
                self.rev_v0_redshift_l = (1+self.v0_redshift) * (1+self.ref_voff_l/299792.458) - 1
                self.rev_v0_redshift = self.rev_v0_redshift_l[0]
                self.rev_v0_redshift_std = self.rev_v0_redshift_l.std()
                # update best-fit voff and fwhm
                best_par_lp[:, self.par_name_p == 'voff'] -= self.ref_voff_l[0]
                best_par_lp[:, self.par_name_p == 'fwhm'] *= (1+self.v0_redshift) / (1+self.rev_v0_redshift)
                # update v0_redshift in each model frame
                for mod_name in rev_model_type.split('+'): 
                    self.model_dict_M[mod_name]['spec_mod'].v0_redshift = self.rev_v0_redshift
                    if self.have_phot: self.model_dict_M[mod_name]['sed_mod'].v0_redshift = self.rev_v0_redshift
                print_log(f"The systemic redshift (v0_redshift) is updated to {self.rev_v0_redshift:.6f}+/-{self.rev_v0_redshift_std:.6f} (from the input {self.v0_redshift}) " + 
                          f"referring to the model and component: '{self.rev_v0_reference}'.", self.log_message, self.if_print_steps)
                print_log(f"The related best-fit results (e.g., shifted velocities) are also updated.", self.log_message, self.if_print_steps)
            else:
                self.rev_v0_redshift = None
                print_log(f"[WARNING] The specified reference component of systemic redshift, '{self.rev_v0_reference}', is not available. The redshift updating will be skipped.", 
                          self.log_message, self.if_print_steps)
        else:
            self.rev_v0_redshift = None
            # recover un-updated v0_redshift, if it is changed, in each model frame
            for mod_name in rev_model_type.split('+'): 
                if self.model_dict_M[mod_name]['spec_mod'].v0_redshift != self.v0_redshift:
                    self.model_dict_M[mod_name]['spec_mod'].v0_redshift = self.v0_redshift
                    if self.have_phot: self.model_dict_M[mod_name]['sed_mod'].v0_redshift = self.v0_redshift

        # format of results
        # output_MC['mod']['comp']['spec_lw'][i_l,i_w]: spectra in observed spectral wavelength
        # output_MC['mod']['comp']['sed_lw'][i_l,i_w]: spectra in full SED wavelength
        # output_MC['mod']['comp']['value_Vl']['name_l'][i_l]: calculated values
        # output_MC['mod']['comp']['par_lp'][i_l,i_p]: copied parameters, sorted in the order in the input model_config_M
        # output_MC['mod']['comp']['coeff_le'][i_l,i_e]: copied coefficients
        # mod: mod0, mod1, ..., tot
        # comp: comp0, comp1, ..., sum
        # i_l: results in the i_l-th loop
        # when comp=sum:
        # output_MC['mod']['sum']['spec_lw'][i_l,i_w]: spectra in observed spectral wavelength
        # output_MC['mod']['sum']['sed_lw'][i_l,i_w]: spectra in full SED wavelength
        # output_MC['mod']['sum']['value_Vl']['name_l'][i_l]: calculated values
        # when mod_name=tot:
        # output_MC['tot']['comp']['spec_lw'][i_l,i_w]: spectra in observed spectral wavelength
        # output_MC['tot']['comp']['sed_lw'][i_l,i_w]: spectra in full SED wavelength
        # output_MC['tot']['comp']['phot_lb'][i_l,i_b]: photometric points in each band
        # comp can be fmod (model), flux (data), fres (residuals), ferr (errors)

        # init the dictionary of the results
        output_MC = {} 
        output_MC['tot'] = {} 
        # write the flux and ferr in each mock loop
        output_MC['tot']['flux'] = {}
        output_MC['tot']['flux']['spec_lw'] = np.array([best_ret_dict_l[i_loop]['flux_w'][:self.num_spec_wave] for i_loop in range(self.num_loops)])
        output_MC['tot']['ferr'] = {}
        output_MC['tot']['ferr']['spec_lw'] = np.array([best_ret_dict_l[i_loop]['ferr_w'][:self.num_spec_wave] for i_loop in range(self.num_loops)])
        if self.have_phot:
            # always use 'joint_fit_3' step since mock phot only created in this step
            output_MC['tot']['flux']['phot_lb'] = np.array([self.output_S['joint_fit_3']['ret_dict_l'][i_loop]['flux_w'][-self.num_phot_band:] for i_loop in range(self.num_loops)])
            output_MC['tot']['ferr']['phot_lb'] = np.array([self.output_S['joint_fit_3']['ret_dict_l'][i_loop]['ferr_w'][-self.num_phot_band:] for i_loop in range(self.num_loops)])

        # init zero formats for models        
        for mod_name in rev_model_type.split('+'): 
            output_MC[mod_name] = {} # init results for mod_name
            comp_name_c = self.model_dict_M[mod_name]['cframe'].comp_name_c
            for (i_comp, comp_name) in enumerate(comp_name_c):
                output_MC[mod_name][comp_name] = {} # init results for each comp of mod_name
                output_MC[mod_name][comp_name]['spec_lw'] = np.zeros((self.num_loops, len(spec_wave_w)))
                if self.have_phot:
                    output_MC[mod_name][comp_name]['sed_lw'] = np.zeros((self.num_loops, len(sed_wave_w)))
            output_MC[mod_name]['sum'] = {} # init results for the comp's sum for each mod_name
            output_MC[mod_name]['sum']['spec_lw'] = np.zeros((self.num_loops, len(spec_wave_w)))
            if self.have_phot:
                output_MC[mod_name]['sum']['sed_lw'] = np.zeros((self.num_loops, len(sed_wave_w)))
        output_MC['tot']['fmod'] = {} # init results for the total model
        output_MC['tot']['fmod']['spec_lw'] = np.zeros((self.num_loops, len(spec_wave_w)))
        if self.have_phot:
            output_MC['tot']['fmod']['sed_lw'] = np.zeros((self.num_loops, len(sed_wave_w)))

        # extract the best-fit models in spec and spec+SED fitting
        for mod_name in rev_model_type.split('+'): 
            comp_name_c = self.model_dict_M[mod_name]['cframe'].comp_name_c
            num_coeffs_c = self.model_dict_M[mod_name]['spec_mod'].num_coeffs_c
            i_pars_0_of_mod, i_pars_1_of_mod, i_coeffs_0_of_mod, i_coeffs_1_of_mod = self.search_mod_index(mod_name, self.full_model_type)
            for i_loop in range(self.num_loops): 
                spec_fmod_ew = self.model_dict_M[mod_name]['spec_func'](spec_wave_w, best_par_lp[i_loop, i_pars_0_of_mod:i_pars_1_of_mod], conv_nbin=self.conv_nbin_max)
                if self.have_phot:
                    sed_fmod_ew = self.model_dict_M[mod_name]['sed_func'](sed_wave_w, best_par_lp[i_loop, i_pars_0_of_mod:i_pars_1_of_mod], conv_nbin=None)
                coeff_fmod_e = best_coeff_le[i_loop, i_coeffs_0_of_mod:i_coeffs_1_of_mod]
                # i_e0 = 0; i_e1 = 0
                for (i_comp, comp_name) in enumerate(comp_name_c):
                    # i_e0 += 0 if i_comp == 0 else num_coeffs_c[i_comp-1]
                    # i_e1 += num_coeffs_c[i_comp]
                    i_coeffs_0_of_comp_in_mod, i_coeffs_1_of_comp_in_mod = self.search_comp_index(comp_name, mod_name)[2:4]
                    spec_fmod_w  = coeff_fmod_e[i_coeffs_0_of_comp_in_mod:i_coeffs_1_of_comp_in_mod] @ spec_fmod_ew[i_coeffs_0_of_comp_in_mod:i_coeffs_1_of_comp_in_mod]
                    output_MC[mod_name][comp_name]['spec_lw'][i_loop, :] = spec_fmod_w
                    output_MC[mod_name]['sum']['spec_lw'][i_loop, :] += spec_fmod_w
                    output_MC['tot']['fmod']['spec_lw'][i_loop, :] += spec_fmod_w
                    if self.have_phot:
                        sed_fmod_w  = coeff_fmod_e[i_coeffs_0_of_comp_in_mod:i_coeffs_1_of_comp_in_mod] @ sed_fmod_ew[i_coeffs_0_of_comp_in_mod:i_coeffs_1_of_comp_in_mod]
                        output_MC[mod_name][comp_name]['sed_lw'][i_loop, :] = sed_fmod_w
                        output_MC[mod_name]['sum']['sed_lw'][i_loop, :] += sed_fmod_w
                        output_MC['tot']['fmod']['sed_lw'][i_loop, :] += sed_fmod_w
        # convert best-fit model SED to phot
        if self.have_phot:
            output_MC['tot']['fmod']['phot_lb'] = self.pframe.spec2phot(sed_wave_w, output_MC['tot']['fmod']['sed_lw'], phot_trans_bw)

        # save fitting residuals
        output_MC['tot']['fres'] = {}
        output_MC['tot']['fres']['spec_lw'] = output_MC['tot']['flux']['spec_lw'] - output_MC['tot']['fmod']['spec_lw']
        if self.have_phot:
            output_MC['tot']['fres']['phot_lb'] = output_MC['tot']['flux']['phot_lb'] - output_MC['tot']['fmod']['phot_lb']

        # save model spectra in flambda to calculate observed flux in later model.extract_results()
        self.output_MC = output_MC

        # convert to flux in mJy if required
        if flux_type in ['Fnu', 'fnu']:
            for mod_name in output_MC:
                for comp_name in output_MC[mod_name]:
                    if 'spec_lw' in output_MC[mod_name][comp_name]: 
                        output_MC[mod_name][comp_name]['spec_lw'] *= self.spec_flux_scale * PhotFrame.rFnuFlam_func(None,spec_wave_w) # None is for 'self' in PhotFrame definition
                    if 'sed_lw' in output_MC[mod_name][comp_name]: 
                        output_MC[mod_name][comp_name]['sed_lw']  *= self.spec_flux_scale * PhotFrame.rFnuFlam_func(None,sed_wave_w)
                    if 'phot_lb' in output_MC[mod_name][comp_name]: 
                        output_MC[mod_name][comp_name]['phot_lb'] *= self.spec_flux_scale * self.pframe.rFnuFlam_b
            self.spec['flux_w'] *= self.spec_flux_scale * PhotFrame.rFnuFlam_func(None,spec_wave_w)
            self.spec['ferr_w'] *= self.spec_flux_scale * PhotFrame.rFnuFlam_func(None,spec_wave_w)
            if self.have_phot:
                self.phot['flux_b'] *= self.spec_flux_scale * self.pframe.rFnuFlam_b
                self.phot['ferr_b'] *= self.spec_flux_scale * self.pframe.rFnuFlam_b
            self.if_input_modified = True

        # calculate average spectra
        for mod_name in rev_model_type.split('+'): 
            self.spec['fmod_'+mod_name+'_w'] = np.average(output_MC[mod_name]['sum']['spec_lw'], weights=1/best_chi_sq_l, axis=0)
        self.spec['fmod_tot_w'] = np.average(output_MC['tot']['fmod']['spec_lw'], weights=1/best_chi_sq_l, axis=0)
        self.spec['fres_w'] = self.spec['flux_w'] - self.spec['fmod_tot_w']
        if self.have_phot:
            for mod_name in rev_model_type.split('+'): 
                self.sed['fmod_'+mod_name+'_w'] = np.average(output_MC[mod_name]['sum']['sed_lw'], weights=1/best_chi_sq_l, axis=0)   
            self.sed['fmod_tot_w'] = np.average(output_MC['tot']['fmod']['sed_lw'], weights=1/best_chi_sq_l, axis=0)
            self.phot['fmod_b'] = self.pframe.spec2phot(sed_wave_w, self.sed['fmod_tot_w'], phot_trans_bw)
            self.phot['fres_b'] = self.phot['flux_b'] - self.phot['fmod_b']

        # save best-fit parameters and coefficients of each model, and calculate properties
        for mod_name in rev_model_type.split('+'):
            if self.have_phot:
                if not (self.model_dict_M[mod_name]['spec_enable'] | self.model_dict_M[mod_name]['sed_enable']): continue
            else:
                if not self.model_dict_M[mod_name]['spec_enable']: continue
            tmp_output_C = self.model_dict_M[mod_name]['spec_mod'].extract_results(step=step, if_print_results=if_print_results, if_return_results=True, 
                                                                                              if_rev_v0_redshift=if_rev_v0_redshift, if_show_average=if_show_average)
            comp_name_c = self.model_dict_M[mod_name]['cframe'].comp_name_c
            for (i_comp, comp_name) in enumerate(comp_name_c):
                output_MC[mod_name][comp_name]['par_lp']   = tmp_output_C[comp_name]['par_lp']
                output_MC[mod_name][comp_name]['coeff_le'] = tmp_output_C[comp_name]['coeff_le']
                output_MC[mod_name][comp_name]['value_Vl'] = tmp_output_C[comp_name]['value_Vl']
            output_MC[mod_name]['sum']['value_Vl'] = tmp_output_C['sum']['value_Vl']

        self.output_MC = output_MC

        ############################################################
        # allow old mod_name, 'ssp' and 'el', in output_MC to be compatible with old version <= 2.2.4
        if 'stellar' in self.full_model_type.split('+'): self.output_MC['ssp'] = output_MC['stellar']
        if 'line'    in self.full_model_type.split('+'): self.output_MC['el']  = output_MC['line']
        # keep aliases for output in old version <= 2.2.4
        for mod_name in rev_model_type.split('+'):
            comp_name_c = self.model_dict_M[mod_name]['cframe'].comp_name_c
            for (i_comp, comp_name) in enumerate(comp_name_c):
                output_MC[mod_name][comp_name]['values'] = output_MC[mod_name][comp_name]['value_Vl']
            output_MC[mod_name]['sum']['values'] = output_MC[mod_name]['sum']['value_Vl']
        self.output_mc = output_MC
        ############################################################

        if if_return_results: return output_MC

    ##########################################################################
    ############################ Plot functions ##############################

    def plot_step(self, flux_w, model_w, ferr_w, mask_w, fit_phot, fit_message, chi_sq, i_loop):
        if self.canvas is None:
            fig, axs = plt.subplots(2, 1, figsize=(9, 3), dpi=100, gridspec_kw={'height_ratios':[2,1]})
            plt.subplots_adjust(bottom=0.15, top=0.9, hspace=0, wspace=0)
        else:
            fig, axs = self.canvas
        ax0, ax1 = axs; ax0.clear(); ax1.clear()

        tmp_z = (1+self.v0_redshift)
        rest_wave_w = self.spec['wave_w']/tmp_z
        mask_spec_w = mask_w[:self.num_spec_wave]
        # ax0.plot(rest_wave_w, self.spec['flux_w'], c='C7', lw=0.3, alpha=0.75, label='Original spectrum')
        ax0.plot(self.spec_wave_w/tmp_z, self.spec_flux_w, c='C7', lw=0.3, alpha=0.75, label='Original spectrum')
        ax0.plot(rest_wave_w[mask_spec_w], flux_w[:self.num_spec_wave][mask_spec_w], c='C0', label='Data used for fitting (spec)')
        ax0.plot(rest_wave_w, model_w[:self.num_spec_wave], c='C1', label='Best-fit model (spec)')
        ax1.plot(rest_wave_w[mask_spec_w], (flux_w-model_w)[:self.num_spec_wave][mask_spec_w], c='C2', alpha=0.6, label='Residuals (spec)')
        ax1.fill_between(rest_wave_w, -3*ferr_w[:self.num_spec_wave], 3*ferr_w[:self.num_spec_wave], color='C5', alpha=0.2, label=r'$\pm 3\sigma$ error') # modified error
        ax1.fill_between(self.spec_wave_w/tmp_z, -3*self.spec_ferr_w, 3*self.spec_ferr_w, color='C5', alpha=0.1) # original error
        ax0.fill_between(rest_wave_w, -self.spec['flux_w'].max()*~mask_w[:self.num_spec_wave], 2*self.spec['flux_w'].max()*~mask_w[:self.num_spec_wave], 
                         hatch='////', fc='None', ec='C5', alpha=0.25)
        ax1.fill_between(rest_wave_w, -self.spec['flux_w'].max()*~mask_w[:self.num_spec_wave], 2*self.spec['flux_w'].max()*~mask_w[:self.num_spec_wave], 
                         hatch='////', fc='None', ec='C5', alpha=0.25)
        if not self.if_keep_invalid:
            ax0.fill_between(self.spec_wave_w/tmp_z, -self.spec['flux_w'].max()*~self.mask_valid_w[:len(self.spec_wave_w)], 2*self.spec['flux_w'].max()*~self.mask_valid_w[:len(self.spec_wave_w)], 
                             hatch='////', fc='None', ec='C5', alpha=0.25)
            ax1.fill_between(self.spec_wave_w/tmp_z, -self.spec['flux_w'].max()*~self.mask_valid_w[:len(self.spec_wave_w)], 2*self.spec['flux_w'].max()*~self.mask_valid_w[:len(self.spec_wave_w)], 
                             hatch='////', fc='None', ec='C5', alpha=0.25)
        ax0.set_xlim(rest_wave_w.min()-50, rest_wave_w.max()+100)
        ax1.set_xlim(rest_wave_w.min()-50, rest_wave_w.max()+100)
        if fit_phot:
            rest_wave_b = self.phot['wave_b']/tmp_z
            mask_phot_b = mask_w[-self.num_phot_band:]
            ind_o_b = np.argsort(rest_wave_b)
            ind_m_b = np.argsort(rest_wave_b[mask_phot_b])
            ax0.plot(rest_wave_b[ind_o_b], self.phot['flux_b'][ind_o_b], '--o', c='C7', label='Original phot-SED')
            ax0.plot(rest_wave_b[mask_phot_b][ind_m_b], flux_w[-self.num_phot_band:][mask_phot_b][ind_m_b],'--o',c='C9',label='Data used for fitting (phot)')
            ax0.plot(rest_wave_b[ind_o_b], model_w[-self.num_phot_band:][ind_o_b], '--s', c='C3', label='Best-fit model (phot)')
            # ax1.plot(rest_wave_b[ind_o_b],  ferr_w[-self.num_phot_band:][ind_o_b], '--o', c='C5')
            # ax1.plot(rest_wave_b[ind_o_b], -ferr_w[-self.num_phot_band:][ind_o_b], '--o', c='C5')
            ax1.errorbar(rest_wave_b, rest_wave_b*0, yerr=3*ferr_w[-self.num_phot_band:], fmt='.', markersize=0.1, linewidth=10, color='C5', alpha=0.4)
            ax1.plot(rest_wave_b[mask_phot_b][ind_m_b], (flux_w-model_w)[-self.num_phot_band:][mask_phot_b][ind_m_b], '--s', c='C6',label='Residuals (phot)')
            ax1.plot(rest_wave_b, rest_wave_b*0, '--', linewidth=1, color='C7', alpha=0.3, zorder=0)
            ax0.set_xscale('log'); ax1.set_xscale('log')
            ax0.set_xlim(np.hstack((rest_wave_w,rest_wave_b)).min()*0.9, np.hstack((rest_wave_w,rest_wave_b)).max()*1.1)
            ax1.set_xlim(np.hstack((rest_wave_w,rest_wave_b)).min()*0.9, np.hstack((rest_wave_w,rest_wave_b)).max()*1.1)

        ax0.legend(ncol=2); ax1.legend(ncol=3, loc='lower right')
        ax0.set_ylim(flux_w[mask_w].min()-0.05*(flux_w[mask_w].max()-flux_w[mask_w].min()), flux_w[mask_w].max()*1.05)
        # tmp_ylim = np.percentile(np.abs(flux_w-model_w)[mask_w], 90) * 1.5
        tmp_ylim = np.median(ferr_w) * 3 * 3 # x3 times of median 3-sigma 
        ax1.set_ylim(-tmp_ylim, tmp_ylim)

        ax0.set_xticks([]); ax1.set_xlabel(r'Rest wavelength ($\AA$)', labelpad=0)
        ax0.set_ylabel(f"Flux ({self.spec_flux_scale:.0e}"+r' erg/s/cm2/$\AA$)')
        ax1.set_ylabel('Res.')
        title = fit_message + r' ($\chi^2_{\nu}$ = ' + f"{chi_sq:.3f}, "
        title += f"loop {i_loop+1}/{self.num_loops}, " + ('original data)' if i_loop == 0 else 'mock data)')
        ax0.set_title(title)

        # attach SEFI cat
        if self.if_plot_icon:
            icon_dir = str(Path(__file__).parent)+'/auxiliaries/'
            icon_file = icon_dir + ('icon_s3fit.dat' if fit_phot else 'icon_s1fit.dat')
            icon = plt.imread(icon_file)
            zoom = min(0.02 * fig.get_figheight(), 0.1)
            abox = AnnotationBbox(OffsetImage(icon, zoom=zoom), (1, -1/fig.get_figheight()), xycoords='axes fraction', box_alignment=(0.5, 0.5), frameon=False)
            ax1.add_artist(abox)

        if self.canvas is not None:
            fig.canvas.draw(); fig.canvas.flush_events() # refresh plot in the given window
        plt.pause(0.0001)  # forces immediate update

    ##################################################

    def plot_results(self, step=None, if_plot_phot=False, if_plot_comp=True, 
                     plot_range=None, wave_type='rest', wave_unit='Angstrom', 
                     flux_type='Flam', res_type='residual', ferr_num=3, 
                     xyscale=('log','log'), figsize=(10, 6), dpi=300, 
                     title=None, legend_loc=None, if_plot_icon=False, 
                     output_plotname=None, **kwargs):
        # step: 'spec', 'pure-spec', 'spec+SED'
        # plot_range = ('spec', 'pure-spec'), 'SED'; check if 'SED'
        # flux_type: ('Flam'), 'Fnu'; check if 'Fnu'
        # wave_type: 'rest', ('obs', 'observed'); check if 'rest'
        # wave_unit: ('Angstrom'), 'um', 'micron'; check if 'um' or 'micron'
        # res_type: 'residual', 'residual/data', 'residual/model'
        # ferr_num: n-sigma error
        # xyscale: 'linear', 'log'

        if (step is None) | (step in ['best', 'final']): step = 'joint_fit_3' if self.have_phot else 'joint_fit_2'
        if  step in ['spec+SED', 'spectrum+SED']:  step = 'joint_fit_3'
        if  step in ['spec', 'pure-spec', 'spectrum', 'pure-spectrum']:  step = 'joint_fit_2'

        output_MC = self.extract_results(step=step, if_return_results=True, flux_type=flux_type, **kwargs) # if_print_results=False, if_rev_v0_redshift=None

        if step == 'joint_fit_2': 
            if_plot_phot = False # forcibly
            plot_range = 'spec' # forcibly

        if wave_type == 'rest':
            z_sys = self.rev_v0_redshift if self.rev_v0_redshift is not None else self.v0_redshift
            z_ratio_wave = (1+z_sys)
            z_ratio_flux = 1 # do not correct flux
        elif wave_type in ['obs', 'observed']:
            z_ratio_wave = 1
            z_ratio_flux = 1
        if wave_unit in ['um', 'micron']: z_ratio_wave *= 1e4

        wave_w = self.spec['wave_w']/z_ratio_wave
        flux_grid = 'spec_lw'
        if if_plot_phot: 
            wave_b = self.phot['wave_b']/z_ratio_wave
            if plot_range == 'SED':
                wave_w = self.sed['wave_w']/z_ratio_wave
                flux_grid = 'sed_lw'

        if res_type == 'residual': 
            ax1_ylabel = 'Residuals'
            fres_lw = output_MC['tot']['fres']['spec_lw']*z_ratio_flux
            ferr_w  = output_MC['tot']['ferr']['spec_lw'][0]*ferr_num*z_ratio_flux
            if if_plot_phot:
                fres_lb = output_MC['tot']['fres']['phot_lb']*z_ratio_flux
                ferr_b  = output_MC['tot']['ferr']['phot_lb'][0]*ferr_num*z_ratio_flux
        elif res_type == 'residual/data':
            ax1_ylabel = 'Residuals / Data'
            fres_lw = np.divide(output_MC['tot']['fres']['spec_lw'], output_MC['tot']['flux']['spec_lw'], where=output_MC['tot']['flux']['spec_lw']>0)
            ferr_w  = np.divide(output_MC['tot']['ferr']['spec_lw'][0]*ferr_num, output_MC['tot']['flux']['spec_lw'][0], where=output_MC['tot']['flux']['spec_lw'][0]>0)
            if if_plot_phot:
                fres_lb = np.divide(output_MC['tot']['fres']['phot_lb'], output_MC['tot']['flux']['phot_lb'], where=output_MC['tot']['flux']['phot_lb']>0)
                ferr_b  = np.divide(output_MC['tot']['ferr']['phot_lb'][0]*ferr_num, output_MC['tot']['flux']['phot_lb'][0], where=output_MC['tot']['flux']['phot_lb'][0]>0)
        elif res_type == 'residual/model':
            ax1_ylabel = 'Residuals / Model'
            fres_lw = output_MC['tot']['fres']['spec_lw'] / output_MC['tot']['fmod']['spec_lw']
            if if_plot_phot:
                fres_lb = output_MC['tot']['fres']['phot_lb'] / output_MC['tot']['fmod']['phot_lb']
                ferr_b  = output_MC['tot']['ferr']['phot_lb'][0]*ferr_num / output_MC['tot']['fmod']['phot_lb'][0]

        alpha_ratio = 10/self.num_loops

        #####################
        fig, axs = plt.subplots(2, 1, figsize=figsize, dpi=100, gridspec_kw={'height_ratios':[3,1]})
        plt.subplots_adjust(bottom=0.08, top=0.94, left=0.08, right=0.98, hspace=0, wspace=0)
        ax0, ax1 = axs
        #####################

        #####################
        # plot original data
        ax0.errorbar(self.spec['wave_w']/z_ratio_wave, self.spec['flux_w']*z_ratio_flux, 
                     linewidth=1.5, color='C7', alpha=1, label='Observed spec.', zorder=1)
        if if_plot_phot:
            ax0.errorbar(wave_b, self.phot['flux_b']*z_ratio_flux, output_MC['tot']['ferr']['phot_lb'][0]*ferr_num*z_ratio_flux, 
                         fmt='o', markersize=8, color='k', alpha=0.5, label='Observed phot.'+r' ($\pm 3\sigma$)', zorder=3)
        #####################

        #####################
        # plot total model spectra
        for i_loop in range(self.num_loops): 
            line, = ax0.plot(wave_w, output_MC['tot']['fmod'][flux_grid][i_loop]*z_ratio_flux, 
                             '-', linewidth=1.5, color='C1', alpha=min(1, 1*alpha_ratio), zorder=2)
            if i_loop == 0: line.set_label('Total models spec.')
        if if_plot_phot:
            for i_loop in range(self.num_loops):
                line = ax0.scatter(wave_b, output_MC['tot']['fmod']['phot_lb'][i_loop]*z_ratio_flux, 
                                   marker='o', s=200, linewidth=2, facecolor='None', edgecolor='C5', alpha=min(1, 0.5*alpha_ratio), zorder=3)
                if i_loop == 0: line.set_label('Total models phot.')
        # plot each model spectra
        for mod_name in self.rev_model_type.split('+'): 
            comp_name_c = copy([*output_MC[mod_name]])
            if if_plot_comp:
                # plot each comp
                if len(comp_name_c) > 2:
                    comp_name_c = comp_name_c[-1:]+comp_name_c[:-1] # move 'sum' to the begining
                else:
                    comp_name_c = comp_name_c[:-1] # hide 'sum' if only one comp
            else:
                comp_name_c = comp_name_c[-1:] # only plot 'sum'
            for comp_name in comp_name_c:
                for i_loop in range(self.num_loops): 
                    line, = ax0.plot(wave_w, output_MC[mod_name][comp_name][flux_grid][i_loop]*z_ratio_flux, 
                                     linestyle=self.model_dict_M[mod_name]['spec_mod'].plot_style_C[comp_name]['linestyle'], 
                                     linewidth=self.model_dict_M[mod_name]['spec_mod'].plot_style_C[comp_name]['linewidth'], 
                                     color=self.model_dict_M[mod_name]['spec_mod'].plot_style_C[comp_name]['color'], 
                                     alpha=min(1, self.model_dict_M[mod_name]['spec_mod'].plot_style_C[comp_name]['alpha']*alpha_ratio), zorder=3)
                    # if i_loop == 0: line.set_label(mod_name+':'+comp)
                # make obvious labels
                ax0.plot([0], [0], 
                         linestyle=self.model_dict_M[mod_name]['spec_mod'].plot_style_C[comp_name]['linestyle'], 
                         linewidth=self.model_dict_M[mod_name]['spec_mod'].plot_style_C[comp_name]['linewidth'], 
                         color=self.model_dict_M[mod_name]['spec_mod'].plot_style_C[comp_name]['color'], 
                         alpha=1, label=(mod_name+':'+comp_name) if if_plot_comp else mod_name)
        #####################

        #####################
        # plot residuals
        ax1.plot([0], [0], '-', linewidth=1, color='C2', alpha=1, label='Residuals spec.') # make obvious label
        for i_loop in range(self.num_loops): 
            line, = ax1.plot(self.spec['wave_w']/z_ratio_wave, fres_lw[i_loop], '-', linewidth=1, color='C2', alpha=min(1, 0.1*alpha_ratio), zorder=1)
            # if i_loop == 0: line.set_label('Residuals spec.')
            if if_plot_phot:
                line, = ax1.plot(wave_b, fres_lb[i_loop], 'o', color='C6', alpha=min(1, 0.25*alpha_ratio), zorder=3)
                # if i_loop == 0: line.set_label('Residuals phot.')
        if if_plot_phot:
            ind_o_b = np.argsort(wave_b)
            ax1.plot(wave_b[ind_o_b], fres_lb[0][ind_o_b], '--o', color='C6', alpha=0.75, label='Residuals phot.', zorder=3)
        ax1.plot(wave_w, wave_w*0, '--', color='C7', alpha=0.3, zorder=0)
        #####################

        #####################
        # plot flux errors
        if self.have_phot:
            if if_plot_phot:
                error_type = '(modified)'
            else:
                error_type = '(original)'
        else:
            error_type = ''
        ax0.fill_between(self.spec['wave_w']/z_ratio_wave, -output_MC['tot']['ferr']['spec_lw'][0]*ferr_num*z_ratio_flux, output_MC['tot']['ferr']['spec_lw'][0]*ferr_num*z_ratio_flux, 
                         facecolor='C5', edgecolor='C5', alpha=0.25, label=r'$\pm$'+f'{ferr_num}'+r'$\sigma$ error spec.'+error_type, zorder=0)
        ax1.fill_between(self.spec['wave_w']/z_ratio_wave, -ferr_w, ferr_w, 
                         facecolor='C5', edgecolor='C5', alpha=0.25, label=r'$\pm$'+f'{ferr_num}'+r'$\sigma$ error spec.'+error_type, zorder=0)
        if if_plot_phot:
            # ax0.errorbar(wave_b, wave_b*0, yerr=output_MC['tot']['ferr']['phot_lb'][0]*ferr_num*z_ratio_flux, 
            #              fmt='.', markersize=0.1, linewidth=10, color='C7', alpha=0.3, label=r'$\pm$'+f'{ferr_num}'+r'$\sigma$ error phot.(modified)', zorder=0)
            ax1.errorbar(wave_b, wave_b*0, yerr=ferr_b, 
                         fmt='.', markersize=0.1, linewidth=10, color='C7', alpha=0.3, label=r'$\pm$'+f'{ferr_num}'+r'$\sigma$ error phot.(modified)', zorder=2)
        #####################

        #####################
        # x- and -y ranges
        xmin = self.spec['wave_w'][self.spec['mask_valid_w']].min() * 0.98
        xmax = self.spec['wave_w'][self.spec['mask_valid_w']].max() * 1.01
        if if_plot_phot & (plot_range == 'SED'):
            xmin = min(xmin, self.phot['wave_b'].min() * 0.9)
            xmax = max(xmax, self.phot['wave_b'].max() * 1.1)
        for ax in axs: ax.set_xlim(xmin/z_ratio_wave, xmax/z_ratio_wave)

        ymax_flux = output_MC['tot']['fmod'][flux_grid][0].max() * 1.03
        if xyscale[1] == 'linear':
            ymin_flux = -0.03 * ymax_flux
            if 'line' in output_MC:
                ymin_flux = min(ymin_flux, output_MC['line']['sum'][flux_grid][0].min() * 1.02) # for absorption lines
        elif xyscale[1] == 'log':
            ymin_flux = np.median(output_MC['tot']['ferr']['spec_lw'][0])
        ax0.set_ylim(ymin_flux*z_ratio_flux, ymax_flux*z_ratio_flux)

        ymax_fres = np.median(ferr_w) * 3
        if if_plot_phot:
            ymax_fres = max(ymax_fres, np.median(ferr_b) * 3)
        ax1.set_ylim(-ymax_fres, ymax_fres) # ferr_w/b is already *z_ratio_flux
        for ax in axs:
            ax.fill_between(self.spec['wave_w']/z_ratio_wave, -ymax_fres*z_ratio_flux*~self.spec['mask_valid_w'], ymax_flux*z_ratio_flux*~self.spec['mask_valid_w'], 
                            hatch='////', facecolor='None', edgecolor='C5', alpha=0.5, zorder=0) 
            ax.set_xscale(xyscale[0])
        ax0.set_yscale(xyscale[1])
        #####################

        if legend_loc is None:
            ax0.legend(ncol=4); ax1.legend(ncol=4, loc=4)
        else:
            ax0.legend(ncol=4, loc=legend_loc[0]); ax1.legend(ncol=4, loc=legend_loc[1])
        ax0.tick_params(axis='x', which='both', labelbottom=False)
        # ax0.set_title('The best-fit spectra in the wavelength range of the data spectrum.')
        if flux_type in ['Fnu', 'fnu']:
            ax0.set_ylabel(f"Flux (mJy)")
        else:
            ax0.set_ylabel(f"Flux ({self.spec_flux_scale:.0e}"+r' erg s$^{-1}$cm$^{-2}\AA^{-1}$)')
        ax1.set_ylabel(ax1_ylabel)
        ax1.set_xlabel(('Rest' if wave_type == 'rest' else 'Observed')+' wavelength '+(r'($\mu$m)' if wave_unit in ['um', 'micron'] else r'($\AA$)')) # , labelpad=0

        if title is not None: ax0.set_title(title)

        # attach SEFI cat
        if if_plot_icon:
            icon_dir = str(Path(__file__).parent)+'/auxiliaries/'
            icon_file = icon_dir + ('icon_s3fit.dat' if step == 'joint_fit_3' else 'icon_s1fit.dat')
            icon = plt.imread(icon_file)
            zoom = 0.01 * fig.get_figheight()
            abox = AnnotationBbox(OffsetImage(icon, zoom=zoom), (0.99, 0.01), xycoords='figure fraction', box_alignment=(1, 0), frameon=False)
            ax1.add_artist(abox)

        if output_plotname is not None:
            if output_plotname.split('.')[-1] in ['pdf', 'PDF']:
                plt.savefig('./tmp.svg', dpi=dpi, transparent=False)
                _ = os.system('svg42pdf ./tmp.svg ' + output_plotname)
                _ = os.system('rm ./tmp.svg')
            else:
                plt.savefig(output_plotname, dpi=dpi, transparent=False) # dpi='figure'

