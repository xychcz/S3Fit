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

from ..auxiliaries.auxiliary_frames import ConfigFrame
from ..auxiliaries.auxiliary_functions import print_log, casefold, color_list_dict, convolve_var_width_fft
# from ..auxiliaries.basic_model_functions import powerlaw_func, blackbody_func, recombination_func
from ..auxiliaries.extinct_laws import ExtLaw

class AGNFrame(object):
    def __init__(self, mod_name=None, fframe=None, 
                 config=None, file_path=None, 
                 v0_redshift=None, R_inst_rw=None, 
                 w_min=None, w_max=None, w_norm=5100, dw_norm=25, 
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

        # check alternative model names
        for i_comp in range(self.num_comps):
            if casefold(self.cframe.info_c[i_comp]['mod_used']) in ['powerlaw', 'pl']: 
                self.cframe.info_c[i_comp]['mod_used'] = 'powerlaw'
            if casefold(self.cframe.info_c[i_comp]['mod_used']) in ['bending-powerlaw', 'bending_powerlaw', 'bending powerlaw', 'bending-pl', 'bending_pl', 'bending pl']: 
                self.cframe.info_c[i_comp]['mod_used'] = 'bending-powerlaw'
            if casefold(self.cframe.info_c[i_comp]['mod_used']) in ['blackbody', 'black_body', 'black body', 'bb']: 
                self.cframe.info_c[i_comp]['mod_used'] = 'blackbody'
            if casefold(self.cframe.info_c[i_comp]['mod_used']) in ['recombination', 'recombination_continuum', 'recombination continuum', 'rec', 'balmer_continuum', 'balmer continuum', 'bac']: 
                self.cframe.info_c[i_comp]['mod_used'] = 'recombination'
            if casefold(self.cframe.info_c[i_comp]['mod_used']) in ['iron', 'feii', 'fe ii']: 
                self.cframe.info_c[i_comp]['mod_used'] = 'iron'

        ############################################################
        # to be compatible with old version <= 2.2.4
        if len(self.cframe.par_index_cP[0]) == 0:
            for i_comp in range(self.num_comps):
                if self.cframe.info_c[i_comp]['mod_used'] == 'powerlaw':
                    self.cframe.par_name_cp[i_comp]  = ['voff', 'fwhm', 'Av', 'alpha_lambda']
                    self.cframe.par_index_cP[i_comp] = {'voff': 0, 'fwhm': 1, 'Av': 2, 'alpha_lambda': 3}
                if self.cframe.info_c[i_comp]['mod_used'] == 'bending-powerlaw':
                    self.cframe.par_name_cp[i_comp]  = ['voff', 'fwhm', 'Av', 'alpha_lambda1', 'alpha_lambda2', 'wave_turn', 'curvature']
                    self.cframe.par_index_cP[i_comp] = {'voff': 0, 'fwhm': 1, 'Av': 2, 'alpha_lambda1': 3, 'alpha_lambda2': 4, 'wave_turn': 5, 'curvature': 6}
                if self.cframe.info_c[i_comp]['mod_used'] == 'blackbody':
                    self.cframe.par_name_cp[i_comp]  = ['voff', 'fwhm', 'Av', 'log_tem',]
                    self.cframe.par_index_cP[i_comp] = {'voff': 0, 'fwhm': 1, 'Av': 2, 'log_tem': 3}
                if self.cframe.info_c[i_comp]['mod_used'] == 'recombination':
                    self.cframe.par_name_cp[i_comp]  = ['voff', 'fwhm', 'Av', 'log_e_tem', 'log_tau_be']
                    self.cframe.par_index_cP[i_comp] = {'voff': 0, 'fwhm': 1, 'Av': 2, 'log_e_tem': 3, 'log_tau_be': 4}
                if self.cframe.info_c[i_comp]['mod_used'] == 'iron':
                    self.cframe.par_name_cp[i_comp]  = ['voff', 'fwhm', 'Av']
                    self.cframe.par_index_cP[i_comp] = {'voff': 0, 'fwhm': 1, 'Av': 2}
        ############################################################

        # set default info if not specified in config
        for i_comp in range(self.num_comps):
            if 'int_wave_range' not in self.cframe.info_c[i_comp]: self.cframe.info_c[i_comp]['int_wave_range'] = [(912,30000)]
            if 'int_wave_unit'  not in self.cframe.info_c[i_comp]: self.cframe.info_c[i_comp]['int_wave_unit']  = 'angstrom'
            if 'int_wave_frame' not in self.cframe.info_c[i_comp]: self.cframe.info_c[i_comp]['int_wave_frame'] = 'rest'
            if 'int_lum_unit'   not in self.cframe.info_c[i_comp]: self.cframe.info_c[i_comp]['int_lum_unit']   = 'erg/s'
            if 'int_lum_type'   not in self.cframe.info_c[i_comp]: self.cframe.info_c[i_comp]['int_lum_type']   = ['intrinsic', 'observed', 'absorbed']
            if self.cframe.info_c[i_comp]['mod_used'] in ['powerlaw', 'bending-powerlaw']:
                # truncated range and index of powerlaw, either nested tuple or dict format, e.g., {'wave_range': (5, None), 'wave_unit': 'micron', 'alpha_lambda': -4}
                if 'truncation' not in self.cframe.info_c[i_comp]: self.cframe.info_c[i_comp]['truncation'] = [((None, 0.01), 'micron', 0.2), ((0.01,  0.1), 'micron',  -1), ((5, None), 'micron',  -4)]
                # the adopted values follow Schartmann et al. 2005; Feltre et al. 2012; Stalevski et al. 2016
            if self.cframe.info_c[i_comp]['mod_used'] == 'recombination':
                if 'H_series' not in self.cframe.info_c[i_comp]: self.cframe.info_c[i_comp]['H_series'] = [2,3,4,5]
            if self.cframe.info_c[i_comp]['mod_used'] == 'iron':
                if 'segments' not in self.cframe.info_c[i_comp]: self.cframe.info_c[i_comp]['segments'] = False

        # group single info to a list
        for i_comp in range(self.num_comps):
            if isinstance(self.cframe.info_c[i_comp]['int_wave_range'], tuple): self.cframe.info_c[i_comp]['int_wave_range'] = [self.cframe.info_c[i_comp]['int_wave_range']]
            if isinstance(self.cframe.info_c[i_comp]['int_wave_range'], list):
                if all( isinstance(i, (int,float)) for i in self.cframe.info_c[i_comp]['int_wave_range'] ):
                    if len(self.cframe.info_c[i_comp]['int_wave_range']) == 2: self.cframe.info_c[i_comp]['int_wave_range'] = [self.cframe.info_c[i_comp]['int_wave_range']]
            if isinstance(self.cframe.info_c[i_comp]['int_lum_type'], str): self.cframe.info_c[i_comp]['int_lum_type'] = [self.cframe.info_c[i_comp]['int_lum_type']] 
            if self.cframe.info_c[i_comp]['mod_used'] in ['powerlaw', 'bending-powerlaw']:
                if isinstance(self.cframe.info_c[i_comp]['truncation'], (tuple, dict)): self.cframe.info_c[i_comp]['truncation'] = [self.cframe.info_c[i_comp]['truncation']]
            if self.cframe.info_c[i_comp]['mod_used'] == 'recombination':
                if isinstance(self.cframe.info_c[i_comp]['H_series'], (str, int)): self.cframe.info_c[i_comp]['H_series'] = [self.cframe.info_c[i_comp]['H_series']]

        # check alternative info
        for i_comp in range(self.num_comps):
            # int_lum_type
            self.cframe.info_c[i_comp]['int_lum_type'] = ['intrinsic' if casefold(lum_type) in ['intrinsic', 'original'] else lum_type for lum_type in self.cframe.info_c[i_comp]['int_lum_type']]
            self.cframe.info_c[i_comp]['int_lum_type'] = ['observed'  if casefold(lum_type) in ['observed', 'reddened', 'attenuated', 'extincted', 'extinct'] else lum_type 
                                                          for lum_type in self.cframe.info_c[i_comp]['int_lum_type']]
            self.cframe.info_c[i_comp]['int_lum_type'] = ['absorbed'  if casefold(lum_type) in ['absorbed', 'dust'] else lum_type for lum_type in self.cframe.info_c[i_comp]['int_lum_type']]
            self.cframe.info_c[i_comp]['int_lum_type'] = list(dict.fromkeys(self.cframe.info_c[i_comp]['int_lum_type'])) # remove duplicates
            if self.cframe.info_c[i_comp]['mod_used'] == 'recombination':
                # translate greek notations to number of lower-level number
                H_series_dict = {'balmer': 2, 'paschen': 3, 'brackett': 4, 'pfund': 5, 'humphreys': 6}
                self.cframe.info_c[i_comp]['H_series'] = [H_series_dict[casefold(series)] if casefold(series) in H_series_dict else series for series in self.cframe.info_c[i_comp]['H_series']]
                self.cframe.info_c[i_comp]['H_series'] = list(dict.fromkeys( self.cframe.info_c[i_comp]['H_series'] )) # remove duplicate

        # set original wavelength grid, required to project iron template
        orig_wave_logbin = 0.05
        orig_wave_num = int(np.round(np.log10(w_max/w_min) / orig_wave_logbin))
        self.orig_wave_w = np.logspace(np.log10(w_min), np.log10(w_max), num=orig_wave_num)
        # load iron template
        if 'iron' in [self.cframe.info_c[i_comp]['mod_used'] for i_comp in range(self.num_comps)]: 
            self.read_iron()

        # count the number of independent model elements
        self.num_coeffs_c = np.zeros(self.num_comps, dtype='int')
        for i_comp in range(self.num_comps):
            if self.cframe.info_c[i_comp]['mod_used'] in ['powerlaw', 'bending-powerlaw', 'blackbody', 'recombination']:
                self.num_coeffs_c[i_comp] = 1 # one independent element per component
            if self.cframe.info_c[i_comp]['mod_used'] == 'iron':
                if self.cframe.info_c[i_comp]['segments']: 
                    self.num_coeffs_c[i_comp] = self.iron_dict['orig_flux_ew'].shape[0] # multiple independent segments
                else:
                    self.num_coeffs_c[i_comp] = 1 # one independent segment
        self.num_coeffs = self.num_coeffs_c.sum()

        # currently do not consider negative spectra 
        self.mask_absorption_e = np.zeros((self.num_coeffs), dtype='bool')
        
        # set plot styles
        self.plot_style_C = {}
        self.plot_style_C['sum'] = {'color': 'C3', 'alpha': 0.75, 'linestyle': '-', 'linewidth': 1.5}
        i_red, i_yellow, i_green, i_purple = 0, 0, 0, 0
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            if self.cframe.info_c[i_comp]['mod_used'] in ['powerlaw', 'bending-powerlaw']:
                self.plot_style_C[comp_name] = {'color': 'None', 'alpha': 0.5, 'linestyle': '--', 'linewidth': 1}
                self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['purple'], i_purple, mode="wrap"))
                i_purple += 1
            if self.cframe.info_c[i_comp]['mod_used'] == 'blackbody':
                self.plot_style_C[comp_name] = {'color': 'None', 'alpha': 0.5, 'linestyle': '--', 'linewidth': 1}
                self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['red'], i_red, mode="wrap"))
                i_red += 1
            if self.cframe.info_c[i_comp]['mod_used'] == 'recombination':
                self.plot_style_C[comp_name] = {'color': 'None', 'alpha': 0.5, 'linestyle': '--', 'linewidth': 1}
                self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['green'], i_green, mode="wrap"))
                i_green += 1
            if self.cframe.info_c[i_comp]['mod_used'] == 'iron':
                self.plot_style_C[comp_name] = {'color': 'None', 'alpha': 0.5, 'linestyle': '-', 'linewidth': 0.75}
                self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['yellow'], i_yellow, mode="wrap"))
                i_yellow += 1

        if self.verbose:
            print_log(f"AGN UV/optical/NIR continuum components: {[self.cframe.info_c[i_comp]['mod_used'] for i_comp in range(self.num_comps)]}", self.log_message)
            for i_comp in range(self.num_comps):
                if self.cframe.info_c[i_comp]['mod_used'] == 'recombination':
                    print_log(f"Lower-level principal quantum number of Hydrogen recombination continuum: {self.cframe.info_c[i_comp]['H_series']}", self.log_message)
                if self.cframe.info_c[i_comp]['mod_used'] == 'iron':
                    if self.cframe.info_c[i_comp]['segments']: 
                        wave_ranges = [wave_range for wave_range in self.iron_dict['wave_segments'] if (wave_range[1] > self.w_min) & (wave_range[0] < self.w_max)]
                        print_log(f"Wavelength segments for fitting of Fe II template: {wave_ranges}", self.log_message)

    ##########################################################################

    def simple_powerlaw(self, wavelength, wave_norm=None, flux_norm=1.0, alpha_lambda=None):
        pl = flux_norm * (wavelength/wave_norm)**alpha_lambda
        return pl

    def bending_powerlaw(self, wavelength, wave_turn=None, flux_trun=1.0, alpha_lambda1=None, alpha_lambda2=None, curvature=None, bending=False):
        # alpha_lambda1, alpha_lambda2: index with wavelength <= wave_turn and wavelength > wave_turn
        # curvature <= 0: broken two-side powerlaw
        # curvature > 0: smoothed bending powerlaw. larger curvature --> smoother break (5: very smooth; 0.1: very sharp)

        if isinstance(wavelength, (int,float)): wavelength = np.array([wavelength])
        if curvature is None: curvature = 0
        if alpha_lambda2 is not None:
            if alpha_lambda1 > alpha_lambda2: curvature = 0 # smoothing does not work in this case

        pl = self.simple_powerlaw(wavelength, wave_turn, flux_trun, alpha_lambda1)

        if bending:
            if curvature <= 0:
                # sharp, continuous broken power law
                mask_w = wavelength > wave_turn
                pl[mask_w] = self.simple_powerlaw(wavelength[mask_w], wave_turn, flux_trun, alpha_lambda2)
            else:
                pl_2 = self.simple_powerlaw(wavelength, wave_turn, 1, (alpha_lambda2-alpha_lambda1)/curvature)
                pl *= ((1+pl_2)/2.0)**curvature

        return pl

    def powerlaw_func(self, wavelength, wave_norm=None, alpha_lambda1=None, alpha_lambda2=None, curvature=None, bending=False, truncation=None):
        # normalized to given flux density (e.g.,the same unit of obs) at rest wave_norm before extinct

        if isinstance(wavelength, (int,float)): wavelength = np.array([wavelength])
        pl_w = self.bending_powerlaw(wavelength, wave_norm, 1, alpha_lambda1, alpha_lambda2, curvature, bending)

        # set cutting index in longer and shorter wavelength ranges
        if truncation is not None:
            slices = []
            for slice in truncation:
                if isinstance(slice, tuple): 
                    wave_range, wave_unit, alpha_truncated = slice
                elif isinstance(slice, dict):
                    wave_range, wave_unit, alpha_truncated = slice['wave_range'], slice['wave_unit'], slice['alpha_lambda']
                wave_ratio = 1e4 if casefold(wave_unit) in ['micron', 'um'] else 1
                wave_left  = wave_range[0]*wave_ratio if wave_range[0] is not None else min(wave_range[1]*wave_ratio*0.9, min(wavelength))
                wave_right = wave_range[1]*wave_ratio if wave_range[1] is not None else max(wave_range[0]*wave_ratio*1.1, max(wavelength))
                slices.append([wave_left, wave_right, alpha_truncated])
            slice_sv = np.array(slices)

            wave_left_0 = 3000
            if sum(slice_sv[:,1] < wave_left_0) > 0:
                slice_short_sv = slice_sv[slice_sv[:,1] < wave_left_0, :]
                slice_short_sv = slice_short_sv[np.argsort(slice_short_sv[:,1])[::-1], :]
                pl_right = self.bending_powerlaw(slice_short_sv[0,1], wave_norm, 1, alpha_lambda1, alpha_lambda2, curvature, bending)
                for i_slice in range(slice_short_sv.shape[0]):
                    wave_left, wave_right, alpha_truncated = slice_short_sv[i_slice]
                    mask_w = (wavelength >= wave_left) & (wavelength <= wave_right)
                    if mask_w.sum() == 0: break
                    pl_w[mask_w] = self.simple_powerlaw(wavelength[mask_w], wave_right, pl_right, alpha_truncated)
                    pl_right = self.simple_powerlaw(wave_left, wave_right, pl_right, alpha_truncated)

            wave_right_0 = 5100
            if sum(slice_sv[:,0] > wave_right_0) > 0:
                slice_long_sv = slice_sv[slice_sv[:,0] > wave_right_0, :]
                slice_long_sv = slice_long_sv[np.argsort(slice_long_sv[:,0]), :]
                pl_left = self.bending_powerlaw(slice_long_sv[0,0], wave_norm, 1, alpha_lambda1, alpha_lambda2, curvature, bending)
                for i_slice in range(slice_long_sv.shape[0]):
                    wave_left, wave_right, alpha_truncated = slice_long_sv[i_slice]
                    mask_w = (wavelength >= wave_left) & (wavelength <= wave_right)
                    if mask_w.sum() == 0: break
                    pl_w[mask_w] = self.simple_powerlaw(wavelength[mask_w], wave_left, pl_left, alpha_truncated)
                    pl_left = self.simple_powerlaw(wave_right, wave_left, pl_left, alpha_truncated)

        return pl_w

    def blackbody_func(self, wavelength, log_tem=None, if_norm=True, wave_norm=None):
        # parameters: temperature (K)

        def get_bb(wavelength):
            # Planck function for the given temperature
            C1 = 1.1910429723971884e27 # 2 * const.h.value * const.c.value**2 * 1e40 * 1e3
            C2 = 1.4387768775039336e8  # const.h.value * const.c.value / const.k_B.value * 1e10
            tmp = C2 / (wavelength * 10.0**log_tem)
            tmp = np.minimum(tmp, 700) # avoid overflow warning in np.exp()
            return C1 / wavelength**5 / (np.exp(tmp) - 1) # in erg/s/cm2/A/sr
        
        ret_bb_w = get_bb(wavelength) 
        if if_norm: ret_bb_w /= get_bb(wave_norm)

        return ret_bb_w

    def recombination_func(self, wavelength, log_e_tem=None, log_tau_be=None, H_series=None, wave_norm=3000):
        # Hydrogen Radiative Recombination Continuum (free-bound)
        # parameters: electron temperature (K), optical depth at balmer edge (3646)
        # temperature range: ~3000--30000 K. lower: neutral H dominated; higher: free-free dominated

        def get_rec(wavelength):
            if isinstance(wavelength, (int,float)): wavelength = np.array([wavelength])
            rec_w = np.zeros_like(wavelength, dtype=float)
            bb_w = self.blackbody_func(wavelength, log_tem=log_e_tem, if_norm=False)
            for lv_n in H_series:
                wave_edge = lv_n**2 / 1.0973731568160e-3 # n**2 / const.Ryd.value * 1e10, in A
                if min(wavelength) > wave_edge: continue
                # assume the bound-free cross section at threshold scales prop to n**(-5)
                tau_edge = 10.0**log_tau_be * (2.0/lv_n)**5
                # calculate the optical depth at each wavelength, τ_λ = τ_BE * (λ_BE / λ)^3 (Grandi 1982)
                tau_w = tau_edge * (wave_edge / wavelength)**3
                tmp_rec_w = bb_w * (1 - np.exp(-tau_w))
                tmp_rec_w[wavelength > wave_edge] = 0.0
                rec_w += tmp_rec_w
            return rec_w

        # normalize at rest 3000 A (default)
        ret_rec_w = get_rec(wavelength) / get_rec(wave_norm)

        return ret_rec_w

    def read_iron(self):
        # combined I Zw 1 Fe II (+ UV Fe III) template, convolving to fwhm = 1100 km/s
        # https://arxiv.org/pdf/astro-ph/0104320, Vestergaard andWilkes, 2001
        # https://arxiv.org/pdf/astro-ph/0312654, Veron-Cetty et al., 2003
        # https://arxiv.org/pdf/astro-ph/0606040, Tsuzuki et al., 2006

        iron_lib = fits.open(self.file_path)
        iron_wave_w = iron_lib[0].data[0] # angstrom, in rest frame
        iron_flux_w = iron_lib[0].data[1] # erg/s/cm2/angstrom
        iron_dw_fwhm_w = iron_wave_w * 1100 / 299792.458
        self.iron_dict = {}
        self.iron_dict['init_wave_w'] = copy(iron_wave_w)

        # normalize models at given wavelength
        mask_norm_w = (iron_wave_w > 2150) & (iron_wave_w < 4000)
        flux_norm_uv = np.trapezoid(iron_flux_w[mask_norm_w], x=iron_wave_w[mask_norm_w])
        mask_norm_w = (iron_wave_w > 4000) & (iron_wave_w < 5600)
        flux_norm_opt = np.trapezoid(iron_flux_w[mask_norm_w], x=iron_wave_w[mask_norm_w])
        iron_flux_w /= flux_norm_uv
        self.iron_dict['init_flux_w'] = copy(iron_flux_w)
        self.iron_dict['flux_opt_uv_ratio'] = flux_norm_opt / flux_norm_uv

        # determine the required model resolution and bin size (in angstrom) to downsample the model
        if self.Rratio_mod is not None:
            ds_R_mod_w = np.interp(iron_wave_w*(1+self.v0_redshift), self.R_inst_rw[0], self.R_inst_rw[1] * self.Rratio_mod) # R_inst_rw in observed frame
            self.dw_fwhm_dsp_w = iron_wave_w / ds_R_mod_w # required resolving width in rest frame
        else:
            if self.dw_fwhm_dsp is not None:
                self.dw_fwhm_dsp_w = np.full(len(iron_wave_w), self.dw_fwhm_dsp)
            else:
                self.dw_fwhm_dsp_w = None
        if self.dw_fwhm_dsp_w is not None:
            if (self.dw_fwhm_dsp_w > iron_dw_fwhm_w).all(): 
                pre_convolving = True
            else:
                pre_convolving = False
                self.dw_fwhm_dsp_w = iron_dw_fwhm_w
            self.dw_dsp = self.dw_fwhm_dsp_w.min() * 0.5 # required min bin wavelength following Nyquist–Shannon sampling
            if self.dw_pix_inst is not None:
                self.dw_dsp = min(self.dw_dsp, self.dw_pix_inst/(1+self.v0_redshift) * 0.5) # also require model bin wavelength <= 0.5 of data bin width (convert to rest frame)
            self.dpix_dsp = int(self.dw_dsp / np.median(np.diff(iron_wave_w))) # required min bin number of pixels
            self.dw_dsp = self.dpix_dsp * np.median(np.diff(iron_wave_w)) # update value
            if self.dpix_dsp > 1:
                if pre_convolving:
                    if self.verbose: 
                        print_log(f'Downsample pre-convolved AGN Fe II pesudo-continuum with bin width of {self.dw_dsp:.3f} Å in a min resolution of {self.dw_fwhm_dsp_w.min():.3f} Å', 
                                  self.log_message)
                    # before downsampling, smooth the model to avoid aliasing (like in ADC or digital signal reduction)
                    # set dw_fwhm_ref as the dispersion in the original model
                    iron_flux_w = convolve_var_width_fft(iron_wave_w, iron_flux_w, dw_fwhm_obj=self.dw_fwhm_dsp_w, dw_fwhm_ref=iron_dw_fwhm_w, num_bins=10, reset_edge=True)
                else:
                    if self.verbose: 
                        print_log(f'Downsample original AGN Fe II pesudo-continuum with bin width of {self.dw_dsp:.3f} Å in a min resolution of {self.dw_fwhm_dsp_w.min():.3f} Å', 
                                  self.log_message)  
                iron_wave_w = iron_wave_w[::self.dpix_dsp]
                iron_flux_w = iron_flux_w[::self.dpix_dsp]
                self.dw_fwhm_dsp_w = self.dw_fwhm_dsp_w[::self.dpix_dsp]

        # select model spectra in given wavelength range
        mask_select_w = (iron_wave_w >= self.w_min) & (iron_wave_w <= self.w_max)
        iron_wave_w = iron_wave_w[mask_select_w]
        iron_flux_w = iron_flux_w[mask_select_w]
        self.dw_fwhm_dsp_w = self.dw_fwhm_dsp_w[mask_select_w]

        self.orig_wave_w = np.hstack((self.orig_wave_w, iron_wave_w))
        self.orig_wave_w = np.sort(self.orig_wave_w)
        self.iron_dict['orig_wave_w'] = self.orig_wave_w
        self.iron_dict['orig_flux_w'] = np.interp(self.orig_wave_w, iron_wave_w, iron_flux_w, left=0, right=0)
        self.dw_fwhm_dsp_w = np.interp(self.orig_wave_w, iron_wave_w, self.dw_fwhm_dsp_w)
        self.iron_dict['dw_fwhm_dsp_w'] = self.dw_fwhm_dsp_w

        # set segments
        self.iron_dict['wave_segments'] = [[1000,2150], [2150,2650], [2650,3020], [3020,4000], [4000,4800], [4800,5600], [5600,6800], [6800,7600]]
        self.iron_dict['init_flux_ew'] = []
        self.iron_dict['orig_flux_ew'] = []
        self.iron_dict['flux_norm_e']  = []
        for wave_range in self.iron_dict['wave_segments']:
            tmp_wave_w = self.iron_dict['init_wave_w']
            tmp_flux_w = np.zeros_like(tmp_wave_w)
            mask_w = (tmp_wave_w >= wave_range[0]) & (tmp_wave_w < wave_range[1])
            tmp_flux_w[mask_w] = copy(self.iron_dict['init_flux_w'][mask_w])
            self.iron_dict['init_flux_ew'].append(tmp_flux_w)
            self.iron_dict['flux_norm_e'].append(np.trapezoid(tmp_flux_w, x=tmp_wave_w))

            tmp_wave_w = self.iron_dict['orig_wave_w']
            tmp_flux_w = np.zeros_like(tmp_wave_w)
            mask_w = (tmp_wave_w >= wave_range[0]) & (tmp_wave_w < wave_range[1])
            tmp_flux_w[mask_w] = copy(self.iron_dict['orig_flux_w'][mask_w])
            self.iron_dict['orig_flux_ew'].append(tmp_flux_w)

        self.iron_dict['init_flux_ew'] = np.array(self.iron_dict['init_flux_ew'])
        self.iron_dict['orig_flux_ew'] = np.array(self.iron_dict['orig_flux_ew'])
        self.iron_dict['flux_norm_e'] = np.array(self.iron_dict['flux_norm_e'])

    ##########################################################################

    def create_models(self, obs_wave_w, par_p, mask_lite_e=None, components=None, 
                      if_dust_ext=True, if_ism_abs=False, if_igm_abs=False, 
                      if_redshift=True, if_convolve=True, conv_nbin=None, if_full_range=False, dpix_resample=None): 

        par_cp = self.cframe.reshape_by_comp(par_p, self.cframe.num_pars_c)
        if isinstance(components, str): components = [components]

        obs_flux_mcomp_ew = None
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            if components is not None:
                if comp_name not in components: continue

            orig_wave_w = copy(self.orig_wave_w)
            # read and append intrinsic templates in rest frame
            if self.cframe.info_c[i_comp]['mod_used'] == 'powerlaw':
                alpha_lambda = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['alpha_lambda']]
                pl = self.powerlaw_func(orig_wave_w, wave_norm=self.w_norm, alpha_lambda1=alpha_lambda, alpha_lambda2=None, curvature=None, bending=False)
                orig_flux_int_ew = pl[None,:] # convert to (1,w) format
            if self.cframe.info_c[i_comp]['mod_used'] == 'bending-powerlaw':
                alpha_lambda1 = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['alpha_lambda1']]
                alpha_lambda2 = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['alpha_lambda2']]
                wave_turn     = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['wave_turn']]
                curvature     = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['curvature']]
                pl = self.powerlaw_func(orig_wave_w, wave_norm=wave_turn, alpha_lambda1=alpha_lambda1, alpha_lambda2=alpha_lambda2, curvature=curvature, bending=True)
                orig_flux_int_ew = pl[None,:] # convert to (1,w) format
            if self.cframe.info_c[i_comp]['mod_used'] == 'blackbody':
                log_tem  = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['log_tem']]
                bb = self.blackbody_func(orig_wave_w, log_tem=log_tem, wave_norm=self.w_norm)
                orig_flux_int_ew = bb[None,:] # convert to (1,w) format
            if self.cframe.info_c[i_comp]['mod_used'] =='recombination':
                log_e_tem  = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['log_e_tem']]
                log_tau_be = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['log_tau_be']]
                rec = self.recombination_func(orig_wave_w, log_e_tem=log_e_tem, log_tau_be=log_tau_be, H_series=self.cframe.info_c[i_comp]['H_series'])
                orig_flux_int_ew = rec[None,:] # convert to (1,w) format
            if self.cframe.info_c[i_comp]['mod_used'] == 'iron':
                if if_full_range:
                    orig_wave_w = copy(self.iron_dict['init_wave_w'])
                    # do not convolve in this case
                    if_convolve = False 
                    if self.cframe.info_c[i_comp]['segments']:
                        iron = self.iron_dict['init_flux_ew']
                        orig_flux_int_ew = copy(iron)
                    else:
                        iron = self.iron_dict['init_flux_w']
                        orig_flux_int_ew = iron[None,:] # convert to (1,w) format
                else:
                    if self.cframe.info_c[i_comp]['segments']:
                        iron = self.iron_dict['orig_flux_ew']
                        orig_flux_int_ew = copy(iron)
                    else:
                        iron = self.iron_dict['orig_flux_w']
                        orig_flux_int_ew = iron[None,:] # convert to (1,w) format

            # dust extinction
            if if_dust_ext & ('Av' in self.cframe.par_index_cP[i_comp]):
                Av = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['Av']]
                orig_flux_d_ew = orig_flux_int_ew * 10.0**(-0.4 * Av * ExtLaw(orig_wave_w))
            else:
                orig_flux_d_ew = orig_flux_int_ew

            # redshift models
            voff = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['voff']]
            z_ratio = (1 + self.v0_redshift) * (1 + voff/299792.458) # (1+z) = (1+zv0) * (1+v/c)
            orig_wave_z_w = orig_wave_w * z_ratio
            if if_redshift:
                orig_flux_dz_ew = orig_flux_d_ew / z_ratio
            else:
                orig_flux_dz_ew = orig_flux_d_ew

            # convolve with intrinsic and instrumental dispersion if self.R_inst_rw is not None; only for iron
            if if_convolve & ('fwhm' in self.cframe.par_index_cP[i_comp]) & (self.R_inst_rw is not None) & (conv_nbin is not None) & (self.cframe.info_c[i_comp]['mod_used'] == 'iron'):
                fwhm = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['fwhm']]
                R_inst_w = np.interp(orig_wave_z_w, self.R_inst_rw[0], self.R_inst_rw[1])
                orig_flux_dzc_ew = convolve_var_width_fft(orig_wave_z_w, orig_flux_dz_ew, dv_fwhm_obj=fwhm, 
                                                          dw_fwhm_ref=self.dw_fwhm_dsp_w*z_ratio, R_inst_w=R_inst_w, num_bins=conv_nbin)
            else:
                orig_flux_dzc_ew = orig_flux_dz_ew # just copy if convlution not required, e.g., for broad-band sed fitting

            # project to observed wavelength
            interp_func = interp1d(orig_wave_z_w, orig_flux_dzc_ew, axis=1, kind='linear', fill_value=(0,0), bounds_error=False)
            obs_flux_scomp_ew = interp_func(obs_wave_w)

            if obs_flux_mcomp_ew is None: 
                obs_flux_mcomp_ew = obs_flux_scomp_ew
            else:
                obs_flux_mcomp_ew = np.vstack((obs_flux_mcomp_ew, obs_flux_scomp_ew))

        if mask_lite_e is not None:
            obs_flux_mcomp_ew = obs_flux_mcomp_ew[mask_lite_e,:]

        return obs_flux_mcomp_ew

    ##########################################################################
    ########################## Output functions ##############################

    def extract_results(self, step=None, if_print_results=True, if_return_results=False, if_rev_v0_redshift=False, if_show_average=False, **kwargs):

        ############################################################
        # check and replace the args to be compatible with old version <= 2.2.4
        if 'print_results'  in kwargs: if_print_results = kwargs['print_results']
        if 'return_results' in kwargs: if_return_results = kwargs['return_results']
        if 'show_average'   in kwargs: if_show_average = kwargs['show_average']
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

        self.num_loops = self.fframe.num_loops # for print_results
        self.spec_flux_scale = self.fframe.spec_flux_scale # to calculate luminosity in printing
        comp_name_c = self.cframe.comp_name_c
        num_comps = self.cframe.num_comps
        par_name_cp = self.cframe.par_name_cp

        # list the properties to be output
        value_names_additive = ['flux_3000', 'flux_5100', 'flux_wavenorm']
        value_names_C = {}
        for (i_comp, comp_name) in enumerate(comp_name_c):
            if self.cframe.info_c[i_comp]['mod_used'] == 'powerlaw':       
                value_names_C[comp_name] = value_names_additive + ['log_lambLum_3000', 'log_lambLum_5100', 'log_lambLum_wavenorm']
            if self.cframe.info_c[i_comp]['mod_used'] == 'bending-powerlaw':       
                value_names_C[comp_name] = value_names_additive + ['log_lambLum_3000', 'log_lambLum_5100', 'log_lambLum_wavenorm', 'log_lambLum_waveturn', 'flux_waveturn']
            if self.cframe.info_c[i_comp]['mod_used'] == 'blackbody':
                value_names_C[comp_name] = value_names_additive + ['log_lambLum_3000', 'log_lambLum_5100', 'log_lambLum_wavenorm', 'log_Lum_int']
            if self.cframe.info_c[i_comp]['mod_used'] == 'recombination':
                value_names_C[comp_name] = value_names_additive + ['log_Lum_int']
            if self.cframe.info_c[i_comp]['mod_used'] == 'iron':      
                value_names_C[comp_name] = value_names_additive + ['log_Lum_uv', 'log_Lum_opt']

            lum_names = []
            wave_unit_str = 'um' if casefold(self.cframe.info_c[i_comp]['int_wave_unit']) in ['micron', 'um'] else 'A'
            for wave_range in self.cframe.info_c[i_comp]['int_wave_range']:
                if wave_range is None: continue
                for lum_type in self.cframe.info_c[i_comp]['int_lum_type']:
                    lum_names.append(f"log_Lum_{wave_range[0]}_{wave_range[1]}{wave_unit_str}_{lum_type}")
            value_names_C[comp_name] += lum_names
            if i_comp == 0: 
                lum_names_additive = lum_names
            else:
                lum_names_additive = [lum_name for lum_name in lum_names_additive if lum_name in value_names_C[comp_name]]
        value_names_additive += lum_names_additive

        # format of results
        # output_C['comp']['par_lp'][i_l,i_p]: parameters
        # output_C['comp']['coeff_le'][i_l,i_e]: coefficients
        # output_C['comp']['value_Vl']['name_l'][i_l]: calculated values
        output_C = {}
        for (i_comp, comp_name) in enumerate(comp_name_c):
            output_C[comp_name] = {} # init results for each comp
            output_C[comp_name]['value_Vl'] = {}
            for value_name in par_name_cp[i_comp] + value_names_C[comp_name]:
                output_C[comp_name]['value_Vl'][value_name] = np.zeros(self.num_loops, dtype='float')
        output_C['sum'] = {}
        output_C['sum']['value_Vl'] = {} # only init values for sum of all comp
        for value_name in value_names_additive:
            output_C['sum']['value_Vl'][value_name] = np.zeros(self.num_loops, dtype='float')

        # locate the results of the model in the full fitting results
        i_pars_0_of_mod, i_pars_1_of_mod, i_coeffs_0_of_mod, i_coeffs_1_of_mod = self.fframe.search_mod_index(self.mod_name, self.fframe.full_mod_type)
        for (i_comp, comp_name) in enumerate(comp_name_c):
            i_pars_0_of_comp_in_mod, i_pars_1_of_comp_in_mod, i_coeffs_0_of_comp_in_mod, i_coeffs_1_of_comp_in_mod = self.fframe.search_comp_index(comp_name, self.mod_name)

            output_C[comp_name]['par_lp']   = best_par_lp[:, i_pars_0_of_mod:i_pars_1_of_mod][:, i_pars_0_of_comp_in_mod:i_pars_1_of_comp_in_mod]
            output_C[comp_name]['coeff_le'] = best_coeff_le[:, i_coeffs_0_of_mod:i_coeffs_1_of_mod][:, i_coeffs_0_of_comp_in_mod:i_coeffs_1_of_comp_in_mod]

            for i_par in range(self.cframe.num_pars_c[i_comp]): 
                output_C[comp_name]['value_Vl'][par_name_cp[i_comp][i_par]] = output_C[comp_name]['par_lp'][:, i_par]
            for i_loop in range(self.num_loops):
                par_p   = output_C[comp_name]['par_lp'][i_loop]
                coeff_e = output_C[comp_name]['coeff_le'][i_loop]

                voff = par_p[self.cframe.par_index_cP[i_comp]['voff']]
                rev_redshift = (1+voff/299792.458)*(1+self.v0_redshift)-1

                tmp_spec_w = self.fframe.output_MC[self.mod_name][comp_name]['spec_lw'][i_loop, :]
                mask_norm_w = np.abs(self.fframe.spec['wave_w']/(1+rev_redshift) - 3000) < 25 # for observed flux at rest 3000 angstrom
                if mask_norm_w.sum() > 0:
                    output_C[comp_name]['value_Vl']['flux_3000'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                    output_C['sum']['value_Vl']['flux_3000'][i_loop] += tmp_spec_w[mask_norm_w].mean()
                mask_norm_w = np.abs(self.fframe.spec['wave_w']/(1+rev_redshift) - 5100) < 25 # for observed flux at rest 5100 angstrom
                if mask_norm_w.sum() > 0:
                    output_C[comp_name]['value_Vl']['flux_5100'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                    output_C['sum']['value_Vl']['flux_5100'][i_loop] += tmp_spec_w[mask_norm_w].mean()
                mask_norm_w = np.abs(self.fframe.spec['wave_w']/(1+rev_redshift) - self.w_norm) < self.dw_norm # for observed flux at user given wavenorm
                if mask_norm_w.sum() > 0:
                    output_C[comp_name]['value_Vl']['flux_wavenorm'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                    output_C['sum']['value_Vl']['flux_wavenorm'][i_loop] += tmp_spec_w[mask_norm_w].mean()

                dist_lum = cosmo.luminosity_distance(rev_redshift).to('cm').value
                unitconv = 4*np.pi*dist_lum**2 * self.spec_flux_scale / const.L_sun.to('erg/s').value # convert intrinsic flux in erg/s/cm2/A to Lum in Lsun/A

                if self.cframe.info_c[i_comp]['mod_used'] == 'powerlaw':
                    alpha_lambda = par_p[self.cframe.par_index_cP[i_comp]['alpha_lambda']]
                    for (wave, wave_str) in zip([3000, 5100, self.w_norm], ['3000', '5100', 'wavenorm']):
                        flux_wave = coeff_e[0] * self.powerlaw_func(wave, wave_norm=self.w_norm, alpha_lambda1=alpha_lambda, alpha_lambda2=None, 
                                                                    curvature=None, bending=False)
                        lambLum_wave = flux_wave * unitconv * wave
                        output_C[comp_name]['value_Vl']['log_lambLum_'+wave_str][i_loop] = np.log10(lambLum_wave)

                if self.cframe.info_c[i_comp]['mod_used'] == 'bending-powerlaw':
                    alpha_lambda1 = par_p[self.cframe.par_index_cP[i_comp]['alpha_lambda1']]
                    alpha_lambda2 = par_p[self.cframe.par_index_cP[i_comp]['alpha_lambda2']]
                    wave_turn     = par_p[self.cframe.par_index_cP[i_comp]['wave_turn']]
                    curvature     = par_p[self.cframe.par_index_cP[i_comp]['curvature']]
                    for (wave, wave_str) in zip([3000, 5100, self.w_norm, wave_turn], ['3000', '5100', 'wavenorm', 'waveturn']):
                        flux_wave = coeff_e[0] * self.powerlaw_func(wave, wave_norm=wave_turn, alpha_lambda1=alpha_lambda1, alpha_lambda2=alpha_lambda2, 
                                                                    curvature=curvature, bending=True)
                        lambLum_wave = flux_wave * unitconv * wave
                        output_C[comp_name]['value_Vl']['log_lambLum_'+wave_str][i_loop] = np.log10(lambLum_wave)
                    mask_norm_w = np.abs(self.fframe.spec['wave_w']/(1+rev_redshift) - wave_turn) < self.dw_norm 
                    if mask_norm_w.sum() > 0:
                        output_C[comp_name]['value_Vl']['flux_waveturn'][i_loop] = tmp_spec_w[mask_norm_w].mean()

                if self.cframe.info_c[i_comp]['mod_used'] == 'blackbody':
                    log_tem  = par_p[self.cframe.par_index_cP[i_comp]['log_tem']]
                    for (wave, wave_str) in zip([3000, 5100, self.w_norm], ['3000', '5100', 'wavenorm']):
                        flux_wave = coeff_e[0] * self.blackbody_func(wave, log_tem=log_tem, wave_norm=self.w_norm)
                        lambLum_wave = flux_wave * unitconv * wave
                        output_C[comp_name]['value_Vl']['log_lambLum_'+wave_str][i_loop] = np.log10(lambLum_wave)
                    tmp_wave_w = np.logspace(np.log10(912), 7.5, num=10000) # till 10 K
                    tmp_bb_w = self.blackbody_func(tmp_wave_w, log_tem=log_tem, wave_norm=self.w_norm)
                    output_C[comp_name]['value_Vl']['log_Lum_int'][i_loop] = np.log10(coeff_e[0] * unitconv * np.trapezoid(tmp_bb_w, x=tmp_wave_w))

                if self.cframe.info_c[i_comp]['mod_used'] == 'recombination':
                    log_e_tem  = par_p[self.cframe.par_index_cP[i_comp]['log_e_tem']]
                    log_tau_be = par_p[self.cframe.par_index_cP[i_comp]['log_tau_be']]
                    tmp_wave_w = np.logspace(np.log10(912), 7.5, num=10000) # till 10 K
                    tmp_rec_w = self.recombination_func(tmp_wave_w, log_e_tem=log_e_tem, log_tau_be=log_tau_be, H_series=self.cframe.info_c[i_comp]['H_series'])
                    output_C[comp_name]['value_Vl']['log_Lum_int'][i_loop] = np.log10(coeff_e[0] * unitconv * np.trapezoid(tmp_rec_w, x=tmp_wave_w))

                if self.cframe.info_c[i_comp]['mod_used'] == 'iron':
                    if self.cframe.info_c[i_comp]['segments']:
                        coeff_uv  = (coeff_e[1:4] * self.iron_dict['flux_norm_e'][1:4])[coeff_e[1:4] > 0].sum()
                        coeff_opt = (coeff_e[4:6] * self.iron_dict['flux_norm_e'][4:6])[coeff_e[4:6] > 0].sum()
                    else:
                        coeff_uv  = coeff_e[0]
                        coeff_opt = coeff_e[0] * self.iron_dict['flux_opt_uv_ratio']
                    output_C[comp_name]['value_Vl']['log_Lum_uv' ][i_loop] = np.log10(coeff_uv  * unitconv)
                    output_C[comp_name]['value_Vl']['log_Lum_opt'][i_loop] = np.log10(coeff_opt * unitconv)

                # calculate integrated lum in given wavelength ranges
                for wave_range in self.cframe.info_c[i_comp]['int_wave_range']: 
                    if wave_range is None: continue
                    wave_unit_str   = 'um' if casefold(self.cframe.info_c[i_comp]['int_wave_unit']) in ['micron', 'um'] else 'A'
                    wave_unit_ratio = 1e4  if casefold(self.cframe.info_c[i_comp]['int_wave_unit']) in ['micron', 'um'] else 1
                    if casefold(self.cframe.info_c[i_comp]['int_wave_frame']) in ['rest']: wave_unit_ratio *= (1+rev_redshift) # rest to obs range
                    int_wave_w = np.logspace(np.log10(wave_range[0]*wave_unit_ratio), np.log10(wave_range[1]*wave_unit_ratio), num=1000) # obs frame grid

                    tmp_coeff_e = best_coeff_le[i_loop, i_coeffs_0_of_mod:i_coeffs_1_of_mod][i_coeffs_0_of_comp_in_mod:i_coeffs_1_of_comp_in_mod]
                    for lum_type in self.cframe.info_c[i_comp]['int_lum_type']:
                        if lum_type in ['intrinsic','absorbed']:
                            tmp_flux_ew = self.create_models(int_wave_w, best_par_lp[i_loop, i_pars_0_of_mod:i_pars_1_of_mod], components=comp_name, 
                                                             if_dust_ext=False, if_redshift=False, if_full_range=True) # flux in rest frame
                            int_lum_intrinsic = np.trapezoid(tmp_coeff_e@tmp_flux_ew, x=int_wave_w/(1+rev_redshift)) * unitconv # from erg/s/cm2/A to Lsun
                        if lum_type in ['observed','absorbed']:
                            tmp_flux_ew = self.create_models(int_wave_w, best_par_lp[i_loop, i_pars_0_of_mod:i_pars_1_of_mod], components=comp_name, 
                                                             if_dust_ext=True, if_redshift=False, if_full_range=True) # flux in rest frame
                            int_lum_observed = np.trapezoid(tmp_coeff_e@tmp_flux_ew, x=int_wave_w/(1+rev_redshift)) * unitconv # from erg/s/cm2/A to Lsun
                        if lum_type == 'intrinsic': int_lum = copy(int_lum_intrinsic)
                        if lum_type == 'observed' : int_lum = copy(int_lum_observed)
                        if lum_type == 'absorbed' : int_lum = int_lum_intrinsic - int_lum_observed

                        value_name = f"log_Lum_{wave_range[0]}_{wave_range[1]}{wave_unit_str}_{lum_type}"
                        output_C[comp_name]['value_Vl'][value_name][i_loop] = np.log10(int_lum)
                        if value_name in output_C['sum']['value_Vl']: output_C['sum']['value_Vl'][value_name][i_loop] += int_lum

        for value_name in output_C['sum']['value_Vl']:
            if value_name[:8] in ['log_Lum_', 'log_lamb']: 
                output_C['sum']['value_Vl'][value_name] = np.log10(output_C['sum']['value_Vl'][value_name])

        # updated to requested lum_unit for each comp
        for (i_comp, comp_name) in enumerate(comp_name_c):
            if casefold(self.cframe.info_c[i_comp]['int_lum_unit']) in ['erg/s', 'erg s-1']: 
                for value_name in output_C[comp_name]['value_Vl']:
                    if value_name[:8] in ['log_Lum_', 'log_lamb']: 
                        output_C[comp_name]['value_Vl'][value_name] += np.log10(const.L_sun.to('erg/s').value) # from log Lsun to log erg/s
        # if all comp have the same lum unit, also update the sum
        if len(set(self.cframe.info_c[i_comp]['int_lum_unit'] for i_comp in range(self.num_comps))) == 1: 
            if casefold(self.cframe.info_c[0]['int_lum_unit']) in ['erg/s', 'erg s-1']:
                for value_name in output_C['sum']['value_Vl']:
                    if value_name[:8] in ['log_Lum_', 'log_lamb']: 
                        output_C['sum']['value_Vl'][value_name] += np.log10(const.L_sun.to('erg/s').value) # from log Lsun to log erg/s

        ############################################################
        # keep aliases for output in old version <= 2.2.4
        for (i_comp, comp_name) in enumerate(comp_name_c):
            output_C[comp_name]['values'] = output_C[comp_name]['value_Vl']
        output_C['sum']['values'] = output_C['sum']['value_Vl']
        ############################################################

        self.output_C = output_C # save to model frame
        if if_print_results: self.print_results(log=self.fframe.log_message, if_show_average=if_show_average)
        if if_return_results: return output_C

    def print_results(self, log=[], if_show_average=False):
        print_log(f"#### Best-fit AGN properties ####", log)

        mask_l = np.ones(self.num_loops, dtype='bool')
        if not if_show_average: mask_l[1:] = False

        # set the print name for each value
        print_name_CV = {}
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            print_name_CV[comp_name] = {}
            for value_name in self.output_C[comp_name]['value_Vl']: print_name_CV[comp_name][value_name] = value_name

            print_name_CV[comp_name]['voff'] = 'Velocity shift in relative to z_sys (km/s)'
            print_name_CV[comp_name]['fwhm'] = 'Velocity FWHM (km/s)'
            print_name_CV[comp_name]['Av'] = 'Extinction (Av)'
            print_name_CV[comp_name]['log_lambLum_3000'] = f"λL3000 (rest,intrinsic) "
            print_name_CV[comp_name]['log_lambLum_5100'] = f"λL5100 (rest,intrinsic) "
            print_name_CV[comp_name]['log_lambLum_wavenorm'] = f"λL{self.w_norm:.0f} (rest,intrinsic) "
            print_name_CV[comp_name]['flux_3000'] = f"F3000 (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)"
            print_name_CV[comp_name]['flux_5100'] = f"F5100 (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)"
            print_name_CV[comp_name]['flux_wavenorm'] = f"F{self.w_norm:.0f} (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)"
            if self.cframe.info_c[i_comp]['mod_used'] == 'powerlaw':
                print_name_CV[comp_name]['alpha_lambda'] = 'Powerlaw α_λ'
            if self.cframe.info_c[i_comp]['mod_used'] == 'bending-powerlaw':
                print_name_CV[comp_name]['curvature'] = 'Curvature'
                print_name_CV[comp_name]['wave_turn'] = 'Turning wavelength (rest, Å)'
                wave_turn = self.output_C[[*self.output_C][i_comp]]['value_Vl']['wave_turn'][0]
                print_name_CV[comp_name]['alpha_lambda1'] = f"Powerlaw α1_λ (λ < {wave_turn:.0f})"
                print_name_CV[comp_name]['alpha_lambda2'] = f"Powerlaw α2_λ (λ > {wave_turn:.0f})"
                print_name_CV[comp_name]['flux_waveturn'] = f"F{wave_turn:.0f} (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)"
                print_name_CV[comp_name]['log_lambLum_waveturn'] = f"λL{wave_turn:.0f} (rest,intrinsic) "
            if self.cframe.info_c[i_comp]['mod_used'] == 'blackbody':
                print_name_CV[comp_name]['log_tem'] = 'Blackbody temperature (log K)'
                print_name_CV[comp_name]['log_Lum_int'] = f"Bolometric Lum. "
            if self.cframe.info_c[i_comp]['mod_used'] == 'recombination':
                print_name_CV[comp_name]['log_e_tem'] = 'Recombination cont. e- temperature (log K)'
                print_name_CV[comp_name]['log_tau_be'] = 'Recombination cont. optical depth at 3646 Å (log τ)'
                print_name_CV[comp_name]['log_Lum_int'] = f"Bolometric Lum. "
            if self.cframe.info_c[i_comp]['mod_used'] == 'iron':
                print_name_CV[comp_name]['log_Lum_uv']  = 'Integrated intrinsic Lum. (UV, 2150-4000 Å) '
                print_name_CV[comp_name]['log_Lum_opt'] = 'Integrated intrinsic Lum. (optical, 4000-5600 Å) '

            for wave_range in self.cframe.info_c[i_comp]['int_wave_range']: 
                wave_unit_str_0 = 'um' if casefold(self.cframe.info_c[i_comp]['int_wave_unit']) in ['micron', 'um'] else 'A'
                wave_unit_str_1 = 'µm' if casefold(self.cframe.info_c[i_comp]['int_wave_unit']) in ['micron', 'um'] else 'Å'
                for lum_type in self.cframe.info_c[i_comp]['int_lum_type']:
                    value_name = f"log_Lum_{wave_range[0]}_{wave_range[1]}{wave_unit_str_0}_{lum_type}"
                    if lum_type == 'absorbed': lum_type = 'dust-absorbed'
                    print_name_CV[comp_name][value_name] = f"Integrated {lum_type} Lum. "+f"({wave_range[0]}-{wave_range[1]} {wave_unit_str_1}) "
        print_name_CV['sum'] = {}
        for value_name in self.output_C['sum']['value_Vl']:
            print_name_CV['sum'][value_name] = copy(print_name_CV[self.comp_name_c[0]][value_name])

        # updated to requested lum_unit for each comp
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            lum_unit_str = '(log Lsun) ' if casefold(self.cframe.info_c[i_comp]['int_lum_unit']) in ['lsun', 'l_sun'] else '(log erg/s)'
            for value_name in self.output_C[comp_name]['value_Vl']:
                if value_name[:8] in ['log_Lum_', 'log_lamb']: 
                    print_name_CV[comp_name][value_name] += lum_unit_str
        # if all comp have the same lum unit, also update the sum
        if len(set(self.cframe.info_c[i_comp]['int_lum_unit'] for i_comp in range(self.num_comps))) == 1: 
            lum_unit_str = '(log Lsun) ' if casefold(self.cframe.info_c[0]['int_lum_unit']) in ['lsun', 'l_sun'] else '(log erg/s)'
        else:
            lum_unit_str = '(log Lsun) ' # default
        for value_name in self.output_C['sum']['value_Vl']:
            if value_name[:8] in ['log_Lum_', 'log_lamb']: 
                print_name_CV['sum'][value_name] += lum_unit_str

        print_length = max([len(print_name_CV[comp_name][value_name]) for comp_name in print_name_CV for value_name in print_name_CV[comp_name]] + [40]) # set min length
        for comp_name in print_name_CV:
            for value_name in print_name_CV[comp_name]:
                print_name_CV[comp_name][value_name] += ' '*(print_length-len(print_name_CV[comp_name][value_name]))

        for (i_comp, comp_name) in enumerate(self.output_C):
            value_Vl = self.output_C[[*self.output_C][i_comp]]['value_Vl']
            # value_names = [*value_Vl]
            value_names = []
            msg = ''
            if i_comp < self.cframe.num_comps: # print best-fit pars for each comp
                print_log(f"# AGN component <{self.cframe.comp_name_c[i_comp]}>:", log)
                if self.cframe.info_c[i_comp]['mod_used'] == 'iron':
                    value_names += ['voff', 'fwhm']
                else:
                    print_log(f"[Note] velocity shift (i.e., redshift) and FWHM are tied following the input model_config.", log)
                value_names += ['Av']
                if self.cframe.info_c[i_comp]['mod_used'] == 'powerlaw':
                    value_names += ['alpha_lambda']
                if self.cframe.info_c[i_comp]['mod_used'] == 'bending-powerlaw':
                    value_names += ['curvature', 'wave_turn', 'alpha_lambda1', 'alpha_lambda2', 'flux_waveturn', 'log_lambLum_waveturn']
                if self.cframe.info_c[i_comp]['mod_used'] == 'blackbody':
                    value_names += ['log_tem', 'log_Lum_int']
                if self.cframe.info_c[i_comp]['mod_used'] == 'recombination':
                    value_names += ['log_e_tem', 'log_tau_be', 'log_Lum_int']
                if self.cframe.info_c[i_comp]['mod_used'] == 'iron':
                    value_names += ['log_Lum_uv', 'log_Lum_opt']
                if self.cframe.info_c[i_comp]['mod_used'] in ['powerlaw', 'bending-powerlaw', 'blackbody']:
                    value_names += ['log_lambLum_3000', 'log_lambLum_5100']
                    if self.w_norm not in [3000,5100]: value_names += ['log_lambLum_wavenorm']
            elif self.cframe.num_comps >= 2: # print sum only if using >= 2 comps
                print_log(f"# Best-fit properties of the sum of all AGN components.", log)
            else: 
                continue
            value_names += ['flux_3000', 'flux_5100']
            if self.w_norm not in [3000,5100]: value_names += ['flux_wavenorm']

            value_names += [value_name for value_name in [*value_Vl] if (value_name[:8] in ['log_Lum_', 'log_lamb']) & (value_name not in value_names)]

            for value_name in value_names:
                msg += '| ' + print_name_CV[comp_name][value_name] + f" = {value_Vl[value_name][mask_l].mean():10.4f}" + f" +/- {value_Vl[value_name].std():<10.4f}|\n"
            msg = msg[:-1] # remove the last \n
            bar = '=' * len(msg.split('\n')[-1])
            print_log(bar, log)
            print_log(msg, log)
            print_log(bar, log)
            print_log('', log)

