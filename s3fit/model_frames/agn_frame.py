# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: s3fit@xychen.me

from copy import deepcopy as copy
import numpy as np
np.set_printoptions(linewidth=10000)
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import astropy.constants as const

from ..auxiliaries.auxiliary_frames import ConfigFrame
from ..auxiliaries.auxiliary_functions import print_log, casefold, color_list_dict, convolve_var_width_fft
# from ..auxiliaries.basic_model_functions import powerlaw_func, blackbody_func, recombination_func
from ..auxiliaries.extinct_laws import ExtLaw

class AGNFrame(object):
    def __init__(self, mod_name=None, fframe=None, config=None, 
                 v0_redshift=None, R_inst_rw=None, 
                 w_min=None, w_max=None, 
                 Rratio_mod=None, dw_fwhm_dsp=None, dw_pix_inst=None, 
                 verbose=True, log_message=[]):

        self.mod_name = mod_name
        self.fframe = fframe
        self.config = config
        self.v0_redshift = v0_redshift
        self.R_inst_rw = R_inst_rw        
        self.w_min = w_min
        self.w_max = w_max
        self.Rratio_mod = Rratio_mod # resolution ratio of model / instrument
        self.dw_fwhm_dsp = dw_fwhm_dsp # model convolving width for downsampling (rest frame)
        self.dw_pix_inst = dw_pix_inst # data sampling width (obs frame)
        self.verbose = verbose
        self.log_message = log_message

        self.cframe=ConfigFrame(self.config)
        self.comp_name_c = self.cframe.comp_name_c
        self.num_comps = self.cframe.num_comps
        self.check_config()

        # set original wavelength grid, required to project iron template
        orig_wave_logbin = 0.05
        orig_wave_num = int(np.round(np.log10(w_max/w_min) / orig_wave_logbin))
        self.orig_wave_w = np.logspace(np.log10(w_min), np.log10(w_max), num=orig_wave_num)
        # load iron template
        if 'iron' in [self.cframe.comp_info_cI[i_comp]['mod_used'] for i_comp in range(self.num_comps)]: 
            self.read_iron()

        # count the number of independent model elements
        self.num_coeffs_c = np.zeros(self.num_comps, dtype='int')
        for i_comp in range(self.num_comps):
            if self.cframe.comp_info_cI[i_comp]['mod_used'] in ['powerlaw', 'bending-powerlaw', 'blackbody', 'recombination']:
                self.num_coeffs_c[i_comp] = 1 # one independent element per component
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'iron':
                if self.cframe.comp_info_cI[i_comp]['segments']: 
                    self.num_coeffs_c[i_comp] = self.iron_dict['orig_flux_ew'].shape[0] # multiple independent segments
                else:
                    self.num_coeffs_c[i_comp] = 1 # one independent segment
        self.num_coeffs = self.num_coeffs_c.sum()

        # currently do not consider negative spectra 
        self.mask_absorption_e = np.zeros((self.num_coeffs), dtype='bool')

        if self.verbose:
            print_log(f"AGN UV/optical/NIR continuum components: {[self.cframe.comp_info_cI[i_comp]['mod_used'] for i_comp in range(self.num_comps)]}", self.log_message)
            for i_comp in range(self.num_comps):
                if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'recombination':
                    print_log(f"Lower-level principal quantum number of Hydrogen recombination continuum: {self.cframe.comp_info_cI[i_comp]['H_series']}", self.log_message)
                if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'iron':
                    if self.cframe.comp_info_cI[i_comp]['segments']: 
                        print_log(f"Wavelength segments for fitting of Fe II template: {self.iron_dict['wave_segments'][self.iron_dict['orig_flux_norm_e']>0].tolist()}", self.log_message)

        # set plot styles
        self.plot_style_C = {}
        self.plot_style_C['sum'] = {'color': 'C3', 'alpha': 0.75, 'linestyle': '-', 'linewidth': 1.5}
        i_red, i_yellow, i_green, i_purple = 0, 0, 0, 0
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            if self.cframe.comp_info_cI[i_comp]['mod_used'] in ['powerlaw', 'bending-powerlaw']:
                self.plot_style_C[comp_name] = {'color': 'None', 'alpha': 0.5, 'linestyle': '--', 'linewidth': 1}
                self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['purple'], i_purple, mode="wrap"))
                i_purple += 1
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'blackbody':
                self.plot_style_C[comp_name] = {'color': 'None', 'alpha': 0.5, 'linestyle': '--', 'linewidth': 1}
                self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['red'], i_red, mode="wrap"))
                i_red += 1
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'recombination':
                self.plot_style_C[comp_name] = {'color': 'None', 'alpha': 0.5, 'linestyle': '--', 'linewidth': 1}
                self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['green'], i_green, mode="wrap"))
                i_green += 1
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'iron':
                self.plot_style_C[comp_name] = {'color': 'None', 'alpha': 0.5, 'linestyle': '-', 'linewidth': 0.75}
                self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['yellow'], i_yellow, mode="wrap"))
                i_yellow += 1

    ##########################################################################

    def check_config(self):

        # check alternative model names
        for i_comp in range(self.num_comps):
            if casefold(self.cframe.comp_info_cI[i_comp]['mod_used']) in ['powerlaw', 'pl']: 
                self.cframe.comp_info_cI[i_comp]['mod_used'] = 'powerlaw'
            if casefold(self.cframe.comp_info_cI[i_comp]['mod_used']) in ['bending-powerlaw', 'bending_powerlaw', 'bending powerlaw', 'bending-pl', 'bending_pl', 'bending pl']: 
                self.cframe.comp_info_cI[i_comp]['mod_used'] = 'bending-powerlaw'
            if casefold(self.cframe.comp_info_cI[i_comp]['mod_used']) in ['blackbody', 'black_body', 'black body', 'bb']: 
                self.cframe.comp_info_cI[i_comp]['mod_used'] = 'blackbody'
            if casefold(self.cframe.comp_info_cI[i_comp]['mod_used']) in ['recombination', 'recombination_continuum', 'recombination continuum', 'rec', 'balmer_continuum', 'balmer continuum', 'bac']: 
                self.cframe.comp_info_cI[i_comp]['mod_used'] = 'recombination'
            if casefold(self.cframe.comp_info_cI[i_comp]['mod_used']) in ['iron', 'feii', 'fe ii']: 
                self.cframe.comp_info_cI[i_comp]['mod_used'] = 'iron'

        ############################################################
        # to be compatible with old version <= 2.2.4
        if len(self.cframe.par_index_cP[0]) == 0:
            for i_comp in range(self.num_comps):
                if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'powerlaw':
                    self.cframe.par_name_cp[i_comp]  = ['voff', 'fwhm', 'Av', 'alpha_lambda']
                    self.cframe.par_index_cP[i_comp] = {'voff': 0, 'fwhm': 1, 'Av': 2, 'alpha_lambda': 3}
                if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'bending-powerlaw':
                    self.cframe.par_name_cp[i_comp]  = ['voff', 'fwhm', 'Av', 'alpha_lambda1', 'alpha_lambda2', 'wave_turn', 'curvature']
                    self.cframe.par_index_cP[i_comp] = {'voff': 0, 'fwhm': 1, 'Av': 2, 'alpha_lambda1': 3, 'alpha_lambda2': 4, 'wave_turn': 5, 'curvature': 6}
                if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'blackbody':
                    self.cframe.par_name_cp[i_comp]  = ['voff', 'fwhm', 'Av', 'log_tem',]
                    self.cframe.par_index_cP[i_comp] = {'voff': 0, 'fwhm': 1, 'Av': 2, 'log_tem': 3}
                if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'recombination':
                    self.cframe.par_name_cp[i_comp]  = ['voff', 'fwhm', 'Av', 'log_e_tem', 'log_tau_be']
                    self.cframe.par_index_cP[i_comp] = {'voff': 0, 'fwhm': 1, 'Av': 2, 'log_e_tem': 3, 'log_tau_be': 4}
                if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'iron':
                    self.cframe.par_name_cp[i_comp]  = ['voff', 'fwhm', 'Av']
                    self.cframe.par_index_cP[i_comp] = {'voff': 0, 'fwhm': 1, 'Av': 2}
        ############################################################

        # set inherited or default info if not specified in config
        # model-level info
        self.cframe.retrieve_inherited_info( 'w_norm', alt_names='norm_wave' , root_info_I=self.fframe.root_info_I, default=5500)
        self.cframe.retrieve_inherited_info('dw_norm', alt_names='norm_width', root_info_I=self.fframe.root_info_I, default=25)

        # component-level info
        # format of returned flux / Lum density or integrated values
        for i_comp in range(self.num_comps):
            # either 2-unit-nested tuples (for wave and value, respectively) or dictionary as follows are supported
            ret_value_formats = [((3000, 10, 'angstrom', 'rest'), ('Flam', 'erg s-1 cm-2 angstrom-1', 'observed')), 
                                 {'wave_center': 5100, 'wave_width': 10, 'wave_unit': 'angstrom', 'wave_frame': 'rest', 
                                  'value_form': 'Flam', 'value_unit': 'erg s-1 cm-2 angstrom-1', 'value_state': 'observed'},
                                 (( 912, 30000, 'angstrom', 'rest'), ('intLum', 'L_sun', 'intrinsic')),
                                 {'wave_min': 912, 'wave_max': 30000, 'wave_unit': 'angstrom', 'wave_frame': 'rest', 
                                  'value_form': 'intLum', 'value_unit': 'L_sun', 'value_state': 'absorbed'} ]
            if self.cframe.comp_info_cI[i_comp]['mod_used'] in ['powerlaw', 'bending-powerlaw', 'blackbody']:
                ret_value_formats += [ ((3000, 10, 'angstrom', 'rest'), ('lamLlam', 'erg s-1', 'intrinsic')),
                                       ((5100, 10, 'angstrom', 'rest'), ('lamLlam', 'erg s-1', 'intrinsic')) ]
            self.cframe.retrieve_inherited_info('ret_value_formats', i_comp=i_comp, root_info_I=self.fframe.root_info_I, default=ret_value_formats)
            # 'wave_unit': any length unit supported by astropy.unit
            # 'value_form': 'Flam', 'lamFlam', 'Fnu', 'nuFnu', 'intFlux'; 'Llam', 'lamLlam', 'Lnu', 'nuLnu', intLum'
            # 'value_state': 'intrinsic', 'observed', 'absorbed' (i.e., dust absorbed)
            # 'value_unit': any flux/luminosity or its density unit supported by astropy.unit
 
            # re-categorize ret_value_formats
            if self.cframe.comp_info_cI[i_comp]['ret_value_formats'] is None: continue # user can set None to skip all of these calculations
            # group line info to a list
            if isinstance(self.cframe.comp_info_cI[i_comp]['ret_value_formats'], (tuple, dict)): 
                self.cframe.comp_info_cI[i_comp]['ret_value_formats'] = [self.cframe.comp_info_cI[i_comp]['ret_value_formats']]
            # convert tuple format to dict
            for i_ret in range(len(self.cframe.comp_info_cI[i_comp]['ret_value_formats'])):
                tmp_tuple = self.cframe.comp_info_cI[i_comp]['ret_value_formats'][i_ret]
                if isinstance(tmp_tuple, tuple):
                    tmp_dict = {}
                    wave_0, wave_1 = tmp_tuple[0][:2]
                    if wave_0 > wave_1:
                        tmp_dict['wave_center'], tmp_dict['wave_width'] = wave_0, wave_1
                    else:
                        tmp_dict['wave_min'], tmp_dict['wave_max'] = wave_0, wave_1 if wave_1 > wave_0 else wave_0+1
                    tmp_dict['wave_unit']  = tmp_tuple[0][2]
                    tmp_dict['wave_frame'] = tmp_tuple[0][3] if len(tmp_tuple[0]) > 3 else 'rest'
                    tmp_dict['value_form'], tmp_dict['value_unit'], tmp_dict['value_state'] = tmp_tuple[1]
                else:
                    tmp_dict = self.cframe.comp_info_cI[i_comp]['ret_value_formats'][i_ret]
                # check alternatives
                tmp_dict['wave_frame'] = 'obs' if casefold(tmp_dict['wave_frame']) in ['observed', 'obs', 'obs.'] else 'rest'
                if tmp_dict['value_form'] == 'flam': tmp_dict['value_form'] = 'Flam'
                if tmp_dict['value_form'] == 'fnu' : tmp_dict['value_form'] = 'Fnu'
                if casefold(tmp_dict['value_state']) in ['intrinsic', 'original']:
                    tmp_dict['value_state'] = 'intrinsic'
                elif casefold(tmp_dict['value_state']) in ['observed', 'reddened', 'attenuated', 'extincted', 'extinct']:
                    tmp_dict['value_state'] = 'observed'
                elif casefold(tmp_dict['value_state']) in ['absorbed', 'dust']:
                    tmp_dict['value_state'] = 'absorbed'
                self.cframe.comp_info_cI[i_comp]['ret_value_formats'][i_ret] = tmp_dict

        # other component-level info
        for i_comp in range(self.num_comps):
            if self.cframe.comp_info_cI[i_comp]['mod_used'] in ['powerlaw', 'bending-powerlaw']:
                # truncated range and index of powerlaw, either nested tuple or dict format, e.g., {'wave_range': (5, None), 'wave_unit': 'micron', 'alpha_lambda': -4}
                self.cframe.retrieve_inherited_info('truncation', i_comp=i_comp, default=[((None, 0.01), 'micron',  0.2), 
                                                                                          ((0.01, 0.10), 'micron', -1.0), 
                                                                                          ((5.00, None), 'micron', -4.0)])
                # the adopted values follow Schartmann et al. 2005; Feltre et al. 2012; Stalevski et al. 2016
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'recombination':
                self.cframe.retrieve_inherited_info('H_series', i_comp=i_comp, default=[2,3,4,5])
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'iron':
                self.cframe.retrieve_inherited_info('segments', i_comp=i_comp, default=False)

        # group single info to a list
        for i_comp in range(self.num_comps):
            if self.cframe.comp_info_cI[i_comp]['mod_used'] in ['powerlaw', 'bending-powerlaw']:
                if isinstance(self.cframe.comp_info_cI[i_comp]['truncation'], (tuple, dict)): self.cframe.comp_info_cI[i_comp]['truncation'] = [self.cframe.comp_info_cI[i_comp]['truncation']]
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'recombination':
                if isinstance(self.cframe.comp_info_cI[i_comp]['H_series'], (str, int)): self.cframe.comp_info_cI[i_comp]['H_series'] = [self.cframe.comp_info_cI[i_comp]['H_series']]

        # check alternative info
        for i_comp in range(self.num_comps):
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'recombination':
                # translate greek notations to number of lower-level number
                H_series_dict = {'balmer': 2, 'paschen': 3, 'brackett': 4, 'pfund': 5, 'humphreys': 6}
                self.cframe.comp_info_cI[i_comp]['H_series'] = [H_series_dict[casefold(series)] if casefold(series) in H_series_dict else series for series in self.cframe.comp_info_cI[i_comp]['H_series']]
                self.cframe.comp_info_cI[i_comp]['H_series'] = list(dict.fromkeys( self.cframe.comp_info_cI[i_comp]['H_series'] )) # remove duplicate

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

    def powerlaw_func(self, wavelength, wave_norm=3000, alpha_lambda1=None, alpha_lambda2=None, curvature=None, bending=False, truncation=None):
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

    def blackbody_func(self, wavelength, log_tem=None, if_norm=True, wave_norm=3000):
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

        for item in ['file', 'file_path']:
            if item in self.cframe.mod_info_I: iron_file = self.cframe.mod_info_I[item]
        iron_lib = fits.open(iron_file)

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
        self.iron_dict['wave_segments'] = np.array([[1000,2150], [2150,2650], [2650,3020], [3020,4000], [4000,4800], [4800,5600], [5600,6800], [6800,7600]])
        self.iron_dict['init_flux_ew'] = []
        self.iron_dict['init_flux_norm_e']  = []
        self.iron_dict['orig_flux_ew'] = []
        self.iron_dict['orig_flux_norm_e']  = []
        for wave_range in self.iron_dict['wave_segments']:
            tmp_init_flux_w = np.zeros_like(self.iron_dict['init_wave_w'])
            mask_w = (self.iron_dict['init_wave_w'] >= wave_range[0]) & (self.iron_dict['init_wave_w'] < wave_range[1])
            tmp_init_flux_w[mask_w] = copy(self.iron_dict['init_flux_w'][mask_w])
            tmp_init_flux_norm = np.trapezoid(tmp_init_flux_w, x=self.iron_dict['init_wave_w'])
            self.iron_dict['init_flux_ew'].append(tmp_init_flux_w)
            self.iron_dict['init_flux_norm_e'].append(tmp_init_flux_norm)

            tmp_orig_flux_w = np.zeros_like(self.iron_dict['orig_wave_w'])
            mask_w = (self.iron_dict['orig_wave_w'] >= wave_range[0]) & (self.iron_dict['orig_wave_w'] < wave_range[1])
            tmp_orig_flux_w[mask_w] = copy(self.iron_dict['orig_flux_w'][mask_w])
            tmp_orig_flux_norm = np.trapezoid(tmp_orig_flux_w, x=self.iron_dict['orig_wave_w'])
            if tmp_orig_flux_norm / tmp_init_flux_norm < 0.05: 
                tmp_orig_flux_w *= 0 # avoid using poorly covered range
                tmp_orig_flux_norm *= 0 
            self.iron_dict['orig_flux_ew'].append(tmp_orig_flux_w)
            self.iron_dict['orig_flux_norm_e'].append(tmp_orig_flux_norm)

        self.iron_dict['init_flux_ew'] = np.array(self.iron_dict['init_flux_ew'])
        self.iron_dict['init_flux_norm_e'] = np.array(self.iron_dict['init_flux_norm_e'])
        self.iron_dict['orig_flux_ew'] = np.array(self.iron_dict['orig_flux_ew'])
        self.iron_dict['orig_flux_norm_e'] = np.array(self.iron_dict['orig_flux_norm_e'])

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
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'powerlaw':
                alpha_lambda = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['alpha_lambda']]
                pl = self.powerlaw_func(orig_wave_w, alpha_lambda1=alpha_lambda, bending=False)
                orig_flux_int_ew = pl[None,:] # convert to (1,w) format
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'bending-powerlaw':
                alpha_lambda1 = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['alpha_lambda1']]
                alpha_lambda2 = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['alpha_lambda2']]
                wave_turn     = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['wave_turn']]
                curvature     = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['curvature']]
                pl = self.powerlaw_func(orig_wave_w, wave_norm=wave_turn, alpha_lambda1=alpha_lambda1, alpha_lambda2=alpha_lambda2, curvature=curvature, bending=True)
                orig_flux_int_ew = pl[None,:] # convert to (1,w) format
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'blackbody':
                log_tem  = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['log_tem']]
                bb = self.blackbody_func(orig_wave_w, log_tem=log_tem)
                orig_flux_int_ew = bb[None,:] # convert to (1,w) format
            if self.cframe.comp_info_cI[i_comp]['mod_used'] =='recombination':
                log_e_tem  = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['log_e_tem']]
                log_tau_be = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['log_tau_be']]
                rec = self.recombination_func(orig_wave_w, log_e_tem=log_e_tem, log_tau_be=log_tau_be, H_series=self.cframe.comp_info_cI[i_comp]['H_series'])
                orig_flux_int_ew = rec[None,:] # convert to (1,w) format
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'iron':
                if if_full_range:
                    orig_wave_w = copy(self.iron_dict['init_wave_w'])
                    # do not convolve in this case
                    if_convolve = False 
                    if self.cframe.comp_info_cI[i_comp]['segments']:
                        iron = self.iron_dict['init_flux_ew']
                        orig_flux_int_ew = copy(iron)
                    else:
                        iron = self.iron_dict['init_flux_w']
                        orig_flux_int_ew = iron[None,:] # convert to (1,w) format
                else:
                    if self.cframe.comp_info_cI[i_comp]['segments']:
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
            if if_convolve & ('fwhm' in self.cframe.par_index_cP[i_comp]) & (self.R_inst_rw is not None) & (conv_nbin is not None) & (self.cframe.comp_info_cI[i_comp]['mod_used'] == 'iron'):
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
        value_names_additive = [] # ['flux_3000', 'flux_5100', 'flux_wavenorm']
        ret_names_additive = None
        value_names_C = {}
        for (i_comp, comp_name) in enumerate(comp_name_c):
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'powerlaw':       
                value_names_C[comp_name] = value_names_additive + [] # ['log_lamLlam_3000', 'log_lamLlam_5100', 'log_lamLlam_wavenorm']
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'bending-powerlaw':       
                value_names_C[comp_name] = value_names_additive + ['flux_waveturn'] # 'log_lamLlam_3000', 'log_lamLlam_5100', 'log_lamLlam_wavenorm', 'log_lamLlam_waveturn', 
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'blackbody':
                value_names_C[comp_name] = value_names_additive + ['log_intLum_bol'] # 'log_lamLlam_3000', 'log_lamLlam_5100', 'log_lamLlam_wavenorm', 
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'recombination':
                value_names_C[comp_name] = value_names_additive + ['log_intLum_bol']
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'iron':      
                value_names_C[comp_name] = value_names_additive + ['log_intLum_uv', 'log_intLum_opt']

            if self.cframe.comp_info_cI[i_comp]['ret_value_formats'] is None: continue
            ret_names = []
            for i_ret in range(len(self.cframe.comp_info_cI[i_comp]['ret_value_formats'])):
                tmp_dict = self.cframe.comp_info_cI[i_comp]['ret_value_formats'][i_ret]
                if 'wave_center' in tmp_dict:
                    wave_name = f"{tmp_dict['wave_center']}{tmp_dict['wave_unit']}"
                else:
                    wave_name = f"{tmp_dict['wave_min']}_{tmp_dict['wave_max']}{tmp_dict['wave_unit']}"
                ret_name = f"log_{tmp_dict['value_form']}_{wave_name}_{tmp_dict['value_state']}_u_{tmp_dict['value_unit']}"
                ret_names.append(ret_name)
            value_names_C[comp_name] += ret_names
            if ret_names_additive is None: 
                ret_names_additive = ret_names
            else:
                ret_names_additive = [ret_name for ret_name in ret_names_additive if ret_name in value_names_C[comp_name]]
        value_names_additive += ret_names_additive

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

                # tmp_spec_w = self.fframe.output_MC[self.mod_name][comp_name]['spec_lw'][i_loop, :]
                # mask_norm_w = np.abs(self.fframe.spec['wave_w']/(1+rev_redshift) - 3000) < 25 # for observed flux at rest 3000 angstrom
                # if mask_norm_w.sum() > 0:
                #     output_C[comp_name]['value_Vl']['flux_3000'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                #     output_C['sum']['value_Vl']['flux_3000'][i_loop] += tmp_spec_w[mask_norm_w].mean()
                # mask_norm_w = np.abs(self.fframe.spec['wave_w']/(1+rev_redshift) - 5100) < 25 # for observed flux at rest 5100 angstrom
                # if mask_norm_w.sum() > 0:
                #     output_C[comp_name]['value_Vl']['flux_5100'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                #     output_C['sum']['value_Vl']['flux_5100'][i_loop] += tmp_spec_w[mask_norm_w].mean()
                # mask_norm_w = np.abs(self.fframe.spec['wave_w']/(1+rev_redshift) - self.cframe.mod_info_I['w_norm']) < self.cframe.mod_info_I['dw_norm'] # for observed flux at user given wavenorm
                # if mask_norm_w.sum() > 0:
                #     output_C[comp_name]['value_Vl']['flux_wavenorm'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                #     output_C['sum']['value_Vl']['flux_wavenorm'][i_loop] += tmp_spec_w[mask_norm_w].mean()

                lum_area = 4*np.pi * cosmo.luminosity_distance(rev_redshift).to('cm').value**2 # in cm2
                unitconv = lum_area * self.spec_flux_scale * u.Unit('erg/s').to('L_sun') # convert intrinsic flux in erg/s/cm2/A to Lum in Lsun/A
 
                # if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'powerlaw':
                #     alpha_lambda = par_p[self.cframe.par_index_cP[i_comp]['alpha_lambda']]
                #     for (wave, wave_str) in zip([3000, 5100, self.cframe.mod_info_I['w_norm']], ['3000', '5100', 'wavenorm']):
                #         flux_wave = coeff_e[0] * self.powerlaw_func(wave, alpha_lambda1=alpha_lambda, bending=False)
                #         lamLlam_wave = flux_wave * unitconv * wave
                #         output_C[comp_name]['value_Vl']['log_lamLlam_'+wave_str][i_loop] = np.log10(lamLlam_wave)

                if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'bending-powerlaw':
                    # alpha_lambda1 = par_p[self.cframe.par_index_cP[i_comp]['alpha_lambda1']]
                    # alpha_lambda2 = par_p[self.cframe.par_index_cP[i_comp]['alpha_lambda2']]
                    # wave_turn     = par_p[self.cframe.par_index_cP[i_comp]['wave_turn']]
                    # curvature     = par_p[self.cframe.par_index_cP[i_comp]['curvature']]
                    # for (wave, wave_str) in zip([3000, 5100, self.cframe.mod_info_I['w_norm'], wave_turn], ['3000', '5100', 'wavenorm', 'waveturn']):
                    #     flux_wave = coeff_e[0] * self.powerlaw_func(wave, wave_norm=wave_turn, alpha_lambda1=alpha_lambda1, alpha_lambda2=alpha_lambda2, curvature=curvature, bending=True)
                    #     lamLlam_wave = flux_wave * unitconv * wave
                    #     output_C[comp_name]['value_Vl']['log_lamLlam_'+wave_str][i_loop] = np.log10(lamLlam_wave)
                    # mask_norm_w = np.abs(self.fframe.spec['wave_w']/(1+rev_redshift) - wave_turn) < self.cframe.mod_info_I['dw_norm'] 
                    # if mask_norm_w.sum() > 0:
                    #     output_C[comp_name]['value_Vl']['flux_waveturn'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                    output_C[comp_name]['value_Vl']['flux_waveturn'][i_loop] = coeff_e[0]

                if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'blackbody':
                    log_tem  = par_p[self.cframe.par_index_cP[i_comp]['log_tem']]
                    # for (wave, wave_str) in zip([3000, 5100, self.cframe.mod_info_I['w_norm']], ['3000', '5100', 'wavenorm']):
                    #     flux_wave = coeff_e[0] * self.blackbody_func(wave, log_tem=log_tem, )
                    #     lamLlam_wave = flux_wave * unitconv * wave
                    #     output_C[comp_name]['value_Vl']['log_lamLlam_'+wave_str][i_loop] = np.log10(lamLlam_wave)
                    tmp_wave_w = np.logspace(np.log10(912), 7.5, num=10000) # till 10 K
                    tmp_bb_w = self.blackbody_func(tmp_wave_w, log_tem=log_tem)
                    output_C[comp_name]['value_Vl']['log_intLum_bol'][i_loop] = np.log10(coeff_e[0] * unitconv * np.trapezoid(tmp_bb_w, x=tmp_wave_w))

                if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'recombination':
                    log_e_tem  = par_p[self.cframe.par_index_cP[i_comp]['log_e_tem']]
                    log_tau_be = par_p[self.cframe.par_index_cP[i_comp]['log_tau_be']]
                    tmp_wave_w = np.logspace(np.log10(912), 7.5, num=10000) # till 10 K
                    tmp_rec_w = self.recombination_func(tmp_wave_w, log_e_tem=log_e_tem, log_tau_be=log_tau_be, H_series=self.cframe.comp_info_cI[i_comp]['H_series'])
                    output_C[comp_name]['value_Vl']['log_intLum_bol'][i_loop] = np.log10(coeff_e[0] * unitconv * np.trapezoid(tmp_rec_w, x=tmp_wave_w))

                if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'iron':
                    if self.cframe.comp_info_cI[i_comp]['segments']:
                        coeff_uv  = (coeff_e[1:4] * self.iron_dict['init_flux_norm_e'][1:4])[coeff_e[1:4] > 0].sum()
                        coeff_opt = (coeff_e[4:6] * self.iron_dict['init_flux_norm_e'][4:6])[coeff_e[4:6] > 0].sum()
                    else:
                        coeff_uv  = coeff_e[0]
                        coeff_opt = coeff_e[0] * self.iron_dict['flux_opt_uv_ratio']
                    output_C[comp_name]['value_Vl']['log_intLum_uv' ][i_loop] = np.log10(coeff_uv  * unitconv)
                    output_C[comp_name]['value_Vl']['log_intLum_opt'][i_loop] = np.log10(coeff_opt * unitconv)

                # calculate requested flux/Lum in given wavelength ranges
                if self.cframe.comp_info_cI[i_comp]['ret_value_formats'] is None: continue
                tmp_coeff_e = best_coeff_le[i_loop, i_coeffs_0_of_mod:i_coeffs_1_of_mod][i_coeffs_0_of_comp_in_mod:i_coeffs_1_of_comp_in_mod]
                for i_ret in range(len(self.cframe.comp_info_cI[i_comp]['ret_value_formats'])):
                    tmp_dict = self.cframe.comp_info_cI[i_comp]['ret_value_formats'][i_ret]

                    if 'wave_center' in tmp_dict:
                        wave_0, wave_1 = tmp_dict['wave_center'] - tmp_dict['wave_width'], tmp_dict['wave_center'] + tmp_dict['wave_width']
                        wave_name = f"{tmp_dict['wave_center']}{tmp_dict['wave_unit']}"
                    else:
                        wave_0, wave_1 = tmp_dict['wave_min'], tmp_dict['wave_max']
                        wave_name = f"{tmp_dict['wave_min']}_{tmp_dict['wave_max']}{tmp_dict['wave_unit']}"
                    wave_ratio = u.Unit(tmp_dict['wave_unit']).to('angstrom')
                    if tmp_dict['wave_frame'] == 'rest': wave_ratio *= (1+rev_redshift) # rest wave to obs wave
                    tmp_wave_w = np.logspace(np.log10(wave_0*wave_ratio), np.log10(wave_1*wave_ratio), num=1000) # obs frame grid

                    if tmp_dict['value_state'] in ['intrinsic','absorbed']:
                        tmp_flux_ew = self.create_models(tmp_wave_w, best_par_lp[i_loop, i_pars_0_of_mod:i_pars_1_of_mod], components=comp_name, 
                                                         if_dust_ext=False, if_redshift=True, if_full_range=True, dpix_resample=300) # flux in obs frame
                        intrinsic_flux_w = tmp_coeff_e @ tmp_flux_ew
                    if tmp_dict['value_state'] in ['observed','absorbed']:
                        tmp_flux_ew = self.create_models(tmp_wave_w, best_par_lp[i_loop, i_pars_0_of_mod:i_pars_1_of_mod], components=comp_name, 
                                                         if_dust_ext=True,  if_redshift=True, if_full_range=True, dpix_resample=300) # flux in obs frame
                        observed_flux_w = tmp_coeff_e @ tmp_flux_ew
                    if tmp_dict['value_state'] == 'intrinsic': tmp_flux_w = intrinsic_flux_w
                    if tmp_dict['value_state'] == 'observed' : tmp_flux_w = observed_flux_w
                    if tmp_dict['value_state'] == 'absorbed' : tmp_flux_w = intrinsic_flux_w - observed_flux_w

                    tmp_Flam = tmp_flux_w.mean()
                    tmp_lamFlam = tmp_flux_w.mean() * tmp_wave_w.mean()
                    tmp_intFlux = np.trapezoid(tmp_flux_w, x=tmp_wave_w)

                    if tmp_dict['value_form'] ==     'Flam'          : ret_value = tmp_Flam    * u.Unit('erg s-1 cm-2 angstrom-1').to(tmp_dict['value_unit'])
                    if tmp_dict['value_form'] in ['lamFlam', 'nuFnu']: ret_value = tmp_lamFlam * u.Unit('erg s-1 cm-2').to(tmp_dict['value_unit'])
                    if tmp_dict['value_form'] ==  'intFlux'          : ret_value = tmp_intFlux * u.Unit('erg s-1 cm-2').to(tmp_dict['value_unit'])

                    if tmp_dict['value_form'] ==     'Llam'          : ret_value = tmp_Flam    * lum_area * u.Unit('erg s-1 angstrom-1').to(tmp_dict['value_unit'])
                    if tmp_dict['value_form'] in ['lamLlam', 'nuLnu']: ret_value = tmp_lamFlam * lum_area * u.Unit('erg s-1').to(tmp_dict['value_unit'])
                    if tmp_dict['value_form'] ==  'intLum'           : ret_value = tmp_intFlux * lum_area * u.Unit('erg s-1').to(tmp_dict['value_unit'])

                    if tmp_dict['value_form'] == 'Fnu' : ret_value = (tmp_Flam       * u.Unit('erg s-1 cm-2 angstrom-1') * (tmp_wave_w.mean()*u.angstrom)**2 / const.c).to(tmp_dict['value_unit']).value
                    if tmp_dict['value_form'] == 'Lnu' : ret_value = (tmp_Flam * lum_area * u.Unit('erg s-1 angstrom-1') * (tmp_wave_w.mean()*u.angstrom)**2 / const.c).to(tmp_dict['value_unit']).value

                    ret_value *= self.spec_flux_scale
                    ret_name = f"log_{tmp_dict['value_form']}_{wave_name}_{tmp_dict['value_state']}_u_{tmp_dict['value_unit']}"
                    output_C[comp_name]['value_Vl'][ret_name][i_loop] = np.log10(ret_value)
                    if ret_name in output_C['sum']['value_Vl']: output_C['sum']['value_Vl'][ret_name][i_loop] += ret_value

        for value_name in output_C['sum']['value_Vl']:
            if (value_name[:8] in ['log_Flam', 'log_Fnu_']) | (value_name[:11] in ['log_lamFlam', 'log_intFlux', 'log_lamLlam', 'log_intLum_']): 
                output_C['sum']['value_Vl'][value_name] = np.log10(output_C['sum']['value_Vl'][value_name])

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

            print_name_CV[comp_name]['voff'] = 'Velocity shift in relative to z_sys (km s-1)'
            print_name_CV[comp_name]['fwhm'] = 'Velocity FWHM (km s-1)'
            print_name_CV[comp_name]['Av'] = 'Extinction (Av)'
            # print_name_CV[comp_name]['log_lamLlam_3000'] = f"λL3000 (rest,intrinsic) "
            # print_name_CV[comp_name]['log_lamLlam_5100'] = f"λL5100 (rest,intrinsic) "
            # print_name_CV[comp_name]['log_lamLlam_wavenorm'] = f"λL{self.cframe.mod_info_I['w_norm']:.0f} (rest,intrinsic) "
            # print_name_CV[comp_name]['flux_3000'] = f"F3000 (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)"
            # print_name_CV[comp_name]['flux_5100'] = f"F5100 (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)"
            # print_name_CV[comp_name]['flux_wavenorm'] = f"F{self.cframe.mod_info_I['w_norm']:.0f} (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)"
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'powerlaw':
                print_name_CV[comp_name]['alpha_lambda'] = 'Powerlaw α_λ'
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'bending-powerlaw':
                print_name_CV[comp_name]['curvature'] = 'Curvature'
                print_name_CV[comp_name]['wave_turn'] = 'Turning wavelength (rest, Å)'
                wave_turn = self.output_C[[*self.output_C][i_comp]]['value_Vl']['wave_turn'][0]
                print_name_CV[comp_name]['alpha_lambda1'] = f"Powerlaw α1_λ (λ < {wave_turn:.0f})"
                print_name_CV[comp_name]['alpha_lambda2'] = f"Powerlaw α2_λ (λ > {wave_turn:.0f})"
                print_name_CV[comp_name]['flux_waveturn'] = f"F{wave_turn:.0f} (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)"
                print_name_CV[comp_name]['log_lamLlam_waveturn'] = f"λL{wave_turn:.0f} (rest,intrinsic) "
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'blackbody':
                print_name_CV[comp_name]['log_tem'] = 'Blackbody temperature (log K)'
                print_name_CV[comp_name]['log_intLum_bol'] = f"Blackbody bolometric lum. (log L☉)"
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'recombination':
                print_name_CV[comp_name]['log_e_tem'] = 'Recombination cont. e- temperature (log K)'
                print_name_CV[comp_name]['log_tau_be'] = 'Recombination cont. optical depth at 3646 Å (log τ)'
                print_name_CV[comp_name]['log_intLum_bol'] = f"Recombination cont. bolometric lum. (log L☉)"
            if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'iron':
                print_name_CV[comp_name]['log_intLum_uv']  = 'Intrinsic lum. (integrated, log L☉) at rest UV (2150-4000 Å)'
                print_name_CV[comp_name]['log_intLum_opt'] = 'Intrinsic lum. (integrated, log L☉) at rest optical (4000-5600 Å)'

            for i_ret in range(len(self.cframe.comp_info_cI[i_comp]['ret_value_formats'])):
                tmp_dict = self.cframe.comp_info_cI[i_comp]['ret_value_formats'][i_ret]

                if 'wave_center' in tmp_dict:
                    wave_name = f"{tmp_dict['wave_center']}{tmp_dict['wave_unit']}"
                else:
                    wave_name = f"{tmp_dict['wave_min']}_{tmp_dict['wave_max']}{tmp_dict['wave_unit']}"
                ret_name = f"log_{tmp_dict['value_form']}_{wave_name}_{tmp_dict['value_state']}_u_{tmp_dict['value_unit']}"

                if tmp_dict['value_state'] == 'absorbed' : tmp_dict['value_state'] = 'dust-'+tmp_dict['value_state']
                print_name_CV[comp_name][ret_name] = f"{tmp_dict['value_state'].capitalize()} "

                tmp_dict['value_unit'] = tmp_dict['value_unit'].replace('L_sun', 'L☉').replace('angstrom', 'Å').replace('Angstrom', 'Å').replace('um', 'µm').replace('micron', 'µm')
                if tmp_dict['value_form'] ==    'Flam': print_name_CV[comp_name][ret_name] += f"flux density (Fλ, log {tmp_dict['value_unit']})"
                if tmp_dict['value_form'] ==    'Llam': print_name_CV[comp_name][ret_name] += f"lum. density (Lλ, log {tmp_dict['value_unit']})"
                if tmp_dict['value_form'] ==    'Fnu' : print_name_CV[comp_name][ret_name] += f"flux density (Fν, log {tmp_dict['value_unit']})"
                if tmp_dict['value_form'] ==    'Lnu' : print_name_CV[comp_name][ret_name] += f"lum. density (Lν, log {tmp_dict['value_unit']})"
                if tmp_dict['value_form'] == 'lamFlam': print_name_CV[comp_name][ret_name] += f"flux (λFλ, log {tmp_dict['value_unit']})"
                if tmp_dict['value_form'] == 'lamLlam': print_name_CV[comp_name][ret_name] += f"lum. (λLλ, log {tmp_dict['value_unit']})"
                if tmp_dict['value_form'] ==  'nuFnu' : print_name_CV[comp_name][ret_name] += f"flux (νFν, log {tmp_dict['value_unit']})"
                if tmp_dict['value_form'] ==  'nuLnu' : print_name_CV[comp_name][ret_name] += f"lum. (νLν, log {tmp_dict['value_unit']})"
                if tmp_dict['value_form'] == 'intFlux': print_name_CV[comp_name][ret_name] += f"flux (integrated, log {tmp_dict['value_unit']})"
                if tmp_dict['value_form'] == 'intLum' : print_name_CV[comp_name][ret_name] += f"lum. (integrated, log {tmp_dict['value_unit']})"

                tmp_dict['wave_unit'] = tmp_dict['wave_unit'].replace('angstrom', 'Å').replace('Angstrom', 'Å').replace('um', 'µm').replace('micron', 'µm')
                if tmp_dict['wave_frame'] == 'obs' : tmp_dict['wave_frame'] += '.'
                if 'wave_center' in tmp_dict:
                    print_name_CV[comp_name][ret_name] += f" at {tmp_dict['wave_frame']} {tmp_dict['wave_center']} {tmp_dict['wave_unit']}"
                else:
                    print_name_CV[comp_name][ret_name] += f" at {tmp_dict['wave_frame']} {tmp_dict['wave_min']}-{tmp_dict['wave_max']} {tmp_dict['wave_unit']}"
        print_name_CV['sum'] = {}
        for value_name in self.output_C['sum']['value_Vl']:
            print_name_CV['sum'][value_name] = copy(print_name_CV[self.comp_name_c[0]][value_name])

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
                value_names += ['voff', 'fwhm', 'Av']
                if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'powerlaw':
                    value_names += ['alpha_lambda']
                if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'bending-powerlaw':
                    value_names += ['curvature', 'wave_turn', 'alpha_lambda1', 'alpha_lambda2', 'flux_waveturn', 'log_lamLlam_waveturn']
                if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'blackbody':
                    value_names += ['log_tem', 'log_intLum_bol']
                if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'recombination':
                    value_names += ['log_e_tem', 'log_tau_be', 'log_intLum_bol']
                if self.cframe.comp_info_cI[i_comp]['mod_used'] == 'iron':
                    value_names += ['log_intLum_uv', 'log_intLum_opt']
                # if self.cframe.comp_info_cI[i_comp]['mod_used'] in ['powerlaw', 'bending-powerlaw', 'blackbody']:
                #     value_names += ['log_lamLlam_3000', 'log_lamLlam_5100']
                #     if self.cframe.mod_info_I['w_norm'] not in [3000,5100]: value_names += ['log_lamLlam_wavenorm']
            elif self.cframe.num_comps >= 2: # print sum only if using >= 2 comps
                print_log(f"# Best-fit properties of the sum of all AGN components.", log)
            else: 
                continue
            # value_names += ['flux_3000', 'flux_5100']
            # if self.cframe.mod_info_I['w_norm'] not in [3000,5100]: value_names += ['flux_wavenorm']

            value_names += [value_name for value_name in [*value_Vl] if value_name not in value_names]

            if i_comp < self.cframe.num_comps:
                if self.cframe.comp_info_cI[i_comp]['mod_used'] != 'iron':
                    value_names.remove('voff')
                    value_names.remove('fwhm')
                    print_log(f"[Note] velocity shift (i.e., redshift) and FWHM are tied to other components following the input model_config.", log)

            for value_name in value_names:
                msg += '| ' + print_name_CV[comp_name][value_name] + f" = {value_Vl[value_name][mask_l].mean():10.4f}" + f" +/- {value_Vl[value_name].std():<10.4f}|\n"
            msg = msg[:-1] # remove the last \n
            bar = '=' * len(msg.split('\n')[-1])
            print_log(bar, log)
            print_log(msg, log)
            print_log(bar, log)
            print_log('', log)

