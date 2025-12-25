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
from ..auxiliaries.extinct_laws import ExtLaw

# alternative component names after casefold
powerlaw_names = ['powerlaw', 'pl']
bending_powerlaw_names = ['bending_powerlaw', 'bending-powerlaw', 'bending powerlaw', 'bending_pl', 'bending-pl', 'bending pl']
blackbody_names = ['bb', 'blackbody', 'black body', 'black_body']
bac_names = ['bac', 'balmer cont.', 'balmer_cont.', 'balmer continuum', 'balmer_continuum']
iron_names = ['iron', 'fe ii', 'feii']

class AGNFrame(object):
    def __init__(self, fframe=None, config=None, filename=None, 
                 v0_redshift=None, R_inst_rw=None, 
                 w_min=None, w_max=None, w_norm=5100, dw_norm=25, 
                 Rratio_mod=None, dw_fwhm_dsp=None, dw_pix_inst=None, 
                 verbose=True, log_message=[]):

        self.fframe = fframe
        self.config = config
        self.filename = filename
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
        self.num_comps = self.cframe.num_comps

        if self.verbose:
            print_log(f"AGN UV/optical continuum components: {[self.cframe.info_c[i_comp]['mod_used'].tolist() for i_comp in range(self.num_comps)]}", self.log_message)
        for i_comp in range(self.num_comps): 
            self.cframe.info_c[i_comp]['mod_used'] = casefold(self.cframe.info_c[i_comp]['mod_used']) # make names case insensitive

        ############################################################
        # to be compatible with old version <= 2.2.4
        if len(self.cframe.par_index_cp[0]) == 0:
            for i_comp in range(self.num_comps):
                if np.isin(powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    self.cframe.par_name_cp[i_comp, :4] = ['voff', 'fwhm', 'Av', 'alpha_lambda']
                    self.cframe.par_index_cp[i_comp] = {'voff': 0, 'fwhm': 1, 'Av': 2, 'alpha_lambda': 3}
                if np.isin(bending_powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    self.cframe.par_name_cp[i_comp, :6] = ['voff', 'fwhm', 'Av', 'alpha_lambda1', 'alpha_lambda2', 'wave_turn', 'curvature']
                    self.cframe.par_index_cp[i_comp] = {'voff': 0, 'fwhm': 1, 'Av': 2, 'alpha_lambda1': 3, 'alpha_lambda2': 4, 'wave_turn': 5, 'curvature': 6}
                if np.isin(blackbody_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    self.cframe.par_name_cp[i_comp, :5] = ['voff', 'fwhm', 'Av', 'log_tem',]
                    self.cframe.par_index_cp[i_comp] = {'voff': 0, 'fwhm': 1, 'Av': 2, 'log_tem': 3}
                if np.isin(bac_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    self.cframe.par_name_cp[i_comp, :5] = ['voff', 'fwhm', 'Av', 'log_e_tem', 'log_tau_be']
                    self.cframe.par_index_cp[i_comp] = {'voff': 0, 'fwhm': 1, 'Av': 2, 'log_e_tem': 3, 'log_tau_be': 4}
                if np.isin(iron_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    self.cframe.par_name_cp[i_comp, :3] = ['voff', 'fwhm', 'Av']
                    self.cframe.par_index_cp[i_comp] = {'voff': 0, 'fwhm': 1, 'Av': 2}
        ############################################################

        self.num_coeffs_c = np.zeros(self.num_comps, dtype='int')
        for i_comp in range(self.num_comps):
            if np.isin(powerlaw_names+bending_powerlaw_names+blackbody_names+bac_names, self.cframe.info_c[i_comp]['mod_used']).any():
                self.num_coeffs_c[i_comp] = 1 # one independent element per component
            if np.isin(iron_names, self.cframe.info_c[i_comp]['mod_used']).any():
                if not ('segment' in [*self.cframe.info_c[i_comp]]): self.cframe.info_c[i_comp]['segment'] = False
                if self.cframe.info_c[i_comp]['segment']: 
                    self.num_coeffs_c[i_comp] = 8 # 8 independent segments
                else:
                    self.num_coeffs_c[i_comp] = 1 # 1 independent element
        self.num_coeffs = self.num_coeffs_c.sum()

        # currently do not consider negative spectra 
        self.mask_absorption_e = np.zeros((self.num_coeffs), dtype='bool')
        
        # set original wavelength grid
        orig_wave_logbin = 0.05
        orig_wave_num = int(np.round(np.log10(w_max/w_min) / orig_wave_logbin))
        self.orig_wave_w = np.logspace(np.log10(w_min), np.log10(w_max), num=orig_wave_num)

        # set iron template
        if np.isin(iron_names, [self.cframe.info_c[i_comp]['mod_used'] for i_comp in range(self.num_comps)]).any(): 
            self.read_iron()

        self.plot_style_c = {}
        self.plot_style_c['sum'] = {'color': 'C3', 'alpha': 1, 'linestyle': '-', 'linewidth': 1.5}
        i_red, i_yellow, i_green, i_purple = 0, 0, 0, 0
        for i_comp in range(self.num_comps):
            if np.isin(powerlaw_names+bending_powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():
                self.plot_style_c[str(self.cframe.comp_c[i_comp])] = {'color': 'None', 'alpha': 0.5, 'linestyle': '--', 'linewidth': 1}
                self.plot_style_c[self.cframe.comp_c[i_comp]]['color'] = str(np.take(color_list_dict['purple'], i_purple, mode="wrap"))
                i_purple += 1
            if np.isin(blackbody_names, self.cframe.info_c[i_comp]['mod_used']).any():
                self.plot_style_c[str(self.cframe.comp_c[i_comp])] = {'color': 'None', 'alpha': 0.5, 'linestyle': '--', 'linewidth': 1}
                self.plot_style_c[self.cframe.comp_c[i_comp]]['color'] = str(np.take(color_list_dict['red'], i_red, mode="wrap"))
                i_red += 1
            if np.isin(bac_names, self.cframe.info_c[i_comp]['mod_used']).any():
                self.plot_style_c[str(self.cframe.comp_c[i_comp])] = {'color': 'None', 'alpha': 0.5, 'linestyle': '--', 'linewidth': 1}
                self.plot_style_c[self.cframe.comp_c[i_comp]]['color'] = str(np.take(color_list_dict['green'], i_green, mode="wrap"))
                i_green += 1
            if np.isin(iron_names, self.cframe.info_c[i_comp]['mod_used']).any():
                self.plot_style_c[str(self.cframe.comp_c[i_comp])] = {'color': 'None', 'alpha': 0.5, 'linestyle': '-', 'linewidth': 0.75}
                self.plot_style_c[self.cframe.comp_c[i_comp]]['color'] = str(np.take(color_list_dict['yellow'], i_yellow, mode="wrap"))
                i_yellow += 1

    ##############################

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

    def powerlaw_func(self, wavelength, wave_norm=None, alpha_lambda1=None, alpha_lambda2=None, curvature=None, bending=False):
        # normalized to given flux density (e.g.,the same unit of obs) at rest wave_norm before extinct

        if isinstance(wavelength, (int,float)): wavelength = np.array([wavelength])
        pl = self.bending_powerlaw(wavelength, wave_norm, 1, alpha_lambda1, alpha_lambda2, curvature, bending)

        # set cutting index in longer and shorter wavelength ranges
        # https://sites.google.com/site/skirtorus/sed-library, Primary source: accretion disk
        alpha_long = -3-1; wave_long = 5e4
        alpha_short1 = 0-1; wave_short1 = 0.1e4
        alpha_short2 = 1.2-1; wave_short2 = 0.01e4

        mask_w = wavelength > wave_long
        if mask_w.sum() > 0: 
            pl_long = self.bending_powerlaw(wave_long, wave_norm, 1, alpha_lambda1, alpha_lambda2, curvature, bending)
            pl[mask_w] = self.simple_powerlaw(wavelength[mask_w], wave_long, pl_long, alpha_long)

        mask_w = wavelength < wave_short1
        if mask_w.sum() > 0: 
            pl_short1 = self.bending_powerlaw(wave_short1, wave_norm, 1, alpha_lambda1, alpha_lambda2, curvature, bending)
            pl[mask_w] = self.simple_powerlaw(wavelength[mask_w], wave_short1, pl_short1, alpha_short1)

        mask_w = wavelength < wave_short2
        if mask_w.sum() > 0: 
            pl_short1 = self.bending_powerlaw(wave_short1, wave_norm, 1, alpha_lambda1, alpha_lambda2, curvature, bending)
            pl_short2 = self.simple_powerlaw(wave_short2, wave_short1, pl_short1, alpha_short1)
            pl[mask_w] = self.simple_powerlaw(wavelength[mask_w], wave_short2, pl_short2, alpha_short2)

        return pl

    def blackbody_func(self, wavelength, log_tem=None, if_norm=True, wave_norm=None):
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

    def bac_func(self, wavelength, log_e_tem=None, log_tau_be=None):
        # parameters: electron temperature (K), optical depth at balmer edge (3646)
        # normalize at rest 3000 AA (default)
        wave_norm = 3000
        balmer_edge = 3646.0

        def get_bac(wavelength):
            planck_flux_w = self.blackbody_func(wavelength, log_tem=log_e_tem, if_norm=False)
            # calculate the optical depth at each wavelength
            # τ_λ = τ_BE * (λ_BE / λ)^3  (as in Grandi 1982)
            # the exponent can vary depending on the specific model
            optical_depth = 10.0**log_tau_be * (balmer_edge / wavelength)**3
            # return the Balmer continuum flux
            return planck_flux_w * (1 - np.exp(-optical_depth))

        bac_flux_w = get_bac(wavelength) / get_bac(wave_norm)
        bac_flux_w[wavelength >= balmer_edge] = 0

        return bac_flux_w

    def recombination_func(self, wavelength, log_e_tem=None, log_tau_be=None):
        # Hydrogen Radiative Recombination Continuum (free-bound)
        # parameters: electron temperature (K), optical depth at balmer edge (3646)
        # temperature range: ~3000--30000 K. lower: neutral H dominated; higher: free-free dominated

        # only consider Balmer, Paschen, Brackett, and Pfund series
        lv_n_series = [2,3,4,5]
        wave_norm = 3000 # normalize at rest 3000 AA (default)

        def get_rec(wavelength):
            if isinstance(wavelength, (int,float)): wavelength = np.array([wavelength])
            rec_w = np.zeros_like(wavelength, dtype=float)
            bb_w = self.blackbody_func(wavelength, log_tem=log_e_tem, if_norm=False)
            for lv_n in lv_n_series:
                wave_edge = lv_n**2 / 1.0973731568157e-3 # n**2/R_H, in AA
                if min(wavelength) > wave_edge: continue
                # assume the bound-free cross section at threshold scales prop to n**(-5)
                tau_edge = 10.0**log_tau_be * (2.0/lv_n)**5
                # calculate the optical depth at each wavelength, τ_λ = τ_BE * (λ_BE / λ)^3 (Grandi 1982)
                tau_w = tau_edge * (wave_edge / wavelength)**3
                tmp_rec_w = bb_w * (1 - np.exp(-tau_w))
                tmp_rec_w[wavelength > wave_edge] = 0.0
                rec_w += tmp_rec_w
            return rec_w

        ret_rec_w = get_rec(wavelength) / get_rec(wave_norm)

        return ret_rec_w

    def read_iron(self):
        # combined I Zw 1 Fe II (+ UV Fe III) template, convolving to fwhm = 1100 km/s
        # https://arxiv.org/pdf/astro-ph/0104320, Vestergaard andWilkes, 2001
        # https://arxiv.org/pdf/astro-ph/0312654, Veron-Cetty et al., 2003
        # https://arxiv.org/pdf/astro-ph/0606040, Tsuzuki et al., 2006

        iron_lib = fits.open(self.filename)
        iron_wave_w = iron_lib[0].data[0] # AA, in rest frame
        iron_flux_w = iron_lib[0].data[1] # erg/s/cm2/AA
        iron_dw_fwhm_w = iron_wave_w * 1100 / 299792.458

        # normalize models at given wavelength
        mask_norm_w = (iron_wave_w > 2150) & (iron_wave_w < 4000)
        flux_norm_uv = np.trapezoid(iron_flux_w[mask_norm_w], x=iron_wave_w[mask_norm_w])
        mask_norm_w = (iron_wave_w > 4000) & (iron_wave_w < 5600)
        flux_norm_opt = np.trapezoid(iron_flux_w[mask_norm_w], x=iron_wave_w[mask_norm_w])
        iron_flux_w /= flux_norm_uv
        self.iron_flux_opt_uv_ratio = flux_norm_opt / flux_norm_uv

        # determine the required model resolution and bin size (in AA) to downsample the model
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
                        print_log(f'Downsample pre-convolved AGN Fe II pesudo-continuum with bin width of {self.dw_dsp:.3f} AA in a min resolution of {self.dw_fwhm_dsp_w.min():.3f} AA', 
                                  self.log_message)
                    # before downsampling, smooth the model to avoid aliasing (like in ADC or digital signal reduction)
                    # set dw_fwhm_ref as the dispersion in the original model
                    iron_flux_w = convolve_var_width_fft(iron_wave_w, iron_flux_w, dw_fwhm_obj=self.dw_fwhm_dsp_w, dw_fwhm_ref=iron_dw_fwhm_w, num_bins=10, reset_edge=True)
                else:
                    if self.verbose: 
                        print_log(f'Downsample original AGN Fe II pesudo-continuum with bin width of {self.dw_dsp:.3f} AA in a min resolution of {self.dw_fwhm_dsp_w.min():.3f} AA', 
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
        self.iron_flux_w = np.interp(self.orig_wave_w, iron_wave_w, iron_flux_w, left=0, right=0)
        self.dw_fwhm_dsp_w = np.interp(self.orig_wave_w, iron_wave_w, self.dw_fwhm_dsp_w)

        # set segments
        iron_flux_ew = []; iron_flux_norm_e = []
        wave_ranges = [[1000,2150], [2150,2650], [2650,3020], [3020,4000], [4000,4800], [4800,5600], [5600,6800], [6800,7600]]
        for wave_range in wave_ranges:
            tmp_w = np.zeros_like(self.orig_wave_w)
            mask_w = (self.orig_wave_w >= wave_range[0]) & (self.orig_wave_w < wave_range[1])
            tmp_w[mask_w] = copy(self.iron_flux_w[mask_w])
            iron_flux_ew.append(tmp_w)
            iron_flux_norm_e.append(np.trapezoid(tmp_w, x=self.orig_wave_w))
        self.iron_flux_ew = np.array(iron_flux_ew)
        self.iron_flux_norm_e = np.array(iron_flux_norm_e)

    def iron_func(self, segment=False):
        # no special parameter needed
        if segment:
            return self.iron_flux_ew
        else:
            return self.iron_flux_w 

    ##############################

    def models_unitnorm_obsframe(self, obs_wave_w, input_pars, if_pars_flat=True, mask_lite_e=None, conv_nbin=None):
        if if_pars_flat: 
            par_cp = self.cframe.flat_to_arr(input_pars)
        else:
            par_cp = copy(input_pars)

        for i_comp in range(par_cp.shape[0]):
            # read and append intrinsic templates in rest frame
            if np.isin(powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():
                alpha_lambda = par_cp[i_comp, self.cframe.par_index_cp[i_comp]['alpha_lambda']]
                pl = self.powerlaw_func(self.orig_wave_w, wave_norm=self.w_norm, alpha_lambda1=alpha_lambda, alpha_lambda2=None, curvature=None, bending=False)
                orig_flux_int_ew = pl[None,:] # convert to (1,w) format
            if np.isin(bending_powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():
                alpha_lambda1 = par_cp[i_comp, self.cframe.par_index_cp[i_comp]['alpha_lambda1']]
                alpha_lambda2 = par_cp[i_comp, self.cframe.par_index_cp[i_comp]['alpha_lambda2']]
                wave_turn     = par_cp[i_comp, self.cframe.par_index_cp[i_comp]['wave_turn']]
                curvature     = par_cp[i_comp, self.cframe.par_index_cp[i_comp]['curvature']]
                pl = self.powerlaw_func(self.orig_wave_w, wave_norm=wave_turn, alpha_lambda1=alpha_lambda1, alpha_lambda2=alpha_lambda2, curvature=curvature, bending=True)
                orig_flux_int_ew = pl[None,:] # convert to (1,w) format
            if np.isin(blackbody_names, self.cframe.info_c[i_comp]['mod_used']).any():
                log_tem  = par_cp[i_comp, self.cframe.par_index_cp[i_comp]['log_tem']]
                bb = self.blackbody_func(self.orig_wave_w, log_tem=log_tem, wave_norm=self.w_norm)
                orig_flux_int_ew = bb[None,:] # convert to (1,w) format
            if np.isin(bac_names, self.cframe.info_c[i_comp]['mod_used']).any():
                log_e_tem  = par_cp[i_comp, self.cframe.par_index_cp[i_comp]['log_e_tem']]
                log_tau_be = par_cp[i_comp, self.cframe.par_index_cp[i_comp]['log_tau_be']]
                bac = self.bac_func(self.orig_wave_w, log_e_tem=log_e_tem, log_tau_be=log_tau_be)
                orig_flux_int_ew = bac[None,:] # convert to (1,w) format
            if np.isin(iron_names, self.cframe.info_c[i_comp]['mod_used']).any():
                if self.cframe.info_c[i_comp]['segment']:
                    iron = self.iron_func(segment=True)
                    orig_flux_int_ew = copy(iron)
                else:
                    iron = self.iron_func(segment=False)
                    orig_flux_int_ew = iron[None,:] # convert to (1,w) format

            # dust extinction
            Av = par_cp[i_comp, self.cframe.par_index_cp[i_comp]['Av']]
            orig_flux_d_ew = orig_flux_int_ew * 10.0**(-0.4 * Av * ExtLaw(self.orig_wave_w))

            # redshift models
            voff = par_cp[i_comp, self.cframe.par_index_cp[i_comp]['voff']]
            z_ratio = (1 + self.v0_redshift) * (1 + voff/299792.458) # (1+z) = (1+zv0) * (1+v/c)
            orig_wave_z_w = self.orig_wave_w * z_ratio
            orig_flux_dz_ew = orig_flux_d_ew / z_ratio

            # convolve with intrinsic and instrumental dispersion if self.R_inst_rw is not None; only for iron
            if (self.R_inst_rw is not None) & (conv_nbin is not None) & np.isin(iron_names, self.cframe.info_c[i_comp]['mod_used']).any():
                fwhm = par_cp[i_comp, self.cframe.par_index_cp[i_comp]['fwhm']]
                R_inst_w = np.interp(orig_wave_z_w, self.R_inst_rw[0], self.R_inst_rw[1])
                orig_flux_dzc_ew = convolve_var_width_fft(orig_wave_z_w, orig_flux_dz_ew, dv_fwhm_obj=fwhm, 
                                                          dw_fwhm_ref=self.dw_fwhm_dsp_w*z_ratio, R_inst_w=R_inst_w, num_bins=conv_nbin)
            else:
                orig_flux_dzc_ew = orig_flux_dz_ew # just copy if convlution not required, e.g., for broad-band sed fitting

            # project to observed wavelength
            interp_func = interp1d(orig_wave_z_w, orig_flux_dzc_ew, axis=1, kind='linear', fill_value='extrapolate')
            obs_flux_scomp_ew = interp_func(obs_wave_w)

            if i_comp == 0: 
                obs_flux_mcomp_ew = obs_flux_scomp_ew
            else:
                obs_flux_mcomp_ew = np.vstack((obs_flux_mcomp_ew, obs_flux_scomp_ew))

        if mask_lite_e is not None:
            obs_flux_mcomp_ew = obs_flux_mcomp_ew[mask_lite_e,:]

        return obs_flux_mcomp_ew

    ##########################################################################
    ########################## Output functions ##############################

    def extract_results(self, step=None, if_print_results=True, if_return_results=False, if_rev_v0_redshift=False, if_show_average=False, lum_unit='erg/s', **kwargs):

        ############################################################
        # check and replace the args to be compatible with old version <= 2.2.4
        if 'print_results'  in [*kwargs]: if_print_results = kwargs['print_results']
        if 'return_results' in [*kwargs]: if_return_results = kwargs['return_results']
        if 'show_average'   in [*kwargs]: if_show_average = kwargs['show_average']
        ############################################################

        if (step is None) | (step in ['best', 'final']): step = 'joint_fit_3' if self.fframe.have_phot else 'joint_fit_2'
        if  step in ['spec+SED', 'spectrum+SED']:  step = 'joint_fit_3'
        if  step in ['spec', 'pure-spec', 'spectrum', 'pure-spectrum']:  step = 'joint_fit_2'

        best_chi_sq_l = copy(self.fframe.output_s[step]['chi_sq_l'])
        best_par_lp   = copy(self.fframe.output_s[step]['par_lp'])
        best_coeff_le = copy(self.fframe.output_s[step]['coeff_le'])

        # update best-fit voff and fwhm if systemic redshift is updated
        if if_rev_v0_redshift & (self.fframe.rev_v0_redshift is not None):
            best_par_lp[:, self.fframe.par_name_p == 'voff'] -= self.fframe.ref_voff_l[0]
            best_par_lp[:, self.fframe.par_name_p == 'fwhm'] *= (1+self.fframe.v0_redshift) / (1+self.fframe.rev_v0_redshift)

        mod = 'agn'
        fp0, fp1, fe0, fe1 = self.fframe.search_model_index(mod, self.fframe.full_model_type)
        spec_wave_w = self.fframe.spec['wave_w']
        spec_flux_scale = self.fframe.spec_flux_scale
        num_loops = self.fframe.num_loops
        comp_c = self.cframe.comp_c
        par_name_cp = self.cframe.par_name_cp
        num_comps = self.cframe.num_comps
        num_pars_per_comp = self.cframe.num_pars_c_max
        num_coeffs_c = self.num_coeffs_c

        # list the properties to be output
        common_val_names = ['flux_3000', 'flux_5100', 'flux_wavenorm']

        # format of results
        # output_c['comp']['par_lp'][i_l,i_p]: parameters
        # output_c['comp']['coeff_le'][i_l,i_e]: coefficients
        # output_c['comp']['values']['name_l'][i_l]: calculated values
        output_c = {}
        i_e0 = 0; i_e1 = 0
        for i_comp in range(num_comps): 
            output_c[str(comp_c[i_comp])] = {} # init results for each comp
            output_c[comp_c[i_comp]]['par_lp']   = best_par_lp[:, fp0:fp1].reshape(num_loops, num_comps, num_pars_per_comp)[:, i_comp, :]
            i_e0 += 0 if i_comp == 0 else num_coeffs_c[i_comp-1]
            i_e1 += num_coeffs_c[i_comp]
            output_c[comp_c[i_comp]]['coeff_le'] = best_coeff_le[:, fe0:fe1][:, i_e0:i_e1]
            output_c[comp_c[i_comp]]['values'] = {}
            if np.isin(powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():       
                comp_val_names = par_name_cp[i_comp].tolist() + common_val_names + ['log_lambLum_3000', 'log_lambLum_5100', 'log_lambLum_wavenorm']
            if np.isin(bending_powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():       
                comp_val_names = par_name_cp[i_comp].tolist() + common_val_names + ['log_lambLum_3000', 'log_lambLum_5100', 'log_lambLum_wavenorm', 'log_lambLum_waveturn', 'flux_waveturn']
            if np.isin(blackbody_names, self.cframe.info_c[i_comp]['mod_used']).any():
                comp_val_names = par_name_cp[i_comp].tolist() + common_val_names + ['log_lambLum_3000', 'log_lambLum_5100', 'log_lambLum_wavenorm', 'log_Lum_int']
            if np.isin(bac_names, self.cframe.info_c[i_comp]['mod_used']).any():
                comp_val_names = par_name_cp[i_comp].tolist() + common_val_names + ['log_Lum_int']
            if np.isin(iron_names, self.cframe.info_c[i_comp]['mod_used']).any():      
                comp_val_names = par_name_cp[i_comp].tolist() + common_val_names + ['log_Lum_uv', 'log_Lum_opt']
            for val_name in comp_val_names:
                output_c[comp_c[i_comp]]['values'][val_name] = np.zeros(num_loops, dtype='float')
        output_c['sum'] = {}
        output_c['sum']['values'] = {} # only init values for sum of all comp
        for val_name in common_val_names:
            output_c['sum']['values'][val_name] = np.zeros(num_loops, dtype='float')

        for i_comp in range(num_comps): 
            for i_par in range(len(self.cframe.config[comp_c[i_comp]]['pars'])): # use the actual valid par number instead of num_pars_per_comp
                comp_val_names = [*output_c[comp_c[i_comp]]['values']]
                output_c[comp_c[i_comp]]['values'][comp_val_names[i_par]] = output_c[comp_c[i_comp]]['par_lp'][:, i_par]
            for i_loop in range(num_loops):
                par_p   = output_c[comp_c[i_comp]]['par_lp'][i_loop]
                coeff_e = output_c[comp_c[i_comp]]['coeff_le'][i_loop]

                voff = par_p[self.cframe.par_index_cp[i_comp]['voff']]
                rev_redshift = (1+voff/299792.458)*(1+self.v0_redshift)-1

                tmp_spec_w = self.fframe.output_mc[mod][comp_c[i_comp]]['spec_lw'][i_loop, :]
                mask_norm_w = np.abs(spec_wave_w/(1+rev_redshift) - 3000) < 25 # for observed flux at rest 3000 AA 
                if mask_norm_w.sum() > 0:
                    output_c[comp_c[i_comp]]['values']['flux_3000'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                    output_c['sum']['values']['flux_3000'][i_loop] += tmp_spec_w[mask_norm_w].mean()
                mask_norm_w = np.abs(spec_wave_w/(1+rev_redshift) - 5100) < 25 # for observed flux at rest 5100 AA
                if mask_norm_w.sum() > 0:
                    output_c[comp_c[i_comp]]['values']['flux_5100'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                    output_c['sum']['values']['flux_5100'][i_loop] += tmp_spec_w[mask_norm_w].mean()
                mask_norm_w = np.abs(spec_wave_w/(1+rev_redshift) - self.w_norm) < self.dw_norm # for observed flux at user given wavenorm
                if mask_norm_w.sum() > 0:
                    output_c[comp_c[i_comp]]['values']['flux_wavenorm'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                    output_c['sum']['values']['flux_wavenorm'][i_loop] += tmp_spec_w[mask_norm_w].mean()

                dist_lum = cosmo.luminosity_distance(rev_redshift).to('cm').value
                unitconv = 4*np.pi*dist_lum**2 * spec_flux_scale # convert intrinsic flux to Lum, in erg/s
                if lum_unit == 'Lsun': unitconv /= const.L_sun.to('erg/s').value

                if np.isin(powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    alpha_lambda = par_p[self.cframe.par_index_cp[i_comp]['alpha_lambda']]
                    for (wave, wave_str) in zip([3000, 5100, self.w_norm], ['3000', '5100', 'wavenorm']):
                        flux_wave = coeff_e[0] * self.powerlaw_func(wave, wave_norm=self.w_norm, alpha_lambda1=alpha_lambda, alpha_lambda2=None, 
                                                                    curvature=None, bending=False)
                        lambLum_wave = flux_wave * unitconv * wave
                        output_c[comp_c[i_comp]]['values']['log_lambLum_'+wave_str][i_loop] = np.log10(lambLum_wave)

                if np.isin(bending_powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    alpha_lambda1 = par_p[self.cframe.par_index_cp[i_comp]['alpha_lambda1']]
                    alpha_lambda2 = par_p[self.cframe.par_index_cp[i_comp]['alpha_lambda2']]
                    wave_turn     = par_p[self.cframe.par_index_cp[i_comp]['wave_turn']]
                    curvature     = par_p[self.cframe.par_index_cp[i_comp]['curvature']]
                    for (wave, wave_str) in zip([3000, 5100, self.w_norm, wave_turn], ['3000', '5100', 'wavenorm', 'waveturn']):
                        flux_wave = coeff_e[0] * self.powerlaw_func(wave, wave_norm=wave_turn, alpha_lambda1=alpha_lambda1, alpha_lambda2=alpha_lambda2, 
                                                                    curvature=curvature, bending=True)
                        lambLum_wave = flux_wave * unitconv * wave
                        output_c[comp_c[i_comp]]['values']['log_lambLum_'+wave_str][i_loop] = np.log10(lambLum_wave)
                    mask_norm_w = np.abs(spec_wave_w/(1+rev_redshift) - wave_turn) < self.dw_norm 
                    if mask_norm_w.sum() > 0:
                        output_c[comp_c[i_comp]]['values']['flux_waveturn'][i_loop] = tmp_spec_w[mask_norm_w].mean()

                if np.isin(blackbody_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    log_tem  = par_p[self.cframe.par_index_cp[i_comp]['log_tem']]
                    for (wave, wave_str) in zip([3000, 5100, self.w_norm], ['3000', '5100', 'wavenorm']):
                        flux_wave = coeff_e[0] * self.blackbody_func(wave, log_tem=log_tem, wave_norm=self.w_norm)
                        lambLum_wave = flux_wave * unitconv * wave
                        output_c[comp_c[i_comp]]['values']['log_lambLum_'+wave_str][i_loop] = np.log10(lambLum_wave)
                    tmp_wave_w = np.logspace(np.log10(912), 7.5, num=10000) # till 10 K
                    tmp_bb_w = self.blackbody_func(tmp_wave_w, log_tem=log_tem, wave_norm=self.w_norm)
                    output_c[comp_c[i_comp]]['values']['log_Lum_int'][i_loop] = np.log10(coeff_e[0] * unitconv * np.trapezoid(tmp_bb_w, x=tmp_wave_w))

                if np.isin(bac_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    log_e_tem  = par_p[self.cframe.par_index_cp[i_comp]['log_e_tem']]
                    log_tau_be = par_p[self.cframe.par_index_cp[i_comp]['log_tau_be']]
                    tmp_wave_w = np.linspace(912.0, 3646.0, 10000)
                    tmp_bac_w = self.bac_func(tmp_wave_w, log_e_tem=log_e_tem, log_tau_be=log_tau_be)
                    output_c[comp_c[i_comp]]['values']['log_Lum_int'][i_loop] = np.log10(coeff_e[0] * unitconv * np.trapezoid(tmp_bac_w, x=tmp_wave_w))

                if np.isin(iron_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    if self.cframe.info_c[i_comp]['segment']:
                        coeff_uv  = (coeff_e[1:4] * self.iron_flux_norm_e[1:4])[coeff_e[1:4] > 0].sum()
                        coeff_opt = (coeff_e[4:6] * self.iron_flux_norm_e[4:6])[coeff_e[4:6] > 0].sum()
                    else:
                        coeff_uv  = coeff_e[0]
                        coeff_opt = coeff_e[0] * self.iron_flux_opt_uv_ratio
                    output_c[comp_c[i_comp]]['values']['log_Lum_uv' ][i_loop] = np.log10(coeff_uv  * unitconv)
                    output_c[comp_c[i_comp]]['values']['log_Lum_opt'][i_loop] = np.log10(coeff_opt * unitconv)

        self.output_c = output_c # save to model frame
        self.num_loops = num_loops # for print_results
        self.spec_flux_scale = spec_flux_scale # to calculate luminosity in printing

        if if_print_results: self.print_results(log=self.fframe.log_message, if_show_average=if_show_average, lum_unit=lum_unit)
        if if_return_results: return output_c

    def print_results(self, log=[], if_show_average=False, lum_unit='erg/s'):
        mask_l = np.ones(self.num_loops, dtype='bool')
        if not if_show_average: mask_l[1:] = False
        lum_unit_str = '(log Lsun) ' if lum_unit == 'Lsun' else '(log erg/s)'

        if self.cframe.num_comps > 1:
            num_comps = len([*self.output_c])
        else:
            num_comps = 1

        for i_comp in range(num_comps):
            tmp_values_vl = self.output_c[[*self.output_c][i_comp]]['values']
            print_log('', log)
            msg = ''
            if i_comp < self.cframe.num_comps:
                print_log(f"Best-fit properties of AGN component: <{self.cframe.comp_c[i_comp]}>", log)
                if np.isin(iron_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    msg += f"| Velocity shift (km/s)                               = {tmp_values_vl['voff'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['voff'].std():<8.4f}|\n"
                    msg += f"| Velocity FWHM (km/s)                                = {tmp_values_vl['fwhm'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['fwhm'].std():<8.4f}|\n"
                else:
                    print_log(f"[Note] velocity shift (i.e., redshift) and FWHM are tied following the input model_config.", log)
                msg += f"| Extinction (Av)                                     = {tmp_values_vl['Av'][mask_l].mean():10.4f}"
                msg += f" +/- {tmp_values_vl['Av'].std():<8.4f}|\n"
                if np.isin(powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    msg += f"| Powerlaw α_λ                                        = {tmp_values_vl['alpha_lambda'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['alpha_lambda'].std():<8.4f}|\n"

                if np.isin(bending_powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    msg += f"| Powerlaw α1_λ (<= turning wavelength)               = {tmp_values_vl['alpha_lambda1'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['alpha_lambda1'].std():<8.4f}|\n"
                    msg += f"| Powerlaw α2_λ ( > turning wavelength)               = {tmp_values_vl['alpha_lambda2'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['alpha_lambda2'].std():<8.4f}|\n"
                    msg += f"| Curvature                                           = {tmp_values_vl['curvature'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['curvature'].std():<8.4f}|\n"
                    msg += f"| Turning wavelength (Å)                              = {tmp_values_vl['wave_turn'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['wave_turn'].std():<8.4f}|\n"
                    msg += f"| F{tmp_values_vl['wave_turn'][0]:.0f} (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)            = {tmp_values_vl['flux_waveturn'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['flux_waveturn'].std():<8.4f}|\n"
                    msg += f"| λL{tmp_values_vl['wave_turn'][0]:.0f} (rest,intrinsic) "+lum_unit_str+f"                 = {tmp_values_vl['log_lambLum_waveturn'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['log_lambLum_waveturn'].std():<8.4f}|\n"

                if np.isin(blackbody_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    msg += f"| Blackbody temperature (log K)                       = {tmp_values_vl['log_tem'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['log_tem'].std():<8.4f}|\n"
                    msg += f"| Blackbody integrated Lum "+lum_unit_str+f"                = {tmp_values_vl['log_Lum_int'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['log_Lum_int'].std():<8.4f}|\n"

                if np.isin(bac_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    msg += f"| Balmer continuum e- temperature (log K)             = {tmp_values_vl['log_e_tem'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['log_e_tem'].std():<8.4f}|\n"
                    msg += f"| Balmer continuum optical depth at 3646 Å (log τ)    = {tmp_values_vl['log_tau_be'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['log_tau_be'].std():<8.4f}|\n"
                    msg += f"| Balmer continuum integrated Lum "+lum_unit_str+f"         = {tmp_values_vl['log_Lum_int'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['log_Lum_int'].std():<8.4f}|\n"

                if np.isin(iron_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    msg += f"| Fe II integrated Lum in 2150-4000 Å "+lum_unit_str+f"     = {tmp_values_vl['log_Lum_uv'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['log_Lum_uv'].std():<8.4f}|\n"
                    msg += f"| Fe II integrated Lum in 4000-5600 Å "+lum_unit_str+f"     = {tmp_values_vl['log_Lum_opt'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['log_Lum_opt'].std():<8.4f}|\n"

                if np.isin(powerlaw_names+bending_powerlaw_names+blackbody_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    msg += f"| λL3000 (rest,intrinsic) "+lum_unit_str+f"                 = {tmp_values_vl['log_lambLum_3000'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['log_lambLum_3000'].std():<8.4f}|\n"
                    msg += f"| λL5100 (rest,intrinsic) "+lum_unit_str+f"                 = {tmp_values_vl['log_lambLum_5100'][mask_l].mean():10.4f}"
                    msg += f" +/- {tmp_values_vl['log_lambLum_5100'].std():<8.4f}|\n"
                    if not (self.w_norm in [3000,5100]):
                        msg += f"| λL{self.w_norm:.0f} (rest,intrinsic) "+lum_unit_str+f"                 = {tmp_values_vl['log_lambLum_wavenorm'][mask_l].mean():10.4f}"
                        msg += f" +/- {tmp_values_vl['log_lambLum_wavenorm'].std():<8.4f}|\n"
            else:
                print_log(f"Best-fit AGN properties of the sum of all components.", log)
            msg += f"| F3000 (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)            = {tmp_values_vl['flux_3000'][mask_l].mean():10.4f}"
            msg += f" +/- {tmp_values_vl['flux_3000'].std():<8.4f}|\n"
            msg += f"| F5100 (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)            = {tmp_values_vl['flux_5100'][mask_l].mean():10.4f}"
            msg += f" +/- {tmp_values_vl['flux_5100'].std():<8.4f}|"
            if not (self.w_norm in [3000,5100]):
                msg += '\n'
                msg += f"| F{self.w_norm:.0f} (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)            = {tmp_values_vl['flux_wavenorm'][mask_l].mean():10.4f}"
                msg += f" +/- {tmp_values_vl['flux_wavenorm'].std():<8.4f}|"

            bar = '=' * len(msg.split('|\n')[-1])
            print_log(bar, log)
            print_log(msg, log)
            print_log(bar, log)



