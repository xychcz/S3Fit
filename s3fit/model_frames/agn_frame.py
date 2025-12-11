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

from ..auxiliary_func import print_log, convolve_var_width_fft
from ..extinct_law import ExtLaw

# alternative component names
powerlaw_names = ['powerlaw', 'pl', 'Powerlaw', 'PL']
bending_powerlaw_names = ['bending_powerlaw', 'bending-powerlaw', 'bending powerlaw', 'bending_pl', 'bending-pl', 'bending pl']
iron_names = ['iron', 'fe ii', 'feii', 'Iron', 'Fe II', 'FeII']
bac_names = ['bac', 'balmer continuum', 'Balmer continuum']

class AGNFrame(object):
    def __init__(self, filename=None, cframe=None, v0_redshift=None, R_inst_rw=None, 
                 w_min=None, w_max=None, w_norm=5100, dw_norm=25, 
                 Rratio_mod=None, dw_fwhm_dsp=None, dw_pix_inst=None, 
                 verbose=True, log_message=[]):

        self.filename = filename
        self.cframe = cframe
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

        self.num_comps = self.cframe.num_comps

        self.num_coeffs_c = np.zeros(self.num_comps, dtype='int')
        for i_comp in range(self.num_comps):
            if np.isin(powerlaw_names+bending_powerlaw_names+bac_names, self.cframe.info_c[i_comp]['mod_used']).any():
                self.num_coeffs_c[i_comp] = 1 # one independent element per component
            if np.isin(iron_names, self.cframe.info_c[i_comp]['mod_used']).any():
                if ~np.isin('segment', [*self.cframe.info_c[i_comp]]): self.cframe.info_c[i_comp]['segment'] = False
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


        if self.verbose:
            print_log(f"AGN UV/optical continuum components: {np.array([self.cframe.info_c[i_comp]['mod_used'] for i_comp in range(self.num_comps)]).T}", self.log_message)

    ##############################

    def simple_powerlaw(self, wavelength, wave_norm=None, flux_norm=1.0, alpha_lambda=None):
        pl = flux_norm * (wavelength/wave_norm)**alpha_lambda
        return pl

    def bending_powerlaw(self, wavelength, wave_turn=None, flux_trun=1.0, alpha_lambda1=None, alpha_lambda2=None, curvature=None, bending=False):
        # alpha_lambda1, alpha_lambda2: index with wavelength <= wave_turn and wavelength > wave_turn
        # curvature <= 0: broken two-side powerlaw
        # curvature > 0: smoothed bending powerlaw. larger curvature --> smoother break (5: very smooth; 0.1: very sharp)
        if curvature is None: curvature = 0
        if alpha_lambda2 is not None:
            if alpha_lambda1 > alpha_lambda2: curvature = 0 # smoothing does not work in this case

        pl = self.simple_powerlaw(wavelength, wave_turn, flux_trun, alpha_lambda1)

        if bending:
            if curvature <= 0:
                # sharp, continuous broken power law
                if np.isscalar(wavelength): wavelength = np.array([wavelength])
                mask_w = wavelength > wave_turn
                pl[mask_w] = self.simple_powerlaw(wavelength[mask_w], wave_turn, flux_trun, alpha_lambda2)
            else:
                pl_2 = self.simple_powerlaw(wavelength, wave_turn, 1, (alpha_lambda2-alpha_lambda1)/curvature)
                pl *= ((1+pl_2)/2.0)**curvature

        return pl

    def powerlaw_func(self, wavelength, wave_norm=None, alpha_lambda1=None, alpha_lambda2=None, curvature=None, bending=False):
        # normalized to given flux density (e.g.,the same unit of obs) at rest wave_norm before extinct

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
                preconvolving = True
            else:
                preconvolving = False
                self.dw_fwhm_dsp_w = iron_dw_fwhm_w
            self.dw_dsp = self.dw_fwhm_dsp_w.min() * 0.5 # required min bin wavelength following Nyquist–Shannon sampling
            if self.dw_pix_inst is not None:
                self.dw_dsp = min(self.dw_dsp, self.dw_pix_inst/(1+self.v0_redshift) * 0.5) # also require model bin wavelength <= 0.5 of data bin width (convert to rest frame)
            self.dpix_dsp = int(self.dw_dsp / np.median(np.diff(iron_wave_w))) # required min bin number of pixels
            self.dw_dsp = self.dpix_dsp * np.median(np.diff(iron_wave_w)) # update value
            if self.dpix_dsp > 1:
                if preconvolving:
                    if self.verbose: 
                        print_log(f'Downsample preconvolved AGN iron pesudo-continuum with bin width of {self.dw_dsp:.3f} AA in a min resolution of {self.dw_fwhm_dsp_w.min():.3f} AA', 
                                  self.log_message)
                    # before downsampling, smooth the model to avoid aliasing (like in ADC or digital signal reduction)
                    # set dw_fwhm_ref as the dispersion in the original model
                    iron_flux_w = convolve_var_width_fft(iron_wave_w, iron_flux_w, dw_fwhm_obj=self.dw_fwhm_dsp_w, dw_fwhm_ref=iron_dw_fwhm_w, num_bins=10, reset_edge=True)
                else:
                    if self.verbose: 
                        print_log(f'Downsample original AGN iron pesudo-continuum with bin width of {self.dw_dsp:.3f} AA in a min resolution of {self.dw_fwhm_dsp_w.min():.3f} AA', 
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

    def bac_func(self, wavelength, logtem=4.0, logtau=1.0, wave_norm=3000):
        # parameters: electron temperature (K), optical depth at balmer edge (3646)
        # normalize at rest 3000 AA

        # Planck function for the given electron temperature
        C1 = 1.1910429723971885e27   # 2 h c^2 * 1e40 * 1e3
        C2 = 1.4387768775039336e8    # hc/k * 1e10
        tmp = C2 / (wavelength * 10.0**logtem)
        planck_flux_w = C1 / wavelength**5 / np.expm1(tmp) # in erg/s/cm2/AA/sr
        # planck_func = BlackBody(temperature=10.0**logtem * u.K, scale=1*u.erg/(u.s*u.cm**2*u.AA*u.sr))
        # planck_flux_w = planck_func(wavelength * u.AA).value 
        
        # calculate the optical depth at each wavelength
        # τ_λ = τ_BE * (λ_BE / λ)^3  (as in Grandi 1982)
        # the exponent can vary depending on the specific model
        balmer_edge = 3646.0
        optical_depth = 10.0**logtau * (balmer_edge / wavelength)**3

        # calculate the Balmer continuum flux
        bac_flux_w = planck_flux_w * (1 - np.exp(-optical_depth))
        bac_flux_w /= np.interp(wave_norm, wavelength, bac_flux_w)
        bac_flux_w[wavelength >= balmer_edge] = 0

        return bac_flux_w

    ##############################

    def models_unitnorm_obsframe(self, obs_wave_w, input_pars, if_pars_flat=True, mask_lite_e=None, conv_nbin=None):
        # comp can be: 'powerlaw', 'bending-powerlaw', 'iron', 'bac'
        # par: voff, fwhm, AV (general); alpha_lambda (pl); logtem, logtau (bac)
        if if_pars_flat: 
            par_cp = self.cframe.flat_to_arr(input_pars)
        else:
            par_cp = copy(input_pars)

        for i_comp in range(par_cp.shape[0]):
            # read and append intrinsic templates in rest frame
            if np.isin(powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():
                pl = self.powerlaw_func(self.orig_wave_w, wave_norm=self.w_norm, alpha_lambda1=par_cp[i_comp,3], alpha_lambda2=None, curvature=None, bending=False)
                orig_flux_int_ew = pl[None,:] # convert to (1,w) format
            if np.isin(bending_powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():
                pl = self.powerlaw_func(self.orig_wave_w, wave_norm=par_cp[i_comp,5], alpha_lambda1=par_cp[i_comp,3], alpha_lambda2=par_cp[i_comp,4], curvature=par_cp[i_comp,6], bending=True)
                orig_flux_int_ew = pl[None,:] # convert to (1,w) format
            if np.isin(iron_names, self.cframe.info_c[i_comp]['mod_used']).any():
                if self.cframe.info_c[i_comp]['segment']:
                    iron = self.iron_func(segment=True)
                    orig_flux_int_ew = copy(iron)
                else:
                    iron = self.iron_func(segment=False)
                    orig_flux_int_ew = iron[None,:] # convert to (1,w) format
            if np.isin(bac_names, self.cframe.info_c[i_comp]['mod_used']).any():
                bac = self.bac_func(self.orig_wave_w, logtem=par_cp[i_comp,3], logtau=par_cp[i_comp,4])
                orig_flux_int_ew = bac[None,:] # convert to (1,w) format

            # dust extinction
            orig_flux_d_ew = orig_flux_int_ew * 10.0**(-0.4 * par_cp[i_comp,2] * ExtLaw(self.orig_wave_w))

            # redshift models
            z_ratio = (1 + self.v0_redshift) * (1 + par_cp[i_comp,0]/299792.458) # (1+z) = (1+zv0) * (1+v/c)
            orig_wave_z_w = self.orig_wave_w * z_ratio
            orig_flux_dz_ew = orig_flux_d_ew / z_ratio

            # convolve with intrinsic and instrumental dispersion if self.R_inst_rw is not None; only for iron
            if (self.R_inst_rw is not None) & (conv_nbin is not None) & np.isin(iron_names, self.cframe.info_c[i_comp]['mod_used']).any():
                R_inst_w = np.interp(orig_wave_z_w, self.R_inst_rw[0], self.R_inst_rw[1])
                orig_flux_dzc_ew = convolve_var_width_fft(orig_wave_z_w, orig_flux_dz_ew, dv_fwhm_obj=par_cp[i_comp,1], 
                                                          dw_fwhm_ref=self.dw_fwhm_dsp_w*z_ratio, R_inst_w=R_inst_w, num_bins=conv_nbin)
            else:
                orig_flux_dzc_ew = orig_flux_dz_ew # just copy if convlution not required, e.g., for broad-band sed fitting

            # project to observed wavelength
            interp_func = interp1d(orig_wave_z_w, orig_flux_dzc_ew, axis=1, kind='linear', fill_value="extrapolate")
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

    def extract_results(self, ff=None, step=None, print_results=True, return_results=False, show_average=False):
        if (step is None) | (step == 'best') | (step == 'final'):
            step = 'joint_fit_3' if ff.have_phot else 'joint_fit_2'
        if (step == 'spec+SED'):  step = 'joint_fit_3'
        if (step == 'spec') | (step == 'pure-spec'): step = 'joint_fit_2'

        best_chi_sq_l = copy(ff.output_s[step]['chi_sq_l'])
        best_par_lp   = copy(ff.output_s[step]['par_lp'])
        best_coeff_le = copy(ff.output_s[step]['coeff_le'])

        mod = 'agn'
        fp0, fp1, fe0, fe1 = ff.search_model_index(mod, ff.full_model_type)
        spec_wave_w = ff.spec['wave_w']
        spec_flux_scale = ff.spec_flux_scale
        num_loops = ff.num_loops
        comp_c = self.cframe.comp_c
        num_comps = self.cframe.num_comps
        num_pars_per_comp = self.cframe.num_pars_per_comp
        num_coeffs_c = self.num_coeffs_c

        # list the properties to be output
        val_names  = ['voff', 'fwhm', 'AV'] # basic fitting parameters

        # format of results
        # output_c['comp']['par_lp'][i_l,i_p]: parameters
        # output_c['comp']['coeff_le'][i_l,i_e]: coefficients
        # output_c['comp']['values']['name_l'][i_l]: calculated values
        output_c = {}
        i_e0 = 0; i_e1 = 0
        for i_comp in range(num_comps): 
            output_c[comp_c[i_comp]] = {} # init results for each comp
            output_c[comp_c[i_comp]]['par_lp']   = best_par_lp[:, fp0:fp1].reshape(num_loops, num_comps, num_pars_per_comp)[:, i_comp, :]
            i_e0 += 0 if i_comp == 0 else num_coeffs_c[i_comp-1]
            i_e1 += num_coeffs_c[i_comp]
            output_c[comp_c[i_comp]]['coeff_le'] = best_coeff_le[:, fe0:fe1][:, i_e0:i_e1]
            output_c[comp_c[i_comp]]['values'] = {}
            if np.isin(powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():       
                comp_val_names = val_names + ['alpha_lambda', 'flux_3000', 'flux_5100', 'flux_wavenorm', 'loglambLum_3000', 'loglambLum_5100', 'loglambLum_wavenorm']
            if np.isin(bending_powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():       
                comp_val_names = val_names + ['alpha_lambda1', 'alpha_lambda2', 'wave_turn', 'curvature', 'flux_3000', 'flux_5100', 'flux_wavenorm', 'loglambLum_3000', 'loglambLum_5100']
            if np.isin(iron_names, self.cframe.info_c[i_comp]['mod_used']).any():      
                comp_val_names = val_names + ['flux_3000', 'flux_5100', 'flux_wavenorm', 'logLum_uv', 'logLum_opt']
            if np.isin(bac_names, self.cframe.info_c[i_comp]['mod_used']).any():
                comp_val_names = val_names + ['logtem', 'logtau', 'flux_3000', 'flux_5100', 'flux_wavenorm', 'logLum_int']
            for val_name in comp_val_names:
                output_c[comp_c[i_comp]]['values'][val_name] = np.zeros(num_loops, dtype='float')
        output_c['sum'] = {}
        output_c['sum']['values'] = {} # only init values for sum of all comp
        for val_name in ['flux_3000', 'flux_5100', 'flux_wavenorm']:
            output_c['sum']['values'][val_name] = np.zeros(num_loops, dtype='float')

        for i_comp in range(num_comps): 
            for i_par in range(len(self.cframe.config[comp_c[i_comp]]['pars'])): # use the actual valid par number instead of num_pars_per_comp
                comp_val_names = [*output_c[comp_c[i_comp]]['values']]
                output_c[comp_c[i_comp]]['values'][comp_val_names[i_par]] = output_c[comp_c[i_comp]]['par_lp'][:, i_par]
            for i_loop in range(num_loops):
                par_p   = output_c[comp_c[i_comp]]['par_lp'][i_loop]
                coeff_e = output_c[comp_c[i_comp]]['coeff_le'][i_loop]
                rev_redshift = (1+par_p[0]/299792.458)*(1+self.v0_redshift)-1

                tmp_spec_w = ff.output_mc[mod][comp_c[i_comp]]['spec_lw'][i_loop, :]
                mask_norm_w = np.abs(spec_wave_w/(1+rev_redshift) - 3000) < 25 # for observed flux at rest 3000 AA 
                if mask_norm_w.sum() > 0:
                    output_c[comp_c[i_comp]]['values']['flux_3000'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                    output_c['sum']['values']['flux_3000'][i_loop] += tmp_spec_w[mask_norm_w].mean()
                mask_norm_w = np.abs(spec_wave_w/(1+rev_redshift) - 5100) < 25 # for observed flux at rest 5100 AA
                if mask_norm_w.sum() > 0:
                    output_c[comp_c[i_comp]]['values']['flux_5100'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                    output_c['sum']['values']['flux_5100'][i_loop] += tmp_spec_w[mask_norm_w].mean()
                mask_norm_w = np.abs(spec_wave_w/(1+rev_redshift) - self.w_norm) < self.dw_norm # for observed flux at user given wavenorm
                output_c[comp_c[i_comp]]['values']['flux_wavenorm'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                output_c['sum']['values']['flux_wavenorm'][i_loop] += tmp_spec_w[mask_norm_w].mean()

                dist_lum = cosmo.luminosity_distance(rev_redshift).to('cm').value
                unitconv = 4*np.pi*dist_lum**2 * spec_flux_scale # convert intrinsic flux to Lum in erg/s
                if np.isin(powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    lambLum_wavenorm = coeff_e[0] * unitconv / const.L_sun.to('erg/s').value * self.w_norm # in Lsun
                    lambLum_3000 = lambLum_wavenorm * (3000/self.w_norm) ** (1+par_p[3])
                    lambLum_5100 = lambLum_wavenorm * (5100/self.w_norm) ** (1+par_p[3])
                    output_c[comp_c[i_comp]]['values']['loglambLum_3000'][i_loop] = np.log10(lambLum_3000)
                    output_c[comp_c[i_comp]]['values']['loglambLum_5100'][i_loop] = np.log10(lambLum_5100)
                    output_c[comp_c[i_comp]]['values']['loglambLum_wavenorm'][i_loop] = np.log10(lambLum_wavenorm)
                if np.isin(bending_powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    alpha_lambda1, alpha_lambda2, wave_turn = par_p[3], par_p[4], par_p[5]
                    lambLum_waveturn = coeff_e[0] * unitconv / const.L_sun.to('erg/s').value * wave_turn # in Lsun
                    for wave in [3000,5100]:
                        lambLum_wave = lambLum_waveturn * (wave/wave_turn) ** (1+(alpha_lambda1 if wave <= wave_turn else alpha_lambda2))
                        output_c[comp_c[i_comp]]['values']['loglambLum_'+str(wave)][i_loop] = np.log10(lambLum_wave)
                if np.isin(iron_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    if self.cframe.info_c[i_comp]['segment']:
                        coeff_uv  = (coeff_e[1:4] * self.iron_flux_norm_e[1:4])[coeff_e[1:4] > 0].sum()
                        coeff_opt = (coeff_e[4:6] * self.iron_flux_norm_e[4:6])[coeff_e[4:6] > 0].sum()
                    else:
                        coeff_uv  = coeff_e[0]
                        coeff_opt = coeff_e[0] * self.iron_flux_opt_uv_ratio
                    output_c[comp_c[i_comp]]['values']['logLum_uv' ][i_loop] = np.log10(coeff_uv  * unitconv)
                    output_c[comp_c[i_comp]]['values']['logLum_opt'][i_loop] = np.log10(coeff_opt * unitconv)
                if np.isin(bac_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    tmp_wave_w = np.linspace(912.0, 3646.0, 10000)
                    tmp_bac_w = self.bac_func(tmp_wave_w, logtem=par_p[3], logtau=par_p[4]) # full bac spectrum with unit flux normalized at rest 3000 AA
                    output_c[comp_c[i_comp]]['values']['logLum_int'][i_loop] = np.log10(coeff_e[0] * unitconv * np.trapezoid(tmp_bac_w, x=tmp_wave_w))
                    
        self.output_c = output_c # save to model frame
        self.num_loops = num_loops # for print_results
        self.spec_flux_scale = spec_flux_scale # to calculate luminosity in printing

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
            print_log('', log)
            msg = ''
            if i_comp < self.cframe.num_comps:
                print_log(f'Best-fit properties of AGN component: <{self.cframe.comp_c[i_comp]}>', log)
                if np.isin(iron_names+bac_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    msg += f'| Velocity shift (km/s)                               = {tmp_values_vl["voff"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["voff"].std():<8.4f}|\n'
                    msg += f'| Velocity FWHM (km/s)                                = {tmp_values_vl["fwhm"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["fwhm"].std():<8.4f}|\n'
                else:
                    print_log(f'[Note] velocity shift (i.e., redshift) and FWHM are tied following the input model_config.', log)
                msg += f'| Extinction (AV)                                     = {tmp_values_vl["AV"][mask_l].mean():10.4f}'
                msg += f' +/- {tmp_values_vl["AV"].std():<8.4f}|\n'
                if np.isin(powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    msg += f'| Powerlaw α_λ                                        = {tmp_values_vl["alpha_lambda"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["alpha_lambda"].std():<8.4f}|\n'  
                    msg += f'| λL3000 (rest,intrinsic) (log Lsun)                  = {tmp_values_vl["loglambLum_3000"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["loglambLum_3000"].std():<8.4f}|\n'
                    msg += f'| λL5100 (rest,intrinsic) (log Lsun)                  = {tmp_values_vl["loglambLum_5100"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["loglambLum_5100"].std():<8.4f}|\n'
                    if ~np.isin(self.w_norm, [3000,5100]):
                        msg += f'| λL{self.w_norm} (rest,intrinsic) (log Lsun)                  = {tmp_values_vl["loglambLum_wavenorm"][mask_l].mean():10.4f}'
                        msg += f' +/- {tmp_values_vl["loglambLum_wavenorm"].std():<8.4f}|\n'
                if np.isin(bending_powerlaw_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    msg += f'| Powerlaw α1_λ (<= turning wavelength)               = {tmp_values_vl["alpha_lambda1"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["alpha_lambda1"].std():<8.4f}|\n'  
                    msg += f'| Powerlaw α2_λ ( > turning wavelength)               = {tmp_values_vl["alpha_lambda2"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["alpha_lambda2"].std():<8.4f}|\n'  
                    msg += f'| Turning wavelength (Å)                              = {tmp_values_vl["wave_turn"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["wave_turn"].std():<8.4f}|\n'  
                    msg += f'| Curvature                                           = {tmp_values_vl["curvature"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["curvature"].std():<8.4f}|\n' 
                    msg += f'| λL3000 (rest,intrinsic) (log Lsun)                  = {tmp_values_vl["loglambLum_3000"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["loglambLum_3000"].std():<8.4f}|\n'
                    msg += f'| λL5100 (rest,intrinsic) (log Lsun)                  = {tmp_values_vl["loglambLum_5100"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["loglambLum_5100"].std():<8.4f}|\n'
                if np.isin(iron_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    msg += f'| Fe II integrated Lum in 2150-4000 Å (log erg/s/cm2) = {tmp_values_vl["logLum_uv"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["logLum_uv"].std():<8.4f}|\n'
                    msg += f'| Fe II integrated Lum in 4000-5600 Å (log erg/s/cm2) = {tmp_values_vl["logLum_opt"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["logLum_opt"].std():<8.4f}|\n'
                if np.isin(bac_names, self.cframe.info_c[i_comp]['mod_used']).any():
                    msg += f'| Balmer continuum e- temperature (log K)             = {tmp_values_vl["logtem"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["logtem"].std():<8.4f}|\n'       
                    msg += f'| Balmer continuum optical depth at 3646 Å (log τ)    = {tmp_values_vl["logtau"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["logtem"].std():<8.4f}|\n'
                    msg += f'| Balmer continuum integrated Lum (log erg/s/cm2)     = {tmp_values_vl["logLum_int"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["logLum_int"].std():<8.4f}|\n'
            else:
                print_log(f'Best-fit stellar properties of the sum of all components.', log)
            msg += f'| F3000 (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)            = {tmp_values_vl["flux_3000"][mask_l].mean():10.4f}'
            msg += f' +/- {tmp_values_vl["flux_3000"].std():<8.4f}|\n'
            msg += f'| F5100 (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)            = {tmp_values_vl["flux_5100"][mask_l].mean():10.4f}'
            msg += f' +/- {tmp_values_vl["flux_5100"].std():<8.4f}|'
            if ~np.isin(self.w_norm, [3000,5100]):
                msg += '\n'
                msg += f'| F{self.w_norm} (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/Å)            = {tmp_values_vl["flux_wavenorm"][mask_l].mean():10.4f}'
                msg += f' +/- {tmp_values_vl["flux_wavenorm"].std():<8.4f}|'      

            bar = '=' * len(msg.split('|\n')[-1])
            print_log(bar, log)
            print_log(msg, log)
            print_log(bar, log)



