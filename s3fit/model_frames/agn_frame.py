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
from astropy.cosmology import WMAP9 as cosmo
# from astropy.modeling.models import BlackBody
from scipy.interpolate import interp1d

from ..auxiliary_func import print_log, convolve_var_width_fft
from ..extinct_law import ExtLaw

class AGNFrame(object):
    def __init__(self, filename=None, cframe=None, v0_redshift=None, R_inst_rw=None, 
                 w_min=None, w_max=None, w_norm=5500, dw_norm=25, 
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
        self.num_coeffs = self.cframe.num_comps # one independent element per component

        # currently do not consider negative spectra 
        self.mask_absorption_e = np.zeros((self.num_coeffs), dtype='bool')
        
        # set original wavelength grid
        # self.orig_wave_w = np.linspace(w_min, w_max, num=100)
        orig_wave_logbin = 0.05
        orig_wave_num = int(np.round(np.log10(w_max/w_min) / orig_wave_logbin))
        self.orig_wave_w = np.logspace(np.log10(w_min), np.log10(w_max), num=orig_wave_num)

        if np.isin('iron', [self.cframe.info_c[i_comp]['mod_used'] for i_comp in range(self.num_comps)]):
            self.read_iron()

        if self.verbose:
            print_log(f"AGN UV/optical continuum components: {np.array([self.cframe.info_c[i_comp]['mod_used'] for i_comp in range(self.num_comps)]).T}", self.log_message)

    ##############################

    def powerlaw_func(self, wavelength, alpha_lambda, wave_norm=5500):
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

    def read_iron(self):
        # combined I Zw 1 Fe II (+ UV Fe III) template, convolving to fwhm = 1100 km/s
        # https://arxiv.org/pdf/astro-ph/0104320
        # https://arxiv.org/pdf/astro-ph/0606040
        # https://arxiv.org/pdf/astro-ph/0312654

        iron_lib = fits.open(self.filename)
        iron_wave_w = iron_lib[0].data[0] # AA, in rest frame
        iron_flux_w = iron_lib[0].data[1] # erg/s/cm2/AA
        iron_dw_fwhm_w = iron_wave_w * 1100 / 299792.458

        # normalize models at given wavelength
        mask_norm_w = (iron_wave_w > 2200) & (iron_wave_w < 4000)
        flux_norm_uv = np.trapezoid(iron_flux_w[mask_norm_w], x=iron_wave_w[mask_norm_w])
        mask_norm_w = (iron_wave_w > 4000) & (iron_wave_w < 5800)
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

    def iron_func(self):
        return self.iron_flux_w # no special parameter needed

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
        # comp can be: 'powerlaw', 'iron', 'bac'
        # par: voff, fwhm, AV (general); alpha_lambda (pl); logtem, logtau (bac)
        if if_pars_flat: 
            par_cp = self.cframe.flat_to_arr(input_pars)
        else:
            par_cp = copy(input_pars)
        if mask_lite_e is not None:
            mask_lite_ce = self.cframe.flat_to_arr(mask_lite_e)

        for i_comp in range(par_cp.shape[0]):
            # read and append intrinsic templates in rest frame
            if np.isin('powerlaw', self.cframe.info_c[i_comp]['mod_used']):
                pl = self.powerlaw_func(self.orig_wave_w, alpha_lambda=par_cp[i_comp,3], wave_norm=self.w_norm)
                orig_flux_int_ew = pl[None,:] # convert to (1,w) format
            if np.isin('iron', self.cframe.info_c[i_comp]['mod_used']):
                iron = self.iron_func()
                orig_flux_int_ew = iron[None,:] # convert to (1,w) format; support multi-element for wavelength bins later
            if np.isin('bac', self.cframe.info_c[i_comp]['mod_used']):
                bac = self.bac_func(self.orig_wave_w, logtem=par_cp[i_comp,3], logtau=par_cp[i_comp,4])
                orig_flux_int_ew = bac[None,:] # convert to (1,w) format

            # limit element number for accelarate calculation
            if mask_lite_e is not None:
                orig_flux_int_ew = orig_flux_int_ew[mask_lite_ce[i_comp,:],:] 

            # dust extinction
            orig_flux_d_ew = orig_flux_int_ew * 10.0**(-0.4 * par_cp[i_comp,2] * ExtLaw(self.orig_wave_w))

            # redshift models
            z_ratio = (1 + self.v0_redshift) * (1 + par_cp[i_comp,0]/299792.458) # (1+z) = (1+zv0) * (1+v/c)
            orig_wave_z_w = self.orig_wave_w * z_ratio
            orig_flux_dz_ew = orig_flux_d_ew / z_ratio

            # convolve with intrinsic and instrumental dispersion if self.R_inst_rw is not None; skip for powerlaw
            if (self.R_inst_rw is not None) & (conv_nbin is not None) & ~np.isin('powerlaw', self.cframe.info_c[i_comp]['mod_used']):
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
        num_coeffs_per_comp = int(self.num_coeffs / num_comps)

        # list the properties to be output
        val_names  = ['voff', 'fwhm', 'AV'] # basic fitting parameters

        # format of results
        # output_c['comp']['par_lp'][i_l,i_p]: parameters
        # output_c['comp']['coeff_le'][i_l,i_e]: coefficients
        # output_c['comp']['values']['name_l'][i_l]: calculated values
        output_c = {}
        for i_comp in range(num_comps): 
            output_c[comp_c[i_comp]] = {} # init results for each comp
            output_c[comp_c[i_comp]]['par_lp']   = best_par_lp[:, fp0:fp1].reshape(num_loops, num_comps, num_pars_per_comp)[:, i_comp, :]
            output_c[comp_c[i_comp]]['coeff_le'] = best_coeff_le[:, fe0:fe1].reshape(num_loops, num_comps, num_coeffs_per_comp)[:, i_comp, :]
            output_c[comp_c[i_comp]]['values'] = {}
            if np.isin('powerlaw', self.cframe.info_c[i_comp]['mod_used']):       
                comp_val_names = val_names + ['index_alpha', 'flux_3000', 'flux_5100', 'flux_wavenorm', 'loglambLum_3000', 'loglambLum_5100', 'loglambLum_wavenorm']
            if np.isin('iron', self.cframe.info_c[i_comp]['mod_used']):      
                comp_val_names = val_names + ['flux_3000', 'flux_5100', 'flux_wavenorm', 'logLum_uv', 'logLum_opt']
            if np.isin('bac', self.cframe.info_c[i_comp]['mod_used']):
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
                if np.isin('powerlaw', self.cframe.info_c[i_comp]['mod_used']):
                    lambLum_wavenorm = coeff_e[0] * unitconv / const.L_sun.to('erg/s').value * self.w_norm # in Lsun
                    lambLum_3000 = lambLum_wavenorm * (3000/self.w_norm) ** (1+par_p[3])
                    lambLum_5100 = lambLum_wavenorm * (5100/self.w_norm) ** (1+par_p[3])
                    output_c[comp_c[i_comp]]['values']['loglambLum_3000'][i_loop] = np.log10(lambLum_3000)
                    output_c[comp_c[i_comp]]['values']['loglambLum_5100'][i_loop] = np.log10(lambLum_5100)
                    output_c[comp_c[i_comp]]['values']['loglambLum_wavenorm'][i_loop] = np.log10(lambLum_wavenorm)
                if np.isin('iron', self.cframe.info_c[i_comp]['mod_used']):
                    output_c[comp_c[i_comp]]['values']['logLum_uv'][i_loop] = np.log10(coeff_e[0] * unitconv)
                    output_c[comp_c[i_comp]]['values']['logLum_opt'][i_loop] = np.log10(coeff_e[0] * unitconv * self.iron_flux_opt_uv_ratio)
                if np.isin('bac', self.cframe.info_c[i_comp]['mod_used']):
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
                if np.isin(['iron', 'bac'], self.cframe.info_c[i_comp]['mod_used']).any(): # != 'powerlaw':
                    msg += f'| Velocity shift (km/s)                               = {tmp_values_vl["voff"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["voff"].std():<8.4f}|\n'
                    msg += f'| Velocity FWHM (km/s)                                = {tmp_values_vl["fwhm"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["fwhm"].std():<8.4f}|\n'
                else:
                    print_log(f'[Note] velocity shift (i.e., redshift) and FWHM are tied following the input model_config.', log)
                msg += f'| Extinction (AV)                                     = {tmp_values_vl["AV"][mask_l].mean():10.4f}'
                msg += f' +/- {tmp_values_vl["AV"].std():<8.4f}|\n'
                if np.isin('powerlaw', self.cframe.info_c[i_comp]['mod_used']):
                    msg += f'| Powerlaw α_λ                                        = {tmp_values_vl["index_alpha"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["index_alpha"].std():<8.4f}|\n'  
                    msg += f'| λL3000 (rest,intrinsic) (log Lsun)                  = {tmp_values_vl["loglambLum_3000"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["loglambLum_3000"].std():<8.4f}|\n'
                    msg += f'| λL5100 (rest,intrinsic) (log Lsun)                  = {tmp_values_vl["loglambLum_5100"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["loglambLum_5100"].std():<8.4f}|\n'
                    if ~np.isin(self.w_norm, [3000,5100]):
                        msg += f'| λL{self.w_norm} (rest,intrinsic) (log Lsun)                  = {tmp_values_vl["loglambLum_wavenorm"][mask_l].mean():10.4f}'
                        msg += f' +/- {tmp_values_vl["loglambLum_wavenorm"].std():<8.4f}|\n'
                if np.isin('iron', self.cframe.info_c[i_comp]['mod_used']):
                    msg += f'| Fe II integrated Lum in 2200-4000 A (log erg/s/cm2) = {tmp_values_vl["logLum_uv"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["logLum_uv"].std():<8.4f}|\n'
                    msg += f'| Fe II integrated Lum in 4000-5800 A (log erg/s/cm2) = {tmp_values_vl["logLum_opt"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["logLum_opt"].std():<8.4f}|\n'
                if np.isin('bac', self.cframe.info_c[i_comp]['mod_used']):
                    msg += f'| Balmer continuum e- temperature (log K)             = {tmp_values_vl["logtem"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["logtem"].std():<8.4f}|\n'       
                    msg += f'| Balmer continuum optical depth at 3646 A (log τ)    = {tmp_values_vl["logtau"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["logtem"].std():<8.4f}|\n'
                    msg += f'| Balmer continuum integrated Lum (log erg/s/cm2)     = {tmp_values_vl["logLum_int"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["logLum_int"].std():<8.4f}|\n'
            else:
                print_log(f'Best-fit stellar properties of the sum of all components.', log)
            msg += f'| F3000 (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/AA)           = {tmp_values_vl["flux_3000"][mask_l].mean():10.4f}'
            msg += f' +/- {tmp_values_vl["flux_3000"].std():<8.4f}|\n'
            msg += f'| F5100 (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/AA)           = {tmp_values_vl["flux_5100"][mask_l].mean():10.4f}'
            msg += f' +/- {tmp_values_vl["flux_5100"].std():<8.4f}|'
            if ~np.isin(self.w_norm, [3000,5100]):
                msg += '\n'
                msg += f'| F{self.w_norm} (rest,extinct) ({self.spec_flux_scale:.0e} erg/s/cm2/AA)           = {tmp_values_vl["flux_wavenorm"][mask_l].mean():10.4f}'
                msg += f' +/- {tmp_values_vl["flux_wavenorm"].std():<8.4f}|'      

            bar = '=' * len(msg.split('|\n')[-1])
            print_log(bar, log)
            print_log(msg, log)
            print_log(bar, log)



