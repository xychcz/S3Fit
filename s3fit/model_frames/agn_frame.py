# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: xiaoyang.chen.cz@gmail.com

import numpy as np
from copy import deepcopy as copy
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import WMAP9 as cosmo
from astropy.modeling.models import BlackBody
from scipy.interpolate import interp1d

from ..auxiliary_func import print_log, convolve_fix_width_fft, convolve_var_width_fft
from ..extinct_law import ExtLaw

class AGNFrame(object):
    def __init__(self, filename=None, w_min=None, w_max=None, w_norm=5500, dw_norm=25, 
                 cframe=None, v0_redshift=None, R_inst_rw=None, resample_width=None, init_smooth_width=None, 
                 verbose=True, log_message=[]):
        
        self.filename = filename
        self.w_min = w_min
        self.w_max = w_max
        self.w_norm = w_norm
        self.dw_norm = dw_norm
        self.cframe = cframe
        self.v0_redshift = v0_redshift
        self.R_inst_rw = R_inst_rw
        self.resample_width = resample_width # resample width
        self.init_smooth_width = init_smooth_width # initial convolving width, e.g., 30AA
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
                
        # # create log grid wavelength (rest) to project intrinsic template and then perform convolution
        # linw_wave = np.linspace(w_min, w_max, num=10)
        # resolution = R_inst_rw * 5 # select a high resampling density
        # self.logw_wave = convert_linw_to_logw(linw_wave, linw_wave, resolution=resolution)[0]

        if np.isin('iron', [self.cframe.info_c[i_comp]['mod_used'] for i_comp in range(self.num_comps)]):
            self.read_iron()

        if self.verbose:
            print_log(f"AGN UV/optical continuum components: {[self.cframe.info_c[i_comp]['mod_used'] for i_comp in range(self.num_comps)]}", self.log_message)

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
        iron_lib = fits.open(self.filename)
        iron_wave_w = iron_lib[0].data[0] # AA
        iron_flux_w = iron_lib[0].data[1] # erg/s/cm2/AA

        # normalize models at given wavelength
        mask_norm_w = (iron_wave_w > 2200) & (iron_wave_w < 4000)
        flux_norm_uv = np.trapz(iron_flux_w[mask_norm_w], x=iron_wave_w[mask_norm_w])
        mask_norm_w = (iron_wave_w > 4000) & (iron_wave_w < 5800)
        flux_norm_opt = np.trapz(iron_flux_w[mask_norm_w], x=iron_wave_w[mask_norm_w])
        iron_flux_w /= flux_norm_uv
        # for any observed spectrum, fuv_obs = fnorm_obs, fopt_obs = fnorm_obs * fopt_mod/fuv_mod

        # determine the resample_width and init_smooth_width, both in wavelength (AA)
        if (self.resample_width is None) & (self.init_smooth_width is not None):
            self.resample_width = self.init_smooth_width / 2 # maximum resample_width to capture smoothed profile
        if (self.init_smooth_width is None) & (self.resample_width is not None):
            self.init_smooth_width = self.resample_width * 2 # minimum init_smooth_width for a well resampling
        if (self.init_smooth_width is not None) & self.verbose: 
            print_log(f'Resample AGN iron pesudo-continuum with bin width of {self.resample_width:.3f} AA in a resolution of {self.init_smooth_width:.3f} AA', 
                      self.log_message)

        # smooth spectra unless both resample_width and init_smooth_width is not input
        if self.init_smooth_width is not None: 
            iron_flux_w = convolve_fix_width_fft(iron_wave_w, iron_flux_w, self.init_smooth_width) 

        # resample model to accelarate calculation
        if self.resample_width is not None: 
            bin_pix = int(self.resample_width / np.diff(iron_wave_w).mean()) # sampling width
            if bin_pix == 0: bin_pix = 1 # in case with high-resolution observed spectrum
        else:
            bin_pix = 1
        iron_wave_w = iron_wave_w[::bin_pix]
        iron_flux_w = iron_flux_w[::bin_pix]

        self.orig_wave_w = np.hstack((self.orig_wave_w, iron_wave_w))
        self.orig_wave_w = np.sort(self.orig_wave_w)
        self.iron_flux_w = np.interp(self.orig_wave_w, iron_wave_w, iron_flux_w, left=0, right=0)

    def iron_func(self):
        return self.iron_flux_w # no special parameter needed

    def bac_func(self, wavelength, logtem=4.0, logtau=1.0, wave_norm=3000):
        # parameters: electron temperature (K), optical depth at balmer edge (3646)
        # normalize at rest 3000 AA

        # Planck function for the given electron temperature
        planck_func = BlackBody(temperature=10.0**logtem * u.K, scale=1*u.erg/(u.s*u.cm**2*u.AA*u.sr))
        planck_flux_w = planck_func(wavelength * u.AA).value 
        
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
                orig_flux_int_ew = np.vstack((pl)).T # convert to (1,w) format
            if np.isin('iron', self.cframe.info_c[i_comp]['mod_used']):
                iron = self.iron_func()
                orig_flux_int_ew = np.vstack((iron)).T # convert to (1,w) format; support multi-element for wavelength bins later
            if np.isin('bac', self.cframe.info_c[i_comp]['mod_used']):
                bac = self.bac_func(self.orig_wave_w, logtem=par_cp[i_comp,3], logtau=par_cp[i_comp,4])
                orig_flux_int_ew = np.vstack((bac)).T # convert to (1,w) format

            if mask_lite_e is not None:
                orig_flux_int_ew = orig_flux_int_ew[mask_lite_ce[i_comp,:],:] # limit element number for accelarate calculation

            # dust extinction
            orig_flux_d_ew = orig_flux_int_ew * 10.0**(-0.4 * par_cp[i_comp,2] * ExtLaw(self.orig_wave_w))
            # redshift models
            z_ratio = (1 + self.v0_redshift) * (1 + par_cp[i_comp,0]/299792.458) # (1+z) = (1+zv0) * (1+v/c)
            orig_wave_z_w = self.orig_wave_w * z_ratio
            orig_flux_dz_ew = orig_flux_d_ew / z_ratio
            # convolve with intrinsic and instrumental dispersion if R_inst_rw is not None 
            # skip for powerlaw
            if (self.R_inst_rw is not None) & (conv_nbin is not None) & ~np.isin('powerlaw', self.cframe.info_c[i_comp]['mod_used']):
                R_inst_w = np.interp(orig_wave_z_w, self.R_inst_rw[0], self.R_inst_rw[1])
                orig_flux_dzc_ew = convolve_var_width_fft(orig_wave_z_w, orig_flux_dz_ew, 
                                                          fwhm_vel_kin=par_cp[i_comp,1], R_inst_w=R_inst_w, num_bins=conv_nbin)
                # convolution in redshifted- or rest-wavelength does not change result
            else:
                orig_flux_dzc_ew = orig_flux_dz_ew 
                # just copy if convlution not required, e.g., for broad-band sed fitting

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
        
        best_chi_sq_l = ff.output_s[step]['chi_sq_l']
        best_par_lp   = ff.output_s[step]['par_lp']
        best_coeff_le = ff.output_s[step]['coeff_le']

        fp0, fp1, fe0, fe1 = ff.search_model_index('agn', ff.full_model_type)
        spec_wave_w = ff.spec['wave_w']
        spec_flux_scale = ff.spec_flux_scale
        num_loops = ff.num_loops
        comp_c = self.cframe.comp_c
        num_comps = self.cframe.num_comps
        num_pars_per_comp = self.cframe.num_pars_per_comp
        num_coeffs_per_comp = int(self.num_coeffs / num_comps)

        # list the properties to be output
        val_names  = ['voff', 'fwhm', 'AV', 'powerlaw_alpha'] # basic fitting parameters
        val_names += ['flux_wavenorm', 'loglambLum_wavenorm']

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
            for val_name in val_names:
                output_c[comp_c[i_comp]]['values'][val_name] = np.zeros(num_loops, dtype='float')
        output_c['sum'] = {}
        output_c['sum']['values'] = {} # only init values for sum of all comp
        for val_name in val_names:
            output_c['sum']['values'][val_name] = np.zeros(num_loops, dtype='float')

        for i_comp in range(num_comps): 
            for i_par in range(num_pars_per_comp):
                output_c[comp_c[i_comp]]['values'][val_names[i_par]] = output_c[comp_c[i_comp]]['par_lp'][:, i_par]
            for i_loop in range(num_loops):
                par_p   = output_c[comp_c[i_comp]]['par_lp'][i_loop]
                coeff_e = output_c[comp_c[i_comp]]['coeff_le'][i_loop]

                rev_redshift = (1+par_p[0]/299792.458)*(1+self.v0_redshift)-1
                # tmp_spec_ew = self.models_unitnorm_obsframe(spec_wave_w, par_p[None,:], if_pars_flat=False)
                # tmp_spec_w = np.dot(coeff_e, tmp_spec_ew)
                tmp_spec_w = ff.output_mc['agn'][comp_c[i_comp]]['spec_lw'][i_loop, :]
                mask_norm_w = np.abs(spec_wave_w/(1+rev_redshift) - self.w_norm) < self.dw_norm # observed flux at wavenorm=5500AA(rest)
                output_c[comp_c[i_comp]]['values']['flux_wavenorm'][i_loop] = tmp_spec_w[mask_norm_w].mean()
                output_c['sum']['values']['flux_wavenorm'][i_loop] += tmp_spec_w[mask_norm_w].mean()
                # output_c[comp_c[i_comp]]['values']['flux_wavenorm'][i_loop] = coeff_e[0] # this is the intrinsic flux_wavenorm
                # output_c['sum']['values']['flux_wavenorm'][i_loop] += coeff_e[0]

                dist_lum = cosmo.luminosity_distance(rev_redshift).to('cm').value
                unitconv = 4*np.pi*dist_lum**2 / const.L_sun.to('erg/s').value * spec_flux_scale # convert intrinsic flux5500(rest) to L5500 
                output_c[comp_c[i_comp]]['values']['loglambLum_wavenorm'][i_loop] = np.log10(coeff_e[0] * unitconv * self.w_norm) # intrinsic λL5500, in Lsun
                output_c['sum']['values']['loglambLum_wavenorm'][i_loop] = coeff_e[0] * unitconv * self.w_norm
        output_c['sum']['values']['loglambLum_wavenorm'] = np.log10(output_c['sum']['values']['loglambLum_wavenorm'])

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
                if np.isin('iron', self.cframe.info_c[i_comp]['mod_used']): # != 'powerlaw':
                    msg += f'| Velocity shift (km/s)                     = {tmp_values_vl["voff"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["voff"].std()[i_comp,1]:<8.4f}|\n'
                    msg += f'| Velocity FWHM (km/s)                      = {tmp_values_vl["fwhm"][mask_l].mean():10.4f}'
                    msg += f' +/- {tmp_values_vl["fwhm"].std()[i_comp,2]:<8.4f}|\n'
                else:
                    print_log(f'[Note] velocity shift (i.e., redshift) and FWHM are tied following the input model_config.', log)
                msg += f'| Extinction (AV)                           = {tmp_values_vl["AV"][mask_l].mean():10.4f}'
                msg += f' +/- {tmp_values_vl["AV"].std():<8.4f}|\n'
                msg += f'| Powerlaw α_λ                              = {tmp_values_vl["powerlaw_alpha"][mask_l].mean():10.4f}'
                msg += f' +/- {tmp_values_vl["powerlaw_alpha"].std():<8.4f}|\n'
            else:
                print_log(f'Best-fit stellar properties of the sum of all components.', log)
            msg += f'| F{self.w_norm} (rest,extinct) ({self.spec_flux_scale} erg/s/cm2/AA) = {tmp_values_vl["flux_wavenorm"][mask_l].mean():10.4f}'
            msg += f' +/- {tmp_values_vl["loglambLum_wavenorm"].std():<8.4f}|\n'
            msg += f'| λL{self.w_norm} (rest,intrinsic) (log Lsun)        = {tmp_values_vl["loglambLum_wavenorm"][mask_l].mean():10.4f}'
            msg += f' +/- {tmp_values_vl["loglambLum_wavenorm"].std():<8.4f}|'
            bar = '=' * len(msg.split('|\n')[-1])
            print_log(bar, log)
            print_log(msg, log)
            print_log(bar, log)
        
