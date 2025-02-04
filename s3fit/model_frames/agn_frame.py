# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Contact: xiaoyang.chen.cz@gmail.com

import numpy as np
from copy import deepcopy as copy
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import WMAP9 as cosmo

from ..auxiliary_func import convert_linw_to_logw, convolve_spec_logw
from ..extinct_law import ExtLaw

class AGNFrame(object):
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
        
    # def read_bac(self)
    # def read_iron(self)
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

    ##########################################################################
    ########################## Output functions ##############################

    def output_results(self, ff=None):
        num_mock_loops = ff.num_mock_loops
        best_chi_sq_l = ff.best_chi_sq
        best_x_lp = ff.best_fits_x
        best_coeff_lm = ff.best_coeffs
        fx0, fx1, fc0, fc1 = ff.model_index('agn', ff.full_model_type)

        num_agn_comps = self.cframe.num_comps
        num_agn_pars = self.cframe.num_pars
        num_agn_coeffs = self.num_coeffs

        output_agn_lcp = np.zeros((num_mock_loops, num_agn_comps, 1 + num_agn_pars + num_agn_coeffs ))
        # p: chi_sq, output_values, ssp_coeffs
        output_agn_lcp[:, :, 0]                  = best_chi_sq_l[:, None]
        output_agn_lcp[:, :, 1:(1+num_agn_pars)] = best_x_lp[:, fx0:fx1].reshape(num_mock_loops, num_agn_comps, num_agn_pars)
        output_agn_lcp[:, :, -num_agn_coeffs:]   = best_coeff_lm[:, fc0:fc1].reshape(num_mock_loops, num_agn_comps, num_agn_coeffs)
        self.output_agn_lcp = output_agn_lcp # save to model frame
        self.spec_flux_scale = ff.spec_flux_scale # to calculate luminosity in printing

        # output to screen
        output_agn_vals = {
            'mean': np.average(output_agn_lcp, weights=1/best_chi_sq_l, axis=0), 
            'rms' : np.std(output_agn_lcp, axis=0, ddof=1) }
        self.output_agn_vals = output_agn_vals # to print result
        self.print_results()

    def print_results(self):
        dist_lum = cosmo.luminosity_distance(self.v0_redshift).to('cm').value
        unitconv = 4*np.pi*dist_lum**2 / const.L_sun.to('erg/s').value * self.spec_flux_scale # convert intrinsic flux5500(rest) to L5500
        num_agn_comps = self.cframe.num_comps

        print('')
        for i_comp in range(num_agn_comps):
            print(f'Best-fit properties of AGN component: <{self.cframe.comp_c[i_comp]}>')
            msg  = f'| Voff (km/s)                      = {self.output_agn_vals["mean"][i_comp,1]:10.4f}'
            msg += f' +/- {self.output_agn_vals["rms"][i_comp,1]:0.4f}\n'
            msg += f'| FWHM (km/s)                      = {self.output_agn_vals["mean"][i_comp,2]:10.4f}'
            msg += f' +/- {self.output_agn_vals["rms"][i_comp,2]:0.4f}\n'
            msg += f'| Extinction (AV)                  = {self.output_agn_vals["mean"][i_comp,3]:10.4f}'
            msg += f' +/- {self.output_agn_vals["rms"][i_comp,3]:0.4f}\n'
            msg += f'| Powerlaw α_λ                     = {self.output_agn_vals["mean"][i_comp,4]:10.4f}'
            msg += f' +/- {self.output_agn_vals["rms"][i_comp,4]:0.4f}\n'
            msg += f'| F{self.w_norm}(rest) ({self.spec_flux_scale} erg/s/cm2/AA) = {self.output_agn_vals["mean"][i_comp,5]:10.4f}'
            msg += f' +/- {self.output_agn_vals["rms"][i_comp,5]:0.4f}\n'
            msg += f'| L{self.w_norm}(rest) (1e8 Lsun/AA)        = {self.output_agn_vals["mean"][i_comp,5]*unitconv/1e8:10.4f}'
            msg += f' +/- {self.output_agn_vals["rms"][i_comp,5]*unitconv/1e8:0.4f}'
            bar = '='*60
            print(bar)
            print(msg)
            print(bar)
