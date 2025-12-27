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
from scipy.interpolate import RegularGridInterpolator

from ..auxiliaries.auxiliary_frames import ConfigFrame
from ..auxiliaries.auxiliary_functions import print_log, casefold, color_list_dict

class TorusFrame(object): 
    def __init__(self, fframe=None, config=None, filename=None, 
                 v0_redshift=None, 
                 w_min=None, w_max=None, 
                 lum_norm=None, flux_scale=None, 
                 verbose=True, log_message=[]): 

        self.fframe = fframe
        self.config = config 
        self.filename = filename        
        self.v0_redshift = v0_redshift        
        self.w_min = w_min # currently not used
        self.w_max = w_max # currently not used
        self.flux_scale = flux_scale
        self.lum_norm = lum_norm if lum_norm is not None else 1e10 # normlize model by 1e10 Lsun
        self.verbose = verbose
        self.log_message = log_message

        self.cframe=ConfigFrame(self.config)
        self.num_comps = self.cframe.num_comps

        ############################################################
        # to be compatible with old version <= 2.2.4
        if len(self.cframe.par_index_cp[0]) == 0:
            self.cframe.par_name_cp = np.array([['voff', 'opt_depth_9.7', 'opening_angle', 'radii_ratio', 'inclination'] for i_comp in range(self.num_comps)])
            self.cframe.par_index_cp = [{'voff': 0, 'opt_depth_9.7': 1, 'opening_angle': 2, 'radii_ratio': 3, 'inclination': 4} for i_comp in range(self.num_comps)]
        for i_comp in range(self.num_comps):
            if 'opening_angle' in self.cframe.par_name_cp[i_comp]:
                self.cframe.par_name_cp[i_comp][self.cframe.par_name_cp[i_comp] == 'opening_angle'] = 'half_open_angle'
                self.cframe.par_index_cp[i_comp]['half_open_angle'] = self.cframe.par_index_cp[i_comp]['opening_angle']        
        ############################################################

        # set default info if not specified in config
        for i_comp in range(self.num_comps):
            if not ('lum_range' in [*self.cframe.info_c[i_comp]]) : self.cframe.info_c[i_comp]['lum_range'] = [(5,38), (8,1000), (1,1000)]
            # group line info to a list
            if isinstance(self.cframe.info_c[i_comp]['lum_range'], tuple): self.cframe.info_c[i_comp]['lum_range'] = [self.cframe.info_c[i_comp]['lum_range']]
            if isinstance(self.cframe.info_c[i_comp]['lum_range'], list):
                if all( isinstance(i, (int,float)) for i in self.cframe.info_c[i_comp]['lum_range'] ):
                    if len(self.cframe.info_c[i_comp]['lum_range']) == 2: self.cframe.info_c[i_comp]['lum_range'] = [self.cframe.info_c[i_comp]['lum_range']]

        # one independent element per component since disc and torus are tied
        self.num_coeffs_c = np.ones(self.num_comps, dtype='int')
        self.num_coeffs = self.num_coeffs_c.sum()

        # currently do not consider negative SED 
        self.mask_absorption_e = np.zeros((self.num_coeffs), dtype='bool')

        self.read_skirtor()

        # set plot styles
        self.plot_style_c = {}
        self.plot_style_c['sum'] = {'color': 'C8', 'alpha': 1, 'linestyle': '-', 'linewidth': 1.5}
        i_yellow = 0
        for i_comp in range(self.num_comps):
            self.plot_style_c[str(self.cframe.comp_c[i_comp])] = {'color': 'None', 'alpha': 0.5, 'linestyle': '--', 'linewidth': 1}
            self.plot_style_c[self.cframe.comp_c[i_comp]]['color'] = str(np.take(color_list_dict['yellow'], i_yellow, mode="wrap"))
            i_yellow += 1

        if self.verbose:
            print_log(f"SKIRTor torus model components: {np.array([self.cframe.info_c[i_comp]['mod_used'] for i_comp in range(self.num_comps)]).T}", self.log_message)
        
    def read_skirtor(self): 
        # https://sites.google.com/site/skirtorus/sed-library
        # skirtor_disc = np.loadtxt(self.file_disc) # [n_wave_ini+6, n_tau*n_oa*n_rrat*n_incl+1]
        # skirtor_torus = np.loadtxt(self.file_dust) # [n_wave_ini+6, n_tau*n_oa*n_rrat*n_incl+1]
        skirtor_lib = fits.open(self.filename)
        skirtor_disc = skirtor_lib[0].data[0]
        skirtor_torus = skirtor_lib[0].data[1]

        wave = skirtor_disc[6:-1,0] # 1e-3 to 1e3 um; omit the last one with zero-value SED
        n_tau = 5; tau = np.array([ 3, 5, 7, 9, 11 ])
        n_oa = 8; oa = np.array([ 10, 20, 30, 40, 50, 60, 70, 80 ])
        n_rrat = 3; rrat = np.array([ 10, 20, 30 ])
        n_incl = 10; incl = np.array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 ])
                
        disc  = np.zeros([n_tau, n_oa, n_rrat, n_incl, len(wave)]) 
        torus = np.zeros([n_tau, n_oa, n_rrat, n_incl, len(wave)]) 
        mass  = np.zeros([n_tau, n_oa, n_rrat]) # torus dust mass
        eb    = np.zeros([n_tau, n_oa, n_rrat]) 
        # All spectra are given in erg/s/um, normalized to disc lum of 1 Lsun
        # Not that the spectra in lum unit should be considered as flux * 4pi * dl2, 
        # where flux is depended on viewing angle and 
        # the 1 Lsun normalization is integrated with anisotropic flux function.
        # Dust mass in Msun
        # eb is energy balance ratio of torus, i.e., inclination integrated Lum_torus/Lum_AGN(intrinsic)

        for i_tau in range(n_tau):
            for i_oa in range(n_oa):
                for i_rrat in range(n_rrat):
                    for i_incl in range(n_incl):
                        mask = skirtor_disc[0,:] == tau[i_tau] 
                        mask &= skirtor_disc[1,:] == oa[i_oa] 
                        mask &= skirtor_disc[2,:] == rrat[i_rrat] 
                        mask &= skirtor_disc[3,:] == incl[i_incl] 
                        mass[i_tau, i_oa, i_rrat] = skirtor_torus[4,mask][0]
                        eb[i_tau, i_oa, i_rrat] = skirtor_torus[5,mask][0]
                        disc[i_tau, i_oa, i_rrat, i_incl, :] = skirtor_disc[6:-1,mask][:,0]
                        torus[i_tau, i_oa, i_rrat, i_incl, :] = skirtor_torus[6:-1,mask][:,0]
                        # in the original library the torus sed and mass is normalized to Lum_AGN of 1 Lsun, 
                        # here renormlized them to Lum_Torus of self.lum_norm Lsun (i.e., Lum_AGN = self.lum_norm Lsun / EB_Torus) 
                        disc[i_tau, i_oa, i_rrat, i_incl, :]  *= self.lum_norm / eb[i_tau, i_oa, i_rrat]
                        torus[i_tau, i_oa, i_rrat, i_incl, :] *= self.lum_norm / eb[i_tau, i_oa, i_rrat]
                        mass[i_tau, i_oa, i_rrat] *= self.lum_norm / eb[i_tau, i_oa, i_rrat]
        
        # convert unit: 1 erg/s/um -> flux_scale * erg/s/AA/cm2
        wave *= 1e4
        lum_dist = cosmo.luminosity_distance(self.v0_redshift).to('cm').value
        lum_area = 4*np.pi * lum_dist**2 # in cm2
        disc *= 1e-4 / lum_area / self.flux_scale
        torus *= 1e-4 / lum_area / self.flux_scale
        disc[disc <= 0]   = disc[disc>0].min()
        torus[torus <= 0] = torus[torus>0].min()
        
        # for interpolation
        ini_pars = (tau, oa, rrat, incl, np.log10(wave))    
        fun_logdisc = RegularGridInterpolator(ini_pars, np.log10(disc), method='linear', bounds_error=False)
        fun_logtorus = RegularGridInterpolator(ini_pars, np.log10(torus), method='linear', bounds_error=False)
        # set bounds_error=False to avoid error by slight exceeding of x-val generated by least_square func
        # but do not use pars outside of initial range
        ini_pars = (tau, oa, rrat)    
        fun_mass = RegularGridInterpolator(ini_pars, mass, method='linear', bounds_error=False)
        fun_eb = RegularGridInterpolator(ini_pars, eb, method='linear', bounds_error=False)

        self.skirtor = {'tau':tau, 'oa':oa, 'rratio':rrat, 'incl':incl, 
                        'wave':wave, 'log_wave':np.log10(wave), 
                        'disc':disc, 'fun_logdisc':fun_logdisc, 
                        'torus':torus, 'fun_logtorus':fun_logtorus, 
                        'mass':mass, 'fun_mass':fun_mass, 
                        'eb':eb, 'fun_eb':fun_eb } 
        
    def get_info(self, tau, oa, rratio):
        fun_mass = self.skirtor['fun_mass']
        fun_eb = self.skirtor['fun_eb']
        gen_pars = np.array([tau, oa, rratio])
        gen_mass = fun_mass(gen_pars)
        gen_eb   = fun_eb(gen_pars)
        return gen_mass, gen_eb

    def models_unitnorm_obsframe(self, wavelength, input_pars, if_pars_flat=True, mask_lite_e=None, conv_nbin=None):
        # conv_nbin is not used for emission lines, it is added to keep a uniform format with other models
        # par: voff (to adjust redshift), tau, oa, rratio, incl
        # comps: 'disc', 'torus'
        if if_pars_flat: 
            par_cp = self.cframe.flat_to_arr(input_pars)
        else:
            par_cp = copy(input_pars)

        for i_comp in range(par_cp.shape[0]):
            voff   = par_cp[i_comp, self.cframe.par_index_cp[i_comp]['voff']]
            tau    = par_cp[i_comp, self.cframe.par_index_cp[i_comp]['opt_depth_9.7']]
            rratio = par_cp[i_comp, self.cframe.par_index_cp[i_comp]['radii_ratio']]
            oa     = par_cp[i_comp, self.cframe.par_index_cp[i_comp]['half_open_angle']]
            incl   = par_cp[i_comp, self.cframe.par_index_cp[i_comp]['inclination']]
            
            # interpolate model for given pars in initial wavelength (rest)
            ini_logwave = self.skirtor['log_wave'].copy()
            fun_logdisc = self.skirtor['fun_logdisc']
            fun_logtorus = self.skirtor['fun_logtorus']
            gen_pars = np.array([[tau, oa, rratio, incl, w] for w in ini_logwave]) # gen: generated
            if 'disc' in casefold(self.cframe.info_c[i_comp]['mod_used']):
                gen_logdisc = fun_logdisc(gen_pars)
            if 'dust' in casefold(self.cframe.info_c[i_comp]['mod_used']):
                gen_logtorus = fun_logtorus(gen_pars)    

            # redshifted to obs-frame
            ret_logwave = np.log10(wavelength) # in AA
            z_ratio = (1 + self.v0_redshift) * (1 + voff/299792.458) # (1+z) = (1+zv0) * (1+v/c)            
            ini_logwave += np.log10(z_ratio)
            if 'disc' in casefold(self.cframe.info_c[i_comp]['mod_used']):
                gen_logdisc -= np.log10(z_ratio)
                ret_logdisc  = np.interp(ret_logwave, ini_logwave, gen_logdisc, 
                                         left=np.minimum(gen_logdisc.min(),-100), right=np.minimum(gen_logdisc.min(),-100))
            if 'dust' in casefold(self.cframe.info_c[i_comp]['mod_used']):
                gen_logtorus -= np.log10(z_ratio)
                ret_logtorus = np.interp(ret_logwave, ini_logwave, gen_logtorus, 
                                         left=np.minimum(gen_logtorus.min(),-100), right=np.minimum(gen_logtorus.min(),-100))
                
            # extended to longer wavelength
            mask_w = ret_logwave > ini_logwave[-1]
            if np.sum(mask_w) > 0:
                if 'disc' in casefold(self.cframe.info_c[i_comp]['mod_used']):
                    index = (gen_logdisc[-2]-gen_logdisc[-1]) / (ini_logwave[-2]-ini_logwave[-1])
                    ret_logdisc[mask_w] = gen_logdisc[-1] + index * (ret_logwave[mask_w]-ini_logwave[-1])
                if 'dust' in casefold(self.cframe.info_c[i_comp]['mod_used']):
                    index = (gen_logtorus[-2]-gen_logtorus[-1]) / (ini_logwave[-2]-ini_logwave[-1])
                    ret_logtorus[mask_w] = gen_logtorus[-1] + index * (ret_logwave[mask_w]-ini_logwave[-1])
                    
            if 'disc' in casefold(self.cframe.info_c[i_comp]['mod_used']):
                ret_disc = 10.0**ret_logdisc
                ret_disc[ret_logdisc <= -100] = 0
            if 'dust' in casefold(self.cframe.info_c[i_comp]['mod_used']):
                ret_torus = 10.0**ret_logtorus
                ret_torus[ret_logtorus <= -100] = 0
                
            obs_flux_scomp_ew = np.zeros_like(ret_logwave)
            if 'disc' in casefold(self.cframe.info_c[i_comp]['mod_used']): obs_flux_scomp_ew += ret_disc
            if 'dust' in casefold(self.cframe.info_c[i_comp]['mod_used']): obs_flux_scomp_ew += ret_torus
                
            obs_flux_scomp_ew = np.vstack((obs_flux_scomp_ew))
            obs_flux_scomp_ew = obs_flux_scomp_ew.T # add .T for a uniform format with other models with n_coeffs > 1
            
            if i_comp == 0: 
                obs_flux_mcomp_ew = obs_flux_scomp_ew
            else:
                obs_flux_mcomp_ew = np.vstack((obs_flux_mcomp_ew, obs_flux_scomp_ew))

        if mask_lite_e is not None:
            obs_flux_mcomp_ew = obs_flux_mcomp_ew[mask_lite_e,:]

        return obs_flux_mcomp_ew

    ##########################################################################
    ########################### Output functions #############################

    def extract_results(self, step=None, if_print_results=True, if_return_results=False, if_rev_v0_redshift=False, if_show_average=False, lum_unit='Lsun', **kwargs):

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

        mod = 'torus'
        fp0, fp1, fe0, fe1 = self.fframe.search_model_index(mod, self.fframe.full_model_type)
        spec_flux_scale = self.fframe.spec_flux_scale
        num_loops = self.fframe.num_loops
        comp_c = self.cframe.comp_c
        par_name_cp = self.cframe.par_name_cp
        num_comps = self.cframe.num_comps
        num_pars_per_comp = self.cframe.num_pars_c_max
        num_coeffs_c = self.num_coeffs_c
        num_coeffs_per_comp = self.num_coeffs_c[0] # components share the same num_coeffs

        # list the properties to be output; the print will follow this order
        value_names_additive = ['log_Lum_total']
        value_names_c = {}
        for i_comp in range(num_comps): 
            value_names_c[comp_c[i_comp]] = value_names_additive + [f"log_Lum_{lum_range[0]}_{lum_range[1]}" for lum_range in self.cframe.info_c[i_comp]['lum_range']]

        # format of results
        # output_c['comp']['par_lp'][i_l,i_p]: parameters
        # output_c['comp']['coeff_le'][i_l,i_e]: coefficients
        # output_c['comp']['values']['name_l'][i_l]: calculated values
        output_c = {}
        for i_comp in range(num_comps): 
            output_c[str(comp_c[i_comp])] = {} # init results for each comp
            output_c[comp_c[i_comp]]['par_lp']   = best_par_lp[:, fp0:fp1].reshape(num_loops, num_comps, num_pars_per_comp)[:, i_comp, :]
            output_c[comp_c[i_comp]]['coeff_le'] = best_coeff_le[:, fe0:fe1].reshape(num_loops, num_comps, num_coeffs_per_comp)[:, i_comp, :]
            output_c[comp_c[i_comp]]['values'] = {}
            for val_name in par_name_cp[i_comp].tolist() + value_names_c[comp_c[i_comp]]:
                output_c[comp_c[i_comp]]['values'][val_name] = np.zeros(num_loops, dtype='float')
        output_c['sum'] = {}
        output_c['sum']['values'] = {} # only init values for sum of all comp
        for val_name in value_names_additive:
            output_c['sum']['values'][val_name] = np.zeros(num_loops, dtype='float')

        i_e0 = 0; i_e1 = 0
        for i_comp in range(num_comps): 
            i_e0 += 0 if i_comp == 0 else num_coeffs_c[i_comp-1]
            i_e1 += num_coeffs_c[i_comp]
            for i_par in range(num_pars_per_comp):
                output_c[comp_c[i_comp]]['values'][par_name_cp[i_comp,i_par]] = output_c[comp_c[i_comp]]['par_lp'][:, i_par]
            for i_loop in range(num_loops):
                par_p = output_c[comp_c[i_comp]]['par_lp'][i_loop]
                coeff_e = output_c[comp_c[i_comp]]['coeff_le'][i_loop]

                lum_0 = coeff_e[0]*self.lum_norm
                if lum_unit == 'erg/s': lum_0 *= const.L_sun.to('erg/s').value
                output_c[comp_c[i_comp]]['values']['log_Lum_total'][i_loop] = np.log10(lum_0)
                output_c['sum']['values']['log_Lum_total'][i_loop] += lum_0

                voff = par_p[self.cframe.par_index_cp[i_comp]['voff']]
                rev_redshift = (1+voff/299792.458)*(1+self.v0_redshift)-1
                dist_lum = cosmo.luminosity_distance(rev_redshift).to('cm').value
                unitconv = 4*np.pi*dist_lum**2 * spec_flux_scale # convert intrinsic flux to Lum, in erg/s
                if lum_unit == 'Lsun': unitconv /= const.L_sun.to('erg/s').value

                for lum_range in self.cframe.info_c[i_comp]['lum_range']: 
                    tmp_wave_w = np.logspace(np.log10(lum_range[0]*1e4), np.log10(lum_range[1]*1e4), num=10000) # rest frame grid
                    tmp_torus_ew = self.models_unitnorm_obsframe(tmp_wave_w * (1+rev_redshift), best_par_lp[i_loop, fp0:fp1])
                    tmp_torus_w  = best_coeff_le[i_loop, fe0:fe1][i_e0:i_e1] @ tmp_torus_ew[i_e0:i_e1] # redshifted flux
                    tmp_torus_w *= 1+rev_redshift # to rest frame, in erg/s/cm2/AA
                    tmp_lum = np.trapezoid(tmp_torus_w, x=tmp_wave_w) * unitconv
                    tmp_name = f"log_Lum_{lum_range[0]}_{lum_range[1]}"
                    output_c[comp_c[i_comp]]['values'][tmp_name][i_loop] = np.log10(tmp_lum)
                    # output_c['sum']['values'][tmp_name][i_loop] += tmp_lum

        output_c['sum']['values']['log_Lum_total'] = np.log10(output_c['sum']['values']['log_Lum_total'])
        # for lum_range in self.cframe.info_c[i_comp]['lum_range']: 
        #     tmp_name = f"log_Lum_{lum_range[0]}_{lum_range[1]}"
        #     output_c['sum']['values'][tmp_name] = np.log10(output_c['sum']['values'][tmp_name])

        self.output_c = output_c # save to model frame
        self.num_loops = num_loops # for print_results
        self.spec_flux_scale = self.fframe.spec_flux_scale # to calculate luminosity in printing

        if if_print_results: self.print_results(log=self.fframe.log_message, if_show_average=if_show_average, lum_unit=lum_unit)
        if if_return_results: return output_c

    def print_results(self, log=[], if_show_average=False, lum_unit='Lsun'):
        print_log(f"#### Best-fit torus properties ####", log)

        mask_l = np.ones(self.num_loops, dtype='bool')
        if not if_show_average: mask_l[1:] = False
        lum_unit_str = '(log Lsun) ' if lum_unit == 'Lsun' else '(log erg/s)'

        # set the print name for each value
        value_names = [value_name for comp in self.output_c for value_name in [*self.output_c[comp]['values']]]
        value_names = list(dict.fromkeys(value_names)) # remove duplicates
        print_names = {}
        for value_name in value_names: print_names[value_name] = value_name
        print_names['voff'] = 'Velocity shift in relative to z_sys (km/s)'
        print_names['opt_depth_9.7'] = 'Optical depth at 9.7 µm'
        print_names['radii_ratio'] = 'Outer/inner radii ratio'
        print_names['half_open_angle'] = 'Half opening angle (degree)'
        print_names['inclination'] = 'Inclination (degree)'
        print_names['log_Lum_total'] = f"Torus Lum. (total) "+lum_unit_str
        for i_comp in range(self.cframe.num_comps): 
            for lum_range in self.cframe.info_c[i_comp]['lum_range']: 
                print_names[f"log_Lum_{lum_range[0]}_{lum_range[1]}"] = f"Torus Lum. "+f"({lum_range[0]}-{lum_range[1]} µm) "+lum_unit_str
        print_length = max([len(print_names[value_name]) for value_name in print_names] + [40]) # set min length
        for value_name in print_names:
            print_names[value_name] += ' '*(print_length-len(print_names[value_name]))

        for i_comp in range(len(self.output_c)):
            values_vl = self.output_c[[*self.output_c][i_comp]]['values']
            value_names = [*values_vl]
            msg = ''
            if i_comp < self.cframe.num_comps: # print best-fit pars for each comp
                
                print_log(f"# Torus component <{self.cframe.comp_c[i_comp]}>:", log)
                value_names = [value_name for value_name in value_names if value_name[:6] != 'Empty_'] # remove unused pars
                print_log(f"[Note] velocity shift (i.e., redshift) is tied following the input model_config.", log)
                value_names.remove('voff') # do not show the tied voff
            elif self.cframe.num_comps >= 2: # print sum only if using >= 2 comps
                print_log(f"# Best-fit properties of the sum of all torus components.", log)
            else: 
                continue
            for value_name in value_names:
                msg += '| ' + print_names[value_name] + f" = {values_vl[value_name][mask_l].mean():10.4f}" + f" +/- {values_vl[value_name].std():<10.4f}|\n"
            msg = msg[:-1] # remove the last \n
            bar = '=' * len(msg.split('\n')[-1])
            print_log(bar, log)
            print_log(msg, log)
            print_log(bar, log)
            print_log('', log)

        # if self.cframe.num_comps > 1:
        #     num_comps = len([*self.output_c])
        # else:
        #     num_comps = 1

        # print_log('', log)
        # msg = ''
        # for i_comp in range(num_comps):
        #     tmp_values_vl = self.output_c[[*self.output_c][i_comp]]['values']
        #     if i_comp < self.cframe.num_comps:
        #         print_log(f"Best-fit properties of torus component: <{self.cframe.comp_c[i_comp]}>", log)
        #         print_log(f"[Note] velocity shift (i.e., redshift) is tied following the input model_config.", log)
        #         # msg  = f"| Voff (km/s)                          = {tmp_values_vl['voff'][mask_l].mean():10.4f}'
        #         # msg += f" +/- {tmp_values_vl['voff'].std():<8.4f}|\n'
        #         msg += f"| Optical depth at 9.7 µm                = {tmp_values_vl['opt_depth_9.7'][mask_l].mean():10.4f}"
        #         msg += f" +/- {tmp_values_vl['opt_depth_9.7'].std():<8.4f}|\n"
        #         msg += f"| Outer/inner radii ratio                = {tmp_values_vl['radii_ratio'][mask_l].mean():10.4f}"
        #         msg += f" +/- {tmp_values_vl['radii_ratio'].std():<8.4f}|\n"
        #         msg += f"| Half opening angle (degree)            = {tmp_values_vl['opening_angle'][mask_l].mean():10.4f}"
        #         msg += f" +/- {tmp_values_vl['opening_angle'].std():<8.4f}|\n"
        #         msg += f"| Inclination (degree)                   = {tmp_values_vl['inclination'][mask_l].mean():10.4f}"
        #         msg += f" +/- {tmp_values_vl['inclination'].std():<8.4f}|\n"
        #     else:
        #         print_log(f"Best-fit stellar properties of the sum of all components.", log)
        #     for lum_range in self.cframe.info_c[i_comp]['lum_range']: 
        #         tmp_name = f"log_Lum_{lum_range[0]}_{lum_range[1]}"
        #         tmp_msg = f"({lum_range[0]}-{lum_range[1]} µm) "+lum_unit_str
        #         msg += f"| Torus Lum "+tmp_msg+' '*(28-len(tmp_msg))+f" = {tmp_values_vl[tmp_name][mask_l].mean():10.4f}"
        #         msg += f" +/- {tmp_values_vl[tmp_name].std():<8.4f}|\n"
        #     msg += f"| Torus Lum (total) "+lum_unit_str+f"          = {tmp_values_vl['log_Lum_total'][mask_l].mean():10.4f}"
        #     msg += f" +/- {tmp_values_vl['log_Lum_total'].std():<8.4f}|"

        #     bar = '=' * len(msg.split('|\n')[-1])
        #     print_log(bar, log)
        #     print_log(msg, log)
        #     print_log(bar, log)

