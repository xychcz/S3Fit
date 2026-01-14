# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: s3fit@xychen.me

from copy import deepcopy as copy
import numpy as np
np.set_printoptions(linewidth=10000)
from scipy.interpolate import RegularGridInterpolator
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import astropy.constants as const

from ..auxiliaries.auxiliary_frames import ConfigFrame
from ..auxiliaries.auxiliary_functions import print_log, casefold, color_list_dict

class TorusFrame(object): 
    def __init__(self, mod_name=None, fframe=None, config=None, 
                 v0_redshift=None, 
                 w_min=None, w_max=None, 
                 verbose=True, log_message=[]): 

        self.mod_name = mod_name
        self.fframe = fframe
        self.config = config 
        self.v0_redshift = v0_redshift        
        self.w_min = w_min # currently not used
        self.w_max = w_max # currently not used
        self.verbose = verbose
        self.log_message = log_message

        self.cframe=ConfigFrame(self.config)
        self.comp_name_c = self.cframe.comp_name_c
        self.num_comps = self.cframe.num_comps
        self.check_config()

        # one independent element per component since disc and torus are tied
        self.num_coeffs_c = np.ones(self.num_comps, dtype='int')
        self.num_coeffs = self.num_coeffs_c.sum()

        # currently do not consider negative SED 
        self.mask_absorption_e = np.zeros((self.num_coeffs), dtype='bool')

        self.read_skirtor()

        if self.verbose:
            print_log(f"SKIRTor torus model components: {np.array([self.cframe.comp_info_cI[i_comp]['mod_used'] for i_comp in range(self.num_comps)]).T}", self.log_message)

        # set plot styles
        self.plot_style_C = {}
        self.plot_style_C['sum'] = {'color': 'C8', 'alpha': 0.75, 'linestyle': '-', 'linewidth': 1.5}
        i_yellow = 0
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            self.plot_style_C[comp_name] = {'color': 'None', 'alpha': 0.5, 'linestyle': '--', 'linewidth': 1}
            self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['yellow'], i_yellow, mode="wrap"))
            i_yellow += 1

    ##########################################################################

    def check_config(self):

        ############################################################
        # to be compatible with old version <= 2.2.4
        if len(self.cframe.par_index_cP[0]) == 0:
            self.cframe.par_name_cp  = [['voff', 'opt_depth_9.7', 'opening_angle', 'radii_ratio', 'inclination'] for i_comp in range(self.num_comps)]
            self.cframe.par_index_cP = [{'voff': 0, 'opt_depth_9.7': 1, 'opening_angle': 2, 'radii_ratio': 3, 'inclination': 4} for i_comp in range(self.num_comps)]
        for i_comp in range(self.num_comps):
            if 'opening_angle' in self.cframe.par_name_cp[i_comp]:
                self.cframe.par_name_cp[i_comp] = ['half_open_angle' if par_name == 'opening_angle' else par_name for par_name in self.cframe.par_name_cp[i_comp]]
                self.cframe.par_index_cP[i_comp]['half_open_angle'] = self.cframe.par_index_cP[i_comp]['opening_angle']        
        ############################################################

        # set inherited or default info if not specified in config
        # component-level info
        # format of returned flux / Lum density or integrated values
        for i_comp in range(self.num_comps):
            # either 2-unit-nested tuples (for wave and value, respectively) or dictionary as follows are supported
            self.cframe.retrieve_inherited_info('ret_emission_set', i_comp=i_comp, root_info_I=self.fframe.root_info_I, 
                                                default=[((5, 38, 'micron', 'rest'), ('intLum', 'L_sun', 'intrinsic')),
                                                         {'wave_min': 1, 'wave_max': 1000, 'wave_unit': 'micron', 'wave_frame': 'rest', 
                                                          'value_form': 'intLum', 'value_unit': 'L_sun', 'value_state': 'intrinsic'},
                                                        ])
            # 'wave_unit': any length unit supported by astropy.unit
            # 'value_form': 'Flam', 'lamFlam', 'Fnu', 'nuFnu', 'intFlux'; 'Llam', 'lamLlam', 'Lnu', 'nuLnu', intLum'
            # 'value_state': 'intrinsic', 'observed', 'absorbed' (i.e., dust absorbed)
            # 'value_unit': any flux/luminosity or its density unit supported by astropy.unit

            if self.cframe.comp_info_cI[i_comp]['ret_emission_set'] is None: continue # user can set None to skip all of these calculations
            # group line info to a list
            if isinstance(self.cframe.comp_info_cI[i_comp]['ret_emission_set'], (tuple, dict)): 
                self.cframe.comp_info_cI[i_comp]['ret_emission_set'] = [self.cframe.comp_info_cI[i_comp]['ret_emission_set']]
            for i_ret in range(len(self.cframe.comp_info_cI[i_comp]['ret_emission_set'])):
                tmp_format = self.cframe.comp_info_cI[i_comp]['ret_emission_set'][i_ret]
                # convert tuple format to dict
                if isinstance(tmp_format, tuple):
                    ret_emi_F = {}
                    wave_0, wave_1 = tmp_format[0][:2]
                    if wave_0 > wave_1:
                        ret_emi_F['wave_center'], ret_emi_F['wave_width'] = wave_0, wave_1
                    else:
                        ret_emi_F['wave_min'], ret_emi_F['wave_max'] = wave_0, wave_1 if wave_1 > wave_0 else wave_0+1

                    if len(tmp_format[0]) > 2: ret_emi_F['wave_unit']  = tmp_format[0][2]
                    if len(tmp_format[0]) > 3: ret_emi_F['wave_frame'] = tmp_format[0][3]
                    ret_emi_F['value_form'], ret_emi_F['value_unit'], ret_emi_F['value_state'] = tmp_format[1]
                elif isinstance(tmp_format, dict):
                    ret_emi_F = tmp_format
                # set default 
                if 'wave_unit'  not in ret_emi_F: ret_emi_F['wave_unit']  = 'micron'
                if 'wave_frame' not in ret_emi_F: ret_emi_F['wave_frame'] = 'rest'
                # check alternatives
                if ret_emi_F['wave_unit'] == 'A': ret_emi_F['wave_unit'] = 'angstrom'
                ret_emi_F['wave_frame'] = 'obs' if casefold(ret_emi_F['wave_frame']) in ['observed', 'obs'] else 'rest'
                if ret_emi_F['value_form'] == 'flam': ret_emi_F['value_form'] = 'Flam'
                if ret_emi_F['value_form'] == 'fnu' : ret_emi_F['value_form'] = 'Fnu'
                if casefold(ret_emi_F['value_state']) in ['intrinsic', 'original']:
                    ret_emi_F['value_state'] = 'intrinsic'
                elif casefold(ret_emi_F['value_state']) in ['observed', 'reddened', 'attenuated', 'extincted', 'extinct']:
                    ret_emi_F['value_state'] = 'observed'
                elif casefold(ret_emi_F['value_state']) in ['absorbed', 'dust']:
                    ret_emi_F['value_state'] = 'absorbed'
                self.cframe.comp_info_cI[i_comp]['ret_emission_set'][i_ret] = ret_emi_F

    ##########################################################################

    def read_skirtor(self): 
        # https://sites.google.com/site/skirtorus/sed-library
        # skirtor_disc = np.loadtxt(self.file_disc) # [n_wave_ini+6, n_tau*n_oa*n_rrat*n_incl+1]
        # skirtor_torus = np.loadtxt(self.file_dust) # [n_wave_ini+6, n_tau*n_oa*n_rrat*n_incl+1]
        for item in ['file', 'file_path']:
            if item in self.cframe.mod_info_I: torus_file = self.cframe.mod_info_I[item]
        skirtor_lib = fits.open(torus_file)
        skirtor_disc = skirtor_lib[0].data[0]
        skirtor_torus = skirtor_lib[0].data[1]

        self.init_wave_w = skirtor_disc[6:-1,0] # 1e-3 to 1e3 um; omit the last one with zero-value SED
        self.init_wave_unit = 'micron'
        self.init_wave_medium = 'vac' # not sensitive due to sampling rate

        n_tau = 5; tau = np.array([ 3, 5, 7, 9, 11 ])
        n_oa = 8; oa = np.array([ 10, 20, 30, 40, 50, 60, 70, 80 ])
        n_rrat = 3; rrat = np.array([ 10, 20, 30 ])
        n_incl = 10; incl = np.array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 ])
                
        disc  = np.zeros([n_tau, n_oa, n_rrat, n_incl, len(self.init_wave_w)]) 
        torus = np.zeros([n_tau, n_oa, n_rrat, n_incl, len(self.init_wave_w)]) 
        mass  = np.zeros([n_tau, n_oa, n_rrat]) # torus dust mass
        eb    = np.zeros([n_tau, n_oa, n_rrat]) 
        # All spectra are given in erg/s/um, normalized to disc lum of 1 Lsun
        # Not that the spectra in lum unit should be considered as flux * 4pi * dl2, 
        # where flux is depended on viewing angle and 
        # the 1 Lsun normalization is integrated with anisotropic flux function.
        # Dust mass in Msun
        # eb is energy balance ratio of torus, i.e., inclination integrated Lum_torus/Lum_AGN(intrinsic)

        # self.lum_norm = 1e10 # normlize model by 1e10 Lsun
        for i_tau in range(n_tau):
            for i_oa in range(n_oa):
                for i_rrat in range(n_rrat):
                    for i_incl in range(n_incl):
                        mask  = skirtor_disc[0,:] == tau[i_tau] 
                        mask &= skirtor_disc[1,:] == oa[i_oa] 
                        mask &= skirtor_disc[2,:] == rrat[i_rrat] 
                        mask &= skirtor_disc[3,:] == incl[i_incl] 
                        mass [i_tau, i_oa, i_rrat] = skirtor_torus[4,mask][0]
                        eb   [i_tau, i_oa, i_rrat] = skirtor_torus[5,mask][0]
                        disc [i_tau, i_oa, i_rrat, i_incl, :] = skirtor_disc [6:-1,mask][:,0]
                        torus[i_tau, i_oa, i_rrat, i_incl, :] = skirtor_torus[6:-1,mask][:,0]
                        # in the original library the torus sed and mass is normalized to Lum_AGN of 1 Lsun, 
                        # here renormlized them to Lum_Torus of self.lum_norm Lsun (i.e., Lum_AGN = self.lum_norm Lsun / EB_Torus) 
                        disc [i_tau, i_oa, i_rrat, i_incl, :] *= 1 / eb[i_tau, i_oa, i_rrat] # self.lum_norm
                        torus[i_tau, i_oa, i_rrat, i_incl, :] *= 1 / eb[i_tau, i_oa, i_rrat] # self.lum_norm
                        mass [i_tau, i_oa, i_rrat]            *= 1 / eb[i_tau, i_oa, i_rrat] # self.lum_norm

        # convert wave unit to angstrom in vacuum
        self.init_wave_w *= u.Unit(self.init_wave_unit).to('angstrom')
        self.init_wave_unit = 'angstrom'
        if self.init_wave_medium == 'air': self.init_wave_w = wave_air_to_vac(self.init_wave_w)

        # convert the normalization from per unit mass to per unit L5500
        self.init_lum_unit = 'erg s-1 micron-1'
        self.init_norm_e = 1 # norm is L_torus
        self.init_norm_unit = 'L_sun'
        self.init_mass_unit = 'M_sun'
        # update normalization to avoid too low values
        scale_lum_e = torus.max()/10
        # scale models by scale_lum_e * init_lum_unit
        disc  /= scale_lum_e
        torus /= scale_lum_e
        mass /= scale_lum_e
        self.init_norm_e /= scale_lum_e
        self.init_norm_unit = str(u.Unit(self.init_norm_unit) / u.Unit(self.init_lum_unit)) # update before init_lum_unit
        self.init_mass_unit = str(u.Unit(self.init_mass_unit) / u.Unit(self.init_lum_unit)) # update before init_lum_unit
        self.init_lum_unit  = str(u.dimensionless_unscaled) # the scaled model is in dimensionless unit

        # # convert unit: 1 erg/s/um -> spec_flux_scale * erg/s/angstrom/cm2
        # lum_dist = cosmo.luminosity_distance(self.v0_redshift).to('cm').value
        # lum_area = 4*np.pi * lum_dist**2 # in cm2
        # disc  *= 1e-4 / lum_area / self.fframe.spec_flux_scale
        # torus *= 1e-4 / lum_area / self.fframe.spec_flux_scale
        disc [disc  <= 0] = disc [disc  > 0].min()
        torus[torus <= 0] = torus[torus > 0].min()
        
        # for interpolation
        ini_pars = (tau, oa, rrat, incl, np.log10(self.init_wave_w))    
        fun_logdisc  = RegularGridInterpolator(ini_pars, np.log10(disc ), method='linear', bounds_error=False)
        fun_logtorus = RegularGridInterpolator(ini_pars, np.log10(torus), method='linear', bounds_error=False)
        # set bounds_error=False to avoid error by slight exceeding of x-val generated by least_square func
        # but do not use pars outside of initial range
        ini_pars = (tau, oa, rrat)    
        fun_mass = RegularGridInterpolator(ini_pars, mass, method='linear', bounds_error=False)
        fun_eb   = RegularGridInterpolator(ini_pars, eb,   method='linear', bounds_error=False)

        self.skirtor = {'tau':tau, 'oa':oa, 'rratio':rrat, 'incl':incl, 
                        'wave':self.init_wave_w, 'log_wave':np.log10(self.init_wave_w), 
                        'disc':disc, 'fun_logdisc':fun_logdisc, 
                        'torus':torus, 'fun_logtorus':fun_logtorus, 
                        'mass':mass, 'fun_mass':fun_mass, 
                        'eb':eb, 'fun_eb':fun_eb } 
        
    def get_info(self, tau, oa, rratio):
        fun_mass = self.skirtor['fun_mass']
        fun_eb   = self.skirtor['fun_eb']
        gen_pars = np.array([tau, oa, rratio])
        gen_mass = fun_mass(gen_pars)
        gen_eb   = fun_eb(gen_pars)
        return gen_mass, gen_eb

    ##########################################################################

    def create_models(self, obs_wave_w, par_p, mask_lite_e=None, components=None, 
                      if_dust_ext=False, if_ism_abs=False, if_igm_abs=False, 
                      if_redshift=True, if_convolve=False, conv_nbin=None, if_full_range=False): 

        # conv_nbin is not used for emission lines, it is added to keep a uniform format with other models
        # par: voff (to adjust redshift), tau, oa, rratio, incl
        # comps: 'disc', 'torus'
        par_cp = self.cframe.reshape_by_comp(par_p, self.cframe.num_pars_c)
        if isinstance(components, str): components = [components]

        obs_flux_mcomp_ew = None
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            if components is not None:
                if comp_name not in components: continue

            tau    = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['opt_depth_9.7']]
            rratio = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['radii_ratio']]
            oa     = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['half_open_angle']]
            incl   = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['inclination']]
            
            # interpolate model for given pars in initial wavelength (rest)
            ini_logwave  = self.skirtor['log_wave'].copy()
            fun_logdisc  = self.skirtor['fun_logdisc']
            fun_logtorus = self.skirtor['fun_logtorus']
            gen_pars = np.array([[tau, oa, rratio, incl, w] for w in ini_logwave]) # gen: generated
            if 'disc' in casefold(self.cframe.comp_info_cI[i_comp]['mod_used']):
                gen_logdisc  = fun_logdisc(gen_pars)
            if 'dust' in casefold(self.cframe.comp_info_cI[i_comp]['mod_used']):
                gen_logtorus = fun_logtorus(gen_pars)    

            # redshift models
            voff = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['voff']]
            z_ratio = (1 + self.v0_redshift) * (1 + voff/299792.458) # (1+z) = (1+zv0) * (1+v/c)
            ini_logwave += np.log10(z_ratio)
            if if_redshift:
                if 'disc' in casefold(self.cframe.comp_info_cI[i_comp]['mod_used']):
                    gen_logdisc  -= np.log10(z_ratio)
                if 'dust' in casefold(self.cframe.comp_info_cI[i_comp]['mod_used']):
                    gen_logtorus -= np.log10(z_ratio)

            # project to observed wavelength
            ret_logwave = np.log10(obs_wave_w) # in angstrom
            if 'disc' in casefold(self.cframe.comp_info_cI[i_comp]['mod_used']):
                ret_logdisc  = np.interp(ret_logwave, ini_logwave, gen_logdisc, 
                                         left=np.minimum(gen_logdisc.min(),-100), right=np.minimum(gen_logdisc.min(),-100))
            if 'dust' in casefold(self.cframe.comp_info_cI[i_comp]['mod_used']):
                ret_logtorus = np.interp(ret_logwave, ini_logwave, gen_logtorus, 
                                         left=np.minimum(gen_logtorus.min(),-100), right=np.minimum(gen_logtorus.min(),-100))

            # extended to longer wavelength
            mask_w = ret_logwave > ini_logwave[-1]
            if np.sum(mask_w) > 0:
                if 'disc' in casefold(self.cframe.comp_info_cI[i_comp]['mod_used']):
                    index = (gen_logdisc[-2]-gen_logdisc[-1]) / (ini_logwave[-2]-ini_logwave[-1])
                    ret_logdisc[mask_w] = gen_logdisc[-1] + index * (ret_logwave[mask_w]-ini_logwave[-1])
                if 'dust' in casefold(self.cframe.comp_info_cI[i_comp]['mod_used']):
                    index = (gen_logtorus[-2]-gen_logtorus[-1]) / (ini_logwave[-2]-ini_logwave[-1])
                    ret_logtorus[mask_w] = gen_logtorus[-1] + index * (ret_logwave[mask_w]-ini_logwave[-1])
                    
            if 'disc' in casefold(self.cframe.comp_info_cI[i_comp]['mod_used']):
                ret_disc = 10.0**ret_logdisc
                ret_disc[ret_logdisc <= -100] = 0
            if 'dust' in casefold(self.cframe.comp_info_cI[i_comp]['mod_used']):
                ret_torus = 10.0**ret_logtorus
                ret_torus[ret_logtorus <= -100] = 0
                
            obs_flux_scomp_ew = np.zeros_like(ret_logwave)
            if 'disc' in casefold(self.cframe.comp_info_cI[i_comp]['mod_used']): obs_flux_scomp_ew += ret_disc
            if 'dust' in casefold(self.cframe.comp_info_cI[i_comp]['mod_used']): obs_flux_scomp_ew += ret_torus
                
            obs_flux_scomp_ew = np.vstack((obs_flux_scomp_ew))
            obs_flux_scomp_ew = obs_flux_scomp_ew.T # add .T for a uniform format with other models with n_coeffs > 1
            
            if obs_flux_mcomp_ew is None: 
                obs_flux_mcomp_ew = obs_flux_scomp_ew
            else:
                obs_flux_mcomp_ew = np.vstack((obs_flux_mcomp_ew, obs_flux_scomp_ew))

        if mask_lite_e is not None:
            obs_flux_mcomp_ew = obs_flux_mcomp_ew[mask_lite_e,:]

        return obs_flux_mcomp_ew

    ##########################################################################
    ########################### Output functions #############################

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
        comp_name_c = self.cframe.comp_name_c
        num_comps = self.cframe.num_comps
        par_name_cp = self.cframe.par_name_cp

        # list the properties to be output; the print will follow this order
        value_names_additive = ['Lumfrac_dust', 'log_Mass_dust', 'log_intLum_dust_bol', 'log_intLum_agn_bol']
        ret_names_additive = None
        value_names_C = {}
        for (i_comp, comp_name) in enumerate(comp_name_c):
            value_names_C[comp_name] = value_names_additive + [] # just copy

            ret_names = []
            if self.cframe.comp_info_cI[i_comp]['ret_emission_set'] is not None:
                for ret_emi_F in self.cframe.comp_info_cI[i_comp]['ret_emission_set']:
                    if 'wave_center' in ret_emi_F:
                        wave_name = f"{ret_emi_F['wave_center']} ({ret_emi_F['wave_frame']}, {ret_emi_F['wave_unit']})"
                    else:
                        wave_name = f"{ret_emi_F['wave_min']}-{ret_emi_F['wave_max']} ({ret_emi_F['wave_frame']}, {ret_emi_F['wave_unit']})"
                    ret_name = f"log_{ret_emi_F['value_form']} ({ret_emi_F['value_state']}, {ret_emi_F['value_unit']}) at {wave_name}"
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
                lum_area = 4*np.pi * cosmo.luminosity_distance(rev_redshift).to('cm')**2 # with unit of cm2
 
                lum_torus = (coeff_e[0] * u.Unit(self.fframe.spec_flux_unit) * lum_area * self.init_norm_e * u.Unit(self.init_norm_unit)).to('L_sun').value
                # coeff_e[0]*self.lum_norm # default unit is Lsun
                output_C[comp_name]['value_Vl']['log_intLum_dust_bol'][i_loop] = np.log10(lum_torus)
                output_C['sum']['value_Vl']['log_intLum_dust_bol'][i_loop] += lum_torus

                tau    = par_p[self.cframe.par_index_cP[i_comp]['opt_depth_9.7']]
                rratio = par_p[self.cframe.par_index_cP[i_comp]['radii_ratio']]
                oa     = par_p[self.cframe.par_index_cP[i_comp]['half_open_angle']]
                mass, eb = self.get_info(tau, oa, rratio)
                mass_torus = (coeff_e[0] * u.Unit(self.fframe.spec_flux_unit) * lum_area * mass * u.Unit(self.init_mass_unit)).to('M_sun').value
                lum_agn = lum_torus / eb
                output_C[comp_name]['value_Vl']['Lumfrac_dust'][i_loop] = eb
                output_C[comp_name]['value_Vl']['log_Mass_dust'][i_loop] = np.log10(mass_torus)
                output_C[comp_name]['value_Vl']['log_intLum_agn_bol'][i_loop] = np.log10(lum_agn)
                output_C['sum']['value_Vl']['Lumfrac_dust'][i_loop] += eb
                output_C['sum']['value_Vl']['log_Mass_dust'][i_loop] += mass_torus
                output_C['sum']['value_Vl']['log_intLum_agn_bol'][i_loop] += lum_agn

                tmp_coeff_e = best_coeff_le[i_loop, i_coeffs_0_of_mod:i_coeffs_1_of_mod][i_coeffs_0_of_comp_in_mod:i_coeffs_1_of_comp_in_mod]
                # calculate requested flux/Lum in given wavelength ranges
                if self.cframe.comp_info_cI[i_comp]['ret_emission_set'] is not None: 
                    for ret_emi_F in self.cframe.comp_info_cI[i_comp]['ret_emission_set']:
                        if 'wave_center' in ret_emi_F:
                            wave_0, wave_1 = ret_emi_F['wave_center'] - ret_emi_F['wave_width'], ret_emi_F['wave_center'] + ret_emi_F['wave_width']
                            wave_name = f"{ret_emi_F['wave_center']} ({ret_emi_F['wave_frame']}, {ret_emi_F['wave_unit']})"
                        else:
                            wave_0, wave_1 = ret_emi_F['wave_min'], ret_emi_F['wave_max']
                            wave_name = f"{ret_emi_F['wave_min']}-{ret_emi_F['wave_max']} ({ret_emi_F['wave_frame']}, {ret_emi_F['wave_unit']})"
                        wave_ratio = u.Unit(ret_emi_F['wave_unit']).to('angstrom')
                        if ret_emi_F['wave_frame'] == 'rest': wave_ratio *= (1+rev_redshift) # rest wave to obs wave, which is required by create_models
                        tmp_wave_w = np.logspace(np.log10(wave_0*wave_ratio), np.log10(wave_1*wave_ratio), num=1000) # obs frame grid

                        if ret_emi_F['value_state'] in ['intrinsic','absorbed']:
                            tmp_flux_ew = self.create_models(tmp_wave_w, best_par_lp[i_loop, i_pars_0_of_mod:i_pars_1_of_mod], components=comp_name, 
                                                             if_dust_ext=False, if_redshift=True, if_full_range=True) # flux in obs frame
                            intrinsic_flux_w = tmp_coeff_e @ tmp_flux_ew
                        if ret_emi_F['value_state'] in ['observed','absorbed']:
                            tmp_flux_ew = self.create_models(tmp_wave_w, best_par_lp[i_loop, i_pars_0_of_mod:i_pars_1_of_mod], components=comp_name, 
                                                             if_dust_ext=True,  if_redshift=True, if_full_range=True) # flux in obs frame
                            observed_flux_w = tmp_coeff_e @ tmp_flux_ew
                        if ret_emi_F['value_state'] == 'intrinsic': tmp_flux_w = intrinsic_flux_w
                        if ret_emi_F['value_state'] == 'observed' : tmp_flux_w = observed_flux_w
                        if ret_emi_F['value_state'] == 'absorbed' : tmp_flux_w = intrinsic_flux_w - observed_flux_w

                        tmp_wave_w *= u.angstrom
                        tmp_flux_w *= u.Unit(self.fframe.spec_flux_unit)
                        tmp_Flam    = tmp_flux_w.mean()
                        tmp_lamFlam = tmp_flux_w.mean() * tmp_wave_w.mean()
                        tmp_intFlux = np.trapezoid(tmp_flux_w, x=tmp_wave_w)

                        if ret_emi_F['value_form'] ==     'Flam'          : ret_value = tmp_Flam
                        if ret_emi_F['value_form'] in ['lamFlam', 'nuFnu']: ret_value = tmp_lamFlam
                        if ret_emi_F['value_form'] ==  'intFlux'          : ret_value = tmp_intFlux

                        if ret_emi_F['value_form'] ==     'Llam'          : ret_value = tmp_Flam    * lum_area
                        if ret_emi_F['value_form'] in ['lamLlam', 'nuLnu']: ret_value = tmp_lamFlam * lum_area
                        if ret_emi_F['value_form'] ==  'intLum'           : ret_value = tmp_intFlux * lum_area

                        if ret_emi_F['value_form'] ==     'Fnu'           : ret_value = tmp_Flam               * tmp_wave_w.mean()**2 / const.c
                        if ret_emi_F['value_form'] ==     'Lnu'           : ret_value = tmp_Flam    * lum_area * tmp_wave_w.mean()**2 / const.c

                        ret_value = ret_value.to(ret_emi_F['value_unit']).value
                        ret_name = f"log_{ret_emi_F['value_form']} ({ret_emi_F['value_state']}, {ret_emi_F['value_unit']}) at {wave_name}"
                        output_C[comp_name]['value_Vl'][ret_name][i_loop] = np.log10(ret_value)
                        if ret_name in output_C['sum']['value_Vl']: output_C['sum']['value_Vl'][ret_name][i_loop] += ret_value

        for value_name in output_C['sum']['value_Vl']:
            if (value_name[:8] in ['log_Flam', 'log_Fnu ', 'log_Mass']) | (value_name[:11] in ['log_lamFlam', 'log_intFlux', 'log_lamLlam', 'log_intLum ']): 
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
        print_log(f"#### Best-fit torus properties ####", log)

        mask_l = np.ones(self.num_loops, dtype='bool')
        if not if_show_average: mask_l[1:] = False

        # set the print name for each value
        print_name_CV = {}
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            print_name_CV[comp_name] = {}
            for value_name in self.output_C[comp_name]['value_Vl']: print_name_CV[comp_name][value_name] = value_name

            print_name_CV[comp_name]['voff'] = 'Velocity shift in relative to z_sys (km s-1)'
            print_name_CV[comp_name]['opt_depth_9.7'] = 'Optical depth at 9.7 µm'
            print_name_CV[comp_name]['radii_ratio'] = 'Outer/inner radii ratio'
            print_name_CV[comp_name]['half_open_angle'] = 'Half opening angle (degree)'
            print_name_CV[comp_name]['inclination'] = 'Inclination (degree)'
            print_name_CV[comp_name]['Lumfrac_dust'] = f"Torus dust radiation absorption fraction"
            print_name_CV[comp_name]['log_Mass_dust'] = f"Torus dust mass (log M☉)"
            print_name_CV[comp_name]['log_intLum_dust_bol'] = f"Torus dust bolometric lum. (log L☉)"
            print_name_CV[comp_name]['log_intLum_agn_bol'] = f"AGN disc bolometric lum. (log L☉)"

            if self.cframe.comp_info_cI[i_comp]['ret_emission_set'] is not None: 
                for ret_emi_F in self.cframe.comp_info_cI[i_comp]['ret_emission_set']:
                    if 'wave_center' in ret_emi_F:
                        wave_name = f"{ret_emi_F['wave_center']} ({ret_emi_F['wave_frame']}, {ret_emi_F['wave_unit']})"
                    else:
                        wave_name = f"{ret_emi_F['wave_min']}-{ret_emi_F['wave_max']} ({ret_emi_F['wave_frame']}, {ret_emi_F['wave_unit']})"
                    ret_name = f"log_{ret_emi_F['value_form']} ({ret_emi_F['value_state']}, {ret_emi_F['value_unit']}) at {wave_name}"

                    tmp_value_state = copy(ret_emi_F['value_state']) # avoid changing the original set, which is used elsewhere
                    if tmp_value_state == 'absorbed' : tmp_value_state = 'dust-'+tmp_value_state
                    print_name_CV[comp_name][ret_name] = f"{tmp_value_state.capitalize()} "

                    ret_emi_F['value_unit'] = ret_emi_F['value_unit'].replace('angstrom', 'Å').replace('Angstrom', 'Å').replace('um', 'µm').replace('micron', 'µm').replace('L_sun', 'L☉')
                    if ret_emi_F['value_form'] ==    'Flam': print_name_CV[comp_name][ret_name] += f"flux density (Fλ, log {ret_emi_F['value_unit']})"
                    if ret_emi_F['value_form'] ==    'Llam': print_name_CV[comp_name][ret_name] += f"lum. density (Lλ, log {ret_emi_F['value_unit']})"
                    if ret_emi_F['value_form'] ==    'Fnu' : print_name_CV[comp_name][ret_name] += f"flux density (Fν, log {ret_emi_F['value_unit']})"
                    if ret_emi_F['value_form'] ==    'Lnu' : print_name_CV[comp_name][ret_name] += f"lum. density (Lν, log {ret_emi_F['value_unit']})"
                    if ret_emi_F['value_form'] == 'lamFlam': print_name_CV[comp_name][ret_name] += f"flux (λFλ, log {ret_emi_F['value_unit']})"
                    if ret_emi_F['value_form'] == 'lamLlam': print_name_CV[comp_name][ret_name] += f"lum. (λLλ, log {ret_emi_F['value_unit']})"
                    if ret_emi_F['value_form'] ==  'nuFnu' : print_name_CV[comp_name][ret_name] += f"flux (νFν, log {ret_emi_F['value_unit']})"
                    if ret_emi_F['value_form'] ==  'nuLnu' : print_name_CV[comp_name][ret_name] += f"lum. (νLν, log {ret_emi_F['value_unit']})"
                    if ret_emi_F['value_form'] == 'intFlux': print_name_CV[comp_name][ret_name] += f"flux (integrated, log {ret_emi_F['value_unit']})"
                    if ret_emi_F['value_form'] == 'intLum' : print_name_CV[comp_name][ret_name] += f"lum. (integrated, log {ret_emi_F['value_unit']})"

                    tmp_wave_unit = ret_emi_F['wave_unit'].replace('angstrom', 'Å').replace('Angstrom', 'Å').replace('um', 'µm').replace('micron', 'µm')
                    tmp_wave_frame = copy(ret_emi_F['wave_frame'])
                    if tmp_wave_frame == 'obs': tmp_wave_frame += '.'
                    if 'wave_center' in ret_emi_F:
                        print_name_CV[comp_name][ret_name] += f" at {tmp_wave_frame} {ret_emi_F['wave_center']} {tmp_wave_unit}"
                    else:
                        print_name_CV[comp_name][ret_name] += f" at {tmp_wave_frame} {ret_emi_F['wave_min']}-{ret_emi_F['wave_max']} {tmp_wave_unit}"

        print_name_CV['sum'] = {}
        for value_name in self.output_C['sum']['value_Vl']:
            print_name_CV['sum'][value_name] = copy(print_name_CV[self.comp_name_c[0]][value_name])

        print_length = max([len(print_name_CV[comp_name][value_name]) for comp_name in print_name_CV for value_name in print_name_CV[comp_name]] + [40]) # set min length
        for comp_name in print_name_CV:
            for value_name in print_name_CV[comp_name]:
                print_name_CV[comp_name][value_name] += ' '*(print_length-len(print_name_CV[comp_name][value_name]))

        for (i_comp, comp_name) in enumerate(self.output_C):
            value_Vl = self.output_C[comp_name]['value_Vl']
            value_names = [*value_Vl]
            msg = ''
            if i_comp < self.cframe.num_comps: # print best-fit pars for each comp
                print_log(f"# Torus component <{self.cframe.comp_name_c[i_comp]}>:", log)
                value_names = [value_name for value_name in value_names if value_name[:6] != 'Empty_'] # remove unused pars
                print_log(f"[Note] velocity shift (i.e., redshift) is tied following the input model_config.", log)
                value_names.remove('voff') # do not show the tied voff
            elif self.cframe.num_comps >= 2: # print sum only if using >= 2 comps
                print_log(f"# Best-fit properties of the sum of all torus components.", log)
            else: 
                continue
            for value_name in value_names:
                msg += '| ' + print_name_CV[comp_name][value_name] + f" = {value_Vl[value_name][mask_l].mean():10.4f}" + f" +/- {value_Vl[value_name].std():<10.4f}|\n"
            msg = msg[:-1] # remove the last \n
            bar = '=' * len(msg.split('\n')[-1])
            print_log(bar, log)
            print_log(msg, log)
            print_log(bar, log)
            print_log('', log)

