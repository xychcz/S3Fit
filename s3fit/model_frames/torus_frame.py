# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: s3fit@xychen.me

from copy import deepcopy as copy
import numpy as np
np.set_printoptions(linewidth=10000)
from scipy.interpolate import RegularGridInterpolator, interp1d
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import astropy.constants as const

from ..auxiliaries.auxiliary_frames import ConfigFrame
from ..auxiliaries.auxiliary_functions import print_log, casefold, color_list_dict

class TorusFrame(object): 
    def __init__(self, mod_name=None, fframe=None, config=None, 
                 v0_redshift=None, R_inst_rw=None, 
                 wave_min=None, wave_max=None, 
                 verbose=True, log_message=[]): 

        self.mod_name = mod_name
        self.fframe = fframe
        self.config = config

        self.v0_redshift = v0_redshift 
        self.R_inst_rw = R_inst_rw        
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.verbose = verbose
        self.log_message = log_message

        self.cframe=ConfigFrame(self.config)
        self.comp_name_c = self.cframe.comp_name_c
        self.num_comps = self.cframe.num_comps
        self.check_config()

        # check if the requested range (wave_min,wave_max) is within the defined range
        if 'disc' in [mod_used for i_comp in range(self.num_comps) for mod_used in self.cframe.comp_info_cI[i_comp]['mod_used']]:
            self.wave_min_def, self.wave_max_def = 912, 1e7 # angstrom
        else:
            self.wave_min_def, self.wave_max_def = 1e4, 1e7 # angstrom
        self.enable = (self.wave_max > self.wave_min_def) & (self.wave_min < self.wave_max_def)

        # load template library
        self.read_torus_library()

        # one independent element per component, since disc and dust spectra are tied for a single component
        self.num_coeffs_c = np.ones(self.num_comps, dtype='int')
        self.num_coeffs_C = {comp_name: num_coeffs for (comp_name, num_coeffs) in zip(self.comp_name_c, self.num_coeffs_c)}
        self.num_coeffs_tot = sum(self.num_coeffs_c)

        self.num_coeffs_max = self.num_coeffs_tot if self.cframe.mod_info_I['num_coeffs_max'] is None else min(self.num_coeffs_tot, self.cframe.mod_info_I['num_coeffs_max'])
        # self.num_coeffs_max = max(self.num_coeffs_max, 3**len(self.init_par_uniq_Pu)) # set min num for each template par # only if nonpara

        # currently do not consider negative SED 
        self.mask_absorption_e = np.zeros((self.num_coeffs_tot), dtype='bool')

        if self.verbose:
            print_log(f"Torus model components: {[self.cframe.comp_info_cI[i_comp]['mod_used'] for i_comp in range(self.num_comps)]}", self.log_message)

        # set plot styles
        self.plot_style_C = {}
        self.plot_style_C['tot'] = {'color': 'C8', 'alpha': 0.75, 'linestyle': '-', 'linewidth': 1.5}
        i_yellow = 0
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            self.plot_style_C[comp_name] = {'color': 'None', 'alpha': 0.5, 'linestyle': '--', 'linewidth': 1}
            self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['yellow'], i_yellow, mode="wrap"))
            i_yellow += 1

    ##########################################################################

    def check_config(self):

        # check alternative model names
        for i_comp in range(self.num_comps):
            mod_used = []
            if any( np.isin(casefold(self.cframe.comp_info_cI[i_comp]['mod_used']), ['disc', 'disk', 'agn']) ): 
                mod_used.append('disc')
            if any( np.isin(casefold(self.cframe.comp_info_cI[i_comp]['mod_used']), ['dust', 'torus']) ): 
                mod_used.append('dust')
            self.cframe.comp_info_cI[i_comp]['mod_used'] = mod_used

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
        # model-level info, call as self.cframe.mod_info_I[info_name]
        self.cframe.retrieve_inherited_info('num_coeffs_max', root_info_I=self.fframe.root_info_I, default=None)
        self.cframe.retrieve_inherited_info('interp_space'  , root_info_I=self.fframe.root_info_I, default='log')

        # component-level info, call as self.cframe.comp_info_cI[i_comp][info_name]
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
                elif casefold(ret_emi_F['value_state']) in ['absorbed', 'dust absorbed', 'dust-absorbed']:
                    ret_emi_F['value_state'] = 'absorbed'
                self.cframe.comp_info_cI[i_comp]['ret_emission_set'][i_ret] = ret_emi_F

    ##########################################################################

    def read_torus_library(self): 
        # https://sites.google.com/site/skirtorus/sed-library
        for item in ['file', 'file_path']:
            if item in self.cframe.mod_info_I: torus_file = self.cframe.mod_info_I[item]
        skirtor_lib = fits.open(torus_file)
        skirtor_disc = skirtor_lib[0].data[0].T # [1 + n_tau_si * n_h_open * n_r_ratio * n_incl, 6 + n_wave_ini]
        skirtor_dust = skirtor_lib[0].data[1].T # [1 + n_tau_si * n_h_open * n_r_ratio * n_incl, 6 + n_wave_ini]

        # mimic the user input from mod_info_I
        template = {}
        template['wave'] = {'value': skirtor_disc[0, 6:], 'unit': 'micron', 'medium': None} # 1e-3 to 1e3 micron
        template['pars'] = {'opt_depth_9.7': skirtor_disc[1:, 0], 'radii_ratio': skirtor_disc[1:, 2], 'half_open_angle': skirtor_disc[1:, 1], 'inclination': skirtor_disc[1:, 3]}
        template['spec'] = {'disc': skirtor_disc[1:, 6:], 'dust': skirtor_dust[1:, 6:], 'unit': 'erg s-1 micron-1'}
        # here 'spec' is short for spectral density
        ######### convert if spec given in fnu

        # attributes
        template['mass_dust' ] = {'value': skirtor_disc[1:, 4]              , 'unit': 'M_sun', 'additive': True } 
        template['frac_abs'  ] = {'value': skirtor_disc[1:, 5]              , 'unit': ''     , 'additive': False}
        template['intLum_agn'] = {'value': np.ones(len(skirtor_disc[1:, 5])), 'unit': 'L_sun', 'additive': True }
        # in the original library the sed and mass_dust is normalized to Lum_AGN of 1 Lsun; Lum_dust = frac_abs * Lum_AGN

        ###############################

        # internal recording
        self.init_spec_wave_w       = template['wave']['value']
        self.init_spec_wave_unit    = template['wave']['unit']
        self.init_spec_wave_medium  = template['wave']['medium']

        self.init_par_Pe       = template['pars']
        self.num_templates     = len(list(self.init_par_Pe.values())[0])

        self.init_spec_dens_B = {key: {'value_ew': value,          'unit': template['spec']['unit']}                     for key, value in template['spec'].items() if key not in ['unit']}
        self.init_attribute_A = {key: {'value_e' : value['value'], 'unit': value['unit'], 'additive': value['additive']} for key, value in template.items()         if key not in ['wave', 'spec', 'pars']}

        # self.init_spec_wave_w
        # self.init_spec_wave_unit
        # self.init_spec_dens_B[bloc_name]['value_ew'] / ['unit']
        # self.init_attribute_A[attr_name]['value_e' ] / ['unit'] / ['additive']
        # 'additive' is short for 'if_is_additive'

        ###############################

        # convert wave unit to angstrom in vacuum
        self.init_spec_wave_w *= u.Unit(self.init_spec_wave_unit).to('angstrom')
        self.init_spec_wave_unit = 'angstrom'
        if self.init_spec_wave_medium == 'air': self.init_spec_wave_w = wave_air_to_vac(self.init_spec_wave_w)

        ###############################

        # add bloc_name of 'tot' if it does not exist
        if 'tot' not in self.init_spec_dens_B.keys():
            tot_value_ew = np.zeros((self.num_templates, len(self.init_spec_wave_w)))
            for bloc_dict in self.init_spec_dens_B.values():
                tot_value_ew += bloc_dict['value_ew']
            self.init_spec_dens_B['tot'] = {'value_ew': tot_value_ew, 'unit': bloc_dict['unit']}

        # update normalization to avoid outlier values
        scale_spec_dens_e    =      self.init_spec_dens_B['tot']['value_ew'].max(axis=1) * 1e-2
        scale_spec_dens_unit = copy(self.init_spec_dens_B['tot']['unit'    ])
        # scale models by scale_spec_dens_e * scale_spec_dens_unit
        for bloc_name, bloc_dict in self.init_spec_dens_B.items():
            bloc_dict['value_ew'] =            bloc_dict['value_ew']  /        scale_spec_dens_e[:, None]
            bloc_dict['unit'    ] = str(u.Unit(bloc_dict['unit'    ]) / u.Unit(scale_spec_dens_unit)) # dimensionless unscaled
            self.init_spec_dens_B[bloc_name] = bloc_dict
        for attr_name, attr_dict in self.init_attribute_A.items():
            attr_dict['orig_unit'] = copy(attr_dict['unit']) # for the output
            if not attr_dict['additive']: continue # skip if the attribute is not additive
            attr_dict['value_e' ] =            attr_dict['value_e' ]  /        scale_spec_dens_e
            attr_dict['unit'    ] = str(u.Unit(attr_dict['unit'    ]) / u.Unit(scale_spec_dens_unit))
            self.init_attribute_A[attr_name] = attr_dict

        ###############################

        # special treatment for this torus model
        # extended to longer wavelength
        if self.wave_max > self.init_spec_wave_w[-1]:
            ext_spec_wave_logbin = np.log10(self.init_spec_wave_w[-1] / self.init_spec_wave_w[-2])
            ext_spec_wave_num = max(2, 1+int(np.log10(self.wave_max / self.init_spec_wave_w[-1]) / ext_spec_wave_logbin))
            ext_spec_wave_w = np.logspace(np.log10(self.init_spec_wave_w[-1]), np.log10(self.wave_max), ext_spec_wave_num)[1:]

            for bloc_name, bloc_dict in self.init_spec_dens_B.items():
                i_w = -2 # avoid zero values at the end
                index_e = np.log10(bloc_dict['value_ew'][:,i_w-1] / bloc_dict['value_ew'][:,i_w]) / np.log10(self.init_spec_wave_w[None,i_w-1] / self.init_spec_wave_w[None,i_w])
                ext_spec_dens_ew  = (ext_spec_wave_w / self.init_spec_wave_w[i_w])[None,:] ** index_e[:,None] * bloc_dict['value_ew'][:,i_w][:,None]
                bloc_dict['value_ew'] = np.hstack((bloc_dict['value_ew'], ext_spec_dens_ew))
                self.init_spec_dens_B[bloc_name] = bloc_dict

            self.init_spec_wave_w = np.hstack((self.init_spec_wave_w, ext_spec_wave_w))

        # avoid non-positive values
        self.spec_dens_min = 0
        if self.cframe.mod_info_I['interp_space'] == 'log':
            for bloc_name, bloc_dict in self.init_spec_dens_B.items():
                tmp_min = bloc_dict['value_ew'][bloc_dict['value_ew'] > 0].min() * 1e-4
                self.spec_dens_min = min(self.spec_dens_min, tmp_min) if self.spec_dens_min > 0 else tmp_min
                bloc_dict['value_ew'][bloc_dict['value_ew'] <= 0] = self.spec_dens_min
                self.init_spec_dens_B[bloc_name] = bloc_dict
            self.log_spec_dens_min = np.log10(self.spec_dens_min)
        
        ###############################

        # sort each par
        index_e = np.lexsort(tuple([par_e for par_e in self.init_par_Pe.values()]))
        self.init_par_Pe = {par_name: par_e[index_e] for par_name, par_e in self.init_par_Pe.items()}
        for bloc_name, bloc_dict in self.init_spec_dens_B.items():
            bloc_dict['value_ew'] = bloc_dict['value_ew'][index_e, :]
            self.init_spec_dens_B[bloc_name] = bloc_dict
        for attr_name, attr_dict in self.init_attribute_A.items():
            attr_dict['value_e' ] = attr_dict['value_e' ][index_e]
            self.init_attribute_A[attr_name] = attr_dict

        # transfer to multi-dimontional par-grid
        self.init_par_uniq_Pu = {par_name: np.unique(par_e) for par_name, par_e in self.init_par_Pe.items()}
        for bloc_name, bloc_dict in self.init_spec_dens_B.items():
            bloc_dict['value_gw'] = np.zeros([len(par_uniq) for par_uniq in self.init_par_uniq_Pu.values()] + [len(self.init_spec_wave_w)]) 
            self.init_spec_dens_B[bloc_name] = bloc_dict
        for attr_name, attr_dict in self.init_attribute_A.items():
            attr_dict['value_g' ] = np.zeros([len(par_uniq) for par_uniq in self.init_par_uniq_Pu.values()]) 
            self.init_attribute_A[attr_name] = attr_dict

        for i_e in range(self.num_templates):
            i_uniq_list = []
            for par_uniq, par_e in zip(self.init_par_uniq_Pu.values(), self.init_par_Pe.values()):
                i_uniq_list.append(np.where(par_uniq == par_e[i_e])[0][0])
            i_uniq_tuple = tuple(i_uniq_list)  
            for bloc_name, bloc_dict in self.init_spec_dens_B.items():
                bloc_dict['value_gw'][i_uniq_tuple] = bloc_dict['value_ew'][i_e, :]
                self.init_spec_dens_B[bloc_name] = bloc_dict
            for attr_name, attr_dict in self.init_attribute_A.items():
                attr_dict['value_g' ][i_uniq_tuple] = attr_dict['value_e' ][i_e]
                self.init_attribute_A[attr_name] = attr_dict

        ###############################

        # build interpolation function
        init_par_uniq_tuple = tuple(self.init_par_uniq_Pu.values())
        if self.cframe.mod_info_I['interp_space'] == 'linear':
            init_par_uniq_wave_tuple = tuple(list(init_par_uniq_tuple) + [self.init_spec_wave_w])
            def tmp_func(x): return x
        elif self.cframe.mod_info_I['interp_space'] == 'log':
            self.init_log_spec_wave_w = np.log10(self.init_spec_wave_w)
            init_par_uniq_wave_tuple = tuple(list(init_par_uniq_tuple) + [self.init_log_spec_wave_w])
            def tmp_func(x): return np.log10(x)
        self.interp_func_R = {}
        for bloc_name, bloc_dict in self.init_spec_dens_B.items():
            self.interp_func_R[('bloc', bloc_name)] = RegularGridInterpolator(init_par_uniq_wave_tuple, tmp_func(bloc_dict['value_gw']), method='linear', bounds_error=False)
        for attr_name, attr_dict in self.init_attribute_A.items():
            self.interp_func_R[('attr', attr_name)] = RegularGridInterpolator(init_par_uniq_tuple,      tmp_func(attr_dict['value_g' ]), method='linear', bounds_error=False)
        # set bounds_error=False to avoid error by slight exceeding of x-val generated by least_square function
        # but should avoid using pars outside of initial range manually

    def interp_model(self, input_par_p, input_par_index_P, ret_name=None):
        input_par_list = [input_par_p[input_par_index_P[par_name]] for par_name in self.init_par_uniq_Pu.keys()]
        if ret_name[0] == 'bloc':
            if self.cframe.mod_info_I['interp_space'] == 'linear':
                input_par_list = [input_par_list + [w] for w in self.init_spec_wave_w]
            elif self.cframe.mod_info_I['interp_space'] == 'log':
                input_par_list = [input_par_list + [logw] for logw in self.init_log_spec_wave_w]
        return self.interp_func_R[ret_name](np.array(input_par_list))

    ##########################################################################

    # def mask_lite_allowed(self, i_comp=None):

    #     i_par_voff = self.cframe.par_index_cP[i_comp]['voff']
    #     voff_min = self.cframe.par_min_cp[i_comp][i_par_voff]


    #     log_age_min, log_age_max = self.cframe.comp_info_cI[i_comp]['log_ssp_age_min'], self.cframe.comp_info_cI[i_comp]['log_ssp_age_max']
    #     age_min = self.age_e.min() if log_age_min is None else 10.0**log_age_min
    #     age_max = cosmo.age(self.v0_redshift).value if log_age_max in ['universe', 'Universe'] else 10.0**log_age_max
    #     mask_lite_ssp_e = (self.age_e >= age_min) & (self.age_e <= age_max)
    #     met_sel = self.cframe.comp_info_cI[i_comp]['ssp_metallicity']
    #     if met_sel != 'all':
    #         if met_sel in ['solar', 'Solar']:
    #             mask_lite_ssp_e &= self.met_e == 0.02
    #         else:
    #             mask_lite_ssp_e &= np.isin(self.met_e, met_sel)
    #     return mask_lite_ssp_e

    # def mask_lite_with_num_mods(self, num_ages_lite=8, num_mets_lite=1, verbose=True):
    #     if self.sfh_name_c[0] == 'nonparametric':
    #         # only used in nonparametic, single component
    #         mask_lite_allowed_e = self.mask_lite_allowed(if_ssp=True, i_comp=0)

    #         ages_full, num_ages_full = np.unique(self.age_e), len(np.unique(self.age_e))
    #         ages_allowed = np.unique(self.age_e[ mask_lite_allowed_e ])
    #         ages_lite = np.logspace(np.log10(ages_allowed.min()), np.log10(ages_allowed.max()), num=num_ages_lite)
    #         ages_lite *= 10.0**((np.random.rand(num_ages_lite)-0.5)*np.log10(ages_lite[1]/ages_lite[0]))
    #         # request log-even ages with random shift
    #         ind_ages_lite = [np.where(np.abs(ages_full-a)==np.min(np.abs(ages_full-a)))[0][0] for a in ages_lite]
    #         # np.round(np.linspace(0, num_ages_full-1, num_ages_lite)).astype(int)
    #         ind_mets_lite = [2,1,3,0][:num_mets_lite] # Z = 0.02 (solar), 0.008, 0.05, 0.004, select with this order
    #         ind_ssp_lite = np.array([ind_met*num_ages_full+np.arange(num_ages_full)[ind_age] 
    #                                  for ind_met in ind_mets_lite for ind_age in ind_ages_lite])
    #         mask_lite_ssp_e = np.zeros_like(self.age_e, dtype='bool')
    #         mask_lite_ssp_e[ind_ssp_lite] = True
    #         mask_lite_ssp_e &= mask_lite_allowed_e
    #         if verbose: print_log(f'Number of used SSP models: {mask_lite_ssp_e.sum()}', self.log_message) 
    #         return mask_lite_ssp_e

    #     else:
    #         mask_lite_csp_e = self.mask_lite_allowed(if_csp=True)
    #         if verbose: print_log(f'Number of used CSP models: {mask_lite_csp_e.sum()}', self.log_message) 
    #         return mask_lite_csp_e

    # def mask_lite_with_coeffs(self, coeffs=None, mask=None, num_mods_min=32, verbose=True):
    #     if self.sfh_name_c[0] == 'nonparametric':
    #         # only used in nonparametic, single component
    #         mask_lite_allowed_e = self.mask_lite_allowed(if_ssp=True, i_comp=0)

    #         coeffs_full = np.zeros(self.num_templates)
    #         coeffs_full[mask if mask is not None else mask_lite_allowed_e] = coeffs
    #         coeffs_sort = np.sort(coeffs_full)
    #         # coeffs_min = coeffs_sort[np.cumsum(coeffs_sort)/np.sum(coeffs_sort) < 0.01].max() 
    #         # # i.e., keep coeffs with sum > 99%
    #         # mask_ssp_lite = coeffs_full >= np.minimum(coeffs_min, coeffs_sort[-num_mods_min]) 
    #         # # keep minimum num of models
    #         # mask_ssp_lite &= mask_lite_allowed_e
    #         # print('Number of used SSP models:', mask_ssp_lite.sum()) #, np.unique(self.age_e[mask_ssp_lite]))
    #         # print('Ages with coeffs.sum > 99%:', np.unique(self.age_e[coeffs_full >= coeffs_min]))
    #         mask_lite_ssp_e = coeffs_full >= coeffs_sort[-num_mods_min]
    #         mask_lite_ssp_e &= mask_lite_allowed_e
    #         if verbose: 
    #             print_log(f'Number of used SSP models: {mask_lite_ssp_e.sum()}', self.log_message) 
    #             print_log(f'Coeffs.sum of used SSP models: {1-np.cumsum(coeffs_sort)[-num_mods_min]/np.sum(coeffs_sort)}', self.log_message) 
    #             print_log(f'Ages of dominant SSP models: {np.unique(self.age_e[coeffs_full >= coeffs_sort[-5]])}', self.log_message) 
    #         return mask_lite_ssp_e

    #     else:
    #         mask_lite_csp_e = self.mask_lite_allowed(if_csp=True)
    #         if verbose: print_log(f'Number of used CSP models: {mask_lite_csp_e.sum()}', self.log_message)             
    #         return mask_lite_csp_e

    ##########################################################################

    def create_models(self, obs_spec_wave_w, par_p, mask_lite_e=None, index_all_t=None, components=None, 
                      if_dust_ext=False, if_ism_abs=False, if_igm_abs=False, 
                      if_z_decline=True, if_convolve=False, conv_nbin=None, if_full_range=False): 

        # conv_nbin is not used for emission lines, it is added to keep a uniform format with other models
        # par: voff (to adjust redshift), tau_si, h_open, r_ratio, incl
        # comps: 'disc', 'dust'
        par_cp = self.cframe.reshape_by_comp(par_p, self.cframe.num_pars_c)
        if mask_lite_e is None:
            if index_all_t is not None:
                mask_lite_e = np.zeros(self.num_coeffs_tot, dtype='bool')
                mask_lite_e[index_all_t[index_all_t >= 0]] = True
        if mask_lite_e is not None: 
            mask_lite_ce = self.cframe.reshape_by_comp(mask_lite_e, self.num_coeffs_c) 
        if isinstance(components, str): components = [components]

        prep_spec_wave_w = copy(self.init_spec_wave_w) # avoid changing the initial 
        if self.cframe.mod_info_I['interp_space'] == 'log':
            prep_log_spec_wave_w = copy(self.init_log_spec_wave_w)
            obs_log_spec_wave_w  = np.log10(obs_spec_wave_w)

        mcomp_spec_dens_tw = None
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            if components is not None:
                if comp_name not in components: continue
            if mask_lite_e is not None:
                if sum(mask_lite_ce[i_comp]) == 0: continue

            intr_spec_dens_tw = np.zeros_like(prep_spec_wave_w[None,:])
            # for bloc_name, bloc_dict in self.init_spec_dens_B.items():
            #     if bloc_name in self.cframe.comp_info_cI[i_comp]['mod_used']:
            #         if mask_lite_e is not None:
            #             intr_spec_dens_tw += bloc_dict['value_ew'][mask_lite_ce[i_comp],:] # limit element number for accelarate calculation
            #         else:
            #             intr_spec_dens_tw += bloc_dict['value_ew']
            # if self.cframe.mod_info_I['interp_space'] == 'log': intr_log_spec_dens_tw = np.log10(intr_log_spec_dens_tw)
            # interpolate model for given pars in initial wavelength (rest)
            for bloc_name in self.init_spec_dens_B.keys():
                if bloc_name in self.cframe.comp_info_cI[i_comp]['mod_used']:
                    intr_spec_dens_tw += self.interp_model(par_cp[i_comp], self.cframe.par_index_cP[i_comp], ret_name=('bloc', bloc_name))[None,:] # convert to (1,w) format
            if self.cframe.mod_info_I['interp_space'] == 'log': intr_log_spec_dens_tw = intr_spec_dens_tw # already log in interp_model

            # dust extinction
            if if_dust_ext & ('Av' in self.cframe.par_index_cP[i_comp]):
                Av = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['Av']]
                dust_ext_factor_w = 10.0**(-0.4 * Av * ExtLaw(prep_spec_wave_w))
            else:
                dust_ext_factor_w = 1

            # redshift models
            voff = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['voff']]
            z_factor = (1 + self.v0_redshift) * (1 + voff/299792.458) # (1+z) = (1+zv0) * (1+v/c)
            z_spec_wave_w  = prep_spec_wave_w * z_factor
            if self.cframe.mod_info_I['interp_space'] == 'log': z_log_spec_wave_w = prep_log_spec_wave_w + np.log10(z_factor)
            if if_z_decline:
                z_decline_factor = 1 / z_factor
            else:
                z_decline_factor = 1

            if self.cframe.mod_info_I['interp_space'] == 'linear':
                zd_spec_dens_tw = intr_spec_dens_tw * dust_ext_factor_w * z_decline_factor
            elif self.cframe.mod_info_I['interp_space'] == 'log':
                zd_log_spec_dens_tw = intr_log_spec_dens_tw + np.log10(dust_ext_factor_w * z_decline_factor)

            # convolve with intrinsic and instrumental dispersion
            if if_convolve & ('fwhm' in self.cframe.par_index_cP[i_comp]) & (self.R_inst_rw is not None) & (conv_nbin is not None):
                fwhm = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['fwhm']]
                R_inst_w = np.interp(z_spec_wave_w, self.R_inst_rw[0], self.R_inst_rw[1])
                if self.cframe.mod_info_I['interp_space'] == 'log': zd_spec_dens_tw = 10.0**zd_log_spec_dens_tw # convolve in linear space
                czd_spec_dens_tw = convolve_var_width_fft(z_spec_wave_w, zd_spec_dens_tw, dv_fwhm_obj=fwhm, 
                                                          dw_fwhm_ref=self.dw_fwhm_dsp_w*z_factor, R_inst_w=R_inst_w, num_bins=conv_nbin)
                if self.cframe.mod_info_I['interp_space'] == 'log': czd_log_spec_dens_tw = np.log10(czd_spec_dens_tw)
            else:
                if self.cframe.mod_info_I['interp_space'] == 'linear':
                    czd_spec_dens_tw = zd_spec_dens_tw # just copy if convlution not required, e.g., for broad-band sed fitting
                elif self.cframe.mod_info_I['interp_space'] == 'log': 
                    czd_log_spec_dens_tw = zd_log_spec_dens_tw

            # project to observed wavelength
            if self.cframe.mod_info_I['interp_space'] == 'linear':
                interp_func = interp1d(z_spec_wave_w, czd_spec_dens_tw, axis=1, kind='linear', fill_value=(0,0), bounds_error=False)
                obs_spec_dens_tw = interp_func(obs_spec_wave_w)
            elif self.cframe.mod_info_I['interp_space'] == 'log': 
                interp_func = interp1d(z_log_spec_wave_w, czd_log_spec_dens_tw, axis=1, kind='linear', fill_value=(self.log_spec_dens_min,self.log_spec_dens_min), bounds_error=False)
                obs_log_spec_dens_tw = interp_func(obs_log_spec_wave_w)
                obs_spec_dens_tw = 10.0**obs_log_spec_dens_tw
                obs_spec_dens_tw[obs_log_spec_dens_tw <= self.log_spec_dens_min] = 0

            if mcomp_spec_dens_tw is None: 
                mcomp_spec_dens_tw = obs_spec_dens_tw
            else:
                mcomp_spec_dens_tw = np.vstack((mcomp_spec_dens_tw, obs_spec_dens_tw))

        return mcomp_spec_dens_tw

    ##########################################################################
    ########################### Output functions #############################

    def extract_results(self, step=None, if_print_results=True, if_return_results=False, if_rev_v0_redshift=False, if_show_average=False, **kwargs):

        ############################################################
        # check and replace the args to be compatible with old version <= 2.2.4
        if 'print_results'  in kwargs: if_print_results  = kwargs['print_results']
        if 'return_results' in kwargs: if_return_results = kwargs['return_results']
        if 'show_average'   in kwargs: if_show_average   = kwargs['show_average']
        ############################################################

        if (step is None) | (step in ['best', 'final']): step = 'joint_fit_3' if self.fframe.have_phot else 'joint_fit_2'
        if  step in ['spec+phot', 'spec+SED', 'spectrum+SED']:  step = 'joint_fit_3'
        if  step in ['spec', 'pure-spec', 'spectrum', 'pure-spectrum']:  step = 'joint_fit_2'
        
        best_chi_sq_l = copy(self.fframe.output_S[step]['chi_sq_l'])
        best_par_lp   = copy(self.fframe.output_S[step]['par_lp'  ])
        best_coeff_lt = copy(self.fframe.output_S[step]['coeff_lt'])
        best_index_lt = copy(self.fframe.output_S[step]['index_lt'])

        # update best-fit voff and fwhm if systemic redshift is updated
        if if_rev_v0_redshift & (self.fframe.rev_v0_redshift is not None):
            best_par_lp[:, self.fframe.par_name_p == 'voff'] -= self.fframe.ref_voff_l[0]
            best_par_lp[:, self.fframe.par_name_p == 'fwhm'] *= (1+self.fframe.v0_redshift) / (1+self.fframe.rev_v0_redshift)

        self.num_loops = self.fframe.num_loops # for print_results
        comp_name_c = self.cframe.comp_name_c
        num_comps = self.cframe.num_comps
        par_name_cp = self.cframe.par_name_cp

        # list the properties to be output; the print will follow this order
        value_names_additive, value_names_weighted = [], []
        for attr_name, attr_dict in self.init_attribute_A.items():
            if attr_dict['additive']: 
                value_names_additive.append('log_'+attr_name) # output log value for additive attributes
            else:
                value_names_weighted.append(attr_name)
        if len(value_names_weighted) > 0: value_names_additive.append('coeff') # use coeff as the weight

        ####################
        # custom additional
        value_names_additive += ['log_intLum_dust']
        ####################

        value_names_C = {}
        for (i_comp, comp_name) in enumerate(comp_name_c):
            value_names_C[comp_name] = value_names_weighted + value_names_additive + [] # add to [] if there is specific value for each comp
        value_names_tot = value_names_weighted + value_names_additive

        ret_names_tot = None
        for (i_comp, comp_name) in enumerate(comp_name_c):
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
            if ret_names_tot is None: 
                ret_names_tot = ret_names
            else:
                ret_names_tot = [ret_name for ret_name in ret_names_tot if ret_name in value_names_C[comp_name]]
        value_names_tot += ret_names_tot

        # format of results
        # output_C['comp']['par_lp'  ][i_l,i_p]: parameters
        # output_C['comp']['coeff_lt'][i_l,i_t]: coefficients
        # output_C['comp']['index_lt'][i_l,i_t]: indexes of original elements sequence in lite elements sequence
        # output_C['comp']['value_Vl']['name_l'][i_l]: calculated values
        output_C = {}
        for (i_comp, comp_name) in enumerate(comp_name_c):
            output_C[comp_name] = {} # init results for each comp
            # output_C[comp_name]['par_lp'  ] = np.zeros((self.num_loops, self.cframe.num_pars_tot), dtype='float')
            # output_C[comp_name]['coeff_lt'] = np.zeros((self.num_loops, self.num_coeffs_max), dtype='float')
            # output_C[comp_name]['index_lt'] = np.zeros((self.num_loops, self.num_coeffs_max), dtype='int') - 1
            output_C[comp_name]['value_Vl'] = {}
            for value_name in par_name_cp[i_comp] + value_names_C[comp_name]:
                output_C[comp_name]['value_Vl'][value_name] = np.zeros(self.num_loops, dtype='float')
        output_C['tot'] = {}
        output_C['tot']['value_Vl'] = {}
        for value_name in value_names_tot:
            output_C['tot']['value_Vl'][value_name] = np.zeros(self.num_loops, dtype='float')

        # locate the results of the model in the full fitting results
        i_pars_0_of_mod, i_pars_1_of_mod, i_coeffs_0_of_mod, i_coeffs_1_of_mod = self.fframe.search_mod_index(self.mod_name, self.fframe.full_mod_type)
        mod_par_lp   = best_par_lp  [:, i_pars_0_of_mod  :i_pars_1_of_mod  ]
        mod_coeff_lt = best_coeff_lt[:, i_coeffs_0_of_mod:i_coeffs_1_of_mod]
        mod_index_lt = best_index_lt[:, i_coeffs_0_of_mod:i_coeffs_1_of_mod]
        mod_coeff_le = np.zeros((self.num_loops, self.num_coeffs_tot), dtype='float')
        for i_loop in range(self.num_loops): mod_coeff_le[i_loop][mod_index_lt[i_loop][mod_index_lt[i_loop] >= 0]] = mod_coeff_lt[i_loop][mod_index_lt[i_loop] >= 0]

        for (i_comp, comp_name) in enumerate(comp_name_c):
            i_pars_0_of_comp_in_mod, i_pars_1_of_comp_in_mod = self.fframe.search_comp_index(comp_name, self.mod_name)[0:2]
            for i_par in range(self.cframe.num_pars_c[i_comp]): 
                output_C[comp_name]['value_Vl'][par_name_cp[i_comp][i_par]] = mod_par_lp[:, i_pars_0_of_comp_in_mod:i_pars_1_of_comp_in_mod][:, i_par]

            for i_loop in range(self.num_loops):
                par_p   = self.cframe.reshape_by_comp(mod_par_lp  [i_loop], self.cframe.num_pars_c)[i_comp]
                coeff_e = self.cframe.reshape_by_comp(mod_coeff_le[i_loop], self.num_coeffs_c     )[i_comp]

                voff = output_C[comp_name]['value_Vl']['voff'][i_loop]
                rev_redshift = (1 + self.v0_redshift) * (1 + voff/299792.458) - 1
                lum_area = 4*np.pi * cosmo.luminosity_distance(rev_redshift)**2 # with area unit

                for attr_name, attr_dict in self.init_attribute_A.items():
                    attr_value_e = self.interp_model(par_p, self.cframe.par_index_cP[i_comp], ret_name=('attr', attr_name))

                    if self.cframe.mod_info_I['interp_space'] == 'log': attr_value_e = 10.0**attr_value_e
                    if attr_dict['additive']: 
                        attr_value_e *= u.Unit(self.fframe.spec_flux_unit) * u.Unit(attr_dict['unit'])           # if original spec_dens is in flam
                        if not attr_value_e.unit.is_equivalent(attr_dict['orig_unit']): attr_value_e *= lum_area # if original spec_dens is in llam
                        attr_value_e = attr_value_e.to(attr_dict['orig_unit']).value
                        output_C[comp_name]['value_Vl']['log_'+attr_name][i_loop]  = np.log10((attr_value_e * coeff_e).sum())
                        output_C['tot'    ]['value_Vl']['log_'+attr_name][i_loop] += (attr_value_e * coeff_e).sum() # keep in linear for total sum
                    else:
                        output_C[comp_name]['value_Vl'][       attr_name][i_loop]  = (attr_value_e * coeff_e).sum() / coeff_e.sum()
                        output_C['tot'    ]['value_Vl'][       attr_name][i_loop] += (attr_value_e * coeff_e).sum() # keep un-weighted for total average

                output_C[comp_name]['value_Vl']['coeff'][i_loop]  = coeff_e.sum()
                output_C['tot'    ]['value_Vl']['coeff'][i_loop] += coeff_e.sum()

                ####################
                # custom additional
                output_C[comp_name]['value_Vl']['log_intLum_dust'][i_loop]  = output_C[comp_name]['value_Vl']['log_intLum_agn'][i_loop] + np.log10(output_C[comp_name]['value_Vl']['frac_abs'][i_loop])
                output_C['tot'    ]['value_Vl']['log_intLum_dust'][i_loop] += 10.0**output_C[comp_name]['value_Vl']['log_intLum_dust'][i_loop] # keep in linear for total sum
                ####################

                # calculate requested flux/Lum in given wavelength ranges
                i_coeffs_0_of_comp_in_mod, i_coeffs_1_of_comp_in_mod = self.fframe.search_comp_index(comp_name, self.mod_name, index_all_t=mod_index_lt[i_loop])[2:4]
                coeff_t = mod_coeff_lt[i_loop, i_coeffs_0_of_comp_in_mod:i_coeffs_1_of_comp_in_mod]
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
                            tmp_flam_tw = self.create_models(tmp_wave_w, mod_par_lp[i_loop], index_all_t=mod_index_lt[i_loop], components=comp_name, 
                                                             if_dust_ext=False, if_z_decline=True, if_full_range=True) # flux in obs frame
                            intrinsic_flam_w = coeff_t @ tmp_flam_tw
                        if ret_emi_F['value_state'] in ['observed','absorbed']:
                            tmp_flam_tw = self.create_models(tmp_wave_w, mod_par_lp[i_loop], index_all_t=mod_index_lt[i_loop], components=comp_name, 
                                                             if_dust_ext=True,  if_z_decline=True, if_full_range=True) # flux in obs frame
                            observed_flam_w = coeff_t @ tmp_flam_tw
                        if ret_emi_F['value_state'] == 'intrinsic': tmp_flam_w = intrinsic_flam_w
                        if ret_emi_F['value_state'] == 'observed' : tmp_flam_w = observed_flam_w
                        if ret_emi_F['value_state'] == 'absorbed' : tmp_flam_w = intrinsic_flam_w - observed_flam_w

                        tmp_wave_w *= u.angstrom
                        tmp_flam_w *= u.Unit(self.fframe.spec_flux_unit)
                        tmp_Flam    = tmp_flam_w.mean()
                        tmp_lamFlam = tmp_flam_w.mean() * tmp_wave_w.mean()
                        tmp_intFlux = np.trapezoid(tmp_flam_w, x=tmp_wave_w)

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
                        if ret_name in output_C['tot']['value_Vl']: output_C['tot']['value_Vl'][ret_name][i_loop] += ret_value

        for value_name in output_C['tot']['value_Vl']:
            # if (value_name[:8] in ['log_Flam', 'log_Fnu ', 'log_mass']) | (value_name[:11] in ['log_lamFlam', 'log_intFlux', 'log_lamLlam', 'log_intLum ']): 
            if (value_name in (value_names_additive + ret_names_tot)) & (value_name[:4] == 'log_'):
                output_C['tot']['value_Vl'][value_name] = np.log10(output_C['tot']['value_Vl'][value_name])
            elif value_name in value_names_weighted:
                output_C['tot']['value_Vl'][value_name] = output_C['tot']['value_Vl'][value_name] / output_C['tot']['value_Vl']['coeff']

        ############################################################
        # keep aliases for output in old version <= 2.2.4
        for (i_comp, comp_name) in enumerate(comp_name_c):
            output_C[comp_name]['values'] = output_C[comp_name]['value_Vl']
        output_C['tot']['values'] = output_C['tot']['value_Vl']
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
            print_name_CV[comp_name]['frac_abs'] = f"Torus dust absorption fraction"
            print_name_CV[comp_name]['log_mass_dust'] = f"Torus dust mass (log {self.init_attribute_A['mass_dust']['orig_unit']})".replace('M_sun', 'M☉')
            print_name_CV[comp_name]['log_intLum_dust'] = f"Torus dust bolometric lum. (log {self.init_attribute_A['intLum_agn']['orig_unit']})".replace('L_sun', 'L☉')
            print_name_CV[comp_name]['log_intLum_agn'] = f"AGN disc bolometric lum. (log {self.init_attribute_A['intLum_agn']['orig_unit']})".replace('L_sun', 'L☉')

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

        print_name_CV['tot'] = {}
        for value_name in self.output_C['tot']['value_Vl']:
            print_name_CV['tot'][value_name] = copy(print_name_CV[self.comp_name_c[0]][value_name])

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
            value_names.remove('coeff')
            for value_name in value_names:
                msg += '| ' + print_name_CV[comp_name][value_name] + f" = {value_Vl[value_name][mask_l].mean():10.4f}" + f" +/- {value_Vl[value_name].std():<10.4f}|\n"
            msg = msg[:-1] # remove the last \n
            bar = '=' * len(msg.split('\n')[-1])
            print_log(bar, log)
            print_log(msg, log)
            print_log(bar, log)
            print_log('', log)

