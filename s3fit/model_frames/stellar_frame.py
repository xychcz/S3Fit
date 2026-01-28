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
import matplotlib.pyplot as plt

from ..auxiliaries.auxiliary_frames import ConfigFrame, PhotFrame
from ..auxiliaries.auxiliary_functions import print_log, casefold, color_list_dict, wave_air_to_vac, convolve_fix_width_fft, convolve_var_width_fft
from ..auxiliaries.extinct_laws import ExtLaw

class StellarFrame(object):
    def __init__(self, mod_name=None, fframe=None, config=None, 
                 v0_redshift=None, R_inst_rw=None, 
                 wave_min=None, wave_max=None, 
                 Rratio_mod=None, dw_fwhm_dsp=None, dw_pix_inst=None, 
                 verbose=True, log_message=[]):

        self.mod_name = mod_name
        self.fframe = fframe
        self.config = config

        self.v0_redshift = v0_redshift
        self.R_inst_rw = R_inst_rw
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.Rratio_mod = Rratio_mod # resolution ratio of model / instrument
        self.dw_fwhm_dsp = dw_fwhm_dsp # model convolving width for downsampling (rest frame)
        self.dw_pix_inst = dw_pix_inst # data sampling width (obs frame)
        self.verbose = verbose
        self.log_message = log_message

        self.cframe=ConfigFrame(self.config)
        self.comp_name_c = self.cframe.comp_name_c
        self.num_comps = self.cframe.num_comps
        self.check_config()

        # check if the requested range (wave_min,wave_max) is within the defined range
        self.wave_min_def, self.wave_max_def = 912, 1e5 # angstrom
        self.enable = (self.wave_max > self.wave_min_def) & (self.wave_min < self.wave_max_def)

        self.sfh_name_c = np.array([info['sfh_name'] for info in self.cframe.comp_info_cI])
        if self.num_comps > 1:
            if np.sum(self.sfh_name_c == 'nonparametric') >= 1:
                raise ValueError((f"Nonparametric SFH can only be used with a single component."))

        # load template library
        self.read_ssp_library()

        # count the number of independent model elements
        self.num_coeffs_c = np.zeros(self.num_comps, dtype='int')
        for i_comp in range(self.num_comps):
            if self.sfh_name_c[i_comp] == 'nonparametric':
                self.num_coeffs_c[i_comp] = self.num_mets * self.num_ages
            else:
                self.num_coeffs_c[i_comp] = self.num_mets
        self.num_coeffs_C = {comp_name: num_coeffs for (comp_name, num_coeffs) in zip(self.comp_name_c, self.num_coeffs_c)}
        self.num_coeffs_tot = sum(self.num_coeffs_c)

        self.num_coeffs_max = self.num_coeffs_tot if self.cframe.mod_info_I['num_coeffs_max'] is None else min(self.num_coeffs_tot, self.cframe.mod_info_I['num_coeffs_max'])
        # self.num_coeffs_max = max(self.num_coeffs_max, 16 * 2) # set min num for each template par # only if nonpara

        # currently do not consider negative spectra 
        self.mask_absorption_e = np.zeros((self.num_coeffs_tot), dtype='bool')

        # check boundaries of stellar ages
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            if 'log_csp_age' in self.cframe.par_name_cp[i_comp]:
                i_par_log_csp_age = self.cframe.par_index_cP[i_comp]['log_csp_age']
                age_universe = cosmo.age(self.v0_redshift).to(self.age_unit).value
                age_min_allowed = self.age_e[self.mask_lite_allowed(if_ssp=True, i_comp=i_comp)].min()
                if self.cframe.par_max_cp[i_comp][i_par_log_csp_age] > np.log10(age_universe):
                    self.cframe.par_max_cp[i_comp][i_par_log_csp_age] = np.log10(age_universe)
                    print_log(f"[WARNING]: Upper bound of log_csp_age of the component '{comp_name}' "
                        +f" is reset to the universe age {age_universe:.3f} {self.age_unit} at z = {self.v0_redshift}.", self.log_message)
                if self.cframe.par_min_cp[i_comp][i_par_log_csp_age] > np.log10(age_universe):
                    self.cframe.par_min_cp[i_comp][i_par_log_csp_age] = np.log10(age_min_allowed*1.0001) # take a factor of 1.0001 to avoid (csp_age-ssp_age) < 0
                    print_log(f"[WARNING]: Lower bound of log_csp_age of the component '{comp_name}' "
                        +f" exceeds the universe age {age_universe:.3f} {self.age_unit} at z = {self.v0_redshift}, "
                        +f" is reset to the available minimum SSP age {age_min_allowed:.3f} {self.age_unit}.", self.log_message)
                if self.cframe.par_min_cp[i_comp][i_par_log_csp_age] < np.log10(age_min_allowed):
                    self.cframe.par_min_cp[i_comp][i_par_log_csp_age] = np.log10(age_min_allowed*1.0001)
                    print_log(f"[WARNING]: Lower bound of log_csp_age of the component '{comp_name}' "
                        +f" is reset to the available minimum SSP age {age_min_allowed:.3f} {self.age_unit}.", self.log_message) 

        if self.verbose:
            for (i_comp, comp_name) in enumerate(self.comp_name_c):
                mask_lite_allowed_e = self.mask_lite_allowed(if_ssp=True, i_comp=i_comp)
                print_log(f"Component ({i_comp}) '{comp_name}':", self.log_message)
                print_log(f"    SSP templates number: {mask_lite_allowed_e.sum()} used in a total of {self.num_templates}", self.log_message)
                print_log(f"    SSP templates age range ({self.age_unit}): from {self.age_e[mask_lite_allowed_e].min():.3f} to {self.age_e[mask_lite_allowed_e].max():.3f}", self.log_message)
                print_log(f"    SSP templates metallicity (Z/H): {np.unique(self.met_e[mask_lite_allowed_e])}", self.log_message) 
                print_log(f"    CSP SFH function: '{self.sfh_name_c[i_comp]}'", self.log_message)

        # set plot styles
        self.plot_style_C = {}
        self.plot_style_C['tot'] = {'color': 'C0', 'alpha': 0.75, 'linestyle': '-', 'linewidth': 1.5}
        i_red, i_green, i_blue = 0, 0, 0
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            self.plot_style_C[comp_name] = {'color': 'None', 'alpha': 0.5, 'linestyle': '-', 'linewidth': 1}
            if 'log_csp_age' in self.cframe.par_name_cp[i_comp]:
                i_par_log_csp_age = self.cframe.par_index_cP[i_comp]['log_csp_age']
                log_csp_age_mid = 0.5 * (self.cframe.par_min_cp[i_comp][i_par_log_csp_age] + self.cframe.par_max_cp[i_comp][i_par_log_csp_age])
                if 10.0**log_csp_age_mid * u.Unit(self.age_unit) > (1 * u.Gyr): # > 1 Gyr
                    self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['red'], i_red, mode="wrap"))
                    i_red += 1
                elif 10.0**log_csp_age_mid * u.Unit(self.age_unit) > (100 * u.Myr): # 100 Myr to 1 Gyr
                    self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['green'], i_green, mode="wrap"))
                    i_green += 1
                else: # < 100 Myr
                    self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['blue'], i_blue, mode="wrap"))
                    i_blue += 1
            else:
                self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['blue'], i_blue, mode="wrap"))
                i_blue += 1
    ##########################################################################

    def check_config(self):

        ############################################################
        # to be compatible with old version <= 2.2.4
        if len(self.cframe.par_index_cP[0]) == 0:
            self.cframe.par_name_cp  = [['voff', 'fwhm', 'Av', 'log_csp_age', 'log_csp_tau'][:self.cframe.num_pars_c[i_comp]] for i_comp in range(self.num_comps)]
            self.cframe.par_index_cP = [{par_name: i_par for (i_par, par_name) in enumerate(self.cframe.par_name_cp[i_comp])} for i_comp in range(self.num_comps)]
        for i_comp in range(self.num_comps):
            if 'age_min' in self.cframe.comp_info_cI[i_comp]: self.cframe.comp_info_cI[i_comp]['log_ssp_age_min'] = self.cframe.comp_info_cI[i_comp]['age_min']
            if 'age_max' in self.cframe.comp_info_cI[i_comp]: self.cframe.comp_info_cI[i_comp]['log_ssp_age_max'] = self.cframe.comp_info_cI[i_comp]['age_max']
            if 'met_sel' in self.cframe.comp_info_cI[i_comp]: self.cframe.comp_info_cI[i_comp]['ssp_metallicity'] = self.cframe.comp_info_cI[i_comp]['met_sel']
        ############################################################

        # set inherited or default info if not specified in config
        # model-level info, call as self.cframe.mod_info_I[info_name]
        self.cframe.retrieve_inherited_info('num_coeffs_max', root_info_I=self.fframe.root_info_I, default=None)

        # component-level info, call as self.cframe.comp_info_cI[i_comp][info_name]
        # format of returned flux / Lum density or integrated values
        for i_comp in range(self.num_comps):
            # either 2-unit-nested tuples (for wave and value, respectively) or dictionary as follows are supported
            self.cframe.retrieve_inherited_info('ret_emission_set', i_comp=i_comp, root_info_I=self.fframe.root_info_I, 
                                                default=[((5500, 25, 'angstrom', 'rest'), ('Flam', 'erg s-1 cm-2 angstrom-1', 'observed')), 
                                                         {'wave_center': 5500, 'wave_width': 25, 'wave_unit': 'angstrom', 'wave_frame': 'rest', 
                                                          'value_form': 'lamLlam', 'value_unit': 'erg s-1', 'value_state': 'intrinsic'},
                                                         ((912, 30000, 'angstrom', 'rest'), ('intLum', 'L_sun', 'intrinsic')),
                                                         {'wave_min': 912, 'wave_max': 30000, 'wave_unit': 'angstrom', 'wave_frame': 'rest', 
                                                          'value_form': 'intLum', 'value_unit': 'L_sun', 'value_state': 'absorbed'},
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
                if 'wave_unit'  not in ret_emi_F: ret_emi_F['wave_unit']  = 'angstrom'
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

        # format of returned SFR 
        for i_comp in range(self.num_comps):
            # either 2-unit-nested tuples (for age and value, respectively) or dictionary as follows are supported
            self.cframe.retrieve_inherited_info('ret_SFR_set', i_comp=i_comp, root_info_I=self.fframe.root_info_I, 
                                                default=[((0, 10, 'Myr'), ('SFR', 'M_sun yr-1', 'intrinsic')), 
                                                         {'age_min': 0, 'age_max': 10, 'age_unit': 'Myr', 
                                                          'value_form': 'SFR', 'value_unit': 'M_sun yr-1', 'value_state': 'absorbed'},
                                                         ((0, 100, 'Myr'), ('SFR', 'M_sun yr-1', 'intrinsic')), 
                                                         ((0, 100, 'Myr'), ('SFR', 'M_sun yr-1', 'absorbed')), 
                                                        ])
            # 'age_unit': any time unit supported by astropy.unit
            # 'value_form': 'SFR', 'sSFR'
            # 'value_state': 'intrinsic', 'observed', 'absorbed' (i.e., dust absorbed)
            # 'value_unit': any SFR/sSFR unit supported by astropy.unit

            if self.cframe.comp_info_cI[i_comp]['ret_SFR_set'] is None: continue # user can set None to skip all of these calculations
            # group line info to a list
            if isinstance(self.cframe.comp_info_cI[i_comp]['ret_SFR_set'], (tuple, dict)): 
                self.cframe.comp_info_cI[i_comp]['ret_SFR_set'] = [self.cframe.comp_info_cI[i_comp]['ret_SFR_set']]
            for i_ret in range(len(self.cframe.comp_info_cI[i_comp]['ret_SFR_set'])):
                tmp_format = self.cframe.comp_info_cI[i_comp]['ret_SFR_set'][i_ret]
                # convert tuple format to dict
                if isinstance(tmp_format, tuple):
                    ret_sfr_F = {}
                    age_0, age_1 = tmp_format[0][:2]
                    if age_0 > age_1:
                        ret_sfr_F['age_center'], ret_sfr_F['age_width'] = age_0, age_1
                    else:
                        ret_sfr_F['age_min'], ret_sfr_F['age_max'] = age_0, age_1 if age_1 > age_0 else age_0+1
                    ret_sfr_F['age_unit']  = tmp_format[0][2]
                    ret_sfr_F['value_form'], ret_sfr_F['value_unit'], ret_sfr_F['value_state'] = tmp_format[1]
                elif isinstance(tmp_format, dict):
                    ret_sfr_F = tmp_format
                # set default 
                if ('wave_center' not in ret_sfr_F) & ('wave_min' not in ret_sfr_F): 
                    ret_sfr_F['wave_center'], ret_sfr_F['wave_width']  = 3000, 25
                if 'wave_frame' not in ret_sfr_F: ret_sfr_F['wave_frame'] = 'rest'
                if 'wave_unit'  not in ret_sfr_F: ret_sfr_F['wave_unit']  = 'angstrom'
                if 'flux_form'  not in ret_sfr_F: ret_sfr_F['flux_form']  = 'intFlux'
                # check alternatives
                if ret_sfr_F['value_form'] == 'sfr'  : ret_sfr_F['value_form'] = 'SFR'
                if ret_sfr_F['value_form'] == 'ssfr' : ret_sfr_F['value_form'] = 'sSFR'
                if casefold(ret_sfr_F['value_state']) in ['intrinsic', 'original']:
                    ret_sfr_F['value_state'] = 'intrinsic'
                elif casefold(ret_sfr_F['value_state']) in ['observed', 'reddened', 'attenuated', 'extincted', 'extinct']:
                    ret_sfr_F['value_state'] = 'observed'
                elif casefold(ret_sfr_F['value_state']) in ['absorbed', 'dust absorbed', 'dust-absorbed']:
                    ret_sfr_F['value_state'] = 'absorbed'
                self.cframe.comp_info_cI[i_comp]['ret_SFR_set'][i_ret] = ret_sfr_F

        # other component-level info
        # check alternative info
        for i_comp in range(self.num_comps):
            # SFH name
            if casefold(self.cframe.comp_info_cI[i_comp]['sfh_name']) in ['exponential']: 
                self.cframe.comp_info_cI[i_comp]['sfh_name'] = 'exponential'
            if casefold(self.cframe.comp_info_cI[i_comp]['sfh_name']) in ['delayed']: 
                self.cframe.comp_info_cI[i_comp]['sfh_name'] = 'delayed'
            if casefold(self.cframe.comp_info_cI[i_comp]['sfh_name']) in ['constant', 'burst', 'starburst']: 
                self.cframe.comp_info_cI[i_comp]['sfh_name'] = 'constant'
            if casefold(self.cframe.comp_info_cI[i_comp]['sfh_name']) in ['nonparametric', 'non-parametric', 'non_parametric']: 
                self.cframe.comp_info_cI[i_comp]['sfh_name'] = 'nonparametric'
            if casefold(self.cframe.comp_info_cI[i_comp]['sfh_name']) in ['user', 'custom', 'customized']: 
                self.cframe.comp_info_cI[i_comp]['sfh_name'] = 'user'

    ##########################################################################

    def read_ssp_library(self):

        ##############################################################
        ###### Modify this section to use a different SSP model ######
        for item in ['file', 'file_path']:
            if item in self.cframe.mod_info_I: ssp_file = self.cframe.mod_info_I[item]
        ssp_lib = fits.open(ssp_file)

        self.header = ssp_lib[0].header
        # load models
        self.init_spec_dens_ew = ssp_lib[0].data
        self.init_spec_dens_unit = 'L_sun angstrom-1'
        # wave_axis = 1
        # crval = self.header[f'CRVAL{wave_axis}']
        # cdelt = self.header[f'CDELT{wave_axis}']
        # naxis = self.header[f'NAXIS{wave_axis}']
        # crpix = self.header[f'CRPIX{wave_axis}']
        # if not cdelt: cdelt = 1
        # self.init_spec_wave_w = crval + cdelt*(np.arange(naxis) + 1 - crpix)
        # use wavelength in data instead of one from header
        self.init_spec_wave_w = ssp_lib[1].data
        self.init_spec_wave_unit = 'angstrom'
        self.init_spec_wave_medium = 'air'
        # template resolution step of 0.1 angstrom, from https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.4781M/abstract
        self.init_dw_fwhm = 0.1 # assume init_dw_fwhm as dw per pix

        self.num_templates = self.init_spec_dens_ew.shape[0]
        self.mass_e = np.ones(self.num_templates, dtype='float') # i.e., self.init_norm_e, initially normalized by mass
        self.mass_unit = 'M_sun' # i.e., self.init_norm_unit
        self.remain_massfrac_e = ssp_lib[2].data # leave remain_massfrac_e = 1 if not provided. 

        self.age_e = np.zeros(self.num_templates, dtype='float')
        self.met_e = np.zeros(self.num_templates, dtype='float')
        for i_e in range(self.num_templates):
            met, age = self.header[f'NAME{i_e}'].split('.dat')[0].split('_')[1:3]
            self.age_e[i_e] = 10.0**float(age.replace('logt',''))
            self.met_e[i_e] = float(met.replace('Z',''))

        init_age_unit = 'yr'
        self.age_unit = 'Gyr'
        self.age_e *= u.Unit(init_age_unit).to(self.age_unit)

        ##############################################################

        # convert wave unit to angstrom in vacuum
        self.init_spec_wave_w *= u.Unit(self.init_spec_wave_unit).to('angstrom')
        self.init_spec_wave_unit = 'angstrom'
        if self.init_spec_wave_medium == 'air': self.init_spec_wave_w = wave_air_to_vac(self.init_spec_wave_w)

        # convert the normalization from per unit mass to per unit L5500
        self.wave_norm, self.dw_norm = 5500, 10
        mask_5500_w = np.abs(self.init_spec_wave_w - self.wave_norm) < self.dw_norm
        scale_spec_dens_e = np.mean(self.init_spec_dens_ew[:, mask_5500_w], axis=1)
        scale_spec_dens_unit = copy(self.init_spec_dens_unit)
        # scale models by scale_spec_dens_e * scale_spec_dens_unit
        self.init_spec_dens_ew /= scale_spec_dens_e[:, None]
        self.init_spec_dens_unit = str(u.Unit(self.init_spec_dens_unit) / u.Unit(scale_spec_dens_unit)) # dimensionless unscaled
        self.mass_e /= scale_spec_dens_e # mass_e is now converted to mass-to-L5500 ratio:
        self.mass_unit = str(u.Unit(self.mass_unit) / u.Unit(scale_spec_dens_unit))

        ##############################################################

        self.age_a, index_ua_a = np.unique(self.age_e, return_inverse=True)
        self.met_m, index_um_m = np.unique(self.met_e, return_inverse=True)
        self.num_ages = len(self.age_a)
        self.num_mets = len(self.met_m)
        # obtain the duration (i.e., bin width) of each ssp if considering a continous SFH
        duration_a = np.gradient(self.age_a)
        self.duration_e = duration_a[index_ua_a] # np.tile(duration_a, (self.num_mets,1)).flatten()

        # assume constant SFH across adjacent age bins
        self.sfr_e = self.mass_e / (self.duration_e * u.Unit(self.age_unit).to('yr'))
        self.sfr_unit = str(u.Unit(self.mass_unit) / u.Unit('yr')) 

        ##############################################################

        # select model spectra in given wavelength range
        mask_select_w = (self.init_spec_wave_w >= self.wave_min) & (self.init_spec_wave_w <= self.wave_max)
        self.prep_spec_wave_w  = self.init_spec_wave_w [  mask_select_w]
        self.prep_spec_dens_ew = self.init_spec_dens_ew[:,mask_select_w]
        # prep_ is short for preprocessed or prepared (e.g., wave-cut, convolved)

        # determine the required model resolution and bin size (in angstrom) to downsample the model
        if self.Rratio_mod is not None:
            R_mod_dsp_w = np.interp(self.prep_spec_wave_w, self.R_inst_rw[0]/(1+self.v0_redshift), self.R_inst_rw[1] * self.Rratio_mod) # R_inst_rw[0] is in observed frame
            self.prep_dw_fwhm_w = self.prep_spec_wave_w / R_mod_dsp_w # required resolving width in rest frame
        elif self.dw_fwhm_dsp is not None:
            self.prep_dw_fwhm_w = np.full(len(self.prep_spec_wave_w), self.dw_fwhm_dsp)
        else:
            self.prep_dw_fwhm_w = None

        if self.prep_dw_fwhm_w is not None:
            if (self.prep_dw_fwhm_w > self.init_dw_fwhm).all(): 
                preconvolving = True
            else:
                preconvolving = False
                self.prep_dw_fwhm_w = np.full(len(self.init_spec_wave_w), self.init_dw_fwhm)
            dw_dsp = self.prep_dw_fwhm_w.min() * 0.5 # required min bin wavelength following Nyquist–Shannon sampling
            if self.dw_pix_inst is not None:
                dw_dsp = min(dw_dsp, self.dw_pix_inst/(1+self.v0_redshift) * 0.5) # also require model bin wavelength <= 0.5 of data bin width (convert to rest frame)
            dpix_dsp = int(dw_dsp / np.median(np.diff(self.prep_spec_wave_w))) # required min bin number of pixels
            dw_dsp = dpix_dsp * np.median(np.diff(self.prep_spec_wave_w)) # update value
            if dpix_dsp > 1:
                if preconvolving:
                    if self.verbose: 
                        print_log(f'Downsample preconvolved SSP templates with bin width of {dw_dsp:.3f} Å in a min resolution of {self.prep_dw_fwhm_w.min():.3f} Å', self.log_message)
                    # before downsampling, smooth the model to avoid aliasing (like in ADC or digital signal reduction)
                    # here assume the internal dispersion in the original model (e.g., in stellar atmosphere) is indepent from the measured dispersion (i.e., stellar motion) in the fitting
                    self.prep_spec_dens_ew = convolve_fix_width_fft(self.prep_spec_wave_w, self.prep_spec_dens_ew, dw_fwhm=self.prep_dw_fwhm_w.min())
                else:
                    if self.verbose: 
                        print_log(f'Downsample original SSP templates with bin width of {dw_dsp:.3f} Å in a min resolution of {self.prep_dw_fwhm_w.min():.3f} Å', self.log_message)  
                self.prep_spec_wave_w  = self.prep_spec_wave_w [  ::dpix_dsp]
                self.prep_spec_dens_ew = self.prep_spec_dens_ew[:,::dpix_dsp]
                self.prep_dw_fwhm_w    = self.prep_dw_fwhm_w   [  ::dpix_dsp]

        ##############################################################

        # extend to longer wavelength in NIR-MIR (e.g., > 3 micron)
        # note that ext_index_e and ext_ratio_e are always required, i.e., to calculate integrated flux
        self.init_spec_wave_min = self.init_spec_wave_w.min()
        self.init_spec_wave_max = self.init_spec_wave_w.max()
        mask_ref_w = (self.init_spec_wave_w > max(1.6e4, self.init_spec_wave_max-5000)) & (self.init_spec_wave_w <= min(2.3e4, self.init_spec_wave_max-1000)) 
        # the longer wavelength end is not a single-temperature blackbody with index_e = -4 (weaker absorption allows radiation from deeper, hotter layer)
        # run linear fit in log-log grid
        logw_w  = np.log10(self.init_spec_wave_w [  mask_ref_w])
        logf_ew = np.log10(self.init_spec_dens_ew[:,mask_ref_w])
        mn_logw   = logw_w.mean()
        mn_logf_e = logf_ew.mean(axis=1)
        d_logw_w  = logw_w  - mn_logw
        d_logf_ew = logf_ew - mn_logf_e[:,None]
        self.ext_index_e = np.dot(d_logw_w, d_logf_ew.T) / np.dot(d_logw_w, d_logw_w)
        self.ext_ratio_e = 10.0**(mn_logf_e - self.ext_index_e * mn_logw)

        if self.wave_max > self.init_spec_wave_max:
            ext_spec_wave_logbin = 0.02
            ext_spec_wave_num = max(2, 1+int(np.log10(self.wave_max / self.init_spec_wave_max) / ext_spec_wave_logbin))
            ext_spec_wave_w = np.logspace(np.log10(self.init_spec_wave_max), np.log10(self.wave_max), ext_spec_wave_num)[1:]
            ext_spec_dens_ew = ext_spec_wave_w[None,:]**self.ext_index_e[:,None] * self.ext_ratio_e[:,None]
            self.prep_spec_wave_w  = np.hstack((self.prep_spec_wave_w , ext_spec_wave_w))
            self.prep_spec_dens_ew = np.hstack((self.prep_spec_dens_ew, ext_spec_dens_ew))

    ##########################################################################

    def mask_lite_allowed(self, i_comp=None, if_ssp=True, if_csp=False):
        if if_csp: if_ssp = False

        if if_ssp: 
            if i_comp is None: i_comp = 0
            # mask for all SSP elements, for an individual comp with i_comp
            log_age_min, log_age_max = self.cframe.comp_info_cI[i_comp]['log_ssp_age_min'], self.cframe.comp_info_cI[i_comp]['log_ssp_age_max']
            age_min = self.age_e.min() if log_age_min is None else 10.0**log_age_min
            age_max = cosmo.age(self.v0_redshift).value if log_age_max in ['universe', 'Universe'] else 10.0**log_age_max
            mask_lite_ssp_e = (self.age_e >= age_min) & (self.age_e <= age_max)
            met_sel = self.cframe.comp_info_cI[i_comp]['ssp_metallicity']
            if met_sel != 'all':
                if met_sel in ['solar', 'Solar']:
                    mask_lite_ssp_e &= self.met_e == 0.02
                else:
                    mask_lite_ssp_e &= np.isin(self.met_e, met_sel)
            return mask_lite_ssp_e

        else: 
            # mask for all CSP elements, loop for all comps
            mask_lite_csp_e = np.array([], dtype='bool')
            for i_comp in range(self.num_comps):
                tmp_mask_e = np.ones(self.num_mets, dtype='bool') 
                met_sel = self.cframe.comp_info_cI[i_comp]['ssp_metallicity']
                if met_sel != 'all':
                    if met_sel == 'solar':
                        tmp_mask_e &= np.unique(self.met_e) == 0.02
                    else:
                        tmp_mask_e &= np.isin(np.unique(self.met_e), met_sel)
                mask_lite_csp_e = np.hstack((mask_lite_csp_e, tmp_mask_e))
            return mask_lite_csp_e

    def mask_lite_with_nums(self, num_ages_lite=8, num_mets_lite=1, verbose=True):
        if self.sfh_name_c[0] == 'nonparametric':
            # only used in nonparametic, single component
            mask_lite_allowed_e = self.mask_lite_allowed(if_ssp=True, i_comp=0)

            ages_full, num_ages_full = np.unique(self.age_e), len(np.unique(self.age_e))
            ages_allowed = np.unique(self.age_e[ mask_lite_allowed_e ])
            ages_lite = np.logspace(np.log10(ages_allowed.min()), np.log10(ages_allowed.max()), num=num_ages_lite)
            ages_lite *= 10.0**((np.random.rand(num_ages_lite)-0.5)*np.log10(ages_lite[1]/ages_lite[0]))
            # request log-even ages with random shift
            ind_ages_lite = [np.where(np.abs(ages_full-a)==np.min(np.abs(ages_full-a)))[0][0] for a in ages_lite]
            # np.round(np.linspace(0, num_ages_full-1, num_ages_lite)).astype(int)
            ind_mets_lite = [2,1,3,0][:num_mets_lite] # Z = 0.02 (solar), 0.008, 0.05, 0.004, select with this order
            ind_ssp_lite = np.array([ind_met*num_ages_full+np.arange(num_ages_full)[ind_age] 
                                     for ind_met in ind_mets_lite for ind_age in ind_ages_lite])
            mask_lite_ssp_e = np.zeros_like(self.age_e, dtype='bool')
            mask_lite_ssp_e[ind_ssp_lite] = True
            mask_lite_ssp_e &= mask_lite_allowed_e
            if verbose: print_log(f'Number of used SSP templates: {mask_lite_ssp_e.sum()}', self.log_message) 
            return mask_lite_ssp_e

        else:
            mask_lite_csp_e = self.mask_lite_allowed(if_csp=True)
            if verbose: print_log(f'Number of used CSP templates: {mask_lite_csp_e.sum()}', self.log_message) 
            return mask_lite_csp_e

    def mask_lite_with_coeffs(self, coeff_t=None, mask_e=None, num_lite=None, frac_lite=None, verbose=True):
        if self.sfh_name_c[0] == 'nonparametric':
            # only used in nonparametic, single component
            mask_lite_allowed_e = self.mask_lite_allowed(if_ssp=True, i_comp=0)

            coeff_e = np.zeros(self.num_templates)
            coeff_e[mask_e if mask_e is not None else mask_lite_allowed_e] = coeff_t
            coeff_sort_s = np.sort(coeff_e)
            coeff_cumnorm_s = np.cumsum(coeff_sort_s)/np.sum(coeff_sort_s)

            if num_lite is not None:
                coeff_min = coeff_sort_s[-num_lite]
            elif frac_lite is not None:
                coeff_min = coeff_sort_s[coeff_cumnorm_s < (1-frac_lite)].max() 
            mask_lite_ssp_e = coeff_e >= coeff_min
            mask_lite_ssp_e &= mask_lite_allowed_e

            if verbose: 
                print_log(f"Number of used SSP templates: {mask_lite_ssp_e.sum()}", self.log_message) 
                print_log(f"Contribution of the {mask_lite_ssp_e.sum()} used SSP templates to Fλ (5500Å): {1-coeff_cumnorm_s[-num_lite]:.6f}", self.log_message) 
                print_log(f"Ages (Gyr) of the top 5 dominant SSP templates: {self.age_e[coeff_e >= coeff_sort_s[-5]]}", self.log_message) 
            return mask_lite_ssp_e

        else:
            mask_lite_csp_e = self.mask_lite_allowed(if_csp=True)
            if verbose: print_log(f'Number of used CSP templates: {mask_lite_csp_e.sum()}', self.log_message)             
            return mask_lite_csp_e

    ##########################################################################

    def llam_weight_from_sfh(self, i_comp, par_p):

        csp_age = 10.0**par_p[self.cframe.par_index_cP[i_comp]['log_csp_age']]
        ssp_age_e = self.age_e
        evo_time_e = csp_age - ssp_age_e

        if self.sfh_name_c[i_comp] == 'exponential': 
            csp_tau = 10.0**par_p[self.cframe.par_index_cP[i_comp]['log_csp_tau']]
            sfh_func_e = np.exp(- evo_time_e / csp_tau)
        if self.sfh_name_c[i_comp] == 'delayed': 
            csp_tau = 10.0**par_p[self.cframe.par_index_cP[i_comp]['log_csp_tau']]
            sfh_func_e = np.exp(- evo_time_e / csp_tau) * evo_time_e
        if self.sfh_name_c[i_comp] == 'constant': 
            sfh_func_e = np.ones_like(evo_time_e)
        if self.sfh_name_c[i_comp] == 'user': 
            sfh_func_e = self.cframe.comp_info_cI[i_comp]['sfh_func'](evo_time_e, csp_age, par_p, i_comp, self)
        ##########################################################################
        # a user defined sfh function (input via config) has the following format
        # def sfh_user(*args):
        #     # please do not touch the following two lines
        #     evo_time, galaxy_age, parameters, i_component, StellarFrame = args
        #     def get_par(par_name): return parameters[StellarFrame.cframe.par_index_cP[i_component][par_name]]
        #     # you can modify the sfh function and parameter names
        #     # parameter names should be the same as those in input config
        #     log_t_peak = get_par('log_t_peak') # epoch with peak SFH
        #     log_csp_tau = get_par('log_csp_tau')
        #     t_peak = 10.0**log_t_peak
        #     csp_tau = 10.0**log_csp_tau
        #     sfh = np.exp(-((galaxy_age-evo_time)-t_peak)**2 / csp_tau**2/2)
        #     return sfh
        # or directly add new SFH function here. 
        ##########################################################################

        sfh_func_e[~self.mask_lite_allowed(if_ssp=True, i_comp=i_comp)] = 0 # do not use ssp out of allowed range
        sfh_func_e[evo_time_e < 0] = 0 # do not use ssp older than csp_age 

        llam_weight_e = sfh_func_e / self.sfr_e # convert SFH_e (in unit of Msun/yr) to L5500_e (sfr_e is normalzied by L5500)
        if any(llam_weight_e > 0): llam_weight_e /= llam_weight_e.sum() # convert to dimensionless llam-weight
        # therefore the csp spectrum, (init_spec_dens_ew * llam_weight_e).sum(axis=0), is still normalized at unit L5500
        # ssp_coeff_e = (csp_coeff * llam_weight_e) for direct usage of init_spec_dens_ew (i.e., nonparametic SFH).

        return llam_weight_e

    def create_models(self, obs_spec_wave_w, par_p, mask_lite_e=None, index_all_t=None, components=None, 
                      if_dust_ext=True, if_ism_abs=False, if_igm_abs=False, 
                      if_z_decline=True, if_convolve=True, conv_nbin=None, if_full_range=False, dpix_resample=300):

        par_cp = self.cframe.reshape_by_comp(par_p, self.cframe.num_pars_c)
        if mask_lite_e is None:
            if index_all_t is not None:
                mask_lite_e = np.zeros(self.num_coeffs_tot, dtype='bool')
                mask_lite_e[index_all_t[index_all_t >= 0]] = True
        if mask_lite_e is not None: 
            mask_lite_ce = self.cframe.reshape_by_comp(mask_lite_e, self.num_coeffs_c) 
        if isinstance(components, str): components = [components]

        # use sparse init spectra if to calculate integrated flux or lum
        if if_full_range:
            prep_spec_dens_ew = self.init_spec_dens_ew[:,::dpix_resample]
            prep_spec_wave_w  = self.init_spec_wave_w [  ::dpix_resample]
            prep_spec_dens_ew = np.hstack((prep_spec_dens_ew[:, prep_spec_wave_w <= self.init_spec_wave_max], self.init_spec_dens_ew[:, self.init_spec_wave_w > self.init_spec_wave_max]))
            prep_spec_wave_w  = np.hstack((prep_spec_wave_w [   prep_spec_wave_w <= self.init_spec_wave_max], self.init_spec_wave_w [   self.init_spec_wave_w > self.init_spec_wave_max]))
            # extrapolate at longer wavelength end
            if max(obs_spec_wave_w)/(1+self.v0_redshift) > max(prep_spec_wave_w):
                ext_spec_wave_w  = np.logspace(np.log10(max(prep_spec_wave_w)), np.log10(max(obs_spec_wave_w)/(1+self.v0_redshift)+1000), 1+50)[1:]
                ext_spec_dens_ew = ext_spec_wave_w[None,:]**self.ext_index_e[:,None] * self.ext_ratio_e[:,None]
                prep_spec_wave_w  = np.hstack((prep_spec_wave_w,  ext_spec_wave_w ))
                prep_spec_dens_ew = np.hstack((prep_spec_dens_ew, ext_spec_dens_ew))
            # do not convolve in this case
            if_convolve = False 
        else:
            prep_spec_wave_w  = self.prep_spec_wave_w
            prep_spec_dens_ew = self.prep_spec_dens_ew

        mcomp_spec_dens_tw = None
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            if components is not None:
                if comp_name not in components: continue
            if mask_lite_e is not None:
                if sum(mask_lite_ce[i_comp]) == 0: continue

            # build models with SFH function
            if self.sfh_name_c[i_comp] == 'nonparametric':
                if mask_lite_e is not None:
                    intr_spec_dens_tw = prep_spec_dens_ew[mask_lite_ce[i_comp],:] # limit element number for accelarate calculation
                else:
                    intr_spec_dens_tw = prep_spec_dens_ew
            else:
                llam_weight_e = self.llam_weight_from_sfh(i_comp, par_cp[i_comp])
                tmp_mask_e = llam_weight_e > 0
                tmp_ew = np.zeros_like(prep_spec_dens_ew)
                tmp_ew[tmp_mask_e,:] = prep_spec_dens_ew[tmp_mask_e,:] * llam_weight_e[tmp_mask_e,None] # weight with llam_weight_e
                intr_spec_dens_ew = tmp_ew.reshape(self.num_mets, self.num_ages, len(prep_spec_wave_w)).sum(axis=1)
                # sum in ages to create csp 
                if mask_lite_e is not None:
                    intr_spec_dens_tw = intr_spec_dens_ew[mask_lite_ce[i_comp],:]

            # dust extinction
            if if_dust_ext & ('Av' in self.cframe.par_index_cP[i_comp]):
                Av = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['Av']]
                d_spec_dens_tw = intr_spec_dens_tw * 10.0**(-0.4 * Av * ExtLaw(prep_spec_wave_w))
            else:
                d_spec_dens_tw = intr_spec_dens_tw

            # redshift models
            voff = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['voff']]
            z_factor = (1 + self.v0_redshift) * (1 + voff/299792.458) # (1+z) = (1+zv0) * (1+v/c)
            z_spec_wave_w = prep_spec_wave_w * z_factor
            if if_z_decline:
                zd_spec_dens_tw = d_spec_dens_tw / z_factor
            else:
                zd_spec_dens_tw = d_spec_dens_tw

            # convolve with intrinsic and instrumental dispersion if self.R_inst_rw is not None
            if if_convolve & ('fwhm' in self.cframe.par_index_cP[i_comp]) & (self.R_inst_rw is not None) & (conv_nbin is not None):
                fwhm = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['fwhm']]
                R_inst_w = np.interp(z_spec_wave_w, self.R_inst_rw[0], self.R_inst_rw[1])
                czd_spec_dens_tw = convolve_var_width_fft(z_spec_wave_w, zd_spec_dens_tw, dv_fwhm_obj=fwhm, 
                                                          dw_fwhm_ref=self.prep_dw_fwhm_w*z_factor, R_inst_w=R_inst_w, num_bins=conv_nbin)
            else:
                czd_spec_dens_tw = zd_spec_dens_tw # just copy if convlution not required, e.g., for broad-band sed fitting

            # project to observed wavelength
            interp_func = interp1d(z_spec_wave_w, czd_spec_dens_tw, axis=1, kind='linear', fill_value=(0,0), bounds_error=False)
            obs_spec_dens_tw = interp_func(obs_spec_wave_w)

            if mcomp_spec_dens_tw is None: 
                mcomp_spec_dens_tw = obs_spec_dens_tw
            else:
                mcomp_spec_dens_tw = np.vstack((mcomp_spec_dens_tw, obs_spec_dens_tw))

        return mcomp_spec_dens_tw
    
    ##########################################################################
    ########################## Output functions ##############################

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

        self.num_loops = self.fframe.num_loops # for reconstruct_sfh and print_results
        comp_name_c = self.cframe.comp_name_c
        num_comps = self.cframe.num_comps
        par_name_cp = self.cframe.par_name_cp

        # list the properties to be output; the print will follow this order
        value_names_additive = ['log_lamLlam_5500', 
                                'log_Mass_formed', 'log_Mass_remaining', 'log_MtoL', 
                                'log_Age_Lweight', 'log_Age_Mweight', 'log_Z_Lweight', 'log_Z_Mweight']
        ret_names_additive = None
        value_names_C = {}
        for (i_comp, comp_name) in enumerate(comp_name_c):
            value_names_C[comp_name] = ['redshift', 'sigma'] + value_names_additive

            ret_names = []
            if self.cframe.comp_info_cI[i_comp]['ret_emission_set'] is not None:
                for ret_emi_F in self.cframe.comp_info_cI[i_comp]['ret_emission_set']:
                    if 'wave_center' in ret_emi_F:
                        wave_name = f"{ret_emi_F['wave_center']} ({ret_emi_F['wave_frame']}, {ret_emi_F['wave_unit']})"
                    else:
                        wave_name = f"{ret_emi_F['wave_min']}-{ret_emi_F['wave_max']} ({ret_emi_F['wave_frame']}, {ret_emi_F['wave_unit']})"
                    ret_name = f"log_{ret_emi_F['value_form']} ({ret_emi_F['value_state']}, {ret_emi_F['value_unit']}) at {wave_name}"
                    ret_names.append(ret_name)

            if self.cframe.comp_info_cI[i_comp]['ret_SFR_set'] is not None:
                for ret_sfr_F in self.cframe.comp_info_cI[i_comp]['ret_SFR_set']:
                    if 'wave_center' in ret_sfr_F:
                        wave_name = f"{ret_sfr_F['wave_center']} ({ret_sfr_F['wave_frame']}, {ret_sfr_F['wave_unit']})"
                    else:
                        wave_name = f"{ret_sfr_F['wave_min']}-{ret_sfr_F['wave_max']} ({ret_sfr_F['wave_frame']}, {ret_sfr_F['wave_unit']})"
                    if 'age_center' in ret_sfr_F:
                        age_name = f"{ret_sfr_F['age_center']} ({ret_sfr_F['age_unit']})"
                    else:
                        age_name = f"{ret_sfr_F['age_min']}-{ret_sfr_F['age_max']} ({ret_sfr_F['age_unit']})"
                    ret_name = f"log_{ret_sfr_F['value_form']} ({ret_sfr_F['value_state']}, {ret_sfr_F['value_unit']}) in {age_name} at {wave_name}"
                    ret_names.append(ret_name)

            value_names_C[comp_name] += ret_names
            if ret_names_additive is None: 
                ret_names_additive = ret_names
            else:
                ret_names_additive = [ret_name for ret_name in ret_names_additive if ret_name in value_names_C[comp_name]]
        value_names_additive += ret_names_additive

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
        output_C['tot']['value_Vl'] = {} # only init values for sum of all comp
        for value_name in value_names_additive:
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
                if self.sfh_name_c[i_comp] != 'nonparametric':
                    coeff_e = np.tile(coeff_e, (self.num_ages,1)).T.flatten() * self.llam_weight_from_sfh(i_comp, par_p) # get ssp_coeff_e from csp_coeff_e

                voff = output_C[comp_name]['value_Vl']['voff'][i_loop]
                rev_redshift = (1 + self.v0_redshift) * (1 + voff/299792.458) - 1
                output_C[comp_name]['value_Vl']['redshift'][i_loop] = copy(rev_redshift)
                fwhm = output_C[comp_name]['value_Vl']['fwhm'][i_loop]
                output_C[comp_name]['value_Vl']['sigma'][i_loop] = fwhm/np.sqrt(np.log(256))

                lum_area = 4*np.pi * cosmo.luminosity_distance(rev_redshift).to('cm')**2 # with unit of cm2
                lamLlam_5500_e = (coeff_e * u.Unit(self.fframe.spec_flux_unit) * lum_area * self.wave_norm * u.angstrom            ).to('L_sun').value
                Mass_formed_e  = (coeff_e * u.Unit(self.fframe.spec_flux_unit) * lum_area * self.mass_e    * u.Unit(self.mass_unit)).to('M_sun').value
                Mass_remaining_e = Mass_formed_e * self.remain_massfrac_e

                output_C[comp_name]['value_Vl']['log_lamLlam_5500'  ][i_loop] = np.log10(lamLlam_5500_e.sum())
                output_C[comp_name]['value_Vl']['log_Mass_formed'   ][i_loop] = np.log10(Mass_formed_e.sum())
                output_C[comp_name]['value_Vl']['log_Mass_remaining'][i_loop] = np.log10(Mass_remaining_e.sum())
                output_C[comp_name]['value_Vl']['log_MtoL'          ][i_loop] = np.log10(Mass_remaining_e.sum() / lamLlam_5500_e.sum())
                output_C[comp_name]['value_Vl']['log_Age_Lweight'   ][i_loop] = (lamLlam_5500_e   * np.log10(self.age_e)).sum() / lamLlam_5500_e.sum()
                output_C[comp_name]['value_Vl']['log_Age_Mweight'   ][i_loop] = (Mass_remaining_e * np.log10(self.age_e)).sum() / Mass_remaining_e.sum()
                output_C[comp_name]['value_Vl']['log_Z_Lweight'     ][i_loop] = (lamLlam_5500_e   * np.log10(self.met_e)).sum() / lamLlam_5500_e.sum()
                output_C[comp_name]['value_Vl']['log_Z_Mweight'     ][i_loop] = (Mass_remaining_e * np.log10(self.met_e)).sum() / Mass_remaining_e.sum()

                output_C['tot'    ]['value_Vl']['log_lamLlam_5500'  ][i_loop] += lamLlam_5500_e.sum()   # keep in linear for sum
                output_C['tot'    ]['value_Vl']['log_Mass_formed'   ][i_loop] += Mass_formed_e.sum()    # keep in linear for sum
                output_C['tot'    ]['value_Vl']['log_Mass_remaining'][i_loop] += Mass_remaining_e.sum() # keep in linear for sum
                output_C['tot'    ]['value_Vl']['log_Age_Lweight'   ][i_loop] += (lamLlam_5500_e   * np.log10(self.age_e)).sum()
                output_C['tot'    ]['value_Vl']['log_Age_Mweight'   ][i_loop] += (Mass_remaining_e * np.log10(self.age_e)).sum()
                output_C['tot'    ]['value_Vl']['log_Z_Lweight'     ][i_loop] += (lamLlam_5500_e   * np.log10(self.met_e)).sum()
                output_C['tot'    ]['value_Vl']['log_Z_Mweight'     ][i_loop] += (Mass_remaining_e * np.log10(self.met_e)).sum()

                i_coeffs_0_of_comp_in_mod, i_coeffs_1_of_comp_in_mod = self.fframe.search_comp_index(comp_name, self.mod_name, index_all_t=mod_index_lt[i_loop])[2:4]
                coeff_t = mod_coeff_lt[i_loop, i_coeffs_0_of_comp_in_mod:i_coeffs_1_of_comp_in_mod]
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

                # calculate requested flux/Lum in given wavelength ranges
                if self.cframe.comp_info_cI[i_comp]['ret_SFR_set'] is not None: 
                    for ret_sfr_F in self.cframe.comp_info_cI[i_comp]['ret_SFR_set']:
                        if 'age_center' in ret_sfr_F:
                            age_0, age_1 = ret_sfr_F['age_center'] - ret_sfr_F['age_width'], ret_sfr_F['age_center'] + ret_sfr_F['age_width']
                            age_name = f"{ret_sfr_F['age_center']} ({ret_sfr_F['age_unit']})"
                        else:
                            age_0, age_1 = ret_sfr_F['age_min'], ret_sfr_F['age_max']
                            age_name = f"{ret_sfr_F['age_min']}-{ret_sfr_F['age_max']} ({ret_sfr_F['age_unit']})"
                        age_ratio = u.Unit(ret_sfr_F['age_unit']).to('Gyr')
                        mask_age_e = (self.age_e >= (age_0*age_ratio)) & (self.age_e <= (age_1*age_ratio))
                        tmp_duration = (age_1-age_0) * u.Unit(ret_sfr_F['age_unit'])
                        ret_value = Mass_remaining_e[mask_age_e].sum() * u.M_sun / tmp_duration

                        if 'wave_center' in ret_sfr_F:
                            wave_0, wave_1 = ret_sfr_F['wave_center'] - ret_sfr_F['wave_width'], ret_sfr_F['wave_center'] + ret_sfr_F['wave_width']
                            wave_name = f"{ret_sfr_F['wave_center']} ({ret_sfr_F['wave_frame']}, {ret_sfr_F['wave_unit']})"
                        else:
                            wave_0, wave_1 = ret_sfr_F['wave_min'], ret_sfr_F['wave_max']
                            wave_name = f"{ret_sfr_F['wave_min']}-{ret_sfr_F['wave_max']} ({ret_sfr_F['wave_frame']}, {ret_sfr_F['wave_unit']})"
                        wave_ratio = u.Unit(ret_sfr_F['wave_unit']).to('angstrom')
                        if ret_sfr_F['wave_frame'] == 'rest': wave_ratio *= (1+rev_redshift) # rest wave to obs wave, which is required by create_models
                        tmp_wave_w = np.logspace(np.log10(wave_0*wave_ratio), np.log10(wave_1*wave_ratio), num=100) # obs frame grid

                        if ret_sfr_F['value_state'] in ['intrinsic','observed','absorbed']:
                            tmp_flam_tw = self.create_models(tmp_wave_w, mod_par_lp[i_loop], index_all_t=mod_index_lt[i_loop], components=comp_name, 
                                                             if_dust_ext=False, if_z_decline=True, if_full_range=True) # flux in obs frame
                            intrinsic_flam_w = coeff_t @ tmp_flam_tw
                        if ret_sfr_F['value_state'] in ['observed','absorbed']:
                            tmp_flam_tw = self.create_models(tmp_wave_w, mod_par_lp[i_loop], index_all_t=mod_index_lt[i_loop], components=comp_name, 
                                                             if_dust_ext=True,  if_z_decline=True, if_full_range=True) # flux in obs frame
                            observed_flam_w = coeff_t @ tmp_flam_tw
                        if ret_sfr_F['value_state'] == 'intrinsic': tmp_flam_w = intrinsic_flam_w
                        if ret_sfr_F['value_state'] == 'observed' : tmp_flam_w = observed_flam_w
                        if ret_sfr_F['value_state'] == 'absorbed' : tmp_flam_w = intrinsic_flam_w - observed_flam_w

                        tmp_Flam    = tmp_flam_w.mean()
                        tmp_lamFlam = tmp_flam_w.mean() * tmp_wave_w.mean()
                        tmp_intFlux = np.trapezoid(tmp_flam_w, x=tmp_wave_w)
                        int_Flam    = intrinsic_flam_w.mean()
                        int_lamFlam = intrinsic_flam_w.mean() * tmp_wave_w.mean()
                        int_intFlux = np.trapezoid(intrinsic_flam_w, x=tmp_wave_w)

                        if ret_sfr_F['flux_form'] ==    'Flam': ret_value *= tmp_Flam    / int_Flam
                        if ret_sfr_F['flux_form'] == 'lamFlam': ret_value *= tmp_lamFlam / int_lamFlam
                        if ret_sfr_F['flux_form'] == 'intFlux': ret_value *= tmp_intFlux / int_intFlux

                        if ret_sfr_F['value_form'] == 'sSFR': ret_value /= Mass_remaining_e.sum()
                        ret_value = ret_value.to(ret_sfr_F['value_unit']).value
                        ret_name = f"log_{ret_sfr_F['value_form']} ({ret_sfr_F['value_state']}, {ret_sfr_F['value_unit']}) in {age_name} at {wave_name}"
                        output_C[comp_name]['value_Vl'][ret_name][i_loop] = np.log10(ret_value)
                        if ret_name in output_C['tot']['value_Vl']: output_C['tot']['value_Vl'][ret_name][i_loop] += ret_value

        output_C['tot']['value_Vl']['log_MtoL'          ] = np.log10(output_C['tot']['value_Vl']['log_Mass_remaining'] / output_C['tot']['value_Vl']['log_lamLlam_5500'])
        output_C['tot']['value_Vl']['log_Age_Lweight'   ] = output_C['tot']['value_Vl']['log_Age_Lweight'] / output_C['tot']['value_Vl']['log_lamLlam_5500']
        output_C['tot']['value_Vl']['log_Age_Mweight'   ] = output_C['tot']['value_Vl']['log_Age_Mweight'] / output_C['tot']['value_Vl']['log_Mass_remaining']
        output_C['tot']['value_Vl']['log_Z_Lweight'     ] = output_C['tot']['value_Vl']['log_Z_Lweight']   / output_C['tot']['value_Vl']['log_lamLlam_5500']
        output_C['tot']['value_Vl']['log_Z_Mweight'     ] = output_C['tot']['value_Vl']['log_Z_Mweight']   / output_C['tot']['value_Vl']['log_Mass_remaining']
        output_C['tot']['value_Vl']['log_Mass_formed'   ] = np.log10(output_C['tot']['value_Vl']['log_Mass_formed'])
        output_C['tot']['value_Vl']['log_Mass_remaining'] = np.log10(output_C['tot']['value_Vl']['log_Mass_remaining'])
        for value_name in output_C['tot']['value_Vl']:
            if (value_name[:8] in ['log_Flam', 'log_Fnu ', 'log_SFR ', 'log_sSFR']) | (value_name[:11] in ['log_lamFlam', 'log_intFlux', 'log_lamLlam', 'log_intLum ']): 
                output_C['tot']['value_Vl'][value_name] = np.log10(output_C['tot']['value_Vl'][value_name])

        i_comp = 0 # only enable one comp if nonparametric SFH is used
        if self.sfh_name_c[i_comp] == 'nonparametric':
            output_C[comp_name_c[i_comp]]['coeff_norm_le'] = mod_coeff_le / mod_coeff_le.sum(axis=1)[:,None]

        ############################################################
        # keep aliases for output in old version <= 2.2.4
        for (i_comp, comp_name) in enumerate(comp_name_c):
            output_C[comp_name]['values'] = output_C[comp_name]['value_Vl']
        output_C['tot']['values'] = output_C['tot']['value_Vl']
        ############################################################

        self.mod_par_lp = mod_par_lp # save for reconstruct_sfh
        self.mod_coeff_le = mod_coeff_le # save for reconstruct_sfh
        self.output_C = output_C # save to model frame
        if if_print_results: self.print_results(log=self.fframe.log_message, if_show_average=if_show_average)
        if if_return_results: return output_C

    def print_results(self, log=[], if_show_average=False):
        print_log(f"#### Best-fit stellar properties ####", log)

        mask_l = np.ones(self.num_loops, dtype='bool')
        if not if_show_average: mask_l[1:] = False

        # set the print name for each value
        print_name_CV = {}
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            print_name_CV[comp_name] = {}
            for value_name in self.output_C[comp_name]['value_Vl']: print_name_CV[comp_name][value_name] = value_name

            print_name_CV[comp_name]['voff'] = 'Velocity shift in relative to z_sys (km s-1)'
            print_name_CV[comp_name]['fwhm'] = 'Velocity FWHM (km s-1)'
            print_name_CV[comp_name]['sigma'] = 'Velocity dispersion (σ) (km s-1)'
            print_name_CV[comp_name]['Av'] = 'Extinction (Av)'
            print_name_CV[comp_name]['log_csp_age'] = f"Maximum age of composite stellar population (log {self.age_unit})"
            print_name_CV[comp_name]['log_csp_tau'] = f"Declining timescale of SFH (log {self.age_unit})"
            print_name_CV[comp_name]['redshift'] = 'Redshift (from continuum absorptions)'
            print_name_CV[comp_name]['log_Mass_formed'] = 'Stellar mass (total mass formed during lifetime) (log M☉)'
            print_name_CV[comp_name]['log_Mass_remaining'] = 'Stellar mass (currently remaining mass) (log M☉)'
            print_name_CV[comp_name]['log_MtoL'] = f"Mass-to-light (λL{self.wave_norm}) ratio (log M☉/L☉)"
            print_name_CV[comp_name]['log_Age_Lweight'] = f"Luminosity-weight age (log {self.age_unit})"
            print_name_CV[comp_name]['log_Age_Mweight'] = f"Mass-weight age (log {self.age_unit})"
            print_name_CV[comp_name]['log_Z_Lweight'] = 'Luminosity-weight metallicity (log Z)'
            print_name_CV[comp_name]['log_Z_Mweight'] = 'Mass-weight metallicity (log Z)'

            if self.cframe.comp_info_cI[i_comp]['ret_emission_set'] is not None: 
                for ret_emi_F in self.cframe.comp_info_cI[i_comp]['ret_emission_set']:
                    if 'wave_center' in ret_emi_F:
                        wave_name = f"{ret_emi_F['wave_center']} ({ret_emi_F['wave_frame']}, {ret_emi_F['wave_unit']})"
                    else:
                        wave_name = f"{ret_emi_F['wave_min']}-{ret_emi_F['wave_max']} ({ret_emi_F['wave_frame']}, {ret_emi_F['wave_unit']})"
                    ret_name = f"log_{ret_emi_F['value_form']} ({ret_emi_F['value_state']}, {ret_emi_F['value_unit']}) at {wave_name}"

                    tmp_value_state = copy(ret_emi_F['value_state']) # avoid changing the original set, which is used elsewhere
                    if tmp_value_state == 'absorbed': tmp_value_state = 'dust-'+tmp_value_state
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

            if self.cframe.comp_info_cI[i_comp]['ret_SFR_set'] is not None: 
                for ret_sfr_F in self.cframe.comp_info_cI[i_comp]['ret_SFR_set']:
                    if 'age_center' in ret_sfr_F:
                        age_name = f"{ret_sfr_F['age_center']} ({ret_sfr_F['age_unit']})"
                    else:
                        age_name = f"{ret_sfr_F['age_min']}-{ret_sfr_F['age_max']} ({ret_sfr_F['age_unit']})"

                    if 'wave_center' in ret_sfr_F:
                        wave_center = ret_sfr_F['wave_center']
                        wave_name = f"{ret_sfr_F['wave_center']} ({ret_sfr_F['wave_frame']}, {ret_sfr_F['wave_unit']})"
                    else:
                        wave_center = (ret_sfr_F['wave_min'] + ret_sfr_F['wave_max']) / 2
                        wave_name = f"{ret_sfr_F['wave_min']}-{ret_sfr_F['wave_max']} ({ret_sfr_F['wave_frame']}, {ret_sfr_F['wave_unit']})"

                    ret_name = f"log_{ret_sfr_F['value_form']} ({ret_sfr_F['value_state']}, {ret_sfr_F['value_unit']}) in {age_name} at {wave_name}"

                    tmp_value_state = copy(ret_sfr_F['value_state']) # avoid changing the original set, which is used elsewhere
                    if tmp_value_state == 'absorbed' : tmp_value_state = 'dust-'+tmp_value_state
                    print_name_CV[comp_name][ret_name] = f"{tmp_value_state.capitalize()} "

                    tmp_value_unit = ret_sfr_F['value_unit'].replace('M_sun', 'M☉')
                    if ret_sfr_F['value_form'] ==  'SFR': print_name_CV[comp_name][ret_name] += f"SFR (log {tmp_value_unit})"
                    if ret_sfr_F['value_form'] == 'sSFR': print_name_CV[comp_name][ret_name] += f"sSFR (log {tmp_value_unit})"

                    if 'age_center' in ret_sfr_F:
                        print_name_CV[comp_name][ret_name] += f" in {ret_sfr_F['age_center']}+/-{ret_sfr_F['age_width']} {ret_sfr_F['age_unit']}"
                    else:
                        print_name_CV[comp_name][ret_name] += f" in {ret_sfr_F['age_min']}-{ret_sfr_F['age_max']} {ret_sfr_F['age_unit']}"

                    tmp_wave_unit = ret_sfr_F['wave_unit'].replace('angstrom', 'Å').replace('Angstrom', 'Å').replace('um', 'µm').replace('micron', 'µm')
                    tmp_wave_frame = copy(ret_sfr_F['wave_frame'])
                    if tmp_wave_frame == 'obs': tmp_wave_frame += '.'
                    print_name_CV[comp_name][ret_name] += f"at {tmp_wave_frame} {wave_center} {tmp_wave_unit}"

        print_name_CV['tot'] = {}
        for value_name in self.output_C['tot']['value_Vl']:
            print_name_CV['tot'][value_name] = copy(print_name_CV[self.comp_name_c[0]][value_name])

        print_length = max([len(print_name_CV[comp_name][value_name]) for comp_name in print_name_CV for value_name in print_name_CV[comp_name]] + [40]) # set min length
        for comp_name in print_name_CV:
            for value_name in print_name_CV[comp_name]:
                print_name_CV[comp_name][value_name] += ' '*(print_length-len(print_name_CV[comp_name][value_name]))

        for (i_comp, comp_name) in enumerate(self.output_C):
            value_Vl = self.output_C[[*self.output_C][i_comp]]['value_Vl']
            value_names = [*value_Vl]
            msg = ''
            if i_comp < self.cframe.num_comps: # print best-fit pars for each comp
                print_log(f"# Stellar component <{self.cframe.comp_name_c[i_comp]}> with {self.sfh_name_c[i_comp]} SFH:", log)
                value_names = [value_name for value_name in value_names if value_name[:6] != 'Empty_'] # remove unused pars
                value_names.remove('redshift'); value_names = ['redshift'] + value_names # move redshift to the begining
                value_names.remove('sigma'); value_names = ['sigma' if value_name == 'fwhm' else value_name for value_name in value_names] # print sigma instead of fwhm
            elif self.cframe.num_comps >= 2: # print sum only if using >= 2 comps
                print_log(f"# Best-fit properties of the sum of all stellar components.", log)
            else: 
                continue
            value_names.remove('log_lamLlam_5500')
            for value_name in value_names:
                msg += '| ' + print_name_CV[comp_name][value_name] + f" = {value_Vl[value_name][mask_l].mean():10.4f}" + f" +/- {value_Vl[value_name].std():<10.4f}|\n"
            msg = msg[:-1] # remove the last \n
            bar = '=' * len(msg.split('\n')[-1])
            print_log(bar, log)
            print_log(msg, log)
            print_log(bar, log)
            print_log('', log)

        i_comp = 0 # only enable one comp if nonparametric SFH is used
        if self.sfh_name_c[i_comp] == 'nonparametric':
            print_log('# Best-fit single stellar populations (SSP) with nonparametric SFH', log)
            cols = f"ID,Age ({self.age_unit}),Metallicity,Coeff.mean,Coeff.rms,log(M/λL{self.wave_norm})"
            fmt_cols = '| {0:^6} | {1:^9} | {2:^11} | {3:^10} | {4:^10} | {5:^14} |'
            fmt_numbers = '| {:^6d} | {:^9.4f} | {:^11.4f} | {:^10.4f} | {:^10.4f} | {:^14.4f} |'
            cols_split = cols.split(',')
            tbl_title = fmt_cols.format(*cols_split)
            tbl_border = len(tbl_title)*'-'
            print_log(tbl_border, log)
            print_log(tbl_title, log)
            print_log(tbl_border, log)
            coeff_norm_mn_e  = self.output_C[self.cframe.comp_name_c[i_comp]]['coeff_norm_le'][mask_l].mean(axis=0)
            coeff_norm_std_e = self.output_C[self.cframe.comp_name_c[i_comp]]['coeff_norm_le'].std(axis=0)
            mtol_e = ((self.mass_e * u.Unit(self.mass_unit)) / (self.wave_norm * u.angstrom)).to('M_sun L_sun-1').value
            for i_e in range(self.num_templates):
                if coeff_norm_mn_e[i_e] < 0.001: continue
                tbl_row = []
                tbl_row.append(i_e)
                tbl_row.append(self.age_e[i_e])
                tbl_row.append(self.met_e[i_e])
                tbl_row.append(coeff_norm_mn_e[i_e]) 
                tbl_row.append(coeff_norm_std_e[i_e])
                tbl_row.append(np.log10(mtol_e[i_e]))
                print_log(fmt_numbers.format(*tbl_row), log)
            print_log(tbl_border, log)
            print_log(f"[Note] Coeff is the normalized fraction of the intrinsic flux at rest {self.wave_norm} Å.", log)
            print_log(f"[Note] only SSPs with Coeff over 0.1% are listed.", log)
            print_log(f"[Note] Mass-to-luminosity ratio is in the unit of log log M☉/L☉.", log)
            print_log('', log)

    def reconstruct_sfh(self, output_C=None, num_bins=None, if_plot_sfh=True, if_return_sfh=False, if_show_average=True, **kwargs):

        ############################################################
        # check and replace the args to be compatible with old version <= 2.2.4
        if 'plot'         in kwargs: if_plot_sfh     = kwargs['plot']
        if 'return_sfh'   in kwargs: if_return_sfh   = kwargs['return_sfh']
        if 'show_average' in kwargs: if_show_average = kwargs['show_average']
        ############################################################

        mask_l = np.ones(self.num_loops, dtype='bool')
        if not if_show_average: mask_l[1:] = False

        if output_C is None: output_C = self.output_C
        comp_name_c = self.cframe.comp_name_c
        num_comps = self.cframe.num_comps

        age_a = self.age_a
        output_sfh_lcza = np.zeros((self.num_loops, num_comps, self.num_mets, self.num_ages))

        for (i_comp, comp_name) in enumerate(comp_name_c):
            for i_loop in range(self.num_loops):
                par_p   = self.cframe.reshape_by_comp(self.mod_par_lp  [i_loop], self.cframe.num_pars_c)[i_comp]
                coeff_e = self.cframe.reshape_by_comp(self.mod_coeff_le[i_loop], self.num_coeffs_c     )[i_comp]
                if self.sfh_name_c[i_comp] != 'nonparametric':
                    coeff_e = np.tile(coeff_e, (self.num_ages,1)).T.flatten() * self.llam_weight_from_sfh(i_comp, par_p) # get ssp_coeff_e from csp_coeff_e

                voff = output_C[comp_name]['value_Vl']['voff'][i_loop]
                rev_redshift = (1 + self.v0_redshift) * (1 + voff/299792.458) - 1
                lum_area = 4*np.pi * cosmo.luminosity_distance(rev_redshift).to('cm')**2 # with unit of cm2

                sfr_e = (coeff_e * u.Unit(self.fframe.spec_flux_unit) * lum_area * self.sfr_e * u.Unit(self.sfr_unit)).to('M_sun yr-1').value
                output_sfh_lcza[i_loop,i_comp,:,:] = sfr_e.reshape(self.num_mets, self.num_ages)

        if num_bins is not None:
            output_sfh_lczb = np.zeros((self.num_loops, num_comps, self.num_mets, num_bins))
            age_b = np.zeros((num_bins))
            log_age_a = np.log10(age_a)
            bwidth = (log_age_a[-1] - log_age_a[0]) / num_bins
            for i_bin in range(num_bins):
                mask_a = (log_age_a > (log_age_a[0]+i_bin*bwidth)) & (log_age_a <= (log_age_a[0]+(i_bin+1)*bwidth))
                duration_b = 10.0**(log_age_a[0]+(i_bin+1)*bwidth) - 10.0**(log_age_a[0]+i_bin*bwidth)
                output_sfh_lczb[:,:,:,i_bin] = (output_sfh_lcza[:,:,:,mask_a] * self.duration_e[:self.num_ages][mask_a]).sum(axis=3) / duration_b
                age_b[i_bin] = 10.0**(log_age_a[0]+(i_bin+1/2)*bwidth)
                
        if if_plot_sfh:
            plt.figure(figsize=(9,3))
            plt.subplots_adjust(bottom=0.15, top=0.9, left=0.08, right=0.98, hspace=0, wspace=0.2)
            ax = plt.subplot(1, 2, 1)
            for i_comp in range(output_sfh_lcza.shape[1]):  
                for i_loop in range(output_sfh_lcza.shape[0]):
                    plt.plot(np.log10(age_a), output_sfh_lcza[i_loop,i_comp,:,:].sum(axis=0), '--')
                plt.plot(np.log10(age_a), output_sfh_lcza[:,i_comp,:,:].sum(axis=1)[mask_l].mean(axis=0), linewidth=4, alpha=0.5, label=f'Mean {self.cframe.comp_name_c[i_comp]}')
            plt.xlim(1.5,-3); plt.ylim(1,1e4); plt.yscale('log')
            plt.xlabel(f"Looking back time (log {self.age_unit})"); plt.ylabel('SFR (M☉/yr)'); plt.legend()
            plt.title('Before binning in log time')

            if num_bins is not None:
                ax = plt.subplot(1, 2, 2)
                for i_comp in range(output_sfh_lczb.shape[1]):  
                    for i_loop in range(output_sfh_lczb.shape[0]):
                        plt.bar(np.log10(age_b), output_sfh_lczb[i_loop,i_comp,:,:].sum(axis=0), bottom=0, width=(np.log10(age_b)[1]-np.log10(age_b)[0])*0.8, 
                        alpha=0.5/output_sfh_lczb.shape[0])
                    plt.bar(np.log10(age_b), output_sfh_lczb[:,i_comp,:,:].sum(axis=1)[mask_l].mean(axis=0), bottom=0, width=(np.log10(age_b)[1]-np.log10(age_b)[0])*0.8,
                           alpha=0.3, hatch='///', ec='C7', linewidth=4, label=f'Mean {self.cframe.comp_name_c[i_comp]}')
                plt.xlim(1.5,-3); plt.ylim(1,1e4); plt.yscale('log')
                plt.xlabel(f"Looking back time ({self.age_unit})"); plt.ylabel('SFR (M☉/yr)'); plt.legend()
                plt.title('After binning in log time')
                
        if if_return_sfh:
            if num_bins is None:
                return output_sfh_lcza, age_a
            else:
                return output_sfh_lczb, age_b
