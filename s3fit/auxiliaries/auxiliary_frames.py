# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: s3fit@xychen.me

import os
from copy import deepcopy as copy
import numpy as np
np.set_printoptions(linewidth=10000)
import astropy.units as u
import astropy.constants as const

from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
def import_module_from_path(path):
    spec = spec_from_file_location('_dynamic', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# import from absolute path to be used by model frame out of installed directory
aux_func = import_module_from_path(str(Path(__file__).parent) + '/auxiliary_functions.py')
casefold, fnu_over_flam, spec_to_phot = aux_func.casefold, aux_func.fnu_over_flam, aux_func.spec_to_phot
# from .auxiliary_functions import casefold, fnu_over_flam, spec_to_phot

class ConfigFrame(object):

    def __init__(self, mod_reg):

        #########################################
        # model level
        if 'mod_info_I' in mod_reg: 
            self.mod_info_I = mod_reg['mod_info_I']
        elif 'mod_info' in mod_reg: 
            self.mod_info_I = mod_reg['mod_info']
        elif 'info' in mod_reg: 
            self.mod_info_I = mod_reg['info']
        else:
            self.mod_info_I = {}
        #########################################

        #########################################
        # components level
        if 'comp_reg_C' in mod_reg: 
            self.comp_reg_C = mod_reg['comp_reg_C']
        elif 'comps' in mod_reg: 
            self.comp_reg_C = mod_reg['comps']
        else:
            self.comp_reg_C = mod_reg

        self.num_comps = len(self.comp_reg_C)
        self.comp_name_c = [*self.comp_reg_C]
        self.comp_index_C = {comp_name: i_comp for (i_comp, comp_name) in enumerate(self.comp_reg_C)}

        self.comp_info_cI = [] 
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            comp_reg = self.comp_reg_C[comp_name]

            if 'comp_info_I' in comp_reg: 
                comp_info_I = comp_reg['comp_info_I']
            elif 'mod_info' in comp_reg: 
                comp_info_I = comp_reg['comp_info']
            elif 'info' in comp_reg: 
                comp_info_I = comp_reg['info']
            else:
                comp_info_I = {}

            self.comp_info_cI.append(comp_info_I)
            self.comp_info_cI[i_comp]['comp_name'] = comp_name

            # group used model elements in an array
            for item in ['mod_used', 'line_used']:
                if item in self.comp_info_cI[i_comp]: 
                    if isinstance(self.comp_info_cI[i_comp][item], str): self.comp_info_cI[i_comp][item] = [self.comp_info_cI[i_comp][item]]
                    self.comp_info_cI[i_comp][item] = np.array(self.comp_info_cI[i_comp][item])

            # rename sign for absorption/emission
            if 'sign' in self.comp_info_cI[i_comp]:
                if casefold(self.comp_info_cI[i_comp]['sign']) in ['absorption', 'negative', '-']:
                    self.comp_info_cI[i_comp]['sign'] = 'absorption'
                if casefold(self.comp_info_cI[i_comp]['sign']) in ['emission', 'positive', '+']:
                    self.comp_info_cI[i_comp]['sign'] = 'emission'
            else:
                self.comp_info_cI[i_comp]['sign'] = 'emission' # default
        #########################################

        #########################################
        # parameters level
        self.num_pars_c = []
        self.par_min_cp = []
        self.par_max_cp = []
        self.par_tie_cp = []
        self.par_name_cp = []
        self.par_index_cP = [] # index of each par_name in each comp

        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            comp_reg = self.comp_reg_C[comp_name]

            if 'par_key_PK' in comp_reg: 
                par_key_PK = comp_reg['par_key_PK']
            elif 'pars' in comp_reg: 
                par_key_PK = comp_reg['pars']

            self.num_pars_c.append(len(par_key_PK))

            par_min_p = []
            par_max_p = []
            par_tie_p = []
            par_name_p = []
            par_index_P = {}

            for i_par in range(len(par_key_PK)):
                if isinstance(par_key_PK, list):
                    par_key_pk = par_key_PK
                    par_min_p.append(par_key_pk[i_par][0])
                    par_max_p.append(par_key_pk[i_par][1])
                    par_tie_p.append(par_key_pk[i_par][2])
                elif isinstance(par_key_PK, dict):
                    par_name = [*par_key_PK][i_par]
                    par_name_p.append(par_name)
                    par_index_P[par_name] = i_par
                    if isinstance(par_key_PK[par_name], list):
                        par_key_Pk = par_key_PK
                        par_min_p.append(par_key_Pk[par_name][0])
                        par_max_p.append(par_key_Pk[par_name][1])
                        par_tie_p.append(par_key_Pk[par_name][2])
                    elif isinstance(par_key_PK[par_name], dict):
                        par_min_p.append(par_key_PK[par_name]['min'])
                        par_max_p.append(par_key_PK[par_name]['max'])
                        par_tie_p.append(par_key_PK[par_name]['tie'])

            self.par_min_cp.append(par_min_p)
            self.par_max_cp.append(par_max_p)
            self.par_tie_cp.append(par_tie_p)
            self.par_name_cp.append(par_name_p)
            self.par_index_cP.append(par_index_P)

        self.num_pars_tot = sum(self.num_pars_c)
        self.num_pars = self.num_pars_tot
        #########################################

    ###################################

    # add the synchronized _C/_CP views of _c/_cp lists
    # they can be callback, e.g., self.comp_info_CI (without '()') but cannot be modified directly
    @property
    def comp_info_CI(self):
        return self.convert_c_to_C(self.comp_info_cI)
    @property
    def num_pars_C(self):
        return self.convert_c_to_C(self.num_pars_c)
    @property
    def par_name_Cp(self):
        return self.convert_c_to_C(self.par_name_cp)
    @property
    def par_index_CP(self):
        return self.convert_c_to_C(self.par_index_cP)
    @property
    def par_min_CP(self):
        return self.convert_c_to_C(self.convert_cp_to_cP(self.par_min_cp))
    @property
    def par_max_CP(self):
        return self.convert_c_to_C(self.convert_cp_to_cP(self.par_max_cp))
    @property
    def par_tie_CP(self):
        return self.convert_c_to_C(self.convert_cp_to_cP(self.par_tie_cp))

    # add the synchronized flattened _p views of _cp lists
    @property
    def par_min_p(self):
        return self.flatten_in_comp(self.par_min_cp)
    @property
    def par_max_p(self):
        return self.flatten_in_comp(self.par_max_cp)
    @property
    def par_tie_p(self):
        return self.flatten_in_comp(self.par_tie_cp)
    @property
    def par_name_p(self):
        return self.flatten_in_comp(self.par_name_cp)
    @property
    def comp_name_p(self):
        return [comp_name for comp_name in self.par_name_Cp for par_name in self.par_name_Cp[comp_name]]

    ###################################

    def convert_c_to_C(self, list_c):
        dict_C = {comp_name:x for (comp_name, x) in zip(self.comp_name_c, list_c)}
        return dict_C

    def convert_C_to_c(self, dict_C):
        list_c = [dict_C[comp_name] for comp_name in dict_C]
        return list_c

    def convert_cp_to_cP(self, input_cp):
        if isinstance(input_cp, list):
            list_cp = copy(input_cp)
            list_cP = copy(input_cp)
            for (i_comp, comp_name) in enumerate(self.comp_name_c):
                list_cP[i_comp] = {par_name:x for (par_name, x) in zip(self.par_name_cp[i_comp], list_cp[i_comp])}
            return list_cP
        elif isinstance(input_cp, dict):
            dict_Cp = copy(input_cp)
            dict_CP = copy(input_cp)
            for (i_comp, comp_name) in enumerate(self.comp_name_c):
                dict_CP[comp_name] = {par_name:x for (par_name, x) in zip(self.par_name_cp[i_comp], dict_Cp[comp_name])}
            return dict_CP

    def convert_cP_to_cp(self, input_cP):
        if isinstance(input_cP, list):
            list_cP = copy(input_cP)
            list_cp = copy(input_cP)
            for (i_comp, comp_name) in enumerate(self.comp_name_c):
                list_cp[i_comp] = [list_cP[i_comp][par_name] for par_name in list_cP[i_comp]]
            return list_cp
        elif isinstance(input_cP, dict):
            dict_CP = copy(input_cP)
            dict_Cp = copy(input_cP)
            for (i_comp, comp_name) in enumerate(self.comp_name_c):
                dict_Cp[comp_name] = [dict_CP[comp_name][par_name] for par_name in dict_CP[comp_name]]
            return dict_Cp

    def flatten_in_comp(self, input_cx):
        if isinstance(input_cx, list): 
            list_cx = copy(input_cx)
            list_flat_x = [x for list_x in list_cx for x in list_x]
        elif isinstance(input_cx, dict): 
            dict_Cx = copy(input_cx)
            list_flat_x = [dict_Cx[comp_name][x] if isinstance(dict_Cx[comp_name], dict) else x for comp_name in dict_Cx for x in dict_Cx[comp_name]]
        return list_flat_x

    def reshape_by_comp(self, list_flat_x, num_x_c=None, num_x_C=None, ret='list_c'):
        if (num_x_c is None) and (num_x_C is not None): num_x_c = self.convert_C_to_c(num_x_C)

        list_cx = []
        i_x_0 = 0
        i_x_1 = 0
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            i_x_0 += 0 if i_comp == 0 else num_x_c[i_comp-1]
            i_x_1 += num_x_c[i_comp]
            list_cx.append(list_flat_x[i_x_0:i_x_1])

        if ret in ['list_c', 'list', '_c']:
            return list_cx
        elif ret in ['dict_C', 'dict', '_C']:
            return self.convert_c_to_C(list_cx)

    def retrieve_inherited_info(self, info_name=None, alt_names=[None], comp_name=None, i_comp=None, root_info_I={}, default=None):
        if isinstance(alt_names, str): alt_names = [alt_names]

        if (comp_name is not None) | (i_comp is not None):
            if i_comp is None: i_comp = self.comp_index_C[comp_name]

            if info_name in self.comp_info_cI[i_comp]:
                return
            elif any(np.isin(alt_names, [*self.comp_info_cI[i_comp]])):
                alt_name = alt_names[np.where(np.isin(alt_names, [*self.comp_info_cI[i_comp]]))[0][0]]
                self.comp_info_cI[i_comp][info_name] = copy(self.comp_info_cI[i_comp][alt_name])

            elif info_name in self.mod_info_I:
                self.comp_info_cI[i_comp][info_name] = copy(self.mod_info_I[info_name])
            elif any(np.isin(alt_names, [*self.mod_info_I])):
                alt_name = alt_names[np.where(np.isin(alt_names, [*self.mod_info_I]))[0][0]]
                self.comp_info_cI[i_comp][info_name] = copy(self.mod_info_I[alt_name])

            elif info_name in root_info_I:
                self.comp_info_cI[i_comp][info_name] = copy(root_info_I[info_name])
            elif any(np.isin(alt_names, [*root_info_I])):
                alt_name = alt_names[np.where(np.isin(alt_names, [*root_info_I]))[0][0]]
                self.comp_info_cI[i_comp][info_name] = copy(root_info_I[alt_name])

            else:
                self.comp_info_cI[i_comp][info_name] = copy(default)

        else:
            if info_name in self.mod_info_I:
                return
            elif any(np.isin(alt_names, [*self.mod_info_I])):
                alt_name = alt_names[np.where(np.isin(alt_names, [*self.mod_info_I]))[0][0]]
                self.mod_info_I[info_name] = copy(self.mod_info_I[alt_name])

            elif info_name in root_info_I:
                self.mod_info_I[info_name] = copy(root_info_I[info_name])
            elif any(np.isin(alt_names, [*root_info_I])):
                alt_name = alt_names[np.where(np.isin(alt_names, [*root_info_I]))[0][0]]
                self.mod_info_I[info_name] = copy(root_info_I[alt_name])

            else:
                self.mod_info_I[info_name] = copy(default)

###################################################################################################
###################################################################################################

class PhotFrame(object):
    def __init__(self, 
                 name_b=None, flux_b=None, ferr_b=None, # on input data
                 input_flux_unit='mJy', output_flux_unit='erg s-1 cm-2 angstrom-1', 
                 trans_dir=None, trans_rsmp=10, # on transmission curves
                 wave_w=None, wave_unit='angstrom', wave_num=None): # on corresonding SED range
        # add file_bac, file_iron later
        
        self.name_b = copy(name_b)
        self.flux_b = copy(flux_b)
        self.ferr_b = copy(ferr_b) 
        self.input_flux_unit  = input_flux_unit
        self.output_flux_unit = output_flux_unit

        self.trans_dir = copy(trans_dir)
        self.trans_rsmp = trans_rsmp

        self.wave_w = copy(wave_w)
        self.wave_unit = wave_unit
        if (self.wave_w is not None): 
            self.wave_w *= u.Unit(self.wave_unit).to('angstrom') # convert to angstrom
        self.wave_num = wave_num
                
        self.trans_dict, self.trans_bw, self.wave_w = self.read_transmission(name_b=self.name_b, 
                                                                             trans_dir=self.trans_dir, trans_rsmp=self.trans_rsmp,  
                                                                             wave_w=self.wave_w, wave_num=self.wave_num)
        self.wave_b = spec_to_phot(self.wave_w, self.wave_w, self.trans_bw)

        # convert flux unit
        if u.Unit(self.input_flux_unit).is_equivalent(self.output_flux_unit):
            flux_ratio_b = u.Unit(self.input_flux_unit).to(self.output_flux_unit)
        elif u.Unit(self.input_flux_unit).is_equivalent('mJy'):
            flux_ratio_b = 1 / fnu_over_flam(self.wave_w, trans_bw=self.trans_bw, flam_unit=self.output_flux_unit, fnu_unit=self.input_flux_unit)
        elif u.Unit(self.input_flux_unit).is_equivalent('erg s-1 cm-2 angstrom-1'):
            flux_ratio_b = fnu_over_flam(self.wave_w, trans_bw=self.trans_bw, flam_unit=self.input_flux_unit, fnu_unit=self.output_flux_unit)
        else:
            raise ValueError((f"The input_flux_unit, {self.input_flux_unit}, is not a valid flux unit."))
        self.flux_b *= flux_ratio_b 
        self.ferr_b *= flux_ratio_b 
        
    def read_transmission(self, name_b=None, trans_dir=None, trans_rsmp=None, wave_w=None, wave_num=None):        
        if trans_dir[-1] != '/': trans_dir += '/'

        if name_b is None:
            file_list = np.array(os.listdir(trans_dir))
            file_list = file_list[np.array([f[-4:] for f in file_list]) == '.dat']
            name_b = np.array([f.split('.dat')[0] for f in file_list])
            trans_bw = None
        else:
            trans_bw = 1

        if wave_w is None:
            w_min, w_max = 1e16, 0
            logw_width_min = 1e16
            for name in name_b: 
                filterdata = np.loadtxt(trans_dir+name+'.dat')
                wave_ini, trans_ini = filterdata[:,0], filterdata[:,1] # wave_ini in angstrom
                w_min = np.minimum(w_min, wave_ini.min())
                w_max = np.maximum(w_max, wave_ini.max())
                mask_w = trans_ini/trans_ini.max() > 0.5 
                logw_width = np.log10(wave_ini[mask_w].max()) - np.log10(wave_ini[mask_w].min()) # FWHM in log
                logw_width_min = np.minimum(logw_width_min, logw_width)
            if wave_num is None: 
                wave_num = int((np.log10(w_max) - np.log10(w_min)) / (logw_width_min / trans_rsmp))
            wave_w = np.logspace(np.log10(w_min)-0.1, np.log10(w_max)+0.1, num=wave_num)

        trans_dict = {}
        for name in name_b:
            filterdata = np.loadtxt(trans_dir+name+'.dat')
            wave_ini, trans_ini = filterdata[:,0], filterdata[:,1]
            trans_ini /= np.trapezoid(trans_ini, x=wave_ini)
            # trans_ini /= np.trapezoid(trans_ini * const.c.to('micron Hz').value / wave_ini**2, x=wave_ini) # in Hz-1
            tmp_center = np.trapezoid(trans_ini*wave_ini, x=wave_ini)
            tmp_trans = np.interp(wave_w, wave_ini, trans_ini, left=0, right=0)
            # tmp_trans = np.interp(wave_w, np.array([wave_w[0]]+list(wave_ini)+[wave_w[-1]]), np.array([0]+list(trans_ini)+[0]), left=0, right=0)
            # tmp_trans[wave_w < wave_ini.min()] = 0; tmp_trans[wave_w > wave_ini.max()] = 0
            tmp_dict = {'center': tmp_center, 'trans': tmp_trans}
            trans_dict[name] = tmp_dict

        if trans_bw is not None:
            trans_bw = []
            for name in name_b:
                trans_bw.append(trans_dict[name]['trans'])
            trans_bw = np.array(trans_bw)

        return trans_dict, trans_bw, wave_w

