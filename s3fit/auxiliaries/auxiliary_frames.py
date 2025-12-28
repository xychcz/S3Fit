# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: s3fit@xychen.me

import os
import numpy as np
np.set_printoptions(linewidth=10000)
from copy import deepcopy as copy
import astropy.units as u
import astropy.constants as const

from .auxiliary_functions import casefold

class ConfigFrame(object):
    def __init__(self, config_C):
        self.config_C = config_C
        self.comp_name_c = np.array([*config_C])
        self.num_comps = len(config_C)
        self.num_pars_c = [len(config_C[comp_name]['pars']) for comp_name in config_C]
        self.num_pars_c_max = max(self.num_pars_c)
        self.num_pars_c_tot = self.num_pars_c_max * self.num_comps # total number of pars of all comps (including placeholders)
        self.num_pars = self.num_pars_c_tot

        self.par_min_cp = np.full((self.num_comps, self.num_pars_c_max), -9999, dtype='float')
        self.par_max_cp = np.full((self.num_comps, self.num_pars_c_max), +9999, dtype='float')
        self.par_tie_cp = np.full((self.num_comps, self.num_pars_c_max), 'None', dtype='<U256')
        self.par_name_cp = np.array([['Empty_'+str(i_par) for i_par in range(self.num_pars_c_max)] for i_comp in range(self.num_comps)]).astype('<U256')
        self.par_index_cP = [] # index of each par in each comp
        self.info_c = [] 

        for i_comp in range(self.num_comps):
            input_pars = config_C[self.comp_name_c[i_comp]]['pars']
            par_index_P = {}
            for i_par in range(self.num_pars_c[i_comp]):
                if isinstance(input_pars, list):
                    par_pk = input_pars
                    self.par_min_cp[i_comp,i_par] = par_pk[i_par][0]
                    self.par_max_cp[i_comp,i_par] = par_pk[i_par][1]
                    self.par_tie_cp[i_comp,i_par] = par_pk[i_par][2]
                elif isinstance(input_pars, dict):
                    par_name = [*input_pars][i_par]
                    par_index_P[par_name] = i_par
                    self.par_name_cp[i_comp,i_par] = par_name
                    if isinstance(input_pars[par_name], list):
                        par_Pk = input_pars
                        self.par_min_cp[i_comp,i_par] = par_Pk[par_name][0]
                        self.par_max_cp[i_comp,i_par] = par_Pk[par_name][1]
                        self.par_tie_cp[i_comp,i_par] = par_Pk[par_name][2]
                    elif isinstance(input_pars[par_name], dict):
                        par_PK = input_pars
                        self.par_min_cp[i_comp,i_par] = par_PK[par_name]['min']
                        self.par_max_cp[i_comp,i_par] = par_PK[par_name]['max']
                        self.par_tie_cp[i_comp,i_par] = par_PK[par_name]['tie']
            self.par_index_cP.append(par_index_P)
                    
            self.info_c.append(config_C[self.comp_name_c[i_comp]]['info'])

            # group used model elements in an array
            for item in ['mod_used', 'line_used']:
                if item in [*self.info_c[i_comp]]: 
                    if isinstance(self.info_c[i_comp][item], str): self.info_c[i_comp][item] = [self.info_c[i_comp][item]]
                    self.info_c[i_comp][item] = np.array(self.info_c[i_comp][item])

            # rename sign for absorption/emission
            if 'sign' in [*self.info_c[i_comp]]:
                if casefold(self.info_c[i_comp]['sign']) in ['absorption', 'negative', '-']:
                    self.info_c[i_comp]['sign'] = 'absorption'
                if casefold(self.info_c[i_comp]['sign']) in ['emission', 'positive', '+']:
                    self.info_c[i_comp]['sign'] = 'emission'
            else:
                self.info_c[i_comp]['sign'] = 'emission' # default

            self.info_c[i_comp]['comp_name'] = self.comp_name_c[i_comp]
        # self.info_c = np.array(self.info_c)

    def flat_to_arr(self, vals_flat):
        return vals_flat.reshape(self.num_comps, int(len(vals_flat)/self.num_comps))

###################################################################################################
###################################################################################################

class PhotFrame(object):
    def __init__(self, 
                 name_b=None, flux_b=None, ferr_b=None, flux_unit='mJy', # on input data
                 trans_dir=None, trans_rsmp=10, # on transmission curves
                 wave_w=None, wave_unit='angstrom', wave_num=None): # on corresonding SED range
        # add file_bac, file_iron later
        
        self.name_b = copy(name_b)
        self.flux_b = copy(flux_b)
        self.ferr_b = copy(ferr_b) 
        self.flux_unit = flux_unit

        self.trans_dir = copy(trans_dir)
        self.trans_rsmp = trans_rsmp

        self.wave_w = copy(wave_w)
        self.wave_unit = wave_unit
        if (self.wave_w is not None) & (self.wave_unit in ['micron', 'um']): self.wave_w *= 1e4 # convert to AA
        self.wave_num = wave_num
                
        self.trans_dict, self.trans_bw, self.wave_w = self.read_transmission(name_b=self.name_b, 
                                                                             trans_dir=self.trans_dir, trans_rsmp=self.trans_rsmp,  
                                                                             wave_w=self.wave_w, wave_num=self.wave_num)
        self.rFnuFlam_w = self.rFnuFlam_func(self.wave_w)
        self.rFnuFlam_b = self.rFnuFlam_func(self.wave_w, self.trans_bw)
        self.wave_b = self.spec2phot(self.wave_w, self.wave_w, self.trans_bw)
        
        if self.flux_unit == 'mJy': 
            self.flux_b /= self.rFnuFlam_b # convert to erg/s/cm2/AA
            self.ferr_b /= self.rFnuFlam_b # convert to erg/s/cm2/AA
        
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
                wave_ini, trans_ini = filterdata[:,0], filterdata[:,1]
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
        # Flam (mean) = INT(Flam * Tlam, x=lam) / INT(Tlam, x=lam), where INT(Tlam, x=lam) = 1
        # Fnu (mean) = INT(Flam * Tlam, x=lam) / INT(Tlam * dnu/dlam, x=lam)
        
    def rFnuFlam_func(self, wave_w, trans_bw=None):
        unitflam = (1 * u.erg/u.s/u.cm**2/u.angstrom)
        rDnuDlam = const.c.to('angstrom Hz') / (wave_w * u.angstrom)**2
        if trans_bw is None:
            # return the ratio of spectrum between Fnu (mJy) and Flam (erg/s/cm2/AA); wave in AA
            return (unitflam / rDnuDlam).to('mJy').value
        else:
            # return the ratio of band flux between Fnu (mJy) and Flam (erg/s/cm2/AA); wave in AA
            unitfint = unitflam * (1 * u.angstrom)
            # here (1 * u.angstrom) = np.trapezoid(trans, x=wave * u.angstrom, axis=axis), since trans is normalized to int=1
            width_nu = np.trapezoid(trans_bw * rDnuDlam, x=wave_w * u.angstrom, axis=trans_bw.ndim-1)
            return (unitfint / width_nu).to('mJy').value
        
    def spec2phot(self, wave_w, spec_mw, trans_bw):
        # convert spectrum in flam (erg/s/cm2/A) to mean flam in band (erg/s/cm2/A)
        if (spec_mw.ndim == 1) & (trans_bw.ndim == 1):
            return np.trapezoid(trans_bw * spec_mw, x=wave_w, axis=0) # return flux, 1-model, 1-band
        if (spec_mw.ndim == 1) & (trans_bw.ndim == 2):
            return np.trapezoid(trans_bw * spec_mw[None,:], x=wave_w, axis=1) # return flux_b, 1-model, multi-band
        if (spec_mw.ndim == 2) & (trans_bw.ndim == 1):
            return np.trapezoid(trans_bw[None,:] * spec_mw, x=wave_w, axis=1) # return flux_m, multi-model, 1-band
        if (spec_mw.ndim == 2) & (trans_bw.ndim == 2):
            return np.trapezoid(trans_bw[None,:,:] * spec_mw[:,None,:], x=wave_w, axis=2) # return flux_mb
        # short for np.trapezoid(trans * spec, x=wave, axis=axis) / np.trapezoid(trans, x=wave, axis=axis), trans is normalized to int=1

