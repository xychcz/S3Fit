# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Contact: xiaoyang.chen.cz@gmail.com

import os
import numpy as np
from copy import deepcopy as copy
import astropy.units as u
import astropy.constants as const

class PhotFrame(object):
    def __init__(self, 
                 name_b=None, flux_b=None, ferr_b=None, fluxunit='mJy', # on input data
                 trans_dir=None, wave_w=None, waveunit='angstrom'): # on transmission curves
        # add file_bac, file_iron later
        
        self.name_b = copy(name_b)
        self.flux_b = copy(flux_b)
        self.ferr_b = copy(ferr_b) 
        self.fluxunit = fluxunit
        
        self.trans_dir = trans_dir
        self.wave_w = copy(wave_w)
        self.waveunit = waveunit
        if (self.wave_w is not None) & (self.waveunit == 'micron'): self.wave_w *= 1e4 # convert to AA
                
        self.trans_dict, self.trans_bw, self.wave_w = self.read_transmission(self.trans_dir, name_b=self.name_b, wave_w=self.wave_w)
        self.rFnuFlam_w = self.rFnuFlam_func(self.wave_w)
        self.rFnuFlam_b = self.rFnuFlam_func(self.wave_w, self.trans_bw)
        self.wave_b = self.spec2phot(self.wave_w, self.wave_w, self.trans_bw)
        
        if self.fluxunit == 'mJy': 
            self.flux_b /= self.rFnuFlam_b # convert to erg/s/cm2/AA
            self.ferr_b /= self.rFnuFlam_b # convert to erg/s/cm2/AA
        
    def read_transmission(self, trans_dir, name_b=None, wave_w=None):        
        if name_b is None:
            file_list = np.array(os.listdir(trans_dir))
            file_list = file_list[np.array([f[-4:] for f in file_list]) == '.dat']
            name_b = np.array([f.split('.dat')[0] for f in file_list])
            trans_bw = None
        else:
            trans_bw = 1

        if wave_w is None:
            w_min, w_max = 1e16, 0
            for name in name_b: 
                filterdata = np.loadtxt(trans_dir+name+'.dat')
                wave_ini, trans_ini = filterdata[:,0], filterdata[:,1]
                w_min = np.minimum(w_min, wave_ini.min())
                w_max = np.maximum(w_max, wave_ini.max())
            wave_w = np.logspace(np.log10(w_min)-0.1, np.log10(w_max)+0.1, num=5000)

        trans_dict = {}
        for name in name_b:
            filterdata = np.loadtxt(trans_dir+name+'.dat')
            wave_ini, trans_ini = filterdata[:,0], filterdata[:,1]
            trans_ini /= np.trapz(trans_ini, x=wave_ini)
            # trans_ini /= np.trapz(trans_ini * const.c.to('micron Hz').value / wave_ini**2, x=wave_ini) # in Hz-1
            tmp_center = np.trapz(trans_ini*wave_ini, x=wave_ini)
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
            # here (1 * u.angstrom) = np.trapz(trans, x=wave * u.angstrom, axis=axis), since trans is normalized to int=1
            width_nu = np.trapz(trans_bw * rDnuDlam, x=wave_w * u.angstrom, axis=trans_bw.ndim-1)
            return (unitfint / width_nu).to('mJy').value
        
    def spec2phot(self, wave_w, spec_mw, trans_bw):
        # convert spectrum in flam (erg/s/cm2/A) to mean flam in band (erg/s/cm2/A)
        if (spec_mw.ndim == 1) & (trans_bw.ndim == 1):
            return np.trapz(trans_bw * spec_mw, x=wave_w, axis=0) # return flux, 1-model, 1-band
        if (spec_mw.ndim == 1) & (trans_bw.ndim == 2):
            return np.trapz(trans_bw * spec_mw[None,:], x=wave_w, axis=1) # return flux_b, 1-model, multi-band
        if (spec_mw.ndim == 2) & (trans_bw.ndim == 1):
            return np.trapz(trans_bw[None,:] * spec_mw, x=wave_w, axis=1) # return flux_m, multi-model, 1-band
        if (spec_mw.ndim == 2) & (trans_bw.ndim == 2):
            return np.trapz(trans_bw[None,:,:] * spec_mw[:,None,:], x=wave_w, axis=2) # return flux_mb
        # short for np.trapz(trans * spec, x=wave, axis=axis) / np.trapz(trans, x=wave, axis=axis), trans is normalized to int=1
