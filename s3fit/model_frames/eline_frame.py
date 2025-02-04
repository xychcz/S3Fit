# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Contact: xiaoyang.chen.cz@gmail.com

import numpy as np
from copy import deepcopy as copy
from scipy.interpolate import RegularGridInterpolator

from ..extinct_law import ExtLaw

class ELineFrame(object):
    def __init__(self, rest_wave_w=None, mask_valid_w=None, 
                 cframe=None, v0_redshift=0, spec_R_inst=1e8, use_pyneb=False, verbose=True):

        self.rest_wave_w = rest_wave_w
        self.mask_valid_w = mask_valid_w
        self.cframe = cframe
        self.v0_redshift = v0_redshift
        self.spec_R_inst = spec_R_inst
        self.use_pyneb = use_pyneb
        self.verbose = verbose

        if self.use_pyneb: 
            self.initialize_linelist_pyneb()
        else:
            self.initialize_linelist_manually()

    def initialize_linelist_manually(self):
        # _n denote line; _l already used for loop id
        # [v3] update with https://physics.nist.gov/PhysRefData/ASD/lines_form.html (H ref: T8637)
        # ratio of doublets calculated with pyneb under ne=100, Te=1e4
        self.linerest_n, self.lineratio_n, self.linename_n = [],[],[]
        self.linerest_n.append(6564.632); self.lineratio_n.append(-1)     ; self.linename_n.append('Ha')
        self.linerest_n.append(4862.691); self.lineratio_n.append(0.349)  ; self.linename_n.append('Hb')
        self.linerest_n.append(4341.691); self.lineratio_n.append(0.164)  ; self.linename_n.append('Hg')
        self.linerest_n.append(4102.899); self.lineratio_n.append(0.0904) ; self.linename_n.append('Hd')
        self.linerest_n.append(3971.202); self.lineratio_n.append(0.0555) ; self.linename_n.append('H7')
        self.linerest_n.append(3890.158); self.lineratio_n.append(0.0367) ; self.linename_n.append('H8')
        self.linerest_n.append(6732.674); self.lineratio_n.append(-1)     ; self.linename_n.append('[S II]:6733')
        self.linerest_n.append(6718.295); self.lineratio_n.append(1.340)  ; self.linename_n.append('[S II]:6718')
        self.linerest_n.append(6585.270); self.lineratio_n.append(-1)     ; self.linename_n.append('[N II]:6585')
        self.linerest_n.append(6549.860); self.lineratio_n.append(0.340)  ; self.linename_n.append('[N II]:6550')
        self.linerest_n.append(6365.536); self.lineratio_n.append(0.319)  ; self.linename_n.append('[O I]:6366')
        self.linerest_n.append(6302.046); self.lineratio_n.append(-1)     ; self.linename_n.append('[O I]:6302')
        self.linerest_n.append(5201.705); self.lineratio_n.append(-1)     ; self.linename_n.append('[N I]:5202')
        self.linerest_n.append(5199.349); self.lineratio_n.append(0.783)  ; self.linename_n.append('[N I]:5199')
        self.linerest_n.append(5099.230); self.lineratio_n.append(-1)     ; self.linename_n.append('[Fe VI]:5099')
        self.linerest_n.append(5008.240); self.lineratio_n.append(-1)     ; self.linename_n.append('[O III]:5008')
        self.linerest_n.append(4960.295); self.lineratio_n.append(0.335)  ; self.linename_n.append('[O III]:4960')
        self.linerest_n.append(3968.590); self.lineratio_n.append(0.301)  ; self.linename_n.append('[Ne III]:3969')
        self.linerest_n.append(3869.860); self.lineratio_n.append(-1)     ; self.linename_n.append('[Ne III]:3870')
        self.linerest_n.append(3729.875); self.lineratio_n.append(-1)     ; self.linename_n.append('[O II]:3730')
        self.linerest_n.append(3727.092); self.lineratio_n.append(0.741)  ; self.linename_n.append('[O II]:3727')
        self.linerest_n.append(3426.864); self.lineratio_n.append(-1)     ; self.linename_n.append('[Ne V]:3427')
        self.linerest_n.append(3346.783); self.lineratio_n.append(0.366)  ; self.linename_n.append('[Ne V]:3347')
        self.linerest_n.append(5877.249); self.lineratio_n.append(-1)     ; self.linename_n.append('He I:5877')
        self.update_linelist()
        
    def initialize_linelist_pyneb(self):
        full_linelist = ['Ha', 'Hb', 'Hg', 'Hd', 'H7', 'H8', '[S II]:6733', '[S II]:6718',
               '[N II]:6585', '[N II]:6550', '[O I]:6366', '[O I]:6302',
               '[N I]:5202', '[N I]:5199', '[Fe VI]:5099', '[O III]:5008',
               '[O III]:4960', '[Ne III]:3969', '[Ne III]:3870', '[O II]:3730',
               '[O II]:3727', '[Ne V]:3427', '[Ne V]:3347']
        self.linerest_n, self.lineratio_n, self.linename_n = [],[],[]
        for linename in full_linelist:
            linename, linerest = self.search_pyneb(linename)
            self.linename_n.append(linename)
            self.linerest_n.append(linerest)
            self.lineratio_n.append(-1.0)  
            
        # not provided: NV_1239, O I:1302, Si IV:1394, Al III:1855, MgII_2796, HeI 5877.249, 
        self.linerest_n.append(5877.249); self.lineratio_n.append(-1)     ; self.linename_n.append('He I:5877')       
        self.update_linelist()
        
    def add_line(self, linenames=None, linerests=None, lineratios=None, force=False, use_pyneb=False):
        if not isinstance(linenames, list): linenames = [linenames]        
        for (i_line, linename) in enumerate(linenames):
            if use_pyneb:
                linename, linerest = self.search_pyneb(linename)
                lineratio = -1.0
            else:
                linerest = linerests[i_line]
                lineratio = lineratios[i_line] if lineratios is not None else -1
            if np.isin(linename, self.linename_n): raise ValueError((f"{linename} is already in the line list: {self.linename_n}."))
            i_close = np.absolute(self.linerest_n - linerest).argmin()
            if (np.abs(self.linerest_n[i_close] - linerest) > 1) | force: 
                self.linename_n  = np.hstack((self.linename_n, [linename]))
                self.linerest_n  = np.hstack((self.linerest_n, [linerest]))
                self.lineratio_n = np.hstack((self.lineratio_n, [lineratio]))
                print(f"{linename, linerest} with linkratio={lineratio} is added into the line list.")
            else:
                print(f"There is a line {self.linename_n[i_close], self.linerest_n[i_close]} close to the input one {linename, linerest}", 
                      ", set force=True to add this line.")
        self.update_linelist()

    def delete_line(self, linenames=None):
        if not isinstance(linenames, list): linenames = [linenames]
        mask_remain_n = np.ones_like(self.linename_n, dtype='bool')
        for linename in linenames:
            mask_remain_n &= self.linename_n != linename
        self.linename_n  = self.linename_n[mask_remain_n]
        self.linerest_n  = self.linerest_n[mask_remain_n]
        self.lineratio_n = self.lineratio_n[mask_remain_n]
        self.update_linelist()

    def update_linelist(self):
        index_line = np.argsort(self.linerest_n)
        self.linerest_n = np.array(self.linerest_n)[index_line]
        self.lineratio_n = np.array(self.lineratio_n)[index_line]
        self.linename_n = np.array(self.linename_n)[index_line]  
        self.initialize_linelink()
            
    def search_pyneb(self, name, ret_atomdata=False):
        import pyneb
        # due to the current coverage of elements and atoms of pyneb, 
        # currently only use pyneb to identify Hydrogen recombination lines
        # add other recombination lines manually
        # also add collisionally excited lines manually if not provided by pyneb        
        # convert roman numbers
        def roman_to_int(roman_num):
            roman_dict = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000,'IV':4,'IX':9,'XL':40,'XC':90,'CD':400,'CM':900}
            i = 0; int_num = 0
            while i < len(roman_num):
                if i+1<len(roman_num) and roman_num[i:i+2] in roman_dict:
                    int_num+=roman_dict[roman_num[i:i+2]]; i+=2
                else:
                    int_num+=roman_dict[roman_num[i]]; i+=1
            return int_num

        HI_lv_low_dict = {'Lya':1,'Lyb':1,'Lyg':1,'Lyd':1,'Ha':2,'Hb':2,'Hg':2,'Hd':2,
                          'Paa':3,'Pab':3,'Pag':3,'Pad':3,'Bra':4,'Brb':4,'Brg':4,'Brd':4}
        num_HI_highorder = 30
        for i in range(num_HI_highorder): HI_lv_low_dict['Ly'+str(6+i)] = 1
        for i in range(num_HI_highorder): HI_lv_low_dict['H'+str(7+i)] = 2
        for i in range(num_HI_highorder): HI_lv_low_dict['Pa'+str(8+i)] = 3
        for i in range(num_HI_highorder): HI_lv_low_dict['Br'+str(9+i)] = 4
        if name in HI_lv_low_dict:
            element = 'H'; notation = 1; line_id = name
            atomdata = pyneb.RecAtom(element, notation) # , extrapolate=False
            lv_low = HI_lv_low_dict[name]
            if np.isin(name[-1], ['a','b','g','d']): 
                lv_up = lv_low + {'a':1,'b':2,'g':3,'d':4}[name[-1]] 
            else:
                lv_up = int(name[1:]) if name[1].isdigit() else int(name[2:])
            wave_vac = 1/(atomdata._Energy[lv_up-1] - atomdata._Energy[lv_low-1])
        else:
            ion, wave = name.split(':')
            if ion[0] == '[': 
                element, notation = ion[1:-1].split(' ')
            else:
                element, notation = ion.split(' ')    
            notation = roman_to_int(notation)
            if np.isin(element+str(notation), ['H1', 'He2']):
                atomdata = pyneb.RecAtom(element, notation) # , extrapolate=False
            else:
                if ~np.isin(element+str(notation), pyneb.atomicData.getAllAtoms(coll=True, rec=False)):
                    raise ValueError((f"{name} not provided in pyneb, please add it into ELineFrame.initialize_linelist() manually."))
                atomdata = pyneb.Atom(element, notation) # , noExtrapol=True
            wave = float(wave[:-2])*1e4 if wave[-2:] == 'um' else float(wave)
            lv_up, lv_low = atomdata.getTransition(wave=wave)
            wave_vac = 1/(atomdata._Energy[lv_up-1] - atomdata._Energy[lv_low-1])
            line_id = f'{ion}:{round(wave_vac)}'
        # line_id = f'{element}{notation}_{int(wave_vac)}'
        # line_id = f'{element}{notation}u{lv_up}l{lv_low}'
        if ret_atomdata: 
            return line_id, wave_vac, atomdata
        else:
            return line_id, wave_vac
                
    def tie_pair(self, tied_names, ref_name=None, ratio=None, use_pyneb=None):
        if use_pyneb is None: use_pyneb = self.use_pyneb
        if not np.isin(ref_name, self.linename_n):
            raise ValueError((f"The reference line '{ref_name}' is not included in the line list."))
        if not isinstance(tied_names, list): tied_names = [tied_names]
        for tied_name in tied_names:
            if not np.isin(tied_name, self.linename_n):
                raise ValueError((f"The tied line '{tied_name}' is not included in the line list."))
            i_tied = np.where(self.linename_n == tied_name)[0][0]
            self.linelink_n[i_tied] = ref_name
            if use_pyneb:
                tied_wave, tied_atomdata = self.search_pyneb(tied_name, ret_atomdata=True)[1:]
                ref_wave,  ref_atomdata  = self.search_pyneb( ref_name, ret_atomdata=True)[1:]
                logdens = np.linspace(0, 5, 11)
                logtems = np.linspace(np.log10(5e2), np.log10(3e4), 11)
                ratio_dt = np.zeros((len(logdens), len(logtems)))
                for i_d in range(len(logdens)):
                    for i_t in range(len(logtems)):
                        tied_emissivity = tied_atomdata.getEmissivity(den=10.0**logdens[i_d], tem=10.0**logtems[i_t], wave=int(tied_wave))
                        ref_emissivity  =  ref_atomdata.getEmissivity(den=10.0**logdens[i_d], tem=10.0**logtems[i_t], wave=int(ref_wave))
                        ratio_dt[i_d,i_t] = tied_emissivity / ref_emissivity
                func_ratio_dt = RegularGridInterpolator((logdens, logtems), ratio_dt, method='linear', bounds_error=False)
                self.linelink_dict[tied_name] = {'ref_name': ref_name, 'func_ratio_dt': func_ratio_dt}
            else:                
                if tied_name == '[S II]:6718':
                    # https://ui.adsabs.harvard.edu/abs/2014A&A...561A..10P 
                    pok14_Rs = np.linspace(1.41, 0.45, 30)
                    pok14_logdens = 0.0543*np.tan(-3.0553*pok14_Rs+2.8506)+6.98-10.6905*pok14_Rs+9.9186*pok14_Rs**2-3.5442*pok14_Rs**3
                    def func_ratio_dt(pars): return [np.interp(pars[0], pok14_logdens, pok14_Rs)]
                else:
                    if ratio is None:
                        tmp = self.lineratio_n[i_tied] * 1.0 # to avoid overwrite in following updating
                    else:
                        if isinstance(ratio, list): raise ValueError((f"Please use a single input ratio."))
                        tmp = ratio * 1.0 # force to input value
                    def func_ratio_dt(pars, ret=tmp): return [ret]
                self.linelink_dict[tied_name] = {'ref_name': ref_name, 'func_ratio_dt': func_ratio_dt}
                
    def release_pair(self, tied_names):
        if not isinstance(tied_names, list): tied_names = [tied_names]
        for tied_name in tied_names:
            if not np.isin(tied_name, self.linename_n):
                raise ValueError((f"The tied line '{tied_name}' is not included in the line list."))
            i_tied = np.where(self.linename_n == tied_name)[0][0]
            self.linelink_n[i_tied] = 'free'
            self.linelink_dict.pop(tied_name)
                    
    def initialize_linelink(self):
        self.linelink_n = np.zeros_like(self.linename_n); self.linelink_n[:] = 'free'
        self.linelink_dict = {}
        # tie line pairs
        self.tie_pair(['Hb','Hg','Hd','H7','H8'], 'Ha')   
        self.tie_pair('[S II]:6718', '[S II]:6733')
        self.tie_pair('[N II]:6550', '[N II]:6585')
        self.tie_pair('[O I]:6366', '[O I]:6302')
        self.tie_pair('[N I]:5199', '[N I]:5202')
        self.tie_pair('[O III]:4960', '[O III]:5008')
        self.tie_pair('[Ne III]:3969', '[Ne III]:3870')
        self.tie_pair('[O II]:3727', '[O II]:3730')
        self.tie_pair('[Ne V]:3347', '[Ne V]:3427')
        self.update_freemask()
        self.update_lineratio()
        
    def update_lineratio(self, AV=0, logden=2, logtem=4):
        for tied_name in self.linelink_dict:
            ref_name = self.linelink_dict[tied_name]['ref_name']
            func_ratio_dt = self.linelink_dict[tied_name]['func_ratio_dt']
            i_tied = np.where(self.linename_n == tied_name)
            self.lineratio_n[i_tied] = func_ratio_dt(np.array([logden, logtem]))[0]
            if (ref_name == 'Ha') | (ref_name == 'H I:6565'):
                tmp = ExtLaw(self.linerest_n[i_tied]) - ExtLaw(np.array([6564.63215]))
                self.lineratio_n[i_tied] *= 10.0**(-0.4 * AV * tmp)

    def update_freemask(self):
        self.num_comps = len(self.cframe.info_c)
        self.num_lines = len(self.linerest_n)
        self.mask_valid_cn = np.zeros((self.num_comps, self.num_lines), dtype='bool')
        self.mask_free_cn  = np.zeros((self.num_comps, self.num_lines), dtype='bool')
        for i_comp in range(self.num_comps):
            mask_valid_n  = self.linerest_n > (self.rest_wave_w.min()-50)
            mask_valid_n &= self.linerest_n < (self.rest_wave_w.max()+50)
            if self.mask_valid_w is not None:
                for i_line in range(self.num_lines):
                    voff_w = (self.rest_wave_w/self.linerest_n[i_line] -1) * 299792.458
                    mask_line_w  = voff_w > self.cframe.min_cp[i_comp,0]
                    mask_line_w &= voff_w < self.cframe.max_cp[i_comp,0]
                    if (mask_line_w & self.mask_valid_w).sum() / mask_line_w.sum() < 0.1: mask_valid_n[i_line] = False
            mask_free_n = mask_valid_n & ~np.isin(self.linename_n, [*self.linelink_dict])
            if self.cframe.info_c[i_comp]['line_used'][0] != 'all': 
                mask_select_n = np.isin(self.linename_n, self.cframe.info_c[i_comp]['line_used'])
                mask_valid_n &= mask_select_n
                mask_free_n  &= mask_select_n
            self.mask_valid_cn[i_comp, mask_valid_n] = True
            self.mask_free_cn[i_comp,  mask_free_n ] = True
        self.num_coeffs = self.mask_free_cn.sum()
        
        # set component name and enable mask for each free line; _f for free or coeffs
        self.component_f = [] # np.zeros((self.num_coeffs), dtype='<U16')
        for i_comp in range(self.num_comps):
            for i_line in range(self.num_lines):
                if self.mask_free_cn[i_comp, i_line]:
                    self.component_f.append(self.cframe.info_c[i_comp]['comp_name'])
        self.component_f = np.array(self.component_f)
        
        if self.verbose:
            for i_comp in range(self.num_comps):
                print(f"Emission line comp{i_comp}:'{self.cframe.info_c[i_comp]['comp_name']}', ", 
                      f"{self.mask_valid_cn[i_comp].sum()} lines in total, ", 
                      f"{self.mask_free_cn[i_comp].sum()} free lines: {self.linename_n[self.mask_free_cn[i_comp]]}")

    def single_gaussian(self, obs_wave_w, lamb_c_rest, voff, fwhm, flux, v0_redshift=0, spec_R_inst=1e8):
        if fwhm <= 0: raise ValueError((f"Non-positive eline fwhm: {fwhm}"))
        if flux < 0: raise ValueError((f"Negative eline flux: {flux}"))
        lamb_c_obs = lamb_c_rest * (1 + v0_redshift)
        mu =    (1 + voff/299792.458) * lamb_c_obs
        sigma_line = fwhm/299792.458  * lamb_c_obs / np.sqrt(np.log(256))
        sigma_inst = 1/spec_R_inst * lamb_c_obs / np.sqrt(np.log(256))
        sigma = np.sqrt(sigma_line**2 + sigma_inst**2)
        model = np.exp(-0.5 * ((obs_wave_w-mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi)) 
        dw = (obs_wave_w[1:]-obs_wave_w[:-1]).min()
        if (model * dw).sum() < 0.10: flux = 0 # disable not well covered emission line
        return model * flux

    def models_single_comp(self, obs_wave_w, pars, list_valid, list_free):
        # update lineratio_n
        voff, fwhm, AV, logden, logtem = pars
        self.update_lineratio(AV, logden, logtem)
        
        models_scomp = []
        for i_free in list_free:
            model_sline = self.single_gaussian(obs_wave_w, self.linerest_n[i_free], voff, fwhm, 
                                               1, self.v0_redshift, self.spec_R_inst) # flux=1
            list_linked = np.where(self.linelink_n == self.linename_n[i_free])[0]
            list_linked = list_linked[np.isin(list_linked, list_valid)]
            for i_linked in list_linked:
                model_sline += self.single_gaussian(obs_wave_w, self.linerest_n[i_linked], voff, fwhm, 
                                                    self.lineratio_n[i_linked], self.v0_redshift, self.spec_R_inst)
            models_scomp.append(model_sline)
        return models_scomp
    
    def models_unitnorm_obsframe(self, obs_wave_w, input_pars, if_pars_flat=True):
        if if_pars_flat: 
            pars = self.cframe.flat_to_arr(input_pars)
        else:
            pars = copy(input_pars)

        for i_comp in range(self.num_comps):
            list_valid = np.arange(len(self.linerest_n))[self.mask_valid_cn[i_comp,:]]
            list_free  = np.arange(len(self.linerest_n))[self.mask_free_cn[i_comp,:]]
            models_scomp = self.models_single_comp(obs_wave_w, pars[i_comp], list_valid, list_free)
            if i_comp == 0: 
                models_mcomp = models_scomp
            else:
                models_mcomp += models_scomp
        return np.array(models_mcomp)

    def mask_el_lite(self, enabled_comps='all'):
        self.enabled_f = np.zeros((self.num_coeffs), dtype='bool')
        if enabled_comps == 'all':
            self.enabled_f[:] = True
        else:
            for comp in enabled_comps:
                self.enabled_f[self.component_f == comp] = True
        return self.enabled_f

    ##########################################################################
    ########################## Output functions ##############################

    def output_results(self, ff=None):
        num_mock_loops = ff.num_mock_loops
        best_chi_sq_l = ff.best_chi_sq
        best_x_lp = ff.best_fits_x
        best_coeff_lm = ff.best_coeffs
        fx0, fx1, fc0, fc1 = ff.model_index('el', ff.full_model_type)

        num_el_comps = self.cframe.num_comps
        num_el_pars = self.cframe.num_pars
        
        output_el_lcp = np.zeros((num_mock_loops, num_el_comps, 1 + num_el_pars + self.num_lines))
        # use self.num_lines instead of self.num_coeffs to cover non-free lines
        output_el_lcp[:, :, 0] = best_chi_sq_l[:, None]

        for i_loop in range(num_mock_loops):
            output_el_lcp[i_loop, :, 1:(1+num_el_pars)] = self.cframe.flat_to_arr( best_x_lp[i_loop, fx0:fx1] )
            output_el_lcp[i_loop, :, -self.num_lines:][self.mask_free_cn] = best_coeff_lm[i_loop, fc0:fc1]
        
        # update lineratio_n to calculate tied lines        
        list_linked = np.where(np.isin(self.linename_n, [*self.linelink_dict]))[0]
        for i_line in list_linked:
            i_main = 1 + num_el_pars + np.where(self.linename_n == self.linelink_n[i_line])[0]
            for i_loop in range(num_mock_loops):
                for i_comp in range(num_el_comps):
                    self.update_lineratio(output_el_lcp[i_loop, i_comp, 1+2], 
                                          output_el_lcp[i_loop, i_comp, 1+3], 
                                          output_el_lcp[i_loop, i_comp, 1+4])
                    output_el_lcp[i_loop, i_comp, 1 + num_el_pars + i_line] = output_el_lcp[i_loop, i_comp, i_main] * self.lineratio_n[i_line] \
                                                                            * self.mask_valid_cn[i_comp, i_line]
        self.output_el_lcp = output_el_lcp # save to model frame
            
        # output to screen
        output_el_vals = {
            'mean': np.average(output_el_lcp, weights=1/best_chi_sq_l, axis=0), 
            'rms' : np.std(output_el_lcp, axis=0, ddof=1) }

        self.output_el_vals = output_el_vals # save to model frame
        self.print_results()

    def print_results(self):
        print('')
        print('Best-fit emission line components')

        cols = 'Par/Line Name'
        fmt_cols = '| {:^20} |'
        fmt_numbers = '| {:^20} |' #fmt_numbers = '| {:=13.4f} |'
        for i_comp in range(self.num_comps): 
            cols += ','+self.cframe.comp_c[i_comp]
            fmt_cols += ' {:^18} |'
            fmt_numbers += ' {:=8.2f} +- {:=6.2f} |'
        cols_split = cols.split(',')
        tbl_title = fmt_cols.format(*cols_split)
        tbl_border = len(tbl_title)*'='
        print(tbl_border)
        print(tbl_title)
        print(tbl_border)

        names = ['Chi^2', 'Voff (km/s)', 'FWHM (km/s)', 'AV (Balmer decre.)', 'log e-density (cm-3)', 'log e-temperature(K)']
        for i_line in range(self.num_lines): 
            names.append('{}'.format(self.linename_n[i_line]))
        for i_par in range(len(names)): 
            tbl_row = []
            tbl_row.append(names[i_par])
            for i_comp in range(self.num_comps):
                tbl_row.append(self.output_el_vals['mean'][i_comp, i_par])
                tbl_row.append(self.output_el_vals['rms'][i_comp, i_par])
            print(fmt_numbers.format(*tbl_row))
        print(tbl_border)  
    