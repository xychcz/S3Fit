# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: xiaoyang.chen.cz@gmail.com

import numpy as np
from copy import deepcopy as copy
from scipy.interpolate import RegularGridInterpolator

from ..auxiliary_func import print_log
from ..extinct_law import ExtLaw

class LineFrame(object):
    def __init__(self, rest_wave_w=None, mask_valid_w=None, 
                 cframe=None, v0_redshift=0, R_inst_rw=None, use_pyneb=False, verbose=True, log_message=[]):

        self.rest_wave_w = rest_wave_w
        self.mask_valid_w = mask_valid_w
        self.cframe = cframe
        self.v0_redshift = v0_redshift
        self.R_inst_rw = R_inst_rw
        self.use_pyneb = use_pyneb
        self.verbose = verbose
        self.log_message = log_message

        self.num_comps = len(self.cframe.info_c)
        # set default profile as gaussian
        for i_comp in range(self.num_comps):
            if ~np.isin('profile', [*self.cframe.info_c[i_comp]]):
                self.cframe.info_c[i_comp]['profile'] = 'Gaussian'

        if self.use_pyneb: 
            self.initialize_linelist_pyneb()
        else:
            self.initialize_linelist_manually()

    def initialize_linelist_manually(self):
        # _n denote line; _l already used for loop id
        # https://physics.nist.gov/PhysRefData/ASD/lines_form.html (H ref: T8637)
        # https://linelist.pa.uky.edu/atomic/
        # https://astronomy.nmsu.edu/drewski/tableofemissionlines.html
        # ratio of doublets calculated with pyneb under ne=100, Te=1e4
        self.linerest_n, self.lineratio_n, self.linename_n = [],[],[]
        # UV
        self.linerest_n.append(1215.670); self.lineratio_n.append(1.0)     ; self.linename_n.append('Lya')
        self.linerest_n.append(1025.722); self.lineratio_n.append(1.0)     ; self.linename_n.append('Lyb')
        self.linerest_n.append( 972.537); self.lineratio_n.append(1.0)     ; self.linename_n.append('Lyg')
        self.linerest_n.append( 949.743); self.lineratio_n.append(1.0)     ; self.linename_n.append('Lyd')
        self.linerest_n.append( 937.803); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly6')
        # self.linerest_n.append( 930.748); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly7')
        # self.linerest_n.append( 926.226); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly8')
        # self.linerest_n.append( 923.150); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly9')
        # self.linerest_n.append( 920.963); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly10')
        self.linerest_n.append( 770.409); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ne VIII:770')
        self.linerest_n.append( 780.324); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ne VIII:780')
        self.linerest_n.append( 977.030); self.lineratio_n.append(1.0)     ; self.linename_n.append('C III:977')
        self.linerest_n.append( 989.790); self.lineratio_n.append(1.0)     ; self.linename_n.append('N III:990')
        self.linerest_n.append( 991.514); self.lineratio_n.append(1.0)     ; self.linename_n.append('N III:991.5')
        self.linerest_n.append( 991.579); self.lineratio_n.append(1.0)     ; self.linename_n.append('N III:991.6')
        self.linerest_n.append(1031.912); self.lineratio_n.append(1.0)     ; self.linename_n.append('O VI:1032')
        self.linerest_n.append(1037.613); self.lineratio_n.append(1.0)     ; self.linename_n.append('O VI:1038')
        self.linerest_n.append(1238.821); self.lineratio_n.append(1.0)     ; self.linename_n.append('N V:1239')
        self.linerest_n.append(1242.804); self.lineratio_n.append(1.0)     ; self.linename_n.append('N V:1243')
        self.linerest_n.append(1260.422); self.lineratio_n.append(1.0)     ; self.linename_n.append('Si II:1260')
        self.linerest_n.append(1264.730); self.lineratio_n.append(1.0)     ; self.linename_n.append('Si II:1265')
        self.linerest_n.append(1302.168); self.lineratio_n.append(1.0)     ; self.linename_n.append('O I:1302')
        self.linerest_n.append(1334.532); self.lineratio_n.append(1.0)     ; self.linename_n.append('C II:1335')
        self.linerest_n.append(1335.708); self.lineratio_n.append(1.0)     ; self.linename_n.append('C II:1336')
        self.linerest_n.append(1393.755); self.lineratio_n.append(1.0)     ; self.linename_n.append('Si IV:1394')
        self.linerest_n.append(1397.232); self.lineratio_n.append(1.0)     ; self.linename_n.append('O IV]:1397')
        self.linerest_n.append(1399.780); self.lineratio_n.append(1.0)     ; self.linename_n.append('O IV]:1400')
        self.linerest_n.append(1402.770); self.lineratio_n.append(1.0)     ; self.linename_n.append('Si IV:1403')
        self.linerest_n.append(1486.496); self.lineratio_n.append(1.0)     ; self.linename_n.append('N IV]:1486')
        self.linerest_n.append(1548.187); self.lineratio_n.append(1.0)     ; self.linename_n.append('C IV:1548')
        self.linerest_n.append(1550.772); self.lineratio_n.append(1.0)     ; self.linename_n.append('C IV:1551')
        self.linerest_n.append(1640.420); self.lineratio_n.append(1.0)     ; self.linename_n.append('He II:1640')
        self.linerest_n.append(1660.809); self.lineratio_n.append(1.0)     ; self.linename_n.append('O III]:1661')
        self.linerest_n.append(1666.150); self.lineratio_n.append(1.0)     ; self.linename_n.append('O III]:1666')
        self.linerest_n.append(1746.823); self.lineratio_n.append(1.0)     ; self.linename_n.append('N III]:1747')
        self.linerest_n.append(1748.656); self.lineratio_n.append(1.0)     ; self.linename_n.append('N III]:1749')
        self.linerest_n.append(1854.716); self.lineratio_n.append(1.0)     ; self.linename_n.append('Al III:1855')
        self.linerest_n.append(1862.790); self.lineratio_n.append(1.0)     ; self.linename_n.append('Al III:1863')
        self.linerest_n.append(1892.030); self.lineratio_n.append(1.0)     ; self.linename_n.append('Si III]:1892')
        self.linerest_n.append(1908.734); self.lineratio_n.append(1.0)     ; self.linename_n.append('C III]:1909')
        self.linerest_n.append(2143.450); self.lineratio_n.append(1.0)     ; self.linename_n.append('N II]:2143')
        self.linerest_n.append(2324.210); self.lineratio_n.append(1.0)     ; self.linename_n.append('C II]:2324')
        self.linerest_n.append(2325.400); self.lineratio_n.append(1.0)     ; self.linename_n.append('C II]:2325')
        self.linerest_n.append(2796.352); self.lineratio_n.append(1.0)     ; self.linename_n.append('Mg II]:2796')
        self.linerest_n.append(2803.531); self.lineratio_n.append(1.0)     ; self.linename_n.append('Mg II]:2804')
        # optical
        self.linerest_n.append(6564.632); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ha')
        self.linerest_n.append(4862.691); self.lineratio_n.append(0.349)   ; self.linename_n.append('Hb')
        self.linerest_n.append(4341.691); self.lineratio_n.append(0.164)   ; self.linename_n.append('Hg')
        self.linerest_n.append(4102.899); self.lineratio_n.append(0.0904)  ; self.linename_n.append('Hd')
        self.linerest_n.append(3971.202); self.lineratio_n.append(0.0555)  ; self.linename_n.append('H7')
        self.linerest_n.append(3890.158); self.lineratio_n.append(0.0367)  ; self.linename_n.append('H8')
        # self.linerest_n.append(3836.479); self.lineratio_n.append(0.0367) ; self.linename_n.append('H9')
        # self.linerest_n.append(3798.983); self.lineratio_n.append(0.0367) ; self.linename_n.append('H10')
        # self.linerest_n.append(3771.708); self.lineratio_n.append(0.0367) ; self.linename_n.append('H11')
        # self.linerest_n.append(3751.224); self.lineratio_n.append(0.0367) ; self.linename_n.append('H12')
        # self.linerest_n.append(3735.437); self.lineratio_n.append(0.0367) ; self.linename_n.append('H13')
        # self.linerest_n.append(3723.004); self.lineratio_n.append(0.0367) ; self.linename_n.append('H14')
        # self.linerest_n.append(3713.033); self.lineratio_n.append(0.0367) ; self.linename_n.append('H15')
        self.linerest_n.append(5877.249); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:5877')
        self.linerest_n.append(3346.783); self.lineratio_n.append(0.366)   ; self.linename_n.append('[Ne V]:3347')
        self.linerest_n.append(3426.864); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ne V]:3427')
        self.linerest_n.append(3727.092); self.lineratio_n.append(0.741)   ; self.linename_n.append('[O II]:3727')
        self.linerest_n.append(3729.875); self.lineratio_n.append(1.0)     ; self.linename_n.append('[O II]:3730')
        self.linerest_n.append(3869.860); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ne III]:3870')
        self.linerest_n.append(3968.590); self.lineratio_n.append(0.301)   ; self.linename_n.append('[Ne III]:3969')
        self.linerest_n.append(4960.295); self.lineratio_n.append(0.335)   ; self.linename_n.append('[O III]:4960')
        self.linerest_n.append(5008.240); self.lineratio_n.append(1.0)     ; self.linename_n.append('[O III]:5008')
        self.linerest_n.append(5099.230); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe VI]:5099')
        self.linerest_n.append(5199.349); self.lineratio_n.append(0.783)   ; self.linename_n.append('[N I]:5199')
        self.linerest_n.append(5201.705); self.lineratio_n.append(1.0)     ; self.linename_n.append('[N I]:5202')
        self.linerest_n.append(6302.046); self.lineratio_n.append(1.0)     ; self.linename_n.append('[O I]:6302')
        self.linerest_n.append(6365.536); self.lineratio_n.append(0.319)   ; self.linename_n.append('[O I]:6366')
        self.linerest_n.append(6549.860); self.lineratio_n.append(0.340)   ; self.linename_n.append('[N II]:6550')
        self.linerest_n.append(6585.270); self.lineratio_n.append(1.0)     ; self.linename_n.append('[N II]:6585')
        self.linerest_n.append(6718.295); self.lineratio_n.append(1.340)   ; self.linename_n.append('[S II]:6718')
        self.linerest_n.append(6732.674); self.lineratio_n.append(1.0)     ; self.linename_n.append('[S II]:6733')

        self.update_linelist()
        
    def initialize_linelist_pyneb(self):
        if self.verbose: 
            print_log('[Note] PyNeb is used in the fitting to derive line emissivities and ratios of line doublets.', self.log_message)
        import pyneb
        self.pyneb = pyneb
        self.pyneblib = {'RecAtom':{'list': pyneb.atomicData.getAllAtoms(coll=False, rec=True)}, 
                         'Atom':{'list': pyneb.atomicData.getAllAtoms(coll=True, rec=False)}} 
        # save to avoid duplicate reading

        # add lines from pyneb
        full_linelist = ['Lya', 'Lyb', 'Lyg', 'Lyd', 'Ly6', 
                         'C III:977', 'N III:990', 'N III:991.5', 'N III:991.6', 'C II:1335', 'C II:1336', 'O IV]:1397', 'O IV]:1400', 'N IV]:1486', 'C IV:1548', 'C IV:1551', 
                         'He II:1640', 'O III]:1661', 'O III]:1666', 'N III]:1747', 'N III]:1749', 'Si III]:1892', 'C III]:1909', 'N II]:2143', 'C II]:2324', 'C II]:2325',
                         'Ha', 'Hb', 'Hg', 'Hd', 'H7', 'H8', 
                         '[Ne V]:3347', '[Ne V]:3427', '[O II]:3727', '[O II]:3730', '[Ne III]:3870', '[Ne III]:3969', '[O III]:4960', '[O III]:5008', 
                         '[Fe VI]:5099', '[N I]:5199', '[N I]:5202', '[O I]:6302','[O I]:6366', '[N II]:6550', '[N II]:6585', '[S II]:6718', '[S II]:6733']

        self.linerest_n, self.lineratio_n, self.linename_n = [],[],[]
        for linename in full_linelist:
            linename, linerest = self.search_pyneb(linename)
            self.linename_n.append(linename)
            self.linerest_n.append(linerest)
            self.lineratio_n.append(1.0)  

        # add lines not provided by pyneb
        self.linerest_n.append( 770.409); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ne VIII:770')
        self.linerest_n.append( 780.324); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ne VIII:780')
        self.linerest_n.append(1031.912); self.lineratio_n.append(1.0)     ; self.linename_n.append('O VI:1032')
        self.linerest_n.append(1037.613); self.lineratio_n.append(1.0)     ; self.linename_n.append('O VI:1038')
        self.linerest_n.append(1238.821); self.lineratio_n.append(1.0)     ; self.linename_n.append('N V:1239')
        self.linerest_n.append(1242.804); self.lineratio_n.append(1.0)     ; self.linename_n.append('N V:1243')
        self.linerest_n.append(1260.422); self.lineratio_n.append(1.0)     ; self.linename_n.append('Si II:1260')
        self.linerest_n.append(1264.730); self.lineratio_n.append(1.0)     ; self.linename_n.append('Si II:1265')
        self.linerest_n.append(1302.168); self.lineratio_n.append(1.0)     ; self.linename_n.append('O I:1302')
        self.linerest_n.append(1393.755); self.lineratio_n.append(1.0)     ; self.linename_n.append('Si IV:1394')
        self.linerest_n.append(1402.770); self.lineratio_n.append(1.0)     ; self.linename_n.append('Si IV:1403')
        self.linerest_n.append(1854.716); self.lineratio_n.append(1.0)     ; self.linename_n.append('Al III:1855')
        self.linerest_n.append(1862.790); self.lineratio_n.append(1.0)     ; self.linename_n.append('Al III:1863')
        self.linerest_n.append(2796.352); self.lineratio_n.append(1.0)     ; self.linename_n.append('Mg II]:2796')
        self.linerest_n.append(2803.531); self.lineratio_n.append(1.0)     ; self.linename_n.append('Mg II]:2804')
        self.linerest_n.append(5877.249); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:5877')

        self.update_linelist()
        
    def add_line(self, linenames=None, linerests=None, lineratios=None, force=False, use_pyneb=False):
        if not isinstance(linenames, list): linenames = [linenames]
        for (i_line, linename) in enumerate(linenames):
            if use_pyneb:
                linename, linerest = self.search_pyneb(linename)
                lineratio = -1.0
            else:
                linerest = linerests[i_line]
                lineratio = lineratios[i_line] if lineratios is not None else 1.0
            if np.isin(linename, self.linename_n): raise ValueError((f"{linename} is already in the line list: {self.linename_n}."))
            i_close = np.absolute(self.linerest_n - linerest).argmin()
            if (np.abs(self.linerest_n[i_close] - linerest) > 1) | force: 
                self.linename_n  = np.hstack((self.linename_n, [linename]))
                self.linerest_n  = np.hstack((self.linerest_n, [linerest]))
                self.lineratio_n = np.hstack((self.lineratio_n, [lineratio]))
                print_log(f"{linename, linerest} with linkratio={lineratio} is added into the line list.", self.log_message)
            else:
                print_log(f"There is a line {self.linename_n[i_close], self.linerest_n[i_close]} close to the input one {linename, linerest}"
                         +f", set force=True to add this line.", self.log_message)
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
        # sort lines with increasing wavelength
        index_line = np.argsort(self.linerest_n)
        self.linerest_n = np.array(self.linerest_n)[index_line]
        self.lineratio_n = np.array(self.lineratio_n)[index_line]
        self.linename_n = np.array(self.linename_n)[index_line]  
        # check lines with valid wavelength coverage
        self.initialize_mask_valid()
        # check the relations between lines
        self.initialize_linelink()

    def initialize_mask_valid(self):
        # only keep covered lines
        mask_valid_n  = self.linerest_n > (self.rest_wave_w.min()-50)
        mask_valid_n &= self.linerest_n < (self.rest_wave_w.max()+50)
        self.linerest_n = self.linerest_n[mask_valid_n]
        self.lineratio_n = self.lineratio_n[mask_valid_n]
        self.linename_n = self.linename_n[mask_valid_n]

        self.num_lines = len(self.linerest_n)
        self.mask_valid_cn = np.zeros((self.num_comps, self.num_lines), dtype='bool')

        # check minimum coverage        
        for i_comp in range(self.num_comps):
            for i_line in range(self.num_lines):
                voff_w = (self.rest_wave_w/self.linerest_n[i_line] -1) * 299792.458
                mask_line_w  = voff_w > self.cframe.min_cp[i_comp,0]
                mask_line_w &= voff_w < self.cframe.max_cp[i_comp,0]
                if mask_line_w.sum() > 0: 
                    self.mask_valid_cn[i_comp, i_line] = True
                    if self.mask_valid_w is not None:
                        if (mask_line_w & self.mask_valid_w).sum() / mask_line_w.sum() < 0.1: # minimum valid coverage fraction
                            self.mask_valid_cn[i_comp, i_line] = False

        # only keep lines if they are specified 
        for i_comp in range(self.num_comps):
            if ~np.isin(self.cframe.info_c[i_comp]['line_used'][0], ['all', 'default']): 
                self.mask_valid_cn[i_comp] &= np.isin(self.linename_n, self.cframe.info_c[i_comp]['line_used'])
            
    def search_pyneb(self, name, ret_atomdata=False):
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
            if ~np.isin(element+str(notation), [*self.pyneblib['RecAtom']]):
                self.pyneblib['RecAtom'][element+str(notation)] = self.pyneb.RecAtom(element, notation) # , extrapolate=False
            atomdata = self.pyneblib['RecAtom'][element+str(notation)]
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
                if ion[-1] == ']': 
                    element, notation = ion[:-1].split(' ')
                else:
                    element, notation = ion.split(' ')
            notation = roman_to_int(notation)
            if np.isin(element+str(notation), ['H1', 'He2']):
                if ~np.isin(element+str(notation), [*self.pyneblib['RecAtom']]):
                    self.pyneblib['RecAtom'][element+str(notation)] = self.pyneb.RecAtom(element, notation) # , extrapolate=False
                atomdata = self.pyneblib['RecAtom'][element+str(notation)]
            else:
                if ~np.isin(element+str(notation), self.pyneblib['Atom']['list']):
                    raise ValueError((f"{name} not provided in pyneb, please add it into LineFrame.initialize_linelist() manually."))
                if ~np.isin(element+str(notation), [*self.pyneblib['Atom']]):
                    self.pyneblib['Atom'][element+str(notation)] = self.pyneb.Atom(element, notation) # , noExtrapol=True
                atomdata = self.pyneblib['Atom'][element+str(notation)]
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
                
    def tie_pair(self, tied_names, ref_names=None, ratio=None, use_pyneb=None):
        # prebuild the tying ratio libraries between tied lines and the reference line (1st valid one if multi given)
        if use_pyneb is None: use_pyneb = self.use_pyneb

        if not isinstance(ref_names, list): ref_names = [ref_names]
        ref_valid = False
        for ref_name in ref_names:
            if not np.isin(ref_name, self.linename_n):
                # raise ValueError((f"The reference line '{ref_name}' is not included in the available line list."))
                continue
            i_ref = np.where(self.linename_n == ref_name)[0][0]
            ref_valid = self.mask_valid_cn[:, i_ref].sum() > 0 # if ref_name exists in any one comp
            if ref_valid: break # pick up the 1st valid ref_name

        if not isinstance(tied_names, list): tied_names = [tied_names]
        for tied_name in tied_names:
            if not np.isin(tied_name, self.linename_n):
                # raise ValueError((f"The tied line '{tied_name}' is not included in the available line list."))
                tied_names.remove(tied_name)
                continue
            i_tied = np.where(self.linename_n == tied_name)[0][0]
            tied_valid = self.mask_valid_cn[:, i_tied].sum() > 0 # if tied_name exists in any one comp

            if ref_valid & tied_valid & (tied_name != ref_name): 
                self.linelink_n[i_tied] = ref_name
                if use_pyneb:
                    tied_wave, tied_atomdata = self.search_pyneb(tied_name, ret_atomdata=True)[1:]
                    ref_wave,   ref_atomdata = self.search_pyneb( ref_name, ret_atomdata=True)[1:]
                    logdens = np.linspace(0, 12, 25)
                    logtems = np.linspace(np.log10(5e2), np.log10(3e4), 11)
                    tied_emi_td = tied_atomdata.getEmissivity(den=10.0**logdens, tem=10.0**logtems, wave=int(tied_wave))
                    ref_emi_td  =  ref_atomdata.getEmissivity(den=10.0**logdens, tem=10.0**logtems, wave=int( ref_wave))
                    ratio_dt = (tied_emi_td / ref_emi_td).T
                    func_ratio_dt = RegularGridInterpolator((logdens, logtems), ratio_dt, method='linear', bounds_error=False)
                    self.linelink_dict[tied_name] = {'ref_name': ref_name, 'func_ratio_dt': func_ratio_dt}
                else:                
                    if tied_name == '[S II]:6718':
                        # https://ui.adsabs.harvard.edu/abs/2014A&A...561A..10P 
                        pok14_Rs = np.linspace(1.41, 0.45, 30)
                        pok14_logdens = 0.0543*np.tan(-3.0553*pok14_Rs+2.8506)+6.98-10.6905*pok14_Rs+9.9186*pok14_Rs**2-3.5442*pok14_Rs**3
                        def func_ratio_dt(pars): return [np.interp(pars[0], pok14_logdens, pok14_Rs)]
                        if self.verbose:
                            print_log(f"Line tying: [S II]:6718 is tied to [S II]:6733 with flux ratio from Proxauf et al.(2014) under the best-fit electron density.", self.log_message)
                    else:
                        if ratio is None:
                            tmp = self.lineratio_n[i_tied] / self.lineratio_n[i_ref] * 1.0 # to avoid overwrite in following updating
                            if self.verbose:
                                print_log(f"Line tying: {tied_name} is tied to {ref_name} with flux ratio, {tmp:.3f}, under electron density of 100 cm-3 and temperature of 1e4 K.", self.log_message)
                        else:
                            if isinstance(ratio, list): raise ValueError((f"Please input a single ratio for {tied_name}, not a list."))
                            tmp = ratio * 1.0 # force to input value
                            if self.verbose:
                                print_log(f"Line tying: {tied_name} is tied to {ref_name} with the input flux ratio, {tmp}.", self.log_message)
                        def func_ratio_dt(pars, ret=tmp): return [ret]
                    self.linelink_dict[tied_name] = {'ref_name': ref_name, 'func_ratio_dt': func_ratio_dt}

        if use_pyneb & self.verbose:
            if len(tied_names) > 1: print_log(f"    {tied_names} --> {ref_name}", self.log_message)
            if len(tied_names) == 1: print_log(f"    {tied_names[0]} --> {ref_name}", self.log_message)
                
    def release_pair(self, tied_names):
        if not isinstance(tied_names, list): tied_names = [tied_names]
        for tied_name in tied_names:
            if not np.isin(tied_name, self.linename_n):
                raise ValueError((f"The tied line '{tied_name}' is not included in the line list."))
            i_tied = np.where(self.linename_n == tied_name)[0][0]
            self.linelink_n[i_tied] = 'free'
            self.linelink_dict.pop(tied_name)
            if self.verbose:
                print_log(f"{tied_name} becomes untied.", self.log_message)
  
    def initialize_linelink(self):
        # initialize the list of refered line names
        self.linelink_n = np.zeros_like(self.linename_n); self.linelink_n[:] = 'free'
        # initialize the ratio library dict of tied lines
        self.linelink_dict = {}

        if self.use_pyneb & self.verbose:
            print_log(f"Line tying (if line available) with flux ratios from pyneb under the best-fit (or fixed) electron density and temperature:", self.log_message)

        self.tie_pair(['Hb','Hg','Hd','H7','H8'], ['Ha','Hb','Hg','Hd']) # use set alternative line is Ha is not covered
        self.tie_pair('[S II]:6718', '[S II]:6733')
        self.tie_pair('[N II]:6550', '[N II]:6585')
        self.tie_pair('[O I]:6366', '[O I]:6302')
        self.tie_pair('[N I]:5199', '[N I]:5202')
        self.tie_pair('[O III]:4960', '[O III]:5008')
        self.tie_pair('[Ne III]:3969', '[Ne III]:3870')
        self.tie_pair('[O II]:3727', '[O II]:3730')
        self.tie_pair('[Ne V]:3347', '[Ne V]:3427')

        # update mask of free lines and count num_coeffs
        self.update_mask_free()
        # update line ratios with default AV, logden, and logtem
        self.update_lineratio()
        
    def update_lineratio(self, AV=0, logden=2, logtem=4):
        for tied_name in self.linelink_dict:
            i_tied = np.where(self.linename_n == tied_name)
            # read flux ratio for given logden and logtem
            func_ratio_dt = self.linelink_dict[tied_name]['func_ratio_dt']
            self.lineratio_n[i_tied] = func_ratio_dt(np.array([logden, logtem]))[0]
            # reflect extinction
            ref_name = self.linelink_dict[tied_name]['ref_name']
            i_ref = np.where(self.linename_n == ref_name)
            tmp = ExtLaw(self.linerest_n[i_tied]) - ExtLaw(self.linerest_n[i_ref])
            self.lineratio_n[i_tied] *= 10.0**(-0.4 * AV * tmp)

    def update_mask_free(self):
        self.mask_free_cn  = np.zeros((self.num_comps, self.num_lines), dtype='bool')
        for i_comp in range(self.num_comps):
            self.mask_free_cn[i_comp] = self.mask_valid_cn[i_comp] & ~np.isin(self.linename_n, [*self.linelink_dict])
            if ~np.isin(self.cframe.info_c[i_comp]['line_used'][0], ['all', 'default']): 
                self.mask_free_cn[i_comp] &= np.isin(self.linename_n, self.cframe.info_c[i_comp]['line_used'])

        self.num_coeffs = self.mask_free_cn.sum()
        
        # set component name and enable mask for each free line; _e denotes free or coeffs
        self.component_e = [] # np.zeros((self.num_coeffs), dtype='<U16')
        for i_comp in range(self.num_comps):
            for i_line in range(self.num_lines):
                if self.mask_free_cn[i_comp, i_line]:
                    self.component_e.append(self.cframe.info_c[i_comp]['comp_name'])
        self.component_e = np.array(self.component_e)

        # mask free absorption lines
        absorption_comp_names = np.array([d['comp_name'] for d in self.cframe.info_c if d['sign'] == 'absorption'])
        self.mask_absorption_e = np.isin(self.component_e, absorption_comp_names)

        if self.verbose:
            print_log(f"Free lines in each components: ", self.log_message)
            for i_comp in range(self.num_comps):
                print_log(f"({i_comp}) '{self.cframe.info_c[i_comp]['comp_name']}' component has "+
                          f"{self.mask_free_cn[i_comp].sum()} free (out of total {self.mask_valid_cn[i_comp].sum()}) "+
                          f"{self.cframe.info_c[i_comp]['profile']}, {self.cframe.info_c[i_comp]['sign']} profiles: \n"+
                          f"    {list(self.linename_n[self.mask_free_cn[i_comp]])}", self.log_message)

    ##################

    def single_line(self, obs_wave_w, lamb_c_rest, voff, fwhm, flux, v0_redshift=0, R_inst_rw=1e8, sign='emission', profile=None):
        if fwhm <= 0: raise ValueError((f"Non-positive line fwhm: {fwhm}"))
        if flux < 0: raise ValueError((f"Negative line flux: {flux}"))

        lamb_c_obs = lamb_c_rest * (1 + v0_redshift)
        mu =   (1 + voff/299792.458) * lamb_c_obs
        fwhm_line = fwhm/299792.458  * lamb_c_obs

        if np.isscalar(R_inst_rw):
            local_R_inst = copy(R_inst_rw)
        else:
            local_R_inst = np.interp(lamb_c_obs, R_inst_rw[0], R_inst_rw[1])
        fwhm_inst = 1 / local_R_inst * lamb_c_obs

        fwhm_tot = np.sqrt(fwhm_line**2 + fwhm_inst**2)

        if np.isin(profile, ['gaussian', 'Gaussian']):
            sigma_tot = fwhm_tot / np.sqrt(np.log(256))
            model = np.exp(-0.5 * ((obs_wave_w-mu)/sigma_tot)**2) / (sigma_tot * np.sqrt(2*np.pi)) 
        else:
            if np.isin(profile, ['lorentz', 'Lorentz']):
                model = 1 / (1 + ((obs_wave_w-mu) / (fwhm_tot/2))**2) / np.pi
            else:
                raise ValueError((f"Please specify either a Gaussian or a Lorentz line profile."))

        if model.sum() <= 0: model[0] = 1e-10 # avoid error from full zero output in case line is not covered
        if sign == 'absorption': model *= -1 # set negative profile for absorption line

        return model * flux

    def models_single_comp(self, obs_wave_w, pars, list_valid, list_free, sign, profile):
        # update lineratio_n
        voff, fwhm, AV, logden, logtem = pars
        self.update_lineratio(AV, logden, logtem)
        
        models_scomp = []
        for i_free in list_free:
            model_sline = self.single_line(obs_wave_w, self.linerest_n[i_free], voff, fwhm, 
                                           1, self.v0_redshift, self.R_inst_rw, sign, profile) # flux=1
            list_linked = np.where(self.linelink_n == self.linename_n[i_free])[0]
            list_linked = list_linked[np.isin(list_linked, list_valid)]
            for i_linked in list_linked:
                model_sline += self.single_line(obs_wave_w, self.linerest_n[i_linked], voff, fwhm, 
                                                self.lineratio_n[i_linked], self.v0_redshift, self.R_inst_rw, sign, profile)
            models_scomp.append(model_sline)
        return np.array(models_scomp)
    
    def models_unitnorm_obsframe(self, obs_wave_w, input_pars, if_pars_flat=True, mask_lite_e=None, conv_nbin=None):
        # conv_nbin is not used for emission lines, it is added to keep a uniform format with other models
        if if_pars_flat: 
            par_cp = self.cframe.flat_to_arr(input_pars)
        else:
            par_cp = copy(input_pars)

        for i_comp in range(self.num_comps):
            list_valid = np.arange(len(self.linerest_n))[self.mask_valid_cn[i_comp,:]]
            list_free  = np.arange(len(self.linerest_n))[self.mask_free_cn[i_comp,:]]
            obs_flux_scomp_ew = self.models_single_comp(obs_wave_w, par_cp[i_comp], list_valid, list_free, 
                                                        self.cframe.info_c[i_comp]['sign'], self.cframe.info_c[i_comp]['profile'])

            if i_comp == 0: 
                obs_flux_mcomp_ew = obs_flux_scomp_ew
            else:
                obs_flux_mcomp_ew = np.vstack((obs_flux_mcomp_ew, obs_flux_scomp_ew))

        if mask_lite_e is not None:
            obs_flux_mcomp_ew = obs_flux_mcomp_ew[mask_lite_e,:]

        return obs_flux_mcomp_ew

    def mask_line_lite(self, enabled_comps='all'):
        self.enabled_e = np.zeros((self.num_coeffs), dtype='bool')
        if enabled_comps == 'all':
            self.enabled_e[:] = True
        else:
            for comp in enabled_comps:
                self.enabled_e[self.component_e == comp] = True
        return self.enabled_e

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

        fp0, fp1, fe0, fe1 = ff.search_model_index('line', ff.full_model_type)
        num_loops = ff.num_loops
        comp_c = self.cframe.comp_c
        num_comps = self.cframe.num_comps
        num_pars_per_comp = self.cframe.num_pars_per_comp

        # extract parameters of emission lines
        par_lcp = best_par_lp[:, fp0:fp1].reshape(num_loops, num_comps, num_pars_per_comp)
        # extract coefficients of all free lines to matrix
        coeff_lcn = np.zeros((num_loops, num_comps, self.num_lines))
        coeff_lcn[:, self.mask_free_cn] = best_coeff_le[:, fe0:fe1]
        # update lineratio_n to calculate tied lines        
        list_linked = np.where(np.isin(self.linename_n, [*self.linelink_dict]))[0]
        for i_line in list_linked:
            i_main = np.where(self.linename_n == self.linelink_n[i_line])[0]
            for i_loop in range(num_loops):
                for i_comp in range(num_comps):
                    self.update_lineratio(*tuple(par_lcp[i_loop, i_comp, 2:5]))
                    coeff_lcn[i_loop, i_comp, i_line] = coeff_lcn[i_loop, i_comp, i_main] * self.lineratio_n[i_line] * self.mask_valid_cn[i_comp, i_line]

        # list the properties to be output
        val_names = ['voff', 'fwhm', 'AV', 'e_den', 'e_temp']
        # append flux of each line
        for i_line in range(self.num_lines): val_names.append('{}'.format(self.linename_n[i_line]))

        # format of results
        # output_c['comp']['par_lp'][i_l,i_p]: parameters
        # output_c['comp']['coeff_le'][i_l,i_n]: coefficients of all emission lines (not only free lines)
        # output_c['comp']['values']['name_l'][i_l]: calculated values
        output_c = {}
        for i_comp in range(num_comps): 
            output_c[comp_c[i_comp]] = {} # init results for each comp
            output_c[comp_c[i_comp]]['par_lp']   = par_lcp[:, i_comp, :]
            output_c[comp_c[i_comp]]['coeff_le'] = coeff_lcn[:, i_comp, :]
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
            for i_line in range(self.num_lines):
                output_c[comp_c[i_comp]]['values'][val_names[i_line+num_pars_per_comp]] = output_c[comp_c[i_comp]]['coeff_le'][:, i_line]
                output_c['sum']['values'][val_names[i_line+num_pars_per_comp]] += output_c[comp_c[i_comp]]['coeff_le'][:, i_line]

        self.output_c = output_c # save to model frame
        self.num_loops = num_loops # for print_results
        self.spec_flux_scale = ff.spec_flux_scale # for print_results

        if print_results: self.print_results(log=ff.log_message, show_average=show_average)
        if return_results: return output_c

    def print_results(self, log=[], show_average=False):
        mask_l = np.ones(self.num_loops, dtype='bool')
        if not show_average: mask_l[1:] = False
        
        print_log('', log)
        print_log('Best-fit emission line components', log)

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
        print_log(tbl_border, log)
        print_log(tbl_title, log)
        print_log(tbl_border, log)

        names = ['Voff (km/s)', 'FWHM (km/s)', 'AV (Balmer decre.)', 'log e-density (cm-3)', 'log e-temperature(K)']
        for i_line in range(self.num_lines): names.append('{}'.format(self.linename_n[i_line]))
        for i_value in range(len(names)): 
            tbl_row = []
            tbl_row.append(names[i_value])
            for i_comp in range(self.num_comps):
                tmp_values_vl = self.output_c[[*self.output_c][i_comp]]['values']
                tbl_row.append(tmp_values_vl[[*tmp_values_vl][i_value]][mask_l].mean())
                tbl_row.append(tmp_values_vl[[*tmp_values_vl][i_value]].std())
            print_log(fmt_numbers.format(*tbl_row), log)
        print_log(tbl_border, log)  
        print_log(f'[Note] Rows starting with a line name show the observed line flux, in unit of {self.spec_flux_scale} erg/s/cm2.', log)
