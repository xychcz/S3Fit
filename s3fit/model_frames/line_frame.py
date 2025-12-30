# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: s3fit@xychen.me

import numpy as np
np.set_printoptions(linewidth=10000)
from copy import deepcopy as copy
from scipy.interpolate import RegularGridInterpolator

from ..auxiliaries.auxiliary_frames import ConfigFrame
from ..auxiliaries.auxiliary_functions import print_log, casefold, greek_letters, roman_to_int, color_list_dict, lamb_air_to_vac, convolve_fix_width_fft
from ..auxiliaries.basic_model_functions import single_line
from ..auxiliaries.extinct_laws import ExtLaw

class LineFrame(object):
    def __init__(self, mod_name=None, fframe=None, 
                 config=None, use_pyneb=False, 
                 v0_redshift=0, R_inst_rw=None, 
                 w_min=None, w_max=None, mask_valid_rw=None, 
                 verbose=True, log_message=[]):

        self.mod_name = mod_name
        self.fframe = fframe
        self.config = config
        self.use_pyneb = use_pyneb
        self.v0_redshift = v0_redshift
        self.R_inst_rw = R_inst_rw
        self.w_min = w_min
        self.w_max = w_max
        self.mask_valid_rw = mask_valid_rw
        self.verbose = verbose
        self.log_message = log_message

        self.cframe=ConfigFrame(self.config)
        self.comp_name_c = self.cframe.comp_name_c
        self.num_comps = len(self.cframe.info_c)

        ############################################################
        # to be compatible with old version <= 2.2.4
        if len(self.cframe.par_index_cP[0]) == 0:
            self.cframe.par_name_cp  = [['voff', 'fwhm', 'Av', 'log_e_den', 'log_e_tem'] for i_comp in range(self.num_comps)]
            self.cframe.par_index_cP = [{'voff': 0, 'fwhm': 1, 'Av': 2, 'log_e_den': 3, 'log_e_tem': 4} for i_comp in range(self.num_comps)]
        self.tie_pair = self.tie_line_fluxes
        self.release_pair = self.untie_line_fluxes
        ############################################################

        # set default info if not specified in config
        for i_comp in range(self.num_comps):
            if not ('line_used'  in self.cframe.info_c[i_comp]) : self.cframe.info_c[i_comp]['line_used'] = np.array(['default'])
            if not ('line_ties'  in self.cframe.info_c[i_comp]) : self.cframe.info_c[i_comp]['line_ties'] = ['default']
            if not ('H_hi_order' in self.cframe.info_c[i_comp]) : self.cframe.info_c[i_comp]['H_hi_order'] = False
            if not ('sign'       in self.cframe.info_c[i_comp]) : self.cframe.info_c[i_comp]['sign'] = 'emission'
            if not ('profile'    in self.cframe.info_c[i_comp]) : self.cframe.info_c[i_comp]['profile'] = 'Gaussian'
            # group line info to a list
            if isinstance(self.cframe.info_c[i_comp]['line_used'], str): self.cframe.info_c[i_comp]['line_used'] = [self.cframe.info_c[i_comp]['line_used']]
            self.cframe.info_c[i_comp]['line_used'] = np.array(self.cframe.info_c[i_comp]['line_used'])
            if isinstance(self.cframe.info_c[i_comp]['line_ties'], str): self.cframe.info_c[i_comp]['line_ties'] = [self.cframe.info_c[i_comp]['line_ties']] # to allow pure hydrogen
            if isinstance(self.cframe.info_c[i_comp]['line_ties'], tuple): self.cframe.info_c[i_comp]['line_ties'] = [self.cframe.info_c[i_comp]['line_ties']]
            if isinstance(self.cframe.info_c[i_comp]['line_ties'], list):
                if all( isinstance(i, str) for i in self.cframe.info_c[i_comp]['line_ties'] ):
                    if len(self.cframe.info_c[i_comp]['line_ties']) > 1: self.cframe.info_c[i_comp]['line_ties'] = [tuple(self.cframe.info_c[i_comp]['line_ties'])] # to allow pure hydrogen

        # load line list and count the number of independent model elements
        self.initialize_linelist()
        self.update_linelist()

        # set plot styles
        self.plot_style_C = {}
        self.plot_style_C['sum'] = {'color': 'C2', 'alpha': 0.75, 'linestyle': '-', 'linewidth': 1.5}
        i_red, i_green, i_purple = 0, 0, 0
        for (i_comp, comp_name) in enumerate(self.comp_name_c):
            self.plot_style_C[comp_name] = {'color': 'None', 'alpha': 0.75, 'linestyle': '-', 'linewidth': 0.75}
            i_par_voff = self.cframe.par_index_cP[i_comp]['voff']
            i_par_fwhm = self.cframe.par_index_cP[i_comp]['fwhm']
            voff_mid = 0.5 * (self.cframe.par_min_cp[i_comp][i_par_voff] + self.cframe.par_max_cp[i_comp][i_par_voff])
            fwhm_mid = 0.5 * (self.cframe.par_min_cp[i_comp][i_par_fwhm] + self.cframe.par_max_cp[i_comp][i_par_fwhm])
            if abs(voff_mid) < 500:
                if fwhm_mid < 1500: # narrow line
                    self.plot_style_C[comp_name]['alpha'] = 1
                    self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['green'][::-1], i_green, mode="wrap"))
                    i_green += 1
                elif fwhm_mid < 5000: # middle-broad line
                    self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['red'][::-1], i_red, mode="wrap"))
                    i_red += 1
                else: # broad line
                    self.plot_style_C[comp_name]['linewidth'] = 1.5
                    self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['red'][::-1], i_red, mode="wrap"))
                    i_red += 1
            else: # outflow line
                self.plot_style_C[comp_name]['color'] = str(np.take(color_list_dict['purple'], i_purple, mode="wrap"))
                i_purple += 1

    def initialize_linelist(self):
        # atomic lines up t0 3.0 micron
        # only include ions with IP <= 150 eV
        # line ratios calculated with pyneb under ne=100, Te=1e4
        # libraries:
        # https://physics.nist.gov/PhysRefData/ASD/lines_form.html (H ref: T8637)
        # https://linelist.pa.uky.edu/atomic/
        # references:
        # https://astronomy.nmsu.edu/drewski/tableofemissionlines.html
        # https://ui.adsabs.harvard.edu/abs/1987ApJ...318..145F/abstract, FO87, Seyfert I galaxy III Zw 77
        # https://ui.adsabs.harvard.edu/abs/2000AJ....120..562T/abstract, T00, hidden quasar IRAS P09104+4109
        # https://ui.adsabs.harvard.edu/abs/2011MNRAS.414.3360R/abstract, R11, type 2 quasar SDSS J11311.05+162739.5
        # https://ui.adsabs.harvard.edu/abs/1990ApJ...352..561O/abstract, O90, Orion nebula, NGC 4151, and other Seyfert galaxies
        # https://ui.adsabs.harvard.edu/abs/1996PASP..108..183O, OF96, NGC 1068
        # https://ui.adsabs.harvard.edu/abs/1997ApJ...475..469Z/abstract, Z97, HST composite quasar spectrum
        # https://ui.adsabs.harvard.edu/abs/2004ApJ...611..107L, LM04, narrow-line Seyfert 1 galaxies, IRAS 13224-3809 and 1H 0707-495
        # https://ui.adsabs.harvard.edu/abs/2001AJ....122..549V/abstract, VB01, SDSS composite quasar spectrum
        # https://ui.adsabs.harvard.edu/abs/2006A%26A...457...61R/abstract, RR06, AGN 0.8-2.4 micron spectra
        # https://ui.adsabs.harvard.edu/abs/1988ApJ...330..751P/abstract, P88, Ca II infrared triplet lines in AGN
        # https://ui.adsabs.harvard.edu/abs/2001A%26A...378L..45I/abstract, I01, blue compact dwarf galaxy SBS 0335-052
        # https://ui.adsabs.harvard.edu/abs/2004A%26A...421..539I/abstract, I04, blue compact dwarf galaxies Tol 1214-277 and Tol 65
        # https://ui.adsabs.harvard.edu/abs/1992A%26A...266..117A/abstract, A92, Seyfert 2, ESO 138 G1
        # https://ui.adsabs.harvard.edu/abs/2006ApJ...640..579G/abstract, G06, near-infrared quasar composite spectrum, 0.58-3.5 micron

        # _n denote line; _l already used for loop id
        self.linerest_n, self.lineratio_n, self.linename_n = [],[],[]

        ########################################################################################################################
        # Hydrogen lines upto upper level of 10 can be tied to estimate extinction; all ratios (except for Lyman series) are in relative to H-alpha
        self.linerest_n.append(1215.670); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly-alpha')
        self.linerest_n.append(1025.722); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly-beta')
        self.linerest_n.append( 972.537); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly-gamma')
        self.linerest_n.append( 949.743); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly-delta')
        self.linerest_n.append( 937.803); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly-epsilon')
        self.linerest_n.append( 930.748); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly-zeta')
        self.linerest_n.append( 926.226); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly-eta')
        self.linerest_n.append( 923.150); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly-theta')
        self.linerest_n.append( 920.963); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly-iota')
        self.linerest_n.append(6564.632); self.lineratio_n.append(1.0)     ; self.linename_n.append('H-alpha')
        self.linerest_n.append(4862.691); self.lineratio_n.append(0.349)   ; self.linename_n.append('H-beta')
        self.linerest_n.append(4341.691); self.lineratio_n.append(0.164)   ; self.linename_n.append('H-gamma')
        self.linerest_n.append(4102.899); self.lineratio_n.append(0.0904)  ; self.linename_n.append('H-delta')
        self.linerest_n.append(3971.202); self.lineratio_n.append(0.0555)  ; self.linename_n.append('H-epsilon')
        self.linerest_n.append(3890.158); self.lineratio_n.append(0.0367)  ; self.linename_n.append('H-zeta')
        self.linerest_n.append(3836.479); self.lineratio_n.append(0.0255)  ; self.linename_n.append('H-eta')
        self.linerest_n.append(3798.983); self.lineratio_n.append(0.0185)  ; self.linename_n.append('H-theta')
        self.linerest_n.append(18756.10); self.lineratio_n.append(0.118)   ; self.linename_n.append('Pa-alpha')
        self.linerest_n.append(12821.58); self.lineratio_n.append(0.0570)  ; self.linename_n.append('Pa-beta')
        self.linerest_n.append(10941.08); self.lineratio_n.append(0.0316)  ; self.linename_n.append('Pa-gamma')
        self.linerest_n.append(10052.12); self.lineratio_n.append(0.0194)  ; self.linename_n.append('Pa-delta')
        self.linerest_n.append(9548.588); self.lineratio_n.append(0.0128)  ; self.linename_n.append('Pa-epsilon')
        self.linerest_n.append(9231.546); self.lineratio_n.append(0.00888) ; self.linename_n.append('Pa-zeta')
        self.linerest_n.append(9017.384); self.lineratio_n.append(0.00644) ; self.linename_n.append('Pa-eta')
        self.linerest_n.append(40522.69); self.lineratio_n.append(0.0280)  ; self.linename_n.append('Br-alpha')
        self.linerest_n.append(26258.68); self.lineratio_n.append(0.0158)  ; self.linename_n.append('Br-beta')
        self.linerest_n.append(21661.21); self.lineratio_n.append(0.00971) ; self.linename_n.append('Br-gamma')
        self.linerest_n.append(19450.89); self.lineratio_n.append(0.00638) ; self.linename_n.append('Br-delta')
        self.linerest_n.append(18179.10); self.lineratio_n.append(0.00442) ; self.linename_n.append('Br-epsilon')
        self.linerest_n.append(17366.87); self.lineratio_n.append(0.00319) ; self.linename_n.append('Br-zeta')
        self.linerest_n.append(74598.41); self.lineratio_n.append(0.00891) ; self.linename_n.append('Pf-alpha')
        self.linerest_n.append(46537.74); self.lineratio_n.append(0.00566) ; self.linename_n.append('Pf-beta')
        self.linerest_n.append(37405.55); self.lineratio_n.append(0.00372) ; self.linename_n.append('Pf-gamma')
        self.linerest_n.append(32969.91); self.lineratio_n.append(0.00257) ; self.linename_n.append('Pf-delta')
        self.linerest_n.append(30392.02); self.lineratio_n.append(0.00186) ; self.linename_n.append('Pf-epsilon')
        ########################################################################################################################
        # only pick up one intermediate wavelength in He I fine/hyperfine structure lines
        self.linerest_n.append(2945.965); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:2946') # T00
        self.linerest_n.append(3188.666); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:3189') # R11
        self.linerest_n.append(3488.725); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:3489') # R11
        self.linerest_n.append(3889.748); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:3890')
        self.linerest_n.append(4027.335); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:4027') # R11
        self.linerest_n.append(4144.928); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:4144') # T00
        self.linerest_n.append(4472.740); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:4472')
        self.linerest_n.append(5877.249); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:5877')
        self.linerest_n.append(7067.165); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:7067')
        self.linerest_n.append(7283.357); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:7283') # O90
        self.linerest_n.append(7818.286); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:7818') # O90
        self.linerest_n.append(10030.46); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:10030') # OF96
        self.linerest_n.append(10033.90); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:10034') # O90
        self.linerest_n.append(10833.22); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:10833') # OF96
        self.linerest_n.append(20586.90); self.lineratio_n.append(1.0)     ; self.linename_n.append('He I:20587') # RR06, G06
        # 54.42 eV
        self.linerest_n.append(1640.420); self.lineratio_n.append(1.0)     ; self.linename_n.append('He II:1640') # FO87
        self.linerest_n.append(2734.106); self.lineratio_n.append(1.0)     ; self.linename_n.append('He II:2734') # T00
        self.linerest_n.append(3204.030); self.lineratio_n.append(1.0)     ; self.linename_n.append('He II:3204') # R11
        self.linerest_n.append(4687.020); self.lineratio_n.append(1.0)     ; self.linename_n.append('He II:4687')
        self.linerest_n.append(8239.050); self.lineratio_n.append(1.0)     ; self.linename_n.append('He II:8239') # OF96
        ########################################################################################################################
        self.linerest_n.append(9826.820); self.lineratio_n.append(1.0)     ; self.linename_n.append('[C I]:9827') # OF96
        self.linerest_n.append(9852.960); self.lineratio_n.append(1.0)     ; self.linename_n.append('[C I]:9853') # OF96
        # 11.26 eV
        self.linerest_n.append(1334.532); self.lineratio_n.append(1.0)     ; self.linename_n.append('C II:1335') # LM04
        self.linerest_n.append(1335.708); self.lineratio_n.append(1.0)     ; self.linename_n.append('C II:1336') # LM04
        self.linerest_n.append(7238.410); self.lineratio_n.append(1.0)     ; self.linename_n.append('C II:7238') # O90
        self.linerest_n.append(2324.210); self.lineratio_n.append(1.0)     ; self.linename_n.append('C II]:2324') # LM04
        self.linerest_n.append(2325.400); self.lineratio_n.append(1.0)     ; self.linename_n.append('C II]:2325') # LM04
        # # 24.38 eV
        self.linerest_n.append( 977.030); self.lineratio_n.append(1.0)     ; self.linename_n.append('C III:977') # Z97
        self.linerest_n.append(1908.734); self.lineratio_n.append(1.0)     ; self.linename_n.append('C III]:1909') # FO87
        # 47.89 eV
        self.linerest_n.append(1548.187); self.lineratio_n.append(1.0)     ; self.linename_n.append('C IV:1548')
        self.linerest_n.append(1550.772); self.lineratio_n.append(1.0)     ; self.linename_n.append('C IV:1551')
        ########################################################################################################################
        self.linerest_n.append(7470.369); self.lineratio_n.append(1.0)     ; self.linename_n.append('N I:7470') # O90
        self.linerest_n.append(8682.666); self.lineratio_n.append(1.0)     ; self.linename_n.append('N I:8683') # O90
        self.linerest_n.append(8705.637); self.lineratio_n.append(1.0)     ; self.linename_n.append('N I:8706') # O90
        self.linerest_n.append(8714.096); self.lineratio_n.append(1.0)     ; self.linename_n.append('N I:8714') # O90
        self.linerest_n.append(10407.55); self.lineratio_n.append(1.0)     ; self.linename_n.append('N I:10408') # RR06
        self.linerest_n.append(3467.490); self.lineratio_n.append(1.0)     ; self.linename_n.append('[N I]:3467.49') # R11
        # self.linerest_n.append(3467.536); self.lineratio_n.append(1.0)     ; self.linename_n.append('[N I]:3467.54') # R11
        self.linerest_n.append(5199.349); self.lineratio_n.append(0.783)   ; self.linename_n.append('[N I]:5199') # ratio in relative to [N I]:5202
        self.linerest_n.append(5201.705); self.lineratio_n.append(1.0)     ; self.linename_n.append('[N I]:5202')
        # 14.53 eV
        self.linerest_n.append(2143.450); self.lineratio_n.append(1.0)     ; self.linename_n.append('N II]:2143') # LM04
        self.linerest_n.append(5756.240); self.lineratio_n.append(1.0)     ; self.linename_n.append('[N II]:5756') # T00
        self.linerest_n.append(6549.860); self.lineratio_n.append(0.340)   ; self.linename_n.append('[N II]:6550') # ratio in relative to [N II]:6585
        self.linerest_n.append(6585.270); self.lineratio_n.append(1.0)     ; self.linename_n.append('[N II]:6585')
        # 29.60 eV
        self.linerest_n.append( 989.790); self.lineratio_n.append(1.0)     ; self.linename_n.append('N III:990') # Z97
        self.linerest_n.append( 991.514); self.lineratio_n.append(1.0)     ; self.linename_n.append('N III:991.5') # Z97
        # self.linerest_n.append( 991.579); self.lineratio_n.append(1.0)     ; self.linename_n.append('N III:991.6') # Z97
        self.linerest_n.append(4512.150); self.lineratio_n.append(1.0)     ; self.linename_n.append('N III:4512') # I01
        self.linerest_n.append(1746.823); self.lineratio_n.append(1.0)     ; self.linename_n.append('N III]:1747') # VB01
        self.linerest_n.append(1748.656); self.lineratio_n.append(1.0)     ; self.linename_n.append('N III]:1749') # VB01
        # 47.45 eV
        self.linerest_n.append(1486.496); self.lineratio_n.append(1.0)     ; self.linename_n.append('N IV]:1486') # FO87
        # 77.47 eV
        self.linerest_n.append(1238.821); self.lineratio_n.append(1.0)     ; self.linename_n.append('N V:1239') # FO87
        self.linerest_n.append(1242.804); self.lineratio_n.append(1.0)     ; self.linename_n.append('N V:1243') # FO87
        ########################################################################################################################
        self.linerest_n.append(1302.168); self.lineratio_n.append(1.0)     ; self.linename_n.append('O I:1302') # VB01
        self.linerest_n.append(6048.112); self.lineratio_n.append(1.0)     ; self.linename_n.append('O I:6048') # I01
        self.linerest_n.append(7004.161); self.lineratio_n.append(1.0)     ; self.linename_n.append('O I:7004') # I01
        self.linerest_n.append(7256.447); self.lineratio_n.append(1.0)     ; self.linename_n.append('O I:7256') # O90
        self.linerest_n.append(8448.680); self.lineratio_n.append(1.0)     ; self.linename_n.append('O I:8449') # P88
        self.linerest_n.append(11290.00); self.lineratio_n.append(1.0)     ; self.linename_n.append('O I:11290') # RR06
        self.linerest_n.append(6302.046); self.lineratio_n.append(1.0)     ; self.linename_n.append('[O I]:6302')
        self.linerest_n.append(6365.536); self.lineratio_n.append(0.319)   ; self.linename_n.append('[O I]:6366') # ratio in relative to [O I]:6302
        # 13.62 eV
        self.linerest_n.append(4318.353); self.lineratio_n.append(1.0)     ; self.linename_n.append('O II:4318') # R11
        self.linerest_n.append(4416.138); self.lineratio_n.append(1.0)     ; self.linename_n.append('O II:4416') # R11
        self.linerest_n.append(3727.092); self.lineratio_n.append(0.741)   ; self.linename_n.append('[O II]:3727') # ratio in relative to [O II]:3730
        self.linerest_n.append(3729.875); self.lineratio_n.append(1.0)     ; self.linename_n.append('[O II]:3730')
        self.linerest_n.append(7322.010); self.lineratio_n.append(1.0)     ; self.linename_n.append('[O II]:7322') # FO87
        self.linerest_n.append(7332.750); self.lineratio_n.append(1.0)     ; self.linename_n.append('[O II]:7333') # FO87
        # 35.12 eV
        self.linerest_n.append(3133.702); self.lineratio_n.append(1.0)     ; self.linename_n.append('O III:3134') # R11
        self.linerest_n.append(3313.282); self.lineratio_n.append(1.0)     ; self.linename_n.append('O III:3313') # R11
        self.linerest_n.append(3445.039); self.lineratio_n.append(1.0)     ; self.linename_n.append('O III:3445') # FO87
        self.linerest_n.append(1660.809); self.lineratio_n.append(1.0)     ; self.linename_n.append('O III]:1661') # FO87
        self.linerest_n.append(1666.150); self.lineratio_n.append(1.0)     ; self.linename_n.append('O III]:1666') # FO87
        self.linerest_n.append(2321.664); self.lineratio_n.append(1.0)     ; self.linename_n.append('[O III]:2322') # FO87
        self.linerest_n.append(4364.436); self.lineratio_n.append(1.0)     ; self.linename_n.append('[O III]:4364')
        self.linerest_n.append(4960.295); self.lineratio_n.append(0.335)   ; self.linename_n.append('[O III]:4960') # ratio in relative to [O III]:5008
        self.linerest_n.append(5008.240); self.lineratio_n.append(1.0)     ; self.linename_n.append('[O III]:5008')
        # 54.93 eV
        self.linerest_n.append(1397.232); self.lineratio_n.append(1.0)     ; self.linename_n.append('O IV]:1397')
        self.linerest_n.append(1399.780); self.lineratio_n.append(1.0)     ; self.linename_n.append('O IV]:1400')
        # 113.90 eV
        self.linerest_n.append(1031.912); self.lineratio_n.append(1.0)     ; self.linename_n.append('O VI:1032') # VB01
        self.linerest_n.append(1037.613); self.lineratio_n.append(1.0)     ; self.linename_n.append('O VI:1038') # VB01
        ########################################################################################################################
        # 40.96 eV
        self.linerest_n.append(3869.860); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ne III]:3870')
        self.linerest_n.append(3968.590); self.lineratio_n.append(0.301)   ; self.linename_n.append('[Ne III]:3969')
        # 97.11 eV
        self.linerest_n.append(3346.783); self.lineratio_n.append(0.366)   ; self.linename_n.append('[Ne V]:3347')
        self.linerest_n.append(3426.864); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ne V]:3427')
        # 207.27 eV
        # self.linerest_n.append( 770.409); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ne VIII:770') # Z97
        # self.linerest_n.append( 780.324); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ne VIII:780') # Z97
        ########################################################################################################################
        # 7.65 eV
        self.linerest_n.append(2796.352); self.lineratio_n.append(1.0)     ; self.linename_n.append('Mg II]:2796') # R11
        self.linerest_n.append(2803.531); self.lineratio_n.append(1.0)     ; self.linename_n.append('Mg II]:2804') # R11
        # 109.24 eV
        self.linerest_n.append(2783.500); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Mg V]:2784') # R11
        self.linerest_n.append(2928.800); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Mg V]:2929') # T00
        ########################################################################################################################
        # 18.83 eV
        self.linerest_n.append(1854.716); self.lineratio_n.append(1.0)     ; self.linename_n.append('Al III:1855') # VB01
        self.linerest_n.append(1862.790); self.lineratio_n.append(1.0)     ; self.linename_n.append('Al III:1863') # VB01
        ########################################################################################################################
        # 8.15 eV
        self.linerest_n.append(1260.422); self.lineratio_n.append(1.0)     ; self.linename_n.append('Si II:1260') # LM04
        self.linerest_n.append(1264.730); self.lineratio_n.append(1.0)     ; self.linename_n.append('Si II:1265') # LM04
        self.linerest_n.append(6348.860); self.lineratio_n.append(1.0)     ; self.linename_n.append('Si II:6349') # I04
        # 16.35 eV
        self.linerest_n.append(1892.030); self.lineratio_n.append(1.0)     ; self.linename_n.append('Si III]:1892') # ref Dhanda et al 2006 not found
        # 33.49 eV
        self.linerest_n.append(1393.755); self.lineratio_n.append(1.0)     ; self.linename_n.append('Si IV:1394') # VB01
        self.linerest_n.append(1402.770); self.lineratio_n.append(1.0)     ; self.linename_n.append('Si IV:1403') # VB01
        # 
        self.linerest_n.append(19650.00); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Si VI]:19650') # RR06
        # 
        self.linerest_n.append(14304.90); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Si X]:14305') # RR06
        ########################################################################################################################
        # 
        self.linerest_n.append(11886.100); self.lineratio_n.append(1.0)     ; self.linename_n.append('[P II]:11886') # RR06
        ########################################################################################################################
        # 10.36 eV
        self.linerest_n.append(4069.749); self.lineratio_n.append(1.0)     ; self.linename_n.append('[S II]:4070') # R11
        self.linerest_n.append(4077.500); self.lineratio_n.append(1.0)     ; self.linename_n.append('[S II]:4078') # R11
        self.linerest_n.append(6718.295); self.lineratio_n.append(1.340)   ; self.linename_n.append('[S II]:6718') # ratio tied to [S II]:6733
        self.linerest_n.append(6732.674); self.lineratio_n.append(1.0)     ; self.linename_n.append('[S II]:6733')
        self.linerest_n.append(10289.55); self.lineratio_n.append(1.0)     ; self.linename_n.append('[S II]:10290') # OF96
        self.linerest_n.append(10323.32); self.lineratio_n.append(1.0)     ; self.linename_n.append('[S II]:10323') # OF96
        self.linerest_n.append(10339.24); self.lineratio_n.append(1.0)     ; self.linename_n.append('[S II]:10339') # O90
        self.linerest_n.append(10373.34); self.lineratio_n.append(1.0)     ; self.linename_n.append('[S II]:10373') # RR06
        # 23.33 eV
        self.linerest_n.append(6313.800); self.lineratio_n.append(1.0)     ; self.linename_n.append('[S III]:6313') # R11
        self.linerest_n.append(9071.100); self.lineratio_n.append(1.0)     ; self.linename_n.append('[S III]:9071') # OF96
        self.linerest_n.append(9533.200); self.lineratio_n.append(1.0)     ; self.linename_n.append('[S III]:9533') # RR06
        # 
        self.linerest_n.append(12523.00); self.lineratio_n.append(1.0)     ; self.linename_n.append('[S IX]:12523') # RR06
        ########################################################################################################################
        # 12.97 eV
        self.linerest_n.append(8581.050); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Cl II]:8581') # OF96
        # 23.81 eV
        self.linerest_n.append(5519.250); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Cl III]:5519') # I04, T00
        self.linerest_n.append(5539.430); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Cl III]:5539') # I04, T00
        ########################################################################################################################
        self.linerest_n.append(7870.359); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ar I:7870') # R11
        # 27.63 eV
        self.linerest_n.append(7137.800); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ar III]:7138') # R11
        self.linerest_n.append(7753.200); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ar III]:7753') # A92
        # 40.74 eV
        self.linerest_n.append(2854.480); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ar IV]:2854') # T00
        self.linerest_n.append(2868.990); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ar IV]:2869') # T00
        self.linerest_n.append(4712.690); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ar IV]:4713')
        self.linerest_n.append(4741.490); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ar IV]:4741')
        self.linerest_n.append(7172.500); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ar IV]:7173') # I04
        self.linerest_n.append(7239.400); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ar IV]:7239') # I04
        self.linerest_n.append(7264.700); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ar IV]:7265') # I04
        # 59.81 eV
        self.linerest_n.append(7007.300); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ar V]:7007') # A92, I04
        ########################################################################################################################
        # 8.88 eV
        self.linerest_n.append(8500.360); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ca II:8500') # P88
        self.linerest_n.append(8544.440); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ca II:8544') # P88
        self.linerest_n.append(8664.520); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ca II:8665') # P88
        # 
        self.linerest_n.append(23211.00); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ca III]:23211') # RR06
        # 67.10 eV
        self.linerest_n.append(5310.590); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ca V]:5311') # R11
        ########################################################################################################################
        # 7.87 eV; UV/optical Fe II lines are reflected in Fe II templates in AGN frame
        self.linerest_n.append(9131.640); self.lineratio_n.append(1.0)     ; self.linename_n.append('Fe II:9132') # RR06
        self.linerest_n.append(9180.324); self.lineratio_n.append(1.0)     ; self.linename_n.append('Fe II:9180') # RR06
        self.linerest_n.append(9205.669); self.lineratio_n.append(1.0)     ; self.linename_n.append('Fe II:9206') # RR06, G06
        self.linerest_n.append(10005.11); self.lineratio_n.append(1.0)     ; self.linename_n.append('Fe II:10005') # RR06
        self.linerest_n.append(4288.600); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe II]:4288') # I01
        self.linerest_n.append(4453.347); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe II]:4453') # I01
        self.linerest_n.append(7157.130); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe II]:7157') # OF96
        self.linerest_n.append(7173.980); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe II]:7174') # OF96
        self.linerest_n.append(7454.590); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe II]:7455') # O90
        self.linerest_n.append(8619.319); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe II]:8619') # OF96
        self.linerest_n.append(8894.354); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe II]:8894') # O90
        self.linerest_n.append(12570.24); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe II]:12570') # RR06
        self.linerest_n.append(13209.15); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe II]:13209') # RR06
        self.linerest_n.append(16439.98); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe II]:16440') # RR06
        # 16.18 eV
        self.linerest_n.append(5086.190); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe III]:5086') # I01
        self.linerest_n.append(5271.870); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe III]:5272') # I04
        # 30.65 eV
        self.linerest_n.append(2830.190); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe IV]:2830') # R11
        self.linerest_n.append(2836.570); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe IV]:2836') # R11
        self.linerest_n.append(4904.440); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe IV]:4904') # R11
        self.linerest_n.append(5238.760); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe IV]:5239') # R11
        # 54.80 eV
        self.linerest_n.append(3840.360); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe V]:3840') # R11
        self.linerest_n.append(3892.380); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe V]:3892') # FO87, R11
        self.linerest_n.append(3912.440); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe V]:3912') # FO87
        self.linerest_n.append(4072.390); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe V]:4072') # FO87
        self.linerest_n.append(4181.770); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe V]:4182') # R11
        self.linerest_n.append(4230.460); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe V]:4230') # I04
        # 75 eV 
        self.linerest_n.append(3663.540); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe VI]:3664') # R11
        self.linerest_n.append(5099.230); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe VI]:5099')
        self.linerest_n.append(5147.180); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe VI]:5147') # R11
        self.linerest_n.append(5177.480); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe VI]:5177') # R11
        self.linerest_n.append(5336.660); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe VI]:5337') # R11
        self.linerest_n.append(5425.730); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe VI]:5426') # R11
        self.linerest_n.append(5639.200); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe VI]:5639') # R11
        self.linerest_n.append(5678.530); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe VI]:5679') # R11
        # 99.00 eV
        self.linerest_n.append(3587.110); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe VII]:3587') # FO87
        self.linerest_n.append(3759.690); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe VII]:3760') # FO87, R11
        self.linerest_n.append(4894.740); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe VII]:4895') # R11
        self.linerest_n.append(5159.850); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe VII]:5160') # R11
        self.linerest_n.append(5277.250); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe VII]:5277') # R11
        self.linerest_n.append(5722.300); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe VII]:5722') # FO87, R11
        self.linerest_n.append(6087.980); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Fe VII]:6087') # FO87, R11
        ########################################################################################################################
        # 7.64 eV
        self.linerest_n.append(7379.860); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ni II]:7380') # A92
        self.linerest_n.append(7413.650); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ni II]:7414') # O90
        # 18.17 eV
        self.linerest_n.append(7892.100); self.lineratio_n.append(1.0)     ; self.linename_n.append('[Ni III]:7892') # VB01
        ########################################################################################################################

        self.linename_n = np.array(self.linename_n)
        self.linerest_n = np.array(self.linerest_n)
        self.lineratio_n = np.array(self.lineratio_n)

        self.linelist_full = copy(self.linename_n)        
        # list main lines in G06 and RR06
        self.linelist_default = np.array(['Ly-beta', 'O VI:1032', 'O VI:1038', 'Ly-alpha', 'N V:1239', 'N V:1243', 'Si IV:1394', 'O IV]:1397', 'O IV]:1400', 
                                          'Si IV:1403', 'C IV:1548', 'He II:1640', 'Al III:1855', 'Al III:1863', 'C III]:1909', 'Mg II]:2796', 'Mg II]:2804', 
                                          '[Ne V]:3347', '[Ne V]:3427', '[O II]:3727', '[O II]:3730', '[Ne III]:3870', 'H-zeta', '[Ne III]:3969', 
                                          'H-epsilon', 'H-delta', 'H-gamma', '[O III]:4364', 'H-beta', '[O III]:4960', '[O III]:5008', '[N I]:5199', '[N I]:5202', 'He I:5877', 
                                          '[O I]:6302', '[O I]:6366', '[N II]:6550', 'H-alpha', '[N II]:6585', '[S II]:6718', '[S II]:6733', 
                                          'He I:7067', '[Ar III]:7138', '[O II]:7322', '[O II]:7333', '[Ni III]:7892', 'O I:8449', '[S III]:9071', 'Fe II:9206', '[S III]:9533', 
                                          'Pa-epsilon', 'Pa-delta', 'He I:10833', 'Pa-gamma', 'O I:11290', '[P II]:11886', '[Fe II]:12570', 'Pa-beta', 
                                          '[Fe II]:13209', '[Si X]:14305', '[Fe II]:16440', 'Pa-alpha', 'Br-delta', '[Si VI]:19650', 'He I:20587', 'Br-gamma', 'Br-beta', 'Br-alpha'])
        
        if any(self.cframe.info_c[i_comp]['H_hi_order'] for i_comp in range(self.num_comps)): 
            self.enable_H_hi_order = True
            self.H_level_upper_max = 40 # limited by pyneb H1._Energy
            if not self.use_pyneb: raise ValueError((f"Please enable pyneb in line config to include high order Hydrogen lines."))
        else:
            self.enable_H_hi_order = False
            self.H_level_upper_max = 10

        self.H_levels_dict = {}
        for u in ['-'+g for g in greek_letters] + [str(i+1) for i in range(self.H_level_upper_max)]: 
            self.H_levels_dict['Ly'+u] = {'lower': 1} # Lyman
            self.H_levels_dict['H' +u] = {'lower': 2} # Balmer
            self.H_levels_dict['Pa'+u] = {'lower': 3} # Paschen
            self.H_levels_dict['Br'+u] = {'lower': 4} # Brackett
            self.H_levels_dict['Pf'+u] = {'lower': 5} # Pfund
            self.H_levels_dict['Hu'+u] = {'lower': 6} # Humphreys
        for name in self.H_levels_dict:
            if len(name.split('-')) == 2: 
                self.H_levels_dict[name]['upper'] = self.H_levels_dict[name]['lower'] + greek_letters.index(name.split('-')[1]) + 1
            else: 
                self.H_levels_dict[name]['upper'] = int(name[1:]) if name[1].isdigit() else int(name[2:])
        
        # initialize pyneb
        if self.use_pyneb: 
            if self.verbose: print_log('PyNeb is used in the fitting to derive line emissivities and ratios of line doublets.', self.log_message)
            import pyneb
            self.pyneb = pyneb
            self.pyneblib = {'RecAtom': {'list': pyneb.atomicData.getAllAtoms(coll=False, rec=True)}, # save to avoid duplicate reading
                             'Atom'   : {'list': pyneb.atomicData.getAllAtoms(coll=True , rec=False)}} 

            # refresh line info with pyneb
            for i_line in range(len(self.linename_n)):
                linename, linerest = self.search_pyneb(self.linename_n[i_line], verbose=False) # show all message of not supported lines with verbose=True
                if linename is not None: 
                    self.linename_n[i_line] = linename
                    self.linerest_n[i_line] = linerest

            if self.enable_H_hi_order: 
                H_level_upper_min = 11 # add lines with lv_up from 11 to 40
                self.linelist_H_hi_order = np.array([n+u for n in ['Ly','H','Pa','Br','Pf'] for u in [str(i+H_level_upper_min) for i in range(self.H_level_upper_max-(H_level_upper_min-1))]]) 
                self.linelist_full = np.hstack((self.linelist_full, self.linelist_H_hi_order))
                for linename in self.linelist_H_hi_order:
                    linename, linerest = self.search_pyneb(linename)
                    self.linename_n  = np.hstack((self.linename_n, linename))
                    self.linerest_n  = np.hstack((self.linerest_n, linerest))
                    self.lineratio_n = np.hstack((self.lineratio_n, 1.0))

    def search_pyneb(self, name, wave_medium='vacuum', ret_atomlib=False, verbose=True):
        # due to the current coverage of elements and atoms of pyneb, 
        # currently only use pyneb to identify Hydrogen recombination lines
        # add other recombination lines manually
        # also add collisionally excited lines manually if not provided by pyneb        

        if name in self.H_levels_dict:
            element = 'H'; notation = 1
            if not (element+str(notation) in self.pyneblib['RecAtom']):
                atomdata = self.pyneb.RecAtom(element, notation) # , extrapolate=False
                dE_ij = atomdata._Energy[:,None]-atomdata._Energy[None,:]
                wave_vac_ij = np.divide(1, dE_ij, where=dE_ij>0, out=np.zeros_like(dE_ij, dtype='float'))
                def func_emissivity(den, tem, wave_vac): 
                    lev_i, lev_j = np.unravel_index(np.argmin(abs(wave_vac_ij - wave_vac)), wave_vac_ij.shape)
                    return atomdata.getEmissivity(den=den, tem=tem, lev_i=lev_i+1, lev_j=lev_j+1)
                self.pyneblib['RecAtom'][element+str(notation)] = {'notation': element+str(notation), 'atomdata': atomdata, 'wave_vac_ij': wave_vac_ij, 'func_emissivity': func_emissivity} 
            atomlib = self.pyneblib['RecAtom'][element+str(notation)]
            wave_vac = atomlib['wave_vac_ij'][self.H_levels_dict[name]['upper']-1, self.H_levels_dict[name]['lower']-1]
            line_id = name
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
            if element+str(notation) in ['H1', 'He2']:
                if not (element+str(notation) in self.pyneblib['RecAtom']):
                    atomdata = self.pyneb.RecAtom(element, notation) # , extrapolate=False
                    dE_ij = atomdata._Energy[:,None]-atomdata._Energy[None,:]
                    wave_vac_ij = np.divide(1, dE_ij, where=dE_ij>0, out=np.zeros_like(dE_ij, dtype='float'))
                    def func_emissivity(den, tem, wave_vac): 
                        lev_i, lev_j = np.unravel_index(np.argmin(abs(wave_vac_ij - wave_vac)), wave_vac_ij.shape)
                        return atomdata.getEmissivity(den=den, tem=tem, lev_i=lev_i+1, lev_j=lev_j+1)
                    self.pyneblib['RecAtom'][element+str(notation)] = {'notation': element+str(notation), 'atomdata': atomdata, 'wave_vac_ij': wave_vac_ij, 'func_emissivity': func_emissivity} 
                atomlib = self.pyneblib['RecAtom'][element+str(notation)]
            else:
                if not (element+str(notation) in self.pyneblib['Atom']['list']):
                    if verbose: 
                        print_log(f"[WARNING] {name} not provided in pyneb, please add it manually with FitFrame.line.add_line(use_pyneb=False).", self.log_message)
                    if ret_atomlib: 
                        return None, None, None
                    else:
                        return None, None
                if not (element+str(notation) in self.pyneblib['Atom']):
                    atomdata = self.pyneb.Atom(element, notation) # , noExtrapol=True
                    dE_ij = atomdata._Energy[:,None]-atomdata._Energy[None,:]
                    wave_vac_ij = np.divide(1, dE_ij, where=dE_ij>0, out=np.zeros_like(dE_ij, dtype='float'))
                    def func_emissivity(den, tem, wave_vac): 
                        lev_i, lev_j = np.unravel_index(np.argmin(abs(wave_vac_ij - wave_vac)), wave_vac_ij.shape)
                        return atomdata.getEmissivity(den=den, tem=tem, lev_i=lev_i+1, lev_j=lev_j+1)
                    self.pyneblib['Atom'][element+str(notation)] = {'notation': element+str(notation), 'atomdata': atomdata, 'wave_vac_ij': wave_vac_ij, 'func_emissivity': func_emissivity} 
                atomlib = self.pyneblib['Atom'][element+str(notation)]
            wave = float(wave[:-2])*1e4 if wave[-2:] == 'um' else float(wave)
            if wave_medium == 'air': wave = lamb_air_to_vac(wave)
            wave_vac = atomlib['wave_vac_ij'].flatten()[np.argmin(np.abs(atomlib['wave_vac_ij'] - wave))]
            line_id = f'{ion}:{round(wave_vac)}'
            if np.abs(wave_vac - wave)/wave > 5e-4: 
                if verbose: 
                    print_log(f"[WARNING] For the input line name {name}, the relative wavelength difference between the input {wave} and the best-match pyneb value {wave_vac} "+
                              f"is larger than 0.05%. Please add it manually with FitFrame.line.add_line(use_pyneb=False).", self.log_message)
                if ret_atomlib: 
                    return None, None, None
                else:
                    return None, None

        if ret_atomlib: 
            return line_id, wave_vac, atomlib
        else:
            return line_id, wave_vac

    def add_line(self, linenames=None, linerests=None, lineratios=None, wave_medium='vacuum', force=False, use_pyneb=False, verbose=True):
        if not isinstance(linenames,  list): linenames  = [linenames]
        if not isinstance(linerests,  list): linerests  = [linerests]
        if not isinstance(lineratios, list): lineratios = [lineratios]
        for (i_line, linename) in enumerate(linenames):
            if use_pyneb:
                linename, linerest = self.search_pyneb(linename, wave_medium=wave_medium, verbose=verbose)
                lineratio = 1.0
                if linename is None:
                    # warning message already print in the above search_pyneb
                    continue # skip the following steps
            else:
                linerest = linerests[i_line] if wave_medium == 'vacuum' else lamb_air_to_vac(linerests[i_line])
                lineratio = lineratios[i_line] if lineratios[i_line] is not None else 1.0
            if linename in self.linelist_full: 
                if verbose: print_log(f"There is already the same line {linename} in the line list, set force=True to add this line forcibly.", self.log_message)
                continue # skip the following steps

            i_close = np.absolute(self.linerest_n - linerest).argmin()
            if (np.abs(self.linerest_n[i_close] - linerest) > 1) | force: 
                self.linename_n  = np.hstack((self.linename_n, linename))
                self.linerest_n  = np.hstack((self.linerest_n, linerest))
                self.lineratio_n = np.hstack((self.lineratio_n, lineratio))
                self.linelist_full = np.hstack((self.linelist_full, linename))
                self.linelist_default = np.hstack((self.linelist_default, linename))
                if verbose: 
                    if use_pyneb | (lineratios[i_line] is None):
                        print_log(f"{linename} with rest wavelength {linerest} (Angstrom) is added into the line list.", self.log_message)
                    else:
                        print_log(f"{linename} with rest wavelength {linerest} (Angstrom) and a fixed flux linkratio={lineratio} is added into the line list.", self.log_message)
            else:
                if verbose: print_log(f"There is a line {self.linename_n[i_close], self.linerest_n[i_close]} close to the input one {linename, linerest} (< 1 Angstrom), "+
                                      f"set force=True to add this line forcibly.", self.log_message)
        self.update_linelist()

    def delete_line(self, linenames=None, verbose=True):
        if not isinstance(linenames, list): linenames = [linenames]
        mask_remain_n = np.ones_like(self.linename_n, dtype='bool')
        for linename in linenames:
            mask_remain_n = self.linename_n != linename
            self.linename_n  = self.linename_n[mask_remain_n]
            self.linerest_n  = self.linerest_n[mask_remain_n]
            self.lineratio_n = self.lineratio_n[mask_remain_n]
            self.linelist_full = self.linelist_full[self.linelist_full != linename]
            self.linelist_default = self.linelist_default[self.linelist_default != linename]
            if verbose: print_log(f"{linename} is deleted from (or does not exist in) the line list.", self.log_message)
        self.update_linelist()

    def update_linelist(self):
        # update self.linename_n, self.linelist_full, and self.linelist_default and the other corresponding linelists

        # only keep covered lines
        mask_valid_n  = self.linerest_n > self.w_min
        mask_valid_n &= self.linerest_n < self.w_max
        self.linename_n = self.linename_n[mask_valid_n]
        self.linerest_n = self.linerest_n[mask_valid_n]
        self.lineratio_n = self.lineratio_n[mask_valid_n]
        # sort self.linename_n with increasing wavelength
        index_line = np.argsort(self.linerest_n)
        self.linename_n = self.linename_n[index_line]
        self.linerest_n = self.linerest_n[index_line]
        self.lineratio_n = self.lineratio_n[index_line]

        ########################################
        # linelists from self.linelist_full
        self.linelist_allowed          = np.array([line for line in self.linelist_full if (line[0] != '[') & (line.split(':')[0][-1] != ']')])
        self.linelist_intercombination = np.array([line for line in self.linelist_full if (line[0] != '[') & (line.split(':')[0][-1] == ']')])
        self.linelist_forbidden        = np.array([line for line in self.linelist_full if  line[0] == '['])

        linelist_H = np.array([n+u for n in ['Ly','H','Pa','Br','Pf'] for u in ['-'+g for g in greek_letters] + [str(i+1) for i in range(self.H_level_upper_max)]])
        linelist_H = linelist_H[np.isin(linelist_H, self.linelist_full)]
        linelist_nonH = self.linelist_full[~np.isin(self.linelist_full, linelist_H)]

        spectra_full = np.array([line.split(':')[0] for line in self.linelist_full])
        spectra_full = spectra_full[~np.isin(spectra_full, linelist_H)]
        spectra_uniq = []
        for spectrum in spectra_full:
            if not (spectrum in spectra_uniq): spectra_uniq.append(str(spectrum))
        spectra_uniq = np.array(spectra_uniq)
        self.linelist_spectra = {}
        self.linelist_spectra['H I'] = linelist_H
        for spectrum in spectra_uniq: self.linelist_spectra[str(spectrum)] = linelist_nonH[spectra_full == spectrum]

        notations_full = copy(spectra_full)
        for i in range(len(notations_full)):
            if notations_full[i][0]  == '[': notations_full[i] = notations_full[i][1:]
            if notations_full[i][-1] == ']': notations_full[i] = notations_full[i][:-1]
        notations_uniq = []
        for notation in notations_full:
            if not (notation in notations_uniq): notations_uniq.append(str(notation))
        notations_uniq = np.array(notations_uniq)
        self.linelist_notations = {}
        self.linelist_notations['H I'] = linelist_H
        for notation in notations_uniq: self.linelist_notations[str(notation)] = linelist_nonH[notations_full == notation]

        elements_full = copy(notations_full)
        for i in range(len(elements_full)):
            elements_full[i] = elements_full[i].split(' ')[0]
        elements_uniq = []
        for element in elements_full:
            if not (element in elements_uniq): elements_uniq.append(str(element))
        elements_uniq = np.array(elements_uniq)
        self.linelist_elements = {}
        self.linelist_elements['H'] = linelist_H
        for element in elements_uniq: self.linelist_elements[str(element)] = linelist_nonH[elements_full == element]
        ########################################

        linelist_used_total = np.array([])
        for i_comp in range(self.num_comps):
            # firstly read any specified lines in input config
            self.cframe.info_c[i_comp]['linelist'] = self.cframe.info_c[i_comp]['line_used'][np.isin(self.cframe.info_c[i_comp]['line_used'], self.linelist_full)]
            if np.isin(self.cframe.info_c[i_comp]['line_used'], ['all']).any():
                enable_all_lines = True
                self.cframe.info_c[i_comp]['linelist'] = np.hstack((self.cframe.info_c[i_comp]['linelist'],  self.linelist_full))
            else:
                enable_all_lines = False
                if np.isin(casefold(self.cframe.info_c[i_comp]['line_used']), casefold(['default', 'NLR', 'AGN_NLR', 'AGN NLR', 'HII', 'outflow'])).any():
                    self.cframe.info_c[i_comp]['linelist'] = np.hstack((self.cframe.info_c[i_comp]['linelist'],  self.linelist_default))
                if np.isin(casefold(self.cframe.info_c[i_comp]['line_used']), casefold(['BLR', 'AGN_BLR', 'AGN BLR'])).any():
                    self.cframe.info_c[i_comp]['linelist'] = np.hstack((self.cframe.info_c[i_comp]['linelist'],  
                                                                        self.linelist_allowed[np.isin(self.linelist_allowed, self.linelist_default)],
                                                                        self.linelist_intercombination[np.isin(self.linelist_intercombination, self.linelist_default)],
                                                                        self.linelist_elements['H']))
                if np.isin(casefold(self.cframe.info_c[i_comp]['line_used']), ['allowed', 'permitted']).any():
                    self.cframe.info_c[i_comp]['linelist'] = np.hstack((self.cframe.info_c[i_comp]['linelist'],  self.linelist_allowed))
                if np.isin(casefold(self.cframe.info_c[i_comp]['line_used']), ['intercombination', 'semiforbidden', 'semi-forbidden', 'semipermitted', 'semi-permitted']).any():
                    self.cframe.info_c[i_comp]['linelist'] = np.hstack((self.cframe.info_c[i_comp]['linelist'],  self.linelist_intercombination))
                if np.isin(casefold(self.cframe.info_c[i_comp]['line_used']), ['forbidden']).any():
                    self.cframe.info_c[i_comp]['linelist'] = np.hstack((self.cframe.info_c[i_comp]['linelist'],  self.linelist_forbidden))  
                if np.isin(self.cframe.info_c[i_comp]['line_used'], [*self.linelist_elements]).any():
                    for element in self.cframe.info_c[i_comp]['line_used'][np.isin(self.cframe.info_c[i_comp]['line_used'], [*self.linelist_elements])]:
                        self.cframe.info_c[i_comp]['linelist'] = np.hstack((self.cframe.info_c[i_comp]['linelist'],  self.linelist_elements[element]))  
                if np.isin(self.cframe.info_c[i_comp]['line_used'], [*self.linelist_spectra]).any():
                    for spectrum in self.cframe.info_c[i_comp]['line_used'][np.isin(self.cframe.info_c[i_comp]['line_used'], [*self.linelist_spectra])]:
                        self.cframe.info_c[i_comp]['linelist'] = np.hstack((self.cframe.info_c[i_comp]['linelist'],  self.linelist_spectra[spectrum])) 
            # remove duplicates
            self.cframe.info_c[i_comp]['linelist'] = self.linename_n[np.isin(self.linename_n, self.cframe.info_c[i_comp]['linelist'])]
            # disable high order Hydrogen lines if not specify in config
            if (not enable_all_lines) & self.enable_H_hi_order:
                if not self.cframe.info_c[i_comp]['H_hi_order']:
                    self.cframe.info_c[i_comp]['linelist'] = self.cframe.info_c[i_comp]['linelist'][~np.isin(self.cframe.info_c[i_comp]['linelist'], self.linelist_H_hi_order)] 
            linelist_used_total = np.hstack((linelist_used_total, self.cframe.info_c[i_comp]['linelist']))

        # only keep used lines
        mask_valid_n = np.isin(self.linename_n, linelist_used_total)
        self.linename_n = self.linename_n[mask_valid_n]
        self.linerest_n = self.linerest_n[mask_valid_n]
        self.lineratio_n = self.lineratio_n[mask_valid_n]
        self.num_lines = len(self.linename_n)

        self.mask_valid_cn = np.zeros((self.num_comps, self.num_lines), dtype='bool')
        # check minimum coverage
        if self.mask_valid_rw is not None:
            rest_wave_w = self.mask_valid_rw[0] / (1+self.v0_redshift)
            for i_comp in range(self.num_comps):
                i_par_voff = self.cframe.par_index_cP[i_comp]['voff']
                i_par_fwhm = self.cframe.par_index_cP[i_comp]['fwhm']
                voff_min = self.cframe.par_min_cp[i_comp][i_par_voff] if ~np.isnan(self.cframe.par_min_cp[i_comp][i_par_voff]) else -500
                voff_max = self.cframe.par_max_cp[i_comp][i_par_voff] if ~np.isnan(self.cframe.par_max_cp[i_comp][i_par_voff]) else  500
                fwhm_max = self.cframe.par_max_cp[i_comp][i_par_fwhm] if ~np.isnan(self.cframe.par_max_cp[i_comp][i_par_fwhm]) else 2000
                # check exceptions for tied pars
                for i_line in range(self.num_lines):
                    voff_w = (rest_wave_w/self.linerest_n[i_line]-1) * 299792.458
                    mask_line_w  = voff_w > (voff_min - fwhm_max) 
                    mask_line_w &= voff_w < (voff_max + fwhm_max) 
                    if mask_line_w.sum() >= 3: # at least 3 points in valid wavelength range
                        self.mask_valid_cn[i_comp,i_line] = (mask_line_w & self.mask_valid_rw[1]).sum() / mask_line_w.sum() >= 0.1 # at least 10% coverage fraction
        # only keep lines if they are specified 
        for i_comp in range(self.num_comps):
            self.mask_valid_cn[i_comp] &= np.isin(self.linename_n, self.cframe.info_c[i_comp]['linelist'])

        # only keep used lines
        mask_valid_n = self.mask_valid_cn.any(axis=0)
        self.linename_n = self.linename_n[mask_valid_n]
        self.linerest_n = self.linerest_n[mask_valid_n]
        self.lineratio_n = self.lineratio_n[mask_valid_n]
        self.mask_valid_cn = self.mask_valid_cn[:,mask_valid_n]
        self.num_lines = len(self.linename_n)

        # check the relations between lines and count num_coeffs
        self.update_linelink()
            
    def update_linelink(self):
        # initialize the list of refered line names
        self.linelink_name_cn = np.tile(np.zeros_like(self.linename_n), (self.num_comps, 1)); self.linelink_name_cn[:,:] = 'free'
        # initialize the ratio library dict of tied lines
        self.linelink_dict_cn = [{} for i_comp in range(self.num_comps)]

        if self.verbose:
            if self.use_pyneb:
                print_log(f"Ties of lines (if covered) with flux ratios from pyneb under the best-fit (or fixed) electron density and temperature:", self.log_message)
            else:
                print_log(f"Ties of lines (if covered) with default (100 cm-3 and 10000 K) or given flux ratios:", self.log_message)

        line_ties_default = ['Hydrogen lines',
                             ('[Ne V]:3427', '[Ne V]:3347'), 
                             ('[O II]:3730', '[O II]:3727'), 
                             ('[Ne III]:3870', '[Ne III]:3969'), 
                             ('[O III]:5008', '[O III]:4960'), 
                             ('[N I]:5202', '[N I]:5199'), 
                             ('[O I]:6302', '[O I]:6366'), 
                             ('[N II]:6585', '[N II]:6550'), 
                             ('[S II]:6733', '[S II]:6718', )]

        line_ties_collect = []
        for i_comp in range(self.num_comps):
            self.cframe.info_c[i_comp]['line_ties'] = list(dict.fromkeys(self.cframe.info_c[i_comp]['line_ties'])) # remove duplicates
            if any(n in self.cframe.info_c[i_comp]['line_ties'] for n in ['default', 'Default']):
                # self.cframe.info_c[i_comp]['line_ties'].remove('default')
                self.cframe.info_c[i_comp]['line_ties'] += line_ties_default
                self.cframe.info_c[i_comp]['line_ties'] = list(dict.fromkeys(self.cframe.info_c[i_comp]['line_ties'])) # remove duplicates
            line_ties_collect += self.cframe.info_c[i_comp]['line_ties']
        line_ties_collect = list(dict.fromkeys(line_ties_collect)) # remove duplicates

        for line_tie in line_ties_collect:
            if line_tie in ['default', 'Default']: continue
            components = [self.cframe.comp_name_c[i_comp] for i_comp in range(self.num_comps) if line_tie in self.cframe.info_c[i_comp]['line_ties']]
            if isinstance(line_tie, str):
                if casefold(line_tie) in ['hydrogen', 'hydrogen lines', 'hydrogen_lines']:
                    linelist_H = [l for l in self.linelist_elements['H'].tolist() if l[:2] != 'Ly' ] # do not include Lyman series due to gas obscuration and Lya forest
                    self.tie_line_fluxes(line_names=linelist_H, components=components) 
            else:
                if isinstance(line_tie, tuple): line_tie = list(line_tie)
                self.tie_line_fluxes(line_names=line_tie, components=components)

        # update mask of free lines and count num_coeffs
        self.update_mask_free()
        # update line ratios for each comp with default Av, log_e_den, and log_e_tem
        self.lineratio_cn = np.tile(np.zeros_like(self.lineratio_n), (self.num_comps, 1))
        for i_comp in range(self.num_comps): 
            self.update_lineratio(Av=0, log_e_den=2, log_e_tem=4, i_comp=i_comp)

    def tie_line_fluxes(self, line_names=None, tied_line_name=None, ref_line_name=None, ratio=None, components=None, use_pyneb=None):
        # prebuild the tying ratio libraries between tied lines and the reference line (1st valid one if multi given)
        # two input types:
        # 1) e.g., line_names = ['[O III]:4960', '[O III]:5008'], use pyneb or lineratio_n to obtain the ratio
        # 2) e.g., tied_line_name='[O III]:4960', ref_line_name='[O III]:5008', ratio=0.3, manually specify the ratio of tied_line_name/ref_line_name

        if not isinstance(line_names, list): line_names = [line_names]
        line_names = np.array(line_names)
        if components is not None:
            if isinstance(components, str): components = [components]
        else:
            components = copy(self.cframe.comp_name_c)
        if use_pyneb is None: use_pyneb = self.use_pyneb

        if line_names[0] is not None:
            if any(~np.isin(line_names, self.linelist_full)):
                print_log(f"[WARNING] The specified line(s), {np.array(ref_names)[~np.isin(ref_names, self.linelist_full)]}, are not available. " + 
                          f"Please check the available line name list with FitFrame.line.linelist_full, or add it manually with FitFrame.line.add_line() .", self.log_message)
        elif ref_line_name is not None:
            if not (ref_line_name in self.linelist_full):
                print_log(f"[WARNING] The reference line, {ref_line_name}, is not available. " + 
                          f"Please check the available line name list with FitFrame.line.linelist_full, or add it manually with FitFrame.line.add_line() .", self.log_message)
                return # skip the following steps
        elif tied_line_name is not None:
            if not (tied_line_name in self.linelist_full):
                print_log(f"[WARNING] The tied line, {tied_line_name}, is not available. " + 
                          f"Please check the available line name list with FitFrame.line.linelist_full, or add it manually with FitFrame.line.add_line() .", self.log_message)
                return # skip the following steps

        # get reference line for each comp
        ref_name_c, tied_names_c = [], []
        for i_comp in range(self.num_comps):
            ref_name_c.append('None')
            tied_names_c.append(['None'])
            if self.cframe.comp_name_c[i_comp] in components: 
                if line_names[0] is not None:
                    line_names_valid = line_names[np.isin(line_names, self.linename_n[self.mask_valid_cn[i_comp]])].tolist()
                    if len(line_names_valid) >= 1: ref_name_c[i_comp] = line_names_valid[0]
                    if len(line_names_valid) >= 2: tied_names_c[i_comp] = line_names_valid[1:]
                else:
                    if ref_line_name  in self.linename_n[self.mask_valid_cn[i_comp]]: ref_name_c[i_comp] = ref_line_name
                    if tied_line_name in self.linename_n[self.mask_valid_cn[i_comp]]: tied_names_c[i_comp] = [tied_line_name]

        mask_finish_c = np.zeros(self.num_comps, dtype='bool')
        for i_comp in range(self.num_comps):
            if mask_finish_c[i_comp]: continue
            ref_name = ref_name_c[i_comp]
            tied_names = tied_names_c[i_comp]
            if (ref_name != 'None') & (tied_names != ['None']):
                mask_sharetie_c = np.array([(ref_name_c[j_comp] == ref_name) & (tied_names_c[j_comp] == tied_names) for j_comp in range(self.num_comps)])
                mask_finish_c[mask_sharetie_c] = True
                for tied_name in tied_names:
                    i_tied = np.where(self.linename_n == tied_name)[0][0]
                    i_ref  = np.where(self.linename_n ==  ref_name)[0][0]
                    if use_pyneb:
                        tied_wave, tied_atomlib = self.search_pyneb(tied_name, ret_atomlib=True)[1:]
                        ref_wave,   ref_atomlib = self.search_pyneb( ref_name, ret_atomlib=True)[1:]
                        if tied_atomlib is None:
                            print_log(f"[WARNING] The tied line, '{tied_name}', is not provided by pyneb. " + 
                                      f"Please add the tying relation manually with FitFrame.line.tie_line_fluxes(ratio=, use_pyneb=False)", self.log_message)
                            tied_names.remove(tied_name)
                            continue
                        if ref_atomlib is None:
                            print_log(f"[WARNING] The reference line, '{ref_name}', is not provided by pyneb. " + 
                                      f"Please add the tying relation manually with FitFrame.line.tie_line_fluxes(ratio=, use_pyneb=False)", self.log_message)
                            tied_names = []
                            break
                        if ref_atomlib['notation'] != tied_atomlib['notation']:
                            print_log(f"[WARNING] The tied lines, '{ref_name}' and '{tied_name}' are from different ions. Only transitions from the same ion are allowed.", self.log_message)
                            tied_names.remove(tied_name)
                            continue
                        log_e_dens = np.linspace(0, 12, 25)
                        log_e_tems = np.linspace(np.log10(5e2), np.log10(3e4), 11)
                        tied_emi_td = tied_atomlib['func_emissivity'](den=10.0**log_e_dens, tem=10.0**log_e_tems, wave_vac=int(tied_wave))
                        ref_emi_td  =  ref_atomlib['func_emissivity'](den=10.0**log_e_dens, tem=10.0**log_e_tems, wave_vac=int( ref_wave))
                        ratio_dt = (tied_emi_td / ref_emi_td).T
                        func_ratio_dt = RegularGridInterpolator((log_e_dens, log_e_tems), ratio_dt, method='linear', bounds_error=False)
                    else:                
                        if (tied_name == '[S II]:6718') | (tied_name == '[S II]:6733'):
                            # https://ui.adsabs.harvard.edu/abs/2014A&A...561A..10P 
                            pok14_Rs = np.linspace(1.41, 0.45, 30) # [S II]:6718/6733
                            pok14_log_e_dens = 0.0543*np.tan(-3.0553*pok14_Rs+2.8506)+6.98-10.6905*pok14_Rs+9.9186*pok14_Rs**2-3.5442*pok14_Rs**3
                            if tied_name == '[S II]:6733': pok14_Rs = 1/pok14_Rs # [S II]:6733/6718
                            def func_ratio_dt(pars): return [np.interp(pars[0], pok14_log_e_dens, pok14_Rs)]
                            if self.verbose:
                                print_log(f"    Line tying: [S II]:6718,6733 are tied with flux ratio from Proxauf et al.(2014) under the best-fit electron density, "+
                                          f" for the components {np.array(self.cframe.comp_name_c)[mask_sharetie_c]}.", self.log_message)
                        else:
                            if ratio is None:
                                tmp = self.lineratio_n[i_tied] / self.lineratio_n[i_ref] * 1.0 # to avoid overwrite in following updating
                                if self.verbose:
                                    print_log(f"    Line tying: {tied_name} is tied to {ref_name} with flux ratio, {tmp:.3f}, under electron density of 100 cm-3 and temperature of 10000 K, "+
                                              f" for the components {np.array(self.cframe.comp_name_c)[mask_sharetie_c]}.", self.log_message)
                            else:
                                if isinstance(ratio, list): raise ValueError((f"Please input a single ratio for {tied_name}, not a list."))
                                tmp = ratio * 1.0 # force to input value
                                if self.verbose:
                                    print_log(f"    Line tying: {tied_name} is tied to {ref_name} with the user input flux ratio, {tmp}, "+
                                              f" for the components {np.array(self.cframe.comp_name_c)[mask_sharetie_c]}.", self.log_message)
                            def func_ratio_dt(pars, ret=tmp): return [ret]
                
                    for j_comp in range(self.num_comps):
                        if mask_sharetie_c[j_comp]: 
                            self.linelink_name_cn[j_comp,i_tied] = ref_name
                            self.linelink_dict_cn[j_comp][tied_name] = {'ref_name': ref_name, 'func_ratio_dt': func_ratio_dt}

                if use_pyneb & self.verbose:
                    if len(tied_names)  > 1: print_log(f"    {tied_names} --> {ref_name} for the components {np.array(self.cframe.comp_name_c)[mask_sharetie_c]}.", self.log_message)
                    if len(tied_names) == 1: print_log(f"    {tied_names[0]} --> {ref_name} for the components {np.array(self.cframe.comp_name_c)[mask_sharetie_c]}.", self.log_message)

    def untie_line_fluxes(self, tied_names, components=None):
        if not isinstance(tied_names, list): tied_names = [tied_names]
        if components is not None:
            if not isinstance(components, list): components = [components]
        else:
            components = copy(self.cframe.comp_name_c)

        for i_comp in range(self.num_comps):
            if self.cframe.comp_name_c[i_comp] in components: 
                for tied_name in tied_names:
                    if tied_name in self.linelink_dict_cn[i_comp]:
                        i_tied = np.where(self.linename_n == tied_name)[0][0]                    
                        self.linelink_name_cn[i_comp,i_tied] = 'free'
                        self.linelink_dict_cn[i_comp].pop(tied_name)
                        if self.verbose: print_log(f"{tied_name} becomes untied for the component {self.cframe.comp_name_c[i_comp]}.", self.log_message)
                    else:
                        if self.verbose: print_log(f"{tied_name} is not tied or nor covered by the spectrum for the component {self.cframe.comp_name_c[i_comp]}..", self.log_message)
        
    def update_lineratio(self, Av=None, log_e_den=None, log_e_tem=None, i_comp=None):
        for tied_name in self.linelink_dict_cn[i_comp]:
            i_tied = np.where(self.linename_n == tied_name)[0][0]
            # read flux ratio for given log_e_den and log_e_tem
            func_ratio_dt = self.linelink_dict_cn[i_comp][tied_name]['func_ratio_dt']
            self.lineratio_cn[i_comp, i_tied] = func_ratio_dt(np.array([log_e_den, log_e_tem]))[0]
            # reflect extinction
            ref_name = self.linelink_dict_cn[i_comp][tied_name]['ref_name']
            i_ref = np.where(self.linename_n == ref_name)[0][0]
            tmp = ExtLaw(self.linerest_n[i_tied]) - ExtLaw(self.linerest_n[i_ref])
            self.lineratio_cn[i_comp, i_tied] *= 10.0**(-0.4 * Av * tmp)

    def update_mask_free(self):
        self.mask_free_cn  = np.zeros((self.num_comps, self.num_lines), dtype='bool')
        for i_comp in range(self.num_comps):
            self.mask_free_cn[i_comp] = self.mask_valid_cn[i_comp] & ~np.isin(self.linename_n, [*self.linelink_dict_cn[i_comp]])
        self.num_coeffs_c = self.mask_free_cn.sum(axis=1)
        self.num_coeffs = self.mask_free_cn.sum()
        
        # set component name and enable mask for each free line; _e denotes free or coeffs
        self.comp_name_e = [] # np.zeros((self.num_coeffs), dtype='<U16')
        for i_comp in range(self.num_comps):
            for i_line in range(self.num_lines):
                if self.mask_free_cn[i_comp, i_line]:
                    self.comp_name_e.append(self.comp_name_c[i_comp])
        self.comp_name_e = np.array(self.comp_name_e)

        # mask free absorption lines
        absorption_comp_names = np.array([d['comp_name'] for d in self.cframe.info_c if d['sign'] == 'absorption'])
        self.mask_absorption_e = np.isin(self.comp_name_e, absorption_comp_names)

        if self.verbose:
            print_log(f"Free lines in each components: ", self.log_message)
            for i_comp in range(self.num_comps):
                print_log(f"({i_comp}) '{self.cframe.info_c[i_comp]['comp_name']}' component has "+
                          f"{self.mask_free_cn[i_comp].sum()} free (out of total {self.mask_valid_cn[i_comp].sum()}) "+
                          f"{self.cframe.info_c[i_comp]['profile']}, {self.cframe.info_c[i_comp]['sign']} profiles: \n"+
                          f"    {self.linename_n[self.mask_free_cn[i_comp]]}", self.log_message)

    ##########################################################################

    def models_single_comp(self, obs_wave_w, par_cp, i_comp):
        voff      = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['voff']]
        fwhm      = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['fwhm']]
        Av        = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['Av']]
        log_e_den = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['log_e_den']]
        log_e_tem = par_cp[i_comp][self.cframe.par_index_cP[i_comp]['log_e_tem']]

        # update lineratio_cn
        self.update_lineratio(Av, log_e_den, log_e_tem, i_comp)
        
        list_valid = np.arange(len(self.linerest_n))[self.mask_valid_cn[i_comp,:]]
        list_free  = np.arange(len(self.linerest_n))[self.mask_free_cn[i_comp,:]]
        models_scomp = []
        for i_free in list_free:
            model_sline = single_line(obs_wave_w, self.linerest_n[i_free], voff, fwhm, 
                                      1,  # flux=1
                                      self.v0_redshift, self.R_inst_rw, self.cframe.info_c[i_comp]['profile'])
            list_linked = np.where(self.linelink_name_cn[i_comp] == self.linename_n[i_free])[0]
            list_linked = list_linked[np.isin(list_linked, list_valid)]
            for i_linked in list_linked:
                model_sline += single_line(obs_wave_w, self.linerest_n[i_linked], voff, fwhm, 
                                           self.lineratio_cn[i_comp, i_linked], 
                                           self.v0_redshift, self.R_inst_rw, self.cframe.info_c[i_comp]['profile'])
            # detect and exclude weak lines
            int_flux_list = [1] + [self.lineratio_cn[i_comp, i_linked] for i_linked in list_linked]
            peak_flux_min = min(int_flux_list) / (fwhm / np.sqrt(np.log(256)) * np.sqrt(2*np.pi)) 
            if not any(model_sline > (peak_flux_min * 0.5)): model_sline *= 0 # require to cover half peak height
            
            models_scomp.append(model_sline)

        models_scomp = np.array(models_scomp)
        if self.cframe.info_c[i_comp]['sign'] == 'absorption': models_scomp *= -1 # set negative profile for absorption line

        return models_scomp
    
    def models_unitnorm_obsframe(self, obs_wave_w, par_p, mask_lite_e=None, conv_nbin=None):
        # conv_nbin is not used for emission lines, it is added to keep a uniform format with other models
        par_cp = self.cframe.reshape_by_comp(par_p, self.cframe.num_pars_c)

        obs_flux_mcomp_ew = None
        for i_comp in range(self.num_comps):
            list_valid = np.arange(len(self.linerest_n))[self.mask_valid_cn[i_comp,:]]
            list_free  = np.arange(len(self.linerest_n))[self.mask_free_cn[i_comp,:]]
            if len(list_free) >= 1:
                obs_flux_scomp_ew = self.models_single_comp(obs_wave_w, par_cp, i_comp)
                if obs_flux_mcomp_ew is None: 
                    obs_flux_mcomp_ew = obs_flux_scomp_ew
                else:
                    obs_flux_mcomp_ew = np.vstack((obs_flux_mcomp_ew, obs_flux_scomp_ew))

        if mask_lite_e is not None:
            obs_flux_mcomp_ew = obs_flux_mcomp_ew[mask_lite_e,:]

        return obs_flux_mcomp_ew

    def mask_lite_with_comps(self, enabled_comps=None, disabled_comps=None):
        if enabled_comps is not None:
            self.enabled_e = np.zeros((self.num_coeffs), dtype='bool')
            for comp_name in enabled_comps: self.enabled_e[self.comp_name_e == comp_name] = True
        else:
            self.enabled_e = np.ones((self.num_coeffs), dtype='bool')
            if disabled_comps is not None:
                for comp_name in disabled_comps: self.enabled_e[self.comp_name_e == comp_name] = False
        return self.enabled_e

    ##########################################################################
    ########################## Output functions ##############################

    def extract_results(self, step=None, if_print_results=True, if_return_results=False, if_rev_v0_redshift=False, if_show_average=False, lum_unit=None, **kwargs):

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
        self.spec_flux_scale = self.fframe.spec_flux_scale # for print_results
        comp_name_c = self.cframe.comp_name_c
        num_comps = self.cframe.num_comps
        par_name_cp = self.cframe.par_name_cp

        # list the properties to be output
        value_names_additive = self.linename_n.tolist()
        value_names_C = {}
        for (i_comp, comp_name) in enumerate(comp_name_c): value_names_C[comp_name] = value_names_additive

        # format of results
        # output_C['comp']['par_lp'][i_l,i_p]: parameters
        # output_C['comp']['coeff_le'][i_l,i_e]: coefficients
        # output_C['comp']['value_Vl']['name_l'][i_l]: calculated values
        output_C = {}
        for (i_comp, comp_name) in enumerate(comp_name_c):
            output_C[comp_name] = {} # init results for each comp
            output_C[comp_name]['value_Vl'] = {}
            for val_name in par_name_cp[i_comp] + value_names_C[comp_name]:
                output_C[comp_name]['value_Vl'][val_name] = np.zeros(self.num_loops, dtype='float')
        output_C['sum'] = {}
        output_C['sum']['value_Vl'] = {} # only init values for sum of all comp
        for val_name in value_names_additive:
            output_C['sum']['value_Vl'][val_name] = np.zeros(self.num_loops, dtype='float')

        # locate the results of the model in the full fitting results
        i_pars_0_of_mod, i_pars_1_of_mod, i_coeffs_0_of_mod, i_coeffs_1_of_mod = self.fframe.search_mod_index(self.mod_name, self.fframe.full_model_type)

        # extract parameters of emission lines; all comp have the same num of pars
        par_lcp = best_par_lp[:, i_pars_0_of_mod:i_pars_1_of_mod].reshape(self.num_loops, num_comps, self.cframe.num_pars_c[0])

        # extract coefficients of all free lines to matrix
        coeff_lcn = np.zeros((self.num_loops, num_comps, self.num_lines))
        coeff_lcn[:, self.mask_free_cn] = best_coeff_le[:, i_coeffs_0_of_mod:i_coeffs_1_of_mod]

        # update lineratio_cn to calculate tied lines
        for i_comp in range(num_comps):
            list_linked = np.where(np.isin(self.linename_n, [*self.linelink_dict_cn[i_comp]]))[0]
            for i_line in list_linked:
                i_main = np.where(self.linename_n == self.linelink_name_cn[i_comp,i_line])[0][0]
                for i_loop in range(self.num_loops):
                    Av        = par_lcp[i_loop, i_comp, self.cframe.par_index_cP[i_comp]['Av']]
                    log_e_den = par_lcp[i_loop, i_comp, self.cframe.par_index_cP[i_comp]['log_e_den']]
                    log_e_tem = par_lcp[i_loop, i_comp, self.cframe.par_index_cP[i_comp]['log_e_tem']]
                    self.update_lineratio(Av, log_e_den, log_e_tem, i_comp)
                    coeff_lcn[i_loop, i_comp, i_line] = coeff_lcn[i_loop, i_comp, i_main] * self.lineratio_cn[i_comp, i_line] * self.mask_valid_cn[i_comp, i_line]

        for (i_comp, comp_name) in enumerate(comp_name_c):
            output_C[comp_name]['par_lp']   = par_lcp[:, i_comp, :]
            output_C[comp_name]['coeff_le'] = coeff_lcn[:, i_comp, :]
            for (i_par, par_name) in enumerate(par_name_cp[i_comp]):
                output_C[comp_name]['value_Vl'][par_name] = par_lcp[:, i_comp, i_par]
            for (i_line, line_name) in enumerate(self.linename_n.tolist()):
                flux_sign = -1.0 if self.cframe.info_c[i_comp]['sign'] == 'absorption' else 1.0
                output_C[comp_name]['value_Vl'][line_name] = coeff_lcn[:, i_comp, i_line] * flux_sign
                output_C['sum']['value_Vl'][line_name] += coeff_lcn[:, i_comp, i_line] * flux_sign

        # output_C['sum'] = output_C.pop('sum') # move sum to the end

        ############################################################
        # keep aliases for output in old version <= 2.2.4
        for (i_comp, comp_name) in enumerate(comp_name_c):
            output_C[comp_name]['values'] = output_C[comp_name]['value_Vl']
        output_C['sum']['values'] = output_C['sum']['value_Vl']
        ############################################################
        
        self.output_C = output_C # save to model frame

        if if_print_results: self.print_results(log=self.fframe.log_message, if_show_average=if_show_average, lum_unit=lum_unit)
        if if_return_results: return output_C

    def print_results(self, log=[], if_show_average=False, lum_unit=None):
        print_log('#### Best-fit line model properties ####', log)

        mask_l = np.ones(self.num_loops, dtype='bool')
        if not if_show_average: mask_l[1:] = False
        # lum_unit_str = '(log Lsun) ' if lum_unit == 'Lsun' else '(log erg/s)'

        cols = 'Par/Line Name'
        fmt_cols = '| {:^20} |'
        fmt_numbers = '| {:^20} |' #fmt_numbers = '| {:=13.4f} |'
        for i_comp in range(self.num_comps): 
            cols += ','+self.cframe.comp_name_c[i_comp]
            fmt_cols += ' {:^18} |'
            fmt_numbers += ' {:=8.2f} +- {:=6.2f} |'
        cols_split = cols.split(',')
        tbl_title = fmt_cols.format(*cols_split)
        tbl_border = len(tbl_title)*'='
        print_log(tbl_border, log)
        print_log(tbl_title, log)
        print_log(tbl_border, log)

        # set the print name for each value
        value_names = [value_name for comp_name in self.output_C for value_name in self.output_C[comp_name]['value_Vl']]
        value_names = list(dict.fromkeys(value_names)) # remove duplicates
        print_names = {}
        for value_name in value_names: print_names[value_name] = value_name
        print_names['voff']      = 'Voff (km/s)'
        print_names['fwhm']      = 'FWHM (km/s)'
        print_names['Av']        = 'Av (Balmer decre.)'
        print_names['log_e_den'] = 'log e-density (cm-3)'
        print_names['log_e_tem'] = 'log e-temperature(K)'

        for i_value in range(len(value_names)): 
            tbl_row = []
            tbl_row.append(print_names[value_names[i_value]])
            for i_comp in range(self.num_comps):
                tmp_values_vl = self.output_C[[*self.output_C][i_comp]]['value_Vl']
                tbl_row.append(tmp_values_vl[[*tmp_values_vl][i_value]][mask_l].mean())
                tbl_row.append(tmp_values_vl[[*tmp_values_vl][i_value]].std())
            print_log(fmt_numbers.format(*tbl_row), log)
        print_log(tbl_border, log)  
        print_log(f'[Note] Rows starting with a line name show the observed line flux, in unit of {self.spec_flux_scale:.0e} erg/s/cm2.', log)
        print_log('', log)


