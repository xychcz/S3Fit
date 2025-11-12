# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: s3fit@xychen.me

import numpy as np
np.set_printoptions(linewidth=10000)
from copy import deepcopy as copy
from scipy.interpolate import RegularGridInterpolator

from ..auxiliary_func import print_log
from ..extinct_law import ExtLaw

class LineFrame(object):
    def __init__(self, cframe=None, v0_redshift=0, R_inst_rw=None, 
                 rest_wave_w=None, mask_valid_w=None, use_pyneb=False, 
                 verbose=True, log_message=[]):

        self.cframe = cframe
        self.v0_redshift = v0_redshift
        self.R_inst_rw = R_inst_rw
        self.rest_wave_w = rest_wave_w
        self.mask_valid_w = mask_valid_w
        self.use_pyneb = use_pyneb
        self.verbose = verbose
        self.log_message = log_message

        self.num_comps = len(self.cframe.info_c)
        # set default info if not specified in config
        for i_comp in range(self.num_comps):
            if ~np.isin('line_used', [*self.cframe.info_c[i_comp]]):
                self.cframe.info_c[i_comp]['line_used'] = np.array(['default'])
        for i_comp in range(self.num_comps):
            if ~np.isin('H_hi_order', [*self.cframe.info_c[i_comp]]):
                self.cframe.info_c[i_comp]['H_hi_order'] = False
        for i_comp in range(self.num_comps):
            if ~np.isin('sign', [*self.cframe.info_c[i_comp]]):
                self.cframe.info_c[i_comp]['sign'] = 'emission'
        for i_comp in range(self.num_comps):
            if ~np.isin('profile', [*self.cframe.info_c[i_comp]]):
                self.cframe.info_c[i_comp]['profile'] = 'Gaussian'

        self.HI_lv_up_max = 40 # limited by pyneb atomdata._Energy of H1, max lv_up = 40
        if np.array([self.cframe.info_c[i_comp]['H_hi_order'] for i_comp in range(self.num_comps)]).any(): 
            self.enable_H_hi_order = True
            if not self.use_pyneb: raise ValueError((f"Please enable pyneb in line config to include high order Hydrogen lines."))
        else:
            self.enable_H_hi_order = False

        self.initialize_linelist()
        if self.use_pyneb: self.initialize_pyneb()
        self.update_linelist()

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
        # Hydrogen lines can be tied to estimate extinction; all ratios (except for Lyman series) are in relative to Ha
        self.linerest_n.append(1215.670); self.lineratio_n.append(1.0)     ; self.linename_n.append('Lya')
        self.linerest_n.append(1025.722); self.lineratio_n.append(1.0)     ; self.linename_n.append('Lyb')
        self.linerest_n.append( 972.537); self.lineratio_n.append(1.0)     ; self.linename_n.append('Lyg')
        self.linerest_n.append( 949.743); self.lineratio_n.append(1.0)     ; self.linename_n.append('Lyd')
        self.linerest_n.append( 937.803); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly6')
        self.linerest_n.append( 930.748); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly7')
        self.linerest_n.append( 926.226); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly8')
        self.linerest_n.append( 923.150); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly9')
        self.linerest_n.append( 920.963); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ly10')
        self.linerest_n.append(6564.632); self.lineratio_n.append(1.0)     ; self.linename_n.append('Ha')
        self.linerest_n.append(4862.691); self.lineratio_n.append(0.349)   ; self.linename_n.append('Hb')
        self.linerest_n.append(4341.691); self.lineratio_n.append(0.164)   ; self.linename_n.append('Hg')
        self.linerest_n.append(4102.899); self.lineratio_n.append(0.0904)  ; self.linename_n.append('Hd')
        self.linerest_n.append(3971.202); self.lineratio_n.append(0.0555)  ; self.linename_n.append('H7')
        self.linerest_n.append(3890.158); self.lineratio_n.append(0.0367)  ; self.linename_n.append('H8')
        self.linerest_n.append(3836.479); self.lineratio_n.append(0.0255)  ; self.linename_n.append('H9')
        self.linerest_n.append(3798.983); self.lineratio_n.append(0.0185)  ; self.linename_n.append('H10')
        self.linerest_n.append(18756.10); self.lineratio_n.append(0.118)   ; self.linename_n.append('Paa')
        self.linerest_n.append(12821.58); self.lineratio_n.append(0.0570)  ; self.linename_n.append('Pab')
        self.linerest_n.append(10941.08); self.lineratio_n.append(0.0316)  ; self.linename_n.append('Pag')
        self.linerest_n.append(10052.12); self.lineratio_n.append(0.0194)  ; self.linename_n.append('Pad')
        self.linerest_n.append(9548.588); self.lineratio_n.append(0.0128)  ; self.linename_n.append('Pa8')
        self.linerest_n.append(9231.546); self.lineratio_n.append(0.00888) ; self.linename_n.append('Pa9')
        self.linerest_n.append(9017.384); self.lineratio_n.append(0.00644) ; self.linename_n.append('Pa10')
        self.linerest_n.append(40522.69); self.lineratio_n.append(0.0280)  ; self.linename_n.append('Bra')
        self.linerest_n.append(26258.68); self.lineratio_n.append(0.0158)  ; self.linename_n.append('Brb')
        self.linerest_n.append(21661.21); self.lineratio_n.append(0.00971) ; self.linename_n.append('Brg')
        self.linerest_n.append(19450.89); self.lineratio_n.append(0.00638) ; self.linename_n.append('Brd')
        self.linerest_n.append(18179.10); self.lineratio_n.append(0.00442) ; self.linename_n.append('Br9')
        self.linerest_n.append(17366.87); self.lineratio_n.append(0.00319) ; self.linename_n.append('Br10')
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
        self.linelist_default = np.array(['Lyb', 'O VI:1032', 'O VI:1038', 'Lya', 'N V:1239', 'N V:1243', 'Si IV:1394', 'O IV]:1397', 'O IV]:1400', 
                                          'Si IV:1403', 'C IV:1548', 'He II:1640', 'Al III:1855', 'Al III:1863', 'C III]:1909', 'Mg II]:2796', 'Mg II]:2804', 
                                          '[Ne V]:3347', '[Ne V]:3427', '[O II]:3727', '[O II]:3730', '[Ne III]:3870', 'H8', '[Ne III]:3969', 
                                          'H7', 'Hd', 'Hg', '[O III]:4364', 'Hb', '[O III]:4960', '[O III]:5008', '[N I]:5199', '[N I]:5202', 'He I:5877', 
                                          '[O I]:6302', '[O I]:6366', '[N II]:6550', 'Ha', '[N II]:6585', '[S II]:6718', '[S II]:6733', 
                                          'He I:7067', '[Ar III]:7138', '[O II]:7322', '[O II]:7333', '[Ni III]:7892', 'O I:8449', '[S III]:9071', 'Fe II:9206', '[S III]:9533', 
                                          'Pa8', 'Pad', 'He I:10833', 'Pag', 'O I:11290', '[P II]:11886', '[Fe II]:12570', 'Pab', '[Fe II]:13209', '[Si X]:14305', '[Fe II]:16440', 
                                          'Paa', 'Brd', '[Si VI]:19650', 'He I:20587', 'Brg', 'Brb'])
        self.linerest_default = self.linerest_n[np.isin(self.linename_n, self.linelist_default)] # used for continuum masking
        
    def initialize_pyneb(self):
        if self.verbose: 
            print_log('[Note] PyNeb is used in the fitting to derive line emissivities and ratios of line doublets.', self.log_message)
        import pyneb
        self.pyneb = pyneb
        self.pyneblib = {'RecAtom':{'list': pyneb.atomicData.getAllAtoms(coll=False, rec=True)}, 
                         'Atom':{'list': pyneb.atomicData.getAllAtoms(coll=True, rec=False)}} 
        # save to avoid duplicate reading

        # these lines cannot match with pyneb lib
        linelist_mismatch = ['C II:7238', 'N I:7470', 'N I:8683', 'N I:8706', 'N I:8714', 'N III:4512', 
                             'O I:1302', 'O I:6048', 'O I:7004', 'O I:7256', 'O I:8449', 'O I:11290', 'O II:4318', 'O II:4416', 'O III:3134', 'O III:3313', 'O III:3445', 
                             'Si II:1260', 'Si II:1265', 'Si II:6349']
        # refresh line info with pyneb
        for i_line in range(len(self.linename_n)):
            if np.isin(self.linename_n[i_line], linelist_mismatch): continue
            linename, linerest = self.search_pyneb(self.linename_n[i_line], verbose=False)
            if linename is not None: 
                self.linename_n[i_line] = linename
                self.linerest_n[i_line] = linerest

        if self.enable_H_hi_order: 
            self.linelist_H_hi_order = np.array([n+u for n in ['Ly','H','Pa','Br','Pf'] if n+'10' in self.linelist_full for u in [str(i+11) for i in range(self.HI_lv_up_max-10)]]) 
            # lv_up from 11 to 40
            self.linelist_full = np.hstack((self.linelist_full, self.linelist_H_hi_order))
            for linename in self.linelist_H_hi_order:
                linename, linerest = self.search_pyneb(linename)
                self.linename_n = np.hstack((self.linename_n, linename))
                self.linerest_n = np.hstack((self.linerest_n, linerest))
                self.lineratio_n = np.hstack((self.lineratio_n, 1.0))
        
    def add_line(self, linenames=None, linerests=None, lineratios=None, force=False, use_pyneb=False):
        if not isinstance(linenames, list): linenames = [linenames]
        for (i_line, linename) in enumerate(linenames):
            if use_pyneb:
                linename, linerest = self.search_pyneb(linename)
                lineratio = 1.0
            else:
                linerest = linerests[i_line]
                lineratio = lineratios[i_line] if lineratios is not None else 1.0
            if np.isin(linename, self.linename_n): raise ValueError((f"{linename} is already in the line list: {self.linename_n}."))
            i_close = np.absolute(self.linerest_n - linerest).argmin()
            if (np.abs(self.linerest_n[i_close] - linerest) > 1) | force: 
                self.linename_n  = np.hstack((self.linename_n, linename))
                self.linerest_n  = np.hstack((self.linerest_n, linerest))
                self.lineratio_n = np.hstack((self.lineratio_n, lineratio))
                self.linelist_full = np.hstack((self.linelist_full, linename))
                self.linelist_default = np.hstack((self.linelist_default, linename))
                print_log(f"{linename, linerest} with linkratio={lineratio} is added into the line list.", self.log_message)
            else:
                print_log(f"There is a line {self.linename_n[i_close], self.linerest_n[i_close]} close to the input one {linename, linerest}"
                         +f", set force=True to add this line.", self.log_message)
        self.update_linelist()

    def delete_line(self, linenames=None):
        if not isinstance(linenames, list): linenames = [linenames]
        mask_remain_n = np.ones_like(self.linename_n, dtype='bool')
        for linename in linenames:
            mask_remain_n = self.linename_n != linename
            self.linename_n  = self.linename_n[mask_remain_n]
            self.linerest_n  = self.linerest_n[mask_remain_n]
            self.lineratio_n = self.lineratio_n[mask_remain_n]
            self.linelist_full = self.linelist_full[self.linelist_full != linename]
            self.linelist_default = self.linelist_default[self.linelist_default != linename]
        self.update_linelist()

    def update_linelist(self):
        # update self.linename_n, self.linelist_full, and self.linelist_default and the other corresponding linelists

        # only keep covered lines
        mask_valid_n  = self.linerest_n > (self.rest_wave_w.min()-50)
        mask_valid_n &= self.linerest_n < (self.rest_wave_w.max()+50)
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

        linelist_H = np.array([n+u for n in ['Ly','H','Pa','Br','Pf'] for u in ['a','b','g','d'] + [str(i+1) for i in range(self.HI_lv_up_max)]])
        linelist_H = linelist_H[np.isin(linelist_H, self.linelist_full)]
        linelist_nonH = self.linelist_full[~np.isin(self.linelist_full, linelist_H)]

        spectra_full = np.array([line.split(':')[0] for line in self.linelist_full])
        spectra_full = spectra_full[~np.isin(spectra_full, linelist_H)]
        spectra_uniq = []
        for spectrum in spectra_full:
            if ~np.isin(spectrum, spectra_uniq): spectra_uniq.append(str(spectrum))
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
            if ~np.isin(notation, notations_uniq): notations_uniq.append(str(notation))
        notations_uniq = np.array(notations_uniq)
        self.linelist_notations = {}
        self.linelist_notations['H I'] = linelist_H
        for notation in notations_uniq: self.linelist_notations[str(notation)] = linelist_nonH[notations_full == notation]

        elements_full = copy(notations_full)
        for i in range(len(elements_full)):
            elements_full[i] = elements_full[i].split(' ')[0]
        elements_uniq = []
        for element in elements_full:
            if ~np.isin(element, elements_uniq): elements_uniq.append(str(element))
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
                if np.isin(self.cframe.info_c[i_comp]['line_used'], ['default', 'NLR', 'AGN_NLR', 'HII', 'outflow']).any():
                    self.cframe.info_c[i_comp]['linelist'] = np.hstack((self.cframe.info_c[i_comp]['linelist'],  self.linelist_default))
                if np.isin(self.cframe.info_c[i_comp]['line_used'], ['BLR', 'AGN_BLR']).any():
                    self.cframe.info_c[i_comp]['linelist'] = np.hstack((self.cframe.info_c[i_comp]['linelist'],  
                                                                        self.linelist_allowed[np.isin(self.linelist_allowed, self.linelist_default)],
                                                                        self.linelist_intercombination[np.isin(self.linelist_intercombination, self.linelist_default)],
                                                                        self.linelist_elements['H']))
                if np.isin(self.cframe.info_c[i_comp]['line_used'], ['allowed', 'permitted']).any():
                    self.cframe.info_c[i_comp]['linelist'] = np.hstack((self.cframe.info_c[i_comp]['linelist'],  self.linelist_allowed))
                if np.isin(self.cframe.info_c[i_comp]['line_used'], ['intercombination', 'semiforbidden', 'semi-forbidden', 'semipermitted', 'semi-permitted']).any():
                    self.cframe.info_c[i_comp]['linelist'] = np.hstack((self.cframe.info_c[i_comp]['linelist'],  self.linelist_intercombination))
                if np.isin(self.cframe.info_c[i_comp]['line_used'], ['forbidden']).any():
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
            self.mask_valid_cn[i_comp] &= np.isin(self.linename_n, self.cframe.info_c[i_comp]['linelist'])

        # check the relations between lines
        self.initialize_linelink()
            
    def search_pyneb(self, name, ret_atomdata=False, verbose=True):
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

        HI_lv_up_max = 40 # limited by atomdata._Energy of H1, max lv_up = 40
        HI_lv_low_dict = {}
        for u in ['a','b','g','d'] + [str(i+1) for i in range(HI_lv_up_max)]: 
            HI_lv_low_dict['Ly'+u] = 1 # Lyman
            HI_lv_low_dict['H' +u] = 2 # Balmer
            HI_lv_low_dict['Pa'+u] = 3 # Paschen
            HI_lv_low_dict['Br'+u] = 4 # Brackett
            HI_lv_low_dict['Pf'+u] = 5 # Pfund

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
                    if verbose: print(f"{name} not provided in pyneb, please add it manually following the github/advanced_usage page.")
                    if ret_atomdata: 
                        return None, None, None
                    else:
                        return None, None
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
            if np.isin(ref_name, self.linename_n):
                i_ref = np.where(self.linename_n == ref_name)[0][0]
                ref_valid = self.mask_valid_cn[:, i_ref].any() # if ref_name exists in any one comp
                if ref_valid: break # pick up the 1st valid ref_name

        if not isinstance(tied_names, list): tied_names = [tied_names]
        tied_names_valid = []
        for tied_name in tied_names:
            if np.isin(tied_name, self.linename_n) & (tied_name != ref_name):
                i_tied = np.where(self.linename_n == tied_name)[0][0]
                tied_valid = self.mask_valid_cn[:, i_tied].any() # if tied_name exists in any one comp
                if ref_valid & tied_valid: 
                    self.linelink_n[i_tied] = ref_name
                    tied_names_valid.append(tied_name)
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
            if len(tied_names_valid)  > 1: print_log(f"    {tied_names_valid} --> {ref_name}", self.log_message)
            if len(tied_names_valid) == 1: print_log(f"    {tied_names_valid[0]} --> {ref_name}", self.log_message)
                
    def release_pair(self, tied_names):
        if not isinstance(tied_names, list): tied_names = [tied_names]
        for tied_name in tied_names:
            if ~np.isin(tied_name, self.linename_n):
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

        H_linenames = [n+u for n in ['H','Pa','Br','Pf'] for u in ['a','b','g','d'] + [str(i+1) for i in range(self.HI_lv_up_max)]] # do not set 'Ly' due to Lya forest
        self.tie_pair(H_linenames, ['Ha','Hb','Hg','Hd', 'Paa','Pab','Pag','Pad']) # use set alternative line is Ha is not covered

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
                          f"    {self.linename_n[self.mask_free_cn[i_comp]]}", self.log_message)

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

        # if model.sum() <= 0: model[0] = 1e-10 # avoid error from full zero output in case line is not covered
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

    def mask_line_lite(self, enabled_comps=None, disabled_comps=None):
        if enabled_comps is not None:
            self.enabled_e = np.zeros((self.num_coeffs), dtype='bool')
            for comp in enabled_comps: self.enabled_e[self.component_e == comp] = True
        else:
            self.enabled_e = np.ones((self.num_coeffs), dtype='bool')
            if disabled_comps is not None:
                for comp in disabled_comps: self.enabled_e[self.component_e == comp] = False
        return self.enabled_e

    ##########################################################################
    ########################## Output functions ##############################

    def extract_results(self, ff=None, step=None, print_results=True, return_results=False, show_average=False):
        if (step is None) | (step == 'best') | (step == 'final'):
            step = 'joint_fit_3' if ff.have_phot else 'joint_fit_2'
        if (step == 'spec+SED'):  step = 'joint_fit_3'
        if (step == 'spec') | (step == 'pure-spec'): step = 'joint_fit_2'
        
        best_chi_sq_l = copy(ff.output_s[step]['chi_sq_l'])
        best_par_lp   = copy(ff.output_s[step]['par_lp'])
        best_coeff_le = copy(ff.output_s[step]['coeff_le'])

        mod = 'line'
        fp0, fp1, fe0, fe1 = ff.search_model_index(mod, ff.full_model_type)
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
        print_log(f'[Note] Rows starting with a line name show the observed line flux, in unit of {self.spec_flux_scale:.0e} erg/s/cm2.', log)
