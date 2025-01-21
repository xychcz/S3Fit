class ELineModels(object):
    def __init__(self, comps, v0_redshift=0, spec_R_inst=1e8):
        self.comps = comps
        self.v0_redshift = v0_redshift
        self.spec_R_inst = spec_R_inst
        
        self.line_rest_n, self.linked_to_n, self.linked_ratio_n, self.line_name_n = [],[],[],[]
        # _n denote line; _l already used for loop id
        # [v2] vacuum wavelength calculated with lamb_air_to_vac and air values in 
        # http://astronomy.nmsu.edu/drewski/tableofemissionlines.html
        # also check https://classic.sdss.org/dr6/algorithms/linestable.php
        # [v3] update with https://physics.nist.gov/PhysRefData/ASD/lines_form.html (H ref: T8637)
        # ratio of doublets calculated with pyneb under ne=100, Te=1e4
        self.line_rest_n.append(6564.632); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('Ha')
        self.line_rest_n.append(4862.691); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.349)  ; self.line_name_n.append('Hb')
        self.line_rest_n.append(4341.691); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.164)  ; self.line_name_n.append('Hg')
        self.line_rest_n.append(4102.899); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.0904) ; self.line_name_n.append('Hd')
        self.line_rest_n.append(3971.202); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.0555) ; self.line_name_n.append('H7')
        self.line_rest_n.append(3890.158); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.0367) ; self.line_name_n.append('H8')
        self.line_rest_n.append(3836.479); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.0255) ; self.line_name_n.append('H9')
        self.line_rest_n.append(3798.983); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.0185) ; self.line_name_n.append('H10')
        self.line_rest_n.append(3771.708); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.0139) ; self.line_name_n.append('H11')
        self.line_rest_n.append(3751.224); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.0107) ; self.line_name_n.append('H12')
        self.line_rest_n.append(3735.437); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.00838); self.line_name_n.append('H13')
        self.line_rest_n.append(3723.004); self.linked_to_n.append(6564.632); self.linked_ratio_n.append(0.00671); self.line_name_n.append('H14')
        self.line_rest_n.append(5877.249); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('HeI')

        self.line_rest_n.append(6732.674); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[SII]b')
        self.line_rest_n.append(6718.295); self.linked_to_n.append(6732.674); self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[SII]a')
        self.line_rest_n.append(6585.270); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[NII]b')
        self.line_rest_n.append(6549.860); self.linked_to_n.append(6585.270); self.linked_ratio_n.append(0.340)  ; self.line_name_n.append('[NII]a')
        self.line_rest_n.append(6365.536); self.linked_to_n.append(6302.046); self.linked_ratio_n.append(0.319)  ; self.line_name_n.append('[OI]b')
        self.line_rest_n.append(6302.046); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[OI]a')
        self.line_rest_n.append(5201.705); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[NI]b')
        self.line_rest_n.append(5199.349); self.linked_to_n.append(5201.705); self.linked_ratio_n.append(0.783)  ; self.line_name_n.append('[NI]a')
        self.line_rest_n.append(5099.230); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[FeVI]')
        self.line_rest_n.append(5008.240); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[OIII]b')
        self.line_rest_n.append(4960.295); self.linked_to_n.append(5008.240); self.linked_ratio_n.append(0.335)  ; self.line_name_n.append('[OIII]a')
        self.line_rest_n.append(3968.590); self.linked_to_n.append(3869.860); self.linked_ratio_n.append(0.301)  ; self.line_name_n.append('[NeIII]b')
        self.line_rest_n.append(3869.860); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[NeIII]a')
        self.line_rest_n.append(3729.875); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[OII]b')
        self.line_rest_n.append(3727.092); self.linked_to_n.append(3729.875); self.linked_ratio_n.append(0.741)  ; self.line_name_n.append('[OII]a')
        self.line_rest_n.append(3426.864); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[NeV]b')
        self.line_rest_n.append(3346.783); self.linked_to_n.append(3426.864); self.linked_ratio_n.append(0.366)  ; self.line_name_n.append('[NeV]a')
        self.line_rest_n = np.array(self.line_rest_n)
        self.linked_to_n = np.array(self.linked_to_n)
        self.linked_ratio_n = np.array(self.linked_ratio_n)
        self.line_name_n = np.array(self.line_name_n)
        
        # recalculate Hydrogen ratios for BLR condition with ne=, Te=
        # add later
        # self.linked_ratio_BLR_n = self.linked_ratio_n.copy() 
        # self.linked_ratio_BLR_n[self.line_name_n=='Hb'] = 

        self.num_comps = len(comps)
        self.num_lines = len(self.line_rest_n)
        self.mask_valid_cn = np.zeros((self.num_comps, self.num_lines), dtype='bool')
        self.mask_free_cn  = np.zeros((self.num_comps, self.num_lines), dtype='bool')
        for i_kin in range(self.num_comps):
            mask_valid_n = self.line_rest_n > 0 # fit all lines
            mask_free_n  = self.linked_to_n == -1
            if comps[i_kin].split(':')[1] != 'all': 
                mask_select_n = np.isin(self.line_name_n, comps[i_kin].split(':')[1].split(',')) 
                mask_valid_n &= mask_select_n
                mask_free_n  &= mask_select_n
            self.mask_valid_cn[i_kin, mask_valid_n] = True
            self.mask_free_cn[i_kin,  mask_free_n ] = True
            print('Emission line complex', i_kin, ', total number:', mask_valid_n.sum(), ', free lines:', self.line_name_n[mask_free_n])
        self.num_coeffs = self.mask_free_cn.sum()
        
        # set component name and enable mask for each free line; _f for free or coeffs
        self.component_f = [] # np.zeros((self.num_coeffs), dtype='<U16')
        for i_kin in range(self.num_comps):
            for i_line in range(self.num_lines):
                if self.mask_free_cn[i_kin, i_line]:
                    self.component_f.append(comps[i_kin].split(':')[0])
        self.component_f = np.array(self.component_f)

        mask_Balmer = self.linked_to_n == 6564.632
        self.extHa_base_n = np.zeros_like(self.linked_ratio_n)
        tmp = Calzetti00_ExtLaw(self.line_rest_n[mask_Balmer], RV=4.05) - Calzetti00_ExtLaw(np.array([6564.632]), RV=4.05)
        self.extHa_base_n[mask_Balmer] = 10.0**(-0.4 * tmp) # the base of extinction (ref to Halpha)
        # repeat for self.extHa_base_BLR_n
        
    def mask_el_lite(self, enabled_comps='all'):
        self.enabled_f = np.zeros((self.num_coeffs), dtype='bool')
        if enabled_comps == 'all':
            self.enabled_f[:] = True
        else:
            for comp in enabled_comps:
                self.enabled_f[self.component_f == comp] = True
        return self.enabled_f
    
    def models_mkin_unitflux(self, wavelength, kins_pars, pf=None):
        if pf is not None: kins_pars = pf.flat_to_arr(kins_pars)
        for i_kin in range(self.num_comps):
            voff, fwhm, el_AV, SII_ratio = kins_pars[i_kin]
            list_valid = np.arange(len(self.line_rest_n))[self.mask_valid_cn[i_kin,:]]
            list_free  = np.arange(len(self.line_rest_n))[self.mask_free_cn[i_kin,:]]
            models_skin = self.models_skin_unitflux(wavelength, voff, fwhm, el_AV, SII_ratio, list_valid, list_free)
            if i_kin == 0: 
                models_mkin = models_skin
            else:
                models_mkin += models_skin
        return np.array(models_mkin)
    
    def models_skin_unitflux(self, wavelength, voff, fwhm, el_AV, SII_ratio, list_valid, list_free):
        # update linked_ratio_n
        mask_Balmer = self.linked_to_n == 6564.632
        new_linked_ratio_n = self.linked_ratio_n.copy() 
        new_linked_ratio_n[mask_Balmer] *= self.extHa_base_n[mask_Balmer] ** el_AV
        # repeat for BLR:
        # new_linked_ratio_BLR_n = self.linked_ratio_BLR_n.copy() 
        # new_linked_ratio_BLR_n[mask_Balmer] *= self.extHa_base_BLR_n[mask_Balmer] ** el_AV
        i_SIIa = np.where(self.line_rest_n == 6718.295)[0][0]
        new_linked_ratio_n[i_SIIa] = SII_ratio # SII_ratio = SIIa/SIIb
        
        models_skin = []
        for i_free in list_free:
            model_scomp = self.single_gaussian(wavelength, self.line_rest_n[i_free], voff, fwhm, 
                                               1, self.v0_redshift, self.spec_R_inst) # flux=1
            list_linked = np.where(self.linked_to_n == self.line_rest_n[i_free])[0]
            list_linked = list_linked[np.isin(list_linked, list_valid)]
            for i_linked in list_linked:
                # determine new_linked_ratio_n or _BLR:
                # add later
                model_scomp += self.single_gaussian(wavelength, self.line_rest_n[i_linked], voff, fwhm, 
                                                    new_linked_ratio_n[i_linked], self.v0_redshift, self.spec_R_inst)
            models_skin.append(model_scomp)
        return models_skin

    def single_gaussian(self, wavelength, lamb_c_rest, voff, fwhm, flux, v0_redshift=0, spec_R_inst=1e8):
        if fwhm <= 0: raise ValueError((f"Non-positive eline fwhm: {fwhm}"))
        if flux < 0: raise ValueError((f"Negative eline flux: {flux}"))
        lamb_c_obs = lamb_c_rest * (1 + v0_redshift)
        mu =    (1 + voff/299792.458) * lamb_c_obs
        sigma_line = fwhm/299792.458  * lamb_c_obs / np.sqrt(np.log(256))
        sigma_inst = 1/spec_R_inst * lamb_c_obs / np.sqrt(np.log(256))
        sigma = np.sqrt(sigma_line**2 + sigma_inst**2)
        model = np.exp(-0.5 * ((wavelength-mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi)) 
        dw = (wavelength[1:]-wavelength[:-1]).min()
        if (model * dw).sum() < 0.10: flux = 0 # disable not well covered emission line
        return model * flux
