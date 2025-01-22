# S<sup>3</sup>Fit
**S<sup>3</sup>Fit**: a **S**imultaneous **S**pectrum and photometric-**S**ED **Fitting** code for extragalaxy

**Install**

**Usage**
```python
# fitting models
ssp_file = '/lwk/xychen/AKARI_ULIRG/GMOS/ifufit_code/ssp/popstar21_stellar_nebular_fullwave.fits'
ssp_pmmc = [[[0, -1000, 1000], [600, 100, 1200], [0.5, 0, 5.0], '5.5e-3, None, solar_met']]
# ssp pars: voff, fwhm, AV; population set
agn_pmmc = [[[0, -1000, 1000], [600, 100, 1200], [3.0, 1.5, 10.0], [-1.7, -1.7, -1.7], 'powerlaw']]
# agn pars: voff, fwhm, AV; alpha_lambda (pl); voff, fwhm not used if only powerlaw
el_pmmc = [[[    0, -500,  500], [ 500,250, 750], [3.0,0,5], [1.2,0.5,1.45], 'NLR:all'], 
           [[ -500,-2000,  100], [1000,750,2500], [1.5,0,5], [1.2,0.5,1.45], 'outflow_1:all'], 
           [[-2500,-3000,-2000], [1000,750,2500], [1.5,0,5], [1.2,0.5,1.45], 'outflow_2:[OIII]a,[OIII]b,[NII]a,Ha,[NII]b'],
           [[    0, -500,  500], [3000,750,9500], [1.5,0,5], [1.2,0.5,1.45], 'BLR:Ha']]
# el pars: voff, fwhm, AV, SIIa/b (n_e: 1e4--1cm-3); 3 kinematic system
torus_disc_file = '../models/skirtor_disc_allincl_p1_nosca_q5.dat'
torus_dust_file = '../models/skirtor_torus_allincl_p1_nosca_q5.dat'
torus_pmmc = [[[0, -1000, 1000], [5, 3, 11], [30, 10, 80], [20, 10, 30], [50, 0, 90], 'disc+dust']] #'disc+dust'
# torus pars: voff, tau, opening angle, radii ratio, inclination angle

mask_b = np.isin(band_name_b, ['SDSS_up', 'SDSS_gp', 'SDSS_rp', 'SDSS_ip', 'SDSS_zp', 
                               '2MASS_J', '2MASS_H', '2MASS_Ks', 'WISE_1', 'WISE_2', 'WISE_3', 'WISE_4', 
                               'Spitzer_IRAC_1', 'Spitzer_IRAC_2', 'Spitzer_IRAC_3', 'Spitzer_IRAC_4', 'Spitzer_MIPS_1'])

FF_s3_n1_disctorus = FitFrame(spec_wave_w=copy(vac_wave_w), spec_flux_w=copy(intspec_flux_w), spec_ferr_w=copy(intspec_ferr_w), 
                              spec_valid_range=valid_wave_range(-1), spec_R_inst=spec_R_inst, spec_flux_scale=flux_scale, 
                              phot_name_b=band_name_b[mask_b], phot_flux_b=band_flux_b[mask_b], 
                              phot_ferr_b=(band_ferr_b - band_flux_b*0.10)[mask_b], # remove 10%flux from input
                              phot_trans_dir='../filters/',
                              v0_redshift=v0_redshift, 
                              ssp_pmmc=ssp_pmmc, ssp_file=ssp_file, 
                              agn_pmmc=agn_pmmc, 
                              el_pmmc=el_pmmc, 
                              torus_pmmc=torus_pmmc, torus_disc_file=torus_disc_file, torus_dust_file=torus_dust_file, 
                              num_mock_loops=1, fitraw=True, plot=1, verbose=False)
```
