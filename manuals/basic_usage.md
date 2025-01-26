# Basic usage

## Initialize the main function
```python
FF = FitFrame(spec_wave_w=spec_wave_w, spec_flux_w=spec_flux_w, spec_ferr_w=spec_ferr_w, 
              spec_valid_range=spec_valid_range, spec_R_inst=spec_R_inst, spec_flux_scale=spec_flux_scale, 
              phot_name_b=phot_name_b, phot_flux_b=phot_flux_b, phot_ferr_b=phot_ferr_b, phot_trans_dir=phot_trans_dir,
              v0_redshift=v0_redshift, model_config=model_config,
              num_mock_loops=0, plot_step=True, print_step=True, verbose=False)
```
[model setup](#model-setup)
`num_mock_loops=0` to only fit the raw data, i.e., no mocked spectra
`plot_step=True` to plot each fitting step

## Input data
#### Spectral data

#### Photometric SED data

## Model setup
```python
# ssp_config = {'main': {'pars': [[-1000, 1000, 'free'], [100, 1200, 'free'], [0, 5.0, 'free'], [-1, -1, 'fix'], [-1, -1, 'fix']], 
#                        'info': {'age_min': 5.5e-3, 'age_max': 'universe', 'met_sel': 'solar', 'sfh': 'nonparametric'} } }
ssp_config = {'main': {'pars': [[-1000, 1000, 'free'], [100, 1200, 'free'], [0, 5.0, 'free'], 
                                [0, 0.94, 'free'], [-1, 1, 'free']], 
                       'info': {'age_min': -2.25, 'age_max': 'universe', 'met_sel': 'solar', 'sfh': 'exponential'} }, 
              'young': {'pars': [[None, None, 'ssp:main:0'], [None, None, 'ssp:main:1'], [None, None, 'ssp:main:2'], 
                                 [-2, -1, 'free'], [-1, -1, 'fix']], 
                        'info': {'age_min': -2.25, 'age_max': 0, 'met_sel': 'solar', 'sfh': 'constant'} } }
# pars: voff, fwhm, AV, log csp_age (Gyr), log sfh_tau (Gyr)
# age_min, age_max: min and max log ssp_age (Gyr)
# met_sel: 'all', 'solar', or any combination of [0.004,0.008,0.02,0.05]
# sfh: 'nonparametric', 'exponential', 'delayed', 'constant'
ssp_file = '/lwk/xychen/AKARI_ULIRG/GMOS/ifufit_code/ssp/popstar21_stellar_nebular_fullwave.fits'
# please use ../models/convert_popstar_ssp.py to create the ssp template library

el_config = {'NLR': {'pars':       [[ -500,   500, 'free'], [250,  750, 'free'], [0, 5, 'free'], [0.5, 1.45, 'free']], 
                     'info': {'line_used': ['all']}}, 
             'outflow_1': {'pars': [[-2000,   100, 'free'], [750, 2500, 'free'], [0, 5, 'free'], [0.5, 1.45, 'free']], 
                           'info': {'line_used': ['all']}}, 
             'outflow_2': {'pars': [[-3000, -2000, 'free'], [750, 2500, 'free'], [0, 5, 'free'], [0.5, 1.45, 'free']], 
                           'info': {'line_used': ['[OIII]a', '[OIII]b', '[NII]a', 'Ha', '[NII]b']} } }
# pars: voff, fwhm, AV, SIIa/b (n_e: 1e4--1cm-3); 3 kinematic system

agn_config = {'main': {'pars': [[None, None, 'el:NLR:0;ssp:main:0'], [None, None, 'el:NLR:1;ssp:main:1'], [1.5, 10.0, 'free'], [-1.7, None, 'fix']],
                       'info': {'mod_used': ['powerlaw']} } }
# pars: voff, fwhm, AV; alpha_lambda of powerlaw at 

torus_config = {'main': {'pars': [[None, None, 'el:NLR:0;ssp:main:0'], [3, 11, 'free'], [10, 80, 'free'], [10, 30, 'free'], [0, 90, 'free']],
                         'info': {'mod_used': ['dust']} } } # 
# pars: voff, tau, opening angle, radii ratio, inclination angle
# set 'mod_used' to ['disc', 'dust'] if use both of disc and dusty torus modules
torus_file = '../models/skirtor_torus.fits'

model_config = {'ssp': {'enable': True, 'config': ssp_config, 'file': ssp_file}, 
                'el': {'enable': True, 'config': el_config}, 
#                 'agn': {'enable': True, 'config': agn_config}, # do not use agn powerlaw since the example is type-2
                'torus': {'enable': True, 'config': torus_config, 'file': torus_file}}
```

#### Stellar component

#### Emission lines

#### AGN central continuum

#### AGN dusty torus

