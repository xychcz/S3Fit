# Basic usage

## Initialize the main function
```python
FF = FitFrame(spec_wave_w=None, spec_flux_w=None, spec_ferr_w=None, 
              spec_valid_range=None, spec_R_inst=None, spec_flux_scale=None, 
              phot_name_b=None, phot_flux_b=None, phot_ferr_b=None, phot_trans_dir=None, phot_fluxunit='mJy', 
              sed_wave_w=None, sed_waveunit='angstrom', 
              v0_redshift=None, model_config=None, 
              num_mock_loops=0, fitraw=True, 
              plot_step=False, print_step=True, verbose=False)
```
#### Spectral data
`spec_wave_w`: Wavelength of the input spectrum, in unit of angstrom.\
`spec_flux_w` and `spec_ferr_w`: Fluxes and measurement errors of the input spectrum, in unit of erg s<sup>-1</sup> cm<sup>-2</sup> angstrom<sup>-1</sup>.\
`spec_valid_range`: Valid wavelength range. For example, if 5000--7000 and 7500--10000 angstrom are used in fitting, set `spec_valid_range=[[5000,7000], [7500,10000]]`.\
`spec_R_inst`: Instrumental spectral resolution of the input spectrum, this is used to convolve the model spectra and estimate the intrinsic velocity width.\
`spec_flux_scale`: Scaling ratio of the flux. The fitting is performed for spec_flux_w/spec_flux_scale to avoid too small value. 
#### Photometric data
`phot_name_b`: List of band names of the input photometric data, e.g., `phot_name_b=['SDSS_gp','2MASS_J','WISE_1']`. The names should be the same as the filenames of the transmission curves in each band. \
`phot_trans_dir`: Directory of the transmission curves. \
`phot_flux_b` and `phot_ferr_b`: Fluxes and measurement errors in each band. \
`phot_fluxunit`: Flux unit of `phot_flux_b` and `phot_ferr_b`, can be `'mJy'` (default) and `'erg/s/cm2/AA'`. If the input data is in unit of 'mJy', they will be converted to 'erg/s/cm2/AA' before fitting. \
`sed_wave_w` and `sed_waveunit`: Wavelength array and its unit of the full SED wavelength range, which are used to create the model spectra and convert them to fluxes in each band. `sed_waveunit` can be `'angstrom'` and `'micron'`; if set to 'micron', they will be converted to 'angstrom'. 
Note that `sed_wave_w` is not mandatory; if it is not set, the code can create the wavelength array to cover all of the transmission curves of the input bands.
#### Model setup 
[model setup](#model-setup)

#### Fitting loop setup
`num_mock_loops=0` to only fit the raw data, i.e., no mocked spectra

#### Auxiliary
`plot_step=True` to plot each fitting step

```python
import sys
sys.path.append(os.path.abspath("PATH_OF_THIS_CODE"))
from s3fit.py import FitFrame

# Spectral data
spec_wave_w, spec_flux_w, spec_ferr_w =
# Wavelength, flux and error of input spectrum, unit in angstorm and erg/s/cm2/angstorm
spec_valid_range = [[5000,7000], [7500,10000]]
# Valid wavelength range. In the example, 5000-7000 and 7500-10000 angstorm are used in fitting.
spec_R_inst = 1000
# Spectral resolution of the data
spec_flux_scale = 1e-15
# Scaling ratio to avoid too small flux values, 

# Photometric data
phot_name_b = ['SDSS_up', 'SDSS_gp', '2MASS_J', 'WISE_1', 'WISE_2']
# Band names, which are the same of the transmission curve files.
phot_flux_b, phot_ferr_b =
# Flux and error in each band, unit in mJy
phot_trans_dir = './filter/'
# directory of transmission curve files.

# Models
# Stellar models from PopSTAR library
ssp_pmmc = [[[0, -1000, 1000], [600, 100, 1200], [0.5, 0, 5.0], '5.5e-3, None, solar_met']]
# Initial guesses and ranges of parameters of voff, fwhm, and AV.
# For example, [0, -1000, 1000] denotes the initial value of velocity shift is 0, and is allowed to vary from -1000 to 1000 km/s.
# The string at the end, '5.5e-3, None, solar_met', denotes the minimum (5,5e-3 Gyr) and maximum (None if set to Universe age) stellar ages,
# and metallicity (currently 'solar_met' or 'all' is supported)
ssp_file =
# location of the PopSTAR library

# Emission line models
el_pmmc = [[[    0, -500,  500], [ 500,250, 750], [3.0,0,5], [1.2,0.5,1.45], 'NLR:all'], 
           [[ -500,-2000,  100], [1000,750,2500], [1.5,0,5], [1.2,0.5,1.45], 'outflow_1:all'], 
           [[-2500,-3000,-2000], [1000,750,2500], [1.5,0,5], [1.2,0.5,1.45], 'outflow_2:[OIII]a,[OIII]b,[NII]a,Ha,[NII]b'],
           [[    0, -500,  500], [3000,750,9500], [1.5,0,5], [1.2,0.5,1.45], 'BLR:Ha']]
# Each line have the same paramerters.
# Support parameters are voff, fwhm, AV, and [SII]6716/6731 ratio (for electron density of 1e4--1cm-3)
# 'NLR:all' denotes for Narrow-Line-Region, all emission lines are used.
# 'outflow_2:[OIII]a,[OIII]b,[NII]a,Ha,[NII]b' denotes a secondary outflow component, for which only [OIII] doublets and Ha-[NII] complex are used.
# 'BLR:Ha' denotes for AGN Broad-Line-Region, only Ha (Halpha) is used.
# Edit ELineModels.__init__() to add more lines.

# AGN continuum models, currently only bended powerlaw is supported
agn_pmmc = [[[0, -1000, 1000], [600, 100, 1200], [3.0, 1.5, 10.0], [-1.7, -1.7, -1.7], 'powerlaw']]
# Initial guesses and ranges of parameters of voff, fwhm, AV, and powerlaw index from 0.1 to 5 micron.

# SKIRTor torus models
torus_pmmc = [[[0, -1000, 1000], [5, 3, 11], [30, 10, 80], [20, 10, 30], [50, 0, 90], 'disc+dust']] #'disc+dust'
# Support parameters are voff, tau, half opening angle of torus, radii ratio, inclination angle
torus_file = 
# location of model library

# initialize 
FF = FitFrame(spec_wave_w=spec_wave_w, spec_flux_w=spec_flux_w, spec_ferr_w=spec_ferr_w, 
              spec_valid_range=spec_valid_range, spec_R_inst=spec_R_inst, spec_flux_scale=spec_flux_scale,
              # input of spectral data 
              phot_name_b=phot_name_b, phot_flux_b=phot_flux_b, phot_ferr_b=phot_ferr_b, phot_trans_dir=phot_trans_dir,
              # input of photometric data, comment this line if pure-spectral fitting is required
              v0_redshift=v0_redshift,
              # initial guess of systemic redshift
              ssp_pmmc=ssp_pmmc, ssp_file=ssp_file, 
              agn_pmmc=agn_pmmc, 
              el_pmmc=el_pmmc, 
              torus_pmmc=torus_pmmc, torus_file=torus_file, 
              # setup of models, comment the corresponding line if a model is not required
              num_mock_loops=10,
              # number of fitting loops for mocked data to estimate uncertainties of parameters; set to 1 if only raw data is fit
              plot=True, verbose=False
              # if showing plots and texts of each steps of the fit
)

# run fit
FF.main_fit()

```

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

