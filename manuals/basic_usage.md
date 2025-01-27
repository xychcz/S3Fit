# Basic usage

## Initialize the main function
```python
import sys
sys.path.append(os.path.abspath("PATH_OF_THIS_CODE"))
from s3fit.py import FitFrame
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
`spec_flux_scale`: Scaling ratio of the flux (e.g., `1e-15`). The fitting is performed for spec_flux_w/spec_flux_scale to avoid too small value. 
#### Photometric data
`phot_name_b`: List of band names of the input photometric data, e.g., `phot_name_b=['SDSS_gp','2MASS_J','WISE_1']`. The names should be the same as the filenames of the transmission curves in each band. \
`phot_trans_dir`: Directory of the transmission curves. \
`phot_flux_b` and `phot_ferr_b`: Fluxes and measurement errors in each band. \
`phot_fluxunit`: Flux unit of `phot_flux_b` and `phot_ferr_b`, can be `'mJy'` (default) and `'erg/s/cm2/AA'`. If the input data is in unit of 'mJy', they will be converted to 'erg/s/cm2/AA' before fitting. \
`sed_wave_w` and `sed_waveunit`: Wavelength array and its unit of the full SED wavelength range, which are used to create the model spectra and convert them to fluxes in each band. `sed_waveunit` can be `'angstrom'` and `'micron'`; if set to 'micron', they will be converted to 'angstrom'. 
Note that `sed_wave_w` is not mandatory; if it is not set, the code can create the wavelength array to cover all of the transmission curves of the input bands.
#### Model setup 
`v0_redshift`: Initial guess of the systemic redshift. The velocity shifts of all models are in relative to the input `v0_redshift`. 
`model_config`: Dictionary of model configurations, see [model setup](#model-setup) section for details. 
#### Fitting loop setup
`num_mock_loops`: Number of the mocked spectra, which is used to estimate the uncertainty of best-fit results. Default is `0`, i.e., only fit the raw data. 
`fitraw`: Whether or not to fit the raw data. Default is `True`. If set to `False`, the code only output results for the mocked spectra. 
#### Auxiliary
`plot_step`: Whether or not to plot the best-fit model spectra and fitting residuals in each intermediate step. Default is `False`. 
`print_step`: Whether or not to print the information each intermediate step (e.g., the examination of each model component). Default is `True`. 
`verbose`: Whether or not to print the running information of least-square solvers. Default is `False`. 


## Model setup
```python
model_config = {'ssp': {'enable': True, 'config': ssp_config, 'file': ssp_file}, 
                'el': {'enable': True, 'config': el_config},
                'agn': {'enable': True, 'config': agn_config}, 
                'torus': {'enable': True, 'config': torus_config, 'file': torus_file}}
```
Current version of S<sup>3</sup>Fit supports stellar continuum (`'ssp'`), emission lines (`'el'`), AGN central continuum (`'agn'`), and AGN dusty torus (`'torus'`). 
If any models are not required, set `'enable'` to `False` or just remove the corresponding lines.
Note that the model names (e.g., 'ssp') are fixed in the code and please do not modity them.
The detailed set up of each model are described as follows. 

#### Stellar continuum
```python
ssp_file = 'DIRECTORY/popstar21_stellar_nebular_fullwave.fits'
```
Current version of S<sup>3</sup>Fit use the [PopSTAR][2] Single Stellar Population (SSP) model library. 
Please run the [converting code](models/convert_popstar_ssp.py) to convert the original PopSTAR models to the format used for this code. 
You may also want to download an example of the converted SSP model for test in [this link][7].

[2]: <https://www.fractal-es.com/PopStar/>
[7]: https://drive.google.com/file/d/1JwdBOnl6APwFmadIX8BYLcLyFNZvnuYg/view?usp=share_link

An example of setup of stellar continuum
```python
ssp_file = 'DIRECTORY/popstar21_stellar_nebular_fullwave.fits'
ssp_config = {'main': {'pars': [[-1000, 1000, 'free'], [100, 1200, 'free'], [0, 5.0, 'free'], 
                                [0, 0.94, 'free'], [-1, 1, 'free']], 
                       'info': {'age_min': -2.25, 'age_max': 'universe', 'met_sel': 'solar', 'sfh': 'exponential'} } }
```
`ssp_file`: Location of the PopSTAR SSP model library. 

pars: voff, fwhm, AV, log csp_age (Gyr), log sfh_tau (Gyr)
age_min, age_max: min and max log ssp_age (Gyr)
met_sel: 'all', 'solar', or any combination of [0.004,0.008,0.02,0.05]
sfh: 'nonparametric', 'exponential', 'delayed', 'constant'


```python
ssp_config = {'main': {'pars': [[-1000, 1000, 'free'], [100, 1200, 'free'], [0, 5.0, 'free'], 
                                [0, 0.94, 'free'], [-1, 1, 'free']], 
                       'info': {'age_min': -2.25, 'age_max': 'universe', 'met_sel': 'solar', 'sfh': 'exponential'} }, 
              'young': {'pars': [[None, None, 'ssp:main:0'], [None, None, 'ssp:main:1'], [None, None, 'ssp:main:2'], 
                                 [-2, -1, 'free'], [-1, -1, 'fix']], 
                        'info': {'age_min': -2.25, 'age_max': 0, 'met_sel': 'solar', 'sfh': 'constant'} } }

```

```python
ssp_config = {'main': {'pars': [[-1000, 1000, 'free'], [100, 1200, 'free'], [0, 5.0, 'free'], [-1, -1, 'fix'], [-1, -1, 'fix']], 
                       'info': {'age_min': 5.5e-3, 'age_max': 'universe', 'met_sel': 'solar', 'sfh': 'nonparametric'} } }
```

#### Emission lines
```python
el_config = {'NLR': {'pars':       [[ -500,   500, 'free'], [250,  750, 'free'], [0, 5, 'free'], [0.5, 1.45, 'free']], 
                     'info': {'line_used': ['all']}}, 
             'outflow_1': {'pars': [[-2000,   100, 'free'], [750, 2500, 'free'], [0, 5, 'free'], [0.5, 1.45, 'free']], 
                           'info': {'line_used': ['all']}}, 
             'outflow_2': {'pars': [[-3000, -2000, 'free'], [750, 2500, 'free'], [0, 5, 'free'], [0.5, 1.45, 'free']], 
                           'info': {'line_used': ['[OIII]a', '[OIII]b', '[NII]a', 'Ha', '[NII]b']} } }
```
pars: voff, fwhm, AV, SIIa/b (n_e: 1e4--1cm-3); 3 kinematic system
Each line have the same paramerters.
Support parameters are voff, fwhm, AV, and [SII]6716/6731 ratio (for electron density of 1e4--1cm-3)
'NLR:all' denotes for Narrow-Line-Region, all emission lines are used.
'outflow_2:[OIII]a,[OIII]b,[NII]a,Ha,[NII]b' denotes a secondary outflow component, for which only [OIII] doublets and Ha-[NII] complex are used.
'BLR:Ha' denotes for AGN Broad-Line-Region, only Ha (Halpha) is used.
Edit ELineModels.__init__() to add more lines.

#### AGN central continuum
```python
agn_config = {'main': {'pars': [[None, None, 'el:NLR:0;ssp:main:0'], [None, None, 'el:NLR:1;ssp:main:1'], [1.5, 10.0, 'free'], [-1.7, None, 'fix']],
                       'info': {'mod_used': ['powerlaw']} } }
```
pars: voff, fwhm, AV; alpha_lambda of powerlaw at 

#### AGN dusty torus

This code uses the [SKIRTor][4] AGN torus model.
Examples of this library are provided in [models](models/), 
which are resampled and reformed to be used by this code. 
Please refer to [SKIRTor][4] website for details and the original library. 

[4]: https://sites.google.com/site/skirtorus/sed-library?authuser=0

```python
torus_config = {'main': {'pars': [[None, None, 'el:NLR:0;ssp:main:0'], [3, 11, 'free'], [10, 80, 'free'], [10, 30, 'free'], [0, 90, 'free']],
                         'info': {'mod_used': ['dust']} } } # 
torus_file = '../models/skirtor_torus.fits'
```
pars: voff, tau, opening angle, radii ratio, inclination angle
set 'mod_used' to ['disc', 'dust'] if use both of disc and dusty torus modules

## Run fitting
```python
FF.main_fit()
```
example

