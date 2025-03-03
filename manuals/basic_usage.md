# Basic usage

> [!NOTE]
> The code is actively under development. Please double-check the manuals archived in the GitHub release for a specific version if you encounter any discrepancies.

## 1. Initialization
As the first step, please initialize the `FitFrame`, which is the main class of S<sup>3</sup>Fit, 
by providing the following input parameters. 
```python
from s3fit import FitFrame
FF = FitFrame(spec_wave_w=None, spec_flux_w=None, spec_ferr_w=None, spec_R_inst_w=None, spec_valid_range=None, 
              phot_name_b=None, phot_flux_b=None, phot_ferr_b=None, phot_flux_unit='mJy', phot_trans_dir=None, 
              v0_redshift=None, model_config=None, 
              num_mocks=0, fit_grid='linear', 
              print_step=True, plot_step=False, canvas=None)
```
#### 1.1 Input spectroscopic data
- `spec_wave_w` (numpy array of floats) \
   Wavelength of the input spectrum, in unit of angstrom.
- `spec_flux_w` and `spec_ferr_w` (numpy array of floats) \
   Fluxes and measurement errors of the input spectrum, in unit of erg s<sup>-1</sup> cm<sup>-2</sup> angstrom<sup>-1</sup>.
- `spec_R_inst_w` (numpy array of floats, or 2-element list) \
   Instrumental spectral resolution ($\lambda/\Delta\lambda$) of the input spectrum,
   this is used to convolve the model spectra and estimate the intrinsic velocity width. 
  `spec_R_inst_w` can be a list of variable resolutions as a function of the input wavelength `spec_wave_w`, 
   or given as a constant value as `spec_R_inst_w=[wave,R]` to specify the resolution `R` at the wavelength `wave` (in angstrom). 
- `spec_valid_range` (nested list of floats) \
   Valid wavelength range.
   For example, if 5000--7000 and 7500--10000 angstrom are used in fitting, set `spec_valid_range=[[5000,7000], [7500,10000]]`.
#### 1.2 Input photometric data
- `phot_name_b` (numpy array of strings) \
   List of band names of the input photometric data, e.g., `phot_name_b=['SDSS_gp','2MASS_J','WISE_1']`.
   The names should be the same as the filenames of the transmission curves in each band, e.g., `'SDSS_gp.dat'`. 
- `phot_flux_b` and `phot_ferr_b` (numpy array of floats) \
   Fluxes and measurement errors in each band.
- `phot_calib_b` (list or numpy array of strings) \
   List of band names of photometric data that is used for calibration of spectrum.
   For example, if 'SDSS_rp' and 'SDSS_ip' bands are covered by the spectrum,
   set `phot_calib_b=['SDSS_rp','SDSS_ip']`
   and S<sup>3</sup>Fit will scale the input `spec_flux_w` and `spec_ferr_w`
   with `phot_flux_b` in the two bands, e.g., to correct for aperture loss of the input spectrum. 
   Set `phot_calib_b=None` (default) if the calibration is not required. 
- `phot_fluxunit` (string) \
   Flux unit of `phot_flux_b` and `phot_ferr_b`, can be `'mJy'` (default) and `'erg/s/cm2/AA'`.
   If the input data is in unit of 'mJy', they will be converted to 'erg/s/cm2/AA' before the fitting.
- `phot_trans_dir` (string) \
   Directory of files of the transmission curves.
> [!TIP]
> If a pure-spectral fitting is required, please set `phot_name_b=None` or just remove all input parameters starting with `phot_` and `sed_` from the input parameters of `FitFrame`. 
#### 1.3 Model setup 
- `v0_redshift` (float) \
   Initial guess of the systemic redshift. The velocity shifts of all models are in relative to the input `v0_redshift`. 
- `model_config` (nested dictionary) \
   Dictionary of model configurations, see [model configuration](#2-model-configuration) section for details. 
#### 1.4 Control of fitting
- `num_mocks` (int) \
   Number of the mock spectra for the Monte Carlo method.  
   The mock spectra are used to estimate the uncertainty of best-fit results. Default is `0`, i.e., only the original data will be fit.
- `fit_grid` (string) \
   Set `fit_grid='linear'` (default) to run the fitting in linear flux grid.
   Set `fit_grid='log'` to run the fitting in logarithmic flux grid.
   Note that if emisison line is the only fitting model (e.g., for the fitting of continuum-subtracted spectrum), `fit_grid` is always set to `'linear'`.
   (please refer to [fitting strategy](./fitting_strategy.md) for details). 
#### 1.5 Auxiliary
- `print_step` (bool) \
   Whether or not to print the information each intermediate step (e.g., the examination of each model component).
   Default is `True`.
- `plot_step` (bool) \
   Whether or not to plot the best-fit model spectra and fitting residuals in each intermediate step.
   Default is `False`. 
- `canvas` (tuple) \
   Matplotlib window with a format of `canvas=(fig,axs)` to display each intermediate step dynamically.
   Please read the [Jupyter Notebook](../example/example.ipynb) for an example case. 
> [!NOTE]
> Please refer to the [list](./full_parameter_list.md) to learn about all of the available parameters of S<sup>3</sup>Fit. 


## 2. Model configuration
Current version of S<sup>3</sup>Fit supports stellar continuum (`'ssp'`), emission lines (`'el'`), AGN central continuum (`'agn'`), and AGN dusty torus (`'torus'`). 
If any models are not required, set `'enable': False` or just delete the corresponding lines.
```python
model_config = {'ssp'  : {'enable': True, 'config': ssp_config,   'file': ssp_file}, 
                'el'   : {'enable': True, 'config': el_config,    'use_pyneb': True},
                'agn'  : {'enable': True, 'config': agn_config}, 
                'torus': {'enable': True, 'config': torus_config, 'file': torus_file}}
```
> [!CAUTION]
> Note that the model names (e.g., 'ssp') are fixed in the code and please do not modify them.

> [!TIP]
> Please read guide in [advanced usage](../manuals/advanced_usage.md) if you want to add a new model type. 

The detailed set up of each model are described as follows. 

#### 2.1 Stellar continuum

```python
ssp_file = 'DIRECTORY/popstar_for_s3fit.fits'
```
Current version of S<sup>3</sup>Fit use the [HR-pyPopStar][2] Single Stellar Population (SSP) model library. 
Please run the [converting code](../model_libraries/convert_popstar_ssp.py) to convert the original HR-pyPopStar models to the format used for S<sup>3</sup>Fit. 
You may also want to download an example of the converted SSP model for test in [this link][7].

[2]: <https://www.fractal-es.com/PopStar/>
[7]: https://drive.google.com/file/d/1JwdBOnl6APwFmadIX8BYLcLyFNZvnuYg/view?usp=share_link

An example of stellar continuum configuration:
```python
ssp_config = {'main': {'pars': [[-1000, 1000, 'free'], # velocity shift (km/s)
                                [100, 1200, 'free'], # velocity FWHM (km/s)
                                [0, 5.0, 'free'], # extinction (AV)
                                [0, 0.94, 'free'], # CSP age (log Gyr)
                                [-1, 1, 'free']], # declining timescale of SFH (log Gyr)
                       'info': {'age_min': -2.25, 'age_max': 'universe', 'met_sel': 'solar', 'sfh_name': 'exponential'} } }
```
In this example, one stellar component (e.g., one stellar population) is set up, which has the name `'main'`; the name can be set freely. 
`'pars'` stores the setup for each parameter. 
Five parameters are used in the example, which are (from the left)
velocity shift (km/s, in relative to the input `v0_redshift`), velocity FWHM (km/s), extinction (AV), log age (Gyr) of the Composite Stellar Population (CSP), 
and the log $\tau$ value (Gyr) of the Star Formation History (SFH), respectively. 
Every parameter setup has the format `[min value, max value, tie condition]`. 
In this example, all parameters are free parameters in fitting. 

`'info'` lists the regulation of the stellar models. 
`'age_min'` and `'age_max'` are the minimum and maximum stellar ages (log Gyr) of the used SSP models.
Set `'age_min': None` if the youngest HR-pyPopStar model is used. 
Set `'age_max': 'universe'` to use the universe age in the input `v0_redshift`. 
`'met_sel'` limit the metallicity, which can be set to 
`'all'` to use all metallicities in HR-pyPopStar models (Z = 0.004, 0.008, 0.02, 0.05), 
`'solar'` to only use solar metallicity (Z = 0.002), 
or any combination of values in a list, e.g., `[0.004,0.008]` or `[0.02,0.05]`. 

`'sfh_name'` denotes the SFH used in this `'main'` component. 
Current version of S<sup>3</sup>Fit supports the following SFH functions:
`'nonparametric'`, `'exponential'`, `'delayed'`, `'constant'`. 
Please read guide in [advanced usage](../manuals/advanced_usage.md) if you want to add a new SFH function. 
An example of `'nonparametric'` SFH is shown as follows:
```python
ssp_config = {'main': {'pars': [[-1000, 1000, 'free'], [100, 1200, 'free'], [0, 5.0, 'free']], 
                       'info': {'age_min': 5.5e-3, 'age_max': 'universe', 'met_sel': 'solar', 'sfh': 'nonparametric'} } }
```
In the case with `'nonparametric'` SFH, only the first three parameters, 
velocity shift, velocity FWHM (km/s), and extinction, are used. 

S<sup>3</sup>Fit supports any combination of the SFH functions with multiple CSP components
(except for `'nonparametric'` SFH that can be only used individually). 
The following example shows the case with two CSP components, 
where the `'main'` component use `'exponential'` SFH and the `'young'` component use `'constant'` SFH. 
```python
ssp_config = {'main': {'pars': [[-1000, 1000, 'free'], [100, 1200, 'free'], [0, 5.0, 'free'], 
                                [0, 0.94, 'free'], [-1, 1, 'free']], 
                       'info': {'age_min': -2.25, 'age_max': 'universe', 'met_sel': 'solar', 'sfh': 'exponential'} }, 
              'young': {'pars': [[None, None, 'ssp:main:0'], [None, None, 'ssp:main:1'], [None, None, 'ssp:main:2'], 
                                 [-2, -1, 'free'], [-1, -1, 'fix']], 
                        'info': {'age_min': -2.25, 'age_max': 0, 'met_sel': 'solar', 'sfh': 'constant'} } }
```
> [!TIP]
> The above example also shows how to tie one parameter to the others. 
For instance, the velocity shift of the `'young'` component has a tie condition of `'ssp:main:0'`, 
indicating that it is tied to the `0`-th parameter of the `'main'` component of the `'ssp'` model.
Also note that the log $\tau$ value (the last parameter) is not used for a `'constant'` SFH, 
set it to any value with `'fix'` to save a free parameter.

> [!CAUTION]
> Please do not directly delete unused parameters with multiple components
> since all stellar components need to have the same number of parameters. 

> [!TIP]
> The best-fit reconstructed SFH of each component can be plotted or output
> by running
> ```python
> FF.model_dict['ssp']['spec_mod'].reconstruct_sfh()
> ```
> Please read the [Jupyter Notebook](../example/example.ipynb) for an example case. 

#### 2.2 Emission lines

S<sup>3</sup>Fit supports combination of multiple emission line components. 
An example of emission line configuration is shown as follows:
```python
el_config = {'NLR': {'pars': [[-500, 500, 'free'], # velocity shift (km/s)
                              [ 250, 750, 'free'], # velocity FWHM (km/s)
                              [0, 5, 'free'], # extinction (AV)
                              [1.3, 4.3, 'free'], # electron density (log cm-3)
                              [4, None, 'fix']], # electron temperature (log K)
                     'info': {'line_used': ['all']}}, 
             'outflow_1': {'pars': [[-2000,  100, 'free'], # velocity shift (km/s)
                                    [  750, 2500, 'free'], # velocity FWHM (km/s)
                                    [0, 5, 'free'], # extinction (AV)
                                    [1.3, 4.3, 'free'], # electron density (log cm-3)
                                    [4, None, 'fix']], # electron temperature (log K)
                           'info': {'line_used': ['all']}}, 
             'outflow_2': {'pars': [[-3000, -2000, 'free'], # velocity shift (km/s)
                                    [  750,  2500, 'free'], # velocity FWHM (km/s)
                                    [0, None, 'fix'], # extinction (AV)
                                    [2, None, 'fix'], # electron density (log cm-3)
                                    [4, None, 'fix']], # electron temperature (log K)
                           'info': {'line_used': ['[O III]:4960', '[O III]:5008', '[N II]:6550', 'Ha', '[N II]:6585'] }},
             'BLR': {'pars': [[ -500,  500, 'free'], # velocity shift (km/s)
                              [  750, 9900, 'free'], # velocity FWHM (km/s)
                              [0, None, 'fix'], # extinction (AV)
                              [9, None, 'fix'], # electron density (log cm-3)
                              [4, None, 'fix']], # electron temperature (log K)
                     'info': {'line_used': ['Ha'] }} }
```
In this example, four components are used:
narrow lines (`'NLR'`), the primary outflow component (`'outflow_1'`), 
the secondary outflow component (`'outflow_2'`, which is faster than `'outflow_1'`), 
and broad lines from AGN Broad-Line-Region (`'BLR'`). 
`'NLR'` and `'outflow_1'` have `'info': {'line_used': ['all']}`, 
which means all available emission lines are used.
For `'outflow_2'` and `'BLR'`, only the emission lines with names specified in `'line_used'` are used. 
> [!TIP]
> Please run `FF.model_dict['el']['spec_mod'].linename_n` and `FF.model_dict['el']['spec_mod'].linerest_n`
to learn about the names and rest wavelengths (in vacuum) of the available emission lines. 
Please read guide in [advanced usage](../manuals/advanced_usage.md) if you want to add new emission lines. 

For a given emission line component (e.g., `'NLR'`), every line shares the same parameter values. 
In the current version of S<sup>3</sup>Fit, there are five parameters for each component (from the left):
velocity shift (km/s, in relative to the input `v0_redshift`), velocity FWHM (km/s), extinction (AV), 
electron density (log cm<sup>-3</sup>) and electron temperature (log K). 
The extinction, electron density and temperature are included to calculated the observed flux ratios of
line transitions, for which the line fitting is affected by other factors, 
such as the effect of absorption of stellar continuum on Hydrogen emission lines, 
and the blurring of neighboring line doublets (e.g., broad components of [OIII]4960 and [OIII]5008 doublets). 
> [!TIP]
> In the above example, the extinction for `'outflow_2'` and `'BLR'` is set to a arbitrarily fixed value `[0, None, 'fix']`
> since only Hα is used among Balmer lines for those components.
> The electron density is also fixed to typical values,
> i.e., 10<sup>2</sup> cm<sup>-3</sup> for outflow and 10<sup>9</sup> cm<sup>-3</sup> for AGN BLR,
> since the flux ratios of used lines are not sensitive to electron density. 

If `model_config['el']['use_pyneb']` is set to `True`, 
S<sup>3</sup>Fit can use [PyNeb](http://research.iac.es/proyecto/PyNeb/) 
to calculate the intrinsic line ratios based on the given electron density and temperature. 
Since the currently line ties are not sensitive to the electron temperature, 
the temperature is fixed to a typical value of the ionized medium, $10^4$ K. 
If PyNeb is not enabled, S<sup>3</sup>Fit use fixed flux ratios for each line pairs, 
which are calculated for an electron temperature of 10<sup>4</sup> K
and an electron density of 10<sup>2</sup> cm<sup>-3</sup>
(except for [SII]6718/6733, which ratio is calculated from electron density using 
the [Proxauf et al. (2014) equation](https://ui.adsabs.harvard.edu/abs/2014A%26A...561A..10P)
). 
Please run the following codes to learn about the default tied lines and their flux ratios. 
```python
el_mod = FF.model_dict['el']['spec_mod']
el_mod.update_lineratio()
for i_line in range(el_mod.num_lines):
    if el_mod.lineratio_n[i_line] < 0: continue # i.e., skip lines witha free flux normalization
    i_ref = np.where(el_mod.linename_n == el_mod.linelink_n[i_line])[0][0]
    print(f'The flux of {el_mod.linename_n[i_line]} is tied to {el_mod.linename_n[i_ref]} with a ratio of {el_mod.lineratio_n[i_line]:2.4f}.')
```
Please read guide in [advanced usage](../manuals/advanced_usage.md) if you want to add or delete line tying relations. 

#### 2.3 AGN central continuum

Current version of S<sup>3</sup>Fit supports a powerlaw model to account for the radiation from the AGN accretion disc. 
The powerlaw model has a flexible spectral index in wavelength from 0.1 micron to 5 micron, 
and is bent with a fixed spectral index of -4 in wavelength longer than 5 micron (i.e., following [SKIRTor][SKIRTor_web]). 
Other modules of AGN central radiation in UV/optical range, e.g., iron pseudo continuum and Balmer continuum, 
will be supported in future versions. 

```python
agn_config = {'main': {'pars': [[None, None, 'el:NLR:0;ssp:main:0'], # velocity shift (km/s)
                                [0, 0, 'fix'], # velocity FWHM (km/s)
                                [1.5, 10.0, 'free'], # extinction (AV)
                                [-1.7, None, 'fix']], # spectral index of powerlaw
                       'info': {'mod_used': ['powerlaw']} } }
```
An example of the powerlaw component is shown as above. 
The listed parameters are 
velocity shift (km/s, in relative to the input `v0_redshift`), velocity FWHM (km/s), extinction (AV), 
and the spectral index of powerlaw. 
Since the velocity shift cannot be determined with only powerlaw, it is tied with `'el:NLR:0;ssp:main:0'`, 
which means the value is tied to the `0`-th parameter (i.e., velocity shift) of the `'NLR'` component of the `'el'` model, 
or the `0`-th parameter of the `'main'` component of the `'ssp'` model when the `'el'` model is not available 
(e.g., in several intermediate fitting steps with only continuum models, see [fitting strategy](../manuals/fitting_strategy.md) for details).
Similarly, the velocity FWHM is also not applicable with only powerlaw model, 
and thus fixed arbitrarily to save a free parameter. 
Both of velocity shift and FWHM could be determined independently when the iron pseudo continuum model is supported. 

In order to reduce the degeneracy between extinction and the spectral index, in this example the later is fixed to -1.7. 

#### 2.4 AGN dusty torus

```python
torus_file = '../model_libraries/skirtor_for_s3fit.fits'
```
S<sup>3</sup>Fit uses the [SKIRTor][SKIRTor_web] AGN torus model.
Please download the [SKIRTor library][SKIRTor_web] and run the [converting code](../model_libraries/convert_skirtor_torus.py) 
to create the torus models used for S<sup>3</sup>Fit. 
Example of this library is also provided in [model libraries](../model_libraries/) for a test of S<sup>3</sup>Fit, 
which contains the templates with a fixed dust density gradient in radial (p = 1) and angular direction (q = 0.5). 
Please refer to [SKIRTor][SKIRTor_web] website for details of the model parameters. 

[SKIRTor_web]: https://sites.google.com/site/skirtorus/sed-library?authuser=0

An example of the configuration of torus model is given as follows:
```python
torus_config = {'main': {'pars': [[None, None, 'el:NLR:0;ssp:main:0'], # velocity shift (km/s)
                                  [3, 11, 'free'], # optical depth at 9.7 micron 
                                  [10, 80, 'free'], # half-opening angle (degree) of torus
                                  [10, 30, 'free'], # ratio of outer to inner radius
                                  [0, 90, 'free']], # inclination angle (degree) from the polar direction
                         'info': {'mod_used': ['dust']} } } 
```
The listed parameters are 
velocity shift (km/s, in relative to the input `v0_redshift`), 
optical depth at 9.7 micron, 
half-opening angle (degree) of the dusty torus, 
ratio of outer to inner radius, 
and inclination angle (degree) from the polar direction. 
Similar to the case of AGN powerlaw component, 
the velocity shift of the torus component is also tied to those of emission line or stellar continuum. 
The `'mod_used'` can be set to `['dust']` to use the pure torus module, 
or `['disc','dust']` to use both of the disc and torus modules
(do not use `'agn'` powerlaw component in this case). 

## 3. Run fitting

After finishing the configuration of all models, run the fitting with the following command:
```python
FF.main_fit()
```
The followng figures denote cases if you would like to check each fitting step by setting `plot_step=True`.
You can choose to list figures in each step (left) or show them dynamically in a given window (right) by specifying the `canvas`. 
<p align="center">
  <img src="https://github.com/user-attachments/assets/de2aaa34-ccdc-419a-ba7d-80660854d012" width="48%" />
  <img src="https://github.com/user-attachments/assets/ae283b48-4097-4c47-901c-34c8c100e69b" width="48%" />
</p>

Please read the [Jupyter Notebook](../example/example.ipynb) for examples of actual fitting. 
The notebook also exhibits 
the method to save the fitting results to a file and reload them from the file, 
and 
the methods to output and display the best-fit results, e.g., model spectra and properties. 

