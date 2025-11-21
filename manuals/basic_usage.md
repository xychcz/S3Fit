# Basic usage

> [!CAUTION]
> The page is being edited for the upcoming release of S<sup>3</sup>Fit **v2.3** (already uploaded in the main repository). The updating till Section 2.1 (included) is finished. 

> [!NOTE]
> This page is for S<sup>3</sup>Fit **v2.3**. S<sup>3</sup>Fit is under active development. Please double-check the manuals archived in the GitHub release for a specific version if you encounter any discrepancies.

> [!TIP]
> Examples of step-by-step usage of S<sup>3</sup>Fit can be found in [example1](https://github.com/xychcz/S3Fit/blob/main/examples/example_galaxy.ipynb) and [example2](https://github.com/xychcz/S3Fit/blob/main/examples/example_quasar.ipynb)

## 1. Initialization
As the first step, please initialize the `FitFrame`, which is the main framework of S<sup>3</sup>Fit, by providing the following input parameters. 
```python
from s3fit import FitFrame
FF = FitFrame(spec_wave_w=None, spec_flux_w=None, spec_ferr_w=None, spec_R_inst_w=None, spec_valid_range=None, 
              phot_name_b=None, phot_flux_b=None, phot_ferr_b=None, phot_trans_dir=None, phot_flux_unit='mJy', 
              v0_redshift=None, model_config=None, 
              num_mocks=0, fit_grid='linear', examine_result=True, 
              print_step=True, plot_step=False)
```
#### 1.1 Input spectroscopic data
- `spec_wave_w` (list or numpy array of floats, <ins>**required**</ins>) \
   Wavelength of the input spectrum, in unit of angstrom.
- `spec_flux_w` and `spec_ferr_w` (list or numpy array of floats, <ins>**required**</ins>) \
   Fluxes and measurement errors of the input spectrum, in unit of erg s<sup>-1</sup> cm<sup>-2</sup> angstrom<sup>-1</sup>.
- `spec_R_inst_w` (list or numpy array of floats, or 2-element list, <ins>**required**</ins>) \
   Instrumental spectral resolution ($\lambda/\Delta\lambda$) of the input spectrum, this is used to convolve the model spectra and estimate the intrinsic velocity width.  `spec_R_inst_w` can be a list of variable resolutions as a function of the input wavelength `spec_wave_w`, or given as a constant value as `spec_R_inst_w=[wave,R]` to specify the resolution `R` at the wavelength `wave` (in angstrom). 
- `spec_valid_range` (nested list of floats, optional) \
   Valid wavelength range. For example, if 5000--7000 and 7500--10000 angstrom are used in fitting, set `spec_valid_range=[[5000,7000], [7500,10000]]`. Default is `None`, in this case the entire input spectrum (except for the wavelengths with non-positive `spec_ferr_w`) will be used in the fitting. 
#### 1.2 Input photometric data
- `phot_name_b` (list or numpy array of strings, required for simultaneous spectrum+SED fitting) \
   List of band names of the input photometric data, e.g., `phot_name_b=['SDSS_gp','2MASS_J','WISE_1']`. The names should be the same as the filenames of the transmission curves in each band, e.g., `'SDSS_gp.dat'`. 
- `phot_flux_b` and `phot_ferr_b` (list or numpy array of floats, required for simultaneous spectrum+SED fitting) \
   Fluxes and measurement errors in each band. The unit is given in `phot_fluxunit`. 
- `phot_trans_dir` (string, required for simultaneous spectrum+SED fitting) \
   Directory of files of the transmission curves.
> [!TIP]
> The above four parameters are only necessary if a simultaneous fitting of spectrum and photometric-SED is performed.
> S<sup>3</sup>Fit will run in a pure-spectral fitting mode if these parameters are set to `None` (default). 
- `phot_fluxunit` (string, optional) \
   Flux unit of `phot_flux_b` and `phot_ferr_b`, can be `'mJy'` (default) and `'erg/s/cm2/AA'`. S<sup>3</sup>Fit run with $f_\lambda$ and it can handle the conversion automatically if the input flux is in $f_\nu$. 
- `phot_calib_b` (list or numpy array of strings, optional) \
   List of band names of photometric data that is used for flux calibration of spectrum (e.g., to correct for aperture loss of the input spectrum). For example, if 'SDSS_rp' and 'SDSS_ip' bands are covered by the spectrum, you can set `phot_calib_b=['SDSS_rp','SDSS_ip']` and S<sup>3</sup>Fit will scale the input `spec_flux_w` and `spec_ferr_w` with `phot_flux_b` in the two bands. Set `phot_calib_b=None` (default) if the calibration is not required. 
#### 1.3 Model setup 
- `v0_redshift` (float, <ins>**required**</ins>) \
   Initial guess of the systemic redshift. The velocity shifts of all models are in relative to the input `v0_redshift`. 
- `model_config` (nested dictionary, <ins>**required**</ins>) \
   Dictionary of model configurations. Please refer to the following [model configuration](#2-model-configuration) section for details. 
#### 1.4 Control of fitting
- `num_mocks` (int, optional) \
   Number of the mock spectra for the Monte Carlo method. The mock spectra are used to estimate the uncertainty of best-fit results. Default is `0`, i.e., only the original data will be fit.
- `fit_grid` (string, optional) \
   Set `fit_grid='linear'` (default) to run the fitting in linear flux grid, or `fit_grid='log'` to run the fitting in logarithmic flux grid. Note that if `line` model is the only fitting model (e.g., for the fitting of continuum-subtracted spectrum), `fit_grid` is always set to `'linear'`. (please refer to [fitting strategy](./fitting_strategy.md) for details).
- `examine_result` (bool, optional) and `accept_model_SN` (float, optional) \
   If `examine_result=True` (default), the best-fit models will be examined. All continuum models and line components with peak S/N < `accept_model_SN` (default: 2) will be automatically disabled. An additional fitting step will be performed with the updated model configuration (i.e., the 2nd fitting steps in [fitting strategy](./fitting_strategy.md)). 
   If `examine_result=False`, the model examinations (except for absorption lines, if included in line configuration) and updated fitting step will be skipped.
-  `accept_absorption_SN` (float, optional) \
   Acceptable minimum peak S/N of absorption line component(s). Any absorption line component(s) with peak S/N < `accept_absorption_SN` will be automatically disabled. The default value is the same as `accept_model_SN`. Note that the examinations of absorption lines is always performed even though `examine_result=False`.
#### 1.5 Auxiliary
- `print_step` (bool, optional) \
   Whether or not to print the information each intermediate step (e.g., the examination of each model component). Default is `True`.
- `plot_step` (bool, optional) \
   Whether or not to plot the best-fit model spectra and fitting residuals in each intermediate step. Default is `False`. 
   
> [!NOTE]
> Please refer to the [list](./full_parameter_list.md) to learn about all of the available parameters of S<sup>3</sup>Fit. 


## 2. Model configuration
Current version of S<sup>3</sup>Fit supports continuum model type of stellar continuum (`'stellar'`), AGN UV/optical continuum (`'agn'`), and AGN torus IR continuum (`'torus'`), as well as line model type (`'line'`). The format of the entire model configurations is a nested dictionary as follows. If any models are not required, set `'enable': False` or just delete the corresponding lines.
```python
model_config = {'stellar': {'enable': True, 'config': stellar_config, 'file': ssp_file}, 
                'agn'    : {'enable': True, 'config': agn_config    , 'file': iron_file}, 
                'torus'  : {'enable': True, 'config': torus_config  , 'file': torus_file}
                'line'   : {'enable': True, 'config': line_config   , 'use_pyneb': True},
			   }
```
> [!CAUTION]
> Note that the model type names (e.g., `'stellar'` and `'agn'`) are hard-coded and please do not modify them. `'stellar'` and `'line'` models were named as, `'ssp'` and `'el'`, respectively, in S<sup>3</sup>Fit v2.2.4 and earlier versions. These old names are still supported in the current version, while they should be considered to be deprecated.

Each model type has its specific `'config'` (i.e., `xxx_config` in the above `model_config`) and the path of the model template file (i.e., `xxx_file` in the `'file'` keys in the above `model_config` ). Every specific `xxx_config` share the same following format:
```python

xxx_config = {'component_0': {'pars': [[min_0, max_0, tie_0], # parameter_0
                                       [min_0, max_0, tie_0], # parameter_0
                                       [min_0, max_0, tie_0], # parameter_0
                                       # ...
                                       # ...
									  ], 
                              'info': {'item_0': value_0,     # infomation_0
                                       'item_1': value_1,     # infomation_1
                                       'item_2': value_2,     # infomation_2
                                       # ...
                                       # ...
									  }
                             }, 
              'component_1': {'pars': [[min_0, max_0, tie_0], # parameter_0
                                       [min_0, max_0, tie_0], # parameter_0
                                       [min_0, max_0, tie_0], # parameter_0
                                       # ...
                                       # ...
									  ], 
                              'info': {'item_0': value_0,     # infomation_0
                                       'item_1': value_1,     # infomation_1
                                       'item_2': value_2,     # infomation_2
                                       # ...
                                       # ...
									  }
                             }, 
			  # ...
			  # ...
			 }
```
A given model type can have multiple components, e.g., `'component_0'` and `'component_1'` as above, . For example, stellar continuum can have old and young population components, line model can have several narrow and broad line components. Each component has its own parameter list (`'pars'`) and constraint information (`'info'`). The difference between `'pars'` and `'info'` are, `'pars'` are fitting parameters (i.e., the non-linear parameters in [fitting strategy](./fitting_strategy.md)), e.g., line width, and thus have the value boundaries (`min_xx, max_xx`) and tying relations (`tie_xx`); while `'info'` gives fixed specifications or assumptions, e.g., which atom lines are used in the fitting. The detailed of each model type are described in following subsections. 

> [!NOTE]
> There is no limit of the number of components for a given model type. The name of components (`component_x` as above) can be defined by users, e.g., 'starburst', 'old stellar population', or 'narrow lines'. 

> [!NOTE]
> Each fitting parameter can be free parameter (if `tie_xx` is `'free'`) or fixed value (if `tie_xx` is `'fix'`). If different components or different model types have the same parameter, the parameter can be tied among components or models. An example of the tying relation is given in the following [stellar continuum configuration](#21-stellar-continuum) with two stellar population components. 

> [!TIP]
> You can import a new model type into S<sup>3</sup>Fit following the guide in [advanced usage](../manuals/advanced_usage.md) (Section 6. Support new types of models). 
#### 2.1 Stellar continuum

```python
ssp_file = 'DIRECTORY/popstar_for_s3fit.fits'
```
The current version of S<sup>3</sup>Fit uses the [HR-pyPopStar][2] Single Stellar Population (SSP) model library. HR-pyPopStar library provides stellar continuum templates for stellar ages (log Gyr) from -4.00 to 1.18, and metallicities (Z) of 0.004, 0.008, 0.02 and 0.05. Please run the [converting code](../model_libraries/convert_popstar_ssp.py) to convert the original HR-pyPopStar library to the format used for S<sup>3</sup>Fit. You may also want to download an example of the SSP model, with an initial mass function (IMF) of Kroupa (2002), in [this link][7] for a test. If you want to use HR-pyPopStar libraries with a different IMF, or use a different SSP library, please follow the guide in Section 4 of the [advanced usage](../manuals/advanced_usage.md).

[2]: <https://www.fractal-es.com/PopStar/>
[7]: https://drive.google.com/file/d/1JwdBOnl6APwFmadIX8BYLcLyFNZvnuYg/view?usp=share_link

A simple example of stellar continuum configuration is as follows:
```python
stellar_config = {'main': {'pars': [[-1000, 1000, 'free'],  # velocity shift (km/s)
                                    [  100, 1200, 'free'],  # velocity FWHM (km/s)
                                    [    0,  5.0, 'free'],  # extinction (AV)
                                    [    0, 0.94, 'free'],  # CSP age (or galaxy age) (log Gyr)
                                    [   -1,    1, 'free'],  # declining timescale of exponential SFH (log Gyr)
								   ],
                           'info': {'age_min' : -2.25,         # min SSP age (log Gyr)
                                    'age_max' : 'universe',    # max SSP age, can be either given in log Gyr, or in the universe age at the given v0_redshift
                                    'met_sel' : 'solar',       # metallicity, can be 'all', 'solar', or any combination of [0.004,0.008,0.02,0.05]
                                    'sfh_name': 'exponential', # name of SFH function, can be 'nonparametric', 'exponential', 'delayed', 'constant', or user defined
						           }
                          }
                 }
```
In this example, one stellar component (e.g., one stellar population) is set up, which has the name `'main'`; the name can be set freely. `'pars'` stores the setup for each parameter. Every parameter setup has the format `[min value, max value, tie condition]`. In this example, all parameters are free parameters in fitting. Five parameters are used in the example, which are (from the top) velocity shift (km/s, in relative to the input `v0_redshift`), velocity FWHM (km/s), extinction (AV), log age (Gyr) of the Composite Stellar Population (CSP) (i.e., the age of the galaxy), and the log $\tau$ value (Gyr) of the Star Formation History (SFH), respectively. 

`'info'` lists the regulation of the stellar models. `'age_min'` and `'age_max'` are the minimum and maximum stellar ages (log Gyr) of the used SSP models. Set `'age_min': None` if the youngest HR-pyPopStar model is used. Set `'age_max': 'universe'` to use the universe age in the input `v0_redshift`. `'met_sel'` limit the metallicity, which can be set to `'all'` to use all metallicities in HR-pyPopStar models (Z = 0.004, 0.008, 0.02, 0.05), or `'solar'` to only use solar metallicity (Z = 0.002), or any combination of values in a list, e.g., `[0.004,0.008]` or `[0.02,0.05]`. 

`'sfh_name'` denotes the SFH used in this `'main'` component. The current version of S<sup>3</sup>Fit supports the following SFH functions: `'nonparametric'`, `'exponential'`, `'delayed'`, `'constant'`. 
An example config with `'nonparametric'` SFH is shown as follows:
```python
stellar_config = {'main': {'pars': [[-1000, 1000, 'free'], # velocity shift (km/s)
                                    [  100, 1200, 'free'], # velocity FWHM (km/s)
                                    [    0,  5.0, 'free'], # extinction (AV)
					           	   ], 
                           'info': {'age_min' : 5.5e-3,          # min SSP age (log Gyr)
					                'age_max' : 'universe',      # max SSP age, can be either given in log Gyr, or in the universe age at the given v0_redshift
					                'met_sel' : 'solar',         # metallicity, can be 'all', 'solar', or any combination of [0.004,0.008,0.02,0.05]
					                'sfh_name': 'nonparametric', # name of SFH function
					               } 
						  } 
			     }
```
In the case with `'nonparametric'` SFH, only the first three parameters, velocity shift, velocity FWHM (km/s), and extinction, are used. The example of fitting with `'nonparametric'` SFH can be found in Section 3.4 and 7.1.2 in the [example](https://github.com/xychcz/S3Fit/blob/main/examples/example_galaxy.ipynb).

> [!TIP]
> You can use a user custom SFH function following the guide in [advanced usage](../manuals/advanced_usage.md) (Section 3. Support new SFH functions). 

S<sup>3</sup>Fit supports any combinations of the SFH functions with multiple CSP components (except for `'nonparametric'` SFH that can be only used individually). The following example shows the case with two CSP components, where the `'main'` component (an old stellar population) uses `'exponential'` SFH and the `'young'` component (a starburst) uses `'constant'` SFH. The example of fitting with the config can be found in the [example](https://github.com/xychcz/S3Fit/blob/main/examples/example_galaxy.ipynb).
```python
stellar_config = {'main' : {'pars': [[-1000, 1000, 'free'], # velocity shift (km/s)
                                     [  100, 1200, 'free'], # velocity FWHM (km/s)
									 [    0,  5.0, 'free'], # extinction (AV)
									 [    0, 0.94, 'free'], # CSP age of old population (or galaxy age) (log Gyr)
									 [   -1,    1, 'free'], # declining timescale of exponential SFH (log Gyr)
									], 
                            'info': {'age_min' : -2.25,         # min SSP age (log Gyr)
							         'age_max' : 'universe',    # max SSP age, can be either given in log Gyr, or in the universe age at the given v0_redshift
									 'met_sel' : 'solar',       # metallicity, can be 'all', 'solar', or any combination of [0.004,0.008,0.02,0.05]
									 'sfh_name': 'exponential', # name of SFH function
									},
						   }, 
                  'young': {'pars': [[None, None, 'stellar:main:0'], # velocity shift (km/s)
				                     [None, None, 'stellar:main:1'], # velocity FWHM (km/s)
									 [None, None, 'stellar:main:2'], # extinction (AV)
									 [  -2,   -1, 'free'          ], # CSP age of young population (log Gyr). 'constant' SFH only has this parameter
									], 
                            'info': {'age_min' : -2.25,      # min SSP age (log Gyr)
							         'age_max' : 0,          # max SSP age of young population (log Gyr)
									 'met_sel' : 'solar',    # metallicity, can be 'all', 'solar', or any combination of [0.004,0.008,0.02,0.05]
									 'sfh_name': 'constant', # name of SFH function
									}
						   }
									
			     }
```

The above example also shows how to tie one parameter to the others. For instance, the velocity shift of the `'young'` component has a tie condition of `'stellar:main:0'`, indicating that it is tied to the `0`th parameter of the `'main'` component of the `'stellar'` model. The same tying relation is set for the velocity FWHM and extinction of the 'young' component.

> [!TIP]
> There are several mode to set tying relation. Here we take the extinction of the `'young'` component, $A_{V\mathrm{,\ young}}$, as an example, which can be tied to the best-fit $A_{V\mathrm{,\ main}}$ with the following patterns: \
> (1) `[None, None, 'stellar:main:2']`, a hard tie, $A_{V\mathrm{,\ young}} = A_{V\mathrm{,\ main}}$; \
> (2) `[0.5, 3, 'stellar:main:2:+']`, a float, additive tie, $A_{V\mathrm{,\ young}}$ can varies from $A_{V\mathrm{,\ main}}+0.5$ to $A_{V\mathrm{,\ main}}+3$; \
> (3) `[0.5, 3, 'stellar:main:2:x']`, a float, multiplicative tie, $A_{V\mathrm{,\ young}}$ can varies from $0.5A_{V\mathrm{,\ main}}$ to $3A_{V\mathrm{,\ main}}$; the marker `':x'` can be replaced with `':*'`; \
> (4) `[1.5, None, 'stellar:main:2:+:fix']`, an additive tie with a fixed factor,  $A_{V\mathrm{,\ young}} = A_{V\mathrm{,\ main}}+1.5$; \
> (5) `[1.5, None, 'stellar:main:2:x:fix']`, a multiplicative tie with a fixed factor,  $A_{V\mathrm{,\ young}} = 1.5A_{V\mathrm{,\ main}}$. \
> You can also set a secondary tying relation in case the primary tied model or component is not available. For example, with `[None, None, 'line:NLR:2;stellar:main:2']`, $A_{V\mathrm{,\ young}}$ is tied to the extinction of the `'NLR'` component of the `'line'` model (i.e., the `2`nd parameter of the component) in the default case, or tied to $A_{V\mathrm{,\ main}}$ when the `line` model is not available (e.g., in the step of pure-continuum fitting). 

> [!TIP]
> The best-fit reconstructed SFH of each component can be plotted or output by running
> ```python
> FF.model_dict['stellar']['spec_mod'].reconstruct_sfh()
> ```
> Please read the [Jupyter Notebook](../examples/example_galaxy.ipynb) for an example case. 

#### 2.2 AGN UV/optical continuum

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

#### 2.3 AGN torus IR continuum

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
#### 2.4  Line model

A simple example of stellar continuum configuration is as follows, which contains one line component with the name of 'narrow line':
```python
line_config = {'narrow line': {'pars': [[-500,  500, 'free'], # velocity shift (km/s)
                                        [ 250,  750, 'free'], # velocity FWHM (km/s)
                                        [   0,    5, 'free'], # extinction (AV)
                                        [ 1.3,  4.3, 'free'], # electron density (log cm-3)
                                        [   4, None, 'fix' ], # electron temperature (log K)
									   ],
                               'info': {'line_used': 'default' , # line preset, or names of lines to be used
							            'sign'     : 'emission', # either 'emission' or 'absorption'
										'profile'  : 'Gaussian', # line profile, can be 'Gaussian' or 'Lorentz'
									   }
							  }
			  }
```


S<sup>3</sup>Fit supports combination of multiple line components. 
An example of line configuration is shown as follows:
```python
el_config = {'NLR': {'pars': [[-500, 500, 'free'], # velocity shift (km/s)
                              [ 250, 750, 'free'], # velocity FWHM (km/s)
                              [0, 5, 'free'], # extinction (AV)
                              [1.3, 4.3, 'free'], # electron density (log cm-3)
                              [4, None, 'fix']], # electron temperature (log K)
                     'info': {'line_used': 'default'}}, 
             'outflow_1': {'pars': [[-2000,  100, 'free'], # velocity shift (km/s)
                                    [  750, 2500, 'free'], # velocity FWHM (km/s)
                                    [0, 5, 'free'], # extinction (AV)
                                    [1.3, 4.3, 'free'], # electron density (log cm-3)
                                    [4, None, 'fix']], # electron temperature (log K)
                           'info': {'line_used': ['[O III]:4960', '[O III]:5008', '[N II]:6550', 'Ha', '[N II]:6585'] }},
             'outflow_2': {'pars': [[-3000, -2000, 'free'], # velocity shift (km/s)
                                    [  750,  2500, 'free'], # velocity FWHM (km/s)
                                    [0, None, 'fix'], # extinction (AV)
                                    [2, None, 'fix'], # electron density (log cm-3)
                                    [4, None, 'fix']], # electron temperature (log K)
                           'info': {'line_used':  }},
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

Please read the [Jupyter Notebook](../examples/example_galaxy.ipynb) for examples of actual fitting. 
The notebook also exhibits 
the method to save the fitting results to a file and reload them from the file, 
and 
the methods to output and display the best-fit results, e.g., model spectra and properties. 

