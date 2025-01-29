# Advanced usage

## Support new band filters

The transmission curve supported by S<sup>3</sup>Fit needs to have 
two columns, wavelengths (in angstrom) and transmission values.
Save the curve with a filename of `Bandname.dat` and put it in the directory set in `phot_trans_dir`, 
and then the new band can be used in S<sup>3</sup>Fit. 

## Switch extinction laws

The default extinction law of S<sup>3</sup>Fit is [Calzetti00](http://www.bo.astro.it/~micol/Hyperz/old_public_v1/hyperz_manual1/node10.html).
If you would like to use another extinction law, please navigate to the `Extinction Functions` section of the S<sup>3</sup>Fit, 
define the the new extinction function that output $A_\lambda/A_V$, 
and remember to specify the new extinction law as the default one by modifying `ExtLaw = ExtLaw_NEW`. 

## Support new Star Formation History (SFH) functions

If you would like to use a new SFH function, please navigete to the `sfh_factor()` function in `SSPModels` class in `s3fit.py`, 
and append the new SFH function after the `'constant'` SFH.
The time variable of SFH function is already defined as `evo_time`. 
The parameters used for SFH starts from the 3rd parameter in the input `ssp_config` 
(the 0th to 2nd parameters are used for velocity shift, FWHM, and extinction). 
> [!IMPORTANT]
> Please remember to confirm the number of the input parameters match the required one in the new SFH function.

## Change to a different Single Stellar Population (SSP) library

The current version of S<sup>3</sup>Fit uses the [HR-pyPopStar][PopSTAR_web] SSP library with an initial mass function (IMF) of Kroupa (2002). 
If you tend to choose another IMF for HR-pyPopStar SSP library, please download the models from [the link](https://www.fractal-es.com/PopStar/#hr_py_download)
and re-run the [converting code](../models/convert_popstar_ssp.py) to create the SSP models used for S<sup>3</sup>Fit. 

[PopSTAR_web]: <https://www.fractal-es.com/PopStar/>

If you would like to utilize a different SSP library, you can modify the `read_ssp_library()` function in the `SSPModels` class.
In order to utilize the auxiliary functions in S<sup>3</sup>Fit, please ensure the new SSP model library has a three-dimentional shape, 
e.g., `orig_flux_zaw[i_z,i_a,i_w]` to represent the flux at the `i_w`-th wavelength value for the model
with the `i_z`-th metallicity and `i_a`-th age. 
The model library `orig_flux_mw` to be used by S<sup>3</sup>Fit can be converted as
```python
orig_flux_mw = orig_flux_zaw.reshape(num_metallicities, num_ages, num_wavelengths)
```
In order to calculate the best-fit total stellar mass and reconstruct the best-fit SFH, 
please make sure the model spectra is normalized by one solar mass in the unit of L<sub>sun</sub> per angstrom. 

## Add new emission lines

Please navigate to the `set_linelist()` function in the `ELineModels` class in `s3fit.py` to add new emission lines.
The rest wavelengths and names are stored in the lists `line_rest_n` and `line_name_n`
(note that the rest wavelength is given in vacuum).
If lineA is tied to lineB with a fixed flux ratio, 
set `linked_to_n` of lineA to `line_rest_n` of lineB, and `linked_ratio_n` of lineA to the flux ratio of lineA/lineB. 
If the flux of lineA is a free parameter, set `linked_to_n` and `linked_ratio_n` of lineA to `-1`. 
The follow coding block exhibits the example with [OIII] doublets, 
where [OIII]a is tied to [OIII]b with a flux ratio of 0.335
(please read the <ins>**Emission lines**</ins> section in [basic usage](../manuals/basic_usage.md) for calculation of the flux ratio).  
```python
self.line_rest_n.append(5008.240); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[OIII]b')
self.line_rest_n.append(4960.295); self.linked_to_n.append(5008.240); self.linked_ratio_n.append(0.335)  ; self.line_name_n.append('[OIII]a')
```

## Support new types of models

Please follow these steps to add a new model into S<sup>3</sup>Fit. 

#### Create a new ModelFrame
`ModelFrame` is the class to handle the model templates and to return the reduced model spectra back to the main fitting functions. 
The following block shows a rough structure for a new model.
```python
class NewModels(object):
    def __init__(self, filename=None, cframe=None, v0_redshift=None, spec_R_inst=None):
        self.filename = filename
        self.cframe = cframe
        self.v0_redshift = v0_redshift
        self.spec_R_inst = spec_R_inst

        self.num_coeffs = # calculate the value

    def models_unitnorm_original(self, pars):
        # For given parameters, read the intrinsic models from the input library file,
        # or calculate them from a given function.

        # Resample the models to log wavelength grid (for later convolution).
        self.logw_wave_w =
        logw_flux_int_mw = 
        return logw_flux_int_mw

    def models_unitnorm_obsframe(self, obs_wave_w, input_pars, if_pars_flat=True, spec_R_inst=None):
        if if_pars_flat: 
            pars = self.cframe.flat_to_arr(input_pars)
        else:
            pars = copy(input_pars)
        if spec_R_inst is None: spec_R_inst = self.spec_R_inst

        for i_comp in range(pars.shape[0]):
            logw_flux_int_mw = self.models_unitnorm_original(pars[i_comp,3:])
            # dust extinction
            logw_flux_e_mw = logw_flux_int_mw * 10.0**(-0.4 * pars[i_comp,2] * ExtLaw(self.logw_wave_w))
            # redshift models
            z_ratio = (1 + self.v0_redshift) * (1 + pars[i_comp,0]/299792.458) 
            logw_wave_z_w = self.logw_wave_w * z_ratio
            logw_flux_ez_mw = logw_flux_e_mw / z_ratio
            # convolve with intrinsic and instrumental dispersion if spec_R_inst is not None
            if spec_R_inst is not None:
                sigma_disp = pars[i_comp,1] / np.sqrt(np.log(256))
                sigma_inst = 299792.458 / spec_R_inst / np.sqrt(np.log(256))
                sigma_conv = np.sqrt(sigma_disp**2+sigma_inst**2)
                logw_flux_ezc_mw = convolve_spec_logw(logw_wave_z_w, logw_flux_ez_mw, sigma_conv, axis=1)
            else:
                logw_flux_ezc_mw = logw_flux_ez_mw 
            # project to observed wavelength
            obs_flux_scomp_mw = []
            for i_model in range(logw_flux_ezc_mw.shape[0]):
                obs_flux_scomp_mw.append(np.interp(obs_wave_w, logw_wave_z_w, logw_flux_ezc_mw[i_model,:]))
            obs_flux_scomp_mw = np.array(obs_flux_scomp_mw)
            if i_comp == 0: 
                obs_flux_mcomp_mw = obs_flux_scomp_mw
            else:
                obs_flux_mcomp_mw = np.vstack((obs_flux_mcomp_mw, obs_flux_scomp_mw))
        return obs_flux_mcomp_mw
```
A `ModelFrame` has three basic functions, `__init__()`, `models_unitnorm_original()`, and `models_unitnorm_obsframe()`. 

The most important value in `__init__()` is `num_coeffs`, which is the number of models for which the normalization factors are free
(i.e., not tied to the other models). 
For the example of stellar continuum models with `'nonparametric'` SFH, `num_coeffs=424` is the number of the full HR-pyPopStar library. 
For the example of stellar continuum models with two components that have `'exponential'` and `'constant'` SFH, respectively, 
`num_coeffs=2*4` since the models with different stellar ages are regulated with the SFH functions. 
For emission line models, `num_coeffs` is the sum of numbers of free lines for which the flux is not tied to the other ones. 
For AGN powerlaw and torus models, `num_coeffs=1` since typically only one component with one model is generated for a given parameter set. 

The purpose of `models_unitnorm_original()` function is to return the original model spectra `logw_flux_int_mw` 
for a given parameter set of a given component.
The original model spectra can be read from an input model library (e.g., SSP models) with the location specified in `filename`;
or created with a function (e.g., gaussian-profile emission lines, powerlaw or blackbody functions). 
`logw_flux_int_mw[i_m,i_w]` denotes the intrinsic model flux at the `i_w`-th wavelength value
for the `i_m`-th model. 
The value of `i_m` starts from `0` to `num_coeffs-1` defined in `__init__()`. 
`logw_` means the original model spectra is re-projected to a wavelength array spaced evenly on a log scale, 
to speed up the convolution in the later steps to account the instrumental and physical dispersion. 
The conversion can be performed with the following function in `s3fit.py`:
```python
logw_wave_w, logw_flux_w = convert_linw_to_logw(linw_wave_w, linw_flux_w, resolution=spec_R_rsmp)
```
, where `spec_R_rsmp` is the resampling resolution, can be set to `spec_R_inst * 3` to achieve a good accuracy. 
It is better to normalize the model spectra to a unit value, e.g., intrinsic flux = 1 at 5500 angstrom for stellar and AGN powerlaw continua, 
and velocity-integrated observed flux = 1 for emission lines.
With the normalization, the best-fit normalization factors are directly the corresponding fluxes of each models. 

The function `models_unitnorm_obsframe()` is used to return the extinct, redshifted, and convolved (with both of instrumental and physical broadening)
model spectra in the observed wavelength grid, to the main fitting functions to calculate the best-fit normalization factors.
Please read the [fitting strategy](../manuals/fitting_strategy.md) to learn about the details. 
You may not need to largely modify the `models_unitnorm_obsframe()` function since it can be used uniformly for multiple types of models. 

#### Create the configuration dictionary
The format of the input model configuration is as follows:
```python
new_config = {'comp0': {'pars': [[min0, max0, tie0], [min1, max1, tie1], [min2, max2, tie2], ... ], 
                        'info': { } }, 
              'comp1': {'pars': [[min0, max0, tie0], [min1, max1, tie1], [min2, max2, tie2], ... ], 
                        'info': { } },
              ..., }
```
Basically the 0th parameter is always the velocity shift to redshift the model spectra to the observed wavelength grid. 
The 1st parameter is velocity width if the spectral feature can be used to determine the physical dispersion. 
The 2nd parameter is typically set to the extinction value. 
The 3rd and later parameters are used to control the model shapes, e.g., the spectral index of AGN powerlaw. 
Please read the <ins>**Configure Models**</ins> section in [basic usage](../manuals/basic_usage.md)
for examples of different configurations. 

Please also remember to update the total `model_config` with:
```python
model_config['new'] = {'enable': True, 'config': new_config, 'file': new_file}
```
`'file'` is not required if the new model does not need an input template, e.g., AGN powerlaw continuum and emission lines. 

#### Initial the new model in FitFrame

The models are initialized in `FitFrame` (the main fitting class) with the `load_models()` function. 
Please add the following coding block into the `load_models()` function with the `NewModels` class defined above. 
```python
mod = 'new'
if np.isin(mod, [*model_config]):
    if model_config[mod]['enable']: 
        self.full_model_type += mod + '+'
        self.model_dict[mod] = {'cf': ConfigFrame(model_config[mod]['config'])}
        self.model_dict[mod]['specmod'] = NewModels(filename=model_config[mod]['file'], w_min=self.spec_wmin, w_max=self.spec_wmax, 
                                                    cframe=self.model_dict[mod]['cf'], v0_redshift=self.v0_redshift, spec_R_inst=self.spec_R_inst) 
        if self.have_phot:
            self.model_dict[mod]['sedmod'] = NewModels(filename=model_config[mod]['file'], w_min=self.sed_wmin, w_max=self.sed_wmax, 
                                                       cframe=self.model_dict[mod]['cf'], v0_redshift=self.v0_redshift, spec_R_inst=self.spec_R_inst) 
```

Now you can re-run the fitting with the new models. Please check [basic usage](../manuals/basic_usage.md) for the setup of `FitFrame`. 
```python
FF = FitFrame(......)
FF.main_fit()
```

#### Handle the best-fit results of the new model

The best-fit results for the new model can be obtained from the following code block. 
```python
self = FF
n_loops = self.num_mock_loops
new_mod, new_cf = self.model_dict['new']['specmod'], self.model_dict['new']['cf']
num_new_comps = new_cf.num_comps
num_new_pars = new_cf.num_pars
num_new_coeffs = int(new_mod.num_coeffs / num_new_comps)
fx0, fx1, fc0, fc1 = self.model_index('new', self.full_model_type)
new_x_lcp = self.best_fits_x[:, fx0:fx1].reshape(n_loops, num_new_comps, num_new_pars)
new_coeff_lcm = self.best_coeffs[:, fc0:fc1].reshape(n_loops, num_new_comps, num_new_coeffs)
```
In the results, `new_x_lcp[i_l, i_c, i_p]` denotes the best-fit value of the `i_p`-th parameter of the `i_c`-th components for the `i_l`-th mocked spectra.
The names of the `i_p`-th parameter follow the order set in the `new_config`. 

`new_coeff_lcm[i_l, i_c, i_m]` denotes the best-fit normalization factor of the `i_m`-th model of the `i_c`-th components for the `i_l`-th mocked spectra.
The meanning and unit of `new_coeff_lcm` depends on the normalization in `models_unitnorm_original()` function in the `NewModels` class. 
For example, if the intrinsic model spectra are normalized to unit flux at rest 5500 angstrom in `models_unitnorm_original()`, 
the value of `new_coeff_lcm[i_l, i_c, i_m]` is the the best-fit intrinsic flux at rest 5500 angstrom
of the `i_m`-th model of the `i_c`-th components for the `i_l`-th mocked spectra.
The intrinsic flux can be converted to intrinsic luminosity by multiplying `unitconv`, where `unitconv` is calculated as:
```python
dist_lum = cosmo.luminosity_distance(self.v0_redshift).to('cm').value
unitconv = 4*np.pi*dist_lum**2 / const.L_sun.to('erg/s').value * self.spec_flux_scale 
```

The best-fit total model spectrum `new_spec_w` of the `i_c`-th components for the `i_l`-th mocked spectra can be obtained with the following code block:
```python
new_spec_mw = self.model_dict['new']['specfunc'](self.spec['wave_w'], self.best_fits_x[i_l, fx0:fx1])
new_spec_cmw = new_spec_mw.reshape(num_new_comps, num_new_coeffs, new_spec_mw.shape[1])
new_spec_w = np.dot(new_coeff_lcm[i_l,i_c], new_spec_cmw[i_c])
```

