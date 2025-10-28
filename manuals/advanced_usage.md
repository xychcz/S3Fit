# Advanced usage

> [!NOTE]
> The code is actively under development. Please double-check the manuals archived in the GitHub release for a specific version if you encounter any discrepancies.

## 1. Support new band filters

The transmission curve supported by S<sup>3</sup>Fit needs to have 
two columns, wavelengths (in angstrom) and transmission values.
Save the curve with a filename of `Bandname.dat` and put it in the directory set in `phot_trans_dir`, 
and then the new band can be used in S<sup>3</sup>Fit. 

## 2. Switch extinction laws

The default extinction law of S<sup>3</sup>Fit is [Calzetti00](http://www.bo.astro.it/~micol/Hyperz/old_public_v1/hyperz_manual1/node10.html).
If you would like to use another extinction law, please 
add the extinction function that output $A_\lambda/A_V$ into `s3fit/extinct_law.py`, 
and remember to specify the new extinction law as the default one by modifying `ExtLaw = ExtLaw_NEW`. 

## 3. Support new Star Formation History (SFH) functions

If you would like to use a new SFH function, you can just define the function 
```python
def sfh_user(time, sfh_pars):
    t_peak = 10.0**sfh_pars[1]
    tau = 10.0**sfh_pars[2]
    return np.exp(-(time-t_peak)**2 / tau**2/2)
```
and add the name of the function into `ssp_config` by setting `'sfh_name'` to `'user'`.
```python
ssp_config = {'main': {'pars': [[-1000, 1000, 'free'], [100, 1200, 'free'], [0, 5.0, 'free'], 
                                [0, 0.94, 'free'], [0, 0.94, 'free'], [-1, 1, 'free']], 
                       'info': {'age_min': -2.25, 'age_max': 'universe', 'met_sel': 'solar', 'sfh_name': 'user', 'sfh_func': sfh_user} }, 
              'young': {'pars': [[None, None, 'ssp:main:0'], [None, None, 'ssp:main:1'], [None, None, 'ssp:main:2'], 
                                 [-2, -0.5, 'free'], [-2, -1.5, 'free'], [-2, -1, 'free']],
                        'info': {'age_min': -2.25, 'age_max': 0, 'met_sel': 'solar', 'sfh_name': 'user', 'sfh_func': sfh_user} } }
model_config['ssp'] = {'enable': True, 'config': ssp_config, 'file': ssp_file}
```
The example uses a Gaussian profile SFH for both of the main and young population (but with different parameter ranges).
In the definition of SFH function, `time` is the evolutionary time from the begining of the component, i.e., 
the age of the oldest single stellar population (SSP). 
`sfh_pars` is the parameter list for a given SFH. 
Note that `sfh_pars[0]` is always set as the age of the composite stellar population (CSP), 
please use `sfh_pars[1:]` for other parameters. 
In the above example, `sfh_pars[1]` is the peak time of the SFH, and `sfh_pars[2]` is the duration of the CSP. 
The parameters used for SFH starts from the 3rd parameter in the input `ssp_config`, e.g., `ssp_config['main']['pars'][3:]`, 
and the 0th to 2nd parameters are always used for velocity shift, FWHM, and extinction. 
> [!IMPORTANT]
> Please remember to confirm the number of the input parameters match the required one in the new SFH function;
> and make sure the multiple components have the same number of parameters, even if they use different SFH functions.
> Just fix the parameter to an arbitrary value if it is not required to calculate the SFH. 

## 4. Change to a different Single Stellar Population (SSP) library

The current version of S<sup>3</sup>Fit uses the [HR-pyPopStar][PopSTAR_web] SSP library with an initial mass function (IMF) of Kroupa (2002). 
If you tend to choose another IMF for HR-pyPopStar SSP library, please download the models from [the link](https://www.fractal-es.com/PopStar/#hr_py_download)
and re-run the [converting code](../model_libraries/convert_popstar_ssp.py) to create the SSP models used for S<sup>3</sup>Fit. 

[PopSTAR_web]: <https://www.fractal-es.com/PopStar/>

If you would like to utilize a different SSP library, you can modify the `read_ssp_library()` function in the `SSPModels` class (`s3fit/model_frames/ssp_frame.py`).
In order to utilize the auxiliary functions in S<sup>3</sup>Fit, please ensure the new SSP model library has a three-dimentional shape, 
e.g., `orig_flux_zaw[i_z,i_a,i_w]` to represent the flux at the `i_w`-th wavelength value for the model
with the `i_z`-th metallicity and `i_a`-th age. 
The model library `orig_flux_ew` to be used by S<sup>3</sup>Fit can be converted as
```python
orig_flux_ew = orig_flux_zaw.reshape(num_metallicities * num_ages, num_wavelengths)
```
In order to calculate the best-fit total stellar mass and reconstruct the best-fit SFH, 
please make sure the model spectra is normalized by one solar mass in the unit of L<sub>sun</sub> per angstrom. 

## 5. Add new emission lines

You can re-edit the emisison line configuration after initializing `FitFrame`.
```python
FF = FitFrame(......)
el_mod = FF.model_dict['el']['spec_mod']
```
If you would like to add new lines into the line list, just run `add_line()`.
If PyNeb is installed and `use_pyneb=True`, only the names of new lines are required, 
otherwise the rest wavelengths (in vacuum) are also required in the input parameters. 
```python
el_mod.add_line(['Lya','[O III]:4364'], use_pyneb=True)
el_mod.add_line(['Lya','[O III]:4364'], linerests=[1215.67,4364.44], use_pyneb=False)
```
You can add as many as new lines. S<sup>3</sup>Fit will only enable the lines that 
can be covered in the wavelength range of the input data spectrum. 
Use `delete_line()` if you would like to delete given lines from the list. 
```python
el_mod.delete_line(['Lya','[O III]:4364'])
```

You can also edit the line ties following the example of [OIII] doublets.
```python
el_mod.tie_pair('[O III]:4960', '[O III]:5008', use_pyneb=True)
el_mod.tie_pair('[O III]:4960', '[O III]:5008', ratio=0.335, use_pyneb=False)
```
In this case, [OIII]4960 is tied to [OIII]5008 with a flux ratio of [OIII]4960/5008. 
If PyNeb is installed and `use_pyneb=True`, only the names of new lines are required.
The line ratio used in the fit will be calculated based on the density and temperature of electrons.
If `use_pyneb=False`, the line ratio needs to be given as input and 
S<sup>3</sup>Fit use the fixed ratio in the fitting
(except for [SII]6718/6733, which ratio is calculated from electron density using 
the [Proxauf et al. (2014) equation](https://ui.adsabs.harvard.edu/abs/2014A%26A...561A..10P)
). 

If you want to remove a tying relation, just run:
```python
el_mod.release_pair('[O III]:4960')
```
Please remember to update the configuration of the emission line models as follows
after adding or removing tying relations.
```python
el_mod.update_freemask()
el_mod.update_lineratio()
```
> [!NOTE]
> In order to utilize PyNeb to identify line transitions and to calculate the emissivities,
> please use a naming format of `'Element Notation:Wavelength'` for permitted lines
> or `'[Element Notation]:Wavelength'` for forbidden lines.
> Wavelength can be given in angstrom or micron, e.g., `'H I:12820'` or `'H I:1.28um'` for Paschen $\beta$.
> For Hydrogen recombination lines, you can also use short names, e.g., `'Pab'` for Paschen $\beta$.
> The prefixes of short names of Lyman, Balmer, Paschen, and Brackett series are
> `'Ly'`, `'H'`, `'Pa'`, and `'Br'`, respectively.
> The suffixes of line transitions can be given as `'a'` ($\alpha$), `'b'` ($\beta$), `'g'` ($\gamma$), `'d'` ($\delta$),
> and upper levels of transitions (e.g., `'H7'` for the transition from n=7 to n=2). 

## 6. Support new types of models

S<sup>3</sup>Fit has a uniform framework for models (see [fitting strategy](./fitting_strategy.md)), 
which enables users to incorporate new models into S<sup>3</sup>Fit 
with the following steps. 

#### 6.1 Create a new ModelFrame
In the first step, please create a `ModelFrame` class for the new model and save it as `s3fit/model_frames/newmodel_frame.py`. 
`ModelFrame` is the class to handle the model templates and to return the reduced model spectra back to the main fitting functions. 
The following block shows a rough structure for a new model.
```python
import numpy as np
from scipy.interpolate import interp1d
from ..auxiliary_func import convolve_var_width_fft
from ..extinct_law import ExtLaw

class NewModelFrame(object):
    def __init__(self, filename=None, cframe=None, v0_redshift=None, R_inst_rw=None):
        self.filename = filename # if the model is input from a library of templates
        self.cframe = cframe # config frame, see section 6.2
        self.v0_redshift = v0_redshift # systemic redshift
        self.R_inst_rw = R_inst_rw # instrumental resolution, provided in FitFrame

        self.num_coeffs = # calculate the value

    def models_unitnorm_original(self, pars):
        # For given parameters, read the original models from the input library file,
        # or calculate them from a given function.
        self.orig_wave_w = # the original wavelength
        self.orig_flux_ew = # the original model spectra, _e represents different model element
        mask_e = select_models(pars) # select the model elements based on input parameters
        return self.orig_flux_ew[mask_e,:]

    def models_unitnorm_obsframe(self, obs_wave_w, input_pars, if_pars_flat=True, mask_lite_e=None, conv_nbin=None):
        if if_pars_flat: 
            pars = self.cframe.flat_to_arr(input_pars)
        else:
            pars = copy(input_pars)
        if mask_lite_e is not None:
            mask_lite_ce = self.cframe.flat_to_arr(mask_lite_e)

        for i_comp in range(pars.shape[0]):
            orig_flux_int_ew = self.models_unitnorm_original(pars[i_comp,3:])
            if mask_lite_e is not None: # limit element number for accelarate calculation
                orig_flux_int_ew = orig_flux_int_ew[mask_lite_ce[i_comp,:],:]

            # dust extinction
            orig_flux_d_ew = orig_flux_int_ew * 10.0**(-0.4 * pars[i_comp,2] * ExtLaw(self.orig_wave_w))

            # redshift models
            z_ratio = (1 + self.v0_redshift) * (1 + pars[i_comp,0]/299792.458) # (1+z) = (1+zv0) * (1+v/c)
            orig_wave_z_w = self.orig_wave_w * z_ratio
            orig_flux_dz_ew = orig_flux_d_ew / z_ratio

            # convolve with intrinsic and instrumental dispersion if self.R_inst_rw is not None
            if (self.R_inst_rw is not None) & (conv_nbin is not None):
                R_inst_w = np.interp(orig_wave_z_w, self.R_inst_rw[0], self.R_inst_rw[1])
                orig_flux_dzc_ew = convolve_var_width_fft(orig_wave_z_w, orig_flux_dz_ew, 
                                                          R_inst_w=R_inst_w, fwhm_vel=pars[i_comp,1], num_bins=conv_nbin)
            else:
                orig_flux_dzc_ew = orig_flux_dz_ew 

           # project to observed wavelength
            interp_func = interp1d(orig_wave_z_w, orig_flux_dzc_ew, axis=1, kind='linear', fill_value="extrapolate")
            obs_flux_scomp_ew = interp_func(obs_wave_w)
            if i_comp == 0: 
                obs_flux_mcomp_ew = obs_flux_scomp_ew
            else:
                obs_flux_mcomp_ew = np.vstack((obs_flux_mcomp_ew, obs_flux_scomp_ew))
        return obs_flux_mcomp_ew
```
A `ModelFrame` has three basic functions, `__init__()`, `models_unitnorm_original()`, and `models_unitnorm_obsframe()`. 

The most important value in `__init__()` is `num_coeffs`, which is the number of model elements 
for which the linear coefficients (i.e., normalization factors) are free (i.e., not tied to the other model elements). 
For the example of stellar continuum models with `'nonparametric'` SFH, `num_coeffs=424` is the number of the SSP templates in the full HR-pyPopStar library. 
For the example of stellar continuum models with two components that have `'exponential'` and `'constant'` SFH, respectively, 
`num_coeffs=2*4` since the models with different stellar ages are regulated with the SFH functions. 
For emission line models, `num_coeffs` is the sum of numbers of free lines for which the flux is not tied to the other ones. 
For AGN powerlaw and torus models, `num_coeffs=1` since typically only one component with one model element is generated for a given parameter set. 

The purpose of `models_unitnorm_original()` function is to return the original model spectra `orig_flux_ew` 
for a given parameter set of a given component.
The original model spectra can be read from an input model library (e.g., SSP models) with the location specified in `filename`;
or created with a function (e.g., gaussian-profile emission lines, powerlaw or blackbody functions). 
`orig_flux_ew[i_e,i_w]` denotes the original model flux at the `i_w`-th wavelength value
for the `i_e`-th model element. 
The value of `i_e` starts from `0` to `num_coeffs-1` defined in `__init__()`. 
It is better to normalize the model spectra to a unit value, e.g., intrinsic flux = 1 at 5500 angstrom for stellar and AGN powerlaw continua, 
and velocity-integrated observed flux = 1 for emission lines.
With the normalization, the best-fit linear coefficients are directly the corresponding fluxes of each models. 

The function `models_unitnorm_obsframe()` is used to calculate the extinct, redshifted, and convolved (with both of instrumental and physical broadening)
model spectra in the observed wavelength grid, and to return the model spectra to `FitFrame` to calculate the best-fit linear coefficients.
Please read the [fitting strategy](../manuals/fitting_strategy.md) to learn about the details. 
You may not need to largely modify the `models_unitnorm_obsframe()` function since it can be used uniformly for multiple types of models. 

#### 6.2 Create the configuration dictionary
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
Please read the [model configuration](../manuals/basic_usage.md#2-model-configuration)
section in the manual of [basic usage](../manuals/basic_usage.md)
for examples of different configurations. 

Please also remember to update the total `model_config` with:
```python
model_config['new'] = {'enable': True, 'config': new_config, 'file': new_file}
```
`'file'` is not required if the new model does not need an input template, e.g., Blackbody template, AGN powerlaw continuum or emission line profiless. 

#### 6.3 Initial the new model in FitFrame

The models are initialized in `FitFrame` (the main fitting class, in `s3fit/fit_frame.py`) with the `load_models()` function. 
Please add the following coding block into the `load_models()` function with the `NewModelFrame` class defined above. 
```python
mod = 'new'
if np.isin(mod, [*model_config]):
    if model_config[mod]['enable']:
        print_log(center_string('Initialize New models', 80), self.log_message)
        self.full_model_type += mod + '+'
        self.model_dict[mod] = {'cf': ConfigFrame(model_config[mod]['config'])}
        from .model_frames.newmodel_frame import NewModelFrame
        self.model_dict[mod]['spec_mod'] = NewModelFrame(filename=self.model_config[mod]['file'],
                                                         cframe=self.model_dict[mod]['cframe'], 
                                                         v0_redshift=self.v0_redshift, 
                                                         R_inst_rw=self.spec['R_inst_rw']) 
        if self.have_phot:
            self.model_dict[mod]['sed_mod'] = NewModelFrame(filename=self.model_config[mod]['file']
                                                            cframe=self.model_dict[mod]['cframe'], 
                                                            v0_redshift=self.v0_redshift, 
                                                            R_inst_rw=self.spec['R_inst_rw'])
```
Note that `NewModelFrame` is called twice to create model frames in the wavelength range of the input spectrum and the full SED range.
You can add additional parameters in `NewModelFrame` to constrain the wavelength range. 

#### 6.4 Handle the best-fit results of the new model

Now you can run the fitting with the new models. Please check [basic usage](../manuals/basic_usage.md) for the initialization of `FitFrame`. 
```python
FF = FitFrame(......)
FF.main_fit()
```

The best-fit results for the new model can be retrieved as:
- `FF.output_mc['new'][comp]['spec_lw'][i_l,i_w]`\
   Best-fit model spectra of the `comp` component of the ``new`` model at the `i_w`-th wavelength 
   from the `i_l`-th fitting loop (`i_l=0` for the original data and `i_l>=1` for the mock data). 
   Replace `comp` with `'sum'` to obtain the sum of all components of the `new` model. 
- `FF.output_mc['new'][comp]['sed_lw'][i_l,i_w]`\
   Best-fit model spectra of the `comp` component of the ``new`` model.
   The difference the above two results is that `'spec_lw'` returns the model spectra in the grid of the input spectral wavelength,
   while `'sed_lw'` returns the model spectra in the full SED wavelength range to cover all the input photometric bands. 
- `FF.output_mc['new'][comp]['par_lp'][i_l,i_p]`\
   Best-fit model parameters (non-linear) for the `i_p`-th parameter (e.g., `i_p=0` is typically the velocity shift) in the `i_l`-th fitting loop.
   The parameters are sorted following the order specified in the input `model_config` dictionary. 
- `FF.output_mc['new'][comp]['coeff_le'][i_l,i_e]`\
   Best-fit model coefficients (linear), i.e., normalization factors of each model elements,
   of the `i_e`-th model element of the `comp` component of the `'new'` models in the `i_l`-th fitting loop. 
   The values of the coefficients depends on how the models are normlized (see Section 6.1). 
