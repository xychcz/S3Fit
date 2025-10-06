# List of all parameters of S<sup>3</sup>Fit
> [!NOTE]
> 'Parameters' here means the input arguments of `FitFrame` (the main framework of S<sup>3</sup>Fit).
> Please do not be confused with the non-linear fitting 'parameters' of models discussed in  [fitting strategy](./fitting_strategy.md).

> [!NOTE]
> The code is actively under development. Please double-check the manuals archived in the GitHub release for a specific version if you encounter any discrepancies.

### Input spectroscopic data
- `spec_wave_w` (list or numpy array of floats, <ins>**required**</ins>) \
   Wavelength of the input spectrum, in unit of angstrom.
- `spec_flux_w` and `spec_ferr_w` (list or numpy array of floats, <ins>**required**</ins>) \
   Fluxes and measurement errors of the input spectrum, in unit of erg s<sup>-1</sup> cm<sup>-2</sup> angstrom<sup>-1</sup>.
- `spec_R_inst_w` (list or numpy array of floats, or 2-element list, <ins>**required**</ins>) \
   Instrumental spectral resolution ($\lambda/\Delta\lambda$) of the input spectrum,
   this is used to convolve the model spectra and estimate the intrinsic velocity width. 
  `spec_R_inst_w` can be a list of variable resolutions as a function of the input wavelength `spec_wave_w`, 
   or given as a constant value as `spec_R_inst_w=[wave,R]` to specify the resolution `R` at the wavelength `wave` (in angstrom). 
- `spec_valid_range` (nested list of floats, optional) \
   Valid wavelength range.
   For example, if 5000--7000 and 7500--10000 angstrom are used in fitting, set `spec_valid_range=[[5000,7000], [7500,10000]]`.
   Default is `None`, in this case the entire input spectrum (except for the wavelengths with non-positive `spec_ferr_w`)
   will be used in the fitting. 
- `spec_flux_scale` (float, optional) \
   Scaling ratio of the input flux (e.g., `spec_flux_scale=1e-15`) to avoid too small values of flux in the fitting. 
  `spec_flux_scale` is not mandatory and it can be determined automatically if setting `spec_flux_scale=None` (default). 
### Input photometric data
- `phot_name_b` (list or numpy array of strings) \
   List of band names of the input photometric data, e.g., `phot_name_b=['SDSS_gp','2MASS_J','WISE_1']`.
   The names should be the same as the filenames of the transmission curves in each band, e.g., `'SDSS_gp.dat'`. 
- `phot_flux_b` and `phot_ferr_b` (list or numpy array of floats) \
   Fluxes and measurement errors in each band.
- `phot_trans_dir` (string) \
   Directory of files of the transmission curves.
> [!TIP]
> The above four parameters are only necessary if a simultaneous fitting of spectrum and photometric-SED is required.
> S<sup>3</sup>Fit will run in a pure-spectral fitting mode if these parameters are set to `None` (default). 
- `phot_fluxunit` (string, optional) \
   Flux unit of `phot_flux_b` and `phot_ferr_b`, can be `'mJy'` (default) and `'erg/s/cm2/AA'`.
   If the input data is in unit of 'mJy', they will be converted to 'erg/s/cm2/AA' before the fitting.
- `phot_calib_b` (list or numpy array of strings, optional) \
   List of band names of photometric data that is used for calibration of spectrum.
   For example, if 'SDSS_rp' and 'SDSS_ip' bands are covered by the spectrum,
   set `phot_calib_b=['SDSS_rp','SDSS_ip']`
   and S<sup>3</sup>Fit will scale the input `spec_flux_w` and `spec_ferr_w`
   with `phot_flux_b` in the two bands, e.g., to correct for aperture loss of the input spectrum. 
   Set `phot_calib_b=None` (default) if the calibration is not required.  
- `sed_wave_w` (list or numpy array of floats, optional) and `sed_waveunit` (string, optional) \
   Wavelength array and its unit of the full SED wavelength range,
   which are used to create the model spectra and convert them to fluxes in each band.
  `sed_waveunit` can be `'angstrom'` (default) and `'micron'`; if set to `'micron'`, the wavelengths will be converted to angstrom before the fitting. 
   Note that `sed_wave_w` is not mandatory.
   S<sup>3</sup>Fit can create the wavelength array to cover all of the transmission curves of the used bands with either of the following two parameters:
- `sed_wave_num` (int, optional) \
   Length of `sed_wave_w`. `sed_wave_w` will be spaced evenly on a log scale. 
- `phot_trans_rsmp` (int, optional) \
   Minimum number of data points to resample the transmission curves within their FWHM.
   Default is 10, i.e., at least 10 data points will be used to resample all of the transmission curves in the ranges with half of their maximum transmission values. 

### Model setup 
- `v0_redshift` (float, <ins>**required**</ins>) \
   Initial guess of the systemic redshift. The velocity shifts of all models are in relative to the input `v0_redshift`. 
- `model_config` (nested dictionary, <ins>**required**</ins>) \
   Dictionary of model configurations.
   Please refer to the [model configuration](../manuals/basic_usage.md#2-model-configuration) section
   in the manual of [basic usage](../manuals/basic_usage.md) for details. 
- `norm_wave` and `norm_width` (float, optional) \
   Wavelength and width (in angstrom) used to normalize continuum models.
   Default values are `5500` and `25` angstrom, respectively. 

### Control of fitting quality
- `num_mocks` (int, optional) \
   Number of the mock spectra for the Monte Carlo method.  
   The mock spectra are used to estimate the uncertainty of best-fit results. Default is `0`, i.e., only the original data will be fit.
- `inst_calib_ratio` (float, optional) \
   Initial ratio to estimate the calibration uncertainties across multiple instruments. 
   Default is `0.1`, i.e., the calibration erros are estimated as 10% of the corresponding fluxes.
-  `inst_calib_ratio_rev` (bool, optional) \
   'inst_calib_ratio' is the initial value of the scaling ratio,
   if `inst_calib_ratio_rev` is set to `True` (Default), the ratio will be iteratively refreshed in the fitting process
   (please refer to [fitting strategy](./fitting_strategy.md) for details).
- `inst_calib_smooth` (float, optional) \
   The convovling width (in velocity, km/s) to smooth the original observed spectrum
   in calculation of revised errors (i.e., to reflect calibration errors).
   The convolution is adopted to avoid involving artifacts into Monte Carlo mock spectra surrounding bright lines.
   Default is `10000` (km/s). Set it to `0` if you want to disable the convolution. 
- `examine_result` (bool, optional) \
   If set `examine_result=False`, the examination of S/N of models and the updating of fitting
   (i.e., the 2nd fitting steps in [fitting strategy](./fitting_strategy.md)) will be skipped.
   Default is `True`.
- `accept_chi_sq` (float, optional) \
   The accepted $\chi^2$ in the initial and intermediate fitting steps. Default is `3`.
   The accepted $\chi^2$ in the final fitting step will be dynamically chosen with $\chi^2$ of the progenitor steps. 
- `nlfit_ntry_max` (int, optional) \
   Maximum number of tries of non-linear fitting process ([fitting strategy](./fitting_strategy.md))
   to achieve the accepted $\chi^2$. Default is `3`.
- `init_annealing` (bool, optional) \
   Whether to perform Dual Annealing optimization to search for rough global minima
   in the initial and the 1st fitting steps ([fitting strategy](./fitting_strategy.md)).
   Default is `True`. 
- `da_niter_max` (int, optional) \
   Maximum number of iterations of the Dual Annealing optimization. Default is `10`.
   Since the purpose of the Dual Annealing optimization is to find a proper initial guess
   for the Non-linear Least-square optimazation, a small number of `da_niter_max` already works well.
   You may examine the results of the Dual Annealing optimization by setting `plot_step=True`. 
- `perturb_scale` (float, optional) \
   Perturbation scaling factor for the transferred parameters. 
   Default is `0.02`, i.e., an array of Gaussian noise with the sigma of 2% of the boundaries of parameters
   will be added to the transferred parameters (please refer to [fitting strategy](./fitting_strategy.md) for details). 
- `nllsq_ftol_ratio` (float, optional) \
   The factor to control the termination condition of the Non-linear Least-square optimazation (i.e., `scipy.optimize.least_squares`). 
   Default is `0.01`, i.e., the `ftol` parameter of `scipy.optimize.least_squares`
   will be given as 0.01*sqrt(n), where n is the number of input data in wavelength (combining both of spectroscopic and photometric data).
- `fit_grid` (string, optional) \
   Set `fit_grid='linear'` (default) to run the fitting in linear flux grid.
   Set `fit_grid='log'` to run the fitting in logarithmic flux grid.
   Note that if emisison line is the only fitting model (e.g., for the fitting of continuum-subtracted spectrum), `fit_grid` is always set to `'linear'`.
   (please refer to [fitting strategy](./fitting_strategy.md) for details). 
- `conv_nbin_max` (int, optional) \
   Maximum bin number to perform FFT accelerated convolution of continuum model spectra with variable Gaussian kernel widths (e.g., wavelength-dependent spectral resolution).
   Default is `5`, i.e., the convolution will be performed with kernel widths at 5 evenly-spaced wavelengths.
   The convolved spectrum at wavelengths between the 5 selected wavelengths will be interploted linearly using the
   spectra convolved with the kernel at the two neighboring selected wavelengths. 
   A small number of `conv_nbin_max` works well for a smooth function of wavelength-dependent resolution.
   Increasing the number will slow down the fitting process significantly. 

### Auxiliary
- `print_step` (bool, optional) \
   Whether or not to print the information each intermediate step (e.g., the examination of each model component).
   Default is `True`.
- `plot_step` (bool, optional) \
   Whether or not to plot the best-fit model spectra and fitting residuals in each intermediate step.
   Default is `False`. 
- `canvas` (tuple, optional) \
   Matplotlib window with a format of `canvas=(fig,axs)` to display each intermediate step dynamically.
   Please read the [Jupyter Notebook](../example/example.ipynb) for an example case. 
- `verbose` (bool, optional) \
   Whether or not to print the running information of the Linear and Non-linear Least-square solvers. Default is `False`.
- `save_per_loop` (bool, optional) and `output_filename` (string, optional)\
   If `save_per_loop` is set to `True`, the fitting results will be saved per Monte Carlo mock fitting loop
   to the file with path specified in `output_filename`. 
   
