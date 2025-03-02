# List of all parameters of S<sup>3</sup>Fit
> [!NOTE]
> 'Parameters' here means the input arguments of `FitFrame` (the main framework of S<sup>3</sup>Fit).
> Please do not be confused with the non-linear fitting 'parameters' of models discussed in  [fitting strategy](./fitting_strategy.md).

### Input spectroscopic data
- `spec_wave_w`: Wavelength of the input spectrum, in unit of angstrom.
- `spec_flux_w` and `spec_ferr_w`: Fluxes and measurement errors of the input spectrum, in unit of erg s<sup>-1</sup> cm<sup>-2</sup> angstrom<sup>-1</sup>.
- `spec_R_inst_w`: Instrumental spectral resolution ($\lambda/\Delta\lambda$) of the input spectrum,
   this is used to convolve the model spectra and estimate the intrinsic velocity width. 
  `spec_R_inst_w` can be a list of variable resolutions as a function of the input wavelength `spec_wave_w`, 
   or given as a constant value as `spec_R_inst_w=[wave,R]` to specify the resolution `R` at the wavelength `wave` (in angstrom). 
- `spec_valid_range`: Valid wavelength range. For example, if 5000--7000 and 7500--10000 angstrom are used in fitting, set `spec_valid_range=[[5000,7000], [7500,10000]]`.
- `spec_flux_scale`: Scaling ratio of the input flux (e.g., `spec_flux_scale=1e-15`) to avoid too small values of flux in the fitting. 
  `spec_flux_scale` is not mandatory and it can be determined automatically if setting `spec_flux_scale=None` (default). 
### Input photometric data
- `phot_name_b`: List of band names of the input photometric data, e.g., `phot_name_b=['SDSS_gp','2MASS_J','WISE_1']`.
   The names should be the same as the filenames of the transmission curves in each band, e.g., `'SDSS_gp.dat'`. 
- `phot_flux_b` and `phot_ferr_b`: Fluxes and measurement errors in each band. 
- `phot_fluxunit`: Flux unit of `phot_flux_b` and `phot_ferr_b`, can be `'mJy'` (default) and `'erg/s/cm2/AA'`.
   If the input data is in unit of 'mJy', they will be converted to 'erg/s/cm2/AA' before the fitting.
- `phot_trans_dir`: Directory of files of the transmission curves.
- `sed_wave_w` and `sed_waveunit`: Wavelength array and its unit of the full SED wavelength range,
   which are used to create the model spectra and convert them to fluxes in each band.
  `sed_waveunit` can be `'angstrom'` and `'micron'`; if set to 'micron', they will be converted to 'angstrom'. 
   Note that `sed_wave_w` is not mandatory.
   S<sup>3</sup>Fit can create the wavelength array to cover all of the transmission curves of the used bands with either of the following two parameters:
   
> [!TIP]
> If a pure spectral fitting is required, please set `phot_name_b=None` or just remove all input parameters starting with `phot_` and `sed_` from the input parameters of `FitFrame`. 

> [!NOTE]
> When the joint fitting for spectrum and photometric SED is performed, the $\chi^2$ value is calculated with modified flux errors by adding 10% of the corresponding fluxes
> for both of the input spectrum and the photometric data in each band. The purpose is to account for the calibration uncertainty among different instruments. 

#### Model setup 
`v0_redshift`: Initial guess of the systemic redshift. The velocity shifts of all models are in relative to the input `v0_redshift`. 
`model_config`: Dictionary of model configurations, see [model setup](#configure-models) section for details. 
#### Fitting setup
`num_mock_loops`: Number of the mocked spectra, which is used to estimate the uncertainty of best-fit results. Default is `0`, i.e., only fit the raw data. \
`fit_raw`: Whether or not to fit the raw data. Default is `True`. If set to `False`, the code only output results for the mocked spectra. \
`multinst_reverr_ratio`: The ratio to scale the original error to account calibration uncertainties across multiple instruments. 
Default is `0.1`, i.e., revised errors by adding 10% of the corresponding fluxes are used in the fitting, 
i.e., to calculate $\chi^2$ values and to search for the best-fit by minimizing $\chi^2$ values 
(please refer to [fitting strategy](./fitting_strategy.md) for details). \
`multinst_reverr_mock`: If set to `True` then the revised errors are also used to create mocked spectra for estimation of parameter uncentainties. 
Default is `False`, i.e., mocked spectra are generated with original measurement errors. \
`fit_grid`: Set `fit_grid='linear'` (default) to run the fitting in linear flux grid.
Set `fit_grid='log'` to run the fitting in logarithmic flux grid. 
The reduced $\chi^2$ value is calculated as follows in the two cases,
```math
\large
\chi_{\nu,\mathrm{linear}}^2 = \sum_i{w_i \left[\frac{d_i}{e_i} \left(\frac{m_i}{d_i}-1 \right) \right]^2}, \ \ \ \ \ \ 
\chi_{\nu,\mathrm{log}}^2 = \sum_i{w_i \left[\frac{d_i}{e_i} \ln{ \left(\frac{m_i}{d_i} \right) } \right]^2},
```
where $d_i$ and $m_i$ are the data and model fluxes in the $i$-th wavelength;
$e_i$ is the error of flux; $w_i$ is the weight to account for the data resampling and degree of freedom
(please refer to [fitting strategy](./fitting_strategy.md) for details). 
Note that if emisison line is the only fitting model (e.g., for continuum subtracted data spectrum), `fit_grid` is always set to `'linear'`.
<!-- `max_fit_ntry`=3 `accept_chi_sq`=5 -->

#### Auxiliary
`plot_step`: Whether or not to plot the best-fit model spectra and fitting residuals in each intermediate step. Default is `False`. \
`print_step`: Whether or not to print the information each intermediate step (e.g., the examination of each model component). Default is `True`. \
`verbose`: Whether or not to print the running information of least-square solvers. Default is `False`. 

