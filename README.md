# S<sup>3</sup>Fit
**S<sup>3</sup>Fit**: a **S**imultaneous **S**pectrum and photometric-**S**ED **Fitting** code for extragalaxy

**Usage**
```python
import os, sys
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
phot_trans_dir =
# directory of transmission curve files.

# Models
# Stellar models from PopSTAR library
ssp_pmmc = [[[0, -1000, 1000], [600, 100, 1200], [0.5, 0, 5.0], '5.5e-3, None, solar_met']]
# Initial guesses and ranges of parameters of voff, fwhm, and AV.
# For example, [0, -1000, 1000] denotes the initial value of velocity shift is 0, and is allowed to vary from -1000 to 1000 km/s.
# The string at the end, '5.5e-3, None, solar_met', denotes the minimum (5,5e-3 Gyr) and maximum (None if set to Universe age) stellar ages,
# and metallicity (currently 'solar_met' or 'all' is supported)
ssp_file =
# location of the PopSTAR library (one example in ...)

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
torus_disc_file = 
torus_dust_file = 
# location of model library

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
              torus_pmmc=torus_pmmc, torus_disc_file=torus_disc_file, torus_dust_file=torus_dust_file,
              # setup of models, comment the corresponding line if a model is not required
              num_mock_loops=10,
              # number of fitting loops for mocked data to estimate uncertainties of parameters; set to 1 if only raw data is fit
              plot=True, verbose=False
              # if showing plots and texts of each steps of the fit
)
```

** Test environment **
```python
python = 3.10
scipy = 1.12.0
numpy = 1.26.4
astropy = 6.0.0
matplotlib = 3.9.1
```
