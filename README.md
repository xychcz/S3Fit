# S<sup>3</sup>Fit
**S<sup>3</sup>Fit**: a <ins>**S**</ins>imultaneous <ins>**S**</ins>pectrum and photometric-<ins>**S**</ins>ED <ins>**Fit**</ins>ting code for observation of galaxies

S<sup>3</sup>Fit is a python code for the analysis of observational data of extragalaxies, which can fit spectrum and multi-band photometric Spectral Energy Distribution (SED) simultaneously. It is written to improve the moderate constraints on properties of continuum models in a pure spectral fitting due to the limited wavelength coverage. 
S<sup>3</sup>Fit support multiple models with multiple components, 
and can handle complex systems with a mixed contribution of Active Galactic Nucleus (AGN) and its host galaxy
in both of continua and emission lines (e.g., narrow lines and broad outflow lines). 
The fitting strategy is optimized to enable an efficient solution of 
the best-fit results for several tens of parameters and model components. 

## Features of S<sup>3</sup>Fit
- Easy switch between pure spectral fitting and spectrum+SED joint fitting modes. 
- Support for flexible combination of multiple stellar populations with different star formation histories (SFH). 
- Support for flexible combination of multiple emission line components.
- Support for AGN continuum models in the UV/optical and IR wavelength range. 
- User-friendly functions to output and plot the fitting results. 
- High extensibility for adding new functions and components by users (e.g., new band filters, new SFH functions, new emission lines, and also new types of models). 

## Fitting strategy
The full fitting pipeline of S<sup>3</sup>Fit is shown in the following flowchart, 
with a detailed description of the [fitting strategy](manuals/fitting_strategy.md) in [manuals](manuals/). 
<p align="center"> <img src="/manuals/Flowcharts.png" width="1200">

An example of the fitting result of S<sup>3</sup>Fit is shown in the following plots. 
<p align="center"> <img src="https://github.com/user-attachments/assets/683f5837-d364-4a53-8113-a05d56f9ef5b" width="600">

## Usage
Please find guides in [manuals](manuals/) for [basic](manuals/basic_usage.md) and [advanced](manuals/advanced_usage.md) usages of this code. 
An example of the usage of S<sup>3</sup>Fit is provided in the [example](example/example.ipynb). 

## Dependency

S<sup>3</sup>Fit mainly depends on several most widely utilized repositories for science calculation, `scipy`, `numpy`, and `astropy`. 
The core requirement of S<sup>3</sup>Fit is the two functions `least_squares` and `lsq_linear` in `scipy.optimize`
(please read the [fitting strategy](manuals/fitting_strategy.md) for details). 
A strong dependency of S<sup>3</sup>Fit on these repositories is not expected. 
It is optional to run S<sup>3</sup>Fit with [PyNeb](http://research.iac.es/proyecto/PyNeb/), 
which is used to calculate intrinsic flux ratios of emission lines.

The test environment is list as follows. 
```
python = 3.10
scipy = 1.12.0
numpy = 1.26.4
astropy = 6.0.0
matplotlib = 3.9.1
pyneb = 1.1.23 (optional)
```

## Future updating
- Add ISM dust and synchrotron models (you may also add them or other models by yourself following the [advanced usage](manuals/advanced_usage.md)) manual.
- Add iron pseudo continuum templates and Balmer continuum of type-1 AGN.
- Test support for absorption lines. 

## Citation
If you would like to use S<sup>3</sup>Fit, please cite the paper [Chen et al. (2025)][1], in which a pure-spectral-fit mode of this code is firstly utilized. 
Please find details of the configuration of star formation history and kinematic parameters of emission lines in the paper. 

## Reference
S<sup>3</sup>Fit uses the Single Stellar Population (SSP) library [HR-pyPopStar][2] ([paper][3]). 
Please download the [HR-pyPopStar library][2] and run the [converting code](model_libraries/convert_popstar_ssp.py) 
to create the SSP models used for S<sup>3</sup>Fit. 
You may also want to download an example of the converted SSP model for test in [this link][7].

S<sup>3</sup>Fit uses the [SKIRTor][4] ([paper1][5], [paper2][6]) AGN torus model. 
Please download the [SKIRTor library][4] and run the [converting code](model_libraries/convert_skirtor_torus.py) 
to create the torus models used for S<sup>3</sup>Fit. 
Example of this library is also provided in [model libraries](model_libraries/) for a test of S<sup>3</sup>Fit, 
which contains the templates with a fixed dust density gradient in radial (p = 1) and angular direction (q = 0.5). 
Please refer to [SKIRTor][4] website for details of the model parameters. 

[1]: <https://iopscience.iop.org/article/10.3847/1538-4357/ad93ab>
[2]: <https://www.fractal-es.com/PopStar/>
[3]: <https://academic.oup.com/mnras/article/506/4/4781/6319511>
[4]: https://sites.google.com/site/skirtorus/sed-library?authuser=0
[5]: http://adsabs.harvard.edu/abs/2012MNRAS.420.2756S
[6]: http://adsabs.harvard.edu/abs/2016MNRAS.458.2288S
[7]: https://drive.google.com/file/d/1JwdBOnl6APwFmadIX8BYLcLyFNZvnuYg/view?usp=share_link

