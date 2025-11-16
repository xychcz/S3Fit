# S<sup>3</sup>Fit
**S<sup>3</sup>Fit**: a <ins>**S**</ins>imultaneous <ins>**S**</ins>pectrum and photometric-<ins>**S**</ins>ED <ins>**Fit**</ins>ting code for observation of galaxies

S<sup>3</sup>Fit is a Python-based tool for analyzing observational data of galaxies.
It offers powerful capabilities for decomposing spectroscopic data 
by supporting multiple continuum and emission line models with multiple components, 
making it well-suited for complex systems with mixed contributions 
from Active Galactic Nuclei (AGNs) and their host galaxies.
By simultaneously fitting the spectrum and multi-band photometric Spectral Energy Distribution (SED), 
S<sup>3</sup>Fit improves constraints on continuum model properties, 
which may be poorly determined when fitting spectral data alone due to its limited wavelength coverage.
With an optimized fitting strategy, 
S<sup>3</sup>Fit efficiently derives the best-fit solution for dozens of model parameters. 
Additionally, it provides an extensible and user-friendly framework, 
allowing users to modify model configurations and incorporate new model components as needed.

## Features of S<sup>3</sup>Fit
- Easy switch between pure spectral fitting and joint spectrum+SED fitting modes.
- Support for flexible combination of multiple stellar populations with different star formation histories (SFH). 
- Support for flexible combination of multiple emission line components.
- Support for AGN continuum models across UV/optical and IR wavelength ranges.
- User-friendly functions for outputting and visualizing fitting results.
- Highly extensible framework, allowing users to add new features such as new SFH functions, emission lines, and custom model types.

## Fitting strategy
The full fitting pipeline of S<sup>3</sup>Fit is shown in the following flowchart, 
with a detailed description of the [fitting strategy](https://github.com/xychcz/S3Fit/blob/main/manuals/fitting_strategy.md) in 
[manuals](https://github.com/xychcz/S3Fit/blob/main/manuals/). 
<p align="center"> <img src="https://github.com/user-attachments/assets/e84119ec-931c-49c8-8639-69217ff8bb38" width="1200">

An example of the fitting result of S<sup>3</sup>Fit is shown in the following plots (and details in [this paper](https://arxiv.org/abs/2510.02801)). 
<p align="center"> <img src="https://github.com/user-attachments/assets/683f5837-d364-4a53-8113-a05d56f9ef5b" width="600">

## Usage
Please find guides in [manuals](https://github.com/xychcz/S3Fit/blob/main/manuals/) 
for [basic](https://github.com/xychcz/S3Fit/blob/main/manuals/basic_usage.md) 
and [advanced](https://github.com/xychcz/S3Fit/blob/main/manuals/advanced_usage.md) usages of this code. 
An example of the usage of S<sup>3</sup>Fit is provided in the 
[example](https://github.com/xychcz/S3Fit/blob/main/examples/example_galaxy.ipynb). 

## Installation

You can install S<sup>3</sup>Fit with [pip](https://pypi.org/project/s3fit/):
```
pip install s3fit
```
S<sup>3</sup>Fit mainly depends on several most widely utilized repositories for science calculation, `scipy`, `numpy`, and `astropy`. 
The core requirement of S<sup>3</sup>Fit is the two functions `least_squares` and `lsq_linear` in `scipy.optimize`
(please read the 
[fitting strategy](https://github.com/xychcz/S3Fit/blob/main/manuals/fitting_strategy.md) for details). 
A strong dependency of S<sup>3</sup>Fit on these repositories is not expected. 
`joblib` is required to run fitting in multithreading. 
It is optional to run S<sup>3</sup>Fit with [PyNeb](http://research.iac.es/proyecto/PyNeb/), 
which is used to calculate intrinsic flux ratios of emission lines:
```
pip install s3fit[pyneb]
```

Dependencies:
```
python >= 3.10
scipy >= 1.12.0
numpy >= 1.26.4
astropy >= 6.0.0
joblib >= 1.5.2
matplotlib >= 3.9.1
pyneb >= 1.1.23 (optional)
```

## Future updating
- Add ISM dust and synchrotron models (you may also add them or other models by yourself following the
  [advanced usage](https://github.com/xychcz/S3Fit/blob/main/manuals/advanced_usage.md) manual).
- Add iron pseudo continuum templates and Balmer continuum of type-1 AGN.
- Test support for absorption lines. 

## Citation
If you would like to use S<sup>3</sup>Fit, please cite with the following BibTeX code:
```
@software{2025ascl.soft03024C,
       author = {{Chen}, Xiaoyang},
        title = "{S3Fit: Simultaneous Spectrum and photometric-SED Fitting code for galaxy observations}",
 howpublished = {Astrophysics Source Code Library, record ascl:2503.024},
         year = 2025,
        month = mar,
          eid = {ascl:2503.024},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025ascl.soft03024C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
 <!--
please cite the paper [Chen et al. (2025)][1], in which a pure-spectral-fit mode of this code is firstly utilized. 
Please find details of the configuration of star formation history and kinematic parameters of emission lines in the paper. 
-->

## Reference
S<sup>3</sup>Fit uses the Single Stellar Population (SSP) library [HR-pyPopStar][2] ([paper][3]). 
Please download the [HR-pyPopStar library][2] and run the 
[converting code](https://github.com/xychcz/S3Fit/blob/main/model_libraries/convert_popstar_ssp.py) 
to create the SSP models used for S<sup>3</sup>Fit. 
You may also want to download an example of the converted SSP model for test in [this link][7].

S<sup>3</sup>Fit uses the [SKIRTor][4] ([paper1][5], [paper2][6]) AGN torus model. 
Please download the [SKIRTor library][4] and run the 
[converting code](https://github.com/xychcz/S3Fit/blob/main/model_libraries/convert_skirtor_torus.py) 
to create the torus models used for S<sup>3</sup>Fit. 
Example of this library is also provided in 
[model libraries](https://github.com/xychcz/S3Fit/blob/main/model_libraries/) for a test of S<sup>3</sup>Fit, 
which contains the templates with a fixed dust density gradient in radial (p = 1) and angular direction (q = 0.5). 
Please refer to [SKIRTor][4] website for details of the model parameters. 

[1]: <https://iopscience.iop.org/article/10.3847/1538-4357/ad93ab>
[2]: <https://www.fractal-es.com/PopStar/>
[3]: <https://academic.oup.com/mnras/article/506/4/4781/6319511>
[4]: https://sites.google.com/site/skirtorus/sed-library?authuser=0
[5]: http://adsabs.harvard.edu/abs/2012MNRAS.420.2756S
[6]: http://adsabs.harvard.edu/abs/2016MNRAS.458.2288S
[7]: https://drive.google.com/file/d/1JwdBOnl6APwFmadIX8BYLcLyFNZvnuYg/view?usp=share_link

