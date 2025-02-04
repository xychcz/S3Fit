# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Contact: xiaoyang.chen.cz@gmail.com

from .fit_frame import FitFrame
from .config_frame import ConfigFrame
from .phot_frame import PhotFrame

from .model_frames import *

from .auxiliary_func import print_time, lamb_air_to_vac, convert_linw_to_logw, convolve_spec_logw
from .extinct_law import ExtLaw

__all__ = ["FitFrame", "ConfigFrame", "PhotFrame", 
           "SSPFrame", "ELineFrame", "AGNFrame", "TorusFrame", 
           "print_time", "lamb_air_to_vac", "convert_linw_to_logw", "convolve_spec_logw", 
           "ExtLaw"
           ]

# print('v2, 240306: (1) [NII] broad 2; (2) add option of fit of raw flux (non-mocked)')
# print('v3, 241029: (1) [NI]5200; (2) tie Balmer lines with AV; (3) limit [SII] ratio')
# print('v4, 241116: (1) AGN PL; (2) rebuild')
# print('v4.1, 241119: (1) AGN PL; (2) rebuild; (3) components examine')
# print('v5, 241211: (1) fit weight cor; (2) add flux_scale; ')
# print('v5.1, 241217: (1) PhotFrame (2) Rename')
# print('v5.2, 250120: (1) Joint fit (2) Add torus')
# print('v1, 250121: (1) S3Fit initialized')
# print('v1.2, 250125: (1) Update ConfigFrame')
# print('v1.4, 250126: (1) Support flexible SFH (2) Add SFH output function'
# print('v1.5, 250128: (1) First release of S3Fit')
# print('v1.6, 250131: (1) Update ELineFrame line setup')
print('v2.0, 250204: (1) Split into multiple files; (2) Support pip installation')