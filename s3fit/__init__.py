# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: s3fit@xychen.me

from .fit_frame import FitFrame
from .config_frame import ConfigFrame
from .phot_frame import PhotFrame
from .model_frames import *
from .auxiliary_func import print_log, center_string, convolve_fix_width_fft, convolve_var_width_fft
from .extinct_law import ExtLaw

__all__ = ["FitFrame", "ConfigFrame", "PhotFrame", 
           "StellarFrame", "LineFrame", "AGNFrame", "TorusFrame", 
           "print_log", "center_string", "convolve_fix_width_fft", "convolve_var_width_fft", 
           "ExtLaw"
           ]

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("your-package-name")
except PackageNotFoundError:
    __version__ = "0.0.0+local"