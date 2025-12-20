# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: s3fit@xychen.me

from .fit_frame import FitFrame, __version__

from .model_frames.stellar_frame import StellarFrame
from .model_frames.agn_frame import AGNFrame
from .model_frames.torus_frame import TorusFrame
from .model_frames.line_frame import LineFrame

from .auxiliaries.auxiliary_frames import ConfigFrame, PhotFrame
from .auxiliaries.auxiliary_functions import print_log, center_string, color_list_dict, convolve_fix_width_fft, convolve_var_width_fft
from .auxiliaries.extinct_laws import ExtLaw

__all__ = [
    '__version__', 'FitFrame', 'ConfigFrame', 'PhotFrame', 
    'StellarFrame', 'LineFrame', 'AGNFrame', 'TorusFrame', 
    'ExtLaw', 'print_log', 'center_string', 'color_list_dict', 'convolve_fix_width_fft', 'convolve_var_width_fft', 
]
