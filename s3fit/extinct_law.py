# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Contact: xiaoyang.chen.cz@gmail.com

import numpy as np

# https://articles.adsabs.harvard.edu/pdf/1989ApJ...345..245C
def ExtLaw_CCM89(wave, RV=3.1):
    # wave in Angstorm
    x = 1e4 / wave # in um-1
    a_out, b_out = np.zeros_like(x), np.zeros_like(x)
    # IR: 0.3 <= x <= 1.1 um-1
    mask_x = (x >= 0.3) & (x <= 1.1)
    a =  0.574*x**1.61
    b = -0.527*x**1.61
    a_out[mask_x], b_out[mask_x] = a[mask_x], b[mask_x]
    # Optical/NIR: 1.1 < x <= 3.3
    mask_x = (x > 1.1) & (x <= 3.3)
    y = x - 1.82    
    a = 1.+ 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b =     1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    a_out[mask_x], b_out[mask_x] = a[mask_x], b[mask_x]
    # UV: 3.3 < x <= 8
    mask_x = (x >= 5.9) & (x <= 8.0)
    F_a = -0.04473*(x - 5.9)**2 - 0.009779*(x - 5.9)**3
    F_b =  0.21300*(x - 5.9)**2 + 0.120700*(x - 5.9)**3
    F_a[~mask_x], F_b[~mask_x] = 0, 0
    mask_x = (x > 3.3) & (x <= 8.0)
    a =  1.752 - 0.316*x - 0.104/((x - 4.67)**2 + 0.341) + F_a
    b = -3.090 + 1.825*x + 1.206/((x - 4.62)**2 + 0.263) + F_b
    a_out[mask_x], b_out[mask_x] = a[mask_x], b[mask_x]
    # FUV: 8 < x <= 10
    mask_x = (x > 8.0) & (x <= 10.0)
    a = -1.073 - 0.628*(x - 8.0) + 0.137*(x - 8.0)**2 - 0.070*(x - 8.0)**3
    b = 13.670 + 4.257*(x - 8.0) - 0.420*(x - 8.0)**2 + 0.374*(x - 8.0)**3
    a_out[mask_x], b_out[mask_x] = a[mask_x], b[mask_x]
    return a_out + b_out / RV # A_lambda / AV

# http://www.bo.astro.it/~micol/Hyperz/old_public_v1/hyperz_manual1/node10.html
def ExtLaw_Calzetti00(wave, RV=4.05):
    # wave in Angstorm
    x = 1e4 / wave # in um-1
    k_out = np.zeros_like(x)
    # extend short-wave edge from 1200 to 90
    mask_w = (wave >= 90) & (wave <= 6300)
    k = 2.659*(-2.156 + 1.509*x - 0.198*x**2 + 0.011*x**3) + RV
    k_out[mask_w] = k[mask_w]
    # 6300 -> 22000 AA
    mask_w = (wave > 6300) & (wave <= 22000)
    k = 2.659*(-1.857 + 1.040*x) + RV
    k_out[mask_w] = k[mask_w]
    if np.sum(wave > 22000) > 0:
        mask_w0 = wave <= 22000
        index, r = np.polyfit(np.log10(wave[mask_w0][-2:]), np.log10(k_out[mask_w0][-2:]), 1)
        mask_w1 = wave > 22000
        k_out[mask_w1] = wave[mask_w1]**index*10.0**r
    return k_out / RV # A_lambda / AV

#################################################
# define the extinction law used in the fit
ExtLaw = ExtLaw_Calzetti00
#################################################
