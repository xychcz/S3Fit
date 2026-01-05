# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: s3fit@xychen.me

import time
import numpy as np
np.set_printoptions(linewidth=10000)
import scipy.sparse as sparse
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as const
import matplotlib.colors as mcolors
import colorsys

#####################################################################
######################### print functions ###########################
def print_log(message, log=[], display=True):
    if display: print(message)
    log.append(message)

def print_time(message, time_last=time.time(), time_init=time.time(), log=[], display=True):
    print_log(f'#### {message} {time.time()-time_last:.1f}s spent in this step; {time.time()-time_init:.1f}s spent in total.', log, display)
    return time.time()

def center_string(s:str, total_length:int=30, pad:str="#", padding_lmin:int=4) -> str:
    text_length = len(s) + 2  # account for spaces around the text
    pad_units = (total_length - text_length) // (2 * len(pad))
    extra_chars = (total_length - text_length) % (2 * len(pad))  # extra characters if division isn't even
    if pad_units < padding_lmin:
        pad_units = padding_lmin
        extra_chars = 0
    left_pad = pad * pad_units
    right_pad = pad * (pad_units + (extra_chars // len(pad)))  # add extra padding on the left if needed
    return f"{left_pad} {s} {right_pad}"

def casefold(x):
    if isinstance(x, str): 
        return x.casefold()
    elif isinstance(x, list): 
        return [i.casefold() for i in x]
    elif isinstance(x, np.ndarray): 
        if x.ndim == 1: return np.array([i.casefold() for i in x])
    elif isinstance(x, (int, float)):
        return x
    raise ValueError((f"casefold() only supports string, list, or 1-d np.ndarray."))

# def var_name(var):
#     return f"{var=}".split('=')[0]
        
# convert roman numbers
def roman_to_int(roman_num):
    roman_dict = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000,'IV':4,'IX':9,'XL':40,'XC':90,'CD':400,'CM':900}
    i = 0; int_num = 0
    while i < len(roman_num):
        if i+1<len(roman_num) and roman_num[i:i+2] in roman_dict:
            int_num+=roman_dict[roman_num[i:i+2]]; i+=2
        else:
            int_num+=roman_dict[roman_num[i]]; i+=1
    return int_num

greek_letters = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
                 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho',
                 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']

#####################################################################
########################## plot functions ###########################
def get_colors_by_hue(hue_center, hue_width=30, min_s=0.35, max_s=0.95, min_v=0.35, max_v=0.95):
    color_list = []
    for name, hexv in mcolors.CSS4_COLORS.items():
        r, g, b = mcolors.to_rgb(hexv)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h_deg = h * 360
        dh = min(abs(h_deg - hue_center), 360 - abs(h_deg - hue_center)) # circular distance in hue
        if dh <= hue_width and s >= min_s and s <= max_s and v >= min_v and v <= max_v: color_list.append(name)
        # color_list = sorted(color_list)
    return color_list

color_list_dict = {'red':    get_colors_by_hue(0,   hue_width=20, min_s=0.3,  max_s=1,    min_v=0.6,  max_v=1),
                   'yellow': get_colors_by_hue(50,  hue_width=25, min_s=0.4,  max_s=1,    min_v=0.5,  max_v=0.99),
                   'green':  get_colors_by_hue(120, hue_width=50, min_s=0.3,  max_s=1,    min_v=0.51, max_v=0.98),
                   'blue':   get_colors_by_hue(190, hue_width=35, min_s=0.41, max_s=0.99, min_v=0.3,  max_v=0.99),
                   'purple': get_colors_by_hue(300, hue_width=50, min_s=0.1,  max_s=0.8,  min_v=0.55, max_v=1),
                  }

#####################################################################
#################### wavelength conversion ##########################
# https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
# input wavelength in angstrom
# extrapolate to < 2000 A to avoid singularity
def wave_vac_to_air(wave_vac, extrapolate=True):
    if np.isscalar(wave_vac): wave_vac = np.array([wave_vac])
    if isinstance(wave_vac, list): wave_vac = np.array(wave_vac)

    s = np.divide(1e4, wave_vac, where=wave_vac>0, out=np.zeros_like(wave_vac, dtype='float'))
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    wave_air = wave_vac / n
    
    mask = wave_vac < 3000
    if any(mask) & extrapolate:
        r = 0.999634933 + 3.52626553e-08*wave_vac[mask] - 3.64256777e-12*wave_vac[mask]**2
        # obtained via fitting wave_air/wave_vac in 2500-5000 A
        wave_air[mask] = wave_vac[mask] * r
    
    return wave_air

def wave_air_to_vac(wave_air, extrapolate=True):
    if np.isscalar(wave_air): wave_air = np.array([wave_air])
    if isinstance(wave_air, list): wave_air = np.array(wave_air)

    s = np.divide(1e4, wave_air, where=wave_air>0, out=np.zeros_like(wave_air, dtype='float'))
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) \
                                + 0.0001599740894897 / (38.92568793293 - s**2)
    wave_vac = wave_air * n
    
    mask = wave_air < 3000
    if any(mask) & extrapolate:
        r = 1.000365120 - 3.52546427e-08*wave_air[mask] + 3.64165604e-12*wave_air[mask]**2
        # obtained via fitting wave_vac/wave_air in 2500-5000 A
        wave_vac[mask] = wave_air[mask] * r
        
    return wave_vac
#####################################################################
###################### Photometric conversion #######################

def Fnu_over_Flam(wave_w, trans_bw=None, Flam_unit='erg s-1 cm-2 angstrom-1', Fnu_unit='mJy'):
    unitFlam = 1.0 * u.Unit(Flam_unit)
    rDnuDlam = const.c / (wave_w * u.angstrom)**2
    if trans_bw is None:
        # return the ratio of spectrum between Fnu (mJy) and Flam (erg/s/cm2/A); wave in angstrom
        return (unitFlam / rDnuDlam).to(Fnu_unit).value
    else:
        # return the ratio of band flux between Fnu (mJy) and Flam (erg/s/cm2/A); wave in angstrom
        unitFint = unitFlam * (1.0 * u.angstrom)
        # here (1 * u.angstrom) = np.trapezoid(trans, x=wave * u.angstrom, axis=axis), since trans is normalized to int=1
        width_nu = np.trapezoid(trans_bw * rDnuDlam, x=wave_w * u.angstrom, axis=trans_bw.ndim-1)
        return (unitFint / width_nu).to(Fnu_unit).value
    
def spec_to_phot(wave_w, spec_mw, trans_bw):
    # convert spectrum in flam (erg/s/cm2/A) to mean flam in band (erg/s/cm2/A)
    if (spec_mw.ndim == 1) & (trans_bw.ndim == 1):
        return np.trapezoid(trans_bw * spec_mw, x=wave_w, axis=0) # return flux, 1-model, 1-band
    if (spec_mw.ndim == 1) & (trans_bw.ndim == 2):
        return np.trapezoid(trans_bw * spec_mw[None,:], x=wave_w, axis=1) # return flux_b, 1-model, multi-band
    if (spec_mw.ndim == 2) & (trans_bw.ndim == 1):
        return np.trapezoid(trans_bw[None,:] * spec_mw, x=wave_w, axis=1) # return flux_m, multi-model, 1-band
    if (spec_mw.ndim == 2) & (trans_bw.ndim == 2):
        return np.trapezoid(trans_bw[None,:,:] * spec_mw[:,None,:], x=wave_w, axis=2) # return flux_mb
    # short for np.trapezoid(trans * spec, x=wave, axis=axis) / np.trapezoid(trans, x=wave, axis=axis), trans is normalized to int=1

#####################################################################
##################### convolution functions #########################
def convert_linw_to_logw(linw_wave, linw_flux, linw_error=None, resolution=None):
    # adopt grid resampling to keep (1/0.8) times original density at the long wavelength end
    if resolution is None: resolution = linw_wave.max()/(0.8*(linw_wave[1]-linw_wave[0]))
    logw_logwidth = np.log(1/resolution + 1)
    logw_wave = np.logspace(np.log(linw_wave.min()), np.log(linw_wave.max()), base=np.e, 
                            num=int(np.log(linw_wave.max()/linw_wave.min()) / logw_logwidth))
    logw_flux = np.interp(logw_wave, linw_wave, linw_flux)
    if linw_error is not None: 
        logw_error = np.interp(logw_wave, linw_wave, linw_error)
        return logw_wave, logw_flux, logw_error
    else:
        return logw_wave, logw_flux

def gaussian_kernel_1d(dpix_sigma, truncate=4.0):
    # 1D Gaussian kernel equivalent to astropy.convolution.Gaussian1DKernel
    radius = int(truncate * dpix_sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / dpix_sigma)**2)
    kernel /= kernel.sum()
    return kernel

def convolve_spec_logw(logw_wave, logw_flux, conv_sigma, axis=0):
    # logw_wave, logw_flux need to be uniform with log_e wavelength
    logw_width = np.log(logw_wave[1])-np.log(logw_wave[0])
    # kernel = Gaussian1DKernel(stddev=conv_sigma/299792.458/logw_width).array
    # kernel /= kernel.sum()
    kernel = gaussian_kernel_1d(conv_sigma/299792.458/logw_width)
    if len(logw_flux.shape) == 2:
        if axis == 0: kernel = kernel[:, None]
        if axis == 1: kernel = kernel[None, :]
    logw_fcon = fftconvolve(logw_flux, kernel, mode='same', axes=axis)
    return logw_fcon

def convolve_fix_width_fft(wave_w, flux_mw, dw_fwhm=None, dpix_sigma=None, reset_edge=True):
    
    if dw_fwhm is not None:
        dw_sigma   = dw_fwhm  / np.sqrt(np.log(256))
        dpix_sigma = dw_sigma / np.median(np.gradient(wave_w))
    
    kernel = gaussian_kernel_1d(dpix_sigma)

    if len(flux_mw.shape) == 1: 
        conv_flux_mw = fftconvolve(flux_mw, kernel, mode='same', axes=0)
    if len(flux_mw.shape) == 2: 
        conv_flux_mw = fftconvolve(flux_mw, kernel[None, :], mode='same', axes=1)
        
    # correct the artifact negative values
    if ~(flux_mw < 0).any() & (conv_flux_mw < 0).any():
        # conv_flux_mw[conv_flux_mw < 0] = 0
        conv_flux_mw[conv_flux_mw <= 0] = flux_mw[flux_mw > 0].min()

    if reset_edge:
        pad_total = len(kernel) - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        start = pad_left
        end = len(wave_w) - pad_right
        if len(flux_mw.shape) == 1: 
            conv_flux_mw[:start] = flux_mw[:start]
            conv_flux_mw[end:] = flux_mw[end:]
        if len(flux_mw.shape) == 2: 
            conv_flux_mw[:,:start] = flux_mw[:,:start]
            conv_flux_mw[:,end:] = flux_mw[:,end:]

    return conv_flux_mw 

def convolve_var_width_fft(wave_w, flux_mw, R_inst_w=None, 
                           dw_fwhm_obj=None, dv_fwhm_obj=None, 
                           dw_fwhm_ref=None, dv_fwhm_ref=None, R_ref=None, 
                           dw_fwhm_func=None, 
                           num_bins=10, reset_edge=True):

    if dw_fwhm_func is not None: 
        if callable(dw_fwhm_func): 
            dw_fwhm_w = dw_fwhm_func(wave_w)
    else:
        if dw_fwhm_obj is not None: dw_fwhm_obj_w = dw_fwhm_obj * 1.0
        if dv_fwhm_obj is not None: dw_fwhm_obj_w = wave_w / (299792.458/dv_fwhm_obj)
        dw2_fwhm_w = dw_fwhm_obj_w**2

        if R_inst_w is not None: 
            dw_fwhm_inst_w = wave_w / R_inst_w
            dw2_fwhm_w += dw_fwhm_inst_w**2

        dw_fwhm_ref_w = None
        if dw_fwhm_ref is not None: dw_fwhm_ref_w = dw_fwhm_ref * 1.0
        if dv_fwhm_ref is not None: dw_fwhm_ref_w = wave_w / (299792.458/dv_fwhm_ref)
        if R_ref       is not None: dw_fwhm_ref_w = wave_w / R_ref
        if dw_fwhm_ref_w is not None:
            dw2_fwhm_w -= dw_fwhm_ref_w**2 # reduce the dispersion of refered template
            
        dw2_fwhm_w[dw2_fwhm_w <= 0] = 1e-8 # avoid non-positive convolving width
        dw_fwhm_w = np.sqrt(dw2_fwhm_w)

    dw_sigma_w   = dw_fwhm_w  / np.sqrt(np.log(256))
    dpix_sigma_w = dw_sigma_w / np.gradient(wave_w)

    if num_bins == 1:
        return convolve_fix_width_fft(wave_w, flux_mw, dpix_sigma=np.median(dpix_sigma_w), reset_edge=reset_edge)
    else:
        ret_flux_mw = np.zeros_like(flux_mw)
        select_wave_w = np.linspace(min(wave_w), max(wave_w), num_bins)
        for wave_0 in select_wave_w:
            dpix_sigma_0 = np.interp(wave_0, wave_w, dpix_sigma_w)
            kernel = gaussian_kernel_1d(dpix_sigma_0)
            if len(flux_mw.shape) == 1: 
                conv_flux_mw = fftconvolve(flux_mw, kernel, mode='same', axes=0)
            if len(flux_mw.shape) == 2: 
                conv_flux_mw = fftconvolve(flux_mw, kernel[None, :], mode='same', axes=1)

            # correct the artifact negative values
            if ~(flux_mw < 0).any() & (conv_flux_mw < 0).any():
                conv_flux_mw[conv_flux_mw <= 0] = flux_mw[flux_mw > 0].min()

            factor_w = 1 - np.abs(wave_w-wave_0) / (select_wave_w[1]-select_wave_w[0])
            mask_w = factor_w >= 0
            if len(flux_mw.shape) == 1: 
                ret_flux_mw[mask_w] += conv_flux_mw[mask_w] * factor_w[mask_w]
            if len(flux_mw.shape) == 2: 
                ret_flux_mw[:, mask_w] += conv_flux_mw[:, mask_w] * factor_w[None, mask_w]

        if reset_edge:
            pad_left  = len(gaussian_kernel_1d(dpix_sigma_w[0])) // 2
            pad_right = len(gaussian_kernel_1d(dpix_sigma_w[-1])) // 2
            start = pad_left
            end = len(wave_w) - pad_right
            if len(flux_mw.shape) == 1: 
                ret_flux_mw[:start] = flux_mw[:start]
                ret_flux_mw[end:] = flux_mw[end:]
            if len(flux_mw.shape) == 2: 
                ret_flux_mw[:,:start] = flux_mw[:,:start]
                ret_flux_mw[:,end:] = flux_mw[:,end:]

        return ret_flux_mw

def convolve_var_width_csr(wave_w, flux_mw, R_inst_w=None, 
                           dw_fwhm_obj=None, dv_fwhm_obj=None, 
                           dw_fwhm_ref=None, dv_fwhm_ref=None, R_ref=None, 
                           dw_fwhm_func=None, 
                           sigma_cutoff=3, num_step=1000):

    if dw_fwhm_func is not None: 
        if callable(dw_fwhm_func): 
            dw_fwhm_w = dw_fwhm_func(wave_w)
    else:
        if dw_fwhm_obj is not None: dw_fwhm_obj_w = dw_fwhm_obj * 1.0
        if dv_fwhm_obj is not None: dw_fwhm_obj_w = wave_w / (299792.458/dv_fwhm_obj)
        dw2_fwhm_w = dw_fwhm_obj_w**2

        if R_inst_w is not None: 
            dw_fwhm_inst_w = wave_w / R_inst_w
            dw2_fwhm_w += dw_fwhm_inst_w**2

        dw_fwhm_ref_w = None
        if dw_fwhm_ref is not None: dw_fwhm_ref_w = dw_fwhm_ref * 1.0
        if dv_fwhm_ref is not None: dw_fwhm_ref_w = wave_w / (299792.458/dv_fwhm_ref)
        if R_ref       is not None: dw_fwhm_ref_w = wave_w / R_ref
        if dw_fwhm_ref_w is not None:
            dw2_fwhm_w -= dw_fwhm_ref_w**2 # reduce the dispersion of refered template
            
        dw2_fwhm_w[dw2_fwhm_w <= 0] = 1e-8 # avoid non-positive convolving width
        dw_fwhm_w = np.sqrt(dw2_fwhm_w)

    dw_sigma_w   = dw_fwhm_w  / np.sqrt(np.log(256))
    dpix_sigma_w = dw_sigma_w / np.gradient(wave_w)

    num_wave = len(wave_w)
    # use LIL format for efficient row-wise updates
    conv_matrix_ww = sparse.lil_matrix((num_wave, num_wave))  
    mask_select_w = np.zeros(num_wave, dtype='bool')

    for i_w in range(0, num_wave, int(num_wave/num_step)):
        mask_select_w[i_w] = True
        wave_0 = wave_w[i_w]
        dw_sigma_0 = dw_sigma_w[i_w]
        dw_dist_w = wave_w - wave_0
        kernel_w = np.exp(-0.5 * (dw_dist_w / dw_sigma_0) ** 2)
        kernel_w /= kernel_w.sum() 
        mask_kn_w = np.abs(dw_dist_w) < (sigma_cutoff * dw_sigma_0)
        conv_matrix_ww[i_w, mask_kn_w] = kernel_w[mask_kn_w]

    # convert sparse matrix to dense for interpolation
    interp_func = interp1d(wave_w[mask_select_w], conv_matrix_ww.toarray()[mask_select_w, :], 
                           axis=0, kind='linear', fill_value='extrapolate')
    # interpolate to the full wave grid and convert back to Compressed Sparse Row
    conv_matrix_ww = sparse.csr_matrix(interp_func(wave_w)) 
    # perform sparse matrix multiplication
    conv_flux_mw = conv_matrix_ww.dot(flux_mw.T).T

    return conv_flux_mw
#####################################################################
#####################################################################