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
# from astropy.convolution import Gaussian1DKernel

#####################################################################
####################### printing functions ##########################
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
#####################################################################
#####################################################################

#####################################################################
#################### wavelength conversion ##########################
# https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
def lamb_vac_to_air(lamb_vac, extrapolate=True):
    s = 1e4 / lamb_vac
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    lamb_air = lamb_vac / n
    
    mask = lamb_vac < 3000
    if (mask.sum() > 0) & extrapolate:
        r = 0.999634933 + 3.52626553e-08*lamb_vac[mask] - 3.64256777e-12*lamb_vac[mask]**2
        # obtained via fitting lamb_air/lamb_vac in 2500-5000 AA
        lamb_air[mask] = lamb_vac[mask] * r
    
    return lamb_air
    
def lamb_air_to_vac(lamb_air, extrapolate=True):
    s = 1e4 / lamb_air
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) \
                                + 0.0001599740894897 / (38.92568793293 - s**2)
    lamb_vac = lamb_air * n
    
    mask = lamb_air < 3000
    if (mask.sum() > 0) & extrapolate:
        r = 1.000365120 - 3.52546427e-08*lamb_air[mask] + 3.64165604e-12*lamb_air[mask]**2
        # obtained via fitting lamb_vac/lamb_air in 2500-5000 AA
        lamb_vac[mask] = lamb_air[mask] * r
        
    return lamb_vac
#####################################################################
#####################################################################

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
                           axis=0, kind='linear', fill_value="extrapolate")
    # interpolate to the full wave grid and convert back to Compressed Sparse Row
    conv_matrix_ww = sparse.csr_matrix(interp_func(wave_w)) 
    # perform sparse matrix multiplication
    conv_flux_mw = conv_matrix_ww.dot(flux_mw.T).T

    return conv_flux_mw
#####################################################################
#####################################################################