import os, sys
import numpy as np
from astropy.io import fits

ret_names, ret_ages, ret_mets, ret_waves, ret_specs = [], [], [], [], []
wmin, wmax = 50, 24050
# real range: 91--24000 AA

# please download the library from https://www.fractal-es.com/PopStar/#hr_py_download
# for the required initial mass function 
dir_prefix = 'pyPopStar_Kroupa/Total/' # 
dir_suffixes = ['Z004/','Z008/','Z02/','Z05/']

for dir_suffix in dir_suffixes: 
    print(dir_suffix)
    files = np.array(os.listdir(dir_prefix+dir_suffix))
    ages = np.array([files[i_f].split('_logt')[1].split('.dat')[0] for i_f in range(len(files))])
    mets = np.array([files[i_f].split('_Z')[1].split('_logt')[0] for i_f in range(len(files))])
    indexes = np.argsort(ages.astype('float'))
    files = files[indexes]
    ages = ages[indexes]
    mets = mets[indexes]
    ret_names.append(files)
    ret_ages.append(ages)
    ret_mets.append(mets)
    for (file, age, met) in zip(files, ages, mets):
        wave, lum = np.genfromtxt(fname=dir_prefix+dir_suffix+file, usecols=(0,1), 
                                  comments="#", delimiter="	", unpack=True) # delimiter -> Tab
        mask_select_w = (wave >= wmin) & (wave <= wmax)
        wave, lum = wave[mask_select_w], lum[mask_select_w]
        if met == '0.004': good_age_min = 7.51
        if met == '0.008': good_age_min = 7.34
        if met == '0.02': good_age_min = 7.34
        if met == '0.05': good_age_min = 7.26
        if float(age) < good_age_min:
            mask_artifact_w = (wave >= 3645) & (wave <= 3682)
            lum[mask_artifact_w] = np.interp(wave[mask_artifact_w], wave[~mask_artifact_w], lum[~mask_artifact_w])
        ret_waves.append(wave)
        ret_specs.append(lum)

ret_names = np.array(ret_names).reshape(4*106)
ret_ages = np.array(ret_ages).reshape(4*106)
ret_mets = np.array(ret_mets).reshape(4*106)
ret_waves = np.array(ret_waves)
ret_specs = np.array(ret_specs)

hdu0 = fits.PrimaryHDU(ret_specs)
hdu0.header['BUNIT'] = ( 'Data Value', ' ' )
hdu0.header['CRPIX1'] = ( 1, ' ' )
hdu0.header['CRVAL1'] = ( ret_waves[0][0], ' ' )
hdu0.header['CDELT1'] = ( (ret_waves[0][1:] - ret_waves[0][:-1]).mean(), ' ' )
for i, ret_name in enumerate(ret_names):
    hdu0.header['NAME'+str(i)] = ( ret_names[i], ' ' )

hdu1 = fits.ImageHDU(ret_waves[0])

# please download the remaining mass function  
# from https://www.fractal-es.com/PopStar/#hr_py_download
# for the required initial mass function 
ssp_Mnow_rmf_mp = np.loadtxt('remaining_stellar_mass_fraction_KRO.dat', skiprows=1)
# logage, met, remaining mass fraction
hdu2 = fits.ImageHDU(ssp_Mnow_rmf_mp[:,2])

hdul = fits.HDUList([hdu0, hdu1, hdu2])
hdul.writeto('popstar_for_s3fit.fits', overwrite=True, output_verify='silentfix')
