import io
import numpy as np
from scipy import interpolate
import scipy.constants as cst
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt

skirtor_dir = '../skirtor/'

# https://sites.google.com/site/skirtorus/sed-library
# Parameters of Stalevski+2016
#t: tau9.7, average edge-on optical depth at 9.7 micron; the actual one along the line of sight may vary depending on the clumps distribution.
tau = [ '3', '5', '7', '9', '11' ]
#p: power-law exponent that sets radial gradient of dust density
radial_index = [ '0', '0.5', '1', '1.5' ]
#q: index that sets dust density gradient with polar angle
polar_index = [ '0', '0.5', '1', '1.5' ]
#oa: angle measured between the equatorial plan and edge of the torus. Half-opening angle of the dust-free cone is 90-oa.
opening_angle = [ '10', '20', '30', '40', '50', '60', '70', '80' ]
#R: ratio of outer to inner radius, R_out/R_in
r_ratio = [ '10', '20', '30' ]
#Mcl: fraction of total dust mass inside clumps. 0.97 means 97% of total mass is inside the clumps and 3% in the interclump dust.
clump_fraction = [ '0.97' ]
#i: inclination, i.e. viewing angle, i.e. position of the instrument w.r.t. the AGN axis. i=0: face-on, type 1 view; i=90: edge-on, type 2 view.
view_angle = [ '0', '10', '20', '30', '40', '50', '60', '70', '80', '90' ]

# Read and convert the wavelength
datafile = open( skirtor_dir + "t{}_p{}_q{}_oa{}_R{}_Mcl{}_i{}_sed.dat"
                .format( tau[0], radial_index[0], polar_index[0], opening_angle[0], r_ratio[0], clump_fraction[0], view_angle[0] ))
data = "".join(datafile.readlines()[-132:])
datafile.close()
wave = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))
# wave in micron

total_dust_mass = np.genfromtxt('total_dust_mass.dat', skip_header = 2, names = [ 'model', 'mass', 'char' ], dtype = 'U30, f8, U4') #, delimiter = ' '
total_dust_lum = np.genfromtxt('total_dust_lum.dat', skip_header = 1, names = [ 'model', 'lumdisc', 'lumtorus', 'eb' ], dtype = 'U30, f8, f8, f8') #, delimiter = ' '

##############################

tau_stack = -9999
oa_stack = -9999
rratio_stack = -9999
incl_stack = -9999
mass_stack = -9999
eb_stack = -9999
lumD_stack = wave
lumT_stack = wave

# Pick up the parameter ranges to be used in s3fit
tau = [ '3', '5', '7', '9', '11' ]
radial_index = ['1'] #[ '0.5' ]
polar_index = [ '0.5' ] #[ '0' ]
opening_angle = [ '10', '20', '30', '40', '50', '60', '70', '80' ]
r_ratio = [ '10', '20', '30' ]
view_angle = [ '0', '10', '20', '30', '40', '50', '60', '70', '80', '90' ]

iter_params = (( t, p, q, oa, R, Mcl, i )
               for t in tau
               for p in radial_index
               for q in polar_index
               for oa in opening_angle
               for R in r_ratio
               for Mcl in clump_fraction
               for i in view_angle
               )

for params in iter_params:
    filename = skirtor_dir + "t{}_p{}_q{}_oa{}_R{}_Mcl{}_i{}_sed.dat".format(*params)
    print("Importing {} ...".format(filename))
    try:
        datafile = open(filename)
    except IOError:
        continue
    data = "".join(datafile.readlines()[-132:])
    datafile.close()

    lamF_total, lamF_dirPri, lamF_scaPri, lamF_dirDust, lamF_scaDust, lamF_iniPri = np.genfromtxt( io.BytesIO(data.encode()), usecols=(1, 2, 3, 4, 5, 6), unpack=True)

    # 1 W/m2 = (1e7 erg/s) / (3.241e-23 Mpc)2 = 9.520e51 erg/s/Mpc2 = 2.487e18 Lsun/Mpc2
    conv = 9.520e51#2.487e18
    
    uL_disc = lamF_dirPri / wave * conv * 4 * 3.1416 * 10**2 / 1e11
    #uL_torus = ( lamF_scaPri + lamF_dirDust + lamF_scaDust ) / wave * conv * 4 * 3.1416 * 10**2 / 1e11
    uL_torus = lamF_dirDust / wave * conv * 4 * 3.1416 * 10**2 / 1e11
    
    mask = total_dust_mass['model'] == "t{}_p{}_q{}_oa{}_R{}_Mcl{}".format(*params)
    uM_dust = total_dust_mass['mass'][ mask ]
    uM_dust /= 1e11
    
    mask = total_dust_lum['model'] == "t{}_p{}_q{}_oa{}_R{}_Mcl{}".format(*params)
    eb = total_dust_lum['eb'][ mask ]
    print(eb)
    
    tau_stack = np.column_stack(( tau_stack, float(params[0]) ))
    oa_stack = np.column_stack(( oa_stack, float(params[3]) ))
    rratio_stack = np.column_stack(( rratio_stack, float(params[4]) ))
    incl_stack = np.column_stack(( incl_stack, float(params[6]) ))
    mass_stack = np.column_stack(( mass_stack, uM_dust )) 
    eb_stack = np.column_stack(( eb_stack, eb )) 
    lumD_stack = np.column_stack(( lumD_stack, uL_disc ))     
    lumT_stack = np.column_stack(( lumT_stack, uL_torus ))     
 
skirtor_disc_output = np.row_stack(( tau_stack, oa_stack, rratio_stack, incl_stack, mass_stack, eb_stack, lumD_stack ))
np.savetxt( "skirtor_disc.dat", skirtor_disc_output, fmt='%8e' )
skirtor_torus_output = np.row_stack(( tau_stack, oa_stack, rratio_stack, incl_stack, mass_stack, eb_stack, lumT_stack ))
np.savetxt( "skirtor_torus.dat", skirtor_torus_output, fmt='%8e' )

fits.PrimaryHDU(np.array([skirtor_disc, skirtor_torus])).writeto('skirtor_for_s3fit.fits', overwrite=True, output_verify='silentfix')
