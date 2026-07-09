##################################################
# imports

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy import units
import pandas as pd
import netCDF4 as nc
import os

import itertools
import paramsurvey
import paramsurvey.params
import paramsurvey.stats as stats
from collections import OrderedDict
import io
import contextlib
import glob

from scipy.interpolate import interp1d

##################################################
# inputs

# sites = ['ALMA','APEX','GLT','IRAM','JCMT','KP','LMT','NOEMA','SMA','SMT','SPT','AGGO','ALI','ARE','ATCA','BAJA','BAN','BAR','BDRY','BGA','BGK','BLDR','BMAC','BOL','BRZ','CAM','CAS','CAT','CNI','CTIO','DomeA','DomeC','DomeF','EFF','ELK','ERB','FAIR','FLWO','FUJI','GAM','GARS','GBT','GGAO','GLTS','GOR','HAN','HART','HAY','HESS','HIT','HOB','HOR','IRK','ISG','ISH','JARE','JELM','JELM2','JOD','KASH','KATH','KEN','KILI','KNM','KOG','KOKEE','KVNPC','KVNTN','KVNUS','KVNYS','LAS','LLA','LOS','MACGO','MAT','MATJ','MED','MET','MIZ','MOR','MUZ','NAN','NOB','NOR','NOTO','NYALE','NZ','OGA','ONS','ONSNE','ONSSW','ORG','OVRO','PAR','PIKE','PRKS','ROEN','ROT','SAN','SEJ','SGO','SHE','SIM','SKS','SMAR','SPX','SRT','STL','SUF','SVET','TAK','TNMA','TOR','TRL','TSU','UDSC','VLA','VLBBR','VLBFD','VLBHN','VLBKP','VLBLA','VLBMK','VLBNL','VLBOV','VLBPT','VLBSC','WARK','WEST','WETTZ','WSRT','XSMC','YAM','YAN','YAR','YBJ','YEB','YEBRG','ZELE','ZUG']
sites = ['AGGO','ARE','BDRY','CAM','HART','JARE','JOD','KASH','KOG','MAT','MED','NOTO','PRKS','ROEN','SEJ','SIM','SRT','SVET','TOR','TSU','UDSC','WARK','WSRT','YAR','ZELE']

# years = ['2025','2026']
years = ['1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022','2023','2024','2025','2026']

MERRA_dirname = '../MERRA2_data/'

fmin = 0            # minimum frequency, in GHz
fmax = 2000         # maximum frequency, in GHz
df = 1.0            # frequency resolution, in GHz
Ncomp = 40          # number of PCA components to use

##################################################
# constants

g0 = 9.80665                # standard gravity; SI units
R = 8.31446261815324        # gas constant; SI units
mu = 0.0289647              # molar mass of air, in kg / mol
k = 1.380649e-23            # boltzmann constant; SI units
rho_H20 = 997.0             # density of water; SI units

# conversion factor from mass mixing ratio to volume mixing ratio for O3
O3_mass_to_volume = 28.9644/47.9982

# conversion factor from mass mixing ratio to volume mixing ratio for H2O
H2O_mass_to_volume = 28.9644/18.0153

##################################################
# useful functions

# linear interpolator
def linear(x,x1,x2,y1,y2):
    m = (y1 - y2)/(x1 - x2)
    b = y1 - (m*x1)
    y = (m*x) + b
    return y

# bilinear interpolator
# z11 = z(x1,y1)
# z12 = z(x1,y2)
# z21 = z(x2,y1)
# z22 = z(x2,y2)
def bilinear(x,y,x1,x2,y1,y2,z11,z12,z21,z22):
    norm = (x2-x1)*(y2-y1)
    w11 = ((x2-x)*(y2-y))/norm
    w12 = ((x2-x)*(y-y1))/norm
    w21 = ((x-x1)*(y2-y))/norm
    w22 = ((x-x1)*(y-y1))/norm
    z = (w11*z11) + (w12*z12) + (w21*z21) + (w22*z22)
    if hasattr(z, '__iter__'):
        if ((z11 == z12) & (z11 == z21) & (z11 == z22)).all():
            return z11
    else:
        if ((z11 == z12) & (z11 == z21) & (z11 == z22)):
            return z11
    return z

##################################################
# primary computation function

def tabulate_weather(pset, system_kwargs, user_kwargs):
    raw_stats = system_kwargs['raw_stats']

    # pass stdout to avoid slowing down ray
    my_stdout = io.StringIO()
    with contextlib.redirect_stdout(my_stdout):

        ##################################################
        # parse parameter set

        input_file = pset['input_file']
        site = pset['site']
        year = pset['year']
        month = pset['month']
        day = pset['day']
        itime = pset['itime']
        am_input_filename = pset['am_input_filename']
        am_output_filename = pset['am_output_filename']
        outfilename_tau = pset['outfilename_tau']
        outfilename_Tb = pset['outfilename_Tb']

        ##################################################
        # read in site info

        sitenames = np.loadtxt('Telescope_Site_Matrix.csv',usecols=(0),skiprows=1,delimiter=',',unpack=True,dtype='str')
        lats, lons, els = np.loadtxt('Telescope_Site_Matrix.csv',usecols=(3,4,5),skiprows=1,delimiter=',',unpack=True)

        ind = (sitenames == site)
        lat = lats[ind][0]
        lon = lons[ind][0]
        el = els[ind][0]

        ##################################################
        # MERRA data

        # load in MERRA data
        with stats.record_wallclock('read MERRA file wallclock', raw_stats):
            with stats.record_iowait('read MERRA file iowait', raw_stats):
                ds = nc.Dataset(input_file)

        # organize the info from the appropriate cells
        with stats.record_wallclock('parse MERRA file wallclock', raw_stats):
        
            times = ds.variables['time'][:].data / 180
            latarr = ds.variables['lat'][:].data
            lonarr = ds.variables['lon'][:].data

            Phi = ds.variables['PHIS'][:][0].data
            elev = Phi / g0

            # pressure in atmospheric levels, x100 to convert from hectoPascals to Pascals
            pressure_levels = ds.variables['lev'][:].data*100.0

            ice_water_temp = ds.variables['QI'][:]
            ice_water_temp[ice_water_temp.mask] = -999
            ice_water = ice_water_temp.data

            liquid_water_temp = ds.variables['QL'][:]
            liquid_water_temp[liquid_water_temp.mask] = -999
            liquid_water = liquid_water_temp.data

            eastward_wind_temp = ds.variables['U'][:]
            eastward_wind_temp[eastward_wind_temp.mask] = -999
            eastward_wind = eastward_wind_temp.data

            northward_wind_temp = ds.variables['V'][:]
            northward_wind_temp[northward_wind_temp.mask] = -999
            northward_wind = northward_wind_temp.data

            # temperature, in K
            temperature_levels_temp = ds.variables['T'][:]
            temperature_levels_temp[temperature_levels_temp.mask] = -999
            temperature_levels = temperature_levels_temp.data

            # atmospheric layer heights, in meters
            layer_heights = ds.variables['H'][:]
            layer_heights[layer_heights.mask] = -1.0e14
            layer_heights = layer_heights.data

            # surface pressure, in Pascals
            Psurf_arr_temp = ds.variables['PS'][:]
            Psurf_arr_temp[Psurf_arr_temp.mask] = -999
            Psurf_arr = Psurf_arr_temp.data

            # ozone mass and volume mixing ratio
            O3_mixing_ratio_temp = ds.variables['O3'][:]
            O3_vmr_temp = O3_mixing_ratio_temp*O3_mass_to_volume
            indhere = np.copy(O3_mixing_ratio_temp.mask)
            O3_mixing_ratio_temp[indhere] = -999
            O3_vmr_temp[indhere] = -999
            O3_mixing_ratio = O3_mixing_ratio_temp.data
            O3_vmr = O3_vmr_temp.data

            # H2O mass and volume mixing ratio
            specific_humidity_temp = ds.variables['QV'][:]
            H2O_mixing_ratio_temp = specific_humidity_temp / (1.0 - specific_humidity_temp)
            H2O_vmr_temp = H2O_mixing_ratio_temp*H2O_mass_to_volume
            indhere = np.copy(specific_humidity_temp.mask)
            specific_humidity_temp[indhere] = -999
            H2O_mixing_ratio_temp[indhere] = -999
            H2O_vmr_temp[indhere] = -999
            specific_humidity = specific_humidity_temp.data
            H2O_mixing_ratio = H2O_mixing_ratio_temp.data
            H2O_vmr = H2O_vmr_temp.data

            ##################################################
            # identify the corners of the box

            ilat1 = np.where(latarr == np.max(latarr[latarr <= lat]))[0][0]
            ilat2 = np.where(latarr == np.min(latarr[latarr >= lat]))[0][0]
            ilon1 = np.where(lonarr == np.max(lonarr[lonarr <= lon]))[0][0]
            ilon2 = np.where(lonarr == np.min(lonarr[lonarr >= lon]))[0][0]

            slice_lat = slice(ilat1,ilat2+1)
            slice_lon = slice(ilon1,ilon2+1)

            # define unified layer height array
            Nlayers = 100
            log_layer_height = np.linspace(np.log10(el),np.log10(70000.0),Nlayers)

            ##################################################
            ##################################################
            # first interpolate in the elevation direction,
            # pulling out the relevant values for this (time,lat,lon) location while only
            # looking at the atmospheric layers that are above the desired elevation level

            ##################################################
            # first corner

            ilat = ilat1
            ilon = ilon1

            # extract the heights of the atmospheric layers bracketing the telescope elevation
            index = (layer_heights[itime,:,ilat,ilon] >= 0.0)
            layer_heights_11 = layer_heights[itime,index,ilat,ilon]

            pressure_levels_11 = pressure_levels[index]
            Pfunc_11 = interp1d(layer_heights_11,np.log10(pressure_levels_11),kind='linear',fill_value='extrapolate')
            P_11 = 10.0**Pfunc_11(10.0**log_layer_height)

            H_11 = 10.0**log_layer_height

            temperature_levels_11 = temperature_levels[itime,index,ilat,ilon]
            ind = (temperature_levels_11 != -999)
            Tfunc_11 = interp1d(layer_heights_11[ind],temperature_levels_11[ind],kind='linear',fill_value='extrapolate')
            T_11 = Tfunc_11(10.0**log_layer_height)

            specific_humidity_11 = specific_humidity[itime,index,ilat,ilon]
            ind = (specific_humidity_11 != -999)
            qfunc_11 = interp1d(np.log10(layer_heights_11[ind]),np.log10(specific_humidity_11[ind]),kind='linear',fill_value='extrapolate')
            q_11 = 10.0**qfunc_11(log_layer_height)

            O3_vmr_11 = O3_vmr[itime,index,ilat,ilon]
            ind = (O3_vmr_11 != -999)
            O3func_11 = interp1d(layer_heights_11[ind],np.log10(O3_vmr_11[ind]),kind='linear',fill_value='extrapolate')
            O3_11 = 10.0**O3func_11(10.0**log_layer_height)

            H2O_vmr_11 = H2O_vmr[itime,index,ilat,ilon]
            ind = (H2O_vmr_11 != -999)
            # H2Ofunc_11 = interp1d(layer_heights_11[ind],np.log10(H2O_vmr_11[ind]),kind='linear',fill_value='extrapolate')
            # H2O_11 = 10.0**H2Ofunc_11(10.0**log_layer_height)
            H2Ofunc_11 = interp1d(layer_heights_11[ind],H2O_vmr_11[ind],kind='linear',fill_value='extrapolate')
            H2O_11 = H2Ofunc_11(10.0**log_layer_height)
            H2O_11[H2O_11 < 0.0] = 0.0

            liquid_water_11 = liquid_water[itime,index,ilat,ilon]
            ind = (liquid_water_11 != -999)
            LWP_mmrfunc_11 = interp1d(layer_heights_11[ind],liquid_water_11[ind],kind='linear',fill_value='extrapolate')
            LWP_mmr_11 = LWP_mmrfunc_11(10.0**log_layer_height)
            LWP_mmr_11[LWP_mmr_11 < 0.0] = 0.0

            ice_water_11 = ice_water[itime,index,ilat,ilon]
            ind = (ice_water_11 != -999)
            IWP_mmrfunc_11 = interp1d(layer_heights_11[ind],ice_water_11[ind],kind='linear',fill_value='extrapolate')
            IWP_mmr_11 = IWP_mmrfunc_11(10.0**log_layer_height)
            IWP_mmr_11[IWP_mmr_11 < 0.0] = 0.0

            eastward_wind_11 = eastward_wind[itime,index,ilat,ilon]
            ind = (eastward_wind_11 != -999)
            Ufunc_11 = interp1d(layer_heights_11[ind],eastward_wind_11[ind],kind='linear',fill_value='extrapolate')
            # U_11 = Ufunc_11(10.0**log_layer_height)
            U_11 = Ufunc_11(el).flatten()[0]

            northward_wind_11 = northward_wind[itime,index,ilat,ilon]
            ind = (northward_wind_11 != -999)
            Vfunc_11 = interp1d(layer_heights_11[ind],northward_wind_11[ind],kind='linear',fill_value='extrapolate')
            # V_11 = Vfunc_11(10.0**log_layer_height)
            V_11 = Vfunc_11(el).flatten()[0]

            # convert LWP and IWP mass mixing ratios to column densities
            dP = P_11[0:-1] - P_11[1:]
            P_mid = np.concatenate((P_11[0:-1] + (dP/2.0),[P_11[-1]+(dP[-1]/2.0)],[P_11[-1]-(dP[-1]/2.0)]))
            dP = P_mid[0:-1] - P_mid[1:]
            LWP_11 = LWP_mmr_11*dP/g0
            IWP_11 = IWP_mmr_11*dP/g0

            # compute the PWV
            integrand = q_11 / (1.0 - q_11)
            integral = -np.sum(0.5*(integrand[1:] + integrand[0:-1])*(P_11[1:] - P_11[0:-1]))
            integral /= (rho_H20*g0)
            integral *= 1000.0
            PWV_11 = integral

            ##################################################
            # second corner

            ilat = ilat1
            ilon = ilon2

            # extract the heights of the atmospheric layers bracketing the telescope elevation
            index = (layer_heights[itime,:,ilat,ilon] >= 0.0)
            layer_heights_12 = layer_heights[itime,index,ilat,ilon]

            pressure_levels_12 = pressure_levels[index]
            Pfunc_12 = interp1d(layer_heights_12,np.log10(pressure_levels_12),kind='linear',fill_value='extrapolate')
            P_12 = 10.0**Pfunc_12(10.0**log_layer_height)

            H_12 = 10.0**log_layer_height

            temperature_levels_12 = temperature_levels[itime,index,ilat,ilon]
            ind = (temperature_levels_12 != -999)
            Tfunc_12 = interp1d(layer_heights_12[ind],temperature_levels_12[ind],kind='linear',fill_value='extrapolate')
            T_12 = Tfunc_12(10.0**log_layer_height)

            specific_humidity_12 = specific_humidity[itime,index,ilat,ilon]
            ind = (specific_humidity_12 != -999)
            qfunc_12 = interp1d(np.log10(layer_heights_12[ind]),np.log10(specific_humidity_12[ind]),kind='linear',fill_value='extrapolate')
            q_12 = 10.0**qfunc_12(log_layer_height)

            O3_vmr_12 = O3_vmr[itime,index,ilat,ilon]
            ind = (O3_vmr_12 != -999)
            O3func_12 = interp1d(layer_heights_12[ind],np.log10(O3_vmr_12[ind]),kind='linear',fill_value='extrapolate')
            O3_12 = 10.0**O3func_12(10.0**log_layer_height)

            H2O_vmr_12 = H2O_vmr[itime,index,ilat,ilon]
            ind = (H2O_vmr_12 != -999)
            # H2Ofunc_12 = interp1d(layer_heights_12[ind],np.log10(H2O_vmr_12[ind]),kind='linear',fill_value='extrapolate')
            # H2O_12 = 10.0**H2Ofunc_12(10.0**log_layer_height)
            H2Ofunc_12 = interp1d(layer_heights_12[ind],H2O_vmr_12[ind],kind='linear',fill_value='extrapolate')
            H2O_12 = H2Ofunc_12(10.0**log_layer_height)
            H2O_12[H2O_12 < 0.0] = 0.0

            liquid_water_12 = liquid_water[itime,index,ilat,ilon]
            ind = (liquid_water_12 != -999)
            LWP_mmrfunc_12 = interp1d(layer_heights_12[ind],liquid_water_12[ind],kind='linear',fill_value='extrapolate')
            LWP_mmr_12 = LWP_mmrfunc_12(10.0**log_layer_height)
            LWP_mmr_12[LWP_mmr_12 < 0.0] = 0.0

            ice_water_12 = ice_water[itime,index,ilat,ilon]
            ind = (ice_water_12 != -999)
            IWP_mmrfunc_12 = interp1d(layer_heights_12[ind],ice_water_12[ind],kind='linear',fill_value='extrapolate')
            IWP_mmr_12 = IWP_mmrfunc_12(10.0**log_layer_height)
            IWP_mmr_12[IWP_mmr_12 < 0.0] = 0.0

            eastward_wind_12 = eastward_wind[itime,index,ilat,ilon]
            ind = (eastward_wind_12 != -999)
            Ufunc_12 = interp1d(layer_heights_12[ind],eastward_wind_12[ind],kind='linear',fill_value='extrapolate')
            # U_12 = Ufunc_12(10.0**log_layer_height)
            U_12 = Ufunc_12(el).flatten()[0]

            northward_wind_12 = northward_wind[itime,index,ilat,ilon]
            ind = (northward_wind_12 != -999)
            Vfunc_12 = interp1d(layer_heights_12[ind],northward_wind_12[ind],kind='linear',fill_value='extrapolate')
            # V_12 = Vfunc_12(10.0**log_layer_height)
            V_12 = Vfunc_12(el).flatten()[0]

            # convert LWP and IWP mass mixing ratios to column densities
            dP = P_12[0:-1] - P_12[1:]
            P_mid = np.concatenate((P_12[0:-1] + (dP/2.0),[P_12[-1]+(dP[-1]/2.0)],[P_12[-1]-(dP[-1]/2.0)]))
            dP = P_mid[0:-1] - P_mid[1:]
            LWP_12 = LWP_mmr_12*dP/g0
            IWP_12 = IWP_mmr_12*dP/g0

            # compute the PWV
            integrand = q_12 / (1.0 - q_12)
            integral = -np.sum(0.5*(integrand[1:] + integrand[0:-1])*(P_12[1:] - P_12[0:-1]))
            integral /= (rho_H20*g0)
            integral *= 1000.0
            PWV_12 = integral

            ##################################################
            # third corner

            ilat = ilat2
            ilon = ilon1

            # extract the heights of the atmospheric layers bracketing the telescope elevation
            index = (layer_heights[itime,:,ilat,ilon] >= 0.0)
            layer_heights_21 = layer_heights[itime,index,ilat,ilon]

            pressure_levels_21 = pressure_levels[index]
            Pfunc_21 = interp1d(layer_heights_21,np.log10(pressure_levels_21),kind='linear',fill_value='extrapolate')
            P_21 = 10.0**Pfunc_21(10.0**log_layer_height)

            H_21 = 10.0**log_layer_height

            temperature_levels_21 = temperature_levels[itime,index,ilat,ilon]
            ind = (temperature_levels_21 != -999)
            Tfunc_21 = interp1d(layer_heights_21[ind],temperature_levels_21[ind],kind='linear',fill_value='extrapolate')
            T_21 = Tfunc_21(10.0**log_layer_height)

            specific_humidity_21 = specific_humidity[itime,index,ilat,ilon]
            ind = (specific_humidity_21 != -999)
            qfunc_21 = interp1d(np.log10(layer_heights_21[ind]),np.log10(specific_humidity_21[ind]),kind='linear',fill_value='extrapolate')
            q_21 = 10.0**qfunc_21(log_layer_height)

            O3_vmr_21 = O3_vmr[itime,index,ilat,ilon]
            ind = (O3_vmr_21 != -999)
            O3func_21 = interp1d(layer_heights_21[ind],np.log10(O3_vmr_21[ind]),kind='linear',fill_value='extrapolate')
            O3_21 = 10.0**O3func_21(10.0**log_layer_height)

            H2O_vmr_21 = H2O_vmr[itime,index,ilat,ilon]
            ind = (H2O_vmr_21 != -999)
            # H2Ofunc_21 = interp1d(layer_heights_21[ind],np.log10(H2O_vmr_21[ind]),kind='linear',fill_value='extrapolate')
            # H2O_21 = 10.0**H2Ofunc_21(10.0**log_layer_height)
            H2Ofunc_21 = interp1d(layer_heights_21[ind],H2O_vmr_21[ind],kind='linear',fill_value='extrapolate')
            H2O_21 = H2Ofunc_21(10.0**log_layer_height)
            H2O_21[H2O_21 < 0.0] = 0.0

            liquid_water_21 = liquid_water[itime,index,ilat,ilon]
            ind = (liquid_water_21 != -999)
            LWP_mmrfunc_21 = interp1d(layer_heights_21[ind],liquid_water_21[ind],kind='linear',fill_value='extrapolate')
            LWP_mmr_21 = LWP_mmrfunc_21(10.0**log_layer_height)
            LWP_mmr_21[LWP_mmr_21 < 0.0] = 0.0

            ice_water_21 = ice_water[itime,index,ilat,ilon]
            ind = (ice_water_21 != -999)
            IWP_mmrfunc_21 = interp1d(layer_heights_21[ind],ice_water_21[ind],kind='linear',fill_value='extrapolate')
            IWP_mmr_21 = IWP_mmrfunc_21(10.0**log_layer_height)
            IWP_mmr_21[IWP_mmr_21 < 0.0] = 0.0

            eastward_wind_21 = eastward_wind[itime,index,ilat,ilon]
            ind = (eastward_wind_21 != -999)
            Ufunc_21 = interp1d(layer_heights_21[ind],eastward_wind_21[ind],kind='linear',fill_value='extrapolate')
            # U_21 = Ufunc_21(10.0**log_layer_height)
            U_21 = Ufunc_21(el).flatten()[0]

            northward_wind_21 = northward_wind[itime,index,ilat,ilon]
            ind = (northward_wind_21 != -999)
            Vfunc_21 = interp1d(layer_heights_21[ind],northward_wind_21[ind],kind='linear',fill_value='extrapolate')
            # V_21 = Vfunc_21(10.0**log_layer_height)
            V_21 = Vfunc_21(el).flatten()[0]

            # convert LWP and IWP mass mixing ratios to column densities
            dP = P_21[0:-1] - P_21[1:]
            P_mid = np.concatenate((P_21[0:-1] + (dP/2.0),[P_21[-1]+(dP[-1]/2.0)],[P_21[-1]-(dP[-1]/2.0)]))
            dP = P_mid[0:-1] - P_mid[1:]
            LWP_21 = LWP_mmr_21*dP/g0
            IWP_21 = IWP_mmr_21*dP/g0

            # compute the PWV
            integrand = q_21 / (1.0 - q_21)
            integral = -np.sum(0.5*(integrand[1:] + integrand[0:-1])*(P_21[1:] - P_21[0:-1]))
            integral /= (rho_H20*g0)
            integral *= 1000.0
            PWV_21 = integral

            ##################################################
            # fourth corner

            ilat = ilat2
            ilon = ilon2

            # extract the heights of the atmospheric layers bracketing the telescope elevation
            index = (layer_heights[itime,:,ilat,ilon] >= 0.0)
            layer_heights_22 = layer_heights[itime,index,ilat,ilon]

            pressure_levels_22 = pressure_levels[index]
            Pfunc_22 = interp1d(layer_heights_22,np.log10(pressure_levels_22),kind='linear',fill_value='extrapolate')
            P_22 = 10.0**Pfunc_22(10.0**log_layer_height)

            H_22 = 10.0**log_layer_height

            temperature_levels_22 = temperature_levels[itime,index,ilat,ilon]
            ind = (temperature_levels_22 != -999)
            Tfunc_22 = interp1d(layer_heights_22[ind],temperature_levels_22[ind],kind='linear',fill_value='extrapolate')
            T_22 = Tfunc_22(10.0**log_layer_height)

            specific_humidity_22 = specific_humidity[itime,index,ilat,ilon]
            ind = (specific_humidity_22 != -999)
            qfunc_22 = interp1d(np.log10(layer_heights_22[ind]),np.log10(specific_humidity_22[ind]),kind='linear',fill_value='extrapolate')
            q_22 = 10.0**qfunc_22(log_layer_height)

            O3_vmr_22 = O3_vmr[itime,index,ilat,ilon]
            ind = (O3_vmr_22 != -999)
            O3func_22 = interp1d(layer_heights_22[ind],np.log10(O3_vmr_22[ind]),kind='linear',fill_value='extrapolate')
            O3_22 = 10.0**O3func_22(10.0**log_layer_height)

            H2O_vmr_22 = H2O_vmr[itime,index,ilat,ilon]
            ind = (H2O_vmr_22 != -999)
            # H2Ofunc_22 = interp1d(layer_heights_22[ind],np.log10(H2O_vmr_22[ind]),kind='linear',fill_value='extrapolate')
            # H2O_22 = 10.0**H2Ofunc_22(10.0**log_layer_height)
            H2Ofunc_22 = interp1d(layer_heights_22[ind],H2O_vmr_22[ind],kind='linear',fill_value='extrapolate')
            H2O_22 = H2Ofunc_22(10.0**log_layer_height)
            H2O_22[H2O_22 < 0.0] = 0.0

            liquid_water_22 = liquid_water[itime,index,ilat,ilon]
            ind = (liquid_water_22 != -999)
            LWP_mmrfunc_22 = interp1d(layer_heights_22[ind],liquid_water_22[ind],kind='linear',fill_value='extrapolate')
            LWP_mmr_22 = LWP_mmrfunc_22(10.0**log_layer_height)
            LWP_mmr_22[LWP_mmr_22 < 0.0] = 0.0

            ice_water_22 = ice_water[itime,index,ilat,ilon]
            ind = (ice_water_22 != -999)
            IWP_mmrfunc_22 = interp1d(layer_heights_22[ind],ice_water_22[ind],kind='linear',fill_value='extrapolate')
            IWP_mmr_22 = IWP_mmrfunc_22(10.0**log_layer_height)
            IWP_mmr_22[IWP_mmr_22 < 0.0] = 0.0

            eastward_wind_22 = eastward_wind[itime,index,ilat,ilon]
            ind = (eastward_wind_22 != -999)
            Ufunc_22 = interp1d(layer_heights_22[ind],eastward_wind_22[ind],kind='linear',fill_value='extrapolate')
            # U_22 = Ufunc_22(10.0**log_layer_height)
            U_22 = Ufunc_22(el).flatten()[0]

            northward_wind_22 = northward_wind[itime,index,ilat,ilon]
            ind = (northward_wind_22 != -999)
            Vfunc_22 = interp1d(layer_heights_22[ind],northward_wind_22[ind],kind='linear',fill_value='extrapolate')
            # V_22 = Vfunc_22(10.0**log_layer_height)
            V_22 = Vfunc_22(el).flatten()[0]

            # convert LWP and IWP mass mixing ratios to column densities
            dP = P_22[0:-1] - P_22[1:]
            P_mid = np.concatenate((P_22[0:-1] + (dP/2.0),[P_22[-1]+(dP[-1]/2.0)],[P_22[-1]-(dP[-1]/2.0)]))
            dP = P_mid[0:-1] - P_mid[1:]
            LWP_22 = LWP_mmr_22*dP/g0
            IWP_22 = IWP_mmr_22*dP/g0

            # compute the PWV
            integrand = q_22 / (1.0 - q_22)
            integral = -np.sum(0.5*(integrand[1:] + integrand[0:-1])*(P_22[1:] - P_22[0:-1]))
            integral /= (rho_H20*g0)
            integral *= 1000.0
            PWV_22 = integral

        ##################################################
        ##################################################
        # next, interpolate in the lat/lon directions

        with stats.record_wallclock('interpolation wallclock', raw_stats):

            x = lat
            y = lon
            x1 = latarr[ilat1]
            x2 = latarr[ilat2]
            y1 = lonarr[ilon1]
            y2 = lonarr[ilon2]

            P = bilinear(x,y,x1,x2,y1,y2,P_11,P_12,P_21,P_22)
            T = bilinear(x,y,x1,x2,y1,y2,T_11,T_12,T_21,T_22)
            q = bilinear(x,y,x1,x2,y1,y2,q_11,q_12,q_21,q_22)
            O3 = bilinear(x,y,x1,x2,y1,y2,O3_11,O3_12,O3_21,O3_22)
            H2O = bilinear(x,y,x1,x2,y1,y2,H2O_11,H2O_12,H2O_21,H2O_22)
            LWP = bilinear(x,y,x1,x2,y1,y2,LWP_11,LWP_12,LWP_21,LWP_22)
            IWP = bilinear(x,y,x1,x2,y1,y2,IWP_11,IWP_12,IWP_21,IWP_22)
            U = bilinear(x,y,x1,x2,y1,y2,U_11,U_12,U_21,U_22)
            V = bilinear(x,y,x1,x2,y1,y2,V_11,V_12,V_21,V_22)
            PWV = bilinear(x,y,x1,x2,y1,y2,PWV_11,PWV_12,PWV_21,PWV_22)

        ##################################################
        ##################################################
        # generate am input file

        with stats.record_wallclock('run am wallclock', raw_stats):
            with stats.record_iowait('run am iowait', raw_stats):

                # convert pressure levels to mbar
                P_here = P[::-1]*(1.0e-2)

                # reorder other quantities to match increasing pressure order
                T_here = T[::-1]
                O3_here = O3[::-1]
                H2O_here = H2O[::-1]
                q_here = q[::-1]
                LWP_here = LWP[::-1]
                IWP_here = IWP[::-1]

                # write am input file
                with open(am_input_filename,'w') as f:

                    # write out header
                    header = ''
                    header += '# File ' + am_input_filename + ' - am model configuration file' + '\n'
                    header += '# ' + '\n'
                    header += '# Date:            ' + year + '-' + month + '-' + day + '\n'
                    header += '# Latitude:        ' + str(lat) + '\n'
                    header += '# Longitude:       ' + str(lon) + '\n'
                    header += '# Time:            ' + str(itime*3) + ' UTC' + '\n'
                    header += '# Elevation:       ' + str(el) + ' meters' +'\n'
                    header += '# Wind (east):     ' + f'{U:.5g}' + ' m/s' +'\n'
                    header += '# Wind (north):    ' + f'{V:.5g}' + ' m/s' +'\n'
                    header += '# PWV:             ' + f'{PWV:.6g}' + ' mm' +'\n'
                    header += '# ' +'\n'

                    header += '\n'
                    header += 'f ' + str(fmin) + ' GHz  ' + str(fmax) + ' GHz  ' + str(df) + ' GHz' + '\n'
                    header += 'output f GHz  tau Tb K' +'\n'
                    header += 'za 0 deg' + '\n'
                    header += 'tol 1e-4' + '\n'
                    header += 'T0 2.725 K' + '\n'
                    f.write(header)

                    for i in range(len(T_here)):
                        strhere = ''
                        strhere += '\n'

                        # record which layer of the atmosphere we're in
                        if P_here[i] < 1.0:
                            layer = 'mesosphere'
                        elif (P_here[i] >= 1.0) & (P_here[i] < 100.0):
                            layer = 'stratosphere'
                        else:
                            layer = 'troposphere'
                        strhere += 'layer ' + layer + '\n'

                        # record the pressure
                        strhere += 'Pbase ' + f'{P_here[i]:.6g}' + ' mbar\n'

                        # record the temperature
                        strhere += 'Tbase ' + f'{T_here[i]:.4g}' + ' K\n'

                        # record the lineshape
                        if P_here[i] <= 1.0:
                            strhere += 'lineshape Voigt-Kielkopf' +'\n'

                        # record the dry air component
                        strhere += 'column dry_air vmr' + '\n'
                        
                        # record the H2O volume mixing ratio
                        if H2O_here[i] > 0.0:
                            strhere += 'column h2o vmr ' + f'{H2O_here[i]:.3e}' + '\n'
                        else:
                            strhere += 'column h2o vmr 0.0' + '\n'

                        # record the O3 volume mixing ratio
                        if O3_here[i] > 0.0:
                            strhere += 'column o3 vmr ' + f'{O3_here[i]:.3e}' + '\n'
                        else:
                            strhere += 'column o3 vmr 0.0' + '\n'
                        
                        # convert liquid water to ice if the temperature is low enough
                        if T_here[i] <= 238.0:
                            liquid_here = 0.0
                            ice_here = IWP_here[i] + LWP_here[i]
                        else:
                            liquid_here = LWP_here[i]
                            ice_here = IWP_here[i]
                        
                        # convert ice water to liquid if the temperature is high enough
                        if T_here[i] >= 299.5:
                            liquid_here += ice_here
                            ice_here = 0.0
                        
                        # record the liquid water component
                        if liquid_here > 0.0:
                            strhere += 'column lwp_abs_Rayleigh ' + f'{liquid_here:.3e}' + ' kg*m-2' + '\n'

                        # record the ice water component
                        if ice_here > 0.0:
                            strhere += 'column iwp_abs_Rayleigh ' + f'{ice_here:.3e}' + ' kg*m-2' + '\n'

                        f.write(strhere)

                ##################################################
                ##################################################
                # run am

                os.system('/n/home00/dpesce/.local/bin/am/am_serial '+am_input_filename+' > '+am_output_filename)

        ##################################################
        ##################################################
        # compute and store PCA components

        with stats.record_wallclock('construct pca wallclock', raw_stats):
            with stats.record_iowait('construct pca iowait', raw_stats):

                # load eigenspectra
                meanspec_tau = np.loadtxt('./eigenspectra/spectrum_mean.txt',unpack=True)
                meanspec_Tb = np.loadtxt('./eigenspectra_Tb/spectrum_mean.txt',unpack=True)
                tau_spectra = list()
                Tb_spectra = list()
                for i in range(Ncomp):
                    spechere = np.loadtxt('./eigenspectra/spectrum_'+str(i).zfill(4)+'.txt',unpack=True)
                    spechere_Tb = np.loadtxt('./eigenspectra_Tb/spectrum_'+str(i).zfill(4)+'.txt',unpack=True)
                    tau_spectra.append(spechere)
                    Tb_spectra.append(spechere_Tb)

                # load am file
                freqhere, tauhere, Tbhere = np.loadtxt(am_output_filename,unpack=True)
                logspec = np.log10(tauhere) - meanspec_tau
                Tbspec = Tbhere - meanspec_Tb

                coeffs = np.zeros(Ncomp)
                for ispec, eigenspec in enumerate(tau_spectra):
                    coeffs[ispec] = np.sum(logspec*eigenspec)

                coeffs_Tb = np.zeros(Ncomp)
                for ispec, eigenspec in enumerate(Tb_spectra):
                    coeffs_Tb[ispec] = np.sum(Tbspec*eigenspec)

                # save opacity spectrum as text file
                with open(outfilename_tau, 'w') as outfile:
                    for coeff in coeffs:
                        outfile.write(str(coeff)+'\n')

                # save Tb spectrum as text file
                with open(outfilename_Tb, 'w') as outfile:
                    for coeff in coeffs_Tb:
                        outfile.write(str(coeff)+'\n')

                # # # reconstruct a spectrum using, e.g.:
                # coeffs_rec = np.loadtxt(outfilename_tau)
                # reconstructed_spectrum = np.zeros(2001)
                # for ispec, eigenspec in enumerate(tau_spectra):
                #     reconstructed_spectrum += coeffs_rec[ispec]*eigenspec
                # reconstructed_spectrum += meanspec_tau
                # reconstructed_spectrum = 10.0**reconstructed_spectrum

##################################################
# make list of inputs for paramsurvey

print('Generating input list....')

months = ['01','02','03','04','05','06','07','08','09','10','11','12']
days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
itimes = [0,1,2,3,4,5,6,7]

# define the parameters to survey over
params = OrderedDict([('site', sites),
                      ('year', years),
                      ('month', months),
                      ('day', days),
                      ('itime', itimes)])

# initialize psets
psets = paramsurvey.params.product(params)

# add additional info that it's useful to pass to the worker function
input_file_list = list()
am_input_filename_list = list()
am_output_filename_list = list()
outfilename_tau_list = list()
outfilename_Tb_list = list()
drop_me = list()
for row in psets.itertuples():

    # input MERRA data file
    year = row.year
    month = row.month
    day = row.day

    if (int(year) < 1992):
        input_file = MERRA_dirname + year +'/MERRA2_100.inst3_3d_asm_Np.'+year+month+day+'.nc4'
    elif (int(year) >= 1992) & (int(year) < 2001):
        input_file = MERRA_dirname + year +'/MERRA2_200.inst3_3d_asm_Np.'+year+month+day+'.nc4'
    elif (int(year) >= 2001) & (int(year) < 2011):
        input_file = MERRA_dirname + year +'/MERRA2_300.inst3_3d_asm_Np.'+year+month+day+'.nc4'
    else:
        input_file = MERRA_dirname + year +'/MERRA2_400.inst3_3d_asm_Np.'+year+month+day+'.nc4'
    input_file_list.append(input_file)

    # check if the input file exists
    if not os.path.exists(input_file):
        drop_me.append(row.Index)

    # am file info
    site = row.site
    itime = row.itime
    dirname = './sites/'+site+'/'+year+'/'+month
    am_input_filename = dirname+'/am_input_file_day'+day+'_time'+str(itime)+'.txt'
    am_output_filename = dirname+'/am_output_file_day'+day+'_time'+str(itime)+'.txt'
    am_input_filename_list.append(am_input_filename)
    am_output_filename_list.append(am_output_filename)

    # create the directory if it doesn't already exist
    os.makedirs(dirname,exist_ok=True)

    # output tau and Tb table info
    outfilename_tau = dirname + '/output_tau_day'+day+'_time'+str(itime)+'.txt'
    outfilename_Tb = dirname + '/output_Tb_day'+day+'_time'+str(itime)+'.txt'
    outfilename_tau_list.append(outfilename_tau)
    outfilename_Tb_list.append(outfilename_Tb)

    # if both of the desired output files exist already, skip this one
    if (os.path.exists(outfilename_tau) & os.path.exists(outfilename_Tb)):
        drop_me.append(row.Index)

# add as columns to the dataframe
psets['input_file'] = input_file_list
psets['am_input_filename'] = am_input_filename_list
psets['am_output_filename'] = am_output_filename_list
psets['outfilename_tau'] = outfilename_tau_list
psets['outfilename_Tb'] = outfilename_Tb_list

# removing pre-completed work
before = len(psets)
psets = psets.drop(drop_me)
after = len(psets)
print('Removed '+str(before-after)+' pre-existing jobs.')

print('Input list created.  There are '+str(len(psets))+' work packages to execute.')

#######################################################
# run primary computation

paramsurvey.init(backend='ray')
results = paramsurvey.map(tabulate_weather, psets, verbose=1, group_size=1)
