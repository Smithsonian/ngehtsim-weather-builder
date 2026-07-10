###############################################################
# imports

import numpy as np
import struct
import glob
import os

import itertools
import paramsurvey
import paramsurvey.params
import paramsurvey.stats as stats
from collections import OrderedDict
import io
import contextlib

from daily_postproc import complete_daily_inputs

###############################################################

toplevel_dir = './sites/'
toplevel_out = './weather_data/'

# number of PCA components to save
Ncomps = 40

###############################################################
# load eigenspectra

meanspec_tau = np.loadtxt('./eigenspectra/spectrum_mean.txt',unpack=True)
meanspec_Tb = np.loadtxt('./eigenspectra_Tb/spectrum_mean.txt',unpack=True)
tau_spectra = list()
Tb_spectra = list()
for i in range(Ncomps):
    spechere = np.loadtxt('./eigenspectra/spectrum_'+str(i).zfill(4)+'.txt',unpack=True)
    spechere_Tb = np.loadtxt('./eigenspectra_Tb/spectrum_'+str(i).zfill(4)+'.txt',unpack=True)
    tau_spectra.append(spechere)
    Tb_spectra.append(spechere_Tb)

###############################################################
# primary computation function

def postproc(pset, system_kwargs, user_kwargs):
    raw_stats = system_kwargs['raw_stats']

    # pass stdout to avoid slowing down ray
    my_stdout = io.StringIO()
    with contextlib.redirect_stdout(my_stdout):

        ###############################################################
        # parse parameter set

        site = pset['site']
        month = pset['month']
        monthlabel = pset['monthlabel']

        ###############################################################
        # extract the data for this month across all years

        # initialize lists
        yearlist = list()
        monthlist = list()
        daylist = list()
        timelist = list()
        coeffs_tau_list = list()
        coeffs_Tb_list = list()
        PWV_list = list()
        windspeed_list = list()
        Pbase_list = list()
        Tbase_list = list()

        # loop over all years
        years = np.sort(glob.glob(toplevel_dir + site + '/*'))
        for iyear in range(len(years)):

            # current year info
            yeardir = years[iyear]
            year = yeardir.split('/')[-1]

            # filename lists
            filenames_tau = np.sort(glob.glob(yeardir+'/'+month+'/output_tau*.txt'))
            filenames_Tb = np.sort(glob.glob(yeardir+'/'+month+'/output_Tb*.txt'))
            filenames_am = np.sort(glob.glob(yeardir+'/'+month+'/am_input_file*.txt'))

            # loop over all files
            for ifile in range(len(filenames_tau)):

                # filenames
                filename_tau = filenames_tau[ifile]
                filename_Tb = filenames_Tb[ifile]
                filename_am = filenames_am[ifile]

                # extract day + time
                day = filename_tau.split('_')[-2][-2:]
                itime = filename_tau.split('_')[-1].split('.')[0][-1]
                time = float(itime)*3.0

                # check that the day + time are consistent among all files
                day2 = filename_Tb.split('_')[-2][-2:]
                day3 = filename_am.split('_')[-2][-2:]
                if ((day2 != day) | (day3 != day)):
                    print('MAJOR ISSUE: the input files are being read in an inconsistent order')
                itime2 = filename_Tb.split('_')[-1].split('.')[0][-1]
                itime3 = filename_am.split('_')[-1].split('.')[0][-1]
                if ((itime2 != itime) | (itime3 != itime)):
                    print('MAJOR ISSUE: the input files are being read in an inconsistent order')

                # PCA coefficients
                coeffs_tau = np.loadtxt(filename_tau)
                coeffs_Tb = np.loadtxt(filename_Tb)

                # append relevant quantities to the running lists
                yearlist.append(int(year))
                monthlist.append(int(month))
                daylist.append(int(day))
                timelist.append(int(itime))
                coeffs_tau_list.append(coeffs_tau)
                coeffs_Tb_list.append(coeffs_Tb)

                # get remaining weather quantities from am input file
                lines = open(filename_am, 'r').readlines()

                # PWV
                PWV = float(lines[9].split(' ')[-2])
                PWV_list.append(PWV)

                # windspeed
                eastwind = float(lines[7].split(' ')[-2])
                northwind = float(lines[8].split(' ')[-2])
                windspeed_list.append(np.sqrt((eastwind**2.0) + (northwind**2.0)))

                # base pressure and temperature
                foundit = False
                for i in range(1,len(lines)):
                    if 'Tbase' in lines[-i]:
                        Tbase = float(lines[-i].split(' ')[-2])
                        Tbase_list.append(Tbase)
                        foundit = True
                    if foundit:
                        break
                foundit = False
                for i in range(1,len(lines)):
                    if 'Pbase' in lines[-i]:
                        Pbase = float(lines[-i].split(' ')[-2])
                        Pbase_list.append(Pbase)
                        foundit = True
                    if foundit:
                        break

        ###############################################################
        # compute daily means

        # initialize lists
        yearlist_dm = list()
        monthlist_dm = list()
        daylist_dm = list()
        coeffs_tau_list_dm = list()
        coeffs_Tb_list_dm = list()
        PWV_list_dm = list()
        windspeed_list_dm = list()
        Pbase_list_dm = list()
        Tbase_list_dm = list()

        # loop over all years
        years = np.sort(glob.glob(toplevel_dir + site + '/*'))
        for iyear in range(len(years)):

            # current year info
            yeardir = years[iyear]
            year = yeardir.split('/')[-1]

            # aggregate only days that have all eight three-hour inputs
            for daily_inputs in complete_daily_inputs(yeardir, month):

                dayhere = str(daily_inputs.day).zfill(2)
                filenames_tau = daily_inputs.tau
                filenames_Tb = daily_inputs.tb
                filenames_am = daily_inputs.am

                # loop over all files and average across the day
                avg_tau_spec = np.zeros(2001)
                avg_Tb_spec = np.zeros(2001)
                avg_PWV = 0.0
                avg_windspeed = 0.0
                avg_Pbase = 0.0
                avg_Tbase = 0.0
                for ifile in range(len(filenames_tau)):

                    # filenames
                    filename_tau = filenames_tau[ifile]
                    filename_Tb = filenames_Tb[ifile]
                    filename_am = filenames_am[ifile]

                    # PCA coefficients
                    coeffs_tau = np.loadtxt(filename_tau)
                    coeffs_Tb = np.loadtxt(filename_Tb)

                    # reconstructed tau spectrum
                    reconstructed_spectrum = np.zeros(2001)
                    for ispec, eigenspec in enumerate(tau_spectra):
                        reconstructed_spectrum += coeffs_tau[ispec]*eigenspec
                    reconstructed_spectrum += meanspec_tau
                    reconstructed_tau_spectrum = 10.0**reconstructed_spectrum
                    avg_tau_spec += reconstructed_tau_spectrum

                    # reconstructed Tb spectrum
                    reconstructed_spectrum = np.zeros(2001)
                    for ispec, eigenspec in enumerate(Tb_spectra):
                        reconstructed_spectrum += coeffs_Tb[ispec]*eigenspec
                    reconstructed_spectrum += meanspec_Tb
                    reconstructed_Tb_spectrum = reconstructed_spectrum
                    avg_Tb_spec += reconstructed_Tb_spectrum

                    # get remaining weather quantities from am input file
                    lines = open(filename_am, 'r').readlines()

                    # PWV
                    PWV = float(lines[9].split(' ')[-2])
                    avg_PWV += PWV

                    # windspeed
                    eastwind = float(lines[7].split(' ')[-2])
                    northwind = float(lines[8].split(' ')[-2])
                    avg_windspeed += np.sqrt((eastwind**2.0) + (northwind**2.0))

                    # base pressure and temperature
                    foundit = False
                    for i in range(1,len(lines)):
                        if 'Tbase' in lines[-i]:
                            Tbase = float(lines[-i].split(' ')[-2])
                            avg_Tbase += Tbase
                            foundit = True
                        if foundit:
                            break
                    foundit = False
                    for i in range(1,len(lines)):
                        if 'Pbase' in lines[-i]:
                            Pbase = float(lines[-i].split(' ')[-2])
                            avg_Pbase += Pbase
                            foundit = True
                        if foundit:
                            break

                sample_count = float(len(filenames_tau))
                avg_tau_spec /= sample_count
                avg_Tb_spec /= sample_count
                avg_PWV /= sample_count
                avg_windspeed /= sample_count
                avg_Pbase /= sample_count
                avg_Tbase /= sample_count

                # recompute PCA components for the average tau spectrum
                logspec = np.log10(avg_tau_spec) - meanspec_tau
                coeffs_tau = np.zeros(Ncomps)
                for ispec, eigenspec in enumerate(tau_spectra):
                    coeffs_tau[ispec] = np.sum(logspec*eigenspec)

                # recompute PCA components for the average tau spectrum
                Tbspec = avg_Tb_spec - meanspec_Tb
                coeffs_Tb = np.zeros(Ncomps)
                for ispec, eigenspec in enumerate(Tb_spectra):
                    coeffs_Tb[ispec] = np.sum(Tbspec*eigenspec)

                # append relevant quantities to the running lists
                yearlist_dm.append(int(year))
                monthlist_dm.append(int(month))
                daylist_dm.append(int(dayhere))
                coeffs_tau_list_dm.append(coeffs_tau)
                coeffs_Tb_list_dm.append(coeffs_Tb)
                PWV_list_dm.append(avg_PWV)
                windspeed_list_dm.append(avg_windspeed)
                Pbase_list_dm.append(avg_Pbase)
                Tbase_list_dm.append(avg_Tbase)

        ###############################################################
        # save files containing the aggregated weather info

        # create the output directory
        outdirname = toplevel_out + site + '/' + monthlabel
        os.makedirs(outdirname,exist_ok=True)

        # loop over all list elements and write files
        with open(outdirname+'/tau_alltimes.txt', 'wb') as binary_file_tau:
            with open(outdirname+'/Tb_alltimes.txt', 'wb') as binary_file_Tb:

                # record the length of a single line
                linelength = np.int16((Ncomps*2) + 5)
                binary_file_tau.write(bytearray(np.array([linelength])))
                binary_file_Tb.write(bytearray(np.array([linelength])))

                for ilist in range(len(yearlist)):

                    # reduce numerical precision
                    yearhere = np.int16(yearlist[ilist])
                    monthhere = np.int8(monthlist[ilist])
                    dayhere = np.int8(daylist[ilist])
                    timehere = np.int8(timelist[ilist])
                    coeffs16_tau = coeffs_tau_list[ilist].astype(np.float16)
                    coeffs16_Tb = coeffs_Tb_list[ilist].astype(np.float16)

                    # make bytearrays
                    ba_tau = bytearray(np.array([yearhere]))
                    ba_tau.append(monthhere)
                    ba_tau.append(dayhere)
                    ba_tau.append(timehere)
                    ba_tau += bytearray(coeffs16_tau)

                    ba_Tb = bytearray(np.array([yearhere]))
                    ba_Tb.append(monthhere)
                    ba_Tb.append(dayhere)
                    ba_Tb.append(timehere)
                    ba_Tb += bytearray(coeffs16_Tb)

                    # write to binary files
                    binary_file_tau.write(ba_tau)
                    binary_file_Tb.write(ba_Tb)

        # other weather info
        with open(outdirname+'/PWV_alltimes.txt', 'wb') as binary_file_PWV:
            with open(outdirname+'/Tbase_alltimes.txt', 'wb') as binary_file_Tbase:
                with open(outdirname+'/Pbase_alltimes.txt', 'wb') as binary_file_Pbase:
                    with open(outdirname+'/windspeed_alltimes.txt', 'wb') as binary_file_windspeed:

                        # record the length of a single line
                        linelength = np.int16(13)
                        binary_file_PWV.write(bytearray(np.array([linelength])))
                        binary_file_windspeed.write(bytearray(np.array([linelength])))
                        binary_file_Pbase.write(bytearray(np.array([linelength])))
                        binary_file_Tbase.write(bytearray(np.array([linelength])))

                        for ilist in range(len(yearlist)):

                            # reduce numerical precision
                            yearhere = np.int16(yearlist[ilist])
                            monthhere = np.int8(monthlist[ilist])
                            dayhere = np.int8(daylist[ilist])
                            timehere = np.int8(timelist[ilist])
                            PWV_here = np.float64(PWV_list[ilist])
                            if not np.isfinite(PWV_here):
                                print('Warning: got a nan for PWV on '+str(yearlist[ilist])+'-'+str(monthlist[ilist])+'-'+str(daylist[ilist])+'-'+str(timelist[ilist]))
                                PWV_here = np.float64(0.0)
                            windspeed_here = np.float64(windspeed_list[ilist])
                            Pbase_here = np.float64(Pbase_list[ilist])
                            Tbase_here = np.float64(Tbase_list[ilist])

                            # make bytearrays
                            ba_PWV = bytearray(np.array([yearhere]))
                            ba_PWV.append(monthhere)
                            ba_PWV.append(dayhere)
                            ba_PWV.append(timehere)
                            ba_PWV += bytes(PWV_here)

                            ba_windspeed = bytearray(np.array([yearhere]))
                            ba_windspeed.append(monthhere)
                            ba_windspeed.append(dayhere)
                            ba_windspeed.append(timehere)
                            ba_windspeed += bytes(windspeed_here)

                            ba_Tbase = bytearray(np.array([yearhere]))
                            ba_Tbase.append(monthhere)
                            ba_Tbase.append(dayhere)
                            ba_Tbase.append(timehere)
                            ba_Tbase +=bytes(Tbase_here)

                            ba_Pbase = bytearray(np.array([yearhere]))
                            ba_Pbase.append(monthhere)
                            ba_Pbase.append(dayhere)
                            ba_Pbase.append(timehere)
                            ba_Pbase += bytes(Pbase_here)

                            # write to binary files
                            binary_file_PWV.write(ba_PWV)
                            binary_file_windspeed.write(ba_windspeed)
                            binary_file_Pbase.write(ba_Pbase)
                            binary_file_Tbase.write(ba_Tbase)

        ###############################################################
        # do the same for the daily averages

        with open(outdirname+'/tau.txt', 'wb') as binary_file_tau:
            with open(outdirname+'/Tb.txt', 'wb') as binary_file_Tb:

                # record the length of a single line
                linelength = np.int16((Ncomps*2) + 4)
                binary_file_tau.write(bytearray(np.array([linelength])))
                binary_file_Tb.write(bytearray(np.array([linelength])))

                for ilist in range(len(yearlist_dm)):

                    # reduce numerical precision
                    yearhere = np.int16(yearlist_dm[ilist])
                    monthhere = np.int8(monthlist_dm[ilist])
                    dayhere = np.int8(daylist_dm[ilist])
                    coeffs16_tau = coeffs_tau_list_dm[ilist].astype(np.float16)
                    coeffs16_Tb = coeffs_Tb_list_dm[ilist].astype(np.float16)

                    # make bytearrays
                    ba_tau = bytearray(np.array([yearhere]))
                    ba_tau.append(monthhere)
                    ba_tau.append(dayhere)
                    ba_tau += bytearray(coeffs16_tau)

                    ba_Tb = bytearray(np.array([yearhere]))
                    ba_Tb.append(monthhere)
                    ba_Tb.append(dayhere)
                    ba_Tb += bytearray(coeffs16_Tb)

                    # write to binary files
                    binary_file_tau.write(ba_tau)
                    binary_file_Tb.write(ba_Tb)

        with open(outdirname+'/PWV.txt', 'wb') as binary_file_PWV:
            with open(outdirname+'/Tbase.txt', 'wb') as binary_file_Tbase:
                with open(outdirname+'/Pbase.txt', 'wb') as binary_file_Pbase:
                    with open(outdirname+'/windspeed.txt', 'wb') as binary_file_windspeed:

                        # record the length of a single line
                        linelength = np.int16(12)
                        binary_file_PWV.write(bytearray(np.array([linelength])))
                        binary_file_windspeed.write(bytearray(np.array([linelength])))
                        binary_file_Pbase.write(bytearray(np.array([linelength])))
                        binary_file_Tbase.write(bytearray(np.array([linelength])))

                        for ilist in range(len(yearlist_dm)):

                            # reduce numerical precision
                            yearhere = np.int16(yearlist_dm[ilist])
                            monthhere = np.int8(monthlist_dm[ilist])
                            dayhere = np.int8(daylist_dm[ilist])
                            PWV_here = np.float64(PWV_list_dm[ilist])
                            if not np.isfinite(PWV_here):
                                print('Warning: got a nan for PWV on '+str(yearlist_dm[ilist])+'-'+str(monthlist_dm[ilist])+'-'+str(daylist_dm[ilist]))
                                PWV_here = np.float64(0.0)
                            windspeed_here = np.float64(windspeed_list_dm[ilist])
                            Pbase_here = np.float64(Pbase_list_dm[ilist])
                            Tbase_here = np.float64(Tbase_list_dm[ilist])

                            # make bytearrays
                            ba_PWV = bytearray(np.array([yearhere]))
                            ba_PWV.append(monthhere)
                            ba_PWV.append(dayhere)
                            ba_PWV += bytes(PWV_here)

                            ba_windspeed = bytearray(np.array([yearhere]))
                            ba_windspeed.append(monthhere)
                            ba_windspeed.append(dayhere)
                            ba_windspeed += bytes(windspeed_here)

                            ba_Tbase = bytearray(np.array([yearhere]))
                            ba_Tbase.append(monthhere)
                            ba_Tbase.append(dayhere)
                            ba_Tbase +=bytes(Tbase_here)

                            ba_Pbase = bytearray(np.array([yearhere]))
                            ba_Pbase.append(monthhere)
                            ba_Pbase.append(dayhere)
                            ba_Pbase += bytes(Pbase_here)

                            # write to binary files
                            binary_file_PWV.write(ba_PWV)
                            binary_file_windspeed.write(ba_windspeed)
                            binary_file_Pbase.write(ba_Pbase)
                            binary_file_Tbase.write(ba_Tbase)

##################################################
# functions to decode the stored binary representation

def read_binary(filename,Ncomps=40):

    with open(filename, 'rb') as binary_file:
        contents = bytearray(binary_file.read())

    linelength = int(contents[0:2][0])
    prelength = linelength - (2*Ncomps)
    Nlines = int((len(contents) - 2) / linelength)

    coeffs = np.zeros((Nlines,Ncomps))
    for i in range(Nlines):

        istart = 2 + (linelength*i)
        iend = istart + linelength
        linehere = contents[istart:iend]

        yearhere = int(struct.unpack('<h', linehere[0:2])[0])
        monthhere = int(struct.unpack('b', linehere[2:3])[0])
        dayhere = int(struct.unpack('b', linehere[3:4])[0])
        if prelength > 4:
            timehere = int(struct.unpack('b', linehere[4:5])[0])
        coeffs[i,:] = np.array(struct.unpack('<'+'e'*Ncomps, linehere[prelength:])).astype(float)

    return coeffs

def read_binary2(filename):

    with open(filename, 'rb') as binary_file:
        contents = bytearray(binary_file.read())

    linelength = int(contents[0:2][0])
    prelength = linelength - 8
    Nlines = int((len(contents) - 2) / linelength)

    vals = np.zeros(Nlines)
    for i in range(Nlines):
        istart = 2 + (linelength*i)
        iend = istart + linelength
        linehere = contents[istart:iend]

        yearhere = int(struct.unpack('<h', linehere[0:2])[0])
        monthhere = int(struct.unpack('b', linehere[2:3])[0])
        dayhere = int(struct.unpack('b', linehere[3:4])[0])
        if prelength > 4:
            timehere = int(struct.unpack('b', linehere[4:5])[0])
        vals[i] = float(struct.unpack('<d', linehere[prelength:])[0])

    return vals

#######################################################
# make list of inputs for paramsurvey

print('Generating input list....')

sites = ['ALMA','APEX','GLT','IRAM','JCMT','KP','LMT','NOEMA','SMA','SMT','SPT','AGGO','ALI','ARE','ATCA','BAJA','BAN','BAR','BDRY','BGA','BGK','BLDR','BMAC','BOL','BRZ','CAM','CAS','CAT','CNI','CTIO','DomeA','DomeC','DomeF','EFF','ELK','ERB','FAIR','FLWO','FUJI','GAM','GARS','GBT','GGAO','GLTS','GOR','HAN','HART','HAY','HESS','HIT','HOB','HOR','IRK','ISG','ISH','JARE','JELM','JELM2','JOD','KASH','KATH','KEN','KILI','KNM','KOG','KOKEE','KVNPC','KVNTN','KVNUS','KVNYS','LAS','LLA','LOS','MACGO','MAT','MATJ','MED','MET','MIZ','MOR','MUZ','NAN','NOB','NOR','NOTO','NYALE','NZ','OGA','ONS','ONSNE','ONSSW','ORG','OVRO','PAR','PIKE','PRKS','ROEN','ROT','SAN','SEJ','SGO','SHE','SIM','SKS','SMAR','SPX','SRT','STL','SUF','SVET','TAK','TNMA','TOR','TRL','TSU','UDSC','VLA','VLBBR','VLBFD','VLBHN','VLBKP','VLBLA','VLBMK','VLBNL','VLBOV','VLBPT','VLBSC','WARK','WEST','WETTZ','WSRT','XSMC','YAM','YAN','YAR','YBJ','YEB','YEBRG','ZELE','ZUG']

months = ['01','02','03','04','05','06','07','08','09','10','11','12']
monthlabels = ['01Jan','02Feb','03Mar','04Apr','05May','06Jun','07Jul','08Aug','09Sep','10Oct','11Nov','12Dec']

# define the parameters to survey over
params = OrderedDict([('site', sites),
                      ('month', months)])

# initialize psets
psets = paramsurvey.params.product(params)

# add additional info that it's useful to pass to the worker function
monthlabels_list = list()
drop_me = list()
for row in psets.itertuples():
    site = row.site
    month = row.month
    ind = (np.array(months) == month)
    monthlabel = np.array(monthlabels)[ind][0]
    monthlabels_list.append(monthlabel)

    outdirname = toplevel_out + site + '/' + monthlabel
    dropit = True
    dropit &= os.path.exists(outdirname+'/tau_alltimes.txt')
    dropit &= os.path.exists(outdirname+'/Tb_alltimes.txt')
    dropit &= os.path.exists(outdirname+'/PWV_alltimes.txt')
    dropit &= os.path.exists(outdirname+'/Tbase_alltimes.txt')
    dropit &= os.path.exists(outdirname+'/Pbase_alltimes.txt')
    dropit &= os.path.exists(outdirname+'/windspeed_alltimes.txt')
    dropit &= os.path.exists(outdirname+'/tau.txt')
    dropit &= os.path.exists(outdirname+'/Tb.txt')
    dropit &= os.path.exists(outdirname+'/PWV.txt')
    dropit &= os.path.exists(outdirname+'/Tbase.txt')
    dropit &= os.path.exists(outdirname+'/Pbase.txt')
    dropit &= os.path.exists(outdirname+'/windspeed.txt')
    if dropit:
        drop_me.append(row.Index)

# add as columns to the dataframe
psets['monthlabel'] = monthlabels_list

# removing pre-completed work
before = len(psets)
psets = psets.drop(drop_me)
after = len(psets)
print('Removed '+str(before-after)+' pre-existing jobs.')

print('Input list created.  There are '+str(len(psets))+' work packages to execute.')

#######################################################
# run primary computation

paramsurvey.init(backend='ray')
results = paramsurvey.map(postproc, psets, verbose=1, group_size=1)




