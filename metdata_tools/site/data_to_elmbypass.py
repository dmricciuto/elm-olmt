import numpy as np
from netCDF4 import Dataset
import os,sys
import gapfill
import write_elm_met
import re
import glob

#------- user input -------------
site = 'US-MMS'
inputdata = '/gpfs/wolf2/cades/cli185/proj-shared/zdr/inputdata'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import OLMTutils
siteinfo = OLMTutils.get_site_info(inputdata) #add site group if needed
time_offset = -5    #Standard time offset from UTC (e.g. EST is -5)
npd = 24            #number of time steps per day (48 = half hourly)
mylon = siteinfo[site]['lon']  #site longitude
mylat = siteinfo[site]['lat']  #site latitude
print('Site lat, lon: ', mylat, mylon)
measurement_height = 30    #tower height (m)
file_path = '/gpfs/wolf2/cades/cli185/proj-shared/zdr/fluxnet/hourly/' 
calc_flds = False    #use T and RH to comput FLDS (use if data missing or sparse)
leapdays = True     #input data has leap days (to be removed for ELM)
outdir = inputdata+'/atm/datm7/CLM1PT_data/1x1pt_'+site+'/'     #Desired directory for ELM met inputs

metdata={}
#outvars   - met variables used as ELM inputs
#invars    - corresponding variables to be read from input file
#conv_add  - offset for converting units (e.g. C to K)
#conv_mult - multiplier for converting units (e.g. hPa to Pa, PAR to FSDS)
#valid_min - minimum acceptable value for this variable (set as NaN outside range)
#valid_max - maximum acceptable value for this variable (set as NaN outside range)

#Note - FLDS not included here (calculated)
outvars  = ['TBOT',      'VPD',  'WIND',    'PSRF',     'FSDS', 'PRECTmms', 'FLDS']
#invars   = ['TA_ERA','VPD_ERA','WS_ERA',  'PA_ERA', 'SW_IN_ERA',   'P_ERA', 'LW_IN_ERA']   #matching header of input file
invars =   ['TA_F_MDS',   'VPD_F_MDS', 'WS_F',      'PA_F',      'SW_IN_F_MDS',   'P_F',     'LW_IN_F_MDS']
conv_add = [273.15,         0,        0,        0,           0,      0,              0]
conv_mult= [     1,         1,        1,     1000,           1.,     1./1800.,       1]
valid_min= [180.00,         0,        0,      5e4,           0,      0,              0]
valid_max= [350.00,       100.,       80,   1.5e5,        2500,      0.1,          1000]

#ELM Variable names and units
#TBOT:     Air temperature at measurement (tower) height (K)
#RH:       Relative humidity at measurment height (%)
#WIND:     Wind speeed at measurement height (m/s)
#PSRF:     air pressure at surface  (Pa)
#FSDS:     Incoming Shortwave radiation  (W/m2)
#FLDS:     Incoming Longwave radiation   (W/m2)
#PRECTmms: Precipitation       (kg/m2/s)


os.system('mkdir -p '+outdir)
for v in outvars:
  metdata[v] = []

# Find all files for the site, ignoring the trailing numbers
#site_files = glob.glob(os.path.join(file_path, f"*{site}*ERA*H*_*-[0-9]*.csv"))
site_files = glob.glob(os.path.join(file_path, f"*{site}*FULLSET*H*_*-[0-9]*.csv"))


years = []
for fname in site_files:
    # Match patterns like AMF_SITE_FLUXNET_ERA5_HH_YYYY-YYYY_*.csv
    match = re.search(r'(\d{4})-(\d{4})', fname)
    if match:
        years.append(int(match.group(1)))
        years.append(int(match.group(2)))
        filename = fname  # Use the last matched filename for processing

if years:
    start_year_file = min(years)
    start_year_out = start_year_file
    end_year = max(years)
    print(f"Detected start year: {start_year_file}, end year: {end_year}")
else:
    print("No files found for site or unable to extract years.")

#Load the data
#for y in range(start_year_out,end_year+1):
#  print(y)
#  isleapyear = False
#  if (y % 4) == 0:
#     isleapyear = True

lnum= 0
myfile = open(filename,'r')
for s in myfile:
    if (lnum == 0):
      header=s.split(',')
    else:   
      data=s.split(',')
      for v in range(0,len(invars)):
        for h in range(0,len(header)):
          if invars[v] in header[h] and int(data[0][0:4]) >= start_year_out and int(data[0][0:4]) <= end_year:
            if (data[0][4:8] == '0229'):
              #skip leap day
              continue
            else:
              try:
                val = float(data[h])*conv_mult[v]+conv_add[v]
                if (val >= valid_min[v] and val <= valid_max[v]):
                  metdata[outvars[v]].append(val)
                else:
                  metdata[outvars[v]].append(np.NaN)
              except:
                metdata[outvars[v]].append(np.NaN)
    lnum=lnum+1
myfile.close()

#Fill missing values with diurnal mean
for key in metdata:
  print(key, len(metdata[key]))
  gapfill.diurnal_mean(metdata[key],npd=npd)


out_fname = outdir+'/all_hourly.nc'
print(out_fname)
write_elm_met.bypass_format(out_fname, metdata, mylat, mylon, start_year_out, end_year, edge=0.1, \
                time_offset=time_offset, calc_qbot = True, calc_lw = calc_flds, zbot=measurement_height)

