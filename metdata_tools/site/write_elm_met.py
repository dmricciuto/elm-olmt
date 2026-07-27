from netCDF4 import Dataset
import numpy as np
import os

#coefficients for calculating saturation vapor pressure
a = [6.107799961, 4.436518521e-01, 1.428945805e-02, 2.650648471e-04, \
         3.031240396e-06, 2.034080948e-08, 6.136820929e-11]
b = [6.109177956, 5.034698970e-01, 1.886013408e-02, 4.176223716e-04, \
         5.824720280e-06, 4.838803174e-08, 1.838826904e-10]


#Function for cacluating saturation specific humidity
def calc_q(e_in, pres):
    myq = 0.622 * e_in / (pres - 0.378*e_in)
    return myq

def esat(t):
    if (t[0] > 0):
        myesat = (a[0]+t*(a[1]+t*(a[2]+t*(a[3]+t*(a[4]+t*(a[5]+t*a[6]))))))
    else:
        myesat = (b[0]+t*(b[1]+t*(b[2]+t*(b[3]+t*(b[4]+t*(b[5]+t*b[6]))))))
    return myesat


def bypass_format(filename, met_data, lat, lon, startyear, endyear, edge=0.1, time_offset=0, calc_qbot = False, calc_lw = False, zbot=10):
  metvars = met_data.keys()

  units={}
  units['TBOT'] = 'K'
  units['TSOIL'] = 'K'
  units['RH'] = '%'
  units['WIND'] = 'm/s'
  units['FSDS'] = 'W/m2'
  units['PAR'] = 'umol/m2/s'
  units['FLDS'] = 'W/m2'
  units['VPD'] = 'Pa'
  units['PSRF'] = 'Pa'
  units['PRECTmms'] = 'kg/m2/s'
  units['QBOT'] = 'kg/kg'
  units['ZBOT'] = 'm'
  long_names = {}
  long_names['TBOT'] = 'temperature at the lowest atm level (TBOT)'
  long_names['RH'] = 'relative humidity at the lowest atm level (RH)'
  long_names['WIND'] = 'wind at the lowest atm level (WIND)'
  long_names['FSDS'] = 'incident solar (FSDS)'
  long_names['FLDS'] = 'incident longwave (FLDS)'
  long_names['VPD'] = 'Vapor pressure deficit (VPD)'
  long_names['PSRF'] = 'pressure at the lowest atm level (PSRF)'
  long_names['PRECTmms'] = 'precipitation (PRECTmms)'
  long_names['QBOT'] = 'specific humidity at the lowest atm level (QBOT)'
  long_names['ZBOT'] = 'observational height (ZBOT)'

  nt = len(met_data['TBOT'])
  npd = np.round(nt/(endyear-startyear+1))/365
  if npd == 48:
      # Convert half-hourly to hourly by averaging every two time steps
      nt_orig = len(met_data['TBOT'])
      nt = nt_orig // 2
      npd = 24
      met_data_hourly = {}
      for v in met_data:
          arr = np.array(met_data[v])
          arr = arr[:nt*2]  # Ensure even number of time steps
          arr_hourly = arr.reshape(nt, 2).mean(axis=1)
          met_data_hourly[v] = arr_hourly
      met_data = met_data_hourly

  from netCDF4 import Dataset
  all_hourly = Dataset(filename,'w')
  all_hourly.createDimension('DTIME', nt)
  all_hourly.createDimension('gridcell',1)
  for v in metvars:
    all_hourly.createVariable(v.strip(), 'f', ('gridcell','DTIME',))
    nshift = int(abs(time_offset*int(npd/24)))
    if (time_offset < 0):
      all_hourly[v][0,nshift:] = met_data[v][:-1*nshift]
      all_hourly[v][0,0:nshift] = met_data[v][0]
    elif (time_offset > 0):
      all_hourly[v][0,:-1*nshift] = met_data[v][nshift:]
      all_hourly[v][0,-1*nshift:] = met_data[v][nt-1]
    else:
      all_hourly[v][0,:] = met_data[v][:]
    all_hourly[v].units = units[v]
    all_hourly[v].long_name = long_names[v]
    all_hourly[v].mode = 'time-dependent'

  if (calc_qbot):
    all_hourly.createVariable('QBOT','f',('gridcell','DTIME',))
    if (not 'VPD' in metvars):
      all_hourly.createVariable('VPD','f',('gridcell','DTIME',))
      e_hourly = esat(all_hourly['TBOT'][0,:]-273.15) * all_hourly['RH'][0,:]/100.
      all_hourly['VPD'][0,:] = esat(all_hourly['TBOT'][0,:]-273.15) * (1.0 - all_hourly['RH'][0,:]/100.)*100.
      all_hourly['VPD'].units = units['VPD']
      all_hourly['VPD'].long_name = long_names['VPD']
      all_hourly['VPD'].mode = 'time_dependent'
    else:
      esat_hourly = esat(all_hourly['TBOT'][0,:]-273.15)
      rh_hourly = (esat_hourly - all_hourly['VPD']) / esat_hourly
      e_hourly = esat_hourly - all_hourly['VPD']
      rh_hourly = np.clip((e_hourly / esat_hourly)*100.0, 0.0, 100.0)
      all_hourly.createVariable('RH','f',('gridcell','DTIME',))
      all_hourly['RH'][0,:] = rh_hourly
      all_hourly['RH'].units = units['RH']
      all_hourly['RH'].long_name = long_names['RH']
      all_hourly['RH'].mode = 'time_dependent'
    all_hourly['QBOT'][0,:] = calc_q(e_hourly, all_hourly['PSRF'][0,:]/100.)      
    all_hourly['QBOT'].units = units['QBOT']
    all_hourly['QBOT'].long_name = long_names['QBOT']
    all_hourly['QBOT'].mode = 'time_dependent'

  if (calc_lw):
    all_hourly.createVariable('FLDS','f',('gridcell','DTIME',))
    stebol = 5.67e-8
    mye = esat(all_hourly['TBOT'][0,:]-273.15) * all_hourly['RH'][0,:]/100.
    ea = 0.70 + 5.95e-5*mye*np.exp(1500.0/all_hourly['TBOT'][0,:])
    all_hourly['FLDS'][0,:] = ea * stebol * (all_hourly['TBOT'][0,:]) ** 4

  if (not 'ZBOT' in metvars):
    all_hourly.createVariable('ZBOT','f',('gridcell','DTIME',))
    all_hourly['ZBOT'][:,:] = zbot
    all_hourly['ZBOT'].units = units['ZBOT']
    all_hourly['ZBOT'].long_name = long_names['ZBOT']
    all_hourly['ZBOT'].mode = 'time_dependent'

  all_hourly.close()

  output_data = Dataset(filename,'a')
  output_data.createDimension('scalar', 1)
  output_data.createVariable('DTIME', 'f8', 'DTIME')
  output_data.variables['DTIME'].long_name = 'observation time'
  output_data.variables['DTIME'].units = 'days since '+str(startyear)+'-01-01 00:00:00'
  output_data.variables['DTIME'].calendar = 'noleap'
  output_data.variables['DTIME'][:] = np.arange(nt) / npd + 0.5/npd  # nt matches the DTIME dimension
  output_data.createVariable('LONGXY', 'f8', 'gridcell')
  output_data.variables['LONGXY'].long_name = "longitude"
  output_data.variables['LONGXY'].units = 'degrees E'
  output_data.variables['LONGXY'][:] = lon 
  output_data.createVariable('LATIXY', 'f8', 'gridcell')
  output_data.variables['LATIXY'].long_name = "latitude"
  output_data.variables['LATIXY'].units = 'degrees N'
  output_data.variables['LATIXY'][:] = lat 
  output_data.createVariable('EDGEE', 'f4', 'scalar')
  output_data.variables['EDGEE'].long_name = "eastern edge in atmospheric data"
  output_data.variables['EDGEE'].units = 'degrees E'
  output_data.variables['EDGEE'][:] = lon + edge/2
  output_data.createVariable('EDGEW', 'f4', 'scalar')
  output_data.variables['EDGEW'].long_name = "western edge in atmospheric data"
  output_data.variables['EDGEW'].units = 'degrees E'
  output_data.variables['EDGEW'][:] = lon - edge/2
  output_data.createVariable('EDGEN', 'f4', 'scalar')
  output_data.variables['EDGEN'].long_name = "northern edge in atmospheric data"
  output_data.variables['EDGEN'].units = 'degrees N'
  output_data.variables['EDGEN'][:] = lat + edge/2
  output_data.createVariable('EDGES', 'f4', 'scalar')
  output_data.variables['EDGES'].long_name = "southern edge in atmospheric data"
  output_data.variables['EDGES'].units = 'degrees N'
  output_data.variables['EDGES'][:] = lat - edge/2
  output_data.createVariable('start_year', 'i4', 'scalar')
  output_data.variables['start_year'][:] = startyear
  output_data.createVariable('end_year', 'i4', 'scalar')
  output_data.variables['end_year'][:] = endyear
  output_data.close()

  # --- Plot each variable ---
  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.use('Agg')
  from netCDF4 import Dataset
  import os

  plot_dir = os.path.join(os.path.dirname(filename), "plots")
  os.makedirs(plot_dir, exist_ok=True)
  with Dataset(filename, 'r') as ds:
      for v in ds.variables:
          # Only plot 2D variables with dimensions ('gridcell', 'DTIME')
          if ds.variables[v].dimensions == ('gridcell', 'DTIME'):
              data = ds.variables[v][0, :]
              nt = data.shape[0]
              # Plot time series (already in your code)
              plt.figure()
              step = max(1, nt // 4000)
              plt.plot(data[::step])
              plt.title(v)
              plt.xlabel('Time step')
              plt.ylabel(v)
              plt.savefig(os.path.join(plot_dir, f"{v}.png"))
              plt.close()

              # --- Summertime mean diurnal cycle (June-August) ---
              # Determine npd (time steps per day)
              npd = ds.variables['DTIME'].shape[0] // ((ds.variables['end_year'][0] - ds.variables['start_year'][0] + 1) * 365)
              if npd not in [24, 48]:
                  continue  # Only plot for hourly or half-hourly data

              # Get start year
              startyear = ds.variables['start_year'][0]
              # Build array of day-of-year for each time step
              dtime = ds.variables['DTIME'][:]
              total_days = len(dtime) // npd
              # For each time step, get day-of-year and month
              doy = np.arange(total_days)
              months = []
              for day in doy:
                  # Calculate month (assuming noleap calendar)
                  # Days in months: Jan=31, Feb=28, Mar=31, Apr=30, May=31, Jun=30, Jul=31, Aug=31, Sep=30, Oct=31, Nov=30, Dec=31
                  month_days = [31,28,31,30,31,30,31,31,30,31,30,31]
                  cum_days = np.cumsum([0]+month_days)
                  m = np.searchsorted(cum_days, day, side='right') - 1
                  months.append(m+1)
              months = np.array(months)

              # Find indices for June-August
              summer_idx = np.where((months >= 6) & (months <= 8))[0]
              # For each summer day, get the diurnal cycle
              summer_data = []
              for idx in summer_idx:
                  start = idx * npd
                  end = start + npd
                  if end <= len(data):
                      summer_data.append(data[start:end])
              if len(summer_data) == 0:
                  continue
              summer_data = np.array(summer_data)  # shape: (num_days, npd)
              mean_diurnal = np.nanmean(summer_data, axis=0)

              # Plot mean diurnal cycle
              print('plotting mean diurnal for '+v)
              plt.figure()
              plt.plot(np.arange(npd), mean_diurnal)
              plt.title(f"{v} Mean Diurnal Cycle (Jun-Aug)")
              plt.xlabel('Hour of day' if npd == 24 else 'Half-hour of day')
              plt.ylabel(v)
              plt.savefig(os.path.join(plot_dir, f"{v}_summer_diurnal.png"))
              plt.close()

  #for y in range(startyear,endyear):
  #  for m in range(0,12):
  #    mst = str(101+m)[1:]
  #    monthly = Dataset('1x1pt_'+site+'/'+str(y)+'-'+mst+'.nc','w')
