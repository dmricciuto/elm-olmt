import numpy as np
import os
import csv
from netCDF4 import Dataset

def _clean(value):
  if value is None:
    return ''
  return str(value).strip().strip("'\"")

def _to_float(value):
  value = _clean(value)
  if value == '' or value.lower() in ['nan', 'na', 'n/a', 'nd']:
    return None
  try:
    return float(value)
  except ValueError:
    return None

def _get_first(row, names):
  for name in names:
    if name in row:
      return row[name]
  return None

def _infer_treatment(self):
  treatment = _clean(getattr(self, 'treatment_name', ''))
  if treatment != '':
    return treatment
  suffix = _clean(getattr(self, 'case_suffix', ''))
  if suffix == '':
    return ''
  for part in suffix.split('_'):
    if part.startswith('T') and any(ch.isdigit() for ch in part):
      return part
    if part == 'TAMB':
      return part
  return ''

def _get_table_obs(self, site='US-UMB', tstep='annual', ystart=-1, yend=9999,
  fluxnet_var='GPP', myobsdir='', valid_months=None, time_average=1,
  error_fraction=0.15, error_floor=1.0):

  obs_file = myobsdir
  if os.path.isdir(obs_file):
    obs_file = os.path.join(obs_file, 'observations.csv')
  if not os.path.exists(obs_file):
    raise FileNotFoundError('Observation table not found: '+obs_file)

  tstep = _clean(tstep).lower()
  if tstep not in ['annual', 'monthly']:
    raise ValueError('Tabular observations support annual or monthly frequency, not '+tstep)

  try:
    error_fraction = float(error_fraction)
  except Exception:
    error_fraction = 0.15
  try:
    error_floor = float(error_floor)
  except Exception:
    error_floor = 1.0

  rows = []
  with open(obs_file, newline='') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      row_var = _clean(_get_first(row, ['model_var', 'variable', 'var']))
      if row_var != fluxnet_var:
        continue
      row_site = _clean(row.get('site', ''))
      if row_site != '' and site != '' and row_site != site:
        continue
      treatment = _infer_treatment(self)
      row_treatment = _clean(row.get('treatment', ''))
      if row_treatment != '':
        if treatment == '' or row_treatment != treatment:
          continue
      rows.append(row)

  if len(rows) == 0:
    print('Warning: no tabular observations matched '+fluxnet_var+' for '+site+
          ' treatment '+_infer_treatment(self)+' in '+obs_file)

  years = []
  for row in rows:
    year = _to_float(row.get('year'))
    if year is not None:
      years.append(int(year))
  if ystart <= 0 and years:
    ystart = min(years)
  if yend >= 9000 and years:
    yend = max(years)
  if ystart <= 0 or yend >= 9000:
    raise ValueError('Observation start/end years must be set for '+obs_file)

  nstep = 1 if tstep == 'annual' else 12
  nrows = (int(yend)-int(ystart)+1)*nstep
  myobs = np.full([nrows], -9999.0, float)
  myobs_err = np.full([nrows], -9999.0, float)

  valid_count = 0
  for row in rows:
    year = _to_float(row.get('year'))
    obs = _to_float(_get_first(row, ['obs', 'value']))
    if year is None or obs is None:
      continue
    year = int(year)
    if year < ystart or year > yend:
      continue
    if tstep == 'annual':
      idx = year-int(ystart)
    else:
      month = _to_float(row.get('month'))
      if month is None:
        continue
      month = int(month)
      if month < 1 or month > 12:
        continue
      idx = (year-int(ystart))*12 + (month-1)
    err = _to_float(_get_first(row, ['obs_err', 'error', 'uncertainty']))
    if err is None or err <= 0:
      err = max(abs(obs)*error_fraction, error_floor)
    myobs[idx] = obs
    myobs_err[idx] = err
    valid_count = valid_count+1

  if tstep == 'monthly':
    if valid_months is None:
      valid_months = list(range(1, 13))
    valid_months = [int(m) for m in valid_months if 1 <= int(m) <= 12]
    mymask = np.zeros([nrows], bool)
    nyears = nrows // 12
    for m in valid_months:
      for y in range(0, nyears):
        mymask[y*12+(m-1)] = True
    myobs[~mymask] = -9999
    myobs_err[~mymask] = -9999

  self.obs[fluxnet_var] = myobs
  self.obs_err[fluxnet_var] = myobs_err
  print('Observation table: '+obs_file)
  print('Loaded '+str(valid_count)+' '+fluxnet_var+' observations for '+site+
        ' treatment '+_infer_treatment(self))

def get_fluxnet_obs(self, site='US-UMB',tstep='monthly',ystart=-1,yend=9999,fluxnet_var='GPP', \
  myobsdir='', valid_months=None, time_average=1, obs_format='fluxnet',
  error_fraction=0.15, error_floor=1.0):

  obs_format = _clean(obs_format).lower()
  if obs_format in ['table', 'csv', 'spruce_c_budget']:
    _get_table_obs(self, site=site, tstep=tstep, ystart=ystart, yend=yend,
      fluxnet_var=fluxnet_var, myobsdir=myobsdir, valid_months=valid_months,
      time_average=time_average, error_fraction=error_fraction,
      error_floor=error_floor)
    return
  
  # Ensure valid_months is a list of integers
  if valid_months is None:
      valid_months = list(range(1, 13))  # [1,2,3,4,5,6,7,8,9,10,11,12]
  # Convert and validate
  valid_months = [int(m) for m in valid_months if 1 <= int(m) <= 12]
  if not valid_months:
      raise ValueError("No valid months provided. Months must be integers between 1 and 12.")

  # Validate time_average parameter
  time_average = int(time_average)
  if time_average < 1:
      time_average = 1

  #myvars = ['TBOT','FSDS','WS','RAIN','VPD','NEE','GPP','ER','EFLX_LH_TOT','FSH']
  #myvars   = ['FPSN','FSH','EFLX_LH_TOT']

  myobsfiles = os.listdir(myobsdir+'/'+tstep+'/')

  vars_elm     = ['NEE',                 'FPSN',           'GPP',           'ER',              'EFLX_LH_TOT','FSH',      'TBOT',    'FSDS',      'WS',  'RAIN', 'VPD']
  vars_fluxnet = ['NEE_CUT_REF',         'GPP_NT_CUT_REF', 'GPP_NT_CUT_REF','RECO_NT_CUT_REF','LE_F_MDS',   'H_F_MDS',  'TA_F_MDS','SW_IN_F_MDS','WS_F','P_F', 'VPD_F_MDS']
  vars_unc     = ['NEE_CUT_REF_JOINTUNC','GPP_NT_CUT_SE',  'GPP_NT_CUT_SE', 'RECO_NT_CUT_SE', 'LE_RANDUNC', 'H_RANDUNC','NA',      'NA',        'NA',  'NA', 'NA']
  vars_qc      = ['NEE_CUT_REF_QC',      'NEE_CUT_REF_QC', 'NEE_CUT_REF_QC','NEE_CUT_REF_QC',  'LE_F_MDS_QC', 'H_F_MDS_QC','TA_F_MDS_QC','SW_IN_F_MDS_QC','WS_F_QC','P_F_QC','VPD_F_MDS_QC']

  ndaysm = [31,28,31,30,31,30,31,31,30,31,30,31]
  if (tstep == 'monthly'):
    nstep = 12
  elif (tstep == 'daily'):
    nstep = 365

  if fluxnet_var not in vars_elm:
      raise ValueError('Unsupported FLUXNET observation variable: '+fluxnet_var)
  for v in range(0,len(vars_elm)):
      if fluxnet_var == vars_elm[v]:
          vnum = v

  for f in myobsfiles:
   if site in f and '.csv' in f and 'FULLSET' in f:
    myobsfile = myobsdir+'/'+tstep+'/'+f
    if (os.path.exists(myobsfile)):
        print('Observation file: '+myobsfile)
        thisrow=0
        myobs_input = open(myobsfile)
        if (ystart <= 0 and yend >= 9000):
          print ('Getting start and end year information from observation file')
          for j in myobs_input:
            if thisrow == 1:
                ystart = int(j[0:4])+1
            elif (thisrow > 1):
                yend = int(j[0:4])
            thisrow=thisrow+1
          myobs_input.close
          nrows = thisrow-1

        print(ystart, yend)
        nrows = (yend-ystart+1)*nstep
        myobs = np.zeros([nrows],float)
        myobs_err = np.zeros([nrows],float)
        myobs_in = open(myobsfile)
        thisrow=0
        thisob=0
        for j in myobs_in:
            if (thisrow == 0):
                header = j.split(',')
            else:
                myvals = j.split(',')
                thiscol=0
                if int(myvals[0][0:4]) >= ystart and int(myvals[0][0:4]) <= yend:
                  isgood=False
                  for h in header:
                    if (h.strip() == vars_fluxnet[vnum]):
                      tempob = float(myvals[thiscol])
                    if (h.strip() == vars_unc[vnum]):
                      tempob_err = float(myvals[thiscol])
                    if (h.strip() == vars_qc[vnum]):
                      #if float(myvals[thiscol]) > 0.8 and int(myvals[0][4:8]) != 229:
                      if int(myvals[0][4:8]) != 229:
                        isgood=True  #only advance if quality flag > 80, not leap day%
                    thiscol=thiscol+1
                  if (isgood):
                    myobs[thisob]     = tempob
                    myobs_err[thisob] = tempob_err
                    if fluxnet_var == 'FPSN' or fluxnet_var == 'GPP':
                       myobs_err[thisob] = max(myobs_err[thisob], 1.0)
                    if fluxnet_var == 'EFLX_LH_TOT':
                       myobs_err[thisob] = max(myobs_err[thisob], 10.0)  
                  else:
                    myobs[thisob] = -9999
                    myobs_err[thisob] = -9999
                  if (int(myvals[0][4:8]) != 229):
                    #only increment if not leap day
                    thisob=thisob+1
            thisrow=thisrow+1
        self.obs[vars_elm[vnum]]=myobs
        self.obs_err[vars_elm[vnum]]=myobs_err
        if (tstep == 'monthly'):
            nmonths = len(self.obs[vars_elm[vnum]])
            nyears = nmonths // 12
            mymask = np.zeros([nmonths],bool)
            for m in valid_months:
              for y in range(0,nyears):
                mymask[y*12+(m-1)] = True
            self.obs[vars_elm[vnum]][~mymask] = -9999
            self.obs_err[vars_elm[vnum]][~mymask] = -9999
        if (tstep == 'daily'):
            #Shift obs by 1 day (Model timestamp repsresents previous day)
            self.obs[vars_elm[vnum]] = np.roll(self.obs[vars_elm[vnum]], 1)
            self.obs_err[vars_elm[vnum]] = np.roll(self.obs_err[vars_elm[vnum]], 1)
            
            #Mask days not in valid months
            ndays = len(self.obs[vars_elm[vnum]]) 
            nyears = ndays // 365
            mymask = np.zeros([ndays],bool)
            month_day_start = 0
            month_day_end = ndaysm[0]
            for m in range(1,13):
              if (m in valid_months):
                for y in range(0,nyears):
                   for day in range(month_day_start, month_day_end):
                        mymask[y*365+day] = True
              month_day_start = month_day_start+ndaysm[m-1]
              month_day_end   = month_day_start+ndaysm[min(m,11)]
            self.obs[vars_elm[vnum]][~mymask] = -9999
            self.obs_err[vars_elm[vnum]][~mymask] = -9999
            
            # ADD TIME AVERAGING FOR DAILY DATA
            if time_average > 1:
                print(f"Applying {time_average}-day averaging to daily observations")
                # Get the original daily data
                daily_obs = self.obs[vars_elm[vnum]].copy()
                daily_obs_err = self.obs_err[vars_elm[vnum]].copy()
                
                # Calculate number of averaged periods
                n_periods = len(daily_obs) // time_average
                # Initialize averaged arrays
                averaged_obs = np.full(n_periods, -9999.0)
                averaged_obs_err = np.full(n_periods, -9999.0)
                
                for i in range(n_periods):
                    start_idx = i * time_average
                    end_idx = start_idx + time_average
                    
                    # Get the chunk of data
                    obs_chunk = daily_obs[start_idx:end_idx]
                    err_chunk = daily_obs_err[start_idx:end_idx]
                    
                    # Only average if we have valid data (not all -9999)
                    valid_obs = obs_chunk[obs_chunk > -9999]
                    valid_err = err_chunk[err_chunk > -9999]
                    
                    if len(valid_obs) > 0:
                        # Calculate mean of valid observations
                        averaged_obs[i] = np.mean(valid_obs)
                        
                        # For errors, use root-mean-square if we have multiple valid values
                        if len(valid_err) > 0:
                            if len(valid_err) == 1:
                                averaged_obs_err[i] = valid_err[0]
                            else:
                                # RMS error for averaged data
                                averaged_obs_err[i] = np.sqrt(np.mean(valid_err**2)) / np.sqrt(len(valid_err))
                        else:
                            averaged_obs_err[i] = -9999
                    # If no valid data, leave as -9999 (already initialized)
                
                # Replace the daily data with averaged data
                self.obs[vars_elm[vnum]] = averaged_obs
                self.obs_err[vars_elm[vnum]] = averaged_obs_err
                
                print(f"Averaged from {len(daily_obs)} daily values to {len(averaged_obs)} {time_average}-day values")

