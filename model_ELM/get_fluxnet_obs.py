import numpy as np
import os
from netCDF4 import Dataset

def get_fluxnet_obs(self, site='US-UMB',tstep='monthly',ystart=-1,yend=9999,fluxnet_var='GPP', \
  myobsdir='', valid_months=None, time_average=1):
  
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


