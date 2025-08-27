import numpy as np
import os
from netCDF4 import Dataset

def get_fluxnet_obs(self, site='US-UMB',tstep='monthly',ystart=-1,yend=9999,fluxnet_var='GPP', myobsdir=''):
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
                      if float(myvals[thiscol]) > 0.8 and int(myvals[0][4:8]) != 229:
                        isgood=True  #only advance if quality flag > 80, not leap day%
                    thiscol=thiscol+1
                  if (isgood):
                    myobs[thisob]     = tempob
                    myobs_err[thisob] = tempob_err
                    if fluxnet_var == 'FPSN' or fluxnet_var == 'GPP':
                       myobs_err[thisob] = max(myobs_err[thisob], 1.0)
                  else:
                    myobs[thisob] = -9999
                    myobs_err[thisob] = -9999
                  thisob=thisob+1
            thisrow=thisrow+1
        self.obs[vars_elm[vnum]]=myobs
        self.obs_err[vars_elm[vnum]]=myobs_err
        if (tstep == 'daily'):
           #Shift obs by 1 day (Model timestamp repsresents previous day)
           self.obs[vars_elm[vnum]] = np.roll(self.obs[vars_elm[vnum]], 1)
           self.obs_err[vars_elm[vnum]] = np.roll(self.obs_err[vars_elm[vnum]], 1)
        print(vars_elm[vnum]+' observations read for '+site+' for '+str(ystart)+' to '+str(yend))
        myobs_in.close()
