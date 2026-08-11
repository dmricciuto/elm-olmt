#!/usr/bin/env python
import re, os, sys, csv, time, math
import numpy

def get_postproc_basevar(var):
    return var.split('_pft')[0].split('_col')[0]

def is_indexed_postproc_var(var):
    return ('_pft' in var or '_col' in var)

def append_unique(values, additions):
    for item in additions:
        if item not in values:
            values.append(item)

def get_surface_file(self):
    if (hasattr(self, 'case_options')):
        if ('surffile' in self.case_options and self.case_options['surffile'] != ''):
            return self.case_options['surffile']
        if ('fsurdat' in self.case_options and self.case_options['fsurdat'] != ''):
            return self.case_options['fsurdat']
    try:
        fsurdat = self.get_namelist_variable('fsurdat')
        return fsurdat.strip().strip('\'"')
    except Exception:
        pass
    if (hasattr(self, 'rundir') and self.rundir != ''):
        return os.path.join(self.rundir, 'surfdata.nc')
    return ''

def is_multitopounit_surface_file(self):
    surface_file = get_surface_file(self)
    if (surface_file == '' or not os.path.exists(surface_file)):
        return False
    try:
        from netCDF4 import Dataset
        with Dataset(surface_file, 'r') as nc:
            if ('topounit' in nc.dimensions and len(nc.dimensions['topounit']) > 1):
                return True
            if ('topoPerGrid' in nc.variables):
                return numpy.nanmax(nc.variables['topoPerGrid'][:]) > 1
    except Exception as err:
        print('Warning: could not inspect surface file for topounits: '+surface_file)
        print(err)
    return False

def option_enabled(self, *names):
    if (not hasattr(self, 'case_options')):
        return False
    for name in names:
        if (name in self.case_options):
            value = self.case_options[name]
            if (isinstance(value, bool)):
                return value
            if (str(value).strip().lower() in ['.true.', 'true', 't', '1']):
                return True
    return False

def is_peatlands_sitegroup(self):
    sitegroup = str(getattr(self, 'sitegroup', '')).strip().strip("'\"")
    return sitegroup.lower() == 'peatlands'

def get_multitopounit_spinup_vars(indexed=False, include_routing=False, include_humhol=False):
    base_vars = ['ZWT', 'ZWT_PERCH', 'WA', 'FSAT', 'H2OSFC', 'QDRAI', 'QDRAI_PERCH',
                 'QDRAI_XS', 'QOVER', 'QH2OSFC', 'QRUNOFF', 'QRUNOFF_NODYNLNDUSE']
    if (include_routing):
        base_vars.extend(['QFROM_UPHILL', 'QTO_DOWNHILL'])
    if (include_humhol):
        base_vars.extend(['QFLX_SURF_INPUT', 'QFLX_LAT_AQU'])
    if (indexed):
        grid_only_vars = ['QRUNOFF']
        indexed_vars = []
        for var in base_vars:
            indexed_vars.append(var)
            if (var not in grid_only_vars):
                indexed_vars.append(var+'_col')
        return indexed_vars
    return base_vars

def set_histvars(self,spinup=-1,hist_mfilt=-9999,hist_nhtfrq=-9999):
   if (spinup>0):
      var_list_spinup = ['PPOOL', 'EFLX_LH_TOT', 'RETRANSN', 'PCO2', 'PBOT', 'TBOT','FSDS','NDEP_TO_SMINN', 'OCDEP', \
                    'BCDEP', 'COL_FIRE_CLOSS', 'HDM', 'LNFM', 'NEE', 'GPP', 'FPSN', 'AR', 'HR', \
                    'MR', 'GR', 'ER', 'NPP', 'TLAI', 'SOIL3C', 'TOTSOMC', 'TOTSOMC_1m', 'LEAFC', \
                    'DEADSTEMC', 'DEADCROOTC', 'FROOTC', 'LIVESTEMC', 'LIVECROOTC', 'TOTVEGC', 'N_ALLOMETRY','P_ALLOMETRY',\
                    'TOTCOLC', 'TOTLITC', 'BTRAN', 'SCALARAVG_vr', 'CWDC', 'QVEGE', 'QVEGT', 'QSOIL', 'QDRAI', \
                    'QRUNOFF', 'FPI', 'FPI_vr', 'FPG', 'FPI_P','FPI_P_vr', 'FPG_P', 'CPOOL','NPOOL', 'PPOOL', 'SMINN', 'HR_vr','ZWT']
      if ('ICBELMBC' in self.compset):
        var_list_spinup = ['FPSN','TLAI','QVEGT','QVEGE','QSOIL','EFLX_LH_TOT','FSH','RH2M','TSA','FSDS','FLDS','PBOT', \
                         'WIND','BTRAN','DAYL','T10','QBOT','TBOT']

      has_postproc_vars = bool(self.postproc_vars)
      use_humhol_routing = option_enabled(self, 'use_humhol', 'humhol')
      use_topounit_routing = use_humhol_routing or option_enabled(self, 'use_IM2_hillslope_hydrology')
      if (use_humhol_routing or is_multitopounit_surface_file(self)):
        append_unique(var_list_spinup, get_multitopounit_spinup_vars( \
          include_routing=use_topounit_routing, include_humhol=use_humhol_routing))
        if (has_postproc_vars or 'ELMBC' in self.compset):
          append_unique(self.postproc_vars, get_multitopounit_spinup_vars(indexed=True, \
            include_routing=use_topounit_routing, include_humhol=use_humhol_routing))


     #if (self.C14):
      #    var_list_spinup.append('C14_TOTSOMC')
      #    var_list_spinup.append('C14_TOTSOMC_1m')
      #    var_list_spinup.append('C14_TOTVEGC')
      if (not 'ECA' in self.compset and not 'ELMBC' in self.compset):
        var_list_spinup.extend(['SOIL4C'])
      vst = ''
      for v in var_list_spinup:
        vst=vst+"'"+v+"',"
      if (not 'ELMBC' in self.compset and not 'FATES' in self.compset):
        self.customize_namelist(variable='hist_empty_htapes',value='.true.')
        self.customize_namelist(variable='hist_fincl1',value=vst[:-1])
        self.customize_namelist(variable='hist_fincl2',value=vst[:-1])
        spinup_nhtfrq = str(self.nyears_spinup*-8760)
        if (not self.postproc_vars):
          self.customize_namelist(variable='hist_dov2xy',value='.true.,.false.')
          self.customize_namelist(variable='hist_mfilt',value='1,1')
          self.customize_namelist(variable='hist_nhtfrq',value=spinup_nhtfrq+','+spinup_nhtfrq)
        else:
          vst_pp=''
          vst_pp_pft=''
          for v in self.postproc_vars:
              if (is_indexed_postproc_var(v)):
                  # PFT- or column-specific outputs (put after the spinup h1 file)
                  pft_var = get_postproc_basevar(v)
                  if ("'"+pft_var+"'," not in vst_pp_pft):
                      vst_pp_pft=vst_pp_pft+"'"+pft_var+"',"
              else:
                  vst_pp=vst_pp+"'"+v+"',"
                  if (is_peatlands_sitegroup(self) and "'"+v+"'," not in vst_pp_pft):
                      vst_pp_pft=vst_pp_pft+"'"+v+"',"
          if (self.postproc_freq == 'hourly'):
            pp_mfilt = '8760'
            pp_nhtfrq = '-1'
          else:
            pp_mfilt = '365'
            pp_nhtfrq = '-24'
          if (vst_pp != '' and vst_pp_pft != ''):
            self.customize_namelist(variable='hist_mfilt',value='1,1,'+pp_mfilt+','+pp_mfilt)
            self.customize_namelist(variable='hist_nhtfrq',value=spinup_nhtfrq+','+spinup_nhtfrq+','+pp_nhtfrq+','+pp_nhtfrq)
            self.customize_namelist(variable='hist_dov2xy',value='.true.,.false.,.true.,.false.')
            self.customize_namelist(variable='hist_fincl3',value=vst_pp[:-1])
            self.customize_namelist(variable='hist_fincl4',value=vst_pp_pft[:-1])
          elif (vst_pp_pft != ''):
            self.customize_namelist(variable='hist_mfilt',value='1,1,'+pp_mfilt)
            self.customize_namelist(variable='hist_nhtfrq',value=spinup_nhtfrq+','+spinup_nhtfrq+','+pp_nhtfrq)
            self.customize_namelist(variable='hist_dov2xy',value='.true.,.false.,.false.')
            self.customize_namelist(variable='hist_fincl3',value=vst_pp_pft[:-1])
          else:
            self.customize_namelist(variable='hist_mfilt',value='1,1,'+pp_mfilt)
            self.customize_namelist(variable='hist_nhtfrq',value=spinup_nhtfrq+','+spinup_nhtfrq+','+pp_nhtfrq)
            self.customize_namelist(variable='hist_dov2xy',value='.true.,.false.,.true.')
            self.customize_namelist(variable='hist_fincl3',value=vst_pp[:-1])
      else:
        if (not self.postproc_vars):
          #Annual output if no postproc vars
          self.customize_namelist(variable='hist_mfilt',value='1')
          self.customize_namelist(variable='hist_nhtfrq',value='-8760')
        else:
          vst_pp=''
          vst_pp_pft=''
          for v in self.postproc_vars:
              if (is_indexed_postproc_var(v)):
                  # PFT- or column-specific outputs (put in h2 file)
                  pft_var = get_postproc_basevar(v)
                  if ("'"+pft_var+"'," not in vst_pp_pft):
                      vst_pp_pft=vst_pp_pft+"'"+pft_var+"',"
              else:
                  vst_pp=vst_pp+"'"+v+"',"
                  if (is_peatlands_sitegroup(self) and "'"+v+"'," not in vst_pp_pft):
                      vst_pp_pft=vst_pp_pft+"'"+v+"',"
          #Write daily for requested postprocessed variables
          if (vst_pp_pft != ''):
              if (self.postproc_freq == 'hourly'):
                self.customize_namelist(variable='hist_mfilt',value='1,8760,8760')
                self.customize_namelist(variable='hist_nhtfrq',value='-8760,-1,-1')
              else:
                self.customize_namelist(variable='hist_mfilt',value='1,365,365')
                self.customize_namelist(variable='hist_nhtfrq',value='-8760,-24,-24')
              self.customize_namelist(variable='hist_dov2xy',value='.true.,.true.,.false.')
              self.customize_namelist(variable='hist_fincl3',value=vst_pp_pft[:-1])
          else:
              if (self.postproc_freq == 'hourly'):
                self.customize_namelist(variable='hist_mfilt',value='1,8760')
                self.customize_namelist(variable='hist_nhtfrq',value='-8760,-1')
              else:
                self.customize_namelist(variable='hist_mfilt',value='1,365')
                self.customize_namelist(variable='hist_nhtfrq',value='-8760,-24')
          self.customize_namelist(variable='hist_fincl2',value=vst_pp[:-1])
   else:
      #Transient simulation
      if (self.postproc_vars == []):
        #Default to daily output for all variables if not postproc vars
        #if ('US-SPR' in self.site):
        #    #For default SPRUCE run, set history variables
        #else:
        self.customize_namelist(variable='hist_mfilt',value='1')
        self.customize_namelist(variable='hist_nhtfrq',value='-8760')
      else:
        #Write annual for all vars, requested postproc vars daily
        vst_pp=''
        vst_pp_pft=''
        for v in self.postproc_vars:
            if (is_indexed_postproc_var(v)):
                # PFT- or column-specific outputs (put in h2 file)
                pft_var = get_postproc_basevar(v)
                if ("'"+pft_var+"'," not in vst_pp_pft):
                    vst_pp_pft=vst_pp_pft+"'"+pft_var+"',"
            else:
                vst_pp=vst_pp+"'"+v+"',"
                if (is_peatlands_sitegroup(self) and "'"+v+"'," not in vst_pp_pft):
                    vst_pp_pft=vst_pp_pft+"'"+v+"',"
        if (vst_pp_pft != ''):
            self.customize_namelist(variable='hist_mfilt',value='1,365,365')
            self.customize_namelist(variable='hist_nhtfrq',value='-8760,-24,-24')
            self.customize_namelist(variable='hist_dov2xy',value='.true.,.true.,.false.')
            self.customize_namelist(variable='hist_fincl3',value=vst_pp_pft[:-1])
        else:
            self.customize_namelist(variable='hist_mfilt',value='1,365')
            self.customize_namelist(variable='hist_nhtfrq',value='-8760,-24')
        self.customize_namelist(variable='hist_fincl2',value=vst_pp[:-1])
