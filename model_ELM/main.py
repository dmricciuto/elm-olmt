import socket, os, sys, csv, time, math, numpy
import glob, re, subprocess, shlex
import pickle
import json
from .makepointdata import makepointdata
from .set_histvars import *
from .ensemble import *
from .get_fluxnet_obs import *
from .surrogate_NN import *
from .run_GSA import *
from .MCMC import *
from .netcdf4_functions import *
from datetime import datetime
import xarray as xr
import code  # For development: code.interact(local=dict(globals(), **locals()))

def smart_loadtxt(filename):
    # Try comma first, then whitespace
    with open(filename) as f:
        first_line = f.readline()
    if ',' in first_line:
        return np.loadtxt(filename, delimiter=',')
    else:
        return np.loadtxt(filename)

def parse_submit_jobnum(output):
    submit_output = output.strip()
    matches = re.findall(r'\bSubmitted\s+batch\s+job\s+(\d+)\b', submit_output)
    if matches:
        return int(matches[-1])

    matches = re.findall(r'\bSubmitted\s+job\s+(\d+)\b', submit_output)
    if matches:
        return int(matches[-1])

    matches = re.findall(r'(?<![\w.])(\d+)(?![\w.])', submit_output)
    if matches:
        return int(matches[-1])

    raise ValueError('Could not parse submitted job id from output:\n'+submit_output)

def write_fates_pft_subset_nc(source_path, output_path, fates_pft, duplicates=1):
    """Write a FATES NetCDF parameter file subset with xarray."""
    with xr.open_dataset(source_path, decode_timedelta=False) as ds:
        if 'fates_pft' not in ds.sizes:
            raise KeyError('Dimension fates_pft not found in '+source_path)
        selected = ds.isel(fates_pft=[int(fates_pft)]).load()
    if int(duplicates) > 1:
        selected = xr.concat([selected.copy(deep=True) for _ in range(int(duplicates))],
                dim='fates_pft')
    tmp_output = output_path+'.tmp'
    selected.to_netcdf(tmp_output, mode='w')
    selected.close()
    os.replace(tmp_output, output_path)

def elm_spatial_cell_count(path, file_kind):
    with xr.open_dataset(path, decode_times=False) as ds:
        sizes = dict(ds.sizes)
    if file_kind == 'domain':
        if 'ni' in sizes and 'nj' in sizes:
            return sizes['ni'] * sizes['nj'], sizes
        if 'gridcell' in sizes:
            return sizes['gridcell'], sizes
    elif file_kind == 'surfdata':
        if 'gridcell' in sizes:
            return sizes['gridcell'], sizes
        if 'lsmlat' in sizes and 'lsmlon' in sizes:
            return sizes['lsmlat'] * sizes['lsmlon'], sizes
        if 'lat' in sizes and 'lon' in sizes:
            return sizes['lat'] * sizes['lon'], sizes
    raise ValueError('Could not determine '+file_kind+' spatial cell count for '+path+
            ' from dimensions '+str(sizes))

class ELMcase():
  def __init__(self,caseid='',compset='ICBELMBC',suffix='',site='',sitegroup='AmeriFlux', \
            res='',tstep=1,np=1,nyears=1,startyear=-1, machine='', queue='', partition='', project = '',\
            exeroot='', modelroot='', runroot='',caseroot='',inputdata='', \
            region_name='', lat_bounds=[-90,90],lon_bounds=[-180,180], \
            point_list=[], namelist_options=[],casename='',mpilib='', olmtdir='', walltime=24, 
            apptainer='', apptainer_bind = '/', offline_driver=False, resubmit_years=0, debug=False):

      if (casename != ''):
        #get case information from pre-existing pkl file:
        print('Loading '+casename)
        #Load case object
        case_file=open('pklfiles/'+casename+'.pkl','rb')
        myinstance=pickle.load(case_file)
        for k in myinstance.__dict__.keys():
          setattr(self, k, getattr(myinstance, k))
        if not hasattr(self, 'mask_grid'):
          self.mask_grid = []
      else:
        self.model_name='elm'
        self.modelroot=modelroot
        self.inputdata_path = inputdata
        self.runroot=runroot
        self.caseroot=caseroot
        self.exeroot=exeroot
        self.debug=debug
        if olmtdir == '':
            self.OLMTdir = os.getcwd()+'/..'
        else:
            self.OLMTdir = olmtdir
        #Set default resolution (site or regional)
        self.site=site
        if (res == ''):
          if (site == ''):
              self.res='r05_r05'
              #Defined rectangular region
          else:
              self.res='ELM_USRDAT'
        else:
          self.res=res
        self.point_list = point_list
        self.region = 'region'
        if (region_name != ''):
          self.region = region_name
        self.lat_bounds=numpy.array(lat_bounds)
        self.lon_bounds=numpy.array(lon_bounds)
        self.sitegroup=sitegroup
        #Set the default case id prefix to the current date
        if (caseid == ''):
          current_date = datetime.now()
          # Format the date as YYYYMMDD
          self.caseid = current_date.strftime('%Y%m%d')
        else:
          self.caseid = caseid
        self.queue=queue
        self.partition=partition
        self.project=project
        self.interactive_build=False
        self.get_machine(machine=machine)
        self.mask_grid=[]
        # Apptainer container image (optional)
        self.apptainer = apptainer
        self.apptainer_bind = apptainer_bind
        self.compiler=''
        self.pio_version=2
        self.compset=compset
        self.case_suffix=suffix   #used for ad_spinup and trans
        #Custom surface/domain info
        self.surffile = ''
        self.use1kmsurf = False
        self.pftdynfile = ''
        self.paramfile = ''
        self.fates_paramfile = ''
        self.nopftdyn=False
        self.domainfile = ''
        self.run_n = nyears
        self.force_full_spinup_cycles = True
        if (startyear == -1):
          if '1850' in self.compset:
              self.startyear=1
          elif '20TR' in self.compset or 'trans' in suffix:
              self.startyear=1850
          else:
              self.startyear=2000
        else:
           self.startyear=startyear
        #Number of processors
        self.np = np
        #Timestep in hours
        self.tstep = tstep
        self.has_finidat = False
        self.cppdefs=''
        self.humhol=False
        self.srcmods=''
        self.output={}
        self.obs={}
        self.obs_err={}
        self.surrogate={}
        self.postproc_vars=[]
        self.postproc_startyear=-1
        self.postproc_endyear=9999
        self.postproc_pfts=[0]
        self.postproc_cols=[0]
        self.postproc_topounit=-1
        self.postproc_freq='monthly'
        self.postproc_timeaverage=1
        self.sens_plot_ntimesteps=None  # None means plot all timesteps
        self.namelist_options=namelist_options
        self.mpilib=''
        self.walltime=walltime
        try:
          self.resubmit_years=int(resubmit_years)
        except Exception:
          self.resubmit_years=0
        self.ensemble_resubmit_years=0
        # Build offline driver (elm_offline_driver) alongside e3sm.exe
        self.offline_driver = offline_driver

  def setup_ensemble(self, sampletype='monte_carlo',parm_list='', ensemble_file='', \
          np_ensemble=64, nsamples=100, obs={}, obs_err={}, finidat_root='', \
          resubmit_years=0):
    read_parm_list(self, parm_list=parm_list)
    if (ensemble_file == ''):
      create_samples(self, sampletype=sampletype, parm_list=parm_list,nsamples=nsamples)
    else:
      self.ensemble_file = ensemble_file
      self.samples = np.transpose(smart_loadtxt(ensemble_file))
      self.nsamples = np.shape(self.samples)[1]
    self.finidat_root = finidat_root
    if (finidat_root != ''):
      self.has_finidat = True
    self.np_ensemble=np_ensemble
    try:
      self.ensemble_resubmit_years = int(resubmit_years)
    except Exception:
      self.ensemble_resubmit_years = 0
    create_ensemble_script(self)
    self.get_default_parms()
    #Variables for surrogate model
    self.pscaler={}
    self.yscaler={}
    self.obs=obs
    self.obs_err=obs_err

  def get_machine(self,machine=''):
    if (machine == ''):
      hostname = socket.gethostname()
      if ('baseline' in hostname):
        self.machine = 'cades-baseline'
    else:
      self.machine=machine
    self.noslurm=False
    self.interactive_build=False
    if ('linux' in self.machine or 'ubuntu' in self.machine or 'docker' in self.machine):
        self.noslurm=True
    if(self.machine == 'pm-cpu'):
        if self.queue == '':
            self.queue='regular'
    elif(self.machine == 'pathfinder'):
        if self.queue == '':
            self.queue='normal'
        # Build on compute node to avoid OOM on login node
        self.interactive_build=True
    else:
        if self.queue == '':
            self.queue='batch'

  def cime_job_queue(self):
    # Pathfinder's CIME queue is the machine-XML queue name (`parallel`).
    # OLMT's `queue` config is used as the Slurm QoS there.
    if (self.machine == 'pathfinder'):
      return 'parallel'
    return self.queue

  def slurm_partition(self):
    if (self.partition != ''):
      return self.partition
    if (self.machine == 'pathfinder'):
      return 'parallel'
    return ''

  def slurm_qos(self):
    if (self.machine == 'pathfinder' and self.slurm_partition() == 'hpcl-cli185'):
      return 'hpcl-cli185'
    return self.queue

  def replace_slurm_submit_option(self, flags, option_names, output_name, value):
    filtered = []
    skip_next = False
    for flag in flags:
      if skip_next:
        skip_next = False
        continue
      if flag in option_names:
        skip_next = True
        continue
      if any(flag.startswith(option+'=') for option in option_names):
        continue
      filtered.append(flag)
    if (value != ''):
      filtered += [output_name, value]
    return filtered

  def cime_walltime_string(self):
    if (isinstance(self.walltime, str) and ':' in self.walltime):
      fields = self.walltime.split(':')
      if (len(fields) == 2):
        return fields[0]+':'+fields[1]+':00'
      return self.walltime
    walltime_hours = float(self.walltime)
    hours = int(walltime_hours)
    minutes = int(round((walltime_hours-hours)*60))
    if (minutes == 60):
      hours += 1
      minutes = 0
    return str(hours)+':'+str(minutes).zfill(2)+':00'

  def pathfinder_batch_command_flags(self, base_flags=''):
    flags = shlex.split(base_flags) if base_flags != '' else []
    walltime = self.cime_walltime_string()
    partition = self.slurm_partition()
    qos = self.slurm_qos()
    if (self.project == ''):
      flags = self.replace_slurm_submit_option(flags, ['-A', '--account'], '--account', '')
    flags = self.replace_slurm_submit_option(flags, ['-t', '--time'], '--time', walltime)
    flags = self.replace_slurm_submit_option(flags, ['-p', '--partition'], '-p', partition)
    flags = self.replace_slurm_submit_option(flags, ['--qos'], '--qos', qos)
    return ' '.join([shlex.quote(str(f)) for f in flags])

  def slurm_submit_args(self, include_time=False, ntasks=None):
    args = []
    partition = self.slurm_partition()
    if (partition != ''):
      args += ['-p', partition]
    if (ntasks is not None):
      args += ['-n', str(ntasks)]
    qos = self.slurm_qos() if self.machine == 'pathfinder' else ''
    if (qos != ''):
      args += ['--qos='+qos]
    if (self.project != ''):
      args += ['--account='+self.project]
    if (include_time):
      args += ['--time='+self.cime_walltime_string()]
    return args

  def cime_batch_command_flags(self, subgroup='case.run'):
    result = subprocess.run(['./xmlquery', '--subgroup', subgroup, '--no-resolve', '--value',
                             'BATCH_COMMAND_FLAGS'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    if (result.returncode > 0):
      return ''
    return result.stdout.strip()

  def cime_case_setup_flags(self):
    flags = []
    if (self.machine == 'pathfinder'):
      flags.append('--disable-git')
    return (' '+' '.join(flags)) if len(flags) > 0 else ''

  def get_model_directories(self):
    if (not os.path.exists(self.modelroot)):
      print('Error:  Model root '+self.modelroot+' does not exist.')
      sys.exit(1)
    #if (not os.path.exists(self.inputdata_path)):
    #  print('Error:  Input data directory '+self.inputdata_path+' does not exist.')
    #  sys.exit(1)
    if (not os.path.exists(self.runroot)):
      print('Error: Run root '+self.runroot+' does not exist.')
      sys.exit(1)
    if (not os.path.exists(self.caseroot)):
      print('Error: Run root '+self.caseroot+' does not exist.')
      sys.exit(1)  
    #if (not os.path.exists(self.exeroot)):
    #  print('No exeroot specified.  Setting to: '+self.runroot+'
    print('Model root directory: '+self.modelroot)
    if (self.inputdata_path != ''):
      print('Input data directory: '+self.inputdata_path)
    print('Run root directory:   '+self.runroot)
    print('Case root directory:  '+self.caseroot)

  def get_forcing(self,metdir='',mettype=''):
    #Get the forcing type and directory
    if (metdir == ''):
        if (self.site != '' and (mettype == '' or mettype == 'site')):
          #Assume the user wants to use site data and set default path
          self.forcing='site'
          self.metdir = self.inputdata_path+'/atm/datm7/CLM1PT_data/1x1pt_'+self.site
        elif (mettype == 'gswp3-daymet4'):
          print('Setting met type to gswp3-daymet4')
          self.forcing='gswp3-daymet4'
          self.metdir='/gpfs/wolf2/cades/cli185/proj-shared/zdr/Daymet_GSWP3_4KM_TESSFA'
        elif ('era5-daymet' in mettype):
          print('Setting met type to era5-daymet4')
          self.forcing = 'era5-daymet4'
          self.metdir = '/gpfs/wolf2/cades/cli185/proj-shared/zdr/Daymet_ERA5_TESSFA2'
        elif (mettype != ''):
          #Met type specified but not metdir.  Get location from metinfo.txt
          self.forcing=mettype
          metinfo = open(self.OLMTdir+'/metinfo.txt','r')
          for s in metinfo:
              if s.split(':')[0] == mettype:
                  self.metdir = self.inputdata_path+'/'+s.split(':')[1].strip()
          metinfo.close()
        else:
          #No site, mettype or metdir specified.  Default to GSWP3
          print('No site, mettype or metdir specified.  Defaulting to GSWP3')
          self.forcing='gswp3'
          self.metdir = self.inputdata_path+'/atm/datm7/atm_forcing.datm7.GSWP3.0.5d.v2.c180716'
        if (self.is_bypass() and not 'site' in self.forcing):
            self.metdir = self.metdir+'/cpl_bypass_full'
    else:
        #Met data directory provided.  Get met type.
        if (self.site != '' and mettype == ''):
            self.forcing='site'
        else:
            self.forcing=mettype
            if mettype == '':
              print('Error: When specifying metdir, Must also specify met type (e.g. gswp3)')
              sys.exit(1)
        self.metdir = metdir
        if self.forcing == 'site':
            self.metdir = metdir+'/1x1pt_'+self.site
    self.get_metdata_year_range()

  def is_bypass(self):
    #Determine whether this is a coupler bypass case from compset name
    if ('CBCN' in self.compset or 'ICB' in self.compset or 'CLM45CB' in self.compset):
      return True
    else:
      return False

  def get_namelist_variable(self,vname):
    #Get the default namelist variable from case directory
    #Must be done AFTER the case.setup
    nfile = open(self.casedir+'/Buildconf/elmconf/lnd_in')
    for line in nfile:
        if (vname in line):
            value = line.split('=')[1]
    return value[:-1]   #avoid new line character


  def modify_jsoninput_file(self, param_file_path, parameters, file_description="parameter"):

    try:
        # Use .strip() and os.path.expanduser to handle whitespace and '~' shortcuts
        clean_path = param_file_path.strip()
    
        with open(clean_path, 'r') as file:
            data = json.load(file)
        
    except FileNotFoundError:
        print(f"Error: The file at {clean_path} was not found.")
        data = {} # Set to empty dict to prevent further crashes
        
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON. Check for syntax errors: {e}")
        print(f"File path: {clean_path}")
        data = {}
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"while attempting to open the the json parameter file: {clean_path}")
        data = {}
        
    for key, value in parameters.items():
        if not(key in data['parameters']):
            print(f"Attempting to modify a parameter that is not in the input file")
            print(f"Requested parameter name: {key}")
            print(f"File-path: {param_file_path}")
        else:

            # Check dimensions of the variable
            var = data['parameters'][key]
            var_data = np.array(var['data'])
            var_shape = var_data.shape

            
            if (not isinstance(value, list)):
                # If this is a scalar input value, simple rules apply
                data['parameters'][key]['data'] = value

            else:
                # If the input value is a list, its a little more complicated

                # Check if first index is -1 (set all indices)
                if len(value) >= 2 and int(value[0]) == -1:
                    if len(value) > 2:
                        print(f"Error: {key} has index -1 but more than 2 values in list. Use [-1, value] only.")
                        continue
                    # Set all indices to the same value
                    all_value = value[1]
                    print(f"  Setting all indices = {all_value}")
                    data['parameters'][key]['data'] = value
                    
                else:
                    # Check if variable is 2D
                    if len(var_shape) == 2:
                        # 2D parameter: expect pairs of [index1, index2, value]
                        if len(value) % 3 != 0:
                            print(f"Error: {key} is 2D but values not in groups of 3 [index1, index2, value]")
                            continue
                        
                        for i in range(0, len(value), 3):
                            if i + 2 >= len(value):
                                break
                            idx1 = int(value[i])
                            idx2 = int(value[i+1]) 
                            param_value = value[i+2]
                            if idx1 < 0 or idx2 < 0:
                                print(f"Error: {key} has negative indices [{idx1}, {idx2}]. Use -1 only to set all.")
                                continue
                            if idx1 >= var_shape[0] or idx2 >= var_shape[1]:
                                print(f"Error: {key} indices [{idx1}, {idx2}] exceed dimensions {var_shape}")
                                continue
                            print(f"  Setting [{idx1}, {idx2}] = {param_value}")
                            data['parameters'][key]['data'][idx1][idx2] = param_value
                            
                    elif len(var_shape) == 1 or file_description == "surface data":
                        # 1D parameter: expect pairs of [index, value]
                        for i in range(0, len(value)-1, 2):
                            idx = int(value[i])
                            param_value = value[i+1]
                            if idx < 0:
                                print(f"Error: {key} has negative index {idx}. Use -1 only to set all.")
                                continue
                            if idx >= var_shape[0]:
                                print(f"Error: {key} index {idx} exceeds dimension {var_shape[0]}")
                                continue
                            data['parameters'][key]['data'][idx] = param_value
                            print(f"  Setting index {idx} = {param_value}")
                    else:
                        print(f"Error: {key} has {len(var_shape)}D shape - only 1D and 2D parameters supported")
                        continue
                    
    # Add metadata about original filename and modifications
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
    # Update global attributes
    # Everything below here assumes that you are going to actually make modification
    # and not just query things. First step then is to modify the changelog of
    # the file
    # ------------------------------------------------------------------------------

    change_log = f'modified by OLMT on: {timestamp}. Parameters changed: '+ ', '.join(parameters.keys())
    
    data['attributes']['history']
    old_hist = data['attributes']['history']
    new_hist = old_hist+'  '+change_log+'.'
    data['attributes']['history'] = new_hist

    try:
        with open(clean_path, 'w') as outfile:
            json.dump(data, outfile, indent=2)
        
    except FileNotFoundError:
        print(f"Error: The file at {clean_path} was not found for writing.")
        data = {} # Set to empty dict to prevent further crashes
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"while attempting to open the the json parameter file: {clean_path}")
        data = {}

        

  def modify_ncinput_file(self, param_file_path, parameters, file_description="parameter"):
    """
    Modify parameters in a NetCDF parameter file.
    """
    import netCDF4
    from datetime import datetime
    with netCDF4.Dataset(param_file_path, 'a') as nc:
        for key, value in parameters.items():
            if key in nc.variables:
                # Parameter exists - modify it
                print(f"Modifying existing {file_description}: {key}")
                
                # Check dimensions of the variable
                var = nc.variables[key]
                var_shape = var.shape

                if isinstance(value, list):
                    # Handle list: indices and values
                    if len(value) % 2 != 0:
                        print(f"Warning: {key} list has odd length, ignoring last element")
                    # Check if first index is -1 (set all indices)
                    if len(value) >= 2 and int(value[0]) == -1:
                        if len(value) > 2:
                            print(f"Error: {key} has index -1 but more than 2 values in list. Use [-1, value] only.")
                            continue
                        # Set all indices to the same value
                        all_value = value[1]
                        print(f"  Setting all indices = {all_value}")
                        nc.variables[key][...] = all_value
                    else:
                        # Check if variable is 2D
                        if len(var_shape) == 2:
                            # 2D parameter: expect pairs of [index1, index2, value]
                            if len(value) % 3 != 0:
                                print(f"Error: {key} is 2D but values not in groups of 3 [index1, index2, value]")
                                continue
                            for i in range(0, len(value), 3):
                                if i + 2 >= len(value):
                                    break
                                idx1 = int(value[i])
                                idx2 = int(value[i+1]) 
                                param_value = value[i+2]
                                if idx1 < 0 or idx2 < 0:
                                    print(f"Error: {key} has negative indices [{idx1}, {idx2}]. Use -1 only to set all.")
                                    continue
                                if idx1 >= var_shape[0] or idx2 >= var_shape[1]:
                                    print(f"Error: {key} indices [{idx1}, {idx2}] exceed dimensions {var_shape}")
                                    continue
                                print(f"  Setting [{idx1}, {idx2}] = {param_value}")
                                nc.variables[key][idx1, idx2] = param_value
                        elif len(var_shape) == 1 or file_description == "surface data":
                            # 1D parameter: expect pairs of [index, value]
                            for i in range(0, len(value)-1, 2):
                                idx = int(value[i])
                                param_value = value[i+1]
                                if key == 'MONTHLY_LAI':
                                    # Special handling for MONTHLY_LAI (set to same LAI for all months for a PFT)
                                    nc.variables[key][:,idx,...] = param_value
                                elif (key == 'PCT_NAT_PFT'):
                                    # Special handling for PCT_NAT_PFT (set PFTs manually)
                                    nc.variables[key][idx,...] = param_value  
                                else:
                                  if idx < 0:
                                    print(f"Error: {key} has negative index {idx}. Use -1 only to set all.")
                                    continue
                                  if idx >= var_shape[0]:
                                    print(f"Error: {key} index {idx} exceeds dimension {var_shape[0]}")
                                    continue
                                  nc.variables[key][idx] = param_value
                                print(f"  Setting index {idx} = {param_value}")
                        else:
                            print(f"Error: {key} has {len(var_shape)}D shape - only 1D and 2D parameters supported")
                            continue
                else:
                    # Scalar value - set all elements
                    nc.variables[key][...] = value
            else:
                # Parameter doesn't exist - create it
                print(f"Creating new {file_description}: {key}")
                if isinstance(value, list):
                    print(f"Warning: Cannot create new parameter {key} with index-specific values. Use scalar for new parameters.")
                    continue
                dtype = 'f4' if isinstance(value, float) else 'i4'  
                nc.createVariable(key, dtype, ())
                nc.variables[key][...] = value
        
        # Add metadata about original filename and modifications
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Update global attributes
        nc.setncattr(f'original_{file_description.replace(" ", "_")}_file', 
                    getattr(self, 'paramfile' if 'parameter' in file_description else 'fates_paramfile', 'unknown'))
        nc.setncattr('modified_by_OLMT', timestamp)
        nc.setncattr('case_name', getattr(self, 'casename', 'unknown'))
        
        modified_params = ', '.join(parameters.keys())
        nc.setncattr(f'modified_{file_description.replace(" ", "_")}s', modified_params)
        
        # Update or add history attribute
        original_file = getattr(self, 'paramfile' if 'parameter' in file_description else 'fates_paramfile', 'unknown')
        history_entry = f"{timestamp}: Modified {file_description}s by OLMT from {original_file}"
        if hasattr(nc, 'history'):
            nc.setncattr('history', nc.getncattr('history') + '\n' + history_entry)
        else:
            nc.setncattr('history', history_entry)

  def set_param_file(self):
    #set the ELM parameter file
    if (self.paramfile == ''):
      #Get parameter filename from case directory
      self.paramfile = self.get_namelist_variable('paramfile')
    print('Parameter file: '+self.paramfile)
    #Copy the parameter file to the temp directory
    os.system('cp '+self.paramfile+' '+self.OLMTdir+'/temp/clm_params.nc')

    if hasattr(self, 'add_parameter') and self.add_parameter:
        param_path = self.OLMTdir + '/temp/clm_params.nc'
        self.modify_ncinput_file(param_path, self.add_parameter, "parameter")

  def set_CNP_param_file(self,filename=''):
    if (filename == ''):
        self.CNPparm_file = self.get_namelist_variable('fsoilordercon')
    else:
        self.CNPparm_file = filename
    os.system('cp '+self.CNPparm_file+' '+self.OLMTdir+'/temp/CNP_parameters.nc')

  def set_fates_param_file(self):
    if (self.fates_paramfile == ''):
        self.fates_paramfile = self.get_namelist_variable('fates_paramfile')
    print('FATES parameter file : '+self.fates_paramfile)
    self.fates_param_type = self.fates_paramfile.split('.')[-1].strip("'").strip('"')  #determine if json or nc

    fbase = self.OLMTdir+'/temp/fates_paramfile.'+self.fates_param_type
    os.system('cp '+self.fates_paramfile+' '+fbase)
    if (self.fates_pft >= 0):
        print('Extracting PFT '+str(self.fates_pft))
        if (self.pft_duplicates > 1):
          if (self.fates_param_type == 'nc'):
            print('Duplicating '+str(self.pft_duplicates)+' times.')
            write_fates_pft_subset_nc(self.OLMTdir+'/temp/fates_paramfile.nc',
                    self.OLMTdir+'/temp/fates_paramfile.nc', self.fates_pft,
                    duplicates=self.pft_duplicates)
          else:
            print('Duplicating '+str(self.pft_duplicates)+' times.')
            fname = self.OLMTdir+'/temp/fates_paramfile.'+self.fates_param_type
            pft_indices = ''
            for pf in range(0,self.pft_duplicates):
                pft_indices = pft_indices+str(self.fates_pft)+','
            swapper_path = self.modelroot+'/components/elm/src/external_models/fates/tools/pft_index_swapper.py'
            swapcmd=swapper_path+' --pft-indices='+pft_indices[:-1]+' --fin='+fbase+' --fout='+fname+' --silent'
            os.system(swapcmd)
        else:
            fname = self.OLMTdir+'/temp/fates_paramfile.'+self.fates_param_type
            if (self.fates_param_type == 'json'):
                swapper_path = self.modelroot+'/components/elm/src/external_models/fates/tools/pft_index_swapper.py'
                swapcmd=swapper_path+' --pft-indices=0,'+f'{self.fates_pft}'+' --fin='+fbase+' --fout='+fname+' --silent'
                os.system(swapcmd)
            else:
                write_fates_pft_subset_nc(self.OLMTdir+'/temp/fates_paramfile.nc',
                        fname, self.fates_pft)


    # Apply FATES parameter modifications
    if hasattr(self, 'add_fates_parameter') and self.add_fates_parameter:
        fates_param_path = self.OLMTdir + '/temp/fates_paramfile.'+self.fates_param_type
        if (self.fates_param_type == 'json'):
            self.modify_jsoninput_file(fates_param_path, self.add_fates_parameter, "FATES parameter")
        else:
            self.modify_ncinput_file(fates_param_path, self.add_fates_parameter, "FATES parameter")

  def set_finidat_file(self, finidat_case='', finidat_year=0, finidat=''):
      if (finidat_case != ''):
        self.finidat_yst = str(10000+finidat_year)[1:]
        self.finidat = self.runroot+'/'+finidat_case+'/run/'+ \
          finidat_case+'.elm.r.'+self.finidat_yst+'-01-01-00000.nc'
        self.finidat_year = finidat_year
      elif (finidat != ''):
        self.finidat = finidat
        self.finidat_year = int(finidat[-19:-15])
        self.finidat_yst=str(10000+finidat_year)[1:]
      self.has_finidat=True

#-----------------------------------------------------------------------------------------
  def create_case(self, machine='',casename='', remove=False):
    if (casename == ''):
      #construct default casename
      if (self.site == ''):
        self.casename = self.caseid+'_'+self.region+'_'+self.compset
      else:
        self.casename = self.caseid+'_'+self.site+"_"+self.compset
      self.casename = '_'.join(filter(None,[self.casename,self.case_suffix]))
    else:
        self.casename = casename
    self.casedir = os.path.abspath(self.caseroot+'/'+self.casename)
    #TODO - replace with a prompt that automacially deletes after 10 seconds
    if (os.path.exists(self.casedir)):
      if (not remove):
        print('Warning:  Case directory exists')
        var = input('proceed (p), remove old (r), or exit (x)? ')
        if var[0] == 'r':
          os.system('rm -rf '+self.casedir)
        if var[0] == 'x':
          sys.exit(1)
      else:
        print('Removing old case directory: '+self.casedir)
        os.system('rm -rf '+self.casedir)    
    print("CASE directory is: "+self.casedir)
    #create the case
    timestr=self.cime_walltime_string()
    #IF the resolution is user defined (site), we will first create a case with 
    #original resolution to get them correct domain, surface and land use files.
    if (self.apptainer != ''):
      cmd = 'apptainer exec --bind '+self.apptainer_bind+' --pwd '+self.modelroot+'/cime/scripts '+ \
              self.apptainer+' ./create_newcase --case '+self.casedir+ \
           ' --mach docker --compset '+self.compset+' --res '+self.res+' --walltime '+timestr+ \
           ' --handle-preexisting-dirs u' 
    else:
      cmd = './create_newcase --case '+self.casedir+' --mach '+self.machine+' --compset '+ \
           self.compset+' --res '+self.res+' --walltime '+timestr+' --handle-preexisting-dirs u' 
    if (self.project != ''):
      cmd = cmd+' --project '+self.project
    if (self.compiler != ''):
      cmd = cmd+' --compiler '+self.compiler
    if (self.mpilib != ''):
      cmd = cmd+' --mpilib '+self.mpilib
    case_queue = self.cime_job_queue()
    if (case_queue != ''):
      cmd = cmd+' --queue '+case_queue
    #ADD MPILIB OPTION HERE
    cmd = cmd+' > '+self.OLMTdir+'/create_newcase.log'
    os.chdir(self.modelroot+'/cime/scripts')

    result = subprocess.run(cmd, stdout=subprocess.PIPE, \
            stderr=subprocess.PIPE, text=True, shell=True)
    if (result.returncode > 0):
      print('Error:  Failed to create case.')
      print(result.stderr)
      sys.exit(1)
    os.chdir(self.casedir)
    if (self.inputdata_path == ''):
      self.inputdata_path=self.xmlquery('DIN_LOC_ROOT')
      print('Input directory not specified.')
      print('Setting to machine default: '+self.inputdata_path)
    self.rundir = self.runroot+'/'+self.casename+'/run'
    self.dobuild = False
    if (self.exeroot == ''):
        self.dobuild = True
        self.exeroot = self.runroot+'/'+self.casename+'/bld'

  def setup_domain_surfdata(self,makedomain=False,makesurfdat=False,makepftdyn=False, pft=-1):
     #------Make domain, surface data and pftdyn files ------------------
    os.chdir(self.OLMTdir)
    mysimyr=1850
    surffile=''
    domainfile=''
    pftdynfile=''
    if ('domainfile' in self.case_options.keys()):
        domainfile = self.case_options['domainfile']
    elif ('fatmlndfrc' in self.case_options.keys()):
        domainfile = self.case_options['fatmlndfrc']
    if ('surffile' in self.case_options.keys()):
        surffile = self.case_options['surffile']
    elif ('fsurdat' in self.case_options.keys()):
        surffile = self.case_options['fsurdat']
    if ('pftdynfile' in self.case_options.keys()):
        pftdynfile = self.case_options['pftdynfile']
        if  pftdynfile == '':
            self.nopftdyn = True
    elif ('flanduse_timeseries' in self.case_options.keys()):
        pftdynfile = self.case_options['flanduse_timeseries']
        if pftdynfile == '':
            self.nopftdyn = True
    if (domainfile == '' and makedomain):
      self.makepointdata(self.domain_global)
    if (surffile == '' and makesurfdat):
      self.makepointdata(self.surfdata_global, pft=pft)
    if (pftdynfile == '' and makepftdyn and not (self.nopftdyn)):
      self.makepointdata(self.pftdyn_global, pft=pft)
    domain_check_file = domainfile
    surf_check_file = surffile
    if (domain_check_file == '' and makedomain):
      domain_check_file = self.OLMTdir+'/temp/domain.nc'
    if (surf_check_file == '' and makesurfdat):
      surf_check_file = self.OLMTdir+'/temp/surfdata.nc'
    if (domain_check_file != '' and surf_check_file != '' and
            os.path.exists(domain_check_file) and os.path.exists(surf_check_file)):
      domain_count, domain_dims = elm_spatial_cell_count(domain_check_file, 'domain')
      surf_count, surf_dims = elm_spatial_cell_count(surf_check_file, 'surfdata')
      if (domain_count != surf_count):
        raise ValueError('Domain/surfdata spatial cell mismatch: domain has '+
                str(domain_count)+' cells '+str(domain_dims)+' in '+domain_check_file+
                '; surfdata has '+str(surf_count)+' cells '+str(surf_dims)+' in '+
                surf_check_file)
    if (domainfile != ''):
      print('\nDomain file:             '+ domainfile)
    if (surffile != ''):
      print('surface data file:       '+ surffile)  
    if (pftdynfile != ''):
      print('20th landuse data file: '+pftdynfile+"'\n")

  def get_metdata_year_range(self):
    #get site year information
    sitedatadir = os.path.abspath(self.inputdata_path+'/lnd/clm2/PTCLM')
    os.chdir(sitedatadir)
    if (self.forcing == 'site'):
      if (self.is_bypass()):
        #Get met data year range from all_hourly file
        mydata = Dataset(self.metdir+'/all_hourly.nc','r')
        self.met_startyear = mydata['start_year'][0]
        self.met_endyear   = mydata['end_year'][0]
        mydata.close()
      else:
        pattern = os.path.join(self.metdir,'????-??.nc')
        matching_files = glob.glob(pattern)
        years=[]
        for f in matching_files:
            years.append(int(f.split('/')[-1].split('-')[0]))
        self.met_startyear = min(years)
        self.met_endyear   = max(years)
      #if (self.met_endyear-self.met_startyear+1 > 20):
      #    self.met_endyear_spinup = self.met_startyear+20-1
      #else:
      self.met_endyear_spinup = self.met_endyear
    else:
        #Assume reanalysis
        self.met_startyear = 1901
        self.met_endyear   = 2014
        if ('daymet' in self.forcing):
            self.met_startyear = 1980
        if ('gfdl' in self.forcing):
            self.met_startyear = 1951
        if ('Qian' in self.forcing):
            self.met_startyear = 1948
        if ('era5' in self.forcing):
            self.met_endyear=2023
        if ('crujra' in self.forcing):
            self.met_endyear=2022
        if ('isimip' in self.forcing):
            self.met_startyear=1951
        #Assume we want a 20-year spinup cycle
        self.met_endyear_spinup = self.met_startyear+20-1
    self.nyears_spinup=self.met_endyear_spinup-self.met_startyear+1
    if (self.run_n == -1):
        #Set the transient run length such that last year is last year of met data
        self.run_n = self.met_endyear-self.startyear+1
    #force run length to be multiple of spinup met data years
    if (getattr(self, 'force_full_spinup_cycles', True) and not 'ICBELM' in self.compset and \
        not '20TR' in self.compset and not 'trans' in self.casename):
      while (self.run_n % self.nyears_spinup != 0):
          self.run_n = self.run_n+1
    if (not self.is_bypass() and not ('phase2' in self.casename)):
        #Figure out the align year
        testyear = self.met_startyear
        while (testyear > self.startyear):
            testyear = testyear - self.nyears_spinup
        self.met_alignyear = (self.startyear-testyear)+self.met_startyear
        if (self.met_endyear != self.met_endyear_spinup):
            self.met_endyear = self.met_endyear_spinup
        if ('20TR' in self.compset):
            self.run_n = self.met_endyear_spinup-self.startyear+1
    elif (not self.is_bypass() and ('phase2') in self.casename):
        self.met_alignyear = self.startyear
        self.met_startyear = self.startyear
    print('\nMet data source: '+self.forcing)
    print('Met data location: '+self.metdir)
    print('Starting met data year: ', self.met_startyear)
    if ('20TR' in self.compset or 'trans' in self.casename or 'ICBELM' in self.casename):
        print('Ending   met data year: ', self.met_endyear)
    else:
        print('Ending   met data year: ', self.met_endyear_spinup)
    if (not self.is_bypass()):
        print('Met data align year: ', self.met_alignyear)
    print('Run length (years): '+str(self.run_n)+'\n')

  def xmlchange(self, variable, value='', append=''):
      os.chdir(self.casedir)
      if (value != ''):
        os.system('./xmlchange '+variable+'='+value)
      elif (append != ''):
        os.system('./xmlchange --append '+variable+'='+append)

  def xmlquery(self, variable):
      result = subprocess.run(['./xmlquery','--value',variable], stdout=subprocess.PIPE)
      return result.stdout.decode('utf-8')

  def setup_case(self):
    os.chdir(self.casedir)
    #env_build
    self.xmlchange('SAVE_TIMING',value='FALSE')
    self.xmlchange('EXEROOT',value=self.exeroot)
    self.xmlchange('PIO_VERSION',value=str(self.pio_version))
    self.xmlchange('MOSART_MODE',value='NULL')
    self.xmlchange('ROF_GRID',value='null')
    if (self.debug):
      self.xmlchange('DEBUG',value='TRUE')
    #-------------- env_run.xml modifications -------------------------
    self.xmlchange('RUNDIR',value=self.rundir)
    self.xmlchange('DIN_LOC_ROOT',value=self.inputdata_path)
    self.xmlchange('DIN_LOC_ROOT_CLMFORC',value=self.inputdata_path+'/atm/datm7/')    
    #define mask and resoultion
    if (self.site != ''):
      self.xmlchange('ELM_USRDAT_NAME',value='1x1pt_'+self.site)
    if ('ad_spinup' in self.casename):
      self.xmlchange('ELM_BLDNML_OPTS',append="'-bgc_spinup on'")
    if ('PHS' in self.compset):
        self.xmlchange('ELM_BLDNML_OPTS',append='-hydrstress')
    #if ('CROP' in self.compset):
    #   self.xmlchange('ELM_BLDNML_OPTS',append='-crop')
    self.xmlchange('RUN_STARTDATE',value=str(self.startyear)+'-01-01')
    #turn off archiving
    self.xmlchange('DOUT_S',value='FALSE')
    #datm options
    if (not self.is_bypass() and not 'default' in self.forcing):
      if (not 'site' in self.forcing):
        self.datm_mode = 'CLMGSWP3v1'   #CLMCRUNCEP
        self.datm_mode = 'uELM_TES'
        self.xmlchange('DATM_MODE',value=self.datm_mode)
      else:
        self.xmlchange('DATM_MODE',value='CLM1PT') 
      self.xmlchange('DATM_CLMNCEP_YR_START',value=str(self.met_startyear))
      if ('phase2' in self.casename):
        self.xmlchange('DATM_CLMNCEP_YR_END',value=str(self.met_endyear))
      else:
        self.xmlchange('DATM_CLMNCEP_YR_END',value=str(self.met_endyear_spinup))
      self.xmlchange('DATM_CLMNCEP_YR_ALIGN',value=str(self.met_alignyear))
    #Change simulation timestep
    if (float(self.tstep) != 0.5):
      self.xmlchange('ATM_NCPL',value=str(int(24/float(self.tstep))))

    if (self.has_finidat):
      self.xmlchange('RUN_REFDATE',value=self.finidat_yst+'-01-01')
    #adds capability to run with transient CO2
    if ('20TR' in self.casename or 'trans' in self.casename):
      self.xmlchange('CCSM_BGC',value='CO2A')
      self.xmlchange('ELM_CO2_TYPE',value='diagnostic')
      #self.xmlchange('DATM_CO2_TSERIES',value="20tr")
    comps = ['ATM','LND','ICE','OCN','CPL','GLC','ROF','WAV','ESP','IAC']
    for c in comps:
      self.xmlchange('NTASKS_'+c,value=str(self.np))
      self.xmlchange('NTHRDS_'+c,value='1')

    # Pathfinder partition-specific override: match requested hardware capacity.
    if (self.machine == 'pathfinder' and self.slurm_partition() == 'hpcl-cli185'):
      self.xmlchange('MAX_TASKS_PER_NODE', value='128')

    self.xmlchange('STOP_OPTION',value='nyears')
    self.xmlchange('STOP_N',value=str(self.run_n))
    self.xmlchange('REST_N',value=str(self.run_n))
    if (self.site == ''):
        self.xmlchange('REST_N',value='20')

    # User-defined PFT count is a build-namelist option, not a runtime
    # namelist field. Surface files with custom lsmpft dimensions need this
    # before preview_namelists generates lnd_in.
    if ('maxpatch_pft' in self.case_options):
      maxpatch_pft = int(self.case_options['maxpatch_pft'])
      if (maxpatch_pft != 17):
        self.xmlchange('ELM_BLDNML_OPTS', append="'-maxpft "+str(maxpatch_pft)+"'")

    # for spinup and transient runs, PIO_TYPENAME is pnetcdf, which now not works well
    if('mac' in self.machine or 'cades' in self.machine or 'linux' in self.machine): 
      self.xmlchange('PIO_TYPENAME',value='netcdf')

    if (self.machine == 'pathfinder'):
      base_flags = self.cime_batch_command_flags('case.run')
      batch_flags = self.pathfinder_batch_command_flags(base_flags)
      if (batch_flags != ''):
        subprocess.run(['./xmlchange', '--subgroup', 'case.run',
                        'BATCH_COMMAND_FLAGS='+batch_flags], check=True)

    if (self.has_finidat):
        self.customize_namelist(variable='finidat',value="'"+self.finidat+"'")
    #Setup the new case
    setup_flags = self.cime_case_setup_flags()
    if (self.apptainer != ''):
        cmd = 'apptainer exec --bind '+self.apptainer_bind+' --pwd '+self.casedir+' '+self.apptainer + \
                    ' ./case.setup'+setup_flags
    else:
        cmd = './case.setup'+setup_flags
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, \
            shell=True)
    if (result.returncode > 0):
        print('Error: runcase.py failed to setup case')
        print(result.stderr)
        sys.exit(1)
    #get the default parameter files for the case
    if ('FATES' in self.compset or 'ED' in self.compset):
        self.set_fates_param_file()
        self.set_param_file()
    else:
        #if (not 'paramfile' in self.case_options.keys()):
        self.set_param_file()
    if (not 'fsoilordercon' in self.case_options.keys()):
        self.set_CNP_param_file()
    #get the default surface and domain files (to pass to makepointdata)
    #Note:  This requires setting a supported resolution
    if ('surffile_global' in self.case_options.keys()):
        self.surfdata_global = self.case_options['surffile_global']
    else:
        self.surfdata_global = self.get_namelist_variable('fsurdat')[2:-1]
    if ('domainfile_global' in self.case_options.keys()):
        self.domain_global = self.case_options['domainfile_global']
    else:
        self.domain_global   = self.get_namelist_variable('fatmlndfrc')[2:-1]
    if ('20TR' in self.casename):
        if ('pftdynfile_global' in self.case_options.keys()):
            self.pftdyn_global = self.case_options['pftdynfile_global']
        else:
            self.pftdyn_global = self.get_namelist_variable('flanduse_timeseries')[2:-1]
    #Set custom surface data information
    surffile=''
    domainfile=''
    pftdynfile=''
    if ('surffile' in self.case_options):
        surffile = self.case_options['surffile']
    elif ('fsurdat' in self.case_options):
        surffile = self.case_options['fsurdat']
    if ('domainfile' in self.case_options):
        domainfile = self.case_options['domainfile']
    elif ('fatmlndfrc' in self.case_options):
        domainfile = self.case_options['fatmlndfrc']
    if ('pftdynfile' in self.case_options):
        pftdynfile = self.case_options['pftdynfile']
    elif ('flanduse_timeseries' in self.case_options):
        pftdynfile = self.case_options['flanduse_timeseries']
    if (surffile==''):
        surffile = self.rundir+'/surfdata.nc'
    if (pftdynfile==''):
       pftdynfile = self.rundir+'/surfdata.pftdyn.nc'
    self.customize_namelist(variable='do_budgets',value='.false.')
    self.customize_namelist(variable='fsurdat',value="'"+surffile+"'")
    if ('20TR' in self.casename):
      if (self.nopftdyn):
          self.customize_namelist(variable='flanduse_timeseries',value="''")
      else:
          self.customize_namelist(variable='flanduse_timeseries',value="'"+pftdynfile+"'")
      self.customize_namelist(variable='check_finidat_fsurdat_consistency',value='.false.')
      self.customize_namelist(variable='check_finidat_year_consistency',value='.false.')
      self.set_histvars()
    else:
      self.set_histvars(spinup=True)
    #if (not 'paramfile' in self.case_options.keys()):
    self.customize_namelist(variable='paramfile',value="'"+self.rundir+"/clm_params.nc'")
    if (not 'fsoilordercon' in self.case_options.keys()):
      self.customize_namelist(variable='fsoilordercon',value="'"+self.rundir+"/CNP_parameters.nc'")
    #Fates options - TODO add nutrient/parteh options
    if ('ED' in self.compset or 'FATES' in self.compset):
        if 'CN' in self.nutrients:
          #Note, if carbon only, use parteh_mode = 1.
          self.customize_namelist(variable='fates_parteh_mode',value='2')
          bldnml = '" -nutrient '+self.nutrients.lower()+' -nutrient_comp_pathway '+ \
                  self.nutrient_comp.lower()+' -soil_decomp '+self.soil_decomp.lower()+'"'
          bldnml = bldnml.replace('cnt','century')
          self.xmlchange('ELM_BLDNML_OPTS',append=bldnml)
        self.customize_namelist(variable='fates_paramfile',value="'"+self.rundir+"/fates_paramfile."+ \
          self.fates_param_type+"'")
        #if (self.fates_logging):
        #    self.customize_namelist(variable='use_fates_logging',value='.true.')
    self.customize_namelist(variable='nyears_ad_carbon_only',value='25')
    self.customize_namelist(variable='spinup_mortality_factor',value='10')
    if (self.is_bypass()):
        #if using coupler bypass, need to add the following
        self.customize_namelist(variable='metdata_type',value="'"+self.forcing+"'")
        self.customize_namelist(variable='metdata_bypass',value="'"+self.metdir+"'")
        if (not 'co2_file' in self.case_options):
            self.customize_namelist(variable='co2_file', value="'"+self.inputdata_path+ \
                    "/atm/datm7/CO2/fco2_datm_rcp4.5_1765-2500_c130312.nc'")
        self.customize_namelist(variable='aero_file', value="'"+self.inputdata_path+"/atm/cam/chem/" \
                +"trop_mozart_aero/aero/aerosoldep_rcp4.5_monthly_1849-2104_1.9x2.5_c100402.nc'")
    #Excluded keys in case_options that are not namelist options (handled elsewhere)
    keys_exclude = ['suffix','surffile','domainfile','pftdynfile','paramfile','fates_paramfile', \
            'humhol','metdir','surffile_global','pftdynfile_global','domainfile_global', \
              'fsurdat', 'flanduse_timeseries', 'fatmlndfrac', 'maxpatch_pft', \
              'peatlands_upland_only', 'peatlands_upland_topounit', \
              'peatlands_upland_pfts', 'peatlands_upland_pft_fractions', \
              'site_npfts', 'site_pft_fractions', \
              'external_mask_file', 'external_mask_var', 'external_mask_min', \
              'external_mask_max', 'external_mask_values', 'external_mask_invert', \
              'external_mask_lat_var', 'external_mask_lon_var', \
              'external_mask_zero_surface', \
              'srcmods', 'variable', 'name', 'nyears']
    #Custom namelist options
    for key in self.case_options.keys():
        if (not key in keys_exclude and not 'restart_' in key):
            if (isinstance(self.case_options[key], str) and not ('hist_' in key) \
                    and not '.true.' in self.case_options[key] and \
                    not '.false.' in self.case_options[key] and (key != 'add_co2')):
                #Make these strings
                self.customize_namelist(variable=key,value="'"+self.case_options[key]+"'")
            else:
                self.customize_namelist(variable=key,value=str(self.case_options[key]))
        elif ('humhol' in key):
            self.humhol=True
    if ('ad_spinup' in self.casename):    #Turn on supplemental P for ad spinup
        self.customize_namelist(variable='suplphos',value="'ALL'")

    #set domain file information
    if (domainfile == ''):
      self.xmlchange('ATM_DOMAIN_PATH',value='"\\${RUNDIR}"')
      self.xmlchange('LND_DOMAIN_PATH',value='"\\${RUNDIR}"')
      self.xmlchange('ATM_DOMAIN_FILE',value='domain.nc')
      self.xmlchange('LND_DOMAIN_FILE',value='domain.nc')
    else:
      domainpath = '/'.join(domainfile.split('/')[:-1])
      domainfilename = domainfile.split('/')[-1]
      self.xmlchange('ATM_DOMAIN_PATH',value=domainpath)
      self.xmlchange('LND_DOMAIN_PATH',value=domainpath)
      self.xmlchange('ATM_DOMAIN_FILE',value=domainfilename)
      self.xmlchange('LND_DOMAIN_FILE',value=domainfilename)

    #global CPPDEF modifications
    if (self.humhol):
        self.cppdefs='HUM_HOL'
    if (self.is_bypass()):
      macrofiles=['./Macros.make','./Macros.cmake']
      for f in macrofiles:
          if (os.path.isfile(f)):
            infile  = open(f)
            outfile = open(f+'.tmp','a')  
            for s in infile:
              if ('CPPDEFS' in s and self.is_bypass()):
                 stemp = s[:-1]+' -DCPL_BYPASS\n'
                 outfile.write(stemp)
              elif ('llapack' in s):
                 outfile.write(s.replace('llapack','llapack -lgfortran'))
              else:
                 outfile.write(s.replace('mcmodel=medium','mcmodel=small'))
            infile.close()
            outfile.close()
            os.system('mv '+f+'.tmp '+f)
      if (os.path.isfile("./cmake_macros/universal.cmake")):
        os.system("echo 'string(APPEND CPPDEFS \" -DCPL_BYPASS\")' >> cmake_macros/universal.cmake")
    if (self.cppdefs != ''):
      #use for HUM_HOL, MARSH, HARVMOD, other cppdefs
      for cppdef in self.cppdefs.split(','):
         print("Turning on "+cppdef+" modification\n")
         self.xmlchange('ELM_CONFIG_OPTS',append='" -cppdefs -D'+cppdef+'"')
    if (self.srcmods != ''):
      if (os.path.exists(self.srcmods) == False):
        print('Invalid srcmods directory.  Exiting')
        sys.exit(1)
      os.system('cp -r '+self.srcmods+'/* '+self.casedir+'/SourceMods')

  def customize_namelist(self, namelist_file='', variable='', value=''):
    output = open("user_nl_elm",'a')
    if (namelist_file != ''):
        mynamelist = open(namelist_file,'r')
        for s in mynamelist:
            output.write(s)
    else:
        output.write(' '+variable+' = '+value+'\n')
    output.close()

  def preview_namelists(self):
      os.chdir(self.casedir)
      if os.path.exists(os.path.abspath(self.modelroot)+'/cime/scripts/Tools/preview_namelists'):
        cmdpath=os.path.abspath(self.modelroot)+'/cime/scripts/Tools'
      else:
        cmdpath=os.path.abspath(self.modelroot)+'/cime/CIME/Tools'
      if (self.apptainer != ''):
          cmd = 'apptainer exec --bind '+self.apptainer_bind+' '+ \
                ' --pwd '+self.casedir+' '+self.apptainer+' '+cmdpath+'/preview_namelists'
      else:
        cmd = cmdpath+'/preview_namelists'

      result = subprocess.run(cmd, stdout=subprocess.PIPE, \
            stderr=subprocess.PIPE, text=True, shell=True)
      if (result.returncode > 0):
        print('Error:  Failed to preview namelists.')
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        sys.exit(1)

  def build_case(self, clean=True):
      os.chdir(self.casedir)
      setup_flags = self.cime_case_setup_flags()
      #If using DATM, set the resolution to ELM_USRDAT
      if (not self.is_bypass()):
        #assume single point
        self.xmlchange('LND_GRID',value='ELM_USRDAT')
        self.xmlchange('ATM_GRID',value='ELM_USRDAT')
        self.xmlchange('LND_NX',value='1')
        self.xmlchange('LND_NY',value='1')
        self.xmlchange('ATM_NX',value='1')
        self.xmlchange('ATM_NY',value='1')
        if (self.apptainer != ''):
          if self.offline_driver:
            cmd = 'apptainer exec --bind '+self.apptainer_bind+' --pwd '+self.casedir+' '+self.apptainer+ \
                ' bash -lc "export CMAKE_ARGS=\"-DBUILD_ELM_OFFLINE_DRIVER=ON\" && ./case.setup'+setup_flags+'"'
          else:
            cmd = 'apptainer exec --bind '+self.apptainer_bind+' --pwd '+self.casedir+' '+self.apptainer+ \
                ' ./case.setup'+setup_flags
        else:
          if self.offline_driver:
            cmd = 'export CMAKE_ARGS="-DBUILD_ELM_OFFLINE_DRIVER=ON" && ./case.setup'+setup_flags
          else:
            cmd = './case.setup'+setup_flags
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, \
            shell=True)

      if (self.dobuild):
        if (self.apptainer != ''):
          if self.offline_driver:
            cmd = 'apptainer exec --bind '+self.apptainer_bind+' --pwd '+self.casedir+' '+self.apptainer + \
                ' bash -lc "export CMAKE_ARGS=\"-DBUILD_ELM_OFFLINE_DRIVER=ON\" && ./case.build"'
          else:
            cmd = 'apptainer exec --bind '+self.apptainer_bind+' --pwd '+self.casedir+' '+self.apptainer + \
                ' ./case.build'
        else:
          if self.offline_driver:
            cmd = 'export CMAKE_ARGS="-DBUILD_ELM_OFFLINE_DRIVER=ON" && ./case.build'
          else:
            cmd = './case.build'
        build_timeout = None
        if (self.interactive_build):
            # Wrap build command with srun to run on a compute node (avoids OOM on login node)
            project_opt = ' --account='+self.project if self.project != '' else ''
            build_partition = self.slurm_partition()
            build_qos = self.slurm_qos()
            qos_opt = ' --qos='+build_qos if build_qos != '' else ''
            resource_wait_minutes = 60
            build_timeout = resource_wait_minutes * 60
            srun_prefix = 'srun --nodes=1 --ntasks=1 --mem=32gb --time=1:00:00'+ \
                    qos_opt+' -c 1 --partition='+build_partition+' --kill-on-bad-exit=1' + \
                    project_opt + ' '
            cmd = srun_prefix+cmd
            print('Building on compute node: '+cmd)
            sinfo = subprocess.run('sinfo -h -p '+shlex.quote(build_partition)+' -t idle -o "%D"', stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, shell=True)
            idle_nodes = sum([int(x) for x in sinfo.stdout.split() if x.isdigit()])
            if idle_nodes == 0:
                print('Warning: no idle nodes in '+build_partition+' partition; waiting for one to become available ' + \
                        '(will time out after '+str(resource_wait_minutes)+' minutes).')
        if (clean):
          result = subprocess.run(cmd+' --clean-all', stdout=subprocess.PIPE, stderr=subprocess.PIPE, \
                  text=True, shell=True, timeout=build_timeout)
        try:
          result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, \
                  text=True, shell=True, timeout=build_timeout)
        except subprocess.TimeoutExpired:
          print('Error:  No compute node became available after '+str(resource_wait_minutes)+ \
                  ' minutes.  Aborting.')
          sys.exit(1)
        if (result.returncode > 0):
          print('Error:  Failed to build case.  Aborting')
          if result.stdout:
              print(result.stdout)
          if result.stderr:
              print(result.stderr)
          sys.exit(1)
      else:
        self.xmlchange('BUILD_COMPLETE',value='TRUE')
      #If using DATM, customize the stream files
      if (not self.is_bypass() and not 'default' in self.forcing):
          self.modify_datm_streamfiles()
      #Copy customized parameter, surface and domain files to run directory
      os.system('mkdir -p '+self.OLMTdir+'/temp')
      #if (not 'paramfile' in self.case_options.keys()):
      os.system('cp '+self.OLMTdir+'/temp/clm_params.nc '+self.rundir)
      if (not 'fsoilordercon' in self.case_options.keys()):
        os.system('cp '+self.OLMTdir+'/temp/CNP_parameters.nc '+self.rundir)
      if ('FATES' in self.compset or 'ED' in self.compset): #and (not 'fates_paramfile' in self.case_options.keys()):
        os.system('cp '+self.OLMTdir+'/temp/fates_paramfile.'+self.fates_param_type+' '+self.rundir)
      if (not 'domainfile' in self.case_options.keys() and not 'fatmlndfrc' in self.case_options.keys()):
         os.system('cp '+self.OLMTdir+'/temp/domain.nc '+self.rundir)
      if (not 'surffile' in self.case_options.keys() and not 'fsurdat' in self.case_options.keys()):
         cmd = 'cp '+self.OLMTdir+'/temp/surfdata.nc '+self.rundir
         execute = subprocess.call(cmd, shell=True)
      if (not 'pftdynfile' in self.case_options.keys() and '20TR' in self.compset and not(self.nopftdyn) \
        and not 'flanduse_timeseries' in self.case_options.keys()):
         os.system('cp '+self.OLMTdir+'/temp/surfdata.pftdyn.nc '+self.rundir)
      if (not self.dobuild):
         self.preview_namelists()

  def modify_datm_streamfiles(self):
    #stream file modifications for datm runs
    #Datm mods/ transient CO2 patch for transient run (datm buildnml mods)
    if (not self.is_bypass()):
      os.chdir(self.casedir)
      myinput  = open('./Buildconf/datmconf/datm_in')
      myoutput = open('user_nl_datm','w')
      for s in myinput:
          if ('streams =' in s):
              if ('trans' in self.casename or '20TR' in self.compset):
                  mypresaero = '"datm.streams.txt.presaero.trans_1850-2000 1850 1850 2000"'
                  myco2      = ', "datm.streams.txt.co2tseries.20tr 1766 1766 2010"'
              elif ('1850' in self.compset):
                  mypresaero = '"datm.streams.txt.presaero.clim_1850 1 1850 1850"'
                  myco2=''
              else:
                  mypresaero = '"datm.streams.txt.presaero.clim_2000 1 2000 2000"'
                  myco2=''
              if (not 'site' in self.forcing):
                  myoutput.write(' streams = "datm.streams.txt.'+self.datm_mode+'.Solar '+ \
                          str(self.met_alignyear)+' '+str(self.met_startyear)+' '+ \
                          str(self.met_endyear)+'  ", '+ \
                          '"datm.streams.txt.'+self.datm_mode+'.Precip '+ \
                          str(self.met_alignyear)+' '+str(self.met_startyear)+ \
                          ' '+str(self.met_endyear)+'  ", '+ \
                          '"datm.streams.txt.'+self.datm_mode+'.TPQW '+ \
                          str(self.met_alignyear)+' '+str(self.met_startyear)+ \
                          ' '+str(self.met_endyear)+'  ", '+mypresaero+myco2+ \
                          ', "datm.streams.txt.topo.observed 1 1 1"\n')
              else:
                  myoutput.write(' streams = "datm.streams.txt.CLM1PT.ELM_USRDAT '+ \
                          str(self.met_alignyear)+' '+str(self.met_startyear)+ \
                          ' '+str(self.met_endyear)+'  ", '+mypresaero+myco2+ \
                          ', "datm.streams.txt.topo.observed 1 1 1"\n')
          elif ('streams' in s):
              continue  #do nothing
          elif ('taxmode' in s):
              if (not 'site' in self.forcing):
                taxst = "taxmode = 'cycle', 'cycle', 'cycle', 'extend', 'extend'"
              else:
                taxst = "taxmode = 'cycle', 'extend', 'extend'"
              if ('trans' in self.casename or '20TR' in self.compset):
                  taxst = taxst+", 'extend'"
              myoutput.write(taxst+'\n')
          else:
              myoutput.write(s)
      myinput.close()
      myoutput.close()
      #Modify aerosol deposition file
      if (not self.is_bypass() and self.site != ''):
        if ('1850' in self.compset):
          myinput  = open('./Buildconf/datmconf/datm.streams.txt.presaero.clim_1850')
          myoutput = open('./user_datm.streams.txt.presaero.clim_1850','w')
        elif ('IELM' in self.compset):
          myinput  = open('./Buildconf/datmconf/datm.streams.txt.presaero.clim_2000')
          myoutput = open('./user_datm.streams.txt.presaero.clim_2000','w')
        if ('1850' in self.compset or 'IELM' in self.compset):
          for s in myinput:
            if ('aerosoldep_monthly' in s):
                myoutput.write('            aerosoldep_monthly_1849-2006_1.9x2.5_c090803.nc\n')
            else:
                myoutput.write(s)
          myinput.close()
          myoutput.close()
      #Modify CO2 file
      if ('20TR' in self.compset):
          myinput  = open('./Buildconf/datmconf/datm.streams.txt.co2tseries.20tr')
          myoutput = open('./user_datm.streams.txt.co2tseries.20tr','w')
          for s in myinput:
              if ('.nc' in s):
                  if (not 'co2_file' in self.case_options):
                    myoutput.write("      fco2_datm_rcp4.5_1765-2500_c130312.nc\n")
                  else:
                    myoutput.write("      "+self.case_options['co2_file']+'\n')
              else:
                  myoutput.write(s)
          myinput.close()
          myoutput.close()
      #Modify forcing file list
      if 'isimip' in self.forcing:
        for vv,vv2 in zip(['Precip', 'Solar', 'TPQW'], ['Prec','Solr','TPQWL']):
          myinput  = open('./Buildconf/datmconf/datm.streams.txt.'+self.datm_mode+'.'+vv)
          myoutput = open('./user_datm.streams.txt.'+self.datm_mode+'.'+vv,'w')
          for s in myinput:
              if 'atm_forcing.datm7.GSWP3.0.5d.v1.c170516' in s:
                  if not (vv in s or 'TPHWL' in s):
                    domainpath = '/'.join(self.metdir.split('/')[:-2])
                    myoutput.write('     '+domainpath+'\n')
                  else:
                    myoutput.write('     '+self.metdir+'/'+vv2+'\n')
              elif 'domain.lnd.360x720_gswp3.0v1.c170606.nc' in s:
                 myoutput.write('     domain.lnd.360x720_isimip.3b.c211109.nc\n')
              elif 'clmforc.GSWP3.c2011' in s:
                 gcm = self.metdir.split('/')[-2]
                 scenario = self.metdir.split('/')[-1]
                 s2 = s.replace('clmforc.GSWP3.c2011', 'clmforc.'+gcm+'.'+scenario+'.c2107'
                                ).replace(vv, vv2).replace('TPQWLL','TPQWL')
                 myoutput.write(s2)
              else:
                 myoutput.write(s)
          myinput.close()
          myoutput.close()

      if (self.forcing == 'site'):
          myinput  = open('./Buildconf/datmconf/datm.streams.txt.CLM1PT.ELM_USRDAT')
          myoutput = open('./user_datm.streams.txt.CLM1PT.ELM_USRDAT','w')
          for s in myinput:
              if ('CLM1PT_data' in s):
                  if (self.metdir != ''):
                    #Replace with user-specified directory
                    myoutput.write(self.metdir+'\n')
                  else:
                    #reverse directories for CLM1PT and site  
                    temp = s.replace('CLM1PT_data', 'TEMPSTRING')
                    s    = temp.replace('1x1pt'+'_'+self.site, 'CLM1PT_data')
                    temp  =s.replace('TEMPSTRING', '1x1pt'+'_'+self.site)
                    myoutput.write(temp)
              elif (('ED' in self.compset or 'FATES' in self.compset) and 'FLDS' in s):
                  print('Not including FLDS in atm stream file')
              else:
                  myoutput.write(s)
          myinput.close()
          myoutput.close()

      # run preview_namelists to copy user_datm.streams.... to CaseDocs
      if os.path.exists(os.path.abspath(self.modelroot)+'/cime/scripts/Tools/preview_namelists'):
        cmdpath=os.path.abspath(self.modelroot)+'/cime/scripts/Tools'
      else:
        cmdpath=os.path.abspath(self.modelroot)+'/cime/CIME/Tools'
      if (self.apptainer != ''):
          cmd = 'apptainer exec --bind '+self.apptainer_bind+' '+ \
                ' --pwd '+self.casedir+' '+self.apptainer+' '+cmdpath+'/preview_namelists'
      else:
        cmd = cmdpath+'/preview_namelists'

      result = subprocess.run(cmd, stdout=subprocess.PIPE, \
            stderr=subprocess.PIPE, text=True, shell=True)
      if (result.returncode > 0):
        print('Error:  Failed to preview namelists.')
        print(result.stderr)
        sys.exit(1)

  def submit_case(self,depend=-1,ensemble=False, multisite_script=''):
    #Create a pickle file of the model object for later use
    #Keep a copy in the case directory and OLMT directory
    self.create_pkl(outdir=self.casedir)
    self.create_pkl(outdir=self.OLMTdir+'/pklfiles')

    mysubmit='sbatch'
    #Submit the case with dependency if requested
    #Return the job id
    if (ensemble):
        #Create the PBS script
        scriptfile = create_ensemble_script(self)
    elif (multisite_script != ''):
        scriptfile = multisite_script
    else:
        scriptfile = './case.submit'
    os.chdir(self.casedir)
    if not isinstance(scriptfile, list):
      scriptfiles = [scriptfile]
    else:
      scriptfiles = scriptfile

    jobnum=0
    jobnum_depend=depend
    for script in scriptfiles:
      raw_sbatch_args = self.slurm_submit_args() if (ensemble or multisite_script != '') else []
      if (jobnum_depend > 0 and not self.noslurm):
        if (ensemble or multisite_script != ''):
            cmd = [mysubmit,'--dependency=afterok:'+str(jobnum_depend)] + raw_sbatch_args + [script]
        else:
            cmd = [script,'--prereq',str(jobnum_depend)]
      else:
        if ((ensemble or multisite_script != '') and not self.noslurm):
            cmd = [mysubmit] + raw_sbatch_args + [script]
        else:
            cmd = [script]
      if (self.noslurm):
          log_file_path='./case_submit.log'
          with open(log_file_path, "a") as log_file:
              result = subprocess.run(cmd, stderr=subprocess.STDOUT, \
                  stdout=log_file)
              jobnum=0
      else:
          #code.interact(local=dict(globals(), **locals()))
          result = subprocess.run(cmd, stderr=subprocess.STDOUT, \
                  stdout=subprocess.PIPE, text=True)
          output = result.stdout.strip()
          if (result.returncode != 0):
              raise RuntimeError('Failed to submit '+script+':\n'+output)
          jobnum = parse_submit_jobnum(output)
          print('\nSubmitted '+str(jobnum)+' from '+script)
          jobnum_depend=jobnum
    if (not ensemble and multisite_script == '' and getattr(self, 'postproc_vars', [])):
      postproc_script = self.create_postprocess_script()
      if (self.noslurm):
          log_file_path='./case_submit.log'
          with open(log_file_path, "a") as log_file:
              subprocess.run([postproc_script], stderr=subprocess.STDOUT, stdout=log_file)
      else:
          cmd = [mysubmit, '--dependency=afterok:'+str(jobnum)] + self.slurm_submit_args(ntasks=1) + [postproc_script]
          result = subprocess.run(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, text=True)
          output = result.stdout.strip()
          if (result.returncode != 0):
              raise RuntimeError('Failed to submit '+postproc_script+':\n'+output)
          postproc_jobnum = parse_submit_jobnum(output)
          print('\nSubmitted '+str(postproc_jobnum)+' from '+postproc_script)
    os.chdir(self.OLMTdir)
    return jobnum

  def create_postprocess_script(self):
    scriptfile = self.casedir+'/case.postprocess'
    log_file = self.rundir+'/postprocess.log'
    with open(scriptfile, 'w') as myfile:
        myfile.write('#!/bin/bash\n')
        partition = self.slurm_partition()
        if (partition != ''):
            myfile.write('#SBATCH -p '+partition+'\n')
        myfile.write('#SBATCH -n 1\n')
        qos = self.slurm_qos() if self.machine == 'pathfinder' else ''
        if (qos != ''):
            myfile.write('#SBATCH --qos='+qos+'\n')
        if (self.project != ''):
            myfile.write('#SBATCH --account='+self.project+'\n')
        myfile.write('#SBATCH --time='+self.cime_walltime_string()+'\n')
        myfile.write('set -e\n')
        myfile.write('cd '+shlex.quote(self.OLMTdir)+'\n')
        myfile.write('python manage_postproc.py --case '+shlex.quote(self.casename)+' > '+
                shlex.quote(log_file)+' 2>&1\n')
    os.chmod(scriptfile, 0o755)
    return scriptfile

  def create_pkl(self, outdir='./pklfiles'):
    os.chdir(self.OLMTdir)
    os.system('mkdir -p pklfiles')
    #print(os.path.abspath(outdir+'/'+self.casename+'.pkl'))
    with open(outdir+'/'+self.casename+'.pkl','wb') as file_out:
        pickle.dump(self, file_out)


# Dynamically import and add methods to ELMcase
def _add_methods_from_module(module):
    for name in dir(module):
        if not name.startswith("_"):
            try:
              method = getattr(module, name)
              if callable(method):
                setattr(ELMcase, name, method)
            except Exception as e:
                print(f"Error adding method {name}: {e}")

# Import modules and add their functions as methods to CLMcase
from . import ensemble, makepointdata, netcdf4_functions, set_histvars, postprocess, get_fluxnet_obs, \
        surrogate_NN, run_GSA, MCMC

_add_methods_from_module(ensemble)
_add_methods_from_module(makepointdata)
_add_methods_from_module(netcdf4_functions)
_add_methods_from_module(set_histvars)
_add_methods_from_module(postprocess)
_add_methods_from_module(get_fluxnet_obs)
_add_methods_from_module(surrogate_NN)
_add_methods_from_module(run_GSA)
_add_methods_from_module(MCMC)
