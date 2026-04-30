#!/usr/bin/env python

import os, sys, csv, time, math
import numpy as np
import datetime
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import xarray as xr
import json
import code  # For development: code.interact(local=dict(globals(), **locals()))


def parse_sbatch(case_run_path):
    """Parse SBATCH directives from a .case.run file and return a dict.

    Returns keys (ints or strings or None):
      nodes, ntasks, ntasks_per_node, cpus_per_task,
      partition, job_name, exclusive (bool), output, error,
      account, time, qos, mem_per_cpu, gres
    """
    parsed = {
        'nodes': None,
        'ntasks': None,
        'ntasks_per_node': None,
        'cpus_per_task': None,
        'partition': None,
        'job_name': None,
        'exclusive': None,
        'output': None,
        'error': None,
        'account': None,
        'time': None,
        'qos': None,
        'mem_per_cpu': None,
        'gres': None,
    }
    if not os.path.exists(case_run_path):
        return parsed
    try:
        import re
        with open(case_run_path, 'r') as cr:
            for line in cr:
                l = line.strip()
                if not l.startswith('#SBATCH'):
                    continue
                line_content = l.lstrip('#').strip()
                m = re.search(r'--nodes[=\s]+(\d+)', line_content)
                if m:
                    parsed['nodes'] = int(m.group(1))
                m = re.search(r'-N\s+(\d+)', line_content)
                if m:
                    parsed['nodes'] = int(m.group(1))
                m = re.search(r'--ntasks[=\s]+(\d+)', line_content)
                if m:
                    parsed['ntasks'] = int(m.group(1))
                m = re.search(r'-n\s+(\d+)', line_content)
                if m:
                    parsed['ntasks'] = int(m.group(1))
                m = re.search(r'--ntasks-per-node[=\s]+(\d+)', line_content)
                if m:
                    parsed['ntasks_per_node'] = int(m.group(1))
                m = re.search(r'--partition[=\s]+(\S+)', line_content)
                if m:
                    parsed['partition'] = m.group(1)
                m = re.search(r'-p\s+(\S+)', line_content)
                if m:
                    parsed['partition'] = m.group(1)
                m = re.search(r'--job-name[=\s]+(\S+)', line_content)
                if m:
                    parsed['job_name'] = m.group(1)
                m = re.search(r'-J\s+(\S+)', line_content)
                if m:
                    parsed['job_name'] = m.group(1)
                if '--exclusive' in line_content:
                    parsed['exclusive'] = True
                m = re.search(r'--output[=\s]+(\S+)', line_content)
                if m:
                    parsed['output'] = m.group(1)
                m = re.search(r'-o\s+(\S+)', line_content)
                if m:
                    parsed['output'] = m.group(1)
                m = re.search(r'--error[=\s]+(\S+)', line_content)
                if m:
                    parsed['error'] = m.group(1)
                m = re.search(r'-e\s+(\S+)', line_content)
                if m:
                    parsed['error'] = m.group(1)
                m = re.search(r'--account[=\s]+(\S+)', line_content)
                if m:
                    parsed['account'] = m.group(1)
                m = re.search(r'-A\s+(\S+)', line_content)
                if m:
                    parsed['account'] = m.group(1)
                m = re.search(r'--time[=\s]+(\S+)', line_content)
                if m:
                    parsed['time'] = m.group(1)
                m = re.search(r'-t\s+(\S+)', line_content)
                if m:
                    parsed['time'] = m.group(1)
                m = re.search(r'--qos[=\s]+(\S+)', line_content)
                if m:
                    parsed['qos'] = m.group(1)
                m = re.search(r'--mem-per-cpu[=\s]+(\S+)', line_content)
                if m:
                    parsed['mem_per_cpu'] = m.group(1)
                m = re.search(r'--gres[=\s]+(\S+)', line_content)
                if m:
                    parsed['gres'] = m.group(1)
                m = re.search(r'--cpus-per-task[=\s]+(\d+)', line_content)
                if m:
                    parsed['cpus_per_task'] = int(m.group(1))
                m = re.search(r'-c\s+(\d+)', line_content)
                if m:
                    parsed['cpus_per_task'] = int(m.group(1))
    except Exception:
        pass
    return parsed


def write_sbatch(parsed, myfile):
    """Write SBATCH directives to `myfile` from `parsed` dict.
    Note: callers may override `nodes` or `job_name` after parsing. This
    function will write `nodes` if present in `parsed` unless the caller
    intends to manage those values itself.
    """
    ps = parsed or {}
    # write nodes and job name if provided
    if ps.get('nodes') is not None:
        myfile.write('#SBATCH -N '+str(ps.get('nodes'))+'\n')
    if ps.get('job_name'):
        myfile.write('#SBATCH -J '+str(ps.get('job_name'))+'\n')
    if ps.get('partition'):
        myfile.write('#SBATCH -p '+str(ps.get('partition'))+'\n')
    if ps.get('account'):
        myfile.write('#SBATCH -A '+str(ps.get('account'))+'\n')
    if ps.get('qos'):
        myfile.write('#SBATCH --qos='+str(ps.get('qos'))+'\n')
    if ps.get('time'):
        myfile.write('#SBATCH --time='+str(ps.get('time'))+'\n')
    # tasks and cpu options
    # Ensure --ntasks-per-node is always emitted when possible (preferred).
    derived_ntasks_per_node = None
    if ps.get('ntasks_per_node') is not None:
        derived_ntasks_per_node = int(ps.get('ntasks_per_node'))
    elif ps.get('ntasks') is not None and ps.get('nodes') is not None:
        # derive ceil(ntasks / nodes)
        try:
            derived_ntasks_per_node = int(math.ceil(float(ps.get('ntasks'))/float(ps.get('nodes'))))
        except Exception:
            derived_ntasks_per_node = None
    if derived_ntasks_per_node is not None:
        myfile.write('#SBATCH --ntasks-per-node='+str(derived_ntasks_per_node)+'\n')
    # Intentionally do NOT emit a global `-n` directive; prefer only
    # `--ntasks-per-node` so downstream tools derive total tasks explicitly.
    if ps.get('cpus_per_task') is not None:
        myfile.write('#SBATCH -c '+str(ps.get('cpus_per_task'))+'\n')
    if ps.get('exclusive'):
        myfile.write('#SBATCH --exclusive\n')
    if ps.get('output'):
        myfile.write('#SBATCH --output='+str(ps.get('output'))+'\n')
    if ps.get('error'):
        myfile.write('#SBATCH --error='+str(ps.get('error'))+'\n')
    if ps.get('mem_per_cpu'):
        myfile.write('#SBATCH --mem-per-cpu='+str(ps.get('mem_per_cpu'))+'\n')
    if ps.get('gres'):
        myfile.write('#SBATCH --gres='+str(ps.get('gres'))+'\n')


def find_latest_restart(finidat_path):
    """
    Search `finidat_path` for files matching `*elm.r.*` and
    return the most recent valid restart file's absolute path.

    A valid restart file here is one that:
      - is a regular file
      - has non-zero size
      - can be opened by netCDF4.Dataset

    Returns `None` if no valid restart file is found.
    """
    import glob

    if not finidat_path:
        return None
    pattern = os.path.join(os.path.abspath(finidat_path), '**', '*elm.r.*')
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None

    # Filter to regular, non-empty files and verify netCDF readability
    valid = []
    for f in candidates:
        try:
            if not os.path.isfile(f):
                continue
            if os.path.getsize(f) == 0:
                continue
            # try opening with netCDF4 to ensure it's a readable restart file
            ds = Dataset(f, 'r')
            ds.close()
            valid.append(f)
        except Exception:
            continue

    if not valid:
        return None

    latest = max(valid, key=lambda x: os.path.getmtime(x))
    return os.path.abspath(latest)

#Read the parameter list file
def read_parm_list(self, parm_list=''):
    os.chdir(self.OLMTdir)
    if (os.path.exists(parm_list)):
        myfile = open(parm_list,'r')
        self.ensemble_parms=[]
        self.ensemble_pfts=[]
        self.ensemble_pmin=[]
        self.ensemble_pmax=[]
        for s in myfile:
            if (not '#' in s[0:3]):
              vals = s.split()
              self.ensemble_parms.append(vals[0].strip())
              self.ensemble_pfts.append(int(vals[1].strip()))
              self.ensemble_pmin.append(float(vals[2].strip()))
              self.ensemble_pmax.append(float(vals[3].strip()))
        myfile.close()
    else:
        print('parm_list file '+parm_list+' does not exist.  Exiting')
        sys.exit(1)
    self.nparms_ensemble = len(self.ensemble_parms)

def get_default_parms(self):
    parm_file = Dataset(self.OLMTdir+'/temp/clm_params.nc','r')
    parm_ds = xr.open_dataset(self.OLMTdir+'/temp/clm_params.nc',decode_timedelta=False)
    data_dict = parm_ds.to_dict()
    
    parm_file = Dataset(self.OLMTdir+'/temp/clm_params.nc','r')
    parm_ds = xr.open_dataset(self.OLMTdir+'/temp/clm_params.nc',decode_timedelta=False)
    data_dict = parm_ds.to_dict()
    
    if ('FATES' in self.compset or 'ED' in self.compset):
      if (self.fates_param_type == 'json'):
        fates_parm_file = json.load(open(self.OLMTdir+'/temp/fates_paramfile.json','r'))
      else:
        fates_parm_file = Dataset(self.OLMTdir+'/temp/fates_paramfile.nc','r')
    
    CNP_parm_file = Dataset(self.OLMTdir+'/temp/CNP_parameters.nc','r')
    self.default_parms=[]
    CNP_parms = ['ks_sorption', 'r_desorp', 'r_weather', 'r_adsorp', 'k_s1_biochem', 'smax', 'k_s3_biochem', \
        'r_occlude', 'k_s4_biochem', 'k_s2_biochem']      
    surfparms = ['MONTHLY_LAI', 'ORGANIC', 'PCT_SAND', 'PCT_CLAY', 'SECONDARY_P', 'APATITE_P', \
            'LABILE_P', 'OCCLUDED_P']
    for i, p in enumerate(self.ensemble_parms):        
        if 'fates' in p:
            if (self.fates_param_type == 'json'):
                param_var = np.array(fates_parm_file['parameters'][p]['data'])
                # Check dimensions of the variable
                ndim = len(param_var.shape)
            else:
                param_var = fates_parm_file[p]
                ndim = len(param_var.dimensions)
        elif p in CNP_parms:
            param_var = CNP_parm_file[p]
            ndim = len(param_var.dimensions)
        elif p in surfparms:
            surffile = Dataset(self.OLMTdir+'/temp/surfdata.nc','r')
            param_var = surffile[p]
            ndim = len(param_var.dimensions)
        else:   
            param_var = parm_file[p]
            ndim = len(param_var.dimensions)    
        # Check the dimensions of the parameter
        if ndim == 0:
            # Scalar parameter - no PFT indexing needed
            self.default_parms.append(param_var[:])  # Read scalar value
        else:
            # Parameter has dimensions - use PFT indexing
            pft_index = self.ensemble_pfts[i]
            self.default_parms.append(param_var[pft_index])
    
    parm_file.close()

#Create the samples file
def create_samples(self,sampletype='monte_carlo',nsamples=100,parm_list=''):
    self.nsamples=nsamples
    self.samples=np.zeros((self.nparms_ensemble,self.nsamples), float)
    for i in range(0,self.nsamples):
        for j in range(0,self.nparms_ensemble):
            self.samples[j,i] = self.ensemble_pmin[j]+(self.ensemble_pmax[j]- \
                    self.ensemble_pmin[j])*np.random.rand(1)
    self.ensemble_file = 'parm_samples/mcsamples_'+self.caseid+'_'+str(self.nsamples)+'.txt'
    os.system('mkdir -p parm_samples')
    np.savetxt(self.ensemble_file,np.transpose(self.samples))

def create_ensemble_script(self):
    #Create the PBS script we will submit to run the ensemble
    os.chdir(self.casedir)
    #Get the LD_LIBRARY_PATH from software environment
    softenv = open('software_environment.txt','r')
    for s in softenv:
        if s.split('=')[0].strip() == 'LD_LIBRARY_PATH':
            ldpath = s.split('=')[1].strip()
    softenv.close()
    self.npernode=int(self.xmlquery('MAX_TASKS_PER_NODE'))
    nnodes = int(np.ceil((self.np_ensemble*self.np)/self.npernode))
    myfile = open('case.submit_ensemble','w')
    myfile.write('#!/bin/bash -e\n\n')
    case_run = os.path.join(self.casedir, '.case.run')
    parsed_sbatch = parse_sbatch(case_run)
    if (self.queue == 'debug'):
        print('Debug queue selected, setting walltime to 2 hours')
        self.walltime=2
    total_tasks = int(self.np_ensemble) * int(self.np)
    parsed_sbatch['qos'] = self.queue
    parsed_sbatch['time'] = str(self.walltime)+':00:00'
    parsed_sbatch['nodes'] = int(nnodes)
    parsed_sbatch['ntasks'] = int(total_tasks)
    parsed_sbatch['ntasks_per_node'] = int(math.ceil(float(parsed_sbatch['ntasks'])/float(parsed_sbatch['nodes'])))
    parsed_sbatch['exclusive'] = False
    # Write SBATCH directives using helper
    write_sbatch(parsed_sbatch, myfile)

    myfile.write('cd '+self.caseroot+'/'+self.casename+'\n')
    myfile.write('export LD_LIBRARY_PATH='+ldpath+'\n\n')
    if (self.apptainer != ''):
      myfile.write('apptainer exec --bind '+self.apptainer_bind+' '+ \
                ' --pwd '+self.caseroot+'/'+self.casename+ ' '+ \
                self.apptainer+' ./preview_namelists\n')
    else:     
      myfile.write('./preview_namelists\n')
    myfile.write('ulimit -n '+str(self.nsamples+1024)+'\n')
    myfile.write('cd '+self.OLMTdir+'\n')
    myfile.write('./manage_ensemble.py --case '+self.casename+'\n')
    myfile.close()  
    os.system('chmod u+x case.submit_ensemble')
    self.rundir_UQ = self.runroot+'/UQ/ensembles/'+self.casename
    os.system('mkdir -p '+self.rundir_UQ)
    self.UQ_output = self.runroot+'/UQ/analysis/'+self.casename
    os.system('mkdir -p '+self.UQ_output)

def create_multisite_script(self,sites,scriptdir,cases_compare=""):
    #Create the PBS script we will submit to run multiple sites
    os.chdir(self.casedir)
    #Get the LD_LIBRARY_PATH from software environment
    softenv = open('software_environment.txt','r')
    for s in softenv:
        if s.split('=')[0].strip() == 'LD_LIBRARY_PATH':
            ldpath = s.split('=')[1].strip()
    softenv.close()
    self.npernode=int(self.xmlquery('MAX_TASKS_PER_NODE'))
    nruns = len(sites) if len(sites) > 0 else 1
    total_tasks = int(self.np) * int(nruns)
    nnodes = max(1, int(np.ceil(float(total_tasks) / float(self.npernode))))
    fname = self.casename.replace('_'+self.site,'')+'.sh'
    myfile = open(fname,'w')
    myfile.write('#!/bin/bash -e\n\n')
    case_run = os.path.join(self.casedir, '.case.run')
    parsed_sbatch = parse_sbatch(case_run)
    if (self.queue == 'debug'):
        print('Debug queue selected, setting walltime to 2 hours')
        self.walltime=2
    parsed_sbatch['qos'] = self.queue
    parsed_sbatch['time'] = str(self.walltime)+':00:00'
    parsed_sbatch['nodes'] = int(nnodes)
    parsed_sbatch['ntasks'] = int(total_tasks)
    parsed_sbatch['ntasks_per_node'] = int(math.ceil(float(total_tasks)/float(nnodes)))
    # Ensure we do not emit --exclusive for multisite submission either.
    parsed_sbatch['exclusive'] = False
    # Write SBATCH directives using helper 
    write_sbatch(parsed_sbatch, myfile)

    myfile.write('cd '+self.caseroot+'/'+self.casename+'\n')
    myfile.write('export LD_LIBRARY_PATH='+ldpath+'\n\n')
    for s in sites:
      site_casename = self.casename.replace(sites[0],s)
      site_casedir = self.caseroot+'/'+site_casename
      site_rundir = self.runroot+'/'+site_casename+'/run'
      myfile.write('cd '+site_casedir+'\n')
      if (self.apptainer != ''):
        myfile.write('apptainer exec --bind '+self.apptainer_bind+' '+ \
                ' --pwd '+site_casedir+ ' '+ \
                self.apptainer+' ./preview_namelists\n')
      else:
        myfile.write('./preview_namelists\n')
      myfile.write('cd '+site_rundir+'\n')
      myfile.write('mkdir -p timing/checkpoints\n')
      #restart file options
      for key in self.case_options.keys():
        if ('restart_' in key):
            var   = key[8:]
            value = str(self.case_options[key])
            if ('*' in value or '+' in value):
                operator=value[0]
                value=value[1:]
                myfile.write('python '+self.OLMTdir+'/modify_netcdf.py --filename '+ \
                    self.finidat+' --var '+var+' --val '+value+ \
                    ' --operator "'+operator+'"\n')
            else:
                myfile.write('python '+self.OLMTdir+'/modify_netcdf.py --filename '+ \
                    self.finidat+' --var '+var+' --val '+value+'\n')
      if (self.noslurm):
        myfile.write('mpiexec -n '+str(self.np)+' '+self.exeroot+'/e3sm.exe > '+ \
                     site_rundir+'/e3sm_log.txt &\n\n')
      else:
          # compute nodes required for this site run and align srun options
          nodes_per_run = max(1, int(np.ceil(float(self.np) / float(self.npernode))))
          ntasks_run = int(self.np)
          tasks_per_node_run = max(1, int(math.ceil(float(ntasks_run) / float(nodes_per_run))))
          cpus_per_task = parsed_sbatch['cpus_per_task'] if parsed_sbatch.get('cpus_per_task') is not None else 1
          if (self.apptainer != ''):
                myfile.write('srun --nodes='+str(nodes_per_run)+' --ntasks='+str(ntasks_run)+' --ntasks-per-node='+str(tasks_per_node_run)+' --cpu-bind=none -c '+str(cpus_per_task)+' apptainer exec '+ \
                    ' --bind '+self.apptainer_bind+' --env OMPI_MCA_pml=ob1 --env OMPI_MCA_btl=self,vader,tcp  ' \
                    +self.apptainer+' '+self.exeroot+'/e3sm.exe > '+ \
                    site_rundir+'/e3sm_log.txt &\n\n')
          else:
                myfile.write('srun --nodes='+str(nodes_per_run)+' --ntasks='+str(ntasks_run)+' --ntasks-per-node='+str(tasks_per_node_run)+' --cpu-bind=none -c '+str(cpus_per_task)+' '+self.exeroot+'/e3sm.exe > '+ \
                    site_rundir+'/e3sm_log.txt &\n\n')
    myfile.write('wait\n')
    myfile.write('cd '+self.OLMTdir+'\n')
    for s in sites:
        # Check if it's a site run (single point simulation)
        is_site_run = hasattr(self, 'site') and self.site != '' and self.site is not None
        if (not 'ICBELM' in self.compset and not '20TR' in self.compset and not 'trans' in self.casename \
            and not 'ad_spinup' in self.casename and not 'FATES' in self.compset and \
            not 'ED' in self.compset and is_site_run):
            #Assume this is a final spinup case, do spinup diagnostic plots
            myfile.write('python manage_postproc.py --case '+self.casename.replace(sites[0],s)+' --plot_spinup\n')
        elif self.postproc_vars:
            #Do requested postprocessing and plotting
            postproc_cmd = 'python manage_postproc.py --case '+self.casename.replace(sites[0],s)
            myfile.write(postproc_cmd + '\n')
            if cases_compare and cases_compare.strip() and s == sites[-1]:
                postproc_cmd += ' --cases_compare "' + cases_compare + '"'
                myfile.write(postproc_cmd + '\n')
    myfile.close()
    os.system('chmod u+x '+fname)
    return os.path.abspath('./'+fname)

def ensemble_copy(self, ens_num):

  gst=str(100000+int(ens_num))

  # create ensemble directory from original case 
  orig_dir = str(os.path.abspath(self.runroot)+'/'+self.casename+'/run')
  ens_dir  = str(os.path.abspath(self.rundir_UQ)+'/g'+gst[1:])
		
  os.system('mkdir -p '+ens_dir+'/timing/checkpoints')
  os.system('rm -f '+ens_dir+'/*.log.* '+ens_dir+'/*.nc '+ens_dir+'/rpointer*')
  os.system('cp  '+orig_dir+'/*_in* '+ens_dir)
  os.system('cp  '+orig_dir+'/*nml '+ens_dir)
  if (not ('CB' in self.casename)):
    os.system('cp  '+orig_dir+'/*stream* '+ens_dir)
  os.system('cp  '+orig_dir+'/*.rc '+ens_dir)
  os.system('cp  '+orig_dir+'/surf*.nc '+ens_dir)
  os.system('cp  '+orig_dir+'/domain*.nc '+ens_dir)
  os.system('cp  '+orig_dir+'/*para*.nc '+ens_dir)


  # loop through all filenames, change directories in namelists, change parameter values
  for f in os.listdir(ens_dir):
    if (os.path.isfile(ens_dir+'/'+f) and (f[-2:] == 'in' or f[-3:] == 'nml' or 'streams' in f)):
        myinput=open(ens_dir+'/'+f)
        myoutput=open(ens_dir+'/'+f+'.tmp','w')
        for s in myinput:
            if ('fates_paramfile' in s):
                paramfile_orig = ((s.split()[2]).strip("'"))
                if (paramfile_orig[0:2] == './'):
                  paramfile_orig = orig_dir+'/'+paramfile_orig[2:]
                paramfile_new  = ens_dir+'/fates_params_'+gst[1:]+'.nc'
                os.system('cp '+paramfile_orig+' '+paramfile_new)
                os.system('nccopy -3 '+paramfile_new+' '+paramfile_new+'_tmp')
                os.system('mv '+paramfile_new+'_tmp '+paramfile_new)
                myoutput.write(" fates_paramfile = '"+paramfile_new+"'\n")
                fates_paramfile = ens_dir+'/fates_params_'+gst[1:]+'.nc'
            elif ('paramfile' in s):
                paramfile_orig = ((s.split()[2]).strip("'"))
                if (paramfile_orig[0:2] == './'):
                   paramfile_orig = orig_dir+'/'+paramfile_orig[2:]
                paramfile_new  = ens_dir+'/clm_params_'+gst[1:]+'.nc'
                os.system('cp '+paramfile_orig+' '+paramfile_new)
                os.system('nccopy -3 '+paramfile_new+' '+paramfile_new+'_tmp')
                os.system('mv '+paramfile_new+'_tmp '+paramfile_new)
                myoutput.write(" paramfile = '"+paramfile_new+"'\n")
                pftfile = ens_dir+'/clm_params_'+gst[1:]+'.nc'
            elif ('ppmv' in s and 'co2' in self.ensemble_parms):
                myoutput.write(" co2_ppmv = "+str(parm_values[pnum_co2])+'\n')
            elif ('fsoilordercon' in s):
                CNPfile_orig = ((s.split()[2]).strip("'"))
                if (CNPfile_orig[0:2] == './'):
                   CNPfile_orig  = orig_dir+'/'+CNPfile_orig[2:]
                CNPfile_new  = ens_dir+'/CNP_parameters_'+gst[1:]+'.nc'
                os.system('cp '+CNPfile_orig+' '+CNPfile_new)
                os.system('nccopy -3 '+CNPfile_new+' '+CNPfile_new+'_tmp')
                os.system('mv '+CNPfile_new+'_tmp '+CNPfile_new)
                myoutput.write(" fsoilordercon = '"+CNPfile_new+"'\n")
                CNPfile = ens_dir+'/CNP_parameters_'+gst[1:]+'.nc'
            elif ('fsurdat =' in s):
                surffile_orig = ((s.split()[2]).strip("'"))
                if (surffile_orig[0:2] == './'):
                  surffile_orig = orig_dir+'/'+surffile_orig[2:]
                surffile_new = ens_dir+'/surfdata_'+gst[1:]+'.nc'
                os.system('cp '+surffile_orig+' '+surffile_new)
                os.system('nccopy -3 '+surffile_new+' '+surffile_new+'_tmp')
                os.system('mv '+surffile_new+'_tmp '+surffile_new)
                myoutput.write(" fsurdat = '"+surffile_new+"'\n")
                surffile = ens_dir+'/surfdata_'+gst[1:]+'.nc'
            elif ('finidat = ' in s and self.has_finidat):
                if (self.finidat_root != ''):
                   finidat_file_path = self.finidat_root+'/g'+gst[1:]
                   finidat_file_name = find_latest_restart(finidat_file_path).split('/')[-1]
                   if finidat_file_name is None:
                          print('No valid restart file found in '+finidat_file_path+'. Exiting.')
                          sys.exit(1)
                else:
                    finidat_file_path = os.path.abspath(self.rundir_UQ)+'/../'+self.dependcase+'/g'+gst[1:]
                    finidat_file_name = self.finidat.split('/')[-1]
                #finidat_file_orig = self.finidat
                finidat_file_new  = finidat_file_path+'/'+finidat_file_name 
                print(finidat_file_new)
                #if ('ad_spinup' in self.dependcase): 
                #        os.system('python adjust_restart.py --rundir '+finidat_file_path+' --casename '+ \
                #            self.dependcase)
                #os.system('cp '+finidat_file_orig+' '+finidat_file_new)
                myoutput.write(" finidat = '"+finidat_file_new+"'\n")
                #Make any requested restart modifications
                for key in self.case_options.keys():
                    if ('restart_' in key):
                        var   = key[8:]
                        value = self.case_options[key]
                        ncval = self.getncvar(finidat_file_new, var)
                        if ('*' in value):
                            value = value*ncval
                        if ('+' in value):
                            value = value+ncval
                        self.putncvar(finidat_file_new, var, value)
            elif ('logfile =' in s):
                #Get the current date and time
                now = datetime.datetime.now()
                #Format the date and time in %y%m%d-%H%M%S format
                date_string = now.strftime("%y%m%d-%H%M%S")
                myoutput.write(s.replace('`date +%y%m%d-%H%M%S`',date_string))
            else:
                myoutput.write(s.replace(orig_dir,ens_dir))
        myoutput.close()
        myinput.close()
        os.system(' mv '+ens_dir+'/'+f+'.tmp '+ens_dir+'/'+f)

  pnum = 0
  CNP_parms = ['ks_sorption', 'r_desorp', 'r_weather', 'r_adsorp', 'k_s1_biochem', 'smax', 'k_s3_biochem', \
             'r_occlude', 'k_s4_biochem', 'k_s2_biochem']

  fates_seed_zeroed=[False,False]
  pnum=0
  parm_values = self.samples[:,ens_num-1]
  parm_indices = self.ensemble_pfts
  for p in self.ensemble_parms:
    if ('INI' in p):
      if ('BGC' in self.casename):
         scalevars = ['soil3c_vr','soil3n_vr','soil3p_vr']
      else:
         scalevars = ['soil4c_vr','soil4n_vr','soil4p_vr']
      sumvars = ['totsomc','totsomp','totcolc','totcoln','totcolp']
      for v in scalevars:
         myvar = self.getncvar(finidat_file_new, v)
         myvar = parm_values[pnum] * myvar
         ierr = self.putncvar(finidat_file_new, v, myvar)
    elif (p == 'MONTHLY_LAI' or p == 'ORGANIC' or p == 'PCT_SAND' or p == 'PCT_CLAY' \
            or 'SECOND_P' in p or 'APATITE_P' in p or 'LABILE_P' in p or 'OCCLUDED_P' in p):
      myfile = surffile
      param = self.getncvar(myfile, p)
      param[:] = parm_values[pnum]
      ierr = self.putncvar(myfile, p, param)
    elif (p != 'co2'):
      if (p in CNP_parms):
         myfile= CNPfile
      elif ('fates' in p):
         myfile = fates_paramfile
      else:
         myfile = pftfile
      param = self.getncvar(myfile,p)
      if (('fates_prt' in p and 'stoich' in p) or ('fates_turnover' in p and 'retrans' in p)):
        #this is a 2D parameter.
         param[parm_indices[pnum] % 12 , parm_indices[pnum] / 12] = parm_values[pnum]
         param[parm_indices[pnum] % 12 , parm_indices[pnum] / 12] = parm_values[pnum]
      elif ('fates_hydr_p50_node' in p or 'fates_hydr_avuln_node' in p or 'fates_hydr_kmax_node' in p or \
            'fates_hydr_pitlp_node' in p or 'fates_hydr_thetas_node' in p):
         param[parm_indices[pnum] / 12 , parm_indices[pnum] % 12] = parm_values[pnum]
         param[parm_indices[pnum] / 12 , parm_indices[pnum] % 12] = parm_values[pnum]
      elif ('fates_leaf_long' in p or 'fates_leaf_vcmax25top' in p):
         param[0,parm_indices[pnum]] = parm_values[pnum]
      #elif (p == 'fates_seed_alloc'):
      #    if (not fates_seed_zeroed[0]):
      #       param[:]=0.
      #       fates_seed_zeroed[0]=True
      #    param[parm_indices[pnum]] = parm_values[pnum]
      #elif (p == 'fates_seed_alloc_mature'):
      #    if (not fates_seed_zeroed[1]):
      #       param[:]=0.
      #       fates_seed_zeroed[1]=True
      #    param[parm_indices[pnum]] = parm_values[pnum]             
      elif (p == 'dayl_scaling' or p == 'vcmaxse'):
        os.system('ncap2 -O -s "'+p+' = flnr" '+myfile+' '+myfile)
        print('Creting netcdf variable for '+p)
        param = self.getncvar(myfile,'flnr')
        param[:] = parm_values[pnum]
      elif (p == 'psi50'):
        param[:,parm_indices[pnum]] = parm_values[pnum]
      elif (parm_indices[pnum] > 0):
         param[parm_indices[pnum]] = parm_values[pnum]
      elif (parm_indices[pnum] == 0):
         try:
           param[:] = parm_values[pnum]
         except:
           param = parm_values[pnum]
      ierr = self.putncvar(myfile, p, param, addvar=True)
      #if ('fr_flig' in p):
      #   param=self.getncvar(myfile, 'fr_fcel')
      #   param[parm_indices[pnum]]=1.0-parm_values[pnum]-parm_values[pnum-1]
      #   ierr = self.putncvar(myfile, 'fr_fcel', param)
    pnum = pnum+1

  #ensure FATES seed allocation paramters sum to one
  #if (fates_seed_zeroed[0]):
  #  param = self.getncvar(myfile,'fates_seed_alloc')
  #  param2 = self.getncvar(myfile,'fates_seed_alloc_mature')
  #  for i in range(0,12):
  #    if (param[i] + param2[i] > 1.0):
  #      sumparam= param[i]+param2[i]
  #      param[i]  = param[i]/sumparam
  #      param2[i] = param2[i]/sumparam
  #  ierr = self.putncvar(myfile, 'fates_seed_alloc', param)      
  #  ierr = self.putncvar(myfile, 'fates_seed_alloc_mature', param2)

def plot_ensemble(self, myvar, percentiles=[1, 5, 25, 50, 75, 95, 99], factor=1):
    UQ_output = self.UQ_output + '/ensemble'
    os.makedirs(UQ_output, exist_ok=True)  # Ensures the directory exists
    """
    Plots percentiles (99th, 95th, 75th, 50th, 25th, 5th, and 1st) for ensemble data.
    
    Parameters:
        data (numpy.ndarray): 2D array of ensemble data (shape: [ensemble_size, num_time_steps]).
        x_axis (list or numpy.ndarray): x-axis values (e.g., time).
        output_file (str): Path to save the plot.
    """
    # Percentiles to calculate
    data=self.output[myvar].transpose()

    # Calculate percentiles along the ensemble axis
    percentile_values = np.nanpercentile(data, percentiles, axis=0)

    # Define line styles for each percentile
    # (Use the same style for matching pairs: 1/99, 5/95, 25/75, and make 50 bold)
    line_styles = {
        1:  {'linestyle': '--', 'color': 'black',  'linewidth': 2},
        99: {'linestyle': '--', 'color': 'black',  'linewidth': 2},
        5:  {'linestyle': '-.', 'color': 'black', 'linewidth': 2},
        95: {'linestyle': '-.', 'color': 'black', 'linewidth': 2},
        25: {'linestyle': ':',  'color': 'black','linewidth': 2},
        75: {'linestyle': ':',  'color': 'black','linewidth': 2},
        50: {'linestyle': '-',  'color': 'black',   'linewidth': 3},  # Bold line
    }

    # Choose x-axis: months if postproc_freq is monthly, else use taxis as is
    if getattr(self, 'postproc_freq', None) == 'monthly':
        months = np.arange(1, len(self.output['taxis']) + 1)
        x_axis = months
        x_label = "Month"
    else:
        x_axis = self.output['taxis']
        x_label = "Time Step"

    y_label = myvar
    # Plot each percentile
    plt.figure(figsize=(10, 6))
    for i, p in enumerate(percentiles):
        style = line_styles[p]
        plt.plot(x_axis,
                 percentile_values[i, :],
                 linestyle=style['linestyle'],
                 color=style['color'],
                 linewidth=style['linewidth'],
                 label=f'{p}th Percentile')

    # Add labels and legend
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Ensemble Percentiles")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(UQ_output+f'/{myvar}_percentiles.png', bbox_inches='tight')
    plt.close()


### END ###
