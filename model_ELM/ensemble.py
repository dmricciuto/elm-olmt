#!/usr/bin/env python

import os, sys, csv, time, math, shlex, re
import numpy as np
import datetime
import matplotlib.pyplot as plt
from netCDF4 import Dataset, chartostring
import xarray as xr
import json
import glob
import shutil
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


def retry_filesystem_op(description, func, retries=None, delay=None):
    """Retry transient filesystem operations on busy shared filesystems."""
    if retries is None:
        retries = int(os.environ.get('OLMT_FS_RETRIES', '5'))
    if delay is None:
        delay = float(os.environ.get('OLMT_FS_RETRY_DELAY', '2.0'))
    retries = max(1, int(retries))
    for attempt in range(1, retries + 1):
        try:
            return func()
        except (OSError, shutil.Error, MemoryError) as e:
            if attempt >= retries:
                print('Filesystem operation failed after '+str(retries)+' attempts: '+description, flush=True)
                raise
            wait = min(delay * attempt, 30.0)
            print(
                'Warning: filesystem operation failed on attempt '
                +str(attempt)+'/'+str(retries)+': '+description+'; '
                +'retrying in '+str(wait)+' s; error: '+repr(e),
                flush=True,
            )
            time.sleep(wait)


def require_directory(path):
    if not os.path.isdir(path):
        raise FileNotFoundError('Expected directory does not exist: '+path)
    return path


def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError('Expected file does not exist: '+path)
    return path


def copy_file_verified(src, dst):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    tmp = dst+'.tmp'

    def _copy():
        require_file(src)
        shutil.copy2(src, tmp)
        require_file(tmp)
        os.replace(tmp, dst)
        return require_file(dst)

    return retry_filesystem_op('copy '+src+' to '+dst, _copy)


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


def get_resubmit_segments(self, resubmit_years):
    """Return segment metadata for logical-case resubmission."""
    total_years = int(self.run_n)
    try:
        resubmit_years = int(resubmit_years or 0)
    except Exception:
        resubmit_years = 0

    if total_years <= 0 or resubmit_years <= 0 or resubmit_years >= total_years:
        return [{
            'index': 1,
            'count': 1,
            'start_offset': 0,
            'end_offset': total_years,
            'run_n': total_years,
            'continue_run': False,
            'final': True,
        }]

    segments = []
    start_offset = 0
    index = 1
    while start_offset < total_years:
        run_n = min(resubmit_years, total_years - start_offset)
        end_offset = start_offset + run_n
        segments.append({
            'index': index,
            'count': 0,
            'start_offset': start_offset,
            'end_offset': end_offset,
            'run_n': run_n,
            'continue_run': index > 1,
            'final': end_offset >= total_years,
        })
        start_offset = end_offset
        index = index + 1

    for segment in segments:
        segment['count'] = len(segments)
    return segments


def get_ensemble_segments(self):
    """Return segment metadata for ensemble resubmission."""
    return get_resubmit_segments(self, getattr(self, 'ensemble_resubmit_years', 0))


def get_multisite_segments(self):
    """Return segment metadata for multi-site resubmission."""
    return get_resubmit_segments(self, getattr(self, 'resubmit_years', 0))


def write_case_xmlchange(myfile, self, variable, value, casedir=None):
    if casedir is None:
        casedir = self.casedir
    cmd = './xmlchange '+variable+'='+str(value)
    if (self.apptainer != ''):
        myfile.write('apptainer exec --bind '+self.apptainer_bind+' --pwd '+casedir+' '+ \
            self.apptainer+' '+cmd+'\n')
    else:
        myfile.write(cmd+'\n')


def write_cime_pythonpath(myfile, self):
    cimeroot = os.path.abspath(self.modelroot)+'/cime'
    myfile.write('export PYTHONPATH='+cimeroot+':${PYTHONPATH:-}\n\n')


def write_cime_env_eval(myfile, self, casedir):
    """Load the CIME case environment in Python and export it to this shell."""
    cimeroot = os.path.abspath(self.modelroot)+'/cime'
    python_exe = shlex.quote(sys.executable)
    myfile.write('# setup environment through CIME Case.load_env, avoiding direct Lmod init/sh sourcing\n')
    myfile.write('eval "$('+python_exe+' - <<\'PYEOF\'\n')
    myfile.write('import contextlib, os, re, shlex, sys\n')
    myfile.write('sys.path.insert(0, '+repr(cimeroot)+')\n')
    myfile.write('from CIME.case import Case\n')
    myfile.write('valid = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")\n')
    myfile.write('before = dict(os.environ)\n')
    myfile.write('with contextlib.redirect_stdout(sys.stderr):\n')
    myfile.write('    with Case('+repr(casedir)+', read_only=False) as case:\n')
    myfile.write('        case.load_env(reset=True)\n')
    myfile.write('keys = sorted(set(before) | set(os.environ))\n')
    myfile.write('for key in keys:\n')
    myfile.write('    if not valid.match(key):\n')
    myfile.write('        continue\n')
    myfile.write('    if key not in os.environ:\n')
    myfile.write('        print("unset "+key)\n')
    myfile.write('    elif before.get(key) != os.environ[key]:\n')
    myfile.write('        print("export "+key+"="+shlex.quote(os.environ[key]))\n')
    myfile.write('PYEOF\n')
    myfile.write(')"\n\n')


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
    self.npernode=int(self.xmlquery('MAX_TASKS_PER_NODE'))
    nnodes = max(1, int(np.ceil((self.np_ensemble*self.np)/self.npernode)))
    case_run = os.path.join(self.casedir, '.case.run')
    parsed_sbatch = parse_sbatch(case_run)
    if (self.queue == 'debug'):
        print('Debug queue selected, setting walltime to 2 hours')
        self.walltime=2
    total_tasks = int(self.np_ensemble) * int(self.np)
    parsed_sbatch['qos'] = self.queue
    if getattr(self, 'partition', '') != '':
        parsed_sbatch['partition'] = self.partition
    parsed_sbatch['time'] = str(self.walltime)+':00:00'
    parsed_sbatch['nodes'] = int(nnodes)
    parsed_sbatch['ntasks'] = int(total_tasks)
    parsed_sbatch['ntasks_per_node'] = int(math.ceil(float(parsed_sbatch['ntasks'])/float(parsed_sbatch['nodes'])))
    # Ensemble jobs can run many E3SM instances at once; request whole nodes
    # so they are not colocated with unrelated jobs on the same node.
    parsed_sbatch['exclusive'] = True

    segments = get_ensemble_segments(self)
    scriptfiles = []
    for segment in segments:
        if len(segments) == 1:
            scriptname = 'case.submit_ensemble'
        else:
            scriptname = 'case.submit_ensemble.seg'+str(1000+segment['index'])[1:]
        myfile = open(scriptname,'w')
        myfile.write('#!/bin/bash -e\n\n')
        # Write SBATCH directives using helper
        write_sbatch(parsed_sbatch.copy(), myfile)

        myfile.write('cd '+self.caseroot+'/'+self.casename+'\n')
        write_cime_pythonpath(myfile, self)
        write_cime_env_eval(myfile, self, self.casedir)
        if len(segments) > 1:
            write_case_xmlchange(myfile, self, 'STOP_OPTION', 'nyears')
            write_case_xmlchange(myfile, self, 'STOP_N', segment['run_n'])
            write_case_xmlchange(myfile, self, 'REST_N', segment['run_n'])
            write_case_xmlchange(myfile, self, 'CONTINUE_RUN', 'TRUE' if segment['continue_run'] else 'FALSE')
        if (self.apptainer != ''):
          myfile.write('apptainer exec --bind '+self.apptainer_bind+' '+ \
                    ' --pwd '+self.caseroot+'/'+self.casename+ ' '+ \
                    self.apptainer+' ./preview_namelists\n')
        else:
          myfile.write('./preview_namelists\n')
        myfile.write('ulimit -n '+str(self.nsamples+1024)+'\n')
        myfile.write('cd '+self.OLMTdir+'\n')
        manage_cmd = shlex.quote(sys.executable)+' ./manage_ensemble.py --case '+self.casename
        if len(segments) > 1:
            segment_end_year = int(self.startyear) + int(segment['end_offset'])
            manage_cmd += ' --segment '+str(segment['index'])
            manage_cmd += ' --segment_years '+str(segment['run_n'])
            manage_cmd += ' --segment_end_year '+str(segment_end_year)
            if segment['continue_run']:
                manage_cmd += ' --continue_segment'
            if not segment['final']:
                manage_cmd += ' --no_final_segment'
        myfile.write(manage_cmd+'\n')
        myfile.close()
        os.system('chmod u+x '+scriptname)
        scriptfiles.append('./'+scriptname)

    self.rundir_UQ = self.runroot+'/UQ/ensembles/'+self.casename
    os.system('mkdir -p '+self.rundir_UQ)
    self.UQ_output = self.runroot+'/UQ/analysis/'+self.casename
    os.system('mkdir -p '+self.UQ_output)
    return scriptfiles

def create_multisite_script(self,sites,scriptdir,cases_compare=""):
    #Create the PBS script we will submit to run multiple sites
    os.chdir(self.casedir)
    self.npernode=int(self.xmlquery('MAX_TASKS_PER_NODE'))
    nruns = len(sites) if len(sites) > 0 else 1
    total_tasks = int(self.np) * int(nruns)
    nnodes = max(1, int(np.ceil(float(total_tasks) / float(self.npernode))))
    case_run = os.path.join(self.casedir, '.case.run')
    parsed_sbatch = parse_sbatch(case_run)
    if (self.queue == 'debug'):
        print('Debug queue selected, setting walltime to 2 hours')
        self.walltime=2
    parsed_sbatch['qos'] = self.queue
    if getattr(self, 'partition', '') != '':
        parsed_sbatch['partition'] = self.partition
    parsed_sbatch['time'] = str(self.walltime)+':00:00'
    parsed_sbatch['nodes'] = int(nnodes)
    parsed_sbatch['ntasks'] = int(total_tasks)
    parsed_sbatch['ntasks_per_node'] = int(math.ceil(float(total_tasks)/float(nnodes)))
    # Ensure we do not emit --exclusive for multisite submission either.
    parsed_sbatch['exclusive'] = False

    segments = get_multisite_segments(self)
    base_fname = self.casename.replace('_'+self.site,'')+'.sh'
    scriptfiles = []
    for segment in segments:
      if len(segments) == 1:
        fname = base_fname
      else:
        fname = base_fname[:-3]+'.seg'+str(1000+segment['index'])[1:]+'.sh'
      myfile = open(fname,'w')
      myfile.write('#!/bin/bash -e\n\n')
      # Write SBATCH directives using helper
      write_sbatch(parsed_sbatch.copy(), myfile)

      myfile.write('cd '+self.caseroot+'/'+self.casename+'\n')
      write_cime_pythonpath(myfile, self)
      for s in sites:
        site_casename = self.casename.replace(sites[0],s)
        site_casedir = self.caseroot+'/'+site_casename
        site_rundir = self.runroot+'/'+site_casename+'/run'
        myfile.write('cd '+site_casedir+'\n')
        write_cime_env_eval(myfile, self, site_casedir)
        if len(segments) > 1:
          write_case_xmlchange(myfile, self, 'STOP_OPTION', 'nyears', casedir=site_casedir)
          write_case_xmlchange(myfile, self, 'STOP_N', segment['run_n'], casedir=site_casedir)
          write_case_xmlchange(myfile, self, 'REST_N', segment['run_n'], casedir=site_casedir)
          write_case_xmlchange(myfile, self, 'CONTINUE_RUN', 'TRUE' if segment['continue_run'] else 'FALSE', casedir=site_casedir)
        if (self.apptainer != ''):
          myfile.write('apptainer exec --bind '+self.apptainer_bind+' '+ \
                  ' --pwd '+site_casedir+ ' '+ \
                  self.apptainer+' ./preview_namelists\n')
        elif len(segments) > 1:
          myfile.write('./preview_namelists\n')
        myfile.write('cd '+site_rundir+'\n')
        myfile.write('mkdir -p timing/checkpoints\n')

        #restart file options only apply to the initial segment
        if not segment['continue_run']:
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
        exe_name = 'elm_offline_driver' if getattr(self, 'offline_driver', False) else 'e3sm.exe'
        exe_path = self.exeroot+'/'+exe_name
        log_path = site_rundir+'/e3sm_log.txt'
        if len(segments) > 1:
          log_path = site_rundir+'/e3sm_log.seg'+str(1000+segment['index'])[1:]+'.txt'
        if (self.noslurm):
          myfile.write('mpiexec -n '+str(self.np)+' '+exe_path+' > '+ \
                       log_path+' &\n\n')
        else:
            if (self.apptainer != ''):
                # Preserve explicit container execution path for multi-site runs.
                nodes_per_run = max(1, int(np.ceil(float(self.np) / float(self.npernode))))
                ntasks_run = int(self.np)
                tasks_per_node_run = max(1, int(math.ceil(float(ntasks_run) / float(nodes_per_run))))
                cpus_per_task = parsed_sbatch['cpus_per_task'] if parsed_sbatch.get('cpus_per_task') is not None else 1
                myfile.write('srun --nodes='+str(nodes_per_run)+' --ntasks='+str(ntasks_run)+' --ntasks-per-node='+str(tasks_per_node_run)+' --cpu-bind=none -c '+str(cpus_per_task)+' apptainer exec '+ \
                    ' --bind '+self.apptainer_bind+' --env OMPI_MCA_pml=ob1 --env OMPI_MCA_btl=self,vader,tcp  ' \
                    +self.apptainer+' '+exe_path+' > '+ \
                    log_path+' &\n\n')
            else:
                direct_cmd = site_casedir+'/.case.run > '+log_path+' 2>&1'
                sbatch_cmd = 'sbatch --wait'
                if parsed_sbatch.get('time'):
                    sbatch_cmd += ' --time='+str(parsed_sbatch.get('time'))
                if parsed_sbatch.get('partition'):
                    sbatch_cmd += ' -p '+str(parsed_sbatch.get('partition'))
                if parsed_sbatch.get('qos'):
                    sbatch_cmd += ' --qos='+str(parsed_sbatch.get('qos'))
                if parsed_sbatch.get('account'):
                    sbatch_cmd += ' -A '+str(parsed_sbatch.get('account'))
                sbatch_cmd += ' --output='+log_path
                sbatch_cmd += ' --error='+log_path
                sbatch_cmd += ' '+site_casedir+'/.case.run'
                myfile.write('if [ -n "${SLURM_JOB_ID:-}" ]; then\n')
                myfile.write('  '+direct_cmd+' &\n')
                myfile.write('else\n')
                myfile.write('  '+sbatch_cmd+' &\n')
                myfile.write('fi\n\n')
      myfile.write('wait\n')
      if segment['final']:
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
      scriptfiles.append(os.path.abspath('./'+fname))
    return scriptfiles

def ensemble_copy(self, ens_num, clean=True):

  gst=str(100000+int(ens_num))

  # create ensemble directory from original case 
  orig_dir = str(os.path.abspath(self.runroot)+'/'+self.casename+'/run')
  ens_dir  = str(os.path.abspath(self.rundir_UQ)+'/g'+gst[1:])

  if not os.path.isdir(orig_dir):
    raise FileNotFoundError('Original run directory not found for ensemble copy: '+orig_dir)
  retry_filesystem_op(
      'create ensemble directory '+ens_dir,
      lambda: os.makedirs(ens_dir+'/timing/checkpoints', exist_ok=True),
  )
  retry_filesystem_op(
      'verify ensemble directory '+ens_dir,
      lambda: require_directory(ens_dir),
  )
  if clean:
    for pattern in ('*.log.*', '*.nc', '*.tmp', 'rpointer*'):
      for path in glob.glob(os.path.join(ens_dir, pattern)):
        if os.path.isfile(path) or os.path.islink(path):
          retry_filesystem_op('remove '+path, lambda path=path: os.remove(path))
        elif os.path.isdir(path):
          retry_filesystem_op('remove directory '+path, lambda path=path: shutil.rmtree(path))

  copy_patterns = ['*_in*', '*nml', '*.rc', 'surf*.nc', 'domain*.nc', '*para*.nc']
  if (not ('CB' in self.casename)):
    copy_patterns.insert(2, '*stream*')
  for pattern in copy_patterns:
    matches = glob.glob(os.path.join(orig_dir, pattern))
    if not matches:
      print('Warning: ensemble_copy found no files for '+os.path.join(orig_dir, pattern))
      continue
    for src in matches:
      if os.path.isfile(src):
        copy_file_verified(src, os.path.join(ens_dir, os.path.basename(src)))


  # loop through all filenames, change directories in namelists, change parameter values
  ensemble_files = retry_filesystem_op(
      'list ensemble directory '+ens_dir,
      lambda: os.listdir(ens_dir),
  )
  for f in ensemble_files:
    if (os.path.isfile(ens_dir+'/'+f) and (f[-2:] == 'in' or f[-3:] == 'nml' or 'streams' in f)):
        myinput = retry_filesystem_op(
            'open '+ens_dir+'/'+f,
            lambda f=f: open(ens_dir+'/'+f),
        )
        myoutput = retry_filesystem_op(
            'open '+ens_dir+'/'+f+'.tmp',
            lambda f=f: open(ens_dir+'/'+f+'.tmp','w'),
        )
        for s in myinput:
            if ('fates_paramfile' in s):
                paramfile_orig = ((s.split()[2]).strip("'"))
                if (paramfile_orig[0:2] == './'):
                  paramfile_orig = orig_dir+'/'+paramfile_orig[2:]
                paramfile_new  = ens_dir+'/fates_params_'+gst[1:]+'.nc'
                copy_file_verified(paramfile_orig, paramfile_new)
                myoutput.write(" fates_paramfile = '"+paramfile_new+"'\n")
                fates_paramfile = ens_dir+'/fates_params_'+gst[1:]+'.nc'
            elif ('paramfile' in s):
                paramfile_orig = ((s.split()[2]).strip("'"))
                if (paramfile_orig[0:2] == './'):
                   paramfile_orig = orig_dir+'/'+paramfile_orig[2:]
                paramfile_new  = ens_dir+'/clm_params_'+gst[1:]+'.nc'
                copy_file_verified(paramfile_orig, paramfile_new)
                myoutput.write(" paramfile = '"+paramfile_new+"'\n")
                pftfile = ens_dir+'/clm_params_'+gst[1:]+'.nc'
            elif ('ppmv' in s and 'co2' in self.ensemble_parms):
                myoutput.write(" co2_ppmv = "+str(parm_values[pnum_co2])+'\n')
            elif ('fsoilordercon' in s):
                CNPfile_orig = ((s.split()[2]).strip("'"))
                if (CNPfile_orig[0:2] == './'):
                   CNPfile_orig  = orig_dir+'/'+CNPfile_orig[2:]
                CNPfile_new  = ens_dir+'/CNP_parameters_'+gst[1:]+'.nc'
                copy_file_verified(CNPfile_orig, CNPfile_new)
                myoutput.write(" fsoilordercon = '"+CNPfile_new+"'\n")
                CNPfile = ens_dir+'/CNP_parameters_'+gst[1:]+'.nc'
            elif ('fsurdat =' in s):
                surffile_orig = ((s.split()[2]).strip("'"))
                if (surffile_orig[0:2] == './'):
                  surffile_orig = orig_dir+'/'+surffile_orig[2:]
                surffile_new = ens_dir+'/surfdata_'+gst[1:]+'.nc'
                copy_file_verified(surffile_orig, surffile_new)
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
                #Make any requested restart modifications on the initial segment only.
                if clean:
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
        retry_filesystem_op(
            'replace '+ens_dir+'/'+f,
            lambda f=f: os.replace(ens_dir+'/'+f+'.tmp', ens_dir+'/'+f),
        )

  pnum = 0
  CNP_parms = ['ks_sorption', 'r_desorp', 'r_weather', 'r_adsorp', 'k_s1_biochem', 'smax', 'k_s3_biochem', \
             'r_occlude', 'k_s4_biochem', 'k_s2_biochem']

  fates_seed_zeroed=[False,False]
  pnum=0
  parm_values = self.samples[:,ens_num-1]
  parm_indices = self.ensemble_pfts
  for p in self.ensemble_parms:
    if ('INI' in p):
      if not clean:
         pnum = pnum+1
         continue
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
        print('Creating netcdf variable for '+p)
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

def _netcdf_var_name(name, used_names):
    """Return a NetCDF-safe variable name while preserving a collision-free map."""
    safe = re.sub(r'[^A-Za-z0-9_]', '_', str(name))
    if safe == '' or safe[0].isdigit():
        safe = 'v_'+safe
    candidate = safe
    n = 2
    while candidate in used_names:
        candidate = safe+'_'+str(n)
        n = n+1
    used_names.add(candidate)
    return candidate

def _indexed_postproc_metadata(var_name):
    match = re.match(r'^(.*_(pft|col))(-?\d+)$', str(var_name))
    if not match:
        return None
    return {
        'source_variable': match.group(1),
        'index_type': match.group(2),
        'index': int(match.group(3)),
    }

def _decode_netcdf_string_array(values):
    arr = np.asarray(values)
    try:
        decoded = chartostring(arr)
        return [str(s).strip() for s in np.asarray(decoded).tolist()]
    except Exception:
        pass
    if arr.dtype.kind == 'S':
        if arr.ndim == 1:
            return [x.decode('utf-8', errors='ignore').strip() for x in arr]
        return [b''.join(row).decode('utf-8', errors='ignore').strip() for row in arr]
    if arr.dtype.kind == 'U':
        if arr.ndim == 1:
            return [str(x).strip() for x in arr]
        return [''.join(row).strip() for row in arr]
    return []

def _decode_pft_names(values):
    names = _decode_netcdf_string_array(values)
    if names:
        return names
    arr = np.asarray(values)
    if arr.ndim == 0:
        return [str(arr.item()).strip()]
    if arr.ndim == 1:
        return [str(v).strip() for v in arr.tolist()]
    return [''.join([str(v) for v in row]).strip() for row in arr.tolist()]

def _read_pft_names_from_file(path, variable_names):
    try:
        if str(path).lower().endswith('.json'):
            with open(path, 'r') as f:
                data = json.load(f)
            parameters = data.get('parameters', data)
            for variable_name in variable_names:
                if variable_name in parameters:
                    raw = parameters[variable_name]
                    if isinstance(raw, dict) and 'data' in raw:
                        raw = raw['data']
                    names = _decode_pft_names(raw)
                    return {i: name for i, name in enumerate(names) if name != ''}
            return {}
        with Dataset(path, 'r') as ds:
            for variable_name in variable_names:
                if variable_name in ds.variables:
                    names = _decode_pft_names(ds.variables[variable_name][:])
                    return {i: name for i, name in enumerate(names) if name != ''}
            return {}
    except Exception as e:
        print('Warning: unable to read PFT names from '+str(path)+': '+str(e))
        return {}

def _is_fates_active(self):
    compset = str(getattr(self, 'compset', ''))
    return 'FATES' in compset or 'ED' in compset

def _add_unique_path(paths, path):
    if path and path not in paths:
        paths.append(path)

def _ensemble_member_dirs(self):
    if not hasattr(self, 'rundir_UQ') or not os.path.isdir(self.rundir_UQ):
        return []
    member_dirs = [
        path for path in glob.glob(os.path.join(self.rundir_UQ, 'g[0-9][0-9][0-9][0-9][0-9]'))
        if os.path.isdir(path)
    ]
    return sorted(member_dirs)

def _find_ensemble_param_pft_names(self):
    """Read PFT names from the parameter file copied into an ensemble member dir."""
    if hasattr(self, 'rundir_UQ'):
        preferred_member = os.path.join(self.rundir_UQ, 'g00001')
        member_dirs = [preferred_member] + [d for d in _ensemble_member_dirs(self) if d != preferred_member]
    else:
        member_dirs = []

    for member_dir in member_dirs:
        candidate_files = []
        if _is_fates_active(self):
            for path in sorted(glob.glob(os.path.join(member_dir, 'fates_params_*'))):
                _add_unique_path(candidate_files, path)
            for suffix in ('nc', 'json'):
                _add_unique_path(candidate_files, os.path.join(member_dir, 'fates_params_00001.'+suffix))
        else:
            for path in sorted(glob.glob(os.path.join(member_dir, 'clm_params_*.nc'))):
                _add_unique_path(candidate_files, path)
            _add_unique_path(candidate_files, os.path.join(member_dir, 'clm_params_00001.nc'))

        for path in candidate_files:
            if path and os.path.isfile(path):
                variable_names = ['fates_pftname', 'pftname'] if _is_fates_active(self) else ['pftname']
                pft_names = _read_pft_names_from_file(path, variable_names)
                if pft_names:
                    return pft_names, path
    return {}, ''

def _find_history_pft_type_map(self):
    """Map history-file PFT slot index to parameter-file PFT index."""
    if not hasattr(self, 'casename'):
        return {}, ''
    for member_dir in _ensemble_member_dirs(self):
        history_files = sorted(glob.glob(os.path.join(member_dir, self.casename+'.elm.h2.*.nc')))
        for path in history_files:
            try:
                with Dataset(path, 'r') as ds:
                    if 'pfts1d_itype_veg' not in ds.variables:
                        continue
                    values = np.asarray(ds.variables['pfts1d_itype_veg'][:]).squeeze()
                    if values.ndim == 0:
                        values = np.asarray([values.item()])
                    return {i: int(values[i]) for i in range(len(values))}, path
            except Exception as e:
                print('Warning: unable to read PFT coordinate metadata from '+str(path)+': '+str(e))
    return {}, ''

def _find_pft_metadata(self):
    param_pft_names, param_path = _find_ensemble_param_pft_names(self)
    if not param_pft_names:
        return {}
    history_to_param, history_path = _find_history_pft_type_map(self)
    if not history_to_param:
        print('Warning: unable to map history PFT indices to parameter PFT names; missing pfts1d_itype_veg')
        return {}

    pft_metadata = {}
    for history_index, param_index in history_to_param.items():
        if param_index in param_pft_names:
            pft_metadata[history_index] = {
                'pft_name': param_pft_names[param_index],
                'parameter_pft_index': param_index,
                'parameter_file': param_path,
                'history_coordinate_file': history_path,
            }
    return pft_metadata

def write_postprocessed_netcdf(self, filename=''):
    """Write ensemble postprocessed output to one NetCDF file."""
    if not hasattr(self, 'output') or not self.output or 'taxis' not in self.output:
        print('No postprocessed output available; skipping NetCDF output')
        return ''
    if filename == '':
        filename = os.path.join(self.rundir_UQ, 'postprocessed_output.nc')
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    taxis = np.asarray(self.output['taxis'], dtype=float)
    nsamples = int(getattr(self, 'nsamples', 0))
    if nsamples <= 0:
        for key, values in self.output.items():
            if key == 'taxis':
                continue
            arr = np.asarray(values)
            if arr.ndim == 2:
                nsamples = arr.shape[1]
                break
    if nsamples <= 0:
        nsamples = 1
    pft_metadata = _find_pft_metadata(self)

    with Dataset(filename, 'w') as ds:
        ds.createDimension('time', len(taxis))
        ds.createDimension('ensemble', nsamples)
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var[:] = taxis
        time_var.long_name = 'postprocessed time axis'
        time_var.units = 'model year'
        member_var = ds.createVariable('ensemble_member', 'i4', ('ensemble',))
        member_var[:] = np.arange(1, nsamples+1)
        member_var.long_name = 'ensemble member number'

        if hasattr(self, 'postprocessed'):
            status = ds.createVariable('postprocessed', 'i4', ('ensemble',))
            status[:] = np.asarray(self.postprocessed, dtype=int)
            status.long_name = 'postprocessing status by ensemble member'
            status.flag_values = '-1, 0, 1'
            status.flag_meanings = 'failed not_processed success'

        ds.title = 'OLMT ensemble postprocessed output'
        ds.case = getattr(self, 'casename', '')
        ds.postprocessing_frequency = getattr(self, 'postproc_freq', '')
        ds.postprocessing_startyear = int(getattr(self, 'postproc_startyear', -1))
        ds.postprocessing_endyear = int(getattr(self, 'postproc_endyear', -1))
        if hasattr(self, 'postproc_vars'):
            ds.postprocessing_variables = ','.join([str(v) for v in self.postproc_vars])
        if hasattr(self, 'postproc_pfts'):
            ds.postprocessing_pfts = ','.join([str(v) for v in self.postproc_pfts])
        if hasattr(self, 'postproc_cols'):
            ds.postprocessing_cols = ','.join([str(v) for v in self.postproc_cols])
        if pft_metadata:
            first_pft_meta = next(iter(pft_metadata.values()))
            ds.pft_name_parameter_file = first_pft_meta['parameter_file']
            ds.pft_history_coordinate_file = first_pft_meta['history_coordinate_file']
            ds.pft_index_mapping = 'PFT variable index attributes are history-file pft indices; parameter_pft_index maps to the parameter-file pftname entry.'

        used_names = set(ds.variables.keys())
        for key in sorted(self.output.keys()):
            if key == 'taxis':
                continue
            data = np.ma.asarray(self.output[key], dtype=float)
            dims = ('time', 'ensemble')
            if data.ndim == 1:
                if len(data) == len(taxis):
                    dims = ('time',)
                elif len(data) == nsamples:
                    dims = ('ensemble',)
                else:
                    dim_name = _netcdf_var_name(key+'_dim', used_names)
                    ds.createDimension(dim_name, len(data))
                    dims = (dim_name,)
            elif data.ndim == 2:
                if data.shape == (len(taxis), nsamples):
                    dims = ('time', 'ensemble')
                elif data.shape == (nsamples, len(taxis)):
                    data = data.T
                    dims = ('time', 'ensemble')
                else:
                    dim0 = _netcdf_var_name(key+'_dim0', used_names)
                    dim1 = _netcdf_var_name(key+'_dim1', used_names)
                    ds.createDimension(dim0, data.shape[0])
                    ds.createDimension(dim1, data.shape[1])
                    dims = (dim0, dim1)
            else:
                print('Skipping NetCDF output for '+str(key)+' with unsupported shape '+str(data.shape))
                continue

            nc_name = _netcdf_var_name(key, used_names)
            out_var = ds.createVariable(nc_name, 'f8', dims, fill_value=-9999.0)
            out_var[:] = data
            out_var.original_name = str(key)
            indexed_meta = _indexed_postproc_metadata(key)
            if indexed_meta is not None:
                out_var.source_variable = indexed_meta['source_variable']
                out_var.index_type = indexed_meta['index_type']
                out_var.index = indexed_meta['index']
                if indexed_meta['index_type'] == 'pft' and indexed_meta['index'] in pft_metadata:
                    out_var.pft_name = pft_metadata[indexed_meta['index']]['pft_name']
                    out_var.parameter_pft_index = pft_metadata[indexed_meta['index']]['parameter_pft_index']

    print('Wrote postprocessed NetCDF output: '+filename)
    return filename

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

    # Mask failed members (sentinel -9999) before computing percentiles
    data = data.astype(float).copy()
    data[data <= -9999] = np.nan
    # Drop columns (members) that are entirely NaN
    valid_mask = ~np.all(np.isnan(data), axis=1)
    data = data[valid_mask, :]
    n_valid = int(valid_mask.sum())
    n_total = valid_mask.shape[0]
    if n_valid < n_total:
        print(f'plot_ensemble {myvar}: using {n_valid} of {n_total} members (excluding failed)')

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

    # --- Density-shaded version (no legend) ---
    # Build a lookup from percentile value -> index in percentile_values
    pct_idx = {p: i for i, p in enumerate(percentiles)}
    # Nested shading bands from outermost to innermost
    bands = []
    if 1  in pct_idx and 99 in pct_idx: bands.append((1,  99,  0.15))
    if 5  in pct_idx and 95 in pct_idx: bands.append((5,  95,  0.20))
    if 25 in pct_idx and 75 in pct_idx: bands.append((25, 75,  0.30))

    fig, ax = plt.subplots(figsize=(10, 6))
    for lo, hi, alpha in bands:
        ax.fill_between(x_axis,
                        percentile_values[pct_idx[lo], :],
                        percentile_values[pct_idx[hi], :],
                        color='steelblue', alpha=alpha)
    if 50 in pct_idx:
        ax.plot(x_axis, percentile_values[pct_idx[50], :],
                color='navy', linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'Ensemble Spread: {myvar}')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(UQ_output+f'/{myvar}_percentiles_shaded.png', bbox_inches='tight')
    plt.close(fig)


### END ###
