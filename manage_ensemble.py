#!/usr/bin/env python
import sys,os, time, math
import numpy as np
import subprocess
import random
import pickle
import shlex
import model_ELM
from optparse import OptionParser

#Python code used to manage the ensemble simulations 
#  and perform post-processing of model output.

parser = OptionParser()


parser.add_option("--case", dest="case", default="", \
                  help="Case name")
parser.add_option("--postproc_only", dest="postproc_only", default=False, \
                  action="store_true")
parser.add_option("--UQ_only", dest="UQ_only", default=False, \
                  action="store_true")
parser.add_option("--segment", dest="segment", default=1, type="int", \
                  help="Ensemble segment number")
parser.add_option("--segment_years", dest="segment_years", default=0, type="int", \
                  help="Years to run in this ensemble segment")
parser.add_option("--segment_end_year", dest="segment_end_year", default=0, type="int", \
                  help="Expected restart year at the end of this segment")
parser.add_option("--continue_segment", dest="continue_segment", default=False, \
                  action="store_true", help="Continue from existing member restart files")
parser.add_option("--no_final_segment", dest="final_segment", default=True, \
                  action="store_false", help="Skip postprocessing and UQ after this segment")
(options, args) = parser.parse_args()

#Load case object
myfile=open('pklfiles/'+options.case+'.pkl','rb')
mycase=pickle.load(myfile)
# default placement: enable pack_single_tasks only for single-task cases unless overridden
if not hasattr(mycase, 'pack_single_tasks'):
  mycase.pack_single_tasks = (getattr(mycase, 'np', 1) == 1)
if not hasattr(mycase, 'rotate_nodes'):
  mycase.rotate_nodes = True

is_final_segment = options.final_segment
is_continue_segment = options.continue_segment or options.segment > 1
defer_postprocess = is_final_segment and mycase.postproc_vars != [] and not options.postproc_only
run_completed = None

def expected_restart_year():
    if options.segment_end_year > 0:
        return options.segment_end_year
    return mycase.startyear+mycase.run_n

#get the node file and parse
def get_nodelist():
  # Prefer Slurm's canonical hostnames to avoid -w mismatches
  try:
    out = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ.get('SLURM_JOB_NODELIST', '')])
    hosts = [h.strip() for h in out.decode().splitlines() if h.strip()]
    if hosts:
      return hosts
  except Exception:
    pass
  # Fallback: try simple parsing of SLURM_JOB_NODELIST
  mynodes=[]
  nodelist = os.environ.get('SLURM_JOB_NODELIST','')
  if not nodelist:
    return mynodes
  # handle formats like node[01-04,07]
  if '[' in nodelist:
    prefix = nodelist.split('[')[0]
    inside = nodelist.split('[')[1].rstrip(']')
    parts = inside.split(',')
    for p in parts:
      if '-' in p:
        a,b = p.split('-')
        for i in range(int(a), int(b)+1):
          mynodes.append(prefix + str(i))
      else:
        mynodes.append(prefix + p)
  else:
    mynodes = nodelist.split(',')
  return mynodes



def check_run_success(n):
    success=False
    jobst = str(100000+n)
    rundir = mycase.rundir_UQ + '/g'+jobst[1:]
    yst = str(10000+expected_restart_year())[1:]
    #yst = '2010'
    if (os.path.isfile(rundir+'/'+mycase.casename+'.elm.r.'+yst+'-01-01-00000.nc')):
        success=True
    return success

def command_to_argv(command):
    """Split a launch command for shell-free subprocess execution."""
    return shlex.split(command)

def env_size_bytes(env):
    return sum(len(k) + len(v) + 2 for k, v in env.items())

def retry_ensemble_copy(member, clean):
    retries = max(1, int(os.environ.get('OLMT_ENSEMBLE_COPY_RETRIES', os.environ.get('OLMT_FS_RETRIES', '5'))))
    delay = float(os.environ.get('OLMT_ENSEMBLE_COPY_RETRY_DELAY', os.environ.get('OLMT_FS_RETRY_DELAY', '2.0')))
    for attempt in range(1, retries + 1):
        try:
            return mycase.ensemble_copy(member, clean=clean)
        except (OSError, MemoryError) as e:
            if attempt >= retries:
                print('ensemble_copy failed after '+str(retries)+' attempts for member '+str(member))
                raise
            wait = min(delay * attempt, 30.0)
            print(
                'Warning: ensemble_copy failed on attempt '
                +str(attempt)+'/'+str(retries)+' for member '+str(member)
                +'; retrying in '+str(wait)+' s; error: '+repr(e),
                flush=True,
            )
            time.sleep(wait)

def active_processes(processes,process_jobnum,process_hang):
    """Returns the number of processes that are still running."""
    pactive=[]
    n=0
    for process in processes:
        if process.poll() is None:  # None means the process is still running
            #Check if final restart file created
            pactive.append(1)
            if (check_run_success(process_jobnum[n])):
                process_hang[n] = process_hang[n]+1
            if (process_hang[n] > 30):
                process.kill()  # Force kill the process
        else:
            pactive.append(0)
            try:
                process.wait(timeout=0)
            except Exception:
                pass
            #Post-process ensemble member if it hasn't yet been done
            member = process_jobnum[n]
            member_idx = member - 1
            if (run_completed is not None and run_completed[member_idx] != 0):
                n=n+1
                continue
            if (mycase.postprocessed[member_idx] == 0):
                print(member, check_run_success(member))
                if (check_run_success(member)):
                    if defer_postprocess:
                        run_completed[member_idx] = 1
                        n=n+1
                        continue
                    if not is_final_segment:
                        if run_completed is not None:
                            run_completed[member_idx] = 1
                        mycase.postprocessed[member_idx] = 1
                        n=n+1
                        continue
                    ierr = postprocess_ensemble(process_jobnum[n])
                    if (ierr != 0):
                        print('Postprocessing failed for ensemble member ' \
                                +str(process_jobnum[n])+', skipping')
                        if run_completed is not None:
                            run_completed[member_idx] = -1
                        mycase.postprocessed[member_idx] = -1
                    else:
                        if run_completed is not None:
                            run_completed[member_idx] = 1
                        mycase.postprocessed[member_idx] = 1
                else:
                    print('Ensemble member '+str(process_jobnum[n])+ \
                            ' failed to complete, skipping')
                    if run_completed is not None:
                        run_completed[member_idx] = -1
                    mycase.postprocessed[member_idx] = -1
        n=n+1
    return pactive

def postprocess_ensemble(n):
  #Postprocess
  if (mycase.postproc_vars != []):
      try:
        if hasattr(mycase, 'postprocess_member'):
          mycase.postprocess_member(ens_num=n, startyear=mycase.postproc_startyear,
                endyear=mycase.postproc_endyear)
          return 0
      except Exception as e:
        print('Postprocessing failed for ensemble member '+str(n)+': '+str(e))
        return 1
      for v in mycase.postproc_vars:
        hnum=1  #default is h1 file (usually daily output with gridcell average)
        if (mycase.postproc_freq == 'annual'):
          hnum=0  #annual output is usually in h0 file (e.g. in spinup)
        mypfts=[0]
        if ('_pft' in v):
            #PFT level outputs requested, usually in h2
            hnum=2
            mypfts=mycase.postproc_pfts
        elif ('_col' in v):
            # Column level outputs requested, usually in h2
            hnum=2
            mypfts=mycase.postproc_cols
        for p in mypfts:
          try:
            if (mycase.postproc_freq == 'daily' or mycase.postproc_freq == 'hourly'):  #default
              mycase.postprocess(v, ens_num=n,startyear=mycase.postproc_startyear, \
                    endyear=mycase.postproc_endyear,index=p,hnum=hnum)
            elif (mycase.postproc_freq == 'monthly'):  #monthly
              mycase.postprocess(v, ens_num=n,startyear=mycase.postproc_startyear, \
                    endyear=mycase.postproc_endyear,index=p,hnum=hnum, dailytomonthly=True)
            elif (mycase.postproc_freq == 'annual'):  #annual
              mycase.postprocess(v, ens_num=n,startyear=mycase.postproc_startyear, \
                    endyear=mycase.postproc_endyear,index=p,hnum=hnum, annualmean=True)
          except Exception as e:
            print('Postprocessing failed for ensemble member '+str(n)+', variable '+v+': '+str(e))
            return 1
  return 0

def _postproc_worker(n):
  """Worker for parallel postprocessing in postproc_only mode.
  Results are written to temp .npy files to avoid large IPC pipe transfers."""
  import tempfile, numpy as np_w
  ierr = postprocess_ensemble(n)
  if ierr != 0:
      return n, ierr, None
  # Write each variable slice to a temp file; return paths to main process
  tmpdir = tempfile.gettempdir()
  saved = {}
  for var_out, data in mycase.output.items():
      fpath = os.path.join(tmpdir, f'olmt_pp_{n}_{var_out}.npy')
      if var_out == 'taxis':
          np_w.save(fpath, data)
      else:
          np_w.save(fpath, data[:, n-1])
      saved[var_out] = fpath
  return n, 0, saved

def postprocess_members(member_numbers):
    if len(member_numbers) == 0:
        return
    import multiprocessing
    n_parallel = int(os.environ.get('OLMT_POSTPROC_WORKERS', int(mycase.np_ensemble)))
    n_parallel = max(1, min(n_parallel, len(member_numbers)))
    print('Postprocessing '+str(len(member_numbers))+' ensemble members with ' \
          +str(n_parallel)+' parallel workers')
    ctx = multiprocessing.get_context('fork')
    with ctx.Pool(processes=n_parallel) as pool:
      all_results = pool.map(_postproc_worker, member_numbers)
    for n, ierr, saved in all_results:
      if ierr != 0 or saved is None:
          print('Postprocessing failed for ensemble member '+str(n)+', skipping')
          mycase.postprocessed[n-1] = -1
          continue
      for var_out, fpath in saved.items():
          try:
              values = np.load(fpath)
              os.remove(fpath)
          except Exception as e:
              print('Failed to load temp file for member '+str(n)+', var '+var_out+': '+str(e))
              mycase.postprocessed[n-1] = -1
              break
          if var_out == 'taxis':
              mycase.output['taxis'] = values
          else:
              if var_out not in mycase.output:
                  mycase.output[var_out] = np.zeros([len(values), mycase.nsamples], float)
              mycase.output[var_out][:, n-1] = values
      else:
          mycase.postprocessed[n-1] = 1

def postprocessed_output_vars(requested_vars):
    expanded = []
    for var in requested_vars:
        if var in mycase.output:
            expanded.append(var)
        elif '_pft' in var:
            for p in mycase.postproc_pfts:
                var_out = var+str(p)
                if var_out in mycase.output:
                    expanded.append(var_out)
                else:
                    print('Warning: postprocessed output '+var_out+' not found; skipping UQ')
        elif '_col' in var:
            for c in mycase.postproc_cols:
                var_out = var+str(c)
                if var_out in mycase.output:
                    expanded.append(var_out)
                else:
                    print('Warning: postprocessed output '+var_out+' not found; skipping UQ')
        else:
            print('Warning: postprocessed output '+var+' not found; skipping UQ')
    return expanded

workdir = os.getcwd()

if (not options.UQ_only):
  mycase.output = {}
  mycase.postprocessed = np.zeros([mycase.nsamples], int)
  run_completed = np.zeros([mycase.nsamples], int)
  if options.segment_years > 0:
    print('Running ensemble segment '+str(options.segment)+' for '+ \
          str(options.segment_years)+' years; expected restart year '+ \
          str(expected_restart_year()))

  if (options.postproc_only):
    # Run postprocessing in parallel using a process pool (no subprocess overhead)
    postprocess_members(list(range(1, mycase.nsamples+1)))

    # Mark failed members with -9999 sentinel so surrogate training excludes them
    failed = np.where(mycase.postprocessed == -1)[0]
    if len(failed) > 0:
        print(f'{len(failed)} ensemble members failed postprocessing and will be excluded from surrogate training')
        for var_out in mycase.output:
            if var_out == 'taxis':
                continue
            mycase.output[var_out][:, failed] = -9999

  else:
    processes=[]
    process_jobnum=[]
    process_hang=[]    #Keep track of how long process has been hanging
    n_job = 1
    rotate_index = 0
    nodes_list = []
    if (mycase.noslurm == False):
      allocated_nodes = None
      if 'SLURM_JOB_NUM_NODES' in os.environ:
          try:
              allocated_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
          except Exception:
              allocated_nodes = None
      # prepare node rotation list if requested
      if getattr(mycase, 'rotate_nodes', False):
        try:
          nodes_list = get_nodelist()
        except Exception:
          nodes_list = []

    #Run the simulations 
    while (n_job <= mycase.nsamples):
      pactive = active_processes(processes,process_jobnum,process_hang)
      if check_run_success(n_job):
        if is_final_segment and defer_postprocess:
          run_completed[n_job-1] = 1
        elif is_final_segment:
          ierr = postprocess_ensemble(n_job)
          if (ierr != 0):
            print('Postprocessing failed for ensemble member '+str(n_job)+', skipping')
            run_completed[n_job-1] = -1
            mycase.postprocessed[n_job-1] = -1
          else:
            run_completed[n_job-1] = 1
            mycase.postprocessed[n_job-1] = 1
        else:
          run_completed[n_job-1] = 1
          mycase.postprocessed[n_job-1] = 1
        n_job = n_job+1
        continue
      if (sum(pactive) < int(mycase.np_ensemble)):
        jobst = str(100000+n_job)
        rundir = mycase.rundir_UQ + '/g'+jobst[1:]+'/'
        log_name = 'e3sm_log.txt'
        if options.segment_years > 0:
          log_name = 'e3sm_log.seg'+str(1000+options.segment)[1:]+'.txt'
        log_file_path = f"{rundir}{log_name}"
        time.sleep(0.2)
        if (mycase.noslurm == False):
          # compute per-member node/task layout
          nodes_per_run = max(1, int(math.ceil(float(mycase.np) / float(mycase.npernode))))
          tpnode = min(int(mycase.npernode), int(mycase.np))
          cpus_per_task = getattr(mycase, 'thread_count', 1) or 1
          # current number running
          current_running = sum(pactive)
          # If packing single-task steps, skip the required_nodes check and
          # allow Slurm to place steps on allocated nodes. Otherwise, ensure
          # there are enough allocated nodes before requesting per-run nodes.
          if not getattr(mycase, 'pack_single_tasks', False):
              required_nodes = nodes_per_run * (current_running + 1)
              if allocated_nodes is not None and required_nodes > allocated_nodes:
                  # not enough nodes available now; wait and retry
                  time.sleep(1)
                  continue
        retry_ensemble_copy(n_job, clean=(not is_continue_segment))
        with open(log_file_path, "w") as log_file:
          if (mycase.noslurm == False):
            # build srun. Two modes:
            # - pack_single_tasks: emit `srun -n 1 -c X` for each member and
            #   optionally add `-w <node>` when rotate_nodes is enabled.
            # - default: request nodes/ntasks-per-node and use distribution to
            #   spread runs across nodes.
            exe_name = 'elm_offline_driver' if getattr(mycase, 'offline_driver', False) else 'e3sm.exe'
            exe_path = mycase.exeroot+'/'+exe_name
            if getattr(mycase, 'pack_single_tasks', False):
              # optional rotate_nodes: pick a node from the allocation
              node_arg = ''
              if getattr(mycase, 'rotate_nodes', False) and len(nodes_list) > 0:
                try:
                  node = nodes_list[rotate_index % len(nodes_list)]
                  rotate_index = rotate_index + 1
                  node_arg = ' -w '+node
                except Exception:
                  node_arg = ''
              if (mycase.apptainer != ''):
                command = 'srun -n 1 -c '+str(cpus_per_task)+ node_arg + \
                  ' apptainer exec --bind '+mycase.apptainer_bind+' --pwd '+rundir+ \
                  ' --env OMPI_MCA_pml=ob1 --env OMPI_MCA_btl=self,vader,tcp  ' +mycase.apptainer+' '+exe_path
              else:
                command = 'srun -n 1 -c '+str(cpus_per_task)+ node_arg + ' '+exe_path
            else:
              # build srun using nodes/ntasks-per-node and distribution to avoid packing
              if (mycase.apptainer != ''):
                command = 'srun --nodes='+str(nodes_per_run)+ ' --ntasks='+str(mycase.np)+ ' --ntasks-per-node='+str(tpnode)+ \
                  ' -c '+str(cpus_per_task)+ \
                  ' apptainer exec --bind '+mycase.apptainer_bind+' --pwd '+rundir+ \
                  ' --env OMPI_MCA_pml=ob1 --env OMPI_MCA_btl=self,vader,tcp  ' +mycase.apptainer+' '+exe_path
              else:
                command = 'srun --nodes='+str(nodes_per_run)+ ' --ntasks='+str(mycase.np)+ ' --ntasks-per-node='+str(tpnode)+ \
                  ' -c '+str(cpus_per_task)+ ' '+exe_path
          else:
            exe_name = 'elm_offline_driver' if getattr(mycase, 'offline_driver', False) else 'e3sm.exe'
            command = mycase.exeroot+'/'+exe_name
          # record the exact launch command for debugging in the member log
          try:
            log_file.write('LAUNCH_CMD: '+command+'\n')
            launch_argv = command_to_argv(command)
            arg_bytes = sum(len(arg) + 1 for arg in launch_argv)
            log_file.write('LAUNCH_ARGC: '+str(len(launch_argv))+'\n')
            log_file.write('LAUNCH_ARG_BYTES: '+str(arg_bytes)+'\n')
            log_file.write('LAUNCH_ENV_BYTES: '+str(env_size_bytes(os.environ))+'\n')
            try:
              log_file.write('LAUNCH_ARG_MAX: '+str(os.sysconf('SC_ARG_MAX'))+'\n')
            except Exception:
              pass
            log_file.flush()
          except Exception:
            launch_argv = command_to_argv(command)
          try:
            process = subprocess.Popen(launch_argv, shell=False, stderr=subprocess.STDOUT, cwd=rundir, stdout=log_file)
          except OSError as e:
            log_file.write('LAUNCH_OSERROR: '+repr(e)+'\n')
            log_file.write('LAUNCH_ARGV: '+repr(launch_argv)+'\n')
            log_file.flush()
            raise
          processes.append(process)
          process_jobnum.append(n_job)
          process_hang.append(0)
        n_job=n_job+1
      else:
        time.sleep(1)

    while (sum(pactive) > 0):
      pactive = active_processes(processes,process_jobnum,process_hang)
      time.sleep(1)

    if defer_postprocess:
      completed_members = [i+1 for i in range(mycase.nsamples) if run_completed[i] == 1]
      failed_members = [i+1 for i in range(mycase.nsamples) if run_completed[i] == -1]
      if len(failed_members) > 0:
          print(str(len(failed_members))+' ensemble members failed before postprocessing')
          for n in failed_members:
              mycase.postprocessed[n-1] = -1
      postprocess_members(completed_members)

    if is_final_segment:
      # Mark failed members with -9999 sentinel so surrogate training excludes them
      failed = np.where(mycase.postprocessed == -1)[0]
      if len(failed) > 0:
          print(f'{len(failed)} ensemble members failed and will be excluded from surrogate training')
          for var_out in mycase.output:
              if var_out == 'taxis':
                  continue
              mycase.output[var_out][:, failed] = -9999

  mycase.create_pkl(outdir=mycase.OLMTdir+'/pklfiles/')

  if not is_final_segment:
    failed = np.where(mycase.postprocessed == -1)[0]
    if len(failed) > 0:
      print(str(len(failed))+' ensemble members failed before the final segment')
      sys.exit(1)

#UQ part of code

if (is_final_segment and mycase.postproc_vars != []):
    # Save postprocessed ensemble outputs in a portable NetCDF file before UQ analysis.
    mycase.write_postprocessed_netcdf()

    uq_vars = postprocessed_output_vars(mycase.postproc_vars)
    if len(uq_vars) == 0:
        print('No postprocessed output variables available for UQ; skipping UQ analysis')
        sys.exit(0)

    #Train surrogate models
    mycase.train_surrogate(uq_vars)
    mycase.plot_surrogate(uq_vars)

    #run GSA
    mycase.GSA(uq_vars)
    mycase.plot_GSA(uq_vars)

    # Plot ensemble percentiles for each postprocessed variable
    for var in uq_vars:
      mycase.plot_ensemble(var)
    
    #Save postprocessed output
    mycase.create_pkl(outdir=mycase.OLMTdir+'/pklfiles/')

    #run MCMC
    #Set intial values for parameters
    if (mycase.obs):
        #Run MCMC for the observation variables
        obs_mcmc = [v for v in uq_vars if v in mycase.obs.keys()]
        # Always run single-site MCMC first
        mycase.nobs_vars = 3
        nwalkers = max(24, (mycase.nparms_ensemble+mycase.nobs_vars)*2)
        mycase.MCMC(obs_mcmc, nwalkers=nwalkers, nsteps=10000, multisite=False)
        
        # Check if multisite MCMC should be run
        if hasattr(mycase, 'all_sites') and mycase.all_sites is not None and len(mycase.all_sites) > 1:
            print(f"Running multisite MCMC for {len(mycase.all_sites)} sites: {mycase.all_sites}")
            mycase.MCMC(obs_mcmc, nwalkers=nwalkers, nsteps=10000, multisite=True)
            
        #Save postprocessed output
        mycase.create_pkl(outdir=mycase.OLMTdir+'/pklfiles/')
