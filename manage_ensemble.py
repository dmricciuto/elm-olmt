#!/usr/bin/env python
import sys,os, time, math
import numpy as np
import subprocess
import random
import pickle
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
(options, args) = parser.parse_args()

#Load case object
myfile=open('pklfiles/'+options.case+'.pkl','rb')
mycase=pickle.load(myfile)
# default placement: enable pack_single_tasks only for single-task cases unless overridden
if not hasattr(mycase, 'pack_single_tasks'):
  mycase.pack_single_tasks = (getattr(mycase, 'np', 1) == 1)
if not hasattr(mycase, 'rotate_nodes'):
  mycase.rotate_nodes = True

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
    yst = str(10000+mycase.startyear+mycase.run_n)[1:]
    #yst = '2010'
    if (os.path.isfile(rundir+'/'+mycase.casename+'.elm.r.'+yst+'-01-01-00000.nc')):
        success=True
    return success

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
            #Post-process ensemble member if it hasn't yet been done
            if (mycase.postprocessed[n] == 0):
                print(n, check_run_success(process_jobnum[n]))
                if (check_run_success(process_jobnum[n])):
                    ierr = postprocess_ensemble(process_jobnum[n])
                else:
                    print('Ensemble member '+str(process_jobnum[n])+ \
                            'Failed to complete')
                mycase.postprocessed[n] = 1
        n=n+1
    return pactive

def postprocess_ensemble(n):
  #Postprocess
  if (mycase.postproc_vars != []):
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
          if (mycase.postproc_freq == 'daily' or mycase.postproc_freq == 'hourly'):  #default
            mycase.postprocess(v, ens_num=n,startyear=mycase.postproc_startyear, \
                  endyear=mycase.postproc_endyear,index=p,hnum=hnum)
          elif (mycase.postproc_freq == 'monthly'):  #monthly
            mycase.postprocess(v, ens_num=n,startyear=mycase.postproc_startyear, \
                  endyear=mycase.postproc_endyear,index=p,hnum=hnum, dailytomonthly=True)
          elif (mycase.postproc_freq == 'annual'):  #annual
            mycase.postprocess(v, ens_num=n,startyear=mycase.postproc_startyear, \
                  endyear=mycase.postproc_endyear,index=p,hnum=hnum, annualmean=True)
  return 0

workdir = os.getcwd()

if (not options.UQ_only):
  mycase.output = {}
  processes=[]
  process_jobnum=[]
  process_hang=[]    #Keep track of how long process has been hanging
  mycase.postprocessed=np.zeros([mycase.nsamples],int)
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
    if (sum(pactive) < int(mycase.np_ensemble)):
      jobst = str(100000+n_job)
      rundir = mycase.rundir_UQ + '/g'+jobst[1:]+'/'
      log_file_path = f"{rundir}e3sm_log.txt"
      if not options.postproc_only:
        time.sleep(0.2)
        mycase.ensemble_copy(n_job)
      with open(log_file_path, "w") as log_file:
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

          # build srun. Two modes:
          # - pack_single_tasks: emit `srun -n 1 -c X` for each member and
          #   optionally add `-w <node>` when rotate_nodes is enabled. 
          # - default: request nodes/ntasks-per-node and use distribution to
          #   spread runs across nodes.
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
                ' --env OMPI_MCA_pml=ob1 --env OMPI_MCA_btl=self,vader,tcp  ' +mycase.apptainer+' '+mycase.exeroot+'/e3sm.exe'
            else:
              command = 'srun -n 1 -c '+str(cpus_per_task)+ node_arg + ' '+mycase.exeroot+'/e3sm.exe'
          else:
            # build srun using nodes/ntasks-per-node and distribution to avoid packing
            if (mycase.apptainer != ''):
              command = 'srun --nodes='+str(nodes_per_run)+ ' --ntasks='+str(mycase.np)+ ' --ntasks-per-node='+str(tpnode)+ \
                ' -c '+str(cpus_per_task)+ \
                ' apptainer exec --bind '+mycase.apptainer_bind+' --pwd '+rundir+ \
                ' --env OMPI_MCA_pml=ob1 --env OMPI_MCA_btl=self,vader,tcp  ' +mycase.apptainer+' '+mycase.exeroot+'/e3sm.exe'
            else:
              command = 'srun --nodes='+str(nodes_per_run)+ ' --ntasks='+str(mycase.np)+ ' --ntasks-per-node='+str(tpnode)+ \
                ' -c '+str(cpus_per_task)+ ' '+mycase.exeroot+'/e3sm.exe'
        else:
          command = mycase.exeroot+'/e3sm.exe'
        if (options.postproc_only):
            command='ls'
        # record the exact launch command for debugging in the member log
        try:
          log_file.write('LAUNCH_CMD: '+command+'\n')
          log_file.flush()
        except Exception:
          pass
        process = subprocess.Popen(command, shell=True, stderr=subprocess.STDOUT, cwd=rundir, stdout=log_file)
        processes.append(process)
        process_jobnum.append(n_job)
        process_hang.append(0)
      n_job=n_job+1
    else:
      time.sleep(1)

  while (sum(pactive) > 0):
    pactive = active_processes(processes,process_jobnum,process_hang)
    time.sleep(1)

  mycase.create_pkl(outdir=mycase.OLMTdir+'/pklfiles/')

#UQ part of code

if (mycase.postproc_vars != []):
    #Train surrogate models
    mycase.train_surrogate(mycase.postproc_vars)
    mycase.plot_surrogate(mycase.postproc_vars)

    #run GSA
    mycase.GSA(mycase.postproc_vars)
    mycase.plot_GSA(mycase.postproc_vars)

    # Plot ensemble percentiles for each postprocessed variable
    for var in mycase.postproc_vars:
      mycase.plot_ensemble(var)
    
    #Save postprocessed output
    mycase.create_pkl(outdir=mycase.OLMTdir+'/pklfiles/')

    #run MCMC
    #Set intial values for parameters
    if (mycase.obs):
        #Run MCMC for the observation variables
        obs_mcmc = [v for v in mycase.postproc_vars if v in mycase.obs.keys()]
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



