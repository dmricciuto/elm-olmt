import sys
sys.path.append('..')
import model_ELM
from OLMTutils import get_machine_info, get_site_info, get_point_list
import os
import numpy as np


#Get default directories, automatically detect machine if machine_name=''
machine, rootdir, inputdata, queue, project, hostname = get_machine_info(machine_name='pm-cpu')

#set rootdir and inputdata below if you want to override defaults
caseroot= rootdir+'/e3sm_cases'
runroot = rootdir+'/e3sm_run'
#TODO:  add option to clone repository

#modelroot = '/gpfs/wolf2/cades/cli185/proj-shared/zdr/E3SM'  #Existing E3SM code directory
modelroot = os.environ['HOME']+'/E3SM' 

#Set the full path of the bld directory to use a pre-built executable. Set exeroot='' to build 
exeroot = ''

project = 'm2420'
account = 'm2420'
queue   = 'shared'   # Defaults to 'regular'  on perlmutter
walltime = 12  

#----------------------Required inputs---------------------------------------------

runtype = 'site'               #site,latlon_list,latlon_bbox
mettype = 'site'               #Site or reanalysis product to use (site, gswp3, crujra)
case_suffix = ''               #Identifier for cases (leave blank if none)

if (runtype == 'site'):
    sites = ['TAM-06']               #Site name, list of site names, or 'all' for all sites in site group
    sitegroup = 'NGEETropics'       #Sites defined in <inputdata>/lnd/clm2/PTCLM/<sitegroup>_sitedata.txt
    numproc = 1
else:
    region_name = 'gradient'   #Set the name of the region/point list to be simulated
    numproc = 5            #Number of processors, must be <= the number of active gridcells
    if (runtype == 'latlon_list'):
        point_list_file = '/home/ac.ricciuto/models/elm-olmt/test_list.txt'   #file with a list of lat lons

#If neither point_list or site is defined, it will use the bounds below.
lat_bounds = [-90,90]
lon_bounds = [-180,180]
res = 'hcru_hcru'          #Resolution of global files to extract from

use_cpl_bypass = False   #True      #Coupler bypass for meteorology
nutrients      = 'CNP'     #none or SP (SP mode), C (carbon only), CN (carbon/nitrogen), or CNP (carbon/nitrogen/phosphorus)
nutrient_comp  = 'RD'      #Relative demand (RD) or equilibirium chemistry approximiation (ECA)
soil_decomp    = 'CTC'     #Convergent trophic cascade (CTC) or Century (CNT) model
#FATES options
use_fates      = True       #Use FATES compsets
fates_pft      = 0          #Extract this PFT index from fates parameter file and only run one (set -1 for all)
                            #Note:  Custom parameter file can be specified instead through case_options (see below)
pft_duplicates = 1          #Construct a file with n pfts, all using the same parameters as fates_pft

#Run lengths/dates
nyears_ad      =  50      #number of years for ad spinup
nyears_final   =  50      #number of years for final spinup, SP run, or FATES C-only
nyears_trans   =  20      #number of years for transient run 
run_startyear  = 1850      #Starting year for transient run, SP run or FATES C-only


#---------------------Optional inputs via namelist variables------------------------
#Define a dictionary to handle namelist options.
#note:  use surffile, domainfile, pftdynfile, metdir instead of the standard namelist variables for those files.
#case_options['option'] = value or [value1, value2, value3] if applying different options to different compsets
case_options={} 
#case_options['metdir'] = '/gpfs/wolf2/cades/cli185/proj-shared/zdr/elm-olmt/runscripts'
#case_options['fates_paramfile'] = inputdata+'/lnd/clm2/paramdata/fates_params_api.32.0.0_pft1_c231215.nc'
#case_options['use_fates_planthydro'] = '.true.'

#--------------------ensemble options------------------------------------------------

parm_list      = ''
#parm_list = 'examples/parm_list_fatesUQ'  #Set parameter list (leave blank for no ensemble)
nsamples       =  256    #number of samples to run
np_ensemble    =  256    #number of ensemble numbers to run in parallel (MUST be <= nsamples)
ensemble_file  = ''     #File containing samples (if blank, OLMT will generate one)
postproc_vars  = ['FATES_NPP','FATES_VEGC']  #Variables to automatically post-process
postproc_startyear = 1860
postproc_endyear   = 1862
postproc_freq      = 'annual'   #Can be daily, monthly, annual, hourly(not tested)

#Observations to use in calibration (must match a model output variable in name/units, and the 
#  length of the postprocessed record 
#This could be replaced with code to read an observation file.
observations = {}
#Note: observations are from a nutrient-enabled simulation
#fates_alloc_storage_cushion = 1.3813409878873
#fates_leaf_vcmax25top = 48.1649986078143 ;
#fates_maintresp_leaf_ryan1991_baserate = 2.07131195058174e-06 ;
#fates_q10_mr = 1.3504971149054 
observations['FATES_NPP'] =  np.array([3.6148E-08,3.4145E-08,3.3922E-08]) #kgC m-2 s-1
observations['FATES_VEGC'] = np.array([1.1578E+01,1.1578E+01,1.1519E+01]) #kgC m-2
observation_error = {}
#For this example, assume 5% error
observation_error['FATES_NPP'] = observations['FATES_NPP']*0.05
observation_error['FATES_VEGC'] = observations['FATES_VEGC']*0.05


#----------------------Define treatment cases ----------------------------------------
#
#Treatmeant cases will use the same compset as the last case, and will inherit case_options
#Specify additional options for treatments as a list (one for each desired treatment)
nyears_treatment   = 85                              #number of years to run treatment simulation (assumed all same)
startyear_treatment = run_startyear + nyears_trans   #Starting year (assuming to start from end of transient
treatment_options={}

#---------------End of user input -----------------------------------------------------

print('\n')
if (runtype == 'site'):
  #Check to see if all reqested sites exist
  if not isinstance(sites,list):
        sites=[sites]

  if (sites[0] != ''):
    siteinfo = get_site_info(inputdata, sitegroup=sitegroup)
    if sites[0] == 'all':
        sites = list(siteinfo.keys())
        print('Running all sites in '+sitegroup+' site group:')
        print(sites)
    else:
        for s in sites:
            if not (s in siteinfo.keys()):
                print(s+' not in '+sitegroup+' site group. Exiting.')
                print('Available sites: ',siteinfo.keys())
                sys.exit(1)
        print('Running site(s): ', sites)
  point_list  = []
  region_name = ''
else:
    sites=['']
    if (runtype == 'latlon_list'):
        point_list = get_point_list(point_list_file)
        print('Running ', len(point_list), 'grid cells')
        print('Points in '+point_list_file)
        if (numproc > len(point_list)):
            numproc = len(point_list)
            print('Warning:  number of proceessors greater than number '\
                    ,'of grid cells. Setting numproc = ',numproc)
    else:
        point_list = []
        print('Running with lat/lon bounding box')
        print('Lat: ', lat_bounds)
        print('Lon: ', lon_bounds)

#Construct the list of compsets and suppring information
compset_type="I"
if (use_cpl_bypass):
    compset_type='ICB'
#Construct the list of compsets and supporting information
twophase=False
compset_base=nutrients+nutrient_comp+soil_decomp+'BC'
if (use_fates):
    compset_base='ELMFATES'
compset_type="I"
if (use_cpl_bypass):
    compset_type='ICB'
elif ((mettype != 'site' or 'PR-LUQ' in sites) and nyears_trans != 0):
    twophase=True       #if using DATM and reanalysis, split into 2 cases

#TODO - move construction of compset lists to a function (in OLMTinfo)
compsets=[]
suffix=[]
startyear=[]
nyears=[]
if (not use_fates and (nutrients == 'none' or nutrients =='SP')):
  compsets.append(compset_type+'ELMBC')
  suffix.append('')
  startyear.append(run_startyear)
  nyears.append(nyears_final)
  depends=[-1]
else:
  if (nyears_ad > 0):
    compsets.append(compset_type+'1850'+compset_base.replace('CNP','CN'))  #ad_spinup
    suffix.append('ad_spinup')
    startyear.append(1)
    nyears.append(nyears_ad)
  if (nyears_final > 0):
    compsets.append(compset_type+'1850'+compset_base)  #Final spinup
    suffix.append('')
    startyear.append(1)
    nyears.append(nyears_final)
  if (nyears_trans != 0):
    compsets.append(compset_type+'20TR'+compset_base)  #Transient
    suffix.append('')
    startyear.append(run_startyear)
    nyears.append(nyears_trans)
  depends = np.cumsum(np.ones([len(compsets)],int))-2
if (twophase):                            #add the phase 2 compset and case info
    compsets.append(compset_type+'20TR'+compset_base)  #Transient phase 2
    nyears.append(nyears[-1])
    suffix.append('phase2')
    depends = np.append(depends, depends[-1]+1)
    startyear.append(run_startyear)
istreatment=np.zeros([len(compsets)],int)
ncases_pretreatment = len(compsets)

ensemble=False
if (parm_list != ''):
    ensemble=True

#Add treatment cases
if ('suffix' in treatment_options.keys()):
  for t in range(0,len(treatment_options['suffix'])):
    nyears.append(nyears_treatment)
    istreatment = np.append(istreatment, 1)
    depends = np.append(depends, ncases_pretreatment-1)
    compsets.append(compsets[-1])
    suffix.append(treatment_options['suffix'][t])
    startyear.append(startyear_treatment)

print('Machine: '+machine)
print('Run root directory:  '+runroot)
print('Case root directory: '+caseroot)
print('Input data directory: '+inputdata)
print('Model root directory: '+modelroot+'\n')

print('\nELM simulation info:')
multisite_scripts=[]
for c in range(0,len(compsets)):
    print('Compset '+str(c+1)+': '+compsets[c])
    print('   Simulation starting year: '+str(startyear[c]))
    if (nyears[c] > 0):
        print('   Simulation length:        '+str(nyears[c]))
    multisite_scripts.append('')
    if (istreatment[c]):
        print('   Treatment:                '+ \
                treatment_options['suffix'][c-ncases_pretreatment])
    print('\n')
if (ensemble):
    print('Ensemble size:  '+str(nsamples))
    print('Parameter list: '+parm_list+'\n')

nsites = len(sites)
jobnum = np.zeros(len(compsets),int)  #list of submitted job ids

for site in sites:
  cases={}
  ncases = len(compsets)  #how many cases we are running
  scriptdir=os.getcwd()

  for c in range(0,ncases):
    mysuffix = '_'.join(filter(None,[suffix[c],case_suffix]))

    cases[c] = model_ELM.ELMcase(caseid='',compset=compsets[c], site=site, \
        caseroot=caseroot,runroot=runroot,inputdata=inputdata,modelroot=modelroot, \
        machine=machine, exeroot=exeroot, suffix=mysuffix, project=project,  \
        res=res, nyears=nyears[c],startyear=startyear[c], region_name=region_name, \
        lat_bounds=lat_bounds, lon_bounds=lon_bounds, np=numproc, point_list=point_list)

    #Create the case
    cases[c].create_case()
    cases[c].case_options={}
    if (site != ''):
        cases[c].siteinfo = siteinfo[site]

    #Get the namelist options for this case
    for key in case_options.keys():
        if isinstance(case_options[key], list):
            cases[c].case_options[key] = case_options[key][c]
        else:
            cases[c].case_options[key] = case_options[key]
    #Add the treatment options (must be list format)
    if (istreatment[c]):
        for key in treatment_options.keys():
            cases[c].case_options[key] = treatment_options[key][c-ncases_pretreatment]
    #Other options
    cases[c].nutrients = nutrients
    cases[c].nutrient_comp = nutrient_comp
    cases[c].soil_decomp = soil_decomp
    cases[c].fates_pft=fates_pft
    cases[c].pft_duplicates = pft_duplicates
    #Set the custom parameter files
    if ('fates_paramfile' in case_options):
      cases[c].fates_paramfile = case_options['fates_paramfile']
    if ('paramfile' in case_options):
      cases[c].paramfile = case_options['paramfile']

    #Get forcing information
    print('Getting forcing information')
    if ('phase2' in suffix[c]):
      #Set the starting year from the last case
      cases[c].startyear = cases[c-1].startyear+cases[c-1].run_n
    if ('metdir' in cases[c].case_options.keys()):
        metdir = cases[c].case_options['metdir']
        cases[c].get_forcing(mettype=mettype, metdir=metdir)
    else:
        cases[c].get_forcing(mettype=mettype)

    #Set the initial data file (if depends on previous case)
    cases[c].dependcase=''
    if (depends[c] >= 0):
      #Set the iniial data file from the last year of the prev case
      finidat_year = cases[depends[c]].run_n+1
      if ('20TR' in cases[depends[c]].compset or 'trans' in cases[depends[c]].compset):
          finidat_year = 1850+cases[depends[c]].run_n
      cases[c].set_finidat_file(finidat_case=cases[depends[c]].casename, \
              finidat_year=finidat_year)
      cases[c].dependcase = cases[depends[c]].casename

    #Set postprocessing variables for ensemble (final case or treatment case)
    if ((c == ncases-1 or istreatment[c]) and ensemble):
      cases[c].postproc_vars = postproc_vars
    else:
      cases[c].postproc_vars=[]
    cases[c].postproc_startyear = postproc_startyear
    cases[c].postproc_endyear = postproc_endyear
    cases[c].postproc_freq = postproc_freq

    #Set up the case (surface, domain and pftdata)
    print('Setting up case for site: '+site)
    cases[c].setup_case()
    if (c == 0):
      #Get the surface and domain data 
      cases[c].setup_domain_surfdata(makesurfdat=True,makedomain=True)
    if (ensemble):
        if (site == sites[0] and c == 0):
            #Get the ensemble file from the first site and case
            cases[c].setup_ensemble(parm_list=parm_list,np_ensemble=np_ensemble,nsamples=nsamples, \
                    ensemble_file = ensemble_file, obs=observations, obs_err=observation_error)
            ensemble_file = cases[c].ensemble_file
        else:
            #Use the ensemble file for subsequent cases and sites
            cases[c].setup_ensemble(parm_list=parm_list,np_ensemble=np_ensemble,nsamples=nsamples, \
                ensemble_file = ensemble_file, obs=observations, obs_err=observation_error)
    if ('20TR' in compsets[c] and not use_fates):
      #Get the dynamic PFT data
      cases[c].mask_grid = cases[0].mask_grid          #Get the mask from the first case
      cases[c].setup_domain_surfdata(makepftdyn=True)

    #Build the case
    print('Building case')
    cases[c].build_case()
    
    #Submit the case
    print('Submitting case')
    jobnum_depend=-1
    if (depends[c] >= 0):
        jobnum_depend = jobnum[depends[c]]
    #Set exeroot for all subsequent cases/sites so we don't have to rebuild
    if (depends[c] < 0 and site == sites[0]):
        exeroot = cases[c].exeroot
    if (ensemble):
        multisite_scripts[c] = cases[c].create_multisite_script([site], scriptdir)
        jobnum[c] = cases[c].submit_case(depend=jobnum_depend, \
            ensemble=ensemble,multisite_script=multisite_scripts[c])
    else:
        if (site == sites[0]):
            #Always use the multi-site script even for one site
            multisite_scripts[c] = cases[c].create_multisite_script(sites, scriptdir)
        if (site == sites[nsites-1]):
            jobnum[c] = cases[c].submit_case(depend=jobnum_depend, \
                ensemble=ensemble,multisite_script=multisite_scripts[c])
    #Return to script directory
    os.chdir(scriptdir)


#archive this script (based on name of first case)
archive_fname='./archive/'+cases[0].casename.replace('_ad_spinup','')
archive_fname=archive_fname.replace('1850','').replace('20TR','')+'_'+machine
if (nsites > 1):
    archive_fname = archive_fname.replace(site,'multisite')
os.system('mkdir -p archive')
if (ensemble):
    archive_fname = archive_fname+'_ensemble'
os.system('cp '+__file__+' '+archive_fname+'.py')


