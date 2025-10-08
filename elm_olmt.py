#!/usr/bin/env python
import sys
import model_ELM
from OLMTutils import get_machine_info, get_site_info, get_point_list, get_default_diag_vars
import os
import numpy as np
import configparser
import argparse

def load_config(config_file):
    """Load configuration from file and return as dictionary"""
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Convert to nested dictionary for easier access
    cfg = {}
    for section in config.sections():
        cfg[section] = {}
        for key, value in config.items(section):
            # Strip quotes from the value first
            value = value.strip().strip('\'"')
            
            # Handle different data types
            if value.lower() in ['true', 'false']:
                cfg[section][key] = value.lower() == 'true'
            elif not ',' in value:
                # Handle single values
                if value.isdigit():
                    cfg[section][key] = int(value)
                elif value.replace('.', '').replace('-', '').isdigit():
                    cfg[section][key] = float(value)
                else:
                    if 'variables' in key:
                        cfg[section][key] = [value] 
                    else:
                        cfg[section][key] = value if value else None
            else:
                # Handle comma-separated lists
                items = [x.strip().strip('\'"') for x in value.split(',')]
                # Try to convert to numeric types
                try:
                    # Try int first
                    cfg[section][key] = [int(x) for x in items]
                except ValueError:
                    try:
                        # Try float
                        cfg[section][key] = [float(x) for x in items]
                    except ValueError:
                        if 'variables' in key or 'sites' in key:
                            cfg[section][key] = [str(x) for x in items]
                        else:
                            # Keep as comma-separated string for string lists (except sites)
                            if 'hist_fincl' in key:
                                # Special handling for hist_fincl to keep quotes
                                items = [f"'{x}'" for x in items]
                            cfg[section][key] = ', '.join(items)    
    return cfg

def main():
    parser = argparse.ArgumentParser(description='Run ELM BGC simulations')
    parser.add_argument('--config', '-c', default='run_config.cfg',
                       help='Configuration file (default: run_config.cfg)')
    parser.add_argument('--gui', action='store_true', help='Launch GUI to create configuration')
    args = parser.parse_args()

    print('\n')
    print(f"    *** OLMT ***")
    if args.gui:
        import subprocess
        subprocess.run([sys.executable, 'GUI_experimental.py'])
        print('GUI launched. Exiting.')
        return

    # Load configuration
    cfg = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Get machine info
    machine_name = cfg['machine'].get('machine_name', '')
    machine, rootdir, inputdata, queue, project, hostname = \
        get_machine_info(machine_name=machine_name)
    print('Machine: '+machine+'\n')
    
    # Override machine defaults with config values if provided
    queue = cfg['machine'].get('queue', queue)
    project = cfg['machine'].get('project', project)
    inputdata = cfg['machine'].get('inputdata', inputdata)
    caseroot = cfg['machine'].get('caseroot', rootdir + '/e3sm_cases')
    runroot = cfg['machine'].get('runroot', rootdir + '/e3sm_run')
    modelroot = cfg['machine'].get('modelroot', '')
    exeroot = cfg['machine'].get('exeroot', '')
    print('Run root directory:  '+runroot)
    print('Case root directory: '+caseroot)
    print('Input data directory: '+inputdata)
    print('Model root directory: '+modelroot+'\n')

    # Extract configuration values
    runtype = cfg['simulation']['runtype']
    mettype = cfg['simulation']['mettype']
    metdir = cfg['simulation'].get('metdir', '')
    case_suffix = cfg['simulation'].get('case_suffix', '')

    # Site configuration
    if runtype == 'site':
        sites = cfg['simulation']['sites']
        if isinstance(sites, str):
            sites = [sites]
        sitegroup = cfg['simulation']['sitegroup']
        numproc = 1
        lat_bounds = [-180,180]
        lon_bounds = [-90, 90]
    else:
        sites = ['']
        region_name = cfg['simulation'].get('name','global')
        numproc = cfg['simulation']['numproc']
        if runtype == 'latlon_list':
            point_list_file = cfg['simulation']['point_list_file']
        lat_bounds = cfg['simulation']['lat_bounds']
        lon_bounds = cfg['simulation']['lon_bounds']
    use_cpl_bypass = cfg['simulation'].get('use_cpl_bypass',True)
    res = cfg['simulation']['res']
    
    # Biogeochemistry options
    nutrients = cfg['biogeochemistry']['nutrients']
    nutrient_comp = cfg['biogeochemistry'].get('nutrient_comp','')
    soil_decomp = cfg['biogeochemistry'].get('soil_decomp','')
    print(f"Running {runtype} simulation with {nutrients} nutrients")
    #print('\n')
    
    # FATES options
    use_fates = cfg['biogeochemistry'].get('use_fates', False)
    if use_fates:
        if 'fates_pft' not in cfg['biogeochemistry']:
            raise ValueError("FATES PFT configuration missing in the config file.")
        else:
            fates_pft = cfg['biogeochemistry']['fates_pft']
        pft_duplicates = cfg['biogeochemistry'].get('pft_duplicates', 1)
    # Crop options
    use_crop = cfg['biogeochemistry'].get('use_crop', False)

    # Run lengths
    nyears_ad = cfg['run_lengths'].get('nyears_ad', 0)
    nyears_final = cfg['run_lengths'].get('nyears_final', 0)
    if (nutrients == 'none'):
        nyears_final = cfg['run_lengths'].get('nyears', nyears_final)
    nyears_trans = cfg['run_lengths'].get('nyears_trans', 0)
    run_startyear = cfg['run_lengths'].get('trans_startyear', 1850)
    if (nutrients == 'none'):
        run_startyear = cfg['run_lengths'].get('startyear', run_startyear)

    # Post-processing
    if ('postprocessing' in cfg):
        postproc_vars = cfg['postprocessing'].get('variables', get_default_diag_vars(nutrients, use_fates))
        postproc_startyear = cfg['postprocessing']['startyear']
        postproc_endyear = cfg['postprocessing']['endyear']
        postproc_freq = cfg['postprocessing']['frequency']

    # Ensemble options
    if ('ensemble' in cfg):
        parm_list = cfg['ensemble'].get('parm_list', '')
        if (parm_list != ''):
            nsamples = cfg['ensemble']['nsamples']
            np_ensemble = cfg['ensemble'].get('np_ensemble',nsamples)
            ensemble_file = cfg['ensemble'].get('ensemble_file','')
    else:
        parm_list = ''

    # Treatment options
    #nyears_treatment = cfg['treatments']['nyears_treatment']

    # Observations 
    has_obs = False
    if ('observations' in cfg):
        obs_dir = cfg['observations']['location']
        obs_vars = cfg['observations']['variables']
        # Ensure obs_vars is always a list
        if isinstance(obs_vars, str):
            obs_vars = [obs_vars]
        elif not isinstance(obs_vars, list):
            obs_vars = list(obs_vars)
        obs_startyear = cfg['observations'].get('startyear', postproc_startyear)
        obs_endyear   = cfg['observations'].get('endyear', postproc_endyear)
        has_obs = True


    # Load case options and treatment options from config file
    case_options = {}
    treatment_options = {}
    
    if 'case_options' in cfg:
        case_options = cfg['case_options'].copy()
   
    if 'add_parameter' in cfg:
        add_parameter = cfg['add_parameter'].copy()

    if 'treatment_options' in cfg:
        treatment_options = cfg['treatment_options'].copy()
        nyears_treatment = cfg['treatment_options']['nyears']
    
    # Wipe the temp directory
    #APW this might be  throwing an error where temp doesn't exist (but also when trying to copy files to temp) 
    os.system('rm temp/*')
    if (runtype == 'site'):
        # Check to see if all reqested sites exist
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
                print('Warning:  number of processors greater than number '\
                    ,'of grid cells. Setting numproc = ',numproc)
        else:
            point_list = []
            print('Running with lat/lon bounding box')
            print('Lat: ', lat_bounds)
            print('Lon: ', lon_bounds)

    #APW: looks duplicated below, commenting out here
    ##Construct the list of compsets and suppring information
    #compset_type="I"
    #if (use_cpl_bypass):
    #    compset_type='ICB'
    # Construct the list of compsets and supporting information
    twophase=False
    compset_base=nutrients+nutrient_comp+soil_decomp+'BC'
    if (use_fates):
        compset_base='ELMFATES'
    if (use_crop):
        compset_base='ELMCNCROP'
    compset_type="I"
    if (use_cpl_bypass):
        compset_type='ICB'
    elif ((mettype != 'site' or 'PR-LUQ' in sites) and nyears_trans != 0):
        twophase=True       # if using DATM and reanalysis, split into 2 cases

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

    # Add treatment cases
    if ('suffix' in treatment_options.keys()):
        for t in range(0,len(treatment_options['suffix'])):
            nyears.append(nyears_treatment)
            istreatment = np.append(istreatment, 1)
            depends = np.append(depends, ncases_pretreatment-1)
            compsets.append(compsets[-1])
            suffix.append(treatment_options['suffix'][t])
            startyear.append(startyear_treatment)

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
            machine=machine, exeroot=exeroot, suffix=mysuffix, queue=queue, project=project,  \
            res=res, nyears=nyears[c],startyear=startyear[c], region_name=region_name, \
            lat_bounds=lat_bounds, lon_bounds=lon_bounds, np=numproc, point_list=point_list, \
            olmtdir=scriptdir)
        #Save the other site names in first site's cases (for use in multi-site calibration)
        if site == sites[0]:
            cases[c].all_sites = [s for s in sites]

        # Create the case
        cases[c].create_case()
        cases[c].case_options={}
        if (site != ''):
            cases[c].siteinfo = siteinfo[site]

        # Get the namelist options for this case
        for key in case_options.keys():
            if isinstance(case_options[key], list):
                cases[c].case_options[key] = case_options[key][c]
            else:
                cases[c].case_options[key] = case_options[key]

        # Add the treatment options (must be list format)
        if (istreatment[c]):
            for key in treatment_options.keys():
                cases[c].case_options[key] = treatment_options[key][c-ncases_pretreatment]
        # Other options
        cases[c].nutrients = nutrients
        cases[c].nutrient_comp = nutrient_comp
        cases[c].soil_decomp = soil_decomp
        if (use_fates):
            cases[c].fates_pft=fates_pft
            cases[c].pft_duplicates = pft_duplicates

        # Set the custom parameter files
        if ('fates_paramfile' in case_options):
            cases[c].fates_paramfile = case_options['fates_paramfile']
        if ('paramfile' in case_options):
            cases[c].paramfile = case_options['paramfile']
        if ('add_parameter' in cfg):
            cases[c].add_parameter = add_parameter

        # Get forcing information
        print('Getting forcing information')
        if ('phase2' in suffix[c]):
            # Set the starting year from the last case
            cases[c].startyear = cases[c-1].startyear+cases[c-1].run_n
        if (metdir != ''):
            cases[c].get_forcing(mettype=mettype, metdir=metdir)
        else:
            cases[c].get_forcing(mettype=mettype)

        # Set the initial data file (if depends on previous case)
        cases[c].dependcase=''
        if (depends[c] >= 0):
            # Set the iniial data file from the last year of the prev case
            finidat_year = cases[depends[c]].run_n+1
            if ('20TR' in cases[depends[c]].compset or 'trans' in cases[depends[c]].compset):
                finidat_year = 1850+cases[depends[c]].run_n
            cases[c].set_finidat_file(finidat_case=cases[depends[c]].casename, \
                  finidat_year=finidat_year)
            cases[c].dependcase = cases[depends[c]].casename

        # Set postprocessing variables (final case or treatment case)
        if (c == ncases-1 or istreatment[c]) and 'postprocessing' in cfg:
            cases[c].postproc_vars = postproc_vars
            cases[c].postproc_startyear = postproc_startyear
            cases[c].postproc_endyear = postproc_endyear
            cases[c].postproc_freq = postproc_freq
            # Also get the observations if requested, use postproc
            if (has_obs and site != ''):
                cases[c].obs = {}
                cases[c].obs_err = {}
                for v in obs_vars:
                    if not v in postproc_vars:
                        print('Adding observation variable to postprocessing: '+v)
                        cases[c].postproc_vars.append(v)
                    print('Getting observations for variable: '+v)
                    cases[c].get_fluxnet_obs(site=site,tstep=postproc_freq,ystart=obs_startyear, \
                        yend=obs_endyear,fluxnet_var=v, myobsdir=obs_dir)
        else:
            cases[c].postproc_vars=[]
        print('Postproc_vars: '+str(cases[c].postproc_vars))

        # Set up the case (surface, domain and pftdata)
        print('Setting up case for site: '+site)
        cases[c].setup_case()
        if (c == 0):
            # Get the surface and domain data 
            cases[c].setup_domain_surfdata(makesurfdat=True,makedomain=True)
        if (ensemble):
            if (site == sites[0] and c == 0):
                # Get the ensemble file from the first site and case
                cases[c].setup_ensemble(parm_list=parm_list,np_ensemble=np_ensemble,nsamples=nsamples, \
                    ensemble_file = ensemble_file, obs=cases[c].obs, obs_err=cases[c].obs_err)
                ensemble_file = cases[c].ensemble_file
            else:
                # Use the ensemble file for subsequent cases and sites
                cases[c].setup_ensemble(parm_list=parm_list,np_ensemble=np_ensemble,nsamples=nsamples, \
                    ensemble_file = ensemble_file, obs=cases[c].obs, obs_err=cases[c].obs_err)
        if ('20TR' in compsets[c] and not use_fates):
            # Get the dynamic PFT data
            cases[c].mask_grid = cases[0].mask_grid          #Get the mask from the first case
            cases[c].setup_domain_surfdata(makepftdyn=True)

        # Build the case
        print('Building case')
        cases[c].build_case()
    
        # Submit the case
        print('Submitting case')
        jobnum_depend=-1
        if (depends[c] >= 0):
            jobnum_depend = jobnum[depends[c]]
        # Set exeroot for all subsequent cases/sites so we don't have to rebuild
        if (depends[c] < 0 and site == sites[0]):
            exeroot = cases[c].exeroot
        if (ensemble):
            multisite_scripts[c] = cases[c].create_multisite_script([site], scriptdir)
            jobnum[c] = cases[c].submit_case(depend=jobnum_depend, \
                ensemble=ensemble,multisite_script=multisite_scripts[c])
        else:
            if (site == sites[0]):
                # Always use the multi-site script even for one site
                multisite_scripts[c] = cases[c].create_multisite_script(sites, scriptdir)
            if (site == sites[nsites-1]):
                jobnum[c] = cases[c].submit_case(depend=jobnum_depend, \
                    ensemble=ensemble,multisite_script=multisite_scripts[c])
        # Return to script directory
        os.chdir(scriptdir)


    #archive this script (based on name of first case)
    #archive_fname='./archive/'+cases[0].casename.replace('_ad_spinup','')
    #archive_fname=archive_fname.replace('1850','').replace('20TR','')+'_'+machine
    #if (nsites > 1):
    #    archive_fname = archive_fname.replace(site,'multisite')
    #os.system('mkdir -p archive')
    #if (ensemble):
    #    archive_fname = archive_fname+'_ensemble'
    #os.system('cp '+__file__+' '+archive_fname+'.py')


if __name__ == "__main__":
    main()
