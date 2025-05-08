import socket, os, sys, re, glob
import subprocess
import numpy as np

#Function to return default directories for supported machines
def get_machine_info(machine_name=''):
    queue = ''
    project = ''
    if (machine_name == ''):
        if ('HOSTNAME' in os.environ):
            machine_name=os.environ['HOSTNAME']
        else:
            result = subprocess.run(['hostname'], capture_output=True, text=True)
            machine_name = result.stdout
    if ('baseline' in machine_name):
        rootdir = '/gpfs/wolf2/cades/cli185/scratch/'+os.environ['USER']
        inputdata = '/gpfs/wolf2/cades/cli185/world-shared/e3sm/inputdata/'
        machine = 'cades-baseline'
        queue = 'batch'
        project = 'CLI185'
    elif  ('chrlogin' in machine_name or 'chrysalis' in machine_name):
        rootdir = '/lcrc/group/e3sm/'+os.environ['USER']+'/scratch'
        inputdata = '/lcrc/group/e3sm/ccsm-data/inputdata'
        machine = 'chrysalis'
        queue = 'compute'
        project = 'e3sm'
    elif ('pm-cpu' in machine_name or 'login' in machine_name):
        rootdir = os.environ['SCRATCH']
        inputdata = '/global/cfs/cdirs/e3sm/inputdata'
        machine = 'pm-cpu'
        queue = 'regular'
        project = 'e3sm'
    elif ('ubuntu' in machine_name or 'linux-generic' in machine_name):
        rootdir = os.environ['HOME']+'/models'
        inputdata = rootdir + '/inputdata'
        machine = 'linux-generic'
    elif ('docker' in machine_name):
        rootdir = '/output'
        inputdata = '/inputdata'
        machine = 'docker'
    else:
        print('Error:  Machine not detected.  Please specify machine name')
        sys.exit(1)
    return machine, rootdir, inputdata, queue, project

#Function to get the available site groups
def get_sitegroups(inputdata):
    PTCLM = inputdata+'/lnd/clm2/PTCLM'
    pattern = os.path.join(PTCLM, "*_sitedata.*")
    files = glob.glob(pattern)

    prefixes = []
    for file in files:
        basename = os.path.basename(file)
        if "_sitedata" in basename:
            prefix = basename.split("_sitedata")[0]
            prefixes.append(prefix)
    return prefixes


def get_site_info(inputdata, sitegroup='AmeriFlux'):
    sitegroup_file = open(inputdata+'/lnd/clm2/PTCLM/'+sitegroup+'_sitedata.txt')
    siteinfo={}
    snum=0
    for s in sitegroup_file:
        if snum > 0:
          sitename = s.split(',')[0]
          siteinfo[sitename]={}
          siteinfo[sitename]['lon'] = float(s.split(',')[3])
          siteinfo[sitename]['lat'] = float(s.split(',')[4])
          siteinfo[sitename]['PCT_NAT_PFT'] = np.zeros([17],float)
          siteinfo[sitename]['PCT_SAND']=-999
          siteinfo[sitename]['PCT_CLAY']=-999
        snum=snum+1
    sitegroup_file.close()
    sitegroup_pftfile = open(inputdata+'/lnd/clm2/PTCLM/'+sitegroup+'_pftdata.txt')
    snum = 0
    #PFTs.  TODO - allow crop PFTs
    for s in sitegroup_pftfile:
        if (snum > 0):
            sitename = s.split(',')[0]
            for p in range(0,5):
                pindex = int(s[:-1].split(',')[p*2+2])
                ppct   = float(s[:-1].split(',')[p*2+1])
                if (ppct > 0):
                    siteinfo[sitename]['PCT_NAT_PFT'][pindex] = ppct
        snum = snum+1
    sitegroup_pftfile.close()
    #Soil texture
    sitegroup_soilfile = open(inputdata+'/lnd/clm2/PTCLM/'+sitegroup+'_soildata.txt')
    snum = 0
    for s in sitegroup_soilfile:
        if (snum > 0):
            sitename = s[:-1].split(',')[0]
            siteinfo[sitename]['PCT_SAND'] = float(s[:-1].split(',')[4])
            siteinfo[sitename]['PCT_CLAY'] = float(s[:-1].split(',')[5])
        snum=snum+1
    sitegroup_soilfile.close()
    return siteinfo

def get_point_list(fname):
    myfile = open(fname, 'r')
    snum=0
    points=[]
    for s in myfile:
        if snum == 0:
            header=s.split(',')
            #header=re.split(r'[,\s\t]+', s)
            hnum=0
            for h in header:
                if ('lat' in h):
                    latcol=hnum
                if ('lon' in h):
                    loncol=hnum
                hnum=hnum+1
        else:
            line=s.split(',') #line=re.split(r'[,\s\t]+', s)
            points.append((float(line[latcol]),float(line[loncol])))
        snum=snum+1
    return points
#TODO:  Function to return met data path for various options

def get_default_diag_vars(nutrients, use_fates):
    if (nutrients == 'none'):
        return ['TLAI','FPSN','QVEGT','QVEGE','QSOIL','EFLX_LH_TOT','FSH','SNOWDP','QRUNOFF','QDRAI','QOVER']
    else:
        return ['NEE','NBP','TLAI','TOTSOMC','CWDC','TOTLITC','TOTECOSYSC','NPP','GPP','QVEGT','QVEGE','EFLX_LH_TOT']
