import socket, os, sys, re, glob
import subprocess
import numpy as np

#Function to return default directories for supported machines
def get_machine_info(machine_name=''):
    queue = ''
    project = ''
    hostname = ''
    apptainer_bind = ''
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
        hostname = 'baseline.ccs.ornl.gov'
        queue = 'batch'
        project = 'CLI185'
        apptainer_bind = '/gpfs/wolf2'
    elif ('cades' in machine_name):
        rootdir = '/lustre/or-scratch/cades-ccsi/'+os.environ['USER']
        inputdata = '/lustre/or-scratch/cades-ccsi/proj-shared/project_acme/e3sm_inputdata/'
        machine = 'cades'
        hostname = 'or-login.ornl.gov'
        queue = 'batch'
        apptainer_bind = '/lustre/or-scratch'
    if ('pflogin' in machine_name or 'pathfinder' in machine_name):
        rootdir = '/projects/hpcl-cli185/users/'+os.environ['USER']
        inputdata = '/projects/hpcl-cli185/world-shared/e3sm/inputdata/'
        machine = 'pathfinder'
        hostname = 'pflogin.ornl.gov'
        queue = 'normal'
        apptainer_bind = '/projects/hpcl-cli185'
    elif  ('chrlogin' in machine_name or 'chrysalis' in machine_name):
        rootdir = '/lcrc/group/e3sm/'+os.environ['USER']+'/scratch'
        inputdata = '/lcrc/group/e3sm/ccsm-data/inputdata'
        machine = 'chrysalis'
        hostname = 'chrysalis.lcrc.anl.gov'
        queue = 'compute'
        project = 'e3sm'
    elif ('pm-cpu' in machine_name or 'login' in machine_name):
        rootdir = os.environ['SCRATCH']
        inputdata = '/global/cfs/cdirs/e3sm/inputdata'
        machine = 'pm-cpu'
        hostname = 'saul-p1.nersc.gov'
        queue = 'regular'
        project = 'e3sm'
    elif ('ubuntu' in machine_name or 'linux-generic' in machine_name):
        rootdir = os.environ['HOME']+'/models'
        inputdata = rootdir + '/inputdata'
        machine = 'linux-generic'
    else:
        if (not 'docker' in machine_name):
            print('Machine not detected.  Assuming docker')
        rootdir = '/output'
        inputdata = '/inputdata'
        machine = 'docker'
    return machine, rootdir, inputdata, queue, project, hostname, apptainer_bind

#Function to get the available site groups
def get_sitegroups(inputdata, sftp=None):
    PTCLM = inputdata + '/lnd/clm2/PTCLM'
    pattern = os.path.join(PTCLM, "*_sitedata.*")
    prefixes = []

    if sftp is not None:
        # Remote: use sftp.listdir to get files
        try:
            files = sftp.listdir(PTCLM)
            for file in files:
                if "_sitedata" in file:
                    prefix = file.split("_sitedata")[0]
                    prefixes.append(prefix)
        except Exception as e:
            print(f"Error reading remote PTCLM directory: {e}")
    else:
        # Local: use glob
        files = glob.glob(pattern)
        for file in files:
            basename = os.path.basename(file)
            if "_sitedata" in basename:
                prefix = basename.split("_sitedata")[0]
                prefixes.append(prefix)
    return prefixes


def get_site_info(inputdata, sitegroup='AmeriFlux', sftp=None, use_crop=False):
    base = inputdata + '/lnd/clm2/PTCLM/'
    siteinfo = {}
    # Helper to read lines from local or remote file
    def readlines(path):
        if sftp is not None:
            with sftp.open(path, 'r') as f:
                return f.readlines()
        else:
            with open(path, 'r') as f:
                return f.readlines()

    # sitedata
    sitedata_path = base + sitegroup + '_sitedata.txt'
    lines = readlines(sitedata_path)
    snum = 0
    if(use_crop):
        npfts = 15
    else:
        npfts = 17
    for s in lines:
        if snum > 0:
            sitename = s.split(',')[0]
            siteinfo[sitename] = {}
            siteinfo[sitename]['lon'] = float(s.split(',')[3])
            siteinfo[sitename]['lat'] = float(s.split(',')[4])
            siteinfo[sitename]['PCT_NAT_PFT'] = np.zeros([npfts], float)
            siteinfo[sitename]['PCT_SAND'] = -999
            siteinfo[sitename]['PCT_CLAY'] = -999
            if(use_crop):
                siteinfo[sitename]['PCT_CFT'] = np.zeros([36],float)
        snum += 1

    # pftdata
    pftdata_path = base + sitegroup + '_pftdata.txt'
    lines = readlines(pftdata_path)
    snum = 0
    for s in lines:
        if snum > 0:
            sitename = s.split(',')[0]
            for p in range(0, 5):
                pindex = int(s[:-1].split(',')[p * 2 + 2])
                ppct = float(s[:-1].split(',')[p * 2 + 1])
                if ppct > 0:
                    #siteinfo[sitename]['PCT_NAT_PFT'][pindex] = ppct
                    if(use_crop):
                        if (pindex < 15):
                            siteinfo[sitename]['PCT_NAT_PFT'][pindex] = ppct
                        else:
                            siteinfo[sitename]['PCT_CFT'][pindex - 15] = ppct
                    else:
                        siteinfo[sitename]['PCT_NAT_PFT'][pindex] = ppct
        snum += 1

    # soildata
    soildata_path = base + sitegroup + '_soildata.txt'
    lines = readlines(soildata_path)
    snum = 0
    for s in lines:
        if snum > 0:
            sitename = s[:-1].split(',')[0]
            siteinfo[sitename]['PCT_SAND'] = float(s[:-1].split(',')[4])
            siteinfo[sitename]['PCT_CLAY'] = float(s[:-1].split(',')[5])
        snum += 1

    #Land use change information
    for sitename in siteinfo.keys():
        landuse_path = base + sitename + '_dynpftdata.txt'
        if os.path.exists(landuse_path):
            lines = readlines(landuse_path)
            snum = 0
            siteinfo[sitename]['transitions'] = {}
            for s in lines:
                if snum > 0:
                    trans_year = s.split(',')[0]
                    siteinfo[sitename]['transitions'][trans_year] = {}
                    siteinfo[sitename]['transitions'][trans_year]['PCT_NAT_PFT'] = np.zeros([npfts], float)
                    for p in range(0, 5):
                        pindex = int(s[:-1].split(',')[p * 2 + 2])
                        ppct = float(s[:-1].split(',')[p * 2 + 1])
                        harvpct = float(s[:-1].split(',')[11])
                        if ppct > 0:
                            if(use_crop):
                                if (pindex < 15):
                                    siteinfo[sitename]['transitions'][trans_year]['PCT_NAT_PFT'][pindex] = ppct
                                else:
                                    siteinfo[sitename]['transitions'][trans_year]['PCT_CFT'][pindex - 15] = ppct
                            else:
                                siteinfo[sitename]['transitions'][trans_year]['PCT_NAT_PFT'][pindex] = ppct
                        siteinfo[sitename]['transitions'][trans_year]['HARVEST'] = harvpct
                snum += 1
        else:
            siteinfo[sitename]['transitions'] = {}
    


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
        return ['NEE','NBP','TLAI','TOTSOMC','CWDC','TOTLITC','TOTECOSYSC','NPP','GPP','QVEGT','QVEGE','EFLX_LH_TOT','TOTVEGC_ABG',\
                'TOTVEGC','QOVER','QSOIL','XR','ER','AR','HR','FSH','SNOWDP','ZWT','CPOOL','NPOOL','PPOOL','FPG','FPI','NDEP_TO_SMINN', \
                'NFIX_TO_SMINN','NEP','QDRAI','QRUNOFF']

def docker_to_host_path(path):
    # Only translate if path starts with /inputdata
    if path.startswith("/inputdata"):
        return path.replace("/inputdata", "/Users/zdr/models/inputdata", 1)
    return path
