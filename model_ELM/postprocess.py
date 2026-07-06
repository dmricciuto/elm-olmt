from netCDF4 import Dataset
import xarray as xr
import re,os,glob
import numpy as np
import matplotlib.pyplot as plt

def get_postproc_basevar(var):
    return var.split('_pft')[0].split('_col')[0]

def is_pft_var(var):
    return '_pft' in var

def is_col_var(var):
    return '_col' in var

def do_dailytomonthly(values):
    dayspermonth=[31,28,31,30,31,30,31,31,30,31,30,31]
    npoints = len(values)
    nmonths = int(npoints/365*12)
    values_out = np.zeros([nmonths],float)
    index=0
    for m in range(0,nmonths):
        mind = m % 12
        values_out[m] = np.mean(values[index:index+dayspermonth[mind]])
        index = index+dayspermonth[mind]
    return values_out

def do_monthlytoannual(values):
    dayspermonth=[31,28,31,30,31,30,31,31,30,31,30,31]
    npoints = len(values)
    nyears = int(npoints/12)
    values_out = np.zeros([nyears],float)
    for y in range(0,nyears):
        values_out[y] = np.sum(values[y*12:(y+1)*12]*dayspermonth)/365
    return values_out

def do_timeaverage(values, nav):
   npoints = len(values)
   values_out = np.zeros([int(npoints/nav)],float)
   for t in range(0,int(npoints/nav)):
       values_out[t] = np.mean(values[t*nav:(t+1)*nav])
   return values_out

def read_postprocess_values(file_list, basevar, var, index=0, gindex=0, xindex=0, yindex=0):
    """Read and concatenate the requested output slice without shelling out to NCO."""
    values = []
    units = ''
    for fname in file_list:
        with Dataset(fname, 'r') as ds:
            if basevar not in ds.variables:
                raise KeyError('Variable '+basevar+' not found in '+fname)
            ncvar = ds.variables[basevar]
            if units == '':
                units = getattr(ncvar, 'units', '')
            ndim = len(ncvar.dimensions)
            if (ndim == 4):
                # 2D output with vertical structure
                data = ncvar[:, index, yindex, xindex]
            elif (ndim == 3):
                # 2D output or 1D output with vertical structure (currently assumes 1D)
                data = ncvar[:, index, gindex]
            else:
                # 1D output (unstructured grid)
                data = ncvar[:, gindex]
                if (is_pft_var(var) or is_col_var(var)):
                    data = ncvar[:, index]
            values.append(np.ma.asarray(data))
    if len(values) == 0:
        return np.ma.array([]), units
    return np.ma.concatenate(values), units

def _postprocess_rundir(self, ens_num=0):
    if (ens_num > 0):
        gst = str(100000+ens_num)[1:]
        return self.rundir_UQ+'/g'+gst
    return self.rundir

def _read_hist_info(rundir, hnum):
    hist_mfilt = None
    hist_nhtfrq = None
    lnd_in = open(os.path.join(rundir, 'lnd_in'))
    for s in lnd_in:
        if (s.split('=')[0].strip() == 'hist_mfilt'):
            hist_mfilt = int((s.split('=')[1].strip()).split(',')[hnum].strip())
        if (s.split('=')[0].strip() == 'hist_nhtfrq'):
            hist_nhtfrq = int((s.split('=')[1].strip()).split(',')[hnum].strip())
    lnd_in.close()
    if hist_nhtfrq is None:
        raise ValueError('hist_nhtfrq not found in '+os.path.join(rundir, 'lnd_in'))
    if (hist_nhtfrq == 0):
        nperyear=12
    else:
        nperyear = max(abs(8760/hist_nhtfrq), 1)
    return hist_mfilt, hist_nhtfrq, nperyear

def _history_file_year(fname, hist_nhtfrq):
    match = re.search(r'\.(\d{4})(?:-|$)', os.path.basename(fname))
    if match:
        return int(match.group(1))
    if (hist_nhtfrq == 0):
        return int(fname.split('-')[-2][-4:])
    return int(fname.split('-')[-4][-4:])

def _postprocess_file_list(self, rundir, hnum, startyear=-1, endyear=9999):
    hist_mfilt, hist_nhtfrq, nperyear = _read_hist_info(rundir, hnum)
    pattern = os.path.join(rundir, self.casename+'.elm.h'+str(hnum)+'.*.nc')
    file_list_all = np.sort(glob.glob(pattern))
    file_list = []
    firstyear = startyear
    lastyr = None
    for f in file_list_all:
        yr = _history_file_year(f, hist_nhtfrq)
        if (firstyear < 0):
            firstyear = yr
        if (endyear >= 9999):
            lastyr = yr
        if (yr >= firstyear and yr <= endyear):
            file_list.append(f)
    if (endyear >= 9999 and lastyr is not None):
        endyear=lastyr
        if (nperyear != 12):
            # If not monthly files, ignore the last file; it only represents one timestep.
            file_list = file_list[:-1]
    return file_list, firstyear, endyear, hist_nhtfrq, nperyear

def _postprocess_requests(self):
    requests = []
    for v in self.postproc_vars:
        hnum=1
        dailytomonthly=False
        annualmean=False
        if (self.postproc_freq == 'annual'):
            hnum=0
            annualmean=True
        elif (self.postproc_freq == 'monthly'):
            dailytomonthly=True
        elif (self.postproc_freq == 'daily' or self.postproc_freq == 'hourly'):
            pass
        else:
            continue
        mypfts=[0]
        if (is_pft_var(v)):
            hnum=2
            mypfts=self.postproc_pfts
        elif (is_col_var(v)):
            hnum=2
            mypfts=self.postproc_cols
        for p in mypfts:
            var_out = v
            if (is_pft_var(v) or is_col_var(v)):
                var_out = var_out+str(p)
            requests.append({
                'var': v,
                'basevar': get_postproc_basevar(v),
                'var_out': var_out,
                'index': p,
                'hnum': hnum,
                'dailytomonthly': dailytomonthly,
                'annualmean': annualmean,
            })
    return requests

def _slice_postprocess_array(data, var, index=0, gindex=0, xindex=0, yindex=0):
    if (data.ndim == 4):
        return data[:, index, yindex, xindex]
    if (data.ndim == 3):
        return data[:, index, gindex]
    values = data[:, gindex]
    if (is_pft_var(var) or is_col_var(var)):
        values = data[:, index]
    return values

def _postprocess_factor(units, annualmean=False):
    factor = 1.0
    if (units == 'gC/m^2/s' or units == 'gN/m^2/s' or units == 'gP/m^2/s'):
        nutrient = units[1]
        if annualmean:
            units = 'g'+nutrient+'/m^2/yr'
            factor = 24*3600*365.
        else:
            units = 'g'+nutrient+'/m^2/day'
            factor = 24*3600
    if (units == 'mm/s'):
        if annualmean:
            units = 'mm/yr'
            factor = 24*3600*365.
        else:
            units = 'mm/day'
            factor = 24*3600
    return factor, units

def _finalize_postprocess_values(self, values, units, startyear, hist_nhtfrq, nperyear,
        dailytomonthly=False, annualmean=False):
    factor, units = _postprocess_factor(units, annualmean=annualmean)
    values = values*factor
    if (dailytomonthly and hist_nhtfrq == -24):
        values_out = do_dailytomonthly(values)
        nperyear_out = 12
    elif (annualmean):
        if (hist_nhtfrq == 0):
            values_out = do_monthlytoannual(values)
        else:
            if (nperyear >= 1):
                values_out = do_timeaverage(values, int(nperyear))
            else:
                values_out = values[:]
        nperyear_out = 1
    else:
        if (self.postproc_timeaverage > 1):
            values_out = do_timeaverage(values, self.postproc_timeaverage)
            nperyear_out = int(nperyear/self.postproc_timeaverage)
        else:
            values_out = values[:]
            nperyear_out = nperyear
    taxis = np.zeros([len(values_out)],float)
    for t in range(0,len(values_out)):
        taxis[t] = startyear+t/nperyear_out
    return values_out, taxis, units

def postprocess_member(self, ens_num=0, startyear=-1, endyear=9999, gindex=0, xindex=0, yindex=0):
    """Postprocess all configured variables for one member with one pass per history stream."""
    requests = _postprocess_requests(self)
    if len(requests) == 0:
        return
    rundir = _postprocess_rundir(self, ens_num=ens_num)
    requests_by_hnum = {}
    for req in requests:
        requests_by_hnum.setdefault(req['hnum'], []).append(req)

    for hnum, h_requests in requests_by_hnum.items():
        file_list, firstyear, _, hist_nhtfrq, nperyear = _postprocess_file_list(
                self, rundir, hnum, startyear=startyear, endyear=endyear)
        if len(file_list) == 0:
            raise FileNotFoundError('No output files found for h'+str(hnum)+' in '+rundir)

        values_by_var = {}
        units_by_var = {}
        requests_by_basevar = {}
        for req in h_requests:
            values_by_var[req['var_out']] = []
            requests_by_basevar.setdefault(req['basevar'], []).append(req)

        for fname in file_list:
            with Dataset(fname, 'r') as ds:
                for basevar, base_requests in requests_by_basevar.items():
                    if basevar not in ds.variables:
                        raise KeyError('Variable '+basevar+' not found in '+fname)
                    ncvar = ds.variables[basevar]
                    if basevar not in units_by_var:
                        units_by_var[basevar] = getattr(ncvar, 'units', '')
                    data = np.ma.asarray(ncvar[:])
                    for req in base_requests:
                        values_by_var[req['var_out']].append(
                                _slice_postprocess_array(data, req['var'], index=req['index'],
                                    gindex=gindex, xindex=xindex, yindex=yindex))

        for req in h_requests:
            values = np.ma.concatenate(values_by_var[req['var_out']])
            values_out, taxis, _ = _finalize_postprocess_values(
                    self, values, units_by_var.get(req['basevar'], ''), firstyear,
                    hist_nhtfrq, nperyear, dailytomonthly=req['dailytomonthly'],
                    annualmean=req['annualmean'])
            if (ens_num > 0 and not req['var_out'] in self.output):
                self.output[req['var_out']] = np.zeros([len(values_out),self.nsamples],float)
            if (ens_num > 0):
                self.output[req['var_out']][:,ens_num-1] = values_out
            else:
                self.output[req['var_out']] = values_out
            self.output['taxis'] = taxis

# -- Get sorted list of .nc files --
def sorted_h0_files(directory):
    files = glob.glob(os.path.join(directory, '*.elm.h0.*-01-01-00000.nc'))
    return sorted(files, key=lambda f: int(re.search(r'\.(\d+)-01-01-00000\.nc', f).group(1)))

#Plot ad and final spinup together
def plot_spinup(self, plotvars=[]):
    if not plotvars:
        plotvars = ['NEE','TOTVEGC','TOTSOMC','GPP','NPP']
    fn_dir = self.rundir
    ad_dir = self.rundir.replace(self.casename,self.dependcase)
    print(fn_dir)
    print(ad_dir)
    #Ignore first file
    files1 = sorted_h0_files(ad_dir)[1:]
    files2 = sorted_h0_files(fn_dir)[1:]

    # -- Load datasets --
    ds1 = xr.open_mfdataset(files1, combine='nested', concat_dim='time')
    ds2 = xr.open_mfdataset(files2, combine='nested', concat_dim='time')

    # -- Extract years from filenames --
    years1 = [int(re.search(r'\.(\d+)-01-01-00000\.nc', f).group(1)) for f in files1]
    years2 = [int(re.search(r'\.(\d+)-01-01-00000\.nc', f).group(1)) for f in files2]
    years2_offset = [y + max(years1) for y in years2]  # offset second series

    # -- Assign time coordinates --
    ds1 = ds1.assign_coords(time=years1)
    ds2 = ds2.assign_coords(time=years2_offset)

    # -- Concatenate datasets --
    ds_combined = xr.concat([ds1, ds2], dim='time')

    for v in plotvars:
    # -- Extract and spatially average NEE --
        vals = ds_combined[v]
        vals_mean = vals.mean(dim=('lndgrid'))  # or 'gridcell' if applicable
        ylabel = v
        if (v == 'NEE' or v == 'GPP' or v == 'NPP'):
            #Convert to gC/m2/yr
            vals_mean=abs(vals_mean)*24*3600*365
        if (v == 'NEE'):
            #Log scale
            vals_mean = np.log10(vals_mean)
            ylabel = "log10(NEE)"

        transition_year = max(years1)

        # -- Plotting --
        plt.figure(figsize=(10, 5))
        plt.plot(ds_combined['time'], vals_mean, marker='o')
        plt.title(v+" during spinup")
        plt.xlabel("Year")
        plt.ylabel(ylabel)
        # Horizontal line at y=0
        if (v == 'NEE'):
            plt.axhline(0, color='red', linestyle='--', label='NEE threshold (1 gC/m2/yr)')

        # Vertical line at transition between dirs
        plt.axvline(transition_year, color='blue', linestyle=':', label=f'Spinup transition')

        plt.grid(True)
        os.system('mkdir -p '+self.rundir+'/../diagnostics')
        plt.savefig(self.rundir+'/../diagnostics/spinup_plot_'+v+'.png')
             
def postprocess(self, var, index=0, gindex=0, startyear=-1, endyear=9999, hnum=0, \
        dailytomonthly=False, annualmean=False,  meanseasonalcycle=False, \
        xindex=0,yindex=0, ens_num=0, plot=False):
    if (ens_num > 0):
        gst = str(100000+ens_num)[1:]
        rundir = self.rundir_UQ+'/g'+gst
    else:
        rundir = self.rundir
    os.chdir(rundir)
    lnd_in = open('./lnd_in')
    #Get history file info from lnd_in
    for s in lnd_in:
        if (s.split('=')[0].strip() == 'hist_mfilt'):
            hist_mfilt = int((s.split('=')[1].strip()).split(',')[hnum].strip())
        if (s.split('=')[0].strip() == 'hist_nhtfrq'):
            hist_nhtfrq = int((s.split('=')[1].strip()).split(',')[hnum].strip())
    lnd_in.close()
    if (hist_nhtfrq == 0):
      nperyear=12
    else:
      nperyear = max(abs(8760/hist_nhtfrq), 1)
    file_list_all = np.sort(glob.glob(self.casename+'.elm.h'+str(hnum)+'.*.nc'))
    file_list = []
    #Filter the requested years
    for f in file_list_all:
        if (hist_nhtfrq == 0):
            yr = int(f.split('-')[-2][-4:])
        else:
            yr = int(f.split('-')[-4][-4:])
        if (startyear < 0):
            startyear = yr
        if (endyear >= 9999):
            lastyr = yr
        if (yr >= startyear and yr <= endyear):
            file_list.append(f)
    if (endyear >= 9999):
        endyear=lastyr
        if (nperyear != 12):
          #If not monthly files, ignore the last file (it only represents a single timestep)
          file_list = file_list[:-1]
    if len(file_list) == 0:
        raise FileNotFoundError('No output files found for variable '+var+' in '+rundir)
    basevar = get_postproc_basevar(var)
    values, units = read_postprocess_values(file_list, basevar, var, index=index, \
            gindex=gindex, xindex=xindex, yindex=yindex)
    #change flux units
    factor = 1.0
    if (units == 'gC/m^2/s' or units == 'gN/m^2/s' or units == 'gP/m^2/s'):
        nutrient = units[1]
        if annualmean:
            units = 'g'+nutrient+'/m^2/yr'
            factor = 24*3600*365.
        else:
            units = 'g'+nutrient+'/m^2/day'
            factor = 24*3600
    if (units == 'mm/s'):
        if annualmean:
            units = 'mm/yr'
            factor = 24*3600*365.
        else:
            units = 'mm/day'
            factor = 24*3600

    values = values*factor

    if (dailytomonthly and hist_nhtfrq == -24):
      values_out = do_dailytomonthly(values)
      nperyear_out = 12
    elif (annualmean):
      if (hist_nhtfrq == 0):
          values_out = do_monthlytoannual(values)
      else:
          if (nperyear >= 1):
            values_out = do_timeaverage(values, int(nperyear))
      nperyear_out = 1
    else:
        if (self.postproc_timeaverage > 1):
            values_out = do_timeaverage(values, self.postproc_timeaverage)
            nperyear_out = int(nperyear/self.postproc_timeaverage)
        else:
            #no averaging requested
            values_out = values[:]
            nperyear_out = nperyear
    var_out = var
    if (is_pft_var(var) or is_col_var(var)):
        var_out = var_out+str(index)
    if (ens_num > 0 and not var_out in self.output):
        self.output[var_out] = np.zeros([len(values_out),self.nsamples],float)
    if (ens_num > 0):
        self.output[var_out][:,ens_num-1] = values_out
    else:
        self.output[var_out]=values_out
    self.output['taxis'] = np.zeros([len(values_out)],float)
    for t in range(0,len(values_out)):
        self.output['taxis'][t] = startyear+t/nperyear_out
    if (plot):
        os.system('mkdir -p '+self.rundir+'/../diagnostics')
        plt.plot(self.output['taxis'],self.output[var_out],'k')
        plt.ylabel(var_out+' ('+units+')')
        plt.xlabel('Years')
        plt.legend([var_out])
        plt.tight_layout()
        plt.savefig(self.rundir+'/../diagnostics/plot_'+var_out+'_'+str(startyear)+'-'+str(endyear)+'.png')
        plt.close()
