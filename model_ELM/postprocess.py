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

def _infer_hist_info_from_files(file_list_all):
    if len(file_list_all) == 0:
        return None, 0, 12
    with Dataset(file_list_all[0], 'r') as ds:
        if 'time' not in ds.dimensions:
            return None, 0, 12
        nperyear = max(len(ds.dimensions['time']), 1)
    if (nperyear == 12):
        hist_nhtfrq = 0
    else:
        hist_nhtfrq = -int(round(8760.0/nperyear))
    return nperyear, hist_nhtfrq, nperyear

def _history_file_year(fname, hist_nhtfrq):
    match = re.search(r'\.(\d{4})(?:-|$)', os.path.basename(fname))
    if match:
        return int(match.group(1))
    if (hist_nhtfrq == 0):
        return int(fname.split('-')[-2][-4:])
    return int(fname.split('-')[-4][-4:])

def _postprocess_file_list(self, rundir, hnum, startyear=-1, endyear=9999):
    pattern = os.path.join(rundir, self.casename+'.elm.h'+str(hnum)+'.*.nc')
    file_list_all = np.sort(glob.glob(pattern))
    try:
        hist_mfilt, hist_nhtfrq, nperyear = _read_hist_info(rundir, hnum)
    except ValueError:
        hist_mfilt, hist_nhtfrq, nperyear = _infer_hist_info_from_files(file_list_all)
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

def _case_is_peatlands(self):
    sitegroup = str(getattr(self, 'sitegroup', '')).strip().strip("'\"")
    return sitegroup.lower() == 'peatlands'

def _requested_peatlands_topounit(self):
    topounit_value = str(getattr(self, 'postproc_topounit', -1)).strip().strip("'\"")
    topounit = int(topounit_value)
    if topounit < 0 and hasattr(self, 'siteinfo'):
        topounit = int(str(self.siteinfo.get('topounit', -1)).strip().strip("'\""))
    if topounit < 0:
        topounit = 1
    return topounit

def _history_topounits_for_request(topounit):
    # Peatlands topoindices are zero-based in Peatlands_pftdata.txt/configs,
    # while ELM's pfts1d_topounit history coordinate is one-based.
    # Peatlands topoindex 1 represents the bog aggregate: hollow+hummock.
    if int(topounit) == 1:
        return [2, 3]
    return [int(topounit) + 1]

def _available_peatlands_history_topounits(pft_topounit=None, pft_active=None, pft_weight=None,
        col_topounit=None, col_active=None, col_weight=None):
    available = []

    def add_topounits(topounit_values, active_values, weight_values):
        if topounit_values is None:
            return
        topounit_values = np.asarray(topounit_values, dtype=int)
        mask = np.ones(topounit_values.shape, dtype=bool)
        if active_values is not None:
            mask = mask & (np.asarray(active_values, dtype=int) > 0)
        if weight_values is not None:
            mask = mask & (np.asarray(weight_values, dtype=float) > 0.0)
        for value in topounit_values[mask]:
            value = int(value)
            if value > 0 and value not in available:
                available.append(value)

    add_topounits(pft_topounit, pft_active, pft_weight)
    add_topounits(col_topounit, col_active, col_weight)
    return available

def _resolve_peatlands_topounit_request(topounit, pft_topounit=None, pft_active=None,
        pft_weight=None, col_topounit=None, col_active=None, col_weight=None):
    available = _available_peatlands_history_topounits(
            pft_topounit=pft_topounit, pft_active=pft_active, pft_weight=pft_weight,
            col_topounit=col_topounit, col_active=col_active, col_weight=col_weight)
    if len(available) == 1:
        return 0, available
    return topounit, _history_topounits_for_request(topounit)

def _unique_preserve_order(values):
    out = []
    for value in values:
        if value not in out:
            out.append(value)
    return out

def _aggregate_history_pfts(data, pft_topounit, pft_type, pft_weight, pft_active,
        requested_topounits, npfts=22):
    values_out = np.ma.masked_all((data.shape[0], npfts), dtype=float)
    pft_topounit = np.asarray(pft_topounit, dtype=int)
    pft_type = np.asarray(pft_type, dtype=int)
    pft_weight = np.asarray(pft_weight, dtype=float)
    pft_active = np.asarray(pft_active, dtype=int)
    requested_topounits = np.asarray(requested_topounits, dtype=int)
    for pft in range(npfts):
        mask = (
            np.isin(pft_topounit, requested_topounits) &
            (pft_type == pft) &
            (pft_active > 0) &
            (pft_weight > 0.0)
        )
        if not np.any(mask):
            continue
        weights = pft_weight[mask]
        values_out[:, pft] = np.ma.average(data[:, mask], axis=1, weights=weights)
    return values_out

def _aggregate_history_columns(data, col_topounit, col_weight, col_active, requested_topounits):
    col_topounit = np.asarray(col_topounit, dtype=int)
    col_weight = np.asarray(col_weight, dtype=float)
    col_active = np.asarray(col_active, dtype=int)
    requested_topounits = np.asarray(requested_topounits, dtype=int)
    mask = (
        np.isin(col_topounit, requested_topounits) &
        (col_active > 0) &
        (col_weight > 0.0)
    )
    if not np.any(mask):
        return np.ma.masked_all((data.shape[0],), dtype=float)
    return np.ma.average(data[:, mask], axis=1, weights=col_weight[mask])

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

def write_peatlands_pft_postprocessed_netcdf(self, filename='', startyear=-1, endyear=9999,
        gindex=0, xindex=0, yindex=0):
    """Write Peatlands postprocessed h2 output as time x 22-PFT arrays."""
    if not _case_is_peatlands(self):
        return ''
    if len(getattr(self, 'postproc_vars', [])) == 0:
        return ''

    rundir = _postprocess_rundir(self, ens_num=0)
    file_list, firstyear, _, hist_nhtfrq, nperyear = _postprocess_file_list(
            self, rundir, 2, startyear=startyear, endyear=endyear)
    if len(file_list) == 0:
        print('No h2 PFT output files found for Peatlands postprocessed NetCDF in '+rundir)
        return ''

    topounit = _requested_peatlands_topounit(self)
    requested_vars = _unique_preserve_order([
        get_postproc_basevar(v) for v in self.postproc_vars
    ])
    dailytomonthly = (str(getattr(self, 'postproc_freq', '')).lower() == 'monthly')
    annualmean = (str(getattr(self, 'postproc_freq', '')).lower() == 'annual')

    pft_values_by_var = {}
    col_values_by_var = {}
    units_by_var = {}
    pft_topounit = None
    pft_type = None
    pft_weight = None
    pft_weight_name = ''
    pft_active = None
    col_topounit = None
    col_weight = None
    col_weight_name = ''
    col_active = None
    skipped_vars = set()
    for fname in file_list:
        with Dataset(fname, 'r') as ds:
            if pft_topounit is None:
                if 'pfts1d_topounit' in ds.variables:
                    pft_topounit = np.asarray(ds.variables['pfts1d_topounit'][:])
                    pft_type = np.asarray(ds.variables['pfts1d_itype_veg'][:])
                    if 'pfts1d_wtgcell' in ds.variables:
                        pft_weight = np.asarray(ds.variables['pfts1d_wtgcell'][:])
                        pft_weight_name = 'pfts1d_wtgcell'
                    else:
                        pft_weight = np.asarray(ds.variables['pfts1d_wttopounit'][:])
                        pft_weight_name = 'pfts1d_wttopounit'
                    if 'pfts1d_active' in ds.variables:
                        pft_active = np.asarray(ds.variables['pfts1d_active'][:])
                    else:
                        pft_active = np.ones_like(pft_type)
            if col_topounit is None:
                if 'cols1d_topounit' in ds.variables:
                    col_topounit = np.asarray(ds.variables['cols1d_topounit'][:])
                    if 'cols1d_wtgcell' in ds.variables:
                        col_weight = np.asarray(ds.variables['cols1d_wtgcell'][:])
                        col_weight_name = 'cols1d_wtgcell'
                    else:
                        col_weight = np.asarray(ds.variables['cols1d_wttopounit'][:])
                        col_weight_name = 'cols1d_wttopounit'
                    if 'cols1d_active' in ds.variables:
                        col_active = np.asarray(ds.variables['cols1d_active'][:])
                    else:
                        col_active = np.ones_like(col_topounit)
            for var in requested_vars:
                if var not in ds.variables:
                    continue
                ncvar = ds.variables[var]
                if var not in units_by_var:
                    units_by_var[var] = getattr(ncvar, 'units', '')
                if 'pft' in ncvar.dimensions:
                    pft_axis = ncvar.dimensions.index('pft')
                    if pft_axis != 1 or len(ncvar.dimensions) != 2:
                        if var not in skipped_vars:
                            print('Skipping Peatlands PFT postprocessing for '+var+
                                    ': expected dimensions (time,pft), found '+str(ncvar.dimensions))
                            skipped_vars.add(var)
                        continue
                    pft_values_by_var.setdefault(var, []).append(np.ma.asarray(ncvar[:]))
                elif 'column' in ncvar.dimensions:
                    col_axis = ncvar.dimensions.index('column')
                    if col_axis != 1 or len(ncvar.dimensions) != 2:
                        if var not in skipped_vars:
                            print('Skipping Peatlands column postprocessing for '+var+
                                    ': expected dimensions (time,column), found '+str(ncvar.dimensions))
                            skipped_vars.add(var)
                        continue
                    col_values_by_var.setdefault(var, []).append(np.ma.asarray(ncvar[:]))
                else:
                    if var not in skipped_vars:
                        print('Skipping Peatlands PFT postprocessing for '+var+
                                ': expected pft or column dimension, found '+str(ncvar.dimensions))
                        skipped_vars.add(var)

    if len(pft_values_by_var) == 0 and len(col_values_by_var) == 0:
        print('No requested PFT or column variables found in h2 output for Peatlands postprocessed NetCDF')
        return ''

    topounit, history_topounits = _resolve_peatlands_topounit_request(
            topounit, pft_topounit=pft_topounit, pft_active=pft_active,
            pft_weight=pft_weight, col_topounit=col_topounit, col_active=col_active,
            col_weight=col_weight)

    processed_pft = {}
    processed_col = {}
    taxis = None
    units_out = {}
    for var, chunks in pft_values_by_var.items():
        raw = np.ma.concatenate(chunks, axis=0)
        pft_values = _aggregate_history_pfts(raw, pft_topounit, pft_type, pft_weight,
                pft_active, history_topounits, npfts=22)
        finalized = []
        for pft in range(22):
            values_out, taxis_var, units = _finalize_postprocess_values(
                    self, pft_values[:, pft], units_by_var.get(var, ''), firstyear,
                    hist_nhtfrq, nperyear, dailytomonthly=dailytomonthly,
                    annualmean=annualmean)
            finalized.append(values_out)
            taxis = taxis_var
            units_out[var] = units
        processed_pft[var] = np.ma.vstack(finalized).T
    for var, chunks in col_values_by_var.items():
        raw = np.ma.concatenate(chunks, axis=0)
        col_values = _aggregate_history_columns(raw, col_topounit, col_weight,
                col_active, history_topounits)
        values_out, taxis_var, units = _finalize_postprocess_values(
                self, col_values, units_by_var.get(var, ''), firstyear,
                hist_nhtfrq, nperyear, dailytomonthly=dailytomonthly,
                annualmean=annualmean)
        processed_col[var] = values_out
        taxis = taxis_var
        units_out[var] = units

    if filename == '':
        outdir = os.path.join(rundir, '..', 'diagnostics')
        os.makedirs(outdir, exist_ok=True)
        filename = os.path.join(outdir, self.casename+
                '_peatlands_topounit'+str(topounit)+'_pft_postprocessed.nc')

    with Dataset(filename, 'w') as ds_out:
        ds_out.createDimension('time', len(taxis))
        ds_out.createDimension('natpft', 22)
        time_var = ds_out.createVariable('time', 'f8', ('time',))
        time_var[:] = taxis
        time_var.long_name = 'postprocessed time axis'
        pft_var = ds_out.createVariable('natpft', 'i4', ('natpft',))
        pft_var[:] = np.arange(22)
        pft_var.long_name = 'ELM natural vegetation PFT index'
        for var, values in processed_pft.items():
            outvar = ds_out.createVariable(var, 'f8', ('time', 'natpft'),
                    fill_value=1.0e36)
            outvar[:, :] = np.ma.filled(values, 1.0e36)
            outvar.units = units_out.get(var, '')
            outvar.long_name = var+' aggregated to requested Peatlands topounit PFT classes'
        for var, values in processed_col.items():
            outvar = ds_out.createVariable(var, 'f8', ('time',), fill_value=1.0e36)
            outvar[:] = np.ma.filled(values, 1.0e36)
            outvar.units = units_out.get(var, '')
            outvar.long_name = var+' aggregated to requested Peatlands topounit columns'
        ds_out.title = 'OLMT Peatlands topounit postprocessed output'
        ds_out.case = getattr(self, 'casename', '')
        ds_out.site = getattr(self, 'site', '')
        ds_out.sitegroup = getattr(self, 'sitegroup', '')
        ds_out.requested_topounit = int(topounit)
        ds_out.requested_topounit_indexing = 'zero-based Peatlands topoindex'
        ds_out.history_topounits = ','.join([str(v) for v in history_topounits])
        ds_out.history_topounit_indexing = 'one-based ELM pfts1d_topounit'
        ds_out.postprocessing_frequency = getattr(self, 'postproc_freq', '')
        ds_out.postprocessing_variables = ','.join(requested_vars)
        ds_out.pft_weight_variable = pft_weight_name
        ds_out.column_weight_variable = col_weight_name
    print('Wrote Peatlands PFT postprocessed NetCDF output: '+filename)
    return filename

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
    rundir = _postprocess_rundir(self, ens_num=ens_num)
    os.chdir(rundir)
    requested_startyear = startyear
    requested_endyear = endyear
    basevar = get_postproc_basevar(var)
    indexed_var = is_pft_var(var) or is_col_var(var)
    file_list, firstyear, endyear, hist_nhtfrq, nperyear = _postprocess_file_list(
            self, rundir, hnum, startyear=startyear, endyear=endyear)
    if len(file_list) == 0 and hnum == 1 and not indexed_var:
        file_list, firstyear, endyear, hist_nhtfrq, nperyear = _postprocess_file_list(
                self, rundir, 0, startyear=requested_startyear, endyear=requested_endyear)
        if len(file_list) > 0:
            hnum = 0
    startyear = firstyear
    if len(file_list) == 0:
        raise FileNotFoundError('No output files found for variable '+var+' in '+rundir)
    try:
        values, units = read_postprocess_values(file_list, basevar, var, index=index, \
                gindex=gindex, xindex=xindex, yindex=yindex)
    except KeyError as err:
        if hnum != 1 or indexed_var:
            raise
        fallback_file_list, fallback_firstyear, fallback_endyear, fallback_hist_nhtfrq, \
                fallback_nperyear = _postprocess_file_list(
                        self, rundir, 0, startyear=requested_startyear,
                        endyear=requested_endyear)
        if len(fallback_file_list) == 0:
            raise
        try:
            values, units = read_postprocess_values(fallback_file_list, basevar, var,
                    index=index, gindex=gindex, xindex=xindex, yindex=yindex)
        except KeyError:
            raise err
        file_list = fallback_file_list
        startyear = fallback_firstyear
        endyear = fallback_endyear
        hist_nhtfrq = fallback_hist_nhtfrq
        nperyear = fallback_nperyear
        hnum = 0
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
