#o!/usr/bin/env python
import re, os, sys, csv, time, math
import numpy as np
from netCDF4 import Dataset
from geopy.distance import geodesic
from scipy.spatial import KDTree
import xarray as xr


#Function to return the indices of the nearest grid cell centers for a list of points
def get_pointindices_list(self, mylat, mylon, lat_grid, lon_grid, mask_grid=[]):
    self.shift_lon=False
    if (max(lon_grid.flatten()) > 180):
        self.shift_lon = True
    lon_grid[lon_grid > 180] -=360
    points = list(zip(lat_grid.flatten(),lon_grid.flatten()))
    #If mask given, only append land points
    if (len(mask_grid) == len(lat_grid)):
        maskf = mask_grid.flatten()
        for index, point in enumerate(points):
            if (maskf[index] == 0):
                points[index]=(-999,-999)
    original_shape = lat_grid.shape
    tree = KDTree(points)
    index_out = []
    for p in range(0,len(mylat)):
        target_point = (mylat[p], mylon[p])
        tree = KDTree(points)
        distance, index = tree.query(target_point)
        nearest_point = points[index]
        distance_km = geodesic(target_point, nearest_point).kilometers
        if (distance_km < 250):
            if (len(original_shape) > 1 and min(original_shape) > 1):
                # Convert the flattened index to a 2D index (row, column)
                row, col = np.unravel_index(index, original_shape)  
                index_out.append((row, col))
            else:
                index_out.append(index)
        else:
            print('Warning: Nearest gridcell to ',target_point)
            print(   'is ', distance_km, 'km away.  Not including')
    return index_out

def get_pointindices_bbox(self, lat_bounds, lon_bounds, lat_grid, lon_grid, mask_grid=[]):
    #Function to return all indices with a rectangular lat/lon bounding box
    # Ensure lon is within the range [-180, 180]
    lon_grid[lon_grid > 180] -= 360
    lon_bounds[lon_bounds > 180] -=360
    # Flatten the lat and lon grids to create a list of points
    points = list(zip(lat_grid.flatten(), lon_grid.flatten()))
    original_shape = lat_grid.shape
    index_out = []
    #If mask given, only append land points
    if (len(mask_grid) == len(lat_grid)):
        maskf = mask_grid.flatten()
    else:
        maskf = np.ones([len(points)],int)
    # Loop through all points in the grid and check if they fall within the bounding box
    for index, (lat, lon) in enumerate(points):
        if lat_bounds[0] <= lat <= lat_bounds[1] and lon_bounds[0] <= lon <= lon_bounds[1] \
                and maskf[index] > 0:
            if (len(original_shape) > 1 and min(original_shape) > 1):
                # Convert the flattened index to a 2D index (row, col)
                row, col = np.unravel_index(index, original_shape)
                index_out.append((row, col))
            else:
                index_out.append(index)
    return index_out


def subset_netcdf(self, index, input_file, output_file, keep2d=False):
    # Load the input NetCDF file
    original_ds = xr.open_dataset(input_file, mode='r')

    # Ensure index is always a list for consistent handling
    if not isinstance(index, list):
        index = [index]
    if len(index) == 0:
        original_ds.close()
        raise ValueError(f'No grid cells selected while subsetting {input_file}')

    # Check if index contains tuples (2D) or integers (1D)
    is_2d_index = len(index) > 0 and isinstance(index[0], (tuple, list))
    if is_2d_index:
        lat_indices = [lat for lat, lon in index]
        lon_indices = [lon for lat, lon in index]
    else:
        point_indices = np.asarray(index, dtype=int)

    if not is_2d_index:
        indexers = {}
        if 'ni' in original_ds.dims:
            indexers['ni'] = xr.DataArray(point_indices, dims='ni')
        if 'gridcell' in original_ds.dims:
            indexers['gridcell'] = xr.DataArray(point_indices, dims='gridcell')
        if indexers:
            ds_subset = original_ds.isel(indexers).load()
            original_ds.close()
            ds_subset.to_netcdf(output_file)
            ds_subset.close()
            return

    # Select the variable and apply subsetting if specified
    for var_name, var_data in original_ds.data_vars.items():
        if ('lsmlat' in var_data.dims and 'lsmlon' in var_data.dims):
            if is_2d_index:
              if keep2d:
                var_subset = var_data.isel(lsmlat=slice(min(lat_indices), max(lat_indices)+1),
                                           lsmlon=slice(min(lon_indices), max(lon_indices)+1))
              else:
                var_subset = var_data.isel(lsmlat=xr.DataArray(lat_indices, dims='gridcell'),
                                           lsmlon=xr.DataArray(lon_indices, dims='gridcell'))
            else:
                var_subset = var_data
        elif ('lat' in var_data.dims and 'lon' in var_data.dims):
            if is_2d_index:
              if keep2d:
                var_subset = var_data.isel(lat=slice(min(lat_indices), max(lat_indices)+1),
                                           lon=slice(min(lon_indices), max(lon_indices)+1))
              else:
                var_subset = var_data.isel(lat=xr.DataArray(lat_indices, dims='gridcell'),
                                           lon=xr.DataArray(lon_indices, dims='gridcell'))
            else:
                var_subset = var_data
        elif ('ni' in var_data.dims and 'nj' in var_data.dims):
              #Domain file
              if not is_2d_index:
                # Vector-domain files commonly have dimensions (nj=1, ni=ncell).
                # In that case bbox/list selection returns 1D ni indices; subset
                # ni directly and preserve nj so the domain and surfdata cell
                # counts remain consistent.
                var_subset = var_data.isel(ni=xr.DataArray(point_indices, dims='ni'))
              elif keep2d:
                # Use original 2D indexing
                var_subset = var_data.isel(nj=slice(min(lat_indices), max(lat_indices)+1),
                                           ni=slice(min(lon_indices), max(lon_indices)+1))
              else:
                # Flatten to 1D
                var_subset = var_data.isel(nj=xr.DataArray(lat_indices, dims='gridcell'),
                                           ni=xr.DataArray(lon_indices, dims='gridcell'))
                var_subset = var_subset.rename({'gridcell': 'ni'})
                var_subset = var_subset.expand_dims(dim={'nj': [1]})
                var_subset = var_subset.transpose('nj', ...)
        elif ('ni' in var_data.dims):
            # Domain corner/bounds variables such as xv(nv, ni) and yv(nv, ni)
            # also need ni subsetting. Leaving them untouched can force the
            # output domain back to the global/vector cell count.
            if is_2d_index:
                if keep2d:
                    var_subset = var_data.isel(ni=slice(min(lon_indices), max(lon_indices)+1))
                else:
                    var_subset = var_data.isel(ni=xr.DataArray(lon_indices, dims='ni'))
            else:
                var_subset = var_data.isel(ni=xr.DataArray(point_indices, dims='ni'))
        elif ('nj' in var_data.dims):
            if is_2d_index:
                if keep2d:
                    var_subset = var_data.isel(nj=slice(min(lat_indices), max(lat_indices)+1))
                else:
                    var_subset = var_data.isel(nj=xr.DataArray(lat_indices, dims='nj'))
            else:
                var_subset = var_data
        elif ('gridcell' in var_data.dims):
            #Source dataset is 1D, simply extract
            #var_subset = var_data.isel({gridcell: index})
            var_subset = var_data.isel(gridcell=index)
        else:
            var_subset = var_data
        var_subset.to_netcdf(output_file,mode='a' if var_name != list(original_ds.data_vars)[0] else 'w')
    original_ds.close()

def setpfts(self, ds, pct_pft, zerootherlandunits=True, year=None, first_bareground=False):
    #Set the PFTs as desired, zero out other landunits
    #If year is specified, set PCT_NAT_PFT for that year and all years after
    # Make a copy to avoid view assignment issues
    ds = ds.copy()
    ds, pct_pft = self.normalize_pct_nat_pft(ds, pct_pft)
    if year is not None and 'time' in ds.dims:
        # Find the time index for the specified year
        years = ds['time'].values
        # Handle different time formats (years since reference, actual years, etc.)
        if hasattr(ds['time'], 'units') and 'since' in ds['time'].units:
            # Convert from time units to actual years
            import pandas as pd
            time_dates = pd.to_datetime(ds['time'].values, unit='D', origin=ds['time'].units.split('since')[1].strip())
            actual_years = time_dates.year.values
        else:
            # Assume time values are already years
            actual_years = years.astype(int)
        # Find the index where year >= specified year
        year_indices = np.where(actual_years >= year)[0]
        if len(year_indices) == 0:
            print(f"Warning: Year {year} not found in dataset. No PFT changes applied.")
            return ds
        print(f"Setting PFT transition in year {year}")
        # Set PCT_NAT_PFT for the specified year and all years after
        for time_idx in year_indices:
            ds['PCT_NAT_PFT'][time_idx, :] = pct_pft.broadcast_like(ds['PCT_NAT_PFT'][time_idx, :])
    else:
        # Original behavior - set for all times/no time dimension
        ds['PCT_NAT_PFT'] = ds['PCT_NAT_PFT'] * 0 + pct_pft.broadcast_like(ds['PCT_NAT_PFT'])
        
    #Assume we want to zero the other land units
    if (zerootherlandunits):
        ds['PCT_NATVEG'] = ds['PCT_NATVEG'] * 0 + 100.0  # Use assignment instead of in-place
        #print('Zeroing out other landunits')
        nonveg=['PCT_WETLAND','PCT_LAKE','PCT_URBAN','PCT_CROP','PCT_GLACIER']
        for v in nonveg:
            if v in ds.variables:
                ds[v] = ds[v] * 0 + 0.0  # Use assignment instead of in-place

    # If requested, make the first topounit bareground: zero all natpft fractions and set natpft 0 to 100
    if first_bareground:
        if 'PCT_NAT_PFT' in ds:
            arr = ds['PCT_NAT_PFT']
            dims = list(arr.dims)
            nd = arr.ndim
            if 'topounit' in dims and 'natpft' in dims:
                top_i = dims.index('topounit')
                nat_i = dims.index('natpft')
                idx_zero = [slice(None)] * nd
                idx_zero[top_i] = 0
                arr.values[tuple(idx_zero)] = 0.0
                idx_nat0 = [slice(None)] * nd
                idx_nat0[top_i] = 0
                idx_nat0[nat_i] = 0
                arr.values[tuple(idx_nat0)] = 100.0
    return ds


def normalize_pct_nat_pft(self, ds, pct_pft):
    """Make a site PFT vector compatible with the surface-data natpft axis."""
    if 'PCT_NAT_PFT' not in ds:
        raise KeyError('PCT_NAT_PFT not found in surface dataset')
    if 'natpft' not in ds['PCT_NAT_PFT'].sizes:
        raise KeyError('PCT_NAT_PFT does not have a natpft dimension')

    pct_values = pct_pft.values if hasattr(pct_pft, 'values') else pct_pft
    pct_values = np.asarray(pct_values, dtype=float)
    if pct_values.ndim != 1:
        raise ValueError(f'Expected 1D PCT_NAT_PFT vector, got shape {pct_values.shape}')

    target_natpft = ds['PCT_NAT_PFT'].sizes['natpft']
    if pct_values.size > target_natpft:
        if target_natpft == 17 and pct_values.size == 22:
            ds = self.expand_natpft_dimension(ds, target_natpft=22)
            target_natpft = ds['PCT_NAT_PFT'].sizes['natpft']
        elif np.allclose(pct_values[target_natpft:], 0.0):
            pct_values = pct_values[:target_natpft]
        else:
            raise ValueError(
                f'Site PFT vector has {pct_values.size} entries but surface data has '
                f'natpft={target_natpft}; refusing to drop nonzero PFT fractions'
            )

    if pct_values.size < target_natpft:
        padded = np.zeros(target_natpft, dtype=float)
        padded[:pct_values.size] = pct_values
        pct_values = padded

    return ds, xr.DataArray(pct_values, dims=['natpft'])


def is_peatlands_sitegroup(self):
    sitegroup = str(getattr(self, 'sitegroup', '')).strip().strip("'\"")
    return sitegroup.lower() == 'peatlands'


def expand_natpft_dimension(self, ds, target_natpft=22):
    """Expand surface-data natpft dimension, preserving existing PFT slices."""
    if 'natpft' not in ds.sizes:
        return ds
    current_natpft = ds.sizes['natpft']
    if current_natpft == target_natpft:
        return ds
    if current_natpft > target_natpft:
        raise ValueError(
            f"Cannot shrink natpft dimension from {current_natpft} to {target_natpft}"
        )
    if current_natpft != 17:
        raise ValueError(
            f"Peatlands surface-data upgrade only supports natpft 17 -> {target_natpft}; "
            f"found natpft={current_natpft}"
        )

    print(f'Expanding natpft dimension from {current_natpft} to {target_natpft}')
    ds = ds.load()
    expanded = xr.Dataset(attrs=dict(ds.attrs))

    for coord_name, coord in ds.coords.items():
        if coord_name == 'natpft':
            expanded.coords[coord_name] = xr.DataArray(np.arange(target_natpft), dims=['natpft'])
        else:
            expanded.coords[coord_name] = coord.copy(deep=True)
    if 'natpft' not in expanded.coords:
        expanded.coords['natpft'] = xr.DataArray(np.arange(target_natpft), dims=['natpft'])

    for var_name, var_data in ds.data_vars.items():
        if 'natpft' not in var_data.dims:
            expanded[var_name] = var_data.copy(deep=True)
            continue

        nat_axis = var_data.dims.index('natpft')
        new_shape = list(var_data.shape)
        new_shape[nat_axis] = target_natpft
        new_values = np.zeros(new_shape, dtype=var_data.dtype)
        old_index = [slice(None)] * var_data.ndim
        new_index = [slice(None)] * var_data.ndim
        old_index[nat_axis] = slice(0, current_natpft)
        new_index[nat_axis] = slice(0, current_natpft)
        new_values[tuple(new_index)] = var_data.values[tuple(old_index)]
        expanded[var_name] = xr.DataArray(new_values, dims=var_data.dims, attrs=var_data.attrs)

    ds.close()
    return expanded


def prepare_peatlands_surface_data(self, ds, latvar, lonvar):
    """Upgrade a standard surface file for the Peatlands sitegroup when needed."""
    if not self.is_peatlands_sitegroup():
        return ds

    if 'topounit' not in ds.sizes:
        print('Adding default 4-topounit Peatlands surface metadata')
        base_elev = self.siteinfo['elev'] if hasattr(self, 'siteinfo') and 'elev' in self.siteinfo else 0.0
        fracarea = [0.25, 0.25, 0.25, 0.25]
        elevations = [base_elev, base_elev + 2.925, base_elev + 3.075, base_elev + 10.0]
        distances = [0, 150, 1, 300]
        is_bog = [0, 1, 1, 0]
        peat_depth = [2.0, 3.0, 3.0, 0.0]
        till_ksat = [0.0, 0.1/86400.0, 0.1/86400.0, 0.0]
        ds = self.add_topounit_dimension(
            ds, latvar, lonvar, num_topounits=4,
            fracarea=fracarea, elevations=elevations, distances=distances,
            is_bog=is_bog, peat_depth=peat_depth, till_ksat=till_ksat
        )
        ds.attrs['topounit_order'] = '1=fen_low_outlet, 2=bog_hollow, 3=bog_hummock, 4=upland_high'
        ds.attrs['topounit_fraction_default'] = 'fen=0.25, bog_hollow=0.25, bog_hummock=0.25, upland=0.25'

    ds = self.expand_natpft_dimension(ds, target_natpft=22)
    return ds

            
def makepointdata(self, filename, pft=-1, mylat=[], mylon=[]):
    #Extract surface, domain, or pftdyn data from a given regional or global file.
    #If mylat and mylon are empty, it will use self.lat_bounds and self.lon_bounds to extract.
    mydata = Dataset(filename,'r')
    lonvar = 'LONGXY'
    latvar = 'LATIXY'
    #Figure out which type of file
    isdomain=False
    ispftdyn=False
    modifysurfdat=False
    if ('domain' in filename.split('/')[-1]):
        print('Creating domain data from ', filename)
        lonvar = 'xc'
        latvar = 'yc'
        infile  = self.domain_global
        outfile = self.OLMTdir+'/temp/domain.nc'
        #Save mask for other datasets
        self.mask_grid = mydata['mask'][:].copy()
        isdomain=True
    elif ('landuse' in filename.split('/')[-1] or 'pftdyn' in filename.split('/')[-1]):
        infile = self.pftdyn_global
        outfile = self.OLMTdir+'/temp/surfdata.pftdyn.nc'
        print('Creating land use data from ', filename)
        ispftdyn=True
    else:
        infile = self.surfdata_global
        outfile = self.OLMTdir+'/temp/surfdata.nc'
        print('Creating surface data from ', filename)
        if hasattr(self, 'add_surfdata') and self.add_surfdata:
            modifysurfdat=True
    
    #Get the site lat/lon
    if (self.site != ''):
        mylat = np.array([self.siteinfo['lat']])
        mylon = np.array([self.siteinfo['lon']])
        mylon[mylon > 180] -= 360
        index = self.get_pointindices_list(mylat, mylon, mydata[latvar][:], mydata[lonvar][:], mask_grid=self.mask_grid) 
        self.subset_netcdf(index, infile,  outfile)
        ds = xr.open_dataset(outfile, mode='r+')

        if (self.is_peatlands_sitegroup() and not isdomain and not ispftdyn):
            ds = self.prepare_peatlands_surface_data(ds, latvar, lonvar)
        
        # Handle HumHol topounit dimension
        if (self.humhol and not self.is_peatlands_sitegroup() and not isdomain):
            if ('SPR' in self.site):
                # SPRUCE is represented as three topounits:
                # 1) boardwalk/fen bare ground, 2) bog hollow, 3) bog hummock.
                # The bog units share a common peat/till interface set 3 m
                # below the hollow surface, so the hummock has deeper peat by
                # its microtopographic offset.
                fracarea = [0.5,0.17,0.33]
                elevations = [464.6,465.0,465.15]   # boardwalk/fen, hollow, hummock
                distances = [0, 3, 1]  # distance to next lower adjacent topounit (m)
                is_bog = [0, 1, 1]
                bog_peat_interface_elev = elevations[1] - 3.0
                peat_depth = [
                    0.0 if not is_bog[i] else elevations[i] - bog_peat_interface_elev
                    for i in range(len(elevations))
                ]
                till_ksat = [0.0, 0.1/86400.0, 0.1/86400.0]  # mm/s
                ds = self.add_topounit_dimension(ds, latvar, lonvar, num_topounits=3, \
                        fracarea=fracarea, elevations=elevations, distances=distances, \
                        is_bog=is_bog, peat_depth=peat_depth, till_ksat=till_ksat)
            else:
                fracarea = [0.5,0.5]
                elevations = [self.siteinfo['elev'], self.siteinfo['elev']+0.15]
                distances = [0, 1]
                #Default fractional areas and elevations
                ds = self.add_topounit_dimension(ds, latvar, lonvar, num_topounits=2, fracarea=fracarea, elevations=elevations, distances=distances)
        if (not isdomain):
            #Set site PFT and soil texture
            if (sum(self.siteinfo['PCT_NAT_PFT']) > 0):
                pct_nat_pft = xr.DataArray(self.siteinfo['PCT_NAT_PFT'], dims=['natpft'])
                if 'SPR' in self.site and not self.is_peatlands_sitegroup():
                    #Set up as 3 topounits, 1 bareground and 2 with the specified PFT fractions
                    ds = self.setpfts(ds, pct_nat_pft, first_bareground=True)  
                else:
                    ds = self.setpfts(ds, pct_nat_pft)
            if (pft >=0):
                npfts = ds['PCT_NAT_PFT'].sizes['natpft']
                #Overrite site info
                pct_pft = np.zeros(npfts, float)
                pct_pft[pft] = 100.0
                pct_nat_pft = xr.DataArray(pct_pft, dims=['natpft'])
                # For SPRUCE HUM_HOL, reserve topounit 1 for boardwalk/fen
                # bare ground rather than assigning the selected vegetation
                # PFT there.
                if 'SPR' in self.site and not self.is_peatlands_sitegroup():
                    ds = self.setpfts(ds, pct_nat_pft, first_bareground=True)
                else:
                    ds = self.setpfts(ds, pct_nat_pft)
            print('Setting PFT_NAT_PFT to: ', self.siteinfo['PCT_NAT_PFT'])
            if (not ispftdyn):
                if (self.siteinfo['PCT_SAND'] >= 0):
                    # Use .values to modify underlying data directly
                    ds['PCT_SAND'].values[:] = self.siteinfo['PCT_SAND']
                    print('Setting %SAND to ',self.siteinfo['PCT_SAND'])
                if (self.siteinfo['PCT_CLAY'] >= 0):
                    ds['PCT_CLAY'].values[:] = self.siteinfo['PCT_CLAY']
                    print('Setting $CLAY to ',self.siteinfo['PCT_CLAY'])
                #SPRUCE-specific properties
                if (self.humhol): #'SPR' in self.site):
                  print('Setting SPRUCE organic, soil order and P variables')
                  ds['ORGANIC'].values[0:9,:] = 130.
                  ds['ORGANIC'].values[9,:] = 65.
                  ds['SOIL_ORDER'].values[:] = 3
                  ds['LABILE_P'].values[:] = 1.0
                  ds['APATITE_P'].values[:] = 0.1
                  ds['SECONDARY_P'].values[:] = 1.0
                  ds['OCCLUDED_P'].values[:] = 1.0
            else:  #handle land use transitions
                #zero harvest
                ds['HARVEST_VH1'].values[:] = 0.0
                ds['HARVEST_VH2'].values[:] = 0.0
                ds['HARVEST_SH1'].values[:] = 0.0
                ds['HARVEST_SH2'].values[:] = 0.0
                ds['HARVEST_SH3'].values[:] = 0.0
                #Apply transitions if given
                years = ds['time'].values
                for year in self.siteinfo['transitions'].keys():
                    #Set PFTS for this year and all subsequent years
                    pct_nat_pft = xr.DataArray(self.siteinfo['transitions'][year]['PCT_NAT_PFT'], dims=['natpft'])
                    if ('SPR' in self.site and not self.is_peatlands_sitegroup()):
                        #Set up as 3 topounits, 1 bareground and 2 with the specified PFT fractions
                        ds = self.setpfts(ds, pct_nat_pft, first_bareground=True, year=int(year))
                    else:
                        ds = self.setpfts(ds, pct_nat_pft, year=int(year))
                    #Set harvest for this year
                    year_indices = np.where(years == int(year))[0]
                    ds['HARVEST_VH1'].values[year_indices] = self.siteinfo['transitions'][year]['HARVEST']

        if (self.shift_lon):
            mylon[mylon < 0] +=360
        ds[latvar].values[:] = mylat
        ds[lonvar].values[:] = mylon
        ds.to_netcdf(outfile+'.tmp')
        ds.close()
        os.system('mv '+outfile+'.tmp '+outfile)
        if (modifysurfdat):
            print('Modifying surface data with user-specified modifications')
            self.modify_ncinput_file(outfile, self.add_surfdata, file_description="surface data")
    
    elif (len(self.point_list) > 0):
        # If only one point and humhol is True, add topounit dimension instead of duplicating
        if len(self.point_list) == 1 and getattr(self, 'humhol', False):
            point_lats = np.array([self.point_list[0][0]])
            point_lons = np.array([self.point_list[0][1]])
        else:
            point_lats = np.array([lat for lat, lon in self.point_list])
            point_lons = np.array([lon for lat, lon in self.point_list])
        point_lons[point_lons > 180] -= 360
        index = self.get_pointindices_list(point_lats, point_lons, mydata[latvar][:], \
                mydata[lonvar][:], mask_grid=self.mask_grid)
        self.subset_netcdf(index, infile,  outfile, keep2d = False)
        ds = xr.open_dataset(outfile, mode='r+')

        if (self.is_peatlands_sitegroup() and not isdomain and not ispftdyn):
            ds = self.prepare_peatlands_surface_data(ds, latvar, lonvar)
        
        # Handle HumHol topounit dimension for point_list case
        if (len(self.point_list) == 1 and getattr(self, 'humhol', False) and
                not self.is_peatlands_sitegroup() and not isdomain):
            print('Adding topounit dimension for HumHol point case')
            ds = self.add_topounit_dimension(ds, latvar, lonvar, num_topounits=2)
        
        if (not isdomain):
            if (pft >=0):
                npfts = ds['PCT_NAT_PFT'].sizes['natpft']
                #Overrite site info
                pct_pft = np.zeros(npfts, float)
                pct_pft[pft] = 100.0
                pct_nat_pft = xr.DataArray(pct_pft, dims=['natpft'])
                ds = self.setpfts(ds, pct_nat_pft);
            if (self.humhol and len(self.point_list) == 1):
                #If humhol, zero other land units but keep nat pfts as is
                pct_nat_pft = ds['PCT_NAT_PFT']
                ds = self.setpfts(ds, pct_nat_pft);
        
        if (self.shift_lon):
            point_lons[point_lons < 0] +=360
        ds[latvar][:] = point_lats
        ds[lonvar][:] = point_lons
        ds.to_netcdf(outfile+'.tmp')
        ds.close()
        os.system('mv '+outfile+'.tmp '+outfile)
        if (modifysurfdat):
            print('Modifying surface data with user-specified modifications')
            self.modify_ncinput_file(outfile, self.add_surfdata, file_description="surface data")
    else:  #USe lat lon bounding box
        if (self.lat_bounds[1]-self.lat_bounds[0] < 180 and self.lon_bounds[1]-self.lon_bounds[0] < 360):
            index = self.get_pointindices_bbox(self.lat_bounds, self.lon_bounds, mydata[latvar][:], mydata[lonvar][:], \
                mask_grid=self.mask_grid)
            keep2d_subset = len(index) > 0 and isinstance(index[0], (tuple, list))
            self.subset_netcdf(index, infile,  outfile, keep2d=keep2d_subset)
            if (self.is_peatlands_sitegroup() and not isdomain and not ispftdyn):
                ds = xr.open_dataset(outfile, mode='r+')
                ds = self.prepare_peatlands_surface_data(ds, latvar, lonvar)
                ds.to_netcdf(outfile+'.tmp')
                ds.close()
                os.system('mv '+outfile+'.tmp '+outfile)
            if (modifysurfdat):
                print('Modifying surface data with user-specified modifications')
                self.modify_ncinput_file(outfile, self.add_surfdata, file_description="surface data")
        else:
            print('Global simulation requested.  Using original file.')
            self.mask_grid=[]
            os.system('cp '+infile+' '+outfile)
            if (self.is_peatlands_sitegroup() and not isdomain and not ispftdyn):
                ds = xr.open_dataset(outfile, mode='r+')
                ds = self.prepare_peatlands_surface_data(ds, latvar, lonvar)
                ds.to_netcdf(outfile+'.tmp')
                ds.close()
                os.system('mv '+outfile+'.tmp '+outfile)


def add_topounit_dimension(self, ds, latvar, lonvar, num_topounits=2, fracarea=None, \
        elevations=None, distances=None, is_bog=None, peat_depth=None, till_ksat=None):
    """
    Add topounit dimension and related variables for topographic simulations
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset to modify
    latvar : str
        Name of latitude variable
    lonvar : str
        Name of longitude variable
    num_topounits : int, optional
        Number of topounits to create (default: 2)
    fracarea : array-like, optional
        Fractional areas for each topounit (must sum to 1.0)
        If None, equal areas will be used
    elevations : array-like, optional
        Elevations for each topounit (in meters asl)
        If None, default elevations will be generated
    distances : array-like, optional
        Lateral distance to the next lower adjacent topounit (m)
    is_bog : array-like, optional
        Integer flags for bog topounits, where 1=bog and 0=non-bog
    peat_depth : array-like, optional
        Peat depth above restrictive till for each topounit (m)
    till_ksat : array-like, optional
        Restrictive till saturated conductivity for each topounit (mm/s)
        
    Returns:
    --------
    xarray.Dataset
        Modified dataset with topounit dimension and variables
    """
    print(f'Adding topounit dimension for {num_topounits} topounits')
    
    # Validate and set default fracarea
    if fracarea is None:
        fracarea = np.full(num_topounits, 1.0/num_topounits)
    else:
        fracarea = np.array(fracarea)
        if len(fracarea) != num_topounits:
            raise ValueError(f"fracarea length ({len(fracarea)}) must equal num_topounits ({num_topounits})")
        if not np.isclose(np.sum(fracarea), 1.0):
            raise ValueError(f"fracarea must sum to 1.0, got {np.sum(fracarea)}")
    
    # Validate and set default elevations
    if elevations is None:
        # Generate default elevations: base elevation + incremental differences
        base_elv = 100.0
        elevations = np.array([base_elv + i*5.0 for i in range(num_topounits)])
    else:
        elevations = np.array(elevations)
        if len(elevations) != num_topounits:
            raise ValueError(f"elevations length ({len(elevations)}) must equal num_topounits ({num_topounits})")

    # Validate and set default lateral distances
    if distances is None:
        # Default lateral distances (meters): zeros
        distances = np.zeros(num_topounits)
    else:
        distances = np.array(distances)
        if len(distances) != num_topounits:
            raise ValueError(f"distances length ({len(distances)}) must equal num_topounits ({num_topounits})")

    # Optional peatland metadata used by the E3SM-Peatlands branch.
    if is_bog is None:
        is_bog = np.zeros(num_topounits, dtype=int)
    else:
        is_bog = np.array(is_bog, dtype=int)
        if len(is_bog) != num_topounits:
            raise ValueError(f"is_bog length ({len(is_bog)}) must equal num_topounits ({num_topounits})")

    if peat_depth is None:
        peat_depth = np.zeros(num_topounits)
    else:
        peat_depth = np.array(peat_depth)
        if len(peat_depth) != num_topounits:
            raise ValueError(f"peat_depth length ({len(peat_depth)}) must equal num_topounits ({num_topounits})")

    if till_ksat is None:
        till_ksat = np.zeros(num_topounits)
    else:
        till_ksat = np.array(till_ksat)
        if len(till_ksat) != num_topounits:
            raise ValueError(f"till_ksat length ({len(till_ksat)}) must equal num_topounits ({num_topounits})")
    
    # Load the dataset into memory first
    ds = ds.load()
    
    # Create new dataset with topounit dimension
    new_ds = xr.Dataset()
    new_ds.coords['topounit'] = xr.DataArray(np.arange(num_topounits), dims=['topounit'])
    
    # Copy all existing dimensions and coordinates
    for dim_name, dim_size in ds.sizes.items():  # Changed from ds.dims to ds.sizes
        if dim_name not in new_ds.sizes:
            # Force copy by converting to numpy and back
            coord_data = ds.coords[dim_name].values
            new_ds.coords[dim_name] = xr.DataArray(coord_data, dims=ds.coords[dim_name].dims, 
                                                   attrs=ds.coords[dim_name].attrs)
    
    # Process each variable
    for var_name, var_data in ds.data_vars.items():
        if var_name in [latvar, lonvar]:
            # Keep lat/lon as is (single gridcell) but force copy
            data_values = var_data.values.copy()
            new_ds[var_name] = xr.DataArray(data_values, dims=var_data.dims, attrs=var_data.attrs)
        elif any(dim in var_data.dims for dim in ['lsmlat', 'lsmlon', 'gridcell']):
            # Only expand variables that have spatial dimensions
            dims = list(var_data.dims)
            # Insert topounit in the correct position in the original dimensions
            if 'gridcell' in dims:
                # For gridcell: topounit should be second to last
                insert_pos = -1
            elif any(dim in dims for dim in ['lsmlat', 'lsmlon']):
                # For lsmlat/lsmlon: topounit should be third to last
                insert_pos = -2
            else:
                # Default: add at the end
                insert_pos = len(dims)
            dims.insert(insert_pos, 'topounit')
            
            # Create new variable with topounit dimension - force copy
            original_data = var_data.values
            # Expand the numpy array manually to avoid views
            expanded_shape = list(original_data.shape)
            insert_axis = insert_pos if insert_pos >= 0 else max(0, len(expanded_shape) + insert_pos)
            expanded_shape.insert(insert_axis, num_topounits)
            expanded_data = np.expand_dims(original_data, axis=insert_axis)
            expanded_data = np.broadcast_to(expanded_data, expanded_shape)
            expanded_data = expanded_data.copy()  # Force a copy to make it writable
            new_ds[var_name] = xr.DataArray(expanded_data, dims=dims, attrs=var_data.attrs)
        else:
            # Keep other variables unchanged but force copy
            data_values = var_data.values.copy()
            new_ds[var_name] = xr.DataArray(data_values, dims=var_data.dims, attrs=var_data.attrs)
    
    # Add topographic variables
    print('Adding topographic variables')
    # Get grid dimensions
    if 'gridcell' in new_ds.sizes:  # Changed from new_ds.dims to new_ds.sizes
        grid_dims = ['gridcell']
        grid_size = new_ds.sizes['gridcell']
    elif 'lsmlat' in new_ds.sizes and 'lsmlon' in new_ds.sizes:  # Changed from new_ds.dims to new_ds.sizes
        grid_dims = ['lsmlat', 'lsmlon']
        lat_size = new_ds.sizes['lsmlat']
        lon_size = new_ds.sizes['lsmlon']
        grid_size = lat_size * lon_size
    else:
        grid_dims = ['gridcell']
        grid_size = 1

    # Calculate weighted average elevation
    weighted_avg_elv = np.sum(fracarea * elevations)
    
    # TOPO2: Weighted average of topounits elevation in a gridcell
    if 'gridcell' in new_ds.sizes:  # Changed from new_ds.dims to new_ds.sizes
        topo2_data = np.full((grid_size,), weighted_avg_elv)  # 1D for gridcell
    else:
        topo2_data = np.full((lat_size, lon_size), weighted_avg_elv)  # 2D for lsmlat/lsmlon
    new_ds['TOPO2'] = xr.DataArray(
        topo2_data, 
        dims=grid_dims,
        attrs={'descriptions': 'Weighted average of topounits elevation in a gridcell'}
    )

    # topoPerGrid: Number of topounit for each grid  
    if 'gridcell' in new_ds.sizes:  # Changed from new_ds.dims to new_ds.sizes
        topopergrid_data = np.full((grid_size,), num_topounits, dtype=int)  # 1D for gridcell
    else:
        topopergrid_data = np.full((lat_size, lon_size), num_topounits, dtype=int)  # 2D for lsmlat/lsmlon
    new_ds['topoPerGrid'] = xr.DataArray(
        topopergrid_data,
        dims=grid_dims, 
        attrs={'descriptions': 'Number of topounit for each grid'}
    )

    # MaxTopounitElv: Maximum topounits elevation in a gridcell
    max_elv = np.max(elevations)
    if 'gridcell' in new_ds.sizes:  # Changed from new_ds.dims to new_ds.sizes
        maxelv_data = np.full((grid_size,), max_elv)  # 1D for gridcell
    else:
        maxelv_data = np.full((lat_size, lon_size), max_elv)  # 2D for lsmlat/lsmlon
    new_ds['MaxTopounitElv'] = xr.DataArray(
        maxelv_data,
        dims=grid_dims,
        attrs={'descriptions': 'Maximum topounits elevation in a gridcell'}
    )

    # TopounitFracArea: Fractional area of topounits
    if 'gridcell' in new_ds.sizes:  # Changed from new_ds.dims to new_ds.sizes
        fracarea_dims = ['topounit', 'gridcell'] 
        fracarea_shape = (num_topounits, grid_size)
    else:
        fracarea_dims = ['topounit', 'lsmlat', 'lsmlon'] 
        fracarea_shape = (num_topounits, lat_size, lon_size)

    # Broadcast fracarea to the correct shape
    fracarea_data = np.broadcast_to(fracarea.reshape(-1, *([1] * (len(fracarea_shape) - 1))), fracarea_shape)
    new_ds['TopounitFracArea'] = xr.DataArray(
        fracarea_data,
        dims=fracarea_dims,
        attrs={
            '_FillValue': -999.0,
            'descriptions': 'Fractional area of the subgrid of the land fraction of grid'
        }
    )

    # TopounitAveElv: Average elevation of each topounit
    if 'gridcell' in new_ds.sizes:  # Changed from new_ds.dims to new_ds.sizes
        elv_dims = ['topounit', 'gridcell'] 
        elv_shape = (num_topounits, grid_size)
    else:
        elv_dims = ['topounit', 'lsmlat', 'lsmlon']
        elv_shape = (num_topounits, lat_size, lon_size)

    # Broadcast elevations to the correct shape
    elv_data = np.broadcast_to(elevations.reshape(-1, *([1] * (len(elv_shape) - 1))), elv_shape)
    new_ds['TopounitAveElv'] = xr.DataArray(
        elv_data,
        dims=elv_dims,
        attrs={
            '_FillValue': -999.0,
            'descriptions': 'Average elevation of subgrid',
            'units': 'meters asl'
        }
    )
    
    # TopounitLateralDist: Lateral distance for each topounit (meters)
    dist_shape = elv_shape
    dist_dims = elv_dims
    dist_data = np.broadcast_to(distances.reshape(-1, *([1] * (len(dist_shape) - 1))), dist_shape)
    new_ds['TopounitLateralDist'] = xr.DataArray(
        dist_data,
        dims=dist_dims,
        attrs={
            '_FillValue': -999.0,
            'descriptions': 'Lateral distance for each topounit within gridcell',
            'units': 'meters'
        }
    )

    is_bog_data = np.broadcast_to(is_bog.reshape(-1, *([1] * (len(dist_shape) - 1))), dist_shape)
    new_ds['TopounitIsBog'] = xr.DataArray(
        is_bog_data,
        dims=dist_dims,
        attrs={
            'long_name': 'topounit bog flag',
            'units': '1',
            'flag_values': np.array([0, 1], dtype=np.int32),
            'flag_meanings': 'false true'
        }
    )

    peat_depth_data = np.broadcast_to(peat_depth.reshape(-1, *([1] * (len(dist_shape) - 1))), dist_shape)
    new_ds['TopounitPeatDepth'] = xr.DataArray(
        peat_depth_data,
        dims=dist_dims,
        attrs={
            '_FillValue': -999.0,
            'long_name': 'topounit peat depth',
            'units': 'm'
        }
    )

    till_ksat_data = np.broadcast_to(till_ksat.reshape(-1, *([1] * (len(dist_shape) - 1))), dist_shape)
    new_ds['TopounitTillKsat'] = xr.DataArray(
        till_ksat_data,
        dims=dist_dims,
        attrs={
            '_FillValue': -999.0,
            'long_name': 'restrictive till saturated hydraulic conductivity below topounit peat',
            'units': 'mm s-1'
        }
    )
    
    # Copy attributes
    new_ds.attrs = dict(ds.attrs)
    if num_topounits == 3 and np.array_equal(is_bog, np.array([0, 1, 1])):
        new_ds.attrs['topounit_order'] = '1=boardwalk_fen_bareground, 2=bog_hollow, 3=bog_hummock'
        new_ds.attrs['topounit_is_bog'] = 'boardwalk_fen=0, bog_hollow=1, bog_hummock=1'
        new_ds.attrs['topounit_peat_depth_m'] = (
            f'boardwalk_fen={float(peat_depth[0]):.2f}, '
            f'bog_hollow={float(peat_depth[1]):.2f}, '
            f'bog_hummock={float(peat_depth[2]):.2f}'
        )
        new_ds.attrs['topounit_till_ksat_mm_per_day'] = 'boardwalk_fen=0.0, bog_hollow=0.1, bog_hummock=0.1'
    # Close old dataset and replace with new one
    ds.close()
    return new_ds
