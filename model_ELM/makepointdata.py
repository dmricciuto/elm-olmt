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
            if (len(original_shape) > 1):
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
            if (len(original_shape) > 1):
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

    # Check if index contains tuples (2D) or integers (1D)
    is_2d_index = len(index) > 0 and isinstance(index[0], (tuple, list))

    # Select the variable and apply subsetting if specified
    for var_name, var_data in original_ds.data_vars.items():
        if ('lsmlat' in var_data.dims and 'lsmlon' in var_data.dims):
            if is_2d_index:
              if keep2d:
                lat_indices = [lat for lat, lon in index]
                lon_indices = [lon for lat, lon in index]
                var_subset = var_data.isel(lsmlat=slice(min(lat_indices), max(lat_indices)+1),
                                           lsmlon=slice(min(lon_indices), max(lon_indices)+1))
              else:
                lat_indices = [lat for lat, lon in index]
                lon_indices = [lon for lat, lon in index]
                var_subset = var_data.isel(lsmlat=xr.DataArray(lat_indices, dims='gridcell'),
                                           lsmlon=xr.DataArray(lon_indices, dims='gridcell'))
            else:
                var_subset = var_data
        elif ('lat' in var_data.dims and 'lon' in var_data.dims):
            if is_2d_index:
              if keep2d:
                lat_indices = [lat for lat, lon in index]
                lon_indices = [lon for lat, lon in index]
                var_subset = var_data.isel(lat=slice(min(lat_indices), max(lat_indices)+1),
                                           lon=slice(min(lon_indices), max(lon_indices)+1))
              else:
                lat_indices = [lat for lat, lon in index]
                lon_indices = [lon for lat, lon in index]
                var_subset = var_data.isel(lat=xr.DataArray(lat_indices, dims='gridcell'),
                                           lon=xr.DataArray(lon_indices, dims='gridcell'))
            else:
                var_subset = var_data
        elif ('ni' in var_data.dims and 'nj' in var_data.dims):
              #Domain file
              if keep2d:
                # Use original 2D indexing
                lat_indices = [lat for lat, lon in index]
                lon_indices = [lon for lat, lon in index]
                var_subset = var_data.isel(nj=slice(min(lat_indices), max(lat_indices)+1),
                                           ni=slice(min(lon_indices), max(lon_indices)+1))
              else:
                # Flatten to 1D
                var_subset = var_data.isel(nj=xr.DataArray([lat for lat, lon in index], dims='gridcell'),
                                           ni=xr.DataArray([lon for lat, lon in index], dims='gridcell'))
                var_subset = var_subset.rename({'gridcell': 'ni'})
                var_subset = var_subset.expand_dims(dim={'nj': [1]})
                var_subset = var_subset.transpose('nj', ...)
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
        
        # Handle HumHol topounit dimension
        if (self.humhol and not isdomain):
            if ('SPR' in self.site):
                #SPRUCE fractional hum-hol areas and elevations
                fracarea = [0.5,0.17,0.33]
                elevations = [464.6,465.0,465.15]   #lagg, hollow, hummock
                distances = [0, 3, 1]  # Example distances for the two topounits (in meters)
                ds = self.add_topounit_dimension(ds, latvar, lonvar, num_topounits=3, fracarea=fracarea, elevations=elevations, distances=distances)
            else:
                #Default fractional areas and elevations
                ds = self.add_topounit_dimension(ds, latvar, lonvar, num_topounits=2)

        if (not isdomain):
            #Set site PFT and soil texture
            if (sum(self.siteinfo['PCT_NAT_PFT']) > 0):
                pct_nat_pft = xr.DataArray(self.siteinfo['PCT_NAT_PFT'], dims=['natpft'])
                if 'SPR' in self.site:
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
                ds = self.setpfts(ds, pct_nat_pft);
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
                if ('SPR' in self.site):
                  print('Setting SPRUCE organic, soil order and P variables')
                  ds['ORGANIC'].values[0:8,:] = 130.
                  ds['ORGANIC'].values[8,:] = 65.
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
                    if ('SPR' in self.site):
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
        
        # Handle HumHol topounit dimension for point_list case
        if (len(self.point_list) == 1 and getattr(self, 'humhol', False) and not isdomain):
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
            self.subset_netcdf(index, infile,  outfile, keep2d=True)
            if (modifysurfdat):
                print('Modifying surface data with user-specified modifications')
                self.modify_ncinput_file(outfile, self.add_surfdata, file_description="surface data")
        else:
            print('Global simulation requested.  Using original file.')
            self.mask_grid=[]
            os.system('cp '+infile+' '+outfile)


def add_topounit_dimension(self, ds, latvar, lonvar, num_topounits=2, fracarea=None, \
        elevations=None, distances=None):
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
            expanded_shape.insert(insert_pos, num_topounits)
            expanded_data = np.broadcast_to(original_data[..., np.newaxis], expanded_shape)
            expanded_data = np.moveaxis(expanded_data, -1, insert_pos)
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
    
    # Copy attributes
    new_ds.attrs = dict(ds.attrs)
    # Close old dataset and replace with new one
    ds.close()
    return new_ds
