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
    #subset_ds = xr.Dataset()
    print(index, input_file)
    # Select the variable and apply subsetting if specified
    for var_name, var_data in original_ds.data_vars.items():
        if ('lsmlat' in var_data.dims and 'lsmlon' in var_data.dims):
            if keep2d:
                lat_indices = [lat for lat, lon in index]
                lon_indices = [lon for lat, lon in index]
                var_subset = var_data.isel(lsmlat=slice(min(lat_indices), max(lat_indices)),
                                           lsmlon=slice(min(lon_indices), max(lon_indices)))
            else:
                var_subset = var_data.isel(lsmlat=xr.DataArray([lat for lat, lon in index], dims='gridcell'),
                                           lsmlon=xr.DataArray([lon for lat, lon in index], dims='gridcell'))
        elif ('ni' in var_data.dims and 'nj' in var_data.dims):
            #Domain file
            if keep2d:
                # Use original 2D indexing
                lat_indices = [lat for lat, lon in index]
                lon_indices = [lon for lat, lon in index]
                var_subset = var_data.isel(nj=slice(min(lat_indices), max(lat_indices)),
                                           ni=slice(min(lon_indices), max(lon_indices)))
            else:
                # Flatten to 1D
                var_subset = var_data.isel(nj=xr.DataArray([lat for lat, lon in index], dims='gridcell'),
                                           ni=xr.DataArray([lon for lat, lon in index], dims='gridcell'))
                var_subset = var_subset.rename({'gridcell': 'ni'})
                var_subset = var_subset.expand_dims(dim={'nj': [1]})
                var_subset = var_subset.transpose('nj', ...)
        elif ('gridcell' in var_data.dims):
            #Source dataset is 1D, simply extract
            var_subset = var_data.isel({gridcell: index})
        else:
            var_subset = var_data
        var_subset.to_netcdf(output_file,mode='a' if var_name != list(original_ds.data_vars)[0] else 'w')
    original_ds.close()

def setpfts(self, ds, pct_pft, zerootherlandunits=True):
    #Set the PFTs as desired, zero out other landunits
    #Assign PCT_NAT_PFT (should work whether 2, 3 or 4 dimensions)
    ds['PCT_NAT_PFT'] = ds['PCT_NAT_PFT'] * 0 + pct_pft.broadcast_like(ds['PCT_NAT_PFT'])
    #Assume we want to zero the other land units
    if (zerootherlandunits):
        ds['PCT_NATVEG'][:] = 100.0
        print('Zeroing out other landunits')
        nonveg=['PCT_WETLAND','PCT_LAKE','PCT_URBAN','PCT_CROP','PCT_GLACIER']
        for v in nonveg:
            ds[v][:] = 0.0
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
    if ('domain' in filename):
        print('Creating domain data from ', filename)
        lonvar = 'xc'
        latvar = 'yc'
        infile  = self.domain_global
        outfile = self.OLMTdir+'/temp/domain.nc'
        #Save mask for other datasets
        self.mask_grid = mydata['mask'][:].copy()
        isdomain=True
    elif ('landuse' in filename or 'pftdyn' in filename):
        infile = self.pftdyn_global
        outfile = self.OLMTdir+'/temp/surfdata.pftdyn.nc'
        print('Creating land use data from ', filename)
        ispftdyn=True
    else:
        infile = self.surfdata_global
        outfile = self.OLMTdir+'/temp/surfdata.nc'
        print('Creating surface data from ', filename)
    
    #Get the site lat/lon
    if (self.site != ''):
        if (self.humhol):
            #If humhol, create two gridcells with same lat lon
            mylat = np.array([self.siteinfo['lat'], self.siteinfo['lat']])
            mylon = np.array([self.siteinfo['lon'], self.siteinfo['lon']])
        else:
            mylat = np.array([self.siteinfo['lat']])
            mylon = np.array([self.siteinfo['lon']])
        mylon[mylon > 180] -= 360
        index = self.get_pointindices_list(mylat, mylon, mydata[latvar][:], mydata[lonvar][:], mask_grid=self.mask_grid) 
        self.subset_netcdf(index, infile,  outfile)
        ds = xr.open_dataset(outfile, mode='r+')
        if (not isdomain):
            #Set site PFT and soil texture
            if (sum(self.siteinfo['PCT_NAT_PFT']) > 0):
                pct_nat_pft = xr.DataArray(self.siteinfo['PCT_NAT_PFT'], dims=['natpft'])
                ds = self.setpfts(ds, pct_nat_pft);
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
                    ds['PCT_SAND'][:] = self.siteinfo['PCT_SAND']
                    print('Setting %SAND to ',self.siteinfo['PCT_SAND'])
                if (self.siteinfo['PCT_CLAY'] >= 0):
                    ds['PCT_CLAY'][:] = self.siteinfo['PCT_CLAY']
                    print('Setting $CLAY to ',self.siteinfo['PCT_CLAY'])
            #else:  TODO - handle land use transitions
        #else:
        #    for p in range(0,len(mylat)):
        #        #Recenter on gridcell lat/lons
        #        ds['xv'][:,p,:] = ds['xv'][:,p,:] + (mylon[p] - ds['xc'][:,p])
        #        ds['yv'][:,p,:] = ds['yv'][:,p,:] + (mylat[p] - ds['yc'][:,p])
        if (self.shift_lon):
            mylon[mylon < 0] +=360
        ds[latvar][:] = mylat
        ds[lonvar][:] = mylon
        ds.to_netcdf(outfile+'.tmp')
        ds.close()
        os.system('mv '+outfile+'.tmp '+outfile)
    elif (len(self.point_list) > 0):
        point_lats = np.array([lat for lat, lon in self.point_list])
        point_lons = np.array([lon for lat, lon in self.point_list])
        point_lons[point_lons > 180] -= 360
        index = self.get_pointindices_list(point_lats, point_lons, mydata[latvar][:], \
                mydata[lonvar][:], mask_grid=self.mask_grid)
        self.subset_netcdf(index, infile,  outfile)
        ds = xr.open_dataset(outfile, mode='r+')
        if (not isdomain):
            if (pft >=0):
                npfts = ds['PCT_NAT_PFT'].sizes['natpft']
                #Overrite site info
                pct_pft = np.zeros(npfts, float)
                pct_pft[pft] = 100.0
                pct_nat_pft = xr.DataArray(pct_pft, dims=['natpft'])
                ds = self.setpfts(ds, pct_nat_pft);
        #if (isdomain):
        #    for p in range(0,len(mylat)):
        #        #Recenter on gridcell lat/lons
        #        ds['xv'][:,p,:] = ds['xv'][:,p,:] + (mylon[p] - ds['xc'][:,p])
        #        ds['yv'][:,p,:] = ds['yv'][:,p,:] + (mylat[p] - ds['yc'][:,p])
        if (self.shift_lon):
            point_lons[point_lons < 0] +=360
        ds[latvar][:] = point_lats
        ds[lonvar][:] = point_lons
        ds.to_netcdf(outfile+'.tmp')
        ds.close()
        os.system('mv '+outfile+'.tmp '+outfile)
    else:  #USe lat lon bounding box
        if (self.lat_bounds[1]-self.lat_bounds[0] < 180 and self.lon_bounds[1]-self.lon_bounds[0] < 360):
            index = self.get_pointindices_bbox(self.lat_bounds, self.lon_bounds, mydata[latvar][:], mydata[lonvar][:], \
                mask_grid=self.mask_grid)
            self.subset_netcdf(index, infile,  outfile, keep2d=True)
        else:
            print('Global simulation requested.  Using original file.')
            self.mask_grid=[]
            os.system('cp '+infile+' '+outfile)
