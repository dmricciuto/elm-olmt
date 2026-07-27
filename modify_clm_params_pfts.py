from netCDF4 import Dataset, chartostring, stringtochar
import numpy as np

src_path = 'clm_params_SPRUCE_20250507.nc'
dst_path = 'clm_params_SPRUCE_20250507_19pfts.nc'

pft_dim_name = 'pft'
old = Dataset(src_path, 'r')
if pft_dim_name not in old.dimensions:
    for dn in old.dimensions:
        if 'pft' in dn.lower():
            pft_dim_name = dn
            break
old_len = len(old.dimensions[pft_dim_name])
new_len = old_len + 2
print('pft dim:', pft_dim_name, 'old_len=', old_len, 'new_len=', new_len)

# mapping function
def map_index(i):
    if i == 12:
        return 15
    if i == 15:
        return 17
    if i >= 16:
        return i + 2
    return i

# create new file
new = Dataset(dst_path, 'w')
# copy dimensions, but replace pft
for dn, dim in old.dimensions.items():
    if dn == pft_dim_name:
        new.createDimension(dn, new_len)
    else:
        new.createDimension(dn, len(dim) if not dim.isunlimited() else None)
# copy global attributes
for an in old.ncattrs():
    new.setncattr(an, old.getncattr(an))

# handle variables
for vn, var in old.variables.items():
    dims = var.dimensions
    out_var = new.createVariable(vn, var.dtype, dims)
    for an in var.ncattrs():
        try:
            out_var.setncattr(an, var.getncattr(an))
        except Exception:
            pass
    if pft_dim_name in dims:
        data = var[:]
        # handle masked arrays by filling masked entries with zeros
        try:
            if hasattr(data, 'mask'):
                data = np.ma.filled(data, 0)
        except Exception:
            pass
        new_shape = list(data.shape)
        pidx = dims.index(pft_dim_name)
        new_shape[pidx] = new_len
        new_data = np.zeros(new_shape, dtype=data.dtype)
        for i in range(old_len):
            tgt = map_index(i)
            src_index = [slice(None)] * data.ndim
            dst_index = [slice(None)] * data.ndim
            src_index[pidx] = i
            dst_index[pidx] = tgt
            new_data[tuple(dst_index)] = data[tuple(src_index)]
        # duplicate index 13 into new index 12
        src_index = [slice(None)] * data.ndim
        dst_index = [slice(None)] * data.ndim
        src_index[pidx] = 13
        dst_index[pidx] = 12
        new_data[tuple(dst_index)] = data[tuple(src_index)]
        # copy index 13 into new index 16 (sedge)
        dst_index = [slice(None)] * data.ndim
        dst_index[pidx] = 16
        new_data[tuple(dst_index)] = data[tuple(src_index)]
        out_var[:] = new_data
    else:
        out_var[:] = var[:]

# add new flags as double with metadata
if 'nonvascular' not in new.variables:
    nv = new.createVariable('nonvascular', 'f8', (pft_dim_name,))
    nv[:] = np.zeros((new_len,), dtype='f8')
    nv[15] = 1.0
    nv.units = "logical flag"
    nv.long_name = "Binary flag for non-vascular PFT"
    nv.flag_values = np.array([0., 1.])
    nv.flag_meanings = "non-vascular"
    nv.coordinates = "pftname"
if 'graminoid' not in new.variables:
    gv = new.createVariable('graminoid', 'f8', (pft_dim_name,))
    gv[:] = np.zeros((new_len,), dtype='f8')
    gv[12:15] = 1.0   #grasses
    gv[16] = 1.0      #sedge
    gv.units = "logical flag"
    gv.long_name = "Binary flag for graminoid PFTs"
    gv.flag_values = np.array([0., 1.])
    gv.flag_meanings = "graminoid"
    gv.coordinates = "pftname"

# Add new parameter fields (use double precision)
if 'climatezone' not in new.variables:
    cz = new.createVariable('climatezone', 'f8', (pft_dim_name,))
    cz[:] = np.zeros((new_len,), dtype='f8')
    #1 = tropical, 2 = temperate, 3 = boreal, 4 = arctic
    cz[1] = 2.0
    cz[2] = 3.0
    cz[3] = 3.0
    cz[4] = 1.0
    cz[5] = 2.0
    cz[6] = 1.0
    cz[7] = 2.0
    cz[8] = 3.0
    cz[9] = 2.0
    cz[10] = 2.0
    cz[11] = 2.0
    cz[12] = 4.0
    cz[13] = 2.0
    cz[14] = 2.0
    cz[15] = 3.0
    cz[16] = 3.0
    cz.units = "category"
    cz.long_name = "Climate zone classification (1=tropical,2=temperate,3=boreal,4=arctic)"
    cz.flag_values = np.array([1., 2., 3., 4.])
    cz.flag_meanings = "tropical temperate boreal arctic"
    cz.coordinates = "pftname"
if 'needleleaf' not in new.variables:
    nl = new.createVariable('needleleaf', 'f8', (pft_dim_name,))
    nl[:] = np.zeros((new_len,), dtype='f8')
    # CLM default needleleaf pfts: 1 and 2 (needleleaf evergreen), 3 (needleleaf deciduous)
    nl[1] = 1.0
    nl[2] = 1.0
    nl[3] = 1.0
    nl.units = "logical flag"
    nl.long_name = "Binary flag for needleleaf PFTs"
    nl.flag_values = np.array([0., 1.])
    nl.flag_meanings = "needleleaf"
    nl.coordinates = "pftname"
if 'iscft' not in new.variables:
    ic = new.createVariable('iscft', 'f8', (pft_dim_name,))
    ic[:] = np.zeros((new_len,), dtype='f8')
    ic[19:] = 1.0
    ic.units = "logical flag"
    ic.long_name = "Binary flag indicating CFT inclusion"
    ic.flag_values = np.array([0., 1.])
    ic.flag_meanings = "iscft"
    ic.coordinates = "pftname"
if 'nfixer' not in new.variables:
    nf = new.createVariable('nfixer', 'f8', (pft_dim_name,))
    nf[:] = np.zeros((new_len,), dtype='f8')
    nf.units = "logical flag"
    nf.long_name = "Binary flag for nitrogen-fixing PFTs"
    nf.flag_values = np.array([0., 1.])
    nf.flag_meanings = "nfixer"
    nf.coordinates = "pftname"

# fix pftname if present
if 'pftname' in new.variables:
    arr = new.variables['pftname'][:]
    try:
        strs = chartostring(arr)
    except Exception:
        strs = np.array([''.join([c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in row]).strip() for row in arr])
    strs = list(strs)
    while len(strs) < new_len:
        strs.append('')
    strs[12] = 'c3_arctic_grass'
    strs[15] = 'moss'
    strs[16] = 'sedge'
    strs[17] = 'c3_crop'
    char_arr = stringtochar(np.array([s.ljust(new.variables['pftname'].shape[1]) for s in strs], dtype='S'))
    new.variables['pftname'][:] = char_arr

new.close()
old.close()
print('Wrote', dst_path)
