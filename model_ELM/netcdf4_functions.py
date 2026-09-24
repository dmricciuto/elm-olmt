#Python utilities for reading and writing variables to a netcdf file
#  using Scientific Python OR scipy, whichever available
import numpy as np

def getncvar(self, fname, varname):
    from netCDF4 import Dataset
    nffile = Dataset(fname,"r")
    if varname in nffile.variables:
      varvals = nffile.variables[varname][:]
    else:
      print('Warning: '+varname+' not in '+fname)
      #raise ValueError('"%s" not in %s'%(varname,fname))
      varvals=[-1]
    nffile.close()
    return varvals

def putncvar(self, fname, varname, varvals, operator='', addvar=False):
    from netCDF4 import Dataset
    import numpy as np
    nffile = Dataset(fname,"a")
    if (varname in nffile.variables):
      if (operator == '*'):
        nffile.variables[varname][...] = nffile.variables[varname][...]*varvals
      else:
        nffile.variables[varname][...] = varvals
      ierr = 0
    else:
      if (addvar):
        #Currently only works for scalars
        newvar = nffile.createVariable(varname, np.float64, ('allpfts',))
        newvar[:] = varvals
        ierr = 0
      else:
        raise ValueError('"%s" not in %s'%(varname,fname))
        ierr = 1
    nffile.close()
    return ierr

# this function is designed specifically for extracting data from a FATES json PFT parameter file
def getjsonvar(self, fname, varname):
    import json
    import numpy as np
    with open(fname, "r", encoding="utf-8") as jsonfile:
        nffile = json.load(jsonfile)
    if "parameters" not in nffile:
        raise ValueError(f'"parameters" not in {fname}')
    parameters = nffile["parameters"]
    if varname not in parameters:
        raise ValueError(f'"{varname}" not in "parameters" in {fname}')
    variable = parameters[varname]
    if not isinstance(variable, dict) or "data" not in variable:
        raise ValueError(f'"parameters.{varname}" in {fname} does not contain a "data" field')
    return np.asarray(variable["data"])


# this function is designed specifically for adding data to a FATES json PFT parameter file
def putjsonvar(self, fname, varname, varvals, operator="", addvar=False):
    import json
    import numpy as np
    with open(fname, "r", encoding="utf-8") as jsonfile:
        nffile = json.load(jsonfile)
    if addvar:
      if "parameters" not in nffile:
          raise ValueError(f'"parameters" not in {fname}')
      parameters = nffile["parameters"]
      if varname not in parameters:
          raise ValueError(f'"{varname}" not in "parameters" in {fname}')
      variable = parameters[varname]
      if not isinstance(variable, dict) or "data" not in variable:
          raise ValueError(f'"parameters.{varname}" in {fname} does not contain a "data" field')
      if isinstance(varvals, np.ndarray):
          variable["data"] = varvals.tolist()
      elif isinstance(varvals, np.generic):
          variable["data"] = varvals.item()
      else:
          variable["data"] = varvals
    elif operator == "*" or not addvar:
        raise ValueError(f'no methods yet in putjsonvar for operator {operator} or addvar False')
    with open(fname, "w", encoding="utf-8") as jsonfile:
        json.dump(nffile, jsonfile, indent=2)
    return 0
