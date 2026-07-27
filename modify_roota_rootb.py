#!/usr/bin/env python3
"""
Script to modify roota_par and rootb_par values at a specific index in a NetCDF file.
"""
import sys
from netCDF4 import Dataset

def modify_root_parameters(filename, index, roota_value=10, rootb_value=0):
    """
    Modify roota_par and rootb_par values at the specified index.
    
    Args:
        filename (str): Path to the NetCDF file
        index (int): Python index to modify (0-based)
        roota_value (float): New value for roota_par
        rootb_value (float): New value for rootb_par
    """
    try:
        print(f"Opening {filename} for modification...")
        
        # Open NetCDF file in append mode
        with Dataset(filename, 'a') as nc_file:
            # Check if variables exist
            if 'roota_par' not in nc_file.variables:
                print("Error: 'roota_par' not found in the file")
                return False
            if 'rootb_par' not in nc_file.variables:
                print("Error: 'rootb_par' not found in the file")
                return False
            
            # Get current values
            roota_var = nc_file.variables['roota_par']
            rootb_var = nc_file.variables['rootb_par']
            
            print(f"Variable shapes:")
            print(f"  roota_par: {roota_var.shape}")
            print(f"  rootb_par: {rootb_var.shape}")
            
            # Check if index is valid
            if index >= roota_var.shape[0] or index >= rootb_var.shape[0]:
                print(f"Error: Index {index} is out of bounds")
                return False
            
            # Show current values
            print(f"Current values at index {index}:")
            print(f"  roota_par[{index}] = {roota_var[index]}")
            print(f"  rootb_par[{index}] = {rootb_var[index]}")
            
            # Modify values
            roota_var[index] = roota_value
            rootb_var[index] = rootb_value
            
            print(f"New values at index {index}:")
            print(f"  roota_par[{index}] = {roota_var[index]}")
            print(f"  rootb_par[{index}] = {rootb_var[index]}")
            
            print("Successfully modified the NetCDF file!")
            return True
            
    except Exception as e:
        print(f"Error modifying file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python modify_roota_rootb.py <netcdf_file> [index] [roota_value] [rootb_value]")
        print("Example: python modify_roota_rootb.py clm_params_SPRUCE_20250507.nc 12 10 0")
        sys.exit(1)
    
    filename = sys.argv[1]
    index = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    roota_value = float(sys.argv[3]) if len(sys.argv) > 3 else 10
    rootb_value = float(sys.argv[4]) if len(sys.argv) > 4 else 0
    
    success = modify_root_parameters(filename, index, roota_value, rootb_value)
    if not success:
        sys.exit(1)