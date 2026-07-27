#!/usr/bin/env python
import sys,os, time
import numpy as np
import subprocess
import pickle
import model_ELM
import matplotlib.pyplot as plt
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--case", dest="case", default="", \
                  help="Case name")
parser.add_option("--cases_compare", dest="cases_compare", default="", \
                  help="Additional cases to plot for comparison (comma-separated)")
parser.add_option("--plot_spinup", dest="spinup", default=False, \
                  action="store_true")
(options, args) = parser.parse_args()

#Post-process and plot variables (multi-case mode)

# Parse case names
case_names = [options.case]
if options.cases_compare:
    case_names.extend(options.cases_compare.split(','))

# Load all case objects
cases = {}
for case_name in case_names:
    case_name = case_name.strip()
    myfile = open('pklfiles/'+case_name+'.pkl','rb')
    cases[case_name] = pickle.load(myfile)
    myfile.close()

# Use the first case as reference for processing parameters
reference_case = cases[case_names[0]]

if options.spinup:
    # Plot spinup for each case (existing functionality)
    for case_name, case_obj in cases.items():
        case_obj.plot_spinup()
else:
    # Process each variable across all cases
    for v in reference_case.postproc_vars:
        if ('_pft' in v):
            hnum = 2 #Assume pft-level output is in h2 file
            myindex = list(reference_case.postproc_pfts)
        elif ('_col' in v):
            hnum = 2 #Assume column-level output is in h2 file
            myindex = list(reference_case.postproc_cols)
        else:
            hnum = 1
            myindex = [0]
        
        # Set up processing options
        annualmean = False
        dailytomonthly = False
        meanseasonalcycle = False
        if (reference_case.postproc_freq.lower() == 'monthly'):
            dailytomonthly = True
        elif (reference_case.postproc_freq.lower() == 'annual'):
            annualmean = True
        
        # Process each index (e.g., different PFTs)
        for i in myindex:
            print(v,myindex)
            # Create a single plot for all cases
            plt.figure(figsize=(12, 6))
            
            # Process each case for this variable and index
            for case_name, case_obj in cases.items():
                # Determine if we should plot individual cases
                plot_individual = len(cases) == 1  # Only plot if single case
                
                # Process the data
                case_obj.postprocess(v, index=i, gindex=0, 
                                   startyear=case_obj.postproc_startyear, 
                                   endyear=case_obj.postproc_endyear, hnum=hnum, 
                                   dailytomonthly=dailytomonthly, annualmean=annualmean,  
                                   meanseasonalcycle=meanseasonalcycle, 
                                   xindex=0, yindex=0, plot=plot_individual)
                
                # Only create multi-case plots if we have multiple cases
                if len(cases) > 1:
                    # Get the processed data
                    var_out = v
                    if ('_pft' in v or '_col' in v):
                        var_out = var_out + str(i)
                    
                    # Plot this case's data on the combined plot
                    plt.plot(case_obj.output['taxis'], case_obj.output[var_out], 
                            label=case_name, linewidth=2)

            # Only finalize and save multi-case plot if we have multiple cases
            if len(cases) > 1:
                # Format the combined plot
                plt.ylabel(var_out) # + ' (' + case_obj.postprocess_get_units(v) + ')')
                plt.xlabel('Years')
                plt.title(f'{var_out} - Multi-Case Comparison')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save the combined plot
                os.system('mkdir -p '+reference_case.rundir+'/../diagnostics')
                plt.savefig(reference_case.rundir+'/../diagnostics/multicase_plot_'+var_out+'_'+
                           str(reference_case.postproc_startyear)+'-'+
                           str(reference_case.postproc_endyear)+'.png', dpi=300)
                plt.close()
                
                print(f"Created multi-case plot for {var_out}")
            else:
                # Close the figure we created but didn't use for single case
                plt.close()

    for case_name, case_obj in cases.items():
        if hasattr(case_obj, 'write_peatlands_pft_postprocessed_netcdf'):
            case_obj.write_peatlands_pft_postprocessed_netcdf(
                    startyear=case_obj.postproc_startyear,
                    endyear=case_obj.postproc_endyear)
