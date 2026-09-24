Offline Land Model Testbed (OLMT)
Contact:  Dan Ricciuto (ricciutodm@ornl.gov)

Updated 2/25/2025

The purpose of the Offline Land Model Testbed (OLMT) is to simplify the workflows for single site, regional and ensemble offline ELM simulations, which are otherwise cumbersome using only CIME. We have been working on a new version with an improved interface.  The user now can run a simulation using a single python run script that will perform the entire workflow.  This usually involves setting up three cases for biogeochemistry-enabled runs: the ad spinup, final spinup and transient simulations.  It is also possible to add additional cases beginning in later years where we apply treatment effects or otherwise modify forcings.  For example, in the SPRUCE study we have 10 experimental treatments (different levels of temperature and CO2 modifications) that begin in 2015.  For those point simulations, we have a single run script that launches and manages 13 cases. 
 
For each case, the runscript will perform the create_newcase, case setup, and submission.  The case.build will be performed on the first case only, and then the same executable will be used for following cases.  When submitting cases, the correct dependencies will be applied, such that the second case will start running after the first has finished, etc.

Generated case inputs are written directly to that case's run directory. The
run-directory path contains the configured case prefix (normally a date) and
the complete case name. In particular, `domain.nc`, `surfdata.nc`,
`clm_params.nc`, `CNP_parameters.nc`, and any generated dynamic-landuse or
FATES parameter file are not staged through a shared `elm-olmt/temp`
directory. This makes setup of different cases safe to run concurrently and
keeps every run's exact inputs beside its output. The executable may still be
shared by build-compatible cases.
 
Users can customize the simulations to run single points, a list of lat/lon coordinates, rectangular regions or global simulations. OLMT assumes surface, land use and domain data already exist (using the defaults for specified compsets and resolution) and will extract points or regions from these data.  The user can also set custom files, for example ultra high-resolution files created from kilocraft.  For single point runs, the PFT and soil information can be set to match observations, for example at AmeriFlux sites.
 
Finally, OLMT also has the capability to perform ensemble simulations.  Users can specify a list of parameters, with allowable ranges for each. Random samples can then be created.  Alternatively, the user can provide their own files with different parameter combinations.  When this ensemble option is enabled, the cases will be set up in the same way as above, but then multiple copies of run directories will be created. We then use another MPI-enabled python script to manage the multiple simulations in parallel. Long multi-site cases can be split into dependent scheduler jobs with `resubmit_years` in the `[run_lengths]` config section; ensemble runs can use the same value or override it with `resubmit_years` in `[ensemble]`. For example, 200 years of ad spinup and 600 years of final spinup with `resubmit_years = 100` submits 8 dependent jobs for each segmented workflow. Users can also specify a list of output variables and time frequency for which to postprocess.  Indexed postprocessing variables can use suffixes like `_pft` or `_col`, with matching `pfts` or `cols` lists in the `postprocessing` config.  A matrix of output values for all ensemble members is then created at the end of the simulation, which can then be used in Uncertainty quantification applications discussed in the other epics.

An optional `[calibration]` section controls automated MCMC after ensemble
postprocessing. `variables` limits surrogate training to calibration targets;
`mcmc_steps`, `nwalkers`, `fit_error`, and `run_gsa` control the analysis; and
`ordered_parameter_constraints` accepts comma-separated rules such as
`vpd_min_moss < vpd_max_moss`. With `mode = multitreatment`, each treatment
job writes its trained surrogate and skips its single-case MCMC. The final
treatment job depends on all other treatments and runs one joint calibration.
Portable `postprocessed_output.nc` files include the exact member-by-parameter
design, parameter names, PFT indices, bounds, and a design checksum.

Please see the wiki page for instructions and examples.
