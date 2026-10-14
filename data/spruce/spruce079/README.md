# SPRUCE water-table observations for treatment diagnostics

The standard SPRUCE diagnostics read
`spruce_daily_water_table_processed_20250408.nc`. It contains daily
plot-level water-table heights relative to the mean hollow elevation,
with positive values above the hollow. The file was processed from
`WEW_Daily_WT_Updated_20250408.csv`: missing values marked `-9999`
were converted to NaN, dates were constructed from year and day of year,
and the source `WTNorm` columns were retained as height above hollow.
The NetCDF also provides the negated depth-below-hollow representation.

Source: Hanson, P. J., J. R. Phillips, W. R. Nettles, K. J. Pearson,
and L. A. Hook (2020), *SPRUCE Plot-Level Water Table Data Assessments
for Absolute Elevations and Height with Respect to Mean Hollows Beginning
in 2015*, Oak Ridge National Laboratory, TES SFA.
https://doi.org/10.25581/spruce.079/1608615

The other observation inputs used by the PDF workflow are already
tracked under `data/spruce`: the annual carbon budget, porewater profiles,
and McFarlane peat-core characteristics. The original download archive,
dataset webpage snapshot, assessment PDF, and generated plots are not
needed to run the diagnostics and are not part of this input set.
