# SPRUCE pretreatment peat-core observations

`Peat_Characteristics_T0_20180425.csv` is the public 2012 SPRUCE S1-Bog
peat-core data product used by McFarlane et al. (2018). It contains physical,
chemical, stable-isotope, and radiocarbon observations from the 16
pretreatment plots.

Primary data citation:

> Iversen, C. M., P. J. Hanson, D. J. Brice, J. R. Phillips, K. J.
> McFarlane, E. A. Hobbie, and R. K. Kolka. 2014. SPRUCE Peat Physical and
> Chemical Characteristics from Experimental Plot Cores, 2012. Oak Ridge
> National Laboratory, TES SFA. https://doi.org/10.3334/CDIAC/spruce.005

Paper citation:

> McFarlane, K. J., P. J. Hanson, C. M. Iversen, J. R. Phillips, and D. J.
> Brice. 2018. Local Spatial Heterogeneity of Holocene Carbon Accumulation
> throughout the Peat Profile of an Ombrotrophic Northern Minnesota Bog.
> *Radiocarbon* 60:941–962. https://doi.org/10.1017/RDC.2018.37

The plotting workflow in
`olmt_diagnostics/plot_spruce_mcfarlane_profiles.py` collapses duplicate
laboratory records for each sampled interval, retains the 16 treed profiles
plus the two non-treed profile locations analyzed in the paper, aligns the
hummock and hollow cores to their respective local surfaces, and reports
spatial variation across core locations. The original CSV is retained
unchanged.
