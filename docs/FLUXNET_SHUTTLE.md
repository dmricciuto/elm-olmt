# FLUXNET Shuttle inputs for OLMT

## Metadata-only sitegroup

`fluxnet_shuttle` is an OLMT sitegroup built from a Shuttle `listall`
catalogue. The 2026-10-28 catalogue includes 246 sites in North and South
America with at least five calendar years between the first and last available
years: 241 AmeriFlux and 5 ICOS sites. AmeriFlux's published-year list confirms
the count for its sites; for the five ICOS sites, the count is the Shuttle
catalogue's first-to-last span. This step does **not** download flux or
meteorological products.

The files are under `inputdata/lnd/clm2/PTCLM/`:

- `fluxnet_shuttle_sitedata.txt`: coordinates, elevation, and available years.
- `fluxnet_shuttle_pftdata.txt`: 17-PFT ELM fractions, using existing OLMT
  site-specific mappings where suitable and approximate IGBP translations
  elsewhere.
- `fluxnet_shuttle_soildata.txt`: seven sites with usable local soil texture.
  Other sites have no soil row, so OLMT retains texture from its gridded
  surface data.
- `fluxnet_shuttle_catalog.csv`: per-site provenance, PFT confidence,
  elevation source, soil source, and year-count basis.
- `fluxnet_shuttle_source_snapshot.csv`: selected source catalogue fields,
  with no contact details.

AmeriFlux elevations come from its site metadata API; four ICOS elevations
come from their station pages. Seven sites lacking a reported elevation use
OpenTopoData Mapzen terrain estimates, identified in the catalogue. The
Shuttle catalogue supplies IGBP class but no soil texture. The IGBP mapping
is a starting point, especially for wetlands, savannas, croplands, and C3/C4
grass mixtures. Review low-confidence PFTs before site-level interpretation.
For US-MOz, the selected local AmeriFlux soil texture conflicts with the older
Fluxnet2015 value; the catalogue records that conflict.

To rebuild the sitegroup after creating a fresh `fluxnet-shuttle listall`
snapshot, run from `elm-olmt`:

```bash
../.venv-fluxnet-shuttle/bin/python tools/build_fluxnet_shuttle_sitegroup.py \
  --snapshot /path/to/fluxnet_shuttle_snapshot_YYYYMMDDTHHMMSS.csv \
  --ptclm ../inputdata/lnd/clm2/PTCLM
```

Use `sitegroup = fluxnet_shuttle` in an OLMT site config to select these sites
and enable Shuttle forcing automatically. Building the catalogue does not
create forcing files for all sites; those are prepared individually when needed.

## Forcing and observations for an individual site

`tools/fluxnet_shuttle_to_olmt.py` discovers one AmeriFlux product through the
official [FLUXNET Shuttle](https://github.com/fluxnet/shuttle), downloads its
ZIP, and writes the ELM cpl-bypass forcing and monthly calibration observations.
The source ZIP and a `provenance.json` with the product citation and SHA-256
stay beside the converted files. Each run downloads into a temporary directory,
compares the ZIP SHA-256 with saved site versions, and discards the download
when unchanged. Changed products get a new dated directory; older versions
remain intact.
The US-MMS Shuttle archive contains CSV tables and no NetCDF forcing file.

Use a Python environment with the dependencies below. OLMT loads the bundled
official Shuttle source from `vendor/fluxnet_shuttle/src` when Shuttle is not
already installed; users do not need to set `PYTHONPATH` or clone the Shuttle
repository. Install the dependencies once in each OLMT Python environment:

```bash
python -m pip install numpy pandas netCDF4 matplotlib requests aiohttp aiofiles 'pydantic>=2' PyYAML
cd /path/to/elm-olmt
python tools/fluxnet_shuttle_to_olmt.py US-MMS \
  --output-root ../inputdata/atm/datm7/CLM1PT_data/fluxnet_shttle
```

The output is:

```text
../inputdata/atm/datm7/CLM1PT_data/fluxnet_shttle/
  current_US-MMS.txt
  1x1pt_US-MMS_YYYYMMDD/
    all_hourly.nc
    observations.csv
    provenance.json
    shuttle_snapshot.csv
    plots/TBOT.png (and one PNG for each other forcing variable)
    raw/AMF_US-MMS_FLUXNET_*.zip
```

The converter accepts `FLUXMET_HH`/`FLUXMET_HR` and
`FULLSET_HH`/`FULLSET_HR` products. For `HH`, it averages each pair of
half-hourly meteorology values to one hourly value and sums the two
precipitation amounts before converting to a rate. It uses the consolidated `*_F`
meteorology fields, removes February 29 for ELM's no-leap calendar, converts
the source units, derives specific humidity from temperature, vapor pressure
deficit, and pressure, and packs the seven required meteorology fields as
`int16` with `scale_factor` and `add_offset`. It reads the measurement height
from the archive's BIF metadata; `--height` overrides that value. If any
hourly forcing value is missing after using the product's consolidated fields,
conversion stops with an error. No interior gap filling is applied.
FLUXNET hourly timestamps are local standard time without daylight saving
time. The converter reads `UTC_OFFSET` from the product's BIF table and shifts
all meteorology to the UTC clock used by ELM's CPL_BYPASS solar geometry before
removing February 29. It requires a whole-hour offset; a missing or fractional
offset stops conversion rather than silently misaligning radiation. The few
UTC hours outside the product's first or last local calendar year are filled
from the nearest available hour, and their count is recorded in provenance.
An unchanged ZIP is reconverted into a new version when an older saved version
lacks this UTC alignment; the old version remains intact.
Cases with an explicit `metdir` pointing to an older version must be updated.
Their existing calibration results used misaligned shortwave and should be
rerun before interpreting fitted photosynthesis parameters.
Each new version includes a PNG for TBOT, QBOT, WIND, PSRF, FSDS, FLDS,
PRECTmms, and ZBOT. Each plot shows monthly hourly means and the mean
seasonal cycle. The blue band is the hourly mean ±1 standard deviation
(bounded by observed minima and maxima); the orange band spans the hourly
minimum to maximum for each month. Seasonal bands pool the same calendar
month across years. Precipitation is plotted in mm per hour on a symmetric
log scale so its mean and extremes remain visible together. To update plots
from saved NetCDF without downloading again, use `--replot` with the site and
output root shown above.

`observations.csv` contains monthly GPP, NEE, ecosystem respiration, latent
heat, and sensible heat where available. Carbon values are in
`g C m-2 d-1`; heat values are in `W m-2`. When an SP run requests `FPSN`,
the table loader matches it to `GPP_NT_CUT_REF` and converts the GPP value
and uncertainty to `umol CO2 m-2 s-1`, the units of ELM's `FPSN` history
field. The saved table retains its original GPP label and values. Missing
uncertainties use OLMT's established error floor and fraction. With
`sitegroup = fluxnet_shuttle`, OLMT refreshes the selected site's inputs when
creating a case. Select matching observations with:

```ini
[observations]
location = shuttle
format = table
variables = FPSN, EFLX_LH_TOT
startyear = 1999
endyear = 2023
```

`[fluxnet_shuttle] enabled = true` also enables Shuttle for another sitegroup.
Set `enabled = false` explicitly to disable it when using the
`fluxnet_shuttle` sitegroup. The `output_root` and `height` options remain
available in this section.

The output root defaults to the path shown above under the configured
`inputdata` directory. `config_files/US-MMS_SP_shuttle_flnr_mbbopt_30.cfg`
sets up a 50-year SP run starting in 1974, one 25-year forcing cycle before
the 1999–2023 observation period. It samples `flnr` and `mbbopt` in a
32-member ensemble and compares monthly FPSN with AmeriFlux GPP, plus latent
heat. Shuttle mode requires `runtype = site`, `mettype = site`, and
`use_cpl_bypass = True`.

For Docker, build the Shuttle dependency layer once. The wrapper selects it
automatically when the sitegroup enables Shuttle. It refreshes the site on the
host before launching Docker, then uses that checked version inside the
container. Set `OLMT_SHUTTLE_PYTHON` to a host Python with the dependencies
above if the default `../.venv-fluxnet-shuttle/bin/python` is unavailable:

```bash
docker build -f docker/Dockerfile.shuttle -t elmv3-shuttle:latest .
./docker/run_olmt.sh config_files/US-MMS_SP_shuttle_flnr_mbbopt_30.cfg
```

The ordinary `./docker/run_olmt.sh --shell` opens the base `elmv3` image.
For an interactive shell with Shuttle available, run
`OLMT_DOCKER_IMAGE=elmv3-shuttle:latest ./docker/run_olmt.sh --shell` on the
host. Shuttle is already present in that image; installing it from GitHub
inside the container is unnecessary.
If `elm_olmt.py` is invoked directly inside Docker, a TLS failure can make
Shuttle's live catalogue empty. OLMT then uses a complete cached site version
and warns that it could not check for updates. The warning prints a command
to refresh only the Shuttle data from the host, without starting OLMT. For
US-MMS in the default layout, exit Docker and run this from `elm-olmt`:

```bash
../.venv-fluxnet-shuttle/bin/python tools/fluxnet_shuttle_to_olmt.py US-MMS \
  --output-root ../inputdata/atm/datm7/CLM1PT_data/fluxnet_shttle
```

The Docker layer uses the official Shuttle source at commit
`028818791fef116db03e97f3f6b0aebdf1a89e90`, copied into
`vendor/fluxnet_shuttle` under its BSD 3-Clause license.

For a Docker forcing smoke run, use the included
`tests/config_files/US-MMS_shuttle_docker_smoke.cfg`:

```bash
./docker/run_olmt.sh tests/config_files/US-MMS_shuttle_docker_smoke.cfg
```

OLMT generates the site domain and surface data during case setup. The smoke
config uses the existing `Fluxnet2015` site metadata table because it includes
US-MMS; the forcing coordinates come from the Shuttle snapshot. The smoke run
simulates one accelerated spinup year and does not compare observations with a
calendar-year simulation. The wrapper prepares the Shuttle forcing and selects
the `elmv3-shuttle:latest` image for this config.

To repeat conversion without another download, pass both `--snapshot-file`
and `--archive` with the saved files. The generated product is licensed CC BY
4.0; cite the exact product shown in `provenance.json` and follow the archive's
`DATA_POLICY_LICENSE_AND_INSTRUCTIONS.txt`.
