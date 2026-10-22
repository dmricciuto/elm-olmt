# Guide to four new ELM-Peatlands physics capabilities

This guide describes four capabilities on the `ELM-Peatlands-Microbe` branch
for colleagues who need to understand, configure, run, or evaluate them. The
commit hashes identify where each capability entered the reconstruction, but
the descriptions below follow the **current branch implementation**, including
subsequent refinements.

1. `a67c1740bd` — hydrologic acrotelm and conservative peat burial
2. `cf1dc3328f` — configurable density-based JULES peat hydraulics
3. `bd01c4e0b4` — prognostic Sphagnum water storage
4. `12ba7a970e` — controlled deep-soil warming treatment

The first two capabilities are tightly coupled: modeled peat carbon density
can determine both burial and hydraulic properties. The Sphagnum store couples
canopy water loss to the upper peat. The warming treatment is independent of
those three capabilities, but is designed for the same SPRUCE treatment
workflow.

## Quick configuration summary

| Capability | Main switch | Default behavior | Required context |
| --- | --- | --- | --- |
| Hydrologic acrotelm and burial | `use_peatland_compaction_profile` | Defaults true when peatland vertical transport is active | `use_peatland_vertical_transport=.true.` and a positive topounit peat depth |
| JULES peat hydraulics | `use_jules_peat_hydraulics` | Defaults true when the compaction profile is active | `use_peatland_compaction_profile=.true.` |
| Prognostic Sphagnum water | `use_prognostic_moss_water` | Defaults true when `use_humhol=.true.` | HUMHOL and a PFT with `nonvascular=1` |
| Controlled deep-soil warming | `use_deep_soil_heating` | False | A reference-temperature stream and a bog topounit with positive peat depth |

For an explicit modern peatland configuration:

```ini
[case_options]
use_humhol = .true.
use_peatland_vertical_transport = .true.
use_peatland_compaction_profile = .true.
use_jules_peat_hydraulics = .true.
use_prognostic_moss_water = .true.
```

The scientific coefficients for peat burial, peat hydraulics, and Sphagnum
water are ELM parameter-file variables. They should be varied through an OLMT
`[parameters]` section or an alternate parameter file, not converted into new
namelist controls.

---

## 1. Hydrologic acrotelm and conservative peat burial

**Introduced in:** `a67c1740bd`

### Purpose

This capability gives peat accumulation a physical storage-and-throughflow
interpretation. Each layer has a target solid-carbon density. New solid carbon
first fills available storage in its source layer; once that storage demand is
met, the remaining carbon is buried into the next layer. Incoming carbon
changes the receiving layer before it can move farther down, so material cannot
pass through multiple layers instantaneously.

The boundary between low-density acrotelm peat and denser catotelm peat is
linked to the water-table regime. This represents the hypothesis that the
depth exposed to sustained growing-season aeration controls the depth of the
more porous, rapidly cycling acrotelm.

### Where it applies

The dynamic calculation is used only when all of the following are true:

- `use_peatland_vertical_transport=.true.`;
- `use_peatland_compaction_profile=.true.`; and
- the column's topounit has `TopounitPeatDepth > 0`.

The implementation uses the column-to-topounit mapping, not a hard-coded
column number. It therefore works with the three-topounit SPRUCE layout and
with other topounit orderings. Non-peat columns retain the standard ELM
vertical-transport path.

### How the acrotelm depth is determined

There are two supported modes.

#### Prescribed mode

The surface-data field `TopounitAcrotelmDepth` supplies a depth below each
topounit's **local surface**. A positive value fixes the density-transition
boundary for that topounit. The value must not exceed `TopounitPeatDepth`.

This is useful at a site such as SPRUCE, where the hummock and hollow have
different local reference surfaces and the observed acrotelm constraint can be
encoded directly. The model still diagnoses its hydrologic acrotelm metrics so
the prescribed value can be evaluated against the simulated water table.

#### Prognostic mode

A zero `TopounitAcrotelmDepth` enables the hydrologic diagnostic:

1. During the configured growing season, instantaneous `ZWT` is smoothed with
   a running mean. The default time scale is 30 days.
2. Only samples with less than 50% ice in the first soil layer and with the
   water table inside the resolved peat column are accepted.
3. The deepest running-mean water table reached during each growing season is
   retained.
4. The model stores five annual values and takes their 90th percentile. This
   represents a recurrent dry-season boundary without allowing one isolated
   drought to define the profile.
5. After a complete five-year window exists, the acrotelm depth approaches
   that statistic over a default 10-year smoothing time scale.

The cold-start boundary is 0.25 m by default. Hydrologic statistics advance in
physical time, not accelerated-decomposition time, so accelerated spinup does
not create hundreds of artificial hydrologic years per model year.

### Target carbon-density profile

The current target is a two-density profile with a smooth transition:

- `peat_compaction_surface_density` above the acrotelm;
- `peat_compaction_deep_density` below the transition; and
- a linear transition over `peat_compaction_transition_width`.

The layer-average target is integrated across the actual layer boundaries.
This matters when a coarse layer straddles the acrotelm boundary.

The standard SPRUCE parameter file currently uses 20 kg C m^-3 for surface
peat, 90 kg C m^-3 for deep peat, and a 0.10 m transition width.

### Conservative burial algorithm

For donor layer *j*, let the physical-equivalent solid-carbon density be
`rho(j)`, the target density be `rho_target(j)`, and layer thickness be
`dz(j)`. The layer's storage demand is

```text
storage_demand(j) = [rho_target(j) - rho(j)] * dz(j) / tau_compaction
```

The local solid-carbon source is the sum of the layer's non-dissolved
decomposition source terms. Internal solid-to-solid cascade transfers cancel
in that sum, while new litter input and losses to DOM or gas remain. The
downward burial flux is

```text
burial_flux(j) = max[0, local_solid_C_source(j) - storage_demand(j)]
```

and the interface velocity is

```text
burial_velocity(j) = burial_flux(j) / rho(j).
```

The flux is capped by the carbon actually available in the donor layer. The
bottom boundary is closed. The same conservative transport matrix moves the
solid C, N, P, C-13, and C-14 state, preserving the transported material's
stoichiometry and isotope composition. DOM is deliberately excluded because
it belongs to the aqueous transport calculation.

A layer below target retains enough of its local production to fill storage.
A layer at target continuously transmits its net new solid-carbon production.
A layer above target also releases existing excess storage over the compaction
time scale. Carbon arriving from above is incorporated into the receiving
layer and can influence burial at the next update; it is not teleported through
the profile.

### Accelerated spinup

Accelerated decomposition reduces raw slow-pool stocks. The model therefore
reconstructs a physical-equivalent density using the same pool acceleration
factors used by the spinup transport calculation, including the accumulated
scalar adjustment after year 40. This prevents the artificial AD pool sizes
from being interpreted as an extremely low-density peat column.

The accelerated transport velocities remain a numerical spinup device. Final
spinup and transient simulations use the common physical burial profile.

### Parameters

| Parameter | Units | Current value | Meaning |
| --- | --- | ---: | --- |
| `peat_compaction_surface_density` | kg C m^-3 | 20 | Acrotelm target carbon density |
| `peat_compaction_deep_density` | kg C m^-3 | 90 | Catotelm target carbon density |
| `peat_compaction_initial_acrotelm_depth` | m | 0.25 | Boundary used before a complete hydrologic window exists |
| `peat_compaction_transition_width` | m | 0.10 | Width of the density transition below the acrotelm boundary |
| `peat_compaction_zwt_running_mean_days` | days | 30 | Time scale used to identify sustained drawdown |
| `peat_compaction_zwt_smoothing_years` | years | 10 | Adjustment time scale for the prognostic boundary |
| `peat_compaction_growing_season_start_doy` | day of year | 121 | First accepted hydrologic sample |
| `peat_compaction_growing_season_end_doy` | day of year | 305 | Last accepted hydrologic sample |
| `peat_compaction_timescale_years` | years | 1 | Relaxation time for density excess/deficit |

`peat_compaction_efolding_depth` remains in the dated parameter file for
compatibility with the earlier target-profile implementation, but it does not
control the current hydrologic acrotelm profile.

### Recommended diagnostics

| History variable | Units | Interpretation |
| --- | --- | --- |
| `PEAT_C_DENSITY` | g C m^-3 | Physical-equivalent modeled solid-C density |
| `PEAT_C_TARGET_DENSITY` | g C m^-3 | Layer-average target density |
| `PEAT_C_BURIAL_FLUX` | g C m^-2 s^-1 | Downward solid-C flux leaving each donor layer |
| `SOM_ADV_COEF` | m s^-1 | Corresponding transport velocity |
| `PEAT_ZWT_RUNMEAN` | m | Sustained water-table diagnostic |
| `PEAT_ZWT_GS_DEEPEST` | m | Deepest sustained value in the current growing season |
| `PEAT_ZWT_GS_P90` | m | 90th percentile of the annual low-water history |
| `PEAT_ACROTELM_DEPTH` | m | Boundary actually used by the density profile |

For evaluation, plot modeled and target density together, then inspect burial
flux and C-14 age. A smooth density profile alone is not sufficient: the
burial flux must also give plausible peat accumulation and age gradients.

### Restart behavior and attribution tests

The running mean, annual low-water history, acrotelm depth, elapsed sampling
time, and statistic year are restartable. A cold start initializes the
hydrologic memory; an older restart without these fields can initialize the new
state, but it does not reproduce a mature five-year hydrologic window.

Use these controlled comparisons:

- `use_peatland_compaction_profile=.false.` recovers the earlier prescribed,
  constant-with-depth peat burial velocity.
- Keep compaction on and prescribe `TopounitAcrotelmDepth` to separate burial
  behavior from water-table errors.
- Keep the hydrologic acrotelm prognostic and inspect `PEAT_ZWT_*` to evaluate
  whether the simulated drought regime is defining a reasonable boundary.

### Scientific status

The formulation is conservative, but its target densities, transition width,
compaction time scale, growing-season window, and 90th-percentile definition
remain calibration choices. Evaluate them jointly against dry bulk density,
carbon density, peat accumulation, C-14 profiles, and observed water tables.
The carbon-density target must be converted consistently before comparison
with observations reported as dry bulk density.

---

## 2. Density-based JULES peat hydraulics

**Introduced in:** `cf1dc3328f`

### Purpose

The historical peatland branch prescribed hydraulic properties as functions
of depth. The JULES option instead derives porosity, retention, and saturated
hydraulic conductivity from dry organic-matter bulk density. When combined
with the burial capability, carbon accumulation changes density and density
changes water movement.

This closes an important feedback:

```text
peat production and burial
    -> carbon density
    -> dry organic-matter bulk density
    -> porosity, suction, b, and Ksat
    -> water table and soil moisture
    -> vegetation and decomposition
```

### Activation and fallback

```ini
use_peatland_compaction_profile = .true.
use_jules_peat_hydraulics = .true.
```

JULES peat hydraulics require the compaction profile. ELM stops with a clear
error if the JULES switch is true while compaction is false. An explicit

```ini
use_jules_peat_hydraulics = .false.
```

retains the earlier depth-based fibric-to-sapric hydraulic profile. Non-peat
columns retain the standard ELM calculation in either case.

### Density conversion

The burial module diagnoses solid carbon density in g C m^-3. It is converted
to dry organic-matter bulk density as

```text
rho_dry = 0.001 * rho_C / f_C
```

where `f_C` is `peat_hydraulic_carbon_fraction`. The result is clamped to the
empirical density range before applying the JULES relationships. At cold
start, the available surface-data organic bulk density supplies the initial
hydraulic properties. During integration, the properties are refreshed from
the modeled physical-equivalent peat carbon density.

### Hydraulic relationships

For dry bulk density `rho` in kg dry organic matter m^-3, the implementation
uses equations 13–16 from [Chadburn et al. (2022)](https://doi.org/10.5194/gmd-15-1633-2022):

```text
saturated suction [mm] = 1000 * exp(0.0230 * rho - 5.08)
Clapp-Hornberger b      = 0.0304 * rho + 1.53
Ksat [mm s-1]          = 1000 * exp(-0.0532 * rho - 6.63)
porosity                = 1 - rho / particle_density
```

ELM then recomputes the dependent water-content thresholds used by its
hydrology. Low-density peat is more porous and conductive; denser peat holds
less pore space and has much lower saturated conductivity.

### Parameters

| Parameter | Units | Current value | Meaning |
| --- | --- | ---: | --- |
| `peat_hydraulic_carbon_fraction` | 1 | 0.56 | Converts carbon mass to dry organic matter |
| `peat_hydraulic_particle_density` | kg m^-3 | 1260 | Organic-matter particle density used for porosity |
| `peat_hydraulic_bulk_density_min` | kg m^-3 | 35 | Lower empirical bound passed to the equations |
| `peat_hydraulic_bulk_density_max` | kg m^-3 | 210 | Upper empirical bound passed to the equations |

`peat_hydraulic_b_max` and `peat_hydraulic_hksat_min` belong to the retained
legacy depth-based path. They are not tuning parameters for the JULES density
equations.

### Recommended diagnostics

At minimum, retain:

- `PEAT_C_DENSITY` and `PEAT_C_TARGET_DENSITY`;
- `ZWT`, `H2OSFC`, `SOILLIQ`, and `SOILICE`;
- porosity/effective porosity and the soil hydraulic state when requested by
  the active history configuration;
- `QDRAI`, `QCHARGE`, runoff, and evapotranspiration; and
- plant productivity and water-stress diagnostics.

Inspect hummock and hollow separately. Similar water tables or stresses do not
necessarily imply identical local saturation because their elevations and
local surface coordinates differ.

### Recommended attribution experiment

Run two otherwise identical mature-restart simulations:

1. compaction on, JULES hydraulics off; and
2. compaction on, JULES hydraulics on.

Compare growing-season water-table distributions, peak ET, recharge, drainage,
soil ice, productivity, and carbon-density evolution. A 50-year accelerated
spinup can reveal large feedbacks, but a 100-year AD plus 10-year final spinup
is a more useful comparison for peat profiles and C-14.

### Scientific status and cautions

The equations are literature-based, but their application to the modeled
carbon density introduces assumptions about peat carbon fraction, particle
density, and empirical density bounds. A hydrologic improvement should not be
judged from water table alone. Confirm that peat density, porosity, storage,
hydraulic conductivity, ET, drainage, and productivity remain mutually
consistent.

---

## 3. Prognostic Sphagnum water storage

**Introduced in:** `bd01c4e0b4`

### Purpose

The retained legacy moss formulation diagnosed tissue water from liquid
volumetric water content in soil layers 3 and 4. That proxy cannot represent
water held in living Sphagnum independently of instantaneous soil VWC.

The new option adds a conserved, restartable living-moss water store. Canopy
evaporation removes water from the store, and peat hydrology recharges it
through a potential-gradient calculation. Photosynthesis receives water per
unit moss dry mass, preserving the units expected by the existing moss
physiology.

### Where it applies

The prognostic store is active only when:

- `use_humhol=.true.`;
- `use_prognostic_moss_water=.true.`; and
- the PFT parameter `nonvascular` equals 1.

The feature is not tied to a hard-coded PFT index. Vascular plants continue to
use the ordinary root-water pathway.

`use_moss_capillary_nutrients` is a separate option. It controls a moss mineral
N access hypothesis and does not turn the moss water store on or off.

### State, capacity, and physiological water content

The state variable `H2O_MOSS_STORAGE` has units kg H2O m^-2. Live moss carbon
is converted to dry biomass using

```text
moss dry mass = TOTVEGC * 0.001 / moss_carbon_fraction_dry_mass.
```

The internal water ratio is

```text
w = H2O_MOSS_STORAGE / moss dry mass,
```

which is numerically equivalent to g H2O g^-1 dry mass. Effective saturation
is

```text
Se = clamp[(w - w_min) / (w_max - w_min), 10^-6, 1].
```

Moss matric potential and conductivity are then

```text
psi_moss = -psi_sat * Se^(-b_moss)
K_moss   = Ksat_moss * Se^(conductivity_exponent).
```

This hydraulic curve is used for moss-soil exchange. The dry-mass water ratio
is also supplied to the existing moss photosynthetic water-response function.

### Conservative exchange with peat

Exchange is operator split against the upper soil layer. The resistance adds
a moss path and a soil path. The head difference between upper peat and the
living moss determines the potential exchange.

Positive `QFLX_MOSS_SOIL` means water moves from soil to moss. The calculation
is limited by:

- available liquid water in the upper soil layer;
- the receiving moss capacity;
- receiving soil pore space when water drains from moss; and
- the hydraulic conductivities of the moss and soil paths.

Below the structural holding capacity, only soil-to-moss recharge is allowed.
Tightly held moss water does not drain back merely because the instantaneous
gradient points downward. Water above `moss_water_drainage_threshold` can
return to the upper peat. If that layer is full, excess water remains in the
moss store and drainage is retried later rather than discarded.

The moss atmospheric loss calculated by canopy fluxes is withdrawn from this
store and remains part of total evapotranspiration. It is not also imposed as a
vascular root-water sink.

### Cold start and restart behavior

Cold start uses a small live Sphagnum seed and a nonzero water ratio so moss can
photosynthesize immediately:

```text
moss_initial_carbon = 10 g C m^-2
moss_carbon_fraction_dry_mass = 0.5
moss_water_initial_content = 5 g H2O g^-1 dry mass
```

This gives 20 g dry mass m^-2 and 0.10 kg H2O m^-2. At 200 g C m^-2, the
default maximum ratio of 8 g g^-1 corresponds to 3.2 kg H2O m^-2.

`H2O_MOSS_STORAGE` is written to restart files. When initializing from an older
restart that lacks the state, ELM initializes it after the carbon state is
available so storage scales with actual live moss biomass.

### Parameters

| Parameter | Units | Current value | Meaning |
| --- | --- | ---: | --- |
| `moss_water_layer_thickness` | m | 0.05 | Hydraulic path length, not a fixed storage volume |
| `moss_water_saturated_suction` | mm | 10 | Magnitude of moss suction at saturation |
| `moss_water_clapp_hornberger_b` | 1 | 3.5 | Retention exponent |
| `moss_water_hydraulic_conductivity_sat` | mm s^-1 | 1.5 | Saturated moss conductivity |
| `moss_water_conductivity_exponent` | 1 | 5 | Conductivity response to effective saturation |
| `moss_initial_carbon` | g C m^-2 | 10 | Cold-start live moss seed |
| `moss_carbon_fraction_dry_mass` | g C g^-1 dry mass | 0.5 | Carbon-to-dry-mass conversion |
| `moss_water_initial_content` | g H2O g^-1 dry mass | 5 | Cold-start water ratio |
| `moss_water_content_min` | g H2O g^-1 dry mass | 0.5 | Ratio mapped to zero effective saturation |
| `moss_water_content_max` | g H2O g^-1 dry mass | 8 | Ratio mapped to full effective saturation |
| `moss_water_drainage_threshold` | g H2O g^-1 dry mass | 8.5 | Structural holding capacity above which drainage is allowed |

These values are initial hypotheses and calibration targets, not a universal
Sphagnum hydraulic parameterization.

### Recommended diagnostics

| History variable | Units | Interpretation |
| --- | --- | --- |
| `H2O_MOSS_STORAGE` | kg H2O m^-2 | Conserved water store |
| `H2O_MOSS_INTER` | g H2O g^-1 dry mass | Internal dry-mass water ratio |
| `H2O_MOSS_WC` | g H2O g^-1 dry mass | Water content passed to moss physiology |
| `MOSS_WATER_POTENTIAL` | mm | Diagnosed living-moss matric potential |
| `QFLX_MOSS_SOIL` | mm H2O s^-1 | Soil-to-moss flux; positive is recharge |

Also retain `GPP_pft`, `NPP_pft`, `TOTVEGC_pft`, moss evaporation/transpiration,
upper-layer liquid water, soil potential, snow, and precipitation.

For a short diagnostic run, plot the terms around the first growing-season dry
event. Check that atmospheric loss lowers storage, soil recharge raises it,
and photosynthesis responds to the water ratio without an unexplained
timestep-scale collapse.

### Legacy comparison

```ini
use_prognostic_moss_water = .false.
```

restores the diagnostic mapping from mean liquid VWC in soil layers 3 and 4.
This is useful for attribution, but it should not be interpreted as a
conserved moss water budget.

### Scientific status and cautions

The major uncertainty is the mapping among dry-mass water content, matric
potential, hydraulic conductivity, and photosynthetic stress. Hummock and
hollow Sphagnum may need separate parameterizations. Validation should use
observed water content or water potential together with GPP/NPP, evaporation,
upper-peat moisture, and water-table position.

Because capacity scales with live biomass, a nearly dead moss PFT has very
little storage and recharge capacity. Always inspect biomass and the seed
initialization before diagnosing the hydraulic parameters.

---

## 4. Controlled deep-soil warming treatment

**Introduced in:** `12ba7a970e`

### Purpose

This capability represents the SPRUCE deep-peat heating treatment while
retaining ELM's prognostic soil-temperature calculation. It does **not** read a
temperature profile and overwrite `TSOI`, and it is not a biogeochemistry-only
temperature offset. Instead, it adds an explicitly budgeted heat source to the
soil thermal matrix.

The treatment follows an untreated T0.00 reference. At each time step, the
controller compares modeled temperature at 2 m with the reference temperature
plus the prescribed treatment increment. Heat is added in the 2–3 m interval
to approach that target.

### Controller calculation

The target temperature is

```text
T_target = T_reference + treatment_offset.
```

`T_control` is interpolated from the modeled soil-temperature profile at
`soil_heating_control_depth`. The requested heater power is

```text
P = min[Pmax, max(0, C_heated * (T_target - T_control) / tau_controller)].
```

This is a one-sided heater: it can add heat but does not actively cool the
profile when the model is above target. The total power is distributed among
layers intersecting the heater interval in proportion to each layer's heat
capacity and overlap with that interval. The layer sources enter the same
implicit thermal solve as the other energy terms and are included in the soil
energy budget.

The controller is applied only to bog topounits with positive peat depth and
only on or after `soil_heating_start_ymd`.

### Namelist controls

| Option | Units | Default | Meaning |
| --- | --- | ---: | --- |
| `use_deep_soil_heating` | logical | false | Enables the controller |
| `soil_heating_reference_file` | path | required when enabled | NetCDF stream containing `T_SOIL_REFERENCE` |
| `soil_heating_stream_year_first` | year | 1 | First year in the reference stream |
| `soil_heating_stream_year_last` | year | 1 | Last year in the reference stream |
| `soil_heating_model_year_align` | year | 1 | Model year aligned with the first stream year |
| `soil_heating_start_ymd` | YYYYMMDD | 0 | First date when heating can occur; SPRUCE uses 20150815 |
| `soil_heating_target_offset` | K | 0 | Treatment increment above T0.00 |
| `soil_heating_control_depth` | m | 2.0 | Depth used for controller error |
| `soil_heating_top_depth` | m | 2.0 | Top of heated interval |
| `soil_heating_bottom_depth` | m | 3.0 | Bottom of heated interval |
| `soil_heating_controller_timescale_days` | days | 1.0 | Response time for correcting a temperature error |
| `soil_heating_max_power` | W m^-2 | 60 | One-sided power cap |

The controller time scale is not the duration of the treatment. It determines
how aggressively the controller requests energy for a given temperature
error. The power cap represents the approximately 59 W m^-2 installed heater
capacity used to motivate the default.

### OLMT treatment workflow

The standard workflow runs TAMB and T0.00 normally. T0.00 writes an hourly,
column-level 2 m temperature history. OLMT converts that history to
`soil_temperature_reference.nc`. Warmed treatments wait for that file and then
use it as their reference stream.

The relevant section of the standard SPRUCE configuration is:

```ini
[treatment_options]
nyears = 9
soil_heating_reference_treatment = T0.00
soil_heating_reference_columns = 1,2
soil_heating_reference_history_tape = 3
soil_heating_start_ymd = 20150815

[treatment1]
name = TAMB
metdir = %(inputdata)s/SPRUCE_data/2024_version/plot07
add_co2 = 0

[treatment2]
name = T0.00
metdir = %(inputdata)s/SPRUCE_data/2024_version/plot06
add_co2 = 0

[treatment3]
name = T2.25
metdir = %(inputdata)s/SPRUCE_data/2024_version/plot20
add_co2 = 0
use_deep_soil_heating = .true.
soil_heating_target_offset = 2.25

[treatment4]
name = T4.50
metdir = %(inputdata)s/SPRUCE_data/2024_version/plot13
add_co2 = 0
use_deep_soil_heating = .true.
soil_heating_target_offset = 4.50

[treatment5]
name = T6.75
metdir = %(inputdata)s/SPRUCE_data/2024_version/plot08
add_co2 = 0
use_deep_soil_heating = .true.
soil_heating_target_offset = 6.75

[treatment6]
name = T9.00
metdir = %(inputdata)s/SPRUCE_data/2024_version/plot17
add_co2 = 0
use_deep_soil_heating = .true.
soil_heating_target_offset = 9.00
```

Columns 1 and 2 are the hollow and hummock in the standard three-topounit
SPRUCE surface data; the auxiliary fen/boardwalk column is excluded from the
reference average. If the surface-data ordering changes, update
`soil_heating_reference_columns` rather than assuming these indices are
universal.

The warmed treatments retain their treatment-specific atmospheric forcing.
Therefore near-surface warming comes from the warmed atmosphere and land
surface, while the controller supplies the experimental deep-heater energy.
After T0.00 has produced the reference file, the warmed treatments can run in
parallel.

### Direct ELM use outside OLMT

When bypassing the OLMT treatment workflow, the user must provide a NetCDF
reference stream with `T_SOIL_REFERENCE` in K and set the stream years and
alignment consistently. The reference file is a runtime-generated dependency,
so it is intentionally not treated as an ordinary CIME input-data file that
must exist during creation of the preceding T0.00 case.

### Recommended diagnostics

| History variable | Units | Interpretation |
| --- | --- | --- |
| `EFLX_SOIL_HEATING` | W m^-2 | Total controller power applied to the column |
| `EFLX_SOIL_HEATING_VR` | W m^-2 | Layer-resolved distribution of heater power |
| `TSOI_HEATING_CONTROL` | K | Modeled temperature at the control depth |
| `TSOI_HEATING_REFERENCE` | K | Untreated stream temperature |
| `TSOI_HEATING_TARGET` | K | Reference plus treatment offset |
| `TSOI` | K | Full prognostic soil-temperature profile |

For initial validation, plot seasonal profiles of `T9.00 - T0.00`, the three
controller temperatures, and heater power. Check the period after 2015-08-15
separately. A target offset at 2 m does not imply a uniform offset throughout
the soil column.

### Common failure modes

- **Warmed cases start before T0.00 finishes.** The reference file does not yet
  exist. Use the OLMT dependency workflow.
- **Wrong reference columns.** Verify the topounit ordering in the generated
  surface and history files.
- **Controller pinned at maximum power.** The target may be unreachable with
  the chosen forcing, time scale, or power cap. Inspect power and temperature
  error together.
- **No heat is applied.** Check the start date, bog flag, peat depth, reference
  file, and that `use_deep_soil_heating` is true for the treatment.
- **Expectation of exactly +9 K everywhere.** The controller targets one depth;
  the vertical response remains a prognostic energy-transport result.

### Scientific status

The implementation conserves and diagnoses the added energy, but the
controller geometry and tuning are a model representation of the experiment.
Validation should compare seasonal soil-temperature differences at multiple
depths, not only the 2 m controller target. Heater power, phase lag, winter
behavior, atmospheric warming, snow, and surface energy fluxes should all be
examined.

---

## Suggested verification sequence

Before using all four features in a calibration ensemble, run the following
small set of controlled tests:

1. **Non-peatland regression:** run US-MOz without HUMHOL and confirm that the
   new peatland options remain inactive.
2. **Burial-only attribution:** compare prescribed acrotelm, prognostic
   acrotelm, and legacy burial from the same mature state.
3. **Hydraulic attribution:** compare JULES on/off with compaction held on.
4. **Moss water attribution:** compare prognostic and diagnostic moss water,
   retaining daily water-store and productivity output.
5. **Heating smoke test:** run T0.00 through reference-file creation, then a
   short T9.00 case spanning 2015-08-15.
6. **Integrated spinup:** use accelerated spinup plus final spinup and inspect
   peat density, C-14, water table, Sphagnum productivity, DOM/acetate/CH4
   profiles, and treatment temperature profiles together.

The standard OLMT file
`config_files/SPRUCE_microbe_20260924.cfg` is the current complete example for
the SPRUCE treatment workflow. Its dated ELM parameter file is
`parameter_files/clm_params_Peatlands_Microbe_20260924.nc`.

## What colleagues should report with results

To make experiments reproducible, record:

- E3SM and elm-olmt commit hashes;
- OLMT configuration and parameter-file names;
- whether the acrotelm was prescribed or prognostic;
- the four feature-switch values;
- spinup stages and simulated years;
- whether the run began from cold start or a named restart;
- topounit ordering and weights; and
- the key diagnostic plots listed in each section.

This is especially important because the scientific parameters remain under
active evaluation and the commit that introduced a capability may not contain
the latest refinements described here.
