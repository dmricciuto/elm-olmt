#!/usr/bin/env python3
"""Download an AmeriFlux FLUXNET product and prepare OLMT site inputs.

The Shuttle snapshot, original ZIP, generated forcing, and observation table
are kept together so that the source and citation remain traceable.
"""

import argparse
import csv
import hashlib
import json
import re
import shlex
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from netCDF4 import Dataset


MET_FIELDS = {
    "TBOT": ("TA_F", "TA_F_MDS", "TA_ERA"),
    "VPD": ("VPD_F", "VPD_F_MDS", "VPD_ERA"),
    "WIND": ("WS_F", "WS_ERA"),
    "PSRF": ("PA_F", "PA_ERA"),
    "FSDS": ("SW_IN_F", "SW_IN_F_MDS", "SW_IN_ERA"),
    "FLDS": ("LW_IN_F", "LW_IN_F_MDS", "LW_IN_ERA"),
    "PRECTmms": ("P_F", "P_ERA"),
}
OBS_FIELDS = {
    "GPP": ("GPP_NT_CUT_REF", "GPP_NT_CUT_SE", "g C m-2 d-1"),
    "NEE": ("NEE_CUT_REF", "NEE_CUT_REF_JOINTUNC", "g C m-2 d-1"),
    "ER": ("RECO_NT_CUT_REF", "RECO_NT_CUT_SE", "g C m-2 d-1"),
    "EFLX_LH_TOT": ("LE_F_MDS", "LE_RANDUNC", "W m-2"),
    "FSH": ("H_F_MDS", "H_RANDUNC", "W m-2"),
}
UNITS = {
    "TBOT": "K", "QBOT": "kg/kg", "WIND": "m/s", "PSRF": "Pa",
    "FSDS": "W/m2", "FLDS": "W/m2", "PRECTmms": "kg/m2/s", "ZBOT": "m",
}
PLOT_STYLE_VERSION = 2
FORCING_TIME_VERSION = 2  # FLUXNET local standard time -> ELM UTC clock
PLOT_VARIABLES = ("TBOT", "QBOT", "WIND", "PSRF", "FSDS", "FLDS", "PRECTmms", "ZBOT")


class ShuttleUnavailableError(RuntimeError):
    """The live Shuttle catalogue could not be read."""


def _shuttle():
    try:
        from fluxnet_shuttle import download, listall
    except ModuleNotFoundError as exc:
        if exc.name != "fluxnet_shuttle":
            raise RuntimeError(
                f"FLUXNET Shuttle needs the Python dependency {exc.name}; "
                "install the Shuttle dependencies in the OLMT environment"
            ) from exc
        vendor = Path(__file__).resolve().parents[1] / "vendor/fluxnet_shuttle/src"
        if not vendor.is_dir():
            raise RuntimeError(
                "FLUXNET Shuttle is unavailable: install the official library or "
                "include vendor/fluxnet_shuttle with the OLMT checkout"
            ) from exc
        sys.path.insert(0, str(vendor))
        try:
            from fluxnet_shuttle import download, listall
        except ModuleNotFoundError as dependency_error:
            raise RuntimeError(
                f"FLUXNET Shuttle needs the Python dependency "
                f"{dependency_error.name}; install the Shuttle dependencies "
                "in the OLMT environment"
            ) from dependency_error
    return download, listall


def _snapshot(site, root, snapshot_file=None):
    live_snapshot = snapshot_file is None
    if snapshot_file is None:
        import asyncio
        # The Shuttle's sync wrapper uses get_event_loop on Python 3.11+.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        _, listall = _shuttle()
        snapshot_file = listall(data_hubs=["ameriflux"], output_dir=str(root))
    snapshot_file = Path(snapshot_file)
    with snapshot_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if live_snapshot and not rows:
        raise ShuttleUnavailableError(
            "FLUXNET Shuttle returned an empty AmeriFlux catalogue; check network/TLS access")
    matches = [r for r in rows
               if r.get("site_id") == site and r.get("data_hub", "").lower() == "ameriflux"]
    if len(matches) != 1:
        raise ValueError(f"Expected one AmeriFlux Shuttle product for {site}; found {len(matches)}")
    return snapshot_file, matches[0]


def _member(archive, frequency):
    pattern = re.compile(rf"(?:FLUXMET|FULLSET)_{frequency}_.*\.csv$", re.IGNORECASE)
    names = [n for n in archive.namelist() if pattern.search(Path(n).name)]
    if len(names) != 1:
        raise ValueError(f"Expected one FLUXMET/FULLSET_{frequency} CSV in ZIP; found {names}")
    return names[0]


def _read_table(archive, frequency):
    needed = {"TIMESTAMP_START"} if frequency in ("HH", "HR") else {"TIMESTAMP"}
    if frequency in ("HH", "HR"):
        needed.update(name for names in MET_FIELDS.values() for name in names)
    else:
        needed.update(name for names in OBS_FIELDS.values() for name in names[:2])
    with archive.open(_member(archive, frequency)) as handle:
        return pd.read_csv(handle, low_memory=False, usecols=lambda col: col in needed)


def _numeric(frame, names):
    for name in names:
        if name in frame:
            values = pd.to_numeric(frame[name], errors="coerce")
            return values.mask(values <= -9990), name
    raise ValueError("Missing required FLUXNET field: " + " or ".join(names))


def _temperature(frame, names):
    """Use the MDS temperature where the preferred product is impossible."""
    values, source = _numeric(frame, names)
    valid = values.between(180 - 273.15, 350 - 273.15)
    invalid = ~valid
    if invalid.any() and source != "TA_F_MDS" and "TA_F_MDS" in frame:
        mds, _ = _numeric(frame, ("TA_F_MDS",))
        replace = invalid & mds.between(180 - 273.15, 350 - 273.15)
        values = values.where(~replace, mds)
        if replace.any():
            source += f"; TA_F_MDS for {int(replace.sum())} invalid records"
    invalid = ~values.between(180 - 273.15, 350 - 273.15)
    if invalid.any():
        index = int(np.flatnonzero(invalid.to_numpy())[0])
        stamp = frame["TIMESTAMP_START"].iloc[index]
        raise ValueError(f"Invalid air temperature at {stamp}: {values.iloc[index]} C "
                         f"(source {source}; no valid TA_F_MDS replacement)")
    return values, source


def _utc_offset_hours(archive):
    """Read the site's standard-time UTC offset from the product BIF."""
    import io

    names = [name for name in archive.namelist() if re.search(r"_BIF_.*\.csv$", Path(name).name, re.I)]
    if len(names) != 1:
        raise ValueError("FLUXNET ZIP must contain one site BIF CSV with UTC_OFFSET")
    with archive.open(names[0]) as raw:
        rows = csv.DictReader(io.TextIOWrapper(raw, encoding="latin1"))
        offsets = {row["DATAVALUE"].strip() for row in rows if row.get("VARIABLE") == "UTC_OFFSET"}
    if len(offsets) != 1:
        raise ValueError(f"Expected one UTC_OFFSET in FLUXNET BIF; found {sorted(offsets)}")
    try:
        offset = float(offsets.pop())
    except ValueError as exc:
        raise ValueError("FLUXNET BIF UTC_OFFSET must be numeric hours") from exc
    if not np.isfinite(offset) or abs(offset) > 14:
        raise ValueError(f"Implausible FLUXNET BIF UTC_OFFSET: {offset}")
    if not offset.is_integer():
        raise ValueError(f"UTC_OFFSET={offset} hours needs subhourly resampling, which is not yet supported")
    return int(offset)


def _hourly_forcing(frame, utc_offset_hours):
    if "TIMESTAMP_START" not in frame:
        raise ValueError("Hourly product has no TIMESTAMP_START")
    stamps = pd.to_datetime(frame["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M", errors="coerce")
    if stamps.isna().any() or stamps.duplicated().any():
        raise ValueError("Invalid or duplicate FLUXNET timestamps")
    data = pd.DataFrame(index=pd.DatetimeIndex(stamps))
    source_fields = {}
    for outvar, names in MET_FIELDS.items():
        if outvar == "TBOT":
            values, source_fields[outvar] = _temperature(frame, names)
        else:
            values, source_fields[outvar] = _numeric(frame, names)
        data[outvar] = values.to_numpy(dtype=float)
    minutes = set(data.index.minute)
    if minutes == {0, 30}:
        counts = data["TBOT"].resample("h").size()
        if (counts != 2).any():
            raise ValueError("Half-hourly product has missing timesteps")
        precip = data["PRECTmms"].resample("h").sum(min_count=2)
        data = data.resample("h").mean()
        data["PRECTmms"] = precip
    elif minutes != {0}:
        raise ValueError(f"Expected hourly or half-hourly timestamps; found minutes {minutes}")
    years = list(range(int(data.index.year.min()), int(data.index.year.max()) + 1))
    expected_local = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31 23:00", freq="h")
    if not data.index.equals(expected_local):
        data = data.reindex(expected_local)
    missing_counts = {}
    for var in MET_FIELDS:
        missing = data[var].isna()
        missing_counts[var] = int(missing.sum())
        if missing.any():
            raise ValueError(f"{var} has {missing.sum()} missing hourly values in the selected product; "
                             "choose a complete FLUXNET product before running ELM")
    data["TBOT"] += 273.15
    data["PSRF"] *= 1000.0  # kPa -> Pa
    vpd_pa = data.pop("VPD") * 100.0  # hPa -> Pa
    data["PRECTmms"] /= 3600.0  # mm per input interval -> kg m-2 s-1
    for name in ("WIND", "FSDS", "FLDS", "PRECTmms"):
        data[name] = data[name].clip(lower=0)
    tc = data["TBOT"] - 273.15
    esat = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    vapor = (esat - vpd_pa).clip(lower=0)
    vapor = np.minimum(vapor, 0.99 * data["PSRF"])
    data["QBOT"] = 0.622 * vapor / (data["PSRF"] - 0.378 * vapor)
    for name, lower, upper in (("TBOT", 180, 350), ("PSRF", 50000, 150000),
                               ("QBOT", 0, 0.1)):
        invalid = ~data[name].between(lower, upper)
        if invalid.any():
            timestamp = data.index[invalid][0]
            value = data.loc[timestamp, name]
            raise ValueError(f"Converted {name}={value:g} at local time {timestamp} "
                             f"is outside [{lower}, {upper}]")
    # FLUXNET hourly TIMESTAMP_START is local standard time (no DST). ELM's
    # CPL_BYPASS solar geometry uses longitude with a UTC model clock.
    data.index = data.index - pd.Timedelta(hours=utc_offset_hours)
    expected_utc = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31 23:00", freq="h")
    expected_utc = expected_utc[~((expected_utc.month == 2) & (expected_utc.day == 29))]
    data = data.reindex(expected_utc)
    edge_hours = abs(utc_offset_hours)
    if utc_offset_hours < 0:
        boundary = data.iloc[:edge_hours]
        interior = data.iloc[edge_hours:]
    elif utc_offset_hours > 0:
        boundary = data.iloc[-edge_hours:]
        interior = data.iloc[:-edge_hours]
    else:
        boundary = data.iloc[:0]
        interior = data
    if interior.isna().any().any() or (edge_hours and not boundary.isna().all().all()):
        raise ValueError("UTC conversion produced missing interior hours; inspect FLUXNET timestamps")
    # The product omits the local hours just outside its first/last year.
    # Fill only those boundary hours with the nearest available meteorology.
    if edge_hours:
        data = data.bfill() if utc_offset_hours < 0 else data.ffill()
    return data, years[0], years[-1], source_fields, missing_counts, edge_hours


def _write_forcing(path, data, first, last, lat, lon, height, utc_offset_hours):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("gridcell", 1)
        ds.createDimension("DTIME", len(data))
        ds.createDimension("scalar", 1)
        for name in ("TBOT", "QBOT", "WIND", "PSRF", "FSDS", "FLDS", "PRECTmms"):
            values = data[name].to_numpy(dtype="f8")
            low, high = float(values.min()), float(values.max())
            offset = (low + high) / 2.0
            scale = (high - low) / 65000.0 if high > low else 1.0
            packed = np.rint((values - offset) / scale).astype("i2")
            var = ds.createVariable(name, "i2", ("gridcell", "DTIME"),
                                    zlib=True, complevel=2, fill_value=np.int16(-32768))
            var.scale_factor = scale
            var.add_offset = offset
            var.set_auto_maskandscale(False)
            var[0, :] = packed
            var.units = UNITS[name]
            var.mode = "time-dependent"
        zbot = ds.createVariable("ZBOT", "f4", ("gridcell", "DTIME"), zlib=True, complevel=2)
        zbot[0, :] = height
        zbot.units = "m"
        zbot.mode = "time-dependent"
        time = ds.createVariable("DTIME", "f8", ("DTIME",))
        time[:] = (np.arange(len(data)) + 0.5) / 24.0
        time.units = f"days since {first}-01-01 00:00:00"
        time.calendar = "noleap"
        time.time_basis = "UTC"
        for name, value in (("LONGXY", lon), ("LATIXY", lat)):
            ds.createVariable(name, "f8", ("gridcell",))[:] = value
        for name, value in (("EDGEE", lon + .05), ("EDGEW", lon - .05),
                            ("EDGEN", lat + .05), ("EDGES", lat - .05),
                            ("start_year", first), ("end_year", last)):
            ds.createVariable(name, "i4" if "year" in name else "f4", ("scalar",))[:] = value
        ds.source = "AmeriFlux FLUXNET product via FLUXNET Shuttle; see provenance.json"
        ds.source_utc_offset_hours = utc_offset_hours
        ds.forcing_time_version = FORCING_TIME_VERSION


def _plot_forcing(site_root, site, data, height):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = site_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for variable in PLOT_VARIABLES:
        values = (pd.Series(height, index=data.index) if variable == "ZBOT"
                  else data[variable])
        if variable == "PRECTmms":
            values = values * 3600.0  # kg m-2 s-1 -> mm h-1
            unit = "mm h$^{-1}$"
        else:
            unit = UNITS[variable]
        monthly = values.resample("MS").agg(["mean", "std", "min", "max"])
        seasonal = values.groupby(values.index.month).agg(["mean", "std", "min", "max"])
        figure, axes = plt.subplots(2, 1, figsize=(10, 6.5), constrained_layout=True)
        for axis, statistics, xvalues in (
            (axes[0], monthly, monthly.index),
            (axes[1], seasonal, seasonal.index),
        ):
            mean = statistics["mean"].to_numpy()
            minimum = statistics["min"].to_numpy()
            maximum = statistics["max"].to_numpy()
            deviation = statistics["std"].fillna(0).to_numpy()
            sd_lower = np.maximum(mean - deviation, minimum)
            sd_upper = np.minimum(mean + deviation, maximum)
            axis.fill_between(xvalues, minimum, maximum, color="#e69f00", alpha=0.26,
                              label="Hourly minimum–maximum", linewidth=0)
            axis.fill_between(xvalues, sd_lower, sd_upper, color="#0072b2", alpha=0.40,
                              label="Hourly mean ±1 SD", linewidth=0)
            axis.plot(xvalues, mean, color="#163d5c", linewidth=1.1,
                      label="Hourly mean")
            axis.set_ylabel(unit)
            axis.grid(alpha=0.2)
            if variable == "PRECTmms":
                axis.set_yscale("symlog", linthresh=0.1)
        axes[0].set_title(f"{site} {variable}: monthly mean and hourly spread")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[1].set_title("Mean seasonal cycle and hourly spread across years")
        axes[1].set_xticks(range(1, 13))
        axes[1].set_xlabel("Month")
        figure.savefig(plot_dir / f"{variable}.png", dpi=150)
        plt.close(figure)


def replot_site(site_root, site):
    """Regenerate diagnostics from existing NetCDF without downloading data."""
    site_root = Path(site_root)
    with Dataset(site_root / "all_hourly.nc") as ds:
        first = int(ds["start_year"][0])
        last = int(ds["end_year"][0])
        dates = pd.date_range(f"{first}-01-01", f"{last}-12-31 23:00", freq="h")
        dates = dates[~((dates.month == 2) & (dates.day == 29))]
        if len(dates) != len(ds.dimensions["DTIME"]):
            raise ValueError("Saved forcing has an unexpected hourly time dimension")
        data = pd.DataFrame({name: np.asarray(ds[name][0, :], dtype=float)
                             for name in PLOT_VARIABLES if name != "ZBOT"}, index=dates)
        height = float(ds["ZBOT"][0, 0])
    _plot_forcing(site_root, site, data, height)
    provenance_file = site_root / "provenance.json"
    provenance = json.loads(provenance_file.read_text())
    provenance["plot_style_version"] = PLOT_STYLE_VERSION
    provenance_file.write_text(json.dumps(provenance, indent=2) + "\n")


def _write_observations(path, frame, site):
    if "TIMESTAMP" not in frame:
        raise ValueError("Monthly product has no TIMESTAMP")
    path.parent.mkdir(parents=True, exist_ok=True)
    stamps = pd.to_datetime(frame["TIMESTAMP"].astype(str), format="%Y%m", errors="coerce")
    if stamps.isna().any():
        raise ValueError("Invalid monthly timestamps")
    rows = []
    for model_var, (source, uncertainty, units) in OBS_FIELDS.items():
        if source not in frame:
            continue
        vals, _ = _numeric(frame, (source,))
        errors = _numeric(frame, (uncertainty,))[0] if uncertainty in frame else pd.Series(np.nan, index=frame.index)
        for stamp, val, err in zip(stamps, vals, errors):
            if not np.isfinite(val):
                continue
            if not np.isfinite(err) or err <= 0:
                err = max(abs(val) * .15, 1.0 if model_var not in ("EFLX_LH_TOT", "FSH") else 10.0)
            rows.append((site, model_var, stamp.year, stamp.month, val, err, source, units))
    if not rows:
        raise ValueError("No supported monthly flux observations in product")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("site", "model_var", "year", "month", "obs", "obs_err", "source_variable", "units"))
        writer.writerows(rows)
    return len(rows)


def _measurement_height(archive):
    names = [name for name in archive.namelist()
             if re.search(r"BIFVARINFO_(?:HH|HR)_", Path(name).name, re.I)]
    if not names:
        return None
    import io
    with archive.open(names[0]) as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="latin1"))
        groups = {}
        for row in reader:
            groups.setdefault(row["GROUP_ID"], {})[row["VARIABLE"]] = row["DATAVALUE"]
    for group in groups.values():
        if group.get("VAR_INFO_VARNAME") in ("TA_F", "TA_F_MDS"):
            try:
                return float(group["VAR_INFO_HEIGHT"])
            except (KeyError, ValueError):
                pass
    return None


def prepare(site, site_root, archive_path, metadata, height):
    with ZipFile(archive_path) as archive:
        utc_offset = _utc_offset_hours(archive)
        met_member = next((f for f in ("HH", "HR") if any(
            re.search(rf"(?:FLUXMET|FULLSET)_{f}_.*\.csv$", Path(n).name, re.I)
            for n in archive.namelist())), None)
        if met_member is None:
            raise ValueError("ZIP has no FLUXMET/FULLSET hourly forcing CSV")
        forcing, first, last, fields, fills, edge_hours = _hourly_forcing(
            _read_table(archive, met_member), utc_offset)
        observations = _read_table(archive, "MM")
        height = height if height is not None else _measurement_height(archive)
    if height is None:
        raise ValueError("Measurement height missing from BIFVARINFO; pass --height")
    lat = float(metadata["location_lat"])
    lon = float(metadata["location_long"])
    met_path = site_root / "all_hourly.nc"
    obs_path = site_root / "observations.csv"
    _write_forcing(met_path, forcing, first, last, lat, lon, height, utc_offset)
    _plot_forcing(site_root, site, forcing, height)
    obs_count = _write_observations(obs_path, observations, site)
    with archive_path.open("rb") as archive_input:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: archive_input.read(1024 * 1024), b""):
            checksum.update(chunk)
    provenance = {
        "site": site, "data_hub": "AmeriFlux", "product": metadata.get("fluxnet_product_name"),
        "citation": metadata.get("product_citation"), "product_id": metadata.get("product_id"),
        "archive_sha256": checksum.hexdigest(),
        "measurement_height_m": height,
        "source_time_basis": "local standard time (no DST)",
        "source_utc_offset_hours": utc_offset,
        "forcing_time_basis": "UTC",
        "forcing_time_version": FORCING_TIME_VERSION,
        "boundary_hours_filled": edge_hours,
        "forcing_source_fields": fields, "missing_hourly_values": fills,
        "forcing_years": [first, last], "observation_rows": obs_count,
        "plot_style_version": PLOT_STYLE_VERSION,
        "license": "CC BY 4.0; cite the product and follow the ZIP data use document",
    }
    (site_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_current(root, site, version):
    pointer = Path(root) / f"current_{site}.txt"
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(version.name + "\n")
    temporary.replace(pointer)


def cached_site(site, output_root):
    """Return the selected complete site version without network access."""
    root = Path(output_root)
    pointer = root / f"current_{site}.txt"
    versions = sorted(root.glob(f"1x1pt_{site}_*"), reverse=True)
    if pointer.is_file():
        selected = root / pointer.read_text().strip()
        if selected.parent == root and selected.name.startswith(f"1x1pt_{site}_"):
            versions = [selected] + [v for v in versions if v != selected]
    for version in versions:
        provenance_file = version / "provenance.json"
        if not (version / "all_hourly.nc").is_file() or not (version / "observations.csv").is_file() or not provenance_file.is_file():
            continue
        try:
            provenance = json.loads(provenance_file.read_text())
        except (OSError, ValueError):
            continue
        if (provenance.get("site") == site and provenance.get("archive_sha256")
                and provenance.get("forcing_time_version", 0) >= FORCING_TIME_VERSION):
            return {"site": site, "directory": str(version.resolve()), "reused": True,
                    "archive_sha256": provenance["archive_sha256"],
                    "forcing_years": provenance.get("forcing_years"),
                    "citation": provenance.get("citation")}
    raise FileNotFoundError(f"No complete UTC-aligned Shuttle version for {site} in {root}")


def _docker_host_refresh_hint(site, output_root):
    """Give a copyable data-only refresh command for the Docker mount layout."""
    if not (Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()):
        return ""
    container_inputdata = Path("/inputdata")
    root = Path(output_root)
    if root.is_relative_to(container_inputdata):
        host_root = Path("../inputdata") / root.relative_to(container_inputdata)
        output_argument = shlex.quote(str(host_root))
    else:
        output_argument = "'<matching path on the host for " + str(root) + ">'"
    command = ("../.venv-fluxnet-shuttle/bin/python "
               "tools/fluxnet_shuttle_to_olmt.py " + shlex.quote(site) +
               " --output-root " + output_argument)
    return ("\nTo refresh only Shuttle data, exit Docker and run from the "
            "elm-olmt directory on the host:\n  " + command +
            "\nThen retry the OLMT command in Docker.")


def sync_site(site, output_root, snapshot_file=None, archive_path=None, height=None):
    """Fetch a fresh product, reusing a complete local version if unchanged.

    A temporary download is always compared with saved archive hashes. Nothing
    from an unchanged fetch is retained; a changed product gets a new dated
    directory while prior versions remain available.
    """
    if not re.fullmatch(r"[A-Za-z]{2}-[A-Za-z0-9]{3}", site):
        raise ValueError("site must look like US-MMS")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"shuttle-{site}-") as temporary:
        temporary = Path(temporary)
        try:
            snapshot, metadata = _snapshot(site, temporary, snapshot_file)
        except ShuttleUnavailableError as exc:
            host_hint = _docker_host_refresh_hint(site, output_root)
            try:
                result = cached_site(site, output_root)
            except FileNotFoundError:
                raise ShuttleUnavailableError(
                    f"{exc}; no UTC-aligned cached {site} product is available.{host_hint}"
                ) from exc
            print(f"Warning: {exc}; using cached {site} data without checking for updates: "
                  f"{result['directory']}{host_hint}", file=sys.stderr)
            return result
        if archive_path is None:
            import asyncio
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
            download, _ = _shuttle()
            files = download(site_ids=[site], snapshot_file=str(snapshot),
                             output_dir=str(temporary))
            if len(files) != 1:
                raise RuntimeError(f"Shuttle downloaded {len(files)} files for {site}")
            archive = Path(files[0])
        else:
            archive = Path(archive_path)
        incoming_hash = _sha256(archive)
        versions = sorted(output_root.glob(f"1x1pt_{site}_*"), reverse=True)
        for version in versions:
            provenance_file = version / "provenance.json"
            if not (version / "all_hourly.nc").is_file() or not (version / "observations.csv").is_file() or not provenance_file.is_file():
                continue
            try:
                provenance = json.loads(provenance_file.read_text())
            except (OSError, ValueError):
                continue
            if (provenance.get("archive_sha256") == incoming_hash
                    and provenance.get("forcing_time_version", 0) >= FORCING_TIME_VERSION):
                if provenance.get("plot_style_version", 0) < PLOT_STYLE_VERSION or any(
                    not (version / "plots" / f"{name}.png").is_file() for name in PLOT_VARIABLES
                ):
                    replot_site(version, site)
                _record_current(output_root, site, version)
                return {"site": site, "directory": str(version.resolve()), "reused": True,
                        "archive_sha256": incoming_hash, "forcing_years": provenance.get("forcing_years"),
                        "citation": provenance.get("citation")}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        destination = output_root / f"1x1pt_{site}_{stamp}"
        if destination.exists():
            destination = output_root / f"1x1pt_{site}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}"
        with tempfile.TemporaryDirectory(prefix=f".1x1pt_{site}_", dir=output_root) as staging_name:
            staging = Path(staging_name)
            provenance = prepare(site, staging, archive, metadata, height)
            raw = staging / "raw"
            raw.mkdir()
            shutil.copy2(archive, raw / archive.name)
            shutil.copy2(snapshot, staging / "shuttle_snapshot.csv")
            staging.rename(destination)
        _record_current(output_root, site, destination)
        return {"site": site, "directory": str(destination.resolve()), "reused": False,
                "archive_sha256": incoming_hash, "forcing_years": provenance["forcing_years"],
                "citation": provenance.get("citation")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", help="AmeriFlux site ID, e.g. US-MMS")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--snapshot-file", type=Path, help="Reuse an existing Shuttle snapshot")
    parser.add_argument("--archive", type=Path, help="Prepare an already downloaded product ZIP")
    parser.add_argument("--height", type=float, help="Override meteorology height from BIF metadata, in meters")
    parser.add_argument("--replot", action="store_true", help="Regenerate plots from the selected saved NetCDF without downloading")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z]{2}-[A-Za-z0-9]{3}", args.site):
        parser.error("site must look like US-MMS")
    if args.replot:
        result = cached_site(args.site, args.output_root)
        replot_site(result["directory"], args.site)
    else:
        result = sync_site(args.site, args.output_root, args.snapshot_file, args.archive, args.height)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
