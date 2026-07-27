"""Run the ELM emulator from an OLMT configuration."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


_TESSFA_MET_TYPES = {"tessfa", "era5-daymet", "era5-daymet4", "gswp3-daymet4"}
_TRENDY_MET_TYPES = {"crujra", "trendy"}
_DEFAULT_CO2_FILE_RELATIVE = Path("atm/datm7/CO2/fco2_datm_rcp4.5_1765-2500_c130312.nc")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _strip_quotes(value: Any) -> str:
    return str(value).strip().strip("'\"")


def _case_value(options: dict[str, Any], key: str, case_index: int, default: Any = "") -> Any:
    value = options.get(key, default)
    if isinstance(value, list):
        if case_index < len(value):
            return value[case_index]
        if value:
            return value[-1]
        return default
    return value


def _first_int_option(options: dict[str, Any], keys: tuple[str, ...], case_index: int) -> int | None:
    for key in keys:
        if key not in options:
            continue
        value = _case_value(options, key, case_index, None)
        if value is None or value == "":
            continue
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is None:
            continue
        token = _strip_quotes(value).split(",", 1)[0].strip()
        if token:
            return int(float(token))
    return None


def _path_from_config(value: Any, *, base: Path | None = None, inputdata: str = "") -> Path | None:
    if value is None or value == "":
        return None
    path_text = os.path.expandvars(os.path.expanduser(_strip_quotes(value)))
    if path_text.startswith("/inputdata") and inputdata:
        path_text = str(Path(inputdata) / path_text.removeprefix("/inputdata").lstrip("/"))
    path = Path(path_text)
    if not path.is_absolute() and base is not None:
        path = base / path
    return path


def _read_metinfo(olmtdir: Path, mettype: str, inputdata: str) -> Path | None:
    metinfo = olmtdir / "metinfo.txt"
    if not mettype or not metinfo.exists():
        return None
    with metinfo.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(":", 1)
            if len(parts) == 2 and parts[0] == mettype:
                return Path(inputdata) / parts[1].strip()
    return None


def _resolve_metdir(
    *,
    mettype: str,
    metdir: Any,
    site: str,
    inputdata: str,
    use_cpl_bypass: bool,
    olmtdir: Path,
) -> Path:
    if metdir:
        resolved = _path_from_config(metdir, inputdata=inputdata)
        if resolved is None:
            raise ValueError("metdir resolved to an empty path")
        if site and (mettype in {"", "site"}):
            site_leaf = "1x1pt_" + site
            if resolved.name != site_leaf:
                resolved = resolved / site_leaf
        return resolved

    if site and (mettype in {"", "site"}):
        return Path(inputdata) / "atm" / "datm7" / "CLM1PT_data" / ("1x1pt_" + site)

    resolved = _read_metinfo(olmtdir, mettype, inputdata)
    if resolved is None:
        if mettype:
            raise ValueError(
                f"No metdir was provided and met type '{mettype}' was not found in {olmtdir / 'metinfo.txt'}"
            )
        resolved = Path(inputdata) / "atm" / "datm7" / "atm_forcing.datm7.GSWP3.0.5d.v2.c180716"

    if use_cpl_bypass and "site" not in mettype and resolved.name != "cpl_bypass_full":
        resolved = resolved / "cpl_bypass_full"
    return resolved


def _surface_data_path(
    *,
    simulation: dict[str, Any],
    case_options: dict[str, Any],
    case_index: int,
    inputdata: str,
    olmtdir: Path,
) -> Path | None:
    explicit = simulation.get("emulator_surfdata", "")
    if explicit:
        return _path_from_config(explicit, base=olmtdir, inputdata=inputdata)

    for key in ("surfdata", "surffile", "fsurdat", "surffile_global", "surfdata_global"):
        value = _case_value(case_options, key, case_index, "")
        if value:
            return _path_from_config(value, base=olmtdir, inputdata=inputdata)
    return None


def _co2_file_path(
    *,
    simulation: dict[str, Any],
    case_options: dict[str, Any],
    case_index: int,
    inputdata: str,
    olmtdir: Path,
) -> Path:
    value = _case_value(case_options, "co2_file", case_index, "")
    if not value:
        value = simulation.get("emulator_co2_file", "")
    if value:
        resolved = _path_from_config(value, base=olmtdir, inputdata=inputdata)
        if resolved is None:
            raise ValueError("co2_file resolved to an empty path")
        return resolved
    return Path(inputdata) / _DEFAULT_CO2_FILE_RELATIVE


def _metdata_type_for_case(case_options: dict[str, Any], case_index: int, mettype: str) -> str:
    for key in ("metdata_type", "datm_metdata_type"):
        value = _case_value(case_options, key, case_index, "")
        if value:
            return _strip_quotes(value).lower()
    return _strip_quotes(mettype).lower()


def _infer_met_format(metdata_type: str, site: str, requested: str) -> str:
    requested = _strip_quotes(requested).lower()
    if requested and requested != "auto":
        return requested
    if site or metdata_type == "site":
        return "site"
    if metdata_type in _TRENDY_MET_TYPES or "trendy" in metdata_type:
        return "trendy"
    if metdata_type in _TESSFA_MET_TYPES or "daymet" in metdata_type:
        return "tessfa"
    return "auto"


def _infer_history_frequency(
    *,
    simulation: dict[str, Any],
    case_options: dict[str, Any],
    case_index: int,
    casename: str,
) -> str:
    explicit = simulation.get("emulator_h0_frequency", "")
    if explicit:
        return _strip_quotes(explicit).lower()

    nhtfrq = _first_int_option(case_options, ("hist_nhtfrq", "nhtfrq"), case_index)
    mfilt = _first_int_option(case_options, ("hist_mfilt", "mfilt"), case_index)
    if nhtfrq is not None and mfilt is not None and nhtfrq > 0 and mfilt < 0:
        print(
            f"Warning: {casename} has hist_nhtfrq={nhtfrq} and hist_mfilt={mfilt}. "
            "This looks swapped; interpreting the negative value as hist_nhtfrq."
        )
        nhtfrq = mfilt
    if nhtfrq is None:
        return "monthly"
    if nhtfrq == 0:
        return "monthly"

    hours = abs(nhtfrq)
    if hours == 1:
        return "hourly"
    if hours == 24:
        return "daily"
    if hours == 120:
        return "5day"
    if hours < 24:
        print(
            f"Warning: {casename} requests subdaily h0 output "
            f"(nhtfrq={nhtfrq}, mfilt={mfilt}); emulator h0 supports hourly, daily, 5day, monthly. "
            "Using daily."
        )
        return "daily"

    print(
        f"Warning: {casename} requests unsupported h0 frequency "
        f"(nhtfrq={nhtfrq}, mfilt={mfilt}); emulator h0 supports hourly, daily, 5day, monthly. "
        "Using monthly."
    )
    return "monthly"


def _olmt_case_name(
    *,
    cfg: dict[str, dict[str, Any]],
    site: str,
    region_name: str,
    compset: str,
    suffix: str,
) -> str:
    simulation = cfg.get("simulation", {})
    caseid = simulation.get("case_prefix", "")
    if caseid == "":
        caseid = datetime.now().strftime("%Y%m%d")
    place = site if site else (region_name if region_name else "region")
    return "_".join(part for part in [f"{caseid}_{place}_{compset}", suffix] if part)


def _add_optional_value(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None and value != "":
        cmd.extend([flag, str(value)])


def _add_optional_bool(cmd: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    cmd.append(flag if _as_bool(value) else "--no-" + flag.removeprefix("--"))


def _build_command(
    *,
    cfg: dict[str, dict[str, Any]],
    case_options: dict[str, Any],
    case_index: int,
    site: str,
    siteinfo: dict[str, Any],
    point_list: list[tuple[float, float]],
    runtype: str,
    metdir: Path,
    met_format: str,
    inputdata: str,
    emulator_root: Path,
    olmtdir: Path,
    rundir: Path,
    casename: str,
    start_year: int,
    nmonths: int,
    lat_bounds: Any,
    lon_bounds: Any,
    restart_in: Path | None,
    restart_out: Path,
    history_frequency: str,
) -> list[str]:
    simulation = cfg.get("simulation", {})
    python_exe = str(simulation.get("emulator_python", sys.executable))
    model_dir = _path_from_config(
        simulation.get("emulator_model_dir", emulator_root / "artifacts" / "canopyflux_models"),
        base=emulator_root,
        inputdata=inputdata,
    )
    surfdata = _surface_data_path(
        simulation=simulation,
        case_options=case_options,
        case_index=case_index,
        inputdata=inputdata,
        olmtdir=olmtdir,
    )
    clm_params = _path_from_config(
        simulation.get("emulator_clm_params", _case_value(case_options, "paramfile", case_index, "")),
        base=emulator_root,
        inputdata=inputdata,
    )
    co2_file = _co2_file_path(
        simulation=simulation,
        case_options=case_options,
        case_index=case_index,
        inputdata=inputdata,
        olmtdir=olmtdir,
    )

    cmd = [
        python_exe,
        "-m",
        "elm_emulator.benchmark_replay",
        "--regional",
        "--device",
        str(simulation.get("emulator_device", "auto")),
        "--dtype",
        str(simulation.get("emulator_dtype", "float32")),
        "--mlp-dtype",
        str(simulation.get("emulator_mlp_dtype", "same")),
        "--model-dir",
        str(model_dir),
        "--met-dir",
        str(metdir),
        "--regional-met-format",
        met_format,
        "--year",
        str(start_year),
        "--month",
        str(simulation.get("emulator_month", 1)),
        "--nmonths",
        str(nmonths),
        "--elm-history-run-dir",
        str(rundir),
        "--elm-history-case",
        casename,
        "--elm-history-frequency",
        history_frequency,
        "--regional-restart-out",
        str(restart_out),
        "--co2-file",
        str(co2_file),
    ]

    if _as_bool(simulation.get("emulator_universal", True), True):
        cmd.append("--universal")
    if not _as_bool(simulation.get("emulator_plot", False), False):
        cmd.append("--no-plot")
    if not _as_bool(simulation.get("emulator_cell_timeseries", False), False):
        cmd.append("--no-regional-cell-timeseries")
    if surfdata is not None:
        cmd.extend(["--regional-surfdata", str(surfdata)])
    if clm_params is not None:
        cmd.extend(["--clm-params", str(clm_params)])
    if restart_in is not None:
        cmd.extend(["--regional-restart-in", str(restart_in)])

    if site:
        info = siteinfo[site]
        cmd.extend(["--lat", str(info["lat"]), "--lon", str(info["lon"])])
    elif runtype == "latlon_list":
        if len(point_list) != 1:
            raise ValueError(
                "emulator=True currently supports site, latlon_bbox, or a single-point latlon_list"
            )
        lat, lon = point_list[0]
        cmd.extend(["--lat", str(lat), "--lon", str(lon)])
    else:
        lat_values = _as_list(lat_bounds)
        lon_values = _as_list(lon_bounds)
        if len(lat_values) >= 2:
            cmd.extend(["--regional-min-lat", str(lat_values[0]), "--regional-max-lat", str(lat_values[1])])
        if len(lon_values) >= 2:
            cmd.extend(["--regional-min-lon", str(lon_values[0]), "--regional-max-lon", str(lon_values[1])])

    scalar_options = {
        "emulator_torch_num_threads": "--torch-num-threads",
        "emulator_hydrology_backend": "--hydrology-backend",
        "emulator_soil_temperature_backend": "--soil-temperature-backend",
        "emulator_physics_step_hours": "--regional-physics-step-hours",
        "emulator_calendar": "--regional-calendar",
        "emulator_met_cycle_years": "--regional-met-cycle-years",
        "emulator_met_cycle_start_year": "--regional-met-cycle-start-year",
        "emulator_ncell": "--regional-ncell",
        "emulator_max_hours": "--regional-max-hours",
        "emulator_met_cache_dir": "--regional-met-cache-dir",
        "emulator_precompute_cache_dir": "--regional-precompute-cache-dir",
        "emulator_progress_stride": "--regional-progress-stride",
        "emulator_regional_ensemble_size": "--regional-ensemble-size",
        "emulator_trait_ensemble_size": "--regional-trait-ensemble-size",
        "emulator_co2_ppm": "--co2-ppm",
        "emulator_output_frequency": "--regional-output-frequency",
        "emulator_daily_output": "--regional-daily-output",
    }
    for key, flag in scalar_options.items():
        _add_optional_value(cmd, flag, simulation.get(key, ""))

    bool_options = {
        "emulator_prefetch": "--regional-prefetch",
        "emulator_met_cache": "--regional-met-cache",
        "emulator_precompute_cache": "--regional-precompute-cache",
        "emulator_safety_clamps": "--regional-safety-clamps",
    }
    for key, flag in bool_options.items():
        _add_optional_bool(cmd, flag, simulation.get(key, ""))

    extra_args = simulation.get("emulator_extra_args", "")
    if extra_args:
        cmd.extend(shlex.split(str(extra_args)))
    return cmd


def _write_case_metadata(
    *,
    casedir: Path,
    rundir: Path,
    casename: str,
    command: list[str],
    metadata: dict[str, Any],
) -> None:
    casedir.mkdir(parents=True, exist_ok=True)
    rundir.mkdir(parents=True, exist_ok=True)
    command_text = " ".join(shlex.quote(part) for part in command)
    (casedir / "emulator_command.sh").write_text(command_text + "\n", encoding="utf-8")
    (rundir / "emulator_command.sh").write_text(command_text + "\n", encoding="utf-8")
    with (casedir / "emulator_case.json").open("w", encoding="utf-8") as handle:
        payload = dict(metadata)
        payload["casename"] = casename
        payload["command"] = command
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def run_emulator_cases(
    *,
    cfg: dict[str, dict[str, Any]],
    sites: list[str],
    siteinfo: dict[str, Any],
    point_list: list[tuple[float, float]],
    runtype: str,
    region_name: str,
    mettype: str,
    metdir: Any,
    use_cpl_bypass: bool,
    inputdata: str,
    runroot: str,
    caseroot: str,
    modelroot: str,
    compsets: list[str],
    suffix: list[str],
    case_suffix: str,
    startyear: list[int],
    nyears: list[int],
    depends: Any,
    istreatment: Any,
    treatment_options: dict[str, dict[str, Any]],
    case_options: dict[str, Any],
    lat_bounds: Any,
    lon_bounds: Any,
    scriptdir: str,
) -> list[dict[str, Any]]:
    """Run emulator-backed OLMT cases and write ELM-style h0 files."""
    simulation = cfg.get("simulation", {})
    olmtdir = Path(scriptdir).resolve()
    emulator_root = _path_from_config(
        simulation.get("emulator_root", modelroot),
        base=olmtdir,
        inputdata=inputdata,
    )
    if emulator_root is None:
        raise ValueError("[machine] modelroot must point to the ELM_emulator checkout when emulator=True")

    dry_run = _as_bool(simulation.get("emulator_dry_run", False), False)
    results: list[dict[str, Any]] = []
    restart_by_case: dict[tuple[str, int], Path] = {}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(emulator_root) + os.pathsep + env.get("PYTHONPATH", "")

    print("\nELM emulator enabled: native ELM create/build/submit will be skipped.")
    print("Emulator root: " + str(emulator_root))

    for site in sites:
        for case_index, compset in enumerate(compsets):
            case_options_this = dict(case_options)
            case_metdir = metdir
            if int(istreatment[case_index]):
                treatment_name = suffix[case_index]
                case_options_this.update(treatment_options.get(treatment_name, {}))
                case_metdir = treatment_options.get(treatment_name, {}).get("metdir", case_metdir)

            mysuffix = "_".join(part for part in [suffix[case_index], case_suffix] if part)
            casename = _olmt_case_name(
                cfg=cfg,
                site=site,
                region_name=region_name,
                compset=compset,
                suffix=mysuffix,
            )
            casedir = Path(caseroot) / casename
            rundir = Path(runroot) / casename / "run"
            nmonths = int(simulation.get("emulator_nmonths", int(nyears[case_index]) * 12))
            if nmonths <= 0:
                print("Skipping zero-length emulator case: " + casename)
                continue

            resolved_metdir = _resolve_metdir(
                mettype=mettype,
                metdir=case_metdir,
                site=site,
                inputdata=inputdata,
                use_cpl_bypass=use_cpl_bypass,
                olmtdir=olmtdir,
            )
            metdata_type = _metdata_type_for_case(case_options_this, case_index, mettype)
            met_format = _infer_met_format(
                metdata_type=metdata_type,
                site=site,
                requested=str(simulation.get("emulator_met_format", "auto")),
            )
            history_frequency = _infer_history_frequency(
                simulation=simulation,
                case_options=case_options_this,
                case_index=case_index,
                casename=casename,
            )

            restart_in = None
            dependency = int(depends[case_index])
            if dependency >= 0:
                restart_in = restart_by_case.get((site, dependency))
            end_year = int(startyear[case_index]) + max(1, (nmonths + 11) // 12)
            restart_out = rundir / f"{casename}.elm.r.{end_year:04d}-01-01-00000.nc"

            command = _build_command(
                cfg=cfg,
                case_options=case_options_this,
                case_index=case_index,
                site=site,
                siteinfo=siteinfo,
                point_list=point_list,
                runtype=runtype,
                metdir=resolved_metdir,
                met_format=met_format,
                inputdata=inputdata,
                emulator_root=emulator_root,
                olmtdir=olmtdir,
                rundir=rundir,
                casename=casename,
                start_year=int(startyear[case_index]),
                nmonths=nmonths,
                lat_bounds=lat_bounds,
                lon_bounds=lon_bounds,
                restart_in=restart_in,
                restart_out=restart_out,
                history_frequency=history_frequency,
            )
            metadata = {
                "compset": compset,
                "site": site,
                "runtype": runtype,
                "mettype": mettype,
                "metdata_type": metdata_type,
                "metdir": str(resolved_metdir),
                "met_format": met_format,
                "co2_file": str(
                    _co2_file_path(
                        simulation=simulation,
                        case_options=case_options_this,
                        case_index=case_index,
                        inputdata=inputdata,
                        olmtdir=olmtdir,
                    )
                ),
                "startyear": int(startyear[case_index]),
                "nyears": int(nyears[case_index]),
                "nmonths": nmonths,
                "restart_in": restart_in,
                "restart_out": restart_out,
                "history_frequency": history_frequency,
                "hist_nhtfrq": _first_int_option(case_options_this, ("hist_nhtfrq", "nhtfrq"), case_index),
                "hist_mfilt": _first_int_option(case_options_this, ("hist_mfilt", "mfilt"), case_index),
            }
            _write_case_metadata(
                casedir=casedir,
                rundir=rundir,
                casename=casename,
                command=command,
                metadata=metadata,
            )

            print("Prepared emulator case: " + casename)
            print("Run directory: " + str(rundir))
            if dry_run:
                print("Dry run: command written but not executed.")
            else:
                log_path = rundir / "emulator.log"
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        command,
                        cwd=emulator_root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Emulator case {casename} failed with exit code {result.returncode}. "
                        f"See {log_path}"
                    )
                print("Finished emulator case: " + casename)

            restart_by_case[(site, case_index)] = restart_out
            results.append(
                {
                    "casename": casename,
                    "casedir": str(casedir),
                    "rundir": str(rundir),
                    "restart_out": str(restart_out),
                    "dry_run": dry_run,
                }
            )

    return results
