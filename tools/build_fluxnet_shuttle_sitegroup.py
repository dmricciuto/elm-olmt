#!/usr/bin/env python3
"""Build a metadata-only OLMT sitegroup from a FLUXNET Shuttle listall snapshot.

No flux or meteorological products are downloaded. AmeriFlux site metadata and
the ICOS station pages provide elevations; OpenTopoData fills the few gaps.
"""

import argparse
import csv
import math
import time
from pathlib import Path

import requests


REGIONS = {"US", "CA", "MX", "GL", "PR", "CR", "PA",
           "AR", "BR", "CL", "PE", "GF"}
AMERIFLUX_INFO = "https://amfcdn.lbl.gov/api/v2/site_info_display/AmeriFlux"
DEM_URL = "https://api.opentopodata.org/v1/mapzen"
ICOS_ELEVATIONS = {
    "GL-ZaH": (41.0, "https://meta.icos-cp.eu/resources/stations/ES_GL-ZaH"),
    "GL-NuF": (43.0, "https://meta.icos-cp.eu/resources/stations/ES_GL-NuF"),
    "GL-Dsk": (8.0, "https://meta.icos-cp.eu/resources/stations/ES_GL-Dsk"),
    "GF-Guy": (40.0, "https://meta.icos-cp.eu/resources/stations/ES_GF-Guy"),
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, skipinitialspace=True))


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def local_pfts(ptclm):
    known = {}
    for group in ("Fluxnet2015", "AmeriFlux"):
        for row in read_csv(ptclm / f"{group}_pftdata.txt"):
            site = row["site_code"]
            # The legacy US-Ton row assigns a tropical tree to a California
            # savanna; use the IGBP-based temperate savanna estimate instead.
            if site == "US-Ton":
                continue
            mix = {}
            for number in range(1, 6):
                pct = float(row[f"pft_f{number}"])
                pft = int(row[f"pft_c{number}"])
                if pct > 0:
                    if pft > 16:
                        raise ValueError(f"{site}: PFT {pft} exceeds standard 17-PFT ELM")
                    mix[pft] = mix.get(pft, 0.0) + pct
            if mix and math.isclose(sum(mix.values()), 100.0, abs_tol=0.01):
                known[site] = (mix, f"local {group}_pftdata.txt", "site-specific")
    return known


def local_soils(ptclm):
    known = {}
    for group in ("Fluxnet2015", "AmeriFlux"):
        for row in read_csv(ptclm / f"{group}_soildata.txt"):
            site = row["site_code"]
            sand = float(row["layer_sand%"])
            clay = float(row["layer_clay%"])
            # Zero/zero in the legacy table is an unknown-texture placeholder.
            if min(sand, clay) < 0 or sand + clay <= 0 or sand + clay > 100:
                continue
            source = f"local {group}_soildata.txt"
            if site in known and (abs(known[site][0] - sand) > 10
                                  or abs(known[site][1] - clay) > 10):
                source += f"; conflicts with {known[site][2]}"
            known[site] = (sand, clay, source)
    return known


def inferred_pfts(igbp, lat, elevation, name):
    """Conservative ELM 17-PFT translation of a single IGBP class."""
    arctic = abs(lat) >= 65
    boreal = abs(lat) >= 50
    tropical = abs(lat) < 25
    needle = 2 if boreal else 1
    broad_decid = 8 if boreal else (6 if tropical else 7)
    broad_ever = 4 if tropical else 5
    grass = 12 if arctic else 13
    shrub = 11 if boreal else 10
    label = name.lower()
    if igbp == "ENF":
        return {needle: 100}, "IGBP ENF + latitude", "medium"
    if igbp == "DNF":
        return {3: 100}, "IGBP DNF", "high"
    if igbp == "EBF":
        return {broad_ever: 100}, "IGBP EBF + latitude", "medium"
    if igbp == "DBF":
        return {broad_decid: 100}, "IGBP DBF + latitude", "medium"
    if igbp == "MF":
        return {needle: 50, broad_decid: 50}, "IGBP MF 50/50 estimate", "low"
    if igbp == "GRA":
        if arctic:
            return {12: 100}, "IGBP GRA; arctic C3 grass", "medium"
        if abs(lat) < 30 and elevation < 1500:
            return {14: 100}, "IGBP GRA; warm-site C4 estimate", "low"
        if abs(lat) < 45:
            return {13: 50, 14: 50}, "IGBP GRA; C3/C4 mix estimate", "low"
        return {13: 100}, "IGBP GRA; cool-site C3 estimate", "medium"
    if igbp == "CRO":
        return {15: 100}, "IGBP CRO; generic C3 crop", "low"
    if igbp == "WET":
        if any(word in label for word in ("forest", "swamp", "wooded")):
            return {(broad_ever if tropical else broad_decid): 100}, "IGBP WET + forest name", "low"
        if any(word in label for word in ("bog", "peat", "mire")):
            return {shrub: 50, grass: 50}, "IGBP WET + peatland name", "low"
        return {grass: 100}, "IGBP WET; herbaceous proxy", "low"
    if igbp == "OSH":
        return {shrub: 100}, "IGBP OSH; deciduous shrub proxy", "low"
    if igbp == "CSH":
        return {(11 if arctic else 9): 100}, "IGBP CSH; shrub proxy", "low"
    if igbp == "WSA":
        return {broad_decid: 45, grass: 55}, "IGBP WSA; 45% tree estimate", "low"
    if igbp == "SAV":
        return {broad_decid: 20, grass: 80}, "IGBP SAV; 20% tree estimate", "low"
    if igbp == "CVM":
        return {15: 50, grass: 50}, "IGBP CVM; 50/50 crop-grass estimate", "low"
    if igbp in ("BSV", "SNO"):
        return {0: 100}, f"IGBP {igbp}; bare ground proxy", "low"
    raise ValueError(f"Unmapped IGBP class: {igbp}")


def terrain_elevations(rows):
    estimates = {}
    for offset in range(0, len(rows), 100):
        batch = rows[offset:offset + 100]
        locations = "|".join(f"{r['location_lat']},{r['location_long']}" for r in batch)
        response = requests.get(DEM_URL, params={"locations": locations}, timeout=45)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "OK" or len(data.get("results", [])) != len(batch):
            raise ValueError("OpenTopoData returned an incomplete elevation batch")
        for row, result in zip(batch, data["results"]):
            value = result.get("elevation")
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"No terrain elevation for {row['site_id']}")
            estimates[row["site_id"]] = float(value)
        if offset + 100 < len(rows):
            time.sleep(1.1)  # Public OpenTopoData limit: one call per second.
    return estimates


def build(snapshot, ptclm):
    catalogue = read_csv(snapshot)
    candidates = []
    for row in catalogue:
        if (row["site_id"].split("-")[0] in REGIONS
                and int(row["last_year"]) - int(row["first_year"]) + 1 >= 5):
            candidates.append(row)
    candidates.sort(key=lambda row: row["site_id"])
    if not candidates:
        raise ValueError("No North or South American sites with a five-year span")

    response = requests.get(AMERIFLUX_INFO, timeout=60)
    response.raise_for_status()
    ameriflux = {site["site_id"]: site for site in response.json().get("values", [])}
    selected = []
    for row in candidates:
        if row["data_hub"] == "AmeriFlux":
            info = ameriflux.get(row["site_id"])
            if info is None:
                raise ValueError(f"No AmeriFlux metadata for {row['site_id']}")
            years = sorted({int(year) for year in info.get("grp_publish_fluxnet", [])})
            if len(years) < 5:
                continue
            if years[0] != int(row["first_year"]) or years[-1] != int(row["last_year"]):
                raise ValueError(f"Catalogue and AmeriFlux years disagree for {row['site_id']}")
            row = {**row, "actual_years": years, "year_basis": "AmeriFlux published years"}
        else:
            row = {**row, "actual_years": [], "year_basis": "Shuttle first-last span"}
        selected.append(row)

    unresolved = []
    for row in selected:
        site = row["site_id"]
        if row["data_hub"] == "AmeriFlux":
            raw = ameriflux[site].get("grp_location", {}).get("location_elev")
            try:
                elevation = float(raw)
            except (TypeError, ValueError):
                elevation = math.nan
            if math.isfinite(elevation):
                row["elevation"] = elevation
                row["elevation_source"] = AMERIFLUX_INFO
                continue
        if site in ICOS_ELEVATIONS:
            row["elevation"], row["elevation_source"] = ICOS_ELEVATIONS[site]
        else:
            unresolved.append(row)
    if unresolved:
        estimates = terrain_elevations(unresolved)
        for row in unresolved:
            row["elevation"] = estimates[row["site_id"]]
            row["elevation_source"] = DEM_URL + " (terrain estimate)"

    pfts = local_pfts(ptclm)
    soils = local_soils(ptclm)
    site_rows, pft_rows, soil_rows, audit_rows = [], [], [], []
    for row in selected:
        site = row["site_id"]
        lat = float(row["location_lat"])
        if site in pfts:
            mix, pft_source, confidence = pfts[site]
        else:
            mix, pft_source, confidence = inferred_pfts(
                row["igbp"], lat, row["elevation"], row["site_name"])
        if not math.isclose(sum(mix.values()), 100, abs_tol=0.01):
            raise ValueError(f"{site}: PFT fractions do not sum to 100")
        site_rows.append({
            "site_code": site, "name": row["site_name"], "state": site.split("-")[0],
            "lon": row["location_long"], "lat": row["location_lat"],
            "elev": f"{row['elevation']:g}", "startyear": row["first_year"],
            "endyear": row["last_year"], "alignyear": 1851,
        })
        pft_row = {"site_code": site}
        for number, (pft, pct) in enumerate(sorted(mix.items()), 1):
            pft_row[f"pft_f{number}"] = f"{pct:g}"
            pft_row[f"pft_c{number}"] = pft
        for number in range(len(mix) + 1, 6):
            pft_row[f"pft_f{number}"] = 0
            pft_row[f"pft_c{number}"] = 0
        pft_rows.append(pft_row)
        soil_source = "not available; retain gridded surface texture"
        sand = clay = ""
        if site in soils:
            sand, clay, soil_source = soils[site]
            soil_rows.append({"site_code": site, "soil_depth": -999, "n_layers": 1,
                              "layer_depth": -999, "layer_sand%": sand,
                              "layer_clay%": clay})
        audit_rows.append({
            "site_code": site, "catalogue_snapshot": Path(snapshot).name,
            "data_hub": row["data_hub"], "igbp": row["igbp"],
            "first_year": row["first_year"], "last_year": row["last_year"],
            "year_count": len(row["actual_years"]) if row["actual_years"] else int(row["last_year"]) - int(row["first_year"]) + 1,
            "year_basis": row["year_basis"],
            "published_years": ";".join(map(str, row["actual_years"])),
            "latitude": row["location_lat"], "longitude": row["location_long"],
            "elevation_m": row["elevation"], "elevation_source": row["elevation_source"],
            "pft_mix": ";".join(f"{pft}:{pct:g}" for pft, pct in sorted(mix.items())),
            "pft_source": pft_source, "pft_confidence": confidence,
            "sand_pct": sand, "clay_pct": clay, "soil_source": soil_source,
            "product_id": row.get("product_id", ""),
            "product_name": row.get("fluxnet_product_name", ""),
        })
    ptclm.mkdir(parents=True, exist_ok=True)
    write_csv(ptclm / "fluxnet_shuttle_sitedata.txt",
              ["site_code", "name", "state", "lon", "lat", "elev", "startyear", "endyear", "alignyear"], site_rows)
    write_csv(ptclm / "fluxnet_shuttle_pftdata.txt",
              ["site_code"] + [field for n in range(1, 6) for field in (f"pft_f{n}", f"pft_c{n}")], pft_rows)
    write_csv(ptclm / "fluxnet_shuttle_soildata.txt",
              ["site_code", "soil_depth", "n_layers", "layer_depth", "layer_sand%", "layer_clay%"], soil_rows)
    write_csv(ptclm / "fluxnet_shuttle_catalog.csv", list(audit_rows[0]), audit_rows)
    source_fields = ("data_hub", "site_id", "site_name", "location_lat",
                     "location_long", "igbp", "first_year", "last_year",
                     "fluxnet_product_name", "product_id")
    write_csv(ptclm / "fluxnet_shuttle_source_snapshot.csv", source_fields,
              [{field: row.get(field, "") for field in source_fields} for row in selected])
    print(f"Wrote {len(site_rows)} sites, {len(pft_rows)} PFT mappings, "
          f"{len(soil_rows)} soil records to {ptclm}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True,
                        help="CSV made by fluxnet-shuttle listall")
    parser.add_argument("--ptclm", type=Path, required=True,
                        help="Inputdata/lnd/clm2/PTCLM directory")
    args = parser.parse_args()
    build(args.snapshot, args.ptclm)


if __name__ == "__main__":
    main()
