#!/usr/bin/env python3
"""Refresh Shuttle sites from an OLMT config before starting Docker."""

import argparse
import configparser
from pathlib import Path

from fluxnet_shuttle_to_olmt import sync_site


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputdata-host", type=Path, required=True)
    parser.add_argument("--inputdata-container", default="/inputdata")
    args = parser.parse_args()
    cfg = configparser.ConfigParser(interpolation=None)
    if not cfg.read(args.config):
        raise FileNotFoundError(args.config)
    sitegroup = cfg.get("simulation", "sitegroup", fallback="").strip().lower()
    enabled = (cfg.getboolean("fluxnet_shuttle", "enabled")
               if cfg.has_option("fluxnet_shuttle", "enabled")
               else sitegroup == "fluxnet_shuttle")
    if not enabled:
        return
    sites = [site.strip() for site in cfg.get("simulation", "sites").split(",") if site.strip()]
    if not sites or "all" in sites:
        raise ValueError("Shuttle requires explicit site IDs")
    container_root = Path(args.inputdata_container)
    configured_root = cfg.get("fluxnet_shuttle", "output_root", fallback=str(
        container_root / "atm/datm7/CLM1PT_data/fluxnet_shttle"))
    configured_root = Path(configured_root)
    if configured_root.is_relative_to(container_root):
        host_root = args.inputdata_host / configured_root.relative_to(container_root)
    elif configured_root.is_absolute():
        raise ValueError("Docker Shuttle output_root must be under " + str(container_root))
    else:
        host_root = args.inputdata_host / configured_root
    height = cfg.getfloat("fluxnet_shuttle", "height", fallback=None)
    for site in sites:
        result = sync_site(site, host_root, height=height)
        print("FLUXNET Shuttle " + site + ": " +
              ("reusing " if result["reused"] else "saved ") + result["directory"], flush=True)


if __name__ == "__main__":
    main()
