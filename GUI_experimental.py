import os
from platform import machine
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import numpy as np
import tkinter.simpledialog as simpledialog
from OLMTutils import get_machine_info, get_site_info, get_sitegroups
import mapselect
#import pickle
import subprocess
import threading

class E3SMConfigurator(tk.Toplevel):   #Tk):
    def __init__(self, master=None):
        super().__init__(master)
        self.selected_site_indices = []
        self.selected_site_names = []  # Track selected site names for persistence
        self._last_tab_index = 0  # Track previous tab index for selection persistence
        self._restoring_selection = False
        self.title("E3SM GUI Configurator")
        self.geometry("1300x900")
        self.result = None
        self.ssh = None
        self.sftp = None
        self.logged_in = False

        # When the window is closed via the window manager, call on_close
        self.protocol("WM_DELETE_WINDOW", self.on_close)        

        # Execute get_machine_info to get defaults
        self.machine, self.rootdir, self.inputdata, self.queue, self.project, self.hostname = \
		get_machine_info()
        if self.machine == 'docker':
            self.sitegroups = get_sitegroups(os.environ['HOME']+'/models/inputdata')
            self.siteinfo = get_site_info(os.environ['HOME']+'/models/inputdata')
        else:
            self.sitegroups = get_sitegroups(self.inputdata)
            self.siteinfo = get_site_info(self.inputdata)

        inputdata = self.inputdata
        modelroot = '/code/E3SM'
        exeroot_default = ''   # default blank
        queue_default = 'batch'
        
        # Top Frame for machine and directory settings
        top_frame = ttk.Frame(self)
        top_frame.pack(fill='x', padx=10, pady=10)
        
        # Row 0: run root and Case Root

        tk.Label(top_frame, text="Case Root:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.caseroot_entry = tk.Entry(top_frame, width=45)
        self.caseroot_entry.insert(0, self.rootdir+'/e3sm_cases')
        self.caseroot_entry.grid(row=0, column=1, padx=5, pady=5)
        browse_caseroot = tk.Button(top_frame, text="Browse",
                            command=lambda: self.smart_browse(self.caseroot_entry))
        browse_caseroot.grid(row=0, column=2, padx=5, pady=5)

        tk.Label(top_frame, text="Run Root:").grid(row=0, column=3, sticky="w", padx=5, pady=5)
        self.runroot_entry = tk.Entry(top_frame, width=45)
        self.runroot_entry.insert(0, self.rootdir+'/e3sm_run')
        self.runroot_entry.grid(row=0, column=4, padx=5, pady=5)
        browse_runroot = tk.Button(top_frame, text="Browse",
                            command=lambda: self.smart_browse(self.runroot_entry))
        browse_runroot.grid(row=0, column=5, padx=5, pady=5)

        # Row 1: Exe Root and Model Root
        tk.Label(top_frame, text="Model Root:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.modelroot_entry = tk.Entry(top_frame, width=45)
        self.modelroot_entry.insert(0, modelroot)
        self.modelroot_entry.grid(row=1, column=1, padx=5, pady=5)
        browse_modelroot = tk.Button(top_frame, text="Browse", command=lambda: self.smart_browse(self.modelroot_entry))
        browse_modelroot.grid(row=1, column=2, padx=5, pady=5)

        tk.Label(top_frame, text="Exeroot:").grid(row=1, column=3, sticky="w", padx=5, pady=5)
        self.exeroot_entry = tk.Entry(top_frame, width=45)
        self.exeroot_entry.insert(0, exeroot_default)
        self.exeroot_entry.grid(row=1, column=4, padx=5, pady=5)
        browse_exeroot = tk.Button(top_frame, text="Browse",
                            command=lambda: self.browse_directory(self.smart_browse(self.exeroot_entry)))
        browse_exeroot.grid(row=1, column=5, padx=5, pady=5)


        # Row 2: Input data
        tk.Label(top_frame, text="Input data:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.inputdata_entry = tk.Entry(top_frame, width=45)
        self.inputdata_entry.insert(0, inputdata)
        self.inputdata_entry.grid(row=2, column=1, padx=5, pady=5)
        browse_inputdata = tk.Button(top_frame, text="Browse",
                            command=lambda: self.smart_browse(self.inputdata_entry))
        browse_inputdata.grid(row=2, column=2, padx=5, pady=5)

        # Row 2b: Met Directory (initially visible)
        self.metdir_label = tk.Label(top_frame, text="Met Directory:")
        self.metdir_label.grid(row=2, column=3, sticky="w", padx=5, pady=5)
        self.metdir_entry = tk.Entry(top_frame, width=45)
        # Set metdir based on initial mettype
        initial_mettype = self.mettype_var.get() if hasattr(self, "mettype_var") else "era5-daymet"
        if initial_mettype == "era5-daymet":
            subdir = "atm/datm7/Daymet_ERA5_TESSFA2/cpl_bypass_full"
        elif initial_mettype == "gswp3":
            subdir = "atm/datm7/atm_forcing.datm7.GSWP3.0.5d.v2.c180716/cpl_bypass_full"
        elif initial_mettype == "crujra":
            subdir = "atm/datm7/atm_forcing.CRUJRA_trendy_2025/cpl_bypass_full"
        else:
            subdir = "atm/datm7"
        default_metdir = os.path.join(self.inputdata_entry.get(), subdir)
        self.metdir_entry.insert(0, default_metdir)
        self.metdir_entry.grid(row=2, column=4, padx=5, pady=5)
        self.browse_metdir = tk.Button(top_frame, text="Browse",
                               command=lambda: self.smart_browse(self.metdir_entry))
        self.browse_metdir.grid(row=2, column=5, padx=5, pady=5)

        # Row 3:  queue and machine
        tk.Label(top_frame, text="Machine:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.machine_var = tk.StringVar(value=self.machine)
        machine_options = ["cades-baseline", "chrysalis", "pm-cpu", "docker"]
        self.machine_menu = ttk.Combobox(top_frame, textvariable=self.machine_var, values=machine_options, state="readonly")
        self.machine_menu.grid(row=3, column=1, padx=5, pady=5)
        self.machine_menu.bind("<<ComboboxSelected>>", self.on_machine_selected)

        tk.Label(top_frame, text="Queue:").grid(row=3, column=3, sticky="w", padx=5, pady=5)
        self.queue_entry = tk.Entry(top_frame, width=45)
        self.queue_entry.insert(0, queue_default)
        self.queue_entry.grid(row=3, column=4, padx=5, pady=5)
        
        self.login_btn = tk.Button(top_frame, text="Login", command=self.remote_login)
        self.login_btn.grid(row=3, column=2, padx=5, pady=5)

        # Row 4: Remote OLMT Path (initially disabled)
        tk.Label(top_frame, text="Remote OLMT Path:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.remote_olmt_var = tk.StringVar()
        self.remote_olmt_entry = tk.Entry(top_frame, textvariable=self.remote_olmt_var, width=45, state="disabled")
        self.remote_olmt_entry.grid(row=4, column=1, padx=5, pady=5)
        self.browse_remote_olmt_btn = tk.Button(top_frame, text="Browse", state="disabled",
                                        command=lambda: self.browse_remote_directory(self.remote_olmt_entry))
        self.browse_remote_olmt_btn.grid(row=4, column=2, padx=5, pady=5)

        # Notebook for other options
        self.notebook = ttk.Notebook(self)
        self.req_frame = ttk.Frame(self.notebook)
        self.ens_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.req_frame, text="Required Inputs")
        self.notebook.add(self.ens_frame, text="Ensemble Options")
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        self.create_required_inputs(self.req_frame)
        self.create_ensemble_options(self.ens_frame)

        # Bind tab change event to preserve/restore site selections
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
    
        # Frame for bottom buttons
        bottom_btn_frame = tk.Frame(self)
        bottom_btn_frame.pack(side=tk.BOTTOM, pady=10)

        save_btn = tk.Button(bottom_btn_frame, text="Save Configuration", command=self.save_configuration)
        save_btn.pack(side=tk.LEFT, padx=10)

        run_docker_btn = tk.Button(bottom_btn_frame, text="Run", command=self.run_button_action)
        run_docker_btn.pack(side=tk.LEFT, padx=10)


        self.use_cpl_bypass_var.trace_add("write", lambda *args: self.update_metdir_visibility(top_frame))


    def on_site_selection(self, event=None):
        print(self._restoring_selection)
        if self._restoring_selection:
            return
        self.selected_site_names = [self.site_menu.get(i) for i in self.site_menu.curselection()]
        print("Selected sites:")
        print(self.selected_site_names)


    def on_tab_changed(self, event=None):
        current_tab = self.notebook.index(self.notebook.select())
        if (current_tab != 0):
            self._restoring_selection = True
        # Only restore selection when returning to Required Inputs tab (tab 0)
        if current_tab == 0 and hasattr(self, 'site_menu'):
            self._restoring_selection = True
            self.site_menu.selection_clear(0, tk.END)
            site_list = [self.site_menu.get(i) for i in range(self.site_menu.size())]
            for name in self.selected_site_names:
                if name in site_list:
                    idx = site_list.index(name)
                    self.site_menu.selection_set(idx)
            self._restoring_selection = False
        self._last_tab_index = current_tab

    def browse_directory(self, entry):
      directory = filedialog.askdirectory(initialdir=entry.get() or "/")
      if directory:
        entry.delete(0, tk.END)
        entry.insert(0, directory)

    def create_required_inputs(self, parent):
        row = 0

        # --- Run Type and Met Type ---
        tk.Label(parent, text="Run Type:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.runtype_var = tk.StringVar(value="latlon_bbox")
        runtype_options = ["site", "latlon_list", "latlon_bbox"]
        self.runtype_menu = ttk.Combobox(parent, textvariable=self.runtype_var, values=runtype_options, state="readonly")
        self.runtype_menu.grid(row=row, column=1, padx=5, pady=5)
        self.runtype_menu.bind("<<ComboboxSelected>>", self.update_runtype_fields)

        tk.Label(parent, text="Met Type:").grid(row=row, column=2, sticky="w", padx=5, pady=5)
        self.mettype_var = tk.StringVar(value="era5-daymet")
        mettype_options = ["era5-daymet", "gswp3", "crujra"]
        self.mettype_menu = ttk.Combobox(parent, textvariable=self.mettype_var, values=mettype_options, state="readonly")
        self.mettype_menu.grid(row=row, column=3, padx=5, pady=5)
        self.mettype_menu.bind("<<ComboboxSelected>>", self.on_mettype_selected)
        row += 1

        # --- Case Suffix ---
        tk.Label(parent, text="Case Suffix:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.case_suffix_entry = tk.Entry(parent)
        self.case_suffix_entry.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        # --- Site selection (if runtype == "site") ---
        self.sites_frame = tk.Frame(parent)
        tk.Label(self.sites_frame, text="Select Site Group:").grid(row=0,column=0,sticky="w")
        sitegroups = self.sitegroups
        self.sitegroup_var = tk.StringVar()
        self.sitegroup_var.set(sitegroups[0])
        self.sitegroup_menu=ttk.Combobox(self.sites_frame, textvariable=self.sitegroup_var, values=sitegroups, state="readonly")
        self.sitegroup_menu.grid(row=0,column=1,padx=5,pady=5)
        self.sitegroup_menu.bind("<<ComboboxSelected>>", self.update_sitelist)
        tk.Label(self.sites_frame, text="Select Site:").grid(row=1, column=0, sticky="w")
        #self.site_var = tk.StringVar(value="US-TDE")
        site_options = list(self.siteinfo.keys())
        #self.site_menu = ttk.Combobox(self.sites_frame, textvariable=self.site_var, values=site_options, state="readonly")
        self.site_menu = tk.Listbox(self.sites_frame, selectmode="multiple", height=6)
        for option in site_options:
             self.site_menu.insert(tk.END, option)
             #self.site_menu.pack(padx=10, pady=10)
        self.site_menu.grid(row=1, column=1, padx=5, pady=5)
        self.site_menu.bind("<<ListboxSelect>>", self.on_site_selection)
        
        # Add "Show Sites on Map" button
        self.show_sites_btn = tk.Button(self.sites_frame, text="Show Sites on Map", command=self.show_sites_on_map)
        self.show_sites_btn.grid(row=2, column=0, columnspan=2, pady=5)
        self.sites_frame.grid(row=row, column=0, columnspan=4, sticky="w", padx=5, pady=5)
        row += 1
        self.REGION_BOUNDS = {
            'Global':      [-180.25, 180.25, -90.25, 90.25],
            'Boreal North America':      [-170.25, -60.25, 49.75, 79.75],
            'Temperate North America':      [-125.25, -66.25, 30.25, 49.75],
            'CONUS':     [-125.25, -66.25, 23.25, 54.75],
            'Columbia Basin':  [-126, -108, 40.0, 55.0],
            'Central America':      [-115.25, -80.25, 9.75, 30.25],
            'South America':      [-80.25, -40.25, -59.75, 12.75],
            'Northern Hemisphere South America':      [-80.25, -50.25, 0.25, 12.75],
            'Southern Hemisphere South America':      [-80.25, -40.25, -59.75, 0.25],
            'Europe':      [-10.25, 30.25, 35.25, 70.25],
            'Middle East':      [-10.25, 60.25, 20.24, 40.25],
            'Africa':      [-20.25, 45.25, -34.75, 20.25],
            'Northern Hemisphere Africa':      [-20.25, 45.25, 0.25, 20.25],
            'Southern Hemisphere Africa':      [10.25, 45.25, -34.75, 0.25],
            'Asia':      [30.25, 179.75, -10.25, 70.25],
            'Boreal Asia':      [30.25, 179.25, 54.75, 70.25],
            'Central Asia':      [30.25, 142.58, 30.25, 54.75],
            'Southeast Asia':      [65.25, 120.25, 5.25, 30.25],
            'Equatorial Asia':      [99.75, 150.25, -10.25, 10.25],
            'Australia':      [112.00, 154.00, -41.25, -10.50],
        }
        # --- Region selection ---
        self.region_label = tk.Label(parent, text="Region:")
        self.region_var = tk.StringVar()
        region_options = ['Custom'] + list(self.REGION_BOUNDS.keys())
        self.region_menu = ttk.Combobox(parent, textvariable=self.region_var, values=region_options, state="readonly")
        self.region_menu.bind("<<ComboboxSelected>>", self.on_region_selected)
        self.region_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.region_menu.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        # --- Lat/Lon bounds (if runtype == "latlon_bbox") ---
        self.latlon_frame = tk.Frame(parent)
        tk.Label(self.latlon_frame, text="Lat Bounds (min, max):").grid(row=0, column=0, sticky="w")
        self.lat_min_entry = tk.Entry(self.latlon_frame, width=5)
        self.lat_min_entry.insert(0, "-90")
        self.lat_min_entry.grid(row=0, column=1, padx=5)
        self.lat_max_entry = tk.Entry(self.latlon_frame, width=5)
        self.lat_max_entry.insert(0, "90")
        self.lat_max_entry.grid(row=0, column=2, padx=5)

        tk.Label(self.latlon_frame, text="Lon Bounds (min, max):").grid(row=1, column=0, sticky="w")
        self.lon_min_entry = tk.Entry(self.latlon_frame, width=5)
        self.lon_min_entry.insert(0, "-180")
        self.lon_min_entry.grid(row=1, column=1, padx=5)
        self.lon_max_entry = tk.Entry(self.latlon_frame, width=5)
        self.lon_max_entry.insert(0, "180")
        self.lon_max_entry.grid(row=1, column=2, padx=5)
    
        # Button to open the Cartopy map selector.
        select_btn = tk.Button(self.latlon_frame, text="Select on Map", command=self.open_map_selector)
        select_btn.grid(row=2, column=0, columnspan=2, pady=10)

        # Add numproc entry (default 1)
        tk.Label(self.latlon_frame, text="Num Processors:").grid(row=3, column=0, sticky="w")
        self.numproc_entry = tk.Entry(self.latlon_frame, width=5)
        self.numproc_entry.insert(0, "1")
        self.numproc_entry.grid(row=3, column=1, padx=5)

        self.latlon_frame.grid(row=row, column=0, columnspan=4, sticky="w", padx=5, pady=5)
        row += 1

        # --- Resolution ---
        tk.Label(parent, text="Resolution:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.resolution_var = tk.StringVar(value="hcru_hcru")
        resolution_options = ["hcru_hcru", "r05_r05","f09_f09", "TESSFA2_4km"]
        self.resolution_menu = ttk.Combobox(parent, textvariable=self.resolution_var, values=resolution_options, state="readonly")
        self.resolution_menu.grid(row=row, column=1, padx=5, pady=5)
        self.resolution_menu.bind("<<ComboboxSelected>>", self.on_resolution_selected)
        row += 1

        # --- Boolean Options ---
        self.use_cpl_bypass_var = tk.BooleanVar(value=True)
        self.use_sp_var = tk.BooleanVar(value=False)
        self.use_fates_var = tk.BooleanVar(value=False)
        self.fates_nutrient_var = tk.BooleanVar(value=True)
        tk.Checkbutton(parent, text="Use Coupler Bypass", variable=self.use_cpl_bypass_var).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        tk.Checkbutton(parent, text="Use SP", variable=self.use_sp_var, command=self.update_sp_fields).grid(row=row, column=1, sticky="w", padx=5, pady=5)
        tk.Checkbutton(parent, text="Use FATES", variable=self.use_fates_var).grid(row=row, column=2, sticky="w", padx=5, pady=5)
        tk.Checkbutton(parent, text="FATES Nutrient", variable=self.fates_nutrient_var).grid(row=row, column=3, sticky="w", padx=5, pady=5)
        row += 1

        # --- SP vs Non-SP Year Options ---
        # Frame for SP-related entries
        self.sp_frame = tk.Frame(parent)
        tk.Label(self.sp_frame, text="Number of years (SP):").grid(row=0, column=0, padx=5, pady=5)
        self.sp_years_entry = tk.Entry(self.sp_frame, width=10)
        self.sp_years_entry.insert(0, "200")
        self.sp_years_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(self.sp_frame, text="Start year (SP):").grid(row=0, column=2, padx=5, pady=5)
        self.sp_start_entry = tk.Entry(self.sp_frame, width=10)
        self.sp_start_entry.insert(0, "1850")
        self.sp_start_entry.grid(row=0, column=3, padx=5, pady=5)

        # Frame for non-SP entries (ad, final, trans)
        self.non_sp_frame = tk.Frame(parent)
        tk.Label(self.non_sp_frame, text="NYears AD:").grid(row=0, column=0, padx=5, pady=5)
        self.nyears_ad_entry = tk.Entry(self.non_sp_frame, width=10)
        self.nyears_ad_entry.insert(0, "200")
        self.nyears_ad_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(self.non_sp_frame, text="NYears Final:").grid(row=0, column=2, padx=5, pady=5)
        self.nyears_final_entry = tk.Entry(self.non_sp_frame, width=10)
        self.nyears_final_entry.insert(0, "400")
        self.nyears_final_entry.grid(row=0, column=3, padx=5, pady=5)
        tk.Label(self.non_sp_frame, text="NYears Trans:").grid(row=0, column=4, padx=5, pady=5)
        self.nyears_trans_entry = tk.Entry(self.non_sp_frame, width=10)
        self.nyears_trans_entry.insert(0, "174")
        self.nyears_trans_entry.grid(row=0, column=5, padx=5, pady=5)

        # Initially, update which frame is shown based on use_SP state
        self.update_sp_fields()

        # Place SP and non-SP frames (they will be shown/hidden accordingly)
        self.sp_frame.grid(row=row, column=0, columnspan=4, sticky="w", padx=5, pady=5)
        self.non_sp_frame.grid(row=row, column=0, columnspan=4, sticky="w", padx=5, pady=5)
        row += 1

        # --- Case Options (multiline text) ---
        tk.Label(parent, text="Case Options:").grid(row=row, column=0, sticky="nw", padx=5, pady=5)
        self.case_options_text = scrolledtext.ScrolledText(parent, width=60, height=10)
        self.case_options_text.grid(row=row, column=1, columnspan=3, padx=5, pady=5)
        # Insert some dummy text (modify as needed)
        self.case_options_text.insert(tk.END, "{\n   variable: value,\n    # ...\n}")
        row += 1

        # Finally, update which runtype frame to show:
        self.update_runtype_fields()

    def open_map_selector(self):
        """Open the Cartopy map selector and update the lat/lon entries with the selection."""
        def update_bounds(min_lon, max_lon, min_lat, max_lat):
            self.lat_min_entry.delete(0, tk.END)
            self.lat_min_entry.insert(0, f"{min_lat:.2f}")
            self.lat_max_entry.delete(0, tk.END)
            self.lat_max_entry.insert(0, f"{max_lat:.2f}")
            self.lon_min_entry.delete(0, tk.END)
            self.lon_min_entry.insert(0, f"{min_lon:.2f}")
            self.lon_max_entry.delete(0, tk.END)
            self.lon_max_entry.insert(0, f"{max_lon:.2f}")
            # Set region to "Custom"
            self.region_var.set("Custom")

        # Get current bounds from the entries
        try:
            min_lat = float(self.lat_min_entry.get())
            max_lat = float(self.lat_max_entry.get())
            min_lon = float(self.lon_min_entry.get())
            max_lon = float(self.lon_max_entry.get())
        except ValueError:
            # Fallback to global if parsing fails
            min_lat, max_lat, min_lon, max_lon = -90, 90, -180, 180

        # Pass initial bounds to the map selector
        mapselect.CartopyMapSelector(
            update_bounds,
            initial_bounds=(min_lon, max_lon, min_lat, max_lat)
        )

    def show_sites_on_map(self):
        selected_sitegroup = self.sitegroup_var.get()
        myinputdata = self.inputdata_entry.get()
        if (self.machine == 'docker'):
            myinputdata = os.path.join(os.environ['HOME'], 'models/inputdata')
        siteinfo = get_site_info(myinputdata, sitegroup=selected_sitegroup, sftp=self.sftp)
        sites = []
        name_to_index = {}
        for idx, (sitename, info) in enumerate(siteinfo.items()):
            lat = info.get("lat")
            lon = info.get("lon")
            if lat is not None and lon is not None:
                sites.append({"name": sitename, "lat": lat, "lon": lon})
                name_to_index[sitename] = idx
        self.selected_site_names = [self.site_menu.get(i) for i in self.site_menu.curselection()]

        def update_selected_sites(selected_sites):
            # selected_sites is a list of site dicts
            self.site_menu.selection_clear(0, tk.END)
            for site in selected_sites:
                idx = name_to_index.get(site["name"])
                if idx is not None:
                    self.site_menu.selection_set(idx)
                    self.site_menu.see(idx)

        mapselect.CartopyMapSelector(update_selected_sites, sites=sites)

    def update_runtype_fields(self, event=None):
        """
        Show/hide the site selection, region, or lat/lon bounds based on runtype.
        """
        runtype = self.runtype_var.get()
        if runtype == "site":
            self.sites_frame.grid()
            self.latlon_frame.grid_remove()
            self.region_label.grid_remove()
            self.region_menu.grid_remove()
            self.update_sitelist()
            self.sitegroup_menu.set(self.sitegroups[0])
            self.numproc_entry.delete(0, tk.END)
            self.numproc_entry.insert(0, "1")
        elif runtype == "latlon_bbox":
            self.latlon_frame.grid()
            self.region_label.grid()
            self.region_menu.grid()
            self.sites_frame.grid_remove()
        else:
            self.sites_frame.grid_remove()
            self.latlon_frame.grid_remove()
            self.region_label.grid_remove()
            self.region_menu.grid_remove()
            self.numproc_entry.delete(0, tk.END)
            self.numproc_entry.insert(0, "1")

    def update_sitelist(self, event=None):
        # Get the currently selected sitegroup from the sitegroup_menu
        selected_sitegroup = self.sitegroup_menu.get()
        if selected_sitegroup:
            # Call get_site_info to update the site information
            myinputdata = self.inputdata_entry.get()
            if (self.machine == 'docker'):
                myinputdata = os.path.join(os.environ['HOME'], 'models/inputdata')
            self.siteinfo = get_site_info(myinputdata, sitegroup=selected_sitegroup, sftp=self.sftp)

            new_site_list = list(self.siteinfo.keys())
            self.site_menu.delete(0, tk.END)
            for option in new_site_list:
                self.site_menu.insert(tk.END, option)
            # Restore selection after updating site list
            for name in self.selected_site_names:
                if name in new_site_list:
                    idx = new_site_list.index(name)
                    self.site_menu.selection_set(idx)

    def update_sp_fields(self):
        """
        If "use SP" is checked, show the SP fields and hide the non-SP fields.
        Otherwise, show the non-SP fields.
        """
        if self.use_sp_var.get():
            self.sp_frame.grid()
            self.non_sp_frame.grid_remove()
        else:
            self.non_sp_frame.grid()
            self.sp_frame.grid_remove()

    def create_ensemble_options(self, parent):
        row = 0

        tk.Label(parent, text="Parameter List:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.parm_list_entry = tk.Entry(parent)
        self.parm_list_entry.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        tk.Label(parent, text="NSamples:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.nsamples_entry = tk.Entry(parent)
        self.nsamples_entry.insert(0, "1000")
        self.nsamples_entry.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        tk.Label(parent, text="N Parallel Ensemble:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.np_ensemble_entry = tk.Entry(parent)
        self.np_ensemble_entry.insert(0, "384")
        self.np_ensemble_entry.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        tk.Label(parent, text="Ensemble File:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.ensemble_file_entry = tk.Entry(parent)
        self.ensemble_file_entry.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        tk.Label(parent, text="Postproc Variables:").grid(row=row, column=0, sticky="nw", padx=5, pady=5)
        self.postproc_vars_text = scrolledtext.ScrolledText(parent, width=40, height=5)
        self.postproc_vars_text.grid(row=row, column=1, padx=5, pady=5)
        self.postproc_vars_text.insert(tk.END, "GPP, ER, NPP, NEE, TLAI, FSH, EFLX_LH_TOT")
        row += 1

        tk.Label(parent, text="Postproc Start Year:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.postproc_start_entry = tk.Entry(parent)
        self.postproc_start_entry.insert(0, "2000")
        self.postproc_start_entry.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        tk.Label(parent, text="Postproc End Year:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.postproc_end_entry = tk.Entry(parent)
        self.postproc_end_entry.insert(0, "2007")
        self.postproc_end_entry.grid(row=row, column=1, padx=5, pady=5)
        row += 1

        tk.Label(parent, text="Postproc Frequency:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.postproc_freq_var = tk.StringVar(value="monthly")
        postproc_freq_options = ["daily", "monthly", "annual"]
        self.postproc_freq_menu = ttk.Combobox(parent, textvariable=self.postproc_freq_var, values=postproc_freq_options, state="readonly")
        self.postproc_freq_menu.grid(row=row, column=1, padx=5, pady=5)
        row += 1

    def save_configuration(self):
        """
        Gather all configuration options and print (or save) them.
        """
        config = {}
        # Top-level machine and directory options
        config["machine"] = self.machine_var.get()
        config["caseroot"] = self.caseroot_entry.get()
        config["runroot"] = self.runroot_entry.get()
        config["modelroot"] = self.modelroot_entry.get()
        config["exeroot"] = self.exeroot_entry.get()
        config["queue"] = self.queue_entry.get()
        config["project"] = self.project
        config["inputdata"] = self.inputdata_entry.get()

        # Required inputs
        config["runtype"] = self.runtype_var.get()
        config["mettype"] = self.mettype_var.get()
        config["case_suffix"] = self.case_suffix_entry.get()

        if config["runtype"] == "site":
            config["sites"] = [self.site_menu.get(i) for i in self.site_menu.curselection()]
            config["sitegroup"] = self.sitegroup_var.get()
            config["numproc"] = 1
        elif config["runtype"] == "latlon_bbox":
            config["lat_bounds"] = [self.lat_min_entry.get(), self.lat_max_entry.get()]
            config["lon_bounds"] = [self.lon_min_entry.get(), self.lon_max_entry.get()]
            config["numproc"] = int(self.numproc_entry.get())
            config["name"] = self.region_var.get()
        else:
            config["numproc"] = 1

        config["res"] = self.resolution_var.get()
        config["use_cpl_bypass"] = self.use_cpl_bypass_var.get()
        config["use_SP"] = self.use_sp_var.get()
        config["use_fates"] = self.use_fates_var.get()
        if config["use_fates"]:
            config["fates_nutrient"] = self.fates_nutrient_var.get()
            config["fates_pft"] = 1 #self.fates_pft_var.get()


        if config["use_SP"]:
            config["nutrients"] = 'none'
            config["nyears"] = self.sp_years_entry.get()
            config["startyear"] = self.sp_start_entry.get()
        else:
            config["nutrients"] = 'CNP'
            config["nutrient_comp"] = 'RD'
            config["soil_decomp"] = 'CTC'
            config["nyears_ad"] = self.nyears_ad_entry.get()
            config["nyears_final"] = self.nyears_final_entry.get()
            config["nyears_trans"] = self.nyears_trans_entry.get()

        # Parse case_options as a dictionary
        case_options_text = self.case_options_text.get("1.0", tk.END).strip()
        case_options_dict = {}
        for line in case_options_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Support both '=' and ':' as separators
            if '=' in line:
                key, value = line.split('=', 1)
            elif ':' in line:
                key, value = line.split(':', 1)
            else:
                continue
            case_options_dict[key.strip()] = value.strip()
        config["case_options"] = case_options_dict
        config["metdir"] = self.metdir_entry.get() if self.use_cpl_bypass_var.get() else ""
        
        # Ensemble options
        config["parm_list"] = self.parm_list_entry.get()
        config["nsamples"] = self.nsamples_entry.get()
        config["np_ensemble"] = self.np_ensemble_entry.get()
        config["ensemble_file"] = self.ensemble_file_entry.get()
        config["postproc_variables"] = self.postproc_vars_text.get("1.0", tk.END).strip()
        config["postproc_startyear"] = self.postproc_start_entry.get()
        config["postproc_endyear"] = self.postproc_end_entry.get()
        config["postproc_frequency"] = self.postproc_freq_var.get()

        self.result = config
        # Save config as a pickle file
        #local_path = "./temp/config.pkl"
        #with open(local_path, "wb") as f:
        #    pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)
        #Save .cfg file
        local_path = "./config_files/config_gui.cfg"
        self.save_cfg_file(config, local_path)


        # If logged in to a remote host, send the file
        if self.sftp is not None and self.ssh is not None:
            # Choose a remote path, e.g., home directory
            remote_path = os.path.join(self.remote_olmt_entry.get(), "config_files/config_gui.cfg")
            try:
                self.sftp.put(local_path, remote_path)
                messagebox.showinfo("Configuration Saved", f"Configuration saved and sent to remote: {remote_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to send config to remote:\n{e}")
        else:
            messagebox.showinfo("Configuration Saved", "Configuration saved successfully!")
    
    def save_cfg_file(self, config, filename="./temp/config.cfg"):
        def write_section(f, section, data):
            filtered = {k: v for k, v in data.items() if v not in [None, '', [], {}]}
            if section == "case_options":
                filtered = {k: v for k, v in filtered.items() if k != "variable"}
            if not filtered:
                return
            f.write(f"[{section}]\n")
            for key, value in filtered.items():
                if isinstance(value, list):
                    value = ', '.join(str(v) for v in value)
                f.write(f"{key.replace('postproc_','')} = {value}\n")
            f.write("\n")

        with open(filename, "w") as f:
            # Machine section
            machine_keys = ["machine", "modelroot", "exeroot", "queue", "project", "inputdata", "caseroot", "runroot"]
            machine_section = {k: config[k] for k in machine_keys if k in config}
            write_section(f, "machine", machine_section)

            # Simulation section
            sim_keys = ["runtype", "case_suffix", "sites", "sitegroup", "res", "mettype", "use_cpl_bypass", "lat_bounds", \
                        "lon_bounds", "numproc", "metdir", "name", "offline_driver"]
            sim_section = {k: config[k] for k in sim_keys if k in config}
            write_section(f, "simulation", sim_section)

            # Biogeochemistry section
            bgc_keys = ["nutrients", "nutrient_comp", "soil_decomp", "use_fates", "fates_pft", "pft_duplicates", "use_crop"]
            bgc_section = {k: config[k] for k in bgc_keys if k in config}
            write_section(f, "biogeochemistry", bgc_section)

            # Run lengths section
            runlen_keys = ["nyears_ad", "nyears_final", "nyears_trans", "trans_startyear", "nyears", "startyear"]
            runlen_section = {k: config[k] for k in runlen_keys if k in config}
            write_section(f, "run_lengths", runlen_section)

            # Case options section
            if "case_options" in config:
                write_section(f, "case_options", config["case_options"])

            # Ensemble section
            ensemble_keys = ["parm_list", "nsamples", "np_ensemble", "ensemble_file"]
            parm_list = config.get("parm_list", "")
            ensemble_section = {}
            if parm_list:
                ensemble_section = {k: config[k] for k in ensemble_keys if k in config and config[k] not in [None, '', [], {}]}
            elif "parm_list" in config and config["parm_list"]:
                ensemble_section["parm_list"] = config["parm_list"]
            write_section(f, "ensemble", ensemble_section)

            # Postprocessing section
            postproc_keys = ["postproc_variables", "postproc_startyear", "postproc_endyear", "postproc_frequency"]
            postproc_section = {k: config[k] for k in postproc_keys if k in config}
            if (parm_list):
                write_section(f, "postprocessing", postproc_section)

            # Observations section (if present)
            if "observations" in config:
                write_section(f, "observations", config["observations"])


    def on_close(self):
        # In case the user closes the window via the window manager,
        # we set result to an empty dictionary (or you could set it to None).
        if self.sftp:
            self.sftp.close()
        if self.ssh:
            self.ssh.close()
        self.result = {}
        self.destroy()

    def on_resolution_selected(self, event=None):
        resolution = self.resolution_var.get()
        if resolution == "TESSFA2_4km":
            # Hide region selection
            self.region_label.grid_remove()
            self.region_menu.grid_remove()
            inputdata = self.inputdata_entry.get()+'/lnd/clm2/surfdata_map'
            inputdata_local = inputdata+'/lnd/clm2/surfdata_map'
            if (self.machine == 'docker'):
                inputdata = self.inputdata_entry.get()+'/lnd/clm2/surfdata_map'
                inputdata_local = os.path.join(os.environ['HOME'], 'models/inputdata/lnd/clm2/surfdata_map')
            # Set custom surface and domain files (update paths as needed)
            surffile = "SEBOX1_surfdata.TES_SE.4km.1d.NLCD.c250202.nc"
            domainfile = "SEBOX1_domain.lnd.TES_SE.4km.1d.c250201.nc"
            # Optionally update case_options
            case_options_text = self.case_options_text.get("1.0", tk.END)
            if "surffile_global" not in case_options_text:
                self.case_options_text.insert(tk.END, f"\nsurffile_global: {inputdata}/{surffile}")
            if "domainfile_global" not in case_options_text:
                self.case_options_text.insert(tk.END, f"\ndomainfile_global: {inputdata}/{domainfile}")
            # Read domain file for bounds
            try:
                import netCDF4
                ds = netCDF4.Dataset(f"{inputdata_local}/{domainfile}")
                lats = ds.variables['yc'][:]
                lons = ds.variables['xc'][:]
                lat_min, lat_max = float(lats.min()), float(lats.max())
                lon_min, lon_max = float(lons.min()), float(lons.max())
                ds.close()
                self.lat_min_entry.delete(0, tk.END)
                self.lat_min_entry.insert(0, str(lat_min))
                self.lat_max_entry.delete(0, tk.END)
                self.lat_max_entry.insert(0, str(lat_max))
                self.lon_min_entry.delete(0, tk.END)
                self.lon_min_entry.insert(0, str(lon_min))
                self.lon_max_entry.delete(0, tk.END)
                self.lon_max_entry.insert(0, str(lon_max))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read domain file:\n{e}", parent=self)
        else:
            # Restore region selection
            self.region_label.grid()
            self.region_menu.grid()
            
    def on_region_selected(self, event=None):
        region = self.region_var.get()
        
        if region in self.REGION_BOUNDS:
            lon_min, lon_max, lat_min, lat_max = self.REGION_BOUNDS[region]
            self.lat_min_entry.delete(0, tk.END)
            self.lat_min_entry.insert(0, str(lat_min))
            self.lat_max_entry.delete(0, tk.END)
            self.lat_max_entry.insert(0, str(lat_max))
            self.lon_min_entry.delete(0, tk.END)
            self.lon_min_entry.insert(0, str(lon_min))
            self.lon_max_entry.delete(0, tk.END)
            self.lon_max_entry.insert(0, str(lon_max))

    def update_metdir_visibility(self, frame=None):
        show = self.use_cpl_bypass_var.get()
        if show:
            self.metdir_label.grid()
            self.metdir_entry.grid()
            self.browse_metdir.grid()
        else:
            self.metdir_label.grid_remove()
            self.metdir_entry.grid_remove()
            self.browse_metdir.grid_remove()

    def remote_login(self):
        if self.logged_in:
            self.remote_logout()
            return

        import paramiko
        username = simpledialog.askstring("Remote Username", "Enter your remote username:", parent=self)
        if username is None:
            return
        password = simpledialog.askstring("Remote Password", "Enter your remote password:", show="*", parent=self)
        if password is None:
            return
        try:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(self.hostname, username=username, password=password)
            self.sftp = self.ssh.open_sftp()
            self.logged_in = True
            self.login_btn.config(text="Logout")
            messagebox.showinfo("Success", f"Logged in to {self.hostname}", parent=self)
            self.remote_homedir = self.sftp.normalize('.')
            self.remote_olmt_var.set(os.path.join(self.remote_homedir, "elm-olmt"))
            self.remote_olmt_entry.config(state="normal")
            self.browse_remote_olmt_btn.config(state="normal")

            # --- Update modelroot entry on login ---
            self.modelroot_entry.delete(0, tk.END)
            self.modelroot_entry.insert(0, os.path.join(self.remote_homedir, 'models/E3SM'))

            # --- Update sitegroups from remote ---
            from OLMTutils import get_sitegroups
            remote_inputdata = self.inputdata_entry.get()
            if (self.machine == 'docker'):
                remote_inputdata = os.path.join(os.environ['HOME'], 'models/inputdata')
            self.sitegroups = get_sitegroups(remote_inputdata, sftp=self.sftp)
            print(self.sitegroups)

            if self.sitegroups:
                self.sitegroup_menu['values'] = self.sitegroups
                self.sitegroup_var.set(self.sitegroups[0])
                self.update_sitelist()
            else:
                messagebox.showwarning("No Sitegroups", "No sitegroups found in remote inputdata.", parent=self)

        except Exception as e:
            self.ssh = None
            self.sftp = None
            self.logged_in = False
            messagebox.showerror("Error", f"Failed to login:\n{e}", parent=self)

    def remote_logout(self):
        if self.sftp:
            self.sftp.close()
            self.sftp = None
        if self.ssh:
            self.ssh.close()
            self.ssh = None
        self.logged_in = False
        self.login_btn.config(text="Login")
        self.remote_olmt_var.set("")
        self.remote_olmt_entry.config(state="disabled")
        self.browse_remote_olmt_btn.config(state="disabled")
        messagebox.showinfo("Logged out", "Disconnected from remote host.", parent=self)

    def browse_local_directory(self, entry):
        directory = filedialog.askdirectory(initialdir=entry.get() or "/")
        if directory:
            entry.delete(0, tk.END)
            entry.insert(0, directory)

    def browse_remote_directory(self, entry):
        if not self.sftp:
            messagebox.showerror("Error", "Not logged in to remote machine.", parent=self)
            return
        class RemoteDirDialog(tk.Toplevel):
            def __init__(dialog_self, sftp, start_path="."):
                super().__init__(self)
                dialog_self.sftp = sftp
                dialog_self.title("Select Remote Directory")
                dialog_self.geometry("500x400")
                dialog_self.selected_dir = None

                dialog_self.path_var = tk.StringVar(value=start_path)
                path_frame = tk.Frame(dialog_self)
                path_frame.pack(fill="x", padx=10, pady=5)
                tk.Label(path_frame, text="Path:").pack(side="left")
                dialog_self.path_entry = tk.Entry(path_frame, textvariable=dialog_self.path_var, width=50)
                dialog_self.path_entry.pack(side="left", fill="x", expand=True)
                tk.Button(path_frame, text="Go", command=dialog_self.update_list).pack(side="left", padx=5)

                dialog_self.listbox = tk.Listbox(dialog_self, selectmode="browse")
                dialog_self.listbox.pack(fill="both", expand=True, padx=10, pady=5)
                dialog_self.listbox.bind("<Double-Button-1>", dialog_self.on_double_click)

                btn_frame = tk.Frame(dialog_self)
                btn_frame.pack(fill="x", padx=10, pady=5)
                tk.Button(btn_frame, text="Select", command=dialog_self.select_dir).pack(side="right")
                tk.Button(btn_frame, text="Cancel", command=dialog_self.destroy).pack(side="right", padx=5)

                dialog_self.update_list()

            def update_list(dialog_self):
                path = dialog_self.path_var.get()
                try:
                    items = dialog_self.sftp.listdir_attr(path)
                    dirs = [item.filename for item in items if str(item.longname).startswith('d')]
                    dirs.sort()
                    dialog_self.listbox.delete(0, tk.END)
                    if path not in ("/", ""):
                        dialog_self.listbox.insert(tk.END, "..")
                    for d in dirs:
                        dialog_self.listbox.insert(tk.END, d)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to list directory:\n{e}", parent=dialog_self)

            def on_double_click(dialog_self, event):
                selection = dialog_self.listbox.curselection()
                if not selection:
                    return
                selected = dialog_self.listbox.get(selection[0])
                path = dialog_self.path_var.get()
                if selected == "..":
                    # Go up one directory
                    new_path = os.path.dirname(path.rstrip("/"))
                    if not new_path:
                        new_path = "/"
                    dialog_self.path_var.set(new_path)
                else:
                    # Go into selected directory
                    new_path = os.path.join(path, selected)
                    dialog_self.path_var.set(new_path)
                dialog_self.update_list()

            def select_dir(dialog_self):
                entry.delete(0, tk.END)
                entry.insert(0, dialog_self.path_var.get())
                dialog_self.destroy()

        # Start at current value or home
        start_path = entry.get() or self.sftp.normalize('.')
        RemoteDirDialog(self.sftp, start_path)

    def smart_browse(self, entry):
        """
        Use remote browse if logged in, otherwise use local browse.
        """
        if self.sftp:
            self.browse_remote_directory(entry)
        else:
            if (self.machine_var.get() == "docker"):
                # For Docker, use local paths for inputdata and modelroot
                if ('inputdata' in entry.get()):
                    old_path = entry.get()
                    old_path = old_path.replace('/inputdata', os.path.join(os.environ['HOME'], 'models/inputdata'))
                    entry.delete(0, tk.END)
                    entry.insert(0, old_path)
                    self.browse_local_directory(entry)
                    new_path = entry.get()
                    entry.delete(0, tk.END)
                    entry.insert(0, new_path.replace(os.path.join(os.environ['HOME'], 'models/inputdata'), '/inputdata'))
                elif ('/code/' in entry.get()):
                    old_path = entry.get()
                    old_path = old_path.replace('/code', os.path.join(os.environ['HOME'], 'models'))
                    entry.delete(0, tk.END)
                    entry.insert(0, old_path)
                    self.browse_local_directory(entry)
                    new_path = entry.get()
                    entry.delete(0, tk.END)
                    entry.insert(0, new_path.replace(os.path.join(os.environ['HOME'], 'models'), '/code'))
                elif ('/output/' in entry.get()):
                    messagebox.showerror("Error", "Cannot modify paths in container output directory.", parent=self)
            else:
                self.browse_local_directory(entry)

    def on_machine_selected(self, event=None):
        # If logged in, log out and reset button
        if self.logged_in:
            self.remote_logout()

        self.machine = self.machine_var.get()
        # Get updated info for the selected machine
        self.machine, self.rootdir, self.inputdata, self.squeue, self.project, self.hostname = \
        get_machine_info(machine_name=self.machine)
        # Update the inputdata entry
        self.inputdata_entry.delete(0, tk.END)
        self.inputdata_entry.insert(0, self.inputdata)
        # Optionally update other fields if you want:
        self.caseroot_entry.delete(0, tk.END)
        self.caseroot_entry.insert(0, os.path.join(self.rootdir, "e3sm_cases")) 
        self.modelroot_entry.delete(0, tk.END)
        if (self.machine == 'docker'):
            self.modelroot_entry.insert(0, "/code/E3SM")
        else:
            self.modelroot_entry.insert(0, os.path.join(os.environ['HOME'], 'models/E3SM'))
        self.runroot_entry.delete(0, tk.END)
        self.runroot_entry.insert(0, os.path.join(self.rootdir, "e3sm_run"))    

        # --- Update metdir based on current mettype ---
        mettype = self.mettype_var.get()
        if mettype == "era5-daymet":
            subdir = "atm/datm7/Daymet_ERA5_TESSFA2/cpl_bypass_full"
        elif mettype == "gswp3":
            subdir = "atm/datm7/atm_forcing.datm7.GSWP3.0.5d.v2.c180716/cpl_bypass_full"
        elif mettype == "crujra":
            subdir = "atm/datm7/atm_forcing.CRUJRA_trendy_2025/cpl_bypass_full"
        else:
            subdir = "atm/datm7"
        self.metdir_entry.delete(0, tk.END)
        self.metdir_entry.insert(0, os.path.join(self.inputdata, subdir))

    def on_mettype_selected(self, event=None):
        mettype = self.mettype_var.get()
        # Example logic: set subdirectory based on mettype
        if mettype == "era5-daymet":
            subdir = "atm/datm7/Daymet_ERA5_TESSFA2/cpl_bypass_full"
        elif mettype == "gswp3":
            subdir = "atm/datm7/atm_forcing.datm7.GSWP3.0.5d.v2.c180716/cpl_bypass_full"
        elif mettype == "crujra":
            subdir = "atm/datm7/atm_forcing.CRUJRA_trendy_2025/cpl_bypass_full"
        else:
            subdir = "atm/datm7"
        self.metdir_entry.delete(0, tk.END)
        self.metdir_entry.insert(0, os.path.join(self.inputdata,subdir))

    def run_button_action(self):
        if self.ssh is not None:
            self.run_remote_gui()
        else:
            self.run_in_docker_container()

    def run_in_docker_container(self):
        def docker_task():
            container_name = "e3sm_gui"
            workdir = "/code/elm-olmt/runscripts"
            image_name = "elmv3"  # Change to your image name
            try:
                subprocess.run(["docker", "rm", "-f", container_name], check=False)
                subprocess.run(
                    ["docker", "run", "-d", "--name", container_name, "--hostname=docker", "--user", "modeluser",
                     "-v", "/Users/zdr/models:/code",
                     "-v", "/Users/zdr/models/inputdata:/inputdata",
                     "-v", "elmoutput:/output",
                     image_name, "tail", "-f", "/dev/null"],
                    check=True
                )
                subprocess.run(
                    ["docker", "exec", "-w", workdir, container_name, "python", "run_GUI.py"],
                    check=True
                )
            except Exception as e:
                print(f"Failed to start or exec in Docker container: {e}")

        threading.Thread(target=docker_task, daemon=True).start()

    def run_remote_gui(self):
        import paramiko
        import tkinter as tk
        from tkinter import scrolledtext, simpledialog, messagebox

        remote_olmt = self.remote_olmt_entry.get()
        # Use bash -l -c to ensure login shell and environment
        command = f"cd {remote_olmt}/runscripts && bash -l -c 'python -u run_GUI.py'"

        username = simpledialog.askstring("Remote Username", "Enter your remote username:", parent=self)
        if username is None:
            return
        password = simpledialog.askstring("Remote Password", "Enter your remote password:", show="*", parent=self)
        if password is None:
            return

        def remote_task(output_widget):
            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(self.hostname, username=username, password=password)
                # Request a PTY for live output
                stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
                while True:
                    line = stdout.readline()
                    if not line:
                        break
                    self.safe_after(0, lambda l=line: output_widget.insert(tk.END, l))
                    self.safe_after(0, output_widget.see, tk.END)
                err = stderr.read().decode()
                if err:
                    self.safe_after(0, lambda: output_widget.insert(tk.END, "\n[stderr]\n" + err))
                    self.safe_after(0, output_widget.see, tk.END)
                self.safe_after(0, lambda: output_widget.insert(tk.END, "\n[Process finished]\n"))
                self.safe_after(0, output_widget.see, tk.END)
                ssh.close()
            except Exception as e:
                self.safe_after(0, lambda: output_widget.insert(tk.END, f"\n[Error] {e}\n"))
                self.safe_after(0, output_widget.see, tk.END)

        # Create the output window
        win = tk.Toplevel(self)
        win.title("Remote Command Output")
        win.geometry("900x500")
        output_widget = scrolledtext.ScrolledText(win, width=110, height=30)
        output_widget.pack(fill="both", expand=True)
        output_widget.insert(tk.END, f"Running remote command:\n{command}\n\n")

        import threading
        threading.Thread(target=remote_task, args=(output_widget,), daemon=True).start()

    def is_ssh_alive(self):
        try:
            transport = self.ssh.get_transport() if self.ssh else None
            return transport and transport.is_active()
        except Exception:
            return False

    def safe_after(self, ms, func, *args, **kwargs):
        try:
            self.after(ms, func, *args, **kwargs)
        except RuntimeError:
            pass

def get_configuration():
    #Opens the configuration GUI, waits for the user to input values and click Save,
    #then returns a dictionary with the configuration.
    #app = E3SMConfigurator()
    ## Run the GUI modally. When the user clicks Save, the window is destroyed.
    #app.mainloop()
    #return app.result

    #Opens the configuration GUI as a modal dialog,
    #waits for the user to input values and click Save,
    #then returns a dictionary with the configuration.
    # Create a hidden root window

    root = tk.Tk()
    root.withdraw()
    
    # Create the configuration GUI as a Toplevel window
    gui = E3SMConfigurator(master=root)
    
    # Wait until the Toplevel window is closed
    gui.wait_window()
    
    # Get the configuration result
    config = gui.result
    

    #hostname = simpledialog.askstring("Remote Host", "Enter your remote hostname:")
    #if hostname:
    #    remote_path = simpledialog.askstring("Remote Path", "Enter remote path for config.pkl:", initialvalue="~/config.pkl", parent=root)
    #    if remote_path:
    #        send_config_to_remote_with_root("./temp/config.pkl", remote_path, hostname, root)
    #
    ## Destroy the hidden root and return the configuration
    root.destroy()

    return config

#Function to send the configuration file to a remote server using SSH and SFTP
def send_config_to_remote_with_root(local_path, remote_path, hostname, parent):
    import paramiko
    username = simpledialog.askstring("Remote Username", "Enter your remote username:", parent=parent)
    if username is None:
        return
    password = simpledialog.askstring("Remote Password", "Enter your remote password:", show="*", parent=parent)
    if password is None:
        return
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password)
        sftp = ssh.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()
        ssh.close()
        messagebox.showinfo("Success", f"File sent to {hostname}:{remote_path}", parent=parent)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to send file:\n{e}", parent=parent)


if __name__ == "__main__":
    config = get_configuration()
  
    # Save config as a pickle file
    #with open("./temp/config.pkl", "wb") as f:
    #    pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)

