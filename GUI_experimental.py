import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import numpy as np
from OLMTutils import get_machine_info, get_site_info, get_sitegroups
import mapselect
import pickle

class E3SMConfigurator(tk.Toplevel):   #Tk):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("E3SM GUI Configurator")
        self.geometry("1200x900")
        self.result = None

        # When the window is closed via the window manager, call on_close
        self.protocol("WM_DELETE_WINDOW", self.on_close)        

        # Execute get_machine_info to get defaults
        self.machine, self.rootdir, self.inputdata = get_machine_info(machine_name='')
        self.sitegroups = get_sitegroups(self.inputdata)
        self.siteinfo = get_site_info(self.inputdata)
        caseroot = self.rootdir + '/e3sm_cases'
        runroot = self.rootdir + '/e3sm_run'
        inputdata = self.inputdata
        modelroot = os.environ.get('HOME', '') + '/models/E3SM'
        exeroot_default = ''   # default blank
        queue_default = 'batch'
        
        # Top Frame for machine and directory settings
        top_frame = ttk.Frame(self)
        top_frame.pack(fill='x', padx=10, pady=10)
        
        # Row 0: run root and Case Root

        tk.Label(top_frame, text="Case Root:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.caseroot_entry = tk.Entry(top_frame, width=45)
        self.caseroot_entry.insert(0, caseroot)
        self.caseroot_entry.grid(row=0, column=1, padx=5, pady=5)
        browse_caseroot = tk.Button(top_frame, text="Browse",
                            command=lambda: self.browse_directory(self.caseroot_entry))
        browse_caseroot.grid(row=0, column=2, padx=5, pady=5)

        tk.Label(top_frame, text="Run Root:").grid(row=0, column=3, sticky="w", padx=5, pady=5)
        self.runroot_entry = tk.Entry(top_frame, width=45)
        self.runroot_entry.insert(0, runroot)
        self.runroot_entry.grid(row=0, column=4, padx=5, pady=5)
        browse_runroot = tk.Button(top_frame, text="Browse",
                            command=lambda: self.browse_directory(self.runroot_entry))
        browse_runroot.grid(row=0, column=5, padx=5, pady=5)

        # Row 1: Exe Root and Model Root
        tk.Label(top_frame, text="Model Root:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.modelroot_entry = tk.Entry(top_frame, width=45)
        self.modelroot_entry.insert(0, modelroot)
        self.modelroot_entry.grid(row=1, column=1, padx=5, pady=5)
        browse_modelroot = tk.Button(top_frame, text="Browse",
                            command=lambda: self.browse_directory(self.modelroot_entry))
        browse_modelroot.grid(row=1, column=2, padx=5, pady=5)

        tk.Label(top_frame, text="Exeroot:").grid(row=1, column=3, sticky="w", padx=5, pady=5)
        self.exeroot_entry = tk.Entry(top_frame, width=45)
        self.exeroot_entry.insert(0, exeroot_default)
        self.exeroot_entry.grid(row=1, column=4, padx=5, pady=5)
        browse_exeroot = tk.Button(top_frame, text="Browse",
                            command=lambda: self.browse_directory(self.exeroot_entry))
        browse_exeroot.grid(row=1, column=5, padx=5, pady=5)


        #Row 2:  Input data
        tk.Label(top_frame, text="Input data:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.inputdata_entry = tk.Entry(top_frame, width=45)
        self.inputdata_entry.insert(0, inputdata)
        self.inputdata_entry.grid(row=2, column=1, padx=5, pady=5)
        browse_inputdata = tk.Button(top_frame, text="Browse",
                            command=lambda: self.browse_directory(self.inputdata_entry))
        browse_inputdata.grid(row=2, column=2, padx=5, pady=5)

        #Row 3:  queue and machine
        tk.Label(top_frame, text="Machine:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.machine_entry = tk.Entry(top_frame, width=45)
        self.machine_entry.insert(0, self.machine)
        self.machine_entry.grid(row=3, column=1, padx=5, pady=5)

        tk.Label(top_frame, text="Queue:").grid(row=3, column=3, sticky="w", padx=5, pady=5)
        self.queue_entry = tk.Entry(top_frame, width=45)
        self.queue_entry.insert(0, queue_default)
        self.queue_entry.grid(row=3, column=4, padx=5, pady=5)
        

        # Notebook for other options
        self.notebook = ttk.Notebook(self)
        self.req_frame = ttk.Frame(self.notebook)
        self.ens_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.req_frame, text="Required Inputs")
        self.notebook.add(self.ens_frame, text="Ensemble Options")
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        self.create_required_inputs(self.req_frame)
        self.create_ensemble_options(self.ens_frame)

        # Save/Print button at the bottom
        save_btn = tk.Button(self, text="Save Configuration", command=self.save_configuration)
        save_btn.pack(side=tk.BOTTOM, pady=10)

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
        self.sites_frame.grid(row=row, column=0, columnspan=4, sticky="w", padx=5, pady=5)
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
        
        self.latlon_frame.grid(row=row, column=0, columnspan=4, sticky="w", padx=5, pady=5)
        row += 1

        # --- Resolution ---
        tk.Label(parent, text="Resolution:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.resolution_var = tk.StringVar(value="r05_r05")
        resolution_options = ["r05_r05", "r10_r10", "r20_r20"]
        self.resolution_menu = ttk.Combobox(parent, textvariable=self.resolution_var, values=resolution_options, state="readonly")
        self.resolution_menu.grid(row=row, column=1, padx=5, pady=5)
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
        self.case_options_text.insert(tk.END, "{\n    'metdir': '/path/to/metdir',\n    # ...\n}")
        row += 1

        # Finally, update which runtype frame to show:
        self.update_runtype_fields()

    def open_map_selector(self):
        """Open the Cartopy map selector and update the lat/lon entries with the selection."""
        def update_bounds(min_lat, max_lat, min_lon, max_lon):
            self.lat_min_entry.delete(0, tk.END)
            self.lat_min_entry.insert(0, f"{min_lat:.2f}")
            self.lat_max_entry.delete(0, tk.END)
            self.lat_max_entry.insert(0, f"{max_lat:.2f}")
            self.lon_min_entry.delete(0, tk.END)
            self.lon_min_entry.insert(0, f"{min_lon:.2f}")
            self.lon_max_entry.delete(0, tk.END)
            self.lon_max_entry.insert(0, f"{max_lon:.2f}")

        # Open the Cartopy map selector window; the callback updates the GUI entries.
        mapselect.CartopyMapSelector(update_bounds)

    def update_runtype_fields(self, event=None):
        """
        Show/hide the site selection or lat/lon bounds based on runtype.
        """
        runtype = self.runtype_var.get()
        if runtype == "site":
            self.sites_frame.grid()
            self.latlon_frame.grid_remove()
            # Optionally, force the combobox to display the default by calling .set()
            self.update_sitelist()
            self.sitegroup_menu.set(self.sitegroups[0])
        elif runtype == "latlon_bbox":
            self.latlon_frame.grid()
            self.sites_frame.grid_remove()
        else:
            # For other run types, hide both (or extend as needed)
            self.sites_frame.grid_remove()
            self.latlon_frame.grid_remove()

    def update_sitelist(self, event=None):
        # Get the currently selected sitegroup from the sitegroup_menu
        selected_sitegroup = self.sitegroup_menu.get()
        if selected_sitegroup:
            # Call get_site_info to update the site information
            self.siteinfo = get_site_info(self.inputdata, sitegroup=selected_sitegroup)
        
            new_site_list = list(self.siteinfo.keys())
            self.site_menu.delete(0, tk.END)
            for option in new_site_list:
              self.site_menu.insert(tk.END, option)

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
        config["machine"] = self.machine_entry.get()
        config["caseroot"] = self.caseroot_entry.get()
        config["runroot"] = self.runroot_entry.get()
        config["modelroot"] = self.modelroot_entry.get()
        config["exeroot"] = self.exeroot_entry.get()
        config["queue"] = self.queue_entry.get()
        config["inputdata"] = self.inputdata_entry.get()

        # Required inputs
        config["runtype"] = self.runtype_var.get()
        config["mettype"] = self.mettype_var.get()
        config["case_suffix"] = self.case_suffix_entry.get()

        if config["runtype"] == "site":
            #config["site"] = self.site_var.get()
            config["site"] = [self.site_menu.get(i) for i in self.site_menu.curselection()]
            config["sitegroup"] = self.sitegroup_var.get()
        elif config["runtype"] == "latlon_bbox":
            config["lat_bounds"] = [self.lat_min_entry.get(), self.lat_max_entry.get()]
            config["lon_bounds"] = [self.lon_min_entry.get(), self.lon_max_entry.get()]

        config["resolution"] = self.resolution_var.get()
        config["use_cpl_bypass"] = self.use_cpl_bypass_var.get()
        config["use_SP"] = self.use_sp_var.get()
        config["use_fates"] = self.use_fates_var.get()
        config["fates_nutrient"] = self.fates_nutrient_var.get()

        if config["use_SP"]:
            config["sp_years"] = self.sp_years_entry.get()
            config["sp_start_year"] = self.sp_start_entry.get()
        else:
            config["nyears_ad"] = self.nyears_ad_entry.get()
            config["nyears_final"] = self.nyears_final_entry.get()
            config["nyears_trans"] = self.nyears_trans_entry.get()

        config["case_options"] = self.case_options_text.get("1.0", tk.END).strip()

        # Ensemble options
        ens = {}
        ens["parm_list"] = self.parm_list_entry.get()
        ens["nsamples"] = self.nsamples_entry.get()
        ens["np_ensemble"] = self.np_ensemble_entry.get()
        ens["ensemble_file"] = self.ensemble_file_entry.get()
        ens["postproc_vars"] = self.postproc_vars_text.get("1.0", tk.END).strip()
        ens["postproc_startyear"] = self.postproc_start_entry.get()
        ens["postproc_endyear"] = self.postproc_end_entry.get()
        ens["postproc_freq"] = self.postproc_freq_var.get()

        config["ensemble_options"] = ens

        self.result = config
        self.destroy()

        #messagebox.showinfo("Configuration Saved", "Configuration saved successfully!")

    def on_close(self):
        # In case the user closes the window via the window manager,
        # we set result to an empty dictionary (or you could set it to None).
        self.result = {}
        self.destroy()

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
    
    # Destroy the hidden root and return the configuration
    root.destroy()
    return config


if __name__ == "__main__":
    config = get_configuration()
  
    # Save config as a pickle file
    with open("./temp/config.pkl", "wb") as f:
        pickle.dump(config, f, protocol=pickle.HIGHEST_PROTOCOL)
