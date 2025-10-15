import tkinter as tk
import time
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ssl
import cartopy.io.img_tiles as cimgt
ssl._create_default_https_context = ssl._create_unverified_context

class CartopyMapSelector:
    def __init__(self, callback, initial_bounds=None, sites=None):
        """
        Opens a Toplevel window with an interactive Cartopy map.
        The user can zoom, pan, and draw a rectangle.
        When the user clicks the "Confirm Selection" button, the callback is called
        with the stored selection: (min_lat, max_lat, min_lon, max_lon).
        Optionally, initial_bounds=(min_lat, max_lat, min_lon, max_lon) can be provided.
        """
        self.callback = callback
        self.selected_bounds = None  # Will store (min_lat, max_lat, min_lon, max_lon)
        self.top = tk.Toplevel()
        self.top.title("Cartopy Map Selector")
        self.last_draw_time = 0
        self.sites = sites or []
        self.selected_sites = set()  # Track indices of selected sites

        
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # If sites are provided, set extent to fit them (with a buffer)
        if self.sites:
            lats = [site["lat"] for site in self.sites]
            lons = [site["lon"] for site in self.sites]
            if lats and lons:
                lat_min, lat_max = min(lats), max(lats)
                lon_min, lon_max = min(lons), max(lons)
                # Add a buffer (e.g., 2 degrees or 10% of the range, whichever is larger)
                lat_buffer = max(2, 0.1 * (lat_max - lat_min))
                lon_buffer = max(2, 0.1 * (lon_max - lon_min))
                self.ax.set_extent([
                    lon_min - lon_buffer, lon_max + lon_buffer,
                    lat_min - lat_buffer, lat_max + lat_buffer
                ], crs=ccrs.PlateCarree())
            else:
                # Fallback to global if no valid lats/lons
                self.ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        else:
            # Default: whole globe
            self.ax.set_extent(initial_bounds if initial_bounds else [-180, 180, -90, 90], crs=ccrs.PlateCarree())
            #self.ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

        # Overlay Cartopy features.
        self.ax.add_feature(cfeature.LAND, facecolor='lightgray')
        self.ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        self.ax.add_feature(cfeature.COASTLINE)
        self.ax.add_feature(cfeature.BORDERS, linestyle=':')
        self.ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')  # US states
        self.ax.add_feature(cfeature.RIVERS, linewidth=0.7, edgecolor='blue')

        # Create a tile object for Stamen Terrain
        #terrain_tiles = cimgt.Stamen('terrain-background')
        # Add the tile layer to your axes; the second parameter is the zoom level
        #self.ax.add_image(terrain_tiles, 8)
        # Create gridlines with labels, etc.
        gl = self.ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

        # Set locators on the gridliner after it's created.
        gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))
        gl.ylocator = mticker.FixedLocator(range(-90, 91, 15))

        # Create the interactive RectangleSelector with customized rectangle properties.
        minspanx = 0.01 * abs(initial_bounds[1] - initial_bounds[0]) if initial_bounds else 5
        minspany = 0.01 * abs(initial_bounds[3] - initial_bounds[2]) if initial_bounds else 5
        rectprops = dict(facecolor='red', edgecolor='red', alpha=0.5, fill=True, zorder=1000)
        self.rectangle_selector = RectangleSelector(
            self.ax, self.onselect,
            useblit=False,          # Disable blitting for better compatibility.
            button=[1],             # Left mouse button only.
            minspanx=minspanx, minspany=minspany,
            spancoords='data',
            interactive=True,
            props=rectprops
        )

        # If initial_bounds are provided, draw the rectangle and set selected_bounds
        if initial_bounds is not None:
            min_lon, max_lon, min_lat, max_lat = initial_bounds
            # Draw rectangle: RectangleSelector expects (x1, y1, x2, y2)
            self.rectangle_selector.extents = (min_lon, max_lon, min_lat, max_lat)
            self.selected_bounds = (min_lon, max_lon, min_lat, max_lat)

        # Embed the Matplotlib figure in the Tkinter Toplevel window.
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add a navigation toolbar for zoom and pan functionality.
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.top)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Connect to the motion event to force canvas updates.
        self.canvas.mpl_connect("motion_notify_event", self.on_move)

        # Connect click event for site selection
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # Start checking the toolbar mode every 100ms.
        self.check_toolbar_mode()

        # Add a Confirm Selection button.
        confirm_btn = tk.Button(self.top, text="Confirm Selection", command=self.confirm_selection)
        confirm_btn.pack(side=tk.TOP, pady=5)

        # Plot the sites on the map if provided.
        self.site_marker_artists = []
        if self.sites:
            for idx, site in enumerate(self.sites):
                marker, = self.ax.plot(site["lon"], site["lat"], marker="o", color="red", markersize=6, transform=ccrs.PlateCarree(), picker=5)
                self.ax.text(site["lon"], site["lat"], site["name"], fontsize=8, color="black", transform=ccrs.PlateCarree(), ha="left", va="bottom")
                self.site_marker_artists.append(marker)

    def on_move(self, event):
       now = time.time()
       if (now - self.last_draw_time > 0.05):
         # Force the canvas to update (redraw) during mouse movement.
         self.canvas.draw_idle()
         self.last_draw_time = now

    def check_toolbar_mode(self):
         """
         Periodically check if the toolbar's mode is active.
         If the zoom (or pan) mode is active, disable the RectangleSelector.
         Otherwise, re-enable it.
         """
         if self.toolbar.mode != "":
             # A tool (like zoom or pan) is active.
             self.rectangle_selector.set_active(False)
         else:
             # No tool active; allow rectangle selection.
             self.rectangle_selector.set_active(True)
         # Check again after 100 milliseconds.
         self.top.after(100, self.check_toolbar_mode)

    def onselect(self, eclick, erelease):
        """Callback when a rectangle is drawn."""
        # Retrieve the coordinates of the press and release events.
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Sort the coordinates to determine bounds.
        min_lon, max_lon = sorted([x1, x2])
        min_lat, max_lat = sorted([y1, y2])
        self.selected_bounds = (min_lon, max_lon, min_lat, max_lat)
        print("Selection updated:", self.selected_bounds)
        # Keep the rectangle visible
        self.rectangle_selector.extents = (min_lon, max_lon, min_lat, max_lat)
        self.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        # Find the closest site within threshold
        min_dist = float("inf")
        selected_idx = None
        for idx, site in enumerate(self.sites):
            dx = event.xdata - site["lon"]
            dy = event.ydata - site["lat"]
            dist = dx*dx + dy*dy
            if dist < min_dist and dist < 1:  # 1 degree squared threshold
                min_dist = dist
                selected_idx = idx
        if selected_idx is not None:
            # Multi-select with Ctrl/Cmd, single select otherwise
            if (event.guiEvent.state & 0x0004) or (event.guiEvent.state & 0x0008):  # Ctrl or Cmd
                if selected_idx in self.selected_sites:
                    self.selected_sites.remove(selected_idx)
                else:
                    self.selected_sites.add(selected_idx)
            else:
                self.selected_sites = {selected_idx}
            self.update_site_highlights()

    def update_site_highlights(self):
        # Highlight selected sites in yellow, others in red
        for idx, marker in enumerate(self.site_marker_artists):
            if idx in self.selected_sites:
                marker.set_color("yellow")
                marker.set_markersize(10)
            else:
                marker.set_color("red")
                marker.set_markersize(6)
        self.canvas.draw_idle()

    def confirm_selection(self):
        """When the user confirms, call the callback with the stored bounds or selected site(s)."""
        if self.selected_sites:
            # User clicked sites (possibly multi-select)
            selected = [self.sites[idx] for idx in sorted(self.selected_sites)]
            print(selected)
            self.callback(selected)
        elif self.selected_bounds is not None and self.sites:
            # User drew a box and sites are present: return all sites in the box
            min_lon, max_lon, min_lat, max_lat = self.selected_bounds
            selected = []
            for site in self.sites:
                lat = site["lat"]
                lon = site["lon"]
                # Convert 0-360 to -180-180 for comparison
                if lon > 180:
                    lon = lon - 360
                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                    selected.append(site)
            print("Selected sites:", selected)
            self.callback(selected)
        elif self.selected_bounds is not None:
            # No sites: just return the bounds
            print("Selected bounds:", self.selected_bounds)
            self.callback(*self.selected_bounds)
        self.top.destroy()

# Example Tkinter application integrating the updated Cartopy map selector.
class ExampleApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cartopy Map Selector Example")

        # Create entry widgets to display the selected lat/lon bounds.
        tk.Label(self, text="Lat Min:").grid(row=0, column=0, padx=5, pady=5)
        self.lat_min_entry = tk.Entry(self)
        self.lat_min_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(self, text="Lat Max:").grid(row=1, column=0, padx=5, pady=5)
        self.lat_max_entry = tk.Entry(self)
        self.lat_max_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self, text="Lon Min:").grid(row=2, column=0, padx=5, pady=5)
        self.lon_min_entry = tk.Entry(self)
        self.lon_min_entry.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(self, text="Lon Max:").grid(row=3, column=0, padx=5, pady=5)
        self.lon_max_entry = tk.Entry(self)
        self.lon_max_entry.grid(row=3, column=1, padx=5, pady=5)

        # Button to open the Cartopy map selector.
        select_btn = tk.Button(self, text="Select on Map", command=self.open_map_selector)
        select_btn.grid(row=4, column=0, columnspan=2, pady=10)

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

        # Open the Cartopy map selector window; the callback updates the GUI entries.
        CartopyMapSelector(update_bounds)

if __name__ == "__main__":
      app = ExampleApp()
      app.mainloop()
