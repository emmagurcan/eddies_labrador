import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.io
import cmocean 
import sys
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import h5py
import numpy as np
from scipy.spatial import cKDTree
import gsw
from matplotlib.colors import Normalize, PowerNorm
from scipy import interpolate
from scipy.signal import find_peaks
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter

max_depth = 800
num_depth = 800

# Define water masses
water_masses = {
    "SAIW": (3.62, 0.43, 34.994, 0.057, 'purple'),
    "LSW": (3.24, 0.32, 35.044, 0.031, 'orange'),
    "ISOW": (3.02, 0.26, 35.098, 0.028, 'green'),
    "DSOW": (1.27, 0.29, 35.052, 0.016, 'blue'),
    "uNADW": (3.33, 0.31, 35.071, 0.027, 'cyan'),
    "lNADW": (2.96, 0.21, 35.083, 0.019, 'magenta')
}

def set_maxdepth(cruise):
    global max_depth
    global num_depth
    if cruise == "MSM74":
        max_depth = 800
        num_depth = 800
    else:
        max_depth = 300
        num_depth = 1001
        
def get_maxdepth():
    global max_depth
    return max_depth

def get_watermasses():
    global water_masses
    return water_masses

def get_numdepth():
    global num_depth
    return num_depth

def filter_points(lat_range, lon_range, latitudes, longitudes):
    """
    Filters points based on given latitude and longitude ranges.
    
    Parameters:
        lat_range (list): List defining the latitude range.
        lon_range (list): List defining the longitude range.
        latitudes (list): List of latitudes.
        longitudes (list): List of longitudes.
    
    Returns:
        list: Indices of points that lie within the specified latitude and longitude range.
    """
    min_lat = np.min(lat_range)
    max_lat = np.max(lat_range)
    min_lon = np.min(lon_range)
    max_lon = np.min(lon_range)
    indices = [
        i for i, (lat, lon) in enumerate(zip(latitudes, longitudes))
        if (min_lat <= lat <= max_lat) & (min_lon <= lon <= max_lon) 
    ]
    
    return indices

def section_MSM74(section_num):
    if section_num == 1:
        start_index = 0
        end_index = 26
        inv_x = 0
    if section_num == 2:
        start_index1 = 26
        end_index1 = 35
        start_index2 = 63
        end_index2 = 64
        start_index3 = 72
        end_index3 = 73
        start_index = (start_index1, start_index2, start_index3)
        end_index = (end_index1, end_index2, end_index3)
        inv_x = 0
    if section_num == 3:
        start_index = 35
        end_index = 43
        inv_x = 0
    if section_num == 4:
        start_index = 43
        end_index = 63
        inv_x = 1
    if section_num == 5:
        start_index = 63
        end_index = 72
        inv_x = 1
    if section_num == 6:
        start_index = 73
        end_index = 84
        inv_x = 0 # manually invert
    if section_num == 7:
        start_index = 84
        end_index = -1
        inv_x = 0
    return start_index, end_index, inv_x

def determine_sigma_multip(S, T, press, lon, lat):
    """
    Compute sigma0 (potential density anomaly) given salinity, temperature, and pressure.
    Supports cases where lon/lat are scalars, 1D arrays, or multiple profile points.

    Parameters:
    - S: 2D array (depth x profiles) or 1D array (depth) - Salinity
    - T: 2D array (depth x profiles) or 1D array (depth) - Temperature
    - press: 1D array (pressure levels)
    - lon: scalar, 1D array, or multiple profile points
    - lat: scalar, 1D array, or multiple profile points

    Returns:
    - sigma0: 1D array (depth) (averaged if multiple profiles)
    """

    # Ensure lon & lat are numpy arrays (handle scalar cases)
    lon = np.atleast_1d(lon)
    lat = np.atleast_1d(lat)

    # Check if we have multiple reference profiles
    multiple_profiles = S.ndim == 2 and len(lon) > 1

    if multiple_profiles:
        # Compute Absolute Salinity (SA) for each profile
        SA = np.array([
            gsw.SA_from_SP(S[:, i], press, lon[i], lat[i]) for i in range(len(lon))
        ]).T  # Transpose to match (depth, profiles) shape

        # Compute Conservative Temperature (CT)
        CT = np.array([
            gsw.CT_from_t(SA[:, i], T[:, i], press) for i in range(len(lon))
        ]).T  

        # Compute Potential Density Anomaly (sigma0)
        sigma0_all = np.array([
            gsw.sigma0(SA[:, i], CT[:, i]) for i in range(len(lon))
        ]).T  

        # Average over all profiles
        sigma0 = np.nanmean(sigma0_all, axis=1)
    
    else:
        # If lon/lat are scalars or we only have one profile, process normally
        SA = gsw.SA_from_SP(S, press, lon[0], lat[0])
        CT = gsw.CT_from_t(SA, T, press)
        sigma0 = gsw.sigma0(SA, CT)

    return sigma0

def determine_sigma(S, T, press, lon, lat):
    # Convert practical salinity to absolute salinity
    SA = gsw.SA_from_SP(S, press, lon, lat)  # Longitude and Latitude needed

    # Convert in situ temperature to conservative temperature
    CT = gsw.CT_from_t(SA, T, press)

    # Calculate in situ density (kg/m³)
    rho = gsw.rho(SA, CT, press)

    # Calculate potential density anomaly referenced to 0 dbar
    sigma0 = gsw.sigma0(SA, CT)  # Potential density anomaly (kg/m³ - 1000)

    return sigma0

def interpol(distance, depth, sigma0):
    nan_indices = np.isnan(sigma0)
    depth_grid = depth 
    # Remove NaNs for interpolation
    valid_points = np.column_stack((distance[~nan_indices], depth[~nan_indices]))
    valid_values = sigma0[~nan_indices]

    # Grid points for interpolation
    grid_distance, grid_depth = np.meshgrid(distance, depth_grid)
    # grid_distance, grid_depth = np.meshgrid(np.unique(distance), depth_grid)
    interp_points = np.column_stack((grid_distance.ravel(), grid_depth.ravel()))

    # Interpolate using nearest-neighbor
    filled_PD = griddata(valid_points, valid_values, interp_points, method='nearest').reshape(grid_depth.shape)
    return grid_distance, grid_depth, filled_PD

def plot_coords(lons, lats, title='Scatter Plot of MSM74', extent=[-70, -25, 50, 70], ax=None):
    valid_indices = ~np.isnan(lats) & ~np.isnan(lons)
    lats_clean = lats[valid_indices]
    lons_clean = lons[valid_indices]

    # Create a map using Cartopy with a PlateCarree projection
    if ax == None:
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set map boundaries (focus on the Labrador Sea)
    ax.set_extent(extent, crs=ccrs.PlateCarree())  # Lon min, Lon max, Lat min, Lat max

    # Add coastlines and gridlines for reference
    ax.coastlines(resolution='50m')
    ax.gridlines(draw_labels=True)

    # Plot the scatter points using the transform argument
    scatter = ax.scatter(lons_clean, lats_clean, color='red', s=10, edgecolor='black', alpha=0.7, transform=ccrs.PlateCarree())

    # Add title
    ax.set_title(title)

    # Show the plot
    plt.show()

def contour_var(lon, depth, var, PD, title, cbar_label, ax=None, colormap = 'coolwarm', x_invt=0, y_invt=1):
    mask = (~np.isnan(lon)) & (~np.isnan(depth)) & (~np.isnan(var)) & (~np.isnan(PD)) & \
           (~np.isinf(lon)) & (~np.isinf(depth)) & (~np.isinf(var)) & (~np.isinf(PD))
    lons_clean = lon[mask]
    depths_clean = depth[mask]
    var_clean = var[mask]
    PD_clean = PD[mask]

    # Check for consistent lengths
    assert len(lons_clean) == len(depths_clean) == len(var_clean) == len(PD_clean), "Mismatch in array lengths!"

    # Plot setup
    if ax is None:
        ax = plt.gca()
    
    # Filled contour plot (tricontourf)
    contour = ax.tricontourf(lons_clean, depths_clean, var_clean, levels=40, cmap=colormap)

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(cbar_label)
    # Set colorbar ticks with 1 decimal point
    cbar.formatter = FormatStrFormatter('%.1f')
    cbar.update_ticks()

    # Contour lines for PD
    contour_levels = np.arange(23, 28, 0.1)
    clines = ax.tricontour(lons_clean, depths_clean, PD_clean, levels=contour_levels, colors='k', linewidths=0.8)
    ax.clabel(clines, inline=True, fmt='%1.1f', fontsize=11)
    
    # Labels and formatting
    ax.set_xlabel("Distance [km]")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)

    # Invert y-axis if depth increases downward
    if y_invt == 1:
        ax.invert_yaxis()
    if x_invt:
        ax.invert_xaxis()

def get_dists(distance, depth, section_num, cruise="MSM74"):
    if section_num == 5 and cruise == "MSM74":
        indices = find_increasing_intervals(depth)
        len_section = len(indices)
        return distance[[idx for idx, idy in indices]]
    else:
        return np.unique(distance)
  
def plot_section_data(distance, depth, distance_adcp, depth_adcp, temperature, salinity, sigma0, v_ortho, section_num, inv_x=0, depth_max_adcp=300, depth_min_adcp=30):
    """
    Create a 2x2 subplot showing temperature, salinity, potential density, and orthogonal velocity.

    Parameters:
    distance (numpy array): Distance array (km).
    depth (numpy array): Depth array (m).
    temperature (numpy array): Temperature data for the section.
    salinity (numpy array): Salinity data for the section.
    sigma0 (numpy array): Potential density data for the section.
    v_ortho (numpy array): Orthogonal velocity data for the section.
    section_num (int): Section number for the title.
    inv_x (bool): Flag to invert x-axis.
    depth_max_adcp (float): Maximum depth for the velocity plot (default: 300).
    """
    
    # Create the figure with 4 subplots (2 rows, 2 columns)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.suptitle(f"MSM74 Section {section_num}")
    axs[0, 0].set_title(f"Temperature")
    contour_var(distance, depth, temperature, sigma0, f"Temperature", "Temperature (°C)", ax=axs[0, 0], colormap='cmo.thermal', x_invt=inv_x)
    
    # mask = (salinity > 30) 
    axs[0, 1].set_title(f"Salinity")
    # contour_var(distance_f, depth_f, salinity, sigma0_, f"Salinity", "Salinity (psu)", ax=axs[0, 1], colormap='cmo.haline', x_invt=inv_x)
    contour_var(distance, depth, salinity, sigma0, f"Salinity", "Salinity (psu)", ax=axs[0, 1], colormap='cmo.haline', x_invt=inv_x)

    axs[1, 0].set_title(f"Potential Density")
    contour_var(distance, depth, sigma0, sigma0, f"Potential Density", "Potential Density (kg/m³)", ax=axs[1, 0], colormap='cmo.dense', x_invt=inv_x)
    
    axs[1, 1].set_title(f"Orthogonal Velocity")
    distance_grid, depth_grid = np.meshgrid(distance_adcp, depth_adcp)
    levels = np.linspace(-0.35, 0.35, 65)  

    # Define a saturation factor (>1 increases saturation, <1 decreases it)
    V_plot = axs[1, 1].contourf(distance_grid, depth_grid, v_ortho, levels=levels, cmap="RdBu_r", norm=Normalize(vmin=-0.35, vmax=0.35), extend='both')

    # V_plot = axs[1, 1].contourf(distance_grid, depth_grid, v_ortho, levels=levels, cmap='RdBu_r', extend='both')
    cbar = plt.colorbar(V_plot, ax=axs[1, 1])
    cbar.set_label('Orthogonal Velocities (m/s)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    # Set colorbar ticks with 1 decimal point
    cbar.formatter = FormatStrFormatter('%.1f')
    cbar.update_ticks()

    contour_levels = np.arange(23, 28, 0.1)

    idxs = np.where(depth < depth_max_adcp)
    sigma0s = sigma0[idxs]
    depths = depth[idxs]
    distances = distance[idxs]

    mask = np.isfinite(sigma0s)
    sigma0s = sigma0s[mask]
    depths = depths[mask]
    distances = distances[mask]

    
    clines = axs[1, 1].tricontour(distances, depths, sigma0s,
                                levels=contour_levels, colors='k', linewidths=0.8)
    clines.clabel(inline=True, fmt='%1.1f', fontsize=11)



    firsts = get_dists(distance, depth, section_num)
    ys = np.ones_like(firsts)
    ys = ys * 8
    axs[0, 0].scatter(firsts, ys, color="r", s=5, zorder=10)
    axs[0, 1].scatter(firsts, ys, color="r", s=5, zorder=10)
    axs[1, 0].scatter(firsts, ys, color="r", s=5, zorder=10)

    axs[1, 1].set_ylim(depth_min_adcp, depth_max_adcp)
    axs[1, 1].set_xlabel('Distance (km)', fontsize=12)
    axs[1, 1].set_ylabel('Depth (m)', fontsize=12)
    axs[1, 1].invert_yaxis()

    if inv_x:
        axs[1, 1].invert_xaxis()

    
    # Tight layout for better spacing
    plt.tight_layout()
    plt.savefig(f'/Users/emmagurcan/Documents/France/ENS/M1/stageM1/analysis/plots/MSM74/section_{section_num}')
    # print(f"Saved figure to plots/section_{section_num} ")
    plt.show()

def plot_sectionMSM40(distance, depth, distance_adcp, depth_adcp, PT, S, PD, v_ortho, section_num, inv_x=0, depth_max_adcp=300, depth_max=300):
    # Plot the figure
    fig, ax = plt.subplots(2, 2, figsize=(20, 16))
    plt.suptitle(f"MSM40 Section {section_num}")
    # Plot Potential Temperature
    PT_plot = ax[0, 0].contourf(distance, depth, PT,levels=20, cmap='cmo.thermal',extend='both')
    fig.colorbar(PT_plot, ax=ax[0, 0], label='Potential Temperature (°C)')
    ax[0, 0].set_ylim(0,depth_max)
    ax[0, 0].set_xlabel('Distance (km)')
    ax[0, 0].set_ylabel('Depth (m)')
    ax[0, 0].set_title('Potential Temperature (°C)')
    ax[0, 0].invert_yaxis()  

    # Plot Potential Density
    PD_plot = ax[1, 0].contourf(distance, depth, PD, levels=20, cmap='cmo.dense',extend='both')
    fig.colorbar(PD_plot, ax=ax[1, 0], label='Potential Density (kg/m3)')
    ax[1, 0].set_ylim(0,depth_max)
    ax[1, 0].set_xlabel('Distance (km)')
    ax[1, 0].set_ylabel('Depth (m)')
    ax[1, 0].set_title('Potential Density')
    ax[1, 0].invert_yaxis()  

    # Plot Salinity
    S_plot = ax[0,1].contourf(distance, depth, S,levels=20, cmap='cmo.haline',extend='both')
    fig.colorbar(S_plot, ax=ax[0,1], label='Salinity (g/kg)')
    ax[0,1].set_ylim(0,depth_max)
    ax[0,1].set_xlabel('Distance (km)')
    ax[0,1].set_ylabel('Depth (m)')
    ax[0,1].set_title('Salinity')
    ax[0,1].invert_yaxis() 

    # Plot Velocity
    levels_velocity=np.linspace(-0.4, 0.4, 100)
    V_plot = ax[1, 1].contourf(distance_adcp, depth_adcp, v_ortho,levels=levels_velocity, cmap='RdBu_r',extend='both')
    fig.colorbar(V_plot, ax=ax[1, 1], label='Velocity (m/s)')
    ax[1, 1].set_ylim(0,depth_max_adcp)
    ax[1, 1].set_xlabel('Distance (km)')
    ax[1, 1].set_ylabel('Depth (m)')
    ax[1, 1].set_title('Velocities')
    ax[1, 1].invert_yaxis() 

    firsts = get_dists(distance, depth, section_num, cruise="MSM40")
    ys = np.ones_like(firsts)
    ys = ys * 8
    ax[0, 0].scatter(firsts, ys, color="r", s=5, zorder=10)
    ax[0, 1].scatter(firsts, ys, color="r", s=5, zorder=10)
    ax[1, 0].scatter(firsts, ys, color="r", s=5, zorder=10)

    # Add isopycnes contours
    contour_levels = np.arange(23, 28, 0.1)

    contour_lines = ax[0, 0].contour(distance, depth, PD, levels=contour_levels, colors='white', linewidths=0.8)
    ax[0, 0].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=11)
    contour_lines = ax[1, 0].contour(distance, depth, PD, levels=contour_levels, colors='white', linewidths=0.8)
    ax[0, 1].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=11)
    contour_lines = ax[0, 1].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.8)
    ax[1, 0].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=11)
    contour_lines = ax[1, 1].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.8)
    ax[1, 1].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'/Users/emmagurcan/Documents/France/ENS/M1/stageM1/analysis/plots/MSM40/section_{section_num}')

def clean_data(dist, depth, section, *vars):
    max_depth = 1000
    idx = np.where(depth > max_depth)[0]  # Get indices where depth > 400
    
    # Apply filtering
    dist, depth = dist[idx], depth[idx]
    vars = [var[idx] for var in vars]  # Apply filtering to all additional variables
    
    # Determine snip value
    snip = 500 if section == 1 else 0

    return (dist[snip:], depth[snip:]) + tuple(var[snip:] for var in vars)

def clean(ds_ctd, lonlatev):
    # Create a mask for lon and lat that matches the ones you want to remove
    lati = np.array(lonlatev['Latitude'][31:35])
    loni = np.array(lonlatev['Longitude'][31:35])

    # Create the 'events' array as given
    exclude_events = np.concatenate([
        np.array(lonlatev['Event'][31:35]),
        np.array(lonlatev['Event'][39:40]),
        np.array(lonlatev['Event'][48:50]),
        np.array(lonlatev['Event'][50:51]),
        np.array(lonlatev['Event'][80:81]),
        np.array(lonlatev['Event'][106:109])
    ])

    # Create a mask where the 'Event' in ds_ctd does not match any in 'events'
    mask = ~np.isin(ds_ctd['Event'], exclude_events)


    press = np.array(ds_ctd['Press [dbar]'][:])
    depth = np.array(ds_ctd['Depth water [m]'][:])
    T = np.array(ds_ctd['Temp [C]'][:])
    S = np.array(ds_ctd['Sal'][:])
    lon = np.array(ds_ctd['Longitude'][:])
    lat = np.array(ds_ctd['Latitude'][:])
    evs = np.array(ds_ctd['Event'][:])
    ts = np.array(ds_ctd['Time'][:])
    o2 = np.array(ds_ctd['O2 [umol_kg]'][:])
    sv = np.array(ds_ctd['SV [m_s]'][:])
    sig = np.array(ds_ctd['Sigma in situ [kg_m**3]'][:])
    fluo = np.array(ds_ctd['Fluorescence [arbitrary units]'][:])
    turb = np.array(ds_ctd['Turbidity [NTU]'][:])
    att = np.array(ds_ctd['Attenuation [1_m]'][:])

    press_f = press[mask]
    depth_f = depth[mask]
    T_f = T[mask]
    S_f = S[mask]
    lon_f = lon[mask]
    lat_f = lat[mask]
    evs_f = evs[mask]
    ts_f = ts[mask]
    o2_f = o2[mask]
    sv_f = sv[mask]
    sig_f = sig[mask]
    fluo_f = fluo[mask]
    turb_f = turb[mask]
    att_f = att[mask]

    new_ctd_file = 'filtered_ctd_data.nc'

    # Create a new NetCDF file
    with h5py.File(new_ctd_file, 'w') as f:
        # Create the datasets in the new file and write the filtered data
        f.create_dataset('Press [dbar]', data=press_f)
        f.create_dataset('Depth water [m]', data=depth_f)
        f.create_dataset('Temp [C]', data=T_f)
        f.create_dataset('Sal', data=S_f)
        f.create_dataset('Longitude', data=lon_f)
        f.create_dataset('Latitude', data=lat_f)
        f.create_dataset('Event', data=evs_f)
        f.create_dataset('Time', data=ts_f)
        f.create_dataset('O2 [umol_kg]', data=o2_f)
        f.create_dataset('SV [m_s]', data=sv_f)
        f.create_dataset('Sigma in situ [kg_m**3]', data=sig_f)
        f.create_dataset('Fluorescence [arbitrary units]', data=fluo_f)
        f.create_dataset('Turbidity [NTU]', data=turb_f)
        f.create_dataset('Attenuation [1_m]', data=att_f)

        print(f"New NetCDF file saved at {new_ctd_file}")

def swap_indices(sigma_grid):
    """
    Swap elements at indices 4/5 with 6/7 in sigma_grid and xs.
    
    Parameters:
    sigma_grid (np.array): 2D array of size (800, 8).
    xs (np.array): 1D array of size (8,).
    
    Returns:
    tuple: Updated sigma_grid and xs.
    """
    if len(sigma_grid.shape) == 2:
        sigma_grid[:, [4, 5, 6, 7]] = sigma_grid[:, [6, 7, 4, 5]]
    elif len(sigma_grid.shape) == 1:
        sigma_grid[[4, 5, 6, 7]] = sigma_grid[[6, 7, 4, 5]]
    
    return sigma_grid

def get_woa_ref(lon_ctd, lat_ctd, lon_woa, lat_woa, var, max_radius=10):
    """
    Selects the most central WOA reference profile relative to the CTD path.
    If the closest profile contains only missing values, expands the search radius.

    Parameters:
    - lon_ctd, lat_ctd: Arrays of CTD longitude and latitude points
    - lon_woa, lat_woa: 1D arrays of WOA reference longitudes and latitudes
    - var: 3D array (depth, lat, lon) of the WOA variable (e.g., temperature or salinity)
    - max_radius: Maximum number of grid points to search outward

    Returns:
    - selected_profile: 1D array of reference values at depth levels
    - selected_lon: Longitude of the selected WOA reference profile
    - selected_lat: Latitude of the selected WOA reference profile
    """
    missing_value = 9.96921e+36  # WOA missing value
    mean_lon, mean_lat = np.mean(lon_ctd), np.mean(lat_ctd)

    # Compute distances from all WOA points to the mean CTD location
    lon_grid, lat_grid = np.meshgrid(lon_woa, lat_woa)
    distances = np.sqrt((lon_grid - mean_lon) ** 2 + (lat_grid - mean_lat) ** 2)

    # Start with a small search radius and progressively expand
    best_profile, best_lat, best_lon = None, None, None

    for radius in range(1, max_radius + 1):
        # Find indices of the closest `radius^2` grid points
        sorted_indices = np.argsort(distances.ravel())[: radius**2]
        lat_indices, lon_indices = np.unravel_index(sorted_indices, distances.shape)

        for i in range(len(sorted_indices)):
            selected_lat_idx, selected_lon_idx = lat_indices[i], lon_indices[i]
            selected_profile = var[:, selected_lat_idx, selected_lon_idx]

            # If this profile is not completely missing, return it immediately
            if not np.any(selected_profile == missing_value):
                return selected_profile, lon_woa[selected_lon_idx], lat_woa[selected_lat_idx]

            # Keep track of the least bad option
            if best_profile is None:
                best_profile = selected_profile
                best_lat, best_lon = lat_woa[selected_lat_idx], lon_woa[selected_lon_idx]

    # If no valid profile is found, return the closest one anyway
    print("Warning: All WOA profiles contain missing values. Returning the closest one anyway.")
    return best_profile, best_lon, best_lat

def get_alternate_woa_ref(lon_ctd, lat_ctd, lon_woa, lat_woa, var, 
                           best_lons, best_lats, max_radius=10):
    """
    Selects a WOA reference profile different from given reference profiles.
    If the closest profile contains only missing values, expands the search radius.

    Parameters:
    - lon_ctd, lat_ctd: Arrays of CTD longitude and latitude points
    - lon_woa, lat_woa: 1D arrays of WOA reference longitudes and latitudes
    - var: 3D array (depth, lat, lon) of the WOA variable (e.g., temperature or salinity)
    - best_lons, best_lats: Arrays of longitudes and latitudes of reference profiles to exclude
    - max_radius: Maximum number of grid points to search outward

    Returns:
    - selected_profile: 1D array of reference values at depth levels
    - selected_lon: Longitude of the selected WOA reference profile
    - selected_lat: Latitude of the selected WOA reference profile
    """
    missing_value = 9.96921e+36  # WOA missing value
    mean_lon, mean_lat = np.mean(lon_ctd), np.mean(lat_ctd)

    # Compute distances from all WOA points to the mean CTD location
    lon_grid, lat_grid = np.meshgrid(lon_woa, lat_woa)
    distances = np.sqrt((lon_grid - mean_lon) ** 2 + (lat_grid - mean_lat) ** 2)
    
    # Exclude the given best profiles
    for best_lon, best_lat in zip(best_lons, best_lats):
        exclude_mask = (lon_grid == best_lon) & (lat_grid == best_lat)
        distances[exclude_mask] = np.inf

    # Start with a small search radius and progressively expand
    alternative_profile, alternative_lat, alternative_lon = None, None, None

    for radius in range(1, max_radius + 1):
        # Find indices of the closest `radius^2` grid points, excluding the given references
        sorted_indices = np.argsort(distances.ravel())[: radius**2]
        lat_indices, lon_indices = np.unravel_index(sorted_indices, distances.shape)

        for i in range(len(sorted_indices)):
            selected_lat_idx, selected_lon_idx = lat_indices[i], lon_indices[i]
            selected_profile = var[:, selected_lat_idx, selected_lon_idx]
            
            # If this profile is not completely missing, return it immediately
            if not np.any(selected_profile == missing_value):
                return selected_profile, lon_woa[selected_lon_idx], lat_woa[selected_lat_idx]

            # Keep track of the least bad option
            if alternative_profile is None:
                alternative_profile = selected_profile
                alternative_lat, alternative_lon = lat_woa[selected_lat_idx], lon_woa[selected_lon_idx]
    
    # If no valid profile is found, return the closest one anyway
    print("Warning: All WOA profiles contain missing values. Returning the closest one anyway.")
    return alternative_profile, alternative_lon, alternative_lat

def get_woa_ref_snip(lon_ctd, lat_ctd, lon_woa, lat_woa, var, radius=0.5):
    """
    Selects WOA reference profiles within a specified radius relative to the CTD path.

    Parameters:
    - lon_ctd, lat_ctd: Arrays of CTD longitude and latitude points
    - lon_woa, lat_woa: 1D arrays of WOA reference longitudes and latitudes
    - var: 3D array (depth, lat, lon) of the WOA variable (e.g., temperature or salinity)
    - radius: The search radius (in degrees) for selecting matching profiles from the WOA dataset

    Returns:
    - selected_profiles: List of 1D arrays of reference values at depth levels for each selected profile
    - selected_lon: List of longitudes of the selected WOA reference profiles
    - selected_lat: List of latitudes of the selected WOA reference profiles
    - ctd_indices: List of indices into the CTD dataset
    - woa_indices: List of indices into the WOA dataset
    """
    # Compute the mean CTD section location
    mean_lon = np.mean(lon_ctd)
    mean_lat = np.mean(lat_ctd)

    # Find all potential matches for longitude and latitude
    lon_indices = np.where(np.abs(lon_woa - mean_lon) <= radius)[0]
    lat_indices = np.where(np.abs(lat_woa - mean_lat) <= radius)[0]

    # Generate all possible (lon, lat) index pairs
    lon_grid, lat_grid = np.meshgrid(lon_indices, lat_indices)
    index_pairs = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))  # Shape (N, 2)

    # Compute distances from each reference profile to the mean CTD location
    distances = [
        np.sqrt((lon_woa[lon] - mean_lon) ** 2 + (lat_woa[lat] - mean_lat) ** 2)
        for lat, lon in index_pairs
    ]

    # Identify the indices that fall within the specified radius
    within_radius_indices = [i for i, dist in enumerate(distances) if dist <= radius]

    # Initialize lists to store results
    selected_profiles = []
    selected_lon = []
    selected_lat = []
    ctd_indices = []
    woa_indices = []

    for idx in within_radius_indices:
        lat_idx, lon_idx = index_pairs[idx]

        # Extract the corresponding reference profile
        selected_profiles.append(var[:, lat_idx, lon_idx])
        selected_lon.append(lon_woa[lon_idx])
        selected_lat.append(lat_woa[lat_idx])
        ctd_indices.append((mean_lat, mean_lon))  # Store the mean CTD location for reference
        woa_indices.append((lat_idx, lon_idx))  # Store the indices in the WOA dataset

    return selected_profiles, selected_lon, selected_lat, ctd_indices, woa_indices


# Haversine function to compute distances (input: degrees, output: km)
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c  # Distance in km

def get_woa_ref_multiple(lon_ctd, lat_ctd, lon_woa, lat_woa, var, distance_threshold=10):
    matched_indices = []
    
    for lo, la in zip(lon_ctd, lat_ctd):
        lon_idx = np.argmin(np.abs(lon_woa - lo))
        lat_idx = np.argmin(np.abs(lat_woa - la))

        # Compute distance between the matched WOA point and the cruise point
        dist = haversine(lo, la, lon_woa[lon_idx], lat_woa[lat_idx])
        
        # Only keep matches within the distance threshold
        if dist <= distance_threshold:
            matched_indices.append((lon_idx, lat_idx))

    matched_indices = np.array(matched_indices)

    if matched_indices.size == 0:
        raise ValueError("No matching (lon, lat) profiles found within the distance threshold!")

    lon_indices = matched_indices[:, 0]
    lat_indices = matched_indices[:, 1]

    print(f"Selected {len(lon_indices)} matching profiles within {distance_threshold} km.")

    # Extract values only at matching (lat, lon) indices
    extracted_values = var[:, lat_indices, lon_indices]  

    # Compute the mean along the lat/lon dimensions
    mean_reference = np.nanmean(extracted_values, axis=1)  

    return mean_reference, lon_woa[lon_indices], lat_woa[lat_indices]

def find_increasing_intervals(arr):
    """
    Finds intervals where the array is strictly increasing.
    
    Parameters:
    arr (list or np.array): Input array.
    
    Returns:
    list of tuples: Each tuple represents the start and end indices (inclusive) of increasing subarrays.
    """
    arr = np.asarray(arr)
    diffs = np.diff(arr) > 0  # Find where elements are increasing
    
    intervals = []
    start = None
    
    for i, increasing in enumerate(diffs):
        if increasing:
            if start is None:
                start = i
        else:
            if start is not None:
                intervals.append((start, i))
                start = None
    
    if start is not None:
        intervals.append((start, len(arr) - 1))
    
    return intervals

def make_3D(distance, depth, var, section_num, cruise):
    mdepth = get_maxdepth()
    ndepth = get_numdepth()
    if section_num == 5 and cruise=="MSM74":
        print("Section 5, using increasing intervals")
        indices = find_increasing_intervals(depth)
        len_section = len(indices)

        depth_grid = np.full((ndepth, len_section), np.nan)

        dist_idx = 0
        for start, stop in indices:
            for idx in range(start, stop+1):
                d = int(depth[idx])
                depth_grid[d, dist_idx] = var[idx]
            dist_idx += 1
        return depth_grid
    
    else:
        len_section = len(np.unique(distance))
        depth_grid = np.full((ndepth, len_section), np.nan)

        prev_dist = distance[0]
        dist_idx = 0

        for i in range(len(var)):
            d = int(depth[i])
            curr_dist = distance[i]

            if curr_dist != prev_dist:
                dist_idx += 1
                prev_dist = curr_dist
                
            depth_grid[d, dist_idx] = var[i]
        return depth_grid

def make_depth_grid(distance):
    # len_section = len(np.unique(distance))
    len_section = len(distance)
    mdepth = get_maxdepth()
    ndepth = get_numdepth()
    depth = np.zeros((ndepth, len_section))
    for i in range(len_section):
        ind_depth = np.arange(0, 800, 1)
        depth[ind_depth, i] = -ind_depth
    return depth

def point_to_line_distance(point, line_start, line_end):
    """
    Calculate the perpendicular distance from a point to a line segment.

    Parameters:
    point (tuple): The (longitude, latitude) of the point.
    line_start (tuple): The (longitude, latitude) of the start of the line segment.
    line_end (tuple): The (longitude, latitude) of the end of the line segment.

    Returns:
    float: The distance from the point to the line segment.
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Vector from point to start of line
    dx0 = x0 - x1
    dy0 = y0 - y1
    
    # Vector from start to end of line
    dx1 = x2 - x1
    dy1 = y2 - y1

    # Project point onto the line,
    line_len_sq = dx1**2 + dy1**2
    if line_len_sq == 0:  # If the start and end are the same point
        return np.sqrt(dx0**2 + dy0**2)  # Distance to start

    t = max(0, min(1, (dx0 * dx1 + dy0 * dy1) / line_len_sq))

    # Nearest point on the segment
    nearest_x = x1 + t * dx1
    nearest_y = y1 + t * dy1

    # Return the distance from the point to the nearest point on the segment
    return np.sqrt((x0 - nearest_x)**2 + (y0 - nearest_y)**2)

def find_adcp_idx_from_ctd(loni, lati, lon_adcp, lat_adcp, max_distance=0.1):
    """
    Find all ADCP points that lie on the path defined by a CTD section (loni, lati).
    
    Parameters:
    loni (numpy array): Longitude array from the CTD dataset.
    lati (numpy array): Latitude array from the CTD dataset.
    lon_adcp (numpy array): Longitude array from the ADCP dataset.
    lat_adcp (numpy array): Latitude array from the ADCP dataset.
    max_distance (float): The maximum distance (in degrees) to consider a point "on the path".

    Returns:
    matched_adcp_indices (list): List of indices of the ADCP points that lie on the path.
    """
    matched_adcp_indices = []

    # Iterate over each segment between consecutive CTD points
    for i in range(len(loni) - 1):
        line_start = (loni[i], lati[i])
        line_end = (loni[i+1], lati[i+1])

        # Iterate over all ADCP points and check if they are close to the path segment
        for j, (lon, lat) in enumerate(zip(lon_adcp, lat_adcp)):
            distance = point_to_line_distance((lon, lat), line_start, line_end)
            if distance <= max_distance:
                matched_adcp_indices.append(j)

    # Remove duplicates (since ADCP points could be close to multiple segments)
    matched_adcp_indices = list(set(matched_adcp_indices))
    print(f"Found {len(matched_adcp_indices)} matches")

    return matched_adcp_indices


def find_adcp_idx_from_ctd2(loni, lati, adcp_lons, adcp_lats, max_distance=10):
    # Remove NaN and infinite values from the datasets
    valid_lonlat_mask = np.isfinite(loni) & np.isfinite(lati)
    loni = loni[valid_lonlat_mask]
    lati = lati[valid_lonlat_mask]

    matched_lons = []
    matched_lats = []
    matched_idx = []
    
    # Iterate over each point in the lonlat dataset
    for lon, lat in zip(loni, lati):
        # Calculate the distance between the lonlat point and all adcp points
        distances = np.sqrt((adcp_lons - lon)**2 + (adcp_lats - lat)**2)
        
        
        # Find the index of the closest point in the adcp dataset
        min_idx = np.argmin(distances)
        
        # Check if the closest point is within the max_distance threshold
        if distances[min_idx] <= max_distance:
            matched_lons.append(adcp_lons[min_idx])
            matched_lats.append(adcp_lats[min_idx])
            matched_idx.append(min_idx)
    
    # Convert to numpy arrays for convenience
    matched_lons = np.array(matched_lons)
    matched_lats = np.array(matched_lats)
    matched_idx = np.array(matched_idx)

    return matched_idx

def load_data(section_num, lonlatev, ds_ctd, ds_adcp):
    ctd_start, ctd_stop, inv_x = section_MSM74(section_num)
    if section_num == 2:
        s1, s2, s3 = ctd_start
        e1, e2, e3 = ctd_stop
        events = np.concatenate([
        np.array(lonlatev['Event'][s1:e1]),
        np.array(lonlatev['Event'][s2:e2]),
        np.array(lonlatev['Event'][s3:e3])])
        loni = np.concatenate([
        np.array(lonlatev['Longitude'][s1:e1]),
        np.array(lonlatev['Longitude'][s2:e2]),
        np.array(lonlatev['Longitude'][s3:e3])])
        lati = np.concatenate([
        np.array(lonlatev['Latitude'][s1:e1]),
        np.array(lonlatev['Latitude'][s2:e2]),
        np.array(lonlatev['Latitude'][s3:e3])])
    else:
        events = np.array(lonlatev['Event'][ctd_start:ctd_stop])
        loni = lonlatev["Longitude"][ctd_start:ctd_stop]
        lati = lonlatev["Latitude"][ctd_start:ctd_stop]

    events_ctd = np.array(ds_ctd['Event'][:])
    matching_indices = np.where(np.isin(events_ctd, events))[0]

    times = np.array(ds_ctd['Time'][:])[matching_indices]
    press = np.array(ds_ctd['Press [dbar]'][:])[matching_indices]
    depth = np.array(ds_ctd['Depth water [m]'][:])[matching_indices]
    T = np.array(ds_ctd['Temp [C]'])[matching_indices]
    S = np.array(ds_ctd['Sal'])[matching_indices]
    lon = np.array(ds_ctd['Longitude'][matching_indices])
    lat = np.array(ds_ctd['Latitude'][matching_indices])

    # Find indices where lon and lat are not NaN
    valid_idx = ~np.isnan(lon) & ~np.isnan(lat)
    
    # Filter lon and lat
    lon = lon[valid_idx]
    lat = lat[valid_idx]
    times = times[valid_idx]
    press = press[valid_idx]
    depth = depth[valid_idx]
    T = T[valid_idx]
    S = S[valid_idx]

    if section_num == 2: 
        sorted_indices = np.argsort(lat)
        times = times[sorted_indices]
        press = press[sorted_indices]
        depth = depth[sorted_indices]
        T = T[sorted_indices]
        S = S[sorted_indices]
        lon = lon[sorted_indices]
        lat = lat[sorted_indices]

    adcp_lats = ds_adcp['LATITUDE']
    adcp_lons = ds_adcp['LONGITUDE']


    depth_adcp = ds_adcp['DEPTH']

    adcp_sec = np.where((adcp_lons<np.nanmax(lon))
                     &(adcp_lons>np.nanmin(lon))
                     &(adcp_lats>np.nanmin(lat))
                     &(adcp_lats<np.nanmax(lat)))[0]
    adcp_range = correct(section_num, adcp_sec, adcp_lons)

    lat_adcp_section = adcp_lats[adcp_range]
    lon_adcp_section = adcp_lons[adcp_range]
    u_adcp = ds_adcp['UCUR'][adcp_range, :].T
    v_adcp = ds_adcp['VCUR'][adcp_range, :].T

    # mask = np.where(np.isfinite(lat_adcp_section) & np.isfinite(lon_adcp_section))
    # lon_adcp_section = lon_adcp_section[mask]
    # lat_adcp_section = lat_adcp_section[mask]
    # u_adcp = u_adcp[:, mask].squeeze()
    # v_adcp = v_adcp[:, mask].squeeze()

    return times, press, depth, T, S, lon, lat, inv_x, lat_adcp_section, lon_adcp_section, depth_adcp, u_adcp, v_adcp

def snip_data(dist, depth, section, ortho_vel, lon_adcp, lat_adcp, distance_adcp, *vars):
    global max_depth
    idx = np.where(depth < max_depth)[0] 
    
    # Apply filtering
    dist, depth = dist[idx], depth[idx]
    vars = [var[idx] for var in vars]  

    # Determine snip value
    if section == 1:
        snip_ctd = 200 
        print(np.shape(ortho_vel))
        snip_adcp = int((snip_ctd / len(dist))) * len(lon_adcp)
    else:
        snip_ctd = 0
        snip_adcp = 0


    # Snip the data
    dist, depth, ortho_vel, lon_adcp, lat_adcp, distance_adcp = dist[snip_ctd:], depth[snip_ctd:], ortho_vel[:, snip_adcp:], lon_adcp[snip_adcp:], lat_adcp[:snip_adcp:], distance_adcp[snip_adcp:]
    vars = [var[snip_ctd:] for var in vars]
    

    return (dist, depth, ortho_vel, lon_adcp, lat_adcp, distance_adcp) + tuple(vars)

def density_anomaly(depth_woa, sigma_woa, sigma_grid, distance):
    mdepth = get_maxdepth()
    ndepth = get_numdepth()

    ind_Z = np.linspace(0, -1*mdepth, ndepth)
    ind_X = np.arange(0, len(np.unique(distance)), 1)
    len_section = len(ind_X)

    Anom = np.full((len(ind_Z),len_section), np.nan)
    ind_depth = np.linspace(0, mdepth, ndepth)
    print(np.max(ind_depth))
    print(np.max(ind_Z))
    interp_func = interpolate.interp1d(depth_woa, sigma_woa, kind='linear', bounds_error=False, fill_value="extrapolate")
    sigma_woa_interp = interp_func(ind_depth)
    for i in ind_X:
        Anom[:, i] = sigma_grid[:, i] - sigma_woa_interp
    return Anom, ind_X, ind_Z


def anom(tref, sref, lon_woa, lat_woa, depth_woa, t_grid, s_grid, sigma_grid, distance, var_name="temp", pas=0.01):
    mdepth = get_maxdepth()
    ndepth = get_numdepth()
    depth_idxs = np.where(depth_woa[:] <= mdepth)
    depth_woa = depth_woa[depth_idxs]
    tref_profile = tref[depth_idxs]
    sref_profile = sref[depth_idxs]

    woa_press = gsw.p_from_z(-depth_woa, lat_woa)
    sigma_woa = determine_sigma(sref_profile, tref_profile, woa_press, lon_woa, lat_woa)

    if var_name in ['density', 'dens']:
        print("Calculating Density Anomalies")
        return density_anomaly(depth_woa, sigma_woa, sigma_grid, distance)

    ind_rho = np.arange(22, 28, pas)
    ind_Z = np.linspace(0, -1 * mdepth, ndepth)
    ind_X = np.arange(len(distance))
    len_section = len(ind_X)

    var_interp = np.full((len(ind_rho), len_section), np.nan)
    Anom_rho = np.full((len(ind_rho), len_section), np.nan)
    depT = np.full((len(ind_rho), len_section), np.nan)
    Anom = np.full((len(ind_Z), len_section), np.nan)
    depth_int = make_depth_grid(distance)

    if var_name in ["temperature", "temp"]:
        print("Calculating Temperature Anomalies")
        ref_profile, var_grid = tref_profile, t_grid
    elif var_name in ["salinity", "sal"]:
        print("Calculating Salinity Anomalies")
        ref_profile, var_grid = sref_profile, s_grid

    ref_interp = np.full(len(ind_rho), np.nan)
    check = np.where(~np.isnan(ref_profile))[0]
    
    if check.size > 0:
        mask = np.where((ind_rho >= np.nanmin(sigma_woa[check])) & (ind_rho <= np.nanmax(sigma_woa[check])))[0]
        if mask.size > 0:
            h = interpolate.interp1d(sigma_woa[check], ref_profile[check], kind='linear', bounds_error=False, fill_value=np.nan)
            ref_interp[mask] = h(ind_rho[mask])
    
    for i in ind_X:
        check = np.where(~np.isnan(var_grid[:, i]))[0]
        if check.size > 1:
            mask = np.where((ind_rho >= np.nanmin(sigma_grid[check, i])) & (ind_rho <= np.nanmax(sigma_grid[check, i])))[0]
            if mask.size > 0:
                f = interpolate.interp1d(sigma_grid[check, i], var_grid[check, i], kind='linear', bounds_error=False, fill_value=np.nan)
                var_interp[mask, i] = f(ind_rho[mask])
                
                for index in mask:
                    if not np.isnan(ref_interp[index]):
                        Anom_rho[index, i] = var_interp[index, i] - ref_interp[index]
                
                g = interpolate.interp1d(sigma_grid[check, i], depth_int[check, i], kind='linear', bounds_error=False, fill_value=np.nan)
                depT[mask, i] = g(ind_rho[mask])

        check = np.where(~np.isnan(depT[:, i]))[0]
        if check.size > 1:
            unique_mask = np.unique(depT[check, i], return_index=True)[1]
            if unique_mask.size > 1:
                m = interpolate.interp1d(depT[check, i][unique_mask], Anom_rho[check, i][unique_mask], kind='linear', bounds_error=False, fill_value=np.nan)
                mask2 = np.where((ind_Z <= np.nanmax(depT[check, i])) & (ind_Z >= np.nanmin(depT[check, i])))[0]
                if mask2.size > 0:
                    Anom[mask2, i] = m(ind_Z[mask2])
    
    return Anom, ind_X, ind_Z

def anom_old(tref, sref, lon_woa, lat_woa, depth_woa, t_grid, s_grid, sigma_grid, distance, var_name="temp", pas=0.01):
    "This works for sure "
    mdepth = get_maxdepth()
    ndepth = get_numdepth()
    depth_idxs = np.where(depth_woa[:] <= mdepth)
    depth_woa = depth_woa[depth_idxs]
    tref_profile = tref[depth_idxs]
    sref_profile = sref[depth_idxs]

    woa_press = gsw.p_from_z(-depth_woa, lat_woa)
    sigma_woa = determine_sigma(sref_profile, tref_profile, woa_press, lon_woa, lat_woa)

    if var_name == 'density' or var_name == 'dens':
        print("Calculating Density Anomalies")
        return density_anomaly(depth_woa, sigma_woa, sigma_grid, distance)

    ind_rho = np.arange(22, 28, pas)
    ind_Z = np.linspace(0, -1*mdepth, ndepth)
    ind_X = np.arange(0, len(np.unique(distance)), 1)
    len_section = len(ind_X)

    var_interp = np.full((len(ind_rho), len_section), np.nan)
    Anom_rho = np.full((len(ind_rho),len_section), np.nan)
    depT = np.full((len(ind_rho),len_section), np.nan)
    Anom = np.full((len(ind_Z),len_section), np.nan)
    depth_int = make_depth_grid(distance)

    if var_name == "temperature" or var_name == "temp" :
        print("Calculating Temperature Anomalies")
        ref_profile = tref_profile
        var_grid = t_grid
    elif var_name == "salinity" or var_name == "sal":
        print("Calculating Salinity Anomalies")
        ref_profile = sref_profile
        var_grid = s_grid        

    # var profile interpolated to sigma
    ref_interp = np.zeros(len(ind_rho))
    check = np.where(np.isnan(ref_profile)== 0)[0]
    ind = np.arange(np.where(ind_rho>=np.nanmin(sigma_woa[check]))[0][0], np.where(ind_rho<=np.nanmax(sigma_woa[check]))[0][-1])
    h = interpolate.interp1d(sigma_woa[check], ref_profile[check], 'linear')
    ref_interp[ind] = h(ind_rho[ind])

    for i in ind_X:
        check = np.where(np.isnan(var_grid[:,i])==0)[0]
        if len(check)>1 :
            # range of rhos that are within our sigmas
            ind2 = np.arange(np.where(ind_rho>=np.nanmin(sigma_grid[check,i]))[0][0],np.where(ind_rho<=np.nanmax(sigma_grid[check,i]))[0][-1])
            # print(ind2)

            # interpolate our temp onto our sigma
            f = interpolate.interp1d(sigma_grid[check,i], var_grid[check,i],'linear')
            var_interp[ind2,i] = f(ind_rho[ind2])
        
            # Calcul de l'anomalie
            for index in ind2:
                if ref_interp[index] != 0:
                    Anom_rho[index,i] = var_interp[index,i] - ref_interp[index]

            # # Interpolation de la profondeur sur les isopycnes
            g = interpolate.interp1d(sigma_grid[check,i], depth_int[check,i],'linear')
            depT[ind2,i] = g(ind_rho[ind2])

        # # Interpolation des anomalies isopycnale sur l'interpolation précédente (anomalies en z)
        check = np.where(np.isnan(depT[:,i])==0)[0]
        # print(np.shape(check))
        if len(check)>1 :
            ind3 = np.arange(np.where(ind_Z<=np.nanmax(depT[check,i]))[0][0],np.where(ind_Z>=np.nanmin(depT[check,i]))[0][-1])
            m = interpolate.interp1d(depT[check,i],Anom_rho[check,i],'linear')
            Anom[ind3,i] = m(ind_Z[ind3])
    return Anom, ind_X, ind_Z

def plot_anomalies1(distance, ind_Z, anom_temp, anom_sal, anom_dens, depth, sigma0, v_ortho, depth_adcp, distance_adcp, section_num, saturation = 0.5, inv_x=0, depth_cutoff=-450, cruise="MSM74"):
    depth = -1 * depth
    ndepth = get_numdepth()
    mdepth = get_maxdepth()
    
    saturationt = saturation
    saturations = saturation
    if cruise == "MSM74":
        indxs = np.where(ind_Z > depth_cutoff)
        anom_temp = anom_temp[indxs, :].squeeze()
        anom_sal = anom_sal[indxs, :].squeeze()
        ind_Z = ind_Z[indxs]

        inddepth = np.where(depth > depth_cutoff)
        contour_dist = distance[inddepth]
        contour_depth = depth[inddepth]
        contour_sigma0 = sigma0[inddepth]
        
        anom_dens = anom_dens[indxs, :].squeeze()

        if section_num == 1:
            # Filter out non-finite values
            finite_indices = np.isfinite(contour_sigma0)
            contour_dist = contour_dist[finite_indices]
            contour_depth = contour_depth[finite_indices]
            contour_sigma0 = contour_sigma0[finite_indices]
    # else:
        # if section_num == 6:
        #     saturationt = saturation
        #     saturations = -1*saturation 

    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Temperature Anomaly
    min1 = saturationt * np.nanmin(anom_temp)
    n2 = Normalize(vmin=min1, vmax=-1 * min1)
    ct2 = axs[0, 0].contourf(np.unique(distance), ind_Z, anom_temp, cmap='coolwarm',
                             levels=np.linspace(min1, -1 * min1, 101), norm=n2, extend='both')
    
    plt.colorbar(ct2, ax=axs[0, 0]).set_label("\u00b0C")
    axs[0, 0].set_title('Temperature Anomaly')
    axs[0, 0].set_ylabel('Depth [m]')
    axs[0, 0].set_xlabel('Distance [km]')
    # axs[0, 0].invert_yaxis()

    # Salinity Anomaly
    min2 = saturations * np.nanmin(anom_sal)
    n1 = Normalize(vmin=min2, vmax=-1 * min2)
    ct1 = axs[0, 1].contourf(np.unique(distance), ind_Z, anom_sal, cmap='coolwarm',
                             levels=np.linspace(min2, -1 * min2, 101), norm=n1, extend='both')
    
    plt.colorbar(ct1, ax=axs[0, 1]).set_label("psu")
    axs[0, 1].set_title('Salinity Anomaly')
    axs[0, 1].set_ylabel('Depth [m]')
    axs[0, 1].set_xlabel('Distance [km]')
    # axs[0, 1].invert_yaxis()

    # Density Anomaly
    if cruise == "MSM40":
        ind_Z = np.linspace(0, -1*ndepth, ndepth)
    min3 = 0.5* np.nanmin(anom_dens)
    n3 = Normalize(vmin=min3, vmax=-1 * min3)
    ct3 = axs[1, 0].contourf(np.unique(distance), ind_Z, anom_dens, cmap='coolwarm',
                             levels=np.linspace(min3, -1 * min3, 101), norm=n3, extend='both') 
    
    plt.colorbar(ct3, ax=axs[1, 0]).set_label("kg/m³")
    if cruise == "MSM40":
        axs[1, 0].set_ylim(-300, 0)
    axs[1, 0].set_title('Density Anomaly')
    axs[1, 0].set_ylabel('Depth [m]')
    axs[1, 0].set_xlabel('Distance [km]')
    # axs[1, 0].invert_yaxis()


    if cruise == "MSM40":
        # Orthogonal Velocity
        distance_grid, depth_grid = np.meshgrid(distance_adcp, depth_adcp)
        V_plot = axs[1, 1].contourf(distance_grid, -1 * depth_grid, v_ortho, levels=np.arange(-1, 1.1, 0.1), cmap='coolwarm', extend='both')
        plt.colorbar(V_plot, ax=axs[1, 1]).set_label('Orthogonal Velocities (m/s)')
        axs[1, 1].set_title("Orthogonal Velocity")
        # axs[1, 1].invert_yaxis()
        
        axs[1, 1].set_ylabel('Depth [m]')
        axs[1, 1].set_xlabel('Distance [km]')
        # Add isopycnes contours
        contour_levels = np.arange(23, 28, 0.2)
        contour_lines1 = axs[0, 0].contour(distance, depth, sigma0, levels=contour_levels, colors='k', linewidths=0.3)
        contour_lines2 = axs[1, 0].contour(distance, depth, sigma0, levels=contour_levels, colors='k', linewidths=0.3)
        contour_lines3 = axs[0, 1].contour(distance, depth, sigma0, levels=contour_levels, colors='k', linewidths=0.3)
        contour_lines4 = axs[1, 1].contour(distance, depth, sigma0, levels=contour_levels, colors='k', linewidths=0.3)
        axs[0, 0].clabel(contour_lines1, inline=True, fmt='%1.1f', fontsize=18)
        axs[0, 1].clabel(contour_lines2, inline=True, fmt='%1.1f', fontsize=18)
        axs[1, 0].clabel(contour_lines3, inline=True, fmt='%1.1f', fontsize=18)
        axs[1, 1].clabel(contour_lines4, inline=True, fmt='%1.1f', fontsize=18)

    else:
        axs[1, 1].set_title(f"Orthogonal Velocity")
        distance_grid, depth_grid = np.meshgrid(distance_adcp, depth_adcp)
        
        levels_velocity = np.arange(-1, 1.1, 0.1)
        V_plot = axs[1, 1].contourf(distance_grid, depth_grid, v_ortho, levels=levels_velocity, cmap='coolwarm', extend='both')

        # levels_velocity=np.arange(-1,1.1,0.1)
        # V_plot = axs[1, 1].contourf(distance_adcp, depth_adcp, v_ortho,levels=levels_velocity, cmap='coolwarm',extend='both')
        cbar4 = fig.colorbar(V_plot, ax=axs[1, 1], label='Velocity (m/s)')
        cbar4.set_label('Velocity (m/s)',fontsize=18)
        cbar4.ax.tick_params(labelsize=18)
        axs[1, 1].set_ylim(0,mdepth)
        axs[1, 1].set_xlabel('Distance [km]',fontsize=18)
        axs[1, 1].set_ylabel('Depth (m)',fontsize=18)
        axs[1, 1].set_title('Orthogonal Velocity',fontsize=18)
        axs[1, 1].invert_yaxis() 
        axs[1, 1].tick_params(axis='both', which='major', labelsize=18)

        clines = axs[0, 0].tricontour(contour_dist, contour_depth, contour_sigma0, levels=np.arange(23, 28, 0.1), colors='k', linewidths=0.8)
        clines2 = axs[0, 1].tricontour(contour_dist, contour_depth, contour_sigma0, levels=np.arange(23, 28, 0.1), colors='k', linewidths=0.8)
        clines3 = axs[1, 0].tricontour(contour_dist, contour_depth, contour_sigma0, levels=np.arange(23, 28, 0.1), colors='k', linewidths=0.8)
        clines4 = axs[1, 1].tricontour(contour_dist, -1*contour_depth, contour_sigma0, levels=np.arange(23, 28, 0.1), colors='k', linewidths=0.8)
        axs[0, 0].clabel(clines, inline=True, fmt='%1.1f', fontsize=11)
        axs[0, 1].clabel(clines2, inline=True, fmt='%1.1f', fontsize=11)
        axs[1, 0].clabel(clines3, inline=True, fmt='%1.1f', fontsize=11)
        axs[1, 1].clabel(clines4, inline=True, fmt='%1.1f', fontsize=11)

    if inv_x:
        axs[0, 0].invert_xaxis()
        axs[0, 1].invert_xaxis()
        axs[1, 0].invert_xaxis()
        axs[1, 1].invert_xaxis()
    plt.suptitle(f"Anomalies for {cruise} Section {section_num}")

    plt.tight_layout()
    plt.show()

def plot_anomalies(distance, ind_Z, anom_temp, anom_sal, anom_dens, depth, sigma0, v_ortho, depth_adcp, distance_adcp, section_num, saturation = 0.25, saturationd=0.6, inv_x=0, depth_cutoff=-450, cruise="MSM74",depth_max_adcp=-300, depth_min_adcp=30):
    depth = -1 * depth
    ndepth = get_numdepth()
    mdepth = get_maxdepth()
    
    saturationt = saturation
    saturations = saturation
    if cruise == "MSM74":
        indxs = np.where(ind_Z > depth_cutoff)
        anom_temp = anom_temp[indxs, :].squeeze()
        anom_sal = anom_sal[indxs, :].squeeze()
        ind_Z = ind_Z[indxs]

        inddepth = np.where(depth > depth_cutoff)
        contour_dist = distance[inddepth]
        contour_depth = depth[inddepth]
        contour_sigma0 = sigma0[inddepth]
        
        anom_dens = anom_dens[indxs, :].squeeze()

        if section_num == 1:
            # Filter out non-finite values
            finite_indices = np.isfinite(contour_sigma0)
            contour_dist = contour_dist[finite_indices]
            contour_depth = contour_depth[finite_indices]
            contour_sigma0 = contour_sigma0[finite_indices]
    # else:
        # if section_num == 6:
        #     saturationt = saturation
        #     saturations = -1*saturation 

    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Temperature Anomaly
    min1 = saturationt * np.nanmin(anom_temp)
    n2 = Normalize(vmin=min1, vmax=-1 * min1)
    ct2 = axs[0, 0].contourf(np.unique(distance), ind_Z, anom_temp, cmap='coolwarm',
                             levels=np.linspace(min1, -1 * min1, 101), norm=n2, extend='both')
    
    plt.colorbar(ct2, ax=axs[0, 0]).set_label("\u00b0C")
    axs[0, 0].set_title('Temperature Anomaly')
    axs[0, 0].set_ylabel('Depth [m]')
    axs[0, 0].set_xlabel('Distance [km]')
    # axs[0, 0].invert_yaxis()

    # Salinity Anomaly
    min2 = saturations * np.nanmin(anom_sal)
    n1 = Normalize(vmin=min2, vmax=-1 * min2)
    ct1 = axs[0, 1].contourf(np.unique(distance), ind_Z, anom_sal, cmap='coolwarm',
                             levels=np.linspace(min2, -1 * min2, 101), norm=n1, extend='both')
    
    plt.colorbar(ct1, ax=axs[0, 1]).set_label("psu")
    axs[0, 1].set_title('Salinity Anomaly')
    axs[0, 1].set_ylabel('Depth [m]')
    axs[0, 1].set_xlabel('Distance [km]')
    # axs[0, 1].invert_yaxis()

    # Density Anomaly
    if cruise == "MSM40":
        ind_Z = np.linspace(0, -1*ndepth, ndepth)
    min3 = saturationd* np.nanmin(anom_dens)
    n3 = Normalize(vmin=min3, vmax=-1 * min3)
    ct3 = axs[1, 0].contourf(np.unique(distance), ind_Z, anom_dens, cmap='coolwarm',
                             levels=np.linspace(min3, -1 * min3, 101), norm=n3, extend='both') 
    
    plt.colorbar(ct3, ax=axs[1, 0]).set_label("kg/m³")
    if cruise == "MSM40":
        axs[0, 0].set_ylim(depth_max_adcp, 0)
        axs[0, 1].set_ylim(depth_max_adcp, 0)
        axs[1, 0].set_ylim(depth_max_adcp, 0)
        axs[1, 1].set_ylim(depth_max_adcp, 0)
    axs[1, 0].set_title('Density Anomaly')
    axs[1, 0].set_ylabel('Depth [m]')
    axs[1, 0].set_xlabel('Distance [km]')
    # axs[1, 0].invert_yaxis()


    if cruise == "MSM40":
        # Orthogonal Velocity
        distance_grid, depth_grid = np.meshgrid(distance_adcp, depth_adcp)
        V_plot = axs[1, 1].contourf(distance_grid, -1 * depth_grid, v_ortho, levels=np.linspace(-0.4, 0.4, 80), cmap='RdBu_r', extend='both')
        plt.colorbar(V_plot, ax=axs[1, 1]).set_label('Orthogonal Velocities (m/s)')
        axs[1, 1].set_title("Orthogonal Velocity")
        # axs[1, 1].invert_yaxis()
        
        axs[1, 1].set_ylabel('Depth [m]')
        axs[1, 1].set_xlabel('Distance [km]')
        # Add isopycnes contours
        contour_levels = np.arange(23.1, 28, 0.1)
        contour_lines1 = axs[0, 0].contour(distance, depth, sigma0, levels=contour_levels, colors='k', linewidths=0.3)
        contour_lines2 = axs[1, 0].contour(distance, depth, sigma0, levels=contour_levels, colors='k', linewidths=0.3)
        contour_lines3 = axs[0, 1].contour(distance, depth, sigma0, levels=contour_levels, colors='k', linewidths=0.3)
        contour_lines4 = axs[1, 1].contour(distance, depth, sigma0, levels=contour_levels, colors='k', linewidths=0.3)
        axs[0, 0].clabel(contour_lines1, inline=True, fmt='%1.1f', fontsize=8)
        axs[0, 1].clabel(contour_lines2, inline=True, fmt='%1.1f', fontsize=8)
        axs[1, 0].clabel(contour_lines3, inline=True, fmt='%1.1f', fontsize=8)
        axs[1, 1].clabel(contour_lines4, inline=True, fmt='%1.1f', fontsize=8)

    else:
        # Define a saturation factor (>1 increases saturation, <1 decreases it)
        levels = np.linspace(-0.35, 0.35, 65)  
        distance_grid, depth_grid = np.meshgrid(distance_adcp, depth_adcp)
        V_plot = axs[1, 1].contourf(distance_grid, depth_grid, v_ortho, levels=levels, cmap="RdBu_r", norm=Normalize(vmin=-0.35, vmax=0.35), extend='both')
        cbar4 = fig.colorbar(V_plot, ax=axs[1, 1], label='Velocity (m/s)')
        cbar4.set_label('Velocity (m/s)')
        cbar4.ax.tick_params(labelsize=18)
        axs[1, 1].set_xlabel('Distance [km]')
        axs[1, 1].set_ylabel('Depth (m)')
        axs[1, 1].set_title('Orthogonal Velocity')
        axs[1, 1].set_ylim(depth_min_adcp, depth_max_adcp)
        # axs[1, 1].set_ylim(depth_min_adcp, depth_max_adcp)
        axs[1, 1].invert_yaxis() 
        axs[1, 1].tick_params(axis='both', which='major')
        

        clines = axs[0, 0].tricontour(contour_dist, contour_depth, contour_sigma0, levels=np.arange(23, 28, 0.1), colors='k', linewidths=0.8)
        clines2 = axs[0, 1].tricontour(contour_dist, contour_depth, contour_sigma0, levels=np.arange(23, 28, 0.1), colors='k', linewidths=0.8)
        clines3 = axs[1, 0].tricontour(contour_dist, contour_depth, contour_sigma0, levels=np.arange(23, 28, 0.1), colors='k', linewidths=0.8)
        clines4 = axs[1, 1].tricontour(contour_dist, -1*contour_depth, contour_sigma0, levels=np.arange(23, 28, 0.1), colors='k', linewidths=0.8)
        axs[0, 0].clabel(clines, inline=True, fmt='%1.1f', fontsize=11)
        axs[0, 1].clabel(clines2, inline=True, fmt='%1.1f', fontsize=11)
        axs[1, 0].clabel(clines3, inline=True, fmt='%1.1f', fontsize=11)
        axs[1, 1].clabel(clines4, inline=True, fmt='%1.1f', fontsize=11)

    if inv_x:
        axs[0, 0].invert_xaxis()
        axs[0, 1].invert_xaxis()
        axs[1, 0].invert_xaxis()
        axs[1, 1].invert_xaxis()
    firsts = get_dists(distance, depth, section_num)
    ys1 = np.ones_like(firsts)
    ys = ys1 * -8
    axs[0, 0].scatter(firsts, ys, color="orange", s=5, zorder=10)
    axs[0, 1].scatter(firsts, ys, color="orange", s=5, zorder=10)
    axs[1, 0].scatter(firsts, ys, color="orange", s=5, zorder=10)
    # axs[1, 1].scatter(firsts, ys, color="orange", s=5, zorder=10)
    plt.suptitle(f"Anomalies for {cruise} Section {section_num}")

    plt.tight_layout()
    plt.show()

def plot_eddy(ax, xs, zs, var, sigma_grid, title, cbar_label, colormap, start_eddy, end_eddy, eddy_z_min, eddy_z, inv_x=0):
    X, Z = np.meshgrid(xs, zs)
    contour = ax.contourf(X, Z, var, levels=40, cmap=colormap)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(cbar_label)
    
    # Extract the slice of xs based on the start and end eddy indices
    xs_slice = xs[start_eddy:end_eddy+1]

    # Create a mask for the zs values within the range of 80 to 300
    z_mask = (zs < eddy_z) & (zs > eddy_z_min)  # Mask for zs within the specified range

    # Now, filter the points within the valid ranges for both xs and zs
    valid_indices = np.where(z_mask[:, None] & (np.isin(xs, xs_slice)))  # Indices where both conditions hold

    # Extract the valid x and z coordinates
    valid_xs = xs[valid_indices[1]]  # Corresponding x values
    valid_zs = zs[valid_indices[0]]  # Corresponding z values

    ax.scatter(valid_xs, valid_zs, s=1, c='red', marker='o', zorder=2)

    contour_levels = np.arange(23, 28, 0.1)
    clines = ax.contour(X, Z, sigma_grid, levels=contour_levels, colors='k', linewidths=0.8)
    ax.clabel(clines, inline=True, fmt='%1.1f', fontsize=11)
    ax.set_xlabel("Distance [km]")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    ax.invert_yaxis()
    
    if inv_x:
        ax.invert_xaxis()

def plot_eddy_ellip(ax, xs, zs, var, col, sigma_grid, title, cbar_label, colormap, 
               eddy_center, a, b, cruise, ylimmax=None, inv_x=0):
    if ylimmax is None:
        ylimmax = get_maxdepth()
    X, Z = np.meshgrid(xs, zs)
    # contour = ax.contourf(X, Z, var, levels=40, cmap=colormap)
    # cbar = plt.colorbar(contour, ax=ax)
    # cbar.set_label(cbar_label)
    
    # Ellipsoid equation: ((x - x0)^2 / a^2) + ((z - z0)^2 / b^2) = 1
    x0, z0 = eddy_center
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipsoid_x = x0 + a * np.cos(theta)
    ellipsoid_z = z0 + b * np.sin(theta)
    
    # Ensure the ellipsoid is within the depth bounds
    # mask = (ellipsoid_z >= eddy_z_min) & (ellipsoid_z <= eddy_z)
    ax.plot(ellipsoid_x, ellipsoid_z, color=col, linewidth=2, zorder=10)
    
    # if cruise == "MSM40":
    #     contour_levels = np.arange(23, 28, 0.1)
    #     # np.append(contour_levels, 27.7)
    #     # print(contour_levels)
    # else:
    #     contour_levels=np.arange(23, 28, 0.1)
    # clines = ax.contour(X, Z, sigma_grid, levels=contour_levels, colors='k', linewidths=0.8)
    # ax.clabel(clines, inline=True, fmt='%1.1f', fontsize=11)
    # ax.set_xlabel("Distance [km]")
    # ax.set_ylabel("Depth (m)")
    # ax.set_title(title)
    # ax.set_ylim(0, ylimmax)
    # ax.invert_yaxis()

    if inv_x:
        ax.invert_xaxis()


def plot_ts_diagrams(sal_eddy, temp_eddy, zs_eddy, sal_other, temp_other, zs_other, xs, zs, t_grid, s_grid, sigma_grid, eddy_center, a, b, cruise, section_num, eddy_num, ylimmax = 4, ylimmin = 3, xlimmax = 35, xlimmin = 34, inv_x=0, cmap='cmo.thermal', clabel="Celsius"):
    original_cmap1 = plt.cm.Reds
    cmap1 = colors.ListedColormap(original_cmap1(np.linspace(0.4, 1, 256)))
    original_cmap2 = plt.cm.Blues
    cmap2 = colors.ListedColormap(original_cmap2(np.linspace(0.4, 1, 256)))
    # Create subplots (3 subplots now: eddy plot, TS diagram with color maps, TS diagram with red/blue colors)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Eddy plot using plot_eddy_ellip
    ax1 = axes[0]
    plot_eddy_ellip(ax1, xs, zs, t_grid, sigma_grid, f"Eddy {eddy_num} in {cruise} Section {section_num}", clabel, cmap, eddy_center, a, b, cruise, inv_x=inv_x)

    # Plot 2: TS Diagram with color maps
    ax2 = axes[1]
    # possibles de S
    SAL_diag = np.arange(19.0, 37.0, 0.01)
    # possibles de T
    TEMP_diag = np.arange(-3, 24, 0.1)
    SAL_diag, TEMP_diag = np.meshgrid(SAL_diag, TEMP_diag)
    SIGMA0_diag = gsw.density.sigma0(SAL_diag, TEMP_diag)

    # Tracer des isopycnes sur le diagramme T-S
    if cruise == "MSM40":
        Contourrange = np.arange(22, 28.5, 0.1)
    else:
        Contourrange = np.arange(22, 28.5, 0.1)
    CS = ax2.contour(SAL_diag, TEMP_diag, SIGMA0_diag, Contourrange, colors='k', linestyles=':', zorder=1)

    ax2.clabel(CS, fontsize=11, inline=1)

    # Scatter plots
    sc1 = ax2.scatter(sal_eddy, temp_eddy, c=zs_eddy, marker='.', cmap=cmap1, zorder=10)  # Intérieur
    sc2 = ax2.scatter(sal_other, temp_other, c=zs_other, marker='.', cmap=cmap2)  # Extérieur gauche

    cb1 = plt.colorbar(sc1, ax=ax2)
    cb2 = plt.colorbar(sc2, ax=ax2)
    cb1.set_label('core of the eddy')
    cb2.set_label('reference profile')

    ax2.set_xlabel(r'$ {\rm Salinity \,  [} {\rm psu]}$', fontsize=11, rasterized=True)
    ax2.set_ylabel(r'$ {\rm Temperature \,  [} {\rm °C]}$', fontsize=11, rasterized=True)
    cb1.ax.invert_yaxis()
    cb2.ax.invert_yaxis()
    ax2.set_title(r'$\theta-S$ diagram', fontsize=20)
    ax2.set_ylim(np.nanmin(t_grid)-0.1, np.nanmax(t_grid)+0.1)
    ax2.set_xlim(np.nanmin(s_grid)-0.1, np.nanmax(s_grid)+0.1)

    # Plot 3: TS Diagram with red/blue colors
    ax3 = axes[2]
    # Scatter plots with explicit colors
    sc1 = ax3.scatter(sal_eddy, temp_eddy, c='red', marker='.', zorder=10)  # Intérieur (core of the eddy in red)
    sc2 = ax3.scatter(sal_other, temp_other, c='blue', marker='.', zorder=1)  # Extérieur gauche (reference profile in blue)

    CS = ax3.contour(SAL_diag, TEMP_diag, SIGMA0_diag, Contourrange, colors='k', linestyles=':', zorder=1)
    ax3.clabel(CS, fontsize=11, inline=1)

    water_masses = {
        "SAIW": (3.62, 0.43, 34.994, 0.057, 27.831, 0.049, 'indigo'),
        "LSW": (3.24, 0.32, 35.044, 0.031, 27.931, 0.042,'orange'),
        "ISOW": (3.02, 0.26, 35.098, 0.028, 28.001, 0.044,'green'),
        "DSOW": (1.27, 0.29, 35.052, 0.016, 28.194, 0.028, 'blue'),
        "uNADW": (3.33, 0.31, 35.071, 0.027, 27.942, 0.027, 'cyan'),
        "lNADW": (2.96, 0.21, 35.083, 0.019, 28.0, 0.029, 'gold'),
        "uENACW": (13.72, 0.5, 36.021, 0.03, 26.887, 0.05, 'purple'),
        "lENACW": (11.36, 0.5, 35.689, 0.03, 27.121, 0.05, 'blueviolet'),
        "uWNACW" : (18.79, 0.5, 36.816, 0.03, 26.344, 0.05, 'lightpink'),
        "lWNACW" : (17.51, 0.5, 36.634, 0.03, 26.554, 0.05, 'coral'),
        "uESACW": (13.60, 0.5, 35.398, 0.03, 26.500, 0.05, 'crimson'),
        "lESACW": (9.44, 0.5, 34.900, 0.03, 26.928, 0.05, 'magenta'),
        "uWSACW": (16.30, 0.5, 35.936, 0.03, 26.295, 0.05, 'darkblue'),
        "lWSACW": (12.30, 0.5, 34.294, 0.03, 26.707, 0.05, 'darkcyan'),
        "AAIW": (1.78, 1.02, 34.206, 0.083, 27.409, 0.111, 'darkgoldenrod'),
        "MW": (12.21, 0.77, 36.682, 0.081, 27.734, 0.150, 'darkgreen')

    }
    for name, (temp_mean, temp_std, sal_mean, sal_std, dens_mean, dens_std, color) in water_masses.items():
        if temp_mean < ylimmax and temp_mean > ylimmin and sal_mean < xlimmax and sal_mean > xlimmin:
            ax3.fill_betweenx(
                [temp_mean - temp_std, temp_mean + temp_std],
                sal_mean - sal_std, sal_mean + sal_std,
                color=color, alpha=0.3, label=name, zorder=9
            )
            ax3.text(sal_mean, temp_mean, name, fontsize=9, ha='center', va='center', color='black', fontweight='bold', zorder=10)

    # Labels and title
    ax3.set_xlabel(r'$ {\rm Salinity \,  [} {\rm psu]}$', fontsize=15, rasterized=True)
    ax3.set_ylabel(r'$ {\rm Temperature \,  [} {\rm °C]}$', fontsize=15, rasterized=True)
    ax3.set_title(r'$\theta-S$ diagram', fontsize=20)
    ax3.set_ylim(ylimmin, ylimmax)
    ax3.set_xlim(xlimmin, xlimmax)
    # ax3.set_xlim(np.nanmin(s_grid)-0.1, np.nanmax(s_grid)+0.1)
    

    sc1.set_label('core of the eddy')
    sc2.set_label('reference profile')

    # Add legends
    ax3.legend(loc='lower left', fontsize='small')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig(f"/Users/emmagurcan/Documents/France/ENS/M1/stageM1/analysis/plots/{cruise}/ts/section_{section_num}_eddy_{eddy_num}")
    plt.show()


# Function to plot 2D contour
def plot_contour(ax, x, y, var, sigma0, title, cbar_label, colormap, inv_x):
    X, Y = np.meshgrid(x, y)
    contour = ax.contourf(X, Y, var, levels=40, cmap=colormap)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(cbar_label)
    contour_levels = np.arange(23, 28, 0.1)
    clines = ax.contour(X, Y, sigma0, levels=contour_levels, colors='k', linewidths=0.8)
    ax.clabel(clines, inline=True, fmt='%1.1f', fontsize=11)
    ax.set_xlabel("Distance [km]")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    ax.invert_yaxis()
    if inv_x:
        ax.invert_xaxis()

def contour_3d(distance, depth, distance_adcp, depth_adcp, temperature, salinity, sigma0, v_ortho, section_num, inv_x=0, depth_max_adcp=300, depth_min_adcp=30):
    """
    Create a 2x2 subplot showing temperature, salinity, potential density, and orthogonal velocity.

    Parameters:
    distance (numpy array): 1D Distance array (km).
    depth (numpy array): 1D Depth array (m).
    distance_adcp (numpy array): 1D Distance array for ADCP data.
    depth_adcp (numpy array): 1D Depth array for ADCP data.
    temperature (numpy array): 2D Temperature data (depth, distance).
    salinity (numpy array): 2D Salinity data (depth, distance).
    sigma0 (numpy array): 2D Potential density data (depth, distance).
    v_ortho (numpy array): 2D Orthogonal velocity data (depth_adcp, distance_adcp).
    section_num (int): Section number for the title.
    inv_x (bool): Flag to invert x-axis.
    depth_max_adcp (float): Maximum depth for the velocity plot (default: 300).
    depth_min_adcp (float): Minimum depth for the velocity plot (default: 30).
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.suptitle(f"MSM74 Section {section_num}")
    
    
    
    # Temperature
    axs[0, 0].set_title("Temperature")
    plot_contour(axs[0, 0], distance, depth, temperature, sigma0, "Temperature", "Temperature (°C)", 'cmo.thermal')
    
    # Salinity
    axs[0, 1].set_title("Salinity")
    plot_contour(axs[0, 1], distance, depth, salinity, sigma0, "Salinity", "Salinity (psu)", 'cmo.haline')
    
    # Potential Density
    axs[1, 0].set_title("Potential Density")
    plot_contour(axs[1, 0], distance, depth, sigma0, sigma0, "Potential Density", "Potential Density (kg/m³)", 'cmo.dense')
    
    # Orthogonal Velocity
    axs[1, 1].set_title("Orthogonal Velocity")
    distance_grid, depth_grid = np.meshgrid(distance_adcp, depth_adcp)
    levels_velocity = np.arange(-1, 1.1, 0.1)
    V_plot = axs[1, 1].contourf(distance_grid, depth_grid, v_ortho, levels=levels_velocity, cmap='coolwarm', extend='both')
    cbar = plt.colorbar(V_plot, ax=axs[1, 1])
    cbar.set_label('Orthogonal Velocities (m/s)', fontsize=12)
    
    contour_levels = np.arange(23, 28, 0.1)
    clines = axs[1, 1].contour(distance_grid, depth_grid, sigma0[:depth_adcp.size, :distance_adcp.size],
                                levels=contour_levels, colors='k', linewidths=0.8)
    axs[1, 1].clabel(clines, inline=True, fmt='%1.1f', fontsize=11)
    axs[1, 1].set_ylim(depth_min_adcp, depth_max_adcp)
    axs[1, 1].set_xlabel('Distance (km)', fontsize=12)
    axs[1, 1].set_ylabel('Depth (m)', fontsize=12)
    axs[1, 1].invert_yaxis()
    if inv_x:
        axs[1, 1].invert_xaxis()
    
    plt.tight_layout()
    # plt.savefig(f'/Users/emmagurcan/Documents/France/ENS/M1/stageM1/analysis/plots/section_{section_num}')
    plt.show()


def calc_ortho_vel(section_num, cruise, LON, LAT, u_section,v_section):
    """METHOD DE YAN"""
    # if section_num == 3 and cruise == "MSM74":
    #     return u_section

    # Calculate orthogonal velocity
    max_lat = LAT[-1]
    max_lon = LON[-1]

    min_lat = LAT[0]
    min_lon = LON[0]
    # R = 6371  # Earth's radius in km

    # plot_coords(np.array([min_lon, max_lon]), np.array([min_lat, max_lat]))
    
    alpha = np.arctan((max_lat-min_lat)/(max_lon-min_lon))
    v_ortho = -u_section*np.sin(alpha) + v_section*np.cos(alpha)
    
    return v_ortho

def compute_orthogonal_velocity(lon, lat, u_adcp, v_adcp):
    """
    METHOD DE GURCAN
    """
    # Compute differences in lon/lat to get along-track direction
    dlon = np.diff(lon)
    dlat = np.diff(lat)
    
    # Convert to unit vectors (along-track direction)
    ds = np.sqrt(dlon**2 + dlat**2)
    tx = dlon / ds  # Along-track x-component
    ty = dlat / ds  # Along-track y-component
    
    # Compute perpendicular vectors (-ty, tx)
    px = ty
    py = -tx
    
    # Extend to match shape of u_adcp and v_adcp (depth dimension)
    px = np.tile(px, (u_adcp.shape[0], 1))
    py = np.tile(py, (u_adcp.shape[0], 1))
    
    # Compute perpendicular velocity using dot product
    v_perp = u_adcp[:, :-1] * px + v_adcp[:, :-1] * py
    
    # Append last column to match original shape
    v_perp = np.hstack((v_perp, v_perp[:, -1][:, np.newaxis]))
    
    return v_perp

# def determine_distance(lon, lat, inv_x, section_num):
#     # d1 = np.sin(lat*(math.pi/180))*np.sin(lat[0]*(math.pi/180))
#     # d2 = np.cos(lat*(math.pi/180))*np.cos(lat[0]*(math.pi/180)) * \
#     #     np.cos(abs(lon[0]-lon)*(math.pi/180))
#     # distance = 6371*np.arccos(d1+d2)
#     # return distance
#     if section_num == 5:
#         inv_x = 0
#     if not inv_x:
#         first_lat = np.nanmin(lat)
#         first_lon = np.nanmin(lon)
#     else:
#         first_lat = np.nanmax(lat)
#         first_lon = np.nanmax(lon)

#     d1 = np.sin(lat*(math.pi/180))*np.sin(first_lat*(math.pi/180))
#     d2 = np.cos(lat*(math.pi/180))*np.cos(first_lat*(math.pi/180)) * \
#         np.cos(abs(first_lon-lon)*(math.pi/180))
#     distance = 6371*np.arccos(d1+d2)
#     return distance

def determine_distance(lon, lat, inv_x, section_num):
    if section_num == 5:
        inv_x = 0
    first_lat =lat[0]
    first_lon = lon[0]

    d1 = np.sin(lat*(math.pi/180))*np.sin(first_lat*(math.pi/180))
    d2 = np.cos(lat*(math.pi/180))*np.cos(first_lat*(math.pi/180)) * \
        np.cos(abs(first_lon-lon)*(math.pi/180))
    
    distance = 6371*np.arccos(d1+d2)
    return distance

def correct(section_num, adcp_sec, LON_75):
    if section_num == 1:
        ind_sort = np.argsort(LON_75[adcp_sec])
        adcp_sec = ind_sort[1:]
        return adcp_sec
    elif section_num == 2:
        ind2 = []
        for i in range(len(adcp_sec)) :
            if (i<=1030) | (i>1140) :
                ind2.append(i)
        return adcp_sec[ind2]
    elif section_num == 5:
        ind5 = []
        for i in range(len(adcp_sec)) :
            if (i>649) & (i<2200) :
                ind5.append(i)
        return adcp_sec[ind5]
    elif section_num == 6:
        ind6 = []
        for i in range(len(adcp_sec)) :
            if (i>300) :
                ind6.append(i)
        return adcp_sec[ind6]
    elif section_num == 7:
        ind7 = []
        for i in range(len(adcp_sec)) :
            if (i<=2200) | (i>2800) :
                ind7.append(i)
        return adcp_sec[ind7]
    else:
        return adcp_sec

def get_eddies(section_num, cruise, sub_section=None):
    if cruise == "MSM74":
        if section_num == 2:
            eddies = {1: (50, 220, 20, 120), 2: (219, 210, 60, 90), 3: (369, 120, 20, 90)}
            return eddies
        if section_num == 3:
            eddies = {1: (255, 200, 70, 140)}
            return eddies
        if section_num == 4:
            print(f"No eddies in {cruise} section {section_num}")
            return None
        if section_num == 5:
            eddies = {1: (165, 85, 29, 35), 2: (80, 110, 8, 50)}
            return eddies
        if section_num == 6:
            print(f"No eddies in {cruise} section {section_num}")
            return None
        if section_num == 7:
            eddies = {1: (115, 280, 10, 150), 2: (200, 100, 10, 60), 3: (268, 190, 10, 100)}
            return eddies
    if cruise == "MSM40":
        if section_num == 1:
            eddies = {1: (120, 15, 10, 5), 2: (143, 20, 10, 8), 3: (280, 90, 20, 69), 4: (390, 70, 25, 50)}
            return eddies
        if section_num == 2:
            print(f"No eddies in {cruise} section {section_num}")
            return None
        if section_num == 3:
            eddies = {1: (200, 45, 10, 45), 2: (265, 40, 10, 25), 3: (375, 40, 10, 25), 4: (445, 30, 10, 24)}
            return eddies
        if section_num == 4:
            eddies = {1: (15, 30, 10, 15), 2: (45, 40, 15, 30), 3: (80, 25, 12, 23), 4: (120, 45, 16, 25), 5: (234, 40, 18, 22)}
            return eddies
        if section_num == 5:
            print(f"See Julian's extensive work")
            return None
        if section_num == 6:
            print(f"No eddies in {cruise} section {section_num}")
            return None
        if section_num == 7:
            if sub_section == 3:
                eddies = {1: (115, 220, 15, 100), 2: (137, 80, 18, 50), 3: (260, 90, 25,70)}
            if sub_section == 2:
                eddies = {1: (25, 65, 14, 20), 2: (180, 90, 18, 55), 3: (260, 70, 18, 30), 4: (335, 120, 50, 90)}
            if sub_section == 1:
                eddies = {1: (50, 165, 30, 140), 2: (150, 200, 30, 140), 3: (325, 160, 30, 140)}
            return eddies 

def plot_eddy_selected(ax, xs, zs, anomalyt, temp_eddy, sigma_grid, distance, ind_Z, title, cbar_label, c, cruise, section_num, depth, cmap=None, ylimmax=300, inv_x=0):
    # if ylimmax is None:
    #     ylimmax = np.max(zs)
    
    X, Z = np.meshgrid(xs, ind_Z)
    # contour = ax.contourf(X, Z, anomalyt, cmap=colormap, levels=levels, extend='both')
    # min1 = 0.7 * np.nanmin(anomalyt)
    # print(min1)
    # n2 = Normalize(vmin=min1, vmax=-1 * min1)
    # contour = ax.contourf(np.unique(distance), ind_Z, anomalyt, cmap='coolwarm',
    #                             levels=np.linspace(min1, -1 * min1, 101), norm=n2, extend='both')
        
    # plt.colorbar(contour)

    # # cbar = plt.clabel(contour)
    # plt.xlabel("Distance [km]")
    # plt.ylabel("Depth [m]")
    # plt.title("Temeprature Anomalies with Mean Section Reference")
    # contour_levels = np.arange(23, 28, 0.1)
    # clines = plt.contour(xs, -1*zs, sigma_grid, levels=contour_levels, colors='k', linewidths=0.8)
    min1 = 0.7 * np.nanmin(anomalyt)
    n2 = Normalize(vmin=min1, vmax=-1 * min1)
    if cruise == "MSM74" and section_num == 5:
        indices = find_increasing_intervals(depth)
        xs = np.array([distance[s] for s, st in indices])
        print(len(xs))
    else:
        xs = np.unique(distance)
    contour = ax.contourf(xs, -1*ind_Z, anomalyt, cmap='coolwarm',
                                levels=np.linspace(min1, -1 * min1, 101), norm=n2, extend='both')
        
    # plt.colorbar(contour)
    if c == "orange":
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label(cbar_label)
        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    

    # Highlight temp_eddy in green
    # Create a binary mask where temp_eddy has values (non-NaN)
    masked_eddy = np.where(np.isnan(temp_eddy), np.nan, 1)  # 1 where data exists, NaN elsewhere

    mask = ~np.isnan(temp_eddy)

    # Smooth/expand the mask slightly to avoid gaps (optional)
    mask_dilated = scipy.ndimage.binary_dilation(mask)

    # Convert mask to float (needed for contouring)
    mask_float = mask_dilated.astype(float)

    ax.contour(xs, -1 * ind_Z, mask_float, levels=[0.5], colors=c, linewidths=2, zorder=11)
    # highlight = ax.pcolormesh(xs, -1*ind_Z, temp_eddy, cmap=c, shading='auto', alpha=0.22, zorder=10)
        
    # Plot density contours
    if cruise == "MSM40":
        contour_levels = np.arange(23, 28, 0.1)
    else:
        contour_levels = np.arange(23, 28, 0.1)
    clines = ax.contour(X, Z, sigma_grid, levels=contour_levels, colors='k', linewidths=0.8)
    firsts = get_dists(distance, depth, section_num, cruise="MSM40")
    ys = np.ones_like(firsts)
    ys = ys * 8
    ax.scatter(firsts, ys, color="red", s=6, zorder=10)
    ax.clabel(clines, inline=True, fmt='%1.1f', fontsize=11)
    
    ax.set_xlabel("Distance [km]", fontsize=15)
    ax.set_ylabel("Depth [m]", fontsize=15)
    ax.set_title(title, fontsize=20)
    ax.set_ylim(0, ylimmax)
    ax.invert_yaxis()
    
    if inv_x:
        ax.invert_xaxis()

def plot_ts_diagrams_anom(anomalyt, sal_eddy, temp_eddy3d, temp_eddy, zs_eddy, sal_other, temp_other, zs_other, xs, zs, t_grid, s_grid, sigma_grid, distance, depth, ind_Z, cruise, section_num, pos_anom, eddy_num, ylimmax = 4, ylimmin = 3, xlimmax = 35, xlimmin = 34, inv_x=0, cmap='coolwarm', clabel="Celsius"):
    if pos_anom:
        c = "green"
    else:
        c = "orange"
    original_cmap1 = plt.cm.Reds
    cmap1 = colors.ListedColormap(original_cmap1(np.linspace(0.4, 1, 256)))
    original_cmap2 = plt.cm.Blues
    cmap2 = colors.ListedColormap(original_cmap2(np.linspace(0.4, 1, 256)))
    # Create subplots (3 subplots now: eddy plot, TS diagram with color maps, TS diagram with red/blue colors)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Eddy plot using plot_eddy_ellip
    ax1 = axes[0]
    # plot_please(ax1, ind_Z, distance, anomalyt)
    plot_eddy_selected(ax1, xs, zs, anomalyt,temp_eddy3d, sigma_grid, distance, ind_Z, f"Eddy {eddy_num} in {cruise} Section {section_num}", clabel,  c, cruise, section_num, depth, inv_x=inv_x, cmap=cmap)

    # Plot 2: TS Diagram with color maps
    ax2 = axes[1]
    # possibles de S
    SAL_diag = np.arange(32, 37, 0.01)
    TEMP_diag = np.arange(-1, 13, 0.1)
    SAL_diag, TEMP_diag = np.meshgrid(SAL_diag, TEMP_diag)
    SIGMA0_diag = gsw.density.sigma0(SAL_diag, TEMP_diag)
    Contourrange = np.arange(21, 33, 0.1)

    CS = ax2.contour(SAL_diag, TEMP_diag, SIGMA0_diag, Contourrange, colors='k', linestyles=':', zorder=1)

    ax2.clabel(CS, fontsize=11, inline=1)

    # Scatter plots
    sc1 = ax2.scatter(sal_eddy, temp_eddy, c=zs_eddy, marker='.', cmap=cmap1, zorder=10)  # Intérieur
    sc2 = ax2.scatter(sal_other, temp_other, c=zs_other, marker='.', cmap=cmap2)  # Extérieur gauche

    cb1 = plt.colorbar(sc1, ax=ax2)
    cb2 = plt.colorbar(sc2, ax=ax2)
    cb1.set_label('core of the eddy')
    cb2.set_label('reference profile')

    ax2.set_xlabel(r'$ {\rm Salinity \,  [} {\rm psu]}$', fontsize=11, rasterized=True)
    ax2.set_ylabel(r'$ {\rm Temperature \,  [} {\rm °C]}$', fontsize=11, rasterized=True)
    cb1.ax.invert_yaxis()
    cb2.ax.invert_yaxis()
    ax2.set_title(r'$\theta-S$ diagram', fontsize=20)
    ax2.set_ylim(np.nanmin(t_grid)-0.1, np.nanmax(t_grid)+0.1)
    ax2.set_xlim(np.nanmin(s_grid)-0.1, np.nanmax(s_grid)+0.1)
    # ax2.set_xlim(34, np.nanmax(s_grid)+0.1)

    # Plot 3: TS Diagram with red/blue colors
    ax3 = axes[2]
    # Scatter plots with explicit colors
    sc1 = ax3.scatter(sal_eddy, temp_eddy, c='red', marker='.', zorder=10)  # Intérieur (core of the eddy in red)
    sc2 = ax3.scatter(sal_other, temp_other, c='blue', marker='.', zorder=1)  # Extérieur gauche (reference profile in blue)

    CS = ax3.contour(SAL_diag, TEMP_diag, SIGMA0_diag, Contourrange, colors='k', linestyles=':', zorder=1)
    ax3.clabel(CS, fontsize=11, inline=1)

    # Manually position contour labels within xlim and ylim
    ax3.clabel(CS, fontsize=11)


    water_masses = {
        "LSW" : (3.2, 0.7, 34.85, 0.1, 'orange'),
        "SAIW": (3.62, 0.43, 34.994, 0.057, 'coral'),
        # "MW" : (12, 0.5, 36.3, 0.2, 'darkgreen'),
        "NEADW" : (3.5, 0.5, 35.7, 0.2, 'green'),
        "AABW" : (-0.5, 0.5, 34.63, 0.2, 'magenta'),
        # "EDW": (18, 0.5, 36.5, 0.2, 'darkcyan'),
        "IBW": (7.5, 0.5, 35.125, 0.02, 'crimson'),
        "RTW": (7.7, 0.3, 35.2, 0.02, 'indigo')

    }

    dens_water_masses = {
        "SPMW27.3" : (9.82, 0.8, 35.41, 0.083, 27.3, 0.05, 'darkblue'),
        "SPMW27.4" : (8.64, 0.8, 35.29, 0.04, 27.4, 0.05, 'darkblue'),
        "SPMW27.5" : (7.53, 0.5, 35.2, 0.09, 27.5, 0.05, 'darkblue'),
        "DSOW" : (0.17, 1, 34.66, 0.08, 27.82, 0.05, 'darkgoldenrod'),
        "AAIW" : (4, 0.5, 34.2, 0.1, 27.1, 0.05, 'blue'),
        "ISOW" : (1.25, 1.75, 34.885, 0.015, 27.9, 0.1, 'purple'),
    }

    if section_num == 4 and cruise == 'MSM40':
        water_masses["S4"] = (3.604, 1.680, 33.934, 0.709, 26.831, 0.50, 'darkgreen')

  
    for name, (temp_mean, temp_std, sal_mean, sal_std, color) in water_masses.items():
        ax3.fill_betweenx(
            [temp_mean - temp_std, temp_mean + temp_std],
            sal_mean - sal_std, sal_mean + sal_std,
            color=color, alpha=0.3, label=name, zorder=10
        )
        ax3.text(sal_mean, temp_mean, name, fontsize=9, ha='center', va='center', color='black', fontweight='bold', zorder=10)

    # Highlight only the specific regions along density contours for dens_water_masses
    for name, (temp_mean, temp_std, sal_mean, sal_std, density, density_std, color) in dens_water_masses.items():
    # Find the closest contour level to the given density
        density_levels = np.array(CS.levels)  # Extract contour levels
        closest_level = density_levels[np.abs(density_levels - density).argmin()]  # Find closest contour

        # Get contour paths corresponding to the closest density level
        for path in CS.collections[np.where(density_levels == closest_level)[0][0]].get_paths():
            vertices = path.vertices
            salinity_contour, temperature_contour = vertices[:, 0], vertices[:, 1]

            # Find the part of the contour that is within the salinity and temperature range
            mask = (
                (temperature_contour >= temp_mean - temp_std) & (temperature_contour <= temp_mean + temp_std) &
                (salinity_contour >= sal_mean - sal_std) & (salinity_contour <= sal_mean + sal_std)
            )

            # Highlight only the selected part of the contour
            if np.any(mask):
                ax3.fill_betweenx(
                    temperature_contour[mask],
                    salinity_contour[mask] - sal_std,
                    salinity_contour[mask] + sal_std,
                    color=color, alpha=0.3, label=name, zorder=10
                )

                # Compute the midpoint for better label placement
                avg_sal = np.mean(salinity_contour[mask])
                avg_temp = np.mean(temperature_contour[mask])

                # Place the label slightly above the highlighted region
                ax3.text(
                    avg_sal, avg_temp + 0.2,  # Shift label slightly upwards
                    name, fontsize=9, ha='center', va='bottom', color='black', fontweight='bold', zorder=10
                )
    # Labels and title
    ax3.set_xlabel(r'$ {\rm Salinity \,  [} {\rm psu]}$', fontsize=15, rasterized=True)
    ax3.set_ylabel(r'$ {\rm Temperature \,  [} {\rm °C]}$', fontsize=15, rasterized=True)
    ax3.set_title(r'$\theta-S$ diagram', fontsize=20)

    ax3.set_xlim(33.5, 35.5)
    # ax3.set_ylim(2, np.nanmax(temp_eddy3d) + 0.5)
    # ax3.set_ylim(ylimmin, ylimmax)
    # ax3.set_xlim(xlimmin, xlimmax)
    # ax3.set_xlim(np.nanmin(s_grid)-0.1, np.nanmax(s_grid)+0.1)
    

    sc1.set_label('core of the eddy')
    sc2.set_label('reference profile')

    # Add legends
    ax3.legend(loc='upper right', fontsize='small')

    # Adjust layout and show the plot
    plt.tight_layout()
    # plt.savefig(f"/Users/emmagurcan/Documents/France/ENS/M1/stageM1/analysis/plots/{cruise}/ts/section_{section_num}_eddy_{eddy_num}")
    plt.show()