#%% Plots final with uCTD and velocities (coordinates)

import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.io
import cmocean 
import fonctions as f
import matplotlib.ticker as ticker

# Load Data
rep = 'C:/Users/J-mar/Documents/MOCIS/Stage M1/Data_MSM40/'
uCTD = scipy.io.loadmat(rep+'UCTD_dataproc.mat')
adcp = scipy.io.loadmat(rep +'adcp_150kHz/TL_GEOMAR_150_STJOHNS_BREST_000_000000_6_hc.mat')

# Define the section (section 2 can't be plotted => no uCTD measures)
section_index = 5
name = f'section_{section_index}'
section = f.section(section_index)
start_index=section[0]
end_index=section[1]
len_section=section[2]
start_index_adcp=section[3]
end_index_adcp=section[4]

# Define depth limit for uCTD
depth_max=300

# Define depth limit for ADCP
depth_max_adcp=300
    
# Create depth array
depth = np.arange(0, 1001)

# Select uCTD data
lon = np.squeeze(uCTD['uctdlon'])[start_index:end_index+1]
lat = np.squeeze(uCTD['uctdlat'])[start_index:end_index+1]
PT = np.squeeze(uCTD['PTgrid'])[:,start_index:end_index+1]
PD = np.squeeze(uCTD['PDgrid'])[:,start_index:end_index+1]
S = np.squeeze(uCTD['Sgrid'])[:,start_index:end_index+1]

# Select adcp data
b = adcp['b']
bb = b[0,0]
ref = bb[0][0,0][0]
nav = bb[2][0,0][0]
vel = bb[1]
u = vel[:,0,:] + ref[0,:] # vitesse zonale, positive vers l'est
v = vel[:,1,:] + ref[1,:] # vitesse méridionale, positive vers le nord
depth_adcp = bb[3][0]
LON_adcp = nav[1,:]
LAT_adcp = nav[2,:]
LAT = LAT_adcp[start_index_adcp:end_index_adcp+1]
LON = LON_adcp[start_index_adcp:end_index_adcp+1]
u_section = u[:,start_index_adcp:end_index_adcp+1]
v_section = v[:,start_index_adcp:end_index_adcp+1]

# Calculate orthogonal velocity
a1 = 6371*np.arccos(np.sin(LAT[-1]*(math.pi/180))*np.sin(LAT[0]*(math.pi/180))+np.cos(LAT[-1]*(math.pi/180))*np.cos(LAT[0]*(math.pi/180)))
b1 = 6371*np.arccos(np.sin(LAT[-1]*(math.pi/180))*np.sin(LAT[-1]*(math.pi/180))+np.cos(LAT[-1]*(math.pi/180))*np.cos(LAT[-1]*(math.pi/180))*np.cos(abs(LON[-1]-LON[0])*(math.pi/180)))
alpha = np.arctan(a1/b1)
v_ortho = np.cos(alpha)*u_section + np.sin(alpha)*v_section

# Plot the figure
fig, ax = plt.subplots(2, 2, figsize=(20, 16))

# Definition of x 
if section_index < 4:
    x_uctd = lat
    x_adcp = LAT
    x_name = 'Latitude (°)'
else:
    x_uctd = lon
    x_adcp = LON
    x_name = 'Longitude (°)'

# Plot Potential Temperature
PT_plot = ax[0, 0].contourf(x_uctd, depth, PT,levels=20, cmap='cmo.thermal',extend='both')
cbar1 = fig.colorbar(PT_plot, ax=ax[0, 0])
cbar1.set_label('Potential Temperature (°C)',fontsize=18)
cbar1.ax.tick_params(labelsize=18)
ax[0, 0].set_ylim(0,depth_max)
#ax[0, 0].set_xlim(-53,-51)
ax[0, 0].set_xlabel(f'{x_name}',fontsize=18)
ax[0, 0].set_ylabel('Depth (m)',fontsize=18)
ax[0, 0].set_title('Potential Temperature (°C)',fontsize=18)
ax[0, 0].invert_yaxis()  
ax[0, 0].tick_params(axis='both', which='major', labelsize=18)
ax[0, 0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))


# Plot Potential Density
PD_plot = ax[0, 1].contourf(x_uctd, depth, PD, levels=np.arange(26,28,0.1), cmap='cmo.dense',extend='both')
cbar2 = fig.colorbar(PD_plot, ax=ax[0, 1])
cbar2.set_label('Potential Density (kg/m3)',fontsize=18)
cbar2.ax.tick_params(labelsize=18)
ax[0, 1].set_ylim(0,depth_max)
#ax[0, 1].set_xlim(-53,-51)
ax[0, 1].set_xlabel(f'{x_name}',fontsize=18)
#ax[0, 1].set_ylabel('Depth (m)',fontsize=18)
ax[0, 1].set_title('Potential Density',fontsize=18)
ax[0, 1].invert_yaxis()
ax[0, 1].tick_params(axis='both', which='major', labelsize=18)  
ax[0, 1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

# Plot Salinity
S_plot = ax[1, 0].contourf(x_uctd, depth, S,levels=20, cmap='cmo.haline',extend='both')
cbar3 = fig.colorbar(S_plot, ax=ax[1, 0], label='Salinity (g/kg)')
cbar3.set_label('Salinity (g/kg)',fontsize=18)
cbar3.ax.tick_params(labelsize=18)
ax[1, 0].set_ylim(0,depth_max)
#ax[1, 0].set_xlim(-53,-51)
ax[1, 0].set_xlabel(f'{x_name}',fontsize=18)
ax[1, 0].set_ylabel('Depth (m)',fontsize=18)
ax[1, 0].set_title('Salinity',fontsize=18)
ax[1, 0].invert_yaxis() 
ax[1, 0].tick_params(axis='both', which='major', labelsize=18)
ax[1, 0].xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

# Plot Velocity
levels_velocity=np.arange(-1,1.1,0.1)
V_plot = ax[1, 1].contourf(x_adcp, depth_adcp, v_ortho,levels=levels_velocity, cmap='coolwarm',extend='both')
cbar4 = fig.colorbar(V_plot, ax=ax[1, 1], label='Velocity (m/s)')
cbar4.set_label('Velocity (m/s)',fontsize=18)
cbar4.ax.tick_params(labelsize=18)
ax[1, 1].set_ylim(0,depth_max_adcp)
#ax[1, 1].set_xlim(-53,-51)
ax[1, 1].set_xlabel(f'{x_name}',fontsize=18)
#ax[1, 1].set_ylabel('Depth (m)',fontsize=18)
ax[1, 1].set_title('Velocities',fontsize=18)
ax[1, 1].invert_yaxis() 
ax[1, 1].tick_params(axis='both', which='major', labelsize=18)
ax[1, 1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

# Add isopycnes contours
contour_levels = np.arange(23, 28, 0.2)
contour_lines1 = ax[0, 0].contour(x_uctd, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
contour_lines2 = ax[1, 0].contour(x_uctd, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
contour_lines3 = ax[0, 1].contour(x_uctd, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
contour_lines4 = ax[1, 1].contour(x_uctd, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[0, 0].clabel(contour_lines1, inline=True, fmt='%1.1f', fontsize=18)
ax[0, 1].clabel(contour_lines2, inline=True, fmt='%1.1f', fontsize=18)
ax[1, 0].clabel(contour_lines3, inline=True, fmt='%1.1f', fontsize=18)
ax[1, 1].clabel(contour_lines4, inline=True, fmt='%1.1f', fontsize=18)

plt.tight_layout()

plt.savefig(f'profiles_{name}.png')

#%% Plots final with uCTD and velocities (coordinates) zoom tourbillon

import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.io
import cmocean 
import fonctions as f
import matplotlib.ticker as ticker

# Load Data
rep = 'C:/Users/J-mar/Documents/MOCIS/Stage M1/Data_MSM40/'
uCTD = scipy.io.loadmat(rep+'UCTD_dataproc.mat')
adcp = scipy.io.loadmat(rep +'adcp_150kHz/TL_GEOMAR_150_STJOHNS_BREST_000_000000_6_hc.mat')

# Define the section (section 2 can't be plotted => no uCTD measures)
section_index = 5
name = f'section_{section_index}'
section = f.section(section_index)
start_index=section[0]
end_index=section[1]
len_section=section[2]
start_index_adcp=section[3]
end_index_adcp=section[4]

# Define depth limit for uCTD
depth_max=300

# Define depth limit for ADCP
depth_max_adcp=300
    
# Create depth array
depth = np.arange(0, 1001)

# Select uCTD data
lon = np.squeeze(uCTD['uctdlon'])[start_index:end_index+1]
lat = np.squeeze(uCTD['uctdlat'])[start_index:end_index+1]
PT = np.squeeze(uCTD['PTgrid'])[:,start_index:end_index+1]
PD = np.squeeze(uCTD['PDgrid'])[:,start_index:end_index+1]
S = np.squeeze(uCTD['Sgrid'])[:,start_index:end_index+1]

# Select adcp data
b = adcp['b']
bb = b[0,0]
ref = bb[0][0,0][0]
nav = bb[2][0,0][0]
vel = bb[1]
u = vel[:,0,:] + ref[0,:] # vitesse zonale, positive vers l'est
v = vel[:,1,:] + ref[1,:] # vitesse méridionale, positive vers le nord
depth_adcp = bb[3][0]
LON_adcp = nav[1,:]
LAT_adcp = nav[2,:]
LAT = LAT_adcp[start_index_adcp:end_index_adcp+1]
LON = LON_adcp[start_index_adcp:end_index_adcp+1]
u_section = u[:,start_index_adcp:end_index_adcp+1]
v_section = v[:,start_index_adcp:end_index_adcp+1]

# Calculate orthogonal velocity
a1 = 6371*np.arccos(np.sin(LAT[-1]*(math.pi/180))*np.sin(LAT[0]*(math.pi/180))+np.cos(LAT[-1]*(math.pi/180))*np.cos(LAT[0]*(math.pi/180)))
b1 = 6371*np.arccos(np.sin(LAT[-1]*(math.pi/180))*np.sin(LAT[-1]*(math.pi/180))+np.cos(LAT[-1]*(math.pi/180))*np.cos(LAT[-1]*(math.pi/180))*np.cos(abs(LON[-1]-LON[0])*(math.pi/180)))
alpha = np.arctan(a1/b1)
v_ortho = np.cos(alpha)*u_section + np.sin(alpha)*v_section

# Plot the figure
fig, ax = plt.subplots(2, 2, figsize=(20, 16))

# Definition of x 
if section_index < 4:
    x_uctd = lat
    x_adcp = LAT
    x_name = 'Latitude (°)'
else:
    x_uctd = lon
    x_adcp = LON
    x_name = 'Longitude (°)'

# Plot Potential Temperature
PT_plot = ax[0, 0].contourf(x_uctd, depth, PT,levels=20, cmap='cmo.thermal',extend='both')
cbar1 = fig.colorbar(PT_plot, ax=ax[0, 0])
cbar1.set_label('Potential Temperature (°C)',fontsize=18)
cbar1.ax.tick_params(labelsize=18)
ax[0, 0].set_ylim(0,depth_max)
ax[0, 0].set_xlabel(f'{x_name}',fontsize=18)
ax[0, 0].set_ylabel('Depth (m)',fontsize=18)
ax[0, 0].set_title('Potential Temperature (°C)',fontsize=18)
ax[0, 0].invert_yaxis()  
ax[0, 0].tick_params(axis='both', which='major', labelsize=18)

# Plot Potential Density
PD_plot = ax[0, 1].contourf(x_uctd, depth, PD, levels=20, cmap='cmo.dense',extend='both')
cbar2 = fig.colorbar(PD_plot, ax=ax[0, 1])
cbar2.set_label('Potential Density (kg/m3)',fontsize=18)
cbar2.ax.tick_params(labelsize=18)
ax[0, 1].set_ylim(0,depth_max)
ax[0, 1].set_xlabel(f'{x_name}',fontsize=18)
ax[0, 1].set_ylabel('Depth (m)',fontsize=18)
ax[0, 1].set_title('Potential Density',fontsize=18)
ax[0, 1].invert_yaxis()
ax[0, 1].tick_params(axis='both', which='major', labelsize=18)  

# Plot Salinity
S_plot = ax[1, 0].contourf(x_uctd, depth, S,levels=20, cmap='cmo.haline',extend='both')
cbar3 = fig.colorbar(S_plot, ax=ax[1, 0], label='Salinity (g/kg)')
cbar3.set_label('Salinity (g/kg)',fontsize=18)
cbar3.ax.tick_params(labelsize=18)
ax[1, 0].set_ylim(0,depth_max)
ax[1, 0].set_xlabel(f'{x_name}',fontsize=18)
ax[1, 0].set_ylabel('Depth (m)',fontsize=18)
ax[1, 0].set_title('Salinity',fontsize=18)
ax[1, 0].invert_yaxis() 
ax[1, 0].tick_params(axis='both', which='major', labelsize=18)

# Plot Velocity
levels_velocity=np.arange(-1,1.1,0.1)
V_plot = ax[1, 1].contourf(x_adcp, depth_adcp, v_ortho,levels=levels_velocity, cmap='coolwarm',extend='both')
cbar4 = fig.colorbar(V_plot, ax=ax[1, 1], label='Velocity (m/s)')
cbar4.set_label('Velocity (m/s)',fontsize=18)
cbar4.ax.tick_params(labelsize=18)
ax[1, 1].set_ylim(0,depth_max_adcp)
ax[1, 1].set_xlabel(f'{x_name}',fontsize=18)
ax[1, 1].set_ylabel('Depth (m)',fontsize=18)
ax[1, 1].set_title('Velocities',fontsize=18)
ax[1, 1].invert_yaxis() 
ax[1, 1].tick_params(axis='both', which='major', labelsize=18)

# Add isopycnes contours
contour_levels = np.arange(23, 28, 0.2)
contour_lines1 = ax[0, 0].contour(x_uctd, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
contour_lines2 = ax[1, 0].contour(x_uctd, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
contour_lines3 = ax[0, 1].contour(x_uctd, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
contour_lines4 = ax[1, 1].contour(x_uctd, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[0, 0].clabel(contour_lines1, inline=True, fmt='%1.1f', fontsize=18)
ax[0, 1].clabel(contour_lines2, inline=True, fmt='%1.1f', fontsize=18)
ax[1, 0].clabel(contour_lines3, inline=True, fmt='%1.1f', fontsize=18)
ax[1, 1].clabel(contour_lines4, inline=True, fmt='%1.1f', fontsize=18)

plt.tight_layout()

plt.savefig(f'profiles_{name}.png')



#%% Plots final with uCTD and velocities (distance)

import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.io
import cmocean 
import fonctions as f

# Load Data
rep = 'C:/Users/J-mar/Documents/MOCIS/Stage M1/Data_MSM40/'
uCTD = scipy.io.loadmat(rep+'UCTD_dataproc.mat')
adcp = scipy.io.loadmat(rep +'adcp_150kHz/TL_GEOMAR_150_STJOHNS_BREST_000_000000_6_hc.mat')

# Define the section (section 2 can't be plotted => no uCTD measures)
section_index = 5
name = f'section_{section_index}'
section = f.section(section_index)
start_index=section[0]
end_index=section[1]
len_section=section[2]
start_index_adcp=section[3]
end_index_adcp=section[4]

# Define depth limit for uCTD
depth_max=300

# Define depth limit for ADCP
depth_max_adcp=300
    
# Create depth array
depth = np.arange(0, 1001)

# Select uCTD data
lon = np.squeeze(uCTD['uctdlon'])[start_index:end_index+1]
lat = np.squeeze(uCTD['uctdlat'])[start_index:end_index+1]
PT = np.squeeze(uCTD['PTgrid'])[:,start_index:end_index+1]
PD = np.squeeze(uCTD['PDgrid'])[:,start_index:end_index+1]
S = np.squeeze(uCTD['Sgrid'])[:,start_index:end_index+1]

# Select adcp data
b = adcp['b']
bb = b[0,0]
ref = bb[0][0,0][0]
nav = bb[2][0,0][0]
vel = bb[1]
u = vel[:,0,:] + ref[0,:] # vitesse zonale, positive vers l'est
v = vel[:,1,:] + ref[1,:] # vitesse méridionale, positive vers le nord
depth_adcp = bb[3][0]
LON_adcp = nav[1,:]
LAT_adcp = nav[2,:]

LAT = LAT_adcp[start_index_adcp:end_index_adcp+1]
LON = LON_adcp[start_index_adcp:end_index_adcp+1]
u_section = u[:,start_index_adcp:end_index_adcp+1]
v_section = v[:,start_index_adcp:end_index_adcp+1]

# Calculate distances
d1 = np.sin(lat*(math.pi/180))*np.sin(lat[0]*(math.pi/180))
d2 = np.cos(lat*(math.pi/180))*np.cos(lat[0]*(math.pi/180)) * \
    np.cos(abs(lon[0]-lon)*(math.pi/180))
distance = 6371*np.arccos(d1+d2)

d1_adcp = np.sin(LAT*(math.pi/180))*np.sin(LAT[0]*(math.pi/180))
d2_adcp = np.cos(LAT*(math.pi/180))*np.cos(LAT[0]*(math.pi/180)) * \
    np.cos(abs(LON[0]-LON)*(math.pi/180))
distance_adcp = 6371*np.arccos(d1_adcp+d2_adcp)

# Calculate orthogonal velocity
a1 = 6371*np.arccos(np.sin(LAT[-1]*(math.pi/180))*np.sin(LAT[0]*(math.pi/180))+np.cos(LAT[-1]*(math.pi/180))*np.cos(LAT[0]*(math.pi/180)))
b1 = 6371*np.arccos(np.sin(LAT[-1]*(math.pi/180))*np.sin(LAT[-1]*(math.pi/180))+np.cos(LAT[-1]*(math.pi/180))*np.cos(LAT[-1]*(math.pi/180))*np.cos(abs(LON[-1]-LON[0])*(math.pi/180)))
alpha = np.arctan(a1/b1)
v_ortho = np.cos(alpha)*u_section + np.sin(alpha)*v_section

# Plot the figure
fig, ax = plt.subplots(2, 2, figsize=(20, 16))

# Plot Potential Temperature
PT_plot = ax[0, 0].contourf(distance, depth, PT,levels=20, cmap='cmo.thermal',extend='both')
fig.colorbar(PT_plot, ax=ax[0, 0], label='Potential Temperature (°C)')
ax[0, 0].set_ylim(0,depth_max)
ax[0, 0].set_xlabel('Distance (km)')
ax[0, 0].set_ylabel('Depth (m)')
ax[0, 0].set_title('Potential Temperature (°C)')
ax[0, 0].invert_yaxis()  

# Plot Potential Density
PD_plot = ax[0, 1].contourf(distance, depth, PD, levels=20, cmap='cmo.dense',extend='both')
fig.colorbar(PD_plot, ax=ax[0, 1], label='Potential Density (kg/m3)')
ax[0, 1].set_ylim(0,depth_max)
ax[0, 1].set_xlabel('Distance (km)')
ax[0, 1].set_ylabel('Depth (m)')
ax[0, 1].set_title('Potential Density')
ax[0, 1].invert_yaxis()  

# Plot Salinity
S_plot = ax[1, 0].contourf(distance, depth, S,levels=20, cmap='cmo.haline',extend='both')
fig.colorbar(S_plot, ax=ax[1, 0], label='Salinity (g/kg)')
ax[1, 0].set_ylim(0,depth_max)
ax[1, 0].set_xlabel('Distance (km)')
ax[1, 0].set_ylabel('Depth (m)')
ax[1, 0].set_title('Salinity')
ax[1, 0].invert_yaxis() 

# Plot Velocity
levels_velocity=np.arange(-1,1.1,0.1)
V_plot = ax[1, 1].contourf(distance_adcp, depth_adcp, v_ortho,levels=levels_velocity, cmap='coolwarm',extend='both')
fig.colorbar(V_plot, ax=ax[1, 1], label='Velocity (m/s)')
ax[1, 1].set_ylim(0,depth_max_adcp)
ax[1, 1].set_xlabel('Distance (km)')
ax[1, 1].set_ylabel('Depth (m)')
ax[1, 1].set_title('Velocities')
ax[1, 1].invert_yaxis() 

# Add isopycnes contours
contour_levels = np.arange(23, 28, 0.2)
contour_lines = ax[0, 0].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[0, 0].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)
contour_lines = ax[1, 0].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[0, 1].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)
contour_lines = ax[0, 1].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[1, 0].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)
contour_lines = ax[1, 1].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[1, 1].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)

plt.savefig(f'profiles_{name}.png')

#%%
plt.figure(figsize=(10,4))

# Plot Velocity
levels_velocity=np.arange(-1,1.1,0.1)
V_plot = plt.contourf(distance_adcp, depth_adcp, v_ortho,levels=levels_velocity, cmap='coolwarm',extend='both')
cbar=plt.colorbar(V_plot)
cbar.set_label('Orthogonal velocities (m/s)',fontsize=15)
cbar.ax.tick_params(labelsize=15)

contour_lines = plt.contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
plt.clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)

plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylim(0,depth_max_adcp)
plt.xlabel('Distance (km)',fontsize=15)
plt.ylabel('Depth (m)',fontsize=15)
plt.gca().invert_yaxis() 

plt.tight_layout()

plt.savefig('section5_velocities.png')


#%%
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

rep='C:/Users/J-mar/Documents/MOCIS/Stage M1/'

# Local map

# Load bathymetric data (example using NetCDF file)
bathymetry_data = xr.open_dataset(rep+'bathy/etopo2.nc')

# Extract relevant variables (e.g., longitude, latitude, depth)
lon_bathy = bathymetry_data['lon']
lat_bathy = bathymetry_data['lat']
depth = bathymetry_data['topo']

# Define region of interest
lon_min, lon_max, lat_min, lat_max = -70, -40, 55, 63.5

# Subset data to region of interest
lon_subset = lon_bathy.sel(lon=slice(lon_min, lon_max))
lat_subset = lat_bathy.sel(lat=slice(lat_min, lat_max))
depth_subset = depth.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

# Define map projection
projection = ccrs.PlateCarree()

# Create a plot with a larger figure size
fig = plt.figure(figsize=(18, 9))

# Add subplot for the map
ax = fig.add_subplot(111, projection=projection,aspect='auto')

# Plot bathymetric relief
c = ax.contourf(lon_subset, lat_subset, depth_subset, transform=projection, cmap='Blues_r',zorder=0)

# Add land feature
ax.add_feature(cfeature.LAND, edgecolor='k', facecolor='lightgray',zorder=1)

# Add coastlines and other features
ax.coastlines(resolution='10m',zorder=2)

# Add grid lines
gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='black',linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

# Add color bar
cbar = plt.colorbar(c, ax=ax, orientation='vertical', fraction=0.05)
cbar.set_label('Depth (m)',fontsize=15)
cbar.ax.tick_params(labelsize=15)

# Add title and labels
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Set extent of the plot
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=projection)

# Load data from campaign
Data = pd.read_csv(rep+"Carte.csv", sep=';')

# Select the variables
uCTD = Data.loc[Data['Gear']=='uCTD',['Lon','Lat']].to_numpy().T
CTD = Data.loc[Data['Gear']=='CTD/RO',['Lon','Lat']].to_numpy().T
moor_dep = Data.loc[Data['Gear']=='MOOR deployed',['Lon','Lat']].to_numpy().T
moor_rec = Data.loc[Data['Gear']=='MOOR deployed',['Lon','Lat']].to_numpy().T
BPS = Data.loc[Data['Gear']=='BPS',['Lon','Lat']].to_numpy().T
argo = Data.iloc[79,0:2].to_numpy().T

# Plot the different measurements and moorings
ax.scatter(lon, lat, marker='o', color='red', edgecolor='white', s=100, label='uCTD', transform=ccrs.Geodetic())
ax.text(lon[0]-0.7, lat[0]+0.2, 'Start',color='red',fontsize=18,weight='bold',transform=ccrs.Geodetic())
ax.text(lon[-1]+0.5, lat[-1], 'End',color='red',fontsize=18,weight='bold')

plt.savefig('carte_zoom.png')

#%% Plots final with uCTD and velocities (distance window)

# Distance window
distance_min=400
distance_max=500

# For only a portion of the graph
indices = np.where((distance>distance_min) & (distance<distance_max))
distance=distance[indices]
PT=PT.T[indices].T
PD=PD.T[indices].T
S=S.T[indices].T
indices_adcp = np.where((distance_adcp>distance_min) & (distance_adcp<distance_max))
distance_adcp=distance_adcp[indices_adcp]
v_section=v_section.T[indices_adcp].T

# Plot the figure
fig, ax = plt.subplots(2, 2, figsize=(20, 16))

# Plot Potential Temperature
PT_plot = ax[0, 0].contourf(distance, depth, PT,levels=20, cmap='cmo.thermal')
fig.colorbar(PT_plot, ax=ax[0, 0], label='Potential Temperature (°C)')

ax[0, 0].set_ylim(0,depth_max)
ax[0, 0].set_xlabel('Distance (km)')
ax[0, 0].set_ylabel('Depth')
ax[0, 0].set_title('Potential Temperature')
ax[0, 0].invert_yaxis()  

# Plot Potential Density
PD_plot = ax[0, 1].contourf(distance, depth, PD, levels=20, cmap='cmo.dense')
fig.colorbar(PD_plot, ax=ax[0, 1], label='Potential Density')

ax[0, 1].set_ylim(0,depth_max)
ax[0, 1].set_xlabel('Distance (km)')
ax[0, 1].set_ylabel('Depth')
ax[0, 1].set_title('Potential Density')
ax[0, 1].invert_yaxis()  

# Plot Salinity
S_plot = ax[1, 0].contourf(distance, depth, S,levels=20, cmap='cmo.haline')
fig.colorbar(S_plot, ax=ax[1, 0], label='Salinity (g/kg)')

ax[1, 0].set_ylim(0,depth_max)
ax[1, 0].set_xlabel('Distance (km)')
ax[1, 0].set_ylabel('Depth')
ax[1, 0].set_title('Salinity')
ax[1, 0].invert_yaxis() 

# Plot Velocity
levels_velocity=np.arange(-1,1,0.1)
V_plot = ax[1, 1].contourf(distance_adcp, depth_adcp, v_ortho,levels=levels_velocity, cmap='coolwarm')
fig.colorbar(V_plot, ax=ax[1, 1], label='Velocity (m/s)')

ax[1, 1].set_ylim(0,depth_max_adcp)
ax[1, 1].set_xlabel('Distance (km)')
ax[1, 1].set_ylabel('Depth')
ax[1, 1].set_title('Velocities')
ax[1, 1].invert_yaxis() 

# Add isopycnes contours
contour_levels = np.arange(23, 28, 0.1)
contour_lines = ax[0, 0].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[0, 0].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)
contour_lines = ax[1, 0].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[0, 1].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)
contour_lines = ax[0, 1].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[1, 0].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)
contour_lines = ax[1, 1].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[1, 1].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)

plt.savefig('tourbillon_'+name+'.pdf')


#%% Plots final with uCTD and velocities

import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.io
import cmocean 

# Load Data
rep = 'C:/Users/J-mar/Documents/MOCIS/Stage M1/Data_MSM40/'
uCTD = scipy.io.loadmat(rep+'UCTD_dataproc.mat')
adcp = scipy.io.loadmat(rep +'adcp_150kHz/TL_GEOMAR_150_STJOHNS_BREST_000_000000_6_hc.mat')

# Define the section (section 2 can't be plotted => no uCTD measures)
section = 7
name = 'section_{}'.format(section)

# Section selection
if (section==1):
    start_index = 0
    end_index = 22
    start_index_adcp = 0
    end_index_adcp = 1899
elif (section==2):
    print("Pas de mesures uCTD")
elif (section==3):
    start_index = 24
    end_index = 45
    start_index_adcp = 8486
    end_index_adcp = 11325
elif (section==4):
    start_index = 45
    end_index = 66
    start_index_adcp = 11325
    end_index_adcp = 13092
elif (section==5):
    start_index = 66
    end_index = 106
    start_index_adcp = 13092
    end_index_adcp = 15925
elif (section==6):
    start_index = 106
    end_index = 120
    start_index_adcp = 15925
    end_index_adcp = 18187
elif (section==7):
    start_index = 120
    end_index = 166
    start_index_adcp = 18187
    end_index_adcp = 25526
    
# Create depth array
max_depth = 1001
depth = np.arange(0, max_depth)

# Distance
distance_min=400
distance_max=500

# Select uCTD data
lon = np.squeeze(uCTD['uctdlon'])[start_index:end_index]
lat = np.squeeze(uCTD['uctdlat'])[start_index:end_index]
PT = np.squeeze(uCTD['PTgrid'])[:, start_index:end_index]
PD = np.squeeze(uCTD['PDgrid'])[:, start_index:end_index]
S = np.squeeze(uCTD['Sgrid'])[:, start_index:end_index]

# Select adcp data
b = adcp['b']
bb = b[0,0]
ref = bb[0][0,0][0]
nav = bb[2][0,0][0]
vel = bb[1]
u = vel[:,0,:] # vitesse zonale, positive vers l'est
v = vel[:,1,:] # vitesse méridionale, positive vers le nord
depth_adcp = bb[3][0]
LON_adcp = nav[1,:]
LAT_adcp = nav[2,:]

LAT = LAT_adcp[start_index_adcp:end_index_adcp]
LON = LON_adcp[start_index_adcp:end_index_adcp]
u_section = u[:,start_index_adcp:end_index_adcp]
v_section = v[:,start_index_adcp:end_index_adcp]

# Calculate distances
d1 = np.sin(lat*(math.pi/180))*np.sin(lat[0]*(math.pi/180))
d2 = np.cos(lat*(math.pi/180))*np.cos(lat[0]*(math.pi/180)) * \
    np.cos(abs(lon[0]-lon)*(math.pi/180))
distance = 6371*np.arccos(d1+d2)

d1_adcp = np.sin(LAT*(math.pi/180))*np.sin(LAT[0]*(math.pi/180))
d2_adcp = np.cos(LAT*(math.pi/180))*np.cos(LAT[0]*(math.pi/180)) * \
    np.cos(abs(LON[0]-LON)*(math.pi/180))
distance_adcp = 6371*np.arccos(d1_adcp+d2_adcp)

# Calculate orthogonal velocity
a1 = distance_adcp[-1]
b1 = np.sin(LAT[-1]*(math.pi/180))*np.sin(LAT[-1]*(math.pi/180))+np.cos(LAT[-1]*(math.pi/180))*np.cos(LAT[-1]*(math.pi/180))*np.cos(abs(LON[-1]-LON[0])*(math.pi/180))
alpha = np.arccos(b1/a1)
v_ortho = np.cos(alpha)*u_section + np.sin(alpha)*v_section

# Plot the figure
fig, ax = plt.subplots(2, 2, figsize=(20, 16))

# Plot Potential Temperature
PT_plot = ax[0, 0].contourf(distance, depth, PT,levels=20, cmap='cmo.thermal')
fig.colorbar(PT_plot, ax=ax[0, 0], label='Potential Temperature (°C)')

ax[0, 0].set_ylim(0,300)
ax[0, 0].set_xlabel('Distance (km)')
ax[0, 0].set_ylabel('Depth')
ax[0, 0].set_title('Potential Temperature')
ax[0, 0].invert_yaxis()  

# Plot Potential Density
PD_plot = ax[0, 1].contourf(distance, depth, PD, levels=20, cmap='cmo.dense')
fig.colorbar(PD_plot, ax=ax[0, 1], label='Potential Density')

ax[0, 1].set_ylim(0,300)
ax[0, 1].set_xlabel('Distance (km)')
ax[0, 1].set_ylabel('Depth')
ax[0, 1].set_title('Potential Density')
ax[0, 1].invert_yaxis()  

# Plot Salinity
S_plot = ax[1, 0].contourf(distance, depth, S,levels=20, cmap='cmo.haline')
fig.colorbar(S_plot, ax=ax[1, 0], label='Salinity (g/kg)')

ax[1, 0].set_ylim(0,300)
ax[1, 0].set_xlabel('Distance (km)')
ax[1, 0].set_ylabel('Depth')
ax[1, 0].set_title('Salinity')
ax[1, 0].invert_yaxis() 

# Plot Velocity
levels_velocity=np.arange(-1,1,0.1)
V_plot = ax[1, 1].contourf(distance_adcp, depth_adcp, v_ortho,levels=levels_velocity, cmap='coolwarm')
fig.colorbar(V_plot, ax=ax[1, 1], label='Velocity (m/s)')
ax[1, 1].set_ylim(16,300)

ax[1, 1].set_xlabel('Distance (km)')
ax[1, 1].set_ylabel('Depth')
ax[1, 1].set_title('Velocities')
ax[1, 1].invert_yaxis() 

# Add isopycnes contours
contour_levels = np.arange(23, 28, 0.1)
contour_lines = ax[0, 0].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[0, 0].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)
contour_lines = ax[1, 0].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[0, 1].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)
contour_lines = ax[0, 1].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[1, 0].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)
contour_lines = ax[1, 1].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[1, 1].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)

plt.tight_layout()

plt.savefig('profiles_'+name+'.png')


# %% Single Plot

rep = 'C:/Users/J-mar/Documents/MOCIS/Stage M1/Data_MSM40/'

uCTD = scipy.io.loadmat(rep+'UCTD_dataproc.mat')

# Define the section (here section 3)
start_index = 66
end_index = 105

# Create depth array
max_depth = 300
depth = np.arange(0, max_depth)

lon = np.squeeze(uCTD['uctdlon'])[start_index:end_index]
lat = np.squeeze(uCTD['uctdlat'])[start_index:end_index]
PT = np.squeeze(uCTD['PTgrid'])[:max_depth, start_index:end_index]
PD = np.squeeze(uCTD['PDgrid'])[:max_depth, start_index:end_index]
S = np.squeeze(uCTD['Sgrid'])[:max_depth, start_index:end_index]

# Calculate distances
d1 = np.sin(lat*(math.pi/180))*np.sin(lat[0]*(math.pi/180))
d2 = np.cos(lat*(math.pi/180))*np.cos(lat[0]*(math.pi/180)) * \
    np.cos(abs(lon[0]-lon)*(math.pi/180))
distance = 6371*np.arccos(d1+d2)

# Create meshgrid of depth and distance
distance_matrix, depth_matrix = np.meshgrid(distance, depth)


# Plotting
plt.figure(figsize=(10, 6))
plt.contourf(distance_matrix, depth_matrix, PT, cmap='cmo.thermal')
plt.colorbar(label='Temperature')

# Add iso-velocity contours
contour_levels = np.arange(0, 14, 3)
contour_lines = plt.contour(distance_matrix, depth_matrix, PT, levels=contour_levels, colors='k', linewidths=0.3)
plt.clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)

plt.xlabel('Distance (km)')
plt.ylabel('Depth')
plt.title('Temperature Contour Plot')
plt.gca().invert_yaxis()  # Invert y-axis to have depth increasing downwards
plt.show()



# Plot Velocity
levels_velocity=np.arange(-1,1,0.1)
V_plot = ax[1, 1].contourf(distance_adcp, depth_adcp, v_section,levels=levels_velocity, cmap='coolwarm')
fig.colorbar(V_plot, ax=ax[1, 1], label='Velocity (m/s)')
ax[1, 1].set_ylim(16,300)

ax[1, 1].set_xlabel('Distance (km)')
ax[1, 1].set_ylabel('Depth')
ax[1, 1].set_title('Velocities')
ax[1, 1].invert_yaxis() 

# Add isopycnes contours
contour_levels = np.arange(23, 28, 0.1)
contour_lines = ax[0, 0].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[0, 0].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)
contour_lines = ax[1, 0].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[0, 1].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)
contour_lines = ax[0, 1].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[1, 0].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)
contour_lines = ax[1, 1].contour(distance, depth, PD, levels=contour_levels, colors='k', linewidths=0.3)
ax[1, 1].clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)

plt.savefig('profiles_'+name+'.pdf')


