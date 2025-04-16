#%% Repertoire pour les données

rep=''

#%% Map very simple

import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

# Creat a figure and define projection
fig = plt.figure()
ax = plt.axes(projection=ccrs.Mercator())   

# Limit the map
ax.set_extent([-70,0,45,65]) 

# Add image for land and ocean
ax.stock_img()

# Add coastlines
ax.coastlines()

# Add gridlines
gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='black',linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Add title and labels
plt.title('Different operations done during msm40 cruise')
plt.xlabel('Longitude',loc='left')
plt.ylabel('Latitude',loc='bottom')

# Load data
Data = pd.read_csv(rep+"Carte.csv", sep=';')
    
# Select data
uCTD = Data.loc[Data['Gear']=='uCTD',['Lon','Lat']].to_numpy().T
CTD = Data.loc[Data['Gear']=='CTD/RO',['Lon','Lat']].to_numpy().T
moor_dep = Data.loc[Data['Gear']=='MOOR deployed',['Lon','Lat']].to_numpy().T
moor_rec = Data.loc[Data['Gear']=='MOOR deployed',['Lon','Lat']].to_numpy().T
BPS = Data.loc[Data['Gear']=='BPS',['Lon','Lat']].to_numpy().T
argo = Data.iloc[79,0:2].to_numpy().T

# Plot data
ax.scatter(uCTD[0], uCTD[1], marker='o', color='red', edgecolor='white', s=50, label='uCTD', transform=ccrs.Geodetic())
ax.scatter(CTD[0], CTD[1], marker='*', color='yellow', edgecolor='black', linewidths=1, s=200,label='CTD', transform=ccrs.Geodetic())
ax.scatter(moor_dep[0], moor_dep[1], marker='s', color='lime', edgecolor='black', linewidths=1, s=200,label='deployed moorings', transform=ccrs.Geodetic())
ax.scatter(moor_rec[0], moor_rec[1], marker='s', color='magenta', edgecolor='black', linewidths=1, s=50,label='recovered moorings', transform=ccrs.Geodetic())
ax.scatter(BPS[0], BPS[1], marker='<', color='white', edgecolor='black', linewidths=1, s=200,label='BPS', transform=ccrs.Geodetic())
ax.scatter(argo[0], argo[1], marker='1', color='black', s=200, linewidths=4, label='argo', transform=ccrs.Geodetic())

ax.legend(fontsize=8)

# Plot St Johns and Brest
sj_lat = 47.5605413
sj_lon =-52.7128315
brest_lat = 48.390394
brest_lon = -4.486076

ax.scatter([sj_lon, brest_lon], [sj_lat, brest_lat], color='black',s=50, linewidth=2, marker='o',transform=ccrs.Geodetic())
ax.text(sj_lon + 3, sj_lat - 1, 'St. Johns', horizontalalignment='left', transform=ccrs.Geodetic(), fontsize=10, weight='bold')
ax.text(brest_lon - 2, brest_lat, 'Brest', horizontalalignment='right', transform=ccrs.Geodetic(), fontsize=10, weight='bold')

plt.tight_layout()

plt.savefig('carte_basse_resolution.pdf')

#%% Map using cartopy and Natural Earth alternative version

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Plot the map
ax = plt.axes(projection=ccrs.Mercator())

# Set extent of the plot
ax.set_extent([-70,0,45,65])

# Define the color level
cmap = plt.get_cmap('Blues')
norm = colors.Normalize(vmin=0, vmax=5000)

# Plot the bathymetry data
for letter, level in [
                      ('L', 0),
                      ('K', 200),
                      ('J', 1000),
                      ('I', 2000),
                      ('H', 3000),
                      ('G', 4000),
                      ('F', 5000),]:
    bathym = cfeature.NaturalEarthFeature(name='bathymetry_{}_{}'.format(letter, level),
                                 scale='10m', category='physical')
    ax.add_feature(bathym,facecolor=cmap(norm(level)), edgecolor='face',zorder=1)

# Add land feature
ax.add_feature(cfeature.LAND, edgecolor='k', facecolor='lightgray',zorder=1)

# Add title and labels
plt.title('Different operations done during msm40 cruise')
plt.xlabel('Longitude',loc='left')
plt.ylabel('Latitude',loc='bottom')

# Add color bar
cbar = plt.colorbar(c, ax=ax, orientation='vertical', fraction=0.05)
cbar.set_label('Depth (m)')

# Load data
Data = pd.read_csv(rep+"Carte.csv", sep=';')
    
# Select data
uCTD = Data.loc[Data['Gear']=='uCTD',['Lon','Lat']].to_numpy().T
CTD = Data.loc[Data['Gear']=='CTD/RO',['Lon','Lat']].to_numpy().T
moor_dep = Data.loc[Data['Gear']=='MOOR deployed',['Lon','Lat']].to_numpy().T
moor_rec = Data.loc[Data['Gear']=='MOOR deployed',['Lon','Lat']].to_numpy().T
BPS = Data.loc[Data['Gear']=='BPS',['Lon','Lat']].to_numpy().T
argo = Data.iloc[79,0:2].to_numpy().T

# Plot data
ax.scatter(uCTD[0], uCTD[1], marker='o', color='red', edgecolor='white', s=50, label='uCTD', transform=ccrs.Geodetic())
ax.scatter(CTD[0], CTD[1], marker='*', color='yellow', edgecolor='black', linewidths=1, s=200,label='CTD', transform=ccrs.Geodetic())
ax.scatter(moor_dep[0], moor_dep[1], marker='s', color='lime', edgecolor='black', linewidths=1, s=200,label='deployed moorings', transform=ccrs.Geodetic())
ax.scatter(moor_rec[0], moor_rec[1], marker='s', color='magenta', edgecolor='black', linewidths=1, s=50,label='recovered moorings', transform=ccrs.Geodetic())
ax.scatter(BPS[0], BPS[1], marker='<', color='white', edgecolor='black', linewidths=1, s=200,label='BPS', transform=ccrs.Geodetic())
ax.scatter(argo[0], argo[1], marker='1', color='black', s=200, linewidths=4, label='argo', transform=ccrs.Geodetic())

ax.legend(fontsize=8)

# Plot St Johns and Brest
sj_lat = 47.5605413
sj_lon =-52.7128315
brest_lat = 48.390394
brest_lon = -4.486076

ax.scatter([sj_lon, brest_lon], [sj_lat, brest_lat], color='black',s=50, linewidth=2, marker='o',transform=ccrs.Geodetic())
ax.text(sj_lon + 3, sj_lat - 1, 'St. Johns', horizontalalignment='left', transform=ccrs.Geodetic(), fontsize=10, weight='bold')
ax.text(brest_lon - 2, brest_lat, 'Brest', horizontalalignment='right', transform=ccrs.Geodetic(), fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig('carte_natural_earth_2.pdf')
    
#%% Map using cartopy and feature from Natural Earth in detail

import pandas as pd
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

def load_bathymetry(zip_file_url):
    """Read zip file from Natural Earth containing bathymetry shapefiles"""
    # Download and extract shapefiles
    import io
    import zipfile

    import requests
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("ne_10m_bathymetry_all/")

    # Read shapefiles, sorted by depth
    shp_dict = {}
    files = glob('ne_10m_bathymetry_all/*.shp')
    assert len(files) > 0
    files.sort()
    depths = []
    for f in files:
        depth = '-' + f.split('_')[-1].split('.')[0]  # depth from file name
        depths.append(depth)
        bbox = (90, -15, 160, 60)  # (x0, y0, x1, y1)
        nei = shpreader.Reader(f, bbox=bbox)
        shp_dict[depth] = nei
    depths = np.array(depths)[::-1]  # sort from surface to bottom
    return depths, shp_dict


if __name__ == "__main__":
    # Load data (14.8 MB file)
    depths_str, shp_dict = load_bathymetry(
        'https://naturalearth.s3.amazonaws.com/' +
        '10m_physical/ne_10m_bathymetry_all.zip')

    # Construct a discrete colormap with colors corresponding to each depth
    depths = depths_str.astype(int)
    N = len(depths)
    nudge = 0.01  # shift bin edge slightly to include data
    boundaries = [min(depths)] + sorted(depths+nudge)  # low to high
    norm = matplotlib.colors.BoundaryNorm(boundaries, N)
    blues_cm = matplotlib.colormaps['Blues_r'].resampled(N)
    colors_depths = blues_cm(norm(depths))

    # Set up plot
    subplot_kw = {'projection': ccrs.Mercator()}
    fig, ax = plt.subplots(subplot_kw=subplot_kw, figsize=(9, 7))
    ax.set_extent([-70,0,45,65])  # x0, x1, y0, y1
    
    # Add title and labels
    plt.title('Different operations done during msm40 cruise')
    plt.xlabel('Longitude',loc='left')
    plt.ylabel('Latitude',loc='bottom')


    # Iterate and plot feature for each depth level
    for i, depth_str in enumerate(depths_str):
        ax.add_geometries(shp_dict[depth_str].geometries(),
                          crs=ccrs.PlateCarree(),
                          color=colors_depths[i],zorder=1)

    # Add standard features
    ax.add_feature(cfeature.LAND, color='grey')
    gl = ax.gridlines(draw_labels=True,linewidth=0.5,color='black',linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    ax.set_position([0.03, 0.05, 0.8, 0.9])

    # Add custom colorbar
    axi = fig.add_axes([0.85, 0.1, 0.025, 0.8])
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    sm = plt.cm.ScalarMappable(cmap=blues_cm, norm=norm)
    fig.colorbar(mappable=sm,
                 cax=axi,
                 spacing='proportional',
                 extend='min',
                 ticks=depths,
                 label='Depth (m)')
    
    # Load data
    Data = pd.read_csv(rep+"Carte.csv", sep=';')
 
    # Select the variables   
    uCTD = Data.loc[Data['Gear']=='uCTD',['Lon','Lat']].to_numpy().T
    CTD = Data.loc[Data['Gear']=='CTD/RO',['Lon','Lat']].to_numpy().T
    moor_dep = Data.loc[Data['Gear']=='MOOR deployed',['Lon','Lat']].to_numpy().T
    moor_rec = Data.loc[Data['Gear']=='MOOR deployed',['Lon','Lat']].to_numpy().T
    BPS = Data.loc[Data['Gear']=='BPS',['Lon','Lat']].to_numpy().T
    argo = Data.iloc[79,0:2].to_numpy().T

    # Plot the variables
    ax.scatter(uCTD[0], uCTD[1], marker='o', color='red', edgecolor='white', s=100, label='uCTD', transform=ccrs.Geodetic())
    ax.scatter(CTD[0], CTD[1], marker='*', color='yellow', edgecolor='black', linewidths=1, s=350,label='CTD', transform=ccrs.Geodetic())
    ax.scatter(moor_dep[0], moor_dep[1], marker='s', color='lime', edgecolor='black', linewidths=1, s=350,label='deployed moorings', transform=ccrs.Geodetic())
    ax.scatter(moor_rec[0], moor_rec[1], marker='s', color='magenta', edgecolor='black', linewidths=1, s=100,label='recovered moorings', transform=ccrs.Geodetic())
    ax.scatter(BPS[0], BPS[1], marker='<', color='white', edgecolor='black', linewidths=1, s=350,label='BPS', transform=ccrs.Geodetic())
    ax.scatter(argo[0], argo[1], marker='1', color='black', s=350, linewidths=4, label='argo', transform=ccrs.Geodetic())

    ax.legend(fontsize=15)

    # Plot St Johns and Brest
    sj_lat = 47.5605413
    sj_lon =-52.7128315
    brest_lat = 48.390394
    brest_lon = -4.486076

    ax.scatter([sj_lon, brest_lon], [sj_lat, brest_lat], color='black',s=150, linewidth=2, marker='o',transform=ccrs.Geodetic())   
    ax.text(sj_lon + 6, sj_lat - 2, 'St. Johns', horizontalalignment='right', transform=ccrs.Geodetic(), fontsize=20, weight='bold')
    ax.text(brest_lon - 1, brest_lat, 'Brest', horizontalalignment='right', transform=ccrs.Geodetic(), fontsize=20, weight='bold')
    
    # Convert vector bathymetries to raster (saves a lot of disk space)
    # while leaving labels as vectors
    ax.set_rasterized(True)
    
    plt.savefig('carte_natural_earth.pdf')

#%% Map using etopo dataset with zoom

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import pandas as pd

# Global map

# Load bathymetric data (example using NetCDF file)
bathymetry_data = xr.open_dataset(rep+'bathy/etopo2.nc')

# Extract relevant variables (e.g., longitude, latitude, depth)
lon = bathymetry_data['lon']
lat = bathymetry_data['lat']
depth = bathymetry_data['topo']

# Define region of interest
lon_min, lon_max, lat_min, lat_max = -70, 0, 45, 65

# Subset data to region of interest
lon_subset = lon.sel(lon=slice(lon_min, lon_max))
lat_subset = lat.sel(lat=slice(lat_min, lat_max))
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
ax.scatter(uCTD[0], uCTD[1], marker='o', color='red', edgecolor='white', s=100, label='uCTD', transform=ccrs.Geodetic())
ax.scatter(CTD[0], CTD[1], marker='*', color='yellow', edgecolor='black', linewidths=1, s=350,label='CTD', transform=ccrs.Geodetic())
ax.scatter(moor_dep[0], moor_dep[1], marker='s', color='lime', edgecolor='black', linewidths=1, s=350,label='deployed moorings', transform=ccrs.Geodetic())
ax.scatter(moor_rec[0], moor_rec[1], marker='s', color='magenta', edgecolor='black', linewidths=1, s=100,label='recovered moorings', transform=ccrs.Geodetic())
ax.scatter(BPS[0], BPS[1], marker='<', color='white', edgecolor='black', linewidths=1, s=350,label='BPS', transform=ccrs.Geodetic())
ax.scatter(argo[0], argo[1], marker='1', color='black', s=350, linewidths=4, label='argo', transform=ccrs.Geodetic())

ax.legend(fontsize=20)

# Place two point of reference (St Johns and Brest)
sj_lat = 47.5605413
sj_lon =-52.7128315
brest_lat = 48.390394
brest_lon = -4.486076

ax.scatter([sj_lon, brest_lon], [sj_lat, brest_lat], color='black',s=150, linewidth=2, marker='o',transform=ccrs.Geodetic())
ax.text(sj_lon + 6, sj_lat - 2, 'St. Johns', horizontalalignment='right', transform=ccrs.Geodetic(), fontsize=20, weight='bold')
ax.text(brest_lon - 1, brest_lat, 'Brest', horizontalalignment='right', transform=ccrs.Geodetic(), fontsize=20, weight='bold')



# Zoom

# Zoomed-in region
zoom_region = [-54, -49, 52, 54]  # [lon_min, lon_max, lat_min, lat_max]

# Define the position and size of the zoomed-in region
zoom_position = [0.4, 0.2, 0.2, 0.2]  # [left, bottom, width, height]

# Draw the zoomed-in region box
ax.plot([zoom_region[0], zoom_region[1], zoom_region[1], zoom_region[0], zoom_region[0]],
        [zoom_region[2], zoom_region[2], zoom_region[3], zoom_region[3], zoom_region[2]],
        color='black', linewidth=2, transform=ccrs.PlateCarree())

# Zoomed-in region
zoom_region = [-54, -49, 52, 54]  # [lon_min, lon_max, lat_min, lat_max]
zoom_position = [0.35, 0.2, 0.25, 0.3]  # [left, bottom, width, height]
ax_zoom = fig.add_axes(zoom_position, projection=ccrs.PlateCarree())

# Plot bathymetric relief for zoomed-in region
ax_zoom.contourf(lon_subset, lat_subset, depth_subset, transform=projection, cmap='Blues_r', zorder=0)

# Add land feature for zoomed-in region
ax_zoom.add_feature(cfeature.LAND, edgecolor='k', facecolor='lightgray', zorder=1)

# Add coastlines and other features for zoomed-in region
ax_zoom.coastlines(resolution='10m', zorder=2)

# Draw the zoomed-in region box on the global map
ax_zoom.plot([zoom_region[0], zoom_region[1], zoom_region[1], zoom_region[0], zoom_region[0]],
              [zoom_region[2], zoom_region[2], zoom_region[3], zoom_region[3], zoom_region[2]],
              color='black', linewidth=3, transform=ccrs.PlateCarree())

# Set extent for the zoomed-in region
ax_zoom.set_extent(zoom_region, crs=ccrs.PlateCarree())

# Plot the measurements on the zoom-in region
ax_zoom.scatter(uCTD[0], uCTD[1], marker='o', color='red', edgecolor='white', s=100, label='uCTD', transform=ccrs.Geodetic())
ax_zoom.scatter(CTD[0], CTD[1], marker='*', color='yellow', edgecolor='black', linewidths=1, s=350,label='CTD', transform=ccrs.Geodetic())
ax_zoom.scatter(moor_dep[0], moor_dep[1], marker='s', color='lime', edgecolor='black', linewidths=1, s=350,label='deployed moorings', transform=ccrs.Geodetic())
ax_zoom.scatter(moor_rec[0], moor_rec[1], marker='s', color='magenta', edgecolor='black', linewidths=1, s=100,label='recovered moorings', transform=ccrs.Geodetic())
ax_zoom.scatter(BPS[0], BPS[1], marker='<', color='white', edgecolor='black', linewidths=1, s=350,label='BPS', transform=ccrs.Geodetic())
ax_zoom.scatter(argo[0], argo[1], marker='1', color='black', s=350, linewidths=4, label='argo', transform=ccrs.Geodetic())

plt.savefig('carte.png')

#%% Map using etopo dataset without zoom

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr


# Load bathymetric data (example using NetCDF file)
bathymetry_data = xr.open_dataset(rep+'/Données bathy/etopo2.nc')

# Extract relevant variables (e.g., longitude, latitude, depth)
lon = bathymetry_data['lon']
lat = bathymetry_data['lat']
depth = bathymetry_data['topo']

# Define region of interest
lon_min, lon_max, lat_min, lat_max = -70, 0, 45, 65

# Subset data to region of interest
lon_subset = lon.sel(lon=slice(lon_min, lon_max))
lat_subset = lat.sel(lat=slice(lat_min, lat_max))
depth_subset = depth.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))

# Define map projection
projection = ccrs.PlateCarree()

# Create a plot
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 1, 1, projection=projection, aspect='auto')

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

# Add color bar
cbar = plt.colorbar(c, ax=ax, orientation='vertical', fraction=0.05)
cbar.set_label('Depth (m)')

# Add title and labels
plt.title('Bathymetric Relief of the Ocean Bottom')
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
ax.scatter(uCTD[0], uCTD[1], marker='o', color='red', edgecolor='white', s=100, label='uCTD', transform=ccrs.Geodetic())
ax.scatter(CTD[0], CTD[1], marker='*', color='yellow', edgecolor='black', linewidths=1, s=350,label='CTD', transform=ccrs.Geodetic())
ax.scatter(moor_dep[0], moor_dep[1], marker='s', color='lime', edgecolor='black', linewidths=1, s=350,label='deployed moorings', transform=ccrs.Geodetic())
ax.scatter(moor_rec[0], moor_rec[1], marker='s', color='magenta', edgecolor='black', linewidths=1, s=100,label='recovered moorings', transform=ccrs.Geodetic())
ax.scatter(BPS[0], BPS[1], marker='<', color='white', edgecolor='black', linewidths=1, s=350,label='BPS', transform=ccrs.Geodetic())
ax.scatter(argo[0], argo[1], marker='1', color='black', s=350, linewidths=4, label='argo', transform=ccrs.Geodetic())

ax.legend(fontsize=20)

# Place two point of reference (St Johns and Brest)
sj_lat = 47.5605413
sj_lon =-52.7128315
brest_lat = 48.390394
brest_lon = -4.486076

ax.scatter([sj_lon, brest_lon], [sj_lat, brest_lat], color='black',s=150, linewidth=2, marker='o',transform=ccrs.Geodetic())
ax.text(sj_lon + 6, sj_lat - 2, 'St. Johns', horizontalalignment='right', transform=ccrs.Geodetic(), fontsize=20, weight='bold')
ax.text(brest_lon - 1, brest_lat, 'Brest', horizontalalignment='right', transform=ccrs.Geodetic(), fontsize=20, weight='bold')


#%% Carte zoom�e section 5

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



