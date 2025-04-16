import numpy as np
import scipy.io
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

rep = 'C:/Users/J-mar/Documents/MOCIS/Stage M1/Data_MSM40/adcp_150kHz/'

# Load data
mat = scipy.io.loadmat(rep + 'TL_GEOMAR_150_STJOHNS_BREST_000_000000_6_hc')

#Extract data
b = mat['b']
bb = b[0,0]
ref = bb[0][0,0][0]
nav = bb[2][0,0][0]
vel = bb[1]
u = vel[:,0,:] # vitesse zonale, positive vers l'est
v = vel[:,1,:] # vitesse méridionale, positive vers le nord
depth = bb[3][0]
LON = nav[1,:]
LAT = nav[2,:]

# Define the section (here section 3)
start_index = np.nanargmax(LAT)
end_index = 15924 

# Slice in the section
LAT_section = LAT[start_index:end_index]
LON_section = LON[start_index:end_index]
u_section = 100*u[:,start_index:end_index]
v_section = 100*v[:,start_index:end_index]

# Calculate distances
d1 = np.sin(LAT_section*(math.pi/180))*np.sin(LAT_section*(math.pi/180))
d2 = np.cos(LAT_section*(math.pi/180))*np.cos(LAT_section*(math.pi/180))*np.cos(abs(LON_section[0]-LON_section)*(math.pi/180))
distance=6371*np.arccos(d1+d2)

# Define the border for the grid
max_depth=np.max(depth)
max_distance=int(np.max(distance))

# Plot the grid
depth, distance = np.meshgrid(depth, distance)
depth = depth.T
distance = distance.T

# Smooth and grid the data (adjust the interpolation method) 
grid_depth, grid_distance = np.mgrid[0:max_depth:100j, 0:max_distance:100j]  # Define grid
grid_velocity = griddata((depth.flatten(), distance.flatten()), v_section.flatten(), (grid_depth, grid_distance), method='linear')      

# Plot the section
plt.figure(figsize=(16, 6))
contour = plt.contourf(grid_distance, grid_depth, grid_velocity, cmap='viridis')  # Contour plot

#Add iso-velocity contours
contour_levels = np.arange(-100, 80, 20)
contour_lines = plt.contour(grid_distance, grid_depth, grid_velocity, levels=contour_levels, colors='k',linewidths=0.3)
plt.clabel(contour_lines, inline=True, fmt='%1.1f', fontsize=12)

plt.colorbar(contour, label='Zonal Velocity')  # Colorbar
plt.xlabel('Distance')
plt.ylabel('Depth')
plt.ylim(16,150)
plt.title('Zonal Velocity Depending on Depth and Distance')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig('v_profile_smooth_gridded.pdf')

plt.show()

#%%

import numpy as np
import scipy.io
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

rep = 'C:/Users/J-mar/Documents/MOCIS/Stage M1/Data_MSM40/adcp_150kHz/'

# Load data
mat = scipy.io.loadmat(rep + 'TL_GEOMAR_150_STJOHNS_BREST_000_000000_6_hc')

#Extract data
b = mat['b']
bb = b[0,0]
ref = bb[0][0,0][0]
nav = bb[2][0,0][0]
vel = bb[1]
u = vel[:,0,:] # vitesse zonale, positive vers l'est
v = vel[:,1,:] # vitesse méridionale, positive vers le nord
depth = bb[3][0]
LON = nav[1,:]
LAT = nav[2,:]

# Define the section 
start_index = np.nanargmax(LAT)     # ADJUST
end_index = 15924                   # ADJUST

# Slice in the section
LAT_section = LAT[start_index:end_index]
LON_section = LON[start_index:end_index]
u_section = 100*u[:,start_index:end_index]
v_section = 100*v[:,start_index:end_index]

# Calculate distances
d1 = np.sin(LAT_section*(math.pi/180))*np.sin(LAT_section*(math.pi/180))
d2 = np.cos(LAT_section*(math.pi/180))*np.cos(LAT_section*(math.pi/180))*np.cos(abs(LON_section[0]-LON_section)*(math.pi/180))
distance=6371*np.arccos(d1+d2)

# Plot the grid
depth, distance = np.meshgrid(depth, distance)
depth = depth.T
distance = distance.T

# Plot the section
plt.figure(figsize=(16, 6))
contour = plt.contourf(distance,depth,v_section, cmap='viridis')  # Contour plot

plt.colorbar(contour, label='Zonal Velocity')  # Colorbar

plt.xlabel('Distance')
plt.ylabel('Depth')
plt.ylim(16,150)
plt.title('Zonal Velocity Depending on Depth and Distance')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig('v_profile.pdf')

plt.show()


#%%

import numpy as np
import scipy.io
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

rep = 'C:/Users/J-mar/Documents/MOCIS/Stage M1/Data_MSM40/adcp_150kHz/'

# Load data
mat = scipy.io.loadmat(rep + 'TL_GEOMAR_150_STJOHNS_BREST_000_000000_6_hc')

#Extract data
b = mat['b']
bb = b[0,0]
ref = bb[0][0,0][0]
nav = bb[2][0,0][0]
vel = bb[1]
u = vel[:,0,:] # vitesse zonale, positive vers l'est
v = vel[:,1,:] # vitesse méridionale, positive vers le nord
depth = bb[3][0]
LON = nav[1,:]
LAT = nav[2,:]

# Define the section 
start_index = np.nanargmax(LAT)     # ADJUST
end_index = 15924                   # ADJUST

# Slice in the section
LAT_section = LAT[start_index:end_index]
LON_section = LON[start_index:end_index]
u_section = 100*u[:,start_index:end_index]
v_section = 100*v[:,start_index:end_index]

# Calculate distances
d1 = np.sin(LAT_section*(math.pi/180))*np.sin(LAT_section*(math.pi/180))
d2 = np.cos(LAT_section*(math.pi/180))*np.cos(LAT_section*(math.pi/180))*np.cos(abs(LON_section[0]-LON_section)*(math.pi/180))
distance=6371*np.arccos(d1+d2)

# Plot the grid
depth, distance = np.meshgrid(depth, distance)
depth = depth.T
distance = distance.T

# Plot the section
plt.figure(figsize=(16, 6))
contour = plt.contourf(distance,depth,v_section, cmap='viridis')  # Contour plot

plt.colorbar(contour, label='Zonal Velocity')  # Colorbar

plt.xlabel('Distance')
plt.ylabel('Depth')
#plt.ylim(16,150)
plt.title('Zonal Velocity Depending on Depth and Distance')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig('v_profile.pdf')

plt.show()



