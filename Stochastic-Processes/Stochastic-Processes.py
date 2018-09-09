# -*- coding: utf-8 -*-
"""
Created on Sun May 13 21:43:57 2018

@author: Cemenenkoff
This code explores the physics of diffusion-limited aggregation (DLA) cluster
growth and is closely based on chapter 7 of Computational Physics by Nicholas
Giordano and Hisao Nakanishi (2nd Edition), pp. 181-211.

Directly from Covey's problem statement:
    Characterize the structure and fractal dimensionality of 2-D DLA clusters
    grown with walkers that begin their walks from a preferred direction
    (i.e., lattice points along the x-axis). Contrast this with walkers that
    begin their walk from no preferred direction, meaning the walkers spawn
    randomly from a circle encompassing the cluster.
    
For purposes of visualization, I think of the growing DLA clusters in this code
as "soot clumps". There is a seed soot particle and we are moving along with it
in its frame, so it appears stationary. Randomly moving smaller surrounding 
soot particles will eventually bump into the seed particle and stick (causing
the clump to grow). This code was written with this specific image in mind.

There are 4 main sections:
    1. Numerical Parameters and Switchboard
        You can adjust which plots are calculated and/or displayed as well as
        their important numerical properties in this section.
    2. Main Functions
        Workhorse functions used in creating the figures are defined here.
    3. Data Generation
        The main code that generates the data of interest is found here.
    4. Figures
        Details for each plot (annotations, labels, etc.) are found here.
"""
#Covey's preamble with a few customizations.
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches #Import mpatches for plot legends.
import numpy as np
plt.style.use('classic') #Use a serif font.
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 10 #Make figures square by default.
plt.rcParams['axes.labelsize'] = 18+4
plt.rcParams['axes.titlesize'] = 20+4
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 6
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor']='white'
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

###############################################################################
# 1. Numerical Parameters and Switchboard #####################################
###############################################################################
#n is the side length of square grid (odd integer).
n = 301
#Choose the maximum number of steps per walker (integer). Note that n*sqrt(2)
#is half the length of the square grid diagonal.
steps = int(n*np.sqrt(2))

#Choose between 'line' or 'circle' for how the walkers spawn. See gen_clump().
spawntype = 'circle'

#Choose whether or not to stop walkers who stray too far to avoid excessively
#long walks.
stop = True

#nwalkers is the number of walkers (integer).
nwalkers = 50*n

#Decide which figures to generate.
gen_fig1 = True #Randomly Generated Soot Clump
gen_fig2 = True #Walker Spawn Positions
gen_fig3 = True #Log-Log Plot of Contained Mass vs Radius from Seed Location

###############################################################################
# 2. Main Functions ###########################################################
###############################################################################
"""
Purpose:
    gen_clump() performs a diffusion-limited aggregation (DLA) random walk to
    simulate how a soot clump grows when soot particles spawn from a preferred
    direction ('line') or no preferred direction ('circle'). In both options,
    particles spawn from a random location on the spawn range. The 'line'
    option has the spawn range just below the clump so as not to spawn
    particles directly on top of the clump. As the clump grows, the spawn range
    widens and moves to stay below the clump dynamically. The 'circle' option
    makes the spawn range a circle that encloses the clump (without touching
    it) that expands while the clump grows.
Inputs:
    [0] n = side length of the square grid available for random walkers
        (odd integer)
    [1] steps = max number of steps each walker may take (integer)
    [2] nwalkers = number of random walkers (integer)
    [3] stop = decide if walkers that stray too far should stop (boolean)
    [4] spawntype = choose between a circular or a linear spawn pattern for the
                    walkers: 'circle' or 'line' (string)
Outputs:
    [0] g = square grid containing a DLA soot clump (2D numpy array)
    [1] x0 = x-location of seed particle (integer)
    [2] y0 = y-location of seed particle (integer)
    [3] width = width of the soot clump (integer)
    [4] height = height of the soot clump (integer)
    [5] h_lo = y-location of the lowest point of the soot clump (integer)
    [6] h_hi = y-location of the tallest point of the soot clump (integer)
    [7] w_lo = x-location of the leftmost point of the soot clump (integer)
    [8] w_hi = x-location of the rightmost point of the soot clump (integer)
    [9] offgrids = number of walkers that walk off of the grid
   [10] x_lo = x-location of the 'line' particle spawn range (integer)
   [11] x_hi = y-location of the 'line' particle spawn range (integer)
   [12] kills = number of walkers terminated because they strayed too far from
                the clump or were abandoned (integer)
   [13] size = total number of soot particles in the clump (integer)
   [14] radius = radius of 'circle' particle spawn range (integer)
   [15] spawn_x = x-positions of walker spawn locations (list of integers)
   [16] spawn_y = y-positions of walker spawn location (list of integers)
"""
def gen_clump(n, nwalkers, steps, stop, spawntype):
    #mid is the location of middle coordinate (odd integer).
    mid = n//2
    
    #Initialize a square grid (g). Each empty location is denoted by a 0.
    #g[i,j] denotes row, column (i.e. [y,x]). The upper left corner is g[0,0]
    #and the lower right corner is g[n-1,n-1].
    g = np.zeros((n,n))
    x0 = mid #Horizontally center the seed.
    if spawntype == 'line':
        y0 = int((9/10)*n) #Center the seed near the top of the grid.
    if spawntype == 'circle':
        y0 = mid #Center the seed in the middle.
    g[y0,x0] = 1 #Change the seed location to a 1.
    
    #The following places on the grid represent spots on the perimeter a
    #soot particle could bind to. Label these bind-able spots with a 2.
    g[y0,x0+1] = 2 #up-neighbor
    g[y0,x0-1] = 2 #down-neighbor
    g[y0+1,x0] = 2 #left-neighbor
    g[y0-1,x0] = 2 #right-neighbor
    
    #Initialize a running count of the width of the clump. At first, the
    #width is just 1, but as it is updated it will be defined as w_hi - w_lo.
    width = 1
    w_hi = x0
    w_lo = x0
    
    #Do something similar for the height.
    height = 1
    h_hi = y0
    h_lo = y0
    
    #Initialize a radius for the circular spawn. If spawntype='line' is chosen,
    #then this value won't increment.
    radius = 1
    
    #Initialize a counter to track how many walkers end off of the grid.
    offgrids = 0

    #We want to track the total number of particles in the clump.
    size = 1
    
    #Set a limit for how far a walker may travel without hitting the clump 
    #before terminating its walk. Track how many walkers are killed.
    kills = 0        
    if spawntype == 'line':
        distcap = 70*height
    elif spawntype == 'circle':
        distcap = int(70*radius)+5
    
    #We want to track all spawn positions for debugging. If truly random,
    #the linearly-generated spawns should map out something vaguely triangular
    #as the spawn line moves downward, and the circularly-generated spawns
    #should map out somethingly vaguely circular as the spawn radius increases.
    spawn_x = []
    spawn_y = []
    
    #Set the growth rate for the variable-width line spawn.
    wrat = 2
    
    #Ensure the spawn is always offset from the growing clump by an appropriate
    #amount (so particles don't spawn on the clump itself).
    offset = 5
    for j in range(nwalkers): #For each walker, do the following:
        if spawntype == 'circle':
            #For the circular spawn, set the largest horizontal or vertical
            #extension of the clump w.r.t. the origin to be the radius of the
            #spawn circle.
            radius_choices = [(w_hi-x0), (w_lo-x0), (h_hi-y0), (h_lo-y0)]
            radius_choices = [abs(i) for i in radius_choices]
            radius = max(radius_choices)+offset
            
            #Define an x-range for the spawn circle. Technically, if 'circle'
            #is chosen, this range isn't required to be able to show the
            #circular spawn on the plot, but since x_lo and x_hi are required
            #outputs of gen_clump(), we might as well make them accurately
            #describe the circle's x-range.
            x_lo = mid-radius
            x_hi = mid+radius
            
            """
            #This is the first (incorrect) way attempted at coding the circular
            #spawn. It is left here as a good example of a subtle error.
            #Choose a random value from the x-range.
            x = np.random.randint(x_lo,x_hi+1) #Make sure x_hi is included.
            #Given a random x, randomly jump to either the top or bottom half
            #of the spawn circle.
            sign = np.random.choice([-1,1])
            y = int(y0+sign*np.sqrt(radius**2 - (x-x0)**2))
            """
            
            #One might be tempted to choose a random value from the x-range,
            #and then plug that random value into a piecewise-defined equation
            #for a circle (randomly choosing a piece), but this sampling method
            #does not ensure walkers spawn along the circle unformly. y-values
            #associated with the endpoints of the x-range will be sampled less.
            #To circumvent this, define x and y in terms of the current radius
            #of the circle and a uniformly drawn random number in [0,2pi).
            theta = np.random.uniform(0,2*np.pi)
            x = int(radius*np.cos(theta) + x0)
            y = int(radius*np.sin(theta) + y0)
            
            #If the diameter of the spawn circle exceeds the side length of the
            #grid, there must be a large clump grown, so stop sending walkers.
            if 2*radius >= n-1:
                #Account for the abandoned walkers.
                kills = (nwalkers - j) + kills
                break
        elif spawntype == 'line':
            #For the linear spawn, the starting x-location of each soot
            #particle is in a random grid location over the center of the clump
            #seed and grows with the width of the clump.
            x_lo = int(mid - wrat*width)
            x_hi = int(mid + wrat*width)
            x = np.random.randint(x_lo,x_hi+1) #Make sure x_hi is included. 
        
            #Adjust the starting y-location of the each soot particle to be an
            #offset number of units away from the minimum height of the clump.
            y = int(h_lo-offset)
        
            #If the spawn position looks like it's about to go off of the grid,
            #there must be a large clump grown, so stop sending walkers.
            if y <= 1:
                #Account for the abandoned walkers.
                kills = (nwalkers - j) + kills
                break
        #Append the spawn position to the appropriate lists.
        spawn_x.append(x)
        spawn_y.append(y)
        
        #Start a counter to track the total distance traveled for each walker.
        dist = 0
        #For each step the current walker will take, do the following:
        for i in range(steps):
            #Draw a random number from a uniform distribution over [0, 1).
            r = np.random.rand()
            #If ever the total distance traveled exceeds the max allowed, stop
            #stepping with this walker and move to the next one.
            if stop == True:
                if dist > distcap:
                    kills+=1
                    break

            if r < 0.25: #If r is in [0,0.25), step left.
                x -= 1         
            if 0.25 <= r < 0.5: #If r is in [0.25,0.5), step right.
                x += 1
            if 0.5 <= r < 0.75: #If r is in [0.5,0.75), step down.
                y -= 1
            if 0.75 <= r < 1.0: #If r is in [0.75,1.0), step up.
                y += 1
            #Every time a step is taken, increment the total distance traveled
            #for the current walker.
            dist += 1
            
            #After its step, if the current walker is within the grid, 
            if (0 <= y < n-1) and (0 <= x < n-1):
                if g[y,x] == 2:#and if currently on a perimeter state,
                    g[y,x] = 1 #then change the grid point to a bound state,
                    size += 1 #and keep a tally of how large the clump is.
                    
                    #If moving the current state right or left bumps into a
                    #clump particle, if the current x-position is greater or
                    #less than the max or min (respectively) x-position of the
                    #clump, adjust the running max (w_hi) or min (w_lo)
                    #x-location of the clump (which will increase width).
                    if g[y,x+1] == 1 or g[y,x-1] == 1:
                        if x > w_hi:
                            w_hi += 1
                        if x < w_lo:
                            w_lo -= 1
                        width = w_hi-w_lo
                    
                    #In a similar fashion, if moving the current state up or
                    #down bumps into a clump particle, adjust the max (h_hi) or
                    #min (h_lo) y-location (which will then increase height).
                    if g[y+1,x] == 1 or g[y-1,x] == 1:
                        if y > h_hi:
                            h_hi += 1
                        if y < h_lo:
                            h_lo -= 1
                        height = h_hi-h_lo
                        
                    #After the walker encounters a perimeter state and changes
                    #it into a bound one, check its up-, down-, left-, and
                    #right-neighbors once again. If they are empty states,
                    #change them to perimeter states.
                    if g[y,x+1] == 0:
                        g[y,x+1] = 2
                    if g[y,x-1] == 0:
                        g[y,x-1] = 2
                    if g[y+1,x] == 0:
                        g[y+1,x] = 2
                    if g[y-1,x] == 0:
                        g[y-1,x] = 2
                    break
                        
        #If the current walker ends up off the grid after taking all of its
        #steps, increment a tally of how many walkers end up totally far away
        #from our zone of interest.
        if y <0 or y >= n or x < 0 or x >= n:
            offgrids+=1

    return g, x0, y0, width, height, h_lo, h_hi, w_lo, w_hi, offgrids, x_lo, \
           x_hi, kills, size, radius, spawn_x, spawn_y

"""
Purpose:
    check_in() checks if a point (x,y) lies within a circle of radius r
    centered at (h,k).
Inputs:
    [0] x = x-position of point (float)
    [1] y = y-position of point (float)
    [2] h = x-position of circle center (float)
    [3] k = y-position of circle center (float)
    [4] r = radius of circle to check within (float)
Output:
    [0] True or False (boolean)
"""
def check_in(x, y, h, k, r):
    #Calculate the distance from the center of the circle to the point
    dist = np.sqrt((x-h)**2 + (y-k)**2)   
    #Check if the distance is less radius of the circle.
    if dist < r:
        return True
    else:
        return False

"""
Purpose:
    Assuming all of a clump's particles have the same mass, find_df() creates
    successively larger rings emanating from the center of a clump and counts
    the number of clump particles within each ring with check_in(). These
    functions allow us to create a log(m) vs log(r) plot wherein we may
    linearly interpolate over a reasonable range to infer how the mass of a
    clump grows with radius from the center if the clump were infinitely large.
    In other words, find_df() uses a linear regression model to estimate the
    soot clump's fractal dimensionality.

Inputs:
    [0] g = square grid containing a DLA soot clump (2D numpy array)
    [1] steps = max number of steps each walker may take (integer)
    [2] x0 = x-location of seed particle (integer)
    [3] y0 = y-location of seed particle (integer)
Outputs:
    [0] m = slope of linear regression model (i.e. fractal dimensionality of
            the soot clump)  (float)
    [1] b = y-intercept of linear regression model (float)
    [2] logr = natural log of radii trimmed to within the fit bounds
               (numpy array)
    [3] logm_best = natural log of masses trimmed to within the fit bounds
                    (numpy array)
    [4] radii = untrimmed array of radii (numpy array)
    [5] masses = untrimmed array of mass data associated with radii
                 (numpy array)
"""
def find_df(g, steps, x0, y0):
    #Get the side length of the square input grid.
    n = len(g)
    
    #Create a list to store total mass as a function of radius from the seed
    #location.
    masses = []
    #Create a list of radii from 1 to the maximum length a walker could walk
    #if it traveled in a straight line from the seed location.
    radii = list(range(1, steps))
    
    for r in range(1, steps): #For each nonzero radius,
        mass = 0 #initialize the mass at zero,
        for i in range(n): #then loop through each lattice point on the grid.
            for j in range(n):
                if g[j,i] == 1: #If a lattice point represents a clump state,
                                #see if it lies within the current radius.
                    if check_in(j, i, y0, x0, r) == True:
                        mass += 1 #If it does, increment the mass associated
                                  #with the current radius.
        #After looping through all lattice points on the grid, append the mass
        #for the current radius to the masses list and then move onto the next
        #(larger) radius.
        masses.append(mass)
    
    #Change the masses and radii lists to numpy arrays so they are more
    #friendly with linear algebra.
    masses = np.asarray(masses)
    radii = np.asarray(radii)
    
    from scipy import stats
    #Since the clump is of finite size, its total mass will be the mode of
    #the masses array. This is because after a certain point, larger and larger
    #radii will count the same number of clump particles within their
    #associated circles because they contain the entire clump.
    mmax = stats.mode(masses)[0][0] #0th element of the 0th structure returned.
    
    #Next, we find the radius that first encloses the clump entirely by
    #looping through the mass array, logging the index of when mmax is first
    #hit, and then breaking out of the loop.
    for i in range(len(masses)):
        if masses[i]==mmax:
            rmax=radii[i] #rmax is the radius associated with the maximum mass.
            break
    
    #We want to limit the linear fitting to radii that are well below the
    #maximum radius, so trim the logm and logr arrays (to be fit) to only
    #contain values associated with radii between two different precentages of
    #rmax. Skip the first few pairs of values as well to ensure an accurate
    #estimate.
    ceil = int(0.65*rmax)
    floor = int(0.15*rmax)
    logm = np.log(masses[floor:ceil])
    logr = np.log(radii[floor:ceil])
    
    #Concatenate the logr data and a list of 1s (and then take the tranpose of
    #the result) to create a design matrix A. Note the columns of this design
    #matrix are the reverse of the usual (i.e. the column of 1s is usually
    #first) so that the first parameter to be returned in the solution array to
    #the normal equations is the best-fit slope, m, rather than the best-fit
    #y-intercept, b.
    A = np.vstack([logr, np.ones(len(logr))]).T
    
    #Solve the normal equations associated with the data: ((A'A)^-1)*(A'y)
    #Note the first element in the returned array is the slope, m, and the
    #second returned element is the y-intercept, b.
    m, b = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, logm))
    
    #Define a linear function log(r) with the best-fit parameters. Note we fit
    #the equation m=r^(df) by taking the log of both sides to linearize it.
    #Define the equation of the line of best fit.
    logm_bestfit_fn = lambda logr: m*logr+b
    logm_fit = logm_bestfit_fn(logr) #Generate the best-fit line data.
    return m, b, logr, logm_fit, radii, masses

###############################################################################
# 3. Data Generation ##########################################################
###############################################################################
data_clump = gen_clump(n, nwalkers, steps, stop, spawntype)
g         = data_clump[0]
x0        = data_clump[1]
y0        = data_clump[2]
width     = data_clump[3]
height    = data_clump[4]
h_lo      = data_clump[5]
h_hi      = data_clump[6]
w_lo      = data_clump[7]
w_hi      = data_clump[8]
offgrids  = data_clump[9]
x_lo      = data_clump[10]
x_hi      = data_clump[11]
kills     = data_clump[12]
size      = data_clump[13]
radius    = data_clump[14]
spawn_x   = data_clump[15]
spawn_y   = data_clump[16]

data_df = find_df(g, steps, x0, y0)
m        = data_df[0]
b        = data_df[1]
logr     = data_df[2]
logm_fit = data_df[3]
radii    = data_df[4]
masses   = data_df[5]

###############################################################################
# 4. Figures ##################################################################
###############################################################################
if gen_fig1 == True:
    fig1 = plt.figure(1, facecolor='white')
    ax1 = fig1.add_subplot(111)
    plt.title(r'$\mathrm{Randomly\ Generated}$'
              +'\n'+r'$\mathrm{Soot\ Clump}$', y=1.05)
    plt.xlabel(r'$\mathrm{x-distance}$')
    plt.ylabel(r'$\mathrm{y-distance}$')
    
    #If the grid is large, remap the seed location so it is visible.
    largegrid = 101 #Define the minimum "large" grid.
    if n>=largegrid:
        plt.plot(x0,y0,'ms', markersize=3)
        #For large grids with the binary colormap, magenta denotes the location
        #of the clump seed, gray denotes soot locations, and black denotes
        #perimeter locations.
        
    #If the grid is relatively small, show the seed on the grid itself by
    #temporarily changing its value to 3.0.
    if n<largegrid:
        g[y0,x0]=3.0
        #For small grids with the binary colormap, black denotes the location
        #of the clump seed, gray denotes soot locations, and dark gray denotes
        #perimeter locations.
    plt.pcolor(g, cmap='binary')
    
    #Once the clump grid plot is created, if the seed location value was
    #changed to 3.0 to highlight it on a small grid, change the seed location
    #back to 1.0 to ensure fractial dimensionality calculations are accurate.
    if g[y0,x0]!=1.0:
        g[y0,x0]=1.0
      
    plt.gca().set_aspect('equal', adjustable='box') #Set square axes.
    plt.xlim(0,len(g)-1) #Limit the plot to the dimensions of the grid.
    plt.ylim(0,len(g)-1)
    
    if spawntype == 'line':
        #Calculate the normalized values of the leftmost and rightmost
        #x-locations of the spawn range.
        norm_x_lo = x_lo/n
        norm_x_hi = x_hi/n
        #Denote the particle spawn range with a cyan line.
        ax1.axhline(y=h_lo-5, xmin=norm_x_lo, xmax=norm_x_hi, linewidth=2,
                    color='c')
    
    if spawntype == 'circle':
        from matplotlib.patches import Circle
        #Denote the particle spawn range with a cyan circle.
        circ = Circle((x0,y0), radius, fill=False, color='c')
        ax1.add_patch(circ)
    
    #Annotate the plot of the soot clump with all relevant information about
    #how it was generated.
    ax1.annotate(r'$\mathrm{width\ =}$'+'{0:5d}'.format(width)
                 +'\n'+r'$\mathrm{height\ =}$'+'{0:5d}'.format(height)
                 +'\n'+r'$\mathrm{max.\ y\ =}$'+'{0:5d}'.format(h_hi)
                 +'\n'+r'$\mathrm{min.\ y\ =}$'+'{0:5d}'.format(h_lo)
                 +'\n'+r'$\mathrm{max.\ x\ =}$'+'{0:5d}'.format(w_hi)
                 +'\n'+r'$\mathrm{min.\ x\ =}$'+'{0:5d}'.format(w_lo)
                 +'\n'+r'$\mathrm{no.\ walkers\ =}$'+'{0:5d}'.format(nwalkers)
                 +'\n'+r'$\mathrm{steps/walker\ =}$'+'{0:5d}'.format(steps)
                 +'\n'+r'$\mathrm{grid\ length\ =}$'+'{0:5d}'.format(n)
                 +'\n'+r'$\mathrm{no.\ offgrids\ =}$'+'{0:5d}'.format(offgrids)
                 +'\n'+r'$\mathrm{kills\ =}$'+'{0:5d}'.format(kills)
                 +'\n'+r'$\mathrm{no.\ particles\ =}$'+'{0:5d}'.format(size),
                         xy=(0.80, 0.97), xycoords='axes fraction',fontsize=10,
                         horizontalalignment='left',verticalalignment='top')
    
    #Define details for a legend. Firstly, define the colors of the plot with
    #hue-saturation-value triples. 
    if n<largegrid: #For small grids,
        seed_hsv = (0,0,0) #seed locations are black,
        perim_hsv =(1/3,1/3,1/3) #perimiter locations are dark gray, and
        soot_hsv = (2/3,2/3,2/3) #soot clump locations are light gray.
    elif n>=largegrid: #For large grids,
        #seed locations are matplotlib magenta,
        seed_hsv = (206/255, 52/255, 195/255)
        #perimiter locations are black,
        perim_hsv = (0,0,0)
        #and soot clump locations are matplotlib gray.
        soot_hsv = (127/255, 127/255, 127/255)
    #Construct the patches for the legend.
    seed_p = mpatches.Patch(color=seed_hsv, label=r'$\mathrm{seed}$')
    perim_p = mpatches.Patch(color=perim_hsv, label=r'$\mathrm{perimeter}$')
    soot_p = mpatches.Patch(color=soot_hsv, label=r'$\mathrm{soot}$')
    spawn_p = mpatches.Patch(color='c', label=r'$\mathrm{spawn}$')
    #Plot the legend.
    plt.legend(handles=[seed_p, perim_p, soot_p, spawn_p], loc=(0.85,0.02),
               prop={'size': 10}, frameon=False)
    plt.tight_layout()
    plt.savefig('fig1') #Save the figure in the working directory.
    plt.show()
    
if gen_fig2 == True:
    fig2=plt.figure(2, facecolor='white')
    ax2 = fig2.add_subplot(111)
    plt.title(r'$\mathrm{Walker\ Spawn\ Positions}$', y=1.05)
    plt.xlabel(r'$\mathrm{x-distance}$')
    plt.ylabel(r'$\mathrm{y-distance}$')
    plt.plot(spawn_x, spawn_y, 'k.', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box') #Set square axes.
    plt.ylim(0,len(g)-1) #Limit the axes to the dimensions of the grid.
    plt.xlim(0,len(g)-1)
    
    #Calculate average spawn coordinates for the annotation.
    avgx = np.average(spawn_x)
    avgy = np.average(spawn_y)
    ax2.annotate(r'$\mathrm{no. walkers\ =}$'+'{0:5d}'.format(nwalkers)
                 +'\n'+r'$\mathrm{grid\ length\ =}$'+'{0:5d}'.format(n)
                 +'\n'+r'$\mathrm{no.\ offgrids\ =}$'+'{0:5d}'.format(offgrids)
                 +'\n'+r'$\mathrm{kills\ =}$'+'{0:5d}'.format(kills)
                 +'\n'+r'$\mathrm{no.\ particles\ =}$'+'{0:5d}'.format(size)
                 +'\n'+r'$\mathrm{avg.\ x\ =\ }$'+'{0:2.2f}'.format(avgx)
                 +'\n'+r'$\mathrm{avg.\ y\ =\ }$'+'{0:2.2f}'.format(avgy),
                         xy=(0.8, 0.97), xycoords='axes fraction', fontsize=10,
                         horizontalalignment='left',verticalalignment='top')
    plt.tight_layout()
    plt.savefig('fig2')
    plt.show

if gen_fig3 == True:
    fig3 = plt.figure(3, facecolor='white')
    ax3 = fig3.add_subplot(111)
    plt.title(r'$\mathrm{Log-Log\ Plot\ of\ Contained\ Mass}$'
              +'\n'+r'$\mathrm{vs.\ Radius\ from\ Seed\ Location}$', y=1.05)
    plt.xlabel(r'$\log(\mathrm{Radius})$')
    plt.ylabel(r'$\log(\mathrm{Mass})$')
    plt.plot(logr, logm_fit,'c')
    plt.plot(np.log(radii),np.log(masses),'k.', alpha=0.5)
    #Wrap the spawntype string with LaTeX so it looks pretty on the plot.
    spawntype_anot = '$\mathrm{'+spawntype+'}$'
    ax3.annotate(r'$\mathrm{fractal\  dim.\ =}$'+'{0:5.2f}'.format(m)
                 +'\n'+r'$\mathrm{no. particles\ =}$'+'{0:5d}'.format(size)
                 +'\n'+r'$\mathrm{spawn\ type\ =\ }$'
                 +r'{0:s}'.format(spawntype_anot),
                 xy=(0.03, 0.9), xycoords='axes fraction', fontsize=10,
                 horizontalalignment='left',verticalalignment='top')
    #Construct the legend and its associated patches.
    fit_p = mpatches.Patch(color='c',
                           label=r'$d_f\ \mathrm{line\ of\ best\ fit}$')
    data_p = mpatches.Patch(color='k', alpha=0.5,
                            label=r'$\mathrm{original\ data}$')
    plt.legend(handles=[fit_p, data_p], loc=(0.03,0.9), prop={'size': 10},
               frameon=False)
    plt.tight_layout()
    plt.savefig('fig3')
    plt.show()