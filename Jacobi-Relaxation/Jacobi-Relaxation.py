# -*- coding: utf-8 -*-
"""
Created on Sun May 6 11:21:21 2018

@author: Cemenenkoff
This code explores the physics of Laplace's equation and is closely based on
chapter 5 of Computational Physics by Nicholas Giordano and Hisao Nakanishi
(2nd Edition), pp. 129-144.

Directly from Covey's problem statement:
    Consider a square conducting plate that is held at a fixed electric
    potential of 1 (in certain units). Far away from the conducting plate, the
    electric potential is zero. Use the relaxation method to solve Laplace's
    equation and determine the potential at all points on a 2-D grid contained
    within a larger square that maintains the system's boundary conditions. The
    goal of this problem is to determine how the precision of your solution
    depends on the geometry of the system, its initial conditions, and the
    number of iterations of the relaxation method that are performed.
    
For purposes of visualization, I think about the conducting plate as "hot"
while the outer boundary as "cold", so over successive iterations of
relaxation, the thermal energy in the hot reservoir "leaks" out across the grid
toward the outer boundary. All variables and functions are named and designed
with a large "icy" parent square and a small "hot plate" inner square (exactly
centered within the larger icy square) in mind.

This code consists of 3 main sections:
    1. Numerical Parameters and Switchboard
        You can adjust which plots are calculated/displayed as well as their
        important numerical properties.
    2. Main Functions
        Workhorse functions used in creating the figures are defined here.
    3. Figures
        Finer details for each plot (annotations, labels, etc.) are found here.
"""
#Covey's preamble with a few customizations.
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#Import the tqdm module to show a loading progress bar for time-consuming
#calculations. This is done by wrapping tdqm() around the range() of a
#for-loop, e.g. for i in tqdm(range(0,steps)).
from tqdm import tqdm
plt.style.use('classic') #Use a serif font.
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
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
#1. Numerical Parameters and Switchboard ######################################
###############################################################################
#dim_hot is the side length of the square hot plate (odd integer) for figs1-5.
#Note this value must be at least 4 less than dim_ice (and/or dim_base) due to
#the nature of discretizing square boundary conditions.
dim_hot = 5
#dim_ice is the side length of the large icy square (odd integer) for fig1,
#fig2, and fig5.
dim_ice = 115
#tol is the relative error tolerance (float) for figs1-4, defined as the
#difference in the total average of the average difference between up-down and
#left-right neighbors for each updated grid sub-square for each full grid
#update. See update() for clarification.
tol = 1e-6

jmax = 1500 #jmax is the maximum number of iterations (integer) for figs1-5.
dim_base=15 #dim_base is the starting grid size for figs3-4 (odd integer).
dim_ceil=115 #dim_ceil is the ending grid size for figs3-4 (odd integer).

#tol_list is the linearly sampled list of error tolerances used in fig5.
tol_list = list(np.linspace(1e-6,1e-5,200))

#Here is a convenient switchboard to selectively generate figures.
gen_fig1 = True #Hot Plate on an Icy Square (heatmap of the relaxation)
gen_fig2 = False #Absolute Relative Error vs. Number of Iterations
gen_fig3 = False #Number of Iterations vs. Dimension of Parent Square
gen_fig4 = False #Average Change over Base Grid vs. Dimension of Parent Square
gen_fig5 = False #Number of Iterations to Achieve Tolerance vs. Tolerance

gen_all = False
if gen_all == True:
    gen_fig1 = True
    gen_fig2 = True
    gen_fig3 = True
    gen_fig4 = True
    gen_fig5 = True

###############################################################################
#2. Main Functions ############################################################
###############################################################################
"""
Inputs:
   dim_ice = dimension of the large icy square  (odd integer)
   dim_hot = dimension of the small square hot plate (odd integer)
Output:
   V = square grid populated with values between 0 and 1 (2D numpy array)
       Values closer to 0 are "colder" whereas values closer to 1 are "hotter".
"""
def generate(dim_ice, dim_hot):    
    #Create a (dim_ice)x(dim_ice) array of random samples from a uniform
    #distribution over [0, 1)
    V = np.random.rand(dim_ice,dim_ice)
    
    #Define the middle of the entire grid with floor division. Note that this
    #is one integer offset from the true middle integer because Python starts
    #indexing at 0. 
    mid = dim_ice//2
    
    #Fill in a centered sub-square of the icy square with 1s to represent the 
    #hot plate. i is the row index, j is column index, the counting index
    #starts at 0, and [0,0] denotes the upper left corner of the grid.
    for i in range((mid-dim_hot//2),(mid+dim_hot//2)+1):
        for j in range((mid-dim_hot//2),(mid+dim_hot//2)+1):
            V[i,j] = 1
    
    #Fill the outside of the grid with 0s to represent the icy outside boundary
    #condition.
    for i in range(0, dim_ice):
        V[i,0]=0 #Fill out the entire first column with 0s.
        V[i,dim_ice-1]=0 #Fill out the entire last column with 0s.
    for j in range(0, dim_ice):
        V[0,j]=0 #Fill out the entire first row with 0s.
        V[dim_ice-1,j]=0 #Fill out the entire last row with 0s.
    return V

"""
Inputs:
   V = configuration to be updated (2D numpy array)
   dim_hot = dimension of the small square hot plate (odd integer)
Outputs:
   V = the updated 2D configuration after updating (2D numpy array)
   avg_of_avgs = average difference between up-down and left-right neighbors
                 post-update (float)
"""
def update(V, dim_hot):
    dim_ice=len(V)
    mid = dim_ice//2
    avg_diff_list = []
    #Since we want to avoid the boundaries of the large icy square, adjust the
    #range to be offset by 1 from the starting and ending integer.
    for i in range(1,dim_ice-1):
        for j in range(1,dim_ice-1):
            #If we are on an index that represents an element of the hot plate,
            #don't touch it and continue to the next iteration.
            if abs(mid-i)<=dim_hot/2 and abs(mid-j)<=dim_hot/2:
                continue
            #Compute one step of relaxation, Giordano eq. 5.10 on p.132
            V[i,j] = (V[i-1,j] + V[i+1,j] + V[i,j-1] + V[i,j+1])/4
            diff_left  = abs(V[i,j]-V[i-1,j]) #Log the new left-difference.
            diff_right = abs(V[i,j]-V[i+1,j]) # '   '   ' right- '
            diff_above = abs(V[i,j]-V[i,j+1]) # '   '   '    up- '
            diff_below = abs(V[i,j]-V[i,j-1]) # '   '   '  down- '
            #Compute the average up-down-left-right difference and append it.
            avg_diff = (diff_left+diff_right+diff_above+diff_below)/4
            avg_diff_list.append(avg_diff)
    #Compute the average of the avg_diff_list. avg_of_avgs represents the 
    #average change for the entire grid and can be used as a measure of
    #convergence to the true equilibrium solution to Laplace's equation.
    avg_of_avgs = np.average(avg_diff_list)
    return V, avg_of_avgs

"""
Inputs:
   V = configuration to be relaxed (2D numpy array)
   dim_hot = dimension of the small square hot plate (odd integer)
   tol = error tolerance (float)
   jmax = maximum number of iterations (integer)
Outputs:
   V = relaxed version of the 2D input array after the error tolerance is
       achieved or the maximum number of iterations reached (2D numpy array)
   count = number of iterations performed (integer)
   err_list = relative error list (list of floats)
"""
def relax(V, dim_hot, tol, jmax):
    count = 0
    #Make a list of 1s to store average differences acquired from update().
    diff_list = [1]*jmax 
    #Initialize a list for the differences of the average differences acquired
    #from update() (i.e. a list for storing relative error).
    err_list = []
    for j in range(0,jmax):
        #Calculate the next V by updating the previous V.
        V, avg_diff = update(V, dim_hot)
        diff_list[j] = avg_diff
        count += 1
        diff_of_diff = abs(diff_list[j]-diff_list[j-1])
        err_list.append(diff_of_diff) #Add to the relative error list.
        #If the difference between errors acquired from update() is less than
        #the error tolerance, break out of the loop
        if diff_of_diff < tol:
            #print(tol, count)
            break
    return V, count, err_list

"""
Inputs:
   V = configuration to obtain base error from (2D numpy array)
   dim_hot = dimension of the small square hot plate (odd integer)
   dim_base = dimension of the base square grid (odd integer)
Outputs:
   avg_of_avgs = average up-down-left-right change over the base grid (float)
"""
def get_err_base(V, dim_hot, dim_base):
    dim_ice=len(V)
    #Note, we can't log the nearest-neighbor changes for the base-grid
    #itself because the outer edges don't have a full set of 4 neighbors
    if dim_ice==dim_base:
        print('Warning, cannot calculate base error for the base-grid itself.')
        return None
    mid = dim_ice//2
    avg_diff_list = []
    #Iterate through grid-values in the center of the expanded grid, but only
    #ones that are shared with the base grid. For example, if the base grid is
    #15x15, for each expanded grid, only count up-down-left-right neighbor
    #differences for the center 15x15 grid excluding the squares representing
    #the hot plate. Average all of these averages to return the average 
    #up-down-left-right change over the base grid which (if small), represents
    #convergence to an equilibrium solution.
    for i in range((mid-dim_base//2),(mid+dim_base//2)+1):
        for j in range((mid-dim_base//2),(mid+dim_base//2)+1):
            #If we are on an index that represents an element of the hot plate,
            #don't touch it and continue to the next iteration.
            if abs(mid-i)<=dim_hot/2 and abs(mid-j)<=dim_hot/2:
                continue
            diff_left  = abs(V[i,j]-V[i-1,j]) #Log the new left-difference.
            diff_right = abs(V[i,j]-V[i+1,j]) # '   '   ' right- '
            diff_above = abs(V[i,j]-V[i,j+1]) # '   '   '    up- '
            diff_below = abs(V[i,j]-V[i,j-1]) # '   '   '  down- '
            #Compute the average up-down-left-right difference and append it.
            avg_diff = (diff_left+diff_right+diff_above+diff_below)/4
            avg_diff_list.append(avg_diff)
    avg_of_avgs = np.average(avg_diff_list)
    #Return the average up-down-left-right change over the base grid
    return avg_of_avgs

###############################################################################
#3. Figures ###################################################################
###############################################################################

if gen_fig1 == True or gen_fig2 == True: 
    V0 = generate(dim_ice, dim_hot) #Create an initial configuration.
    V, count, err = relax(V0, dim_hot, tol, jmax) #Relax V0.
    
    if gen_fig1 == True:
        #######################################################################
        #Hot Plate on an Icy Square (heatmap of the relaxation)
        fig1 = plt.figure(1, facecolor='white')
        ax1 = fig1.add_subplot(111)
        plt.title(r'$\mathrm{Conducting\ Plate\ at\ }V=1$'+'\n'
                  +r'$\mathrm{Bounded\ By\ }V=0\mathrm{\ at\ Equilibrium}$',
                  y=1.05) #Ensure the title doesn't overlap the colorbar.
        plt.pcolor(V,cmap='plasma') #plasma, cool, gist_heat are cool too.
        plt.gca().set_aspect('equal',adjustable='box') #Set square axes.
        plt.colorbar() #Put a color gradient alongside to act like a legend.
        plt.xlabel(r'$\mathrm{x-Distance}$')
        plt.ylabel(r'$\mathrm{y-Distance}$')
        plt.xlim(0,len(V))#Trim to the dimensions of the large icy square.
        plt.ylim(0,len(V))
        ax1.annotate('{0:5>d}x{0:5<d}'.format(dim_ice)
                     +r'$\mathrm{\ parent\ grid;}$'
                     +' {0:5>d}x{0:5<d}'.format(dim_hot)
                     +r'$\mathrm{\ center\ grid}$'
                     +'\n'+'{0:.2e}'.format(tol)
                     +r'$\mathrm{\ error\ tolerance;}$'
                     +' {0:5>d}/{1:<5d}'.format(count,jmax)
                     +r'$\mathrm{iterations}$',
                     xy=(0.0, -0.2), xycoords='axes fraction', fontsize=12,
                     horizontalalignment='left',verticalalignment='top')
        plt.tight_layout()
        plt.show()

    if gen_fig2 == True:
        #######################################################################
        #Create a list of integers from 0 to the number of relative errors to
        #serve as the x-axis array for the next plot.
        err_iter = list(range(0,len(err)))
        
        #Absolute Relative Error vs. Number of Iterations
        fig2 = plt.figure(2, facecolor='white')
        ax2 = fig2.add_subplot(111)
        plt.title(r'$\mathrm{Absolute\ Relative\ Error\ }$'
                  +r'$\mathrm{vs.\ Number\ of\ Iterations}$')
        import matplotlib.ticker as mtick #Format the y-tick labels.
        ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        plt.plot(err_iter,err, 'b.')
        plt.xlabel(r'$\mathrm{Number\ of\ Iterations}$')
        plt.ylabel(r'$\mathrm{Absolute\ Relative\ Error}$')
        plt.xlim(1,50) #Skip the first because it is wildly inaccurate.
        plt.ylim(0,1.05*err[1])
        ax2.annotate('{0:5>d}x{0:5<d}'.format(dim_ice)
                     +r'$\mathrm{\ parent\ grid;}$'
                     +' {0:5>d}x{0:5<d}'.format(dim_hot)
                     +r'$\mathrm{\ center\ grid}$'
                     +'\n'+'{0:.2e}'.format(tol)
                     +r'$\mathrm{\ error\ tolerance}$',
                     xy=(0.6, 0.95), xycoords='axes fraction', fontsize=12,
                     horizontalalignment='left',verticalalignment='top')
        plt.tight_layout()
        plt.show()

if gen_fig3 == True or gen_fig4 == True:
    ###########################################################################
    #The following iteration gathers data on the number of iterations to
    #achieve tolerance vs. the dimension of the large icy square.
    grid_exps=(dim_ceil-dim_base)//2 #number of grid expansions (integer)
    dim_ice_list = []
    count_list = []
    err_base_list = []
    for i in tqdm(range(grid_exps+1)):
        dim_temp = 2*i+dim_base #The dimension jumps up by 2 each time.
        dim_ice_list.append(dim_temp)
        V0 = generate(dim_temp,dim_hot)
        V, count, err_full = relax(V0, dim_hot, tol, jmax)
        count_list.append(count)
        #Note, we can't log the nearest-neighbor changes for the base-grid
        #itself because the outer edges don't have a full set of 4 neighbors.
        if dim_temp != dim_base:
            err_base = get_err_base(V, dim_hot, dim_base)
            err_base_list.append(err_base)
    
    if gen_fig3 == True:
        #Number of Iterations to Achieve Tolerance vs. Dimension of Parent
        #Square
        fig3 = plt.figure(3, facecolor='white')
        ax3 = fig3.add_subplot(111)
        plt.title(r'$\mathrm{Number\ of\ Iterations\ to \ Achieve\ Tolerance}$'
                  +'\n'+r'$\mathrm{vs. \ Dimension\ of\ Parent\ Square}$')
        plt.plot(dim_ice_list, count_list, 'b.')
        plt.xlabel(r'$\mathrm{Dimension\ of\ Parent\ Square}$')
        plt.xlim(dim_base,dim_ceil)
        plt.ylabel(r'$\mathrm{Number\ of\ Iterations}$')
        ax3.annotate('{0:5>d}x{0:5<d}'.format(dim_hot)
                     +r'$\mathrm{\ center\ grid;}$'+' {0:.2e}'.format(tol)
                     +r'$\mathrm{\ error\ tolerance}$',
                     xy=(0.6, 0.1), xycoords='axes fraction', fontsize=12,
                     horizontalalignment='left',verticalalignment='top')
        plt.tight_layout()
        plt.show()
    
    if gen_fig4 == True:
        #Average Change over Base Grid vs. Dimension of Parent Square
        fig4 = plt.figure(4, facecolor='white')
        ax4 = fig4.add_subplot(111)
        plt.title(r'$\mathrm{Average\ Change\ over\ Base\ Grid}$'
                  +'\n'+r'$\mathrm{vs. \ Dimension\ of\ Parent\ Square}$')
        plt.plot(dim_ice_list[1:], err_base_list, 'b.')
        plt.xlabel(r'$\mathrm{Dimension\ of\ Parent\ Square}$')
        plt.ylabel(r'$\mathrm{Average\ Absolute\ Average\ }$'
                   +r'$\uparrow\downarrow\leftarrow\rightarrow$'+'\n'
                   +r'$\mathrm{\ Nearest\ Neighbor\ Base\ Grid\ Changes}$')
        plt.xlim(dim_ice_list[1], dim_ice_list[-1])
        ax4.annotate('{0:5>d}x{0:5<d}'.format(dim_hot)
                     +r'$\mathrm{\ center\ grid;}$'+' {0:.2e}'.format(tol)
                     +r'$\mathrm{\ error\ tolerance}$',
                     xy=(0.5, 0.95), xycoords='axes fraction', fontsize=12,
                     horizontalalignment='left',verticalalignment='top')
        plt.tight_layout()
        plt.tight_layout()
        plt.show()

if gen_fig5 == True:
    ###########################################################################
    #The following iteration gathers data on the number of iterations to
    #achieve tolerance vs. tolerance.
    
    #Note that a new V0 is generated and then subsequently deleted for each
    #tolerance value. Without this, the first value in tol_list is computed
    #correctly, but because V0 is saved in the system RAM, the next iterations
    #have some of their computation "saved", hence the true number of counts
    #associated with them is flawed. Python 3.6 tries to save computational
    #time by recyling previously computed values associated with pre-existing
    #data structures. We circumvent this by removing V0 from the RAM and then 
    #reintroducing it for each iteration.
    count_list = []
    for i in tqdm(range(len(tol_list))):
        V0 = generate(dim_ice, dim_hot)
        count = relax(V0, dim_hot, tol_list[i], jmax)[1]
        count_list.append(count)
        del V0

    #Number of Iterations to Achieve Tolerance vs. Tolerance
    fig5 = plt.figure(5, facecolor='white')
    ax5 = fig5.add_subplot(111)
    plt.title(r'$\mathrm{Number\ of\ Iterations\ to \ Achieve\ Tolerance}$'
                         +'\n'+r'$\mathrm{\ vs.\ Tolerance}$')
    import matplotlib.ticker as mtick #Format the x-tick labels.
    ax5.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.xticks(rotation=20)
    plt.plot(tol_list, count_list, 'b.')
    plt.xlabel(r'$\mathrm{Tolerance}$')
    plt.ylabel(r'$\mathrm{Number\ of\ Iterations}$')
    ax5.annotate('{0:5>d}x{0:5<d}'.format(dim_ice)
                 +r'$\mathrm{\ parent\ grid;}$'
                 +' {0:5>d}x{0:5<d}'.format(dim_hot)
                 +r'$\mathrm{\ center\ grid}$',
                 xy=(0.6, 0.95), xycoords='axes fraction', fontsize=12,
                 horizontalalignment='left',verticalalignment='top')
    plt.tight_layout()
    plt.show()