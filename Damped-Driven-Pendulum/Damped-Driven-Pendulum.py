# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:27:28 2018

@author: Cemenenkoff
This code explores the physics of a damped driven pendulum and recreates
figures found in Computational Physics by Nicholas Giordano and Hisao Nakanishi
(2nd Edition), pp. 59-69. It may also generate a tab-delimited .txt file for
certain initial conditions. Choose which figures to generate and/or whether to
generate an output text file under the Switchboard section below.

Arrays for angular position and velocity of the pendulum are generated using
the RK-2 method of numerically solving ODEs. See the Main RK-2 Routine section
below for specifics.
"""

###############################################################################
#Preamble######################################################################
###############################################################################
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
plt.rcParams['savefig.dpi'] = 1200
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
#Switchboard###################################################################
###############################################################################
#Decide which figures to generate.
gen_fig_3p6_left   = False #theta vs t for F_D = 0, 0.5, 1.2
gen_fig_3p6_center = False #theta vs t for F_D = 1.2 with and without resets
gen_fig_3p6_right  = False #w vs t for F_D = 0, 0.5, 1.2
gen_fig_3p8_left   = False #w vs theta for F_D = 0.5
gen_fig_3p8_right  = False #w vs theta for F_D = 1.2
gen_fig_3p9        = False #stroboscopic w vs theta for F_D = 1.2
gen_fig_3p11       = True #bifurcation diagram of theta vs F_D
gen_ex_318         = False #stroboscopic w vs theta for F_D = 1.4, 1.44, 1.465

gen_all_plots = False #Here's a convenient switch to generate all plots.
if gen_all_plots == True:
    gen_fig_3p6_left   = True
    gen_fig_3p6_center = True
    gen_fig_3p6_right  = True
    gen_fig_3p8_left   = True
    gen_fig_3p8_right  = True
    gen_fig_3p9        = True
    gen_fig_3p11       = True
    gen_ex_318         = True

#Decide if you want to generate a tab-delimited text output file to then
#compare with Covey's results via verify_Pendulum_HW3.txt. Ensure hires = True
#so the comparison is performed with high resolution results.
gen_text_file = False

#If gen_text_file = True, decide if you want to enable a numerical convergence
#test which analyzes how the model differs from Covey's data at t=60.0 for
#various time step sizes.
if gen_text_file == True:
    gen_num_comp  = True
if gen_text_file == False:
    gen_num_comp = False

###############################################################################
#Numerical Parameters##########################################################
###############################################################################
#Initial Conditions: _std denotes "standard" values. Anything with _std is
#hard-coded and not meant to change unless you know what you are doing.
w0_std = 0 #initial angular velocity (radians/s)
theta0_std = 0.2 #initial angular position (radians)
F_D_std = [0, 0.5, 1.2] #driving force amplitudes (radians/s^2)
g = 9.8 #magnitude of gravitational acceleration (m/s)
l = 9.8 #length of the pendulum (m)
q = 0.5 #friction coefficient (1/s)
w_D = 2/3 #driving force frequency (1/s)
T_D = (2*np.pi)/w_D #driving period (s) (recall that w=2pi/T)
dt_std = 0.04 #time step (s)
t0_std = 0.0 #start time (s)
intmult_std = 20 #integer as a multiplicative factor
tf_std = intmult_std*T_D #end time (s) for standard runs

#For the stroboscopic plots, you only need to adjust intmult_strb and skip.
intmult_strb = 300
tf_strb = intmult_strb*T_D #end time (s) for stroboscopic plots

#For the bifurcation diagram, run tests to find an appropriate end time,
#time step, tol, and skip parameter for something like samp_bif = 20 first.
#Once you think you're obtaining a clean resolution, crank up the number of
#F_D samples.
intmult_bif = 300
tf_bif = intmult_bif*T_D #end time (s) for the bifurcation diagram
#Note end times are integer multiples of T_D to ensure there will be t-values
#at (or close to) integer multiples of T_D. The time step used by Giordano to
#create Figure 3.11 is 0.01, but instead we force dt to be an integer fraction
#of T_D to guarantee the existence of t-values that occur at integer multiples
#of T_D (makes for a cleaner plot).
intmult_dt_bif = 600
dt_bif = T_D/intmult_dt_bif
#samp_bif is the (even) number of linearly spaced F_D samples used in creating
#the bifurcation diagram. To change the range of F_D, go to where figure 3.11
#is generated farther down in the code.
samp_bif = 12000
#tol is the tolerance for how far away a time value can be from a perfect
#integer multiple of T_D and still be registered as a stroboscopic time, thus
#causing its index in the time-array to be appended to ind_strb in the main()
#function below.
tol = 0.0015
#skip details how many values to skip in each array containing stroboscopic
#data to get rid of transients on the plot. Physically, this represents the
#small period of time the pendulum's dampener needs to "settle down" the
#initial motion.
skip=290
#If higher resolution is needed, hires = True changes all important variables
#to double precision floats. This means each value will be assigned a sign bit,
#11 bits for the exponent, and 52 bits for the mantissa.
hires = True
if hires == True:
    w0_std = np.float_(w0_std)
    theta0_std = np.float_(theta0_std)
    F_D_std = np.float_(F_D_std)
    g = np.float_(g)
    l = np.float_(l)
    q = np.float_(q)
    w_D = np.float_(w_D)
    T_D = (2*np.pi)/w_D
    dt_std = np.float_(dt_std)
    t0_std = np.float_(t0_std)
    tf_std = intmult_std*T_D
    tf_strb = intmult_strb*T_D
    tf_bif = intmult_bif*T_D
    dt_bif = T_D/intmult_dt_bif
    tol = np.float_(tol)

###############################################################################
#Main RK-2 Routine#############################################################
###############################################################################
#2nd derivative of theta, inputs are floats, output is a float.
def d2(w,theta,t,F_D):
    #Giordano p. 59, eq. 3.19 top line
    return -g/l*np.sin(theta) - q*w + F_D*np.sin(w_D*t)

#1st derivative of theta, input is a float, output is a float.
def d1(w): #first derivative of theta
    return w #Giordano p. 59, eq. 3.19 bottom line

#main() takes in initial conditions and then outputs arrays of time values,
#theta values, omega values, and stroboscopic index indices (integers).
#Specifically,
#    inputs: float, float, float, float, float, float, bool
#   outputs:  list,  list,  list,  list
def main(w0, theta0, F_D, dt, t0, tf, reset):
    steps = int(round((tf-t0)/dt)) #total number of time steps (dimensionless)
    t = np.linspace(t0, tf, steps+1) #Generate the time list.
    w = [w0] #Initialize a list with initial conditions for angular velocity.
    theta = [theta0]#Do the same for angular position.
    #Initialize a list to store indices associated with time values that are
    #equal to the driving period.
    ind_strb = []
    for i in range(0, steps):
        #Calculate w and theta after half a time step of evolution.
        w_hw = w[i] + d2(w[i],theta[i],t[i],F_D)*(dt/2)
        theta_hw = theta[i] + d1(w[i])*(dt/2)
        #Use the halfway values to compute w and theta after a full time step.
        w_ip1 = w[i] + d2(w_hw,theta_hw,t[i] + dt/2,F_D)*dt
        theta_ip1 = theta[i] + w_hw*dt
        if reset == True:
            #If the pendulum swings to the right up and past vertical, change
            #theta into a negative angle.
            if theta_ip1 > np.pi:
                theta_ip1 = theta_ip1 - 2*np.pi
            #If the pendulum swings to the left up and past vertical, change
            #theta into a positive angle.
            if theta_ip1 < -np.pi:
                theta_ip1 = theta_ip1 + 2*np.pi
        #Append the i+1th values to the appropriate lists.
        w.append(w_ip1)
        theta.append(theta_ip1)
        #If the remainder after dividing the current time step by T_D is
        #essentially zero, note the index of the current step and append it to
        #the stroboscopic index list.
        if (t[i]%T_D) <= tol and t[i] != 0:
            ind_strb.append(i)
    return t, theta, w, ind_strb

###############################################################################
#Text File and Covey Comparison################################################
###############################################################################
#An email from Covey:
#Second, I've just posted files that you can use to explore the accuracy and
#precision of our calculations -- especially if you uncover another bug in my
#code!

#The file for my pendulum trajectory is available at verify_Pendulum_HW3.txt.
#The columns of this file are time (in seconds), non-remapped theta values
#(in radians), and angular velocities (in radians / sec). This was calculated
#using my RK-2 implementation, with initial conditions of:

#* theta_0 = 0.2 , omega_0 = 0, time_0 = 0, g = -9.8, l = 9.8, q = 0.5,
#driveForce = 1.2, driveFrequency = 2/3, RK2timeStep = 0.4, and FullTime = 60.

#Since we want our results to be as accurate as possible for this comparison,
#ensure calculations are done in high resolution.
if gen_text_file == True and hires == False:
    print('Please ensure hires = True and then try again.')
if gen_text_file == True and hires == True:
    #Generate data for Covey's initial conditions. Note the False to obtain
    #"non-remapped theta values". Change dt_std to something nonstandard to
    #perform a convergence test.
    data = main(w0_std, theta0_std, F_D_std[2], 0.04, t0_std,
                np.float_(60.0), False)
    t_tofile     = data[0]
    theta_tofile = data[1]
    w_tofile     = data[2]
    f = open('Cemenenkoff-PHYS486-HW3.txt','w')
    for i in range(len(t_tofile)):
        f.write('%3.15f'%t_tofile[i]+'\t'+'%3.15f'%theta_tofile[i]+'\t'
                +'%3.15f\n'%w_tofile[i])
    f.close() #Close the file after writing to it.

    #We can quantify how much our calculations differ from Covey's by finding
    #the absolute difference between each set of values.
    t_Cov = [] #Initialize empty lists to import Covey's .txt file data.
    theta_Cov = []
    w_Cov = []
    with open('verify_Pendulum_HW3.txt', 'r') as file:
        next(file) #Skip the header line.
        for row in file:
            row = row.strip()
            #Each time a space is hit in the current row, store the element in
            #a list called column.
            column = row.split()
            #Append elements from column to the appropriate lists. Ensure the
            #values are stored at double precision and that hires == True.
            t_Cov.append(np.float_(column[0]))
            theta_Cov.append(np.float_(column[1]))
            w_Cov.append(np.float_(column[2]))

    w_diffs = []
    theta_diffs = []
    for i in range(len(w_Cov)):
        abserr_w = abs(w_tofile[i]-w_Cov[i])
        abserr_theta = abs(theta_tofile[i]-theta_Cov[i])
        w_diffs.append(abserr_w)
        theta_diffs.append(abserr_theta)
    
    #We can graph the absolute deviations over time to witness the descent into
    #the chaos.
    #abserr_w vs t_Cov
    fig13 = plt.figure(13, facecolor='white')
    plt.title(r'$\omega\ \mathrm{Absolute\ Deviation\ vs.\ }t$')
    ###########################################################################
    plt.plot(t_Cov, w_diffs, 'k')
    plt.ylabel(r'$\mathrm{\left|\omega-\omega_\mathrm{Cov}\right|\ }$'
               +r'$\mathrm{(radians/second)}$')
    plt.ylim(0,1.05*max(w_diffs))
    plt.xlabel(r'$t$')
    
    #abserr_theta vs t_Cov
    fig14 = plt.figure(14, facecolor='white')
    plt.title(r'$\theta\ \mathrm{Absolute\ Deviation\ vs.\ }t$')
    ###########################################################################
    plt.plot(t_Cov, theta_diffs, 'k')
    plt.ylabel(r'$\mathrm{\left|\theta-\theta_\mathrm{Cov}\right|\ }$'
               +r'$\mathrm{(radians)}$')
    plt.ylim(0,1.05*max(theta_diffs))
    plt.xlabel(r'$t$')

    w_avgdev = np.average(abserr_w)
    theta_avgdev = np.average(abserr_theta)
    print('Absolute average theta deviation = %.12e'%theta_avgdev)
    print('Absolute average omega deviation = %.12e'%w_avgdev)

#Below, we explore how changing the time step affects the numerical
#precision of the model at a specific time value (60.0 here). This assumes
#that Covey's text file is approximately correct.
if gen_num_comp == True:
    #find_nearest_ind() finds the index in an array closest to a given value.
    import math
    def find_nearest_ind(array,value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array)
        or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx
    samp_dt = 1000
    dt_list = np.linspace(0.001,5.0,samp_dt) #Initialize a range of dt's to test.
    theta_diffs_t60 = []
    w_diffs_t60 = []
    #For each dt, find the index of the t-value closest to 60.0 (the last time
    #value in Covey's text file), and then use that index to pull the
    #associated w and theta for that t-value. Then compute the difference
    #between the indexed values and Covey's values.
    for dt in dt_list:
        t = main(w0_std, theta0_std, F_D_std[2], dt, t0_std,
                 np.float_(60.0), False)[0]
        theta = main(w0_std, theta0_std, F_D_std[2], dt, t0_std,
                     np.float_(60.0), False)[1]
        w = main(w0_std, theta0_std, F_D_std[2], dt, t0_std,
                 np.float_(60.0), False)[2]
        ind = find_nearest_ind(t, 60.0)
        theta_d60 = abs(theta_Cov[-1]-theta[ind])
        w_d60 = abs(w_Cov[-1]-w[ind])
        theta_diffs_t60.append(theta_d60)
        w_diffs_t60.append(w_d60)

    #Generate an Absolute w Deviation at t=60.0 vs dt plot.
    fig11 = plt.figure(11, facecolor='white')
    plt.title(r'$\mathrm{Absolute\ }$'+r'$\theta\ $'
              +r'$\mathrm{Deviation\ at\ }t=60.0\ \mathrm{vs\ }dt$')
    ###########################################################################
    plt.plot(dt_list, theta_diffs_t60, 'k.', alpha=0.3)
    plt.ylabel(r'$\mathrm{Absolute\ Deviation\ (radians)}$')
    plt.xlabel(r'$dt$')
    ax11 = fig11.add_subplot(111)
    xannot = 0.01
    yannot = 0.98
    ax11.annotate('%d '%samp_dt+r'$dt\ \mathrm{samples}$',
                 xy=(xannot, yannot), xycoords='axes fraction',
                 fontsize=16, horizontalalignment='left',
                 verticalalignment='top')

    #Generate an Absolute theta Deviation at t=60.0 vs dt plot.
    fig12 = plt.figure(12, facecolor='white')
    plt.title(r'$\mathrm{Absolute\ }$'+r'$\omega\ $'
              +r'$\mathrm{Deviation\ at\ }t=60.0\ \mathrm{vs\ }dt$')
    ###########################################################################
    plt.plot(dt_list, w_diffs_t60, 'k.', alpha=0.3)
    plt.ylabel(r'$\mathrm{Absolute\ Deviation\ (radians/second)}$')
    plt.xlabel(r'$dt$')
    ax12 = fig12.add_subplot(111)
    ax12.annotate('%d '%samp_dt+r'$dt\ \mathrm{samples}$',
                 xy=(xannot, yannot), xycoords='axes fraction',
                 fontsize=16, horizontalalignment='left',
                 verticalalignment='top')

###############################################################################
#Figures#######################################################################
###############################################################################
#Recreate Figure 3.6 from Giordano page 60.
if gen_fig_3p6_left == True:
    #Do the left panel as a 3x1 matrix of subfigures.
    fig1 = plt.figure(1, facecolor='white')
    plt.title(r'$\mathrm{Fig.\ 3.6\ (left):\ }\theta$'+r'$\mathrm{\ vs.\ }t$')
    plt.axis('off') #Remove the frame from the parent figure.
    ###########################################################################
    data_0   = main(w0_std, theta0_std, F_D_std[0], dt_std, t0_std, tf_std,
                    True)
    t_0     = data_0[0]
    theta_0 = data_0[1]
    ax1_311 = fig1.add_subplot(311)
    ax1_311.locator_params(tight=True, nbins=5)
    plt.plot(t_0,theta_0,'k')
    plt.xlim(0,60)
    plt.ylim(-0.2,0.2)
    ax1_311.axes.get_xaxis().set_ticks([]) #Turn off x-axis tick marks.
    ax1_311.spines['top'].set_visible(False) #Remove the top frame line.
    ax1_311.spines['right'].set_visible(False)#Remove the right frame line.
    ax1_311.spines['bottom'].set_visible(False)#Remove the bottom frame line.
    ax1_311.yaxis.set_ticks_position('left')#Set all y-ticks to be on the left.
    xannot = 0.5
    yannot = 0.95
    ax1_311.annotate(r'$F_D$ = %2.2f'%F_D_std[0],
                 xy=(xannot, yannot-0.15), xycoords='axes fraction',
                 fontsize=16, horizontalalignment='left',
                 verticalalignment='top')
    ###########################################################################
    data_0p5 = main(w0_std, theta0_std, F_D_std[1], dt_std, t0_std, tf_std,
                    True)
    t_0p5     = data_0p5[0]
    theta_0p5 = data_0p5[1]
    ax1_312 = fig1.add_subplot(312)
    ax1_312.locator_params(tight=True, nbins=5)
    plt.plot(t_0p5,theta_0p5,'k')
    plt.ylabel(r'$\theta\ (\mathrm{radians})$')
    plt.xlim(0,60)
    plt.ylim(-2,2)
    ax1_312.axes.get_xaxis().set_ticks([])
    ax1_312.spines['top'].set_visible(False)
    ax1_312.spines['right'].set_visible(False)
    ax1_312.spines['bottom'].set_visible(False)
    ax1_312.yaxis.set_ticks_position('left')
    ax1_312.annotate(r'$F_D$ = %2.2f'%F_D_std[1],
                 xy=(xannot, yannot), xycoords='axes fraction',
                 fontsize=16, horizontalalignment='left',
                 verticalalignment='top')
    ###########################################################################
    data_1p2 = main(w0_std, theta0_std, F_D_std[2], dt_std, t0_std, tf_std,
                    True)
    t_1p2     = data_1p2[0]
    theta_1p2 = data_1p2[1]
    ax1_313 = fig1.add_subplot(313)
    ax1_313.locator_params(tight=True, nbins=5)
    plt.plot(t_1p2,theta_1p2,'k')
    plt.xlabel(r'$t\ (\mathrm{seconds})$')
    plt.xlim(0,60)
    plt.ylim(-4,4)
    ax1_313.spines['top'].set_visible(False)
    ax1_313.spines['right'].set_visible(False)
    ax1_313.yaxis.set_ticks_position('left')
    ax1_313.xaxis.set_ticks_position('bottom')
    ax1_313.annotate(r'$F_D$ = %2.2f'%F_D_std[2],
                 xy=(xannot, yannot+0.12), xycoords='axes fraction',
                 fontsize=16, horizontalalignment='left',
                 verticalalignment='top')

if gen_fig_3p6_center == True:
    #theta vs t for F_D = 1.2 with and without resets. Do this as a 2x1 matrix
    #of subfigures.
    fig2 = plt.figure(2, facecolor='white')
    plt.title(r'$\mathrm{Fig.\ 3.6\ (center):\ }\theta$'
              +r'$\mathrm{\ vs.\ }t$')
    #Make a big overlaid subplot so axes labels can be shared.
    ax2 = fig2.add_subplot(111)
    #Turn off axis lines and ticks of the big subplot.
    ax2.spines['top'].set_color('none')
    ax2.spines['bottom'].set_color('none')
    ax2.spines['left'].set_color('none')
    ax2.spines['right'].set_color('none')
    ax2.tick_params(labelcolor='w', top='off', bottom='off', left='off',
                   right='off')
    ax2.set_ylabel(r'$\theta\ (\mathrm{radians})$') #Set a common y-label.
    ###########################################################################
    data  = main(w0_std, theta0_std, F_D_std[2], dt_std, t0_std, tf_std, True)
    t     = data[0]
    theta = data[1]
    w     = data[2]
    ax2_211 = fig2.add_subplot(211)
    plt.plot(t,theta,'k')
    plt.yticks([0,3])
    plt.ylim(-1,3.5)
    ax2_211.axes.get_xaxis().set_ticks([])
    ax2_211.spines['top'].set_visible(False)
    ax2_211.spines['right'].set_visible(False)
    ax2_211.spines['bottom'].set_visible(False)
    ax2_211.yaxis.set_ticks_position('left')
    ###########################################################################
    data_noreset  = main(w0_std, theta0_std, F_D_std[2], dt_std, t0_std,
                         tf_std, False)
    t_noreset     = data_noreset[0]
    theta_noreset = data_noreset[1]
    w_noreset     = data_noreset[2]
    ax2_212 = fig2.add_subplot(212)
    plt.plot(t_noreset, theta_noreset,'k')
    plt.xlabel(r'$t\ (\mathrm{seconds})$')
    plt.xlim(0,60)
    plt.ylim(-11,3)
    ax2_212.spines['top'].set_visible(False)
    ax2_212.spines['right'].set_visible(False)
    ax2_212.yaxis.set_ticks_position('left')
    ax2_212.xaxis.set_ticks_position('bottom')
    xannot = 0.5
    yannot = 0.95
    plt.xticks([0,20,40,60])
    plt.yticks([-10,-5,0,2])
    ax2_212.annotate(r'$F_D$ = %2.2f'%F_D_std[2],
                 xy=(xannot, yannot), xycoords='axes fraction', fontsize=16,
                 horizontalalignment='left',verticalalignment='top')
    plt.tight_layout()
    plt.show()

if gen_fig_3p6_right == True:
    #Do the right panel as another 3x1 matrix of subfigures.
    fig3 = plt.figure(3, facecolor='white')
    plt.title(r'$\mathrm{Fig.\ 3.6\ (right):\ }\omega$'
              +r'$\mathrm{\ vs.\ }t$')
    plt.axis('off')
    ###########################################################################
    data_0   = main(w0_std, theta0_std, F_D_std[0], dt_std, t0_std, tf_std,
                    True)
    t_0   = data_0[0]
    w_0   = data_0[2]
    ax3_311 = fig3.add_subplot(311)
    ax3_311.locator_params(tight=True, nbins=3)
    plt.plot(t_0,w_0,'k')
    plt.xlim(0,60)
    plt.ylim(-0.2,0.2)
    ax3_311.axes.get_xaxis().set_ticks([]) #Turn off x-axis tick marks.
    ax3_311.spines['top'].set_visible(False) #Remove the top frame line.
    ax3_311.spines['right'].set_visible(False)#Remove the right frame line.
    ax3_311.spines['bottom'].set_visible(False)#Remove the bottom frame line.
    ax3_311.yaxis.set_ticks_position('left')#Set all y-ticks to be on the left.
    xannot = 0.5
    yannot = 0.95
    ax3_311.annotate(r'$F_D$ = %2.2f'%0.0,
                 xy=(xannot, yannot-0.15), xycoords='axes fraction',
                 fontsize=16, horizontalalignment='left',
                 verticalalignment='top')
    ###########################################################################
    data_0p5 = main(w0_std, theta0_std, F_D_std[1], dt_std, t0_std, tf_std,
                    True)
    t_0p5 = data_0p5[0]
    w_0p5 = data_0p5[2]
    ax3_312 = fig3.add_subplot(312)
    ax3_312.locator_params(tight=True, nbins=3)
    plt.plot(t_0p5,w_0p5,'k')
    plt.ylabel(r'$\omega\ (\mathrm{radians/second})$')
    #Adjust the number of points between the y-axis and its label to prevent
    #overlap.
    ax3_312.yaxis.labelpad = 15
    plt.xlim(0,60)
    plt.ylim(-1,1)
    ax3_312.axes.get_xaxis().set_ticks([])
    ax3_312.spines['top'].set_visible(False)
    ax3_312.spines['right'].set_visible(False)
    ax3_312.spines['bottom'].set_visible(False)
    ax3_312.yaxis.set_ticks_position('left')
    ax3_312.annotate(r'$F_D$ = %2.2f'%0.5,
                 xy=(xannot, yannot+0.05), xycoords='axes fraction',
                 fontsize=16, horizontalalignment='left',
                 verticalalignment='top')
    ###########################################################################
    data_1p2 = main(w0_std, theta0_std, F_D_std[2], dt_std, t0_std, tf_std,
                    True)
    t_1p2 = data_1p2[0]
    w_1p2 = data_1p2[2]
    ax3_313 = fig3.add_subplot(313)
    ax3_313.locator_params(tight=True, nbins=3)
    plt.plot(t_1p2,w_1p2,'k')
    plt.xlabel(r'$t\ (\mathrm{seconds})$')
    plt.xlim(0,60)
    plt.ylim(-3,3)
    ax3_313.spines['top'].set_visible(False)
    ax3_313.spines['right'].set_visible(False)
    ax3_313.yaxis.set_ticks_position('left')
    ax3_313.xaxis.set_ticks_position('bottom')
    ax3_313.annotate(r'$F_D$ = %2.2f'%1.2,
                 xy=(xannot, yannot), xycoords='axes fraction', fontsize=16,
                 horizontalalignment='left',verticalalignment='top')

#Recreate Figure 3.8 from Giordano page 63.
if gen_fig_3p8_left == True:
    #w vs t for F_D=0.5
    fig4 = plt.figure(4, facecolor='white')
    plt.title(r'$\mathrm{Fig.\ 3.8\ (left):\ }\omega$'
              +r'$\mathrm{\ vs.\ }\theta$')
    ###########################################################################
    data_0p5 = main(w0_std, theta0_std, F_D_std[1], dt_std, t0_std, tf_std,
                    True)
    t_0p5     = data_0p5[0]
    theta_0p5 = data_0p5[1]
    w_0p5     = data_0p5[2]
    ax4 = fig4.add_subplot(111)
    plt.plot(theta_0p5[skip:],w_0p5[skip:],'k.', markersize=3, alpha=0.5)
    plt.ylabel(r'$\omega\ (\mathrm{radians/second})$')
    plt.xlabel(r'$\theta\ (\mathrm{radians})$')
    xannot = 0.02
    yannot = 0.08
    ax4.annotate(r'$F_D$ = %2.2f'%F_D_std[1]
                 + r'$,\ $' + r'$\mathrm{tol}$ = %2.3f'%tol
                 + r'$,\ $' + r'$\mathrm{skip}$ = %d'%skip
                 + r'$,\ $' + r'$t_f$ = %d$T_D$'%intmult_std,
                 xy=(xannot, yannot), xycoords='axes fraction', fontsize=16,
                 horizontalalignment='left',verticalalignment='top')
    plt.tight_layout()
    plt.show()

if gen_fig_3p8_right == True:
    #w vs t for F_D=1.2
    fig5 = plt.figure(5, facecolor='white')
    plt.title(r'$\mathrm{Fig.\ 3.8\ (right):\ }\omega$'
              +r'$\mathrm{\ vs.\ }\theta$')
    ###########################################################################
    data_1p2 = main(w0_std, theta0_std, F_D_std[2], dt_std, t0_std, tf_std,
                    True)
    t_1p2     = data_1p2[0]
    theta_1p2 = data_1p2[1]
    w_1p2     = data_1p2[2]
    ax5 = fig5.add_subplot(111)
    plt.plot(theta_1p2[skip:],w_1p2[skip:],'k.', markersize=3, alpha=0.5)
    plt.ylabel(r'$\omega\ (\mathrm{radians/second})$')
    plt.xlabel(r'$\theta\ (\mathrm{radians})$')
    xannot = 0.02
    yannot = 0.08
    ax5.annotate(r'$F_D$ = %2.2f'%F_D_std[2]
                 + r'$,\ $' + r'$\mathrm{tol}$ = %2.3f'%tol
                 + r'$,\ $' + r'$\mathrm{skip}$ = %d'%skip
                 + r'$,\ $' + r'$t_f$ = %d$T_D$'%intmult_std,
                 xy=(xannot, yannot), xycoords='axes fraction', fontsize=16,
                 horizontalalignment='left',verticalalignment='top')
    plt.tight_layout()
    plt.show()

#Recreate of Figure 3.9 from Giordano page 64.
if gen_fig_3p9 == True:
    #stroboscopic w vs theta for F_D = 1.2
    fig6 = plt.figure(6, facecolor='white')
    plt.title(r'$\mathrm{Fig.\ 3.9:\ Stroboscopic\ }\omega$'
              +r'$\mathrm{\ vs.\ }\theta$')
    ###########################################################################
    #Generate arrays for F_D = 1.2 and a large tf-value.
    data_strb = main(w0_std, theta0_std, F_D_std[2], dt_std, t0_std, tf_strb,
                     True)
    t_strb     = data_strb[0]
    theta_strb = data_strb[1]
    w_strb     = data_strb[2]
    ind_strb   = data_strb[3]
    #Trim the above arrays according to the list of stroboscopic indices.
    t_strb     = [t_strb[i] for i in ind_strb]
    theta_strb = [theta_strb[i] for i in ind_strb]
    w_strb     = [w_strb[i] for i in ind_strb]
    ax6 = fig6.add_subplot(111)
    plt.plot(theta_strb[skip:],w_strb[skip:],'k.', markersize=3, alpha=0.5)
    plt.ylabel(r'$\omega\ (\mathrm{radians/second})$')
    plt.xlabel(r'$\theta\ (\mathrm{radians})$')
    plt.xlim(-4,4)
    plt.ylim(-2,1)
    xannot = 0.02
    yannot = 0.13
    ax6.annotate(r'$F_D$ = %2.2f'%F_D_std[2]
                 + r'$,\ $' + r'$\mathrm{tol}$ = %2.4f'%tol
                 + '\n' + r'$\mathrm{skip}$ = %d'%skip
                 + r'$,\ $' + r'$t_f$ = %d$T_D$'%intmult_strb,
                 xy=(xannot, yannot), xycoords='axes fraction', fontsize=16,
                 horizontalalignment='left',verticalalignment='top')
    plt.tight_layout()
    plt.show()

#Recreate Figure 3.11 from Giordano page 68.
if gen_fig_3p11 == True:
    #for each F_D value, run main(), get ind_strb, and then use it to append
    #len(ind_strb) values of the current F_D value to F_D_parent. Use ind_strb
    #again to trim theta down to only stroboscopic indices, and then append
    #these values theta_parent. Note the use of skip to remove transients.
    F_D_parent = []
    theta_parent = []
    #Below we create linearly spaced samples, but heavily weighted such that
    #detailed part of the bifurcation diagram is sampled more.
    a = np.linspace(1.3,1.472,samp_bif//2)
    b = np.linspace(1.472,1.5,samp_bif//2)
    F_D = np.concatenate((a,b)) 

    #Wrap tdqm() around range() to incorporate a progress bar.
    for i in tqdm(range(0,len(F_D))):
        data = main(w0_std, theta0_std, F_D[i], dt_bif, t0_std, tf_bif, True)
        theta = data[1]
        index = data[3]
        theta_strb = [theta[i] for i in index]
        for j in range(skip,len(index)):
            F_D_parent.append(F_D[i])
        theta_parent = theta_parent + theta_strb[skip:]
    fig7 = plt.figure(7, facecolor='white')
    plt.title(r'$\mathrm{Fig.\ 3.11:\ }\theta\ $'+r'$\mathrm{\ vs.\ }F_D$'
              +r'$\mathrm{\ Bifurcation\ Diagram}$')
    ###########################################################################
    ax7 = fig7.add_subplot(111)
    plt.plot(F_D_parent,theta_parent,'k.', markersize=1, alpha=0.3)
    plt.ylabel(r'$\theta\ (\mathrm{radians})$')
    plt.xlabel(r'$F_D\ \mathrm{(radians/second^2)}$')
    plt.xlim(1.35,1.49)
    plt.ylim(0.9,3)
    xannot = 0.02
    yannot = 0.08
    ax7.annotate(r'$\mathrm{tol}$ = %2.4f'%tol
                 + r'$,\ $' + r'$\mathrm{skip}$ = %d'%skip
                 + r'$,\ $' + r'$t_f$ = %d$T_D$'%intmult_bif
                 + r'$,\ $' + r'$dt$ = $T_D$/%d'%intmult_dt_bif
                 + r'$,\ $' + '%d '%samp_bif + r'$F_D\mathrm{\ samples}$',
                 xy=(xannot, yannot), xycoords='axes fraction', fontsize=10,
                 horizontalalignment='left',verticalalignment='top')
    plt.tight_layout()
    plt.show()

#Giordano exercise 3.18, stroboscopic w vs theta for F_D = 1.4, 1.44, 1.465,
#as given on p.69.
if gen_ex_318 == True:
    #Initialize an array of nonstandard F_D amplitudes.
    F_D_strb = [1.4, 1.44, 1.465]

    #Generate a stroboscopic w vs theta plot for F_D = 1.4.
    fig8 = plt.figure(8, facecolor='white')
    plt.title(r'$\mathrm{Ex.\ 3.18(a):\ Stroboscopic\ }\omega$'
                  +r'$\mathrm{\ vs.\ }\theta$')
    ###########################################################################
    data_strb = main(w0_std, theta0_std, F_D_strb[0], dt_std, t0_std, tf_strb,
                     True)
    t_strb     = data_strb[0] #Store each output list separately.
    theta_strb = data_strb[1]
    w_strb     = data_strb[2]
    ind_strb   = data_strb[3]
    #Trim these lists according to the list of stroboscopic indices.
    t_strb     = [t_strb[i] for i in ind_strb]
    theta_strb = [theta_strb[i] for i in ind_strb]
    w_strb     = [w_strb[i] for i in ind_strb]
    ax8 = fig8.add_subplot(111)
    plt.plot(theta_strb[skip:],w_strb[skip:],'k.', markersize=3, alpha=0.3)
    ax8.ticklabel_format(useOffset=False)#Ensure yticks dont overlap the title.
    plt.ylabel(r'$\omega\ (\mathrm{radians/second})$')
    plt.xlabel(r'$\theta\ (\mathrm{radians})$')
    plt.xlim(1.3,1.7)
    plt.ylim(-1.8,-1.76)
    xannot = 0.63 #Define relative coordinates for an annotation.
    yannot = 0.16
    ax8.annotate(r'$F_D$ = %2.2f'%F_D_strb[0]
                 + r'$,\ $' + r'$\mathrm{tol}$ = %2.4f'%tol
                 + '\n' + r'$\mathrm{skip}$ = %d'%skip
                 + r'$,\ $' + r'$t_f$ = %d$T_D$'%intmult_strb,
                 xy=(xannot, yannot), xycoords='axes fraction', fontsize=16,
                 horizontalalignment='left',verticalalignment='top')
    plt.tight_layout()
    plt.show()

    #Generate another plot for F_D = 1.44.
    fig9 = plt.figure(9, facecolor='white')
    plt.title(r'$\mathrm{Ex.\ 3.18(b):\ Stroboscopic\ }\omega$'
                  +r'$\mathrm{\ vs.\ }\theta$')
    ###########################################################################
    data_strb = main(w0_std, theta0_std, F_D_strb[1], dt_std, t0_std, tf_strb,
                     True)
    t_strb     = data_strb[0]
    theta_strb = data_strb[1]
    w_strb     = data_strb[2]
    ind_strb   = data_strb[3]
    t_strb     = [t_strb[i] for i in ind_strb]
    theta_strb = [theta_strb[i] for i in ind_strb]
    w_strb     = [w_strb[i] for i in ind_strb]
    ax9 = fig9.add_subplot(111)
    plt.plot(theta_strb[skip:],w_strb[skip:],'k.', markersize=3, alpha=0.3)
    plt.ylabel(r'$\omega\ (\mathrm{radians/second})$')
    plt.xlabel(r'$\theta\ (\mathrm{radians})$')
    ax9.annotate(r'$F_D$ = %2.2f'%F_D_strb[1]
                 + r'$,\ $' + r'$\mathrm{tol}$ = %2.4f'%tol
                 + '\n' + r'$\mathrm{skip}$ = %d'%skip
                 + r'$,\ $' + r'$t_f$ = %d$T_D$'%intmult_strb,
                 xy=(xannot+0.02, yannot), xycoords='axes fraction',
                 fontsize=16, horizontalalignment='left',
                 verticalalignment='top')
    plt.tight_layout()
    plt.show()

    #Generate another plot for F_D = 1.465.
    fig10 = plt.figure(10, facecolor='white')
    plt.title(r'$\mathrm{Ex.\ 3.18(c):\ Stroboscopic\ }\omega$'
                  +r'$\mathrm{\ vs.\ }\theta$')
    ###########################################################################
    data_strb = main(w0_std, theta0_std, F_D_strb[2], dt_std, t0_std, tf_strb,
                     True)
    t_strb     = data_strb[0]
    theta_strb = data_strb[1]
    w_strb     = data_strb[2]
    ind_strb   = data_strb[3]
    t_strb     = [t_strb[i] for i in ind_strb]
    theta_strb = [theta_strb[i] for i in ind_strb]
    w_strb     = [w_strb[i] for i in ind_strb]
    ax10 = fig10.add_subplot(111)
    plt.plot(theta_strb[skip:],w_strb[skip:],'k.', markersize=3, alpha=0.3)
    plt.ylabel(r'$\omega\ (\mathrm{radians/second})$')
    plt.xlabel(r'$\theta\ (\mathrm{radians})$')
    ax10.annotate(r'$F_D$ = %2.2f'%F_D_strb[2]
                 + r'$,\ $' + r'$\mathrm{tol}$ = %2.4f'%tol
                 + '\n' + r'$\mathrm{skip}$ = %d'%skip
                 + r'$,\ $' + r'$t_f$ = %d$T_D$'%intmult_strb,
                 xy=(xannot+0.03, yannot), xycoords='axes fraction',
                 fontsize=16, horizontalalignment='left',
                 verticalalignment='top')
    plt.tight_layout()
    plt.show()