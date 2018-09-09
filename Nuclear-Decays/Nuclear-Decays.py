# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 18:51:48 2018

@author: Cemenenkoff

This code is primarily adjusted via the steps parameter, and how the g list is
defined. If the g list only contains one value, then a full analysis ensues. If
the g list contains multiple values, then only Numerical Nb/Na vs. T is plotted
for each value of g.
"""

import matplotlib.pyplot as plt
import numpy as np
import math
plt.style.use('classic')
title_font = '20'
label_font = '16'

#Try out Covey's preamble
import matplotlib
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.autolayout'] = False 
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor']='white'
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

"""
calc_g_vs_tdist is a switch between two scripts.
If True:
    The code generates a gamma vs Absolute Error Bound for T plot
    NOTE: takes ~30 mins for current settings.

If False, and if there is one gamma value in the g list:
    The code does a full numerical analysis for that (single) gamma value.
    If that value of gamma is 1, then a tab delimited .txt file of T and Nb/Na
    is generated and saved as Cemenenkoff-PHYS486-HW1.txt as long as there is
    not a pre-existing file of the same name in the working directory.
If False, and if there are multiple gamma values in the g list:
    The code creates a stacked figure for Nb/Na vs. T but skips the other
    figures.
    NOTE: The stacked figure is calibrated for FIVE gamma values. You may
          calibrate these values on lines 120-121.
"""
calc_g_vs_tdist = False

#If you aren't sure what you're doing with the calc_g_vs_tdist plot,
#don't touch anything between the **** lines! The calibration is finicky.
#******************************************************************************
if calc_g_vs_tdist == True:
    Na_start = 1 #percentage of the starting pop of Na, normalized by Na
    Nb_start = 0 #percentage of the starting pop of Nb, normalized by Na, given
                 #in the handout as Nb(0)=0.
    
    #g=tau_a/tau_b, where tau_a and tau_b are time constants associated with the
    #individual decay rates of Na and Nb.
    
    #g = np.arange(0, 5, 0.001) #linear sampling
    g = np.logspace(-2,0.69897000, 500) #log10 sampling. start power, end power,
                                        #steps.
    #10**3 steps seems like too many. Try 500 on the next run.
    
    #choose parameters for the scaled time data.
    start = 0  #start time in years, normalized by tau_a
    stop  = 10 #end time in years, also normalized by tau_a
    steps = 10**7 +1 #number of time steps
    
    #generate a linear array of scaled time values (T=t/tau_a).
    T  = np.linspace(start,stop,steps,retstep=True,endpoint=True) 
    #Since retstep=True, np.linspace returns both the T array and the step size.
    dT = T[1] #grab the step size, defined as Delta(t)/tau_a
    T  = T[0] #grab just the (scaled) T values in a numpy array
    
    Nrat_meas = 0.2
    prec = 0.005
    Tdist_list = [] #list to store the absolute error bound for T for each g.
    plt.figure(facecolor="white")
    fig1 = plt.figure(1)
    for j in range(len(g)):
        Na_list = [Na_start]
        Nb_list = [Nb_start]
        for i in range(steps-1):
            Na = Na_list[i] - Na_list[i]*(dT)
            Nb = Na_list[i]*(dT) - g[j]*Nb_list[i]*(dT) + Nb_list[i]
            Na_list.append(Na)
            Nb_list.append(Nb)
        Na_arr = np.asarray(Na_list)
        Nb_arr = np.asarray(Nb_list)   
        Nratio = Nb_arr/Na_arr
        Nrat_meas_hi = (1+prec)*Nrat_meas
        Nrat_meas_lo = (1-prec)*Nrat_meas
        Nratio_list = Nratio.tolist()
        T_list = T.tolist()
        
        meas_ind_hi = min(range(len(Nratio_list)),
                          key=lambda i: abs(Nratio_list[i]-Nrat_meas_hi))
        meas_ind_lo = min(range(len(Nratio_list)),
                          key=lambda i: abs(Nratio_list[i]-Nrat_meas_lo))
        
        Tdist = abs( T_list[meas_ind_hi] - T_list[meas_ind_lo] )
        print('%2.8f, %2.8f'%(g[j],Tdist))
        Tdist_list.append(Tdist)
    
    plt.plot(g,Tdist_list, '-')
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'$T_{\mathrm{hi}}-T_{\mathrm{lo}}$')
    plt.title(r'$\gamma \mathrm{\ vs.\ Absolute\ Error\ Bound\ } T_{\mathrm{hi}}-T_{\mathrm{lo}}$')
#******************************************************************************

if calc_g_vs_tdist == False:
    Na_start = 1 #percentage of the starting pop of Na, normalized by Na
    Nb_start = 0 #percentage of the starting pop of Nb, normalized by Na, given
                 #in the handout as Nb(0)=0.
    
    #g=tau_a/tau_b, where tau_a and tau_b are time constants associated with the
    #individual decay rates of Na and Nb.
    
    g = np.linspace(1.0,2.0, 5) #5 gammas greater than or equal to 1.0
    #g = np.linspace(0.0,1.0, 5) #5 gammas less than or equal to 1.0
    #g = [1] #when g=[1], the errors get so tiny that machine precision comes
            #into play. To avoid this, use 0.9999 for the g=1 case if you
            #want a pretty %Error in Numerical Nb/Na vs T plot.
    
    #choose parameters for the scaled time data.
    start = 0  #start time in years, normalized by tau_a
    stop  = 10 #end time in years, also normalized by tau_a
    steps = 10**5 +1 #number of time steps
    #Use steps = 10**4+1 when emulating Covey's table
    
    #generate a linear array of scaled time values (T=t/tau_a).
    T  = np.linspace(start,stop,steps,retstep=True,endpoint=True) 
    dT = T[1] #grab the step size, defined as Delta(t)/tau_a
    T  = T[0] #grab just the (scaled) T values in a numpy array
    #print(T, type(T), len(T))
    plt.figure(facecolor="white")
    fig1 = plt.figure(1)
    max_yval = 0.0 #Initialize the max_yval for scaling the plot.
    for j in range(len(g)):
        #Below are hand-coded RGB gradients, calibrated for 5 gamma values to be
        #plotted simultaneously.
        if max(g)<=1:
            clr = ((255-60*j)/255,0,(0+60*j)/255) #gradient from red to blue.
        if max(g)>1:
            clr = (0,(0+60*j)/255,(255-60*j)/255) #gradient from blue to green
        #The first value in each list that we are filling up is defined via the
        #Na_start, Nb_start initial conditions given on lines 16 and 17.
        Na_list = [Na_start]
        Nb_list = [Nb_start]
        #t runs from 0 to (steps-1), so there are (steps-1)-start+1 steps total
        for i in range(steps-1):
            #print('%4.2d, %4.2d' % (i, len(Na_list)))
            Na = Na_list[i] - Na_list[i]*(dT)
            Nb = Na_list[i]*(dT) - g[j]*Nb_list[i]*(dT) + Nb_list[i]
            Na_list.append(Na)
            Nb_list.append(Nb)
            #The next Na value is determined by the i-th value in the Na_list the
            #inner for loop is iterating through. The next Nb value is determined
            #by i-th value in both the Na and Nb lists.
        Na_arr = np.asarray(Na_list) #Change to numpy arrays for graphing.
        Nb_arr = np.asarray(Nb_list)   
        Nratio = Nb_arr/Na_arr
        if max(Nratio)>max_yval:
            max_yval = max(Nratio) #log the max Nratio to scale the y-axis later.
        #print(len(Nb_arr),len(Na_arr), len(t), len(Nratio))
        plt.plot(T, Nratio, '-', c = clr,
                 label=r'$\gamma=\frac{\tau_a}{\tau_b}=$'+'%1.1f' % (g[j]))
        
    #is_equal compares two floats and declares them equal if they are within the
    #defined maximum relative difference.
    def is_equal(x, y, maxreldiff):
        diff = float(abs(x-y))
        x = float(abs(x))
        y = float(abs(y))
        largest = float(max(x, y))
        if diff <= maxreldiff*largest:
            return True
        else:
            return False
            
    #Write tab-delimited results to a text file for g=1.
    if len(g)==1 and is_equal(g[0],1,0.01)==True:
        f = open('Cemenenkoff-PHYS486-HW1.txt','w')
        for i in range(len(T)):
            #print(T[i],Nratio[i])
            f.write('%f'%T[i]+'\t'+'%f\n'%Nratio[i])
        f.close()
    
    #After all of the lines are plotted, format and label and add a legend.
    plt.title(r'$\mathrm{Numerical\ }N_B/N_A \mathrm{\ vs.\ } T$',
              fontsize = title_font)
    plt.xlabel(r'$T$', fontsize = label_font)
    plt.ylabel(r'$N_B/N_A$', fontsize = label_font)
    plt.xlim(0,T[-1])
    ax1 = fig1.add_subplot(111)
    #Define the plot details for the two major classes of gamma.
    #(xannot, yannot) represents the coordinates for an annotation.
    #Case 1: gammas less than one, analyzing multiple values.
    if max(g) <= 1 and len(g)!=1:
        xannot = 0.02
        plt.ylim(0,0.005*max_yval)
    #Case 2: gammas greater than one or when analyzing just one gamma value.
    if max(g) > 1 or len(g)==1:
        xannot = 0.02
        plt.ylim(0,max_yval)
    yannot = 0.87
    yoffset= 0.08
    ax1.annotate(r'$\mathrm{time\ steps}=$'+'%d' % (steps-1),
                 xy=(xannot, yannot-(len(g)-1)*yoffset), xycoords='axes fraction',
                 fontsize=14, horizontalalignment='left', verticalalignment='top')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    #This part of the code is designed for only ONE value in the g list.
    if len(g)==1:
        #Pick an Nb/Na value to use as the measured value for this thought experiment.
        #Make sure this is a reasonably attainable value for the given gamma.
        Nrat_meas = Nratio[math.floor(steps/2)]
        #The code will output an associated T interval given the Nrat_meas value's
        #precision of plus or minus 0.5% after the numerical solution is constructed.
        
        Nrat_meas_hi = (1+0.005)*Nrat_meas #The high end of the Nb/Na measurement.
        Nrat_meas_lo = (1-0.005)*Nrat_meas #The low end of the Nb/Na measurement.
        Nratio_list = Nratio.tolist() #Change the arrays to lists so we can easily
        T_list = T.tolist()           #retrieve important indices.
        
        #Below, we return the index of the value in Nratio_list that is closest to the
        #Nrat_meas_hi and Nrat_meas_lo values. We then put those indices into the T
        #array to get the associated error bounds on T.
        meas_ind_hi = min(range(len(Nratio_list)),
                          key=lambda i: abs(Nratio_list[i]-Nrat_meas_hi))
        meas_ind_lo = min(range(len(Nratio_list)),
                          key=lambda i: abs(Nratio_list[i]-Nrat_meas_lo))
        
        print('The associated T-interval for Nb/Na=%4.3f,'%(Nrat_meas))
        print('given 0.5% measurement precision, is')
        print('(%f,%f) years/tau_a'%(T_list[meas_ind_lo], T_list[meas_ind_hi]))
        
        T_dist = T_list[meas_ind_hi] - T_list[meas_ind_lo]
        
        #Below returns the (index, value) combo for the value in Nratio_list that is
        #closest to the Nrat_meas value. Use this for debugging.
        #TEST = min(enumerate(Nratio_list), key=lambda x: abs(x[1]-Nrat_meas))
        #print('TEST:', end="")
        #print(TEST)
        
        Na_true = np.exp(-T)
        if g[0]!=1:
            Nb_true = Nb_start/Na_start*np.exp(-g[0]*T)+(np.exp(-T)-np.exp(-g[0]*T))/(g[0]-1)
        if g[0]==1:
            Nb_true = Nb_start/Na_start*np.exp(-T)+(T)*np.exp(-T)
        Nratio_true = Nb_true/Na_true
        
        
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)
        xannot = 0.88
        yannot = 0.82
        ax2.annotate(r'$\gamma=$'+'%1.1f' % (g[0]), xy=(xannot, yannot),
                                       xycoords='axes fraction', fontsize=14,
                                       horizontalalignment='left',
                                       verticalalignment='top')
        plt.plot(T, Na_true, '-', c = 'green', label=r'$N_A$')
        plt.plot(T, Nb_true, '-', c = 'black', label=r'$N_B$')
        plt.title(r'$\mathrm{True\ } N_B \mathrm{\ and\ True\ }N_A \mathrm{\ vs.\ }T$',
                  fontsize = title_font)
        plt.xlabel(r'$T$', fontsize = label_font)
        plt.ylabel(r'$\%\mathrm{\ Nuclei\ Remaining}$', fontsize = label_font)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        
        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(111)
        if g[0] <=1:
            xannot = 0.02
        if g[0] > 1:
            xannot = 0.60
        yannot = 0.80
        ax3.annotate(r'$\gamma=$'+'%1.1f' % (g[0])+r'$\mathrm{,\ time\ steps}=$'+'%d'%(steps),
                                             xy=(xannot, yannot),
                                             xycoords='axes fraction', fontsize=14,
                                             horizontalalignment='left',
                                             verticalalignment='top')
        plt.plot(T, Nratio_true, '-', c = 'red', label=r'$\mathrm{True\ }N_B/N_A$')
        plt.plot(T, Nratio, '--', c = 'blue',
                 label=r'$\mathrm{Numerical\ }N_B/N_A$', alpha=0.75)
        plt.title(r'$\mathrm{True\ and\ Numerical\ } N_B/N_A \mathrm{\ vs.\ }T$',
                  fontsize = title_font)
        plt.xlabel(r'$T$', fontsize = label_font)
        plt.ylabel(r'$N_B/N_A$', fontsize = label_font)
        if g[0] <=1:
            plt.legend(loc='upper left')
        if g[0] > 1:
            plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
        
        #We use a for loop here to circumvent the first error value because it
        #causes division by zero in the absolute relative error formula.
        error = []
        for x in range(1,len(Nratio)): #start from 1 because Nratio_true[0]=0 which
                                       #breaks the error formula
            #Each error value is PERCENT error.
            error_val = abs(Nratio_true[x]-Nratio[x])*100/Nratio_true[x]
            error.append(error_val)
        #print(len(error), len(T)) #len(T) should be 1 more than len(error)
        #print(error)
        
        from matplotlib.ticker import FormatStrFormatter
        fig4 = plt.figure(4)
        ax4 = fig4.add_subplot(111)
        xannot = 0.02
        yannot = 0.87
        ax4.annotate(r'$\gamma=$'+'%1.1f'%(g[0])+r'$\mathrm{,\ time\ steps}=$'+'%d'%(steps),
                                           xy=(xannot, yannot),
                                           xycoords='axes fraction', fontsize=14,
                                           horizontalalignment='left',
                                           verticalalignment='top')
        #Skip the T=0 value because the error isn't physically meaningful at that time.
        plt.plot(T[1:], error, '--', c = 'grey',
                 label=r'$\%\mathrm{\ \ Error\ in\ }N_B/N_A$')
        plt.title(r'$\%\mathrm{\ Error\ in\ Numerical\ }N_B/N_A\mathrm{\ vs.\ }T$',
                  fontsize = title_font, y=1.06)
        plt.xlabel(r'$T$', fontsize = label_font)
        plt.ylabel(r'$\%\mathrm{\ Error\ in\ Numerical\ }N_B/N_A$',
                   fontsize = label_font)
        ax4.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
        
        print('Minimum %% Error: %2.6f, Maximum %% Error: %2.6f'%(min(error),max(error)))
        if max(error)<0.05:
            print('Results are within 0.5% error tolerance.')
            print('You may trust these results.')
        if max(error)>=0.05:
            print('Results are NOT within 0.5% error tolerance.')
            print('Do not trust these results!')