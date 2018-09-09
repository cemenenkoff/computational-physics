# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 07:38:02 2018

@author: Cemenenkoff
This code models two identical objects: one is a free-falling projectile while
the other is a horizontally shot projectile. Both objects start at the same
height, but the shot object is given a nonzero initial velocity. Both
projectiles stop after collision with the ground.

This code creates the following five figures:
    1. Difference in Collision Times (Dt) vs. Object Parameter (C/m)
    2. Dt vs. Initial Velocity (vx0)
    3. Range (x) vs. Time (t)
    4. Height (y) vs. t
    5. Shot Object Absolute Overshoot Error (e) vs Time Step (dt)

A tab-delimited .txt file of t-, x-, and y-values may also be created for
specific initial conditions (governed by the Com_cov variable). Whether or not
to generate the .txt file is regulated by the gen_text_file boolean below.
"""
gen_text_file = False

#Covey's preamble with a few customizations.
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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

#Quick-and-Dirty References:
#https://en.wikipedia.org/wiki/Drag_(physics)
#http://hyperphysics.phy-astr.gsu.edu/hbase/Mechanics/quadrag.html

"""
The main() function has good looping capability for any of the Tuneable
Parameters below (but will need to be adjusted accordingly). Main Function
Inputs are specially chosen from Tuneable Parameters as they are of some
analytical interest, e.g.
       for i in range(len(vx0)):
           do something,
to be able to see how varying one parameter (like vx0) can change the behavior
of the system.
"""

#Tuneable Parameters:
x0 = 0.0 #initial x-position (m) for both objects
vy0 = 0.0 #initial y-velocity (m/s) for both objects
g = 9.807 #magnitude of gravitational acceleration on Earth (m/s^2)
Cd = 0.47 #drag coefficient (dimensionless). 0.47 is for a sphere.
          #see https://en.wikipedia.org/wiki/Drag_coefficient
p = 1.225 #density of the fluid (kg/m^3). 1.225 is for air.
A = 1.0 #cross-sectional area (m^2)
C = 0.5*Cd*p*A #total coefficient of the drag force. (kg/m)
m = 100.0 #projectile mass (kg)
t0 = 0.0 #start time (s)
tf = 600 #end time (s)
#IMPORTANT NOTE:
#Make sure tf is large to ensure a collision. If not, the code will run into
#an error when trying to construct Figure 1 and Figure 2 because they both rely
#on Dt (i.e. no collision means the hit tuple will return empty).

#Standard Input Values for main():
y0_std = 10**(3) #initial y-position (m)
dt_std = 10**(-3) #standard time step.
vx0_std = 10**(3) #standard initial x-velocity for the shot object
Com_std = C/m #standard C "over" m for a 100kg sphere with a 1.0 m^2
              #cross-section in air. Note that Dividing C by m collapes Cd, p,
              #A, C, and m into one variable: Com (units of 1/m)
              
#Below is the Com needed to generate Covey's requested .txt file for a 100kg
#sphere with cross-sectional area 0.1 square meters, launched from a height of
#10**(3.5) meters, and with an initial velocity of 10.0 meters/second.
Com_cov = 0.5*0.5*1.225*0.1/100.0
        # 0.5* Cd*  p  * A / m

#Main Function Inputs:             
vx0 = np.linspace(0.0,3*10**(3),100+1) #initial x-velocity list (m/s)
Com = np.linspace(10**(-3),10**(0),100+1) #C/m list
dt = np.linspace(10**(-4),10**(-2), 500+1) #dt list (s)

"""
Math for the main() Function:

We want to solve the vector equation (vectors are in parentheses):
 m(d(v)/dt) = m(g)-k||(v)||*(v)
 where
  m is the mass of the projectile
 (v) is the velocity
 (g) is the acceleration due to gravity
  k is the drag coefficient
 (d(v)/dt) is the time-derivative of velocity, and
 ||(v)|| is the magnitude of velocity.

 Define a(t,(v)) = (g) - k/m||(v)||(v),
 thus (d(v)/dt) = a(t,(v)).

We are going to use this vector form, but split it into x- and y-cases. Once
we obtain dv/dt in the x- and y-cases, we can find the x- and y-positions.
"""

#Note: The inputs of main are meant to be floating points. The outputs are five
#lists and a 6-element tuple if there is a collision (and 0-element if not).
def main(vx0, Com, dt, y0):
    steps = int(round((tf-t0)/dt)) #total number of time steps (dimensionless)
    t = np.linspace(t0, tf, steps+1) #Generate the time list.
    
    #Initialize lists to store values to during the iteration.
    x = [x0]   #x-position values, initialized with the starting value
    y = [y0]   #y-position values, "
    vx = [vx0] #x-velocity values, "
    vy = [vy0] #y-velocity values, "
    hit = () #hit is an empty tuple to store collision information.
    
    #Define a function describing dvx/dt (the acceleration in the x-direction,
    #ax.) The negative is out front because (air) drag always opposes the
    #x-motion.
    def ax(vx, vy):
        return -Com*np.sqrt(vx**2+vy**2)*vx
    
    #Define a function describing dvy/dt (the acceleration in the y-direction,
    #ay.) A negative is in front of g because it is given as a magnitude in
    #Tuneable Parameters. The other negative in front of (C/m) is there because
    #drag always opposes the y-motion. If we launch the projectile downward,
    #y-drag will push upward. If we launch upward, y-drag will push downward. 
    def ay(vx, vy):
        return -g-Com*np.sqrt(vx**2+vy**2)*vy
          
    #The for loop below uses the Runge-Kutta 2nd Order Method for solving ODEs.
    #Specifically, the values for the i+1th position in the arrays being built
    #are calculated halfway between the ith and i+1th time steps to reduce
    #numerical error.
    for i in range(0,steps):
        #On the ith time step, find the x- and y- accelerations given the ith
        #x- and y-velocities, and then multiply those accelerations by half of
        #a time step to estimate the x- and y- velocities halfway before the
        #next time step. This is justified by the definition of average
        #acceleration: a=(vf-vi)/dt -> vf = vi+a*dt
        vx_hw = vx[i] + ax(vx[i],vy[i])*(dt/2)
        vy_hw = vy[i] + ay(vx[i],vy[i])*(dt/2)
        
        #Using the halfway (hw) x- and y- velocities, find the next x- and
        #y-velocities a full time step away from the ith time step
        #(i.e. the i+1th step).
        vx_ip1 = vx[i] + ax(vx_hw,vy_hw)*dt
        vy_ip1 = vy[i] + ay(vx_hw,vy_hw)*dt
        
        #Using the definition of average velocity:
        #v=(sf-si)/dt -> sf = si+v*dt for a position s,
        #find the next (i.e. i+1th) x- and y-positions a full time step away
        #from the ith time step using the halfway x- and y-velocities.
        x_ip1 = x[i] + vx_hw*dt
        y_ip1 = y[i] + vy_hw*dt
        
        #Append each i+1th value to its corresponding list.
        x.append(x_ip1)
        y.append(y_ip1)
        vx.append(vx_ip1)
        vy.append(vy_ip1)
        
        #If the projectile hits the ground, make it stop.
        if y_ip1<=0:
            #The numerical error of the time to hit is directly associated with
            #the absolute value of the negative y-value in the iteration that
            #signifies a collision.
            e_ip1 = abs(y_ip1) 
            
            #linearly interpolate between the i+1th y-value that breaches 0 and
            #the previous value to estimate the collision time.
            t_ip1 = t0+(i+1)*dt #the i+1th time value
            t_i = t0+(i)*dt #the ith time value
            y_i = y[-2] #the ith y-value is now the 2nd-to-last value in the y
                        #list
            t_hit = t_ip1 - y_ip1*(t_ip1 - t_i)/(y_ip1-y_i) #point-slope form
            #Store relevant collision information in the hit tuple below.
            hit = tuple( [t_hit, x_ip1, y_ip1, vx_ip1, vy_ip1, e_ip1] )
                         #    0,     1,     2,      3,      4,     5
            break #Since a collision occurred, break out of the loop.

    return t[0:len(y)],x,y,vx,vy,hit #Return relevant lists and the hit tuple.
          #          0,1,2, 3, 4,  5 #Note the truncation of t.

if gen_text_file == True:
    #Write tab-delimited results to a text file for a 100kg sphere with cross-
    #sectional area 0.1, launched from a height of 10**(3.5) with vx0=10.0.
    t_tofile = main(10.0, Com_cov, 0.001, 10**(3.5))[0]
    x_tofile = main(10.0, Com_cov, 0.001, 10**(3.5))[1]
    y_tofile = main(10.0, Com_cov, 0.001, 10**(3.5))[2]
    
    #Get the index of the value in the time-array which corresponds to t=3.0.
    t_3s_ind = min(range(len(t_tofile)), key=lambda i: abs(t_tofile[i]-3.0))
    f = open('Cemenenkoff-PHYS486-HW2.txt','w')
    #Write associated (x,y) data from t0=0.0 to tf=3.0 for Covey's table.
    for i in range(t_3s_ind+1):
        f.write('%f'%t_tofile[i]+'\t'+'%f'%x_tofile[i]+'\t'+'%f\n'%y_tofile[i])
    f.close()

#Figure 1: Dt vs C/m 
Dt = [] #Initialize a list to store differences between collision times.
for i in range(len(Com)):
    shot = main(vx0_std, Com[i], dt_std, y0_std) #shot object variables have _s
    fall = main(0.0, Com[i], dt_std, y0_std) #falling object variables have _f
    
    t_s = shot[0] #time list
    x_s = shot[1] #x-position list
    y_s = shot[2] #y-position list
    vx_s = shot[3] #x-velocity list
    vy_s = shot[4] #y-velocity list
    hit_s = shot[5] #collision information (tuple)
    t_hit_s = hit_s[0] #time associated with the collision
    
    t_f = fall[0]
    x_f = fall[1]
    y_f = fall[2]
    vx_f = fall[3]
    vy_f = fall[4]
    hit_f = fall[5]
    t_hit_f = hit_f[0]
    
    Dt.append(abs(t_hit_s-t_hit_f)) #Append the absolute value of the
    #difference between collision times for the shot and falling objects.
plt.figure(facecolor="white")
plt.plot(Com,Dt, color='blue', linestyle='-')
plt.title(r'$\mathrm{Difference\ in\ Collision\ Times\ vs.}$'
          +r'$\mathrm{\ Object\ Parameter}$')
plt.xlabel(r'$C/m\ (\mathrm{meters}^{-1})$')
plt.xlim(Com[0],Com[-1])
plt.ylabel(r'$\Delta t\ \mathrm{(seconds)}$')
#Annotate relevant information. Coordinates are in data units.
plt.text(0.85, 2.3,r'$y_0$ = '+str(y0_std)+
         '\n'+r'$v_{x_0}$ = '+str(vx0_std)+
         '\n'+r'$t_0$ = '+str(t0)+
         '\n'+r'$t_f$ = '+str(tf)+
         '\n'+r'$dt$ = '+str(dt_std))
plt.tight_layout()
plt.show()

#Figure 2: vx0 vs Dt
Dt = []
for i in range(len(vx0)):
    shot = main(vx0[i], Com_std, dt_std, y0_std) #shot object variables have _s
    fall = main(0.0, Com_std, dt_std, y0_std) #falling object variables have _f
    t_s = shot[0]
    x_s = shot[1]
    y_s = shot[2]
    vx_s = shot[3]
    vy_s = shot[4]
    hit_s = shot[5]
    t_hit_s = hit_s[0]
    t_f = fall[0]
    x_f = fall[1]
    y_f = fall[2]
    vx_f = fall[3]
    vy_f = fall[4]
    hit_f = fall[5]
    t_hit_f = hit_f[0]
    Dt.append(abs(t_hit_s-t_hit_f))
plt.figure(facecolor="white")
plt.plot(vx0,Dt, color='blue', linestyle='-', label='')
plt.title(r'$\mathrm{Difference\ in\ Collision\ Times\ vs.}$'
          +r'$\mathrm{\ Initial\ Velocity}$')
plt.xlabel(r'$v_{x_0}\ \left(\mathrm{\frac{meters}{second}}\right)$')
plt.xlim(0,vx0[-1])
plt.ylabel(r'$\Delta t\ \mathrm{(seconds)}$')
#Annotate relevant information. Coordinates are in data units.
#Limit Com_std to 2 decimal places for annotations.
Com_anot = "{0:.2e}".format(Com_std)
plt.text(2340, 0.10,r'$y_0$ = '+str(y0_std)+
         '\n'+r'$C/m$ = '+str(Com_anot)+
         '\n'+r'$t_0$ = '+str(t0)+
         '\n'+r'$t_f$ = '+str(tf)+
         '\n'+r'$dt$ = '+str(dt_std))
plt.tight_layout()
plt.show()

#Calculations for x/y vs t figures.
shot = main(vx0_std, Com_std, dt_std, y0_std)
fall = main(0.0, Com_std, dt_std, y0_std)
t_s = shot[0]
x_s = shot[1]
y_s = shot[2]
vx_s = shot[3]
vy_s = shot[4]
hit_s = shot[5]
t_hit_s = hit_s[0]
t_f = fall[0]
x_f = fall[1]
y_f = fall[2]
vx_f = fall[3]
vy_f = fall[4]
hit_f = fall[5]
t_hit_f = hit_f[0]
Dt = abs(t_hit_s-t_hit_f)
Dt_anot = "{0:.2f}".format(Dt) #Limit Dt to 2 decimal places for annotations.

#Figure 3: x vs t
plt.figure(facecolor="white") 
plt.plot(t_s,x_s, color='blue', linestyle='-', label=r'$\mathrm{shot}$')
plt.plot(t_s,np.zeros(len(x_s)), color='red', linestyle='-',
         label=r'$\mathrm{falling}$')
plt.title(r'$\mathrm{Range\ vs.\ Time}$')
plt.xlabel(r'$t\ \mathrm{(seconds)}$')
plt.xlim(0,t_s[-1])
plt.ylabel(r'$x\ \mathrm{(meters)}$')
plt.legend(loc='best', #Put the annotation for this figure in the legend.
           title=r'$y_0$ = '+str(y0_std)+
           '\n'+r'$v_{x_0}$= '+str(vx0_std)+
           '\n'+r'$C/m$ = '+str(Com_anot)+
           '\n'+r'$t_0$ = '+str(t0)+
           '\n'+r'$t_f$ = '+str(tf)+
           '\n'+r'$dt$ = '+str(dt_std))
plt.tight_layout()
plt.show()

#Figure 4: y vs. t
plt.figure(facecolor="white") 
plt.plot(t_s,y_s, color='blue', linestyle='-', label=r'$\mathrm{shot}$')
plt.plot(t_f,y_f, color='red', linestyle='-', label=r'$\mathrm{falling}$')
plt.axvline(t_hit_s, color='grey', linestyle='-', alpha=0.25)
plt.title(r'$\mathrm{Height\ vs.\ Time}$')
plt.xlabel(r'$t\ \mathrm{(seconds)}$')
plt.xlim(0,t_s[-1])
plt.ylabel(r'$y\ \mathrm{(meters)}$')
plt.ylim(0,y0_std)
plt.legend(loc='best',
           title=r'$y_0$ = '+str(y0_std)+
           '\n'+r'$v_{x_0}$= '+str(vx0_std)+
           '\n'+r'$C/m$ = '+str(Com_anot)+
           '\n'+r'$t_0$ = '+str(t0)+
           '\n'+r'$t_f$ = '+str(tf)+
           '\n'+r'$dt$ = '+str(dt_std)+
           '\n'+r'$\Delta t$ = '+str(Dt_anot))
plt.tight_layout()
plt.show()

#Figure 5: e vs. dt
e = [] #Initialize a list to store absolute overshoot error.
for i in range(len(dt)):
    shot = main(vx0_std, Com_std, dt[i], y0_std)
    hit_s = shot[5] #Grab the collision information for the shot object.
    e_hit_s = hit_s[5] #Get the shot object error value from the hit tuple.
    e.append(e_hit_s) #Append the overshoot error to the e list.
plt.figure(facecolor="white")
plt.plot(dt,e, color='blue', linestyle = '-')
plt.title(r'$\mathrm{Shot\ Object\ Absolute\ }$'+
          r'$y\mathrm{-Overshoot\ Error\ vs.\ Time\ Step}$')
plt.xlabel(r'$dt\ \mathrm{(seconds)}$')
plt.ylabel(r'$e\ \mathrm{(meters)}$')
#Annotate relevant information. Coordinates are in data units.
plt.text(0.00025, 0.4,r'$y_0$ = '+str(y0_std)+
         '\n'+r'$v_{x_0}$ = '+str(vx0_std)+
         '\n'+r'$C/m$ = '+str(Com_anot)+
         '\n'+r'$t_0$ = '+str(t0)+
         '\n'+r'$t_f$ = '+str(tf))
plt.tight_layout()
plt.show()