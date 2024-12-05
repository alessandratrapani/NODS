import numpy as np
import math as m
from numpy import arange as arange
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

text_color = 'black'
rcParams['text.color'] = text_color
rcParams['axes.labelcolor'] = text_color
rcParams['xtick.color'] = text_color
rcParams['ytick.color'] = text_color

font_size = 28
plt.rc('font', size=font_size)          # controls default text sizes
plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_size)    # legend fontsize
plt.rc('figure', titlesize=font_size)  # fontsize of the figure title

NO_color = (0.521568627, 0.709803922, 0.415686275)
Calm2C_color=(0.6, 0.458823529, 0.662745098)
spike_color=(1.0,1.0,0.533333333)
Garthwaite_color=(0.850980392, 0.850980392, 0.850980392)

dt     = 1 # size of a time step [ms]
t_sim  = 1000 # simulation time [ms] 

ds     = 0.1# space step size 100[nm]
r_max  = 5 # micrometer

distances = arange(-r_max, r_max+0.1, ds)
time = arange(0,t_sim,dt)

D      = 0.848 # diffusion coefficient [um^2/ms]
I      = 0.15 # inactivation coefficient [1/ms] 
r_2 = distances**2

def production_function_real(Calm2C_old,nNOS_old,dt,Ca_spike):
    
    #parameters chosen after comparison with results for NO synthesis simulated in NEURON 
    tau = -150
    tauNOS1 = 200
    tauNOS2 = 25
    
    Calm2C = Calm2C_old + (((Calm2C_old/tau) + Ca_spike)*dt)
    nNOS = nNOS_old+((((1/tauNOS1)*((Calm2C_old)/((Calm2C_old)+1)))-(nNOS_old/tauNOS2))*dt)
      
    return nNOS, Calm2C
def Green_function(t,r_2,D,I):
    
    if t == 0:
        t = 0.1
        
    a = 1/(4*m.pi*D*t)
    e1 = (-1*r_2)/(4*D*t)
    exp_diffusion = np.exp(e1)
    exp_inactivation = np.exp(-I*t)
    G = (m.pow(a,3/2))*exp_diffusion*exp_inactivation
    
    return G

Green_LUT = np.zeros((len(distances),2))
Green_LUT[:,0] = Green_function(0,r_2,D,I)
Green_LUT[:,1] = Green_function(dt,r_2,D,I)

# costants to be tuned
A=1.35e-9
B=1e12
C=20
single_spike = np.loadtxt('reproduce_garthwaite_figure/single_spike.txt')

Calm2C_old = 0
nNOS_old = 0
NO_ti = 0
NO_tf = NO_ti
NO_conc = np.zeros((len(distances),len(time)))
u0 = np.zeros((len(distances)))

for i in range(0,len(time)-1,1):    
    NO_ti = NO_tf    
    nNOS_old, Calm2C_old = production_function_real(Calm2C_old,nNOS_old,dt,single_spike[i])
    NO_tf = nNOS_old*A    
    spacial_conv = np.convolve(Green_LUT[:,1], u0, 'same')    
    u = spacial_conv +(((Green_LUT[:,0]*NO_tf) + (Green_LUT[:,1]*NO_ti))*((dt)/2))    
    NO_conc[:,i+1] = u*B    
    u0 = u


garth = np.loadtxt('reproduce_garthwaite_figure/garthwaite.txt')
#-----------------------------------------------------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 9))

plt.plot(time, NO_conc[53,:], color='g',linewidth=4, label='NO sim')
plt.plot(time, garth, 'o',color='orange',markersize=8, label='Garthwaite data')
plt.plot(single_spike*C,color='black',linewidth=4, label='spike')
plt.xlabel('time [ms]')
plt.ylabel('NO concentration [pM]')
plt.title('NO time profile', fontweight ='bold')


plt.legend(loc='upper right')
plt.savefig('garth.png', transparent = False)

