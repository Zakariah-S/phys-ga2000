'''
Exercise 10.4 (Pg. 460) in Newman
'''
import numpy as np
import matplotlib.pyplot as plt

save = False #Determines whether the plot is saved-- I don't want to accidentally the image I made!

#Simulation parameters
initial_tl: int     = 1000                  #Starting number of thorium atoms
initial_pb: int     = 0                     #Starting number of lead atoms
t_tl208: float      = 3.053 * 60            #Half-life of Thallium-208
dt: float           = 1.0                   #Time-step
steps: int          = 1000                  #Number of time-steps taken

#Set up arrays for the independent and dependent variables
times = np.linspace(0, steps * dt, steps + 1, dtype=np.float64)
N_Tl = np.full(times.shape, initial_tl, dtype=np.int32)
N_Pb = np.full(times.shape, initial_pb, dtype=np.int32)

#Generate random number x, distributed according to the exponential P(x) = (ln2/t_tl208) * exp(-(ln2/t_tl208)x)
#We simply take x as the set of times at which each thallium atom decays.
z = np.random.rand(initial_tl)
x = -(t_tl208/np.log(2)) * np.log(1 - z)
decay_times = np.sort(x)

#Set up a histogram with bins for each time-step to see how many decays occur within the spans of size dt.
#Then, use cumsum to find the TOTAL decays that have occurred by the end of each timestep
decay_hist = np.histogram(decay_times, times)[0]
decays_after_t = np.cumsum(decay_hist)

#Update the numbers of each atom for all times accordingly
N_Pb[1:] += decays_after_t
N_Tl[1:] -= decays_after_t

#Plot data
plt.plot(times, N_Tl, label="Simulated Thallium")
plt.plot(times, N_Pb, label="Simulated Lead")

#Plot expected result
plt.plot(times, initial_tl * np.power(2, -(times)/t_tl208), ls='dotted', label="Expected Thallium")
plt.plot(times, initial_tl - initial_tl * np.power(2, -(times)/t_tl208), ls='dotted', label="Expected Lead")

#Labels
plt.xlabel("Time (s)")
plt.ylabel("Number of atoms")
plt.title("Radioactive Decay of Thallium-208")
plt.legend()

#Save and show
if save: plt.savefig("thalliumdecay.eps", format="eps")
plt.show()