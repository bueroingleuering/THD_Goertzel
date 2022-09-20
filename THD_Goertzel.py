"""@author: robin leuering"""

import numpy as np
import matplotlib.pyplot as plt

"""Goertzel algorithm"""

def goertzel(x,k):
    N   = np.size(x)
    A   = 2*np.pi*k/N
    B   = 2*np.pi*k*(N-1)/N
    f   = 2*np.cos(A)
    y1  = 0
    y2  = 0
    for n in range(N):
        y0 = x[n]+y1*f-y2
        y2 = y1
        y1 = y0
    return 2*np.hypot(y1*np.cos(B)-y2*np.cos(A*N),y2*np.sin(A*N)-y1*np.sin(B))/N

"""Setup"""

f   = 50                                    # Frequency [Hz]
w   = 2*f*np.pi                             # Omega [1/s]
phase = 0*np.pi/180;                        # Phase delay
n   = 4096                                  # Time Steps
N   = 16                                    # Samples to measure
k   = N/2                                   # Wave number
t   = np.arange(0,1/f,1/(f*n))              # Time vector
y   = 0                                     # DC offset
y1  = y + 325*np.sin(w*t+phase)             # Sinus with base frequency
y   = y1 + 7*np.sin(3*(w*t+phase))          # Sinus with base frequency + disturbance
y   = y + 5*np.sin(5*(w*t+phase))
y   = y + 20*np.sin(7*(w*t+phase))

"""Discrete"""

td  = np.arange(0,1/f,1/(f*N))              # Discrete time vector
yd  = np.zeros(N)                           # Discrete signal vector prepare
for i in range(N):
    yd[i] = y[t.tolist().index(td[i])]      # Discrete signal vector

"""THD"""

P_n = 2*np.mean(np.square(yd))              # Power time domain
U_1 = goertzel(yd,1)                        # Amplitude k=1 frequency domain
THD_E = P_n/U_1**2 - 1                      # THD energy relatetd
THD_U = np.sqrt(THD_E)                      # THD voltage related

"""Plot"""

plt.figure()
plt.plot(t,y1,c='#000000',label='fundamental',linewidth=1.0)
plt.plot(t,y,c='#284b64',label='continuous',linewidth=3.0)
plt.plot(td,yd,'ro',label='discrete',linewidth=3.0)
plt.title('Signal time domain',fontsize=20)
plt.legend(loc='best',fontsize=16)
plt.grid(True)
plt.xlabel('t [s]',fontsize=16)
plt.show()