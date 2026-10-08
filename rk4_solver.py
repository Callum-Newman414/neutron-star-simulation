import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from numpy import log10

m_p = 1.672e-27
m_e = 9.11e-31
c = 3e8
h = 6.626e-34

def ode_solve_rk(f, y0, t):
    # Author: Callum Newman , Date: 04/05/2026
    # Solve ODE problem, i.e. dy/dt = f(y,t), using Runge-Kutta algorithm.
    # Input:
    # * f: a function that receives the current state, y, and the current position/time,
    #      t, and returns the derivative value of the state, dy/dt.
    # * y0: the initial state of the system, given in as a tuple of M elements
    # * t: 1D numpy array of position/time steps with length N where the values of y will be
    #      returned.
    #
    # Output:
    # * y: (M x N) numpy array that contains the values of y at every position/time step.
    #      Columns correspond to the position/time and rows to the element of y.
    #
    # Example use:
    # >> # Example of solving the harmonic oscillation equation, d2x/dt2 = -x
    # >> # The ODE problem is posed with 2 element vector: y = (dx/dt, x)
    # >> # Therefore, dy/dt = (d2x/dt2, dx/dt) = (-y[1], y[0])
    # >> f = lambda y, t : (-y[1], y[0])
    # >> y0 = (0, 1)
    # >> t = np.arange(0,10.01,0.01)
    # >> y = ode_solve_rk(f, y0, t)
    # >> plt.plot(t, y[1,:]) # plot x vs t
    # >> plt.show()

    M = len(y0)
    N = t.size
    dt = t[2] - t[1]
    y = np.zeros((M,N))
    y_initial = y0

    # Runge-Kutta algorithm (RK4):

    # Create an (M x 4) matrix F to hold the values of the function f for the 4
    # intermediate stages of the calculation for each derivative of the M elements of y.

    F = np.zeros((M,4))    
    y1 = np.zeros(M) # array to hold the values of y(t+dt)

    y[:,0] = y0     # set initial values in array

    for i in range(0,N-1):
        F[:,0] = f(y0,t[i])
        F[:,1] = f((y0 + dt/2 * F[:,0]),t[i] + dt/2)
        F[:,2] = f((y0 + dt/2 * F[:,1]),t[i] + dt/2)
        F[:,3] = f((y0 + dt   * F[:,2]),t[i] + dt)

        y1 = y0 + dt/6 * (F[:,0] + 2*F[:,1] + 2*F[:,2] + F[:,3])

        # For all of the uses of this function in the project, we can ignore sufficiently
        # small values of y (i.e. density and mass)
        for j in range(M):
            if y1[j] < y_initial[j] * 1e-4:
                return y

        # Store values in output matrix and set y0 for next iteration
        y[:,i+1] = y1
        y0 = y1.copy()
    return y