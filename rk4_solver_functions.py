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

# density and mass solver

def get_density(rho0, r, rel):
    # Author: Callum Newman , Date: 05/05/2026
    # Obtain the density, rho, as function of the radial distance, r, using the
    # implemented ODE solver and the non-relativistic or relativistic equation.
    # Input:
    # * rho: the central density at r = 0.
    # * r: the grid points of radial distance where the density is calculated in form
    #      of a 1D numpy array with N elements.
    # * rel: boolean distinguising relativistic and non-relativistic cases.
    #
    # Output:
    # * rho: an N-element 1D numpy array that contains the density at the radial grid points
    #        given in r.
    # * mass: the cumulative mass of the white dwarf from r = 0 to the radial grid
    #         point given in r (a 1D numpy array with N elements).
    #
    # Example use:
    # >> rho0 = 1e13
    # >> r = np.arange(5e4, 5.005e7, 5e4) # NB this range should not start at or too close
    # >>                                   to zero so the value of rho[1] doesn't inflate.
    # >> [rho, mass] = get_density(rho0, r, 0) # non-relativistic
    # >> plt.plot(r, rho) # plot density vs radial distance
    # >> plt.show()

    m_p = 1.672e-27
    m_e = 9.11e-31
    c = 3e8
    h = 6.626e-34

    N = r.size
    dr = r[2] - r[1]

    rho = np.zeros(N)
    mass = np.zeros(N)
    a = -1.9140e-7 # c = -d(rho)/dp, the negative reciprocal of the rate of change of pressure
                  # with respect to density (from equation 8 in the labscript)
    G = 6.6743e-11 # Gravitational constant
    d = -6 * m_p / (m_e * c**2)
    coef = (3*h**3)/(16*pi * m_p * m_e**3 * c**3)

    # Create lambda functions that gives the derivative functions of density and mass for the 
    # relativistic and non-relativistic cases
    # Since the function for the derivative of density raises density to a non-integer power, 
    # the function makes sure to only pass values of density greater than 0
    # y = [rho, mass]
    if rel:
        f = lambda y, r: np.array([(d * (1 + (coef * y[0])**(2/3))**0.5 / (coef*y[0])**(2/3) * G * y[0] * y[1] / r**2),
                                   (4*pi*r**2 * y[0])]) if np.all(np.array(y) > 0) \
                         else np.array([0,(4*pi*r**2 * y[0])])
    else:
        f = lambda y, r: np.array([(a * y[0]**(-2/3) * G * y[0] * y[1] / r**2),
                                   (4*pi*r**2 * y[0])]) if np.all(np.array(y) > 0) \
                         else np.array([0, (4*pi*r**2 * y[0])])
    m0 = 4*pi/3 * dr**3 * rho0
    y0 = (rho0, m0)
    y = ode_solve_rk(f, y0, r)
    rho = y[0,:]
    mass = y[1,:]

    return [rho, mass]