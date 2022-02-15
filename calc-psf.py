import numpy as np
import scipy
from scipy.integrate import quad
from scipy.integrate import dblquad
from numba import complex128, float64, jit

@jit(complex128(float64, float64,float64,float64,float64,float64), nopython=True, cache=True)
def vOut(phi, t, z2, r2, p2, v):
    NA = 1.4  # Numerical Apperture of objective
    n = 1.518  # Refractive index (water)
    h_max = 30e6  # Max. height of incoming beam (radius of beam) here 5 mm
    A1 = 0.8
    lambda_sted = 775  # STED wavelength (nm)
    theta_max = np.arcsin(NA / n)  # maximum openening angle of objective
    foc = h_max / np.sin(theta_max)  # effective focal length of objective
    k = (2 * np.pi) / lambda_sted
    a = np.sqrt(np.cos(t))
    pol = 1 / np.sqrt(2) * np.array([1, 1j, 0])  # right circular polarization
    # pol = [0, 1, 0]; # linear y polarization
    # pol = 1/sqrt(5)*[2, 1i, 0]; # Ellipse x
    Vx = (1 + (np.cos(t) - 1) * np.cos(phi) ** 2) * pol[0] + ((np.cos(t) - 1) * np.cos(phi) * np.sin(phi)) * pol[1] - (
                np.sin(t) * np.cos(phi)) * pol[2]
    Vy = ((np.cos(t) - 1) * np.cos(phi) * np.sin(phi)) * pol[0] + (1 + (np.cos(t) - 1) * np.sin(phi) ** 2) * pol[1] - (
                np.sin(t) * np.sin(phi)) * pol[2]
    Vz = (np.sin(t) * np.cos(phi)) * pol[0] + (np.sin(t) * np.sin(phi)) * pol[1] + (np.cos(t)) * pol[2]
    R = h_max / np.sqrt(2)
    f = 1

    # xShift = 0; # phase plate shift in x direction
    # yShift = 0; # phase plate shift in y direction
    h_act = foc * np.sin(t)

    # For no phase plate, use:
    # d_alpha = 0;

    # For a vortex phase plate(no shift), use:
    # d_alpha = phi;

    # # For an annular phase plate, use:
    #d_alpha = (h_act < R) * np.pi

    # for i=1:length(h_act)
    if abs(h_act) < R:
        d_alpha = np.pi
    else:
        d_alpha = 0

    b = k * n * (z2 * np.cos(t) + r2 * np.sin(t) * np.cos(phi - p2))
    if v == 1:
        f = np.sin(t) * A1 * a * Vx * np.exp(1j * d_alpha) * np.exp(1j * b)
    elif v == 2:
        f = np.sin(t) * A1 * a * Vy * np.exp(1j * d_alpha) * np.exp(1j * b)
    elif v == 3:
        f = np.sin(t) * A1 * a * Vz * np.exp(1j * d_alpha) * np.exp(1j * b)

    return f

def complex_dblquad(func, amin, amax, bmin, bmax, argsl, epsrell):
    def real_func(phi, t, z2, r2, p2, v):
        return np.real(func(phi, t, z2, r2, p2, v))

    def imag_func(phi, t, z2, r2, p2, v):
        return np.imag(func(phi, t, z2, r2, p2, v))

    real_integral = dblquad(real_func, amin, amax, bmin, bmax, args=argsl, epsrel=epsrell)
    imag_integral = dblquad(imag_func, amin, amax, bmin, bmax, args=argsl, epsrel=epsrell)
    return (real_integral[0] + 1j * imag_integral[0])


if __name__ == '__main__':
    plane = 0  # XY plane = 1, XZ plane = 0
    NA = 1.4  # / sqrt(2); # Numerical Apperture
    n = 1.518  # Refractive index(water)

    C = 1
    A1 = 1
    step = 10  # simulation step size
    z2_max = 1500  # size of z in simulated image(nm)
    r2_max = 1000  # size of radius  r in simulated image(nm)

    theta_max = np.arcsin(NA / n)  # maximum opening angle  of objective
    z2_array = np.arange(-z2_max, z2_max, step)
    p2_array = np.arange(0, 2 * np.pi, 0.2)
    if plane == 1:
        r2_array = np.arange(0, r2_max, step)
    else:
        r2_array = np.arange(-r2_max, r2_max, step)

    tol = 1e-3  # integration tolerance
    if plane == 1:
        E_x = np.zeros((len(r2_array), len(p2_array)), dtype=np.complex_)
        E_y = np.zeros((len(r2_array), len(p2_array)), dtype=np.complex_)
        E_z = np.zeros((len(r2_array), len(p2_array)), dtype=np.complex_)
        for q in range(len(p2_array)):
            p2 = p2_array[q]
            for m in range(len(r2_array)):
                r2 = r2_array[m]
                z2 = 0  # We want a XY cross-section at z=0
                ### Double integral calculation (calls on function vOut):
                E_x[m, q] = 1j * complex_dblquad(vOut, 0, theta_max, 0, 2 * np.pi, [z2, r2, p2, 1], tol)
                E_y[m, q] = 1j * complex_dblquad(vOut, 0, theta_max, 0, 2 * np.pi, [z2, r2, p2, 2], tol)
                E_z[m, q] = 1j * complex_dblquad(vOut, 0, theta_max, 0, 2 * np.pi, [z2, r2, p2, 3], tol)

        I = abs(E_x) ** 2 + abs(E_y) ** 2 + abs(E_z) ** 2  # Intensity profile
        np.savetxt("out_xy.txt", I, delimiter=',')

    if plane == 0:
        EZ_x = np.zeros((len(z2_array), len(r2_array)), dtype=np.complex_)
        EZ_y = np.zeros((len(z2_array), len(r2_array)), dtype=np.complex_)
        EZ_z = np.zeros((len(z2_array), len(r2_array)), dtype=np.complex_)
        p2 = 1.5 * np.pi
        for m in range(len(r2_array)):
            r2 = r2_array[m]
            for k in range(len(z2_array)):
                z2 = z2_array[k]
                ### Double integral calculation (calls on function vOut):
                EZ_x[k, m] = 1j * complex_dblquad(vOut, 0, theta_max, 0, 2 * np.pi, [z2, r2, p2, 1], tol)
                EZ_y[k, m] = 1j * complex_dblquad(vOut, 0, theta_max, 0, 2 * np.pi, [z2, r2, p2, 2], tol)
                EZ_z[k, m] = 1j * complex_dblquad(vOut, 0, theta_max, 0, 2 * np.pi, [z2, r2, p2, 3], tol)

        Iz = abs(EZ_x) ** 2 + abs(EZ_y) ** 2 + abs(EZ_z) ** 2  # Intensity profile
        np.savetxt("out_z.txt", Iz, delimiter=',')
