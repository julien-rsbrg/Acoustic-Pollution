# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os
import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)

# MRG packages
import _env
import preprocessing
import processing
import postprocessing
# import solutions


def clip(values, vmin, vmax):
    clipped_values = numpy.where(values >= vmin, values, vmin)
    clipped_values = numpy.where(clipped_values >= vmax, vmax, clipped_values)
    return clipped_values


def your_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                Alpha, mu, chi, V_obj, mu1, V_0):
    """This function return the optimized density.

    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi;
        mu1: float, it characterizes the importance of the volume constraint on
        the domain (not really important for our case, you can set it up to 0);
        V_0: float, volume constraint on the domain (you can set it up to 1).
    """
    min_mu = config["OPTIMIZATION"]["MIN_MU"]
    tol_err_vol_chi = config["OPTIMIZATION"]["TOLERANCE_ERROR_VOLUME_CHI"]
    step_lagrange_mult = config["OPTIMIZATION"]["STEP_LAGRANGE_MULTIPLIER"]
    S = numpy.sum(numpy.where(domain_omega == _env.NODE_ROBIN, 1, 0))
    k = 0
    numb_iter = config["OPTIMIZATION"]["NBRE_ITER"]

    # energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)
    while k < numb_iter and mu > min_mu:
        print('---- iteration number = ', k)
        print('1. computing solution of Helmholtz problem, i.e., u')
        alpha_rob = Alpha * chi
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        u = processing.enable_trace_robin_fn(u, domain_omega)

        print('2. computing solution of adjoint problem, i.e., q')
        f_adj = -2*numpy.conjugate(u)
        f_dir_adj = 0*f_dir
        q = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f_adj, f_dir_adj, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        q = processing.enable_trace_robin_fn(q, domain_omega)

        print('3. computing objective function, i.e., energy')
        energy_k = your_compute_objective_function(
            domain_omega, u, spacestep, mu1, V_0)
        if k == 0:
            energy = numpy.array([[energy_k]])
        else:
            energy_k = numpy.array([[energy_k]])
            energy = numpy.concatenate([energy, energy_k], axis=0)
        print("     ---energy:", energy, energy_k)

        print('4. computing parametric gradient')
        grad = numpy.real(Alpha*u*q)
        grad = processing.set2zero(grad, domain_omega)
        print("    grad:", numpy.max(numpy.abs(grad)),
              numpy.min(numpy.abs(grad)))

        ene = energy_k
        while ene >= energy[-1, 0] and mu > min_mu:
            lagrange_mult = 0

            print('    a. computing gradient descent')
            new_chi_without_constraint = chi-mu*grad
            new_chi_without_constraint = preprocessing.set2zero(
                new_chi_without_constraint, domain_omega)
            # print('    b. computing projected gradient')

            new_chi = clip(
                new_chi_without_constraint+lagrange_mult, 0, 1)

            integre_chi = numpy.sum(
                new_chi)/S
            while numpy.abs(integre_chi-V_obj) >= tol_err_vol_chi:

                print(f"V_obj {V_obj} integre_chi {integre_chi}",
                      'lag:', lagrange_mult, end='\r')

                if integre_chi >= V_obj:
                    lagrange_mult -= step_lagrange_mult
                else:
                    lagrange_mult += step_lagrange_mult
                new_chi = clip(
                    new_chi_without_constraint+lagrange_mult, 0, 1)
                # assert numpy.min(chi) >= 0, numpy.min(chi)
                new_chi = preprocessing.set2zero(
                    new_chi, domain_omega)
                integre_chi = numpy.sum(new_chi)/S
            print()
            # print('    c. computing solution of Helmholtz problem, i.e., u')
            alpha_rob = Alpha*new_chi
            u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f_adj, f_dir, f_neu, f_rob,
                                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
            u = processing.enable_trace_robin_fn(u, domain_omega)
            print('    d. computing objective function, i.e., energy (E)')
            ene = your_compute_objective_function(
                domain_omega, u, spacestep, mu1, V_0)
            print('           ---ene', ene)
            print("           ---vs energy_k:", energy[k, 0])
            if ene < energy[-1, 0]:
                # The step is increased if the energy decreased
                mu = mu * 1.1
            else:
                # The step is decreased if the energy increased
                mu = mu / 2
            print("           ---mu ", mu)

        chi = new_chi
        k += 1

    print('end. computing solution of Helmholtz problem, i.e., u')

    return chi, energy, u, grad


def get_norm_L2(u):
    # print("           numpy.max(u@numpy.conjugate(u.T))",
    #       numpy.max(numpy.abs(u@numpy.conjugate(u.T))))
    return numpy.abs(numpy.trace(u@numpy.conjugate(u.T)))


def your_compute_objective_function(domain_omega, u, spacestep, mu1, V_0):
    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2 + mu1*(Vol(domain_omega)-V_0)

    Parameter:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the
        Robin frontier;
        u: Matrix (NxP), it is the solution of the Helmholtz problem, we are
        computing its energy;
        spacestep: float, it corresponds to the step used to solve the Helmholtz
        equation;
        mu1: float, it is the constant that defines the importance of the volume
        constraint;
        V_0: float, it is a reference volume.
    """

    raw_energy = get_norm_L2(u)*(spacestep**2)

    # N*N = V_0 intially
    N = domain_omega.shape[1]
    in_interior = numpy.where(domain_omega == _env.NODE_INTERIOR, 1, 0)
    vol_domain_omega = numpy.sum(in_interior)/(N**2)
    vol_constraint = vol_domain_omega*(spacestep**2)-V_0

    return raw_energy + mu1*vol_constraint


if __name__ == '__main__':

    # -- set parameters of the geometry
    N = config["GEOMETRY"]["N_POINTS_AXIS_X"]  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = config["GEOMETRY"]["LEVEL"]  # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    # kx = config["PDE"]["KX"]
    # ky = config["PDE"]["KY"]
    # wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber = config["PDE"]["WAVENUMBER"]

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(
        M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(
        M, N, level)

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- define boundary conditions
    # planar wave defined on top
    if config["PDE"]["INCIDENT_WAVE"] == "spherical":
        # spherical wave defined on top
        f_dir[:, :] = 0.0
        f_dir[0, int(N/2)] = 10.0
    else:
        f_dir[:, :] = 0.0
        f_dir[0, 0:N] = 1.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    # -- define absorbing material
    # -- this is the function you have written during your project
    from compute_alpha_folder.compute_alpha import compute_alpha
    material = config["GEOMETRY"]["MATERIAL"]
    Alpha = compute_alpha(wavenumber, material)[0]
    alpha_rob = Alpha * chi

    # -- set parameters for optimization
    S = numpy.sum(numpy.where(domain_omega == _env.NODE_ROBIN, 1, 0))
    V_0 = 1  # initial volume of the domain
    V_obj = numpy.sum(chi) / S  # constraint on the density
    print("V_obj:", V_obj)
    mu = config["OPTIMIZATION"]["INIT_MU"]  # initial gradient step
    # parameter of the volume functional
    mu1 = config["OPTIMIZATION"]["MU1_VOLUME_CONSTRAINT"]

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    u = processing.enable_trace_robin_fn(u, domain_omega)
    chi0 = chi.copy()
    u0 = u.copy()

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    energy = numpy.zeros((100+1, 1), dtype=numpy.float64)
    chi, energy, u, grad = your_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                                       Alpha, mu, chi, V_obj, mu1, V_0)

    # --- end of optimization
    chin = chi.copy()
    un = u.copy()

    # -- plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un, chin)
    err = un - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)
    postprocessing._plot_comparison(u0, un)

    # print('min energy:', numpy.min(energy))
    print("max|chi0-chin|:", numpy.max(numpy.abs(chi0-chin)))
    print("energy:", energy)
    print('End.')
