# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os
from copy import deepcopy

import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)

# MRG packages
import _env
import preprocessing
import processing
import postprocessing
import utils
# import solutions


def discretize_chi(domain_omega, chi, step_threshold=1e-3, nbre_iter_max=1000):
    S = numpy.sum(numpy.where(domain_omega == _env.NODE_ROBIN, 1, 0))
    V_obj = numpy.sum(chi)/S

    threshold = 0.5
    discretized_chi = deepcopy(chi)
    discretized_chi = numpy.where(chi > threshold, 1, 0)
    V_disc = numpy.sum(discretized_chi)/S
    k = 0
    while V_disc != V_obj and k < nbre_iter_max:
        if V_disc < V_obj:
            threshold -= step_threshold
        else:
            threshold += step_threshold
        discretized_chi = numpy.where(chi > threshold, 1, 0)
        V_disc = numpy.sum(discretized_chi)/S
        k += 1
    return discretized_chi


def optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
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
    min_mu = config["OPTIMIZATION"]["GRAD_DESCENT_CHI"]["MIN_MU"]
    tol_err_vol_chi = config["OPTIMIZATION"]["GRAD_DESCENT_CHI"]["TOLERANCE_ERROR_VOLUME_CHI"]
    step_lagrange_mult = config["OPTIMIZATION"]["GRAD_DESCENT_CHI"]["STEP_LAGRANGE_MULTIPLIER"]
    S = numpy.sum(numpy.where(domain_omega == _env.NODE_ROBIN, 1, 0))
    k = 0
    numb_iter = config["OPTIMIZATION"]["GRAD_DESCENT_CHI"]["NBRE_ITER"]

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
        energy_k = utils.compute_energy(u, spacestep)
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

            new_chi = utils.clip(
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
                new_chi = utils.clip(
                    new_chi_without_constraint+lagrange_mult, 0, 1)
                # assert numpy.min(chi) >= 0, numpy.min(chi)
                new_chi = preprocessing.set2zero(
                    new_chi, domain_omega)
                integre_chi = numpy.sum(new_chi)/S
            print()
            # print('    c. computing solution of Helmholtz problem, i.e., u')
            alpha_rob = Alpha*new_chi
            u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
            u = processing.enable_trace_robin_fn(u, domain_omega)
            print('    d. computing objective function, i.e., energy (E)')
            ene = utils.compute_energy(u, spacestep)
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
    chi = discretize_chi(
        domain_omega, chi, step_threshold=1e-3, nbre_iter_max=1000)
    return chi, energy, u, grad


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
    # initial gradient step
    mu = config["OPTIMIZATION"]["GRAD_DESCENT_CHI"]["INIT_MU"]
    # parameter of the volume functional
    mu1 = config["OPTIMIZATION"]["GRAD_DESCENT_CHI"]["MU1_VOLUME_CONSTRAINT"]

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
    chi, energy, u, grad = optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
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
    print('volume diff of chi:', (numpy.sum(chin)-numpy.sum(chi0))/S)
    print("energy:", energy)
    u0, un = numpy.real(u0), numpy.real(un)
    print("min u0, max u0, min un, max un:\n", numpy.min(
        u0), numpy.max(u0), numpy.min(un), numpy.max(un))
    print('End.')
