# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os


# MRG packages
import _env
import preprocessing
import processing
import postprocessing

# import solutions


def your_optimization_procedure(
    domain_omega,
    spacestep,
    omega,
    f,
    f_dir,
    f_neu,
    f_rob,
    beta_pde,
    alpha_pde,
    alpha_dir,
    beta_neu,
    beta_rob,
    alpha_rob,
    Alpha,
    mu,
    chi,
    V_obj,
    S,
    mu1=0,
    V_0=1,
    numb_iter=100,
):
    """This function return the optimized density.

    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi;
        mu1: float, it characterizes the importance of the volume constraint on
        the domain (not really important for our case, you can set it up to 0);
        V_0: float, volume constraint on the domain (you can set it up to 1).
        S : surface of the fractal
    """
    eps0 = 10 ** (-5)
    eps1 = 10 ** (-3)
    eps2 = 10 ** (-5)
    eps3 = 1.1

    k = 0
    (M, N) = numpy.shape(domain_omega)
    energy = numpy.zeros((numb_iter + 1, 1), dtype=numpy.float64)
    while k < numb_iter and mu > eps0:
        print("---- iteration number = ", k)
        alpha_rob = chi * Alpha
        print("1. computing solution of Helmholtz problem, i.e., u")
        u = processing.solve_helmholtz(
            domain_omega,
            spacestep,
            omega,
            f,
            f_dir,
            f_neu,
            f_rob,
            beta_pde,
            alpha_pde,
            alpha_dir,
            beta_neu,
            beta_rob,
            alpha_rob,
        )
        print("2. computing solution of adjoint problem, i.e., q")
        f_adjoint = -2 * numpy.conjugate(u)
        q = processing.solve_helmholtz(
            domain_omega,
            spacestep,
            omega,
            f_adjoint,
            f_dir,
            f_neu,
            f_rob,
            beta_pde,
            alpha_pde,
            alpha_dir,
            beta_neu,
            beta_rob,
            alpha_rob,
        )
        print("4. computing parametric gradient")
        grad = numpy.real(Alpha * u * q)
        print("3. computing objective function, i.e., energy")
        energy[k] = your_compute_objective_function(
            domain_omega, u, spacestep, mu1, V_0
        )
        ene = energy[k]
        chi = chi - mu * grad
        chi = preprocessing.set2zero(chi, domain_omega)
        while ene >= energy[k] and mu > eps0:
            l = 0
            print("    a. computing gradient descent")
            print("    b. computing projected gradient")
            chikp1 = numpy.where(chi + l < 0, 0, chi + l)
            chikp1 = numpy.where(chikp1 > 1, 1, chikp1)
            print("    c. computing solution of Helmholtz problem, i.e., u")
            print("    d. computing objective function, i.e., energy (E)")
            chi_contour = abs(numpy.sum(chikp1) / S - V_obj)
            while chi_contour > eps1:
                if numpy.sum(chikp1) / S >= V_obj:
                    l -= eps2
                else:
                    l += eps2
                new_chikp1 = numpy.where(chi + l < 0, 0, chi + l)
                new_chikp1 = numpy.where(new_chikp1 > 1, 1, new_chikp1)
                chikp1 = new_chikp1
                chi_contour = abs(numpy.sum(chikp1) / S - V_obj)
            chi = chikp1
            alpha_rob = chi * Alpha
            u = processing.solve_helmholtz(
                domain_omega,
                spacestep,
                omega,
                f,
                f_dir,
                f_neu,
                f_rob,
                beta_pde,
                alpha_pde,
                alpha_dir,
                beta_neu,
                beta_rob,
                alpha_rob,
            )
            ene = your_compute_objective_function(domain_omega, u, spacestep, mu1, V_0)
            print(f"ene = {ene}, enek = {energy[k]}")
            if ene < energy[k]:
                print("a")
                # The step is increased if the energy decreased
                mu = mu * eps3
            else:
                print("b")
                # The step is decreased if the energy increased
                mu = mu / 2
        k += 1

    print("end. computing solution of Helmholtz problem, i.e., u")

    return chi, energy, u, grad


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

    energy = numpy.sum((spacestep * numpy.absolute(u)) ** 2)
    return energy


if __name__ == "__main__":

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 50  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 2  # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber = 10.0

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    (
        beta_pde,
        alpha_pde,
        alpha_dir,
        beta_neu,
        alpha_rob,
        beta_rob,
    ) = preprocessing._set_coefficients_of_pde(M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(M, N, level)

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0
    # spherical wave defined on top
    # f_dir[:, :] = 0.0
    # f_dir[0, int(N/2)] = 10.0

    # -- initialize
    alpha_rob[:, :] = -wavenumber * 1j

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)

    # -- define absorbing material
    Alpha = 10.0 - 10.0 * 1j
    # -- this is the function you have written during your project
    # import compute_alpha
    # Alpha = compute_alpha.compute_alpha(...)
    alpha_rob = Alpha * chi

    # -- set parameters for optimization
    S = 0  # surface of the fractal
    for i in range(0, M):
        for j in range(0, N):
            if domain_omega[i, j] == _env.NODE_ROBIN:
                S += 1
    V_0 = 1  # initial volume of the domain
    V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
    mu = 0.001  # initial gradient step
    mu1 = 10 ** (-5)  # parameter of the volume functional

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u = processing.solve_helmholtz(
        domain_omega,
        spacestep,
        wavenumber,
        f,
        f_dir,
        f_neu,
        f_rob,
        beta_pde,
        alpha_pde,
        alpha_dir,
        beta_neu,
        beta_rob,
        alpha_rob,
    )
    chi0 = chi.copy()
    u0 = u.copy()

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    energy = numpy.zeros((100 + 1, 1), dtype=numpy.float64)
    # chi, energy, u, grad = your_optimization_procedure(...)
    chi, energy, u, grad = your_optimization_procedure(
        domain_omega,
        spacestep,
        wavenumber,
        f,
        f_dir,
        f_neu,
        f_rob,
        beta_pde,
        alpha_pde,
        alpha_dir,
        beta_neu,
        beta_rob,
        alpha_rob,
        Alpha,
        mu,
        chi,
        V_obj,
        S,
        mu1,
        V_0,
    )
    # --- end of optimization

    chin = chi.copy()
    un = u.copy()

    # -- plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un, chin)
    err = un - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)

    print("End.")
