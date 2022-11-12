# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot as plt
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
            chi = compute_projected(chi, domain_omega, V_obj)
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


def compute_projected(chi, domain, V_obj):
    """This function performs the projection of $\chi^n - mu*grad

    To perform the optimization, we use a projected gradient algorithm. This
    function caracterizes the projection of chi onto the admissible space
    (the space of $L^{infty}$ function which volume is equal to $V_{obj}$ and whose
    values are located between 0 and 1).

    :param chi: density matrix
    :param domain: domain of definition of the equations
    :param V_obj: characterizes the volume constraint
    :type chi: numpy.array((M,N), dtype=float64)
    :type domain: numpy.array((M,N), dtype=complex128)
    :type float: float
    :return:
    :rtype:
    """

    (M, N) = numpy.shape(domain)
    S = 0
    for i in range(M):
        for j in range(N):
            if domain[i, j] == _env.NODE_ROBIN:
                S = S + 1

    B = chi.copy()
    l = 0
    chi = preprocessing.set2zero(chi, domain)

    V = numpy.sum(numpy.sum(chi)) / S
    debut = -numpy.max(chi)
    fin = numpy.max(chi)
    ecart = fin - debut
    # We use dichotomy to find a constant such that chi^{n+1}=max(0,min(chi^{n}+l,1)) is an element of the admissible space
    while ecart > 10**-4:
        # calcul du milieu
        l = (debut + fin) / 2
        for i in range(M):
            for j in range(N):
                chi[i, j] = numpy.maximum(0, numpy.minimum(B[i, j] + l, 1))
        chi = preprocessing.set2zero(chi, domain)
        V = sum(sum(chi)) / S
        if V > V_obj:
            fin = l
        else:
            debut = l
        ecart = fin - debut
        # print('le volume est', V, 'le volume objectif est', V_obj)

    return chi


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
    # wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    # wavenumber = 10.0
    wavenumbers = numpy.arange(start=7, stop=12, step=1 / 8)

    energies_before = []
    energies_after = []
    for wavenumber in wavenumbers:
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
        mu = 0.01  # initial gradient step
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

        energies_before.append(
            your_compute_objective_function(domain_omega, u, spacestep, mu1, V_0)
        )
        # chi = preprocessing._set_chi(M, N, x, y)
        # chi = preprocessing.set2zero(chi, domain_omega)
        # alpha_rob = Alpha * chi
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

        energies_after.append(
            your_compute_objective_function(domain_omega, un, spacestep, mu1, V_0)
        )

        # -- plot chi, u, and energy
        postprocessing._plot_uncontroled_solution(u0, chi0)
        postprocessing._plot_controled_solution(un, chin)
        err = un - u0
        postprocessing._plot_error(err)
        postprocessing._plot_energy_history(energy)

        print("End.")
    print("hey")
    plt.close()
    plt.plot(wavenumbers, energies_before, label="avant")
    plt.plot(wavenumbers, energies_after, label="après")
    plt.xlabel("Wavenumber")
    plt.ylabel("Energie")
    plt.legend()
    filename = "full_hist.jpg"
    dst_file_path = os.path.join("./results/", filename)
    plt.savefig(dst_file_path)
    plt.close()
    print("EndEndEnd.")
