# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot as plt
import numpy
import os
import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)

# MRG packages
import processing
import _env
import preprocessing
from utils import clip
import utils
import processing
import postprocessing


nb_points_etude = 1


def etude_freq_show(domain_omega, spacestep, chi, f, f_dir, f_neu,
                    f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha, mu1, V_0, wavenumber_min, wavenumber_max):
    l_J = []
    # l_wavenumber = numpy.exp(numpy.linspace(
    #    numpy.log(wavenumber_min), numpy.log(wavenumber_max), nb_points_etude))
    l_wavenumber = numpy.linspace(
        wavenumber_min, wavenumber_max, nb_points_etude)
    alpha_rob_final = Alpha*chi
    for wavenumber_var in l_wavenumber:

        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber_var, f, f_dir, f_neu,
                                       f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob_final)
        J = utils.compute_energy(
            u, spacestep)
        l_J.append(J)
    plt.plot(l_wavenumber, l_J, 'bo-')
    plt.title('Energie en fonction de la fréquence', fontsize=20)
    #plt.text(1, -0.6, r'$\sum_{i=0}^\infty x_i$', fontsize=20)
    # plt.text(0.6, 0.6, r'$\mathcal{A}\mathrm{sin}(2 \omega t)$',
    #        fontsize=20)
    plt.xlabel('Wavenumber')
    plt.ylabel(r'$J(\chi)$')
    plt.grid()
    plt.show()


def etude_freq(domain_omega, spacestep, chi, f, f_dir, f_neu,
               f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha, mu1, V_0, l_wavenumber):
    l_J = []
    alpha_rob_final = Alpha*chi
    for wavenumber_var in l_wavenumber:

        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber_var, f, f_dir, f_neu,
                                       f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob_final)
        J = utils.compute_energy(
            u, spacestep)
        l_J.append(J)
    return l_J


def objective_function_freq(domain_omega, spacestep, chi, f, f_dir, f_neu,
                            f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha, mu1, V_0, l_wavenumber):
    l_J = etude_freq(domain_omega, spacestep, chi, f, f_dir, f_neu,
                     f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha, mu1, V_0, l_wavenumber)
    res = sum(l_J)/len(l_J)
    return res


def freq_optimization_procedure(domain_omega, spacestep, l_wavenumber, f, f_dir, f_neu, f_rob,
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

        alpha_rob = Alpha * chi
        l_u = []
        l_q = []
        for wavenumber in l_wavenumber:
            print('1. computing solution of Helmholtz problem, i.e., u')
            u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
            u = processing.enable_trace_robin_fn(u, domain_omega)
            l_u.append(u)
            print('2. computing solution of adjoint problem, i.e., q')
            f_adj = -2*numpy.conjugate(u)
            f_dir_adj = 0*f_dir
            q = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f_adj, f_dir_adj, f_neu, f_rob,
                                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
            q = processing.enable_trace_robin_fn(q, domain_omega)
            l_q.append(q)
        l_u = numpy.array(l_u)
        l_q = numpy.array(l_q)
        print('3. computing objective function, i.e., energy')
        energy_k = objective_function_freq(domain_omega, spacestep, chi, f, f_dir, f_neu,
                                           f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha, mu1, V_0, l_wavenumber)
        if k == 0:
            energy = numpy.array([[energy_k]])
        else:
            energy_k = numpy.array([[energy_k]])
            energy = numpy.concatenate([energy, energy_k], axis=0)
        print("     ---energy:", energy, energy_k)

        print('4. computing parametric gradient')
        grad = sum([numpy.real(Alpha*l_u[i]*l_q[i])
                   for i in range(len(l_u))])/len(l_u)
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
            ene = objective_function_freq(domain_omega, spacestep, chi, f, f_dir, f_neu,
                                          f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, Alpha, mu1, V_0, l_wavenumber)
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
    u_fin = l_u[-1]
    return chi, energy, u_fin, grad


if __name__ == '__main__':
    # -- set parameters of the geometry
    N = config["GEOMETRY"]["N_POINTS_AXIS_X"]  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = config["GEOMETRY"]["LEVEL"]  # level of the fractal
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    # kx = config["PDE"]["KX"]
    # ky = config["PDE"]["KY"]
    # wavenumber = numpy.sqrt(kx*2 + ky*2)  # wavenumber
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

    # -- plot J(chin) vs wavenumber
    wavenumber_min, wavenumber_max = 10, 11
    l_wavenumber = numpy.linspace(
        wavenumber_min, wavenumber_max, nb_points_etude)

    u = processing.solve_helmholtz(domain_omega, spacestep, l_wavenumber[-1], f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    u = processing.enable_trace_robin_fn(u, domain_omega)
    u0 = u.copy()
    chi0 = chi.copy()

    chi, energy, u, grad = freq_optimization_procedure(domain_omega, spacestep, l_wavenumber, f, f_dir, f_neu, f_rob,
                                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                                       Alpha, mu, chi, V_obj, mu1, V_0)
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
