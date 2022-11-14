####### standard #######
import numpy
import matplotlib.pyplot as plt
from copy import deepcopy
import os

####### my packages #######
import _env
import frontier_generation as fro_gen
import preprocessing
import utils
import chi_gradient_descent

###########################
import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)


def get_indices_chi(chi):
    i_points, j_points = numpy.where(
        chi == 1)
    i_points = numpy.expand_dims(i_points, axis=1)
    j_points = numpy.expand_dims(j_points, axis=1)
    return numpy.concatenate([i_points, j_points], axis=-1)


def plot_energy_chi_beta(energies, betas):
    # , cmap = 'jet')#, vmin = 1e-4, vmax = 1e-0)
    plt.plot(betas[:, 0], energies[:, 0], 'bo-')
    plt.title(
        'Energy after optimization with gradient descent for different chi integral')
    plt.xlabel('percentage of covered frontier')
    plt.ylabel('Energy')
    # matplotlib.pyplot.colorbar()
    # matplotlib.pyplot.show()
    filename = 'fig_energy_chi_beta_study.jpg'
    dst_file_path = os.path.join(dst_folder, filename)
    plt.savefig(dst_file_path)
    plt.close()


def compare_chi_beta(level, beta_min, beta_max, N_betas=100):
    # -- set parameters of the geometry
    N = config["GEOMETRY"]["N_POINTS_AXIS_X"]  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
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

    # -- define absorbing material
    # -- this is the function you have written during your project
    from compute_alpha_folder.compute_alpha import compute_alpha
    material = config["GEOMETRY"]["MATERIAL"]
    Alpha = compute_alpha(wavenumber, material)[0]

    # -- set parameters for optimization
    S = numpy.sum(numpy.where(domain_omega == _env.NODE_ROBIN, 1, 0))
    V_0 = 1  # initial volume of the domain
    chi = preprocessing._set_chi(M, N, x, y)
    raw_V_obj = int(numpy.sum(chi))  # constraint on the density
    V_obj = numpy.sum(chi)/S
    print("raw_V_obj:", raw_V_obj)
    # initial gradient step
    mu = config["OPTIMIZATION"]["GRAD_DESCENT_CHI"]["INIT_MU"]
    # parameter of the volume functional
    mu1 = config["OPTIMIZATION"]["GRAD_DESCENT_CHI"]["MU1_VOLUME_CONSTRAINT"]

    energies = numpy.zeros((N_betas, 1))
    betas = numpy.zeros((N_betas, 1))
    number_epochs = numpy.zeros((N_betas, 1))
    S = numpy.sum(numpy.where(domain_omega == _env.NODE_ROBIN, 1, 0))
    for beta_i in range(N_betas):
        print("--beta_i:", beta_i)
        # -- define material density matrix
        beta = (beta_max-beta_min)*beta_i/max(N_betas-1, 1)+beta_min
        betas[beta_i, 0] = beta
        raw_V_obj = beta*S

        chi = utils.set_random_chi(domain_omega, raw_V_obj)
        alpha_rob = Alpha * chi

        chi, energy, u, grad = chi_gradient_descent.optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                                                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                                                           Alpha, mu, chi, V_obj, mu1, V_0)

        energies[beta_i, 0] = energy[-1, 0]
        number_epochs[beta_i, 0] = energy.shape[0]
        betas[beta_i, 0] = beta

    plot_energy_chi_beta(energies, betas)


if __name__ == '__main__':
    dst_folder = f'./results/chi_gradient_descent_beta_study_level1/'
    if not (os.path.exists(dst_folder)):
        os.mkdir(dst_folder)
    compare_chi_beta(1, 0.1, 1, N_betas=30)
    print('End.')
