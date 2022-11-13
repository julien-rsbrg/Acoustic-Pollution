import numpy as np
import yaml
from yaml.loader import SafeLoader

with open("ACO.yaml") as f:
    config = yaml.load(f, Loader=SafeLoader)

import processing
import preprocessing
import postprocessing
import _env


def aco(
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
        m : number of ants
    """
    ALPHA = config["ALPHA"]
    BETA = config["BETA"]
    GAMMA = config["GAMMA"]
    Q = config["Q"]
    RHO = config["RHO"]

    numb_iter = config["NB_ITER"]
    m = config["NB_ANTS"]

    (M, N) = np.shape(domain_omega)
    energy = np.zeros((numb_iter, 1), dtype=np.float64)
    boundary_coords = np.array(
        [
            (i, j)
            for i in range(M)
            for j in range(N)
            if domain_omega[i, j] == _env.NODE_ROBIN
        ]
    )
    S = len(boundary_coords)

    T = np.zeros(S)  # pheromones

    choice = None

    for k in range(numb_iter):
        print("---- iteration number = ", k)

        energies = []

        for ant in range(m):
            print("---- ant number = ", ant)
            initial_points = np.random.choice(5, 1)[0]
            chosen_indexes = list(np.random.choice(S, initial_points))
            chosen_coords = [boundary_coords[i] for i in chosen_indexes]
            for _ in range(
                int(1 / spacestep) - initial_points
            ):  # ant chi boundary building

                p = np.zeros(S)  # probability to choose a node for the boundary chi
                possible_nodes = find_possible_nodes(boundary_coords, chosen_coords)
                if not possible_nodes:
                    p = np.ones(S)
                for i in possible_nodes:
                    t = T[i]
                    p[i] = GAMMA + (t**ALPHA)
                p = normalize(p)
                index_chosen = np.random.choice(S, 1, p=p)[0]
                chosen_indexes.append(index_chosen)
                chosen_coords.append(boundary_coords[index_chosen])

            # Now the boundary is created, we compute its energy
            chi_ant = build_chi(M, N, chosen_coords)
            alpha_rob_ant = Alpha * chi_ant

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
                alpha_rob_ant,
            )

            ene = energy_function(domain_omega, u, spacestep, mu1, V_0)
            # ene = existence_surface(spacestep, u)

            energies.append((ene, chosen_indexes))

        energies.sort()
        energy[k] = energies[0][0]
        # The energy at step k is considered to be the minimal one

        if choice is None:
            chi_coords = [boundary_coords[i] for i in energies[0][1]]
            choice = (energies[0][0], build_chi(M, N, chi_coords))
        else:
            if energy[k] > choice[0]:
                chi_coords = [boundary_coords[i] for i in energies[0][1]]
                choice = (energies[0][0], build_chi(M, N, chi_coords))

        # Now we update pheromones
        T = (1 - RHO) * T

        for e, indexes in energies:
            for index in indexes:
                T[index] += Q / e

    return choice[1], energy, u


def existence_surface(spacestep, u):
    v = u / np.linalg.norm(u)
    s = (spacestep**2) * np.sum(abs(v) ** 4)
    print(f"-------------- surface existence : {1/s}")
    return 1 / s


def find_possible_nodes(boundary_coords, chosen_coords):
    res = []
    for k, coord in enumerate(boundary_coords):
        i, j = coord

        if (
            belongs((i + 1, j), chosen_coords)
            or belongs((i - 1, j), chosen_coords)
            or belongs((i, j - 1), chosen_coords)
            or belongs((i, j + 1), chosen_coords)
        ) and (not belongs(coord, chosen_coords)):
            res.append(k)

    return res


def energy_function(domain_omega, u, spacestep, mu1, V_0):
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

    energy = np.sum((spacestep * np.absolute(u)) ** 2)
    return energy


def build_chi(M, N, coords):
    chi = np.zeros((M, N), dtype=np.float64)
    val = 1.0
    for k in coords:
        chi[k[0], k[1]] = val
    return chi


def belongs(x, nparr):
    for y in nparr:
        if x[0] == y[0] and x[1] == y[1]:
            return True
    return False


def dist(coord1, coord2):
    distance = np.sqrt(np.sum((coord1 - coord2) ** 2))
    return distance


def normalize(vector):
    return vector / np.sum(vector)


def visibility(coord: tuple, boundary: list[tuple]):
    if not boundary:
        return 1
    min_dist = np.inf
    for bound_coord in boundary:
        d = dist(coord, bound_coord)
        if (d < min_dist) and (d > 0):
            min_dist = d
    return 1 / min_dist


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
    wavenumber = np.sqrt(kx**2 + ky**2)  # wavenumber
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
    V_obj = np.sum(np.sum(chi)) / S  # constraint on the density
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
    chi0 = chi.copy()
    u0 = u.copy()

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization
    energy = np.zeros((100 + 1, 1), dtype=np.float64)
    # chi, energy, u, grad = your_optimization_procedure(...)
    chi, energy, u = aco(
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
