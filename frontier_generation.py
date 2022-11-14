import numpy
import matplotlib.pyplot as plt
from copy import deepcopy


###############
import _env
import processing
import preprocessing
import postprocessing
import utils

import logging
logging.basicConfig(filename="log.txt", level=logging.DEBUG,
                    format="%(asctime)s-%(levelname)s-%(message)s")
###############


def remove_these_points(points, points_to_remove):
    flag_matrix = numpy.zeros(points.shape[0])
    for point in points_to_remove:
        flag_matrix_point = numpy.zeros(points.shape)
        for i in range(points.shape[-1]):
            flag_matrix_point[:, i] = numpy.where(
                points[:, i] == point[i], 1, 0)
        flag_matrix_point = numpy.prod(flag_matrix_point, axis=-1)
        flag_matrix = flag_matrix+flag_matrix_point

    # print("flag_matrix\n", flag_matrix)
    points = points[flag_matrix == 0]
    return points


def remove_out_of_bound(points, imin, imax, jmin, jmax):
    bounds = numpy.array([[imin, imax], [jmin, jmax]])
    flag_matrix = numpy.zeros(points.shape)
    for i in range(points.shape[-1]):
        flag_matrix[:, i] = numpy.where(
            points[:, i] < bounds[i, 0], 1, 0)
        flag_matrix[:, i] += numpy.where(
            points[:, i] > bounds[i, 1], 1, 0)
    flag_matrix = numpy.sum(flag_matrix, axis=-1)
    points = points[flag_matrix == 0]
    return points


def get_neighbours_points_in_grid(points_indices, imin, imax, jmin, jmax):
    """
    points_indices (np.array of shape (N_points,2))
    """
    N_points = points_indices.shape[0]
    translation = numpy.ones((N_points, 1), dtype=int)
    zeros_translation = 0*translation
    i_translation = numpy.concatenate([translation, zeros_translation], axis=1)
    j_translation = numpy.concatenate([zeros_translation, translation], axis=1)
    neigh_points_indices = points_indices
    for coef in [-1, 1]:
        neigh_points_indices_i = points_indices+coef*i_translation
        neigh_points_indices_j = points_indices+coef*j_translation
        neigh_points_indices = numpy.concatenate(
            [neigh_points_indices, neigh_points_indices_i, neigh_points_indices_j], axis=0)

    neigh_points_indices = remove_these_points(
        neigh_points_indices, points_indices)
    neigh_points_indices = remove_out_of_bound(
        neigh_points_indices, imin, imax, jmin, jmax)
    return neigh_points_indices


def test_get_neighbours_points_in_grid():
    points = numpy.array([
        [0, 0],
        [-1, 0],
        [1, 3],
        [1, 0]
    ])

    neighbours_points = get_neighbours_points_in_grid(points, -3, 3, -3, 3)
    neighbours_points = remove_out_of_bound(
        neighbours_points, -3, 3, -3, 3)


def compute_raw_volume_interior(domain_omega):
    return numpy.sum(numpy.where(
        domain_omega == _env.NODE_INTERIOR, 1, 0))


def get_indices_frontier_robin(domain_omega):
    i_robin_points, j_robin_points = numpy.where(
        domain_omega == _env.NODE_ROBIN)
    i_robin_points = numpy.expand_dims(i_robin_points, axis=1)
    j_robin_points = numpy.expand_dims(j_robin_points, axis=1)
    return numpy.concatenate([i_robin_points, j_robin_points], axis=-1)


def test_get_indices_frontier_robin():
    N = 50
    M = 2*N
    domain_omega, _, _, _, _ = preprocessing._set_geometry_of_domain(M, N)
    print(get_indices_frontier_robin(domain_omega))


def filter_points(points, domain_omega, info_env):
    env_values = domain_omega[points[:, 0], points[:, 1]]
    filtered_points = points[env_values == info_env]
    return filtered_points


def test_filter_points():
    domain_omega = numpy.array([[0, -1, 0],
                                [-1, 0, 0]])
    points = numpy.array([[0, 0],
                          [0, 1],
                          [1, 0],
                          [1, 1]])

    print("filter_points\n", filter_points(points, domain_omega, 0))


def get_values_around_point(domain_omega, point_indices, cross_only=False):
    values_around = numpy.zeros((8, 3), dtype=int)
    window = {-1, 0, 1}
    count = 0
    for i_add in window:
        for j_add in window:
            if i_add != 0 or j_add != 0:
                if (cross_only and (i_add == 0 or j_add == 0)) or not (cross_only):
                    i, j = point_indices[0]+i_add, point_indices[1]+j_add
                    if i >= 0 and i < domain_omega.shape[0] and j >= 0 and j < domain_omega.shape[1]:
                        values_around[count, :] = [domain_omega[i, j], i, j]
                        count += 1
    return values_around[values_around[:, 0] != 0]


def test_get_values_around_point():
    N = 50
    M = 2*N
    domain_omega, x, y, x_plot, y_plot = preprocessing._set_geometry_of_domain(
        M, N, level=0)
    point_indices = numpy.array([N, N//2])
    values_around = get_values_around_point(domain_omega, point_indices)
    print(values_around, values_around.shape)
    print(set(values_around[:, 0]))

    values_around = get_values_around_point(
        domain_omega, point_indices, cross_only=True)
    print(values_around, values_around.shape)
    print(set(values_around[:, 0]))


def is_in_dirichlet_or_neumann(domain_omega, point_indices):
    info_env = domain_omega[point_indices[0], point_indices[1]]
    if info_env in [_env.NODE_DIRICHLET, _env.NODE_NEUMANN]:
        return True
    return False


def change_one_neighbour(domain_omega, cross_values_around, neigh_value_to_give):
    for neighbour in cross_values_around:
        if neighbour[0] == _env.NODE_ROBIN:
            neigh_indices = neighbour[1:]
            neigh_cross_values_around = get_values_around_point(
                domain_omega, neigh_indices, cross_only=True)
            set_neigh_cross_values_around = set(
                neigh_cross_values_around[:, 0])
            if not ({_env.NODE_INTERIOR, _env.NODE_COMPLEMENTARY}.issubset(set_neigh_cross_values_around)):
                domain_omega[neigh_indices[0],
                             neigh_indices[1]] = neigh_value_to_give
                return domain_omega

    return domain_omega


def clean_domain(domain_omega):
    M, N = domain_omega.shape
    frontier = get_indices_frontier_robin(domain_omega)

    # check that NODE_COMPLEMENTARY and NODE_INTERIOR nodes do not touch each
    # add a NODE_ROBIN otherwise
    neigh_indices = get_neighbours_points_in_grid(
        frontier, 0, M-1, 0, N-1)
    in_interior_neigh_indices = filter_points(
        neigh_indices, domain_omega, _env.NODE_INTERIOR)
    for point_to_check in in_interior_neigh_indices:
        cross_values_around = get_values_around_point(
            domain_omega, point_to_check, cross_only=True)
        if _env.NODE_COMPLEMENTARY in cross_values_around[:, 0]:
            domain_omega[point_to_check[0],
                         point_to_check[1]] = _env.NODE_ROBIN

    # remove NODE_ROBIN not being a separating node between NODE_INTERIOR and NODE_COMPLEMENTARY
    for point_on_frontier in frontier:
        cross_values_around = get_values_around_point(
            domain_omega, point_on_frontier, cross_only=True)
        set_cross_values_around = set(cross_values_around[:, 0])
        if not ({_env.NODE_INTERIOR, _env.NODE_COMPLEMENTARY}.issubset(set_cross_values_around)):
            # not real frontier
            if _env.NODE_INTERIOR in set_cross_values_around:
                domain_omega[point_on_frontier[0],
                             point_on_frontier[1]] = _env.NODE_INTERIOR
            else:
                domain_omega[point_on_frontier[0],
                             point_on_frontier[1]] = _env.NODE_COMPLEMENTARY
    return domain_omega


def update_frontier(domain_omega, points_chosen):
    # WARNING : change points_chosen
    # remove points located on forbidden locations
    M, N = domain_omega.shape
    new_domain_omega = deepcopy(domain_omega)
    not_too_low = numpy.where(points_chosen[:, 0] >= N//3)
    points_chosen = points_chosen[not_too_low]
    not_too_high = numpy.where(points_chosen[:, 0] <= M-1-5)
    points_chosen = points_chosen[not_too_high]

    for point_indices in points_chosen:
        if not (is_in_dirichlet_or_neumann(domain_omega, point_indices)):
            suppressed_info_env = new_domain_omega[point_indices[0],
                                                   point_indices[1]]
            new_domain_omega[point_indices[0],
                             point_indices[1]] = _env.NODE_ROBIN
            # print(("   after putting 1 node to ROBIN : {}>1 ? ".format(
            #     utils.get_norm_L1(new_domain_omega-domain_omega))))

            cross_values_around = get_values_around_point(
                domain_omega, point_indices, cross_only=True)

            if _env.NODE_INTERIOR == suppressed_info_env:
                # point_indices is in interior
                new_domain_omega = change_one_neighbour(
                    new_domain_omega, cross_values_around, _env.NODE_COMPLEMENTARY)
            elif _env.NODE_COMPLEMENTARY == suppressed_info_env:
                new_domain_omega = change_one_neighbour(
                    new_domain_omega, cross_values_around, _env.NODE_INTERIOR)

    new_domain_omega = preprocessing.reset_boundary_env(new_domain_omega)

    new_domain_omega = clean_domain(new_domain_omega)

    return new_domain_omega


def test_update_frontier():
    N = 50
    M = 2*N
    domain_omega, _, _, _, _ = preprocessing._set_geometry_of_domain(
        M, N, level=0)

    plt.imshow(domain_omega)
    plt.show()
    points_chosen = numpy.array([[N+1, i] for i in range(0, N)])
    domain_omega = update_frontier(domain_omega, points_chosen)
    plt.imshow(domain_omega)
    plt.show()


def set_xy_from_domain(domain_omega):
    # WARNING : not checked with a tortuous domain (ie 2 frontier points along the same j-axis and with a distance>=2)
    # does the order of the elements in x and y matter ?
    frontier = get_indices_frontier_robin(domain_omega)
    return frontier[:, 1], frontier[:, 0]


def generate_frontier_random(M, N, nbre_iter):
    print("--- generate_frontier_random ---")
    logging.info("--- generate_frontier_random ---")
    domain_omega, _, _, _, _ = preprocessing._set_geometry_of_domain(M, N)
    raw_vol_domain_0 = compute_raw_volume_interior(domain_omega)

    frontier = get_indices_frontier_robin(domain_omega)
    for k in range(nbre_iter):
        print(f"iteration {k}/{nbre_iter-1}        ", end="\r")

        neigh_indices = get_neighbours_points_in_grid(
            frontier, 0, M-1, 0, N-1)
        in_interior_neigh_indices = filter_points(
            neigh_indices, domain_omega, _env.NODE_INTERIOR)
        nbre_chosen = numpy.random.randint(
            5, in_interior_neigh_indices.shape[0]//4)
        points_chosen_in_interior_idx = numpy.random.choice(
            in_interior_neigh_indices.shape[0], size=nbre_chosen)
        points_chosen_in_interior = in_interior_neigh_indices[points_chosen_in_interior_idx, :]
        domain_omega = update_frontier(domain_omega, points_chosen_in_interior)
        frontier = get_indices_frontier_robin(domain_omega)

        raw_vol_domain = compute_raw_volume_interior(domain_omega)
        while numpy.abs(raw_vol_domain-raw_vol_domain_0) > 1:
            neigh_indices = get_neighbours_points_in_grid(
                frontier, 0, M-1, 0, N-1)
            in_complement_neigh_indices = filter_points(
                neigh_indices, domain_omega, _env.NODE_COMPLEMENTARY)
            point_chosen_in_complement_idx = [numpy.random.randint(
                1, in_complement_neigh_indices.shape[0])]
            point_chosen_in_complement = in_complement_neigh_indices[
                point_chosen_in_complement_idx, :]
            domain_omega = update_frontier(
                domain_omega, point_chosen_in_complement)
            frontier = get_indices_frontier_robin(domain_omega)
            raw_vol_domain = compute_raw_volume_interior(domain_omega)
    logging.info(f"stopped at iteration {k}/{nbre_iter-1}")
    print()
    x, y = set_xy_from_domain(domain_omega)
    return domain_omega, x, y


def test_generate_frontier_random():
    N = 50
    M = 2*N
    nbre_iter = 1000
    domain_omega, x, y = generate_frontier_random(M, N, nbre_iter)
    plt.imshow(domain_omega)
    plt.show()
    return domain_omega, x, y

# generate_frontier_random(100, 50)


if __name__ == '__main__':
    import preprocessing
    import yaml
    from yaml.loader import SafeLoader

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)

    # test_filter_points()
    # test_get_values_around_point()
    # test_update_frontier()
    # test_get_indices_frontier_robin()
    domain_omega, x, y = test_generate_frontier_random()
    print("x\n", x, x.shape)
    print("y\n", y, y.shape)

    ##########################################################

    N = config["GEOMETRY"]["N_POINTS_AXIS_X"]  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    wavenumber = config["PDE"]["WAVENUMBER"]

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(
        M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

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

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    postprocessing.myimshow(numpy.real(u), title='$\operatorname{Re}(u_{n})$ in $\Omega$',
                            colorbar='colorbar', cmap='jet', vmin=-1, vmax=1, filename='fig_un_re.jpg')
    postprocessing.myimshow(numpy.imag(u), title='$\operatorname{Im}(u_{n})$ in $\Omega$',
                            colorbar='colorbar', cmap='jet', vmin=-1, vmax=1, filename='fig_un_im.jpg')
