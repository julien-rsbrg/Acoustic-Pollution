####### standard #######
import numpy
import matplotlib.pyplot as plt
####### my packages #######
import preprocessing
import processing
###########################
import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)
####################### classic #######################


def clip(values, vmin, vmax):
    clipped_values = numpy.where(values >= vmin, values, vmin)
    clipped_values = numpy.where(clipped_values >= vmax, vmax, clipped_values)
    return clipped_values


def get_norm_L2(A):
    return numpy.abs(numpy.trace(A@numpy.conjugate(A.T)))


def get_norm_L1(A):
    return numpy.sum(numpy.abs(A))


def get_distance_sets(setA, setB):
    A_without_B = setA.intersection(setA.symmetric_difference(setB))
    B_without_A = setB.intersection(setB.symmetric_difference(setA))
    return len(B_without_A)+len(A_without_B)


def test_distance_sets():
    A = {(3, 3), (4, 5), (5, 6), (7, 6)}
    B = {(4, 5), (8, 1), (7, 6)}
    print("distance_sets(A, B)", get_distance_sets(A, B))


def get_shifted_cubic_matrix_axis1(B):
    res = B.copy()
    res = numpy.expand_dims(res, axis=-1)
    A = numpy.arange(B.shape[1])
    for _ in range(B.shape[1]-1):
        A = numpy.concatenate([A[1:], [A[0]]], axis=0)
        shifted_B = numpy.expand_dims(B[:, A], axis=-1)
        res = numpy.concatenate([res, shifted_B], axis=-1)
    return res


# def get_distance_L1_between_columns(B):
#     shifted_B = get_shifted_cubic_matrix_axis1(B)
#     augmented_B = numpy.expand_dims(B, axis=-1)@numpy.ones((1, B.shape[1]))
#     res = numpy.sum(numpy.abs(augmented_B-shifted_B), axis=0)
#     return res


# def get_distance_L1(B, column_0, column_1):
#     return B[column_0, column_1-column_0]


def get_id_n_nearest_nodes(node_coords, new_node_coords, n=2):
    assert node_coords.shape[0] >= n
    assert new_node_coords.shape[-1] == new_node_coords.shape[-1], "not the same dimension of nodes"
    Mdist = numpy.ones((node_coords.shape[0], 1))@new_node_coords-node_coords
    Mdist = Mdist**2@numpy.ones((node_coords.shape[-1], 1))
    # print("Mdist:", Mdist[:, 0], "shape:", Mdist.shape)
    # print("nearest nodes id:", np.argpartition(Mdist[:, 0], n)[:n])
    return numpy.argpartition(Mdist[:, 0], n)[:n]


def get_nearest_distance(node_coords, new_node_coords):
    assert new_node_coords.shape[-1] == new_node_coords.shape[-1], "not the same dimension of nodes"
    Mdist = numpy.ones((node_coords.shape[0], 1))@new_node_coords-node_coords
    Mdist = Mdist**2@numpy.ones((node_coords.shape[-1], 1))
    Mdist = numpy.sqrt(Mdist)
    return numpy.sort(Mdist, axis=0)[0, 0]


def get_mean_point_wise_distance(src_node_coords, bis_node_coords):
    # WARNING: it is not symmetric !
    assert src_node_coords.shape[-1] == bis_node_coords.shape[-1], "not the same dimension of nodes"
    src_node_coords_cp = src_node_coords.copy()
    bis_node_coords_cp = bis_node_coords.copy()
    n = src_node_coords_cp.shape[0]
    bis_n = bis_node_coords_cp.shape[0]

    bis_node_coords_cp = numpy.expand_dims(bis_node_coords, axis=-1)
    bis_node_coords_cp = numpy.matmul(
        bis_node_coords_cp, numpy.ones((bis_n, 1, n)))
    bis_node_coords_cp = bis_node_coords_cp.T

    src_node_coords_cp = numpy.expand_dims(src_node_coords_cp, axis=-1)
    src_node_coords_cp = numpy.matmul(
        src_node_coords_cp, numpy.ones((n, 1, bis_n)))

    Mdist = (src_node_coords_cp - bis_node_coords_cp)**2
    Mdist = numpy.sqrt(numpy.sum(Mdist, axis=1))

    Mdist = numpy.amin(Mdist, axis=1)
    mean_dist = numpy.mean(Mdist)
    return mean_dist


def is_same_matrix(A, B):
    same_shape = A.shape == B.shape
    if not (same_shape):
        return False

    if len(A.shape) == 1:
        return all(A == B)
    else:
        for i in range(A.shape[0]):
            same_matrix = is_same_matrix(A[i], B[i])
            if not (same_matrix):
                return False
        return True


def is_elem_in(elem, elems):
    for e in elems:
        if is_same_matrix(elem, e):
            return True
    return False

####################### objective functions #######################


def compute_energy(u, spacestep):
    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2

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
    return raw_energy


def set_PDE_params(M, N):
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(
        M, N)
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- define boundary conditions
    # planar wave defined on top
    if config["PDE"]["INCIDENT_WAVE"] == "spherical":
        # spherical wave defined on top
        f_dir[:, :] = 0.0
        f_dir[0, int(N/2)] = 10.0
    else:
        f_dir[:, :] = 0.0
        f_dir[0, 0:N] = 1.0

    return [f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob]


def convert_into_set_tuple(A):
    assert len(A.shape) == 2
    A = {tuple([a[i] for i in range(A.shape[1])]) for a in A}
    return A


def score_energy_minus(domain_omega, PDE_params, Alpha, wavenumber, chi):
    M, N = domain_omega.shape
    spacestep = 1.0 / N

    [f, f_dir, f_neu, f_rob, beta_pde, alpha_pde,
        alpha_dir, beta_neu, beta_rob, alpha_rob] = PDE_params

    alpha_rob = Alpha*chi

    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    energy = compute_energy(u, spacestep)

    return - energy
####################### domain optimization #######################
