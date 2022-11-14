####### standard #######
import numpy
import matplotlib.pyplot as plt
from copy import deepcopy

####### my packages #######
import _env
import frontier_generation as fro_gen
import preprocessing
import processing
import postprocessing
import utils
from utils import score_energy_minus, set_PDE_params

###########################
import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)

import logging
logging.basicConfig(filename="log.txt", level=logging.DEBUG,
                    format="%(asctime)s-%(levelname)s-%(message)s", filemode="w")
###########################


class IndivAlgoGenDomainOpti():
    def __init__(self, Alpha, wavenumber, score_fn, init_domain, init_score, velocity_factor):
        self.M, self.N = init_domain.shape
        self.domain = init_domain
        self.Alpha = Alpha
        self.chi = preprocessing.set_full_chi(init_domain)
        self.wavenumber = wavenumber

        self.score_fn = score_fn
        self.score = init_score

        self.velocity_factor = velocity_factor

    def generate_new_location(self, nbre_iter, distance_max, distance_min_to_overcome=+numpy.infty):
        # this function is exactly the same as the one in WolfDomainOpti
        M, N = self.M, self.N
        domain_omega = deepcopy(self.domain)
        original_frontier = fro_gen.get_indices_frontier_robin(self.domain)
        raw_vol_domain_0 = fro_gen.compute_raw_volume_interior(domain_omega)

        frontier = fro_gen.get_indices_frontier_robin(domain_omega)

        distance = 0
        k = 0
        while k < nbre_iter and distance < min(distance_min_to_overcome, distance_max):
            print(
                f"generate_frontier_random: iteration {k}/{nbre_iter-1} {distance}", end='\r')
            neigh_indices = fro_gen.get_neighbours_points_in_grid(
                frontier, 0, M-1, 0, N-1)
            nbre_chosen = numpy.random.randint(
                1, 3)
            points_chosen_idx = numpy.random.choice(
                neigh_indices.shape[0], size=nbre_chosen)
            points_chosen_neigh = neigh_indices[points_chosen_idx, :]
            domain_omega = fro_gen.update_frontier(
                domain_omega, points_chosen_neigh)
            frontier = fro_gen.get_indices_frontier_robin(domain_omega)

            raw_vol_domain = fro_gen.compute_raw_volume_interior(domain_omega)
            while numpy.abs(raw_vol_domain-raw_vol_domain_0) > 1:
                neigh_indices = fro_gen.get_neighbours_points_in_grid(
                    frontier, 0, M-1, 0, N-1)
                if raw_vol_domain < raw_vol_domain_0:
                    info_env_to_dig = _env.NODE_COMPLEMENTARY
                else:
                    info_env_to_dig = _env.NODE_INTERIOR

                neigh_indices_to_dig = fro_gen.filter_points(
                    neigh_indices, domain_omega, info_env_to_dig)
                point_chosen_idx = [numpy.random.randint(
                    1, neigh_indices_to_dig.shape[0])]
                point_chosen = neigh_indices_to_dig[
                    point_chosen_idx, :]
                domain_omega = fro_gen.update_frontier(
                    domain_omega, point_chosen)
                frontier = fro_gen.get_indices_frontier_robin(domain_omega)
                raw_vol_domain = fro_gen.compute_raw_volume_interior(
                    domain_omega)
            k += 1
            # choosing this order in arguments, I penalize new frontier with too many points
            distance = utils.get_mean_point_wise_distance(
                frontier, original_frontier)

        logging.info(
            f"generate_frontier_random: iteration {k}/{nbre_iter-1} {distance}")
        print()

        # x, y = fro_gen.set_xy_from_domain(domain_omega)
        return domain_omega

    def move_to(self, new_domain, new_frontier, PDE_params):
        self.domain = new_domain
        self.chi = preprocessing.set_full_chi(new_domain)
        self.score = self.score_fn(
            new_domain, PDE_params, self.Alpha, self.wavenumber, self.chi)

    def brownian_motion(self, nbre_iter, PDE_params, velocity_factor=None):
        if velocity_factor is None:
            velocity_factor = self.velocity_factor

        domain_omega = self.generate_new_location(
            nbre_iter, velocity_factor)
        frontier = fro_gen.get_indices_frontier_robin(domain_omega)
        self.move_to(domain_omega, frontier, PDE_params)


def optimization_procedure(M, N, wavenumber, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, N_indiv, epochs):
    print('--- optimization procedure algo gen ---')
    logging.info('--- optimization procedure ---')
    PDE_params = [f, f_dir, f_neu, f_rob, beta_pde,
                  alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob]

    score_fn_to_maximise = dico_score_fn_name_to_fn[config["OPTIMIZATION"]["ALGO_GEN_DOMAIN"]
                                                    ["score_fn_to_maximise"]]
    nbre_iter_gen_domain = config["OPTIMIZATION"]["ALGO_GEN_DOMAIN"]["nbre_iter_gen_domain"]
    velocity_factor_max = config["OPTIMIZATION"]["ALGO_GEN_DOMAIN"]["velocity_factor_max"]
    velocity_factor_min = config["OPTIMIZATION"]["ALGO_GEN_DOMAIN"]["velocity_factor_min"]
    ratio_reproduce = config["OPTIMIZATION"]["ALGO_GEN_DOMAIN"]["RATIO_REPRODUCE"]

    assert ratio_reproduce < 0.5
    N_reproduce = int(N_indiv*ratio_reproduce)

    all_indivs = []
    all_indivs_score = []
    for indiv_i in range(N_indiv):
        logging.info(f"Initializing indiv {indiv_i}")
        init_domain, _, _ = fro_gen.generate_frontier_random(M, N, 100)
        init_chi = preprocessing.set_full_chi(init_domain)
        init_score = score_fn_to_maximise(
            init_domain, PDE_params, Alpha, wavenumber, init_chi)
        all_indivs += [IndivAlgoGenDomainOpti(Alpha,
                                              wavenumber, score_fn_to_maximise, init_domain, init_score, velocity_factor_max)]
        all_indivs_score += [init_score]

    history = {"best domains": [], "best scores": []}
    for epoch in range(epochs):
        print(f"epoch:{epoch}")
        logging.info(f"epoch:{epoch}")
        # choose N_reproduce individuals
        N_reproduce_best_indices = numpy.argpartition(
            all_indivs_score, -N_reproduce)[-N_reproduce:]
        N_reproduce_to_replace_indices = numpy.argpartition(
            all_indivs_score, N_reproduce)[:N_reproduce]

        # they reproduce with themselves
        n_replaced = 0
        velocity_factor = (velocity_factor_min-velocity_factor_max) * \
            epoch/max(1, epochs-1)+velocity_factor_max
        for indiv_i in N_reproduce_best_indices:
            parent_indiv = all_indivs[indiv_i]
            child_indiv = IndivAlgoGenDomainOpti(Alpha,
                                                 wavenumber, score_fn_to_maximise, parent_indiv.domain, parent_indiv.score, velocity_factor)
            # mutation
            child_indiv.brownian_motion(
                nbre_iter_gen_domain, PDE_params)

            # add new indiv
            child_i = N_reproduce_to_replace_indices[n_replaced]
            all_indivs[child_i] = child_indiv
            all_indivs_score[child_i] = child_indiv.score
            n_replaced += 1

        # update history
        best_score_i = numpy.argmax(all_indivs_score)
        history["best domains"].append(all_indivs[best_score_i].domain)
        history["best scores"].append(all_indivs_score[best_score_i])
        postprocessing.save_array(
            all_indivs[best_score_i].domain, f"epoch_{epoch}_best_indiv_domain")

        scores = numpy.array([[-s] for s in history["best scores"]])
        print("best energy of epoch:", scores[-1, 0])
        logging.info("  best energy of epoch:" + str(scores[-1, 0]))
        postprocessing._plot_energy_history(scores)

    return history


def run_optimization_procedure():
    N = config["GEOMETRY"]["N_POINTS_AXIS_X"]  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    spacestep = 1.0/N

    # -- define absorbing material
    # -- this is the function you have written during your project
    from compute_alpha_folder.compute_alpha import compute_alpha
    material = config["GEOMETRY"]["MATERIAL"]
    wavenumber = config["PDE"]["WAVENUMBER"]
    Alpha = compute_alpha(wavenumber, material)[0]

    N_indiv = max(config['OPTIMIZATION']["ALGO_GEN_DOMAIN"]["N_INDIV"], 1)
    [f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir,
        beta_neu, beta_rob, alpha_rob] = set_PDE_params(M, N)

    epochs = config['OPTIMIZATION']["ALGO_GEN_DOMAIN"]["EPOCHS"]

    history = optimization_procedure(M, N, wavenumber, f, f_dir, f_neu, f_rob,
                                     beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                     Alpha, N_indiv, epochs)

    energy = numpy.array([[-s] for s in history["best scores"]])
    print("energy\n", energy)
    postprocessing._plot_energy_history(energy)

    un = processing.solve_helmholtz(history["best domains"][-1], spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)

    lowest_energy_id = int(numpy.argmin(energy, axis=0))
    lowest_energy = energy[lowest_energy_id][0]
    lowest_domain = history["best domains"][lowest_energy_id]
    postprocessing._plot_domain_study(
        lowest_domain, lowest_energy, un)


if __name__ == '__main__':
    dico_score_fn_name_to_fn = {"energy_minus": score_energy_minus}

    run_optimization_procedure()
