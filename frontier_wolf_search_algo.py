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
from utils import score_energy_minus, set_PDE_params, convert_into_set_tuple

###########################
import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)

import logging
logging.basicConfig(filename="log.txt", level=logging.DEBUG,
                    format="%(asctime)s-%(levelname)s-%(message)s", filemode="w")
###########################


class WolfDomainOpti():
    def __init__(self, Alpha, wavenumber, radius_visual_area, escape_step_size, velocity_factor, proba_enemy_appear, prey_search_perseverance, prey_iter, prey_score_gap_to_hunt, score_fn_to_maximise, max_memory_len, init_domain, init_score, first_randomize=False):
        self.M, self.N = init_domain.shape
        self.domain = init_domain
        self.Alpha = Alpha
        self.chi = preprocessing.set_full_chi(init_domain)
        self.wavenumber = wavenumber

        self.score_fn = score_fn_to_maximise
        self.score = init_score

        self.radius_visual_area = radius_visual_area
        self.escape_step_size = escape_step_size
        self.velocity_factor = velocity_factor
        self.proba_enemy_appear = proba_enemy_appear

        self.prey_search_perseverance = max(prey_search_perseverance, 1)
        self.prey_iter = prey_iter
        self.prey_score_gap_to_hunt = prey_score_gap_to_hunt

        frontier = fro_gen.get_indices_frontier_robin(self.domain)
        # #  as we know, frontier will only move along axis 0
        # frontier = frontier[numpy.argsort(frontier[:, 1]), :]

        self.memory = [frontier]
        self.max_memory_len = max(max_memory_len, 1)

        # self.prey_memory = [previous best prey score, best prey domain]
        self.prey_memory = [-numpy.infty, None]

        if first_randomize:
            domain_omega = self.generate_new_location(
                600, 4*self.radius_visual_area)
            frontier = fro_gen.get_indices_frontier_robin(domain_omega)
            PDE_params = set_PDE_params(self.M, self.N)
            self.move_to(domain_omega, frontier, PDE_params)

    def get_set_frontier(self):
        return convert_into_set_tuple(self.memory[-1])

    def update_memory(self, new_frontier):
        if len(self.memory) == self.max_memory_len:
            self.memory = self.memory[1:]
            self.memory.append(new_frontier)
        else:
            self.memory.append(new_frontier)

    def move_to(self, new_domain, new_frontier, PDE_params):
        self.update_memory(new_frontier)
        self.domain = new_domain
        self.chi = preprocessing.set_full_chi(new_domain)
        self.score = self.score_fn(
            new_domain, PDE_params, self.Alpha, self.wavenumber, self.chi)

    def generate_new_location(self, nbre_iter, distance_max, distance_min_to_overcome=+numpy.infty):
        # this function is very similar to frontier_generation.generate_frontier_random(...)
        M, N = self.M, self.N
        domain_omega = deepcopy(self.domain)
        raw_vol_domain_0 = fro_gen.compute_raw_volume_interior(domain_omega)

        frontier = fro_gen.get_indices_frontier_robin(domain_omega)

        distance = 0
        k = 0
        already_known = True
        while (k < nbre_iter or already_known) and distance < min(distance_min_to_overcome, distance_max):
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
                frontier, self.memory[-1])
            already_known = utils.is_elem_in(frontier, self.memory)

        logging.info(
            f"generate_frontier_random: iteration {k}/{nbre_iter-1} {distance}")
        print()

        # x, y = fro_gen.set_xy_from_domain(domain_omega)
        return domain_omega

    def update_prey_memory(self, best_score_prey, best_domain_prey):
        if best_score_prey < self.prey_memory[0]:
            best_score_prey = self.prey_memory[0]
            best_domain_prey = self.prey_memory[1]
        else:
            self.prey_memory[0] = best_score_prey
            self.prey_memory[1] = best_domain_prey
        return best_score_prey, best_domain_prey

    def prey_new_food_initiatively(self, PDE_params):
        found_good_prey = False

        best_score_prey = -numpy.infty
        best_domain_prey = None

        for _ in range(self.prey_search_perseverance):
            domain_prey = self.generate_new_location(
                self.prey_iter, self.radius_visual_area)
            score_prey = self.score_fn(
                domain_prey, PDE_params, self.Alpha, self.wavenumber, self.chi)
            if score_prey > best_score_prey:
                best_score_prey = score_prey
                best_domain_prey = domain_prey

        best_score_prey, best_domain_prey = self.update_prey_memory(
            best_score_prey, best_domain_prey)

        if best_score_prey > self.prey_score_gap_to_hunt+self.score:
            # hunt prey
            found_good_prey = True

        return found_good_prey, best_domain_prey, best_score_prey

    def move_close_to(self, destination_frontier, PDE_params, nbre_modif_max=50, nbre_iter=20):
        # TODO optimize it
        original_domain = deepcopy(self.domain)
        original_frontier = deepcopy(self.memory[-1])

        new_frontier = deepcopy(self.memory[-1])
        new_domain = deepcopy(self.domain)

        distance = 0
        k = 0
        while k < nbre_modif_max and distance < self.velocity_factor*self.radius_visual_area:
            kbis, already_equal = 0, True
            while kbis < nbre_iter and already_equal:
                idx = numpy.random.randint(0, new_frontier.shape[0])
                point_i, point_j = new_frontier[idx, :]
                point_dest_id = utils.get_id_n_nearest_nodes(
                    destination_frontier, numpy.array([[point_i, point_j]]), 1)
                dest_i, dest_j = destination_frontier[point_dest_id,
                                                      0], destination_frontier[point_dest_id, 1]
                move_point_i, move_point_j = int(numpy.sign(
                    dest_i-point_i)), int(numpy.sign(dest_j-point_j))
                already_equal = (move_point_i == 0 and move_point_j == 0)
                kbis += 1
            point_i, point_j = point_i+move_point_i, move_point_j
            new_frontier[idx] = [point_i, point_j]

            new_domain = fro_gen.update_frontier(
                new_domain, numpy.array([[point_i, point_j]]))
            new_frontier = fro_gen.get_indices_frontier_robin(new_domain)
            # print("move close to step {} changed something in domain: {}>1 ?   ".format(
            #     k, utils.get_norm_L1(new_domain-original_domain)), end="\r")
            self.move_to(new_domain, new_frontier, PDE_params)

            # choosing this order in arguments, I penalize new frontier with too many points
            distance = utils.get_mean_point_wise_distance(
                new_frontier, original_frontier)

            k += 1

        return self.domain

    def brownian_motion(self, nbre_iter, PDE_params, velocity_factor=None):
        if velocity_factor is None:
            velocity_factor = self.velocity_factor

        domain_omega = self.generate_new_location(
            nbre_iter, velocity_factor*self.radius_visual_area)
        frontier = fro_gen.get_indices_frontier_robin(domain_omega)
        self.move_to(domain_omega, frontier, PDE_params)

    def escape(self, nbre_iter, PDE_params):
        domain_omega = self.generate_new_location(
            nbre_iter, distance_max=self.escape_step_size, distance_min_to_overcome=self.escape_step_size)
        frontier = fro_gen.get_indices_frontier_robin(domain_omega)
        self.move_to(domain_omega, frontier, PDE_params)


def optimization_procedure(M, N, wavenumber, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, N_wolves, epochs, patience, tol_early_stopping):
    logging.info('---Start optimization procedure---')

    PDE_params = [f, f_dir, f_neu, f_rob, beta_pde,
                  alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob]

    init_level = config["OPTIMIZATION"]["WSA_DOMAIN"]["INIT_LEVEL"]

    radius_visual_area = config["OPTIMIZATION"]["WSA_DOMAIN"]["radius_visual_area"]
    escape_step_size = config["OPTIMIZATION"]["WSA_DOMAIN"]["escape_step_size"]
    velocity_factor = config["OPTIMIZATION"]["WSA_DOMAIN"]["velocity_factor"]
    proba_enemy_appear = config["OPTIMIZATION"]["WSA_DOMAIN"]["proba_enemy_appear"]
    nbre_iter_gen_domain = config["OPTIMIZATION"]["WSA_DOMAIN"]["nbre_iter_gen_domain"]
    prey_search_perseverance = config["OPTIMIZATION"]["WSA_DOMAIN"]["prey_search_perseverance"]
    prey_iter = config["OPTIMIZATION"]["WSA_DOMAIN"]["prey_iter"]
    prey_score_gap_to_hunt = config["OPTIMIZATION"]["WSA_DOMAIN"]["prey_score_gap_to_hunt"]
    assert config["OPTIMIZATION"]["WSA_DOMAIN"]["score_fn_to_maximise"] in dico_score_fn_name_to_fn
    score_fn_to_maximise = dico_score_fn_name_to_fn[config["OPTIMIZATION"]["WSA_DOMAIN"]
                                                    ["score_fn_to_maximise"]]
    max_memory_len = config["OPTIMIZATION"]["WSA_DOMAIN"]["max_memory_len"]

    wolf_pack = []
    for wolf_i in range(N_wolves):
        logging.info(f"Initializing wolf {wolf_i}")
        init_domain, _, _, _, _ = preprocessing._set_geometry_of_domain(
            M, N, level=init_level)
        init_chi = preprocessing.set_full_chi(init_domain)
        init_score = score_fn_to_maximise(
            init_domain, PDE_params, Alpha, wavenumber, init_chi)
        new_wolf = WolfDomainOpti(Alpha, wavenumber, radius_visual_area, escape_step_size, velocity_factor, proba_enemy_appear,
                                  prey_search_perseverance, prey_iter, prey_score_gap_to_hunt, score_fn_to_maximise, max_memory_len, init_domain, init_score, first_randomize=True)
        wolf_pack += [new_wolf]

    # WARNING: distances are not symmetric
    wolf_pack_distances = numpy.zeros((N_wolves, N_wolves))
    for i in range(N_wolves):
        for j in range(N_wolves):
            distance = utils.get_mean_point_wise_distance(
                wolf_pack[i].memory[-1], wolf_pack[j].memory[-1])
            wolf_pack_distances[i, j] = distance
    postprocessing.save_array(
        wolf_pack_distances, "epoch_00_wolf_pack_distances")

    epoch = 0
    history = {"best domains": [], "best scores": []}
    activate_ealy_stopping = False
    best_wolf_i = -1
    while epoch < epochs and not (activate_ealy_stopping):
        print(" epoch ", epoch)
        logging.info(f' epoch: {epoch}')
        for wolf_i in range(N_wolves):
            print("wolf number :", wolf_i)
            logging.info(f"  wolf number: {wolf_i}")
            logging.info("    seek prey")
            found_good_prey, best_domain_prey, best_score_prey = wolf_pack[wolf_i].prey_new_food_initiatively(
                PDE_params)

            if wolf_i == best_wolf_i:
                # greedy policy
                if found_good_prey:
                    logging.info("    hunt prey greedy")
                    frontier_prey = fro_gen.get_indices_frontier_robin(
                        best_domain_prey)
                    wolf_pack[wolf_i].move_to(
                        best_domain_prey, frontier_prey, PDE_params)

            else:
                if found_good_prey:
                    logging.info("    hunt prey")
                    frontier_prey = fro_gen.get_indices_frontier_robin(
                        best_domain_prey)
                    wolf_pack[wolf_i].move_close_to(
                        frontier_prey, PDE_params)

                # Get closer to companion if better food
                logging.info("    seek companion")
                # choosing this order in arguments, I penalize new frontier with too many points
                distance_from_wolf_i = wolf_pack_distances[:, wolf_i]
                seen_companions = numpy.where(distance_from_wolf_i <=
                                              wolf_pack[wolf_i].radius_visual_area)[0]
                # for this initialization, one could have thought of putting wolf_i itself and remove the next "if".
                # however, move_close_to would run for nothing
                best_score_companion = -numpy.infty
                best_companion = -1
                for seen_companion in seen_companions:
                    if wolf_pack[seen_companion].score > best_score_companion:
                        best_score_companion = wolf_pack[seen_companion].score
                        best_companion = int(seen_companion)

                if best_score_companion > wolf_pack[wolf_i].score and best_score_companion > best_score_prey:
                    logging.info("    join companion")
                    frontier_comp = fro_gen.get_indices_frontier_robin(
                        wolf_pack[best_companion].domain)
                    # print("frontier_comp\n", frontier_comp, frontier_comp.shape)
                    wolf_pack[wolf_i].move_close_to(frontier_comp, PDE_params)

                elif not (found_good_prey):
                    logging.info("    brownian motion")
                    wolf_pack[wolf_i].brownian_motion(
                        nbre_iter_gen_domain, PDE_params)

                if numpy.random.rand(1) <= wolf_pack[wolf_i].proba_enemy_appear:
                    logging.info("    escape")
                    wolf_pack[wolf_i].escape(nbre_iter_gen_domain, PDE_params)

            for j in range(N_wolves):
                distance = utils.get_mean_point_wise_distance(
                    wolf_pack[i].memory[-1], wolf_pack[j].memory[-1])
                wolf_pack_distances[i, j] = distance

        # Update history
        best_wolf = wolf_pack[0]
        for wolf_i in range(1, N_wolves):
            if wolf_pack[wolf_i].score > best_wolf.score:
                best_wolf = wolf_pack[wolf_i]
                best_wolf_i = wolf_i

        history["best domains"].append(best_wolf.domain)
        history["best scores"].append(best_wolf.score)
        postprocessing.save_array(
            wolf_pack_distances, f"epoch_{epoch}_wolf_pack_distances")
        postprocessing.save_array(
            best_wolf.domain, f"epoch_{epoch}_best_wolf_domain")

        if len(history["best scores"]) > patience:
            if best_wolf.score < history["best scores"][-patience] - tol_early_stopping:
                activate_ealy_stopping = True
                logging.info("  Activated early stopping")
                print("Activated early stopping")

        scores = numpy.array([[-s] for s in history["best scores"]])
        print("last energy\n", scores[-1, 0])
        logging.info("  last energy" + str(scores[-1, 0]))
        postprocessing._plot_energy_history(scores)

        epoch += 1

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

    N_wolves = max(config['OPTIMIZATION']["WSA_DOMAIN"]["N_WOLVES"], 1)
    epochs = max(config['OPTIMIZATION']["WSA_DOMAIN"]["EPOCHS"], 1)
    patience = max(
        config['OPTIMIZATION']["WSA_DOMAIN"]["PATIENCE_EARLY_STOPPING"], 1)
    tol_early_stopping = config['OPTIMIZATION']["WSA_DOMAIN"]["TOL_EARLY_STOPPING"]

    [f, f_dir, f_neu, f_rob, beta_pde, alpha_pde, alpha_dir,
        beta_neu, beta_rob, alpha_rob] = set_PDE_params(M, N)
    history = optimization_procedure(M, N, wavenumber, f, f_dir, f_neu, f_rob,
                                     beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                     Alpha, N_wolves, epochs, patience, tol_early_stopping)

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
    postprocessing._plot_all_distances_arrays()


def test_WolfDomain():
    M, N = 100, 50
    from compute_alpha_folder.compute_alpha import compute_alpha
    material = config["GEOMETRY"]["MATERIAL"]
    wavenumber = config["PDE"]["WAVENUMBER"]
    Alpha = compute_alpha(wavenumber, material)[0]
    PDE_params = set_PDE_params(M, N)

    radius_visual_area = config["OPTIMIZATION"]["WSA_DOMAIN"]["radius_visual_area"]
    escape_step_size = config["OPTIMIZATION"]["WSA_DOMAIN"]["escape_step_size"]
    velocity_factor = config["OPTIMIZATION"]["WSA_DOMAIN"]["velocity_factor"]
    proba_enemy_appear = config["OPTIMIZATION"]["WSA_DOMAIN"]["proba_enemy_appear"]
    nbre_iter_gen_domain = config["OPTIMIZATION"]["WSA_DOMAIN"]["nbre_iter_gen_domain"]
    prey_search_perseverance = config["OPTIMIZATION"]["WSA_DOMAIN"]["prey_search_perseverance"]
    prey_iter = config["OPTIMIZATION"]["WSA_DOMAIN"]["prey_iter"]
    prey_score_gap_to_hunt = config["OPTIMIZATION"]["WSA_DOMAIN"]["prey_score_gap_to_hunt"]
    assert config["OPTIMIZATION"]["WSA_DOMAIN"]["score_fn_to_maximise"] in dico_score_fn_name_to_fn
    score_fn_to_maximise = dico_score_fn_name_to_fn[config["OPTIMIZATION"]["WSA_DOMAIN"]
                                                    ["score_fn_to_maximise"]]
    max_memory_len = config["OPTIMIZATION"]["WSA_DOMAIN"]["max_memory_len"]

    init_domain, _, _ = fro_gen.generate_frontier_random(M, N, 100)
    init_chi = preprocessing.set_full_chi(init_domain)
    init_score = score_fn_to_maximise(
        init_domain, PDE_params, Alpha, wavenumber, init_chi)

    wolf = WolfDomainOpti(Alpha, wavenumber, radius_visual_area, escape_step_size, velocity_factor, proba_enemy_appear,
                          prey_search_perseverance, prey_iter, prey_score_gap_to_hunt, score_fn_to_maximise, max_memory_len, init_domain, init_score)
    domain_found = wolf.generate_new_location(
        3000, distance_max=2, distance_min_to_overcome=1)
    frontier_found = fro_gen.get_indices_frontier_robin(domain_found)
    print("utils.get_norm_L1(domain_found-wolf.domain)",
          utils.get_norm_L1(domain_found-wolf.domain))
    frontier_wolf = fro_gen.get_indices_frontier_robin(wolf.domain)
    print('distance:', utils.get_mean_point_wise_distance(
        frontier_found, frontier_wolf))

    domain_found = wolf.generate_new_location(
        20, distance_max=0.2, distance_min_to_overcome=10)
    frontier_found = fro_gen.get_indices_frontier_robin(domain_found)
    print("utils.get_norm_L1(domain_found-wolf.domain)",
          utils.get_norm_L1(domain_found-wolf.domain))
    frontier_wolf = fro_gen.get_indices_frontier_robin(wolf.domain)
    print('distance:', utils.get_mean_point_wise_distance(
        frontier_found, frontier_wolf))

    domain_found = wolf.generate_new_location(
        1000, distance_max=4, distance_min_to_overcome=2)
    frontier_found = fro_gen.get_indices_frontier_robin(domain_found)
    print("utils.get_norm_L1(domain_found-wolf.domain)",
          utils.get_norm_L1(domain_found-wolf.domain))
    frontier_wolf = fro_gen.get_indices_frontier_robin(wolf.domain)
    print('distance:', utils.get_mean_point_wise_distance(
        frontier_found, frontier_wolf))


if __name__ == '__main__':
    dico_score_fn_name_to_fn = {"energy_minus": score_energy_minus}

    # test_WolfDomain()
    run_optimization_procedure()
