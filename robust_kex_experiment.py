"""
Fairness experiments, varying
- edge success probability
- chain cap
- fairness rule
"""
import argparse
import os
import kidney_ip
import numpy as np
import re
import random

from kidney_graph_io import get_UNOS_graphs, get_cmu_graphs
from utils import generate_filepath
from kidney_ip import OptConfig, optimise_robust_picef, optimize_picef


def robust_kex_experiment(args):
    # run an experiment with edge-weight robust kidney exchange, testing the method of Ren et al. (2019) and
    # McElfresh et al, (2018).

    output_dir = args.output_dir
    seed = args.seed
    input_dir = args.input_dir
    graph_type = args.graph_type
    verbose = args.verbose
    cycle_cap = args.cycle_cap
    chain_cap = args.chain_cap
    num_weight_measurements = args.num_weight_measurements
    num_weight_realizations = args.num_weight_realizations
    alpha_list = args.alpha_list
    theta_list = args.theta_list
    # protection_level = args.protection_level
    gamma_list = args.gamma_list
    num_trials = args.num_trials
    dist_type = args.dist_type

    rs = np.random.RandomState(seed=seed)

    # generate an output file
    output_file = generate_filepath(output_dir, 'robust_kex_experiment', 'csv')

    # list of output column names
    output_columns = ['graph_name',
                        'trial_num',
                        'alpha',
                        'realization_num',
                        'cycle_cap',
                        'chain_cap',
                        'method',
                        'parameter_name',
                        'parameter_value',
                        'realized_score']

    # output file header, and write experiment parameters to the first line
    with open(output_file, 'w') as f:
        f.write(str(args) + '\n')
        f.write((','.join(len(output_columns) * ['%15s ']) + '\n') % tuple(output_columns))

    if graph_type == 'UNOS':
        graph_generator = get_UNOS_graphs(input_dir)
    elif graph_type == 'CMU':
        graph_generator = get_cmu_graphs(input_dir)

    # run the experiment for each graph
    for digraph, ndd_list, graph_name in graph_generator:

        for i_trial in range(num_trials):

            for alpha in alpha_list:
                # initialize edge types, weight function, and edge_weight_list
                initialize_edge_weights(digraph, ndd_list, num_weight_measurements, alpha,
                                        rs=rs,
                                        dist_type=dist_type)

                # method 1: solve the non-robust approach with *true edge means*
                set_nominal_edge_weight(digraph, ndd_list)
                sol_nonrobust_truemean = optimize_picef(OptConfig(digraph, ndd_list, cycle_cap, chain_cap,
                                                         verbose=verbose))

                # method 1: solve the non-robust approach
                # for this method we treat the *mean* of all measurements as the true edge weights
                set_sample_mean_edge_weight(digraph, ndd_list)
                sol_nonrobust_samplemean = optimize_picef(OptConfig(digraph, ndd_list, cycle_cap, chain_cap,
                                                         verbose=verbose))

                # method 2: solve the RO approach of McElfresh et al. (2018), assuming a symmetric edge weight distribution
                # for this method, as with the non-robust method, we set the nominal value for each edge to be the mean of all
                # measurements. we also set the discount value to be the difference between the mean and the *min* measurement.
                # that is, we assume that we observe the minimum-possible edge weight in the measurements.
                #
                # we could improve this, by making a truthful assumption about the edge weight distribution.
                set_RO_weight_parameters(digraph, ndd_list)
                sol_RO_list = []
                for gamma in gamma_list:
                    sol_RO_list.append(optimise_robust_picef(OptConfig(digraph, ndd_list, cycle_cap, chain_cap,
                                                 verbose=verbose,
                                                 gamma=gamma)))

                # method 3: solve the DRO approach.
                sol_DRO_list = []
                for theta in theta_list:
                    sol_DRO_list.append(kidney_ip.optimize_DROinf_picef(OptConfig(digraph, ndd_list, cycle_cap, chain_cap,
                                                                verbose=verbose), theta=theta))

                # if verbose:
                #     print("protection level epsilon = %f" % protection_level)
                #     print("non-robust solution:")
                #     print(sol_nonrobust.display())
                #     print("edge-weight robust solution:")
                #     print(sol_RO.display())
                #     print("distributionally-robust solution:")
                #     print(sol_DRO.display())

                with open(output_file, 'a') as f:
                    for i_realization in range(num_weight_realizations):

                        # apply a realization to each edge
                        realize_edge_weights(digraph, ndd_list, rs=rs)

                        realized_nonrobust_truemean_score = sum([e.weight for e in sol_nonrobust_truemean.matching_edges])

                        write_line(f,
                                   graph_name,
                                   i_trial,
                                   alpha,
                                   i_realization,
                                   cycle_cap,
                                   chain_cap,
                                   'nonrobust_truemean',
                                   'None',
                                   0,
                                   realized_nonrobust_truemean_score)

                        realized_nonrobust_samplemean_score = sum([e.weight for e in sol_nonrobust_samplemean.matching_edges])

                        write_line(f,
                                   graph_name,
                                   i_trial,
                                   alpha,
                                   i_realization,
                                   cycle_cap,
                                   chain_cap,
                                   'nonrobust_samplemean',
                                   'None',
                                   0,
                                   realized_nonrobust_samplemean_score)

                        # find the optimal matching, given true weights
                        sol_opt = optimize_picef(OptConfig(digraph, ndd_list, cycle_cap, chain_cap,
                                                                 verbose=verbose))
                        realized_optimal_score = sum([e.weight for e in sol_opt.matching_edges])

                        write_line(f,
                                   graph_name,
                                   i_trial,
                                   alpha,
                                   i_realization,
                                   cycle_cap,
                                   chain_cap,
                                   'optimal',
                                   'None',
                                   0,
                                   realized_optimal_score)

                        for sol, gamma in zip(sol_RO_list, gamma_list):
                            score = sum([e.weight for e in sol.matching_edges])
                            write_line(f,
                                       graph_name,
                                       i_trial,
                                       alpha,
                                       i_realization,
                                       cycle_cap,
                                       chain_cap,
                                       'RO',
                                       'gamma',
                                       gamma,
                                       score)

                        for sol, theta in zip(sol_DRO_list, theta_list):
                            score = sum([e.weight for e in sol.matching_edges])
                            write_line(f,
                                       graph_name,
                                       i_trial,
                                       alpha,
                                       i_realization,
                                       cycle_cap,
                                       chain_cap,
                                       'DRO',
                                       'theta',
                                       theta,
                                       score)


def write_line(f,
               graph_name,
               i_trial,
               alpha,
               i_realization,
               cycle_cap,
               chain_cap,
               method,
               parameter_name,
               parameter_value,
               score):

    f.write((','.join(10 * ['%15s ']) + '\n') %
        (graph_name,
        '%d' % i_trial,
        '%.3e' % alpha,
        '%d' % i_realization,
        '%d' % cycle_cap,
        '%d' % chain_cap,
        method,
        parameter_name,
        '%.4e' % parameter_value,
        '%.4e' % score))

def initialize_edge_weights(digraph, ndd_list, num_weight_measurements, alpha,
                                rs=None,
                                dist_type='binary'):
    # initialize the "weight_type" of each edge, and the function draw_edge_weight
    #
    # set the following properties for each edge:
    # - type : (int) the type of edge weight distribution
    # - alpha : the fraction of edges that are random (type = 0 is deterministic)
    # - draw_edge_weight : (function handle) take a random state and return an edge weight
    # - edge_weight_list : a list of draws (measurements) from draw_edge_weight
    # - dist_type : the type of edge weight distribution ('binary' or 'unos')
    if rs is None:
        rs = np.random.RandomState(0)

    if dist_type == 'binary':
        initialize_edge = initialize_edge_binary
    elif dist_type == 'unos':
        initialize_edge = initialize_edge_unos
    else:
        raise Warning("edge distribution type not recognized")

    for e in digraph.es:
        initialize_edge(e, alpha, num_weight_measurements, rs=rs)
    for n in ndd_list:
        for e in n.edges:
            initialize_edge(e, alpha, num_weight_measurements, rs=rs)


def initialize_edge_binary(e, alpha, num_weight_measurements,
                    rs=None):
    if rs is None:
        rs = np.random.RandomState(0)

    # set a type : with probability alpha, the edge is random
    if rs.rand() < alpha:
        e.type = 1
    else:
        # deterministic edge
        e.type = 0

    e.draw_edge_weight = lambda x: edge_weight_distribution_binary(e.type, x)
    e.weight_list = [e.draw_edge_weight(rs) for _ in range(num_weight_measurements)]
    e.true_mean_weight = 0.5


def initialize_edge_unos(e, alpha, num_weight_measurements,
                    rs=None):
    if rs is None:
        rs = np.random.RandomState(0)

    # probabilities of meeting each criteria
    p_list = [1.0, # base points (100)
              0.005, # exact tissue type match (200)
              0.12, # highly sensitized (125)
              0.5, # at least one antibody mismatch (-5)
              0.01, # patient is <18 (100)
              0.001, # prior organ donor (150)
              0.5] # geographic proximity (0, 25, 50, 75)]

    # weights for each criteria
    w_list = [100, # base points (100)
              200, # exact tissue type match (200)
              125, # highly sensitized (125)
              -5, # at least one antibody mismatch (-5)
              100, # patient is <18 (100)
              150, # prior organ donor (150)
              75] # geographic proximity

    # max edge weight is 750

    # set a type : with probability alpha, the edge is random
    if rs.rand() < alpha:
        # probabilistic edge
        e.type = 1

        # for each criteria, draw an initial value; this will be equal to the deterministic edge weight
        _, b_realized = sample_edge_weight_distribution_unos(rs, w_list, p_list)

        # fix the bernoulli variables for the last three criteria (these should be certain)
        p_list_fixed = np.copy(p_list)
        p_list_fixed[4] = b_realized[4]
        p_list_fixed[5] = b_realized[5]
        p_list_fixed[6] = b_realized[6]

        e.draw_edge_weight = lambda x: sample_edge_weight_distribution_unos(x, w_list, p_list_fixed)[0]
        e.weight_list = [e.draw_edge_weight(rs) for _ in range(num_weight_measurements)]
        e.true_mean_weight = np.dot(p_list_fixed, w_list)

    else:
        # deterministic edge
        e.type = 0

        # for each criteria, draw an initial value; this will be equal to the deterministic edge weight
        fixed_weight, _ = sample_edge_weight_distribution_unos(rs, w_list, p_list)
        e.draw_edge_weight = lambda x: fixed_weight
        e.weight_list = [fixed_weight] * num_weight_measurements
        e.true_mean_weight = fixed_weight


def set_nominal_edge_weight(digraph, ndd_list):
    # for each edge, set the property edge.weight to be the mean of edge.weight_list
    # this is used for the *nominal* (non-robust) approach
    for e in digraph.es:
        e.weight = e.true_mean_weight
    for n in ndd_list:
        for e in n.edges:
            e.weight = e.true_mean_weight


def set_sample_mean_edge_weight(digraph, ndd_list):
    # for each edge, set the property edge.weight to be the mean of edge.weight_list
    # this is used for the *nominal* (non-robust) approach
    for e in digraph.es:
        e.weight = np.mean(e.weight_list)
    for n in ndd_list:
        for e in n.edges:
            e.weight = np.mean(e.weight_list)


def set_RO_weight_parameters(digraph, ndd_list):
    # for each edge, set e.weight =( max(e.weight_list) - min(e.weight_list)) / 2
    # set discount = weight - min(e.weight_list)
    for e in digraph.es:
        e.weight = np.max(e.weight_list) - np.min(e.weight_list)
        e.discount = e.weight - np.min(e.weight_list)
    for n in ndd_list:
        for e in n.edges:
            e.weight = np.max(e.weight_list) - np.min(e.weight_list)
            e.discount = e.weight - np.min(e.weight_list)



def realize_edge_weights(digraph, ndd_list, rs=None):
    # simulate a realization of each edge weight, by drawing from a distribution and setting the property edge.weight
    if rs is None:
        rs = np.random.RandomState(0)

    for e in digraph.es:
        e.weight = e.draw_edge_weight(rs)
    for n in ndd_list:
        for e in n.edges:
            e.weight = e.draw_edge_weight(rs)


# ---- edge weight distribution functions ----
# each function should generate a random list of edge weights for some edge
#
# shared arguments:
# - alpha \in [0,1] : the fraction of edges that are uncertain (1 - alpha)% of the edges are constant-weight
# - num_measurements : the number of edge measurements to provide
# all constant edges have

def edge_weight_distribution_binary(type, rs):
    # return an edge weight for a certain type of distribution
    #
    # type:
    # 0 : deterministic
    # not 0 : 0 or 1, with prob. 0.5 each
    if type == 0:
        return 0.5
    else:
        return rs.choice([0.0, 1.0], p=[0.5, 0.5], size=1)[0]

def sample_edge_weight_distribution_unos(rs, w_list, p_list):
    # return an edge weight for a unos-inspired distribution
    #
    # total edge distribution is:
    # w_e ~ \sum_c p_{c} w_c
    #
    # each p_{c} is a bernoulli r.v.
    b_realized = (rs.rand(len(p_list)) <= p_list).astype(int)
    return np.dot(w_list, b_realized), b_realized

# def edge_dist_binary(alpha, num_measurements,
#                      rs=None):
#     # return a list of constant edge weights (with prob. 1-alpha) or binary edge weights (with prob. alpha)
#     # constant edges have weight 0.5; binary edge weights have weight 0 or 1 (each with prob. 0.5)
#     #
#     # inputs:
#     # - alpha \in [0,1] : the fraction of edges that are uncertain (1 - alpha)% of the edges are constant-weight
#     # - num_measurements : the number of weight measurements to generate
#     # - rs : (optional) to seed all randomness
#     if rs is None:
#         rs = np.random.RandomState(0)
#
#     # generate random edge weights
#     if rs.rand() < alpha:
#         return rs.choice([0.0, 1.0], p=[0.5, 0.5], size=num_measurements)
#     else:
#         return np.array([0.5] * num_measurements)


def main():
    # run the experiment. sample usage:
    # >>> python robust_kex_experiment.py  --num-weight-measurements=5  --output-dir /Users/duncan/research/ --graph-type CMU --input-dir /Users/duncan/research/graphs/graphs_from_john/graphs_64
    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose',
                        action='store_true',
                        help='verbose output')
    parser.add_argument('--seed',
                        type=int,
                        help='random seed for experiments',
                        default=0)
    parser.add_argument('--alpha-list',
                        type=float,
                        help='a list of alpha values : the fraction of edges that are random.',
                        default=[0.5],
                        nargs='+')
    parser.add_argument('--num-trials',
                        type=int,
                        help='number of times to randomly assign edge distributions & measurements',
                        default=1)
    parser.add_argument('--num-weight-realizations',
                        type=int,
                        help='number of times to randomly draw edge weights and calculate matching score',
                        default=10)
    parser.add_argument('--num-weight-measurements',
                        type=int,
                        default=10,
                        help='number of weight measurements to use for all robust methods')
    parser.add_argument('--chain-cap',
                        type=int,
                        default=4,
                        help='chain cap')
    parser.add_argument('--cycle-cap',
                        type=int,
                        default=3,
                        help='cycle cap')
    parser.add_argument('--gamma-list',
                        type=float,
                        default=[1],
                        nargs="+",
                        help='list of gamma values (used only by edge-weight-robust')
    parser.add_argument('--theta-list',
                        type=float,
                        default=[0.1],
                        nargs="+",
                        help='list of theta values(used only by DRO')
    parser.add_argument('--graph-type',
                        type=str,
                        default='CMU',
                        choices=['CMU', 'UNOS'],
                        help='type of exchange graphs: CMU format or UNOS format')
    parser.add_argument('--dist-type',
                        type=str,
                        default='unos',
                        choices=['unos', 'binary'],
                        help='type of edge weight distribution: unos or binary')
    parser.add_argument('--input-dir',
                        type=str,
                        default=None,
                        help='input directory, containing exchange graph files (in either CMU or UNOS format)')
    parser.add_argument('--output-dir',
                        type=str,
                        default=None,
                        help='output directory, where an output csv will be written')

    args = parser.parse_args()

    # UNCOMMENT FOR TESTING ARGPARSE / DEBUGGING
    # arg_string = "--num-weight-measurements=3 --gamma-list 0 1 5 10 15 --theta-list 0.1 10 100 500 600 --alpha-list 0.9 --output-dir /Users/duncan/research/DistRobustKex_output --graph-type CMU --input-dir /Users/duncan/research/example_graphs"
    # args = parser.parse_args(arg_string.split())

    robust_kex_experiment(args)


if __name__ == "__main__":
    main()
