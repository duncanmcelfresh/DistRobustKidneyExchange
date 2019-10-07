# function for running experiments with different robust kidney exchange formulations

import argparse
import numpy as np

from edge_weight_distributions import initialize_edge_weights
from kidney_graph_io import get_unos_graphs, get_cmu_graphs
from utils import generate_filepath
from kidney_ip import OptConfig, optimise_robust_picef, optimize_picef, optimize_SAA_picef, optimize_DRO_SAA_picef


def robust_kex_experiment(args):
    rs = np.random.RandomState(seed=args.seed)

    if not any([args.use_truemean, args.use_samplemean, args.use_saa, args.use_ro]):
        raise Exception("at least one matching method must be used")

    # generate an output file
    output_file = generate_filepath(args.output_dir, 'robust_kex_experiment', 'csv')

    # list of output column names
    output_columns = ['graph_name',
                      'trial_num',
                      'alpha',
                      'noise_scale',
                      'realization_num',
                      'cycle_cap',
                      'chain_cap',
                      'method',
                      'realized_score']

    # output file header, and write experiment parameters to the first line
    with open(output_file, 'w') as f:
        f.write(str(args) + '\n')
        f.write((','.join(len(output_columns) * ['%s']) + '\n') % tuple(output_columns))

    if args.graph_type == 'unos':
        graph_generator = get_unos_graphs(args.input_dir)
    elif args.graph_type == 'cmu':
        graph_generator = get_cmu_graphs(args.input_dir)

    # run the experiment for each graph
    for digraph, ndd_list, graph_name in graph_generator:

        print("running tests for graph: %s" % graph_name)

        for i_trial in range(args.num_trials):

            for alpha in args.alpha_list:

                # store all solutions (matchings) in a dict
                sol_dict = {}

                # initialize edge types, weight function, and edge_weight_list
                initialize_edge_weights(digraph, ndd_list, args.num_weight_measurements, alpha, rs, args.dist_type)

                # method 1: solve the non-robust approach with *true edge means*
                set_nominal_edge_weight(digraph, ndd_list)
                if args.use_truemean:
                    sol_dict['nonrobust_truemean'] = optimize_picef(OptConfig(digraph, ndd_list, args.cycle_cap,
                                                                              args.chain_cap,
                                                                              verbose=args.verbose))

                # method 1a: solve the non-robust approach with *sample edge means*
                if args.use_samplemean:
                    set_sample_mean_edge_weight(digraph, ndd_list)
                    sol_dict['nonrobust_samplemean'] = optimize_picef(OptConfig(digraph, ndd_list, args.cycle_cap,
                                                                                args.chain_cap,
                                                                                verbose=args.verbose))

                # method 2: solve the RO approach of McElfresh et al. (2018)
                if args.use_ro:
                    set_RO_weight_parameters(digraph, ndd_list)
                    for gamma in args.gamma_list:
                        sol_dict[('ro_gamma_%s' % str(gamma))] = optimise_robust_picef(
                            OptConfig(digraph, ndd_list, args.cycle_cap, args.chain_cap,
                                      verbose=args.verbose,
                                      gamma=gamma))

                # method 3: solve the SAA-CVar approach
                if args.use_saa:
                    for saa_gamma in args.saa_gamma_list:
                        for saa_alpha in args.saa_alpha_list:
                            sol_dict[('saa_gamma_%s_alpha_%s' % (str(saa_gamma), str(saa_alpha)))] = optimize_SAA_picef(
                                OptConfig(digraph, ndd_list, args.cycle_cap, args.chain_cap,
                                          verbose=args.verbose), args.num_weight_measurements, saa_gamma, saa_alpha)

                # method 4: solve the DRO approach
                if args.use_dro_saa:
                    for theta in args.dro_theta_list:
                        pair_e_list = digraph.es
                        ndd_e_list = []
                        for n in ndd_list:
                            ndd_e_list.extend(n.edges)
                        weights = [wt for e in pair_e_list for wt in e.weight_list] + [wt for e in ndd_e_list for wt in
                                                                                       e.weight_list]
                        w_min = min(weights)
                        w_max = max(weights)
                        for saa_gamma in args.saa_gamma_list:
                            for saa_alpha in args.saa_alpha_list:
                                sol_dict[('dro_saa_gamma_%s_alpha_%s_theta_%s' % (
                                str(saa_gamma), str(saa_alpha), str(theta)))] = \
                                    optimize_DRO_SAA_picef(OptConfig(digraph, ndd_list, args.cycle_cap, args.chain_cap,
                                                                     verbose=args.verbose),
                                                           args.num_weight_measurements, saa_gamma, saa_alpha, theta,
                                                           w_min,
                                                           w_max)

                with open(output_file, 'a') as f:
                    for i_realization in range(args.num_weight_realizations):

                        for noise_scale in args.noise_scale_list:
                            # apply a realization to each edge
                            realize_edge_weights(digraph, ndd_list, rs, noise_scale)

                            if args.use_omniscient:
                                # solve for the (omniscient) optimal edge weight
                                sol_dict['omniscient'] = optimize_picef(OptConfig(digraph,
                                                                                  ndd_list,
                                                                                  args.cycle_cap,
                                                                                  args.chain_cap,
                                                                                  verbose=args.verbose))

                            for sol_name, (sol, matched_edges) in sol_dict.items():
                                score = sum([e.weight for e in matched_edges])

                                f.write((','.join(len(output_columns) * ['%s']) + '\n') %
                                        (graph_name,
                                         '%d' % i_trial,
                                         '%.3e' % alpha,
                                         '%.3e' % noise_scale,
                                         '%d' % i_realization,
                                         '%d' % args.cycle_cap,
                                         '%d' % args.chain_cap,
                                         sol_name,
                                         '%.4e' % score))


def set_nominal_edge_weight(digraph, ndd_list):
    """for each edge in the digraph, and for all ndd edges for each ndd, set the property edge.weight
     to be the property edge.true_mean_weight

     Args:
         digraph: (kidney_digraph.Graph).
         ndd_list: (list(kidney_ndds.Ndd)).
     """
    for e in digraph.es:
        e.weight = e.true_mean_weight
    for n in ndd_list:
        for e in n.edges:
            e.weight = e.true_mean_weight


def set_sample_mean_edge_weight(digraph, ndd_list):
    """for each edge in the digraph, and for all ndd edges for each ndd, set the property edge.weight
    to be the mean of edge.weight_list

    Args:
        digraph: (kidney_digraph.Graph).
        ndd_list: (list(kidney_ndds.Ndd)).
    """
    for e in digraph.es:
        e.weight = np.mean(e.weight_list)
    for n in ndd_list:
        for e in n.edges:
            e.weight = np.mean(e.weight_list)


def set_RO_weight_parameters(digraph, ndd_list):
    """set the edge weight for the robust optimization approach.

    for each edge, set e.weight = ( max(e.weight_list) + min(e.weight_list)) / 2
    set discount = weight - min(e.weight_list)

    Args:
        digraph: (kidney_digraph.Graph).
        ndd_list: (list(kidney_ndds.Ndd)).
    """
    for e in digraph.es:
        e.weight = (np.max(e.weight_list) + np.min(e.weight_list)) / 2
        e.discount = e.weight - np.min(e.weight_list)
    for n in ndd_list:
        for e in n.edges:
            e.weight = (np.max(e.weight_list) + np.min(e.weight_list)) / 2
            e.discount = e.weight - np.min(e.weight_list)


def realize_edge_weights(digraph, ndd_list, rs, noise_scale=0.0):
    """simulate a realization of each edge weight, by drawing from a distribution and setting the property edge.weight

    Args:
        digraph: (kidney_digraph.Graph).
        ndd_list: (list(kidney_ndds.Ndd)).
        rs: (numpy.random.RandomState).
        noise_scale: (float) if >0, Gaussian noise with mean 0 and variance noise_scale * E[edge weight] is added
            to each realization. must be >= 0. noise is only added to stochastic edges. deterministic edges always have
            zero noise.
    """

    # randomly assign edges to receive positive or negative weight noise
    noise_magnitude = 10.0
    for e in digraph.es:
        if rs.rand() < 0.5:
            noise = noise_magnitude
        else:
            noise = - noise_magnitude
        e.weight = e.draw_edge_weight(rs) + noise
    for n in ndd_list:
        for e in n.edges:
            if rs.rand() < 0.5:
                noise = noise_magnitude
            else:
                noise = - noise_magnitude
            e.weight = e.draw_edge_weight(rs) + noise

    # this is old...
    # if noise_scale > 0:
    #
    #     # # draw noise separately for each donor node
    #     # for node in digraph.vs + ndd_list:
    #     #     if node.type == 0:
    #     #         # deterministic edge (no noise)
    #     #         node.weight = node.draw_edge_weight(rs)
    #     #     elif node.type == 1:
    #     #         # stochastic edge (add normal noise)
    #     #         node.weight = max(node.draw_edge_weight(rs) + rs.normal(0.0, noise_scale * node.true_mean_weight), 0.0)
    #     #
    #     # for e in digraph.es:
    #     #     e.weight = e.src.weight
    #     # for n in ndd_list:
    #     #     for e in n.edges:
    #     #         e.weight = n.weight
    #
    #     # # below is used only if edges are independent
    #     # noise = lambda e: 0 if e.type == 0 else rs.normal(0.0, noise_scale * np.mean(e.weight_list))
    #     #
    #     # for e in digraph.es:
    #     #     e.weight = e.draw_edge_weight(rs) + noise(e)
    #     # for n in ndd_list:
    #     #     for e in n.edges:
    #     #         e.weight = e.draw_edge_weight(rs) + noise(e)
    # else:
    #     for e in digraph.es:
    #         e.weight = e.draw_edge_weight(rs)
    #     for n in ndd_list:
    #         for e in n.edges:
    #             e.weight = e.draw_edge_weight(rs)

def main():
    # run the experiment. sample usage:
    # >>> python robust_kex_experiment.py  --num-weight-measurements=5  --output-dir /Users/duncan/research/ --graph-type cmu --input-dir /Users/duncan/research/graphs/graphs_from_john/graphs_64
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
                        help='a list of alpha values : the fraction of edges that are random (for weight type = binary or unos) or the number of probabilistic edges (if weight_type = dro).',
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
    parser.add_argument('--saa-alpha-list',
                        type=float,
                        default=[0.01, 0.05, 0.1],
                        nargs="+",
                        help='list of alpha values (used only by saa-robust')
    parser.add_argument('--saa-gamma-list',
                        type=float,
                        default=[0.3, 0.5, 0.7],
                        nargs="+",
                        help='list of gamma values (used only by saa-robust')
    parser.add_argument('--dro-theta-list',
                        type=float,
                        default=[1.0],
                        nargs="+",
                        help='list of theta values (used only by dro-robust')
    parser.add_argument('--graph-type',
                        type=str,
                        default='cmu',
                        choices=['cmu', 'unos'],
                        help='type of exchange graphs: cmu format or unos format')
    parser.add_argument('--dist-type',
                        type=str,
                        default='unos',
                        choices=['unos', 'binary', 'lkdpi'],
                        help='type of edge weight distribution')
    parser.add_argument('--input-dir',
                        type=str,
                        default=None,
                        help='input directory, containing exchange graph files (in either cmu or unos format)')
    parser.add_argument('--output-dir',
                        type=str,
                        default=None,
                        help='output directory, where an output csv will be written')
    parser.add_argument('--use-omniscient',
                        action='store_true',
                        help='if set, calculate the omniscient max-weight matching for each realization.',
                        default=False)
    parser.add_argument('--use-ro',
                        action='store_true',
                        help='if set, calculate the RO optimal matching (McElfresh 2019).',
                        default=False)
    parser.add_argument('--use-saa',
                        action='store_true',
                        help='if set, calculate the SAA-CVar optimal matching (Ren 2020).',
                        default=False)
    parser.add_argument('--use-dro-saa',
                        action='store_true',
                        help='if set, calculate the DRO-SAA-CVar optimal matching (Ren 2020).',
                        default=False)
    parser.add_argument('--use-samplemean',
                        action='store_true',
                        help='if set, use the non-robust with the true mean edge weight.',
                        default=False)
    parser.add_argument('--use-truemean',
                        action='store_true',
                        help='if set, use the non-robust with true mean edge weight.',
                        default=False)
    parser.add_argument('--noise-scale-list',
                        type=float,
                        nargs="+",
                        default=[0.0],
                        help='amount of noise to add to the realizations, on [0, 1].')
    parser.add_argument('--DEBUG',
                        action='store_true',
                        help='if set, use a fixed arg string for debugging. otherwise, parse args.',
                        default=False)

    args = parser.parse_args()

    if args.DEBUG:
        # fixed set of parameters, for debugging:
        arg_str = '--num-weight-measurements 3'
        arg_str += ' --use-samplemean'
        arg_str += ' --num-trials 1'
        arg_str += ' --use-dro-saa'
        arg_str += ' --noise-scale-list 0.0'  # 0.0 0.1 0.5 0.7'
        arg_str += ' --use-saa'
        arg_str += ' --use-ro'
        arg_str += ' --dist-type lkdpi'
        arg_str += ' --alpha-list 0.5'
        arg_str += ' --num-weight-realizations 1000'
        arg_str += ' --saa-alpha-list 0.5'
        arg_str += ' --dro-theta-list 0.01'  # 0.01 0.1 1.0 10 100 1000 10000'
        arg_str += ' --saa-gamma-list 10'
        arg_str += ' --gamma-list 5'
        arg_str += ' --output-dir /Users/duncan/research/DistRobustKidneyExchange_output/debug'
        arg_str += ' --graph-type cmu'
        arg_str += ' --input-dir /Users/duncan/research/graphs/graphs_from_john/graphs_64'

        args_fixed = parser.parse_args(arg_str.split())
        robust_kex_experiment(args_fixed)

    else:
        args = parser.parse_args()
        robust_kex_experiment(args)


if __name__ == "__main__":
    main()
