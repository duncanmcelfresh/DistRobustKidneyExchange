# function for running experiments with different robust kidney exchange formulations

import argparse
import numpy as np

from edge_weight_distributions import initialize_edge_weights
from kidney_graph_io import get_unos_graphs, get_cmu_graphs
from utils import generate_filepath
from kidney_ip import OptConfig, optimise_robust_picef, optimize_picef, optimize_SAA_picef


def robust_kex_experiment(args):

    rs = np.random.RandomState(seed=args.seed)

    # generate an output file
    output_file = generate_filepath(args.output_dir, 'robust_kex_experiment', 'csv')

    # list of output column names
    output_columns = ['graph_name',
                      'trial_num',
                      'alpha',
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

        for i_trial in range(args.num_trials):

            for alpha in args.alpha_list:

                # store all solutions (matchings) in a dict
                sol_dict = {}

                # initialize edge types, weight function, and edge_weight_list
                initialize_edge_weights(digraph, ndd_list, args.num_weight_measurements, alpha, rs, args.dist_type)

                # method 1: solve the non-robust approach with *true edge means*
                set_nominal_edge_weight(digraph, ndd_list)
                sol_dict['nonrobust_truemean'] = optimize_picef(OptConfig(digraph, ndd_list, args.cycle_cap,
                                                                          args.chain_cap,
                                                                          verbose=args.verbose))

                # method 1a: solve the non-robust approach with *sample edge means*
                set_sample_mean_edge_weight(digraph, ndd_list)
                sol_dict['nonrobust_samplemean'] = optimize_picef(OptConfig(digraph, ndd_list, args.cycle_cap,
                                                                            args.chain_cap,
                                                                            verbose=args.verbose))

                # method 2: solve the RO approach of McElfresh et al. (2018)
                set_RO_weight_parameters(digraph, ndd_list)
                for gamma in args.gamma_list:
                    sol_dict[('ro_gamma_%s' % gamma)] = optimise_robust_picef(
                        OptConfig(digraph, ndd_list, args.cycle_cap, args.chain_cap,
                                  verbose=args.verbose,
                                  gamma=gamma))

                # method 3: solve the DRO approach with SAA
                for ssa_gamma in args.ssa_gamma_list:
                    for ssa_alpha in args.ssa_alpha_list:
                        sol_dict[('ssa_gamma_%s_alpha_%s' % (ssa_gamma, ssa_alpha))] = optimize_SAA_picef(
                            OptConfig(digraph, ndd_list, args.cycle_cap, args.chain_cap,
                                      verbose=args.verbose), args.num_weight_measurements, ssa_gamma, ssa_alpha)
                # # method 3: solve the DRO approach.
                # set_sample_mean_edge_weight(digraph, ndd_list)
                # for theta in theta_list:
                #     sol_dict[('dro_theta_%s' % theta )] = \
                    #         kidney_ip.optimize_DROinf_picef(OptConfig(digraph, ndd_list, cycle_cap, chain_cap,
                #                                                   verbose=verbose), theta=theta)

                # if verbose:
                #     print("protection level epsilon = %f" % protection_level)
                #     print("non-robust solution:")
                #     print(sol_nonrobust.display())
                #     print("edge-weight robust solution:")
                #     print(sol_RO.display())
                #     print("distributionally-robust solution:")
                #     print(sol_DRO.display())

                with open(output_file, 'a') as f:
                    for i_realization in range(args.num_weight_realizations):

                        # apply a realization to each edge
                        realize_edge_weights(digraph, ndd_list, rs=rs)

                        # solve for the (omniscient) optimal edge weight
                        sol_dict['omniscient'] = optimize_picef(OptConfig(digraph,
                                                                          ndd_list,
                                                                          args.cycle_cap,
                                                                          args.chain_cap,
                                                                          verbose=args.verbose))

                        for sol_name, (sol, matched_edges) in sol_dict.items():
                            score = sum([e.weight for e in matched_edges])

                            f.write((','.join(8 * ['%s']) + '\n') %
                                    (graph_name,
                                     '%d' % i_trial,
                                     '%.3e' % alpha,
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


def realize_edge_weights(digraph, ndd_list, rs):
    """simulate a realization of each edge weight, by drawing from a distribution and setting the property edge.weight

    Args:
        digraph: (kidney_digraph.Graph).
        ndd_list: (list(kidney_ndds.Ndd)).
        rs: (numpy.random.RandomState).
    """

    for e in digraph.es:
        e.weight = e.draw_edge_weight(rs)
    for n in ndd_list:
        for e in n.edges:
            e.weight = e.draw_edge_weight(rs)


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
    parser.add_argument('--ssa-alpha-list',
                        type=float,
                        default=[0.01, 0.05, 0.1],
                        nargs="+",
                        help='list of alpha values (used only by ssa-robust')
    parser.add_argument('--ssa-gamma-list',
                        type=float,
                        default=[0.3, 0.5, 0.7],
                        nargs="+",
                        help='list of gamma values (used only by ssa-robust')
    parser.add_argument('--graph-type',
                        type=str,
                        default='cmu',
                        choices=['cmu', 'unos'],
                        help='type of exchange graphs: cmu format or unos format')
    parser.add_argument('--dist-type',
                        type=str,
                        default='unos',
                        choices=['unos', 'binary', 'dro'],
                        help='type of edge weight distribution')
    parser.add_argument('--input-dir',
                        type=str,
                        default=None,
                        help='input directory, containing exchange graph files (in either cmu or unos format)')
    parser.add_argument('--output-dir',
                        type=str,
                        default=None,
                        help='output directory, where an output csv will be written')
    parser.add_argument('--DEBUG',
                        action='store_true',
                        help='if set, use a fixed arg string for debugging. otherwise, parse args.',
                        default=False)

    args = parser.parse_args()

    if args.DEBUG:
        # fixed set of parameters, for debugging:
        arg_str = '--num-weight-measurements 3'
        arg_str += ' --dist-type unos'
        arg_str += ' --alpha-list 1.0'
        arg_str += ' --num-weight-realizations 3'
        arg_str += ' --ssa-alpha-list 0.5'
        arg_str += ' --ssa-gamma-list 5'
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
