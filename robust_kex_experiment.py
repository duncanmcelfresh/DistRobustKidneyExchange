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
from kidney_ip import OptConfig, solve_edge_weight_uncertainty, optimize_picef


def robust_kex_experiment(args):
    # run an experiment with edge-weight robust kidney exchange, testing the method of Ren et al. (2019) and
    # McElfresh et al, (2018).

    output_dir = args.output_dir
    seed = args.seed
    input_dir = args.input_dir
    graph_type = args.graph_type
    verbose = args.verbose
    protection_level = args.protection_level
    cycle_cap = args.cycle_cap
    chain_cap = args.chain_cap
    num_weight_measurements = args.num_weight_measurements
    num_weight_realizations = args.num_weight_realizations

    rs = np.random.RandomState(seed=seed)

    # generate an output file
    output_file = generate_filepath(output_dir, 'robust_kex_experiment', 'csv')

    # list of output column names
    output_columns = ['graph_name',
                        'realization_num',
                        'cycle_cap',
                        'chain_cap',
                        'protection_level',
                        'nonrobust_score',
                        'edge_weight_robust_score']

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

        # simulate measurements of edge weight, to be used by each method
        simulate_edge_measurements(digraph, ndd_list,
                                   rs=rs,
                                   num_weight_measurements=num_weight_measurements)

        # method 1: solve the non-robust approach
        # for this method we treat the *mean* of all measurements as the true edge weights
        set_mean_edge_weight(digraph, ndd_list)
        sol_nonrobust = optimize_picef(OptConfig(digraph, ndd_list, cycle_cap, chain_cap,
                                                 verbose=verbose))

        # method 2: solve the RO approach of McElfresh et al. (2018), assuming a symmetric edge weight distribution
        # for this method, as with the non-robust method, we set the nominal value for each edge to be the mean of all
        # measurements. we also set the discount value to be the difference between the mean and the *min* measurement.
        # that is, we assume that we observe the minimum-possible edge weight in the measurements.
        #
        # we could improve this, by making a truthful assumption about the edge weight distribution.
        set_mean_edge_weight(digraph, ndd_list)
        set_discount_values(digraph, ndd_list)
        sol_weight_robust = solve_edge_weight_uncertainty(OptConfig(digraph, ndd_list, cycle_cap, chain_cap,
                                     verbose=verbose,
                                     protection_level=protection_level))


        if verbose:
            print("protection level epsilon = %f" % protection_level)
            print("non-robust solution:")
            print(sol_nonrobust.display())
            print("edge-weight robust solution:")
            print(sol_weight_robust.display())


        with open(output_file, 'a') as f:
            for i_realization in range(num_weight_realizations):

                # apply a realization to each edge
                realize_edge_weights(digraph, ndd_list,
                                     rs=None)

                realized_nonrobust_score = sum([e.weight for e in sol_nonrobust.matching_edges])
                realized_edge_weight_robust_score = sum([e.weight for e in sol_weight_robust.matching_edges])

                f.write((','.join(len(output_columns) * ['%15s ']) + '\n') %
                    (graph_name,
                    '%d' % i_realization,
                    '%d' % cycle_cap,
                    '%d' % chain_cap,
                    '%.2e' % protection_level,
                    '%.3e' % realized_nonrobust_score,
                    '%.3e' % realized_edge_weight_robust_score))


def simulate_edge_measurements(digraph, ndd_list,
                                rs=None,
                                num_weight_measurements=10):
    # sample num_measurements measurements for each edge. put these in a list, and set it to the property
    # edge.weight_list
    if rs is None:
        rs = np.random.RandomState(0)

    # for this, use a simple weight distribution: random integer between 1 and 10.
    generate_weight_list = lambda: rs.randint(1, 10, size=num_weight_measurements)

    for e in digraph.es:
        e.weight_list = generate_weight_list()
    for n in ndd_list:
        for e in n.edges:
            e.weight_list = generate_weight_list()


def set_mean_edge_weight(digraph, ndd_list):
    # for each edge, set the property edge.weight to be the mean of edge.weight_list
    # this is used for the *nominal* (non-robust) approach
    for e in digraph.es:
        e.weight = np.mean(e.weight_list)
    for n in ndd_list:
        for e in n.edges:
            e.weight = np.mean(e.weight_list)


def set_discount_values(digraph, ndd_list):
    # for each edge, set the property edge.discount to be  mean(edge.weight_list) - min(edge.weight_list)
    # this is used for the edge-weight robust approach (McElfresh et al. 2018)
    for e in digraph.es:
        e.discount = np.mean(e.weight_list) - np.min(e.weight_list)
    for n in ndd_list:
        for e in n.edges:
            e.discount = np.mean(e.weight_list) - np.min(e.weight_list)


def realized_weight_function(edge, rs=None):
    # draw from a distribution of true edge weights, and it to e.weight
    if rs is None:
        rs = np.random.RandomState(0)

    return rs.randint(1, 10)


def realize_edge_weights(digraph, ndd_list, rs=None):
    # simulate a realization of each edge weight, by drawing from a distribution and setting the property edge.weight
    if rs is None:
        rs = np.random.RandomState(0)

    for e in digraph.es:
        e.weight = realized_weight_function(e, rs=rs)
    for n in ndd_list:
        for e in n.edges:
            e.weight = realized_weight_function(e, rs=rs)


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
    parser.add_argument('--protection-level',
                        type=int,
                        default=0.1,
                        help='protection level (used only by edge-weight-robust')
    parser.add_argument('--graph-type',
                        type=str,
                        default='CMU',
                        choices=['CMU', 'UNOS'],
                        help='type of exchange graphs: CMU format or UNOS format')
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
    # arg_string = "--num-weight-measurements=5  --output-dir /Users/duncan/research/ --graph-type CMU --input-dir /Users/duncan/research/graphs/graphs_from_john/graphs_64"
    # args = parser.parse_args(arg_string.split())

    robust_kex_experiment(args)


if __name__ == "__main__":
    main()
