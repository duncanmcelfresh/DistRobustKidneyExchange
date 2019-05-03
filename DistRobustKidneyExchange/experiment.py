"""
Fairness experiments, varying
- edge success probability
- chain cap
- fairness rule
"""

import os

import kidney_digraph
import kidney_ip
import kidney_utils
import kidney_ndds
from read_CMU_format import read_CMU_format

import glob
import numpy as np
import time
import re
import random

def solve_kep(cfg, formulation, use_relabelled=True):
    formulations = {
        "picef": ("PICEF", kidney_ip.optimise_picef),
        "pctsp": ("PC-TSP", kidney_ip.optimise_pctsp),
        "pitsp": ("PI-TSP", kidney_ip.optimize_pitsp),
        "rob_picef": ("ROBUST PICEF", kidney_ip.optimise_robust_picef),
        "edge_weight_uncertainty": ("var. budget edge weight uncertainty", kidney_ip.solve_edge_weight_uncertainty),
        "rob_pctsp": ("ROBUST PC-TSP", kidney_ip.optimize_robust_pctsp)
    }

    if formulation in formulations:
        formulation_name, formulation_fun = formulations[formulation]
        if use_relabelled:
            opt_result = kidney_ip.optimise_relabelled(formulation_fun, cfg)
        else:
            opt_result = formulation_fun(cfg)

        kidney_utils.check_validity(opt_result, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain)
        opt_result.formulation_name = formulation_name
        return opt_result
    else:
        raise ValueError("Unrecognised IP formulation name")


def start():

    # experiment_type should be  'wt' (edge weight uncertainty), 'ex' (edge existence uncertainty), "gamma" (constant gamma)
    experiment_type = 'wt'

    # output folder for results file
    out_base = './results/'

    # folder to place matchings
    matchings_base = './matchings/'

    # verbosity
    verbose = 2

    # run experiment? (create output csv)
    run_experiment = True

    # get current time for output file
    timestr = time.strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(out_base,'kpd_'+experiment_type+"_"+timestr+".csv")

    # print header to file
    if experiment_type == 'ex':
        print_header_existence(outfile)
    elif experiment_type == 'wt':
        print_header_weight(outfile)
    elif experiment_type == 'gamma':
        print_header_constant_gamma(outfile)

    input_dir = './data'

    maxcard_filename = glob.glob(input_dir+os.sep+'*maxcard.input')[0]
    file_base = '_'.join(maxcard_filename.split('_')[:-1])
    name = maxcard_filename.split('/')[-1]

    details_files = glob.glob(file_base + '_*details.input')

    if len(details_files)>0:

        details_filename = details_files[0]
        d, altruists = read_CMU_format(details_filename, maxcard_filename)

        experiment_func(matchings_base, outfile, d, altruists, name, experiment_type,
                        verbose=verbose,
                        run_experiment=run_experiment)
    else:
        print("could not find *details.input file for: {}\n".format(maxcard_filename))

def experiment_func(matchings_base, outfile, d, a, dirname, experiment_type,
                    verbose = 0,
                    cycle_file = None,
                    remove_subtours=False,
                    run_experiment=True):

    cycle_cap = 3
    chain_cap_list =  [10, 3, 0]
    N_trials = 100

    # constant gamma

    if experiment_type=='ex':

        protection_levels_ex = [1e-1, 0.5, 1 - 2 ** -2, 1 - 2 ** -3]
        edge_success_prob_list = [0.3, 0.5, 0.7]  # for edge existence only

        find_edge_existence_matchings(matchings_base, d, a, dirname,
                                      verbose=verbose,
                                      cycle_cap=cycle_cap,
                                      chain_cap_list=chain_cap_list,
                                      edge_success_prob_list=edge_success_prob_list,
                                      protection_levels=protection_levels_ex,
                                      N_trials=N_trials,
                                      remove_subtours=remove_subtours,
                                      cycle_file=cycle_file,
                                      run_experiment=run_experiment,
                                      outfile=outfile)
    elif experiment_type == 'wt':

        protection_levels_weight = [1e-4, 1e-3, 1e-2, 1e-1, 0.5]
        alpha_list = [0.3, 0.5, 0.9]  # for edge weight only

        find_edge_weight_matchings(matchings_base, d, a, dirname, verbose=verbose, chain_cap_list=chain_cap_list,
                                   dist='flat', protection_levels=protection_levels_weight, alpha_list=alpha_list,
                                   cycle_file=cycle_file,
                                   outfile=outfile,
                                   run_experiment=run_experiment,
                                   N_trials=N_trials)

    if experiment_type=='gamma':

        gamma_list = [1, 2, 3, 4, 5]

        find_edge_existence_matchings_const_gamma(matchings_base, d, a, dirname, verbose=verbose, chain_cap_list=chain_cap_list,
                                                  gamma_list=gamma_list,
                                                  N_trials=N_trials,
                                                  run_experiment=run_experiment,
                                                  outfile=outfile)

def build_filename(name, type, ch_cap,
                   wt_string=None,
                   edge_success_prob=None,
                   protection_level=None,
                   gamma=None):
    '''
    make a filename for an optimization run, of the following format:
    name = exchange filename
    type = 'ex' | 'wt' | 'nr' | 'fa' (existence, weight, non-robust, failure-aware)
    ch_cap = chain cap
    wt_str = string indicating edge weight distribution (only for 'wt')
    edge_success_prob = edge success probability (only for 'fa', 'ex')
    protection_level = protection level (only for 'ex', 'wt')

    non-robust:
    NAME_nr_ch00.sol

    failure-aware:
    NAME_fa_ch00_p03.sol

    edge-weight uncertainty:
    NAME_wt_ch00_a01_ep0001.sol

    ex. uncertainty:
    NAME_ex_ch00_p05_ep05.sol
    '''

    ch_string = 'ch{:02d}'.format(ch_cap)

    if type == 'nr':
        return "_".join([name,type,ch_string]) + ".sol"

    if type == 'fa':
        if edge_success_prob is None:
            raise Warning("edge_success_prob not given")
        p_string = re.sub('\\.','','p{:g}'.format(edge_success_prob))
        return "_".join([name,type,ch_string,p_string]) + ".sol"

    if type == 'wt':
        if wt_string is None:
            raise Warning("wt_string not given")
        if protection_level is None:
            raise Warning("protection_level not given")
        eps_string = re.sub('\\.','','ep{:g}'.format(protection_level))
        return "_".join([name,type,ch_string,wt_string,eps_string]) + ".sol"

    if type == 'ex':
        if edge_success_prob is None:
            raise Warning("edge_success_prob not given")
        if protection_level is None:
            raise Warning("protection_level not given")
        p_string = re.sub('\\.','','p{:g}'.format(edge_success_prob))
        eps_string = re.sub('\\.','','ep{:g}'.format(protection_level))
        return "_".join([name,type,ch_string,p_string,eps_string]) + ".sol"

    if type == 'gamma':
        if gamma is None:
            raise Warning("gamma not given")
        gam_string = re.sub('\\.','','g{:g}'.format(gamma))
        return "_".join([name,type,ch_string,gam_string]) + ".sol"


def assign_flat_edges(seed, alpha, d, a):
    # start random state
    edge_assigner = np.random.RandomState(seed)
    # some edge weights are 0.5, others are either 0 or 1 (0.5 in expectation)
    low_frac = 0.0
    high_frac = 2.0
    # assign some edges to be constant (0.5), and others to be either 0 or 1
    # weight_type 0 = constant
    # weight_type 1 = bimodal
    # alpha = fraction of bimodal edges
    discount_func = lambda edge: 0.0 if edge.weight_type == 0 else 0.5

    for e in d.es:
        e.weight_type = edge_assigner.choice([0, 1], p=[1.0 - alpha, alpha])
        e.discount = discount_func(e)
        e.score = 0.5
    for n in a:
        for e in n.edges:
            e.weight_type = edge_assigner.choice([0, 1], p=[1.0 - alpha, alpha])
            e.discount = discount_func(e)
            e.score = 0.5


def find_edge_weight_matchings(matchings_dir, d, a, name,
                               verbose=0,
                               cycle_cap=3,
                               chain_cap_list=[3],
                               dist='uniform',
                               protection_levels=[0.01],
                               alpha_list=[0.5],
                               cycle_file=None,
                               run_experiment=False,
                               N_trials=None,
                               outfile=None):
    timelimit = None
    edge_success_prob = 1

    for alpha in alpha_list:

        # weight distribution:
        wt_string = dist + re.sub('\\.', '', '{:g}'.format(alpha))

        if dist == 'uniform':
            # edge weights are uniformly distributed on [w(1-a),w(1+a)], where a is between 0 and 1
            low_frac = 1.0 - alpha
            high_frac = 1.0 + alpha
            discount_func = lambda edge: edge.score * alpha
            realized_weight_func = lambda edge: np.random.uniform(low=edge.score*low_frac, high=edge.score*high_frac)

        elif dist == 'bimodal':
            # edge weights are either low or high
            low_frac = 1.0 - alpha
            high_frac = 1.0 + alpha
            discount_func = lambda edge: edge.score * alpha
            realized_weight_func = lambda edge: np.random.choice([edge.score * low_frac, edge.score * high_frac])

        elif dist == 'flat':
            # save state for re-generating edge states
            edge_assign_seed = random.randint(0, 2**31)
            # assign some edges to be constant (0.5), and others to be either 0 or 1
            # sets edge discount and weight_type
            assign_flat_edges(edge_assign_seed,alpha,d,a)

            # weight_type 0 = constant
            # weight_type 1 = bimodal

            # realized edge weights will be added later
            realized_weight_func = lambda edge: 0.5 if edge.weight_type==0 else np.random.choice([0.0,1.0])

        for chain_cap in chain_cap_list:

            type = 'nr'
            nr_file = os.path.join(matchings_dir,build_filename(name, type, chain_cap))

            cfg = kidney_ip.OptConfig(d, a, cycle_cap, chain_cap, verbose=(verbose >= 2),
                                      timelimit=timelimit, edge_success_prob=edge_success_prob, cycle_file=cycle_file)

            # # if os.path.exists(nr_file):
            # #     if run_experiment:
            # #         sol = kidney_ip.OptSolution.from_file(cfg, nr_file)
            # #         nonrob_score = sum(e.score for e in sol.matching_edges)
            # #     else:
            # #         pass
            # else:

            sol = kidney_ip.optimise_picef(cfg)
            sol.save_to_file(nr_file)
            nonrob_score = sum(e.score for e in sol.matching_edges)

            max_card = 0
            for p in protection_levels:

                type = 'wt'
                wt_file = os.path.join(matchings_dir, build_filename(name, type, chain_cap, wt_string=wt_string, edge_success_prob=None, protection_level=p))

                cfg_rob = kidney_ip.OptConfig(d, a, cycle_cap, chain_cap, verbose=(verbose>=2),
                                              timelimit=timelimit, edge_success_prob=edge_success_prob,
                                              protection_level=p,cycle_file=cycle_file, edge_assign_seed=edge_assign_seed)

                if os.path.exists(wt_file):
                    if run_experiment:
                        sol_rob = kidney_ip.OptSolution.from_file(cfg_rob, wt_file)
                        rob_optimistic_score = sum(e.score for e in sol_rob.matching_edges)
                    else:
                        pass
                else:
                    sol_rob = kidney_ip.solve_edge_weight_uncertainty(cfg_rob, max_card=max_card)
                    sol_rob.save_to_file(wt_file)
                    rob_optimistic_score = sum(e.score for e in sol_rob.matching_edges)
                    max_card = sol_rob.max_card  # don't recalculate the maximum cardinality...

                if verbose >= 1:
                    print "protection level epsilon = %f" % p
                    print "non-robust solution:"
                    print sol.display()
                    print "robust solution:"
                    print sol_rob.display()

                if run_experiment:
                    # assign edges their original weight_type using the seed
                    assign_flat_edges(sol_rob.edge_assign_seed, alpha, d, a)

                    with open(outfile, 'a') as csvfile:
                        for i in range(N_trials):

                            realized_robust_score = sol_rob.score_with_edge_weight_func(realized_weight_func)
                            realized_nonrob_score = sol.score_with_edge_weight_func(realized_weight_func)

                            csvfile.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                                name,
                                i,
                                cycle_cap,
                                chain_cap,
                                alpha,
                                p,
                                sol_rob.gamma,
                                rob_optimistic_score,
                                nonrob_score,
                                realized_robust_score,
                                realized_nonrob_score))

def find_edge_existence_matchings(matchings_dir, d, a, name,
                                    verbose=0,
                                    cycle_cap=3,
                                    chain_cap_list=[3],
                                    edge_success_prob_list = [0.4],
                                    gamma_list=[],
                                    protection_levels=[0.01],
                                    N_trials=10,
                                    subtour_dir = None,
                                    remove_subtours=False,
                                    cycle_file=None,
                                    run_experiment=False,
                                    outfile=None):
    timelimit = None

    if len(edge_success_prob_list) > 0:
        prob = True
        use_gamma = False
        par_list = edge_success_prob_list
    elif len(gamma_list) > 0:
        use_gamma = True
        prob = False
        par_list = gamma_list

    if ( len(edge_success_prob_list) > 0 and len(gamma_list) > 0):
        raise Warning("cannot specify both edge success prob list and gamma list")

    for chain_cap in chain_cap_list:

        for par in par_list:

            if prob:
                edge_fail_func = lambda e: np.random.rand() > par
            elif use_gamma:
                gamma = par

            type = 'nr'
            nr_file = os.path.join(matchings_dir,build_filename(name, type, chain_cap))

            cfg = kidney_ip.OptConfig(d, a, cycle_cap, chain_cap, verbose=(verbose >= 2),
                                      timelimit=timelimit, edge_success_prob=1, cycle_file=cycle_file)

            if os.path.exists(nr_file):
                if run_experiment:
                    sol = kidney_ip.OptSolution.from_file(cfg, nr_file)
                    nonrob_score = sum(e.score for e in sol.matching_edges)
            else:
                sol = kidney_ip.optimise_picef(cfg)
                sol.save_to_file(nr_file)
                nonrob_score = sum(e.score for e in sol.matching_edges)
                if verbose >= 1:
                    print "protection level epsilon = %f" % p
                    print "non-robust solution:"
                    print sol.display()

            if prob:
                type = 'fa'
                fa_file = os.path.join(matchings_dir,build_filename(name, type, chain_cap,edge_success_prob=par))

                cfg_failaware = kidney_ip.OptConfig(d, a, cycle_cap, chain_cap, verbose=(verbose>=2),
                                           timelimit=timelimit, edge_success_prob=par,cycle_file=cycle_file)

            if os.path.exists(fa_file):
                if run_experiment:
                    sol_failaware = kidney_ip.OptSolution.from_file(cfg_failaware, fa_file)
                    failaware_score = sum(e.score for e in sol_failaware.matching_edges)
                else:
                    pass
            else:
                sol_failaware = kidney_ip.optimise_picef(cfg_failaware)
                sol_failaware.save_to_file(fa_file)
                failaware_score = sol_failaware.get_total_score()
                if verbose >= 1:
                    print "failure-aware solution:"
                    print sol_failaware.display()
            if run_experiment and (sol.total_score == 0):
                if verbose:
                    print "optimal matching has zero weight for exchange: %s" % name
                with open(outfile, 'a') as csvfile:
                    csvfile.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        name,
                        0, # i
                        cycle_cap,
                        chain_cap,
                        par,
                        p,
                        0, # sol_rob.gamma,
                        0, # sol_rob.optimistic_score,
                        0, # failaware_nominal_score,
                        0, # sol.total_score,
                        0, # realized_robust_score,
                        0, # realized_failaware_score,
                        0 ))# realized_nonrob_score))
            else:
                max_num_cycles = 0
                for p in protection_levels:

                    type = 'ex'
                    ex_file = os.path.join(matchings_dir, build_filename(name, type, chain_cap,
                                                                         edge_success_prob=par,
                                                                         protection_level=p))

                    cfg_rob = kidney_ip.OptConfig(d, a, cycle_cap, chain_cap, verbose=(verbose >= 2),
                                                  timelimit=timelimit, edge_success_prob=par,
                                                  protection_level=p, name=name, subtour_dir=subtour_dir,
                                                  cycle_file=cycle_file,
                                                  remove_subtours=remove_subtours)

                    if os.path.exists(ex_file):
                        if run_experiment:
                            sol_rob = kidney_ip.OptSolution.from_file(cfg_rob, ex_file)
                            rob_optimistic_score = sum(e.score for e in sol_rob.matching_edges)
                            gamma = sol_rob.gamma
                        else:
                            pass
                    else:
                        sol_rob = kidney_ip.solve_edge_existence_uncertainty(cfg_rob, max_num_cycles=max_num_cycles)
                        sol_rob.save_to_file(ex_file)
                        rob_optimistic_score = sum(e.score for e in sol_rob.matching_edges)
                        gamma = sol_rob.gamma
                        max_num_cycles = sol_rob.max_num_cycles  # don't recalculate the maximum number of cycles...
                        if verbose:
                            print "robust solution:"
                            print sol_rob.display()



                    if run_experiment:
                        # get the robust and non-robust scores for N_trials realizations
                        with open(outfile, 'a') as csvfile:
                            for i in range(N_trials):

                                    # add edge failures (these don't need to be replicated, they are Bernouli random)
                                    for e in d.es:
                                        e.fail = edge_fail_func(e)

                                    for n in a:
                                        for e in n.edges:
                                            e.fail = edge_fail_func(e)

                                    if verbose >= 2:
                                        print "MATCHINGS AFTER FAILURE"
                                        print "non-robust solution:"
                                        print sol.display()
                                        print "failure-aware solution:"
                                        print sol_failaware.display()
                                        print "robust solution:"
                                        print sol_rob.display()

                                    realized_failaware_score = sol_failaware.score_after_edge_failure(a)
                                    realized_robust_score = sol_rob.score_after_edge_failure(a)
                                    realized_nonrob_score = sol.score_after_edge_failure(a)

                                    csvfile.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                                        name,
                                        i,
                                        cycle_cap,
                                        chain_cap,
                                        par,
                                        p,
                                        gamma,
                                        rob_optimistic_score,
                                        failaware_score,
                                        nonrob_score,
                                        realized_robust_score,
                                        realized_failaware_score,
                                        realized_nonrob_score))

def find_edge_existence_matchings_const_gamma(matchings_dir, d, a,
                                    name,verbose=0,
                                    cycle_cap=3,
                                    chain_cap_list=[3],
                                    gamma_list=[1],
                                    N_trials=10,
                                    run_experiment=False,
                                    outfile=None):
    timelimit = None

    for chain_cap in chain_cap_list:

        empty_matching = False
        for gamma in sorted(gamma_list):

            type = 'nr'
            nr_file = os.path.join(matchings_dir, build_filename(name, type, chain_cap))

            cfg = kidney_ip.OptConfig(d, a, cycle_cap, chain_cap, verbose=(verbose >= 2),
                                      timelimit=timelimit, edge_success_prob=1,gamma=0)

            if os.path.exists(nr_file):
                if run_experiment:
                    sol = kidney_ip.OptSolution.from_file(cfg, nr_file)
                    nonrob_score = sum(e.score for e in sol.matching_edges)
            else:
                sol = kidney_ip.optimise_pctsp(cfg) # THIS USED TO USE PICEF
                sol.save_to_file(nr_file)
                nonrob_score = sum(e.score for e in sol.matching_edges)
                if verbose >= 1:
                    print "non-robust solution:"
                    print sol.display()

            type = 'gamma'
            gam_file = os.path.join(matchings_dir, build_filename(name, type, chain_cap,gamma=gamma))

            cfg_rob = kidney_ip.OptConfig(d, a, cycle_cap, chain_cap, verbose=(verbose >= 2),
                                          timelimit=timelimit,
                                          name=name,
                                          gamma=gamma)

            if os.path.exists(gam_file):
                if run_experiment:
                    sol_rob = kidney_ip.OptSolution.from_file(cfg_rob, gam_file)
                    rob_optimistic_score = sum(e.score for e in sol_rob.matching_edges)
                else:
                    pass
            elif empty_matching:
                sol_rob.gamma = gamma # use the previous matching, but change gamma...
            else:
                sol_rob = kidney_ip.optimize_robust_pctsp(cfg_rob)
                sol_rob.save_to_file(gam_file)
                rob_optimistic_score = sum(e.score for e in sol_rob.matching_edges)
                if rob_optimistic_score == 0:
                    empty_matching = True


                if verbose:
                    print "robust solution (gamma = %d):" % gamma
                    print sol_rob.display()

            if run_experiment:
                # get the robust and non-robust scores for N_trials realizations
                with open(outfile, 'a') as csvfile:
                    for i in range(N_trials):

                        # scores after a set number of failures
                        realized_robust_score = sol_rob.score_after_num_failures(a,gamma,seed=i)
                        realized_nonrob_score = sol.score_after_num_failures(a,gamma,seed=i)

                        csvfile.write("{},{},{},{},{},{},{},{},{}\n".format(
                                name,
                                i,
                                cycle_cap,
                                chain_cap,
                                gamma,
                                rob_optimistic_score,
                                nonrob_score,
                                realized_robust_score,
                                realized_nonrob_score))



def print_header_weight(outfile):
    with open(outfile,'w') as csvfile:
        colnames = ['kpd_dirname',
                'trial_num',
                'cycle_cap',
                'chain_cap',
                'alpha',
                'protection_level',
                'best_gamma',
                'nominal_robust_matching_score',
                'nominal_nonrob_matching_score',
                'realized_robust_matching_score',
                'realized_nonrob_matching_score']
        # 11 fields
        csvfile.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(*colnames))


def print_header_existence(outfile):
    with open(outfile,'w') as csvfile:
        colnames = ['kpd_dirname',
                'trial_num',
                'cycle_cap',
                'chain_cap',
                'edge_success_prob',
                'protection_level',
                'best_gamma',
                'nominal_robust_matching_score',
                'nominal_failaware_matching_score',
                'nominal_nonrob_matching_score',
                'realized_robust_matching_score',
                'realized_failaware_matching_score',
                'realized_nonrob_matching_score']
        # 13 fields
        csvfile.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(*colnames))


def print_header_constant_gamma(outfile):
    with open(outfile,'w') as csvfile:
        colnames = ['kpd_dirname',
                'trial_num',
                'cycle_cap',
                'chain_cap',
                'gamma',
                'nominal_robust_matching_score',
                'nominal_nonrob_matching_score',
                'realized_robust_matching_score',
                'realized_nonrob_matching_score']
        # 9 fields
        csvfile.write("{},{},{},{},{},{},{},{},{}\n".format(*colnames))


if __name__ == "__main__":
    start()
