'''Solving the kidney exchange problem with PICEF, PC-TSP, and PI-TSP'''

import kidney_utils
from gurobi_functions import optimize, create_mip_model

from kidney_digraph import Edge
from kidney_ndds import NddEdge

from kidney_digraph import Cycle, failure_aware_cycle_weight, cycle_weight
from kidney_ndds import Chain

from gurobipy import *

# from graph_tool import Graph
# from graph_tool.flow import boykov_kolmogorov_max_flow, min_st_cut

import numpy as np
import sys
import random
from guppy import hpy
import json

from kidney_digraph import reg_failure_aware_cycle_weight

h = hpy()

EPS = 1e-12
EPS_mid = 0.01
EPS_large = 0.1
W = 1e7
W_small = 1e4

# for testing modifications to the formulations
test = True

relax_discount = 0.99  # total discount is 1.0


###################################################################################################
#                                                                                                 #
#                                  Code used by all formulations                                  #
#                                                                                                 #
###################################################################################################

class MaxIterConstraintGenException(Exception):
    pass


class OptConfig(object):
    """The inputs (problem instance and parameters) for an optimisation run

    Data members:
        digraph
        ndds
        max_cycle
        max_chain
        verbose: True if and only if Gurobi output should be written to screen and log file
        timelimit
        edge_success_prob
        lp_file: The name of a .lp file to write, or None if the file should not be written
        relax: True if and only if the LP relaxation should be solved also
        gamma: uncertainty budget (robust implementation)
    """

    def __init__(self, digraph, ndds, max_cycle, max_chain,
                 verbose=False,
                 timelimit=None,
                 edge_success_prob=1,
                 lp_file=None,
                 relax=False,
                 gamma=0,
                 cardinality_restriction=None,
                 protection_level=0.1,
                 chain_restriction=None,
                 cycle_restriction=None,
                 name=None):
        self.digraph = digraph
        self.ndds = ndds
        self.max_cycle = max_cycle
        self.max_chain = max_chain
        self.verbose = verbose
        self.timelimit = timelimit
        self.edge_success_prob = edge_success_prob
        self.edge_failure_prob = 1.0 - self.edge_success_prob
        self.lp_file = lp_file
        self.relax = relax
        self.gamma = gamma  # robust uncertainty budget
        self.cardinality_restriction = cardinality_restriction
        self.chain_restriction = chain_restriction
        self.cycle_restriction = cycle_restriction
        self.protection_level = protection_level  # for variable budget uncertainty: probability that U-set does not contain true edge weights
        self.name = name

        # are chains used?
        self.use_chains = (self.max_chain > 0) and (self.chain_restriction < len(self.ndds))


class OptSolution(object):
    """An optimal solution for a kidney-exchange problem instance.

    Data members:
        ip_model: The Gurobi Model object
        cycles: A list of cycles in the optimal solution, each represented
            as a list of vertices
        chains: A list of chains in the optimal solution, each represented
            as a Chain object
        total_weight: The total weight of the solution
    """

    def __init__(self, ip_model, cycles, chains, digraph,
                 edge_success_prob=1,
                 infeasible=False,
                 gamma=0,
                 robust_weight=0,
                 optimistic_weight=0,
                 cycle_obj=None,
                 chain_restriction=None,
                 cycle_restriction=None,
                 cardinality_restriction=None,
                 cycle_cap=None,
                 chain_cap=None,
                 matching_edges=None,
                 alpha_var=None):
        self.ip_model = ip_model
        self.cycles = cycles
        self.chains = chains
        self.digraph = digraph
        self.infeasible = infeasible
        if self.infeasible:
            self.total_weight = 0
        else:
            self.total_weight = (sum(c.weight for c in chains) +
                                 sum(failure_aware_cycle_weight(c, digraph, edge_success_prob) for c in cycles))
        self.edge_success_prob = edge_success_prob
        self.cycle_obj = cycle_obj
        self.matching_edges = matching_edges
        self.gamma = gamma
        self.robust_weight = robust_weight
        self.optimistic_weight = optimistic_weight
        self.cycle_restriction = cycle_restriction
        self.chain_restriction = chain_restriction
        self.cardinality_restriction = cardinality_restriction
        self.cycle_cap = cycle_cap
        self.chain_cap = chain_cap
        self.alpha_var = alpha_var

        if ip_model is not None:
            self.timeout = (ip_model.status == GRB.TIME_LIMIT)
        else:
            self.timeout = False

    def same_matching_edges(self, other):
        if len(self.matching_edges) != len(other.matching_edges):
            return False
        for self_e in self.matching_edges:
            edge_found = False
            for other_e in other.matching_edges:
                if (self_e.src_id == other_e.src_id) and (self_e.tgt.id == other_e.tgt.id):
                    edge_found = True
                    break
            if not edge_found:
                return False
        return True

    def add_matching_edges(self, ndds):
        '''Set attribute 'matching_edges' using self.cycle_obj, self.chains, and self.digraph'''

        matching_edges = []

        for ch in self.chains:
            chain_edges = []
            tgt_id = ch.vtx_indices[0]
            for e in ndds[ch.ndd_index].edges:
                if e.tgt.id == tgt_id:
                    chain_edges.append(e)
            if len(chain_edges) == 0:
                raise Warning("NDD edge not found")
            for i in range(len(ch.vtx_indices) - 1):
                chain_edges.append(self.digraph.adj_mat[ch.vtx_indices[i]][ch.vtx_indices[i + 1]])
            if len(chain_edges) != (len(ch.vtx_indices)):
                raise Warning("Chain contains %d edges, but only %d edges found" %
                              (len(ch.vtx_indices), len(chain_edges)))
            matching_edges.extend(chain_edges)

        for cy in self.cycle_obj:
            cycle_edges = []
            for i in range(len(cy.vs) - 1):
                cycle_edges.append(self.digraph.adj_mat[cy.vs[i].id][cy.vs[i + 1].id])
            # add final edge
            cycle_edges.append(self.digraph.adj_mat[cy.vs[-1].id][cy.vs[0].id])
            if len(cycle_edges) != len(cy.vs):
                raise Warning("Cycle contains %d vertices, but only %d edges found" %
                              (len(cy.vs), len(cycle_edges)))
            matching_edges.extend(cycle_edges)

        self.matching_edges = matching_edges

    def display(self):
        """Print the optimal cycles and chains to standard output."""

        print("cycle_count: {}".format(len(self.cycles)))
        print("chain_count: {}".format(len(self.chains)))
        print("cycles:")
        # # cs is a list of cycles, with each cycle represented as a list of vertex IDs
        # # Sort the cycles
        if len(self.cycle_obj) > 0:
            for c in sorted(self.cycle_obj):
                print(c.display())
        else:
            cs = [[v.id for v in c] for c in self.cycles]
            # Put the lowest-indexed vertex at the start of each cycle
            for i in range(len(cs)):
                min_index_pos = cs[i].index(min(cs[i]))
                cs[i] = cs[i][min_index_pos:] + cs[i][:min_index_pos]
                print("\t".join(str(v_id) for v_id in cs[i]))
        print("chains:")
        for c in self.chains:
            print(c.display())

        print("edges:")
        for e in sorted(self.matching_edges, key=lambda x: x.weight, reverse=True):
            print(e.display(self.gamma))

        print("total weight:")
        print(self.total_weight)
        if self.gamma > 0:
            d_sum = np.sum(e.discount * e.discount_frac for e in self.matching_edges)
            print("total discount value = {}".format(d_sum))
            print("robust matching weight: {}".format(self.robust_weight))
            print("optimistic matching weight: {}".format(self.optimistic_weight))

    # added by Duncan
    def vertex_mask(self):
        """Returns a numpy array of length |V| containing 1 if the vertex
        participates in the solution, and zero otherwise."""
        # cs is a list of cycles, with each cycle represented as a list of vertex IDs
        v_list = list()
        for cy in self.cycles:
            v_list.append([v.id for v in cy])
        for ch in self.chains:
            v_list.append([v for v in ch.vtx_indices])
        # cy_verts = [v.id for v in self.cycles if self.cycles]
        #        ch_verts = [v.id for v in self.chains if self.chains]
        v_mask = np.zeros(self.digraph.n, dtype=np.int)
        if np.sum(v_list) != 0:
            v_mask[np.concatenate(v_list)] = 1
        return v_mask

    # get weight using a digraph with (possibly) different weights
    def get_weight(self, digraph, ndds, edge_success_prob=1.0):
        weight = (sum(c.get_weight(digraph, ndds, edge_success_prob) for c in self.chains) +
                  sum(failure_aware_cycle_weight(c, digraph, edge_success_prob) for c in self.cycles))
        return weight


###################################################################################################
#                                                                                                 #
#                  Chain vars and constraints (used by HPIEF', HPIEF'' and PICEF)                 #
#                                                                                                 #
###################################################################################################

def add_chain_vars_and_constraints(digraph, ndds, use_chains, max_chain, m, vtx_to_vars,
                                   store_edge_positions=False):
    """Add the IP variables and constraints for chains in PICEF and HPIEF'.

    Args:
        ndds: a list of NDDs in the instance
        use_chains: boolean: True if chains should be used
        max_chain: the chain cap
        m: The Gurobi model
        vtx_to_vars: A list such that for each Vertex v in the Digraph,
            vtx_to_vars[v.id] will contain the Gurobi variables representing
            edges pointing to v.
        store_edge_positions: if this is True, then an attribute grb_edge_positions
            will be added to edges that have associated Gurobi variables.
            edge.grb_edge_positions[i] will indicate the position of the edge respresented
            by edge.grb_vars[i]. (default: False)
    """

    if use_chains:  # max_chain > 0:
        for v in digraph.vs:
            v.grb_vars_in = [[] for i in range(max_chain - 1)]
            v.grb_vars_out = [[] for i in range(max_chain - 1)]

        for ndd in ndds:
            ndd_edge_vars = []
            for e in ndd.edges:
                edge_var = m.addVar(vtype=GRB.BINARY)
                e.edge_var = edge_var
                ndd_edge_vars.append(edge_var)
                vtx_to_vars[e.tgt.id].append(edge_var)
                if max_chain > 1: e.tgt.grb_vars_in[0].append(edge_var)
            m.update()
            m.addConstr(quicksum(ndd_edge_vars) <= 1)

        dists_from_ndd = kidney_utils.get_dist_from_nearest_ndd(digraph, ndds)

        # Add pair->pair edge variables, indexed by position in chain
        # e.grb_var are the chain variables for each edge.
        for e in digraph.es:
            e.grb_vars = []
            if store_edge_positions:
                e.grb_var_positions = []
            for i in range(max_chain - 1):
                if dists_from_ndd[e.src.id] <= i + 1:
                    edge_var = m.addVar(vtype=GRB.BINARY)
                    e.grb_vars.append(edge_var)
                    if store_edge_positions:
                        e.grb_var_positions.append(i + 1)
                    vtx_to_vars[e.tgt.id].append(edge_var)
                    e.src.grb_vars_out[i].append(edge_var)
                    if i < max_chain - 2:
                        e.tgt.grb_vars_in[i + 1].append(edge_var)
            # m.update()

        # At each chain position, sum of edges into a vertex must be >= sum of edges out
        for i in range(max_chain - 1):
            for v in digraph.vs:
                m.addConstr(quicksum(v.grb_vars_in[i]) >= quicksum(v.grb_vars_out[i]))

        m.update()


###################################################################################################
#                                                                                                 #
#                                              PICEF                                              #
#                                                                                                 #
###################################################################################################

def create_picef_model(cfg):
    """Optimise using the PICEF formulation.

    Args:
        cfg: an OptConfig object

    Returns:
        an OptSolution object
    """

    cycles = cfg.digraph.find_cycles(cfg.max_cycle)

    m = create_mip_model(time_lim=cfg.timelimit, verbose=cfg.verbose)
    m.params.method = -1

    cycle_vars = [m.addVar(vtype=GRB.BINARY) for __ in cycles]

    vtx_to_vars = [[] for __ in cfg.digraph.vs]

    add_chain_vars_and_constraints(cfg.digraph, cfg.ndds, cfg.use_chains, cfg.max_chain, m,
                                   vtx_to_vars, store_edge_positions=cfg.edge_success_prob != 1)

    for i, c in enumerate(cycles):
        for v in c:
            vtx_to_vars[v.id].append(cycle_vars[i])

    for l in vtx_to_vars:
        if len(l) > 0:
            m.addConstr(quicksum(l) <= 1)

    # add variables for each pair-pair edge indicating whether it is used in a cycle or chain
    for e in cfg.digraph.es:
        used_in_cycle = []
        for var, c in zip(cycle_vars, cycles):
            if kidney_utils.cycle_contains_edge(c, e):
                used_in_cycle.append(var)
        used_var = m.addVar(vtype=GRB.INTEGER)
        if cfg.use_chains:
            m.addConstr(used_var == quicksum(used_in_cycle) + quicksum(e.grb_vars))
        else:
            m.addConstr(used_var == quicksum(used_in_cycle))
        e.used_var = used_var

    # number of edges in the matching
    num_edges_var = m.addVar(vtype=GRB.INTEGER)
    pair_edge_count = [e.used_var for e in cfg.digraph.es]
    if cfg.use_chains:
        ndd_edge_count = [e.edge_var for ndd in cfg.ndds for e in ndd.edges]
        m.addConstr(num_edges_var == quicksum(pair_edge_count + ndd_edge_count))
    else:
        m.addConstr(num_edges_var == quicksum(pair_edge_count))

    # add a cardinality restriction if necessary
    if cfg.cardinality_restriction is not None:
        m.addConstr(num_edges_var <= cfg.cardinality_restriction)

    m.update()

    return m, cycles, cycle_vars, num_edges_var


def optimize_picef(cfg):
    m, cycles, cycle_vars, _ = create_picef_model(cfg)

    # add cycle objects
    cycle_list = []
    for c, var in zip(cycles, cycle_vars):
        c_obj = Cycle(c)
        c_obj.add_edges(cfg.digraph.es)
        c_obj.weight = failure_aware_cycle_weight(c_obj.vs, cfg.digraph, cfg.edge_success_prob)
        c_obj.grb_var = var
        cycle_list.append(c_obj)

    if not cfg.use_chains:
        obj_expr = quicksum(failure_aware_cycle_weight(c, cfg.digraph, cfg.edge_success_prob) * var
                            for c, var in zip(cycles, cycle_vars))
    elif cfg.edge_success_prob == 1:
        obj_expr = (quicksum(cycle_weight(c, cfg.digraph) * var for c, var in zip(cycles, cycle_vars)) +
                    quicksum(e.weight * e.edge_var for ndd in cfg.ndds for e in ndd.edges) +
                    quicksum(e.weight * var for e in cfg.digraph.es for var in e.grb_vars))
    else:
        obj_expr = (quicksum(failure_aware_cycle_weight(c, cfg.digraph, cfg.edge_success_prob) * var
                             for c, var in zip(cycles, cycle_vars)) +
                    quicksum(e.weight * cfg.edge_success_prob * e.edge_var
                             for ndd in cfg.ndds for e in ndd.edges) +
                    quicksum(e.weight * cfg.edge_success_prob ** (pos + 1) * var
                             for e in cfg.digraph.es for var, pos in zip(e.grb_vars, e.grb_var_positions)))
    m.setObjective(obj_expr, GRB.MAXIMIZE)

    optimize(m)

    pair_edges = [e for e in cfg.digraph.es if e.used_var.x > 0.5]

    if cfg.use_chains:
        matching_chains = kidney_utils.get_optimal_chains(
            cfg.digraph, cfg.ndds, cfg.edge_success_prob)
        ndd_chain_edges = [e for ndd in cfg.ndds for e in ndd.edges if e.edge_var.x > 0.5]
    else:
        ndd_chain_edges = []
        matching_chains = []

    matching_edges = pair_edges + ndd_chain_edges

    if cfg.cardinality_restriction is not None:
        if len(matching_edges) > cfg.cardinality_restriction:
            raise Warning("cardinality restriction is violated: restriction = %d edges, matching uses %d edges" % (
            cfg.cardinality_restriction, len(matching_edges)))

    cycles_used = [c for c, v in zip(cycles, cycle_vars) if v.x > 0.5]
    cycle_obj = [c for c in cycle_list if c.grb_var.x > 0.5]

    sol = OptSolution(ip_model=m,
                      cycles=cycles_used,
                      cycle_obj=cycle_obj,
                      chains=matching_chains,
                      digraph=cfg.digraph,
                      edge_success_prob=cfg.edge_success_prob,
                      chain_restriction=cfg.chain_restriction,
                      cycle_restriction=cfg.cycle_restriction,
                      cycle_cap=cfg.max_chain,
                      chain_cap=cfg.max_cycle,
                      cardinality_restriction=cfg.cardinality_restriction)
    sol.add_matching_edges(cfg.ndds)
    kidney_utils.check_validity(sol, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain)
    return sol, matching_edges


def max_cycles(cfg):
    '''
    Use PICEF to find the maximum number of cycles in a matching...
    '''
    m, _, cycle_vars, _ = create_picef_model(cfg)

    num_cycles = quicksum(cycle_vars)

    m.setObjective(num_cycles, GRB.MAXIMIZE)

    optimize(m)
    if cfg.verbose:
        print("maximum number of cycles = %d" % m.objVal)
    if m.objVal != int(m.objVal):
        raise Warning("number of cycles is not integer")
    return int(m.objVal)


def optimise_robust_picef(cfg):
    m, cycles, cycle_vars, num_edges_var = create_picef_model(cfg)

    # for use later
    floor_gamma = np.floor(cfg.gamma)
    ceil_gamma = np.ceil(cfg.gamma)
    gamma_frac = cfg.gamma - floor_gamma

    # add cycle vars
    cycle_list = []
    for c, var in zip(cycles, cycle_vars):
        c_obj = Cycle(c)
        c_obj.add_edges(cfg.digraph.es)
        c_obj.weight = cycle_weight(c_obj.vs, cfg.digraph)
        c_obj.grb_var = var
        cycle_list.append(c_obj)

    m.update()

    # gamma is integer
    if gamma_frac == 0:

        if cfg.use_chains:
            for ndd in cfg.ndds:
                for e in ndd.edges:
                    g_var = m.addVar(vtype=GRB.BINARY)
                    d_var = m.addVar(vtype=GRB.BINARY)
                    e.g_var = g_var
                    e.d_var = d_var
                    m.addGenConstrAnd(e.d_var, [e.g_var, e.edge_var])

            m.update()

        # add g and d variables for pair-pair edges
        for e in cfg.digraph.es:
            g_var = m.addVar(vtype=GRB.BINARY)
            d_var = m.addVar(vtype=GRB.BINARY)
            e.g_var = g_var
            e.d_var = d_var
            m.addGenConstrAnd(e.d_var, [e.g_var, e.used_var])

        m.update()

    # gamma is not integer
    else:

        if cfg.use_chains:
            # use both gf (full discount if gf=1, gp=0) and gp (partial discount, if gf=gp=1)
            for ndd in cfg.ndds:
                for e in ndd.edges:
                    gf_var = m.addVar(vtype=GRB.BINARY)
                    df_var = m.addVar(vtype=GRB.BINARY)
                    e.gf_var = gf_var
                    e.df_var = df_var
                    m.addGenConstrAnd(e.df_var, [e.gf_var, e.edge_var])
                    gp_var = m.addVar(vtype=GRB.BINARY)
                    dp_var = m.addVar(vtype=GRB.BINARY)
                    e.gp_var = gp_var
                    e.dp_var = dp_var
                    m.addGenConstrAnd(e.dp_var, [e.gp_var, e.edge_var])

            m.update()

        for e in cfg.digraph.es:
            gf_var = m.addVar(vtype=GRB.BINARY)
            df_var = m.addVar(vtype=GRB.BINARY)
            e.gf_var = gf_var
            e.df_var = df_var
            m.addGenConstrAnd(e.df_var, [e.gf_var, e.used_var])
            gp_var = m.addVar(vtype=GRB.BINARY)
            dp_var = m.addVar(vtype=GRB.BINARY)
            e.gp_var = gp_var
            e.dp_var = dp_var
            m.addGenConstrAnd(e.dp_var, [e.gp_var, e.used_var])

        m.update()

    # discount indicators g follow same ordering as the edge discount values (sort in increasing order)
    if cfg.use_chains:
        ndd_e = [e for ndd in cfg.ndds for e in ndd.edges]
    else:
        ndd_e = []
    all_edges = cfg.digraph.es + ndd_e
    e_sorted = sorted(all_edges, key=lambda x: x.discount, reverse=False)

    # ordering constraints over g
    # gamma is integer
    if gamma_frac == 0:
        for i in range(len(e_sorted) - 1):
            m.addConstr(e_sorted[i].g_var <= e_sorted[i + 1].g_var)

    # gamma is not integer
    else:
        for i in range(len(e_sorted) - 1):
            m.addConstr(e_sorted[i].gf_var <= e_sorted[i + 1].gf_var)
            m.addConstr(e_sorted[i].gp_var <= e_sorted[i + 1].gp_var)

    # number of edges used in matching (include all position-indexed vars)

    # uncertainty budget (number of discounted edges)
    gamma_var = m.addVar(vtype=GRB.CONTINUOUS)
    m.addGenConstrMin(gamma_var, [num_edges_var, cfg.gamma])

    # add a cardinality restriction if necessary
    if cfg.cardinality_restriction is not None:
        m.addConstr(num_edges_var <= cfg.cardinality_restriction)

    m.update()

    # limit number of discounted variables
    # gamma is integer
    if gamma_frac == 0:
        m.addConstr(quicksum(e.d_var for e in all_edges) == gamma_var)

    # gamma is not integer
    else:
        h_var = m.addVar(vtype=GRB.BINARY)
        m.addConstr(cfg.gamma - num_edges_var <= W_small * h_var)
        m.addConstr(num_edges_var - cfg.gamma <= W_small * (1 - h_var))
        m.addConstr(quicksum(e.dp_var for e in all_edges) == h_var * num_edges_var + (1 - h_var) * ceil_gamma)
        m.addConstr(quicksum(e.df_var for e in all_edges) == h_var * num_edges_var + (1 - h_var) * floor_gamma)

    # total discount (by edge)
    # gamma is integer
    if gamma_frac == 0:
        total_discount = quicksum(e.discount * e.d_var for e in all_edges)

    # gamma is not integer
    else:
        total_discount = quicksum((1 - gamma_frac) * e.discount * e.df_var for e in all_edges) + \
                         quicksum(gamma_frac * e.discount * e.dp_var for e in all_edges)

    # set a variable for the total (optimistic matching weight)
    total_weight = m.addVar(vtype=GRB.CONTINUOUS)

    m.update()

    if not cfg.use_chains:
        m.addConstr(total_weight == quicksum(failure_aware_cycle_weight(c, cfg.digraph, cfg.edge_success_prob) * var
                                             for c, var in zip(cycles, cycle_vars)))
        obj_expr = total_weight - total_discount
    elif cfg.edge_success_prob == 1:
        m.addConstr(
            total_weight == (quicksum(cycle_weight(c, cfg.digraph) * var for c, var in zip(cycles, cycle_vars)) +
                             quicksum(e.weight * e.edge_var for ndd in cfg.ndds for e in ndd.edges) +
                             quicksum(e.weight * var for e in cfg.digraph.es for var in e.grb_vars)))
        obj_expr = total_weight - total_discount
    else:
        raise Warning("not implemented")

    m.setObjective(obj_expr, GRB.MAXIMIZE)

    optimize(m)

    if gamma_frac == 0:  # gamma is integer
        discounted_pair_edges = [e for e in cfg.digraph.es if e.d_var.x > 0]

        for e in discounted_pair_edges:
            e.discount_frac = e.d_var.x

        if cfg.use_chains:
            discounted_ndd_edges = [(i_ndd, e) for i_ndd, ndd in enumerate(cfg.ndds)
                                    for e in ndd.edges if e.d_var.x > 0.0]

            for _, e in discounted_ndd_edges:
                e.discount_frac = e.d_var.x

    else:  # gamma is not integer

        discounted_pair_edges = [e for e in cfg.digraph.es \
                                 if ((e.df_var.x > 0.0) or (e.dp_var.x > 0.0))]

        for e in discounted_pair_edges:
            e.discount_frac = (1 - gamma_frac) * e.df_var.x + gamma_frac * e.dp_var.x

        if cfg.use_chains:
            discounted_ndd_edges = [(i_ndd, e) for i_ndd, ndd in enumerate(cfg.ndds) for e in ndd.edges \
                                    if ((e.df_var.x > 0.0) or (e.dp_var.x > 0.0))]
            for _, e in discounted_ndd_edges:
                e.discount_frac = (1 - gamma_frac) * e.df_var.x + gamma_frac * e.dp_var.x

    if cfg.use_chains:
        ndd_matching_edges = [e for ndd in cfg.ndds for e in ndd.edges if e.edge_var.x > 0.5]
    else:
        ndd_matching_edges = []

    used_matching_edges = [e for e in cfg.digraph.es if e.used_var.x > 0.5]

    matching_edges = ndd_matching_edges + used_matching_edges

    if cfg.cardinality_restriction is not None:
        if len(matching_edges) > cfg.cardinality_restriction:
            raise Warning("cardinality restriction is violated: restriction = %d edges, matching uses %d edges" % (
            cfg.cardinality_restriction, len(matching_edges)))

    chains_used = kidney_utils.get_optimal_chains(cfg.digraph, cfg.ndds, cfg.edge_success_prob) if cfg.use_chains \
        else []

    cycles_used = [c for c, v in zip(cycles, cycle_vars) if v.x > 0.5]
    cycle_obj = [c for c in cycle_list if c.grb_var.x > 0.5]

    sol = OptSolution(ip_model=m,
                      cycles=cycles_used,
                      cycle_obj=cycle_obj,
                      chains=chains_used,
                      digraph=cfg.digraph,
                      edge_success_prob=cfg.edge_success_prob,
                      gamma=cfg.gamma,
                      robust_weight=m.objVal,
                      optimistic_weight=total_weight.x,
                      chain_restriction=cfg.chain_restriction,
                      cycle_restriction=cfg.cycle_restriction,
                      cycle_cap=cfg.max_chain,
                      chain_cap=cfg.max_cycle,
                      cardinality_restriction=cfg.cardinality_restriction)
    sol.add_matching_edges(cfg.ndds)
    kidney_utils.check_validity(sol, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain)

    return sol, matching_edges


def solve_edge_weight_uncertainty(cfg, max_card=None):
    # Solves the robust kidney exchange problem with a variable-budget edge weight uncertainty.
    #     - uses the cardinality-restriction method of Poss.
    #     - uses the constant-budget edge-weight-uncertainty robust formulation of PICEF
    #
    # inputs:
    #     - cfg               : OptConfig object
    #     - max_card          : maximum number of edges in a feasible solution (None if this is not known)

    # define gamma (variable uncertainty budget) function
    gamma_func = lambda x_norm: kidney_utils.gamma_symmetric_edge_weights(x_norm, cfg.protection_level)

    if cfg.verbose:
        print("solving edge weight uncertainty ")

    if max_card is None:
        # find maximum-cardinality solution (max edge-count)
        d_uniform = cfg.digraph.uniform_copy()
        ndds_uniform = [n.uniform_copy() for n in cfg.ndds]
        cfg_maxcard = OptConfig(d_uniform, ndds_uniform, cfg.max_cycle, cfg.max_chain, cfg.verbose,
                                cfg.timelimit, cfg.edge_success_prob, gamma=0)
        sol_maxcard = optimize_picef(cfg_maxcard)

        # the number of edges in the maximum cardinality solution
        max_card = len(sol_maxcard.matching_edges)

    if cfg.verbose:
        print("maximum cardinality = %d" % max_card)

    # now find all card-restricted solutions to the constant-budget robust problem,
    # and take the best one

    best_gamma = 0

    # if there is no feasible solution...
    if max_card == 0:
        sol_maxcard.max_card = 0
        return sol_maxcard

    for card_restriction in range(1, max_card + 1):

        # solve the k-cardinality-restricted problem, with Gamma = gamma(k)
        cfg.cardinality_restriction = card_restriction
        cfg.gamma = gamma_func(card_restriction)
        if cfg.gamma == 0:
            new_sol = optimize_picef(cfg)
            new_sol.robust_weight = new_sol.total_weight
            new_sol.optimistic_weight = new_sol.total_weight
        else:
            new_sol = optimise_robust_picef(cfg)

        if cfg.verbose:
            print("%d edges; gamma = %f; robust obj = %f" % (card_restriction, cfg.gamma, new_sol.robust_weight))

        if card_restriction == 1:
            best_sol = new_sol
            best_gamma = cfg.gamma
        elif new_sol.robust_weight > best_sol.robust_weight:
            best_sol = new_sol
            best_gamma = cfg.gamma

    # return the best solution and save the best gamma value
    cfg.gamma = best_gamma
    best_sol.max_card = max_card
    return best_sol


# #####################################DRO###############################################################
#
# # Infty-ball
#
# ######################################################################################################
#
# def optimize_DROinf_picef(cfg, theta=0.1):
#     m, cycles, cycle_vars, _ = create_picef_model(cfg)
#
#     # add cycle objects
#     cycle_list = []
#     for c, var in zip(cycles, cycle_vars):
#         c_obj = Cycle(c)
#         c_obj.add_edges(cfg.digraph.es)
#         c_obj.weight = failure_aware_cycle_weight(c_obj.vs, cfg.digraph, cfg.edge_success_prob)
#         c_obj.grb_var = var
#         cycle_list.append(c_obj)
#
#     if not cfg.use_chains:  # define the weights by including the regularization terms
#         obj_expr = quicksum(reg_failure_aware_cycle_weight(c, cfg.digraph, cfg.edge_success_prob, theta) * var
#                             for c, var in zip(cycles, cycle_vars))
#     elif cfg.edge_success_prob == 1:
#         obj_expr = (quicksum(
#             reg_failure_aware_cycle_weight(c, cfg.digraph, cfg.edge_success_prob, theta) * var for c, var in
#             zip(cycles, cycle_vars)) +
#                     quicksum((e.weight - theta) * e.edge_var for ndd in cfg.ndds for e in ndd.edges) +
#                     quicksum((e.weight - theta) * var for e in cfg.digraph.es for var in e.grb_vars))
#     else:
#         obj_expr = (quicksum(reg_failure_aware_cycle_weight(c, cfg.digraph, cfg.edge_success_prob, theta) * var
#                              for c, var in zip(cycles, cycle_vars)) +
#                     quicksum((e.weight - theta) * cfg.edge_success_prob * e.edge_var
#                              for ndd in cfg.ndds for e in ndd.edges) +
#                     quicksum((e.weight - theta) * cfg.edge_success_prob ** (pos + 1) * var
#                              for e in cfg.digraph.es for var, pos in zip(e.grb_vars, e.grb_var_positions)))
#     m.setObjective(obj_expr, GRB.MAXIMIZE)
#
#     optimize(m)
#
#     pair_edges = [e for e in cfg.digraph.es if e.used_var.x > 0.5]
#
#     if cfg.use_chains:
#         matching_chains = kidney_utils.get_optimal_chains(
#             cfg.digraph, cfg.ndds, cfg.edge_success_prob)
#         ndd_chain_edges = [e for ndd in cfg.ndds for e in ndd.edges if e.edge_var.x > 0.5]
#     else:
#         ndd_chain_edges = []
#         matching_chains = []
#
#     matching_edges = pair_edges + ndd_chain_edges
#
#     if cfg.cardinality_restriction is not None:
#         if len(matching_edges) > cfg.cardinality_restriction:
#             raise Warning("cardinality restriction is violated: restriction = %d edges, matching uses %d edges" % (
#                 cfg.cardinality_restriction, len(matching_edges)))
#
#     cycles_used = [c for c, v in zip(cycles, cycle_vars) if v.x > 0.5]
#     cycle_obj = [c for c in cycle_list if c.grb_var.x > 0.5]
#
#     sol = OptSolution(ip_model=m,
#                       cycles=cycles_used,
#                       cycle_obj=cycle_obj,
#                       chains=matching_chains,
#                       digraph=cfg.digraph,
#                       edge_success_prob=cfg.edge_success_prob,
#                       chain_restriction=cfg.chain_restriction,
#                       cycle_restriction=cfg.cycle_restriction,
#                       cycle_cap=cfg.max_chain,
#                       chain_cap=cfg.max_cycle,
#                       cardinality_restriction=cfg.cardinality_restriction)
#     sol.add_matching_edges(cfg.ndds)
#     kidney_utils.check_validity(sol, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain)
#     return sol, matching_edges


#####################################DRO###############################################################

# SAA formulation

######################################################################################################


def optimize_SAA_picef(cfg, num_weight_measurements, gamma, alpha):
    m, cycles, cycle_vars, _ = create_picef_model(cfg)

    # add cycle objects
    cycle_list = []
    for c, var in zip(cycles, cycle_vars):
        c_obj = Cycle(c)
        c_obj.add_edges(cfg.digraph.es)
        c_obj.weight = failure_aware_cycle_weight(c_obj.vs, cfg.digraph, cfg.edge_success_prob)
        c_obj.grb_var = var
        cycle_list.append(c_obj)

    # add variables for each edge weight measurement
    weight_vars = m.addVars(num_weight_measurements, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    for i in range(num_weight_measurements):
        m.addConstr(weight_vars[i] == - (quicksum(e.used_var * e.weight_list[i] for e in cfg.digraph.es) +
                                         quicksum(
                                             e.weight_list[i] * e.edge_var for ndd in cfg.ndds for e in ndd.edges)))

    # auxiliary variable
    d_var = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)

    # add pi variables & constraints for SAA
    pi_vars = m.addVars(num_weight_measurements, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    for i in range(num_weight_measurements):
        m.addConstr(pi_vars[i] >= weight_vars[i])
        m.addConstr(pi_vars[i] >= (1 + gamma / alpha) * weight_vars[i] - (d_var * gamma) / alpha)

    # objective
    obj = (1.0 / float(num_weight_measurements)) * quicksum(pi_vars) + gamma * d_var

    m.setObjective(obj, sense=GRB.MINIMIZE)

    if not cfg.use_chains:
        raise Exception("not implemented")
    elif cfg.edge_success_prob == 1:
        pass
    else:
        raise Exception("not implemented")

    optimize(m)

    pair_edges = [e for e in cfg.digraph.es if e.used_var.x > 0.5]

    if cfg.use_chains:
        matching_chains = kidney_utils.get_optimal_chains(
            cfg.digraph, cfg.ndds, cfg.edge_success_prob)
        ndd_chain_edges = [e for ndd in cfg.ndds for e in ndd.edges if e.edge_var.x > 0.5]
    else:
        ndd_chain_edges = []
        matching_chains = []

    matching_edges = pair_edges + ndd_chain_edges

    if cfg.cardinality_restriction is not None:
        if len(matching_edges) > cfg.cardinality_restriction:
            raise Warning("cardinality restriction is violated: restriction = %d edges, matching uses %d edges" % (
            cfg.cardinality_restriction, len(matching_edges)))

    cycles_used = [c for c, v in zip(cycles, cycle_vars) if v.x > 0.5]
    cycle_obj = [c for c in cycle_list if c.grb_var.x > 0.5]

    sol = OptSolution(ip_model=m,
                      cycles=cycles_used,
                      cycle_obj=cycle_obj,
                      chains=matching_chains,
                      digraph=cfg.digraph,
                      edge_success_prob=cfg.edge_success_prob,
                      chain_restriction=cfg.chain_restriction,
                      cycle_restriction=cfg.cycle_restriction,
                      cycle_cap=cfg.max_chain,
                      chain_cap=cfg.max_cycle,
                      cardinality_restriction=cfg.cardinality_restriction)
    sol.add_matching_edges(cfg.ndds)
    kidney_utils.check_validity(sol, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain)
    return None, matching_edges


#####################################DRO###############################################################

# DRO-SAA formulation

######################################################################################################


def optimize_DRO_SAA_picef(cfg, num_weight_measurements, gamma, alpha, theta, w_min, w_max):
    """Solve the DRO-SAA formulation of (Ren, 2020)
    
    Arguments:
        cfg: (OptConfig object)
        num_weight_measurements: (int). number of weight measurements associated with each edge
        gamma: (float). parameter balancing between a pure CVar objective (gamma->infinity) and a pure max-expectation
            objective (gamma=0)
        alpha: (float). CVar protection level, should be on [0, 1]
        theta: prediction of distance between assumed distribution and true distribution
        w_min: (float). assumed minimum edge weight of unknown distribution 
        w_max: (float). assumed maximum edge weight of unknown distribution 
    """

    m, cycles, cycle_vars, _ = create_picef_model(cfg)

    # add cycle objects
    cycle_list = []
    for c, var in zip(cycles, cycle_vars):
        c_obj = Cycle(c)
        c_obj.add_edges(cfg.digraph.es)
        c_obj.weight = failure_aware_cycle_weight(c_obj.vs, cfg.digraph, cfg.edge_success_prob)
        c_obj.grb_var = var
        cycle_list.append(c_obj)

    # add variables for each edge weight measurement
    weight_vars = m.addVars(num_weight_measurements, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    for i in range(num_weight_measurements):
        m.addConstr(weight_vars[i] == - (quicksum(e.used_var * e.weight_list[i] for e in cfg.digraph.es) +
                                         quicksum(
                                             e.weight_list[i] * e.edge_var for ndd in cfg.ndds for e in ndd.edges)))

    # auxiliary variables
    d_var = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='d')
    lam_var = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='lambda')
    s_vars = m.addVars(num_weight_measurements, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='s')

    # add eta and mu vars for each edge
    for e in cfg.digraph.es:
        e.eta_vars = m.addVars(num_weight_measurements, 2, vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY, name='eta')
        e.mu_vars = m.addVars(num_weight_measurements, 2, vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY, name='mu')
    for n in cfg.ndds:
        for e in n.edges:
            # also add the used_var here, for convenience
            e.used_var = e.edge_var
            e.eta_vars = m.addVars(num_weight_measurements, 2, vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY,
                                   name='eta')
            e.mu_vars = m.addVars(num_weight_measurements, 2, vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY,
                                  name='mu')

    # add variables for each edge weight measurement
    weight_vars = m.addVars(num_weight_measurements, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    for i in range(num_weight_measurements):
        m.addConstr(weight_vars[i] == - (quicksum(e.used_var * e.weight_list[i] for e in cfg.digraph.es) +
                                         quicksum(
                                             e.weight_list[i] * e.edge_var for ndd in cfg.ndds for e in ndd.edges)))

    b1 = 0
    b2 = - d_var * gamma / alpha

    # construct a list of all edges, for convenience
    e_list = cfg.digraph.es
    for n in cfg.ndds:
        e_list.extend(n.edges)

    # add main constraints
    for i_measurement in range(num_weight_measurements):
        # k = 1
        edge_sum_1a = quicksum([e.eta_vars[i_measurement, 0] * (w_max - e.weight_list[i_measurement])
                                for e in e_list])
        edge_sum_1b = quicksum([e.mu_vars[i_measurement, 0] * (e.weight_list[i_measurement] - w_min)
                                for e in e_list])
        m.addConstr(b1 + weight_vars[i_measurement] + edge_sum_1a + edge_sum_1b <= s_vars[i_measurement],
                    name=("s_constr_k1_i%d" % i_measurement))

        # k = 2
        edge_sum_2a = quicksum([e.eta_vars[i_measurement, 1] * (w_max - e.weight_list[i_measurement])
                                for e in e_list])
        edge_sum_2b = quicksum([e.mu_vars[i_measurement, 1] * (e.weight_list[i_measurement] - w_min)
                                for e in e_list])
        m.addConstr(b2 + (1 + gamma / alpha) * weight_vars[i_measurement] + edge_sum_2a + edge_sum_2b <=
                    s_vars[i_measurement],
                    name=("s_constr__k2_i%d" % i_measurement))

        # now for the norm sum (using the 1-norm)
        # for each edge, get |\eta_ik - \mu_ik  - a_k|. then sum all of these to obtain the 1-norm
        e_norm_plus_vars_k1 = m.addVars(len(e_list), vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY,
                                        name=("e_norm_plus_k1_i%d" % i_measurement))
        e_norm_minus_vars_k1 = m.addVars(len(e_list), vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY,
                                         name=("e_norm_minus_k1_i%d" % i_measurement))
        e_norm_plus_vars_k2 = m.addVars(len(e_list), vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY,
                                        name=("e_norm_plus_k2_i%d" % i_measurement))
        e_norm_minus_vars_k2 = m.addVars(len(e_list), vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY,
                                         name=("e_norm_minus_k2_i%d" % i_measurement))
        for i_e, e in enumerate(e_list):
            # k = 1
            m.addConstr(
                e_norm_plus_vars_k1[i_e] - e_norm_minus_vars_k1[i_e] == e.eta_vars[i_measurement, 0] - e.mu_vars[
                    i_measurement, 0] + e.used_var)
            m.addConstr(quicksum(e_norm_plus_vars_k1) + quicksum(e_norm_minus_vars_k1) <= lam_var)
            # k = 2
            m.addConstr(
                e_norm_plus_vars_k2[i_e] - e_norm_minus_vars_k2[i_e] == e.eta_vars[i_measurement, 1] - e.mu_vars[
                    i_measurement, 1] + e.used_var)
            m.addConstr(quicksum(e_norm_plus_vars_k2) + quicksum(e_norm_minus_vars_k2) <= lam_var)

    # objective
    obj = lam_var * theta + (1.0 / float(num_weight_measurements)) * quicksum(s_vars) + gamma * d_var

    m.setObjective(obj, sense=GRB.MINIMIZE)

    if not cfg.use_chains:
        raise Exception("not implemented")
    elif cfg.edge_success_prob == 1:
        pass
    else:
        raise Exception("not implemented")

    optimize(m)

    pair_edges = [e for e in cfg.digraph.es if e.used_var.x > 0.5]

    if cfg.use_chains:
        matching_chains = kidney_utils.get_optimal_chains(
            cfg.digraph, cfg.ndds, cfg.edge_success_prob)
        ndd_chain_edges = [e for ndd in cfg.ndds for e in ndd.edges if e.edge_var.x > 0.5]
    else:
        ndd_chain_edges = []
        matching_chains = []

    matching_edges = pair_edges + ndd_chain_edges

    if cfg.cardinality_restriction is not None:
        if len(matching_edges) > cfg.cardinality_restriction:
            raise Warning("cardinality restriction is violated: restriction = %d edges, matching uses %d edges" % (
                cfg.cardinality_restriction, len(matching_edges)))

    cycles_used = [c for c, v in zip(cycles, cycle_vars) if v.x > 0.5]
    cycle_obj = [c for c in cycle_list if c.grb_var.x > 0.5]

    sol = OptSolution(ip_model=m,
                      cycles=cycles_used,
                      cycle_obj=cycle_obj,
                      chains=matching_chains,
                      digraph=cfg.digraph,
                      edge_success_prob=cfg.edge_success_prob,
                      chain_restriction=cfg.chain_restriction,
                      cycle_restriction=cfg.cycle_restriction,
                      cycle_cap=cfg.max_chain,
                      chain_cap=cfg.max_cycle,
                      cardinality_restriction=cfg.cardinality_restriction)
    sol.add_matching_edges(cfg.ndds)
    kidney_utils.check_validity(sol, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain)
    return None, matching_edges
