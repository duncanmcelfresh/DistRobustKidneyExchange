'''Solving the kidney exchange problem with PICEF, PC-TSP, and PI-TSP'''

import kidney_utils

from kidney_digraph import Edge
from kidney_ndds import NddEdge

from kidney_digraph import Cycle, failure_aware_cycle_score, cycle_score
from kidney_ndds import Chain

from gurobipy import *

# from graph_tool import Graph
# from graph_tool.flow import boykov_kolmogorov_max_flow, min_st_cut

import numpy as np
import sys
import random
from guppy import hpy
import json

h = hpy()

EPS = 1e-12
EPS_mid = 0.01
EPS_large = 0.1
W = 1e7
W_small = 1e4

# for testing modifications to the formulations
test = True

relax_discount = 0.99 # total discount is 1.0

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
        verbose: True if and only if Gurobi output should be writtent to screen and log file
        timelimit
        edge_success_prob
        lp_file: The name of a .lp file to write, or None if the file should not be written
        relax: True if and only if the LP relaxation should be solved also
        gamma: uncertainty budget (robust implementation)
    """

    def __init__(self, digraph, ndds, max_cycle, max_chain, verbose=False,
                 timelimit=None, edge_success_prob=1,
                 lp_file=None, relax=False, gamma=0, cardinality_restriction=None, protection_level=0.1,
                 chain_restriction = None, cycle_restriction = None, subtour_dir = None, cycle_file=None, name = None,
                 remove_subtours=False,
                 edge_assign_seed=None,
                 wt_fair_alpha = None,
                 max_pof = None,
                 min_fair = None,
                 UH_max = None,
                 U_max = None,
                 min_UH_fair = None,
                 min_chain_len = None):
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
        self.gamma = gamma # robust uncertainty budget
        self.cardinality_restriction = cardinality_restriction
        self.chain_restriction = chain_restriction
        self.cycle_restriction = cycle_restriction
        self.protection_level = protection_level # for variable budget uncertainty: probability that U-set does not contain true edge weights
        self.name = name
        self.subtour_dir = subtour_dir
        self.subtour_file = None
        self.remove_subtours = remove_subtours
        self.edge_assign_seed=edge_assign_seed
        self.cycle_file = cycle_file
        self.wt_fair_alpha = wt_fair_alpha # weighted fairness parameter (fixed)
        self.max_pof = max_pof # for weighted fairness
        self.min_fair = min_fair # for weighted fairness
        self.UH_max = UH_max # for weighted fairness (pof bounded)
        self.U_max = U_max # for weighted fairness (pof bounded)
        self.min_UH_fair = min_UH_fair # alpha-lex fairness...
        self.min_chain_len = min_chain_len
        # find cycle file if it exists
        # if (self.cycle_dir is not None) and (self.name is not None):
        #     # find cycle pickle file matching name
        #     c_file = glob.glob(os.path.join(self.cycle_dir,self.name+'.pkl'))
        #     if len(c_file) == 1:
        #         self.cycle_file = c_file[0]
        #     elif len(c_file) > 1:
        #         raise Warning("multiple cycle files found for filename: %s" % self.name)
            # elif len(c_file) == 0:
            #     # write cycle file with this name
            #     self.cycle_file = os.path.join(self.cycle_dir, self.name + '.pkl')
            #     if self.verbose:
            #         print "No cycle file found, creating file: %s" % self.cycle_file
        # find subtour file if it exists
        if (self.subtour_dir is not None) and (self.name is not None):
            # find subtour constraint pickle file matching name
            sub_file = glob.glob(os.path.join(self.subtour_dir,self.name+'.pkl'))
            if len(sub_file) == 1:
                self.subtour_file = sub_file[0]
            elif len(sub_file) > 1:
                raise Warning("multiple subtour files found for filename: %s" % self.name)
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
        total_score: The total score of the solution
    """

    def __init__(self, ip_model, cycles, chains, digraph,
                 edge_success_prob=1,
                 infeasible=False,
                 gamma=0,
                 robust_score=0,
                 optimistic_score=0,
                 cycle_obj = None,
                 chain_restriction=None,
                 cycle_restriction=None,
                 cardinality_restriction=None,
                 cycle_cap = None,
                 chain_cap = None,
                 matching_edges=None,
                 edge_assign_seed=None,
                 wt_fair_alpha=None,
                 alpha_var = None,
                 min_chain_len = None):
        self.ip_model = ip_model
        self.cycles = cycles
        self.chains = chains
        self.digraph = digraph
        self.infeasible = infeasible
        self.min_chain_len = min_chain_len
        if self.infeasible:
            self.total_score = 0
        else:
            self.total_score = (sum(c.score for c in chains) +
                                sum(failure_aware_cycle_score(c, digraph, edge_success_prob) for c in cycles))
        self.edge_success_prob = edge_success_prob
        self.cycle_obj = cycle_obj
        self.matching_edges = matching_edges
        self.gamma = gamma
        self.robust_score = robust_score
        self.optimistic_score = optimistic_score
        self.cycle_restriction = cycle_restriction
        self.chain_restriction = chain_restriction
        self.cardinality_restriction=cardinality_restriction
        self.cycle_cap = cycle_cap
        self.chain_cap = chain_cap
        self.edge_assign_seed=edge_assign_seed
        self.wt_fair_alpha=wt_fair_alpha
        self.alpha_var = alpha_var

        if ip_model is not None:
            self.timeout = (ip_model.status == GRB.TIME_LIMIT)
        else:
            self.timeout = False

    def score_with_N_failures(self,num_fail,d,edge_success_prob=1.0):
        '''return the score of the solution when exactly N cycles or chains fail (uniformly at random)'''
        num_struct = len(self.cycles) + len(self.chains)
        if num_fail >= num_struct:
            return 0.0
        else:
            successes = random.sample(range(num_struct),num_struct-num_fail)
            scores = [c.score for c in self.chains] + [failure_aware_cycle_score(c, d, edge_success_prob) for c in self.cycles]
            return sum(scores[i] for i in successes)

    def to_dict(self):
        sol_dict = {'cycle_obj':[c.to_dict() for c in self.cycle_obj],
                    'chains':[c.to_dict() for c in self.chains],
                    'cycle_cap':self.cycle_cap,
                    'chain_cap': self.chain_cap,
                    'chain_restriction':self.chain_restriction,
                    'cycle_restriction':self.cycle_restriction,
                    'cardinality_restriction':self.cardinality_restriction,
                    'edge_assign_seed':self.edge_assign_seed,
                    'edge_success_prob':self.edge_success_prob,
                    'gamma':self.gamma,
                    'matching_edges':[e.to_dict() for e in self.matching_edges],
                    'wt_fair_alpha':self.wt_fair_alpha,
                    # 'robust_score':self.robust_score,
                    # 'optimistic_store':self.optimistic_score
                    }
        return sol_dict

    def pof(self,U_max):
        '''the price of fairness, compared with U_max'''
        if U_max == 0:
            return -1
        else:
            return (float(U_max) - self.total_score)/float(U_max)

    def frac_f(self,ndds,UH_max):
        '''fraction of the fair score (UH_max)'''
        if UH_max == 0:
            return -1
        else:
            return self.get_fair_score(ndds, unfair=False)/float(UH_max)


    @classmethod
    def from_file(cls, cfg, infile):

        with open(infile, 'r') as f:
            sol_dict = json.load(f)

        cycle_obj = [Cycle.from_dict(c,cfg.digraph) for c in sol_dict['cycle_obj']]
        cycles = [c.vs for c in cycle_obj]
        # add edges from cfg.digraph and cfg.ndds
        ndd_edges = [NddEdge.from_dict(e,cfg.ndds) for e in sol_dict['matching_edges'] if e['type']=='ndd_edge']
        pair_edges = [Edge.from_dict(e,cfg.digraph) for e in sol_dict['matching_edges'] if e['type']=='pair_edge']
        matching_edges = ndd_edges + pair_edges
        ip_model = None
        opt_sol = cls(ip_model,
                cycles,
                [Chain.from_dict(c) for c in sol_dict['chains']],
                cfg.digraph,
                cycle_obj=cycle_obj,
                edge_success_prob=sol_dict['edge_success_prob'],
                gamma=sol_dict['gamma'],
                # robust_score=sol_dict['robust_score'],
                # optimistic_score=sol_dict['optimistic_score'],
                cycle_cap = sol_dict['cycle_cap'],
                chain_cap=sol_dict['chain_cap'],
                cycle_restriction=sol_dict['cycle_restriction'],
                chain_restriction=sol_dict['chain_restriction'],
                cardinality_restriction=sol_dict['cardinality_restriction'],
                edge_assign_seed=sol_dict['edge_assign_seed'],
                matching_edges=matching_edges)

        if sol_dict.has_key('wt_fair_alpha'):
            opt_sol.wt_fair_alpha = sol_dict['wt_fair_alpha']

        # opt_sol.add_matching_edges(cfg.ndds)
        return opt_sol

    def save_to_file(self,outfile):
        out_dict = self.to_dict()
        with open(outfile,'wb') as f:
            json.dump(out_dict,f,indent=4)

    def same_matching_edges(self,other):
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

    def add_matching_edges(self,ndds):
        '''
        Set attribute 'matching_edges' using 'self.cycle_obj', 'self.chains', and 'self.digraph'
        '''
        matching_edges = []

        for ch in self.chains:
            chain_edges = []
            tgt_id = ch.vtx_indices[0]
            for e in ndds[ch.ndd_index].edges:
                if e.tgt.id == tgt_id:
                    chain_edges.append(e)
            if len(chain_edges) == 0:
                raise Warning("NDD edge not found")
            for i in range(len(ch.vtx_indices)-1):
                chain_edges.append(self.digraph.adj_mat[ch.vtx_indices[i]][ch.vtx_indices[i + 1]])
            if len(chain_edges) != (len(ch.vtx_indices)):
                raise Warning("Chain contains %d edges, but only %d edges found" %
                              (len(ch.vtx_indices), len(chain_edges)))
            matching_edges.extend(chain_edges)

        for cy in self.cycle_obj:
            cycle_edges = []
            for i in range(len(cy.vs)-1):
                cycle_edges.append(self.digraph.adj_mat[cy.vs[i].id][cy.vs[i+1].id])
            # add final edge
            cycle_edges.append(self.digraph.adj_mat[cy.vs[-1].id][cy.vs[0].id])
            if len(cycle_edges) != len(cy.vs):
                raise Warning("Cycle contains %d vertices, but only %d edges found" %
                              (len(cy.vs), len(cycle_edges)))
            matching_edges.extend(cycle_edges)

        self.matching_edges = matching_edges

    # given a weighting function weight_func(e), which takes one edge (ndd or pair) as an argument,
    # calculate the value of the solution given the new weights
    def score_with_edge_weight_func(self, weight_func):
        return sum(weight_func(e) for e in self.matching_edges)

    # after discounting certain edges
    def get_discounted_score(self):
        return sum(e.score - e.discount * e.discount_frac for e in self.matching_edges)

    # non-discounted score
    def get_total_score(self):
        return sum(e.score for e in self.matching_edges)

    # after the property 'fail' has been added, calculate the new matching score
    def score_after_edge_failure(self,ndds):
        # if the cycles does not have failed edges, add it
        cycle_score = 0
        for c in self.cycle_obj:
            if not any(e.fail for e in c.edges):
                # this needs to be the optimistic (not expected) score
                # cycle_score += c.score
                cycle_score += sum(e.score for e in c.edges)

        # calculate the truncated chain score if an edge failed
        chain_score = 0
        for ch in self.chains:
            chain_score += ch.score_after_failure(self.digraph, ndds)

        return cycle_score + chain_score

    # after the property 'fail' has been added, return the set of remaining cycles and chains
    def cy_ch_after_edge_failure(self,ndds):
        # if the cycles does not have failed edges, add it
        # cycles = []
        # for c in self.cycle_obj:
        #     if not any(e.fail for e in c.edges):
        #         cycles.append(c)
        cycles = [c for c in self.cycle_obj if not any(e.fail for e in c.edges)]

        # calculate the truncated chain score if an edge failed
        # for ch in self.chains:
        #     new_ch = chain_after_failure(ch, self.digraph, ndds)

        chains = [chain_after_failure(ch, self.digraph, ndds) for ch in self.chains]
        chains_exist = [ch for ch in chains if len(ch.vtx_indices) > 0]

        return cycles,chains_exist

    # score a certain number of edge failures
    def score_after_num_failures(self,ndds,num_fail,seed=0):
        # save the "true" failure state of each edge
        rs = np.random.RandomState(seed)
        for e in self.matching_edges:
            e.true_fail = e.fail
            e.fail = False

        # indices of failed edges:
        if num_fail > len(self.matching_edges):
            return 0
        failures = rs.choice(len(self.matching_edges), size=num_fail,replace=False)
        # set num_fail edges to fail
        for i,e in enumerate(self.matching_edges):
            if i in failures:
                e.fail = True

        # get score after failures
        score = self.score_after_edge_failure(ndds)

        for e in self.matching_edges:
            e.fail = e.true_fail
            e.true_fail = None

        return score

    def display(self):
        """Print the optimal cycles and chains to standard output."""

        print "cycle_count: {}".format(len(self.cycles))
        print "chain_count: {}".format(len(self.chains))
        print "cycles:"
        # # cs is a list of cycles, with each cycle represented as a list of vertex IDs
        # # Sort the cycles
        if len(self.cycle_obj)>0:
            for c in sorted(self.cycle_obj):
                print c.display()
        else:
            cs = [[v.id for v in c] for c in self.cycles]
            # Put the lowest-indexed vertex at the start of each cycle
            for i in range(len(cs)):
                min_index_pos = cs[i].index(min(cs[i]))
                cs[i] = cs[i][min_index_pos:] + cs[i][:min_index_pos]
                print "\t".join(str(v_id) for v_id in cs[i])
        print "chains:"
        for c in self.chains:
            print c.display()

        print "edges:"
        for e in sorted(self.matching_edges, key=lambda x: x.score, reverse=True):
            print(e.display(self.gamma))

        print "total score:"
        print self.total_score
        if self.gamma > 0:
            # print "gamma: {}".format(self.gamma)
            # print "discounted pair edges ({}):".format(len(self.discounted_pair_edges))
            # for e in self.discounted_pair_edges:
            #     print "\t".join([str(e.src.id), str(e.tgt.id)]) + "\t" + "score={}".format(e.score) + "\t" + "discount={}".format(e.discount) + "\t" + "discount_frac={}".format(e.discount_frac)
            # print "discounted NDD edges ({}):".format(len(self.discounted_ndd_edges))
            # for n,e in self.discounted_ndd_edges:
            #     print str(n) + "\t" + str(e.tgt.id) + "\t" + "score={}".format(e.score) + "\t" + "discount={}".format(e.discount) + "\t" + "discount_frac={}".format(e.discount_frac)
            # ndd_d_sum = np.sum(e.discount*e.discount_frac for _,e in self.discounted_ndd_edges)
            # pair_d_sum = np.sum(e.discount*e.discount_frac for e in self.discounted_pair_edges)
            d_sum = np.sum( e.discount * e.discount_frac for e in self.matching_edges)
            print "total discount value = {}".format(d_sum)
            print "robust matching weight: {}".format(self.robust_score)
            print "optimistic matching weight: {}".format(self.optimistic_score)
            # print "ratio of robust/optimistic: {}".format(self.robust_score/self.optimistic_score)
            # print "difference between robust/optimistic: {}".format(self.optimistic_score-self.robust_score)

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

        # added by Duncan

    def num_matched(self):
        return np.sum(self.vertex_mask())

    # added by Duncan
    # returns percentage of sensitized pairs included in solution. if graph contains no sensitized pairs, returns -1
    def num_sensitized(self):
        sens = [1 if v.sensitized else 0 for v in self.digraph.vs]
        num_sens = sens.count(1)
        if num_sens == 0:
            return -1
        else:
            return np.dot(sens, self.vertex_mask())  # for % instead: float( ) / num_sens

    def relabelled_copy(self, old_to_new_vertices, new_digraph):
        """Create a copy of the solution with vertices relabelled.

        If the solution was found on a relabelled copy of the instance digraph, this
        method can be used to transform the solution back to the original digraph. Each
        Vertex v in the OptSolution on which this method is called is replaced in the
        returned copy by old_to_new_vertices[v.id].
        """

        relabelled_cycles = [[old_to_new_vertices[v.id] for v in c] for c in self.cycles]
        relabelled_chains = [Chain(c.ndd_index,
                                   [old_to_new_vertices[i].id for i in c.vtx_indices],
                                   c.score)
                             for c in self.chains]
        return OptSolution(self.ip_model, relabelled_cycles, relabelled_chains,
                           new_digraph, self.edge_success_prob)

    # get score using a digraph with (possibly) different weights
    def get_score(self, digraph, ndds, edge_success_prob=1.0):
        score = (sum(c.get_score(digraph, ndds, edge_success_prob) for c in self.chains) +
                 sum(failure_aware_cycle_score(c, digraph, edge_success_prob) for c in self.cycles))
        return score

    def get_score_from_edges(self,sens=None):
        '''get total score from self.matching_edges
        DOES NOT CONSIDER EDGE SUCCESS PROBABILITY (assumes 1.0)'''
        if sens is not None:
            if sens==True:
                return sum(e.score for e in self.matching_edges if e.sensitized)
            if sens==False:
                return sum(e.score for e in self.matching_edges if not e.sensitized)
        else:
            return sum(e.score for e in self.matching_edges)

    # calculate and update the score, using the current ndds and digraph
    def update_score(self, ndds):
        self.total_score = self.get_score(self.digraph, ndds, self.edge_success_prob)

    def get_fair_score(self, ndds, unfair=False):
        d_fair = self.digraph.fair_copy(unfair=unfair)
        ndds_fair = [n.fair_copy(unfair=unfair) for n in ndds]
        fair_score = self.get_score(d_fair, ndds_fair, self.edge_success_prob)
        return fair_score


def optimise(model, cfg):
    if cfg.lp_file:
        model.update()
        model.write(cfg.lp_file)
        sys.exit(0)
    elif cfg.relax:
        model.update()
        r = model.relax()
        r.optimize()
        print "lp_relax_obj_val:", r.obj_val
        print "lp_relax_solver_status:", r.status
        sys.exit(0)
    else:
        model.optimize()


def optimise_relabelled(formulation_fun, cfg):
    """Optimise on a relabelled graph such that vertices are sorted in descending
        order of (indegree + outdegree)"""

    in_degs = [0] * cfg.digraph.n
    for e in cfg.digraph.es:
        in_degs[e.tgt.id] += 1

    sorted_vertices = sorted(cfg.digraph.vs,
                             key=lambda v: len(v.edges) + in_degs[v.id],
                             reverse=True)

    relabelled_digraph = cfg.digraph.induced_subgraph(sorted_vertices)

    # old_to_new_vtx[i] is the vertex in the new graph corresponding to vertex
    # i in the original digraph
    old_to_new_vtx = [None] * cfg.digraph.n
    for i, v in enumerate(sorted_vertices):
        old_to_new_vtx[v.id] = relabelled_digraph.vs[i]

    relabelled_ndds = create_relabelled_ndds(cfg.ndds, old_to_new_vtx)
    relabelled_cfg = copy.copy(cfg)
    relabelled_cfg.digraph = relabelled_digraph
    relabelled_cfg.ndds = relabelled_ndds

    opt_result = formulation_fun(relabelled_cfg)
    return opt_result.relabelled_copy(sorted_vertices, cfg.digraph)


# def create_ip_model(time_limit, verbose): # changed by Duncan
def create_ip_model(time_limit, verbose, multi=1, gap=0):
    """Create a Gurobi Model."""

    m = Model("kidney-mip")
    # m.params.mipGap = 0 # solve to optimality
    if not verbose:
        m.params.outputflag = 0
    if multi > 1:
        m.setParam(GRB.Param.PoolSolutions, multi)  # number of solutions to collect
        m.setParam(GRB.Param.PoolGap, gap)  # only collect optimal solutions (gap = 0)
        m.setParam(GRB.Param.PoolSearchMode, 2)  # exhaustive search
    if time_limit is not None:
        m.params.timelimit = time_limit
        # m.setParam('TimeLimit', time_limit)
    return m

###################################################################################################
#                                                                                                 #
#                  Chain vars and constraints (used by HPIEF', HPIEF'' and PICEF)                 #
#                                                                                                 #
###################################################################################################

def add_chain_vars_and_constraints(digraph, ndds,use_chains, max_chain, m, vtx_to_vars,
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

    if use_chains:  #max_chain > 0:
        for v in digraph.vs:
            v.grb_vars_in  = [[] for i in range(max_chain-1)]
            v.grb_vars_out = [[] for i in range(max_chain-1)]

        for ndd in ndds:
            ndd_edge_vars = []
            for e in ndd.edges:
                edge_var = m.addVar(vtype=GRB.BINARY)
                e.edge_var = edge_var
                ndd_edge_vars.append(edge_var)
                vtx_to_vars[e.tgt.id].append(edge_var)
                if max_chain>1: e.tgt.grb_vars_in[0].append(edge_var)
            m.update()
            m.addConstr(quicksum(ndd_edge_vars) <= 1)

        dists_from_ndd = kidney_utils.get_dist_from_nearest_ndd(digraph, ndds)

        # Add pair->pair edge variables, indexed by position in chain
        for e in digraph.es:
            e.grb_vars = []
            if store_edge_positions:
                e.grb_var_positions = []
            for i in range(max_chain-1):
                if dists_from_ndd[e.src.id] <= i+1:
                    edge_var = m.addVar(vtype=GRB.BINARY)
                    e.grb_vars.append(edge_var)
                    if store_edge_positions:
                        e.grb_var_positions.append(i+1)
                    vtx_to_vars[e.tgt.id].append(edge_var)
                    e.src.grb_vars_out[i].append(edge_var)
                    if i < max_chain-2:
                        e.tgt.grb_vars_in[i+1].append(edge_var)
            # m.update()

        # At each chain position, sum of edges into a vertex must be >= sum of edges out
        for i in range(max_chain-1):
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

    cycles = cfg.digraph.find_cycles(cfg.max_cycle,cycle_file=cfg.cycle_file)

    m = create_ip_model(cfg.timelimit, cfg.verbose)
    m.params.method = -1

    cycle_vars = [m.addVar(vtype=GRB.BINARY) for __ in cycles]
    # m.update()

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
    pair_edge_count = [e.used_var for e in cfg.digraph.es ]
    if cfg.use_chains:
        ndd_edge_count = [e.edge_var for ndd in cfg.ndds for e in ndd.edges ]
        m.addConstr( num_edges_var == quicksum(pair_edge_count + ndd_edge_count))
    else:
        m.addConstr( num_edges_var == quicksum(pair_edge_count))


    # add a cardinality restriction if necessary
    if cfg.cardinality_restriction is not None:
        m.addConstr( num_edges_var <= cfg.cardinality_restriction )


    m.update()

    return m,cycles, cycle_vars, num_edges_var

def optimise_picef(cfg):

    m,cycles, cycle_vars, _ = create_picef_model(cfg)

    # add cycle objects
    cycle_list = []
    for c,var in zip(cycles,cycle_vars):
        c_obj = Cycle(c)
        c_obj.add_edges(cfg.digraph.es)
        c_obj.score = failure_aware_cycle_score(c_obj.vs, cfg.digraph,cfg.edge_success_prob)
        c_obj.grb_var = var
        cycle_list.append(c_obj)

    if not cfg.use_chains:# cfg.max_chain == 0:
        obj_expr = quicksum(failure_aware_cycle_score(c, cfg.digraph, cfg.edge_success_prob) * var
                            for c, var in zip(cycles, cycle_vars))
    elif cfg.edge_success_prob == 1:
        obj_expr = (quicksum(cycle_score(c, cfg.digraph) * var for c, var in zip(cycles, cycle_vars)) +
                    quicksum(e.score * e.edge_var for ndd in cfg.ndds for e in ndd.edges) +
                    quicksum(e.score * var for e in cfg.digraph.es for var in e.grb_vars))
    else:
        obj_expr = (quicksum(failure_aware_cycle_score(c, cfg.digraph, cfg.edge_success_prob) * var
                             for c, var in zip(cycles, cycle_vars)) +
                    quicksum(e.score * cfg.edge_success_prob * e.edge_var
                             for ndd in cfg.ndds for e in ndd.edges) +
                    quicksum(e.score * cfg.edge_success_prob ** (pos + 1) * var
                             for e in cfg.digraph.es for var, pos in zip(e.grb_vars, e.grb_var_positions)))
    m.setObjective(obj_expr, GRB.MAXIMIZE)

    # print "current memory usage (picef): %f GB" % (h.heap().size/1e9)
    # print "size of model (picef): %f GB" % (sys.getsizeof(m)/1e9)

    optimise(m, cfg)


    # if m.status == GRB.status.INFEASIBLE:
    #     return OptSolution(ip_model=m,
    #                        cycles=[],
    #                        chains=[],
    #                        digraph=cfg.digraph,
    #                        edge_success_prob=cfg.edge_success_prob,
    #                        infeasible=True)
    # else:

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
            raise Warning("cardinality restriction is violated: restriction = %d edges, matching uses %d edges" % (cfg.cardinality_restriction, len(matching_edges)))

    cycles_used = [c for c, v in zip(cycles, cycle_vars) if v.x > 0.5]
    cycle_obj = [c for c in cycle_list if c.grb_var.x > 0.5]

    sol = OptSolution(ip_model=m,
                       cycles=cycles_used,
                       cycle_obj = cycle_obj,
                       chains=matching_chains,
                       digraph=cfg.digraph,
                       edge_success_prob=cfg.edge_success_prob,
                       chain_restriction=cfg.chain_restriction,
                       cycle_restriction=cfg.cycle_restriction,
                       cycle_cap=cfg.max_chain,
                       chain_cap=cfg.max_cycle,
                      cardinality_restriction=cfg.cardinality_restriction,
                      edge_assign_seed=cfg.edge_assign_seed
                      )
                       # matching_edges=matching_edges)
    sol.add_matching_edges(cfg.ndds)
    kidney_utils.check_validity(sol, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain)
    return sol

def max_cycles(cfg):
    '''
    Use PICEF to find the maximum number of cycles in a matching...
    '''
    m,_, cycle_vars, _ = create_picef_model(cfg)

    num_cycles = quicksum(cycle_vars)

    m.setObjective(num_cycles, GRB.MAXIMIZE)

    optimise(m, cfg)
    if cfg.verbose:
        print "maximum number of cycles = %d" % m.objVal
    if m.objVal != int(m.objVal):
        raise Warning("number of cycles is not integer")
    return int(m.objVal)


def optimise_robust_picef(cfg):

    m,cycles, cycle_vars, num_edges_var = create_picef_model(cfg)

    # for use later
    floor_gamma = np.floor(cfg.gamma)
    ceil_gamma = np.ceil(cfg.gamma)
    gamma_frac = cfg.gamma - floor_gamma


    # add cycle vars
    cycle_list = []
    for c,var in zip(cycles,cycle_vars):
        c_obj = Cycle(c)
        c_obj.add_edges(cfg.digraph.es)
        c_obj.score = cycle_score(c_obj.vs, cfg.digraph)
        c_obj.grb_var = var
        cycle_list.append(c_obj)

    m.update()

    # add ordering indicator variables g and discount indicators d ( d = y*g ) for NDD edges

    if gamma_frac == 0: # gamma is integer

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

    else: # gamma is not integer

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
    e_sorted = sorted(all_edges, key=lambda e: e.discount, reverse=False)

    # ordering constraints over g
    if gamma_frac == 0: # gamma is integer
        for i in range(len(e_sorted)-1):
            m.addConstr(e_sorted[i].g_var <= e_sorted[i+1].g_var)
    else: # gamma is not integer
        for i in range(len(e_sorted) - 1):
            m.addConstr(e_sorted[i].gf_var <= e_sorted[i + 1].gf_var)
            m.addConstr(e_sorted[i].gp_var <= e_sorted[i + 1].gp_var)

    # number of edges used in matching (include all position-indexed vars)

    # uncertainty budget (number of discounted edges)
    gamma_var = m.addVar(vtype=GRB.CONTINUOUS)
    m.addGenConstrMin( gamma_var, [num_edges_var,cfg.gamma])

    # add a cardinality restriction if necessary
    if cfg.cardinality_restriction is not None:
        m.addConstr( num_edges_var <= cfg.cardinality_restriction )

    m.update()

    # limit number of discounted variables
    if gamma_frac == 0: # gamma is integer
        m.addConstr( quicksum(e.d_var for e in all_edges) == gamma_var )
    else: # gamma is not integer
        # introduce gamma_frac_var: if gamma_var is integer, gamma_frac_var = 0, otherwise, gamma_frac_var = 1
        # gamma_frac_var = m.addVar(vtype=GRB.BINARY)
        # # if gamma_var is integer, gamma - gamma_var > 0, otherwise gamma - gamma_var = 0
        # m.addConstr( cfg.gamma - gamma_var <= W_small * gamma_frac_var ) # if "!=",gamma_frac_var = 1, (otherwise 0 or 1)
        # m.addConstr( gamma_frac_var <=  W_small* (cfg.gamma - gamma_var) ) # if "==", gamma_frac_var = 0 (otherwise, 0 or 1)

        h_var = m.addVar(vtype=GRB.BINARY)
        m.addConstr( cfg.gamma - num_edges_var <= W_small*h_var )
        m.addConstr( num_edges_var - cfg.gamma <= W_small*(1-h_var) )
        m.addConstr( quicksum(e.dp_var for e in all_edges) == h_var * num_edges_var + (1-h_var)*ceil_gamma )
        m.addConstr( quicksum(e.df_var for e in all_edges) == h_var * num_edges_var + (1-h_var)*floor_gamma )

    # total discount (by edge)
    if gamma_frac == 0: # gamma is integer
        total_discount = quicksum( e.discount * e.d_var for e in all_edges)
    else: # gamma is not integer
        total_discount = quicksum( ( 1 - gamma_frac ) * e.discount * e.df_var for e in all_edges) + \
                        quicksum( gamma_frac * e.discount * e.dp_var for e in all_edges)

    # set a variable for the total (optimistic matching weight)
    total_weight = m.addVar(vtype=GRB.CONTINUOUS)

    m.update()

    if not cfg.use_chains:
         m.addConstr( total_weight == quicksum(failure_aware_cycle_score(c, cfg.digraph, cfg.edge_success_prob) * var
                            for c, var in zip(cycles, cycle_vars)))
         obj_expr = total_weight - total_discount
    elif cfg.edge_success_prob == 1:
        m.addConstr( total_weight == (quicksum(cycle_score(c, cfg.digraph) * var for c, var in zip(cycles, cycle_vars)) +
                    quicksum(e.score * e.edge_var for ndd in cfg.ndds for e in ndd.edges) +
                    quicksum(e.score * var for e in cfg.digraph.es for var in e.grb_vars)) )
        obj_expr =  total_weight - total_discount
    # else:
    #     obj_expr = (quicksum(failure_aware_cycle_score(c, cfg.digraph, cfg.edge_success_prob) * var
    #                          for c, var in zip(cycles, cycle_vars)) +
    #                 quicksum(e.score * cfg.edge_success_prob * e.edge_var
    #                          for ndd in cfg.ndds for e in ndd.edges) +
    #                 quicksum(e.score * cfg.edge_success_prob ** (pos + 1) * var
    #                          for e in cfg.digraph.es for var, pos in zip(e.grb_vars, e.grb_var_positions)))

    m.setObjective(obj_expr, GRB.MAXIMIZE)

    optimise(m, cfg)

    # if m.status == GRB.status.INFEASIBLE:
    #     return OptSolution(ip_model=m,
    #                        cycles=[],
    #                        chains=[],
    #                        digraph=cfg.digraph,
    #                        edge_success_prob=cfg.edge_success_prob,
    #                        infeasible=True)
    # else:


    if gamma_frac == 0: # gamma is integer
        discounted_pair_edges = [ e for e in cfg.digraph.es if e.d_var.x > 0]

        for e in discounted_pair_edges:
            e.discount_frac = e.d_var.x

        if cfg.use_chains:
            discounted_ndd_edges = [ (i_ndd, e) for i_ndd,ndd in enumerate(cfg.ndds) for e in ndd.edges if e.d_var.x > 0.0]

            for _,e in discounted_ndd_edges:
                e.discount_frac = e.d_var.x
    else: # gamma is not integer

        discounted_pair_edges = [e for e in cfg.digraph.es \
                                 if ((e.df_var.x > 0.0) or (e.dp_var.x > 0.0))]

        for e in discounted_pair_edges:
            e.discount_frac = (1-gamma_frac) * e.df_var.x + gamma_frac * e.dp_var.x

        if cfg.use_chains:
            discounted_ndd_edges = [(i_ndd, e) for i_ndd, ndd in enumerate(cfg.ndds) for e in ndd.edges \
                                    if ((e.df_var.x > 0.0) or (e.dp_var.x > 0.0))]
            for _, e in discounted_ndd_edges:
                e.discount_frac = (1-gamma_frac) * e.df_var.x  + gamma_frac * e.dp_var.x


    if cfg.use_chains:
        ndd_matching_edges = [e for ndd in cfg.ndds for e in ndd.edges if e.edge_var.x > 0.5 ]
    else:
        ndd_matching_edges = []

    used_matching_edges = [ e for e in cfg.digraph.es if e.used_var.x > 0.5]

    matching_edges = ndd_matching_edges + used_matching_edges

    if cfg.cardinality_restriction is not None:
        if len(matching_edges) > cfg.cardinality_restriction:
            raise Warning("cardinality restriction is violated: restriction = %d edges, matching uses %d edges" % (cfg.cardinality_restriction, len(matching_edges)))

    chains_used = kidney_utils.get_optimal_chains(cfg.digraph, cfg.ndds, cfg.edge_success_prob) if cfg.use_chains \
        else []

    cycles_used = [c for c, v in zip(cycles, cycle_vars) if v.x > 0.5]
    cycle_obj = [c for c in cycle_list if c.grb_var.x > 0.5]

    sol = OptSolution(ip_model=m,
                       cycles=cycles_used,
                       cycle_obj=cycle_obj,
                       chains= chains_used,
                       digraph=cfg.digraph,
                       edge_success_prob=cfg.edge_success_prob,
                       gamma=cfg.gamma,
                       robust_score= m.objVal,
                       optimistic_score = total_weight.x,
                      chain_restriction=cfg.chain_restriction,
                      cycle_restriction=cfg.cycle_restriction,
                      cycle_cap=cfg.max_chain,
                      chain_cap=cfg.max_cycle,
                      cardinality_restriction=cfg.cardinality_restriction,
                      edge_assign_seed=cfg.edge_assign_seed
                      )
                       # matching_edges=matching_edges)
    sol.add_matching_edges(cfg.ndds)
    kidney_utils.check_validity(sol, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain)

    return sol

def optimise_weighted_fairness(cfg):

    m,cycles, cycle_vars, num_edges_var = create_picef_model(cfg)

    if cfg.max_pof is not None:
        max_pof = cfg.max_pof
        alpha_var = m.addVar(vtype=GRB.CONTINUOUS)
        pos_weighted_fairness = True
        if cfg.UH_max is None:
            raise Warning("cfg.UH_max is required")
    elif cfg.min_fair is not None:
        alpha_var = m.addVar(vtype=GRB.CONTINUOUS)
        pos_weighted_fairness = True
        if cfg.UH_max is None:
            raise Warning("cfg.UH_max is required")
    elif cfg.wt_fair_alpha is not None:
        pos_weighted_fairness = (cfg.wt_fair_alpha > 0)
        neg_weighted_fairness = (cfg.wt_fair_alpha < 0)
        alpha = abs(cfg.wt_fair_alpha)
        alpha_var = m.addVar(vtype=GRB.CONTINUOUS,lb=alpha,ub=alpha)
    elif cfg.min_UH_fair is not None:
        alpha_lex_fairness = True
    else:
        raise Warning("Weight parameter alpha is required : cfg.{ wt_fair_alpha | max_pof | min_fair }")

    # add cycle vars
    cycle_list = []
    for c,var in zip(cycles,cycle_vars):
        c_obj = Cycle(c)
        c_obj.add_edges(cfg.digraph.es)
        c_obj.score = cycle_score(c_obj.vs, cfg.digraph)
        c_obj.grb_var = var
        cycle_list.append(c_obj)

    m.update()

    # add a cardinality restriction if necessary
    if cfg.cardinality_restriction is not None:
        m.addConstr( num_edges_var <= cfg.cardinality_restriction )

    m.update()

    # set a variable for the total (optimistic matching weight)
    total_weight = m.addVar(vtype=GRB.CONTINUOUS)

    m.update()


    # utilities for H and L
    d_fair = cfg.digraph.fair_copy()
    d_unfair = cfg.digraph.fair_copy(unfair=True)

    if cfg.max_chain == 0:
        UH = quicksum(failure_aware_cycle_score(c, d_fair, cfg.edge_success_prob) * var
                      for c, var in zip(cycles, cycle_vars))
    elif cfg.edge_success_prob == 1:
        UH = (quicksum(cycle_score(c, d_fair) * var for c, var in zip(cycles, cycle_vars)) +
              quicksum(e.score * e.edge_var * e.sensitized for ndd in cfg.ndds for e in ndd.edges) +
              quicksum(e.score * var * e.sensitized for e in cfg.digraph.es for var in e.grb_vars))
    else:
        UH = (quicksum(failure_aware_cycle_score(c, d_fair, cfg.edge_success_prob) * var
                       for c, var in zip(cycles, cycle_vars)) +
              quicksum(e.score * cfg.edge_success_prob * e.edge_var * e.sensitized
                       for ndd in cfg.ndds for e in ndd.edges) +
              quicksum(e.score * cfg.edge_success_prob ** (pos + 1) * var * e.sensitized
                       for e in cfg.digraph.es for var, pos in zip(e.grb_vars, e.grb_var_positions)))

    if cfg.max_chain == 0:
        UL = quicksum(failure_aware_cycle_score(c, d_unfair, cfg.edge_success_prob) * var
                      for c, var in zip(cycles, cycle_vars))
    elif cfg.edge_success_prob == 1:
        UL = (quicksum(cycle_score(c, d_unfair) * var for c, var in zip(cycles, cycle_vars)) +
              quicksum(e.score * e.edge_var * (1 - e.sensitized) for ndd in cfg.ndds for e in ndd.edges) +
              quicksum(e.score * var * (1 - e.sensitized) for e in cfg.digraph.es for var in e.grb_vars))
    else:
        UL = (quicksum(failure_aware_cycle_score(c, d_unfair, cfg.edge_success_prob) * var
                       for c, var in zip(cycles, cycle_vars)) +
              quicksum(e.score * cfg.edge_success_prob * e.edge_var * (1 - e.sensitized)
                       for ndd in cfg.ndds for e in ndd.edges) +
              quicksum(e.score * cfg.edge_success_prob ** (pos + 1) * var * (1 - e.sensitized)
                       for e in cfg.digraph.es for var, pos in zip(e.grb_vars, e.grb_var_positions)))

    # limit alpha by max pof
    if cfg.max_pof is not None:
        # UL = m.addVar(vtype=GRB.CONTINUOUS)  # utility to lowly sensitized patients
        # m.addConstr(UL == quicksum(e.score * e.used_var for e in cfg.digraph.es if not e.sensitized)
        #             + quicksum(e.score * e.edge_var for ndd in cfg.ndds for e in ndd.edges if not e.sensitized))
        # UH = m.addVar(vtype=GRB.CONTINUOUS)  # utility to highly sensitized patients
        # m.addConstr(UH == quicksum(e.score * e.used_var for e in cfg.digraph.es if e.sensitized) \
        #             + quicksum(e.score * e.edge_var for ndd in cfg.ndds for e in ndd.edges if e.sensitized))
        # bound alpha
        # attempt 1:
        U_ratio = m.addVar(vtype=GRB.CONTINUOUS,lb=0)
        m.addConstr(U_ratio * cfg.UH_max == UL )
        # m.addConstr(U_ratio * cfg.UH_max <= UL )
        m.addConstr(alpha_var == (max_pof / (1.0 - max_pof)) * (1.0 + U_ratio))
        # m.addConstr(alpha_var == (max_pof / (1.0 - max_pof)) * (1.0 + U_ratio))

        # attempt 2:
        # U_ratio_2 = m.addVar(vtype=GRB.CONTINUOUS,lb=1)
        # m.addConstr(U_ratio_2 * cfg.UH_max == UH+UL )
        # m.addConstr(alpha_var == (max_pof / (1.0 - max_pof)) * U_ratio_2)

        # attempt 3:
        # m.addConstr(alpha_var * UH <= max_pof * 10)

        # U_ratio_var = m.addVar(vtype=GRB.CONTINUOUS,lb=0)
        # m.addGenConstrMax(U_ratio_var,[1,U_ratio])
        # x1 = m.addVar(vtype=GRB.CONTINUOUS)
        # m.addConstr(x1 == (U_ratio + UH)/2)
        # x2 = m.addVar(vtype=GRB.CONTINUOUS)
        # m.addConstr(x2 == (U_ratio - UH)/2)
        # m.addQConstr(x1 * x1 - x2*x2 <= UL)
        # m.addQConstr(UL <= x1 * x1 - x2*x2 )

        # this doesn't work, because it's a quadratic equality constraint...
        # m.addQConstr(UL == U_ratio * UH) # define U_ratio =  UL/UH (i.e. UH * U_ratio == UL)
        # m.addQConstr(U_ratio * UH <= UL)


    elif cfg.min_fair is not None:
        # UL = m.addVar(vtype=GRB.CONTINUOUS)  # utility to lowly sensitized patients
        # m.addConstr(UL == quicksum(e.score * e.used_var for e in cfg.digraph.es if not e.sensitized)
        #             + quicksum(e.score * e.edge_var for ndd in cfg.ndds for e in ndd.edges if not e.sensitized))
        # bound alpha
        m.addConstr(alpha_var == UL/(cfg.UH_max*(1.0-cfg.min_fair))-1.0)
    # alpha is fixed

    # alpha-lex fairness
    if cfg.min_UH_fair is not None:

      m.addConstr(UH >= cfg.min_UH_fair)

      for e in cfg.digraph.es:
          if e.sensitized:
              e.alpha = 0
          else:
              e.alpha = 0

      for n in cfg.ndds:
          for e in n.edges:
              if e.sensitized:
                  e.alpha = 0
              else:
                  e.alpha = 0

    elif pos_weighted_fairness: # positive weighted fairness (fixed alpha)
        # priority_weight = alpha_var * ( quicksum( e.score * e.used_var for e in cfg.digraph.es if e.sensitized)
        #                     + quicksum(e.score * e.edge_var for ndd in cfg.ndds for e in ndd.edges if e.sensitized))

        for e in cfg.digraph.es:
            if e.sensitized:
                e.alpha = alpha_var + 0.01
            else:
                e.alpha = 0

        for n in cfg.ndds:
            for e in n.edges:
                if e.sensitized:
                    e.alpha = alpha_var + 0.01
                else:
                    e.alpha = 0

    elif neg_weighted_fairness: # negative weighted fairness (fixed alpha)
        # priority_weight = -1.0 * alpha_var * ( quicksum( e.score * e.used_var for e in cfg.digraph.es if not e.sensitized) \
        #                     + quicksum(e.score * e.edge_var for ndd in cfg.ndds for e in ndd.edges if not e.sensitized))
        for e in cfg.digraph.es:
            if e.sensitized:
                e.alpha = 0
            else:
                e.alpha = alpha_var + 0.01

        for n in cfg.ndds:
            for e in n.edges:
                if e.sensitized:
                    e.alpha = 0
                else:
                    e.alpha = alpha_var + 0.01

    if not cfg.use_chains:
         m.addConstr( total_weight == quicksum(failure_aware_cycle_score_weighted(c, cfg.digraph, cfg.edge_success_prob) * var
                            for c, var in zip(cycles, cycle_vars)))
         # obj_expr = total_weight  # + priority_weight
    elif cfg.edge_success_prob == 1:
        m.addConstr( total_weight == (quicksum(cycle_score_weighted(c, cfg.digraph) * var for c, var in zip(cycles, cycle_vars)) +
                    quicksum( e.score * (1.0 + e.alpha) * e.edge_var for ndd in cfg.ndds for e in ndd.edges) +
                    quicksum(e.score * (1.0 + e.alpha)  * var for e in cfg.digraph.es for var in e.grb_vars)) )
        # obj_expr =  total_weight # + priority_weight
    else:
        total_weight = (quicksum(failure_aware_cycle_score_weighted(c, cfg.digraph, cfg.edge_success_prob) * var
                             for c, var in zip(cycles, cycle_vars)) +
                    quicksum(e.score * (1.0 + e.alpha) * cfg.edge_success_prob * e.edge_var
                             for ndd in cfg.ndds for e in ndd.edges) +
                    quicksum(e.score * (1.0 + e.alpha)* cfg.edge_success_prob ** (pos + 1) * var
                             for e in cfg.digraph.es for var, pos in zip(e.grb_vars, e.grb_var_positions)))

    m.setObjective(total_weight, GRB.MAXIMIZE)

    optimise(m, cfg)

    if m.SolCount < 1:
        raise Warning("no feasible solutions")

    # if cfg.use_chains:
    #     ndd_matching_edges = [e for ndd in cfg.ndds for e in ndd.edges if e.edge_var.x > 0.5 ]
    # else:
    #     ndd_matching_edges = []

    # used_matching_edges = [ e for e in cfg.digraph.es if e.used_var.x > 0.5]
    #
    # matching_edges = ndd_matching_edges + used_matching_edges
    #
    # if cfg.cardinality_restriction is not None:
    #     if len(matching_edges) > cfg.cardinality_restriction:
    #         raise Warning("cardinality restriction is violated: restriction = %d edges, matching uses %d edges" % (cfg.cardinality_restriction, len(matching_edges)))

    chains_used = kidney_utils.get_optimal_chains(cfg.digraph, cfg.ndds, cfg.edge_success_prob) if cfg.use_chains \
        else []

    cycles_used = [c for c, v in zip(cycles, cycle_vars) if v.x > 0.5]
    cycle_obj = [c for c in cycle_list if c.grb_var.x > 0.5]

    sol = OptSolution(ip_model=m,
                       cycles=cycles_used,
                       cycle_obj=cycle_obj,
                       chains= chains_used,
                       digraph=cfg.digraph,
                       edge_success_prob=cfg.edge_success_prob,
                      cycle_cap=cfg.max_chain,
                      chain_cap=cfg.max_cycle,
                      wt_fair_alpha=cfg.wt_fair_alpha,
                      alpha_var = None if cfg.min_UH_fair is not None else alpha_var.x)

    sol.add_matching_edges(cfg.ndds)
    kidney_utils.check_validity(sol, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain)

    return sol

def solve_edge_weight_uncertainty(cfg, max_card=0):
    '''
    Solves the robust kidney exchange problem with a variable-budget edge weight uncertainty.
        - uses the cardinality-restriction method of Poss.
        - uses the constant-budget edge-weight-uncertainty robust formulation of PICEF

    inputs:
        - cfg               : OptConfig object
        - max_card          : maximum number of edges in a feasible solution
    '''

    # define gamma (variable uncertainty budget) function
    gamma_func = lambda x_norm: kidney_utils.gamma_symmetric_edge_weights(x_norm,cfg.protection_level)

    if cfg.verbose:
        print "solving edge weight uncertainty "

    if max_card == 0:
        # find maximum-cardinality solution (cardinality = edge count)
        d_uniform = cfg.digraph.uniform_copy()
        ndds_uniform = [ n.uniform_copy() for n in cfg.ndds ]
        cfg_maxcard = OptConfig(d_uniform, ndds_uniform, cfg.max_cycle, cfg.max_chain, cfg.verbose,
                                   cfg.timelimit, cfg.edge_success_prob,gamma=0)
        sol_maxcard = optimise_picef(cfg_maxcard)

        # the number of edges in the maximum cardinality solution
        max_card = len(sol_maxcard.matching_edges)

    if cfg.verbose:
        print "maximum cardinality = %d" % max_card

    # now find all card-restricted solutions to the constant-budget robust problem,
    # and take the best one

    best_gamma = 0

    # if there is no feasible solution...
    if max_card == 0:
        sol_maxcard.max_card = 0
        return sol_maxcard

    for card_restriction in range(1,max_card+1):

        # solve the k-cardinality-restricted problem, with Gamma = gamma(k)
        cfg.cardinality_restriction = card_restriction
        cfg.gamma = gamma_func(card_restriction)
        if cfg.gamma == 0:
            new_sol = optimise_picef(cfg)
            new_sol.robust_score = new_sol.total_score
            new_sol.optimistic_score = new_sol.total_score
        else:
            new_sol = optimise_robust_picef(cfg)

        if cfg.verbose:
            print "%d edges; gamma = %f; robust obj = %f" % (card_restriction, cfg.gamma, new_sol.robust_score)

        if card_restriction == 1:
            best_sol = new_sol
            best_gamma = cfg.gamma
        elif new_sol.robust_score > best_sol.robust_score:
                best_sol = new_sol
                best_gamma = cfg.gamma

    # return the best solution and save the best gamma value
    cfg.gamma = best_gamma
    best_sol.max_card = max_card
    return best_sol






###################################################################################################
#                                                                                                 #
#                                             PC-TSP                                              #
#                                                                                                 #
###################################################################################################

def optimise_pctsp(cfg):
    """Optimise using the PC-TSP formulation.

    Args:
        cfg: an OptConfig object

    Returns:
        an OptSolution object
    """

    # initialize PC-TSP model
    m = create_pctsp_model(cfg) # ,cycles, cycle_vars
    # if chains are used, solve using constraint generation
    # otherwise, it's just cycle formulation, so optimize directly
    if (cfg.use_chains) and (not cfg.remove_subtours):

        try:
            pctsp_constraint_gen(cfg,m)
        except MaxIterConstraintGenException as ex:
            # just remove the subtours if constraint generation takes too long...
            cfg.remove_subtours = True
            optimise(m, cfg)

        chains_used = kidney_utils.get_optimal_chains_pctsp(cfg.digraph, cfg.ndds)
        # find edges used
        chain_edges_pair = [e for e in cfg.digraph.es if e.edge_var.x > 0.5]
        chain_edges_ndd = [e for n in cfg.ndds for e in n.edges if e.edge_var.x > 0.5]
        chain_edges_used = chain_edges_pair + chain_edges_ndd

    elif cfg.use_chains and cfg.remove_subtours:
        optimise(m, cfg)

        chains_used = kidney_utils.get_optimal_chains_pctsp(cfg.digraph, cfg.ndds)
        # find edges used
        chain_edges_pair = [e for e in cfg.digraph.es if e.edge_var.x > 0.5]
        chain_edges_ndd = [e for n in cfg.ndds for e in n.edges if e.edge_var.x > 0.5]
        chain_edges_used = chain_edges_pair + chain_edges_ndd
    else:
        optimise(m, cfg)
        chains_used = []
        chain_edges_used = []

    # print "current memory usage (pc-tsp): %f GB" % (h.heap().size/1e9)
    # print "size of model (pc-tsp): %f GB" % (sys.getsizeof(m)/1e9)

    # find all cycles used in the matching
    cycles_used = [ c for c in cfg.digraph.cycles if c.grb_var.x > 0.5]

    matching_edges = chain_edges_used + [e for c in cycles_used for e in c.edges ]

    # check cardinality restrictions
    if cfg.use_chains and  (cfg.chain_restriction is not None):
        if len(chains_used) > cfg.chain_restriction:
            raise Warning("chain restriction violated: # chains = %d, restriction = %d"
                          % (len(chains_used),cfg.chain_restriction))
    if cfg.cycle_restriction is not None:
        if len(cycles_used) > cfg.cycle_restriction:
            raise Warning("cycle restriction violated: # cycles = %d, restriction = %d"
                              % (len(cycles_used),cfg.cycle_restriction))

    sol = OptSolution(ip_model=m,
                       cycles=[c.vs for c in cfg.digraph.cycles if c.grb_var.x > 0.5],
                       cycle_obj=cycles_used,
                       chains=chains_used,
                       digraph=cfg.digraph,
                       edge_success_prob=cfg.edge_success_prob,
                      chain_restriction=cfg.chain_restriction,
                      cycle_restriction=cfg.cycle_restriction,
                      cycle_cap=cfg.max_chain,
                      chain_cap=cfg.max_cycle,
                      cardinality_restriction=cfg.cardinality_restriction,
                      edge_assign_seed=cfg.edge_assign_seed
                      )
                       # matching_edges=matching_edges)
    sol.add_matching_edges(cfg.ndds)
    kidney_utils.check_validity(sol, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain)

    return sol

def create_pitsp_model(cfg):
    """Create the PI-TSP IP model

    Args:
        cfg: an OptConfig object

    Returns:
        m: a gurobipy model for PC-TSP

        vtx_to_vars: A list such that for each Vertex v in the Digraph,
            vtx_to_vars[v.id] will contain the Gurobi variables representing
            edges pointing to v.
    """

    # create graph-tool graph
    # digraph_GT, v_GT, v_src, v_map, e_map, e_weight = create_GT_digraph(cfg)

    # cfg.remove_subtours = True


    all_cycles = cfg.digraph.find_cycles(cfg.max_cycle,cycle_file=cfg.cycle_file)
    cycle_list = [Cycle(c) for c in all_cycles]

    m = create_ip_model(cfg.timelimit, cfg.verbose)

    m.params.method = -1 # 2 EDIT: CHANGED FROM 2
    # m.setParam(GRB.Param.MIPFocus, 2)
    # add cycle vars
    for c in cycle_list:
        # c.add_edges(cfg.digraph.es)
        c.score = cycle_score(c.vs, cfg.digraph)
        c.grb_var = m.addVar(vtype=GRB.BINARY)

    cfg.digraph.cycles = cycle_list

    m.update()

    if cfg.use_chains: #cfg.max_chain > 0:

        # assign edge in/out variables for each vertex, for each chain (ndd)
        for v in cfg.digraph.vs:
            v.grb_vars_in_ndd  = [[] for _ in range(len(cfg.ndds))]
            v.grb_vars_out_ndd = [[] for _ in range(len(cfg.ndds))]
            v.es_in  = []
            v.es_out = []

        for i_ndd,ndd in enumerate(cfg.ndds):
            ndd_edge_vars = []
            for e in ndd.edges:
                edge_var = m.addVar(vtype=GRB.BINARY)
                edge_pos_var = m.addVar(vtype=GRB.CONTINUOUS,lb=1,ub=1)
                e.edge_var = edge_var
                e.edge_pos_var = edge_pos_var
                e.edge_pos_var_prime = e.edge_var # ADDED BY DUNCAN FOR POSITION CONSTRAINTS
                ndd_edge_vars.append(edge_var)
                e.tgt.grb_vars_in_ndd[i_ndd].append(edge_var)
                e.tgt.es_in.append(e) # ADDED BY DUNCAN: FOR POSITION CONSTRAINTS
            ndd_used = m.addVar(vtype=GRB.CONTINUOUS) # GRB.BINARY # (changed from binary by Duncan)
            m.addConstr(ndd_used == quicksum(e.edge_var for e in ndd.edges))
            m.addConstr(ndd_used <= 1)
            ndd.used_var = ndd_used
            m.update()

        kidney_utils.find_vertex_chain_participation(cfg.digraph, cfg.ndds, cfg.max_chain)

        # add pair-pair edge vars for each NDD, and each edge overall (y^n_e and y_e)
        for e in cfg.digraph.es:
            e.edge_var_ndd = []
            if any(e.src.can_be_in_chain_list) and any(e.tgt.can_be_in_chain_list):
                edge_var = m.addVar(vtype=GRB.CONTINUOUS)
                edge_pos_var = m.addVar(vtype=GRB.CONTINUOUS)
            else:
                edge_var = m.addVar(vtype=GRB.CONTINUOUS,lb=0,ub=0)
                edge_pos_var = m.addVar(vtype=GRB.CONTINUOUS,ub=1,lb=1)

            e.edge_var = edge_var
            e.edge_pos_var = edge_pos_var
            e.tgt.es_in.append(e)  # ADDED BY DUNCAN: FOR POSITION CONSTRAINTS
            e.src.es_out.append(e)  # ADDED BY DUNCAN: FOR POSITION CONSTRAINTS
            for i_ndd,ndd in enumerate(cfg.ndds):
                edge_var_n = m.addVar(vtype=GRB.BINARY)
                e.tgt.grb_vars_in_ndd[i_ndd].append(edge_var_n)
                e.src.grb_vars_out_ndd[i_ndd].append(edge_var_n)
                e.edge_var_ndd.append(edge_var_n)

            # set edge vars using edge_var_ndds (sum over all NDDs)
            m.addConstr( quicksum(e.edge_var_ndd) == e.edge_var )
            # ADDED BY DUNCAN FOR EDGE POSITIONS:
            # add edge_pos_var_prime = e.edge_pos_var* e.edge_var
            e_vp = m.addVar(vtype=GRB.CONTINUOUS,lb = 0)
            m.addConstr( e_vp <= e.edge_var * (len(cfg.digraph.vs)+1))
            m.addConstr( e_vp <= e.edge_pos_var)
            m.addConstr( e.edge_pos_var - (1 - e.edge_var)*(len(cfg.digraph.vs)+1) <= e_vp)
            e.edge_pos_var_prime = e_vp


        m.update()

        # define pair-pair flow
        for v in cfg.digraph.vs:
            flow_in = m.addVar(vtype=GRB.CONTINUOUS) # GRB.CONTINUOUS
            flow_out = m.addVar(vtype=GRB.CONTINUOUS)# GRB.CONTINUOUS
            m.addConstr(quicksum( e.edge_var for e in v.es_in ) == flow_in)
            m.addConstr(quicksum( e.edge_var for e in v.es_out ) == flow_out)
            v.grb_flow_in = flow_in
            v.grb_flow_out = flow_out

        # m.update()

        # flow constraints
        for v in cfg.digraph.vs:
            zc_sum = quicksum( c.grb_var for c in cfg.digraph.cycles if v in c.vs) # quicksum( zc for c, zc in zip(cycles,cycle_vars) if v in c)    # add cycle variables to vertex vars
            m.addConstr( v.grb_flow_out + zc_sum <= v.grb_flow_in + zc_sum )
            m.addConstr( v.grb_flow_in + zc_sum <= 1)
            # ADDED BY DUNCAN:
            # downstream constraints (edge position in chain)
            # the position of v is the position of its incoming edge
            pos_var = m.addVar(vtype=GRB.CONTINUOUS)
            v.pos_var = pos_var
            m.addConstr(v.pos_var == quicksum( e.edge_pos_var_prime for e in v.es_in ))
            # the position of v's outgoing edges is v's pos + 1
            for e in v.es_out:
                m.addConstr(e.edge_pos_var == v.pos_var + 1)

        # flow constraints by NDD
        for v in cfg.digraph.vs:
             for i_ndd, ndd in enumerate(cfg.ndds):
                f_out_n = quicksum(v.grb_vars_out_ndd[i_ndd])
                f_in_n = quicksum(v.grb_vars_in_ndd[i_ndd])
                m.addConstr( f_out_n <= f_in_n )
                # m.addConstr(f_in_n <= ndd.used_var)  # ADDED BY DUNCAN: FLOW IN CAN ONLY BE POSITIVE IF NDD IS USED
                # m.addConstr(f_out_n <= ndd.used_var)  # ADDED BY DUNCAN: FLOW IN CAN ONLY BE POSITIVE IF NDD IS USED

        # chain length constraints
        for i_ndd, ndd in enumerate(cfg.ndds):
            ndd_edges = quicksum(e.edge_var for e in ndd.edges)
            pair_edges = quicksum(e.edge_var_ndd[i_ndd] for e in cfg.digraph.es)
            m.addConstr(ndd_edges + pair_edges <= cfg.max_chain)
            if cfg.min_chain_len is not None:
                m.addConstr(ndd_edges + pair_edges >= cfg.min_chain_len)

            # m.addConstr(quicksum(e.edge_var_ndd[i_ndd] for e in cfg.digraph.es) <= cfg.max_chain - 1)
                        # quicksum(e.edge_var for e in ndd.edges) <= cfg.max_chain )

        m.update()

        # add chain weight variables w^N_n
        for i_ndd, ndd in enumerate(cfg.ndds):
            ndd_weight = m.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
            weight_ndd_edge = quicksum(e.score * e.edge_var for e in ndd.edges) # ndd edge
            weight_pair_edges = quicksum(e.score * e.edge_var_ndd[i_ndd] for e in cfg.digraph.es) # pair-pair edges
            ndd.weight_var = ndd_weight
            m.addConstr(weight_ndd_edge + weight_pair_edges == ndd.weight_var)  # define chain weights
            m.update()

        # objective
        obj_expr = quicksum(n.weight_var for n in cfg.ndds) + \
                   quicksum( c.score * c.grb_var for c in cfg.digraph.cycles) # quicksum(cycle_score(c, cfg.digraph) * var for c, var in zip(cycles, cycle_vars))

    else: # no chains used

        # flow constraints
        for v in cfg.digraph.vs:
            zc_sum = quicksum( c.grb_var for c in cfg.digraph.cycles if v in c.vs) # add cycle variables to vertex vars
            m.addConstr( zc_sum <= 1)

        obj_expr = quicksum(c.score * c.grb_var for c in cfg.digraph.cycles)

    # obj_expr2 = (quicksum(cycle_score(c, cfg.digraph) * var for c, var in zip(cycles, cycle_vars)) +
    #  quicksum(e.score * e.edge_var for ndd in cfg.ndds for e in ndd.edges) +
    #  quicksum(e.score * e.edge_var for e in cfg.digraph.es))

    #m.setObjective(quicksum(cycle_score(c, cfg.digraph) * var for c, var in zip(cycles, cycle_vars)))
    # m.setObjective(quicksum(ndd_weights))
    m.setObjective(obj_expr,GRB.MAXIMIZE)
    m.update()

    return m

def create_pctsp_model(cfg):
    """Create the PC-TSP IP model without cut set constraints

    Args:
        cfg: an OptConfig object

    Returns:
        m: a gurobipy model for PC-TSP

        vtx_to_vars: A list such that for each Vertex v in the Digraph,
            vtx_to_vars[v.id] will contain the Gurobi variables representing
            edges pointing to v.
    """

    # create graph-tool graph
    # digraph_GT, v_GT, v_src, v_map, e_map, e_weight = create_GT_digraph(cfg)

    # cfg.remove_subtours = True



    all_cycles = cfg.digraph.find_cycles(cfg.max_cycle,cycle_file=cfg.cycle_file)
    cycle_list = [Cycle(c) for c in all_cycles]

    m = create_ip_model(cfg.timelimit, cfg.verbose)

    m.params.method = -1 # 2 EDIT: CHANGED FROM 2

    # add cycle vars
    for c in cycle_list:
        c.add_edges(cfg.digraph.es)
        c.score = cycle_score(c.vs, cfg.digraph)
        c.grb_var = m.addVar(vtype=GRB.BINARY)

    cfg.digraph.cycles = cycle_list

    m.update()

    # vtx_to_vars = [[] for __ in cfg.digraph.vs]

    if cfg.use_chains: #cfg.max_chain > 0:

        # assign edge in/out variables for each vertex, for each chain (ndd)
        for v in cfg.digraph.vs:
            v.grb_vars_in_ndd  = [[] for _ in range(len(cfg.ndds))]
            v.grb_vars_out_ndd = [[] for _ in range(len(cfg.ndds))]
            v.es_in  = []
            v.es_out = []

        for i_ndd,ndd in enumerate(cfg.ndds):
            ndd_edge_vars = []
            for e in ndd.edges:
                edge_var = m.addVar(vtype=GRB.BINARY)
                # edge_pos_var = m.addVar(vtype=GRB.CONTINUOUS,lb=1,ub=1)
                e.edge_var = edge_var
                # e.edge_pos_var = edge_pos_var
                # e.edge_pos_var_prime = e.edge_var # ADDED BY DUNCAN FOR POSITION CONSTRAINTS
                ndd_edge_vars.append(edge_var)
                e.tgt.grb_vars_in_ndd[i_ndd].append(edge_var)
                e.tgt.es_in.append(e) # ADDED BY DUNCAN: FOR POSITION CONSTRAINTS
            ndd_used = m.addVar(vtype=GRB.CONTINUOUS) # GRB.BINARY # (changed from binary by Duncan)
            m.addConstr(ndd_used == quicksum(e.edge_var for e in ndd.edges))
            # m.addConstr( quicksum(e.edge_var for e in ndd.edges) <= 1)
            m.addConstr(ndd_used <= 1)
            ndd.used_var = ndd_used
            m.update()

        kidney_utils.find_vertex_chain_participation(cfg.digraph, cfg.ndds, cfg.max_chain)

        # add pair-pair edge vars for each NDD, and each edge overall (y^n_e and y_e)
        for e in cfg.digraph.es:
            e.edge_var_ndd = []
            if any(e.src.can_be_in_chain_list) and any(e.tgt.can_be_in_chain_list):
                edge_var = m.addVar(vtype=GRB.CONTINUOUS)
                # edge_pos_var = m.addVar(vtype=GRB.CONTINUOUS)
            else:
                edge_var = m.addVar(vtype=GRB.CONTINUOUS,lb=0,ub=0)

            e.edge_var = edge_var
            # e.edge_pos_var = edge_pos_var
            e.tgt.es_in.append(e)  # ADDED BY DUNCAN: FOR POSITION CONSTRAINTS
            e.src.es_out.append(e)  # ADDED BY DUNCAN: FOR POSITION CONSTRAINTS
            for i_ndd,ndd in enumerate(cfg.ndds):
                # if e.src.can_be_in_chain_list[i_ndd] and e.tgt.can_be_in_chain_list[i_ndd]:
                edge_var_n = m.addVar(vtype=GRB.BINARY)
                e.tgt.grb_vars_in_ndd[i_ndd].append(edge_var_n)
                e.src.grb_vars_out_ndd[i_ndd].append(edge_var_n)
                e.edge_var_ndd.append(edge_var_n)

        m.update()

        # set edge vars using edge_var_ndds (sum over all NDDs)
        for e in cfg.digraph.es:
            m.addConstr( quicksum(e.edge_var_ndd) == e.edge_var )
            # ADDED BY DUNCAN FOR EDGE POSITIONS:
            # add edge_pos_var_prime = e.edge_pos_var* e.edge_var
            # e_vp = m.addVar(vtype=GRB.CONTINUOUS,lb = 0)
            # m.addConstr( e_vp <= e.edge_var * (len(cfg.digraph.vs)+1))
            # m.addConstr( e_vp <= e.edge_pos_var)
            # m.addConstr( e.edge_pos_var - (1 - e.edge_var)*(len(cfg.digraph.vs)+1) <= e_vp)
            # e.edge_pos_var_prime = e_vp

        # define pair-pair flow
        for v in cfg.digraph.vs:
            flow_in = m.addVar(vtype=GRB.CONTINUOUS) # GRB.CONTINUOUS
            flow_out = m.addVar(vtype=GRB.CONTINUOUS)# GRB.CONTINUOUS
            m.addConstr(quicksum( e.edge_var for e in v.es_in ) == flow_in)
            m.addConstr(quicksum( e.edge_var for e in v.es_out ) == flow_out)
            v.grb_flow_in = flow_in
            v.grb_flow_out = flow_out

        m.update()

        # flow constraints
        for v in cfg.digraph.vs:
            zc_sum = quicksum( c.grb_var for c in cfg.digraph.cycles if v in c.vs) # quicksum( zc for c, zc in zip(cycles,cycle_vars) if v in c)    # add cycle variables to vertex vars
            m.addConstr( v.grb_flow_out + zc_sum <= v.grb_flow_in + zc_sum )
            m.addConstr( v.grb_flow_in + zc_sum <= 1)
            # # ADDED BY DUNCAN:
            # # downstream constraints (edge position in chain)
            # # the position of v is the position of its incoming edge
            # pos_var = m.addVar(vtype=GRB.CONTINUOUS)
            # v.pos_var = pos_var
            # m.addConstr(v.pos_var == quicksum( e.edge_pos_var_prime for e in v.es_in ))
            # # the position of v's outgoing edges is v's pos + 1
            # for e in v.es_out:
            #     m.addConstr(e.edge_pos_var == v.pos_var + 1)

        # flow constraints by NDD
        for v in cfg.digraph.vs:
             for i_ndd, ndd in enumerate(cfg.ndds):
                f_out_n = quicksum(v.grb_vars_out_ndd[i_ndd])
                f_in_n = quicksum(v.grb_vars_in_ndd[i_ndd])
                m.addConstr( f_out_n <= f_in_n )
                # m.addConstr(f_in_n <= ndd.used_var)  # ADDED BY DUNCAN: FLOW IN CAN ONLY BE POSITIVE IF NDD IS USED
                # m.addConstr(f_out_n <= ndd.used_var)  # ADDED BY DUNCAN: FLOW IN CAN ONLY BE POSITIVE IF NDD IS USED

        # chain length constraints
        for i_ndd, ndd in enumerate(cfg.ndds):
            ndd_edges = quicksum(e.edge_var for e in ndd.edges)
            pair_edges = quicksum(e.edge_var_ndd[i_ndd] for e in cfg.digraph.es)
            m.addConstr(ndd_edges + pair_edges <= cfg.max_chain)
            # m.addConstr(quicksum(e.edge_var_ndd[i_ndd] for e in cfg.digraph.es) <= cfg.max_chain - 1)
                        # quicksum(e.edge_var for e in ndd.edges) <= cfg.max_chain )

        m.update()

        # add chain weight variables w^N_n
        for i_ndd, ndd in enumerate(cfg.ndds):
            ndd_weight = m.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
            weight_ndd_edge = quicksum(e.score * e.edge_var for e in ndd.edges) # ndd edge
            weight_pair_edges = quicksum(e.score * e.edge_var_ndd[i_ndd] for e in cfg.digraph.es) # pair-pair edges
            ndd.weight_var = ndd_weight
            m.addConstr(weight_ndd_edge + weight_pair_edges == ndd.weight_var)  # define chain weights
            m.update()

        # objective
        obj_expr = quicksum(n.weight_var for n in cfg.ndds) + \
                   quicksum( c.score * c.grb_var for c in cfg.digraph.cycles) # quicksum(cycle_score(c, cfg.digraph) * var for c, var in zip(cycles, cycle_vars))

    else: # no chains used

        # flow constraints
        for v in cfg.digraph.vs:
            zc_sum = quicksum( c.grb_var for c in cfg.digraph.cycles if v in c.vs) # add cycle variables to vertex vars
            m.addConstr( zc_sum <= 1)

        obj_expr = quicksum(c.score * c.grb_var for c in cfg.digraph.cycles)

    # obj_expr2 = (quicksum(cycle_score(c, cfg.digraph) * var for c, var in zip(cycles, cycle_vars)) +
    #  quicksum(e.score * e.edge_var for ndd in cfg.ndds for e in ndd.edges) +
    #  quicksum(e.score * e.edge_var for e in cfg.digraph.es))

    #m.setObjective(quicksum(cycle_score(c, cfg.digraph) * var for c, var in zip(cycles, cycle_vars)))
    # m.setObjective(quicksum(ndd_weights))
    m.setObjective(obj_expr,GRB.MAXIMIZE)
    m.update()

    return m


## NEW FORMULATION : PI-TSP
def optimize_pitsp(cfg):

    m = create_pitsp_model(cfg) # ,cycles, cycle_vars

    # optimise(m, cfg)

    # # finally, solve
    # if cfg.use_chains and (not cfg.remove_subtours):
    #
    #     try:
    #         pctsp_constraint_gen(cfg,m)
    #     except MaxIterConstraintGenException as ex:
    #         # just remove the subtours if constraint generation takes too long...
    #         print "constraint gen took too long, removing all subtours"
    #         cfg.remove_subtours = True
    #         optimise(m, cfg)
    #
    # else:
    # print "current memory usage (pi-tsp): %f GB" % (h.heap().size/1e9)
    # print "size of model (pi-tsp): %f GB" % (sys.getsizeof(m)/1e9)

    optimise(m,cfg)

    if m.status in [GRB.status.INFEASIBLE, GRB.status.INF_OR_UNBD]:
        sol = OptSolution(ip_model=m,
                          cycles=[],
                          chains=[],
                          digraph=cfg.digraph,
                          infeasible=True
                          )
    else:
        # find all cycles used in the matching
        cycles_used = [ c for c in cfg.digraph.cycles if c.grb_var.x > 0.1]

        # get chains
        chains_used = kidney_utils.get_optimal_chains_pctsp(cfg.digraph, cfg.ndds) if cfg.use_chains \
            else []

        sol = OptSolution(ip_model=m,
                           cycles=[c.vs for c in cfg.digraph.cycles if c.grb_var.x > 0.5],
                           cycle_obj = cycles_used,
                           chains=chains_used,
                           digraph=cfg.digraph,
                           edge_success_prob=cfg.edge_success_prob,
                          chain_restriction=cfg.chain_restriction,
                          cycle_restriction=cfg.cycle_restriction,
                          cycle_cap=cfg.max_cycle,
                          chain_cap=cfg.max_chain,
                          min_chain_len=cfg.min_chain_len,
                          cardinality_restriction=cfg.cardinality_restriction,
                          edge_assign_seed=cfg.edge_assign_seed
                          )
        sol.add_matching_edges(cfg.ndds)

        kidney_utils.check_validity(sol, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain, cfg.min_chain_len)

    return sol

import pickle
def pctsp_constraint_gen(cfg, m, max_iter = 200):
    """Constraint generation method for solving PC-TSP formulation, given PC-TSP model m"""

    # write the subtour constraint file after finding violated constraints
    write_subtour_file = False

    if not cfg.use_chains:
        optimise(m, cfg)
    else:
        # if subtour constraints have been saved, use them
        if cfg.subtour_file is not None:
            eS_list,v_list = load_subtours_from_pickle(cfg)
            # with open(cfg.subtour_file, "rb") as f:
            #     pickle_d = pickle.load(f)
            #     v_idx = pickle_d['v_idx']
            #     ndd_edges_idx = pickle_d['ndd_edges_idx']
            #     pair_edges_idx = pickle_d['pair_edges_idx']
            #     ndd_edge_list =ixd_list_to_nddedge_list(cfg, ndd_edges_idx)
            #     pair_edge_list =ixd_list_to_pairedge_list(cfg, pair_edges_idx)
            #     v_list = ixd_list_to_v_list(cfg,v_idx)
            #     eS_list = [ndd_l + pair_l for ndd_l,pair_l in zip(ndd_edge_list,pair_edge_list)]
            if cfg.verbose:
                print "read %d subtour constraints from file" % len(v_list)
            for eS, v in zip(eS_list, v_list):
                m.addConstr(v.grb_flow_in <= quicksum(e.edge_var for e in eS))
        # elif cfg.subtour_dir is not None:
        elif cfg.subtour_dir is None: # otherwise, create file, if subtour dir is defined
            # cfg.subtour_file = os.path.join(cfg.subtour_dir,cfg.name + ".pkl")
            write_subtour_file = False
        # else:
        #     write_subtour_file = False

        # solve model without subtour elimination constraints
        optimise(m, cfg)

        # if cfg.max_chain > 0 and len(cfg.ndds) > 0:
        #     # find initial violated constraints, add them to the model
        #     eS_list,v_list = find_violated(cfg)
        #
        #     if len(v_list) > 0:
        #         if cfg.verbose:
        #             print("iter {}: found {} violated constraints:".format(0,len(v_list)))
        #         for eS, v in zip(eS_list,v_list):
        #             m.addConstr(v.grb_flow_in <= quicksum(e.edge_var for e in eS))
        #         optimise(m,cfg)
        #         violated = True
        #     else:
        #         if cfg.verbose:
        #             print("iter %d: no violated constraints found!" % 0)
        #         violated = False

        # if test:
        #     for c in cfg.digraph.cycles:
        #         c.banned = False

        violated = True
        iter = 1
        while iter < max_iter and violated:
            eS_cstr, v_cstr, v_tgt_list = find_violated(cfg)
            if len(v_cstr) > 0:

                if test:
                    # # ban all known cycles that contain vertices in v_tgt_list (the offending vertices)
                    # for c in [cy for cy in cfg.digraph.cycles if (any(v in cy.vs for v in v_tgt_list) and (not cy.banned) )]:
                    #     m.addConstr(quicksum(e.edge_var for e in c.edges) <= c.length - 1)
                    #     c.banned = True

                    # find all cycles that the offending vertices (v_tgt_list) participate in, and ban them
                    current_banned = []
                    for v_offend in v_tgt_list:
                        if v_offend not in current_banned:
                            # find the cycle created with v_offend:
                            v_ch = [v_offend]
                            e_ch = []
                            chain_complete = False
                            while not chain_complete:
                                e_next = [e for e in v_ch[-1].edges if np.abs(e.edge_var.x - 1.0) < EPS_mid]
                                if len(e_next) > 1:
                                    raise Warning("vertex %s has more than one outgoing chain edge" % str(v_ch[-1]))
                                elif len(e_next) == 0:
                                    raise Warning("expected outgoing chain edge for vertex %s" % str(v_ch[-1]))
                                else:
                                    e_ch.append(e_next[0])
                                    v_ch.append(e_next[0].tgt)
                                # check for complete chain
                                if v_ch[-1] == v_ch[0]:
                                    chain_complete = True
                                    # ban other cycles created by replacing one vertex in this cycle
                                    m.addConstr(quicksum(e.edge_var for e in e_ch) <= len(e_ch) - 1)
                                    ban_other_cycles(e_ch,m,verbose=cfg.verbose)
                                    current_banned.extend(v_ch[:-1])
                                    if cfg.verbose:
                                        print "banned chain: %s" % " ".join([str(v) for v in v_ch])


                if (not write_subtour_file) and (cfg.subtour_dir is not None):
                    write_subtour_file = True
                    cfg.subtour_file = os.path.join(cfg.subtour_dir, cfg.name + ".pkl")
                if cfg.verbose:
                    print("iter {}: found {} violated constraints:".format(iter,len(v_cstr)))
                # for eS, v in zip(eS_cstr, v_cstr):
                #     print("v = %d" % v.id)
                #     print("S = ")
                #     for e in eS:
                #         print(e)
                for eS, v_list in zip(eS_cstr, v_cstr):
                    for v in v_list: # TODO: CHECK FOR REDUNDENCIES IN eS and v_list
                        m.addConstr(v.grb_flow_in <= quicksum(e.edge_var for e in eS))
                iter += 1
                if write_subtour_file:
                    write_subtours_to_pickle(cfg, eS_cstr, v_cstr)
                # eS_list.extend(eS_cstr)
                # v_list.extend(v_cstr)
                optimise(m, cfg)
            elif len(v_cstr) == 0:
                violated = False
                if cfg.verbose:
                    print("iter %d: no violated constraints found!" % iter)

        # if write_subtour_file:
        #     write_to_pickle(cfg,eS_list,v_list)
            # # pickle_d = {'eS_list':eS_list,'v_list':v_list}
            # pickle_d = edge_list_to_index_dict(eS_list)
            # v_d = v_list_to_index_dict(v_list)
            # pickle_d.update(v_d)
            # with open(cfg.subtour_file, "ab") as f:
            #     pickle.dump(pickle_d, f)

        if iter >= max_iter:
            raise MaxIterConstraintGenException
            # raise Warning("maximum number of iterations reached")

def ban_other_cycles(e_ch,m, verbose=False,recursive=False):
    '''
    e_ch is a list of edges forming an illegal chain; m is the gurobi model

    assume e_ch has been banned already; find other cycles created
    by replacing one vertex in v_ch with a different one
    '''
    # search for a replacement for e_ch[i] and e_ch[i+1]
    # old: (e_ch[i].src -> e_ch[i].tgt), and (e_ch[i+1].src -> e_ch[i+1].tgt)
    # 1) e1_new = (e_ch[i].src -> v_new); e2_new = (v_new -> e_ch[i+1].tgt)
    for i in range(len(e_ch)-1):
        # for all potential new targets
        for e1_new in [e for e in e_ch[i].src.edges if e != e_ch[i]]:
            # if potential v_new can points to c_ch[i+1].tgt edge:
            v_new = e1_new.tgt
            for e2_new in v_new.edges:
                if e2_new.tgt == e_ch[i+1].tgt:
                    # this is a potential cycle
                    e_ch_tmp = e_ch[:]
                    e_ch_tmp[i] = e1_new
                    e_ch_tmp[i+1] = e2_new
                    m.addConstr(quicksum(e.edge_var for e in e_ch_tmp) <= len(e_ch_tmp) - 1)
                    if verbose:
                        print "banned additional chain: %s" % " ".join([str(e) for e in e_ch_tmp])
                    break


def edge_list_to_index_dict(e_list_list):
    '''
    turn a list (of lists) of both Edge and NddEdge objects and turn it into two lists of (src_id,tgt_it):
    first separate the edge list into ndd edges and pair-pair edges,
    then put them in a dictionary
    '''
    ndd_edges_idx = [ [(e.src_id,e.tgt.id) for e in e_list if isinstance(e,NddEdge) ]
                     for e_list in e_list_list]

    pair_edges_idx = [ [(e.src_id,e.tgt.id) for e in e_list if isinstance(e,Edge)]
                      for e_list in e_list_list]
    return {'ndd_edges_idx':ndd_edges_idx,'pair_edges_idx':pair_edges_idx}

def ixd_list_to_nddedge_list(cfg, e_idx):
    '''
    turn a list of (ndd_idx,tgt_idx) into a list of NddEdge objects
    '''
    return [ [cfg.ndds[ndd_idx].get_edge(tgt_idx) for ndd_idx,tgt_idx in e_idx_list]
            for e_idx_list in e_idx]

def ixd_list_to_pairedge_list(cfg, e_idx):
    '''
    turn a list of (src_id,tgt_id) into a list of Edge objects
    '''
    return [ [cfg.digraph.adj_mat[src_id][tgt_id] for src_id,tgt_id in e_idx_list]
             for e_idx_list in e_idx]

def v_list_to_index_dict(v_idx_list):
    return {'v_idx':[ v.id for v in v_idx_list]}

def ixd_list_to_v_list(cfg, v_idx_list):
    return [ cfg.digraph.vs[i] for i in v_idx_list]


def write_subtours_to_pickle(cfg,eS_list,v_list):
    # pickle_d = {'eS_list':eS_list,'v_list':v_list}
    pickle_d = edge_list_to_index_dict(eS_list)
    v_d = v_list_to_index_dict(v_list)
    pickle_d.update(v_d)
    with open(cfg.subtour_file, "ab+") as f:
        pickle.dump(pickle_d, f)


def load_subtours_from_pickle(cfg):
    '''
    load a list of (lists of) edges,
    '''
    v_list = []
    eS_list = []
    with open(cfg.subtour_file, "rb") as f:
        while True:
            try:
                pickle_d = pickle.load(f)
                v_idx = pickle_d['v_idx']
                ndd_edges_idx = pickle_d['ndd_edges_idx']
                pair_edges_idx = pickle_d['pair_edges_idx']
                ndd_edge_list = ixd_list_to_nddedge_list(cfg, ndd_edges_idx)
                pair_edge_list = ixd_list_to_pairedge_list(cfg, pair_edges_idx)
                v = ixd_list_to_v_list(cfg, v_idx)
                eS = [ndd_l + pair_l for ndd_l, pair_l in zip(ndd_edge_list, pair_edge_list)]
                v_list.extend(v)
                eS_list.extend(eS)
            except EOFError:
                break
    if len(eS_list) != len(v_list):
        raise Warning("subtour constraint lists eS and V are not the same length")
    return eS_list,v_list

def optimize_robust_pctsp(cfg):

    # for use later
    floor_gamma = np.floor(cfg.gamma)
    ceil_gamma = np.ceil(cfg.gamma)
    gamma_frac = cfg.gamma - floor_gamma


    m = create_pctsp_model(cfg) # ,cycles, cycle_vars

    # edge discount is edge weight
    for ndd in cfg.ndds:
        for e in ndd.edges:
            e.discount = e.score

    for e in cfg.digraph.es:
        e.discount = e.score

    # add cycle/chain discount indicators g and d = y*g
    if gamma_frac == 0:  # gamma is integer

        if cfg.use_chains:
            # add ordering indicator variables g and discount indicators d ( d = y*g ) for chains (NDDs)
            for ndd in cfg.ndds:
                g_var = m.addVar(vtype=GRB.BINARY)
                d_var = m.addVar(vtype=GRB.BINARY)
                ndd.g_var = g_var
                ndd.d_var = d_var
                m.addGenConstrAnd(ndd.d_var, [ndd.g_var, ndd.used_var])
                m.update()

        # add g and d variables for cycles
        for c in cfg.digraph.cycles:
            c.g_var = m.addVar(vtype=GRB.BINARY)
            c.d_var = m.addVar(vtype=GRB.BINARY)
            m.addGenConstrAnd(c.d_var, [c.g_var, c.grb_var])
            m.update()

    else:  # gamma is not integer

        if cfg.use_chains:
            # use both gf (full discount if gf=1, gp=0) and gp (partial discount, if gf=gp=1)
            for ndd in cfg.ndds:
                ndd_used = m.addVar(vtype=GRB.BINARY)
                m.addConstr(ndd_used == quicksum(e.edge_var for e in ndd.edges))
                ndd.used_var = ndd_used
                gp_var = m.addVar(vtype=GRB.BINARY)
                gf_var = m.addVar(vtype=GRB.BINARY)
                dp_var = m.addVar(vtype=GRB.BINARY)
                df_var = m.addVar(vtype=GRB.BINARY)

                ndd.gp_var = gp_var
                ndd.gf_var = gf_var

                ndd.dp_var = dp_var
                ndd.df_var = df_var

                m.addGenConstrAnd(ndd.dp_var, [ndd.gp_var, ndd.used_var])
                m.addGenConstrAnd(ndd.df_var, [ndd.gf_var, ndd.used_var])

                m.update()

        # add g and d variables for cycles
        for c in cfg.digraph.cycles:
            c.gp_var = m.addVar(vtype=GRB.BINARY)
            c.gf_var = m.addVar(vtype=GRB.BINARY)
            c.dp_var = m.addVar(vtype=GRB.BINARY)
            c.df_var = m.addVar(vtype=GRB.BINARY)

            m.addGenConstrAnd(c.dp_var, [c.gp_var, c.grb_var])
            m.addGenConstrAnd(c.df_var, [c.gf_var, c.grb_var])

            m.update()

    # discount indicators g follow same ordering as the edge discount values (sort in increasing order)

    if gamma_frac == 0:

        # first for cycles (easy)
        c_sorted = sorted(cfg.digraph.cycles, key=lambda c: c.score, reverse=False)
        for i in range(len(c_sorted)-1):
            m.addConstr(c_sorted[i].g_var <= c_sorted[i+1].g_var)

        if cfg.use_chains:
            # for chains v. chains
            for n1,n2 in itertools.combinations(cfg.ndds,2):
                # add q_cn indicator variables
                q = m.addVar(vtype=GRB.BINARY)
                m.addConstr(n1.g_var <= n2.g_var + q)
                m.addConstr( n2.weight_var - n1.weight_var <= W*(1-q))
                m.addConstr(n2.g_var <= n1.g_var + (1-q))
                m.addConstr( n1.weight_var - n2.weight_var <= W*q)

            # for chains v. cycles
            for ndd in cfg.ndds:
                for c in cfg.digraph.cycles:
                    qcn = m.addVar(vtype=GRB.BINARY)
                    m.addConstr(ndd.g_var <= c.g_var + qcn)
                    m.addConstr(c.score - ndd.weight_var <= W * (1 - qcn))
                    m.addConstr(c.g_var <= ndd.g_var + (1 - qcn))
                    m.addConstr(ndd.weight_var - c.score <= W * qcn)

        m.update()

    else: # gamma is not integer

        # first for cycles (easy)
        c_sorted = sorted(cfg.digraph.cycles, key=lambda c: c.score, reverse=False)
        for i in range(len(c_sorted) - 1):
            m.addConstr(c_sorted[i].gp_var <= c_sorted[i + 1].gp_var)
            m.addConstr(c_sorted[i].gf_var <= c_sorted[i + 1].gf_var)

        if cfg.use_chains:
            # for chains v. chains
            for n1, n2 in itertools.combinations(cfg.ndds, 2):
                # add q_cn indicator variables
                q = m.addVar(vtype=GRB.BINARY)

                m.addConstr(n1.gp_var <= n2.gp_var + q)
                m.addConstr(n1.gf_var <= n2.gf_var + q)

                m.addConstr( n2.weight_var - n1.weight_var <= W*(1-q))

                m.addConstr(n2.gp_var <= n1.gp_var + (1-q))
                m.addConstr(n2.gf_var <= n1.gf_var + (1-q))

                m.addConstr( n1.weight_var - n2.weight_var <= W*q)

            # for chains v. cycles
            for ndd in cfg.ndds:
                for c in cfg.digraph.cycles:
                    qcn = m.addVar(vtype=GRB.BINARY)
                    m.addConstr( ndd.gp_var <= c.gp_var + qcn )
                    m.addConstr( ndd.gf_var <= c.gf_var + qcn )

                    m.addConstr( c.score - ndd.weight_var <= W*(1-qcn))

                    m.addConstr( c.gp_var <= ndd.gp_var + (1-qcn) )
                    m.addConstr( c.gf_var <= ndd.gf_var + (1-qcn) )

                    m.addConstr( ndd.weight_var - c.score <= W*qcn )

        m.update()

    m.update()

    # number of objects used in matching (include all position-indexed vars)
    G = m.addVar(vtype=GRB.INTEGER)
    cycle_count = [c.grb_var for c in cfg.digraph.cycles]
    if cfg.use_chains:
        ndd_count = [ ndd.used_var for ndd in cfg.ndds ]
        m.addConstr( G == quicksum(cycle_count + ndd_count))
    else:
        m.addConstr( G == quicksum(cycle_count))


    # add a cardinality restriction if necessary
    if cfg.cycle_restriction is not None:
        m.addConstr( quicksum(cycle_count) <= cfg.cycle_restriction )
    if cfg.use_chains and (cfg.chain_restriction is not None):
        m.addConstr( quicksum(ndd_count) <= cfg.chain_restriction )

    # uncertainty budget (number of discounted edges ### THIS CAN BE GRB.INTEGER INSTEAD ###
    # G_eq_zero = m.addVar(vtype=GRB.BINARY)
    # m.addConstr(1 - G_eq_zero <= W_small * G)
    # m.addConstr( G <= W_small * (1 - G_eq_zero))
    # G_m_1 = m.addVar(vtype=GRB.INTEGER)
    # m.addConstr(G_m_1 == G - 1 + G_eq_zero)
    gamma_var = m.addVar(vtype=GRB.CONTINUOUS)
    # m.addGenConstrMin( gamma_var, [G_m_1,cfg.gamma])
    m.addGenConstrMin( gamma_var, [G,cfg.gamma])

    m.update()

    # limit number of discounted variables

    if gamma_frac == 0: # gamma is integer
        if cfg.use_chains:
            m.addConstr( quicksum(c.d_var for c in cfg.digraph.cycles)
                         + quicksum(ndd.d_var for ndd in cfg.ndds)
                         == gamma_var )
        else:
            m.addConstr( quicksum(c.d_var for c in cfg.digraph.cycles)
                         == gamma_var )
    else: # gamma is not integer

        h_var = m.addVar(vtype=GRB.BINARY)
        m.addConstr( cfg.gamma - G <= W_small*h_var )
        m.addConstr( G - cfg.gamma <= W_small*(1-h_var) )

        if cfg.use_chains:
            m.addConstr( quicksum(c.dp_var for c in cfg.digraph.cycles)
                         + quicksum(ndd.dp_var for ndd in cfg.ndds)
                         == h_var * G + (1 - h_var) * ceil_gamma)

            m.addConstr( quicksum(c.df_var for c in cfg.digraph.cycles)
                         + quicksum(ndd.df_var for ndd in cfg.ndds)
                         == h_var * G + (1 - h_var) * floor_gamma)
        else:
            m.addConstr(quicksum(c.dp_var for c in cfg.digraph.cycles)
                        == h_var * G + (1 - h_var) * ceil_gamma)

            m.addConstr(quicksum(c.df_var for c in cfg.digraph.cycles)
                        == h_var * G + (1 - h_var) * floor_gamma)

    # total discount (by object)
    if gamma_frac == 0: # gamma is integer
        if cfg.use_chains:
            total_discount = quicksum(n.weight_var * n.d_var for n in cfg.ndds)  \
                             +quicksum(c.score * c.grb_var * c.d_var for c in cfg.digraph.cycles)
        else:
            total_discount = quicksum(c.score * c.grb_var * c.d_var for c in cfg.digraph.cycles)
    else: # gamma is not integer
          if cfg.use_chains:
              total_discount = (1 - gamma_frac)*quicksum(n.weight_var * n.df_var for n in cfg.ndds) \
                         +(1 - gamma_frac)*quicksum(c.score * c.grb_var * c.df_var for c in cfg.digraph.cycles) \
                         +gamma_frac *quicksum(n.weight_var * n.dp_var for n in cfg.ndds) \
                         +gamma_frac * quicksum(c.score * c.grb_var * c.dp_var for c in cfg.digraph.cycles)
          else:
              total_discount = (1 - gamma_frac) * quicksum(c.score * c.grb_var * c.df_var for c in cfg.digraph.cycles) \
                              + gamma_frac * quicksum(c.score * c.grb_var * c.dp_var for c in cfg.digraph.cycles)

    # set a variable for the total (optimistic matching weight)
    total_weight = m.addVar(vtype=GRB.CONTINUOUS)

    if cfg.use_chains:
        optimistic_obj = quicksum(n.weight_var for n in cfg.ndds) + \
                   quicksum( c.score * c.grb_var for c in cfg.digraph.cycles)
    else:
        optimistic_obj = quicksum(c.score * c.grb_var for c in cfg.digraph.cycles)

    m.addConstr(optimistic_obj == total_weight)

    obj_expr = optimistic_obj - relax_discount * total_discount

    # set the objective
    m.setObjective(obj_expr, GRB.MAXIMIZE)

    m.update()

    # optimise(m, cfg)

    # finally, solve
    if cfg.use_chains and (not cfg.remove_subtours):

        try:
            pctsp_constraint_gen(cfg,m)
        except MaxIterConstraintGenException as ex:
            # just remove the subtours if constraint generation takes too long...
            print "constraint gen took too long, removing all subtours"
            cfg.remove_subtours = True
            optimise(m, cfg)

    else:
        optimise(m,cfg)

    # find all cycles used in the matching
    cycles_used = [ c for c in cfg.digraph.cycles if c.grb_var.x > 0.1]

    if cfg.use_chains:
        # find all edges used in chains. (edge_var property is only used for chains)
        chain_edges_pair = [e for e in cfg.digraph.es if e.edge_var.x > 0.5]
        chain_edges_ndd = [e for n in cfg.ndds for e in n.edges if e.edge_var.x > 0.5]

        chain_edges_used = chain_edges_pair + chain_edges_ndd
        matching_edges = chain_edges_used + [e for c in cycles_used for e in c.edges ]
    else:
        matching_edges =  [e for c in cycles_used for e in c.edges ]


    if gamma_frac == 0: # gamma is integer

        discounted_cycles = [ c for c in cfg.digraph.cycles if c.d_var.x > 0.5]

        for c in discounted_cycles:
            for e in c.edges:
                e.discount_frac = c.d_var.x
            c.discount_frac = c.d_var.x

        if cfg.use_chains:
            discounted_ndd_chains = [ (i_ndd,ndd) for i_ndd,ndd in enumerate(cfg.ndds) if ndd.d_var.x > 0.5]

            # discount all edges involved in these chains
            for i_ndd,ndd in discounted_ndd_chains:
                discount_frac = ndd.d_var.x
                ndd.discount_frac = discount_frac
                for e in ndd.edges:
                    if e.edge_var.x > 0.5:
                        e.discount_frac = discount_frac
                for e in cfg.digraph.es:
                    if e.edge_var_ndd[i_ndd].x > 0.5: # edge is used in chain i_ndd
                        e.discount_frac = discount_frac


    else: # gamma is not integer

        discounted_cycles = [ c for c in cfg.digraph.cycles
                              if ((c.df_var.x > 0.5) or (c.dp_var.x > 0.5))]

        for c in discounted_cycles:
            discount_frac = (1-gamma_frac) * c.df_var.x + gamma_frac * c.dp_var.x
            for e in c.edges:
                e.discount_frac = discount_frac
            c.discount_frac = discount_frac

        if cfg.use_chains:
            discounted_ndd_chains = [ (i_ndd,ndd) for i_ndd,ndd in enumerate(cfg.ndds)
                                      if ((ndd.df_var.x > 0.5) or (ndd.dp_var.x > 0.5))]
            # discount all edges involved in these chains
            for i_ndd,ndd in discounted_ndd_chains:
                discount_frac = (1-gamma_frac) * ndd.df_var.x + gamma_frac * ndd.dp_var.x
                ndd.discount_frac = discount_frac
                for e in ndd.edges:
                    if e.edge_var.x > 0.5:
                        e.discount_frac = discount_frac
                for e in cfg.digraph.es:
                    if e.edge_var_ndd[i_ndd].x > 0.5: # edge is used in chain i_ndd
                        e.discount_frac = discount_frac


    # get chains
    chains_used = kidney_utils.get_optimal_chains_pctsp(cfg.digraph, cfg.ndds) if cfg.use_chains \
        else []

    for ch in chains_used:
        ch.discount_frac = cfg.ndds[ch.ndd_index].discount_frac

    # check cardinality restrictions
    if cfg.use_chains and (cfg.chain_restriction is not None):
        if len(chains_used) > cfg.chain_restriction:
            raise Warning("chain restriction violated: # chains = %d, restriction = %d"
                          % (len(chains_used),cfg.chain_restriction))
    if cfg.cycle_restriction is not None:
        if len(cycles_used) > cfg.cycle_restriction:
            raise Warning("cycle restriction violated: # cycles = %d, restriction = %d"
                              % (len(cycles_used),cfg.cycle_restriction))
    sol = OptSolution(ip_model=m,
                       cycles=[c.vs for c in cfg.digraph.cycles if c.grb_var.x > 0.5],
                       cycle_obj = cycles_used,
                       chains=chains_used,
                       digraph=cfg.digraph,
                       edge_success_prob=cfg.edge_success_prob,
                       gamma=cfg.gamma,
                       robust_score= m.objVal,
                       optimistic_score = total_weight.x,
                      chain_restriction=cfg.chain_restriction,
                      cycle_restriction=cfg.cycle_restriction,
                      cycle_cap=cfg.max_cycle,
                      chain_cap=cfg.max_chain,
                      cardinality_restriction=cfg.cardinality_restriction,
                      edge_assign_seed=cfg.edge_assign_seed
                      )
                       # matching_edges=matching_edges)
    sol.add_matching_edges(cfg.ndds)



    kidney_utils.check_validity(sol, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain)

    return sol

def solve_edge_existence_uncertainty(cfg, max_num_cycles = 0):
    '''
    Solves the robust kidney exchange problem with a variable-budget edge existence uncertainty.
        - uses the cardinality-restriction method of Poss.
        - uses the constant-budget edge-existence-uncertainty robust formulation of PC-TSP

    inputs:
        - cfg               : OptConfig object
        - max_cycles        : maximum number of cycles in a feasible matching
    '''

    # define gamma (variable uncertainty budget) function
    gamma_func = lambda n_cy,n_ch: kidney_utils.gamma_homogeneous_edge_failure(n_cy,n_ch,cfg.edge_failure_prob,cfg.max_cycle,cfg.protection_level)

    if cfg.verbose:
        print "solving edge existence uncertainty"

    if max_num_cycles == 0:
        # find maximum-cardinality solution (cardinality = edge count)
        max_num_cycles = max_cycles(cfg)

    if cfg.verbose:
        print "maximum cycle count = %d" % max_num_cycles

    # now find all card-restricted solutions to the constant-budget robust problem,
    # and take the best one

    best_gamma = 0

    # if there is no feasible solution...
    if (max_num_cycles == 0) and (len(cfg.ndds) == 0):
        opt_sol = OptSolution(ip_model=[],
                           cycles=[],
                           cycle_obj=[],
                           chains=[],
                           matching_edges=[],
                           gamma = 0,
                           digraph=cfg.digraph,
                           edge_success_prob=cfg.edge_success_prob,
                              chain_restriction=cfg.chain_restriction,
                              cycle_restriction=cfg.cycle_restriction,
                              cycle_cap=cfg.max_chain,
                              chain_cap=cfg.max_cycle,
                      cardinality_restriction=cfg.cardinality_restriction,
                      edge_assign_seed=cfg.edge_assign_seed
                              )
                           # matching_edges=[])

        opt_sol.max_num_cycles = 0
        return opt_sol

    update_new_sol = True # flag
    for cycle_restriction in range(0,max_num_cycles+1):
        for chain_restriction in range(0,len(cfg.ndds)+1):
            if cycle_restriction+chain_restriction > 0:
                # solve the cardinality-restricted problem, with Gamma = gamma(k)
                cfg.cycle_restriction = cycle_restriction
                cfg.chain_restriction = chain_restriction
                cfg.gamma = gamma_func(cycle_restriction,chain_restriction)
                if cfg.gamma == 0:
                    new_sol = optimise_picef(cfg)
                    new_sol.robust_score = new_sol.total_score
                    new_sol.optimistic_score = new_sol.total_score
                else:
                    new_sol = optimize_robust_pctsp(cfg)

                if cfg.verbose:
                    print "max %d cycles, max %d chains; gamma = %f; robust obj = %f" % (cycle_restriction, chain_restriction, cfg.gamma, new_sol.robust_score)

                if update_new_sol:
                    best_sol = new_sol
                    best_gamma = cfg.gamma
                    update_new_sol = False
                elif new_sol.robust_score > best_sol.robust_score:
                        best_sol = new_sol
                        best_gamma = cfg.gamma

    # return the best solution and save the best gamma value
    cfg.gamma = best_gamma
    best_sol.max_num_cycles = max_num_cycles
    return best_sol

def find_violated(cfg):
    """Find violated subtour elimination constraints.
    For every vertex v with positive NDD flow into v, create the
    auxiliary graph described in Anderson et al. (supplemental),
    and solve max flow/ min cut problem. If the min cut is smaller
    than the flow into v, then taking S as the nodes on the sink
    side (v side) of the cut, and vertex v, this constitutes a
    violated constraint.

    Inputs:
        m: solved gurobi model
        cfg: config object

    Outputs:
        eS: list of sets (lists) of edges into S for each violated constraint
        v: list of vertices for each violated constraint
    """

    eS = []
    v_sink_list = []
    v_tgt_list = []
    digraph_GT, v_GT, v_src, v_map, v_obj, e_map, e_weight = create_GT_digraph(cfg)

    # for each vertex with positive chain flow in...
    for v_tgt in cfg.digraph.vs:
        if v_tgt.grb_flow_in.x > 0.5:
            # solve min cut
            src = v_src
            tgt = v_map[v_tgt]
            cap = e_weight
            res = boykov_kolmogorov_max_flow(digraph_GT, src, tgt, cap)
            part = min_st_cut(digraph_GT, src, cap, res) # part[vertex] is True if vertex is on source side of the cut
            mc = sum([cap[e] - res[e] for e in digraph_GT.edges() if (part[e.source()] != part[e.target()])])

            # if mc < v_tgt.grb_flow_in.X: # constraint violated
            if (mc + EPS_mid) < v_tgt.grb_flow_in.X :
                # find edges over cut
                # print('constraint violated:')
                # print("mc = %f" % mc)
                # print("flow_in = %f" % v_tgt.grb_flow_in.X)
                # for e in digraph_GT.edges():
                    # if part[e.source()] != part[e.target()]:
                    #     print("edge to add:")
                    #     print(e_obj[e].tgt)
                # we need all edges pointing INTO S (the sink side of the cut)
                # so the target should be in S and the source should be not in S
                # i.e. part[e.target()] == False and part[e.source()] == True
                # eS.append([ e_map[e] for e in digraph_GT.edges()
                #             if (part[e.source()] == True)
                #             and (part[e.target()] == False)
                #             and (v_src != e.source()) ] )
                eS.append([ e_map[e] for e in digraph_GT.edges()
                            if (part[e.source()] != part[e.target()])])
                v_sink = [v_obj[v] for v in digraph_GT.vertices() if part[v] == False]

                # check that sum of edge grb_vars for cut is actually less than the flow
                if (sum(e.edge_var.x for e in eS[-1]) + EPS_mid) > v_tgt.grb_flow_in.X:
                    raise Warning("min cut is larger than flow")

                v_sink_list.append(v_sink)
                v_tgt_list.append(v_tgt)
    return eS, v_sink_list, v_tgt_list

import graph_stuff
def create_GT_digraph(cfg):
    """Create a graph-tool digraph from the kex digraph and NDDs.
    Add edge weights as per Anderson et al."""

    # create graph_tool graph
    # these dicts are { id : vertex }
    digraph_GT = Graph()

    # for FlowNetwork class
    digraph_flow = graph_stuff.FlowNetwork()
    v_obj_flow = {}
    e_map_flow = {}
    # edge weights
    e_weight = digraph_GT.new_edge_property("double")
    # e_map = digraph_GT.new_edge_property("object")

    # edge map: keys are tuples (src,tgt), and values are both kidney_digraph.Edge and kidney_ndd.NDDEdge type
    e_map = {}

    # vertex map to object v_obj[v] for a GT vertex v returns the Vertex in the kex digraph
    v_obj = digraph_GT.new_vertex_property("object")

    # pair vertices
    v_GT = {}
    for v in cfg.digraph.vs:
        v_GT[v.id] = digraph_GT.add_vertex()
        v_obj[v_GT[v.id]] = v
        # for FlowNetwork class
        digraph_flow.AddVertex(name=str(v.id))

    # pair-pair edges
    for e in cfg.digraph.es:
        e_GT = digraph_GT.add_edge(v_GT[e.src_id], v_GT[e.tgt_id]) # from src.id & tgt.id
        e_weight[e_GT] = round(e.edge_var.x) # get rid of numerical noise
        e_map[e_GT] = e

    # ndd vertices
    if len(cfg.ndds) == 1:
        ndd_GT = [ digraph_GT.add_vertex() ]
    elif len(cfg.ndds)>1:
        ndd_GT = list(digraph_GT.add_vertex(len(cfg.ndds)))
    else:
        ndd_GT = []

    # ndd edges
    for i_ndd,ndd in enumerate(cfg.ndds):
        for e in ndd.edges:
            e_GT = digraph_GT.add_edge(ndd_GT[i_ndd], v_GT[e.tgt_id])
            e_weight[e_GT] = round(e.edge_var.x)
            e_map[e_GT] = e

    # add additional vertex
    v_src = digraph_GT.add_vertex()

    for ndd in ndd_GT:
        e_GT = digraph_GT.add_edge(v_src, ndd)
        e_weight[e_GT] = 1.0
        # e_map[e_GT] = e

    # create a property map where v_map[v] for a pair-pair vertex v returns the GT vetex
    v_map = {v_obj[v]: v for v,_ in v_GT.iteritems()}

    return digraph_GT, v_GT, v_src, v_map, v_obj, e_map, e_weight


def edge_failure_from_seed(src_uid, tgt_uid, seed, failure_prob):
    '''
    return a 0 or 1 indicating edge failure from (src_uid -> tgt_uid)
    - use a random seed, such that these failures can be reproduced
    - for any pair, and any seed, returns 1 with prob. = failure_prob
    '''
    rs = np.random.RandomState(((src_uid+1)*((tgt_uid+2)**2)*(seed+src_uid+tgt_uid+1) ) % (2**32 - 1))
    return rs.binomial(1, failure_prob)

