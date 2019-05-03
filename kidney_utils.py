import kidney_ndds
from kidney_digraph import *

EPS = 0.00001

class KidneyOptimException(Exception):
    pass

def check_validity(opt_result, digraph, ndds, max_cycle, max_chain, min_chain = None):
    """Check that the solution is valid.

    This method checks that:
      - all used edges exist
      - no vertex or NDD is used twice (which also ensures that no edge is used twice)
      - cycle and chain caps are respected
      - chain does not contain cycle (check for repeated tgt vertices)
    """

    # all used edges exist
    for chain in opt_result.chains:
        if chain.vtx_indices[0] not in [e.tgt.id for e in ndds[chain.ndd_index].edges]:
            raise KidneyOptimException("Edge from NDD {} to vertex {} is used but does not exist".format(
                    chain.ndd_index, chain.vtx_indices[0]))
    for cycle in opt_result.cycles:
        for i in range(len(cycle)):
            if digraph.adj_mat[cycle[i-1].id][cycle[i].id] is None:
                raise KidneyOptimException("Edge from vertex {} to vertex {} is used but does not exist".format(
                        cycle[i-1].id, cycle[i].id))
                
    # no vertex or NDD is used twice
    ndd_used = [False] * len(ndds)
    vtx_used = [False] * len(digraph.vs)
    for chain in opt_result.chains:
        if ndd_used[chain.ndd_index]:
            raise KidneyOptimException("NDD {} used more than once".format(chain.ndd_index))
        ndd_used[chain.ndd_index] = True
        for vtx_index in chain.vtx_indices:
            if vtx_used[vtx_index]:
                raise KidneyOptimException("Vertex {} used more than once".format(vtx_index))
            vtx_used[vtx_index] = True
            
    for cycle in opt_result.cycles:
        for vtx in cycle:
            if vtx_used[vtx.id]:
                raise KidneyOptimException("Vertex {} used more than once".format(vtx.id))
            vtx_used[vtx.id] = True

    # cycle and chain caps are respected
    for chain in opt_result.chains:
        if len(chain.vtx_indices) > max_chain:
            raise KidneyOptimException("The chain cap is violated")
    for cycle in opt_result.cycles:
        if len(cycle) > max_cycle:
            raise KidneyOptimException("The cycle cap is violated")
    if not min_chain is None:
        for chain in opt_result.chains:
            if len(chain.vtx_indices) < min_chain:
                raise KidneyOptimException("The min-chain cap is violated")

    # # min chain length is respected
    # if cfg.min_chain_len is not None:
    #     for chain in opt_result.chains:
    #         if len(set(chain.vtx_indices)) < cfg.min_chain_len:
    #             raise KidneyOptimException("The chain is below the min length (%d):\n %s" %
    #                                        (cfg.min_chain_len,chain.display()))

    # chains do not contain loops
    for chain in opt_result.chains:
        if len(set(chain.vtx_indices)) < len(chain.vtx_indices):
            raise KidneyOptimException("The chain contains loops:\n %s" % chain.display())

def get_dist_from_nearest_ndd(digraph, ndds):
    """ For each donor-patient pair V, this returns the length of the
    shortest path from an NDD to V, or 999999999 if no path from an NDD
    to V exists.
    """
    
    # Get a set of donor-patient pairs who are the target of an edge from an NDD
    ndd_targets = set()
    for ndd in ndds:
        for edge in ndd.edges:
            ndd_targets.add(edge.tgt)

    # Breadth-first search
    q = deque(ndd_targets)
    distances = [999999999] * len(digraph.vs)
    for v in ndd_targets:
        distances[v.id] = 1

    while q:
        v = q.popleft()
        for e in v.edges:
            w = e.tgt
            if distances[w.id] == 999999999:
                distances[w.id] = distances[v.id] + 1
                q.append(w)

    return distances


def find_vertex_chain_participation(digraph, ndds,max_chain):
    """ For each donor-patient pair V, add a property "can_be_in_chain_list",
    which is a list of booleans: can_be_in_chain_list[i] = True if v can be in a chain
    initiated by ndd i (True if v is within the chain cap of ndd i, False otherwise)
    """

    for v in digraph.vs:
        v.can_be_in_chain_list = [False for _ in ndds]

    for i_ndd,ndd in enumerate(ndds):
        # Get a set of donor-patient pairs who are the target of an edge from an NDD
        ndd_targets = set()
        for edge in ndd.edges:
            ndd_targets.add(edge.tgt)

        # Breadth-first search
        q = deque(ndd_targets)
        distances = [999999999] * len(digraph.vs)
        for v in ndd_targets:
            distances[v.id] = 1

        while q:
            v = q.popleft()
            for e in v.edges:
                w = e.tgt
                if distances[w.id] == 999999999:
                    distances[w.id] = distances[v.id] + 1
                    q.append(w)

        for v,dist in zip(digraph.vs,distances):
            if dist <= max_chain:
                v.can_be_in_chain_list[i_ndd] = True


def find_selected_path(v_id, next_vv):
    path = [v_id]
    while v_id in next_vv:
        v_id = next_vv[v_id]
        path.append(v_id)
    return path
        
def find_selected_cycle(v_id, next_vv):
    cycle = [v_id]
    while v_id in next_vv:
        v_id = next_vv[v_id]
        if v_id in cycle:
            return cycle
        else:
            cycle.append(v_id)
    return None

def get_optimal_chains(digraph, ndds, edge_success_prob=1):
    # Chain edges
    chain_next_vv = {e.src.id: e.tgt.id
                     for e in digraph.es
                     for var in e.grb_vars
                     if var.x > 0.1}  # changed to Xn from x by Duncan

    optimal_chains = []
    for i, ndd in enumerate(ndds):
        for e in ndd.edges:
            if e.edge_var.x > 0.1:
                vtx_indices = find_selected_path(e.tgt.id, chain_next_vv)
                # Get score of edge from NDD
                score = e.score * edge_success_prob
                # Add scores of edges between vertices
                for j in range(len(vtx_indices) - 1):
                    score += digraph.adj_mat[vtx_indices[j]][vtx_indices[j + 1]].score * edge_success_prob ** (j + 2)
                optimal_chains.append(kidney_ndds.Chain(i, vtx_indices, score))

    return optimal_chains

# added by duncan
def get_optimal_chains_pctsp(digraph, ndds):
    # Chain edges
    edge_success_prob = 1.0
    chain_next_vv = {e.src.id: e.tgt.id
                     for e in digraph.es
                     if e.edge_var.x > 0.5}

    optimal_chains = []
    for i, ndd in enumerate(ndds):
        for e in ndd.edges:
            if e.edge_var.x > 0.5:
                vtx_indices = find_selected_path(e.tgt.id, chain_next_vv)
                # Get score of edge from NDD
                score = e.score * edge_success_prob
                # Add scores of edges between vertices
                for j in range(len(vtx_indices) - 1):
                    score += digraph.adj_mat[vtx_indices[j]][vtx_indices[j + 1]].score * edge_success_prob ** (j + 2)
                optimal_chains.append(kidney_ndds.Chain(i, vtx_indices, score))

    return optimal_chains


def selected_edges_to_cycles(digraph, cycle_start_vv, cycle_next_vv):
    cycles = [find_selected_cycle(start_v, cycle_next_vv) for start_v in cycle_start_vv]
    # Remove "cycles" that are really part of a chain
    cycles = [c for c in cycles if c is not None]
    # Remove duplicated cycles
    cycles = [c for c in cycles if c[0] == min(c)]
    # Use vertices instead of indices
    return [[digraph.vs[v_id] for v_id in c] for c in cycles]

# return True if cycle c contains edge e
# c is a list of kidney_digraph.Vertex objects (with the first vertex not repeated
# edge is a kidney_digraph.Edge objects
def cycle_contains_edge(c,e):
    if e.src in c:
        i = c.index(e.src)
        if e.tgt == c[(i+1) % len(c)]:
            return True
        else:
            return False
    return False


# -------------------------------------------------------------------------------------------------
#
#                           Functions for Variable Uncertainty Budget
#
# -------------------------------------------------------------------------------------------------
from scipy.special import binom
from scipy.optimize import minimize
import math

def B_bound(num_E,gamma):
    '''
    The upper-bound on probability that realized edge weights fall outside of the U-set:
    Assuming symmetric interval uncertainty, and realized edge weights symmetrically distributed about
    their nominal value.
    From Bertsimas, Price of Robustness
    '''
    eta = (gamma + num_E)/2.0
    fl_eta = int(math.floor(eta))
    mu = float(eta - fl_eta)
    return math.pow(2,-num_E)*((1.0-mu)*binom(num_E,fl_eta)
                              + sum( binom(num_E,l) for l in range(fl_eta+1,int(num_E)+1) ))


def gamma_symmetric_edge_weights(x_norm,epsilon):
    '''
    Variable budget function for symmetric cost uncertainty (from Poss & Bergamo)

    input:
        - x_norm    : number of edges in the solution
        - epsilon   : protection level (realized edge weights will be outside of U-set with prob. epsilon
    '''

    # the first constraint is that B_bound <= epsilon,
    # the second is that gamma >= 0
    # the third is that gamma <= x_norm
    constr = ({'type':'ineq',
               'fun':lambda g: epsilon - B_bound(x_norm,g)
              },
              {'type':'ineq',
               'fun':lambda g: g},
              {'type': 'ineq',
               'fun': lambda g: x_norm - g})

    func = lambda gamma: gamma # we just want to minimize gamma

    # method = Constrained Optimization BY Linear Approximation (COBYLA)
    res = minimize(func,0.01, constraints=constr,method='COBYLA')

    # if the minimization is succesful, return the result. otherwise, return x_norm
    if res.success:
        return max(round(res.fun,4),0)
    else:
        return x_norm


from scipy.special import betainc  # betainc(a,b,x)
from scipy.stats import binom as binomial_dist  # pmf(x,n,p)
from math import floor


def G_bound(n, m, p, k, gamma):
    pk = math.pow(1 - p, k)
    floor_gamma = int(floor(gamma))
    if gamma >= m:
        s1 = sum(binomial_dist.pmf(y, n, 1 - pk) for y in range(0, floor_gamma - m + 1))
        s2 = sum(betainc(m - gamma + y, gamma - y + 1, 1 - p) * binomial_dist.pmf(y, n, 1 - pk) for y in
                 range(floor_gamma - m + 1, min(n, floor_gamma) + 1))
        return s1 + s2
    else:
        return sum(betainc(m - gamma + y, gamma - y + 1, 1 - p) * binomial_dist.pmf(y, n, 1 - pk) for y in
                   range(0, min(n, floor_gamma) + 1))


def G_bound_2(n, m, p, k, gamma):
    pk = math.pow(1 - p, k)
    floor_gamma = int(floor(gamma))
    return sum(binom_cdf(gamma - y, m, p) * binomial_dist.pmf(y, n, 1 - pk) for y in range(0, n + 1))


def binom_cdf(y, n, p):
    '''
    CDF of the binomial distribution, using the regularized incomplete beta function
    '''
    if y >= n:
        return 1.0
    elif y < 0:
        return 0.0
    else:
        return betainc(n - y, y + 1, 1 - p)


# @np.vectorize
def gamma_homogeneous_edge_failure(n, m, p, k, epsilon):
    '''
    Variable budget function for homogeneous edge failure probability p

    input:
        - n         : number of cycles in the matching
        - m         : number of chains in the matching
        - k         : cycle cap
        - p         : edge failure probability
        - epsilon   : protection level (realized cycle/chain weights will be outside of U-set with prob. epsilon)
    '''

    # the first constraint is that 1 - G_bound <= epsilon,
    # the second is that gamma >= 0
    # the third is that gamma <= n+m
    constr = ({'type': 'ineq',
               'fun': lambda g: epsilon - (1 - G_bound_2(n, m, p, k, g))
               },
              {'type': 'ineq',
               'fun': lambda g: g},
              {'type': 'ineq',
               'fun': lambda g: n + m - g})

    func = lambda gamma: gamma  # we just want to minimize gamma

    # method = Constrained Optimization BY Linear Approximation (COBYLA)
    res = minimize(func, 0.01, constraints=constr, method='COBYLA')

    # if the minimization is succesful, return the result. otherwise, return x_norm
    if res.success:
        return max(round(res.fun, 4), 0)
    else:
        return n + m
