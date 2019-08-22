# functions for defining and sampling edge weight distributions
import numpy as np


def edge_weight_distribution_binary(edge_type, rs):
    if edge_type == 0:
        return 0.5
    else:
        return rs.choice([0.0, 1.0], p=[0.5, 0.5], size=1)[0]


def sample_unos_distribution(rs, w_list, p_list):
    """return an edge weight for a unos-inspired distribution
    total edge distribution is:
    w_e ~ \sum_c p_{c} w_c

    Args:
        rs: numpy.random.Randomstate
        w_list: (list/array). list of weights for each criteria
        p_list: (list/array). list of probabilities for each criteria. these are used to draw bernoulli random variables
            for each criteria.
    """
    realized_vars = (rs.rand(len(p_list)) <= p_list).astype(int)
    return np.dot(w_list, realized_vars), realized_vars


def initialize_edge_weights(digraph, ndd_list, num_weight_measurements, alpha, rs, dist_type):
    """
    initialize the edge weight distribution for each edge.

    nothing is returned. the following fields are added to each edge of the digraph and each ndd in ndd_list:
    - e.type: 1 if edge is probabilistic, 0 if deterministic
    - e.draw_edge_weight: function handle that takes a single argument (a random state) and returns a float drawn from
        the edge weight distribution
    - e.weight_list: a list of edge wieght draws, from the edge's distribution.
    - e.true_mean_weight: the edge's true mean weight

    Args:
        digraph: (kidney_digraph.Graph). the edges of this graph will be initialized
        ndd_list: (list(kidney_ndds.Ndd). each edge of these Ndds will be initialized
        num_weight_measurements: (int). number of edge weight measurements to draw
        alpha: (float). probability that each edge will be random
        rs: (numpy.random.Randomstate)
        dist_type: (str). edge distribution type. either "unos", "binary", or "dro"
    """

    assert dist_type in ['binary', 'unos', 'lkdpi']

    if dist_type == 'binary':
        initialize_edge = initialize_edge_binary
    elif dist_type == 'unos':
        initialize_edge = initialize_edge_unos
    elif dist_type == 'lkdpi':
        initialize_edge = initialize_edge_lkdpi

        # initiate a fixed lkdpi for each edge
        for e in digraph.es:
            initial_lkdpi(e, rs)
        for n in ndd_list:
            for e in n.edges:
                initial_lkdpi(e, rs)

    for e in digraph.es:
        initialize_edge(e, alpha, num_weight_measurements, rs)
    for n in ndd_list:
        for e in n.edges:
            initialize_edge(e, alpha, num_weight_measurements, rs)


########################################################################################################################
# functions for initializing edge weight distributions
########################################################################################################################

def initialize_edge_binary(e, alpha, num_weight_measurements, rs):
    """
     initialize the edge distribution for an Edge with a binary distribution.
     edges are randomly chosen to tbe deterministic or random.

     random edges have weight 0 or 1 with probability 0.5 each. deterministic edges have weight 0.5

     nothing is returned. the following fields are added to Edge e:
     - e.type: 1 if edge is probabilistic, 0 if deterministic
     - e.draw_edge_weight: function handle that takes a single argument (a random state) and returns a float drawn from
         the edge weight distribution
     - e.weight_list: a list of edge wieght draws, from the edge's distribution.
     - e.true_mean_weight: the edge's true mean weight

     Args
         e: (kidney_digraph.Edge)
         alpha: (float). probability that the edge is random (as opposed to deterministic)
         num_weight_measurements: (int). number of draws from the edge distribution. these will be set to
             e.weight_list
         rs: (numpy.random.Randomstate)
     """
    e.type = 1 if rs.rand() < alpha else 0
    e.draw_edge_weight = lambda x: edge_weight_distribution_binary(e.type, x)
    e.weight_list = [e.draw_edge_weight(rs) for _ in range(num_weight_measurements)]
    e.true_mean_weight = 0.5


def initialize_edge_unos(e, alpha, num_weight_measurements, rs):
    """
    initialize the edge distribution for an Edge using the unos-type distribution.
    edges are randomly chosen to tbe deterministic or random.

    nothing is returned. the following fields are added to Edge e:
    - e.type: 1 if edge is probabilistic, 0 if deterministic
    - e.draw_edge_weight: function handle that takes a single argument (a random state) and returns a float drawn from
        the edge weight distribution
    - e.weight_list: a list of edge wieght draws, from the edge's distribution.
    - e.true_mean_weight: the edge's true mean weight

    Args
        e: (kidney_digraph.Edge)
        alpha: (float). probability that the edge is random (as opposed to deterministic)
        num_weight_measurements: (int). number of draws from the edge distribution. these will be set to
            e.weight_list
        rs: (numpy.random.Randomstate)
    """

    # probabilities of meeting each criteria
    p_list = [1.0,  # base points (100)
              0.5,  # exact tissue type match (200)
              0.5,  # highly sensitized (125)
              0.5,  # at least one antibody mismatch (-5)
              0.3,  # patient is <18 (100)
              0.3,  # prior organ donor (150)
              0.5,  # geographic proximity (1) (0, 25, 50, 75)]
              0.5,  # geographic proximity (2) (0, 25, 50, 75)]
              0.5]  # geographic proximity (3) (0, 25, 50, 75)]

    # weights for each criteria
    w_list = [1,  # base points (100)
              200,  # exact tissue type match (200)
              125,  # highly sensitized (125)
              100,  # at least one antibody mismatch (-5)
              100,  # patient is <18 (100)
              150,  # prior organ donor (150)
              25,  # geographic proximity (1)
              25,  # geographic proximity (2)
              25]  # geographic proximity (3)

    # set a type : with probability alpha, the edge is random
    if rs.rand() < alpha:
        # probabilistic edge
        e.type = 1

        # for each criteria, draw an initial value; this will be equal to the deterministic edge weight
        _, vars_realized = sample_unos_distribution(rs, w_list, p_list)

        # these criteria are fixed
        # p_list_fixed = np.copy(p_list)
        # p_list_fixed[4] = vars_realized[4]
        # p_list_fixed[5] = vars_realized[5]
        # p_list_fixed[6] = vars_realized[6]
        # p_list_fixed[7] = vars_realized[7]
        # p_list_fixed[8] = vars_realized[8]
        p_list_fixed = p_list

        e.draw_edge_weight = lambda x: sample_unos_distribution(x, w_list, p_list_fixed)[0]
        e.weight_list = [e.draw_edge_weight(rs) for _ in range(num_weight_measurements)]
        e.true_mean_weight = np.dot(p_list_fixed, w_list)

    else:
        # deterministic edge
        e.type = 0

        # for each criteria, draw an initial value; this will be equal to the deterministic edge weight
        fixed_weight, _ = sample_unos_distribution(rs, w_list, p_list)
        e.draw_edge_weight = lambda x: fixed_weight
        e.weight_list = [fixed_weight] * num_weight_measurements
        e.true_mean_weight = fixed_weight


def initial_lkdpi(e, rs):
    e.lkdpi = rs.normal(37.1506024096, 22.2170610307)


def initialize_edge_lkdpi(e, type, num_weight_measurements, rs):
    e.true_mean_weight = 14.78 * np.exp(-0.01239 * e.lkdpi)
    e.draw_edge_weight = lambda x: x.exponential(e.true_mean_weight)
    e.weight_list = [e.draw_edge_weight(rs) for _ in range(num_weight_measurements)]
