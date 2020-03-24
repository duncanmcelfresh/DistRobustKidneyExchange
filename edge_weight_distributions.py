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

        # initiate features for each donor node
        for e in digraph.es:
            if not hasattr(e.tgt, 'p_list_fixed'):
                initialize_recip_unos(e.tgt, rs)
            e.p_list_fixed = e.tgt.p_list_fixed
            e.w_list = e.tgt.w_list
        for n in ndd_list:
            initialize_recip_unos(n, rs)
            for e in n.edges:
                e.p_list_fixed = n.p_list_fixed
                e.w_list = n.w_list

    elif dist_type == 'lkdpi':

        # initialize donor LKDPI & weight measurements (edges inherit weights from donors)
        for node in digraph.vs + ndd_list:
            initialize_lkdpi_node(node, alpha, num_weight_measurements, rs)

        for e in digraph.es:
            assert hasattr(e.src, 'lkdpi')

            # inherit these properties from the source (donor) node
            e.type = e.src.type
            e.lkdpi = e.src.lkdpi
            e.true_mean_weight = e.src.true_mean_weight
            e.draw_edge_weight = e.src.draw_edge_weight
            e.weight_list = e.src.weight_list

        for n in ndd_list:
            assert hasattr(n, 'lkdpi')
            for e in n.edges:
                # inherit these properties from the source (donor) node
                e.type = n.type
                e.type = n.lkdpi
                e.true_mean_weight = n.true_mean_weight
                e.draw_edge_weight = n.draw_edge_weight
                e.weight_list = n.weight_list

    # # this was needed only when edges had independent weights.
    # for e in digraph.es:
    #     initialize_edge(e, alpha, num_weight_measurements, rs)
    # for n in ndd_list:
    #     for e in n.edges:
    #         initialize_edge(e, alpha, num_weight_measurements, rs)


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


def initialize_recip_unos(recip, rs):
    """
    initialize the recipient features for the unos-type distribution.

    nothing is returned. the following fields are added to recipient Node recip:
    - e.p_list_fixed: probability of each criteria being met for this donor (1 is always met, 0 is never met)
    - e.w_list: weights associated with each criteria

    Args
        recip: (kidney_digraph.Node)
        rs: (numpy.random.Randomstate)
    """

    # probabilities of meeting each criteria
    p_list = [1.0,  # base points (100)
              0.005,  # exact tissue type match (200)
              0.12,  # highly sensitized (125)
              0.5,  # at least one antibody mismatch (-5)
              0.3,  # patient is <18 (100)
              0.5,  # prior organ donor (150)
              0.5,  # geographic proximity (1) (0, 25, 50, 75)]
              0.5,  # geographic proximity (2) (0, 25, 50, 75)]
              0.5]  # geographic proximity (3) (0, 25, 50, 75)]

    # weights for each criteria
    w_list = [1,  # base points (100)
              200,  # exact tissue type match (200)
              125,  # highly sensitized (125)
              -5,  # at least one antibody mismatch (-5)
              100,  # patient is <18 (100)
              150,  # prior organ donor (150)
              25,  # geographic proximity (1)
              25,  # geographic proximity (2)
              25]  # geographic proximity (3)

    # for each criteria, draw an initial value; this will be equal to the deterministic edge weight
    _, vars_realized = sample_unos_distribution(rs, w_list, p_list)

    # these criteria are fixed
    p_list_fixed = np.copy(p_list)
    p_list_fixed[2] = vars_realized[2]
    p_list_fixed[4] = vars_realized[4]
    p_list_fixed[5] = vars_realized[5]

    recip.p_list_fixed = p_list_fixed
    recip.w_list = w_list


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

    if rs.rand() < alpha:
        e.type = 1
        e.draw_edge_weight = lambda x: sample_unos_distribution(x, e.w_list, e.p_list_fixed)[0]
        e.weight_list = [e.draw_edge_weight(rs) for _ in range(num_weight_measurements)]
        e.true_mean_weight = np.dot(e.p_list_fixed, e.w_list)

    else:
        # deterministic edge
        e.type = 0

        # for each criteria, draw an initial value; this will be equal to the deterministic edge weight
        fixed_weight, _ = sample_unos_distribution(rs, e.w_list, e.p_list_fixed)
        e.draw_edge_weight = lambda x: fixed_weight
        e.weight_list = [fixed_weight] * num_weight_measurements
        e.true_mean_weight = fixed_weight


def initialize_lkdpi(e, rs):
    # initialize the LKDPI for an edge. this is the mean LKDPI (from a known distribution) +/- 1 SD
    if rs.rand() < 0.5:
        e.lkdpi = 14.93
    else:
        e.lkdpi = 59.37


def initialize_lkdpi_node(node, alpha, num_weight_measurements, rs):
    # initialize the LKDPI for a node. this is the mean LKDPI (from a known distribution) +/- 1 SD

    # determine whether the node is high or low LKDPI (equal probability of each)
    if rs.rand() < 0.5:
        node.lkdpi = 14.93
    else:
        node.lkdpi = 59.37

    # determine whether the node is stochastic or deterministic
    if rs.rand() < alpha:
        # stochastic node
        node.type = 1
        node.true_mean_weight = 14.78 * np.exp(-0.01239 * node.lkdpi)
        node.draw_edge_weight = lambda x: x.exponential(node.true_mean_weight)
        node.weight_list = [node.draw_edge_weight(rs) for _ in range(num_weight_measurements)]
    else:
        # deterministic node
        node.type = 0
        node.true_mean_weight = 14.78 * np.exp(-0.01239 * node.lkdpi)
        node.draw_edge_weight = lambda x: node.true_mean_weight
        node.weight_list = [node.draw_edge_weight(rs) for _ in range(num_weight_measurements)]


def full_lkdpi(x, rs):
    x.lkdpi = rs.normal(37.1506024096, 22.2170610307)


def initialize_edge_lkdpi(e, alpha, num_weight_measurements, rs):
    if rs.rand() < alpha:
        # stochastic edge
        e.type = 1
        e.true_mean_weight = 14.78 * np.exp(-0.01239 * e.lkdpi)
        e.draw_edge_weight = lambda x: x.exponential(e.true_mean_weight)
        e.weight_list = [e.draw_edge_weight(rs) for _ in range(num_weight_measurements)]
    else:
        # deterministic edge
        e.type = 0
        e.true_mean_weight = 14.78 * np.exp(-0.01239 * e.lkdpi)
        e.draw_edge_weight = lambda x: 14.78 * np.exp(-0.01239 * e.lkdpi)
        e.weight_list = [e.draw_edge_weight(rs) for _ in range(num_weight_measurements)]
