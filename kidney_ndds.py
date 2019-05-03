"""This module has objects for non-directed donors (NDDs) and chains initiated by NDDs.

In contrast to many kidney-exchange formulations, we do not include in the main directed
graph a vertex for each NDD. Rather, the program maintains a separate array
of Ndd objects. Each of these Ndd objects maintains a list of outgoing edges, and
each of these edges points towards a vertex (which represents a donor-patient pair)
in the directed graph.
"""

from kidney_digraph import KidneyReadException
import pandas

import copy

class Ndd:
    """A non-directed donor"""
    def __init__(self,id = None):
        self.edges = []
        self.discount_frac = 0
        self.chain_weight = 0
        self.id = id
    def add_edge(self, ndd_edge):
        """Add an edge representing compatibility with a patient who appears as a
        vertex in the directed graph."""
        self.edges.append(ndd_edge)

    def remove_edge(self, ndd_edge):
        """Remove the edge."""
        self.edges.remove(ndd_edge)

    def get_edge(self,tgt_idx):
        '''
        return the edge that points to vertex with id tgt_idx, if it exists
        '''
        tgt_edges = [e for e in self.edges if e.tgt.id == tgt_idx]
        if len(tgt_edges) == 1:
            return tgt_edges[0]
        elif len(tgt_edges) == 0:
            raise Warning("edge not found to target index %d" % tgt_idx)
        else:
            raise Warning("multiple edges found to target index %d" % tgt_idx)

    # multiply edge weights ending in highly sensitized patients by a factor of (1+beta)
    def augment_weights(self, beta):
        for e in self.edges:
            if e.tgt.sensitized:
                new_score = (1.0 + beta) * e.score
                e.score = new_score

    # inverse of augment_weights
    def unaugment_weights(self, beta):
        for e in self.edges:
            if e.tgt.sensitized:
                new_score = e.score / (1.0 + beta)
                e.score = new_score

    def fair_copy(self, unfair=False):
        '''
        Return a copy of the ndd that only awards weights to edges that benefit highly sensitized patients
        '''

        if unfair:
            include_vertex = lambda v: not v.sensitized
        else:
            include_vertex = lambda v: v.sensitized

        n = Ndd(id=self.id)
        for e in self.edges:
            tgt = e.tgt
            new_score = e.score if include_vertex(tgt) else 0
            n.add_edge(NddEdge(tgt,new_score,src_id=n.id))
        return n

    def uniform_copy(self):
        '''
        Return a copy of the ndd with all weights set to 1
        '''
        n = Ndd(id=self.id)
        for e in self.edges:
            tgt = e.tgt
            new_score = 1.0
            n.add_edge(NddEdge(tgt,new_score,src_id=n.id))
        return n

class NddEdge:
    """An edge pointing from an NDD to a vertex in the directed graph"""
    def __init__(self, tgt, score, discount=0,fail=False, discount_frac=0,src_id = None):
        self.tgt = tgt
        self.tgt_id = tgt.id
        self.src_id = src_id
        self.score = score # edge weight
        self.discount = discount # discount value for the robust case
        self.fail = fail
        self.discount_frac = discount_frac
        self.sensitized = tgt.sensitized

    def __str__(self):
        return("NDD edge to V{}".format(self.tgt.id))

    def to_dict(self):
        e_dict = {'type':'ndd_edge',
                  'score':self.score,
                  'discount':self.discount,
                  'discount_frac':self.discount_frac,
                  'src_id':self.src_id,
                  'tgt_id': self.tgt_id,
                  'sensitized': self.sensitized
                  }
        return e_dict

    @classmethod
    def from_dict(cls, e_dict,ndds):
        # find NddEdge among provided ndds...
        # THIS DOESN'T HAVE TO BE A NddEdge FUNCTION

        # tgt = digraph.vs[e_dict['tgt_id']]
        # e = cls(tgt, e_dict['score'], discount=e_dict['discount'], discount_frac=e_dict['discount_frac'],src_id = e_dict['src_id'])
        # return e

        e_tgt_id = e_dict['tgt_id']
        for e in ndds[e_dict['src_id']].edges:
            if e.tgt_id == e_tgt_id:
                return e
        raise Warning("NddEdge not found")


    def display(self,gamma):
        # if gamma == 0:
        #     return "NDD Edge: tgt=%d, score=%f" % ( self.tgt.id, self.score)
        # else:
        return "NDD Edge: tgt=%d, score=%f, sens=%s, max_discount=%f, discount_frac=%f" % \
               (self.tgt.id, self.score, self.sensitized, self.discount, self.discount_frac)


def create_relabelled_ndds(ndds, old_to_new_vtx):
    """Creates a copy of a n array of NDDs, with target vertices changed.

    If a target vertex in an original NDD had ID i, then the new target vertex
    will be old_to_new_vtx[i].
    """
    new_ndds = [Ndd(id=ndd.id) for ndd in ndds]
    for i, ndd in enumerate(ndds):
        for edge in ndd.edges:
            new_ndds[i].add_edge(NddEdge(old_to_new_vtx[edge.tgt.id], edge.score,src_id=ndd.id))

    return new_ndds

def read_ndds(lines, digraph):
    """Reads NDDs from an array of strings in the .ndd format."""

    ndds = []
    ndd_count, edge_count = [int(x) for x in lines[0].split()]
    ndds = [Ndd(id=i) for i in range(ndd_count)]

    # Keep track of which edges have been created already so that we can
    # detect duplicates
    edge_exists = [[False for v in digraph.vs] for ndd in ndds]

    for line in lines[1:edge_count+1]:
        tokens = [t for t in line.split()]
        src_id = int(tokens[0])
        tgt_id = int(tokens[1])
        score = float(tokens[2])
        if src_id < 0 or src_id >= ndd_count:
            raise KidneyReadException("NDD index {} out of range.".format(src_id))
        if tgt_id < 0 or tgt_id >= digraph.n:
            raise KidneyReadException("Vertex index {} out of range.".format(tgt_id))
        if edge_exists[src_id][tgt_id]:
            raise KidneyReadException(
                    "Duplicate edge from NDD {0} to vertex {1}.".format(src_id, tgt_id))
        ndds[src_id].add_edge(NddEdge(digraph.vs[tgt_id], score,src_id=ndds[src_id].id))
        edge_exists[src_id][tgt_id] = True

    if lines[edge_count+1].split()[0] != "-1" or len(lines) < edge_count+2:
        raise KidneyReadException("Incorrect edge count")

    return ndds

class Chain(object):
    """A chain initiated by an NDD.
    
    Data members:
        ndd_index: The index of the NDD
        vtx_indices: The indices of the vertices in the chain, in order
        score: the chain's score
    """

    def __init__(self, ndd_index, vtx_indices, score):
        self.ndd_index = ndd_index
        self.vtx_indices = vtx_indices
        self.score = score
        self.discount_frac = 0.0

    def to_dict(self):
        ch_dict = {'ndd_index':self.ndd_index,
                   'vtx_indices':self.vtx_indices,
                   'discount_frac':self.discount_frac,
                   'score':self.score}
        return ch_dict

    @property
    def length(self):
        return len(self.vtx_indices)

    @classmethod
    def from_dict(cls, ch_dict):
        ch = cls(ch_dict['ndd_index'], ch_dict['vtx_indices'], ch_dict['score'])
        ch.discount_frac = ch_dict['discount_frac']
        return ch

    def __repr__(self):
        return ("Chain NDD{} ".format(self.ndd_index) +
                        " ".join(str(v) for v in self.vtx_indices) +
                        " with score " + str(self.score))

    def display(self):
        vtx_str = " ".join(str(v) for v in self.vtx_indices)
        return "Ndd %d, vtx = %s; score = %f, discount_frac = %f" % (self.ndd_index, vtx_str, self.score,self.discount_frac)



    def __cmp__(self, other):
        # Compare on NDD ID, then chain length, then on score, then
        # lexicographically on vtx indices
        if self.ndd_index < other.ndd_index:
            return -1
        elif self.ndd_index > other.ndd_index:
            return 1
        elif len(self.vtx_indices) < len(other.vtx_indices):
            return -1
        elif len(self.vtx_indices) > len(other.vtx_indices):
            return 1
        elif self.score < other.score:
            return -1
        elif self.score > other.score:
            return 1
        else:
            for i, j in zip(self.vtx_indices, other.vtx_indices):
                if i < j:
                    return -1
                elif i > j:
                    return 1
        return 0

    # return chain score with (possibly) different edge weights
    def get_score(self, digraph, ndds, edge_success_prob):
        # chain score is equal to e1.score * p + e2.score * p**2 + ... + en.score * p**n
        ndd = ndds[self.ndd_index]
        # find the vertex that the NDD first donates to
        tgt_id = self.vtx_indices[0]
        e1 = []
        for e in ndd.edges:
            if e.tgt.id == tgt_id:
                # get edge...
                e1 = e
                break
        if e1 == []:
            print("chain.update_score: could not find vertex id")
        score = e1.score * edge_success_prob # add score from ndd to first pair
        for j in range(len(self.vtx_indices) - 1):
            score += digraph.adj_mat[self.vtx_indices[j]][self.vtx_indices[j + 1]].score * edge_success_prob ** (j + 2)
        return score

    def score_after_failure(self, digraph, ndds):
        ndd = ndds[self.ndd_index]
        # find the vertex that the NDD first donates to
        tgt_id = self.vtx_indices[0]
        e1 = []
        for e in ndd.edges:
            if e.tgt.id == tgt_id:
                # get edge...
                e1 = e
                break
        if e1 == []:
            raise Warning("could not find vertex id")
        score = 0
        if e1.fail:
            return score
        else:
            score += e1.score  # add score from ndd to first pair
            for j in range(len(self.vtx_indices) - 1):
                if digraph.adj_mat[self.vtx_indices[j]][self.vtx_indices[j + 1]].fail:
                    return score
                else:
                    score += digraph.adj_mat[self.vtx_indices[j]][self.vtx_indices[j + 1]].score
        return score

# return the remaining chain after the property 'fail' has been added to the edges
def chain_after_failure(chain, digraph, ndds):
    ndd = ndds[chain.ndd_index]
    # find the vertex that the NDD first donates to
    tgt_id = chain.vtx_indices[0]
    e1 = []
    for e in ndd.edges:
        if e.tgt.id == tgt_id:
            # get edge...
            e1 = e
            break
    if e1 == []:
        raise Warning("could not find vertex id")
    score = 0
    max_j = 0
    vtx_indices = []
    if e1.fail:
        return Chain(chain.ndd_index, [], 0)
    else:
        j=0
        score += e1.score  # add score from ndd to first pair
        for j in range(len(chain.vtx_indices) - 1):
            if digraph.adj_mat[chain.vtx_indices[j]][chain.vtx_indices[j + 1]].fail:
                return Chain(chain.ndd_index, chain.vtx_indices[:(j+1)], score)
            else:
                score += digraph.adj_mat[chain.vtx_indices[j]][chain.vtx_indices[j + 1]].score

    return Chain(chain.ndd_index, chain.vtx_indices[:(j+1)], score)


def find_chains(digraph, ndds, max_chain, edge_success_prob=1):
    """Generate all chains with up to max_chain edges."""

    def find_chains_recurse(vertices, score):
        chains.append(Chain(ndd_idx, vertices[:], score))
        if len(vertices) < max_chain:
            for e in digraph.vs[vertices[-1]].edges:
                if e.tgt.id not in vertices:
                    vertices.append(e.tgt.id)
                    find_chains_recurse(vertices, score+e.score*edge_success_prob**len(vertices))
                    del vertices[-1]
    chains = []
    if max_chain == 0:
        return chains
    for ndd_idx, ndd in enumerate(ndds):
        for e in ndd.edges:
            vertices = [e.tgt.id]
            find_chains_recurse(vertices, e.score*edge_success_prob)
    return chains

# read altruists from *edgeweights.csv file
def read_from_kpd(edgeweights_filename, digraph, vtx_index):
    col_names = ['match_run','patient_id', 'patient_pair_id', 'donor_id', 'donor_pair_id', 'weight']
    df = pandas.read_csv(edgeweights_filename , names = col_names, skiprows=1)
    nonzero_edges = df.loc[df['weight'] > 0 ]  # last column is edge weights -- only take nonzero edges

    ndd_edges = nonzero_edges.loc[ nonzero_edges['donor_pair_id'].isnull()]# take only NDD edges
    ndd_id = set(list(ndd_edges['donor_id'].unique()))

    ndd_count = len(ndd_id)

    if ndd_count > 0:
        ndds = [Ndd(id=i) for i in range(ndd_count)]
        ndd_index = dict(zip( ndd_id, range(len(ndd_id)) )) # ndd_index[id] gives the index in the digraph

        # Keep track of which edges have been created already so that we can
        # detect duplicates
        edge_exists = [[False for v in digraph.vs] for ndd in ndds]

        # warned = False
        for index, row in ndd_edges.iterrows():
            src_id = ndd_index[row['donor_id']]
            tgt_id = vtx_index[row['patient_pair_id']]
            score = row['weight']
            if src_id < 0 or src_id >= ndd_count:
                raise KidneyReadException("NDD index {} out of range.".format(src_id))
            if tgt_id < 0 or tgt_id >= digraph.n:
                raise KidneyReadException("Vertex index {} out of range.".format(tgt_id))
            # if edge_exists[src_id][tgt_id] & ~warned:
                # print "# WARNING: Duplicate edge in file: {}".format(edgeweights_filename)
                # warned = True
                # raise KidneyReadException(
                #         "Duplicate edge from NDD {0} to vertex {1}.".format(src_id, tgt_id))
            ndds[src_id].add_edge(NddEdge(digraph.vs[tgt_id], score,src_id=ndds[src_id].id))
            edge_exists[src_id][tgt_id] = True
    else:
        ndds = []
        ndd_index = []

    return ndds,ndd_index

