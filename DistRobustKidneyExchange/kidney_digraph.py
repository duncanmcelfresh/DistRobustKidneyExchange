"""A Digraph class which can be used for representing donor-patient pairs
(as vertices) and their compatibilities (as weighted edges), along with
some related methods.
Modified by Duncan from https://github.com/jamestrimble/kidney_solver
"""

from collections import deque
import pandas
import json
import os

class KidneyReadException(Exception):
    pass

def cycle_score(cycle, digraph):
    """Calculate the sum of a cycle's edge scores.

    Args:
        cycle: A list of Vertex objects in the cycle, with the first Vertex not repeated.
        digraph: The digraph in which this cycle appears.
    """

    return sum(digraph.adj_mat[cycle[i-1].id][cycle[i].id].score
                        for i in range(len(cycle)))

def cycle_score_weighted(cycle, digraph):
    """Calculate the sum of a cycle's edge scores.

    Args:
        cycle: A list of Vertex objects in the cycle, with the first Vertex not repeated.
        digraph: The digraph in which this cycle appears.
    """

    return sum(digraph.adj_mat[cycle[i-1].id][cycle[i].id].score *(1.0 + digraph.adj_mat[cycle[i-1].id][cycle[i].id].alpha)
                        for i in range(len(cycle)))


def failure_aware_cycle_score(cycle, digraph, edge_success_prob):
    """Calculate a cycle's total score, with edge failures and no backarc recourse.

    Args:
        cycle: A list of Vertex objects in the cycle, with the first Vertex not repeated.
            UPDATE: cycle = a Cycle object
        digraph: The digraph in which this cycle appears.
        edge_success_prob: The problem that any given edge will NOT fail
    """

    return sum(digraph.adj_mat[cycle[i-1].id][cycle[i].id].score
                    for i in range(len(cycle))) * edge_success_prob**len(cycle)

def failure_aware_cycle_score_weighted(cycle, digraph, edge_success_prob):
    """Calculate a cycle's total score, with edge failures and no backarc recourse.

    Args:
        cycle: A list of Vertex objects in the cycle, with the first Vertex not repeated.
            UPDATE: cycle = a Cycle object
        digraph: The digraph in which this cycle appears.
        edge_success_prob: The problem that any given edge will NOT fail
    """
    return sum(digraph.adj_mat[cycle[i-1].id][cycle[i].id].score * (1.0 + digraph.adj_mat[cycle[i-1].id][cycle[i].id].alpha)
                    for i in range(len(cycle))) * edge_success_prob**len(cycle)


class Vertex:
    """A vertex in a directed graph (see the Digraph class)."""

    def __init__(self, id):
        self.id = id
        self.edges = []
        self.sensitized = False

    def __str__(self):
        return ("V{}".format(self.id))

class Cycle:
    """A cycle in a directed graph.

    Contains:
    - list of vertices, in order
    - list of edges
    - cycle score
    """
    def __init__(self,vs):
        self.vs = vs
        self.score = 0
        self.length = len(vs)
        self.discount_frac = 0
        self.edges = []

    def to_dict(self):
        cy_dict = {'vs':[v.id for v in self.vs],
                   'discount_frac':self.discount_frac,
                   'score':self.score}
        return cy_dict

    def to_dict_dynamic(self):
        '''
        for dynamic experiments, save different info
        '''
        cy_dict = {'vs':[v.to_dict_dynamic() for v in self.vs],
                   'score':self.score}
        return cy_dict

    @classmethod
    def from_dict(cls, cy_dict,digraph):
        cy = cls([digraph.vs[vi] for vi in cy_dict['vs']])
        cy.discount_frac = cy_dict['discount_frac']
        cy.score = cy_dict['score']
        cy.add_edges(digraph.es)
        return cy


    def __cmp__(self,other):
        if min(self.vs) < min(other.vs):
            return -1
        elif min(self.vs) > min(other.vs):
            return 1
        else:
            return 1

    def __len__(self):
        return self.length

    def display(self):
        vtx_str = " ".join(str(v) for v in self.vs)
        return "L=%2d; %15s; score = %f; discount_frac = %f" % (self.length,vtx_str, self.score, self.discount_frac )

    def contains_edge(self, e):
        if e.src in self.vs:
            i = self.vs.index(e.src)
            if e.tgt == self.vs[(i + 1) % self.length]:
                return True
            else:
                return False
        return False

    def add_edges(self,es):
        # create an unordered list of edges in the cycle
        self.edges = [e for e in es if self.contains_edge(e)]
    #
    # def cycle_score(self, digraph):
    #     """Calculate the sum of a cycle's edge scores.
    #
    #     Args:
    #         cycle: A list of Vertex objects in the cycle, with the first Vertex not repeated.
    #         digraph: The digraph in which this cycle appears.
    #     """
    #
    #     return sum(digraph.adj_mat[cycle[i-1].id][cycle[i].id].score
    #                         for i in range(len(cycle)))


class Edge:
    """An edge in a directed graph (see the Digraph class)."""

    def __init__(self, id, score, src, tgt, discount=0,fail=False, discount_frac=0):
        self.id = id
        self.score = score # edge weight
        self.discount = discount # maximum discount value for the robust case
        self.discount_frac = discount_frac # amount of discount (between 0,1)
        self.fail = fail # whether edge failed
        self.src = src   # source vertex
        self.src_id = src.id
        self.tgt = tgt # target vertex
        self.tgt_id = tgt.id
        self.sensitized = tgt.sensitized

    def __str__(self):
        return ("V" + str(self.src.id) + "-V" + str(self.tgt.id))

    def to_dict(self):
        e_dict = {'type':'pair_edge',
                  'id':self.id,
                  'score':self.score,
                  'discount':self.discount,
                  'discount_frac':self.discount_frac,
                  'src_id':self.src_id,
                  'tgt_id': self.tgt_id,
                  'sensitized': self.sensitized
                  }
        return e_dict

    @classmethod
    def from_dict(cls, e_dict,digraph):
        # find edge among digraph.es
        # THIS DOESN'T HAVE TO BE A CLASS METHOD FOR Edge
        return digraph.adj_mat[e_dict['src_id']][e_dict['tgt_id']]
        # src = digraph.vs[e_dict['src_id']]
        # tgt = digraph.vs[e_dict['tgt_id']]
        # e = cls(e_dict['id'], e_dict['score'], src, tgt, discount=e_dict['discount'], discount_frac=e_dict['discount_frac'])
        # return e


    def display(self,gamma):
        # if gamma == 0:
        #     return "src=%d, tgt=%d, score=%f" % (self.src.id, self.tgt.id, self.score)
        # else:
        return "src=%d, tgt=%d, score=%f, sens=%s, max_discount=%f, discount_frac=%f" % (
        self.src.id, self.tgt.id, self.score, self.sensitized, self.discount, self.discount_frac)

class Digraph:
    """A directed graph, in which each edge has a numeric score.

    Data members:
        n: the number of vertices in the digraph
        vs: an array of Vertex objects, such that vs[i].id == i
        es: an array of Edge objects, such that es[i].id = i
    """

    def __init__(self, n,
                 vs_ids = None,
                 adj_mat = None):
        """Create a Digraph with n vertices"""
        self.n = n

        # add vertices with specified IDs
        if vs_ids is not None:
            # check length of vs
            if len(vs_ids) != n:
                raise Warning("length of vs_ids is %d but n is %d" % (len(vs_ids),n))
            else:
                self.vs = [Vertex(id) for id in vs_ids]
        else:
            self.vs = [Vertex(i) for i in range(n)]

        # add an existing adj_mat
        if adj_mat is not None:
            # check dimensions
            if np.array(adj_mat).shape != (n,n):
                raise Warning("adj_mat is not the correct size for n=%d: %s" % (n,str(np.array(adj_mat).shape)))
            # check diag (no self-edges)
            for i in range(n):
                if adj_mat[i][i] is not None:
                    raise Warning("adj_mat has a diagonal entry at (%d,d%): %s" % (i,i,str(adj_mat[i][i])))
        else:
            self.adj_mat = [[None for x in range(n)] for x in range(n)]

        self.es = []
        self.cycles = []

    def add_edge(self, score, source, tgt):
        """Add an edge to the digraph

        Args:
            score: the edge's score, as a float
            source: the source Vertex
            tgt: the edge's target Vertex
        """

        id = len(self.es)
        e = Edge(id, score, source, tgt)
        self.es.append(e)
        source.edges.append(e)
        self.adj_mat[source.id][tgt.id] = e

    def remove_edge(self,e):
        self.es.remove(e)
        e.src.edges.remove(e)
        self.adj_mat[e.src.id][e.tgt.id] = None

    import os
    def find_cycles(self, max_length, cycle_file=None):
        """Find cycles of length up to max_length in the digraph.
            - If cycle_file is given and exists, read cycles and max length
                - If the file's max length is < max_length, find cycles and write cycle file
            - If file is given and does not exist, write it
        Returns:
            a list of cycles. Each cycle is represented as a list of
            vertices, with the first vertex _not_ repeated at the end.
        """

        if cycle_file is None: # no cycle file given
            cycle_list = [cycle for cycle in self.generate_cycles(max_length)]
        elif os.path.isfile(cycle_file): # cycle file given and exists
            # read cycles
            max_len_file = self.read_cycle_maxlen_from_file(cycle_file)
            if max_len_file < max_length:
                # max len is too short, re-generate cycles and write
                cycle_list = [cycle for cycle in self.generate_cycles(max_length)]
                self.write_cycles_to_file(cycle_file, max_length, cycle_list)
            else:
                _, cycle_ind_list = self.read_cycles_from_file(cycle_file,max_length)
                # get cycles
                cycle_list =  [[self.vs[ic] for ic in c] for c in cycle_ind_list] # [[self.vs[ic] for ic in c] for c in cycle_ind_list if len(c) <= max_length]
        else: # cycle file is given and does not exist
            # write cycles
            cycle_list = [cycle for cycle in self.generate_cycles(max_length)]
            self.write_cycles_to_file(cycle_file, max_length, cycle_list)

        return cycle_list

    def write_cycles_to_file(self,cycle_file,max_length,cycle_list):
        cycle_ind_list = [[v.id for v in c] for c in cycle_list]
        # cycle_data = {'max_length': max_length, 'cycle_ind_list': cycle_ind_list}
        maxlen_data = {'max_length': max_length}
        # cycle_data = {'cycle_ind_list': cycle_ind_list}
        with open(cycle_file, "wb") as f:
            # json.dump(cycle_data, f, indent=4)
            f.write(json.dumps(max_length))
            f.write("\n")
            for c in cycle_list: # in cycle_ind_list:
                f.write(json.dumps([v.id for v in c]))
                f.write("\n")

    def read_cycle_maxlen_from_file(self,cycle_file):
        with open(cycle_file, "rb") as f:
            max_length = json.loads(f.readline())
        return max_length

    def read_cycles_from_file(self,cycle_file, max_len):
        '''
        Read cycle index lists from file, line by line.
        Only read cycles that are <= max_len
        '''
        cycle_ind_list = []
        with open(cycle_file, "rb") as f:
            max_length = json.loads(f.readline())
            line = f.readline()
            while line:
                cycle_list = json.loads(line)
                if len(cycle_list) <= max_len:
                    cycle_ind_list.append(cycle_list)
                line = f.readline()

        return max_length, cycle_ind_list


    def generate_cycles(self, max_length):
        """Generate cycles of length up to max_length in the digraph.

        Each cycle yielded by this generator is represented as a list of
        vertices, with the first vertex _not_ repeated at the end.
        """

        vtx_used = [False] * len(self.vs)  # vtx_used[i]==True iff vertex i is in current path

        def cycle(current_path):
            last_vtx = current_path[-1]
            if self.edge_exists(last_vtx, current_path[0]):
                yield current_path[:]
            if len(current_path) < max_length:
                for e in last_vtx.edges: 
                    v = e.tgt
                    if (len(current_path) + shortest_paths_to_low_vtx[v.id] <= max_length
                                and not vtx_used[v.id]):
                        current_path.append(v)
                        vtx_used[v.id] = True
                        for c in cycle(current_path):
                            yield c
                        vtx_used[v.id] = False
                        del current_path[-1]

        # Adjacency lists for transpose graph
        transp_adj_lists = [[] for v in self.vs]
        for edge in self.es:
            transp_adj_lists[edge.tgt.id].append(edge.src)

        for v in self.vs:
            shortest_paths_to_low_vtx = self.calculate_shortest_path_lengths(
                    v, max_length - 1,
                    lambda u: (w for w in transp_adj_lists[u.id] if w.id > v.id))
            vtx_used[v.id] = True
            for c in cycle([v]):
                yield c
            vtx_used[v.id] = False
    
    def get_shortest_path_from_low_vtx(self, low_vtx, max_path):
        """ Returns an array of path lengths. For each v > low_vtx, if the shortest
            path from low_vtx to v is shorter than max_path, then element v of the array
            will be the length of this shortest path. Otherwise, element v will be
            999999999."""
        return self.calculate_shortest_path_lengths(self.vs[low_vtx], max_path,
                    adj_list_accessor=lambda v: (e.tgt for e in v.edges if e.tgt.id >= low_vtx))

    def get_shortest_path_to_low_vtx(self, low_vtx, max_path):
        """ Returns an array of path lengths. For each v > low_vtx, if the shortest
            path to low_vtx from v is shorter than max_path, then element v of the array
            will be the length of this shortest path. Otherwise, element v will be
            999999999."""
        def adj_list_accessor(v):
            for i in range(low_vtx, len(self.vs)):
                if self.adj_mat[i][v.id]:
                    yield self.vs[i]
            
        return self.calculate_shortest_path_lengths(self.vs[low_vtx], max_path,
                    adj_list_accessor=adj_list_accessor)

    def calculate_shortest_path_lengths(self, from_v, max_dist,
                adj_list_accessor=lambda v: (e.tgt for e in v.edges)):
        """Calculate the length of the shortest path from vertex from_v to each
        vertex with a greater or equal index, using paths containing
        only vertices indexed greater than or equal to from_v.

        Return value: a list of distances of length equal to the number of vertices.
        If the shortest path to a vertex is greater than max_dist, the list element
        will be 999999999.

        Args:
            from_v: The starting vertex
            max_dist: The maximum distance we're interested in
            adj_list_accessor: A function taking a vertex and returning an
                iterable of out-edge targets
        """
        # Breadth-first search
        q = deque([from_v])
        distances = [999999999] * len(self.vs)
        distances[from_v.id] = 0

        while q:
            v = q.popleft()
            #Note: >= is used instead of == on next line in case max_dist<0
            if distances[v.id] >= max_dist:
                break
            for w in adj_list_accessor(v):
                if distances[w.id] == 999999999:
                    distances[w.id] = distances[v.id] + 1
                    q.append(w)

        return distances

    def edge_exists(self, v1, v2):
        """Returns true if and only if an edge exists from Vertex v1 to Vertex v2."""

        return self.adj_mat[v1.id][v2.id] is not None
                    
    def induced_subgraph(self, vertices):
        """Returns the subgraph indiced by a given list of vertices."""

        subgraph = Digraph(len(vertices))
        for i, v in enumerate(vertices):
            for j, w in enumerate(vertices):
                e = self.adj_mat[v.id][w.id]
                if e is not None:
                    new_src = subgraph.vs[i]
                    new_tgt = subgraph.vs[j]
                    subgraph.add_edge(e.score, new_src, new_tgt)
        return subgraph

    # read the *recipient.csv file to label the sensitized recipients
    def KPD_label_sensitized(self, recipient_filename, vtx_index):
        df = pandas.read_csv(recipient_filename)
        df.columns = map(str.lower, df.columns)
        sensitized_ids = df['kpd_candidate_id'].loc[df['highly_sensitized'] == 'Y']
        for n in set(sensitized_ids).intersection(vtx_index.keys()):
            vi = vtx_index[n]
            self.vs[vi].sensitized = True

    # multiply edge weights ending in highly sensitized patients by a factor of (1+beta)
    def augment_weights(self, beta):
        for e in self.es:
            if self.vs[e.tgt.id].sensitized:
                new_score = (1.0 + beta) * e.score
                e.score = new_score

    # inverse of augment_weights
    def unaugment_weights(self, beta):
        for e in self.es:
            if self.vs[e.tgt.id].sensitized:
                new_score = e.score/(1.0 + beta)
                e.score = new_score

    def get_num_sensitized(self):
        num = 0
        for v in self.vs:
            if v.sensitized:
                num += 1
        return num

    def get_num_pairs(self):
        return len(self.vs)

    def fair_copy(self, unfair=False):
        '''
        Return a copy of the graph that only assigns weight to edges
        that benefit highly sensitized patients
        '''
        if unfair:
            include_vertex = lambda v: not v.sensitized
        else:
            include_vertex = lambda v: v.sensitized

        d = Digraph(self.n)
        d.vs = self.vs
        for e in self.es:
            tgt_v = e.tgt
            new_score = e.score if include_vertex(tgt_v) else 0
            d.add_edge(new_score, e.src, tgt_v)
        return d

    def uniform_copy(self):
        '''
        Return a copy of the graph with all edge weights set to 1
        '''
        d = Digraph(self.n)
        d.vs = self.vs
        for e in self.es:
            tgt_v = e.tgt
            new_score = 1.0
            d.add_edge(new_score, e.src, tgt_v)
        return d

    def __str__(self):
        return "\n".join([str(v) for v in self.vs])
        
def read_digraph(lines):
    """Reads a digraph from an array of strings in the input format."""

    vtx_count, edge_count = [int(x) for x in lines[0].split()]
    digraph = Digraph(vtx_count)
    for line in lines[1:edge_count+1]:
        tokens = [x for x in line.split()]
        src_id = int(tokens[0])
        tgt_id = int(tokens[1])
        if src_id < 0 or src_id >= vtx_count:
            raise KidneyReadException("Vertex index {} out of range.".format(src_id))
        if tgt_id < 0 or tgt_id >= vtx_count:
            raise KidneyReadException("Vertex index {} out of range.".format(tgt_id))
        if src_id == tgt_id:
            raise KidneyReadException("Self-loop from {0} to {0} not permitted".format(src_id))
        if digraph.edge_exists(digraph.vs[src_id], digraph.vs[tgt_id]):
            raise KidneyReadException("Duplicate edge from {} to {}".format(src_id, tgt_id))
        score = float(tokens[2])
            
        digraph.add_edge(score, digraph.vs[src_id], digraph.vs[tgt_id])

    if lines[edge_count+1].split()[0] != "-1" or len(lines) < edge_count+2:
        raise KidneyReadException("Incorrect edge count")

    return digraph

# read a KPD graph from *edgeweights.csv file
def read_from_kpd(edgeweights_filename):
    col_names = ['match_run','patient_id', 'patient_pair_id', 'donor_id', 'donor_pair_id', 'weight']
    df = pandas.read_csv(edgeweights_filename , names = col_names, skiprows=1)
    nonzero_edges = df.loc[df['weight'] > 0 ]  # last column is edge weights -- only take nonzero edges
    kpd_edges = nonzero_edges.loc[ ~nonzero_edges['donor_pair_id'].isnull()]# remove NDD edges
    vtx_id = set(list(kpd_edges['patient_id'].unique()) + list(kpd_edges['donor_pair_id'].unique())) # get unique vertex ids

    vtx_count = len(vtx_id)
    digraph = Digraph(vtx_count)
    vtx_index = dict(zip( vtx_id, range(len(vtx_id) ))) # vtx_index[id] gives the index in the digraph

    warned = False
    for index, row in kpd_edges.iterrows():
        src_id = vtx_index[row['donor_pair_id']]
        tgt_id = vtx_index[row['patient_id']]
        score = row['weight']
        if src_id < 0 or src_id >= vtx_count:
            raise KidneyReadException("Vertex index {} out of range.".format(src_id))
        if tgt_id < 0 or tgt_id >= vtx_count:
            raise KidneyReadException("Vertex index {} out of range.".format(tgt_id))
        if src_id == tgt_id:
            raise KidneyReadException("Self-loop from {0} to {0} not permitted".format(src_id))
        if digraph.edge_exists(digraph.vs[src_id], digraph.vs[tgt_id]) & ~warned:
            print "# WARNING: Duplicate edge in file: {}".format(edgeweights_filename)
            warned = True
            # raise KidneyReadException("Duplicate edge from {} to {}".format(src_id, tgt_id))
        if score == 0:
            raise KidneyReadException("Zero-weight edge from {} to {}".format(src_id, tgt_id))

        digraph.add_edge(score, digraph.vs[src_id], digraph.vs[tgt_id])

    return digraph,vtx_index

