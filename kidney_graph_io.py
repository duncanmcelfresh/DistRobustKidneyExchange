import glob
import os
import pandas as pd

from kidney_digraph import Digraph, KidneyReadException
from kidney_ndds import Ndd, NddEdge


def read_UNOS_graph(directory):
    # read a UNOS-format exchange, and return a list of kidney_ndd.Ndd objects and a kidney_digraph.Digraph object.
    #
    # each UNOS-format exchange is contained in a subdirectory with the naming format 'KPD_CSV_IO_######'. Each exchange
    # subdirectory must contain a file of the format ########_edgeweights.csv

    # look for edge & recipient files
    edge_files = glob.glob(directory + os.sep + '*edgeweights.csv')

    name = os.path.basename(directory)

    # there should only be one edgeweights file
    assert len(edge_files) == 1

    edge_filename = edge_files[0]

    col_names = ['match_run','patient_id', 'patient_pair_id', 'donor_id', 'donor_pair_id', 'weight']
    df = pd.read_csv(edge_filename, names=col_names, skiprows=1)

    # last column is edge weights -- only take nonzero edges
    nonzero_edges = df.loc[df['weight'] > 0]

    # remove NDD edges
    kpd_edges = nonzero_edges.loc[~nonzero_edges['donor_pair_id'].isnull()]

    # get unique vertex ids
    vtx_id = set(list(kpd_edges['patient_id'].unique()) + list(kpd_edges['donor_pair_id'].unique()))

    vtx_count = len(vtx_id)
    digraph = Digraph(vtx_count)

    # vtx_index[id] gives the index in the digraph
    vtx_index = dict(zip( vtx_id, range(len(vtx_id))))

    warned = False
    for index, row in kpd_edges.iterrows():
        src_id = vtx_index[row['donor_pair_id']]
        tgt_id = vtx_index[row['patient_id']]
        weight = row['weight']
        if src_id < 0 or src_id >= vtx_count:
            raise KidneyReadException("Vertex index {} out of range.".format(src_id))
        if tgt_id < 0 or tgt_id >= vtx_count:
            raise KidneyReadException("Vertex index {} out of range.".format(tgt_id))
        if src_id == tgt_id:
            raise KidneyReadException("Self-loop from {0} to {0} not permitted".format(src_id))
        if digraph.edge_exists(digraph.vs[src_id], digraph.vs[tgt_id]) & ~warned:
            print "# WARNING: Duplicate edge in file: {}".format(edge_filename)
            warned = True
        if weight == 0:
            raise KidneyReadException("Zero-weight edge from {} to {}".format(src_id, tgt_id))

        digraph.add_edge(weight, digraph.vs[src_id], digraph.vs[tgt_id])

        # now read NDDs - take only NDD edges
        ndd_edges = nonzero_edges.loc[nonzero_edges['donor_pair_id'].isnull()]
        ndd_id = set(list(ndd_edges['donor_id'].unique()))

        ndd_count = len(ndd_id)

        if ndd_count > 0:
            ndd_list = [Ndd(id=i) for i in range(ndd_count)]
            ndd_index = dict(zip(ndd_id, range(len(ndd_id))))  # ndd_index[id] gives the index in the digraph

            # Keep track of which edges have been created already, to detect duplicates
            edge_exists = [[False for v in digraph.vs] for ndd in ndd_list]

            for index, row in ndd_edges.iterrows():
                src_id = ndd_index[row['donor_id']]
                tgt_id = vtx_index[row['patient_pair_id']]
                weight = row['weight']
                if src_id < 0 or src_id >= ndd_count:
                    raise KidneyReadException("NDD index {} out of range.".format(src_id))
                if tgt_id < 0 or tgt_id >= digraph.n:
                    raise KidneyReadException("Vertex index {} out of range.".format(tgt_id))

                ndd_list[src_id].add_edge(NddEdge(digraph.vs[tgt_id], weight, src_id=ndd_list[src_id].id))
                edge_exists[src_id][tgt_id] = True
        else:
            ndd_list = []
            ndd_index = []

    return digraph, ndd_list, name


# TODO: TEST THIS GENERATOR. CURRENTLY UNTESTED.
def get_UNOS_graphs(directory):
    # create a generator that produces kidney exchange graphs, given a directory containing UNOS-format exchange graph
    # files.

    # get all subdirectories that look like UNOS files (KPD_CSV_IO_...)
    kpd_dirs = glob.glob(os.path.join(directory, 'KPD_CSV_IO*/'))

    for sub_directory in sorted(kpd_dirs):

        # read the graph
        digraph, vtx_index, ndd_list, ndd_index = read_UNOS_graph(sub_directory)

        yield digraph, ndd_list



def read_CMU_format(details_filename, maxcard_filename,
                     frac_edges=None,
                     seed=101):
    # read a "CMU" format exchange graph, using the details and maxcard files
    #
    # optional : frac_edges in (0, 1) adds only a fraction of the edges to the Digraph.

    name = os.path.basename(maxcard_filename)

    # read details.input file
    col_names = ['id', 'abo_patient', 'abo_fonor', 'wife_patient', 'pra', 'in_deg', 'out_deg', 'is_ndd', 'is_marginalized']
    df_details = pd.read_csv(details_filename, names=col_names, skiprows=1, delim_whitespace=True)

    pair_details = df_details.loc[df_details['is_ndd'] == 0]
    pair_id = list(pair_details['id'].unique())

    # vtx_index[id] gives the index in the digraph
    vtx_index = dict( zip( pair_id, range(len(pair_id))))

    vtx_count = len(vtx_index)
    digraph = Digraph(vtx_count)

    # label sensitized pairs
    for index, row in pair_details.iterrows():
        if row['is_marginalized']:
            digraph.vs[vtx_index[row['id']]].sensitized = True

    # read maxcard.inuput file (edges)
    col_names = ['src_id','tgt_id', 'weight', 'c4', 'c5']
    df_edges = pd.read_csv(maxcard_filename, names=col_names, skiprows=1, delim_whitespace=True)

    # drop the last column
    df_edges.drop(df_edges.index[-1])

    # take only nonzero edges
    nonzero_edges = df_edges.loc[df_edges['weight'] > 0]

    # optional: sample from the edges
    if frac_edges is not None:
        assert (frac_edges < 1.0) and (frac_edges > 0.0)
        nonzero_edges = nonzero_edges.sample(frac=frac_edges, random_state=seed)

    # ind ndds if they exist
    ndd_details = df_details.loc[df_details['is_ndd'] == 1]
    ndd_count = len(ndd_details)

    if ndd_count > 0:
        ndd_list = [Ndd(id=i) for i in range(ndd_count)]
        ndd_id = list(ndd_details['id'].unique())

        # ndd_index[id] gives the index in the ndd list
        ndd_index = dict( zip( ndd_id, range(len(ndd_id))))
    else:
        ndd_list = []
        ndd_index = []

    use_ndds = ndd_count > 0

    # add edges to pairs and ndds
    for index, row in nonzero_edges.iterrows():
        src = row['src_id']
        tgt_id = vtx_index[ row['tgt_id'] ]
        weight = row['weight']
        if use_ndds and ndd_index.has_key(src): # this is an ndd edge
            src_id = ndd_index[src]
            ndd_list[src_id].add_edge(NddEdge(digraph.vs[tgt_id], weight, src_id=ndd_list[src_id].id))
        else: # this edge is a pair edge
            src_id = vtx_index[src]
            digraph.add_edge(weight, digraph.vs[src_id], digraph.vs[tgt_id])

    return digraph, ndd_list, name


# TODO: TEST THIS GENERATOR. CURRENTLY UNTESTED
def get_cmu_graphs(directory):
    # create a generator that produces kidney exchange graphs, given a directory containing "CMU" format exchange
    # graph files.

    # find all *maxcard.input files in the directory -- each corresponds to an exchange graph
    maxcard_files = glob.glob(os.path.join(directory, '*maxcard.input'))

    for maxcard_file in maxcard_files:

        file_base = '_'.join(maxcard_file.split('_')[:-1])

        # find the details file; there can be only one
        details_files = glob.glob(os.path.join(directory, file_base + '_*details.input'))
        assert len(details_files) == 1
        details_file = details_files[0]

        digraph, ndd_list, name = read_CMU_format(details_file, maxcard_file)

        yield digraph, ndd_list, name


