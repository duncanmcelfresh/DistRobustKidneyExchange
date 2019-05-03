# These functions calculate the following indices:
#
# - Disparate Impact Discrimination Index for categorical outcomes (DIDI_categorical)
# - Disparate Impact Discrimination Index for continuous outcomes (DIDI_continuous)
# - Disparate Treatment Discrimination Index for categorical outcomes (DTDI_categorical)
# - Disparate Treatment Discrimination Index for continuous outcomes (DTDI_continuous)
#
# Let N be the number of individuals/data points.
#
# --- DIDI Functions ---
# Each DIDI function takes a list of N categorical outcomes (labels), and a list of protected features (x_protect).
# x_protect can contain a single feature (i.e. x_protect = ['A','B','B',...]) or multiple features in a tuple
# (i.e. x_protect = [('A',1),('A',6),('B',3),...]).
#
# --- DTDI Functions ---
# Each DTDI function takes a list of N values rather than categorical outcomes. These functions also take a NxN array
# of *distances* between each individual in the input data. Be careful to specify this appropriately.
#
# Duncan McElfresh
# July 2018

import numpy as np


def DIDI_categorical(labels,x_protect):
    '''
    The inputs are lists/arrays with one entry for each data point/individual

    inputs:
    - labels : a list of categorical labels for each data point
    - x_protect : a list of the protected feature values

    output:
    - disparate impact discrimination index
    '''

    if len(labels) != len(x_protect):
        raise Warning("labels and x_protect must have the same length")

    # this is written for lists
    labels = list(labels)
    x_protect = list(x_protect)

    didi = 0

    n = float(len(labels))

    unique_labels = set(labels)
    unique_x_protect = set(x_protect)

    # for each individual label, calculate the different from average for each protected class
    for l in unique_labels:

        # fraction of overall data points with label l
        frac_label = float(labels.count(l))/n

        # didi += | <fraction of data points with protected feature x and label l> - < fraction of data with label l> |
        for x in unique_x_protect:
            didi += np.abs( float(zip(labels,x_protect).count((l,x)))/float(x_protect.count(x)) - frac_label )

    return didi


def DIDI_continuous(values, x_protect):
    '''
    The inputs are lists/arrays with one entry for each data point/individual

    inputs:
    - values : a list of numerical values for each data point (treated as floats)
    - x_protect : a list of the protected feature values

    output:
    - disparate impact discrimination index
    '''

    if len(values) != len(x_protect):
        raise Warning("labels and x_protect must have the same length")

    # this is written for lists
    values = list(values)
    x_protect = list(x_protect)

    didi = 0

    n = float(len(values))

    unique_x_protect = set(x_protect)

    # mean value over all data points
    mean_value = np.mean(values)

    # for each protected feature, calculate the different from average

    # didi += | <average value of data points with protected feature x> - <average value over all data points> |
    for x in unique_x_protect:
        didi += np.abs( np.mean([values[i] for i in range(n) if x_protect[i] == x]) - mean_value )

    return didi


# disparate treatment

def DTDI_categorical(labels, x_protect, distances):
    '''
    The inputs are lists/arrays with one entry for each data point/individual

    inputs:
    - labels : a list of categorical labels for each data point
    - x_protect : a list of the protected feature values. each entry can be a single value, or a tuple (multiple values)
    - distances : a list of distances between data points; these distances should be based on the NONprotected features

    output:
    - disparate treatment discrimination index
    '''

    n = len(labels)

    # this is written for numpy arrays
    labels = np.array(labels)
    x_protect = np.array(x_protect)
    distances = np.array(distances,dtype=float)

    if len(labels) != len(x_protect):
        raise Warning("labels and x_protect must have the same length")

    if distances.shape != (n,n):
        raise Warning("distances must be nxn square")

    if not (distances == distances.T).all():
        raise Warning("distances must be symmetric")

    dtdi = 0

    unique_labels = set(labels)
    unique_x_protect = set(x_protect)

    # for each point, for each individual label, calculate the difference between the sample average from the
    # overall average
    for i in range(n):

        d_i_sum = distances[i,:].sum()

        for l in unique_labels:

            # fraction of overall data points with label l -- weighted by distances
            avg_label = distances[i,labels == l].sum()/d_i_sum

            # dtdi += | <sample average for class x> - < overall average> |
            for x in unique_x_protect:

                dtdi += np.abs( distances[i, (labels == l) & (x_protect == x)].sum()/ distances[i,x_protect == x].sum()
                                - avg_label )

    return dtdi


def DTDI_continuous(values, x_protect, distances):
    '''
    The inputs are lists/arrays with one entry for each data point/individual

    inputs:
    - values : a list of numerical values for each data point (treated as floats)
    - x_protect : a list of the protected feature values
    - distances : a list of distances between data points; these distances should be based on the NONprotected features

    output:
    - disparate treatment discrimination index
    '''

    n = len(values)

    # this is written for numpy arrays
    values = np.array(values, dtype=float)
    x_protect = np.array(x_protect)
    distances = np.array(distances, dtype=float)

    if len(values) != len(x_protect):
        raise Warning("values and x_protect must have the same length")

    if distances.shape != (n,n):
        raise Warning("distances must be nxn square")

    if not (distances == distances.T).all():
        raise Warning("distances must be symmetric")

    dtdi = 0

    unique_x_protect = set(x_protect)

    # for each point, for each protected feature, calculate the difference between the distance-weighted
    # sample average from the overall distance-weighted average
    for i in range(n):

        d_i_sum = distances[i,:].sum()

        # dtdi += | <average value of data points with protected feature x, weighted by distance from i > -
        # <average value over all data points, weighted by distance from i> |
        for x in unique_x_protect:
            dtdi += np.abs(
                (distances[i, x_protect == x]*values[x_protect == x]).sum() / distances[i, x_protect == x].sum() - \
                            (distances[i,:]*values).sum() / d_i_sum )

    return dtdi

