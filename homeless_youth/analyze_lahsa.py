# analyze LAHSA data

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

family_file = '/Users/duncan/research/homeless_youth/LAHSA_data/Dataset/Family.csv'
singles_file = '/Users/duncan/research/homeless_youth/LAHSA_data/Dataset/Singles.csv'

family_df = pd.read_csv(family_file, delimiter=',')
singles_df = pd.read_csv(singles_file, delimiter=',')

# check that all PersonalIDs are unique in both files
assert len(family_df['PersonalID'].unique()) == len(family_df)
assert len(singles_df['PersonalID'].unique()) == len(singles_df)

# check for overlap in the PersonalIDs (should be zero -- they should not overlap)
print "Number of PersonalIDs that appear in both Family_Data.csv and Singles_data.csv: %d" % \
      len(set(family_df['PersonalID']).intersection(singles_df['PersonalID']))

# "First_Exit_Date" is the date of the first exit from homelessness
# "Exit_type" is the exit type for the first exit
# "Return_Date" is the date when the client LEFT the first housing exit (to anywhere)

# df = singles_df
# name = 'singles'

df = family_df
name = 'family'

# add race field
race_list = ['AmIndAKNative', 'Asian', 'BlackAfAmerican', 'NativeHIOtherPacific', 'White', 'RaceNone']

# ethnicity == 1 is hispanic/latino

df['race'] = ''
for r in race_list:
    df.loc[df[r] == 1, 'race'] = r

# now set hispanic/latino
df.loc[df['Ethnicity']==1,'race'] = "HispanicLatino"

# how many do not have race labels ("" or "RaceNone")
print "the number of rows no race label is %d (%3.1f pct.) " % (len(df[df['race'].isin(['','RaceNone'])]),100*float(len(df[df['race'].isin(['','RaceNone'])]))/float(len(df)))

# convert everything to datetimes
df['First_Exit_Date'] = pd.to_datetime(df['First_Exit_Date'])
df['Return_Date'] = pd.to_datetime(df['Return_Date'])

# set a final date:
final_date = pd.to_datetime('2018-07-01')
cutoff_date = pd.to_datetime('2017-07-01') # one year prior to the final date

# add a "Still_Housed" column -- default is false
df['Still_Housed'] = False

# for those with an exit date, but no return date, set the exit date to the final date
# for these rows, set Still_Housed = True
df.loc[(df['Return_Date'].isna() & df['First_Exit_Date'].notna()),'Still_Housed'] = True
df.loc[(df['Return_Date'].isna() & df['First_Exit_Date'].notna()),'Return_Date'] = final_date

# calculate the exit duration
df['First_Exit_Duration'] = (df['Return_Date'] - df['First_Exit_Date']).dt.days

# for those with no exit date, set duration to 0 (they never exited)
# zero_time = pd.to_datetime(0) - pd.to_datetime(0)
df.loc[df['First_Exit_Date'].isna(),'First_Exit_Duration'] = 0

# are there any rows with a return date but no exit date? should be none
assert len(df.loc[(df['First_Exit_Date'].isna() & df['Return_Date'].notna())]) == 0

# how many rows never had an exit
print "the number of rows with NO first exit is %d (%3.1f pct.) " % ((df['First_Exit_Duration'] == 0).sum(),100*float((df['First_Exit_Duration'] == 0).sum())/float(len(df)))

# plot a hist of those that did exit
df[df['First_Exit_Duration'] > 0]['First_Exit_Duration'].hist(bins=50)

# define a stable exit -- housed at least 365 days
# ONLY LOOK AT THOSE WITH ASSESSMENT DATES ON OR AFTER THE CUTOFF DATE
df['STABLE_EXIT'] = np.nan # those with assessments on or after this get nan

assess_date_col = 'Assessment Date_SINGLES'

df[assess_date_col] = pd.to_datetime(df[assess_date_col])

stable_limit = 365 # number of days for a stable exit
df.loc[df[assess_date_col]<= cutoff_date, 'STABLE_EXIT'] = 0 # this is default
df.loc[(df[assess_date_col]<= cutoff_date) & (df['First_Exit_Duration'] >= stable_limit), 'STABLE_EXIT'] = 1

# how many stable exits were there?
print "the number of rows with a stable exit is  %d (%3.1f pct.) " % ((df['STABLE_EXIT'] == 1).sum(),100*float((df['STABLE_EXIT'] == 1).sum())/float(len(df['STABLE_EXIT'].notna())))

# create a df with only the stable exit rows
df_st_ex = df.loc[df['STABLE_EXIT'].notna()].copy()

# --- look at exit types ---

# overall exit types
df['Exit_type'].value_counts(normalize=True,dropna=False)

# by race
df.groupby('race')['Exit_type'].value_counts(normalize=True,dropna=False)

# now make a "received_resource" column (RRH or PSH)
df['received_resource'] = df['Exit_type'].isin(['RRH','PSH'])
resource_by_race = df.groupby('race')['received_resource'].value_counts(normalize=True,dropna=False)
resource_by_race = pd.DataFrame(df.groupby('race')['received_resource'].value_counts(normalize=True,dropna=False))
resource_by_race.columns = ['pct']
resource_by_race = resource_by_race.reset_index()

didi_resource_race = DIDI_categorical(df['race'],df['received_resource'])
# it's 0.41853160743517714

# calculate the disparity index

# --- calculate disparities ---

# disparate impact, for stable exits:

# by race
df_test = df_st_ex[~df_st_ex['race'].isin(['','RaceNone'])].copy()

didi_race = DIDI_categorical(df_test['race'],df_test['STABLE_EXIT'])
# ----------------------------------------------------------------------------------------------------------------------
# Functions to calculate disparity indices

# for categorical variables (e.g. successful exit, or type of exit)

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

#TODO: not done...

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



results_df.columns = [c.replace(" ", "") for c in results_df.columns]

results_long = pd.melt(results_df,id_vars=['q#','trial'],value_vars=['AC_robust','us_robust','AC_true','us_true'])

# results_df.hist(column=['AC_robust'],by=['q#'])

results_long_rob = results_long[results_long['variable'].isin(['AC_robust','us_robust'])]
results_long_true = results_long[results_long['variable'].isin(['AC_true','us_true'])]

sns.catplot(x='q#', y='value',data=results_long_rob , hue='variable', kind="box", whis='range')
plt.title('Robust recommended utility: % difference from optimal\n (10 indep. trials. whiskers are min/max)')
plt.ylabel('%-diff. from opt')
plt.tight_layout()


sns.catplot(x='q#', y='value',data=results_long_true , hue='variable', kind="box", whis='range')
plt.title('True recommended utility: % difference from optimal\n (10 indep. trials. whiskers are min/max)')
plt.ylabel('%-diff. from opt')
plt.tight_layout()


