# plot results

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def lower_20pct_trimmed_mean(arr):
    """calculate the the mean of the array, including elements up to the p^th percentile. that is: the mean of the
    lowest (100 * p)% of the array."""
    p = 0.2
    return np.mean(sorted(arr)[:int(round(len(arr) * p, 0))])


# test file (3 realizations)
# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/debug/robust_kex_experiment_20190821_171748.csv'

# results with 100 samples - alpha = 1.0
# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/robust_kex_experiment_20190821_204305.csv'

# new results using unos edge weights
# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/robust_unos.csv'

# new results using lkdpi edge weights
# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/robust_lkdpi.csv'

# new results using lkdpi edge weights & 2k measurements per edge
# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/robust_lkdpi_2k_measurements.csv'

# # # new results with lkdpi and 128-node graphs, N=100 measurements per edge (PARTIAL
# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/debug/robust_kex_experiment_20190904_130132.csv'

# new results with lkdpi and 128-node graphs, N=200 measurements per edge (PARTIAL)
output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/lkdpi_128_partial.csv'


# results
df = pd.read_csv(output_file, skiprows=1)

# strip whitespace from cols
df.columns = [str.strip(c) for c in df.columns]

# strip whitespace from the 'method' columns
df['method'] = df['method'].str.replace(" ", "")

# add a column for the parameter value for each method
df['ro_gamma'] = df['method'].apply(lambda x: float(x.split('_')[2]) if x.startswith('ro') else '')
df['ssa_gamma'] = df['method'].apply(lambda x: float(x.split('_')[2]) if x.startswith('ssa') else '')
df['ssa_alpha'] = df['method'].apply(lambda x: float(x.split('_')[4]) if x.startswith('ssa') else '')

# add a column to identify each method
df['method_base'] = df['method']
df.loc[df['method'].str.contains('ro_gamma'), ['method_base']] = 'ro'
df.loc[df['method'].str.contains('ssa_'), ['method_base']] = 'ssa'

# --- aggregate results for each graph-distribution pair, over all realizations ---
# create a uid for identifying each graph-distribution and each method

# create an id for each graph-distribution pair
id_columns = ['graph_name', 'trial_num', 'alpha', 'cycle_cap', 'chain_cap']
df['uid'] = df[id_columns].apply(tuple, axis=1)

# for each method and each uid, get the max, min, mean, and mean-lowest-p% of the realized edge weights

df_grouped = df.groupby(by=['uid', 'method_base', 'ro_gamma', 'ssa_gamma', 'ssa_alpha']).agg(
    {
        'realized_score': ['min', 'max', 'mean', lower_20pct_trimmed_mean]
    }).reset_index()

new_cols = list(df_grouped.columns.droplevel(0))
new_cols[:5] = list(df_grouped.columns.droplevel(1)[:5])
df_grouped.columns = new_cols

df_baseline = df_grouped.loc[
    df_grouped['method_base'] == 'nonrobust_samplemean', ['uid', 'min', 'max', 'mean', 'lower_20pct_trimmed_mean']]
df_baseline.rename(columns={'min': 'baseline_min',
                            'max': 'baseline_max',
                            'mean': 'baseline_mean',
                            'lower_20pct_trimmed_mean': 'baseline_lower_20pct_trimmed_mean'}, inplace=True)
df_clean = pd.merge(df_grouped, df_baseline, on='uid')

df_clean['min_pct_diff'] = (df_clean['min'] - df_clean['baseline_min']) / df_clean['baseline_min']
df_clean['max__pct_diff'] = (df_clean['max'] - df_clean['baseline_max']) / df_clean['baseline_max']
df_clean['mean_pct_diff'] = (df_clean['mean'] - df_clean['baseline_mean']) / df_clean['baseline_mean']
df_clean['20pct_pct_diff'] = (df_clean['lower_20pct_trimmed_mean'] - df_clean['baseline_lower_20pct_trimmed_mean']) / \
                             df_clean['baseline_lower_20pct_trimmed_mean']

plot_field = '20pct_pct_diff'
# plot_field = 'mean_pct_diff'

remove_zeros = False

# create three subplots
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

# --- gamma - RO ---
if remove_zeros:
    df_ro = df_clean[(df_clean['method_base'] == 'ro') & (df_clean[plot_field] != 0.0)]
else:
    df_ro = df_clean[df_clean['method_base'] == 'ro']

ax_ro = sns.boxplot(x='ro_gamma', y=plot_field, data=df_ro, ax=ax1)

# --- nonrobust ---
df_nonrobust = df_clean.loc[df_clean['method_base'].str.contains('nonrobust')]
df_nonrobust['mean_type'] = 'sample mean'
df_nonrobust.loc[df_nonrobust['method_base'].str.contains('true'), ['mean_type']] = 'true mean'

ax_nonrobust = sns.boxplot(x='mean_type', y=plot_field, data=df_nonrobust, ax=ax2)

# --- gamma - SSA ---
if remove_zeros:
    df_ssa = df_clean[(df_clean['method_base'] == 'ssa') & (df_clean[plot_field] != 0.0)]
else:
    df_ssa = df_clean[(df_clean['method_base'] == 'ssa')]

ax_ssa = sns.boxplot(x='ssa_gamma', y=plot_field, data=df_ssa, ax=ax3)

# --- make plots for individual kex graphs ---

uid_list = df_clean['uid'].unique()

uid = uid_list[19]
print('uid: %s' % str(uid))

# create three subplots
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

# --- gamma - RO ---
df_ro = df[(df['method_base'] == 'ro') & (df['uid'] == uid)]

ax_ro = sns.boxplot(x='ro_gamma', y='realized_score', data=df_ro, ax=ax1)

# --- nonrobust ---
df_nonrobust = df.loc[df['method_base'].str.contains('nonrobust') & (df['uid'] == uid)]
df_nonrobust['mean_type'] = 'sample mean'
df_nonrobust.loc[df_nonrobust['method_base'].str.contains('true'), ['mean_type']] = 'true mean'

ax_nonrobust = sns.boxplot(x='mean_type', y='realized_score', data=df_nonrobust, ax=ax2)

# --- gamma - SSA ---
df_ssa = df[(df['method_base'] == 'ssa') & (df['uid'] == uid)]

ax_ssa = sns.boxplot(x='ssa_gamma', y='realized_score', data=df_ssa, ax=ax3)

plt.title(uid[0])
