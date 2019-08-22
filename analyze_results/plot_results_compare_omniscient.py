# plot results

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# test file (3 realizations)
output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/debug/robust_kex_experiment_20190821_171748.csv'

# results with 100 samples - alpha = 1.0
output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/robust_kex_experiment_20190821_204305.csv'
df = pd.read_csv(output_file, skiprows=1)

# strip whitespace from cols
df.columns = [str.strip(c) for c in df.columns]

# strip whitespace from the 'method' columns
df['method'] = df['method'].str.replace(" ","")

# add a column for the parameter value for each method
df['ro_gamma'] = df['method'].apply(lambda x: float(x.split('_')[2]) if x.startswith('ro') else '')
df['ssa_gamma'] = df['method'].apply(lambda x: float(x.split('_')[2]) if x.startswith('ssa') else '')
df['ssa_alpha'] = df['method'].apply(lambda x: float(x.split('_')[4]) if x.startswith('ssa') else '')

# add a column to identify each method
df['method_base'] = df['method']
df.loc[df['method'].str.contains('ro_gamma'), ['method_base']] = 'ro'
df.loc[df['method'].str.contains('ssa_'), ['method_base']] = 'ssa'

# create a col for the omniscient edge weight for this realization
id_columns = ['graph_name', 'trial_num', 'alpha', 'realization_num', 'cycle_cap', 'chain_cap']
df['uid'] = df[id_columns].apply(tuple, axis=1)

# get the best possible , and add them as columns to the df
df_best = df.loc[df['method'] == 'omniscient', ['uid', 'realized_score']]
df_best.rename(columns={'realized_score': 'best_realized_score'}, inplace=True)
df_clean = pd.merge(df, df_best, on='uid')

# calculate the absolute optimality gap for all methods
df_clean['abs_opt_gap'] = df_clean['realized_score'] - df_clean['best_realized_score']

# calculate the relative opt gap
df_clean['relative_opt_gap'] = df_clean['abs_opt_gap'] / df_clean['best_realized_score']

# make sure that omniscient always has a zero optimality gap
assert df_clean.loc[df_clean['method'] == 'omniscient', ['abs_opt_gap']].max()[0] == 0.0
assert df_clean.loc[df_clean['method'] == 'omniscient', ['abs_opt_gap']].min()[0] == 0.0

# create three subplots
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

# --- gamma - RO ---
df_ro = df_clean[df_clean['method_base'] == 'ro']

ax_ro = sns.boxplot(x='ro_gamma', y='relative_opt_gap', data=df_ro, ax=ax1)

# --- nonrobust ---
df_nonrobust = df_clean.loc[df_clean['method_base'].str.contains('nonrobust')]
df_nonrobust['mean_type'] = 'sample mean'
df_nonrobust.loc[df_nonrobust['method_base'].str.contains('true'), ['mean_type']] = 'true mean'

ax_nonrobust = sns.boxplot(x='mean_type', y='relative_opt_gap', data=df_nonrobust, ax=ax2)

# --- gamma - SSA ---
df_ssa = df_clean[(df_clean['method_base'] == 'ssa') & (df_clean['ssa_gamma'] == 5)]

ax_ssa = sns.boxplot(x='ssa_alpha', y='relative_opt_gap', data=df_ssa, ax=ax3)
