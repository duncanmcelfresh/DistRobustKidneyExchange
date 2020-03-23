# plot results from DRO
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.font_manager as font_manager

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'stix'


# plt.rc('text', usetex=False)
# plt.rc('font', family='serif')

def lower_pct_trimmed_mean(arr):
    """calculate the the mean of the array, including elements up to the p^th percentile. that is: the mean of the
    lowest (100 * p)% of the array."""
    p = 0.4
    return np.mean(sorted(arr)[:int(round(len(arr) * p, 0))])


# dro results
output_file = '/Users/duncan/research/old_projects/DistRobustKidneyExchange/DistRobustKidneyExchange_output/dro/robust_kex_experiment_20191008_144745.csv'

fig_dir = '/Users/duncan/research/old_projects/DistRobustKidneyExchange/DistRobustKidneyExchange_output/fig'

# results
df = pd.read_csv(output_file, skiprows=1)

# strip whitespace from cols
df.columns = [str.strip(c) for c in df.columns]

# strip whitespace from the 'method' columns
df['method'] = df['method'].str.replace(" ", "")

# for sanity's sake, take only look at one of the noise scale values
# noise_scale = 10.0
# df = df.loc[df['noise_scale'].isin([noise_scale])]

# take only one gamma value saa & dro
plot_gamma = 10.0

# take only one alpha value
plot_alpha = 0.4

# optional: take only one trial
# trial_num = 2
# df = df.loc[df['trial_num'] == trial_num]

# add a column for the parameter value for each method
df['ro_gamma'] = df['method'].apply(lambda x: float(x.split('_')[2]) if x.startswith('ro') else '')
df['saa_gamma'] = df['method'].apply(lambda x: float(x.split('_')[2]) if x.startswith('saa') else '')
df['saa_alpha'] = df['method'].apply(lambda x: float(x.split('_')[4]) if x.startswith('saa') else '')
df['dro_gamma'] = df['method'].apply(lambda x: float(x.split('_')[3]) if x.startswith('dro') else '')
df['dro_alpha'] = df['method'].apply(lambda x: float(x.split('_')[5]) if x.startswith('dro') else '')
df['dro_theta'] = df['method'].apply(lambda x: float(x.split('_')[7]) if x.startswith('dro') else '')

# add a column to identify each method
df['method_base'] = df['method']
df.loc[df['method'].str.contains('ro_gamma'), ['method_base']] = 'ro'
df.loc[df['method'].str.startswith('saa_'), ['method_base']] = 'saa'
df.loc[df['method'].str.startswith('dro_'), ['method_base']] = 'dro'

# remove all dro or saa rows with gamma or alpha not equal to this value
df = df.loc[df['dro_gamma'].isin([plot_gamma, ''])]
df = df.loc[df['saa_gamma'].isin([plot_gamma, ''])]
# df = df.loc[~df['saa_gamma'].isin([0.1, 1.0])]

df = df.loc[df['dro_alpha'].isin([plot_alpha, ''])]
df = df.loc[df['saa_alpha'].isin([plot_alpha, ''])]
# --- aggregate results for each graph-distribution pair, over all realizations ---
# create a uid for identifying each graph-distribution and each method

# create an id for each graph-distribution pair
id_columns = ['graph_name', 'trial_num', 'alpha', 'cycle_cap', 'chain_cap', 'noise_scale']
df['uid'] = df[id_columns].apply(tuple, axis=1)

# for each method and each uid, get the max, min, mean, and mean-lowest-p% of the realized edge weights

df_grouped = df.groupby(
    by=['uid', 'method_base', 'ro_gamma', 'saa_gamma', 'saa_alpha', 'dro_gamma', 'dro_alpha', 'dro_theta']).agg(
    {
        'realized_score': ['min', 'max', 'mean', lower_pct_trimmed_mean]
    }).reset_index()

new_cols = list(df_grouped.columns.droplevel(0))
new_cols[:8] = list(df_grouped.columns.droplevel(1)[:8])
df_grouped.columns = new_cols

df_baseline = df_grouped.loc[
    df_grouped['method_base'] == 'nonrobust_samplemean', ['uid', 'min', 'max', 'mean', 'lower_pct_trimmed_mean']]
df_baseline.rename(columns={'min': 'baseline_min',
                            'max': 'baseline_max',
                            'mean': 'baseline_mean',
                            'lower_pct_trimmed_mean': 'baseline_lower_pct_trimmed_mean'}, inplace=True)
df_clean = pd.merge(df_grouped, df_baseline, on='uid')

# # manual cleanup (optional)
# del df, df_grouped, df_baseline

# add the noise scale back in, for reference
df_clean['noise_scale'] = df_clean['uid'].apply(lambda x: x[5])

df_clean['min_pct_diff'] = (df_clean['min'] - df_clean['baseline_min']) / df_clean['baseline_min']
df_clean['max__pct_diff'] = (df_clean['max'] - df_clean['baseline_max']) / df_clean['baseline_max']
df_clean['mean_pct_diff'] = (df_clean['mean'] - df_clean['baseline_mean']) / df_clean['baseline_mean']
df_clean['lowerpct_pct_diff'] = (df_clean['lower_pct_trimmed_mean'] - df_clean['baseline_lower_pct_trimmed_mean']) / \
                                df_clean['baseline_lower_pct_trimmed_mean']

plot_field = 'lowerpct_pct_diff'
# plot_field = 'mean_pct_diff'


font = font_manager.FontProperties(family='Courier New')

remove_zeros = True

# create 3 subplots
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(5.5, 3))

# only use one gamma value
df_clean['saa_gamma'].unique()
df_clean['dro_gamma'].unique()

# --- gamma - RO ---
if remove_zeros:
    df_ro = df_clean[(df_clean['method_base'] == 'ro') & (df_clean[plot_field] != 0.0)]
else:
    df_ro = df_clean[df_clean['method_base'] == 'ro']

ax_ro = sns.boxplot(x='ro_gamma', y=plot_field, data=df_ro, ax=ax1)  # , label="_nolegend_")
# ax_ro = sns.catplot(kind='box', x='ro_gamma', y=plot_field, row='noise_scale', data=df_ro, ax=ax1)  # , label="_nolegend_")

ax1.set_title("RO", fontname='Courier New')
ax1.set_ylabel("FRAC-NR")
ax1.set_xlabel(r"$\Gamma$")
baseline = ax1.plot([-100, 100], [0.0, 0.0], 'r:', linewidth=2, label="NR (baseline)")

ax1.legend()  # ([baseline], ["NR (baseline)"])  # ,bbox_to_anchor=(1.05, 1), loc=2)

# --- gamma - SSA ---
if remove_zeros:
    df_saa = df_clean[
        (df_clean['method_base'] == 'saa') & (df_clean[plot_field] != 0.0)]
    # (df_clean['method_base'] == 'saa') & (df_clean[plot_field] != 0.0) & (df_clean['saa_gamma'] == plot_gamma)]
else:
    df_saa = df_clean[(df_clean['method_base'] == 'saa')]
    # df_saa = df_clean[(df_clean['method_base'] == 'saa') & (df_clean['saa_gamma'] == plot_gamma)]

ax_saa = sns.boxplot(x='saa_gamma', y=plot_field, data=df_saa, ax=ax2)

# plot a line of the means
# gamma_vals = sorted(list(df_saa['saa_gamma'].unique()))
# mean_vals = []
# for gamma in gamma_vals:
#     mean_vals.append(df_saa[df_saa['saa_gamma'] == gamma][plot_field].mean())

# ax2.plot(gamma_vals, mean_vals, 'g', linewidth=2)
ax2.plot([-100, 100], [0.0, 0.0], 'r:', linewidth=2)
ax2.set_title("SAA", fontname='Courier New')
ax2.set_ylabel("")
ax2.set_xlabel("$\\gamma$")

# --- theta - dro ---

# plot only these theta values
plot_theta = [0.1, 1.0, 2, 5, 10.0, 20.0]
# plot_theta = df_clean['dro_theta'].unique()

if remove_zeros:
    df_dro = df_clean[
        (df_clean['method_base'] == 'dro') & (df_clean[plot_field] != 0.0) & df_clean['dro_theta'].isin(plot_theta)]
    # (df_clean['method_base'] == 'dro') & (df_clean[plot_field] != 0.0)]
    # (df_clean['method_base'] == 'dro') & (df_clean[plot_field] != 0.0) & (df_clean['dro_gamma'] == plot_gamma)]
else:
    df_dro = df_clean[(df_clean['method_base'] == 'dro') & df_clean['dro_theta'].isin(plot_theta)]
    # df_dro = df_clean[(df_clean['method_base'] == 'dro')]
    # df_dro = df_clean[(df_clean['method_base'] == 'dro') & (df_clean['dro_gamma'] == plot_gamma)]

ax_dro = sns.boxplot(x='dro_theta', y=plot_field, data=df_dro, ax=ax3)

# plot a line of the means
gamma_vals = sorted(list(df_dro['dro_theta'].unique()))
mean_vals = []
for gamma in gamma_vals:
    mean_vals.append(df_dro[df_dro['dro_theta'] == gamma][plot_field].mean())

# ax3.plot(gamma_vals, mean_vals, 'g', linewidth=2)
ax3.plot([-100, 100], [0.0, 0.0], 'r:', linewidth=2)
ax3.set_title("DRO", fontname='Courier New')
ax3.set_ylabel("")
ax3.set_xlabel("$\\theta$ ($\\gamma=%d$)" % plot_gamma)
ax3.xaxis.set_tick_params(rotation=30)

# plt.suptitle("$\\alpha=%3.1f$" %  plot_alpha)
plt.tight_layout()

plt.savefig(os.path.join(fig_dir, "dro_results_new.pdf"))
