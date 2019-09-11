# plot results

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.font_manager as font_manager

plt.rcParams["font.family"] = "Times New Roman"

def lower_20pct_trimmed_mean(arr):
    """calculate the the mean of the array, including elements up to the p^th percentile. that is: the mean of the
    lowest (100 * p)% of the array."""
    p = 0.5
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
# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/lkdpi_128_partial.csv'

# new results with lkdpi and 64-node graphs, N=100 measurements per edge, and LKDPI is a property of donors (PARTIAL)
# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/debug/robust_kex_experiment_20190904_152427.csv'

# new results with LKDPI as a property of donors and 64-node graphs and N=1000 measurements
# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/robust_kex_experiment_20190905_083550.csv'

# new results with UNOS as a property of donors and 64-node graphs and N=1000 measurements
# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/robust_kex_experiment_20190905_103333.csv'

# # new results with LKDPI as a property of donors and 64-node graphs and N=5000 measurements (PARTIAL)
# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/robust_kex_experiment_20190905_130738.csv'

# yet another LKDPI test... now with alpha = 0.5...
# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/robust_kex_experiment_20190905_170049.csv'

# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/debug/robust_kex_experiment_20190905_172132.csv'

# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/debug/robust_kex_experiment_20190905_173309.csv'

# good!!!!
output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/gold/robust_kex_experiment_20190905_175226.csv'

# this is the unos results
output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/gold/robust_kex_experiment_20190905_195359.csv'


# this one is bad...
# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/debug/robust_kex_experiment_20190905_180715.csv'


# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/debug/robust_kex_experiment_20190905_183023.csv'


# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/debug/robust_kex_experiment_20190905_184917.csv'

# THIS IS THE DRO RESULTS
# output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/debug/robust_kex_experiment_20190905_202420.csv'

# better DRO results
output_file = '/Users/duncan/research/DistRobustKidneyExchange_output/robust_kex_experiment_20190907_121809.csv'



# results
df = pd.read_csv(output_file, skiprows=1)

# strip whitespace from cols
df.columns = [str.strip(c) for c in df.columns]

# strip whitespace from the 'method' columns
df['method'] = df['method'].str.replace(" ", "")

# add a column for the parameter value for each method
df['ro_gamma'] = df['method'].apply(lambda x: float(x.split('_')[2]) if x.startswith('ro') else '')
df['saa_gamma'] = df['method'].apply(lambda x: float(x.split('_')[2]) if x.startswith('saa') else '')
df['saa_alpha'] = df['method'].apply(lambda x: float(x.split('_')[4]) if x.startswith('saa') else '')
df['dro_gamma'] = df['method'].apply(lambda x: float(x.split('_')[3]) if x.startswith('dro') else '')
df['dro_alpha'] = df['method'].apply(lambda x: float(x.split('_')[5]) if x.startswith('dro') else '')

# add a column to identify each method
df['method_base'] = df['method']
df.loc[df['method'].str.contains('ro_gamma'), ['method_base']] = 'ro'
df.loc[df['method'].str.startswith('saa_'), ['method_base']] = 'saa'
df.loc[df['method'].str.startswith('dro_'), ['method_base']] = 'dro'

# --- aggregate results for each graph-distribution pair, over all realizations ---
# create a uid for identifying each graph-distribution and each method

# create an id for each graph-distribution pair
id_columns = ['graph_name', 'trial_num', 'alpha', 'cycle_cap', 'chain_cap']
df['uid'] = df[id_columns].apply(tuple, axis=1)

# for each method and each uid, get the max, min, mean, and mean-lowest-p% of the realized edge weights

df_grouped = df.groupby(by=['uid', 'method_base', 'ro_gamma', 'saa_gamma', 'saa_alpha', 'dro_gamma', 'dro_alpha']).agg(
    {
        'realized_score': ['min', 'max', 'mean', lower_20pct_trimmed_mean]
    }).reset_index()

new_cols = list(df_grouped.columns.droplevel(0))
new_cols[:7] = list(df_grouped.columns.droplevel(1)[:7])
# new_cols[:5] = list(df_grouped.columns.droplevel(1)[:5])
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

font = font_manager.FontProperties(family='Courier New')

remove_zeros = False

# create two subplots
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(5, 3))

# --- gamma - RO ---
if remove_zeros:
    df_ro = df_clean[(df_clean['method_base'] == 'ro') & (df_clean[plot_field] != 0.0)]
else:
    df_ro = df_clean[df_clean['method_base'] == 'ro']

ax_ro = sns.boxplot(x='ro_gamma', y=plot_field, data=df_ro, ax=ax1)  # , label="_nolegend_")

ax1.set_title("RO")
ax1.set_ylabel("$\\%NR$")
ax1.set_xlabel("$\\Gamma$")
baseline = ax1.plot([-100, 100], [0.0, 0.0], 'r:', linewidth=2, label="NR (baseline)")

ax1.legend()  # ([baseline], ["NR (baseline)"])  # ,bbox_to_anchor=(1.05, 1), loc=2)

# below - this is the baseline, so always zero
# # --- nonrobust ---
# df_nonrobust = df_clean.loc[df_clean['method_base'].str.contains('nonrobust')]
# df_nonrobust['mean_type'] = 'sample mean'
# df_nonrobust.loc[df_nonrobust['method_base'].str.contains('true'), ['mean_type']] = 'true mean'
#
# ax_nonrobust = sns.boxplot(x='mean_type', y=plot_field, data=df_nonrobust, ax=ax2)

# --- gamma - SSA ---
if remove_zeros:
    df_saa = df_clean[(df_clean['method_base'] == 'saa') & (df_clean[plot_field] != 0.0)]
else:
    df_saa = df_clean[(df_clean['method_base'] == 'saa')]

ax_saa = sns.boxplot(x='saa_gamma', y=plot_field, data=df_saa, ax=ax2)

# plot a line of the means
gamma_vals = sorted(list(df_saa['saa_gamma'].unique()))
mean_vals = []
for gamma in gamma_vals:
    mean_vals.append(df_saa[df_saa['saa_gamma'] == gamma][plot_field].mean())

ax2.plot(gamma_vals, mean_vals, 'g', linewidth=2)
ax2.plot([-100, 100], [0.0, 0.0], 'r:', linewidth=2)
ax2.set_title("CVar")
ax2.set_ylabel("")
ax2.set_xlabel("$\\gamma$")

plt.tight_layout()
# plt.savefig("/Users/duncan/Downloads/lkdpi_results.pdf")
plt.savefig("/Users/duncan/Downloads/unos_results.pdf")

# --- for dro ---


plot_field = '20pct_pct_diff'
# plot_field = 'mean_pct_diff'

font = font_manager.FontProperties(family='Courier New')

remove_zeros = True

# create two subplots
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(6, 3))

# --- gamma - RO ---
if remove_zeros:
    df_ro = df_clean[(df_clean['method_base'] == 'ro') & (df_clean[plot_field] != 0.0)]
else:
    df_ro = df_clean[df_clean['method_base'] == 'ro']

ax_ro = sns.boxplot(x='ro_gamma', y=plot_field, data=df_ro, ax=ax1)  # , label="_nolegend_")

ax1.set_title("RO")
ax1.set_ylabel("$\\%NR$")
ax1.set_xlabel("$\\Gamma$")
baseline = ax1.plot([-100, 100], [0.0, 0.0], 'r:', linewidth=2, label="NR (baseline)")

ax1.legend()  # ([baseline], ["NR (baseline)"])  # ,bbox_to_anchor=(1.05, 1), loc=2)

# below - this is the baseline, so always zero
# # --- nonrobust ---
# df_nonrobust = df_clean.loc[df_clean['method_base'].str.contains('nonrobust')]
# df_nonrobust['mean_type'] = 'sample mean'
# df_nonrobust.loc[df_nonrobust['method_base'].str.contains('true'), ['mean_type']] = 'true mean'
#
# ax_nonrobust = sns.boxplot(x='mean_type', y=plot_field, data=df_nonrobust, ax=ax2)

# --- gamma - SSA ---
if remove_zeros:
    df_saa = df_clean[(df_clean['method_base'] == 'saa') & (df_clean[plot_field] != 0.0)]
else:
    df_saa = df_clean[(df_clean['method_base'] == 'saa')]

ax_saa = sns.boxplot(x='saa_gamma', y=plot_field, data=df_saa, ax=ax2)

# plot a line of the means
gamma_vals = sorted(list(df_saa['saa_gamma'].unique()))
mean_vals = []
for gamma in gamma_vals:
    mean_vals.append(df_saa[df_saa['saa_gamma'] == gamma][plot_field].mean())

ax2.plot(gamma_vals, mean_vals, 'g', linewidth=2)
ax2.plot([-100, 100], [0.0, 0.0], 'r:', linewidth=2)
ax2.set_title("CVar")
ax2.set_ylabel("")
ax2.set_xlabel("$\\gamma$")


# --- gamma - dro ---
if remove_zeros:
    df_saa = df_clean[(df_clean['method_base'] == 'dro') & (df_clean[plot_field] != 0.0)]
else:
    df_saa = df_clean[(df_clean['method_base'] == 'dro')]

ax_saa = sns.boxplot(x='dro_gamma', y=plot_field, data=df_saa, ax=ax3)

# plot a line of the means
gamma_vals = sorted(list(df_saa['dro_gamma'].unique()))
mean_vals = []
for gamma in gamma_vals:
    mean_vals.append(df_saa[df_saa['dro_gamma'] == gamma][plot_field].mean())

ax3.plot(gamma_vals, mean_vals, 'g', linewidth=2)
ax3.plot([-100, 100], [0.0, 0.0], 'r:', linewidth=2)
ax3.set_title("DRO")
ax3.set_ylabel("")
ax3.set_xlabel("$\\gamma$")

plt.tight_layout()
plt.savefig("/Users/duncan/Downloads/dro_results.pdf")

#
#
# # --- make plots for individual kex graphs ---
#
# uid_list = df_clean['uid'].unique()
#
# uid = uid_list[7]
# print('uid: %s' % str(uid))
#
# # create three subplots
# f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
#
# # --- gamma - RO ---
# df_ro = df[(df['method_base'] == 'ro') & (df['uid'] == uid)]
#
# ax_ro = sns.boxplot(x='ro_gamma', y='realized_score', data=df_ro, ax=ax1)
#
# # --- nonrobust ---
# df_nonrobust = df.loc[df['method_base'].str.contains('nonrobust') & (df['uid'] == uid)]
# df_nonrobust['mean_type'] = 'sample mean'
# df_nonrobust.loc[df_nonrobust['method_base'].str.contains('true'), ['mean_type']] = 'true mean'
#
# ax_nonrobust = sns.boxplot(x='mean_type', y='realized_score', data=df_nonrobust, ax=ax2)
#
# # --- gamma - SSA ---
# df_saa = df[(df['method_base'] == 'saa') & (df['uid'] == uid)]
#
# ax_saa = sns.boxplot(x='saa_gamma', y='realized_score', data=df_saa, ax=ax3)
#
# plt.title(uid[0])
