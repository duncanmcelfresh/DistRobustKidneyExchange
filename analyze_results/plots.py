# plot results

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# output_file = '/Users/duncan/research/DistRobustKex_output/robust_kex_experiment_20190519_182539.csv'
output_file = '/Users/duncan/research/DistRobustKex_output/robust_kex_experiment_20190520_195155.csv'

df = pd.read_csv(output_file, skiprows=1)

# strip whitespace from cols
df.columns = [str.strip(c) for c in df.columns]

# strip whitespace from the 'method' columns
df['method'] = df['method'].str.replace(" ","")

# add a unique identifier for each independent realization, which consists of:
# (graph_name, trial_num, alpha, realization_num, cycle_cap, chain_cap) tuple
id_columns = ['graph_name', 'trial_num', 'alpha', 'realization_num', 'cycle_cap', 'chain_cap']
df['uid'] = df[id_columns].apply(tuple, axis=1)

# get the best possible objval, objval_scaled, and rank, and add them as columns to the df
df_best = df.loc[df['method'] == 'optimal', ['uid', 'realized_score']]

df_best.rename(columns={'realized_score': 'best_realized_score'}, inplace=True)

df_clean = pd.merge(df, df_best, on='uid')

# calculate the absolute optimality gap for all methods
df_clean['abs_opt_gap'] = df_clean['realized_score'] - df_clean['best_realized_score']

# calculate the relative opt gap
df_clean['relative_opt_gap'] = df_clean['abs_opt_gap'] / df_clean['best_realized_score']

# change all parameter_value = None to 0, and make this column floats
df_clean[df_clean['parameter_value'].isna()]

# make sure relative_opt_gap is always zero for optimal
assert len(df_clean[df_clean['method'] == 'optimal']['relative_opt_gap'].unique()) == 1
assert df_clean[df_clean['method'] == 'optimal']['relative_opt_gap'].unique()[0] == 0.0






# -----------------------------------
# --------- initial plots -----------
# -----------------------------------

# show opt gap by theta, for each alpha


df_plot = df_clean[df_clean['alpha'] == 0.9]

fig, axs = plt.subplots(1, 4, figsize=(9, 3), sharey=True)

# plot optimal values
df_method = df_plot[df_plot['method'] == 'optimal']
g = sns.catplot(x="parameter_value",
                y="relative_opt_gap",
                ax=axs[0],
                data=df_method,
                kind="box",
                sharex=False,
                height=3) #, aspect=.7)

df_method = df_plot[df_plot['method'] == 'nonrobust']
g = sns.catplot(x="parameter_value",
                y="relative_opt_gap",
                ax=axs[1],
                data=df_method,
                kind="box",
                sharex=False,
                height=3) #, aspect=.7)

df_method = df_plot[df_plot['method'] == 'RO']
g = sns.catplot(x="parameter_value",
                y="relative_opt_gap",
                ax=axs[2],
                data=df_method,
                kind="box",
                sharex=False,
                height=3) #, aspect=.7)

df_method = df_plot[df_plot['method'] == 'DRO']
g = sns.catplot(x="parameter_value",
                y="relative_opt_gap",
                ax=axs[3],
                data=df_method,
                kind="box",
                sharex=False,
                height=3) #, aspect=.7)






# df_plot = df_clean[df_clean['num_items'].isin(num_items_list) & df_clean['method'].isin(display_methods)]
alpha_list = [0.5, 0.9]
df_plot = df_clean[df_clean['alpha'] == 0.5]

method_order = ['optimal',
                'nonrobust',
                'DRO']
                # 'RO']
g = sns.catplot(x="parameter_value",
                y="relative_opt_gap",
                col="method",
                col_order=method_order,
                row='alpha',
                data=df_plot,
                kind="box",
                sharex=False,
                height=3) #, aspect=.7)

# reset x-axis limits
for i, ax in enumerate(g.axes.flatten()):
    # rescale x-axis
    max_x = df_plot[df_plot['method'] == method_order[i]]['parameter_value'].max() * 1.05
    min_x = df_plot[df_plot['method'] == method_order[i]]['parameter_value'].min() * 0.95
    print('min:%f, max:%f' % (min_x, max_x))
    ax.set_xlim(min_x, max_x)

# reset axes
plt.tight_layout()


# -----------------------------------
# --------- final plots -------------
# -----------------------------------
from matplotlib import rc
# rc('text', usetex=False)
import matplotlib.font_manager as font_manager
plt.rcParams["font.family"] = "Times New Roman"

# palette = sns.color_palette("cubehelix", 3)
palette = sns.color_palette("OrRd", 3)

# modify a seaborn catplot
num_items_list = [20]
num_features_list = [2, 5, 10, 15]
display_methods = ['random', 'ac', 'opt']
labels = ['Rand', 'AC', 'Opt']

df_plot = df_clean[df_clean['num_items'].isin(num_items_list)
                   & df_clean['num_features'].isin(num_features_list)
                   & df_clean['method'].isin(display_methods)]

# ----
# plot 1: absolute objective gap (boxplots), by number of features and number of items.
# x axis: k (num queries)
# y axis: absolute objective gap (should approach zero)
# grid row : num. items
# grid column: num. features
# ----

g = sns.catplot(x="k", y="opt_gap",
                    hue="method",
                    col="num_features",
                    row='num_items',
                    data=df_plot,
                    kind="box",
                    palette=palette,
                    height=3,
                    flierprops={'marker': '+'},
                    legend=False,
                    hue_order=display_methods) #, aspect=.7)

# set title for each subplot
for i, ax in enumerate(g.axes.flatten()):
    ax.set_title('%d Features' % num_features_list[i])

# set legend text
font = font_manager.FontProperties(family='Courier New')
leg_ax = g.axes.flatten()[0]
handles, _ = leg_ax.get_legend_handles_labels()
leg_ax.legend(title='Method', handles=handles, prop=font , labels=labels)

# set x axis label
leg_ax.set_ylabel('Absolute Optimality Gap')

# plt.tight_layout()

# ----
# plot 2: true rank of recommended item (boxplots), by number of features and number of items.
#  (same as plot 1, but for rank)
# x axis: k (num queries)
# y axis: rank of recommended item
# grid row : num. items
# grid column: num. features
# ----


g = sns.catplot(x="k", y="true_recommendation_rank",
                    hue="method",
                    col="num_features",
                    row='num_items',
                    data=df_plot,
                    kind="box",
                    palette=palette,
                    height=3,
                    flierprops={'marker': '+'},
                    legend=False,
                    hue_order=display_methods)

# set title for each subplot
for i, ax in enumerate(g.axes.flatten()):
    ax.set_title('%d Features' % num_features_list[i])
    # REVERSE Y AXIS
    ax.set_ylim(20, 0)

# set legend text
font = font_manager.FontProperties(family='Courier New')
leg_ax = g.axes.flatten()[0]
handles, _ = leg_ax.get_legend_handles_labels()
leg_ax.legend(title='Method', handles=handles, prop=font , labels=labels)

# set x axis label
leg_ax.set_ylabel('True Rank of Recommended Item')


# --- now worst-case plots

df_worstcase_rank = df_plot.groupby(['num_items', 'num_features', 'method', 'k'])['true_recommendation_rank'].max().reset_index()
df_worstcase_obj_gap = df_plot.groupby(['num_items', 'num_features', 'method', 'k'])['opt_gap'].max().reset_index()

markers = ["x", "+", "o"]
linestyles = [":", "--", '-']


# ----
# plot 3: WORST CASE objective gap
#  (same as plot 1, but for rank)
# x axis: k (num queries)
# y axis: rank of recommended item
# grid row : num. items
# grid column: num. features
# ----

g = sns.catplot(x="k", y="opt_gap",
                    hue="method",
                    col="num_features",
                    row='num_items',
                    data=df_worstcase_obj_gap,
                    kind="point",
                    palette=palette,
                    height=3,
                    legend=False,
                    hue_order=display_methods,
                    markers=markers,
                    linestyles=linestyles)

# set title for each subplot
for i, ax in enumerate(g.axes.flatten()):
    ax.set_title('%d Features' % num_features_list[i])

# set legend text
font = font_manager.FontProperties(family='Courier New')
leg_ax = g.axes.flatten()[0]
handles, _ = leg_ax.get_legend_handles_labels()
leg_ax.legend(title='Method', handles=handles, prop=font, labels=labels)

# set x axis label
leg_ax.set_ylabel('Wost-Case Absolute Optimality Gap')


# ----
# plot 4: WORST CASE rank of recommended item (line plot), by number of features and number of items.
#  (same as plot 1, but for rank)
# x axis: k (num queries)
# y axis: rank of recommended item
# grid row : num. items
# grid column: num. features
# ----


g = sns.catplot(x="k", y="true_recommendation_rank",
                    hue="method",
                    col="num_features",
                    row='num_items',
                    data=df_worstcase_rank,
                    kind="point",
                    palette=palette,
                    height=3,
                    legend=False,
                    hue_order=display_methods,
                    markers=markers,
                    linestyles=linestyles)

# set title for each subplot
for i, ax in enumerate(g.axes.flatten()):
    ax.set_title('%d Features' % num_features_list[i])
    # REVERSE Y AXIS
    ax.set_ylim(20, 0)

# set legend text
font = font_manager.FontProperties(family='Courier New')
leg_ax = g.axes.flatten()[0]
handles, _ = leg_ax.get_legend_handles_labels()
leg_ax.legend(title='Method', handles=handles, prop=font, labels=labels)

# set x axis label
leg_ax.set_ylabel('Worst-Case True Rank of Recommended Item')












# -----------------------------------
# --- plots, for initial analysis ---
# -----------------------------------

# offset_scaled = - 10 # df_clean['objval_scaled'].min()
# df_clean['pct_objval_scaled'] = 100 * (- df_clean['objval_scaled'] + df_clean['best_objval_scaled']) / (- offset_scaled + df_clean['best_objval_scaled'].abs())
# offset_unscaled = df_clean['objval'].min()
# df_clean['pct_objval'] = 100 * (- df_clean['objval'] + df_clean['best_objval']) / (- offset_unscaled + df_clean['best_objval'].abs())
#
# df_clean['pct_objval_no_offset'] = 100 * (- df_clean['objval'] + df_clean['best_objval']) / (df_clean['best_objval'])
# df_clean['objval_diff'] = (- df_clean['objval'] + df_clean['best_objval'])

# spot check for errors
# df_clean[df_clean['uid'] == (10, 6, 0, 0)][['method', 'objval_scaled', 'best_objval_scaled', 'pct_objval_scaled']]


# -- objval_scaled --

num_items_list = [10, 20, 30]
display_methods = ['random', 'ac', 'opt']

df_plot = df_clean[df_clean['num_items'].isin(num_items_list) & df_clean['method'].isin(display_methods)]

g = sns.catplot(x="k", y="pct_objval_scaled",
                    hue="method", col="num_features", row='num_items',
                    data=df_plot, kind="box",
                    height=3) #, aspect=.7)

plt.tight_layout()
plt.savefig('/Users/duncan/Desktop/objval_scaled.png')


# -- pct_objval unscaled --


num_items_list = [10, 20, 30]
display_methods = ['random', 'ac', 'opt']

df_plot = df_clean[df_clean['num_items'].isin(num_items_list) & df_clean['method'].isin(display_methods)]

g = sns.catplot(x="k", y="pct_objval",
                    hue="method", col="num_features", row='num_items',
                    data=df_plot, kind="box",
                    height=3) #, aspect=.7)

plt.tight_layout()

plt.savefig('/Users/duncan/Desktop/pct_objval_unscaled.png')


# -- objval unscaled, without offset --

num_items_list = [10, 20, 30]
display_methods = ['random', 'ac', 'opt']

df_plot = df_clean[df_clean['num_items'].isin(num_items_list) & df_clean['method'].isin(display_methods)]

g = sns.catplot(x="k", y="pct_objval_no_offset",
                    hue="method", col="num_features", row='num_items',
                    data=df_plot, kind="box",
                    height=3) #, aspect=.7)

plt.tight_layout()
plt.savefig('/Users/duncan/Desktop/objval_no_offset.png')


# -- objval unscaled, without offset --

num_items_list = [10, 20, 30]
display_methods = ['random', 'ac', 'opt', 'best_possible']

df_plot = df_clean[df_clean['num_items'].isin(num_items_list) & df_clean['method'].isin(display_methods)]

g = sns.catplot(x="k", y="objval",
                    hue="method", col="num_features", row='num_items',
                    data=df_plot, kind="box",
                    height=3) #, aspect=.7)

plt.tight_layout()
plt.savefig('/Users/duncan/Desktop/actual_objval.png')

# -- objval unscaled, without offset --

num_items_list = [10, 20, 30]
display_methods = ['random', 'ac', 'opt', 'best_possible']

df_plot = df_clean[df_clean['num_items'].isin(num_items_list) & df_clean['method'].isin(display_methods)]

g = sns.catplot(x="k", y="objval_scaled",
                    hue="method", col="num_features", row='num_items',
                    data=df_plot, kind="box",
                    height=3) #, aspect=.7)

plt.tight_layout()
plt.savefig('/Users/duncan/Desktop/actual_objval_scaled.png')


# -- difference between optimal objval and actual --

num_items_list = [10, 20, 30]
display_methods = ['random', 'ac', 'opt']

df_plot = df_clean[df_clean['num_items'].isin(num_items_list) & df_clean['method'].isin(display_methods)]

g = sns.catplot(x="k", y="objval_diff",
                    hue="method", col="num_features", row='num_items',
                    data=df_plot, kind="box",
                    height=3) #, aspect=.7)

plt.tight_layout()
plt.savefig('/Users/duncan/Desktop/objval_diff.png')






# -- rank --


num_items_list = [10, 20, 30]
display_methods = ['random', 'ac', 'opt']

df_plot = df_clean[df_clean['num_items'].isin(num_items_list) & df_clean['method'].isin(display_methods)]

g = sns.catplot(x="k", y="true_recommendation_rank",
                    hue="method", col="num_features", row='num_items',
                    data=df_plot, kind="box",
                    height=3) #, aspect=.7)

for ax in g.axes.flat:
    ax.invert_yaxis()

plt.tight_layout()

plt.savefig('/Users/duncan/Desktop/rank.png')


# --- plot worst-case rank ---

num_items_list = [10, 20, 30]
display_methods = ['random', 'ac', 'opt']

df_tmp = df_clean[df_clean['num_items'].isin(num_items_list) & df_clean['method'].isin(display_methods)]

df_plot = df_tmp.groupby(['num_items', 'num_features', 'method', 'k'])['true_recommendation_rank'].max().reset_index()

g = sns.catplot(x="k", y="true_recommendation_rank",
                    hue="method", col="num_features", row='num_items',
                    data=df_plot, kind="point",
                    height=3) #, aspect=.7)

for ax in g.axes.flat:
    ax.invert_yaxis()

plt.tight_layout()

plt.savefig('/Users/duncan/Desktop/worstcase_rank.png')


# --- plot worst-case optimality gap ---

num_items_list = [10, 20, 30]
display_methods = ['random', 'ac', 'opt']

df_tmp = df_clean[df_clean['num_items'].isin(num_items_list) & df_clean['method'].isin(display_methods)]

df_plot = df_tmp.groupby(['num_items', 'num_features', 'method', 'k'])['pct_objval_scaled'].max().reset_index()

g = sns.catplot(x="k", y="pct_objval_scaled",
                    hue="method", col="num_features", row='num_items',
                    data=df_plot, kind="point",
                    height=3) #, aspect=.7)

plt.tight_layout()

plt.savefig('/Users/duncan/Desktop/worstcase_optgap.png')