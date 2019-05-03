

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt


results_file_KPD = r'/Users/duncan/research/robust_kex/results/gold/KPD_min_chain_len_20180903_131654.csv'
outname_KPD = 'kpd_chainlen'

outdir = r'/Users/duncan/research/robust_kex/results/img/'


df = pd.read_csv(results_file_KPD, delimiter=';')

# get the number of NDDs for each name -- using the cap_chain_list
df['num_ndds'] = df.apply(lambda row: len(ast.literal_eval(row['cap_chain_list'])),axis=1)

# first get the chain lengths for all uncapped formulations
df_uncap = df.loc[:,['name','num_ndds','nocap_chain_list']].copy().drop_duplicates()

# all of the filenames should be unique...
assert len(df_uncap['name'].unique()) == len(df_uncap)

# now get the min-chain-capped results
df_cap = df.loc[:,['name','min_chain_len','num_ndds','cap_chain_list','sol_nocap_total_score','sol_cap_total_score']].copy()
# add the optimality gap
df_cap['pct_opt'] = (df_cap['sol_cap_total_score'] - df_cap['sol_nocap_total_score'])/df_cap['sol_nocap_total_score']

# remove any where there are no chains in the capped version (i.e. infeasible to enforce min-chain-cap)
# NOTE: there may be instances where it is optimal to have NO chains, in which case the uncapped row will have []
name_drop = df_cap[df_cap['num_ndds']==0]['name'].unique()

df_cap = df_cap.loc[~df_cap['name'].isin(name_drop)]
df_uncap = df_uncap.loc[~df_uncap['name'].isin(name_drop)]

# make sure they have the same names
assert sorted(df_cap['name'].unique()) == sorted(df_uncap['name'].unique())

# make sure the min. num of NDDs is 1
assert (df_cap['num_ndds'].min() > 0) & (df_uncap['num_ndds'].min() > 0)


# get the total chain length for both df
df_cap['total_chain_len'] = df_cap['cap_chain_list'].apply(lambda x: np.sum(ast.literal_eval(x)))
df_uncap['total_chain_len'] = df_uncap['nocap_chain_list'].apply(lambda x: np.sum(ast.literal_eval(x)))

# get all of the chain lengths for the uncapped rows
uncap_list = []
for idx,row in df_uncap.iterrows():
    chain_list = ast.literal_eval(row['nocap_chain_list'])
    uncap_list.extend(chain_list)
    # if the number of chains is smaller than num_ndds, add so many zeros
    uncap_list.extend(np.zeros(int(row['num_ndds']-len(chain_list))))

# get the chain lengths for the 1-capped chains
cap1_list = []
for idx,row in df_cap[df_cap['min_chain_len']==1].iterrows():
    chain_list = ast.literal_eval(row['cap_chain_list'])
    cap1_list.extend(chain_list)
    # if the number of chains is smaller than num_ndds, add so many zeros
    cap1_list.extend(np.zeros(int(row['num_ndds']-len(chain_list))))

# get the chain lengths for the 2-capped chains
cap2_list = []
for idx,row in df_cap[df_cap['min_chain_len']==2].iterrows():
    chain_list = ast.literal_eval(row['cap_chain_list'])
    cap2_list.extend(chain_list)
    # if the number of chains is smaller than num_ndds, add so many zeros
    cap2_list.extend(np.zeros(int(row['num_ndds']-len(chain_list))))

# get the chain lengths for the 3-capped chains
cap3_list = []
for idx,row in df_cap[df_cap['min_chain_len']==3].iterrows():
    chain_list = ast.literal_eval(row['cap_chain_list'])
    cap3_list.extend(chain_list)
    # if the number of chains is smaller than num_ndds, add so many zeros
    cap3_list.extend(np.zeros(int(row['num_ndds']-len(chain_list))))

# plot histograms of each
bins = [-0.01,0.99,1.99,2.99,3.99]
bins_pct = np.linspace(-0.4,0.05,10)

import matplotlib
matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('font', family='serif', serif='Times New Roman')
matplotlib.rc('text', usetex=True)


fig, axes = plt.subplots(2, 4, sharex='row',sharey='row',figsize=(10,5))

axes[1,0].hist(uncap_list, bins=bins, align='left',density=True, rwidth=0.8)
axes[1,1].hist(cap1_list, bins=bins, align='left',density=True, rwidth=0.8)
axes[1,2].hist(cap2_list, bins=bins, align='left',density=True, rwidth=0.8)
axes[1,3].hist(cap3_list, bins=bins, align='left',density=True, rwidth=0.8)


# now plot the relative loss in efficiency
# plt.hist(df_cap[df_cap['min_chain_len']==3]['pct_opt'], align='left',normed=True,color='red')
l1 = len(df_cap[df_cap['min_chain_len']==1])
l2 = len(df_cap[df_cap['min_chain_len']==2])
l3 = len(df_cap[df_cap['min_chain_len']==3])

axes[0,0].hist(np.zeros(l1),range=(-0.4,0.0),bins=10, align='right',density=False,color='red', rwidth=0.8,weights=np.ones(l1)/l1)
axes[0,1].hist(df_cap[df_cap['min_chain_len']==1]['pct_opt'],range=(-0.4,0.0),bins=10, align='right',density=False,color='red', rwidth=0.8,weights=np.ones(l1)/l1)
axes[0,2].hist(df_cap[df_cap['min_chain_len']==2]['pct_opt'],range=(-0.4,0.0),bins=10, align='right',density=False,color='red', rwidth=0.8,weights=np.ones(l2)/l2)
axes[0,3].hist(df_cap[df_cap['min_chain_len']==3]['pct_opt'],range=(-0.4,0.0),bins=10, align='right',density=False,color='red', rwidth=0.8,weights=np.ones(l3)/l3)

axes[0,0].set_ylabel('Fraction of Matchings')
axes[1,0].set_ylabel('Fraction of Chains')

axes[0,0].set_xlabel('$\Delta OPT$')
axes[1,0].set_xlabel('Chain Length')
axes[0,1].set_xlabel('$\Delta OPT$')
axes[1,1].set_xlabel('Chain Length')
axes[0,2].set_xlabel('$\Delta OPT$')
axes[1,2].set_xlabel('Chain Length')
axes[0,3].set_xlabel('$\Delta OPT$')
axes[1,3].set_xlabel('Chain Length')

axes[0,0].set_title('$L_{min}=0$')
axes[0,1].set_title('$L_{min}=1$')
axes[0,2].set_title('$L_{min}=2$')
axes[0,3].set_title('$L_{min}=3$')

axes[0,0].xaxis.set_ticks([-0.3,-0.2,-0.1,0.0])
axes[1,0].xaxis.set_ticks([0,1,2,3])

fig.tight_layout()

plt.savefig(outdir+'min_chain_len.pdf',additional_artists=[], bbox_inches='tight')

# get the average chain length for uncapped chains
print "mean chain length for UN-CAPPED CHAINS: %f" % float(df_uncap['total_chain_len'].sum()/df_uncap['num_ndds'].sum())

print "mean chain length for MIN-1 CHAINS: %f" %
