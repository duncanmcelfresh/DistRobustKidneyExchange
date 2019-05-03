import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import scipy
import string
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

data_dir = "/Users/duncan/research/HealthyMe_app"
mturk_csv = "mturk_healthyme_batch_results.csv"
topic_json = "HF-all.json"

# --- read and clean MTurk data ---

# read mturk data
infile = os.path.join(data_dir,mturk_csv)
mturk_data = pd.read_csv(infile)

# clean mturk data
keep_cols = []

# Look at app-relevant demo factors:
# age groups (5)
# sex at birth
# hispanic/latino
# race (4)
# smoke?
keep_cols.extend(['Answer.userAge',
                  'Answer.assignedSex',
                  'Answer.hispanicLatinoOrSpanish',
                  'Answer.userRace',
                  'Answer.currentlySmoke'])

# also, look at goals
keep_cols.append('Answer.HealthGoals')

########################################################################################################################
### Question 1:
### What is the different in ratings of a) suggested articles (related to goals) and b) user-selected?
########################################################################################################################

# It seems that goals 1, 2, 3 correspond to articles from the self-selected goals, while **goal 4 is the self-browse**

# aggregate all ratings
rating_cols = []

# it looks like goal 1, 2, 3 are from the self-selected goals, while **goal 4 is the self-browse**...
rating_cols.extend(['Answer.goal1ImportanceA',
                  'Answer.goal1ImportanceB',
                  'Answer.goal1ImportanceC',
                  'Answer.goal2ImportanceA',
                  'Answer.goal2ImportanceB',
                  'Answer.goal2ImportanceC',
                  'Answer.goal3ImportanceA',
                  'Answer.goal3ImportanceB',
                  'Answer.goal3ImportanceC',
                  'Answer.goal4ImportanceA',
                  'Answer.goal4ImportanceB',
                  'Answer.goal4ImportanceC'])

# also get feasibility
rating_cols.extend(['Answer.goal1FeasibilityA',
                  'Answer.goal1FeasibilityB',
                  'Answer.goal1FeasibilityC',
                  'Answer.goal2FeasibilityA',
                  'Answer.goal2FeasibilityB',
                  'Answer.goal2FeasibilityC',
                  'Answer.goal3FeasibilityA',
                  'Answer.goal3FeasibilityB',
                  'Answer.goal3FeasibilityC',
                  'Answer.goal4FeasibilityA',
                  'Answer.goal4FeasibilityB',
                  'Answer.goal4FeasibilityC'])

# whether or not you'd include in your personal library
# I ASSUME THIS IS CORRECT
rating_cols.extend(['Answer.includeOrNot1A',
                  'Answer.includeOrNot1B',
                  'Answer.includeOrNot1C',
                  'Answer.includeOrNot2A',
                  'Answer.includeOrNot2B',
                  'Answer.includeOrNot2C',
                  'Answer.includeOrNot3A',
                  'Answer.includeOrNot3B',
                  'Answer.includeOrNot3C',
                  'Answer.includeOrNot4A',
                  'Answer.includeOrNot4B',
                  'Answer.includeOrNot4C'])

# grab only the ratings of each article
ratings_data = mturk_data[rating_cols]

# make a long dataset of all ratings
ratings_data_long = ratings_data.stack().reset_index()
ratings_data_long.columns = ['user_num','rating_name','rating']

# make a column indicating which goal #: 1,2,3,4 (0 is NA)
ratings_data_long['goal_num'] = 0
ratings_data_long.loc[ratings_data_long['rating_name'].str.contains('1'),'goal_num'] = 1
ratings_data_long.loc[ratings_data_long['rating_name'].str.contains('2'),'goal_num'] = 2
ratings_data_long.loc[ratings_data_long['rating_name'].str.contains('3'),'goal_num'] = 3
ratings_data_long.loc[ratings_data_long['rating_name'].str.contains('4'),'goal_num'] = 4

# make a column indicating which article (A,B,C)
ratings_data_long['article_label'] = 'N'
ratings_data_long.loc[ratings_data_long['rating_name'].str.endswith('A'),'article_label'] = 'A'
ratings_data_long.loc[ratings_data_long['rating_name'].str.endswith('B'),'article_label'] = 'B'
ratings_data_long.loc[ratings_data_long['rating_name'].str.endswith('C'),'article_label'] = 'C'


# make a column for the type of rating
ratings_data_long['type'] = 'NA'
ratings_data_long.loc[ratings_data_long['rating_name'].str.contains('Feasibility'),'type'] = 'Feasibility'
ratings_data_long.loc[ratings_data_long['rating_name'].str.contains('Importance'),'type'] = 'Importance'
ratings_data_long.loc[ratings_data_long['rating_name'].str.contains('include'),'type'] = 'include'

# recode each rating on the following scale (ALL POSITIVE CODING): 0 is null value
ratings_data_long['rating_num'] = -1

# Importance:
# - Very unimportant = 1
# - Unimportant = 2
# - Moderately important = 3
# - Important = 4
# - Very important = 5
ratings_data_long.loc[(ratings_data_long['rating'].str.contains('veryUnimportant')
                       & (ratings_data_long['type'] == 'Importance')),'rating_num'] = 1
ratings_data_long.loc[(ratings_data_long['rating'].str.contains('unimportant')
                       & (ratings_data_long['type'] == 'Importance')),'rating_num'] = 2
ratings_data_long.loc[(ratings_data_long['rating'].str.contains('moderatelyImportant')
                       & (ratings_data_long['type'] == 'Importance')),'rating_num'] = 3
ratings_data_long.loc[(ratings_data_long['rating'].str.contains('important')
                       & (ratings_data_long['type'] == 'Importance')),'rating_num'] = 4
ratings_data_long.loc[(ratings_data_long['rating'].str.contains('veryImportant')
                       & (ratings_data_long['type'] == 'Importance')),'rating_num'] = 5

# Feasibility:
# - Not at all likely = 1
# - Unlikely = 2
# - I don't know = 3
# - Likely = 4
# - Very likely = 5
ratings_data_long.loc[(ratings_data_long['rating'].str.contains('notAtAllLikely')
                       & (ratings_data_long['type'] == 'Feasibility')),'rating_num'] = 1
ratings_data_long.loc[(ratings_data_long['rating'].str.contains('unlikely')
                       & (ratings_data_long['type'] == 'Feasibility')),'rating_num'] = 2
ratings_data_long.loc[(ratings_data_long['rating'].str.contains('iDontKnow')
                       & (ratings_data_long['type'] == 'Feasibility')),'rating_num'] = 3
ratings_data_long.loc[(ratings_data_long['rating'].str.contains('likely')
                       & (ratings_data_long['type'] == 'Feasibility')),'rating_num'] = 4
ratings_data_long.loc[(ratings_data_long['rating'].str.contains('veryLikely')
                       & (ratings_data_long['type'] == 'Feasibility')),'rating_num'] = 5

# Include in library?:
# - yes = 1
# - no = 0
ratings_data_long.loc[(ratings_data_long['rating'].str.contains('no')
                       & (ratings_data_long['type'] == 'include')),'rating_num'] = 1
ratings_data_long.loc[(ratings_data_long['rating'].str.contains('yes')
                       & (ratings_data_long['type'] == 'include')), 'rating_num'] = 2

# check that the ratings are valid
ratings_data_long['rating_num'].value_counts()

# output:
# 4   4958
# 2    4162
# 5    3938
# 1    2169
# 3    1743

# also mark which goals correspond to self, and rec
ratings_data_long['goal_type'] = 'rec'
ratings_data_long.loc[ratings_data_long['goal_num']==4,'goal_type'] = 'self'

# add string encoding for goal number
ratings_data_long['goal_num_str'] = ratings_data_long['goal_num'].astype(str)

# plot histogram of replies by each goal num
ratings_data_long[(ratings_data_long['type']=='include')]['rating_num'].hist(by=ratings_data_long['goal_num'])
ratings_data_long[(ratings_data_long['type']=='include')]['rating_num'].hist(by=ratings_data_long['goal_type'],normed=True)


# --- Histogram of 'would you include in library...'

data_self = ratings_data_long[((ratings_data_long['type'] == 'include') & (ratings_data_long['goal_type'] == 'self'))]['rating_num']
weights_self = np.ones_like(data_self) / float(len(data_self))
# plt.hist(data_self,weights=weights_self, label='self-selected')
data_rec = ratings_data_long[((ratings_data_long['type'] == 'include') & (ratings_data_long['goal_type'] == 'rec'))]['rating_num']
weights_rec = np.ones_like(data_rec) / float(len(data_rec))
# plt.hist(data_rec,weights=weights_rec, label='goal-related')

plt.hist([data_self,data_rec], weights = [weights_self,weights_rec], label=['self-selected','goal-related'], bins=[0,1.5,3])
plt.title("Responses to 'Would you include <article> in your personal library?'")
plt.legend()
plt.xticks([0.75,2.25],['no','yes'])
plt.tight_layout()

# --- Histogram of Feasibility

data_self = ratings_data_long[((ratings_data_long['type'] == 'Feasibility') & (ratings_data_long['goal_type'] == 'self'))]['rating_num']
weights_self = np.ones_like(data_self) / float(len(data_self))

data_rec = ratings_data_long[((ratings_data_long['type'] == 'Feasibility') & (ratings_data_long['goal_type'] == 'rec'))]['rating_num']
weights_rec = np.ones_like(data_rec) / float(len(data_rec))

plt.hist([data_self,data_rec], weights = [weights_self,weights_rec], label=['self-selected','goal-related'])
plt.title("Responses to Feasibility question")
plt.legend()
# plt.xticks([0.75,2.25],['no','yes'])
plt.tight_layout()


# --- Histogram of Importance

data_self = ratings_data_long[((ratings_data_long['type'] == 'Importance') & (ratings_data_long['goal_type'] == 'self'))]['rating_num']
weights_self = np.ones_like(data_self) / float(len(data_self))

data_rec = ratings_data_long[((ratings_data_long['type'] == 'Importance') & (ratings_data_long['goal_type'] == 'rec'))]['rating_num']
weights_rec = np.ones_like(data_rec) / float(len(data_rec))

plt.hist([data_self,data_rec], weights = [weights_self,weights_rec], label=['self-selected','goal-related'])
plt.title("Responses to Importance question")
plt.legend()
# plt.xticks([0.75,2.25],['no','yes'])
plt.tight_layout()

# --- Now plot by each goal number...



# --- Statistical Tests ---

# Use Wilcoxon Signed-Rank test to determine significance of difference in ratings of each goal

# for each goal, take the average rating score for each metric (positively coded)

ratings_avg = ratings_data_long[['user_num','type','goal_num','rating_num']].groupby(['user_num', 'type','goal_num']).mean().reset_index()

# -- Importance

# Compare goals 1 and 2
# to get the difference in scores, take only these goal scores,
# and set one goal to negative, and then sum by user & rating type

goal1 = 1
goal2 = 3
ratings_12 = ratings_avg[ratings_avg['goal_num'].isin([goal1,goal2])]

# set one of the goals' ratings to negative
ratings_12.loc[ratings_12['goal_num']==goal1,'rating_num'] = -1.0 * ratings_12[ratings_12['goal_num']==goal1]['rating_num']

# now get the differences, by summing over goal_num
rating_diffs = ratings_12.groupby(['user_num','type']).sum().reset_index()

# histograms
rating_diffs['rating_num'].hist(by=rating_diffs['type'])

wx_feas = scipy.stats.wilcoxon(rating_diffs[rating_diffs['type']=='Feasibility']['rating_num'], zero_method="pratt")
print wx_feas

wx_impr = scipy.stats.wilcoxon(rating_diffs[rating_diffs['type']=='Importance']['rating_num'], zero_method="pratt")
print wx_impr

wx_incl = scipy.stats.wilcoxon(rating_diffs[rating_diffs['type']=='include']['rating_num'], zero_method="pratt")
print wx_incl



# --- Now use a mixed linear model

# importance
ratings_tmp = ratings_data_long[ratings_data_long['type']=='Importance']
md = smf.mixedlm("rating_num ~ goal_num_str", ratings_tmp, groups=ratings_tmp["user_num"])
mdf = md.fit()
print(mdf.summary())
#             Mixed Linear Model Regression Results
# ============================================================
# Model:              MixedLM  Dependent Variable:  rating_num
# No. Observations:   5572     Method:              REML
# No. Groups:         479      Scale:               0.3519
# Min. group size:    2        Likelihood:          -5433.7379
# Max. group size:    12       Converged:           Yes
# Mean group size:    11.6
# ------------------------------------------------------------
#                   Coef. Std.Err.    z    P>|z| [0.025 0.975]
# ------------------------------------------------------------
# Intercept         3.988    0.024 168.259 0.000  3.941  4.034
# goal_num_str[T.2] 0.079    0.022   3.531 0.000  0.035  0.122
# goal_num_str[T.3] 0.189    0.023   8.308 0.000  0.144  0.234
# goal_num_str[T.4] 0.406    0.022  18.215 0.000  0.362  0.450
# groups RE         0.150    0.021
# ============================================================

# Feasibility
ratings_tmp = ratings_data_long[ratings_data_long['type']=='Feasibility']
md = smf.mixedlm("rating_num ~ goal_num_str", ratings_tmp, groups=ratings_tmp["user_num"])
mdf = md.fit()
print(mdf.summary())
#            Mixed Linear Model Regression Results
# ===========================================================
# Model:             MixedLM  Dependent Variable:  rating_num
# No. Observations:  5690     Method:              REML
# No. Groups:        479      Scale:               1.0192
# Min. group size:   3        Likelihood:          -8509.2252
# Max. group size:   12       Converged:           Yes
# Mean group size:   11.9
# -----------------------------------------------------------
#                   Coef. Std.Err.   z    P>|z| [0.025 0.975]
# -----------------------------------------------------------
# Intercept         3.549    0.037 95.149 0.000  3.476  3.623
# goal_num_str[T.2] 0.187    0.038  4.946 0.000  0.113  0.262
# goal_num_str[T.3] 0.386    0.038 10.197 0.000  0.312  0.460
# goal_num_str[T.4] 0.840    0.038 22.199 0.000  0.766  0.914
# groups RE         0.323    0.027
# ===========================================================

# include
ratings_tmp = ratings_data_long[ratings_data_long['type']=='include']
md = smf.mixedlm("rating_num ~ goal_num_str", ratings_tmp, groups=ratings_tmp["user_num"])
mdf = md.fit()
print(mdf.summary())
#             Mixed Linear Model Regression Results
# ============================================================
# Model:              MixedLM  Dependent Variable:  rating_num
# No. Observations:   5708     Method:              REML
# No. Groups:         479      Scale:               0.1403
# Min. group size:    3        Likelihood:          -2887.1117
# Max. group size:    12       Converged:           Yes
# Mean group size:    11.9
# ------------------------------------------------------------
#                   Coef. Std.Err.    z    P>|z| [0.025 0.975]
# ------------------------------------------------------------
# Intercept         1.604    0.014 114.769 0.000  1.577  1.631
# goal_num_str[T.2] 0.090    0.014   6.409 0.000  0.062  0.117
# goal_num_str[T.3] 0.121    0.014   8.635 0.000  0.094  0.149
# goal_num_str[T.4] 0.292    0.014  20.783 0.000  0.264  0.319
# groups RE         0.046    0.010
# ============================================================

# try plotting these...

# ratings_tmp = ratings_data_long[ratings_data_long['type']=='include']
# ratings_tmp['rating_num'].hist(by=ratings_data_long['goal_num_str'],normed=True)
# weights_rec = np.ones_like(data_rec) / float(len(data_rec))
# # plt.hist(data_rec,weights=weights_rec, label='goal-related')
#
# plt.hist([data_self,data_rec], weights = [weights_self,weights_rec], label=['self-selected','goal-related'], bins=[0,1.5,3])




########################################################################################################################
### Question 2:
### What are the selected health goals?
########################################################################################################################




# Answer.HealthGoals

goals_data = mturk_data['Answer.HealthGoals']

# split the goals into 3
goals_split = goals_data.str.split("|", expand = True).reset_index()
goals_split.columns = ['user_num', 'goal1', 'goal2', 'goal3']

# first look at all the goals...
goals_long = goals_split[:][[ 'goal1', 'goal2', 'goal3']].stack().reset_index()
goals_long.columns = ['user_num', 'goal_num', 'goal']

# what is the frequency of each goal?
goals_long['goal'].value_counts()

# results:
# physicallyFit         388
# healthyEating         265
# nutritionAndWeight    186
# mentalHealth          133
# healthyAging           88
# backPain               54
# oralHealth             37
# foodSafety             35
# reduceAlcohol          30
# sexualHealth           25
# visionHearing          23
# learnDiabetes          22
# tobaccoControl         20
# injuryPrevention       19
# careOfDiabetes         17
# cancer                 16
# strokePrevention       14
# infectiousDiseases     12
# heartDisease            9
# learnImmunizations      7
# osteoporosis            7
# preventingSTDs          4
# violencePrevention      1

# plot this -- normalized by the number of responses
fig, ax = plt.subplots()
(goals_long['goal'].value_counts()/500).plot(ax=ax, kind='barh')
# goals_long['goal'].value_counts().plot(ax=ax, kind='barh')
ax.invert_yaxis()  # labels read top-to-bottom
plt.title('Fraction of responses that selected each goal')
plt.tight_layout()

# what are the predictors of selecting each goal?...



# mturk_short =

# --- article text (json) ---

with open(os.path.join(data_dir,topic_json)) as f:
    topic_data = json.load(f)

resource_data = topic_data['Result']['Resources']['Resource']

# now make a list of all tools/topics, including title/id

resource_list = []
for resource in resource_data:
    resource_list.append({k:resource[k] for k in
                               ['Title','Type','Id','Categories']})

id_list = [r['Id'] for r in resource_data]

# shoot, these IDs are useless...
print "There are %d unique IDs, and %d total resources." % (len(np.unique(id_list)), len(id_list))

# re-assign new IDs -- these are unique
for i,r in enumerate(resource_list):
    r['UID'] = i

# function to extract string-categories
# ASSUMES THE CATEGORIES DON'T HAVE COMMAS
sep_cat = lambda cat_str: string.replace(cat_str, ' ','').split(',') if cat_str is not None else ['None']

# add categories
for r in resource_list:
    r['Category_list'] = sep_cat(r['Categories'])

# get all the unique categories, assign them IDs
cat_list_nested = np.unique([r['Category_list'] for r in resource_list])
cat_list = [ci for c in cat_list_nested for ci in c] # un-nest
cat_dict = {cat_list[i]:i for i,c in enumerate(cat_list)}

# label the category numbers
for r in resource_list:
    r['Category_labels'] = [cat_dict[cat_str] for cat_str in r['Category_list']]

# how often do labels appear?
for key,val in cat_dict.iteritems():
    print "%s: count = %d" % (key, sum([1 for r in resource_list if val in r['Category_labels']]))

# frequency of categories of article, by demographic group...