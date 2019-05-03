# Duncan McElfresh
# June 2018

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines import NelsonAalenFitter
from lifelines import KaplanMeierFitter

from lifelines.utils import k_fold_cross_validation

from lifelines.plotting import plot_lifetimes
'''
Survival analysis of two time parameters:
- time-in-housing (first epoch only)
- waiting time

both of these are censored

I'll use a Cox regression analysis to identify the predictors of these times
'''

# read the person and resource df
data_dir = '/Users/duncan/research/homeless_youth/data_products'
data_dir = '/Users/duncan/research/homeless_youth/data_products'
# data_dir = '/home/dmcelfre/homeless_youth/data_products'

# p_filename = 'person_socal_metro.csv'
# h_filename = 'resource_socal_metro.csv'
p_filename = 'person_all_data.csv'

person_file = os.path.join(data_dir, p_filename)

person_df = pd.read_csv(person_file, index_col=0)

# cast the arrival time and housed time as dates
person_df['ARRIVAL_DT'] = pd.to_datetime(person_df['ARRIVAL_DATE'])
person_df['EXIT_DT'] = pd.to_datetime(person_df['DATE_OF_EXIT_HOMELESSNESS'])
person_df['EXIT_HOUSING_DT'] = pd.to_datetime(person_df['DATE_OF_EXIT_FROM_HOUSING'])

# these are the valid exit types
exit_types =    ['TYPE_OF_EXIT_Family',
                'TYPE_OF_EXIT_PSH',
                'TYPE_OF_EXIT_RRH',
                'TYPE_OF_EXIT_Self Resolve']

# create a df for only housing exits
df_housing_time = person_df.loc[person_df[exit_types].sum(1) > 0].copy()
df_not_housed = person_df.loc[person_df[exit_types].sum(1) == 0].copy()
df_not_housed['DAYS_IN_HOUSING'] = 0

# ensure that all agents have an exit date
if np.sum(df_housing_time['EXIT_DT'].isnull()) > 0:
    raise Warning("some exited agents have no exit date")

# calculate time-in-housing:
# - A) if agent has an exit date, AND still housed, they have been housed until June 1 2017
# - if agent has an exit date, and has a 'DATE_OF_EXIT_FROM_HOUSING', then this is their exit date

last_date = pd.to_datetime('06-01-2017')

# add a censored variable
df_housing_time['CENSORED']=False

# - A) if agent has an exit date, AND still housed, they have been housed until June 1 2017
df_housing_time.loc[df_housing_time['STILL_HOUSED']==1,'LAST_HOUSING_DATE'] = last_date
df_housing_time.loc[df_housing_time['STILL_HOUSED']==1,'CENSORED'] = True

# - if agent has an exit date, and has a 'DATE_OF_EXIT_FROM_HOUSING', then this is their exit date
df_housing_time.loc[~df_housing_time['EXIT_HOUSING_DT'].isnull(),'LAST_HOUSING_DATE'] =  \
    df_housing_time.loc[~df_housing_time['EXIT_HOUSING_DT'].isnull()]['EXIT_HOUSING_DT']


# ensure all agents have a last housing date
if np.sum(df_housing_time['LAST_HOUSING_DATE'].isnull()) > 0:
    raise Warning("some exited agents have no last-housing date")

# calculate time in housing
df_housing_time['TIME_IN_HOUSING'] = df_housing_time['LAST_HOUSING_DATE'] - df_housing_time['EXIT_DT']
df_housing_time['DAYS_IN_HOUSING'] = df_housing_time['TIME_IN_HOUSING'].apply(lambda x: x.days)


# is the 'death' observed?
df_housing_time['OBSERVED'] = ~df_housing_time['CENSORED']

# if there are negative days in housing, remove these rows
# ---- THIS IS THE FINAL DF FOR HOUSING TIME
df_housing_time_final = df_housing_time.loc[df_housing_time['DAYS_IN_HOUSING'] >= 0].copy()


df_rrh = df_housing_time_final.loc[df_housing_time_final['TYPE_OF_EXIT_RRH'] == 1]
df_psh = df_housing_time_final.loc[df_housing_time_final['TYPE_OF_EXIT_PSH']==1]
df_self = df_housing_time_final.loc[df_housing_time_final['TYPE_OF_EXIT_Self Resolve'] == 1]
df_fam = df_housing_time_final.loc[df_housing_time_final['TYPE_OF_EXIT_Family'] == 1]


# Cox regression analaysis of time in housing

img_dir = '/Users/duncan/research/homeless_youth/survival_analysis/Cox_analysis_new'

# which df to work on?
# df = df_rrh.copy()
# df_name = 'RRH'

# df = df_psh.copy()
# df_name = 'PSH'

# df = df_fam.copy()
# df_name = 'FAM'

df = df_self.copy()
df_name = 'SELF'

# initialize the Cox model
cph = CoxPHFitter(penalizer=0.2)

# encode race as (black, white, latino, other).
df.loc[:,'RACE_ETHNICITY_other'] = 1 - ( (df['RACE_ETHNICITY_White']==1) | (df['RACE_ETHNICITY_Black']==1) |  (df['RACE_ETHNICITY_Hispanic']==1) )

# ensure race variables are mutually exclusive
# BLACK IS CONTROL
assert 1 == (df['RACE_ETHNICITY_other'] + df['RACE_ETHNICITY_Black'] + df['RACE_ETHNICITY_White'] + df['RACE_ETHNICITY_Hispanic']).min()
assert 1 == (df['RACE_ETHNICITY_other'] + df['RACE_ETHNICITY_Black'] + df['RACE_ETHNICITY_White'] + df['RACE_ETHNICITY_Hispanic']).max()

# ensure that "where housed" variables are mutually exclusive
# CAR IS CONTROL
assert 1 == (df['WHERE_SLEEP_car'] + df['WHERE_SLEEP_couch'] + df['WHERE_SLEEP_outdoors'] + df['WHERE_SLEEP_shelter'] + df['WHERE_SLEEP_transitional housing']).min()
assert 1 == (df['WHERE_SLEEP_car'] + df['WHERE_SLEEP_couch'] + df['WHERE_SLEEP_outdoors'] + df['WHERE_SLEEP_shelter'] + df['WHERE_SLEEP_transitional housing']).max()

# ensure that community type is mutually exclusive
# RURAL IS CONTROL
assert 1 == (df['TYPE_OF_COMMUNITY_suburban'] + df['TYPE_OF_COMMUNITY_rural'] + df['TYPE_OF_COMMUNITY_urban'] ).min()
assert 1 == (df['TYPE_OF_COMMUNITY_suburban'] + df['TYPE_OF_COMMUNITY_rural'] + df['TYPE_OF_COMMUNITY_urban'] ).max()

# ensure that gender is mutually exclusive
# MALE IS CONTROL
assert 1 == (df['SELF_IDENTIFIED_GENDER_f'] + df['SELF_IDENTIFIED_GENDER_m']).min()
assert 1 == (df['SELF_IDENTIFIED_GENDER_f'] + df['SELF_IDENTIFIED_GENDER_m']).max()


# 1) test all dependent variables separately
#   - keep all with p<0.1
# 2) add ALL remaining dependent vars from (1) to a model, WITH controls
#   - iteratively remove one feature (the highest p-val) until ALL are p<0.05

# dependent variables
dep_vars = ['DAYS_IN_HOUSING','OBSERVED']


# control variables
control_vars = ['17_OR_YOUNGER',
                'IDENTIFIES_AS_LGBTQQI2',
                'RACE_ETHNICITY_White',
                # 'RACE_ETHNICITY_Black', # this is the control
                'RACE_ETHNICITY_Hispanic',
                'RACE_ETHNICITY_other',   # only do white & non-white
                'SELF_IDENTIFIED_GENDER_f'
                # 'SELF_IDENTIFIED_GENDER_m' # this is control
                ]

# independent variables to include

indep_vars = [
'DAYS_SINCE_PSH',
'NUM_TIMES_HOMELESS_LAST_3_YEARS',
'FINAL_ACUITY_SCORE',
'WHERE_SLEEP_car', # this is control # also, keep where sleep together (unless ALL are insignificant)
'WHERE_SLEEP_couch', # REMOVE FOR PSH ONLY
'WHERE_SLEEP_outdoors',
'WHERE_SLEEP_shelter',
'WHERE_SLEEP_transitional housing',
# 'WHERE_SLEEP_nan', ## REMOVED BECAUSE NOT ENOUGH INSTANCES
# 'TYPE_OF_COMMUNITY_rural',# this is control # also, keep type of community together (unless ALL are insignificant; rural is reference)
'TYPE_OF_COMMUNITY_suburban',
'TYPE_OF_COMMUNITY_urban',
# 'TYPE_OF_COMMUNITY_nan', ## REMOVED BECAUSE NOT ENOUGH INSTANCES
'BEEN_ATTACKED_SINCE_HOMELESS',
'TRIED_TO_HARM_SELF_OR_OTHERS',
'LEGAL_ISSUES',
'INCARCERATED_BEFORE_18_YRS',
'FORCED_TO_DO_THINGS',
'DO_RISKY_THINGS',
'SOMEONE_THINKS_U_OWE_MONEY',
'GET_ANY_MONEY_LEGAL_OR_OTHER',
'NICE_ACTIVITIES_MAKE_YOU_HAPPY',
'ABLE_SATISFY_BASIC_NEEDS',
'BECAUSE_RAN_AWAY',
'BECAUSE_DIFFERENCE_CULTURAL_BELIEFS',
'BECAUSE_KICKED_OUT',
'BECAUSE_GENDER_ID',
'BECAUSE_VIOLENCE_FAMILY',
'BECAUSE_ABUSE',
'HAD_TO_LEAVE_BECAUSE_PHYSICAL_HEALTH',
'CHRONIC_HEALTH_LKSLH',
'INTERESTED_PROGRAM_HIV_AIDS',
'PHYSICAL_DISABILITIES',
'AVOID_MEDICAL_HELP',
'PREGNANT',
'KICKED_OUT_DRINKING_DRUGS',
'DIFFICULT_DRINKING_DRUG',
'MARIJUANA_12_OR_YOUNGER',
'MENTAL_HEALTH_ISSUE',
'PAST_HEAD_INJURY',
'DISABILITY',
'MENTAL_HEALTH_NEED_HELP',
'MEDICATIONS_NOT_TAKING',
'MEDICATIONS_NOT_TAKING_OR_SELLING',
'HEALTHCARE_AT_ED_ER',
'TOOK_AMBULANCE',
'USED_CRISIS_SERVICE',
'TALKED_TO_POLICE',
'BEEN_IN_JAIL'
    ]

# type_of_community vars
community_vars = [ # keep type of community together (unless ALL are insignificant; rural is reference)
'TYPE_OF_COMMUNITY_suburban',
'TYPE_OF_COMMUNITY_urban']
# where_sleep vars
where_sleep_vars = [ # keep where sleep together (unless ALL are insignificant)
'WHERE_SLEEP_couch',
'WHERE_SLEEP_outdoors',
'WHERE_SLEEP_shelter',
'WHERE_SLEEP_transitional housing']



# 1) test all features separately; keep those with p<0.1

# build a dict with the p-vals for each independent var
p_val_dict = {}

p_val_thresh_1 = 0.1
for var in indep_vars:
    df_tmp = df.loc[:, [var] + dep_vars]
    # scores = k_fold_cross_validation(cph_psh, df_tmp_psh, 'DAYS_IN_HOUSING', event_col='OBSERVED', k=3)
    try:
        cph.fit(df_tmp, duration_col='DAYS_IN_HOUSING', event_col='OBSERVED')
    except:
        pass
    # cph.print_summary()

    # IF p-value is <thresh, add it to the dict
    if float(cph.summary['p'])< p_val_thresh_1:
        p_val_dict[var] = float(cph.summary['p'])

# ------ RESULTS OF PART 1 ------
for k,v in p_val_dict.iteritems():
    print "'%s':%7.5e," % (k,v)

# --  RRH --
# this is the result:
p_val_dict = {
'BECAUSE_KICKED_OUT': 1.74424e-02,
'DO_RISKY_THINGS': 7.89532e-02,
'BEEN_IN_JAIL': 1.59938e-02,
'LEGAL_ISSUES': 1.68252e-04,
'DAYS_SINCE_PSH': 1.30368e-02,
'USED_CRISIS_SERVICE': 7.04748e-16,
'BECAUSE_RAN_AWAY': 1.56274e-04,
'WHERE_SLEEP_transitional housing': 8.82103e-75, # because where_sleep is significant, we keep all other where_sleep variables
'TALKED_TO_POLICE': 5.10635e-03,
'FINAL_ACUITY_SCORE': 4.42274e-06,
'MENTAL_HEALTH_ISSUE': 1.53508e-02,
'NUM_TIMES_HOMELESS_LAST_3_YEARS': 1.42129e-10,
'KICKED_OUT_DRINKING_DRUGS': 2.82133e-02,
'WHERE_SLEEP_couch': 1.53203e-18,
'WHERE_SLEEP_shelter': 2.53166e-72,
'BECAUSE_ABUSE': 4.11577e-02,
'NICE_ACTIVITIES_MAKE_YOU_HAPPY': 6.35971e-12
}

# -- PSH --
p_val_dict = {
'BECAUSE_KICKED_OUT':2.47538e-03,
'BECAUSE_GENDER_ID':2.70872e-02,
'AVOID_MEDICAL_HELP':1.24445e-02,
'BECAUSE_VIOLENCE_FAMILY':3.14590e-02,
'TOOK_AMBULANCE':3.22336e-02,
'WHERE_SLEEP_outdoors':2.99105e-04, # ADD ALL WHERE SLEEP
'FINAL_ACUITY_SCORE':2.84164e-08,
'MARIJUANA_12_OR_YOUNGER':1.78672e-02,
'WHERE_SLEEP_shelter':4.19528e-04,
'BECAUSE_ABUSE':1.42516e-03
}

# now with white vs. non-white (SAME RESULT
# 'BECAUSE_KICKED_OUT':2.47538e-03,
# 'BECAUSE_GENDER_ID':2.70872e-02,
# 'AVOID_MEDICAL_HELP':1.24445e-02,
# 'BECAUSE_VIOLENCE_FAMILY':3.14590e-02,
# 'TOOK_AMBULANCE':3.22336e-02,
# 'WHERE_SLEEP_outdoors':2.99105e-04,
# 'FINAL_ACUITY_SCORE':2.84164e-08,
# 'MARIJUANA_12_OR_YOUNGER':1.78672e-02,
# 'WHERE_SLEEP_shelter':4.19528e-04,
# 'BECAUSE_ABUSE':1.42516e-03,

# -- FAM --
p_val_dict = {
'BECAUSE_KICKED_OUT':8.58423e-02,
'TRIED_TO_HARM_SELF_OR_OTHERS':9.71343e-07,
'ABLE_SATISFY_BASIC_NEEDS':6.59248e-03,
'BECAUSE_RAN_AWAY':1.81611e-02,
'BECAUSE_GENDER_ID':6.08767e-04,
'BECAUSE_DIFFERENCE_CULTURAL_BELIEFS':9.27461e-02,
'LEGAL_ISSUES':8.80073e-03,
'PREGNANT':7.07133e-06,
'DAYS_SINCE_PSH':9.21172e-04,
'DISABILITY':4.49235e-02,
'INCARCERATED_BEFORE_18_YRS':1.79297e-03,
'FORCED_TO_DO_THINGS':1.36141e-02,
'MEDICATIONS_NOT_TAKING_OR_SELLING':7.88092e-03,
'DO_RISKY_THINGS':2.81350e-02,
'WHERE_SLEEP_shelter':1.99872e-06,
'NICE_ACTIVITIES_MAKE_YOU_HAPPY':8.46173e-02,
'WHERE_SLEEP_car':8.01395e-06,
'BECAUSE_VIOLENCE_FAMILY':1.75184e-04,
'USED_CRISIS_SERVICE':2.54384e-03,
'WHERE_SLEEP_outdoors':1.07624e-03,
'TALKED_TO_POLICE':7.29973e-07,
'MEDICATIONS_NOT_TAKING':3.07619e-02,
'PAST_HEAD_INJURY':1.19149e-02,
'BEEN_ATTACKED_SINCE_HOMELESS':7.80892e-08,
'BECAUSE_ABUSE':4.04627e-02,
'HEALTHCARE_AT_ED_ER':3.77497e-04,
'TOOK_AMBULANCE':1.71719e-03,
'GET_ANY_MONEY_LEGAL_OR_OTHER':1.08406e-03,
'DIFFICULT_DRINKING_DRUG':5.36864e-02,
'FINAL_ACUITY_SCORE':1.33052e-55,
'MENTAL_HEALTH_NEED_HELP':7.28252e-02,
'MARIJUANA_12_OR_YOUNGER':1.62083e-02,
'BEEN_IN_JAIL':7.78048e-05
}

# -- SELF --
p_val_dict = {
'GET_ANY_MONEY_LEGAL_OR_OTHER':4.97729e-05,
'BEEN_IN_JAIL':6.25526e-02,
'TYPE_OF_COMMUNITY_suburban':3.89170e-03,
'BECAUSE_VIOLENCE_FAMILY':5.95102e-04,
'MEDICATIONS_NOT_TAKING':7.55417e-02,
'WHERE_SLEEP_outdoors':4.94540e-02,
'TYPE_OF_COMMUNITY_urban':3.60773e-03,
'WHERE_SLEEP_transitional housing':4.74989e-04,
'TALKED_TO_POLICE':3.78519e-02,
'FINAL_ACUITY_SCORE':2.25858e-49,
'MENTAL_HEALTH_ISSUE':7.61083e-02,
'NUM_TIMES_HOMELESS_LAST_3_YEARS':2.00413e-03,
'MEDICATIONS_NOT_TAKING_OR_SELLING':4.02849e-02,
'BEEN_ATTACKED_SINCE_HOMELESS':1.02322e-03,
'BECAUSE_RAN_AWAY':2.86421e-04,
'WHERE_SLEEP_shelter':9.92259e-02,
'BECAUSE_ABUSE':1.58256e-12,
'BECAUSE_GENDER_ID':7.77303e-02
}

# --------------- end of part 1 result ----------------

# create the complete variable list
part_2_vars = p_val_dict.keys()

# add the where_sleep and community vars, if any is included.
if len(set(where_sleep_vars).intersection(part_2_vars)) > 0:
    part_2_vars = list(set(part_2_vars + where_sleep_vars))

if len(set(community_vars).intersection(part_2_vars)) > 0:
    part_2_vars = list(set(part_2_vars + community_vars))

# now add all variables, and remove the highest-p var until all are within the threshold
# INCLUDE CONTROL VARS NOWs
p_val_thresh_2 = 0.05

# check for nan values:
# df.loc[:,part_2_vars].isna().sum()

done = False
while not done:
    df_tmp = df.loc[:, part_2_vars + dep_vars + control_vars]
    # scores = k_fold_cross_validation(cph_psh, df_tmp_psh, 'DAYS_IN_HOUSING', event_col='OBSERVED', k=3)

    cph.fit(df_tmp, duration_col='DAYS_IN_HOUSING', event_col='OBSERVED') # ,step_size=0.05) # include this if convergence errors
    # cph.print_summary()

    # if all p-values are within threshold, we're done
    if cph.summary.loc[part_2_vars,'p'].max()< p_val_thresh_2:
        done = True
    else:
        # otherwise, remove the highest-p feature
        remove_feat = cph.summary.loc[part_2_vars,'p'].idxmax()
        part_2_vars.remove(remove_feat)

    if len(part_2_vars) == 0:
        done = True
        print "WARNING: REMOVED ALL VARIABLES"

cph.print_summary()

# # RESULTS FOR RRH
# n=2883, number of events=674
#                                     coef  exp(coef)  se(coef)       z      p  lower 0.95  upper 0.95
# WHERE_SLEEP_couch                 1.0337     2.8113    0.2285  4.5235 0.0000      0.5858      1.4815  ***
# WHERE_SLEEP_transitional housing  1.1624     3.1976    0.1894  6.1367 0.0000      0.7911      1.5336  ***
# FINAL_ACUITY_SCORE                0.1735     1.1894    0.0356  4.8711 0.0000      0.1037      0.2433  ***
# WHERE_SLEEP_shelter              -0.5045     0.6038    0.1856 -2.7183 0.0066     -0.8682     -0.1407   **
# WHERE_SLEEP_outdoors             -0.3074     0.7353    0.3050 -1.0080 0.3135     -0.9052      0.2903
# 17_OR_YOUNGER                     0.3182     1.3747    0.0875  3.6349 0.0003      0.1466      0.4898  ***
# IDENTIFIES_AS_LGBTQQI2           -0.1518     0.8592    0.0886 -1.7127 0.0868     -0.3255      0.0219    .
# RACE_ETHNICITY_White             -0.2457     0.7822    0.0899 -2.7319 0.0063     -0.4219     -0.0694   **
# RACE_ETHNICITY_Hispanic           0.0182     1.0184    0.1102  0.1654 0.8686     -0.1977      0.2341
# RACE_ETHNICITY_other              0.0129     1.0130    0.1723  0.0750 0.9402     -0.3248      0.3506
# SELF_IDENTIFIED_GENDER_f         -0.1137     0.8925    0.0994 -1.1442 0.2525     -0.3085      0.0811
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# Concordance = 0.705

# RESULTS FOR PSH
# n=577, number of events=103
#                             coef  exp(coef)  se(coef)       z      p  lower 0.95  upper 0.95
# FINAL_ACUITY_SCORE        0.3219     1.3798    0.0606  5.3163 0.0000      0.2032      0.4406  ***
# 17_OR_YOUNGER            -0.2374     0.7887    0.3010 -0.7885 0.4304     -0.8273      0.3526
# IDENTIFIES_AS_LGBTQQI2    0.1057     1.1115    0.2032  0.5201 0.6030     -0.2925      0.5038
# RACE_ETHNICITY_White     -0.0226     0.9777    0.2213 -0.1020 0.9188     -0.4563      0.4111
# RACE_ETHNICITY_Hispanic   0.2618     1.2992    0.3227  0.8113 0.4172     -0.3707      0.8942
# RACE_ETHNICITY_other      0.2280     1.2561    0.3921  0.5814 0.5610     -0.5406      0.9966
# SELF_IDENTIFIED_GENDER_f  0.2184     1.2441    0.2447  0.8927 0.3720     -0.2612      0.6980
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# Concordance = 0.657
#
# ---- NOW WITH RACE as WHITE vs NON-WHITE
# n=577, number of events=103
#                             coef  exp(coef)  se(coef)       z      p  lower 0.95  upper 0.95
# FINAL_ACUITY_SCORE        0.3189     1.3756    0.0602  5.2977 0.0000      0.2009      0.4369  ***
# 17_OR_YOUNGER            -0.2193     0.8031    0.2999 -0.7313 0.4646     -0.8072      0.3685
# IDENTIFIES_AS_LGBTQQI2    0.1027     1.1081    0.2028  0.5062 0.6127     -0.2949      0.5002
# RACE_ETHNICITY_White     -0.1033     0.9018    0.1993 -0.5186 0.6040     -0.4939      0.2872
# SELF_IDENTIFIED_GENDER_f  0.2332     1.2627    0.2425  0.9620 0.3361     -0.2420      0.7084
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# Concordance = 0.657
#
#
# RESULTS FOR FAM
# n=1259, number of events=253
#                             coef  exp(coef)  se(coef)       z      p  lower 0.95  upper 0.95
# BECAUSE_GENDER_ID         0.5425     1.7203    0.2266  2.3941 0.0167      0.0984      0.9866    *
# PREGNANT                  0.5666     1.7622    0.2166  2.6160 0.0089      0.1421      0.9911   **
# BECAUSE_ABUSE            -0.5128     0.5988    0.1875 -2.7341 0.0063     -0.8803     -0.1452   **
# FINAL_ACUITY_SCORE        0.4781     1.6131    0.0339 14.0898 0.0000      0.4116      0.5446  ***
# 17_OR_YOUNGER             0.3104     1.3640    0.1387  2.2390 0.0252      0.0387      0.5822    *
# IDENTIFIES_AS_LGBTQQI2   -0.2756     0.7591    0.1656 -1.6642 0.0961     -0.6002      0.0490    .
# RACE_ETHNICITY_White     -0.4972     0.6082    0.1503 -3.3073 0.0009     -0.7918     -0.2025  ***
# RACE_ETHNICITY_Hispanic  -0.0193     0.9809    0.2070 -0.0931 0.9258     -0.4249      0.3864
# RACE_ETHNICITY_other     -0.0959     0.9085    0.2512 -0.3820 0.7025     -0.5882      0.3963
# SELF_IDENTIFIED_GENDER_f  0.0551     1.0566    0.1434  0.3841 0.7009     -0.2259      0.3361
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# Concordance = 0.742

#
# RESULTS FOR SELF
# n=1145, number of events=272
#                                    coef  exp(coef)  se(coef)       z      p  lower 0.95  upper 0.95
# BECAUSE_GENDER_ID               -0.5453     0.5796    0.2518 -2.1659 0.0303     -1.0388     -0.0519    *
# FINAL_ACUITY_SCORE               0.7266     2.0681    0.0579 12.5479 0.0000      0.6131      0.8401  ***
# WHERE_SLEEP_outdoors            -1.9430     0.1433    0.8007 -2.4265 0.0152     -3.5124     -0.3736    *
# TYPE_OF_COMMUNITY_urban          0.3410     1.4063    0.1726  1.9754 0.0482      0.0027      0.6793    *
# NUM_TIMES_HOMELESS_LAST_3_YEARS -0.1087     0.8970    0.0472 -2.3033 0.0213     -0.2011     -0.0162    *
# TYPE_OF_COMMUNITY_suburban       0.5095     1.6644    0.1937  2.6297 0.0085      0.1298      0.8892   **
# 17_OR_YOUNGER                    1.6914     5.4270    0.1313 12.8808 0.0000      1.4340      1.9488  ***
# IDENTIFIES_AS_LGBTQQI2           0.4644     1.5910    0.1688  2.7507 0.0059      0.1335      0.7952   **
# RACE_ETHNICITY_White            -0.1780     0.8369    0.1498 -1.1885 0.2346     -0.4715      0.1155
# RACE_ETHNICITY_Hispanic         -0.4296     0.6508    0.2055 -2.0902 0.0366     -0.8324     -0.0268    *
# RACE_ETHNICITY_other             0.0741     1.0770    0.2804  0.2644 0.7915     -0.4754      0.6237
# SELF_IDENTIFIED_GENDER_f         0.1303     1.1391    0.1416  0.9202 0.3575     -0.1472      0.4077
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# Concordance = 0.812
#
# It looks like WHERE_SLEEP and COMMUNITY_TYPE are both significant, adding them back in:
# n=1145, number of events=272
#                                     coef  exp(coef)  se(coef)       z      p  lower 0.95  upper 0.95
# TYPE_OF_COMMUNITY_suburban        0.4897     1.6318    0.1940  2.5236 0.0116      0.1094      0.8700    *
# WHERE_SLEEP_outdoors             -2.7002     0.0672    0.9412 -2.8689 0.0041     -4.5448     -0.8555   **
# TYPE_OF_COMMUNITY_urban           0.3430     1.4092    0.1736  1.9763 0.0481      0.0028      0.6832    *
# WHERE_SLEEP_transitional housing -1.2896     0.2754    0.6975 -1.8488 0.0645     -2.6568      0.0775    .
# FINAL_ACUITY_SCORE                0.7007     2.0151    0.0600 11.6875 0.0000      0.5832      0.8182  ***
# NUM_TIMES_HOMELESS_LAST_3_YEARS  -0.0770     0.9259    0.0510 -1.5084 0.1315     -0.1770      0.0231
# WHERE_SLEEP_couch                -0.9372     0.3917    0.6733 -1.3919 0.1640     -2.2568      0.3825
# WHERE_SLEEP_shelter              -0.9074     0.4036    0.6451 -1.4065 0.1596     -2.1718      0.3571
# BECAUSE_GENDER_ID                -0.6136     0.5414    0.2568 -2.3895 0.0169     -1.1168     -0.1103    *
# 17_OR_YOUNGER                     1.7122     5.5410    0.1332 12.8536 0.0000      1.4511      1.9733  ***
# IDENTIFIES_AS_LGBTQQI2            0.4931     1.6374    0.1701  2.8982 0.0038      0.1596      0.8266   **
# RACE_ETHNICITY_White             -0.1898     0.8271    0.1503 -1.2623 0.2069     -0.4845      0.1049
# RACE_ETHNICITY_Hispanic          -0.4333     0.6484    0.2062 -2.1018 0.0356     -0.8373     -0.0292    *
# RACE_ETHNICITY_other              0.1023     1.1077    0.2809  0.3643 0.7156     -0.4482      0.6528
# SELF_IDENTIFIED_GENDER_f          0.1130     1.1196    0.1420  0.7960 0.4260     -0.1652      0.3913
# ---
# Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# Concordance = 0.813

# make plots of the significant variables

# by race
ax = cph.plot_covariate_groups('RACE_ETHNICITY_White',[1])
cph.plot_covariate_groups('RACE_ETHNICITY_Hispanic',[1], ax=ax)
cph.plot_covariate_groups('RACE_ETHNICITY_other',[1], ax=ax)
ax.set_title('Cox Regression model for '+df_name)
ax.lines[-1].remove()
ax.lines[-2].remove()
ax.legend()

ax.set_xlabel('Days in housing')
ax.set_ylabel('Probability of still being housed')
plt.savefig(os.path.join(img_dir,df_name+'_by_race.png'))

# by LGBTQ
ax = cph.plot_covariate_groups('IDENTIFIES_AS_LGBTQQI2',[1])
ax.set_title('Cox Regression model for '+ df_name)
plt.savefig(os.path.join(img_dir,df_name+'_by_LGBTQ.png'))

# by PREGNANT
ax = cph.plot_covariate_groups('PREGNANT', [1])
ax.set_title('Cox Regression model for ' + df_name)
plt.savefig(os.path.join(img_dir, df_name + '_by_PREGNANT.png'))

# by ABUSE
ax = cph.plot_covariate_groups('BECAUSE_ABUSE', [1])
ax.set_title('Cox Regression model for ' + df_name)
plt.savefig(os.path.join(img_dir, df_name + '_by_BY_ABUSE.png'))

# by 17-younger
ax = cph.plot_covariate_groups('17_OR_YOUNGER',[1])
ax.set_title('Cox Regression model for '+df_name)
plt.savefig(os.path.join(img_dir,df_name+'_by_17.png'))

# by where-sleep
ax = cph.plot_covariate_groups('WHERE_SLEEP_couch',[1])
cph.plot_covariate_groups('WHERE_SLEEP_transitional housing',[1], ax=ax)
cph.plot_covariate_groups('WHERE_SLEEP_shelter',[1], ax=ax)
cph.plot_covariate_groups('WHERE_SLEEP_outdoors',[1], ax=ax)
ax.set_title('Cox Regression model for '+df_name)
ax.lines[-1].remove()
ax.lines[-2].remove()
ax.lines[-3].remove()
ax.legend()
plt.savefig(os.path.join(img_dir,df_name+'_by_where_sleep.png'))


# by type of community
ax = cph.plot_covariate_groups('TYPE_OF_COMMUNITY_suburban',[1])
cph.plot_covariate_groups('TYPE_OF_COMMUNITY_urban',[1], ax=ax)
ax.set_title('Cox Regression model for '+df_name)
ax.lines[-1].remove()
ax.legend()
plt.savefig(os.path.join(img_dir,df_name+'_by_community_type.png'))


# by final acuity score
# by race
ax = cph.plot_covariate_groups('FINAL_ACUITY_SCORE',[0,5,7,12,17])
ax.set_title('Cox Regression model for '+df_name)
plt.savefig(os.path.join(img_dir,df_name+'_by_score.png'))

# by number times homeless
ax = cph.plot_covariate_groups('NUM_TIMES_HOMELESS_LAST_3_YEARS',[0,1,3,10,20])
ax.set_title('Cox Regression model for '+df_name)
plt.savefig(os.path.join(img_dir,df_name+'_by_num_times_homeless.png'))

