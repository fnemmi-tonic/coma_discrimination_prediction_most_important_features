# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:28:50 2018

@author: federico nemmi
"""
import os
os.chdir("/home/zipat/Documents/Python Scripts/multiparametric_patrice/group_discrimination")
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.svm import LinearSVC
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt
from helper_functions_mean_ci import mean_confidence_interval
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle





whole_data = pd.read_table("../FC_SC_PCC_MPC_Cingulum_v17072018_SubRegions.csv", decimal = ",", sep = ",")


outcome = whole_data.loc[:,"Group"].values



num_data = whole_data.iloc[:,5:].values
num_date = scale(num_data)
colnames = whole_data.columns[5:]


random_state = 12072018

subject_id = np.arange(num_data.shape[0])



n_reps_true = 100
n_reps_shuff = 1000


k_best = [.9, .8, .7, .6, .5, .4, .3, .2, .1]
outer_corr_coeff = {}
outer_acc = {}

    
preds = list()
oob_outcomes = list()
acc = list()
acc_shuffled = list()
k_selected = []
selected_features = []

for n in np.arange(n_reps_true):
    print("Advancement {}%".format(((n+1)/n_reps_true)*100))
    X_res, Y_res, subj_id_res = resample(num_data, outcome, subject_id, random_state = random_state + n)
    oob_subjects = [s for s in subject_id if s not in subj_id_res]
    oob_outcome = outcome[oob_subjects]
    oob_features = num_data[oob_subjects, :]
    grid_search_results = list()
    models = list()
    
    rfk = StratifiedKFold(n_splits=10 , random_state = 100)
    
    cv_k_best = []
    selectors = []
    
    for val in k_best:
        cv_acc = []
        selector = SelectKBest(k = round(len(colnames) * val))
        
        for train, test in rfk.split(X_res, Y_res):
            X_res_int = X_res[train,:]
            Y_res_int = Y_res[train]
            selector.fit(X_res_int, Y_res_int.ravel())
            X_res_selected = selector.transform(X_res_int)
            clf = LinearSVC()
            clf.fit(X_res_selected, Y_res_int.ravel())
            inner_preds = clf.predict(selector.transform(X_res[test,:]))
            cv_acc.append(((((inner_preds[Y_res[test] == 1] == np.transpose(Y_res[test][Y_res[test] == 1])) * 1).mean() + ((inner_preds[Y_res[test] == 2] == np.transpose(Y_res[test][Y_res[test] == 2])) * 1).mean()) / 2))
        cv_k_best.append(np.array(cv_acc).mean())
        selectors.append(selector)
    
    best_selector = np.array(selectors)[np.array(cv_k_best).argsort()][-1]
    best_selector.fit(X_res, Y_res.ravel())
    k_selected.append(np.array(k_best)[np.array(cv_k_best).argsort()][-1])
    selected_features.append(colnames[best_selector.get_support()].tolist())
    #for g in ParameterGrid(grid):
    #    SVRmod = SVR(**g)
    #    SVRmod.fit(X_res, Y_res)
    #    Rsq = SVRmod.score(X_res, Y_res)
    #    grid_search_results.append(Rsq)
    #    models.append(SVRmod)
    X_res_selected_outern = best_selector.transform(X_res)    
    clf_outern = LinearSVC()
    
    clf_outern.fit(X_res_selected_outern, Y_res)
    
    oob_pred = clf_outern.predict(best_selector.transform(oob_features))
    
        
    preds.append(oob_pred)
    
    
    oob_outcomes.append(oob_outcome)
    
       
    acc.append((((oob_outcome[oob_outcome == 1] == oob_pred[oob_outcome == 1]) * 1).mean() + ((oob_outcome[oob_outcome == 2] == oob_pred[oob_outcome == 2]) * 1).mean())/2)

    





for n in np.arange(n_reps_shuff):
    print("Advancement {}%".format(((n+1)/n_reps_shuff)*100))
    X_res, Y_res, subj_id_res = resample(num_data, outcome, subject_id, random_state = random_state + n)
    oob_subjects = [s for s in subject_id if s not in subj_id_res]
    oob_outcome = outcome[oob_subjects]
    oob_features = num_data[oob_subjects, :]
    grid_search_results = list()
    models = list()
    X_res = shuffle(X_res, random_state = random_state + n)
    rfk = StratifiedKFold(n_splits = 10 , random_state = 100)
    
    cv_k_best = []
    selectors = []
    
    for val in k_best:
        cv_acc = []
        selector = SelectKBest(k = round(len(colnames) * val))
        
        for train, test in rfk.split(X_res, Y_res):
            X_res_int = X_res[train,:]
            Y_res_int = Y_res[train]
            selector.fit(X_res_int, Y_res_int.ravel())
            X_res_selected = selector.transform(X_res_int)
            clf = LinearSVC()
            clf.fit(X_res_selected, Y_res_int.ravel())
            inner_preds = clf.predict(selector.transform(X_res[test,:]))
            cv_acc.append(((((inner_preds[Y_res[test] == 1] == np.transpose(Y_res[test][Y_res[test] == 1])) * 1).mean() + ((inner_preds[Y_res[test] == 2] == np.transpose(Y_res[test][Y_res[test] == 2])) * 1).mean()) / 2))
        cv_k_best.append(np.array(cv_acc).mean())
        selectors.append(selector)
    
    best_selector = np.array(selectors)[np.array(cv_k_best).argsort()][-1]
    best_selector.fit(X_res, Y_res.ravel())
    
    
    X_res_selected_outern = best_selector.transform(X_res)
    
    clf_outern = LinearSVC()
    
    clf_outern.fit(X_res_selected_outern, Y_res)
    
    oob_pred = clf_outern.predict(best_selector.transform(oob_features))
        
    shuffled_pred = clf_outern.predict(best_selector.transform(oob_features))
    
    acc_shuffled.append((((oob_outcome[oob_outcome == 1] == shuffled_pred[oob_outcome == 1]) * 1).mean() + ((oob_outcome[oob_outcome == 2] == shuffled_pred[oob_outcome == 2]) * 1).mean())/2)


import pickle

with (open("group_LinearSVC_select_k_best_independent_balanced_inner.pkl", "wb")) as f:
    pickle.dump([acc, acc_shuffled, preds, oob_outcomes, selected_features, k_selected], f)

import pickle

with (open("group_LinearSVC_select_k_best_independent_balanced_inner.pkl", "rb")) as f:
    tt = pickle.load(f)


selected_features = tt[4]

selected_features = [[x.replace("DP", "RD") for x in ll] for ll in selected_features]
selected_features = [[x.replace("FC_", "FC ") for x in ll] for ll in selected_features]
selected_features = [[x.replace("_", " ") for x in ll] for ll in selected_features]
selected_features = [[x.replace("PCC", " PMC") for x in ll] for ll in selected_features]

mean_confidence_interval(acc)
mean_confidence_interval(acc_shuffled)
((acc_shuffled > np.mean(acc))*1).mean()

lab_and_counts = np.unique(np.hstack(np.array(selected_features).ravel()), return_counts = True)
features_bigger_than_thr = [el for el, val in zip(lab_and_counts[0], lab_and_counts[1]) if val > n_reps_true * .50]
features_bigger_than_thr = features_bigger_than_thr[:15]
counts_bigger_than_thr = [val for val in lab_and_counts[1] if val > n_reps_true * .50]
counts_bigger_than_thr = counts_bigger_than_thr[:15]
plt.rcParams.update({'font.size': 30})
sorted_features = np.array(features_bigger_than_thr)[np.array(counts_bigger_than_thr).argsort()[::-1]]
counts_bigger_than_thr.sort(reverse = True)
plt.figure(num = 1, figsize = (18,18), dpi = 180)
plt.bar(sorted_features, counts_bigger_than_thr)
plt.xticks(rotation=75, horizontalalignment="right")
plt.tight_layout()
plt.title("Group discrimination")
plt.savefig("group_most_selected_features.tiff")

plt.hist(k_selected)       