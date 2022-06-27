import numpy as np
import sys
from scipy.stats import loguniform
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import pandas as pd

from experimental_fitting import CurveFit
from experimental_setup import Experiment

if __name__ == '__main__':

    letter_dataset_id = 6
    breast_tissue_dataset_id = 1465
    blood_transfusion_id = 1464
    #Extra_trees_classifier_error
    sick_dataset_id = 38 # SKip
    ova_dataset_id = 1142 # doesnt work?

    #svc error
    mfeat_dataset_id = 971 #skip
    musk_dataset_id = 1116
    libras_dataset_id = 299
    drama_dataset_id = 273 # skip
    soil_dataset_id = 923
    yeast_dataset_id = 181
    telescope_dataset_id = 1120

    #SGDC error
    houses_dataset_id = 823
    jm_dataset_id = 1053
    space_dataset_id = 737
    fri_dataset_id = 715
    click_dataset_id = 1216 #Massive dataset (DO LATER)
    fri_ci_dataset_id = 910
    anal_dataset_id = 966 #doest work

    #Ridge
    credit_dataset_id = 1597 #doesnt work

    #Kneighbours
    credit_g_dataset_id = 31

    datasets = [737]
    # Number of splits
    n_splits = 5

    # Learners
    learners = [DecisionTreeClassifier(), DecisionTreeClassifier(), KNeighborsClassifier(), KNeighborsClassifier()]
    learners = [KNeighborsClassifier(), KNeighborsClassifier()]

    # Hyperparameter distribution/grid
    svc_distribution = {'C': [loguniform(1e0, 1e3)],
                        'gamma':[ loguniform(1e-4, 1e-3)],
                        'kernel': ['rbf'],
                        'class_weight': ['balanced', None],
                        'random_state' : [42]}

    # svc_distribution = {'C': [1,2,3,4,5,6,7,8,9,10,11],
    #                     'kernel': ['rbf', 'sigmoid'],
    #                     'class_weight': ['balanced', None],
    #                     'random_state': [42]}

    svc_distribution2 = {'kernel': ['linear'],
                        'random_state': [42]}

    dt_distribution = {'criterion': ['gini', 'entropy'],
                       'splitter': ['best', 'random'],
                       'max_depth': [None, 1, 2,3, 4,5, 6,7,8, 9,10],
                       'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12],
                       'random_state' :[42]}

    kneighbours_distribution = {'n_neighbors': [1,2,3,4,5, 6,7,8, 9,10,11,12],
                                'weights': ['distance', 'uniform']}

    lr_distribution = {'C': [1, 3, 5, 7, 9, 11],
                       'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                       'penalty': ['none', 'l2'],
                       'random_state' : [42]}

    extra_dt_distribution = {
                             'criterion': ['gini', 'entropy'],
                             'max_depth': [None, 2,4,6,8],
                             'min_samples_split': [2,3,4,5],
                             'min_samples_leaf' : [1,3,5,7,9],
                             'random_state' : [42]}

    sgd_distribution = {'loss': ['hinge', 'log', 'huber', 'modified_huber', 'squared_hinge'],
                         'penalty': ['l2', 'l1'],
                         'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                         'random_state': [42]}

    ridge_distribution = {'alpha': [1,1.2,1.4,1.6,1.8,2],
                          'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'],
                          'random_state': [42]}


    tuning_params = [kneighbours_distribution, None]
    # tuning_strategy = RandomizedSearchCV
    tuning_strategy = GridSearchCV

    # Create a new instance of an experiment
    e = Experiment(datasets, learners, tuning_params, tuning_strategy, n_splits=n_splits)

    print("Running experiments...")
    e.run_all_experiments()
    print("Finished running all experiments")

    df = pd.read_pickle("experiment_results.gz")
    f = CurveFit(df)
    f.extrapolate(0.2)