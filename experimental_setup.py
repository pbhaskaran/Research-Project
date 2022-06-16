import sys

from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, KFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Common imports
import numpy as np
import pandas as pd
from typing import List


def all_elems_equal(s):
    a = s.to_numpy()
    return (a[0] == a).all()


def sqrt_schedule_function(k) -> int:
    """
    Create a schedule according to a geometric sequence of square root of two starting from 16
    :param k: the current increment to compute the training set size
    :return: an integer indicating the training set size
    """
    return int(np.ceil(2.0 ** ((7.0 + k) / 2.0)))


def get_schedule(max_size, schedule_function=sqrt_schedule_function, min_size=16) -> List[int]:
    """
    Create a schedule for increasing the dataset on which a model is evaluated.
    :param max_size: the maximum size of the training set
    :param schedule_function: the scheduling function
    :param min_size: the minimum size of the training set
    :return: a schedule (i.e. a list) of increasing training set sizes.
    """
    res = []
    k = 1.0
    training_size = schedule_function(k)

    while training_size < max_size:
        if training_size >= min_size:
            res.append(training_size)
        k += 1
        training_size = schedule_function(k)

    return res


def split_kfold_stratified(X, y, n_splits, random_state) -> list:
    """
    Split the datasets into k stratified folds
    :param X: all instances in the dataset
    :param y: the corresponding label for each instance
    :param n_splits: the number of splits to generate
    :param random_state: seed for splitting reproducibility
    :return: k training and test folds
    """
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []

    for train_index, test_index in skfolds.split(X, y):
        X_train_folds = X.iloc[train_index]
        y_train_folds = y.iloc[train_index]
        X_test_folds = X.iloc[test_index]
        y_test_fold = y.iloc[test_index]

        # Reshuffle to ensure that there are at least two classes in the first 16 elements of the fold
        while all_elems_equal(y_train_folds[:16]):
            shuffled_indices = np.random.permutation(len(y_train_folds))
            X_train_folds = X_train_folds.iloc[shuffled_indices]
            y_train_folds = y_train_folds.iloc[shuffled_indices]

        folds.append((X_train_folds, y_train_folds, X_test_folds, y_test_fold))

    return folds


def get_datasets_curve(training_size, X_train_split, y_train_split, X_test_split, y_test_split):
    """
    Given the current training size, divide the current training set into one that has the `training_size` and add
    the unused data to the test set.
    :param training_size: size that the training set needs to have
    :param X_train_split: the current stratified train inputs
    :param y_train_split: the current stratified train labels
    :param X_test_split: the current stratified test inputs
    :param y_test_split: the current stratified test labels
    :return: four splits which conform to the `training_size`
    """
    X_train_k = X_train_split[:training_size]
    y_train_k = y_train_split[:training_size]

    # Append unused data to the test set
    X_test_k = np.append(X_test_split, X_train_split[training_size:], axis=0)
    y_test_k = np.append(y_test_split, y_train_split[training_size:], axis=0)

    # X_test_k = X_test_split
    # y_test_k = y_test_split

    return X_train_k, y_train_k, X_test_k, y_test_k


class Experiment:
    """
    A class for learning curve experiments, which runs each dataset against each learner and generate learning
    curves for increasing dataset size according to a schedule given by the `schedule_function`. The results are
    all saved to a file called `experiment_results.gz`.
    """

    def __init__(self, datasets, learners, tuning_params=None, tuning_strategy=None, n_splits=10,
                 performance_metric=accuracy_score, schedule_function=sqrt_schedule_function):
        """
        Create a new class for experiments, which runs each dataset against each learner and generate learning curves
        for increasing dataset size according to a schedule given by the `schedule_function`. The results are all saved
        to a file called `experiment_results.gz`.
        :param datasets: a list of id for datasets which are on OpenML
        :param learners: a list of instantiated learners
        :param tuning_params: a list of dictionaries for tuning parameter, each of which corresponding a learner
        of the same index, default is `None` if hyperparameters should not be tuned
        :param tuning_strategy: an uninstantiated class of the strategy for tuning (e.g. `RandomSearchCV`), default is
        `None` if hyperparameters should not be tuned (only one class should be passed into this parameter)
        :param n_splits: the number of splits to generate which corresponds to the number of learning curves generated
        for each dataset and classifier
        :param performance_metric: the performance metric for the learner's predictions
        :param schedule_function: the function to schedule the increase of the training set size
        """
        self.datasets: list[int] = datasets
        self.learners: list = learners
        self.tuning_params = tuning_params
        self.tuning_strategy = tuning_strategy
        self.n_splits: int = n_splits
        self.performance_metric = performance_metric
        self.schedule_function = schedule_function
        self.best_params_split = []

        # If either the tuning_params or tuning_strategy is None/empty then create an array of None for tuning params
        if tuning_params is None or tuning_strategy is None or len(tuning_params) == 0:
            self.tuning_params = [None] * len(learners)

    def __preprocess_data(self, df):
        df_num = df.select_dtypes(include=[np.number])
        num_attribs = list(df_num)

        # build num pipeline
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ])

        # build full pipeline
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
        ])

        full_pipeline.fit(df)

        return full_pipeline

    def __evaluate_learner(self, openmlid, dataset, og_learner, hyperparameters):
        """
        Helper function to evaluate a particular dataset on a particular learner, with or without tuning
        :param openmlid: the openmlid of the dataset
        :param dataset: the fetched dataset from OpenML
        :param og_learner: the learner the dataset is to be evaluated on
        :param hyperparameters: the distribution or grid of hyperparameters used for tuning
        :return: a table containing 1. learner name, 2. openmlid, 3. training_size, 4. labels,
        5. predictions, 6. performance
        """
        X, y = dataset["data"], dataset["target"]

        # Do one-hot encoding for categorical attributes
        cat_attributes = list(X.select_dtypes(include=['category', 'object']))
        X = pd.get_dummies(X, columns=cat_attributes)

        splits = split_kfold_stratified(X, y, self.n_splits, 42)
        prediction_table = []
        for split in splits:
            X_train_split, y_train_split, X_test_split, y_test_split = split

            # Preprocess data
            preprocessor = self.__preprocess_data(X_train_split)
            X_train_split = preprocessor.transform(X_train_split)
            X_test_split = preprocessor.transform(X_test_split)
            y_train_split = y_train_split.to_numpy()
            y_test_split = y_test_split.to_numpy()

            schedule = get_schedule(len(y_train_split), self.schedule_function, 16)

            for j in range(0, len(schedule)):
                learner = clone(og_learner)
                X_train_k, y_train_k, X_test_k, y_test_k = \
                    get_datasets_curve(schedule[j], X_train_split, y_train_split, X_test_split, y_test_split)
                if hyperparameters:
                    param_search = self.tuning_strategy(learner, hyperparameters, n_jobs=-1,
                                                        cv=KFold(n_splits=5, random_state=None, shuffle=False),
                                                        error_score="raise")

                    param_search.fit(X_train_k, y_train_k)
                    self.best_params_split.append(param_search.best_estimator_)
                    learner = param_search.best_estimator_
                else:
                    learner.fit(X_train_k, y_train_k)

                predictions = learner.predict(X_test_k)
                performance = self.performance_metric(y_test_k, predictions)

                if hyperparameters is not None:
                    prediction_table.append((og_learner.__class__.__name__ + "tuned", openmlid, schedule[j], y_test_k,
                                             predictions, performance, 42, 42))
                else:
                    prediction_table.append((og_learner.__class__.__name__, openmlid, schedule[j], y_test_k,
                                         predictions, performance, 42, 42))
        return prediction_table

    def run_all_experiments(self):
        """
        Generate results for each dataset and for each learner and save it to "experiment_results.gz"
        """
        all_results = []
        for openmlid in self.datasets:
            dataset = fetch_openml(data_id=openmlid, as_frame=True)
            for learner_index in range(0, len(self.learners)):
                learner = self.learners[learner_index]
                hyperparameters = self.tuning_params[learner_index]
                all_results = all_results + self.__evaluate_learner(openmlid, dataset, learner, hyperparameters)

        df_all_results = pd.DataFrame(np.array(all_results),
                                      columns=['learner', 'openmlid', 'training_size',
                                               'labels', 'predictions', self.performance_metric.__name__])

        # df_all_results = pd.DataFrame(np.array(all_results),
        #                               columns=['learner', 'openmlid', 'size_train',
        #                                        'labels', 'predictions', 'score_valid', 'outer_seed', 'inner_seed'])

        df_all_results.to_pickle("temp_data/experiment_results737.gz")
        df_all_results.to_csv("temptest.csv")