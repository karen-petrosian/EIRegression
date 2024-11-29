# import random
#
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
# from collections import Counter
#
#
# class ModelOptimizer:
#     def __init__(self, search_method='grid', n_iter=10):
#         """
#         Initialize the ModelOptimizer with a specified search method and number of iterations for random search.
#         :param search_method: Method for hyperparameter tuning ('grid' or 'random').
#         :param n_iter: Number of iterations for random search.
#         """
#         self.search_method = search_method
#         self.n_iter = n_iter
#
#     def optimize(self, pipeline, hp_grid, X_train, y_train, scoring='accuracy', cv=5, lower_search_bound=5):
#         """
#         Perform hyperparameter tuning on the given pipeline.
#         :param pipeline: Pipeline including preprocessing and the model.
#         :param hp_grid: Hyperparameter grid for the model.
#         :param X_train: Training data features.
#         :param y_train: Training data target.
#         :param scoring: Scoring metric for optimization.
#         :param cv: Number of folds for cross-validation.
#         :return: The pipeline with the best found parameters.
#         """
#         # Adjust the number of splits based on the number of samples
#         n_splits = min(cv, len(y_train))
#
#         if len(y_train) <= lower_search_bound:
#             random_params = {k: random.choice(v) for k, v in hp_grid.items()}
#             pipeline.set_params(**random_params)
#             pipeline.fit(X_train, y_train)
#             return pipeline
#
#         # Check for sufficient samples per class for StratifiedKFold
#         if "classifier" in pipeline.named_steps:
#             class_counts = Counter(y_train)
#             if any(count < n_splits for count in class_counts.values()):
#                 raise ValueError(f"One or more classes have fewer members ({min(class_counts.values())}) than n_splits={n_splits}.")
#
#             cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True)
#         else:
#             cv_strategy = KFold(n_splits=n_splits, shuffle=True)
#
#         if self.search_method == 'grid':
#             search = GridSearchCV(pipeline, hp_grid, cv=cv_strategy, scoring=scoring, error_score='raise')
#         elif self.search_method == 'random':
#             search = RandomizedSearchCV(pipeline, hp_grid, n_iter=self.n_iter, cv=cv_strategy, scoring=scoring,
#                                         random_state=42)
#         else:
#             raise ValueError("Invalid search method. Choose 'grid', 'random' or 'stratified'.")
#
#         search.fit(X_train, y_train)
#         return search.best_estimator_


import optuna
from optuna.pruners import BasePruner
import optuna.integration
import optuna.trial
from optuna.study import StudyDirection
from optuna.trial import TrialState
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
optuna.logging.set_verbosity(optuna.logging.DEBUG)


class ModelOptimizer:
    def __init__(
            self,
            n_trials: int = 500,
            timeout: int = 600,
            metric: str = 'r2',
            early_stopping_rounds: Optional[int] = 20,
            random_state: int = 42
    ):
        self.n_trials = n_trials
        self.timeout = timeout
        self.metric = metric
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.study = None
        self.best_pipeline = None
        self.cv_results_ = None

    def _get_metric_func(self):
        if self.metric == 'r2':
            return r2_score
        elif self.metric == 'neg_mse':
            return lambda y, y_pred: -mean_squared_error(y, y_pred)
        elif self.metric == 'neg_mae':
            return lambda y, y_pred: -mean_absolute_error(y, y_pred)
        else:
            return self.metric

    def _create_objective(self, pipeline, param_distributions, X_train, y_train, cv_strategy, scoring):
        def objective(trial):
            params = {}
            for param_name, param_info in param_distributions.items():
                if isinstance(param_info, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_info)
                elif isinstance(param_info, tuple):
                    if isinstance(param_info[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_info[0], param_info[1])
                    elif param_name.endswith(('learning_rate', 'alpha', 'lambda')):
                        params[param_name] = trial.suggest_float(param_name, param_info[0], param_info[1], log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_info[0], param_info[1])

            pipeline.set_params(**params)

            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                scoring=scoring if scoring is not None else self._get_metric_func(),
                cv=cv_strategy,
                n_jobs=-1
            )

            trial.set_user_attr('cv_scores_mean', scores.mean())
            trial.set_user_attr('cv_scores_std', scores.std())

            return scores.mean()

        return objective

    class EarlyStoppingCallback:
        def __init__(self, early_stopping_rounds: int):
            self.early_stopping_rounds = early_stopping_rounds
            self.best_value = None
            self.best_trial = None
            self.stagnant_trials = 0

        def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
            if trial.state != TrialState.COMPLETE:
                return

            current_value = trial.value

            if self.best_value is None or current_value > self.best_value:
                self.best_value = current_value
                self.best_trial = trial.number
                self.stagnant_trials = 0
            else:
                self.stagnant_trials += 1

            if self.stagnant_trials >= self.early_stopping_rounds:
                study.stop()

    def optimize(
            self,
            pipeline,
            param_distributions: Dict[str, Any],
            X_train,
            y_train,
            cv: int = 5,
            scoring: Optional[str] = None,
            lower_search_bound: int = 5,
            **kwargs
    ):
        """
        Perform hyperparameter tuning using Optuna.

        Args:
            pipeline: Pipeline including preprocessing and the model
            param_distributions: Dictionary of parameter distributions for optimization
            X_train: Training data features
            y_train: Training data target
            cv: Number of folds for cross-validation
            scoring: Scoring metric to use (if None, uses the metric specified in __init__)
            lower_search_bound: Minimum number of samples required for optimization
            **kwargs: Additional keyword arguments (ignored)

        Returns:
            The pipeline with the best found parameters
        """
        if len(y_train) <= lower_search_bound:
            random_params = {
                k: np.random.choice(v) if isinstance(v, list) else
                np.random.uniform(v[0], v[1]) for k, v in param_distributions.items()
            }
            pipeline.set_params(**random_params)
            pipeline.fit(X_train, y_train)
            return pipeline

        n_splits = min(cv, len(y_train))
        cv_strategy = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        self.study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )

        objective = self._create_objective(
            pipeline, param_distributions, X_train, y_train, cv_strategy, scoring
        )

        # Create custom early stopping callback
        callbacks = []
        if self.early_stopping_rounds is not None:
            early_stopping = self.EarlyStoppingCallback(self.early_stopping_rounds)
            callbacks.append(early_stopping)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=callbacks,
            n_jobs=-1
        )

        self.cv_results_ = pd.DataFrame({
            'trial': range(len(self.study.trials)),
            'value': [t.value for t in self.study.trials],
            'cv_scores_mean': [t.user_attrs.get('cv_scores_mean') for t in self.study.trials],
            'cv_scores_std': [t.user_attrs.get('cv_scores_std') for t in self.study.trials],
            **{k: [t.params.get(k) for t in self.study.trials] for k in param_distributions.keys()}
        })

        best_params = self.study.best_params
        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)
        self.best_pipeline = pipeline

        return pipeline