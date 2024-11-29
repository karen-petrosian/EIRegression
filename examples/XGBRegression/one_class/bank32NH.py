import os
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
from EIRegressor.EmbeddedInterpreter_1bucket import EmbeddedInterpreter
from EIRegressor.model_optimizer import ModelOptimizer


def execute(save_dir, n_buckets=3, i=None, bucketing_method="quantile", single_rules_breaks=3):
    # Load dataframes
    current_file = Path(__file__)  # Gets the path of the current script
    project_root = current_file.parents[2]
    train_data = pd.read_csv(project_root / "datasets/bank32NH/bank32nh.data", delim_whitespace=True, header=None)
    test_data = pd.read_csv(project_root / "datasets/bank32NH/bank32nh.test", delim_whitespace=True, header=None)

    domain_data = pd.read_csv(project_root / "datasets/bank32NH/bank32nh.domain", delim_whitespace=True, header=None)
    column_names = domain_data.iloc[:, 0].apply(lambda x: x.split()[0]).tolist()

    # Apply column names to the train and test dataframes
    train_data.columns = column_names
    test_data.columns = column_names

    # Set the target column (assuming 'b2call' for this example)
    target = "rej"

    # Data Preprocessing
    train_data = train_data.apply(pd.to_numeric, errors='ignore')
    test_data = test_data.apply(pd.to_numeric, errors='ignore')

    train_data = pd.get_dummies(train_data, drop_first=True)
    test_data = pd.get_dummies(test_data, drop_first=True)

    train_data = train_data[train_data[target].notna()]
    test_data = test_data[test_data[target].notna()]

    X_train, y_train = train_data.drop(target, axis=1).values, train_data[target].values
    X_test, y_test = test_data.drop(target, axis=1).values, test_data[target].values

    regressor_hp_grid = {
        'regressor__learning_rate': {"type": "float", "low": 0.01, "high": 0.3, "step": 0.01},  # Finer granularity
        'regressor__n_estimators': {"type": "int", "low": 50, "high": 500, "step": 50},  # Wider range for trees
        'regressor__max_depth': {"type": "int", "low": 3, "high": 10, "step": 1},  # More depth options
        'regressor__min_child_weight': {"type": "int", "low": 1, "high": 10, "step": 1},  # Regularization term
        'regressor__subsample': {"type": "float", "low": 0.5, "high": 1.0, "step": 0.1},  # Controls sampling of rows
        'regressor__colsample_bytree': {"type": "float", "low": 0.5, "high": 1.0, "step": 0.1},  # Column sampling
        'regressor__gamma': {"type": "float", "low": 0, "high": 0.5, "step": 0.05},  # Minimum loss reduction
        'regressor__reg_alpha': {"type": "float", "low": 0, "high": 1.0, "step": 0.1},  # L1 regularization
        'regressor__reg_lambda': {"type": "float", "low": 0.1, "high": 2.0, "step": 0.1},  # L2 regularization
        'regressor__scale_pos_weight': {"type": "float", "low": 1, "high": 10, "step": 1},  # Class imbalance handling
        'regressor__max_delta_step': {"type": "float", "low": 0, "high": 10, "step": 1}
    }

    regressor_default_args = {
        "learning_rate": 0.05,
        "n_estimators": 100,
        "max_depth": 5,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1
    }
    regressor_optimizer = ModelOptimizer(n_trials=1000)

    # Creation of EI Regression with Random Forest
    eiReg = EmbeddedInterpreter(regressor=xgb.XGBRegressor,
                                model_optimizer=regressor_optimizer,
                                model_preprocessor=None,
                                n_buckets=n_buckets,
                                bucketing_method=bucketing_method,
                                reg_default_args=regressor_default_args,
                                reg_hp_args=regressor_hp_grid,
                                max_iter=4000, lossfn="MSE",
                                min_dloss=0.0001, lr=0.005, precompute_rules=True,
                                force_precompute=True, device="cuda")

    eiReg.fit(X_train, y_train,
              add_single_rules=True, single_rules_breaks=single_rules_breaks, add_multi_rules=True,
              column_names=train_data.drop(target, axis=1).columns)
    buck_pred, y_pred = eiReg.predict(X_test, return_buckets=True)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    acc, f1, cm = eiReg.evaluate_classifier(X_test, y_test)

    top_uncertainties = eiReg.get_top_uncertainties()

    results = {"R2": r2,
               "MAE": mae,
               "MSE": mse,
               "Accuracy": acc,
               "F1": f1,
               "Confusion Matrix": cm.tolist(),
               "Uncertainties": top_uncertainties}

    save_results = os.path.join(save_dir, "rules")
    os.makedirs(save_results, exist_ok=True)
    eiReg.rules_to_txt(os.path.join(save_results, f"rule_results_{n_buckets}_buckets_{i}_iterations.txt"),
                       results=results)

    return results


def run_multiple_executions(save_dir, num_buckets, num_iterations, single_rules_breaks=3):
    os.makedirs(save_dir, exist_ok=True)
    all_results_file_path = os.path.join(save_dir, f"results_{num_buckets}_buckets_{num_iterations}_iterations.json")

    all_results = {}
    # Check if the consolidated results file exists and load it
    if os.path.exists(all_results_file_path):
        with open(all_results_file_path, 'r') as json_file:
            all_results = json.load(json_file)

    for n_buckets in range(1, num_buckets + 1):
        bucket_results = all_results.get(f"{n_buckets}_buckets", [])

        for iteration in range(1, num_iterations + 1):
            # Construct the expected path for the results of this iteration
            expected_result_path = os.path.join(save_dir, "rules",
                                                f"rule_results_{n_buckets}_buckets_{iteration}_iterations.txt")

            # Check if this experiment's results already exist
            if not os.path.exists(expected_result_path):
                print(f"Running execution for {n_buckets} buckets, iteration {iteration}")
                results = execute(save_dir=save_dir, n_buckets=n_buckets, i=iteration, single_rules_breaks=single_rules_breaks)
                bucket_results.append(results)
                all_results[f"{n_buckets}_buckets"] = bucket_results
                with open(all_results_file_path, 'w') as json_file:
                    json.dump(all_results, json_file, indent=4)
