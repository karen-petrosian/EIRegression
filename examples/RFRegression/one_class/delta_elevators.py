import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from EIRegressor.EmbeddedInterpreter_1bucket import EmbeddedInterpreter
from EIRegressor.model_optimizer import ModelOptimizer
from pathlib import Path



def execute(save_dir, n_buckets=3, i=None, bucketing_method="quantile", single_rules_breaks=3):
    # Load dataframe
    current_file = Path(__file__)  # Gets the path of the current script
    project_root = current_file.parents[2]

    domain_data = pd.read_csv(project_root / "datasets/Elevators/delta_elevators.domain",
                              delim_whitespace=True, header=None)

    column_names = domain_data.iloc[:, 0].apply(lambda x: x.split()[0]).tolist()

    data = pd.read_csv(project_root / "datasets/Elevators/delta_elevators.data",
                       delim_whitespace=True, header=None)
    data.columns = column_names
    target = "Se"
    data = data[data[target].notna()]

    X, y = data.drop(target, axis=1).values, data[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33)

    regressor_hp_grid = {
        'regressor__n_estimators': {"type": "int", "low": 50, "high": 500, "step": 25},
        'regressor__max_depth': {"type": "int", "low": 3, "high": 30, "step": 1},
        'regressor__min_samples_split': {"type": "int", "low": 2, "high": 20, "step": 2},
        'regressor__min_samples_leaf': {"type": "int", "low": 1, "high": 10, "step": 1},
        'regressor__max_features': {"type": "float", "low": 0.1, "high": 1.0, "step": 0.1},
        'regressor__bootstrap': {"type": "categorical", "choices": [True, False]},
        'regressor__min_weight_fraction_leaf': {"type": "float", "low": 0.0, "high": 0.5, "step": 0.1},
        'regressor__max_leaf_nodes': {"type": "int", "low": 10, "high": 1000, "step": 50},
        'regressor__min_impurity_decrease': {"type": "float", "low": 0.0, "high": 0.1, "step": 0.01}
    }

    regressor_default_args = {
        "n_estimators": 100,  # "criterion": "squared_error",
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "n_jobs": -1
    }

    regressor_optimizer = ModelOptimizer(n_trials=1000)

    # Creation of EI Regression with Random Forest
    eiReg = EmbeddedInterpreter(regressor=RandomForestRegressor,
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
              column_names=data.drop(target, axis=1).columns)
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
            expected_result_path = os.path.join(save_dir, "rules", f"rule_results_{n_buckets}_buckets_{iteration}_iterations.txt")

            # Check if this experiment's results already exist
            if not os.path.exists(expected_result_path):
                print(f"Running execution for {n_buckets} bucket(s), iteration {iteration}")
                results = execute(save_dir=save_dir, n_buckets=n_buckets, i=iteration, single_rules_breaks=single_rules_breaks)
                bucket_results.append(results)
                all_results[f"{n_buckets}_buckets"] = bucket_results
                with open(all_results_file_path, 'w') as json_file:
                    json.dump(all_results, json_file, indent=4)


if __name__ == '__main__':
    save_dir = "examples/RFRegression/results/delta_elevators_3_breaks"
    run_multiple_executions(save_dir=save_dir,
                            num_buckets=10,
                            num_iterations=50,
                            dataset_name='delta_elevators_3_breaks')

    print("\n" + "="*47)
    print("✨🎉   Thank you for using this program!   🎉✨")
    print("        🚀 Program executed successfully 🚀")
    print("="*47 + "\n")