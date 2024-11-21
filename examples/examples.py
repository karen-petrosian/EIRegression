import os

from XGBRegression.one_class.movies import run_multiple_executions as xbg_movies_example_one
from XGBRegression.one_class.housing import run_multiple_executions as xbg_housing_example_one
from XGBRegression.one_class.concrete import run_multiple_executions as xbg_concrete_example_one
from XGBRegression.one_class.insurance import run_multiple_executions as xbg_insurance_example_one
from XGBRegression.one_class.house_16 import run_multiple_executions as xbg_house_16_example_one
from XGBRegression.one_class.bank32NH import run_multiple_executions as xbg_bank_32_example_one
from XGBRegression.one_class.delta_elevators import run_multiple_executions as xgb_delta_elevator_example_one

from XGBRegression.three_class.movies import run_multiple_executions as xbg_movies_example_three
from XGBRegression.three_class.housing import run_multiple_executions as xbg_housing_example_three
from XGBRegression.three_class.concrete import run_multiple_executions as xbg_concrete_example_three
from XGBRegression.three_class.insurance import run_multiple_executions as xbg_insurance_example_three
from XGBRegression.three_class.house_16 import run_multiple_executions as xbg_house_16_example_three
from XGBRegression.three_class.bank32NH import run_multiple_executions as xbg_bank_32_example_three
from XGBRegression.three_class.delta_elevators import run_multiple_executions as xgb_delta_elevator_example_three

from RFRegression.one_class.movies import run_multiple_executions as rf_movies_example_one
from RFRegression.one_class.housing import run_multiple_executions as rf_housing_example_one
from RFRegression.one_class.concrete import run_multiple_executions as rf_concrete_example_one
from RFRegression.one_class.insurance import run_multiple_executions as rf_insurance_example_one
from RFRegression.one_class.house_16 import run_multiple_executions as rf_house_16_example_one
from RFRegression.one_class.bank32NH import run_multiple_executions as rf_bank_32_example_one
from RFRegression.one_class.delta_elevators import run_multiple_executions as rf_delta_elevator_example_one

from RFRegression.three_class.movies import run_multiple_executions as rf_movies_example_three
from RFRegression.three_class.housing import run_multiple_executions as rf_housing_example_three
from RFRegression.three_class.concrete import run_multiple_executions as rf_concrete_example_three
from RFRegression.three_class.insurance import run_multiple_executions as rf_insurance_example_three
from RFRegression.three_class.house_16 import run_multiple_executions as rf_house_16_example_three
from RFRegression.three_class.bank32NH import run_multiple_executions as rf_bank_32_example_three
from RFRegression.three_class.delta_elevators import run_multiple_executions as rf_delta_elevator_example_three

def main():
    num_buckets = 2
    num_iterations = 1
    save_dir = "results"

    # Run all three examples one after the other

    print("xgboost movies (one class)")
    xbg_movies_example_one(num_buckets=num_buckets,
                           num_iterations=num_iterations,
                           save_dir=os.path.join(save_dir, "xgboost/one_class/movies"))

    print("xgboost housing (one class)")
    xbg_housing_example_one(num_buckets=num_buckets,
                            num_iterations=num_iterations,
                            save_dir=os.path.join(save_dir, "xgboost/one_class/housing"))

    print("xgboost concrete (one class)")
    xbg_concrete_example_one(num_buckets=num_buckets,
                             num_iterations=num_iterations,
                             save_dir=os.path.join(save_dir, "xgboost/one_class/concrete"))

    print("xgboost insurance (one class)")
    xbg_insurance_example_one(num_buckets=num_buckets,
                              num_iterations=num_iterations,
                              save_dir=os.path.join(save_dir, "xgboost/one_class/insurance"))

    print("xgboost house_16 (one class)")
    xbg_house_16_example_one(num_buckets=num_buckets,
                             num_iterations=num_iterations,
                             save_dir=os.path.join(save_dir, "xgboost/one_class/house_16"))

    print("xgboost bank32NH (one class)")
    xbg_bank_32_example_one(num_buckets=num_buckets,
                            num_iterations=num_iterations,
                            save_dir=os.path.join(save_dir, "xgboost/one_class/bank32NH"))

    print("xgboost delta elevators (one class)")
    xgb_delta_elevator_example_one(num_buckets=num_buckets,
                                   num_iterations=num_iterations,
                                   save_dir=os.path.join(save_dir, "xgboost/one_class/delta_elevators"))

    print("random forest movies (one class)")
    rf_movies_example_one(num_buckets=num_buckets,
                          num_iterations=num_iterations,
                          save_dir=os.path.join(save_dir, "rf/one_class/movies"))

    print("random forest housing (one class)")
    rf_housing_example_one(num_buckets=num_buckets,
                           num_iterations=num_iterations,
                           save_dir=os.path.join(save_dir, "rf/one_class/housing"))

    print("random forest concrete (one class)")
    rf_concrete_example_one(num_buckets=num_buckets,
                            num_iterations=num_iterations,
                            save_dir=os.path.join(save_dir, "rf/one_class/concrete"))

    print("random forest insurance (one class)")
    rf_insurance_example_one(num_buckets=num_buckets,
                             num_iterations=num_iterations,
                             save_dir=os.path.join(save_dir, "rf/one_class/insurance"))

    print("random forest house_16 (one class)")
    rf_house_16_example_one(num_buckets=num_buckets,
                            num_iterations=num_iterations,
                            save_dir=os.path.join(save_dir, "rf/one_class/house_16"))

    print("random forest bank32NH (one class)")
    rf_bank_32_example_one(num_buckets=num_buckets,
                           num_iterations=num_iterations,
                           save_dir=os.path.join(save_dir, "rf/one_class/bank32NH"))

    print("random forest delta elevators (one class)")
    rf_delta_elevator_example_one(num_buckets=num_buckets,
                                  num_iterations=num_iterations,
                                  save_dir=os.path.join(save_dir, "rf/one_class/delta_elevators"))

    # Three-class examples
    print("xgboost movies (three class)")
    xbg_movies_example_three(num_buckets=num_buckets,
                             num_iterations=num_iterations,
                             save_dir=os.path.join(save_dir, "xgboost/three_class/movies"))

    print("xgboost housing (three class)")
    xbg_housing_example_three(num_buckets=num_buckets,
                              num_iterations=num_iterations,
                              save_dir=os.path.join(save_dir, "xgboost/three_class/housing"))

    print("xgboost concrete (three class)")
    xbg_concrete_example_three(num_buckets=num_buckets,
                               num_iterations=num_iterations,
                               save_dir=os.path.join(save_dir, "xgboost/three_class/concrete"))

    print("xgboost insurance (three class)")
    xbg_insurance_example_three(num_buckets=num_buckets,
                                num_iterations=num_iterations,
                                save_dir=os.path.join(save_dir, "xgboost/three_class/insurance"))

    print("xgboost house_16 (three class)")
    xbg_house_16_example_three(num_buckets=num_buckets,
                               num_iterations=num_iterations,
                               save_dir=os.path.join(save_dir, "xgboost/three_class/house_16"))

    print("xgboost bank32NH (three class)")
    xbg_bank_32_example_three(num_buckets=num_buckets,
                              num_iterations=num_iterations,
                              save_dir=os.path.join(save_dir, "xgboost/three_class/bank32NH"))

    print("xgboost delta elevators (three class)")
    xgb_delta_elevator_example_three(num_buckets=num_buckets,
                                     num_iterations=num_iterations,
                                     save_dir=os.path.join(save_dir, "xgboost/three_class/delta_elevators"))

    print("random forest movies (three class)")
    rf_movies_example_three(num_buckets=num_buckets,
                            num_iterations=num_iterations,
                            save_dir=os.path.join(save_dir, "rf/three_class/movies"))

    print("random forest housing (three class)")
    rf_housing_example_three(num_buckets=num_buckets,
                             num_iterations=num_iterations,
                             save_dir=os.path.join(save_dir, "rf/three_class/housing"))

    print("random forest concrete (three class)")
    rf_concrete_example_three(num_buckets=num_buckets,
                              num_iterations=num_iterations,
                              save_dir=os.path.join(save_dir, "rf/three_class/concrete"))

    print("random forest insurance (three class)")
    rf_insurance_example_three(num_buckets=num_buckets,
                               num_iterations=num_iterations,
                               save_dir=os.path.join(save_dir, "rf/three_class/insurance"))

    print("random forest house_16 (three class)")
    rf_house_16_example_three(num_buckets=num_buckets,
                              num_iterations=num_iterations,
                              save_dir=os.path.join(save_dir, "rf/three_class/house_16"))

    print("random forest bank32NH (three class)")
    rf_bank_32_example_three(num_buckets=num_buckets,
                             num_iterations=num_iterations,
                             save_dir=os.path.join(save_dir, "rf/three_class/bank32NH"))

    print("random forest delta elevators (three class)")
    rf_delta_elevator_example_three(num_buckets=num_buckets,
                                    num_iterations=num_iterations,
                                    save_dir=os.path.join(save_dir, "rf/three_class/delta_elevators"))






if __name__ == '__main__':
    main()
