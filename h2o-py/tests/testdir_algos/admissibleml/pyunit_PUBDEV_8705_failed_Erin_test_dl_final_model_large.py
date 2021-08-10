from __future__ import print_function
import os
import sys

from h2o.estimators.infogram import H2OInfoGramEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

sys.path.insert(1, os.path.join("..","..",".."))
import h2o
from tests import pyunit_utils
    
def test_infogram_iris():
    """
    Test by Erin that was failing due to variable not initialized before using.  I fixed it.
    """
    data = h2o.import_file(pyunit_utils.locate("bigdata/laptop/admissibleml_test/KDDCup09_appetency.csv"))
    y = "APPETENCY"

    data[y] = data[y].asfactor()
    data[y].levels()
    ss = data.split_frame(ratios = [0.8], seed = 1)
    train = ss[0]
    model_params = {'reproducible':True, 'epochs':2}
    infogram_model = H2OInfoGramEstimator(seed = 1, model_algorithm='deeplearning', model_algorithm_params=model_params)
    infogram_model.train(y = y, training_frame = train)
    infogram_final_scoring_history = infogram_model._model_json["output"]["scoring_history"]
    manual_model_drf = H2ODeepLearningEstimator(seed=1, reproducible=True, epochs=2)
    manual_model_drf.train(x=infogram_model, y=y, training_frame=train)
    manual_scoring_history = manual_model_drf._model_json["output"]["scoring_history"]
    # make sure final model in infogram is the same as manually built one.
    pyunit_utils.assert_H2OTwoDimTable_equal_upto(infogram_final_scoring_history,
                                                  manual_scoring_history,['number_of_trees', 'training_rmse', 'training_logloss',
                                                                          'training_classification_error'])

if __name__ == "__main__":
    pyunit_utils.standalone_test(test_infogram_iris)
else:
    test_infogram_iris()
