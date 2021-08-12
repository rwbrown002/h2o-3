from __future__ import print_function
import os
import sys

from h2o.estimators.infogram import H2OInfoGramEstimator

sys.path.insert(1, os.path.join("..","..",".."))
import h2o
from tests import pyunit_utils
    
def test_infogram_german_data():
    """
    Simple breast cancer data test to check that core infogram is working:
     1. it generates the correct lists as Deep's original code.  
     2. when model and infogram parameters are specified, it uses the correct specification.
    :return: 
    """
    deep_rel = [0.0040477989, 0.0974455315, 0.0086303713, 0.0041002103, 0.0037914745,
                0.0036801151, 0.0257819346, 0.2808010416, 0.0005372569, 0.0036280018, 0.0032444598, 0.0002943119,
                0.0026430897, 0.0262074332, 0.0033317064, 0.0068812603, 0.0006185385, 0.0082121491, 0.0014562177,
                0.0081786997, 1.0000000000, 0.0894895310, 0.6187801784, 0.3302352775, 0.0021346433, 0.0016077771,
                0.0260198502, 0.3404628948, 0.0041384517, 0.0019399743]
    deep_cmi = [0.00000000, 0.31823883, 0.52769230, 0.00000000, 0.00000000, 0.00000000,
                0.01183309, 0.67430653, 0.00000000, 0.00000000, 0.45443221, 0.00000000, 0.24561013, 0.87720587,
                0.31939378, 0.19370515, 0.00000000, 0.16463918, 0.00000000, 0.00000000, 0.44830772, 1.00000000,
                0.00000000, 0.00000000, 0.62478098, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.64466111]
    fr = h2o.import_file(path=pyunit_utils.locate("smalldata/admissibleml_test/wdbc_changed.csv"))
    target = "diagnosis"
    fr[target] = fr[target].asfactor()
    x = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean",
         "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
         "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
         "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
         "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
         "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"]
    infogram_model = H2OInfoGramEstimator(seed = 12345, ntop=50)
    infogram_model.train(x=x, y=target, training_frame=fr)

    # make sure our result matches Deep's
    pred_names, rel = infogram_model.get_all_predictor_relevance()
    x, cmi = infogram_model.get_all_predictor_cmi()
    assert deep_rel.sort() == rel.sort(), "Expected: {0}, actual: {1}".format(deep_rel, rel)
    assert deep_cmi.sort() == cmi.sort(), "Expected: {0}, actual: {1}".format(deep_cmi, cmi)

    gbm_params = {'ntrees':3}
    glm_params = {'family':'binomial'}
    infogram_model_gbm_glm = H2OInfoGramEstimator(seed = 12345, ntop=50, 
                                                  infogram_algorithm='gbm', infogram_algorithm_params=gbm_params, 
                                                  model_algorithm='glm', model_algorithm_params=glm_params)
    infogram_model_gbm_glm.train(x=x, y=target, training_frame=fr)
    x, cmi_gbm_glm = infogram_model_gbm_glm.get_all_predictor_cmi()
    assert abs(cmi_gbm_glm[1]-cmi[1]) > 0.01, "CMI from infogram model with gbm using different number of trees should" \
                                              " be different but are not."
    
def assert_list_frame_equal(cmi, rel, predictor_rel_cmi_frame, tol=1e-6):
    rel_frame = predictor_rel_cmi_frame["Relevance"].as_data_frame(use_pandas=False)
    cmi_frame = predictor_rel_cmi_frame["CMI"].as_data_frame(use_pandas=False)
    count = 1
    for one_cmi in cmi:
        assert abs(float(cmi_frame[count][0])-one_cmi) < tol, "expected: {0}, actual: {1} and they are " \
                                                              "different".format(float(cmi_frame[count][0]), one_cmi) 
        assert abs(float(rel_frame[count][0])-rel[count-1]) < tol, "expected: {0}, actual: {1} and they are " \
                                                                   "different".format(float(rel_frame[count][0]), rel[count-1])
        count += 1


if __name__ == "__main__":
    pyunit_utils.standalone_test(test_infogram_german_data)
else:
    test_infogram_german_data()
