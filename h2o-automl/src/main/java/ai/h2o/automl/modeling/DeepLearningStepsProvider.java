package ai.h2o.automl.modeling;

import ai.h2o.automl.*;
import ai.h2o.automl.preprocessing.PreprocessingConfig;
import ai.h2o.automl.preprocessing.TargetEncoding;
import hex.deeplearning.DeepLearningModel;
import hex.deeplearning.DeepLearningModel.DeepLearningParameters;
import hex.grid.Grid;
import water.Job;

import java.util.HashMap;
import java.util.Map;

import static ai.h2o.automl.ModelingStep.GridStep.DEFAULT_GRID_TRAINING_WEIGHT;
import static ai.h2o.automl.ModelingStep.ModelStep.DEFAULT_MODEL_TRAINING_WEIGHT;

public class DeepLearningStepsProvider
        implements ModelingStepsProvider<DeepLearningStepsProvider.DeepLearningSteps>
                 , ModelParametersProvider<DeepLearningParameters> {

    public static class DeepLearningSteps extends ModelingSteps {

        static abstract class DeepLearningModelStep extends ModelingStep.ModelStep<DeepLearningModel> {
            public DeepLearningModelStep(String id, int weight, int priorityGroup, AutoML autoML) {
                super(Algo.DeepLearning, id, weight, priorityGroup, autoML);
            }
            
            @Override
            protected PreprocessingConfig getPreprocessingConfig() {
                //TE useless for DNN
                PreprocessingConfig config = super.getPreprocessingConfig();
                config.put(TargetEncoding.CONFIG_PREPARE_CV_ONLY, aml().isCVEnabled());
                return config;
            }
        }

        static abstract class DeepLearningGridStep extends ModelingStep.GridStep<DeepLearningModel> {

            DeepLearningGridStep(String id, int weight, int priorityGroup, AutoML autoML) {
                super(Algo.DeepLearning, id, weight, priorityGroup, autoML);
            }

            DeepLearningParameters prepareModelParameters() {
                DeepLearningParameters dlParameters = new DeepLearningParameters();

                dlParameters._epochs = 10000; // early stopping takes care of epochs - no need to tune!
                dlParameters._adaptive_rate = true;
                dlParameters._activation = DeepLearningParameters.Activation.RectifierWithDropout;

                return dlParameters;
            }
            
            @Override
            protected PreprocessingConfig getPreprocessingConfig() {
                //TE useless for DNN
                PreprocessingConfig config = super.getPreprocessingConfig();
                config.put(TargetEncoding.CONFIG_PREPARE_CV_ONLY, aml().isCVEnabled());
                return config;
            }

            Map<String, Object[]> prepareSearchParams() {
                Map<String, Object[]> searchParams = new HashMap<>();

                searchParams.put("_rho", new Double[] { 0.9, 0.95, 0.99 });
                searchParams.put("_epsilon", new Double[] { 1e-6, 1e-7, 1e-8, 1e-9 });
                searchParams.put("_input_dropout_ratio", new Double[] { 0.0, 0.05, 0.1, 0.15, 0.2 });

                return searchParams;
            }
        }


        private ModelingStep[] defaults = new DeepLearningModelStep[] {
                new DeepLearningModelStep("def_1", DEFAULT_MODEL_TRAINING_WEIGHT, 10, aml()) {
                    @Override
                    protected Job<DeepLearningModel> startJob() {
                        DeepLearningParameters dlParameters = new DeepLearningParameters();  // don't use common params for default DL
                        dlParameters._hidden = new int[]{ 10, 10, 10 };
                        return trainModel(dlParameters);
                    }
                },
        };

        private ModelingStep[] grids = new DeepLearningGridStep[] {
                new DeepLearningGridStep("grid_1", DEFAULT_GRID_TRAINING_WEIGHT, 90,  aml()) {
                    @Override
                    protected Job<Grid> startJob() {
                        DeepLearningParameters dlParameters = prepareModelParameters();

                        Map<String, Object[]> searchParams = prepareSearchParams();
                        searchParams.put("_hidden", new Integer[][] {
                                {  20 },
                                {  50 },
                                { 100 }
                        });
                        searchParams.put("_hidden_dropout_ratios", new Double[][] {
                                { 0.0 },
                                { 0.1 },
                                { 0.2 },
                                { 0.3 },
                                { 0.4 },
                                { 0.5 }
                        });

                        return hyperparameterSearch(dlParameters, searchParams);
                    }
                },
                new DeepLearningGridStep("grid_2", DEFAULT_GRID_TRAINING_WEIGHT, 100,aml()) {
                    @Override
                    protected Job<Grid> startJob() {
                        DeepLearningParameters dlParameters = prepareModelParameters();

                        Map<String, Object[]> searchParams = prepareSearchParams();
                        searchParams.put("_hidden", new Integer[][] {
                                {  20,  20 },
                                {  50,  50 },
                                { 100, 100 }
                        });
                        searchParams.put("_hidden_dropout_ratios", new Double[][] {
                                { 0.0, 0.0 },
                                { 0.1, 0.1 },
                                { 0.2, 0.2 },
                                { 0.3, 0.3 },
                                { 0.4, 0.4 },
                                { 0.5, 0.5 }
                        });
                        return hyperparameterSearch(dlParameters, searchParams);
                    }
                },
                new DeepLearningGridStep("grid_3", DEFAULT_GRID_TRAINING_WEIGHT, 100,aml()) {
                    @Override
                    protected Job<Grid> startJob() {
                        DeepLearningParameters dlParameters = prepareModelParameters();

                        Map<String, Object[]> searchParams = prepareSearchParams();
                        searchParams.put("_hidden", new Integer[][] {
                                {  20,  20,  20 },
                                {  50,  50,  50 },
                                { 100, 100, 100 }
                        });
                        searchParams.put("_hidden_dropout_ratios", new Double[][] {
                                { 0.0, 0.0, 0.0 },
                                { 0.1, 0.1, 0.1 },
                                { 0.2, 0.2, 0.2 },
                                { 0.3, 0.3, 0.3 },
                                { 0.4, 0.4, 0.4 },
                                { 0.5, 0.5, 0.5 }
                        });

                        return hyperparameterSearch(dlParameters, searchParams);
                    }
                },
        };

        public DeepLearningSteps(AutoML autoML) {
            super(autoML);
        }

        @Override
        protected ModelingStep[] getDefaultModels() {
            return defaults;
        }

        @Override
        protected ModelingStep[] getGrids() {
            return grids;
        }
    }

    @Override
    public String getName() {
        return Algo.DeepLearning.name();
    }

    @Override
    public DeepLearningSteps newInstance(AutoML aml) {
        return new DeepLearningSteps(aml);
    }

    @Override
    public DeepLearningParameters newDefaultParameters() {
        return new DeepLearningParameters();
    }
}

