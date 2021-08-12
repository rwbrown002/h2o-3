package hex.schemas;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import hex.InfoGram.InfoGram;
import hex.InfoGram.InfoGramModel;
import hex.Model;
import hex.deeplearning.DeepLearningModel;
import hex.glm.GLMModel;
import hex.tree.drf.DRFModel;
import hex.tree.gbm.GBMModel;
import hex.tree.xgboost.XGBoostModel;
import water.api.API;
import water.api.schemas3.KeyV3;
import water.api.schemas3.ModelParametersSchemaV3;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

public class InfoGramV3 extends ModelBuilderSchema<InfoGram, InfoGramV3, InfoGramV3.InfoGramParametersV3> {
  public static final class InfoGramParametersV3 extends ModelParametersSchemaV3<InfoGramModel.InfoGramParameters, InfoGramParametersV3> {
    public static final String[] fields = new String[] {
            "model_id",
            "training_frame",
            "validation_frame",
            "seed",
            "keep_cross_validation_models",
            "keep_cross_validation_predictions",
            "keep_cross_validation_fold_assignment",
            "fold_assignment",
            "fold_column",
            "response_column",
            "ignored_columns",
            "ignore_const_cols",
            "score_each_iteration",
            "offset_column",
            "weights_column",
            "standardize",
            "distribution",
            "plug_values",
            "max_iterations",
            "stopping_rounds",
            "stopping_metric",
            "stopping_tolerance",
            // dead unused args forced here by backwards compatibility, remove in V4
            "balance_classes",
            "class_sampling_factors",
            "max_after_balance_size",
            "max_confusion_matrix_size",
            "max_runtime_secs",
            "custom_metric_func",
            "auc_type",
            // new parameters for INFOGRAMs only
            "infogram_algorithm", // choose algo and parameter to generate infogram
            "infogram_algorithm_params",
            "model_algorithm",    // choose algo and parameter to generate final model
            "model_algorithm_params",
            "sensitive_attributes",
            "conditional_info_threshold",
            "varimp_threshold",
            "data_fraction",
            "nparallelism",
            "ntop",
            "compute_p_values"
    };

    @API(help = "Seed for pseudo random number generator (if applicable)", gridable = true)
    public long seed;

    // Input fields
    @API(help = "Standardize numeric columns to have zero mean and unit variance", level = API.Level.critical)
    public boolean standardize;

    @API(help = "Plug Values (a single row frame containing values that will be used to impute missing values of the training/validation frame, use with conjunction missing_values_handling = PlugValues)", direction = API.Direction.INPUT)
    public KeyV3.FrameKeyV3 plug_values;

/*    @API(help = "Restrict coefficients (not intercept) to be non-negative")
    public boolean non_negative;*/

    @API(help = "Maximum number of iterations", level = API.Level.secondary)
    public int max_iterations;

    @API(help = "Prior probability for y==1. To be used only for logistic regression iff the data has been sampled and the mean of response does not reflect reality.", level = API.Level.expert)
    public double prior;

    // dead unused args, formely inherited from supervised model schema
    /**
     * For imbalanced data, balance training data class counts via
     * over/under-sampling. This can result in improved predictive accuracy.
     */
    @API(help = "Balance training data class counts via over/under-sampling (for imbalanced data).", level = API.Level.secondary, direction = API.Direction.INOUT)
    public boolean balance_classes;

    /**
     * Desired over/under-sampling ratios per class (lexicographic order).
     * Only when balance_classes is enabled.
     * If not specified, they will be automatically computed to obtain class balance during training.
     */
    @API(help = "Desired over/under-sampling ratios per class (in lexicographic order). If not specified, sampling factors will be automatically computed to obtain class balance during training. Requires balance_classes.", level = API.Level.expert, direction = API.Direction.INOUT)
    public float[] class_sampling_factors;

    /**
     * When classes are balanced, limit the resulting dataset size to the
     * specified multiple of the original dataset size.
     */
    @API(help = "Maximum relative size of the training data after balancing class counts (can be less than 1.0). Requires balance_classes.", /* dmin=1e-3, */ level = API.Level.expert, direction = API.Direction.INOUT)
    public float max_after_balance_size;

    /** For classification models, the maximum size (in terms of classes) of
     *  the confusion matrix for it to be printed. This option is meant to
     *  avoid printing extremely large confusion matrices.  */
    @API(help = "[Deprecated] Maximum size (# classes) for confusion matrices to be printed in the Logs", level = API.Level.secondary, direction = API.Direction.INOUT)
    public int max_confusion_matrix_size;

    @API(help = "Machine learning algorithm chosen to build the infogram.  AUTO default to GBM", values={"AUTO",
            "deeplearning", "drf", "gbm", "glm", "xgboost"}, level = API.Level.expert, 
            direction = API.Direction.INOUT, gridable=true)
    public InfoGramModel.InfoGramParameters.Algorithm infogram_algorithm;

    @API(help = "parameters specified to the chosen algorithm can be passed to infogram using algorithm_params",
            level = API.Level.expert, gridable=true)
    public String infogram_algorithm_params;

    @API(help = "Machine learning algorithm chosen to build the final model.  Default to AUTO.  If you do not specify" +
            "model_algorithm_params, this will turn off final model building.  If you want to build a final model," +
            " make sure you specify model_algorithm not to AUTO or specify model_algorithm_params.  If you specify" +
            "model_algorithm_params but did not specify model_algorithm, a final gbm will be built with parameters" +
            "specified in model_algorithm_params.", values={"AUTO", "deeplearning", "drf", "gbm", "glm", "xgboost"},
            level = API.Level.critical, direction = API.Direction.INOUT, gridable=true)
    public InfoGramModel.InfoGramParameters.Algorithm model_algorithm;

    @API(help = "parameters specified to the chosen final algorithm", level = API.Level.secondary, gridable=true)
    public String model_algorithm_params;

    @API(help = "predictors that are to be excluded from model due to them being discriminatory or inappropriate for" +
            " whatever reason.", level = API.Level.secondary, gridable=true)
    public String[] sensitive_attributes;

    @API(help = "conditional information threshold between 0 and 1 that is used to decide whether a predictor's " +
            "conditional information is high enough.  Default to 0.1", level = API.Level.secondary, gridable = true)
    public double conditional_info_threshold;

    @API(help = "variable importance threshold between 0 and 1 that is used to decide whether a predictor's relevance" +
            " level is high enough.  Default to 0.1", level = API.Level.secondary, gridable = true)
    public double varimp_threshold;

    @API(help = "fraction of training frame to use to build the infogram model.  Default to 1.0",
            level = API.Level.secondary, gridable = true)
    public double data_fraction;

    @API(help = "number of models to build in parallel.  Default to 0.0 which is adaptive to the system capability",
            level = API.Level.secondary, gridable = true)
    public int nparallelism;

    @API(help = "number of top k variables to consider based on the varimp.  Default to 0.0 which is to consider" +
            " all predictors",
            level = API.Level.secondary, gridable = true)
    public int ntop;
    
    @API(help = "If true will calculate the p-value. Default to false",
            level = API.Level.secondary, gridable = false)
    public boolean compute_p_values;  // todo implement this option

    public InfoGramModel.InfoGramParameters fillImpl(InfoGramModel.InfoGramParameters impl) {
      super.fillImpl(impl);
 
      if (infogram_algorithm_params != null && !infogram_algorithm_params.isEmpty()) {
        Properties p = generateProperties(infogram_algorithm_params);
        ParamNParamSchema schemaParams = generateParamsSchema(infogram_algorithm);
        schemaParams._paramsSchema.init_meta();
        impl._infogram_algorithm_parameters = (Model.Parameters) schemaParams._paramsSchema
                .fillFromImpl(schemaParams._params)
                .fillFromParms(p, true)
                .createAndFillImpl();
        super.fillImpl(impl);
      }

      if (model_algorithm_params != null && !model_algorithm_params.isEmpty()) {
        Properties p = generateProperties(model_algorithm_params);
        ParamNParamSchema schemaParams = generateParamsSchema(model_algorithm);
        schemaParams._paramsSchema.init_meta();
        impl._model_algorithm_parameters = (Model.Parameters) schemaParams._paramsSchema
                .fillFromImpl(schemaParams._params)
                .fillFromParms(p, true)
                .createAndFillImpl();
        super.fillImpl(impl);
      }
      
      return impl;
    }
    
    Properties generateProperties(String algoParms) {
      Properties p = new Properties();
      HashMap<String, String[]> map = new Gson().fromJson(algoParms, new TypeToken<HashMap<String, String[]>>() {
      }.getType());
      for (Map.Entry<String, String[]> param : map.entrySet()) {
        String[] paramVal = param.getValue();
        if (paramVal.length == 1) {
          p.setProperty(param.getKey(), paramVal[0]);
        } else {
          p.setProperty(param.getKey(), Arrays.toString(paramVal));
        }
      }
      return p;
    }
    
    private class ParamNParamSchema {
      private ModelParametersSchemaV3 _paramsSchema;
      private Model.Parameters _params;
      
      public ParamNParamSchema(ModelParametersSchemaV3 schema, Model.Parameters params) {
        _paramsSchema = schema;
        _params = params;
      }
    }

    ParamNParamSchema generateParamsSchema(InfoGramModel.InfoGramParameters.Algorithm chosenAlgo) {
      ModelParametersSchemaV3 paramsSchema;
      Model.Parameters params;
      switch (chosenAlgo) {
        case AUTO:
        case glm:
          paramsSchema = new GLMV3.GLMParametersV3();
          params = new GLMModel.GLMParameters();
          // FIXME: This is here because there is no Family.AUTO. It enables us to know if the user specified family or not.
          // FIXME: Family.AUTO will be implemented in https://0xdata.atlassian.net/projects/PUBDEV/issues/PUBDEV-7444
          ((GLMModel.GLMParameters) params)._family = null;
          break;
        case gbm:
          paramsSchema = new GBMV3.GBMParametersV3();
          params = new GBMModel.GBMParameters();
          break;
        case drf:
          paramsSchema = new DRFV3.DRFParametersV3();
          params = new DRFModel.DRFParameters();
          break;
        case deeplearning:
          paramsSchema = new DeepLearningV3.DeepLearningParametersV3();
          params = new DeepLearningModel.DeepLearningParameters();
          break;
        case xgboost:
          paramsSchema = new XGBoostV3.XGBoostParametersV3();
          params = new XGBoostModel.XGBoostParameters();
          break;
        default:
          throw new UnsupportedOperationException("Unknown given algo: " + chosenAlgo);
      }
      return new ParamNParamSchema(paramsSchema, params);
    }
  }
}
