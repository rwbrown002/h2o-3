package hex.InfoGram;

import hex.Model;
import hex.ModelBuilder;
import hex.ModelBuilderHelper;
import hex.ModelCategory;
import water.H2O;
import water.Key;
import water.Scope;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.TwoDimTable;
import water.DKV;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import hex.genmodel.utils.DistributionFamily;

import static hex.InfoGram.InfoGramModel.InfoGramParameters.Algorithm.AUTO;
import static hex.InfoGram.InfoGramModel.InfoGramParameters.Algorithm.gbm;
import static hex.InfoGram.InfoGramUtils.*;
import static hex.gam.MatrixFrameUtils.GamUtils.keepFrameKeys;



public class InfoGram extends ModelBuilder<InfoGramModel, InfoGramModel.InfoGramParameters,
        InfoGramModel.InfoGramModelOutput> {
  boolean _buildCore; // true to find core predictors, false to find admissible predictors
  String[] _topKPredictors; // contain the names of top predictors to consider for infogram
  Frame _baseOrSensitiveFrame = null;
  String[] _modelDescription; // describe each model in terms of predictors used
  int _numModels; // number of models to build
  double[] _cmi;  // store conditional mutual information
  double[] _cmiRaw;  // raw conditional mutual information
  TwoDimTable _varImp;
  int _numPredictors; // number of predictors in training dataset
  Key<Frame> _cmiRelKey;
  List<Key<Frame>> _generatedFrameKeys; // keep track of all keys generated

  public InfoGram(boolean startup_once) { super(new InfoGramModel.InfoGramParameters(), startup_once);}

  public InfoGram(InfoGramModel.InfoGramParameters parms) {
    super(parms);
    init(false);
  }

  public InfoGram(InfoGramModel.InfoGramParameters parms, Key<InfoGramModel> key) {
    super(parms, key);
    init(false);
  }

  @Override
  protected Driver trainModelImpl() {
    return new InfoGramDriver();
  }

  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[] { ModelCategory.Binomial, ModelCategory.Multinomial};
  }

  @Override
  public boolean isSupervised() {
    return true;
  }

  @Override
  public boolean havePojo() {
    return false;
  }

  @Override
  public boolean haveMojo() {
    return false;
  }

  @Override
  public BuilderVisibility builderVisibility() {
    return BuilderVisibility.Experimental;
  }

  @Override
  public void init(boolean expensive) {
    super.init(expensive);
    if (expensive)
      validateInfoGramParameters();
  }

  private void validateInfoGramParameters() {
    Frame dataset = _parms.train();

    if (!_parms.train().vec(_parms._response_column).isCategorical())
      error("response_column", " only classification is allowed.  Change your response column " +
              "to be categorical before calling InfoGram.");

    // make sure sensitive_attributes are true predictor columns
    if (_parms._sensitive_attributes != null) {
      List<String> colNames = Arrays.asList(dataset.names());
      for (String senAttribute : _parms._sensitive_attributes)
        if (!colNames.contains(senAttribute))
          error("sensitive_attributes", "sensitive attribute: "+senAttribute+" is not a valid " +
                  "column in the training dataset.");
    }

    _buildCore = _parms._sensitive_attributes == null;
    // make sure conditional_info threshold is between 0 and 1
    if (_parms._conditional_info_threshold < 0 || _parms._conditional_info_threshold > 1)
      error("conditional_info_thresold", "conditional info threshold must be between 0 and 1.");

    // make sure varimp threshold is between 0 and 1
    if (_parms._varimp_threshold < 0 || _parms._varimp_threshold > 1)
      error("varimp_threshold", "varimp threshold must be between 0 and 1.");

    // check top k to be between 0 and training dataset column number
    if (_parms._ntop < 0)
      error("_topk", "topk must be between 0 and the number of predictor columns in your training dataset.");

    _numPredictors = _parms.train().numCols()-1;
    if (_parms._weights_column != null)
      _numPredictors--;
    if (_parms._offset_column != null)
      _numPredictors--;
    if ( _parms._ntop > _numPredictors) {
      warn("_topk", "topk exceed the actual number of predictor columns in your training dataset." +
              "  It will be set to the number of predictors in your training dataset.");
      _parms._ntop = _numPredictors;
    }

    if (_parms._nfolds > 1)
      error("nfolds", "please specify nfolds as part of the algorithm specific parameter in " +
              "_info_algorithm_parms or _model_algorithm_parms");

    if (_parms._nparallelism < 0)
      error("nparallelism", "must be >= 0.  If 0, it is adaptive");

    if (_parms._nparallelism == 0) // adaptively set nparallelism
      _parms._nparallelism = 2* H2O.NUMCPUS;
    
    if (_parms._compute_p_values)
      error("compute_p_values", " compute_p_values calculation is not yet implemented.");
    
    if (nclasses() < 2)
      error("distribution", " infogram currently only supports classification models");
    
    if (DistributionFamily.AUTO.equals(_parms._distribution)) {
      _parms._distribution = (nclasses() == 2) ? DistributionFamily.bernoulli : DistributionFamily.multinomial;
    }
    
    if (!AUTO.equals(_parms._model_algorithm) || null != _parms._model_algorithm_parameters)
      _parms._build_final_model = true;
  }

  private class InfoGramDriver extends Driver {
    void generateBasicFrame() {
      String[] eligiblePredictors = extractPredictors(_parms);  // exclude senstive attributes if applicable
      _baseOrSensitiveFrame = extractTrainingFrame(_parms, _parms._sensitive_attributes, 1, _parms.train().clone());
      _parms.fillImpl(true); // copy over model specific parameters to build infogram
      if (_parms._build_final_model)
        _parms.fillImpl(false); // copy over model specific parameters for final model
      _topKPredictors = extractTopKPredictors(_parms, _parms.train(), eligiblePredictors, _generatedFrameKeys); // extract topK predictors
    }

    @Override
    public void computeImpl() {
      init(true);
      if (error_count() > 0)
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(InfoGram.this);
      _job.update(0, "Initializing model training");
      _generatedFrameKeys = new ArrayList<>(); // generated infogram model plus one for safe Infogram
      generateBasicFrame(); // generate tranining frame with predictors and sensitive features (if specified)
      _numModels = 1 + _topKPredictors.length;
      _modelDescription = generateModelDescription(_topKPredictors, _parms._sensitive_attributes);
      buildModel();
    }
    // todo:  add max_runtime_secs restrictions
    public final void buildModel() {
      InfoGramModel model = null;
      try {
        model = new InfoGramModel(dest(), _parms, new InfoGramModel.InfoGramModelOutput(InfoGram.this));
        model.write_lock(_job);
        _cmiRaw = new double[_numModels];
        buildInfoGramsNRelevance(); // calculate mean CMI
        _job.update(1, "finished building models for InfoGram ...");
        model._output.copyCMIRelevance(_cmiRaw, _cmi, _topKPredictors, _varImp); // copy over cmi, relevance of all predictors
        _cmi = model._output._cmi;
        _cmiRelKey = model._output.generateCMIRelFrame();
        model._output.extractAdmissibleFeatures(_varImp, model._output._all_predictor_names, _cmi, _cmiRaw,
                _parms._conditional_info_threshold, _parms._varimp_threshold);  // extract admissible information model output
        _job.update(1, "finished building final model with admissible features ...");
        if (_parms._build_final_model) {
          Model finalModel = buildFinalModel(model._output._admissible_features);
          Scope.track_generic(finalModel);
          fillModelMetrics(model, finalModel, _parms._model_algorithm_parameters._train.get(), _parms._model_algorithm);
        }
        _job.update(0, "InfoGram building completed...");
        model.update(_job);
      } finally {
        DKV.remove(_baseOrSensitiveFrame._key);
        removeFromDKV(_generatedFrameKeys);
        final List<Key<Vec>> keep = new ArrayList<>();
        if (model != null) {
          keepFrameKeys(keep, _cmiRelKey);
        }
        Scope.exit(keep.toArray(new Key[keep.size()]));
        model.update(_job);
        model.unlock(_job);
      }
    }

    private Model buildFinalModel(String[] admissibleFeatures) { // build final model with admissible features only
      Model.Parameters finalParams = _parms._model_algorithm_parameters;
      Frame trainingFrameFinal = extractTrainingFrame(_parms, admissibleFeatures, 1, _parms.train());
      _generatedFrameKeys.add(trainingFrameFinal._key);
      finalParams._train = trainingFrameFinal._key;
      if (_parms._valid != null)
        _parms._model_algorithm_parameters._valid = extractTrainingFrame(_parms, admissibleFeatures,
                1, _parms.valid()).getKey();

      ModelBuilder builder = ModelBuilder.make(finalParams);
      return (Model) builder.trainModel().get();
    }

    private void buildInfoGramsNRelevance() {
      int outerLoop = (int) Math.floor(_numModels/_parms._nparallelism); // last model is build special
      int modelCount = 0;
      int lastModelInd = _numModels - 1;
      if (outerLoop > 0) {  // build parallel models but limit it to parms._nparallelism at a time
        for (int outerInd = 0; outerInd < outerLoop; outerInd++) {
          buildModelCMINRelevance(modelCount, _parms._nparallelism, lastModelInd);
          modelCount += _parms._nparallelism;
        }
      }
      int leftOver = _numModels - modelCount;
      if (leftOver > 0) // finish building the leftover models
        buildModelCMINRelevance(modelCount, leftOver, lastModelInd);
      _cmi = calculateFinalCMI(_cmiRaw, _buildCore);  // scale cmi to be from 0 to 1, ignore last one
    }

    private void buildModelCMINRelevance(int modelCount, int numModel, int lastModelInd) {
      boolean lastModelIndcluded = (modelCount+numModel >= lastModelInd);
      Frame[] trainingFrames = buildTrainingFrames(_topKPredictors, _parms.train(), _baseOrSensitiveFrame, modelCount,
              numModel, _buildCore, lastModelInd, _generatedFrameKeys); // generate training frame
      Model.Parameters[] modelParams = buildModelParameters(trainingFrames, _parms._infogram_algorithm_parameters,
              numModel, _parms._infogram_algorithm); // generate parameters
      ModelBuilder[] builders = ModelBuilderHelper.trainModelsParallel(buildModelBuilders(modelParams),
              numModel); // build models in parallel
      if (lastModelIndcluded) // extract relevance here for core infogram
        extractRelevance(builders[numModel-1].get(), modelParams[numModel-1]);
      generateInfoGrams(builders, trainingFrames, _cmiRaw, modelCount, numModel, _parms._response_column, 
              _generatedFrameKeys); // extract model, score, generate infogram
    }

    private void extractRelevance(Model model, Model.Parameters parms) {
      if (_buildCore) { // full model is last one, just extract varImp
        _varImp = extractVarImp(_parms._infogram_algorithm, model);
      } else {  // need to build model for fair info grame
        Frame fullFrame = subtractAdd2Frame(_baseOrSensitiveFrame, _parms.train(), _parms._sensitive_attributes,
                _topKPredictors); // training frame is topKPredictors minus sensitive_attributes
        parms._train = fullFrame._key;
        _generatedFrameKeys.add(fullFrame._key);
        ModelBuilder builder = ModelBuilder.make(parms);
        Model fairModel = (Model) builder.trainModel().get();
        _varImp = extractVarImp(_parms._infogram_algorithm, fairModel);
        Scope.track_generic(fairModel);
      }
    }
  }
}
