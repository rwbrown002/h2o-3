extensions = dict(
    set_required_params="""
parms$training_frame <- training_frame
args <- .verify_dataxy(training_frame, x, y)
parms$ignored_columns <- args$x_ignore
parms$response_column <- args$y
""",
    module="""
#' @export   
h2o.get_best_r2_values<- function(object) {
  if( is(object, "H2OModel") && (object@algorithm=='maxrglm'))
    return(return(object@model$best_r2_values))
}

#' @export 
h2o.get_best_model_predictors<-function(object) {
  if ( is(object, "H2OModel") && (object@algorithm=='maxrglm'))
    return(object@model$best_model_predictors)
}    
    """
)

doc = dict(
    preamble="""
H2O MaxRGLM is used to build test best model with one predictor, two predictors, ... up to max_predictor_number 
specified in the algorithm parameters.  The best model is the one with the highest R2 value.
""",
    examples="""
library(h2o)
h2o.init()
# Run MaxRGLM of VOL ~ all predictors
prostate_path <- system.file("extdata", "prostate.csv", package = "h2o")
prostate <- h2o.uploadFile(path = prostate_path)
prostate$CAPSULE <- as.factor(prostate$CAPSULE)
model <- h2o.maxrglm(y = "VOL", x = c("CAPSULE", "RACE", "AGE", "RACE", "DPROS", "DCAPS", "PSA", "GLEASON:), training_frame = prostate)
"""
)
