from market_analysis.deep_q_learning.agents_behavior_saver import AgentsBehaviorSaver
from market_analysis.postprocessing import PostprocessingSteps
from market_analysis.predictions import ModelEvaluator, MultivariateAnalysisTests, RollingForwardCrossValidation
from market_analysis.predictions.algorithm.arima import Arima
from market_analysis.preprocessing import PreprocessingSteps


def train(dataframe):
    configuration = ModelConfiguration()

    preprocessing_steps = PreprocessingSteps(configuration)
    postprocessing_steps = PostprocessingSteps(configuration)
    model_evaluator = ModelEvaluator()
    multivariate = MultivariateAnalysisTests()

    # multivariate.cointegration_exists(dataframe)

    preprocessed = preprocessing_steps.preprocess_data(dataframe)
    predictor = Arima()
    predictor.r = dataframe
    predictor.create_model(preprocessed)

    # sabrati sa normalizovanim podacima
    postprocessed = postprocessing_steps.postprocess(predictor.get_fitted_values(), dataframe)
    p = postprocessing_steps.postprocess(preprocessed, dataframe)
    multivariate.granger_causality_test(preprocessed, predictor.get_model_results())
    # model_evaluator.evaluate_model(predictor.get_fitted_values(), preprocessed)
    model_evaluator.plot_scatter_real_predicted(preprocessed, predictor.get_fitted_values())
    # model_evaluator.evaluate_model(postprocessed, dataframe)
    print model_evaluator.get_rmse(predictor.get_fitted_values(), preprocessed)
    mape = predictor.get_mape()
    print mape

    predicted = predictor.predict(10)

  # postprocessing_steps.postprocess(dataframe, predicted)

# data_reader = DataReaderImpl()
data = []
if data.size != 0:
    print data.describe()
    behavior_saver = AgentsBehaviorSaver()
    # plt.close('all')
    # dataframe =
    actions = behavior_saver.load()[0]

    train(dataframe)
    roll = RollingForwardCrossValidation()
    roll.validation(0, dataframe)
