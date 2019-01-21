import analysis
from predictions import ModelEvaluator
from predictions import MultivariateAnalysisTests
from predictions import RollingForwardCrossValidation
from predictions import VARPredictor, ModelConfiguration
from data_reader import DataReaderImpl
from features import DateFrameBuilder
from market_analysis.deep_q_learning.data_api.db_worker import DBWorker
from market_analysis.postprocessing import PostprocessingSteps
from preprocessing import PreprocessingSteps
from datetime import datetime as dt

def build_dataframe():
    dataframe = (
        DateFrameBuilder(data)
            # .add_bolinger_bands_diff(10)
            .add_returns()
            .add_bolinger_bands_diff(10)
            .add_sharp_ratio(5)
        # .add_cummulative_daily_returns()
            # .add_rolling_volatility(10)
            .add_roc(10)
                     .add_momentum()
            .add_rsi_index()
            .add_distance_from_sma(10)
                    # .add_trend(70)
                    # .add_volume()
            .add_price_volume_trend()
                    # .add_acceleration_bands()
                    # .add_high_low_ratio()

            .build()
    )
    return dataframe

def plot_data(data, dataframe):
    analysis.plot_correlation_matrix(dataframe)
    analysis.plot_close(data)
    analysis.plot_price_and_volume(data)
    # analysis.plot_close_high_low(data.ix["2017-04-10":])
    analysis.plot_daily_returns_hist(data)
    analysis.plot_daily_returns(data)
    analysis.plot_monthly_returns(data)
    analysis.plot_rolling_mean_and_std(data, 20)

    # market_data = data_reader.read_data("/home/nissatech/Documents/Market Analysis Data/prices/Data/ETFs/",
    #                                     "spy",
    #                                     '2014-01-24')

    # analysis.get_daily_returns_related_to_market(data, market_data)
    analysis.plot_trend(data)
    analysis.plot_bolinger_bands(data)
    analysis.plot_volatiliity(data)
    analysis.plot(dataframe)
    # analysis.plot_open_close(data)

def train(dataframe):
    configuration = ModelConfiguration()

    preprocessing_steps = PreprocessingSteps(configuration)
    postprocessing_steps = PostprocessingSteps(configuration)
    model_evaluator = ModelEvaluator()
    multivariate = MultivariateAnalysisTests()

    multivariate.cointegration_exists(dataframe)

    preprocessed = preprocessing_steps.preprocess_data(dataframe)
    predictor = VARPredictor()
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
    r = predictor.predict(10)
    predicted = predictor.predict(10)
    # postprocessing_steps.postprocess(dataframe, predicted)

# data_reader = DataReaderImpl()
#
# data = data_reader.read_data("/home/nissatech/Documents/Market Analysis Data/prices/Data/Stocks/",
#                               "ego",
#                               '2016-06-04')


db_worker = DBWorker()
start_date = dt.strptime("2018-11-22 00:00:00", '%Y-%m-%d %H:%M:%S')
data = db_worker.get_trades_for_period('BTC-EUR', start_date)

# data =daily_stock_prices_getter.get_data("2018-04-30", "EIA/PET_RWTC_D")
# data = daily_stock_prices_getter.get_table("MCD", "2018-05-11")
if data.size != 0:
    print data.describe()
    # plt.close('all')
    dataframe = build_dataframe()

    plot_data(data, dataframe)
    # train(dataframe)
    # roll = RollingForwardCrossValidation()
    # roll.validation(0, dataframe)





