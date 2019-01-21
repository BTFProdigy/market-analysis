import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from market_analysis.features import ReturnsStatistics
from market_analysis.features import TechnicalFeaturesCalculator

returns_statistics= ReturnsStatistics()
tfc = TechnicalFeaturesCalculator()
# plt.set_cmap('viridis')

matplotlib.rcParams['axes.color_cycle'] = ['orchid', 'darkblue', 'gold']

def plot_close(data):
    data['Price'].plot()
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title("Price")
    plt.show()

def plot_volume(data):
    data['Volume'].plot()
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.title("Volume")
    plt.show()

def plot_rolling_mean_and_std(data, window):
    price = data['Price']
    rolmean = price.rolling(window=window, center=False).mean()

    rolstd = price.rolling(window=window).std()

    price.plot( color='blue', label='Original')
    rolmean.plot(color='red', label='Rolling Mean')
    rolstd.plot(color='black', label = 'Rolling Std')

    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

def get_daily_returns_related_to_market(prices, market_prices):
    prices = prices["Price"]
    market_prices = market_prices["Price"]

    daily_returns = returns_statistics.compute_returns(prices)
    market_daily_returns = returns_statistics.compute_returns(market_prices)

    daily_returns.plot(label = "Stock Daily Returns")
    market_daily_returns.plot(label = "Market Daily Returns")
    plt.title("Stock & Market Daily Returns")
    plt.legend()
    plt.show()

    ratio = daily_returns / market_daily_returns
    ratio.plot()
    plt.title("Stock/Market Daily Returns Ratio")

    plt.show()

    return

def plot_daily_returns_hist(data):
    prices = data['Price']
    daily_returns = returns_statistics.compute_returns(prices)

    histogram_daily_return(daily_returns)

def plot_daily_returns(data):
    prices = data['Price']
    daily_returns = returns_statistics.compute_returns(prices)
    daily_returns*=100

    daily_returns.plot()
    plt.xlabel('Time')
    plt.ylabel("%")
    plt.title("Daily Returns")
    plt.show()

def histogram_daily_return(daily_returns):
    mean = daily_returns.mean()
    std = daily_returns.std()

    daily_returns.hist()
    plt.axvline(mean, linestyle='dashed', linewidth = 2, color = 'r')
    plt.axvline(std, linestyle='dashed', linewidth = 2, color = 'g')
    plt.axvline(-std, linestyle='dashed', linewidth = 2, color = 'g')
    plt.title("Histogram of Daily Returns")
    plt.show()

def plot_correlation_matrix(data):
    corrs = data.corr()
    sns.heatmap(corrs,
                xticklabels=corrs.columns,
                yticklabels=corrs.columns)
    plt.show()


def plot_price_and_volume(data):
    data['Price'].plot(label = "Price")
    normalized_volume = data['Volume']*data['Price'].ix[0]/data['Volume'].ix[0]
    normalized_volume.plot(label = "Normalized Volume")
    plt.title("Price and Volume Normalized")
    plt.legend()
    plt.show()

def plot_close_high_low(data):
    data['Price'].plot(label = "Price")
    data['Low'].plot(label = "Low")
    data['High'].plot(label = "High")
    data['Open'].plot(label = "Open")
    plt.title("Close, High & Low")
    plt.legend()
    plt.show()

def plot_monthly_returns(data):
    monthly = returns_statistics.get_monthly_returns(data["Price"])
    monthly*=100
    monthly.plot()

    plt.title("Monthly returns")
    plt.xlabel('Time')
    plt.ylabel("%")
    plt.show()

def plot_open_close(data):
    data['Price'].plot(label = "Price")
    data['Open'].plot(label = "Open")
    plt.title("Open & Close")
    plt.legend()
    plt.show()

def plot_trend(data):
    price = data['Price']
    rolmean = price.rolling(window=90).mean()

    price.plot( color='blue', label='Original')
    rolmean.plot(color='red', label='Rolling Mean')

    plt.legend(loc='best')
    plt.title('Trend')
    plt.show()

def plot_bolinger_bands(data):
    close = data["Price"]
    bolinger_bands = tfc.get_bolinger_bands(close)
    bolinger_bands[0].plot(label = "Lower Band", linestyle="dashed")
    bolinger_bands[1].plot(label = "Upper Band", linestyle="dashed")
    close.plot(label = "Price")

    plt.legend()
    plt.title("Bolinger Bands")
    plt.show()


def plot_volatiliity(data):
    volatility = tfc.get_rolling_volatility(data["Price"], 10)
    volatility*=100
    volatility.plot(label = "Volatility")

    plt.legend()
    plt.title("Volatility")
    plt.xlabel('Time')
    plt.ylabel("%")
    plt.show()

def plot(data):
    for column in data.columns:
        data[column].plot()
        plt.title(column)

        plt.show()