# trading-analysis

#### Contents

- [Background](https://github.com/michael-tojong/trading-analysis#background)
- [Output](https://github.com/michael-tojong/trading-analysis#output)
- [Usage](https://github.com/michael-tojong/trading-analysis#usage)
- [Data Set](https://github.com/michael-tojong/trading-analysis#data-set)
- [Generating Candlestick Charts via Matplotlib](https://github.com/michael-tojong/trading-analysis#generating-candlestick-charts-via-matplotlib)
- [Feature Engineering](https://github.com/michael-tojong/trading-analysis#feature-engineering)
- [Implementing a Convolutional Neural Network](https://github.com/michael-tojong/trading-analysis#implementing-a-convolutional-neural-network)
- [Using Regression to Draw Trendlines](https://github.com/michael-tojong/trading-analysis#using-regression-to-draw-trendlines)
- [Using Mean Shift Clustering to Draw Support and Resistance Levels](https://github.com/michael-tojong/trading-analysis#using-mean-shift-clustering-to-draw-support-and-resistance-levels)

## Background

The goal of this personal project was to automate financial chart generation with technical analysis considering trends underlying short-term and long-term time frames, support and resistance levels, and price action. To aid in identifying visual patterns, I had also implemented a convolutional neural network to label individual candlestick shapes.

Beginning in 2014, I had an interest in learning to trade financial securities and other assets, such as cryptocurrencies. Because I didn't have much trading capital at the time, my interest first took me to understanding the foreign exchange market along with how swing trading strategies might be utilized by both retail and institutional traders. Typical retail brokers within this market offer dangerously attractive leverage ratios for holding large trading positions that the average person might not have actual funds for, and small fluctuations in the price rate between a pair of currencies had outsized effects when handling such large positions with leverage. In other words, with a relatively small amount of trading capital, forex traders can borrow more capital to have the potential to earn big or lose big with each currency trade. There's nuance to why the majority of retail forex traders actually lose money over time, but this repository description focuses on my motivation for starting this project and functionalities that I wanted the automated analysis to fulfill.

Identifying and watching specific patterns and indicators in financial charts are major activities within trading. I perused through books and online resources to learn that some of the most foundational patterns for traders to know are visual ones, and they include candlestick shapes, trendlines, formations, support levels, and resistance levels. To conduct technical analyses, my tool of choice was TradingView.com, a free site that provides graphical tools to manually analyze financial charts with. There, I was producing visualizations in the same manner that this project would eventually automate. I started this project, however, a few years after I had first learned the ways of retail trading.

There are other widely used indicators that are calculated from price, volume, and other numerical data, such as moving averages, Bollinger Bands, RSI, and MACD. I didn't include the calculations and visualizations for such indicators in this project. As an amateur retail trader, I had subscribed to a simple trading philosophy, having less distractions in my visualizations.

I am no longer actively working on this project, which I had developed on my own time during 2017 through 2018. At the time, this repository was privately [housed on Bitbucket](https://bitbucket.org/mtoj91/the-forex-ai/src/master/). I then migrated it to GitHub in June 2021 for showcase purposes. To help track and prioritize items for development, I had used a free tool called KanbanFlow. My boards still have a few ideas left for me to implement into this project, but for now, they're all on the back burner.

#### Disclaimer

Technical analysis is a useful methodology to intuit how the market is feeling and reacting to events pertaining to the asset at hand. It's also useful for predicting when to enter a trade and at what levels the price might go towards. Fundamental analysis, on the other hand, is more research-oriented, and it requires an appreciation for the ongoing business and economic dynamics surrounding an asset. Technical analysis should be used to augment fundamental analysis, as it doesn't make sense to pay attention to price values and visual patterns alone. As such, the availability of this personal project is for showcase purposes only, and I would like to warn that its outputs should **not** be used as a basis for trading, as it automates only one part of the overall decision-making process in trading. It also doesn't have concerns for money management, a critical aspect of trading, among others. I am not responsible for any financial decision made by any party accessing this showcase.
<br>
<br>
## Output

The following image shows an example of the automated technical analysis that this project fulfills, using *open-high-low-close* price data for the Euro-US Dollar currency pair.

[![Example output - financial chart with technical analysis elements overlaid - EURUSD 4 hour data](https://github.com/michael-tojong/trading-analysis/blob/master/Reference/Example%20output%20-%20financial%20chart%20with%20technical%20analysis%20elements%20overlaid%20-%20EURUSD%204%20hour%20data.jpg?raw=true "Example output - financial chart with technical analysis elements overlaid - EURUSD 4 hour data")](https://github.com/michael-tojong/trading-analysis/blob/master/Reference/Example%20output%20-%20financial%20chart%20with%20technical%20analysis%20elements%20overlaid%20-%20EURUSD%204%20hour%20data.jpg "Example output - financial chart with technical analysis elements overlaid - EURUSD 4 hour data")

The vertical axis signifies price, and the horizontal axis signifies the timestamps that each candle represents. A candle is colored green if its close price was higher than the open price for that timestamp. A red candle signifies price closing lower. The automated analysis does consider data before the first timestamp in the chart, but the candles belonging to that long-term data aren't displayed in the output. The output window only shows the last *x* candles, which a user can configure when generating a chart.

The purple vertical highlight on the center-right of the chart shows the last closed candle. The current candle follows after it, and price can continue to increase or decrease after the data was last downloaded (see the [Data Set](https://github.com/michael-tojong/trading-analysis#data-set) section).

There are two trends calculated from the data — short-term and long-term. The short-term data spans the candles that are displayed in the output window, and the long-term data includes the candles made before the first timestamp in the output window. In the output, the short-term trendlines have an aqua color, and the long-term trendlines have a dark green color. For each trend, there are upper and lower limits, and these limits are shaded darker than the middle "control" trendline. The idea is that whenever price is below the lower limit, it signifies an oversold environment. Correspondingly, when price is above the upper limit, it signifies an overbought environment.

Purple dashed lines following the short-term trend originate from candlesticks that displayed a price rejection from either a support or resistance line or as signified by certain candlestick shapes.

Lastly, the bold red horizontal lines are the long-term support and resistance levels. These levels indicate price values that have shown strong rejections in the past. In other words, price either bounced back up or back down after touching those levels, and it would take a big event for price to break through those levels. The corresponding short-term levels are colored blue, and they show points that price tended to bounce from within the window.
<br>
<br>
## Usage

The "back end" code to conduct an automated analysis are contained in the [*Modules*](https://github.com/michael-tojong/trading-analysis/tree/master/Modules) directory. The root folder has three Python scripts showing different parameter values that a user can configure when generating output charts. [One script](https://github.com/michael-tojong/trading-analysis/blob/master/analyze_forex.py) loops through seven currency pairs for five time frames, ranging from hourly charts to monthly charts. [Another script](https://github.com/michael-tojong/trading-analysis/blob/master/analyze_bitcoin.py) analyzes Bitcoin price data for two time frames at three different short-term and long-term windows. [The last file](https://github.com/michael-tojong/trading-analysis/blob/master/generate_single_chart.py) can be executed from a command-line interface to generate a single chart, requiring two arguments for specifying the asset to analyze along with the time frame. Generating one chart generally takes a few seconds to complete.

When a chart is generated, the code checks for whether there are data sets available within the time frame that the analysis requires. If not, an API call is made for the data broker to provide the required files (see the next section [Data Set](https://github.com/michael-tojong/trading-analysis#data-set)), and a _Datasets_ directory is created if it's not already available for the files to be stored under. After an analysis is made, a .jpg file of the output chart is placed under the _Analyses_ directory. This directory is also automatically created if it's not already available.
<br>
<br>
## Data Set

Data used in this project require the following columns:
- Date & time
- Open price
- High price
- Low price
- Close price

Files should be in .csv format and named as `[asset] [time frame].csv`. For example, a file having hourly data tracking the Canadian Dollar-Australian Dollar currency pair should be named as `CAD_AUD H1.csv`.

For the forex analyses, I had an account with OANDA, a retail broker that also provides an API for accessing financial data. The modules under the [*Oanda*](https://github.com/michael-tojong/trading-analysis/tree/master/Modules/Oanda) sub-directory are used to conduct an automated analysis while ingesting and storing required data from OANDA. If there are existing price data points for the currency pair stored under the _Datasets_ directory, the ingestion appends the data coming after the last time stamp that was stored into the existing .csv file. It also parses relevant columns and prepares the data file structure for the analysis modules to further consume.

Using OANDA's API requires an authentication token tied to the user account ([see here for their documentation](https://developer.oanda.com/rest-live-v20/authentication/)). To use the modules under the *Oanda* sub-directory, the token needs to be provided via the `get_oanda_candles` function.

Data not sourced from OANDA can be placed under the _Datasets_ directory, as long as the files conform to the format mentioned above. A _Datasets_ directory also needs to be created if it doesn't already exist in the root folder. From there, the module under the [*Manual*](https://github.com/michael-tojong/trading-analysis/tree/master/Modules/Manual) sub-directory can be used to conduct an analysis. An [example usage script](https://github.com/michael-tojong/trading-analysis/blob/master/analyze_bitcoin.py) analyzes hourly and daily Bitcoin data obtained manually.
<br>
<br>
## Generating Candlestick Charts via Matplotlib

As seen in the [`candlePlotting` module](https://github.com/michael-tojong/trading-analysis/blob/master/Modules/candlePlotting.py), I utilized functions from Matplotlib's [mplfinance](https://github.com/matplotlib/mplfinance) library to plot the *open-high-low-close* financial charts provided in the analysis outputs. Additionally, a trained convolutional neural network (CNN) model [analyzes each individual candlestick's shape](https://github.com/michael-tojong/trading-analysis/blob/d12ffdac87807ec84fcc6d9712f0e7702fe5812b/Modules/features.py#L14). In order to do so, one image for each candle is separately generated using the same function that I use to plot the financial charts in the outputs.
<br>
<br>
## Feature Engineering

The [`features` module](https://github.com/michael-tojong/trading-analysis/blob/master/Modules/features.py) contains code to derive features from the OHLC price data. This module has fourteen functions that each compute a number of such features. Each individual candle is analyzed via the [`new_datetime_alpha`](https://github.com/michael-tojong/trading-analysis/blob/ba33566f0fe3f0d312c3600e976dff47210cc701/Modules/features.py#L215) and [`new_datetime_complete`](https://github.com/michael-tojong/trading-analysis/blob/ba33566f0fe3f0d312c3600e976dff47210cc701/Modules/features.py#L244) functions, which in turn call the feature engineering functions.

Features include whether a candlestick's shape signifies an important price action behavior, whether a price point had displayed a rejection (see the [Output](https://github.com/michael-tojong/trading-analysis#output) section), what directions the short-term and long-term trends are going in, and whether price is near a trend's middle "control" or near its upper and lower limits. For a full list of the features that I had the automated analysis track, see this [reference file](https://github.com/michael-tojong/trading-analysis/blob/master/Reference/Features%20on%20df_window.txt).
<br>
<br>
## Implementing a Convolutional Neural Network

One feature that the automated analysis considers is the shape of a candlestick. To get this shape, an image is generated using the `mplfinance.candlestick2_ohlc` function to plot data from one timestamp. For the trained CNN model to interpret this image, a greyscale copy of the image is then converted into a NumPy array, where each element represents a pixel with a percentage value ranging from 0 (a white pixel) to 1 (a black pixel).

This [reference file](https://github.com/michael-tojong/trading-analysis/blob/master/Reference/Single%20candle%20patterns%20(categories).txt) tracks candlestick shapes signifying important price action behaviors that the automated analysis considers. Some of the shapes signify a price rejection, and as mentioned in the [Output](https://github.com/michael-tojong/trading-analysis#output) section, the purple dashed lines in the analysis outputs can signify these types of price rejections.

To train the CNN model, I [wrote a program](https://github.com/michael-tojong/trading-analysis/blob/master/Modules/CNN/make%20y%20candles.py) to loop through hundreds of timestamps from an initial subset of price data for me to work with. The program generated financial charts for me to inspect, while asking me to label each candle's shape. Having a training set with human-provided labels, I then used objects from the [keras](https://keras.io/) library to [architect a CNN model](https://github.com/michael-tojong/trading-analysis/blob/master/Modules/CNN/generate_model.py). Before fitting the model to the training data, I represented each candle as a NumPy array in the same manner mentioned above. I also tested each CNN model that I trained by having it analyze each candle from a test set and then generating an image of the candle with the label that the model predicted. I validated each image and iteratively re-architected the CNN model until I got satisfactory results.

This [reference plot](https://github.com/michael-tojong/trading-analysis/blob/master/Reference/standard%20plot%201.png) shows a generated financial chart with color-coded vertical highlights on any candle that the CNN model predicted as having a [relevant candlestick shape](https://github.com/michael-tojong/trading-analysis/blob/master/Reference/Single%20candle%20patterns%20(categories).txt).
<br>
<br>
## Using Regression to Draw Trendlines

To draw the short-term and long-term trendlines, I used [scikit-learn's LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) algorithm to [fit a regression model](https://github.com/michael-tojong/trading-analysis/blob/ba33566f0fe3f0d312c3600e976dff47210cc701/Modules/features.py#L142) to the closing price values. The middle "control" line is the output of the values that the regression model predicted for each timestamp. The upper and lower limits result from the bounds of a confidence interval dynamically computed from the regression model's predicted values. Refer to the [Output](https://github.com/michael-tojong/trading-analysis#output) section for how to identify these trendlines.
<br>
<br>
## Using Mean Shift Clustering to Draw Support and Resistance Levels

For the support and resistance levels, I utilized [scikit-learn's MeanShift](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html) algorithm to [fit a clustering model](https://github.com/michael-tojong/trading-analysis/blob/ba33566f0fe3f0d312c3600e976dff47210cc701/Modules/features.py#L92) finding price points with high activity. The model's hyperparameter values depend on whether it would predict the short-term or long-term price levels. The automated analysis smooths out the levels by averaging price points that border each other and removing any short-term levels that are too close to the major long-term levels. Refer to the [Output](https://github.com/michael-tojong/trading-analysis#output) section for how to identify these support and resistance levels.
