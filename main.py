# the web framework
import streamlit as st 
import pandas as pd

# for forecasting / prediction
from statsmodels.tsa.statespace.sarimax import SARIMAX

# yahoo finance API/ package for getting stock data
import yfinance as yf

# a figure / graph plotting package
from plotly import graph_objs as go

# date time package 
from datetime import date

# additional imports for differencing the time series data
from statsmodels.tsa.statespace.tools import diff

# setting up a start date and getting the current date
# this is for the stocks
START = "2012-01-01" 
TODAY = date.today().strftime("%Y-%m-%d")

# choosing stocks that we want to display / deal with 
unsortedStocksCodes = ("JNJ", "GOOG", "AAPL", "BRK-A", "AMZN", "MSFT", "JPM", "NFLX", "META", "BAC", "GME", "MCD", "KO")
unsortedStocksList = ("Johnson & Johnson", "Alphabet Inc Class C", "Apple Inc", "Berkshire Hathaway Inc. Class A", 
                      "Amazon.com, Inc.", "Microsoft Corporation", "JPMorgan Chase & Co", "Netflix Inc", "Meta Platforms Inc", 
                      "Bank of America Corp", "GameStop Corp.", "McDonald's Corp", "Coca-Cola Co")
stocks_dictionary = dict(zip(list(unsortedStocksList), list(unsortedStocksCodes)))
stocks = sorted(unsortedStocksList)
stocks = tuple(stocks)


@st.cache
def load_stock_data(stock_code):
    """
    The "@st.cache" helps for the data to be present already if the data was selected before and is being selected again
    The function loads stock data through the yahoo finance library 
    
    Args:
        stock_code (string): a code or abbreviation for identifying a stock
        
    Returns: the data retrieved from yahoo finance
    """
    # this will give a pandas dataframe for a particular stock with the specified start date and end date
    data = yf.download(stock_code, START, TODAY)
    # the reset_index method will not create a new DataFrame. 
    # Instead, it will directly modify and overwrite the original DataFrame
    data.reset_index(inplace=True)
    return data


def plot_data(stock_data):
    """
    Creates scatter plots for a particular stock
    The y axis contains the opening and closing price and the x axis contains the date/time
    """
    st.subheader("Plotting Scatter Plots")
    # creating a plotly graph object figure
    figure = go.Figure()
    
    # plotting 2 scatter plot lines on one graph
    # the first box contains the graph and is the plot
    # the second box is the slider
    # we have appropriate labels and titles
    figure.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Open'], name='Open Price'))
    figure.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name='Closing Price'))
    #figure.layout.update(title_text="Plotting Scatter Plots", xaxis_rangeslider_visible=True)
    figure.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(figure)
    
    
def main():
    # giving the web app a title and a subheader
    st.title("The Stock Forecast App")
    st.subheader("A Web Application for Stock Forecast\n")
    st.markdown("""
    **Source Code:** [https://github.com/g4ze/Stock-Prediction-Web-App](https://github.com/g4ze/stock-predictor).
    
    **Instructions-**
    * Select a Stock
    * Select the number of years for prediction
    * Wait for the forecast to happen!
    """)
    # for adding space
    for i in range(3):
        st.text("")

    # creating a dropdown box for user selection
    # whenever the company name is selected in the dropdown, its looked up in the dictionary to find the stocks code
    dropdown_box_selection = st.selectbox("\nSelect a stock for prediction\n", stocks)
    dropdown_box_selection_stock_code = stocks_dictionary[dropdown_box_selection]

    # creating a slider for selecting the number of years of stock data to predict
    # calculating the no. of days based on the slider selection
    n_years = st.slider("\nYears of prediction", 1, 10)
    period = n_years * 365

    # for adding space
    for i in range(3):
        st.text("")

    # loads the data and has placeholders for before and after loading the data
    # loads the data through load_stock_data() which we made
    s = "Loading data for " + dropdown_box_selection + "..." 
    stock_data_state = st.text(s)
    stock_data = load_stock_data(dropdown_box_selection_stock_code)    
    s = "Data Loaded for " + dropdown_box_selection + "!"
    stock_data_state.text(s)

    # writes a subheading
    # displays the stock data as a pandas dataframe
    s = "\nDisplaying first 5 rows for the " + "'" + dropdown_box_selection + "'" + " Stock Data"
    st.subheader(s)
    st.write(stock_data.head())
    s = "\nDisplaying last 5 rows for the " + "'" + dropdown_box_selection + "'" + " Stock Data"
    st.subheader(s)
    st.write(stock_data.tail())
    
    # calls the plot_data() function which we made for plotting the data
    plot_data(stock_data)

    # placing placeholders for forecasting
    s = "Forecasting data for " + "'" + dropdown_box_selection + "'" + "..." 
    forecast_state = st.text(s) 

    # Forecasting closing price for stocks using SARIMAX
    # Slicing the columns into a new dataframe and then renaming the columns in it
    # Renaming is necessary for the SARIMAX model
    df_train = stock_data[['Date', 'Close']]
    df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})

    # Ensure the time series is stationary, if needed
    # Uncomment the next line if differencing is required
    # df_train['y'] = diff(df_train['y'], k_diff=1)

    # Creating a SARIMAX object
    model = SARIMAX(df_train['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

    # Fitting the model
    results = model.fit(disp=False)

    # Doing the forecast
    forecast_values = results.get_forecast(steps=period).predicted_mean

    # Creating a future dataframe for the forecast
    forecast_index = pd.date_range(start=df_train['ds'].iloc[-1] + pd.DateOffset(1), periods=period, freq='D')
    forecast_dataframe = pd.DataFrame({'ds': forecast_index, 'y': forecast_values})

    # Displaying the last 5 forecast / predicted rows
    s = "\nForecast Data for " + "'" + dropdown_box_selection + "'" + " Stock Data"
    st.subheader(s)
    st.write(forecast_dataframe.tail())

    # Plotting the forecast we got
    st.text("")
    st.subheader('\n\n\nPlotting Forecast Data')
    st.write('ds here means datestamp and y is the stock price')
    figure1 = go.Figure()
    figure1.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], name='Observed', mode='lines'))
    figure1.add_trace(go.Scatter(x=forecast_dataframe['ds'], y=forecast_dataframe['y'], name='Forecast', mode='lines', line=dict(dash='dash')))
    st.plotly_chart(figure1)

    # Placing placeholders for forecasting
    s = "Forecasting done for " + "'" + dropdown_box_selection + "'" + "!"
    forecast_state.text(s)


if __name__ == '__main__':
    main()
