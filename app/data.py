import json
import requests
import pandas as pd
from io import StringIO
import os
import matplotlib.pyplot as plt

class dataHelper():
        
    def __init__(self, outputfile, api_key=None):
        self.api_key = api_key if api_key else self.getKey()
        self.FILENAME = outputfile
        
    def getKey(self):
        # Load API key from config.json in the parent directory
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
        try:
            with open(config_path) as f:
                config = json.load(f)
                return config.get("alphaVantageKey")
        except FileNotFoundError:
            print("config.json not found.")
            return None
        except json.JSONDecodeError:
            print("Error decoding JSON in config.json.")
            return None
        
    def getDaily(self, symbol, period="full", type="csv"):
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.api_key}&outputsize={period}&datatype={type}"
        response = requests.get(url)
        if response.status_code == 200:
            # Convert the response text into a DataFrame
            data = pd.read_csv(StringIO(response.text))
            
            # Write the DataFrame to the file (FILENAME)
            data.to_csv(self.FILENAME, index=False)  

            print(f"Data for {symbol} has been written to {self.FILENAME}")
            return data
        else:
            print(f"Failed to retrieve data for {symbol}. Status code: {response.status_code}")
            return None
    
    def calculateSMA(self, period, feature):
        full_data = pd.read_csv(self.FILENAME, index_col='timestamp', parse_dates=True)
        data = full_data[feature].to_frame()
        
        data[f'SMA{period}'] = data[feature].rolling(period).mean()
        
        full_data = full_data.join(data[f'SMA{period}'])
        full_data.loc[:full_data.index[period-1], f'SMA{period}'] = 0
        
        full_data.to_csv(self.FILENAME)
        
        return full_data
    
    
    def calculateSMADirection(self, SMA1, SMA2):
        # 1 = up, 0 = no change, -1 = down
        full_data = pd.read_csv(self.FILENAME, index_col='timestamp', parse_dates=True)
        full_data[f'SMADirection_{SMA1}_{SMA2}'] = 0
        
        first_non_zero_index = full_data.loc[full_data[f'SMA{SMA2}'] != 0].index[0]
        start_row = full_data.loc[first_non_zero_index:].index[0]
        
        full_data.loc[start_row:, f'SMADirection_{SMA1}_{SMA2}'] = 0  # Initialize the column from the start row
        full_data.loc[start_row:, f'SMADirection_{SMA1}_{SMA2}'] = (
            (full_data.loc[start_row:, f'SMA{SMA1}'] > full_data.loc[start_row:, f'SMA{SMA2}']).astype(int) -
            (full_data.loc[start_row:, f'SMA{SMA1}'] < full_data.loc[start_row:, f'SMA{SMA2}']).astype(int)
        )
        
        # assign the actual direction values
        full_data.loc[full_data[f'SMADirection_{SMA1}_{SMA2}'] == 1, f'SMADirection_{SMA1}_{SMA2}'] = 1  # SMA2 > SMA1 -> up
        full_data.loc[full_data[f'SMADirection_{SMA1}_{SMA2}'] == -1, f'SMADirection_{SMA1}_{SMA2}'] = -1  # SMA2 < SMA1 -> down
        
        full_data.to_csv(self.FILENAME)
        
        return full_data
        
    def calculateMomentum(self, period):
        # 1 = up, 0 = no change, -1 = down
        full_data = pd.read_csv(self.FILENAME, index_col='timestamp', parse_dates=True)
        full_data[f'Momentum_{period}'] = 0.0
        full_data[f'Momentum_{period}'] = full_data[f'Momentum_{period}'].astype('float64')
        
        for i in range(period-1, len(full_data)):
            close_today = full_data.iloc[i]['close']
            close_period_ago = full_data.iloc[i - (period-1)]['close']
            momentum = (close_today - close_period_ago) / close_today
            full_data.at[full_data.index[i], f'Momentum_{period}'] = float(momentum)
        
        full_data.to_csv(self.FILENAME)
        return full_data
    
    def calculateMomentumDirection(self, period):
        # if Momentum_{period} > 0, then 1, if < 0, then -1, else 0
        full_data = pd.read_csv(self.FILENAME, index_col='timestamp', parse_dates=True)
        full_data[f'MomentumDirection_{period}'] = 0
        
        full_data.loc[full_data[f'Momentum_{period}'] > 0, f'MomentumDirection_{period}'] = 1
        full_data.loc[full_data[f'Momentum_{period}'] < 0, f'MomentumDirection_{period}'] = -1
        
        full_data.to_csv(self.FILENAME)
        return full_data
    
    def prepareData(self, symbol, period="full", type="csv"):
        print("-----------------------------------------------------------------------------")
        print(f"Preparing data for {symbol}:\n")
        
        # get the data from the alphavantage API
        try:
            self.getDaily(symbol, period, type)
            print(f"Data for {symbol} has been fetched from AlphaVantage API and written to {self.FILENAME}")
        except Exception as e:
            print(f"Error: {e}")
            pass
        
        # add features to the data
        # calculate the SMA for 5 days
        try:
            self.calculateSMA(5, 'close')
            print(f"SMA for 5 days has been calculated and written to {self.FILENAME}")
        except Exception as e:
            print(f"Error: {e}")
            pass
        
        # calculate the SMA for 20 days
        try:
            self.calculateSMA(20, 'close')
            print(f"SMA for 20 days has been calculated and written to {self.FILENAME}")
        except Exception as e:
            print(f"Error: {e}")
            pass
        
        # calculate the SMA for 60 days
        try:
            self.calculateSMA(60, 'close')
            print(f"SMA for 60 days has been calculated and written to {self.FILENAME}")
        except Exception as e:
            print(f"Error: {e}")
            pass
        
        # calculate the SMA direction for 5 and 20 days
        try:
            self.calculateSMADirection(5, 20)
            print(f"SMA direction for 5 and 20 days has been calculated and written to {self.FILENAME}")
        except Exception as e:
            print(f"Error: {e}")
            pass
        
        # calculate the SMA direction for 20 and 60 days
        try:
            self.calculateSMADirection(20, 60)
            print(f"SMA direction for 20 and 60 days has been calculated and written to {self.FILENAME}")
        except Exception as e:
            print(f"Error: {e}")
            pass
        
        # calculate the momentum for 5 days
        try:
            self.calculateMomentum(5)
            print(f"Momentum for 5 days has been calculated and written to {self.FILENAME}")
        except Exception as e:
            print(f"Error: {e}")
            pass
        
        # calculate the momentum direction for 5 days
        try:
            self.calculateMomentumDirection(5)
            print(f"Momentum direction for 5 days has been calculated and written to {self.FILENAME}")
        except Exception as e:
            print(f"Error: {e}")
            pass
        
        # calculate the momentum for 20 days
        try:
            self.calculateMomentum(20)
            print(f"Momentum for 20 days has been calculated and written to {self.FILENAME}")
        except Exception as e:
            print(f"Error: {e}")
            pass
        
        # calculate the momentum direction for 20 days
        try:
            self.calculateMomentumDirection(20)
            print(f"Momentum direction for 20 days has been calculated and written to {self.FILENAME}")
        except Exception as e:
            print(f"Error: {e}")
            pass
        
        # calculate the momentum for 60 days
        try:
            self.calculateMomentum(60)
            print(f"Momentum for 60 days has been calculated and written to {self.FILENAME}")
        except Exception as e:
            print(f"Error: {e}")
            pass
        
        # calculate the momentum direction for 60 days
        try:
            self.calculateMomentumDirection(60)
            print(f"Momentum direction for 60 days has been calculated and written to {self.FILENAME}")
        except Exception as e:
            print(f"Error: {e}")
            pass
        
        print(f"Data preparation for {symbol} is complete, and the data has been written to {self.FILENAME}")
        print("-----------------------------------------------------------------------------\n")
        
    def loadDataToDF(self):
        try:
            data = pd.read_csv(self.FILENAME, index_col='timestamp', parse_dates=True)
            print(f"Data successfully loaded from {self.FILENAME}.\n")
            return data
        except FileNotFoundError:
            print(f"Error: File {self.FILENAME} not found.\n")
            return None
        except pd.errors.EmptyDataError:
            print(f"Error: The CSV file {self.FILENAME} is empty.\n")
            return None
        except pd.errors.ParserError:
            print(f"Error: There was a parsing issue with the CSV file {self.FILENAME}.\n")
            return None
        
    def plotBacktestedData(self, image_file='backtested_plot.png'):
        df = pd.read_csv(self.FILENAME, parse_dates=[0], index_col=0)

        # Plot the Close and test_close columns
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Actual Close', color='blue')
        plt.plot(df.index, df['predictedClose'], label='Predicted Close', color='orange')

        # Adding titles and labels
        plt.title('Actual vs Predicted Closing Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        #save and show the plot
        plt.savefig(image_file)
        print(f"Plot saved as {image_file}")
        plt.show()
    
    
         
