import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from randomForestRegressor import model


DATA_PATH = "data.json"
DATA_CSV = "data.csv"
ticker = "MSFT"

def fetchAndSaveData():
    msft = yf.Ticker(ticker)
    msftHist = msft.history(period="max")
    msftHist.to_csv(DATA_CSV)

    selectedColumns = ['Open', 'High', 'Low', 'Close']
    msftHistFiltered = msftHist[selectedColumns]

    msftHistFiltered.to_json(DATA_PATH)
    
    #msftHist.to_json(DATA_PATH)

    return msftHistFiltered
    #return msftHist

def saveDataFrameToCSV(df, file_name):
    # Set index=True to keep the datetime index in the file
    df.to_csv(file_name, index=True)  
    print(f"DataFrame saved to {file_name}")


def plotBacktestedData(csv_file, image_file='backtested_plot.png'):
    df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)

    # Plot the Close and test_close columns
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Actual Close', color='blue')
    plt.plot(df.index, df['test_close'], label='Predicted Close', color='orange')

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


if __name__ == "__main__":
    print("starting...\n")
    
    # Get data from yfinance and save it to a JSON file
    fetchAndSaveData()
    
    # Create an instance of the model
    reg_model = model(DATA_PATH)
    
    # Load and preprocess the data
    df = reg_model.load_data()
    reg_model.preprocess_data(df)
    
    # Train the model
    reg_model.train()
    
    # Evaluate the model
    mse = reg_model.evaluate()
    print(f"Mean Squared Error: {mse}\n")
    
    # Backtest the model to predict closing prices for the past data
    df = reg_model.backtest(df)
    saveDataFrameToCSV(df[['Open', 'Close', 'test_close']], "backtested_data.csv")
    
    # Create a DataFrame with sample features including new ones (with placeholders for testing)
    example_features = pd.DataFrame([[430, 435, 420, 431, 429, 0.002, 1.5]],
                                    columns=['Open', 'High', 'Low', 'Moving_Avg_5', 'Moving_Avg_10', 'Daily_Return', 'Volatility'])
    
    predicted_closing_price = reg_model.predict(example_features)
    print(f"Predicted Closing Price: {predicted_closing_price[0]}\n")
    
    
    opening_price = example_features['Open'][0]
    predicted_closing_price_value = predicted_closing_price.item()
    price_change = abs(predicted_closing_price_value - opening_price)
    
    if predicted_closing_price_value < opening_price:
        print(f"Price went down: Opening Price = {opening_price}, Predicted Closing Price = {predicted_closing_price_value}, Change = {price_change:.2f}")
    elif predicted_closing_price_value > opening_price:
        print(f"Price went up: Opening Price = {opening_price}, Predicted Closing Price = {predicted_closing_price_value}, Change = {price_change:.2f}")
    else:
        print(f"No price change: Opening Price = {opening_price}, Predicted Closing Price = {predicted_closing_price_value}")

    
    plotBacktestedData('backtested_data.csv', 'backtested_plot.png')


  