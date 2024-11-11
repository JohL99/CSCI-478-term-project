import pandas as pd
from pytictoc import TicToc
from randomForestRegressor import model
from data import dataHelper

DATA_PATH = "data.json"
DATA_CSV = "data.csv"
ticker = "MSFT"

def main(): 
    print("starting...\n")
    
    #timer = TicToc()
    
    dh = dataHelper(outputfile=DATA_CSV)
    dh.prepareData(ticker, "full", "csv")
    
    df = dh.loadDataToDF()
    
    if df is None:
        print("Error: unable to load data from csv file.")
        exit()
    
    mdl = model(df)
    print("Model created.")
    
    mdl.prepareData()
    print("Data prepared.")
    
    mdl.train()
    print("Model trained.")
    
    print(mdl.evaluate())
    
    print("Backtesting...")
    #timer.tic()
    mdl.backtest(DATA_CSV)
    #timer.toc()
    print("Backtesting done.")
    
    dh.plotBacktestedData('backtested_plot.png')
    
if __name__ == "__main__":
    main()
    
    
    
    """ # Create an instance of the model
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

    
    plotBacktestedData('backtested_data.csv', 'backtested_plot.png') """


  