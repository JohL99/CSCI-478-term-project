import sys
import matplotlib
import numpy as np
import pandas as pd
from randomForestRegressor import model
from data import dataHelper

class StockPredictionCommandLine:
    def __init__(self, trainTicker, DATA_CSV):
        #matplotlib.use('Agg')
        self.dh = dataHelper(outputfile="data.csv")
        self.mdl = None
        self.trainTicker = trainTicker
        self.TRAIN_BACKUP_CSV = "backup_data_train.csv"
        self.TEST_BACKUP_CSV = "backup_data_test.csv"
        self.DATA_CSV = DATA_CSV

        if not self.dh.api_key:
            print("API key is missing. Please provide your Alpha Vantage API key.")
            sys.exit(1)

    def display_welcome_message(self):
        print("\nWelcome to Stock Locker!")
        print("You can either train a new model or make a prediction.")
        print("Please choose an option:")
        print("1. Train a new model")
        print("2. Test an existing model")
        print("3. Make a prediction")
        print("4. Help")
        print("5. Quit")

    def get_user_choice(self):
        while True:
            try:
                choice = int(input("Enter your choice (1/2/3/4/5): "))
                if choice in [1, 2, 3, 4, 5]:
                    return choice
                else:
                    print("Invalid input. Please enter 1, 2, 3, 4, or 5.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
                
    def getTicker(self):
        while True:
            ticker = input("Enter the stock code (e.g., MSFT): ").upper()
            if not ticker.isalnum():
                print("Invalid stock code format. Please enter a valid stock code (e.g., MSFT).")
                continue
            return ticker
        
    def getPeriod(self):
        while True:
            period = int(input("Enter the period in days (e.g., 1, 5, 20, etc): "))
            if not period > 0:
                print("Invalid period. Please enter a positive number.")
                continue
            return period
                

    def handle_train_model(self):
        ticker = self.getTicker()
        
        if ticker == "QUIT":
            return
        
        print("\nTraining a new model using ", ticker ,"...\n")
        
        # in actual use this would use getDaily() to get the latest data
        # but for testing there is a backup csv file with data that gets copied 
        # due to the daily limit on the API
        self.dh.getDaily(ticker, "full", "csv")
        #self.dh.copyDataFromExisting(self.TRAIN_BACKUP_CSV, self.DATA_CSV)
        self.dh.prepareData(ticker)
        df = self.dh.loadDataToDF()
        print("Data prepared.")
        
        if df is None:
            print("Error: unable to load data from csv file.")
            return
        
        
        mdl = model(df)
        print("Model created.")
        
        mdl.setUnstandardizedData(df)
        print("Original open, high, low, and close values saved.")
        
        mdl.standardiseData(True)
        print("Data standardised.")
        mdl.prepareData()
        
        try:
            mdl.train()
            print("Model trained.")
            print(mdl.evaluate())
        except Exception as e:
            print(f"Error during model training: {e}")
            return
        try:
            print("Backtesting...")
            mdl.backtest(self.DATA_CSV)
            print("Backtesting done.")
            self.dh.plotBacktestedData("Actual vs Predicted Closing Prices - Standardised", 'train_backtested_plot_std.png')
            mdl.unstandardizeData(self.DATA_CSV)
            print(f"Data unstandardised and saved to {self.DATA_CSV}.")
            self.dh.plotBacktestedData("Actual vs Predicted Closing Prices - Normalised",'train_backtested_plot_normal.png') 
            self.mdl = mdl
        except Exception as e:
            print(f"Error during backtesting: {e}")
            return
    
        print("done.")
        
        
    def handle_test_model(self):
        ticker = self.getTicker()
        
        if ticker == "QUIT":
            return
        
        print("\nTesting an existing model using ", ticker ,"...\n")
        
        # in actual use this would use getDaily() to get the latest data
        # but for testing there is a backup csv file with data that gets copied 
        # due to the daily limit on the API
        self.dh.getDaily(ticker, "full", "csv")
        #self.dh.copyDataFromExisting(self.TEST_BACKUP_CSV, self.DATA_CSV)
        self.dh.prepareData(ticker)
        df = self.dh.loadDataToDF()
        print("Data prepared.")
        
        if df is None:
            print("Error: unable to load data from csv file.")
            return
        
        
        mdl = model(df)
        print("Model created.")
        
        mdl.setUnstandardizedData(df)
        print("Original open, high, low, and close values saved.")
        
        mdl.loadModel()
        mdl.loadScaler()
        
        mdl.standardiseData(True)
        print("Data standardised.")
        mdl.prepareData()
        
        try:
            print("Backtesting...")
            mdl.backtest(self.DATA_CSV)
            print("Backtesting done.")
            self.dh.plotBacktestedData("Actual vs Predicted Closing Prices - Standardised", 'train_backtested_plot_std.png')
            mdl.unstandardizeData(self.DATA_CSV)
            print(f"Data unstandardised and saved to {self.DATA_CSV}.")
            self.dh.plotBacktestedData("Actual vs Predicted Closing Prices - Normalised",'train_backtested_plot_normal.png') 
        except Exception as e:
            print(f"Error during backtesting: {e}")
            return
        
        print("done.")

    def handle_make_prediction(self):
        
        ticker = self.getTicker()
            
        if ticker == "QUIT":
            return
    
        # in actual use this would use getDaily() to get the latest data
        # but for testing there is a backup csv file with data that gets copied 
        # due to the daily limit on the API
        self.dh.getDaily(ticker, "full", "csv")
        #self.dh.copyDataFromExisting(self.TEST_BACKUP_CSV, self.DATA_CSV)
        
        
        # load data
        # the new data will override data.csv!
        self.dh.prepareData(ticker)
        df = self.dh.loadDataToDF() 

        if df is None:
            print("Error: unable to load data from csv file.")
            return
            
            
        # make sure model is trained
        if self.mdl is None:
            try:
                mdl = model(df)
                print("Model object created.")

                self.mdl = mdl
                
                self.mdl.setUnstandardizedData(df)
                print("Original open, high, low, and close values saved.")
                
                self.mdl.loadModel()
                self.mdl.loadScaler()
            except Exception as e:
                print(f"Error loading model: {e} - please train a model first.")
                return
            
        print("\nMaking a stock prediction...\n")
        while True:

            try:
                self.mdl.standardiseData(False)
                print("Data standardised.")
                
                self.mdl.prepareData()
                print("Data prepared.")
            except Exception as e:
                print(f"Error during data preparation: {e}")
                return

            try:
                prediction_data = df.iloc[-1:].drop(['close'], axis=1)
                # make sure features exisits
                feature_columns = [
                    'open', 'high', 'low', 'volume', 
                    'SMA5', 'SMA20', 'SMA60', 
                    'SMADirection_5_20',
                    'SMADirection_20_60',
                    'Momentum_5',
                    'MomentumDirection_5',
                    'Momentum_20',
                    'MomentumDirection_20',
                    'Momentum_60',
                    'MomentumDirection_60'
                ]
                missing_features = [col for col in feature_columns if col not in prediction_data.columns]
                if missing_features:
                    print(f"Error: Missing features for prediction: {missing_features}")
                    return

                prediction_data = prediction_data[feature_columns]
                

                if prediction_data.isnull().values.any():
                    print(f"Error: Prediction data for {ticker} contains NaN values. Please try a different stock code.")
                    continue

                if np.isinf(prediction_data.values).any():
                    print(f"Error: Prediction data for {ticker} contains infinite values. Please try a different stock code.")
                    continue
                
                predicted_price = self.mdl.predict(prediction_data)
                
                # Ensure predicted_price is scalar and assign it to the DataFrame
                #predicted_close = predicted_price[0]  # Extract scalar value from predicted price
                
                # Create a new DataFrame with predicted value
                predicted_df = pd.DataFrame({
                    'open': [0],
                    'high': [0],
                    'low': [0],
                    'volume': [0],
                    'predictedClose': [0]
                })
                predicted_df['predictedClose'] = predicted_price                
                
                predicted_price_norm = self.mdl.unstandardizeVal(predicted_price)
                
                print(f"Predicted closing price for {ticker}: {predicted_price_norm:.2f}")
                
            except Exception as e:
                #print(f"Error during prediction: {e}")
                return
            break

    def display_help_message(self):
        print("\nHelp:")
        print("1. Train a new model: This will fetch data for a stock, prepare the data, and train a Random Forest model.")
        print("2. Test an existing model: This will fetch data for a stock, prepare the data and existing model using the new data.")
        print("2. Make a prediction: This will allow you to input a stock code and predict the next closing price based on the trained model.")
        print("3. Quit: Exits the program.")
        print("Please follow the on-screen instructions.")

    def quit_program(self):
        print("Exiting the program. Goodbye!")
        sys.exit()

    def start(self):
        while True:
            self.display_welcome_message()
            choice = self.get_user_choice()

            if choice == 1:
                self.handle_train_model()
            elif choice == 2:
                self.handle_test_model()
            elif choice == 3:
                self.handle_make_prediction()
            elif choice == 4:
                self.display_help_message()
            elif choice == 5:
                self.quit_program()