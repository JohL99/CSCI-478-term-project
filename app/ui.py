import sys
import matplotlib
import numpy as np
import pandas as pd
import datetime as dt
from randomForestRegressor import model
from data import dataHelper

class StockPredictionCommandLine:
    
    
    def __init__(self, DATA_CSV, type):
        #matplotlib.use('Agg')
        self.dh = dataHelper(outputfile="data.csv")
        self.mdl = None
        self.type = type
        self.trainTicker = "MSFT"
        self.testTicker = "BA"
        self.predictTicker = "AAPL"
        self.TRAIN_BACKUP_CSV = "backup_data_train.csv"
        self.TEST_BACKUP_CSV = "backup_data_test.csv"
        self.PREDICT_BACKUP_CSV = "backup_data_predict.csv"
        self.DATA_CSV = DATA_CSV


# -----------------------------------------------------------------------------------------------------------------------------------
# Drivers for the cli

    def start(self):
        while True:
            self.display_welcome_message()
            choice = self.get_user_choice()

            if choice == 1:
                print("\n")
                self.handle_train_model()
            elif choice == 2:
                print("\n")
                self.handle_test_model()
            elif choice == 3:
                print("\n")
                self.handle_make_prediction()
            elif choice == 4:
                print("\n")
                self.display_help_message()
            elif choice == 5:
                print("\n")
                self.quit_program()
    
    
    def display_welcome_message(self):
        print("\nWelcome to Stock Locker!")
        print("You can either train a new model or make a prediction.")
        print("Please choose an option:")
        print("1. Train a new model")
        print("2. Test an existing model")
        print("3. Make a prediction")
        print("4. Help")
        print("5. Quit")
        
        
    def display_help_message(self):
        print("\nHelp:")
        print("1. Train a new model: This will allow you to enter a stock ticker and will fetch data for the stock, prepare the data, and train the model.")
        print("2. Test an existing model: This will allow you to enter a stock ticker and fetch data for the stock, prepare the data and test an existing model using the new data.")
        print("2. Make a prediction: This will allow you to enter a stock ticker and predict the next closing price an existing trained model.")
        print("3. Help: Prints this prompt.")
        print("4. Quit: Exits the program.")
        print("-----------------------------------------------------------------------------")


    def quit_program(self):
        print("Exiting the program. Goodbye!")
        sys.exit()
    
    
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# Getters for user input


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
            
            if ticker == "QUIT":
                self.quit_program()
                
            if not ticker.isalnum():
                print("Invalid stock code format. Please enter a valid stock code (e.g., MSFT).")
                continue
            return ticker
        
        
    def getPeriod(self):
        while True:
            try:
                period = int(input("Enter the period in days you would like to predict for (e.g., 1, 5, 20, etc): "))
                
                if period <= 0:
                    print("Invalid period. Please enter a positive number.")
                    continue
                
                return period
            
            except ValueError:
                print("Invalid input. Please enter a valid number.")
                
                
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# Handlers for user choices


    def handle_train_model(self):
        
        if self.type == 0:
            ticker = self.trainTicker
            self.dh.copyDataFromExisting(self.TRAIN_BACKUP_CSV, self.DATA_CSV)
            print("\nTraining a new model using MSFT...\n")
        else:
            ticker = self.getTicker()
            self.dh.getDaily(ticker, "full", "csv")
            print(f"\n\nTraining a new model using {ticker}...\n")
            
        # Load data
        self.dh.prepareData(ticker)
        df = self.dh.loadDataToDF()
        print("Data prepared.")
        
        if df is None:
            print("Error: unable to load data from csv file.")
            return
        
        # Create model and do necessary steps to prepare data and train model
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
            # backtest the model
            print("Backtesting...")
            mdl.backtest(self.DATA_CSV)
            print("Backtesting done.")
            
            self.dh.plotBacktestedData("Actual vs Predicted Closing Prices - Standardised", 'train_backtested_plot_std.png')
            
            mdl.unstandardiseData(self.DATA_CSV)
            print(f"Data unstandardised and saved to {self.DATA_CSV}.")
            
            self.dh.plotBacktestedData("Actual vs Predicted Closing Prices - Normalised",'train_backtested_plot_normal.png') 
            self.mdl = mdl # save the model locally for future use to avoid the need to load it from the file
        except Exception as e:
            print(f"Error during backtesting: {e}")
            return
    
        print("Done training the model.")
        print("-----------------------------------------------------------------------------")
        
        
    def handle_test_model(self):
        
        if self.type == 0:
            ticker = self.testTicker
            self.dh.copyDataFromExisting(self.TEST_BACKUP_CSV, self.DATA_CSV)
            print("\nTesting an existing model using BA...\n")
        else:
            ticker = self.getTicker()
            self.dh.getDaily(ticker, "full", "csv")
            print(f"\n\nTesting an existing model using {ticker}...\n")
            
        # Load data
        self.dh.prepareData(ticker)
        df = self.dh.loadDataToDF()
        print("Data prepared.")
        
        if df is None:
            print("Error: unable to load data from csv file.")
            return
        
        # Ensure model is trained
        if self.mdl is None:
            # Create model object and do necessary steps to prepare data and load the trained model
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
        
        
        self.mdl.standardiseData(False)
        print("Data standardised.")
        self.mdl.prepareData()
        
        try:
            print("Backtesting...")
            self.mdl.backtest(self.DATA_CSV)
            print("Backtesting done.")
            
            self.dh.plotBacktestedData("Actual vs Predicted Closing Prices - Standardised", 'test_backtested_plot_std.png')
            
            self.mdl.unstandardiseData(self.DATA_CSV)
            print(f"Data unstandardised and saved to {self.DATA_CSV}.")
            
            self.dh.plotBacktestedData("Actual vs Predicted Closing Prices - Normalised",'test_backtested_plot_normal.png') 
        except Exception as e:
            print(f"Error during backtesting: {e}")
            return
        
        print("Done testing model.")
        print("-----------------------------------------------------------------------------")


    def handle_make_prediction(self):
        
        if self.type == 0:
            ticker = self.predictTicker
            self.dh.copyDataFromExisting(self.PREDICT_BACKUP_CSV, self.DATA_CSV)
            period = self.getPeriod()
            print("\n")
        else:
            ticker = self.getTicker()
            period = self.getPeriod()
            self.dh.getDaily(ticker, "full", "csv")
            print(f"\n\nMaking predictions for {ticker}, over {period} days...\n")
            
        # Load data
        self.dh.prepareData(ticker)
        df = self.dh.loadDataToDF()

        if df is None:
            print("Error: unable to load data from CSV file.")
            return

        # Ensure model is trained
        if self.mdl is None:
            # Create model object and do necessary steps to prepare data and load the trained model
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

        try:
            self.mdl.standardiseData(False)
            print("Data standardized.")

            self.mdl.prepareData()
            print("Data prepared.\n")
            
        except Exception as e:
            print(f"Error during data preparation: {e}")
            return
        
        print("\nMaking a stock prediction...\n")
        
        print("-----------------------------------------------------------------------------")
        
        try:
            self.predict_for_period(df, period)
            
        except Exception as e:
            print(f"\nError preparing prediction: {e}")
            return
        
        print("\n-----------------------------------------------------------------------------")


# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# Helper function for making predictions


    def predict_for_period(self, df, period):
            # Prepare the data by excluding the 'close' column and making sure the features exist
            prediction_data = df.iloc[-1:].drop(['close'], axis=1)
            volume = prediction_data['volume'].iloc[0] # Assume volume stays constant
            today = dt.date.today()
            
            # Define the feature columns used for prediction
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
            
            # Check for missing features
            missing_features = [col for col in feature_columns if col not in prediction_data.columns]
            if missing_features:
                print(f"Error: Missing features for prediction: {missing_features}")
                return

            prediction_data = prediction_data[feature_columns]
            
            # Ensure there are no NaN or infinite values
            if prediction_data.isnull().values.any():
                print("Error: Prediction data contains NaN values. Please try a different stock.")
                return
            if np.isinf(prediction_data.values).any():
                print("Error: Prediction data contains infinite values. Please try a different stock.")
                return
            
            # List to store predicted prices and unstandardized values
            predicted_prices = []
            unstandardized_prices = []

            # Iterate over the prediction period
            for _ in range(period):
                
                # Make the prediction
                predicted_price = self.mdl.predict(prediction_data)
                
                # Ensure predicted_price is scalar 
                predicted_close = predicted_price[0]
                
                # Unstandardize the predicted close value
                predicted_price_norm = self.mdl.unstandardiseVal(predicted_close)
                
                # Store the unstandardized value
                predicted_prices.append(predicted_close)
                unstandardized_prices.append(predicted_price_norm)
                
                # Update prediction_data for the next prediction
                prediction_data['open'] = predicted_price_norm
                prediction_data['high'] = max(predicted_close, prediction_data['high'].iloc[0])
                prediction_data['low'] = min(predicted_close, prediction_data['low'].iloc[0])
                prediction_data['volume'] = volume  # Assume volume stays constant
                        
            
            # Print the predicted prices
            print(f"Predicted closing prices for {period} days:\n")
            for i, price in enumerate(unstandardized_prices):
                nDay = today + dt.timedelta(days=i)
                print(f"{nDay}: {price:.2f}")
            
            return unstandardized_prices


# -----------------------------------------------------------------------------------------------------------------------------------

    
                
    
