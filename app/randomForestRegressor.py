from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
import pandas as pd
import numpy as np
from data import dataHelper 

FILE = "data.csv"

class model():
    
    
    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.model = RandomForestRegressor(n_estimators=100, random_state=42) 
        self.scaler = StandardScaler()
        self.cols_std_unstd = ['open', 'high', 'low', 'close']
        self.original_values = None # used to store the original values of open, high, low and close
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scalerName = 'scaler.pk1' # name of the scaler file to save
        self.modelName = 'model.pk1' # name of the model file to save
        
        
    # ---------------------------------------------------------------------------------------------------------------
    # These functions are used for training
    
    # used to populate the data_frame from data.csv
    def prepareData(self):
        self.data_frame['predictedClose'] = 0.0
        self.data_frame['predictedClose'] = self.data_frame['predictedClose'].astype(float)
        
        
        X = self.data_frame[
            ['open', 
             'high', 
             'low', 
             'volume', 
             'SMA5', 
             'SMA20', 
             'SMA60',
             'SMADirection_5_20',
             'SMADirection_20_60',
             'Momentum_5',
             'MomentumDirection_5',
             'Momentum_20',
             'MomentumDirection_20',
             'Momentum_60',
             'MomentumDirection_60'
            ]].fillna(0)  # Filling NaN values
        
        y = self.data_frame['close']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.8, random_state=42)


    # Train and save the model
    def train(self):
        self.model.fit(self.X_train, self.y_train)
        self.saveModel()
    
    
    # evaluate the model
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = root_mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        output = (
            "\n--------------------------------------------------------\n"
            "Model Evaluation:\n\n"
            f"Mean Squared Error: {mse}\n"
            f"Root Mean Squared Error: {rmse}\n"
            f"R2 Score: {r2}\n"
            f"Mean Absolute Error: {mae}\n"
            "--------------------------------------------------------\n"
            )
        return output
    
    
    # predict a closing price
    def predict(self, new_data):
        predicted_closing_price = self.model.predict(new_data)
        return predicted_closing_price
    
    
    # backtest the model on the data
    def backtest(self, filename):
        df = self.data_frame
        
        # Define columns 
        columns = [
            'open', 'high', 'low', 'volume', 
            'SMA5', 'SMA20', 'SMA60', 
            'SMADirection_5_20', 'SMADirection_20_60', 
            'Momentum_5', 'MomentumDirection_5', 
            'Momentum_20', 'MomentumDirection_20', 
            'Momentum_60', 'MomentumDirection_60'
        ]
        
        # Iterate through each row 
        for index, row in df.iterrows():
            # Prepare data as a dataframe
            new_data = pd.DataFrame([[
                row['open'], row['high'], row['low'], row['volume'], 
                row['SMA5'], row['SMA20'], row['SMA60'], 
                row['SMADirection_5_20'], row['SMADirection_20_60'], 
                row['Momentum_5'], row['MomentumDirection_5'], 
                row['Momentum_20'], row['MomentumDirection_20'], 
                row['Momentum_60'], row['MomentumDirection_60']
            ]], columns=columns).fillna(0)

            # Predict the closing price for the current row and update the dataframe
            predicted_close = self.predict(new_data)
            df.at[index, 'predictedClose'] = predicted_close[0] if predicted_close else None

        # Save the DataFrame to CSV 
        df.to_csv(filename)
        
        
    # ---------------------------------------------------------------------------------------------------------------
    # These functions are used for saving and loading the model and saving the unstandardised data
    
    
    # save the scaler
    def saveScaler(self):
        os.makedirs('model_data', exist_ok=True)
        scaler_filename = os.path.join('model_data', self.scalerName)
        joblib.dump(self.scaler, scaler_filename)
        print(f"Trained scaler saved to {scaler_filename}")
    
    
    # load the scaler
    def loadScaler(self):
        scaler_filename = os.path.join('model_data', self.scalerName)
        self.scaler = joblib.load(scaler_filename)
        print(f"Scaler loaded from {scaler_filename}")
    
    
    # save the model
    def saveModel(self):
        os.makedirs('model_data', exist_ok=True)
        model_filename = os.path.join('model_data', self.modelName)
        joblib.dump(self.model, model_filename)
        print(f"Trained model saved to {model_filename}")
    
    
    # load the model
    def loadModel(self):
        model_filename = os.path.join('model_data', self.modelName)
        self.model = joblib.load(model_filename)
        print(f"Model loaded from {model_filename}")
    
    
    # save the unstandardised data
    def setUnstandardizedData(self, data):
        cols_to_save = self.cols_std_unstd
        self.original_values = data[cols_to_save].copy()


    # get the unstandardised data 
    # primarily used for testing
    def getUnstandardizedData(self):
        return self.original_values.head()
    
    def getStandardizedData(self):
        needed_columns = ['open', 'high', 'low', 'close']  # Modify this list as needed
    
        # Return only the first row with the specified columns
        return self.data_frame[needed_columns].iloc[0:1]
        
        
    # ---------------------------------------------------------------------------------------------------------------
    # These functions are used for loading, using the trained model, and unstandardizing the data once backtesting is 
    # done to make it more human readable
        
        
    # unstandardize the data
    def unstandardiseData(self, filename):
        # Ensure original values are available
        if self.original_values is None:
            raise ValueError("Original values are not available to revert standardization.")

        # Iterate through each row to unstandardize the predictedClose
        for index, row in self.data_frame.iterrows():
            # Extract the standardized row values from the original data
            standardized_row = [
                row['open'], row['high'], row['low'], row['close']
            ]

            # Use the scaler to unstandardize the predictedClose based on the standardized_row
            predicted_close_standardized = self.data_frame.at[index, 'predictedClose']
            
            predicted_close_unstandardized = self.scaler.inverse_transform([standardized_row])[0][0]

            # Update the predictedClose column with the unstandardized value
            self.data_frame.at[index, 'predictedClose'] = predicted_close_unstandardized
            
            # Columns to unstandardize ['open', 'high', 'low', 'close']
            self.data_frame[self.cols_std_unstd] = self.original_values[self.cols_std_unstd].copy()
            
        self.data_frame.to_csv(filename)


    def unstandardiseVal(self, predicted_close):
        # Ensure predicted_close is a 2D array (one row, one feature)
        predicted_close_2d = np.array([[predicted_close]])  # Shape: (1, 1)

        # Create a row with 4 columns, where the first 3 columns are placeholders (e.g., zeros)
        # The 4th column will contain the predicted close value
        unstandardized_row = np.zeros((1, 4))  
        unstandardized_row[0, 3] = predicted_close_2d[0, 0]  # Place predicted close in the 4th column (close)

        # Use the scaler to unstandardize the entire row
        unstandardized_value = self.scaler.inverse_transform(unstandardized_row)

        # Return the unstandardized value of the predicted close (the 4th column)
        return unstandardized_value[0, 3]  # Extract the unstandardized predicted close


    # standardise the data
    def standardiseData(self, train):
        
        # train is a flag to indicate whether this is a training run or a testing run 
        # if it is a training run, we iniitialize a new scaler and save it 
        if train:
            self.scaler = StandardScaler()
            print("Scaler created.")
        
        # Select columns to standardize ['open', 'high', 'low', 'close']
        cols_to_standardize = self.cols_std_unstd
        
        self.original_values[self.cols_std_unstd] = self.data_frame[self.cols_std_unstd].copy()
        
        # Apply scaler to specified columns
        self.data_frame[cols_to_standardize] = self.scaler.fit_transform(self.data_frame[cols_to_standardize])
        
        # save the scaler
        if train:
            self.saveScaler()
        
        
# ---------------------------------------------------------------------------------------------------------------

    