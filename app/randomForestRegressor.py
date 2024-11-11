from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
import pandas as pd

class model():
    
    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.model = RandomForestRegressor(n_estimators=100, random_state=42) 
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scalerName = 'scaler.pk1' # name of the scaler file to save
        self.modelName = 'model.pk1' # name of the model file to save
        
    
    # used to populate the data_frame from data.csv
    def prepareData(self):
        self.data_frame['predictedClose'] = 0.0
        self.data_frame['predictedClose'] = self.data_frame['predictedClose'].astype(float)
        
        # Select columns to standardize
        cols_to_standardize = ['open', 'high', 'low', 'close']
        self.scaler = StandardScaler()
        
        # Apply scaler to specified columns
        self.data_frame[cols_to_standardize] = self.scaler.fit_transform(self.data_frame[cols_to_standardize])
        
        # save the scaler
        self.saveScaler()
        
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

    # Train the model
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
            "--------------------------------------------------------\n"
            f"Mean Squared Error: {mse}\n"
            f"Root Mean Squared Error: {rmse}\n"
            f"R2 Score: {r2}\n"
            f"Mean Absolute Error: {mae}\n"
            "--------------------------------------------------------"
            )
        return output
    
    def predict(self, new_data):
        predicted_closing_price = self.model.predict(new_data)
        return predicted_closing_price
    
    def backtest(self, filename):
        df = self.data_frame
        
        # Define columns once to avoid repetition
        columns = [
            'open', 'high', 'low', 'volume', 
            'SMA5', 'SMA20', 'SMA60', 
            'SMADirection_5_20', 'SMADirection_20_60', 
            'Momentum_5', 'MomentumDirection_5', 
            'Momentum_20', 'MomentumDirection_20', 
            'Momentum_60', 'MomentumDirection_60'
        ]
        
        # Iterate through each row for backtesting
        for index, row in df.iterrows():
            # Prepare data for prediction as a DataFrame
            new_data = pd.DataFrame([[
                row['open'], row['high'], row['low'], row['volume'], 
                row['SMA5'], row['SMA20'], row['SMA60'], 
                row['SMADirection_5_20'], row['SMADirection_20_60'], 
                row['Momentum_5'], row['MomentumDirection_5'], 
                row['Momentum_20'], row['MomentumDirection_20'], 
                row['Momentum_60'], row['MomentumDirection_60']
            ]], columns=columns).fillna(0)

            # Predict the closing price for the current row and check if prediction is in expected format
            predicted_close = self.predict(new_data)
            df.at[index, 'predictedClose'] = predicted_close[0] if predicted_close else None

        # Save the DataFrame to CSV once after the loop
        df.to_csv(filename)
        
    def saveScaler(self):
        os.makedirs('model_data', exist_ok=True)
        scaler_filename = os.path.join('model_data', self.scalerName)
        joblib.dump(self.scaler, scaler_filename)
        print(f"Scaler saved to {scaler_filename}")
        
    def loadScaler(self):
        scaler_filename = os.path.join('model_data', self.scalerName)
        self.scaler = joblib.load(scaler_filename)
        print(f"Scaler loaded from {scaler_filename}")
        
    def saveModel(self):
        os.makedirs('model_data', exist_ok=True)
        model_filename = os.path.join('model_data', self.modelName)
        joblib.dump(self.model, model_filename)
        print(f"Model saved to {model_filename}")
        
    def loadModel(self):
        model_filename = os.path.join('model_data', self.modelName)
        self.model = joblib.load(model_filename)
        print(f"Model loaded from {model_filename}")
