from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import json
import pandas as pd

class model():
    
    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)  # Use RandomForestRegressor
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepareData(self):
        self.data_frame['predictedClose'] = 0.0
        self.data_frame['predictedClose'] = self.data_frame['predictedClose'].astype(float)
        
        # Select columns to standardize
        cols_to_standardize = ['open', 'high', 'low', 'close']
        scaler = StandardScaler()
        
        # Apply scaler to specified columns
        self.data_frame[cols_to_standardize] = scaler.fit_transform(self.data_frame[cols_to_standardize])
        
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
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize the features
        #self.X_train = self.scaler.fit_transform(self.X_train)
        #self.X_test = self.scaler.transform(self.X_test)

    def train(self):
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        return mse
    
    def predict(self, new_data):
        #new_data_scaled = self.scaler.transform(new_data[['Open', 'High', 'Low', 'Moving_Avg_5', 'Moving_Avg_10', 'Daily_Return', 'Volatility']].fillna(0))
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

            # Predict the closing price for the current row
            predicted_close = self.predict(new_data)
            
            # Check if prediction is in expected format
            df.at[index, 'predictedClose'] = predicted_close[0] if predicted_close else None

        # Save the DataFrame to CSV once after the loop
        df.to_csv(filename)
