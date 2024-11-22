from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import json
import pandas as pd

class model():
    
    data_path = "data.json"
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)  # Use RandomForestRegressor
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        # Load data from a JSON file
        with open(self.data_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        # Ensure the index is numeric and convert to datetime
        df.index = pd.to_numeric(df.index)
        df.index = pd.to_datetime(df.index, unit="ms")
        
        df['Moving_Avg_5'] = df['Close'].rolling(window=20).mean()
        df['Moving_Avg_10'] = df['Close'].rolling(window=60).mean()
        # check when Moving_Avg_5 crosses Moving_Avg_10
        # if Moving_Avg_5 > Moving_Avg_10, market decline bear market
        # if Moving_Avg_5 < Moving_Avg_10, market rise bull market


        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Close'].rolling(window=5).std()
        df.fillna(0, inplace=True)
        
        df['test_close'] = None
        
        return df
    
    def preprocess_data(self, df):
        # Prepare features
        X = df[['Open', 'High', 'Low', 'Moving_Avg_5', 'Moving_Avg_10', 'Daily_Return', 'Volatility']].fillna(0)  # Filling NaN values
        y = df['Close']
        #y_trinary = (y > x['open']) up
        #y_trinary = (y < x['open']) down
        #y_trinary = (y == x['open']) no change

        # Split the data into training and test sets (80% train, 20% test)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # y_trinary = (y_pred > x) up
        # y_trinary = (y_pred < x) down
        # y_trinary = (y_pred == x) no change

    def train(self):
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        return mse
    
    def predict(self, new_data):
        new_data_scaled = self.scaler.transform(new_data[['Open', 'High', 'Low', 'Moving_Avg_5', 'Moving_Avg_10', 'Daily_Return', 'Volatility']].fillna(0))
        predicted_closing_price = self.model.predict(new_data_scaled)
        return predicted_closing_price
    
    def backtest(self, df):
        for index, row in df.iterrows():
            opening_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            
            moving_avg_5 = row['Moving_Avg_5']
            moving_avg_10 = row['Moving_Avg_10']
            daily_return = row['Daily_Return']
            volatility = row['Volatility']

            new_data = pd.DataFrame([[opening_price, high_price, low_price, moving_avg_5, moving_avg_10, daily_return, volatility]],
                                    columns=['Open', 'High', 'Low', 'Moving_Avg_5', 'Moving_Avg_10', 'Daily_Return', 'Volatility']).fillna(0)

            # Predict the closing price
            predicted_close = self.predict(new_data)

            # Populate the 'test_close' column with the predicted closing price
            df.at[index, 'test_close'] = predicted_close[0]  # Insert the predicted value

        return df
