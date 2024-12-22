import pandas as pd
from pytictoc import TicToc
import argparse as ap
import sys
from randomForestRegressor import model
from data import dataHelper
from ui import StockPredictionCommandLine

DATA_CSV = "data.csv"
DATA_TEST_CSV = "data_test.csv"
TRAIN_BACKUP_CSV = "backup_data_train.csv"
TEST_BACKUP_CSV = "backup_data_test.csv"
tickerTrain = "MSFT"
tickerTest = "BA"
    
    
if __name__ == "__main__":
    
    parser = ap.ArgumentParser(
        prog="Stock Locker",
        description="Train or test an existing model to predict stock prices.",
        add_help=False  
    )
    
    parser.add_argument(
        "mode",
        choices=["test", "live"],
        help="Choose 'test' to use backup stock data files\n'live' to retrieve new data from the API.\nIf you encounter an error during the data preparation step, try 'test' mode."
    )
    parser.add_argument(
        "-h", "--help",
        action="help",
        help="Choose 'test' to use backup stock data files\n'live' to retrieve new data from the API.\nIf you encounter an error during the data preparation step, try 'test' mode."
    )
    
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    if args.mode == "test":
        mode = 0
    elif args.mode == "live":
        mode = 1
    else:
        parser.print_help()
        sys.exit(1)
        
    cl = StockPredictionCommandLine(DATA_CSV, mode)
    cl.start()
    
    
    

  