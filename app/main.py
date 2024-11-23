import pandas as pd
from pytictoc import TicToc
import argparse as ap
import sys
from randomForestRegressor import model
from data import dataHelper

DATA_CSV = "data.csv"
DATA_TEST_CSV = "data_test.csv"
TRAIN_BACKUP_CSV = "backup_data_train.csv"
TEST_BACKUP_CSV = "backup_data_test.csv"
tickerTrain = "MSFT"
tickerTest = "BA"


# ---------------------------------------------------------------------------------------------------------------
# mainTrain creates a new model, trains it, evaluates it, backtests it, and saves the model and scaler files

def mainTrain(): 
    print("starting...\n")
    
    dh = dataHelper(outputfile=DATA_CSV)
    
    # in actual use this would use getDaily() to get the latest data
    # but for testing there is a backup csv file with data that gets copied 
    # due to the daily limit on the API
    #dh.getDaily(tickerTest, "full", "csv")
    dh.copyDataFromExisting(TRAIN_BACKUP_CSV, DATA_CSV)
    
    dh.prepareData(tickerTrain)
    
    df = dh.loadDataToDF()
    
    if df is None:
        print("Error: unable to load data from csv file.")
        exit()
    
    mdl = model(df)
    print("Model created.")
    
    mdl.setUnstandardizedData(df)
    print("Original open, high, low, and close values saved.")
    
    
    mdl.standardiseData(True)
    print("Data standardised.")
    mdl.prepareData()
    
    mdl.train()
    print("Model trained.")
    
    print(mdl.evaluate())
    
    print("Backtesting...")
    mdl.backtest(DATA_CSV)
    print("Backtesting done.")

    dh.plotBacktestedData("Actual vs Predicted Closing Prices - Standardised", 'train_backtested_plot_std.png')
    
    mdl.unstandardizeData(DATA_CSV)
    print(f"Data unstandardised and saved to {DATA_CSV}.")
    
    dh.plotBacktestedData("Actual vs Predicted Closing Prices - Normalised",'train_backtested_plot_normal.png')   
    
    print("done.")
# ---------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------
# mainTest loads an existing model and scaler, and tests it on new data    

def mainTest():
    print("starting...\n")

    dh = dataHelper(outputfile=DATA_TEST_CSV)
    
    # in actual use this would use getDaily() to get the latest data
    # but for testing there is a backup csv file with data that gets copied 
    # due to the daily limit on the API
    #dh.getDaily(tickerTest, "full", "csv")
    dh.copyDataFromExisting(TEST_BACKUP_CSV, DATA_TEST_CSV)
    
    dh.prepareData(tickerTest)
    
    df = dh.loadDataToDF()
    
    if df is None:
        print("Error: unable to load data from csv file.")
        exit()
    
    mdl = model(df)
    print("Model created.")
    
    mdl.setUnstandardizedData(df)
    print("Original open, high, low, and close values saved.")
    
    mdl.loadModel()
    mdl.loadScaler()
    
    mdl.standardiseData(False)
    print("Data standardised.")
    
    print("Backtesting...")
    mdl.backtest(DATA_TEST_CSV)
    print("Backtesting done.")

    dh.plotBacktestedData("Actual vs Predicted Closing Prices - Standardised",'test_backtested_plot_std.png')
    
    mdl.unstandardizeData(DATA_TEST_CSV)
    print(f"Data unstandardised and saved to {DATA_TEST_CSV}.")
    
    dh.plotBacktestedData("Actual vs Predicted Closing Prices - Normalised", 'test_backtested_plot_normal.png')   
    
    print("done.")
    
# ---------------------------------------------------------------------------------------------------------------
    
    
if __name__ == "__main__":
    
    parser = ap.ArgumentParser(
        prog="Stock price prediction",
        description="Train or test an existing model to predict stock prices.",
        add_help=False  # Disable default help to customize error handling.
    )
    
    parser.add_argument(
        "mode",
        choices=["train", "test"],
        help="Choose 'train' to train a model or 'test' to test an existing model."
    )
    parser.add_argument(
        "-h", "--help",
        action="help",
        help="Show this help message and exit."
    )
    
    # Parse arguments
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    if args.mode == "train":
        mainTrain()
    elif args.mode == "test":
        mainTest()
    
    

  