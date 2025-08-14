import numpy as np
def heuristics_v2(df):
    # Calculate Intraday Volatility
    df['Intraday_Volatility'] = (df['high'] - df['low']) / df['open']
    
    # Measure Overnight Gap
    df['Overnight_Gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Compute Volume Momentum
    df['Volume_Momentum'] = df['volume'] / df['volume'].rolling(window=5).mean()
    
    # Evaluate Amount to Volume Ratio for Buying Power
    df['Amount_Volume_Ratio'] = df['amount'] / df['volume']
    
    # Create Simple Moving Averages (SMA) for Trend Analysis
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    
    # Design Exponential Moving Averages (EMA) for Recent Trend Weighting
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # Construct Moving Average Convergence Divergence (MACD)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    
    # Adjusted Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # On-Balance Volume (OBV) for Cumulative Buying/Selling Pressure
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Stock-to-Market Return Ratio
    df['Stock_to_Market_Return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) - 0.0  # Assuming market return is 0 for simplicity
