import numpy as np
def heuristics_v2(df):
    # Calculate moving averages
    df['MA_5'] = df['close'].rolling(window=5).mean()
    df['MA_10'] = df['close'].rolling(window=10).mean()
    df['MA_20'] = df['close'].rolling(window=20).mean()
    df['MA_50'] = df['close'].rolling(window=50).mean()
    df['MA_100'] = df['close'].rolling(window=100).mean()
    
    # Compute the difference between different moving averages
    df['MA_5_10'] = df['MA_5'] - df['MA_10']
    df['MA_10_20'] = df['MA_10'] - df['MA_20']
    df['MA_20_50'] = df['MA_20'] - df['MA_50']
    df['MA_50_100'] = df['MA_50'] - df['MA_100']
    
    # Use moving average crossovers to generate signals
    df['MA_5_cross_MA_20'] = (df['MA_5'] > df['MA_20']).astype(int)
    
    # Implement momentum indicators using closing prices
    df['ROC_14'] = df['close'].pct_change(periods=14)
    
    # Calculate on-balance volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # Develop a volume-weighted moving average (VWMA)
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # Evaluate the relationship between volume and price changes
    df['vol_pct_change'] = df['volume'].pct_change()
    df['price_pct_change'] = df['close'].pct_change()
    df['vol_price_ratio'] = df['vol_pct_change'] / df['price_pct_change'].replace(0, np.nan)
    
    # Compute average true range (ATR) over a given period
    df['TR'] = df[['high', 'low']].sub(df['close'].shift(), axis=0).abs().max(axis=1)
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    
    # Calculate historical volatility using standard deviation of daily returns
