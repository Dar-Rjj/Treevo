import pandas as pd

def heuristics_v2(df):
    # Factor 1: Simple momentum
    f1 = df['close'].pct_change(periods=20)
    
    # Factor 2: Relative strength compared to average close price over the last 50 days
    f2 = (df['close'] - df['close'].rolling(window=50).mean()) / df['close'].rolling(window=50).std()
    
    # Factor 3: Average volume surprise in the past 10 days
    avg_volume = df['volume'].rolling(window=10).mean()
    f3 = (df['volume'] - avg_volume) / avg_volume
    
    # Factor 4: Volatility of the last 7 days' returns
    daily_returns = df['close'].pct_change()
    f4 = daily_returns.rolling(window=7).std()
    
    # Factor 5: Price-to-amount ratio
    f5 = df['close'] / df['amount']
    
    # Combine all factors into a matrix with dates as index
    heuristics_matrix = pd.DataFrame({'Momentum_20D': f1, 'RS_vs_Avg50D': f2, 'Volume_Surprise_10D': f3, 'Volatility_7D': f4, 'Price_to_Amount_Ratio': f5})
    
    return heuristics_matrix
