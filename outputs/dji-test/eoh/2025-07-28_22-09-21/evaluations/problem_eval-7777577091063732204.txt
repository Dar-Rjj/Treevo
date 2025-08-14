import pandas as pd

def heuristics_v2(df):
    # Custom alpha 1: Difference between the daily high and low prices
    alpha1 = df['high'] - df['low']
    
    # Custom alpha 2: Daily return volatility (standard deviation over a 5-day window)
    alpha2 = df['close'].rolling(window=5).std()
    
    # Custom alpha 3: Ratio of closing price to average volume over the last 7 days
    alpha3 = df['close'] / df['volume'].rolling(window=7).mean()
    
    # Custom alpha 4: Price change momentum (difference between today's close and 10-day moving average of close)
    alpha4 = df['close'] - df['close'].rolling(window=10).mean()
    
    # Custom alpha 5: Volume growth rate (percentage change in volume from the previous day)
    alpha5 = df['volume'].pct_change()
    
    # Combining all alphas into a DataFrame
    heuristics_matrix = pd.DataFrame({'alpha1': alpha1, 'alpha2': alpha2, 'alpha3': alpha3, 'alpha4': alpha4, 'alpha5': alpha5})
    
    return heuristics_matrix
