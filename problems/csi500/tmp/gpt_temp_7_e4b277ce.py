import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Define lookback periods
    sma_short_period = 20
    sma_long_period = 200
    vol_adj_volatility_period = 10
    price_change_period = 5
    liquidity_period = 30
    
    # Calculate Simple Moving Average (SMA) of Close Prices
    df['SMA_Short'] = df['close'].rolling(window=sma_short_period).mean()
    df['SMA_Long'] = df['close'].rolling(window=sma_long_period).mean()
    
    # Compute Volume-Adjusted Volatility
    df['High_Low_Diff'] = df['high'] - df['low']
    df['Volume_Adjusted_Volatility'] = df['High_Low_Diff'] * df['volume']
    df['Volume_Adjusted_Volatility_Rolling'] = df['Volume_Adjusted_Volatility'].rolling(window=vol_adj_volatility_period).mean()
    
    # Compute Price Momentum
    df['Price_Momentum'] = (df['close'] - df['SMA_Short']) / df['close'].rolling(window=sma_short_period).mean()
    
    # Incorporate Additional Price Change Metrics
    df['Percentage_Change'] = df['close'].pct_change(periods=price_change_period)
    df['High_Low_Range'] = df['high'] - df['low']
    
    # Consider Market Trend Alignment
    df['Market_Trend'] = (df['SMA_Short'] > df['SMA_Long']).astype(int)
    
    # Incorporate Liquidity Measures
    df['Daily_Turnover'] = df['volume'] * df['close']
    df['Liquidity'] = df['Daily_Turnover'].rolling(window=liquidity_period).mean()
    
    # Adjust Weights Based on Market Trend
    initial_weights = {
        'Price_Momentum': 0.4,
        'Volume_Adjusted_Volatility_Rolling': 0.3,
        'Percentage_Change': 0.1,
        'High_Low_Range': 0.1,
        'Market_Trend': 0.1
    }
    
    bullish_weights = {
        'Price_Momentum': 0.5,
        'Volume_Adjusted_Volatility_Rolling': 0.25,
        'Percentage_Change': 0.1,
        'High_Low_Range': 0.05,
        'Market_Trend': 0.1
    }
    
    bearish_weights = {
        'Price_Momentum': 0.3,
        'Volume_Adjusted_Volatility_Rolling': 0.4,
        'Percentage_Change': 0.1,
        'High_Low_Range': 0.1,
        'Market_Trend': 0.1
    }
    
    # Adjust Weights Based on Liquidity
    high_liquidity_multiplier = 1.2
    low_liquidity_multiplier = 0.8
    
    # Final Alpha Factor
    df['Alpha_Factor'] = 0
    for date in df.index:
        if df.loc[date, 'Market_Trend'] == 1:
            weights = bullish_weights
        else:
            weights = bearish_weights
        
        if df.loc[date, 'Liquidity'] > df['Liquidity'].mean():
            multiplier = high_liquidity_multiplier
        else:
            multiplier = low_liquidity_multiplier
        
        for component, weight in weights.items():
            df.loc[date, 'Alpha_Factor'] += df.loc[date, component] * weight * multiplier
    
    return df['Alpha_Factor'].dropna()

# Example usage
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
