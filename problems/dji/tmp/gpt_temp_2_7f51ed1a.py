import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate 50-day and 200-day Simple Moving Averages (SMA)
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    
    # Determine the crossover points between 50-day and 200-day SMAs
    df['SMA_Crossover'] = (df['SMA_50'] > df['SMA_200']).astype(int)
    
    # Calculate Price Rate of Change (ROC) over 14 days
    df['ROC'] = df['close'].pct_change(periods=14)
    
    # Identify positive and negative ROC trends
    df['ROC_Trend'] = (df['ROC'] > 0).astype(int)
    
    # Calculate Volume Weighted Average Price (VWAP)
    df['Cum_Vol'] = df['volume'].cumsum()
    df['Cum_Vol_Price'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum()
    df['VWAP'] = df['Cum_Vol_Price'] / df['Cum_Vol']
    
    # Analyze the relationship between VWAP and close price
    df['VWAP_Diff'] = df['close'] - df['VWAP']
    
    # Create a Volume-based Momentum Indicator
    df['Volume_Momentum'] = df['volume'].pct_change(periods=14)
    
    # Evaluate the predictive power of candlestick patterns
    df['Candle_Body'] = abs(df['close'] - df['open'])
    df['Upper_Shadow'] = df['high'] - df['close'] if df['close'] > df['open'] else df['high'] - df['open']
    df['Lower_Shadow'] = df['open'] - df['low'] if df['close'] > df['open'] else df['close'] - df['low']
    
    # Assess the trend and volatility of the open-close spread
    df['Open_Close_Spread'] = df['close'] - df['open']
    df['OC_Spread_Mean'] = df['Open_Close_Spread'].rolling(window=14).mean()
    df['OC_Spread_Std'] = df['Open_Close_Spread'].rolling(window=14).std()
    
    # Generate a composite score for each day
    df['Composite_Factor'] = (
        df['SMA_Crossover'] * 0.2 +
        df['ROC_Trend'] * 0.2 +
        (df['VWAP_Diff'] > 0).astype(int) * 0.2 +
        (df['Volume_Momentum'] > 0).astype(int) * 0.2 +
        (df['Candle_Body'] > (df['Upper_Shadow'] + df['Lower_Shadow'])).astype(int) * 0.2
    )
    
    # Return the composite factor as the alpha factor
    return df['Composite_Factor']
