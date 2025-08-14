import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, lookback_period=10):
    # Calculate Volume-Weighted Price Movement
    df['Volume_Weighted_Price_Movement'] = (df['High'] + df['Low']) / 2 * df['Volume']
    
    # Evaluate Intraday Momentum Components
    df['Daily_High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    df['Intraday_Momentum'] = (df['Daily_High_Low_Ratio'] + df['Close_Open_Ratio']) / 2
    
    # Assess Overnight Return Sentiment
    df['Overnight_Return'] = np.log(df['Open']) - np.log(df['Close'].shift(1))
    df['Log_Volume'] = np.log(df['Volume'])
    df['Overnight_Sentiment'] = df['Overnight_Return'] * df['Log_Volume']
    
    # Integrate Intraday and Overnight Momentum
    df['Momentum_Composite'] = df['Intraday_Momentum'] - df['Overnight_Return']
    
    # Generate Composite Momentum Indicator
    sum_volume_weighted_price_movement = df['Volume_Weighted_Price_Movement'].rolling(window=lookback_period).sum()
    sum_volume = df['Volume'].rolling(window=lookback_period).sum()
    df['Composite_Momentum_Indicator'] = sum_volume_weighted_price_movement / sum_volume
    
    # Apply Dynamic Volume Adjustment
    df['EMA_Volume'] = df['Volume'].ewm(span=lookback_period, adjust=False).mean()
    df['Recent_Volume_Volatility'] = df['Volume'].std()
    df['Adjusted_Volume'] = df['EMA_Volume'] - df['Volume'].iloc[-1] + df['Recent_Volume_Volatility']
    
    # Incorporate High-Low Volatility and Reversal Potential
    df['Intraday_High_Low_Difference'] = df['High'] - df['Low']
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].shift(1)
    df['Volume_Direction'] = np.where(df['Volume_Ratio'] > 1, 1, -1)
    df['Daily_Momentum_Change'] = df['Close'] - df['Close'].shift(1)
    df['Reversal_Potential'] = df['Intraday_High_Low_Difference'] * df['Volume_Direction'] - df['Daily_Momentum_Change']
    
    # Calculate Intraday High-Low Spread Z-Score
    intraday_high_low_spread = df['High'] - df['Low']
    historical_spread_std = intraday_high_low_spread.rolling(window=lookback_period).std()
    df['Intraday_High_Low_Spread_Z_Score'] = intraday_high_low_spread / historical_spread_std
    
    # Combine All Indicators
    df['Alpha_Factor'] = df['Momentum_Composite'] * df['Composite_Momentum_Indicator'] * df['Intraday_High_Low_Spread_Z_Score']
    
    return df['Alpha_Factor']

# Example usage:
# df = pd.read_csv('stock_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
