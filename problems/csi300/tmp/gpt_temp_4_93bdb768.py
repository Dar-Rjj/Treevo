import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Price Angle
    df['Price_Angle'] = df['High'] - df['Low']
    
    # Determine Lookback Period Based on Volatility
    def determine_lookback_period(rolling_volatility, threshold=0.5):
        if rolling_volatility > threshold:
            return 5  # Shorter lookback for higher volatility
        else:
            return 10  # Longer lookback for lower volatility
    
    # Calculate Rolling Volatility (standard deviation of returns)
    df['Return'] = df['Close'].pct_change()
    df['Rolling_Volatility'] = df['Return'].rolling(window=20).std()
    
    # Dynamic Cumulative Price Angle
    df['Cumulative_Price_Angle'] = 0
    for i in range(len(df)):
        lookback_period = determine_lookback_period(df['Rolling_Volatility'].iloc[i])
        df.loc[df.index[i], 'Cumulative_Price_Angle'] = df['Price_Angle'].iloc[max(0, i-lookback_period+1):i+1].sum()
    
    # Evaluate Volume Trend
    df['Volume_Change'] = df['Volume'] - df['Volume'].shift(lookback_period)
    df['Volume_Weight'] = np.where(df['Volume_Change'] > 0, 1, -1)
    
    # Incorporate Trade Direction
    df['Trade_Direction'] = np.where(df['Close'] > df['Open'], 1, -1)
    
    # Combine Dynamic Cumulative Price Angle, Volume Trend, and Trade Direction
    df['Alpha_Factor'] = df['Cumulative_Price_Angle'] * df['Volume_Weight'] * df['Trade_Direction']
    
    # Adjust for Volatility
    df['Alpha_Factor'] = df['Alpha_Factor'] * df['Rolling_Volatility']
    
    return df['Alpha_Factor']

# Example usage
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
