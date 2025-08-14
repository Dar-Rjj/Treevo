import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Compute Intraday Momentum Intensity
    df['High_Low_Range'] = df['high'] - df['low']
    df['Open_Close_Diff'] = df['close'] - df['open']
    df['Intraday_Momentum_Intensity'] = (df['High_Low_Range'] + df['Open_Close_Diff']) * df['volume']

    # Analyze Day-to-Day Momentum Continuation
    df['Prev_Close_to_Open'] = df['open'].shift(-1) - df['close']
    df['Multi_Day_Close_Trend'] = df['close'].rolling(window=5).mean()
    df['Trend_Direction'] = df['Multi_Day_Close_Trend'].diff()

    # Combine with Volume Spikes and Price Volatility
    df['Exp_Moving_Avg_Volume'] = df['volume'].ewm(span=5, adjust=False).mean()
    df['Volume_Spike'] = (df['volume'] > 1.5 * df['Exp_Moving_Avg_Volume']).astype(int)
    
    df['Price_Volatility'] = df['close'].rolling(window=5).std()
    volatility_bins = [0, df['Price_Volatility'].quantile(0.33), df['Price_Volatility'].quantile(0.66), float('inf')]
    df['Volatility_Category'] = pd.cut(df['Price_Volatility'], bins=volatility_bins, labels=[0, 1, 2])

    # Adjust the Factor Value Based on Volume and Volatility
    df['Factor_Strength_Adjustment'] = 0
    df.loc[(df['Intraday_Momentum_Intensity'] > 0) & (df['Volume_Spike'] == 1) & (df['Volatility_Category'] == 0), 'Factor_Strength_Adjustment'] = 1
    df.loc[(df['Intraday_Momentum_Intensity'] < 0) & (df['Volume_Spike'] == 1) & (df['Volatility_Category'] == 2), 'Factor_Strength_Adjustment'] = -1

    # Final Alpha Factor
    df['Alpha_Factor'] = df['Intraday_Momentum_Intensity'] + df['Prev_Close_to_Open'] + df['Trend_Direction'] + df['Factor_Strength_Adjustment']

    return df['Alpha_Factor']
