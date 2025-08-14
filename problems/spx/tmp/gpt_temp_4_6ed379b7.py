import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the difference between high and low prices of day t
    factor1 = (df['high'] - df['low']).abs()
    
    # Create a factor from the difference (close - open) on day t
    factor2 = df['close'] - df['open']
    
    # Develop a factor using the ratio (close / open) on day t
    factor3 = df['close'] / df['open']
    
    # Construct a factor from the daily volume on day t
    factor4 = df['volume']
    
    # Build a factor by dividing the amount by the volume on day t
    factor5 = df['amount'] / df['volume']
    
    # Create a factor from the change in volume compared to the previous day
    factor6 = df['volume'].pct_change()
    
    # Develop a composite factor from (high - low) and (close - open)
    factor7 = (factor1 + factor2) / 2
    
    # Design a composite factor using (close / open) and (amount / volume)
    factor8 = (factor3 + factor5) / 2
    
    # Calculate the simple moving average (SMA) of closing prices over a 5-day period ending on day t
    sma_close = df['close'].rolling(window=5).mean()
    
    # Calculate the SMA of volumes over a 5-day period ending on day t
    sma_volume = df['volume'].rolling(window=5).mean()
    
    # Derive a factor from the difference between the SMA of closing prices and the closing price on day t
    factor9 = sma_close - df['close']
    
    # Determine the rate of change (ROC) of the closing price over the last 5 days
    roc_close = df['close'].pct_change(periods=5)
    
    # Compute the ROC of the volume over the last 5 days
    roc_volume = df['volume'].pct_change(periods=5)
    
    # Combine all factors into a single DataFrame
    factors_df = pd.DataFrame({
        'Factor1': factor1,
        'Factor2': factor2,
        'Factor3': factor3,
        'Factor4': factor4,
        'Factor5': factor5,
        'Factor6': factor6,
        'Factor7': factor7,
        'Factor8': factor8,
        'Factor9': factor9,
        'ROC_Close': roc_close,
        'ROC_Volume': roc_volume
    })
    
    return factors_df
