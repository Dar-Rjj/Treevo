import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=20, m=5, p=14, volume_threshold=2.0, intraday_range_threshold=1.5, volatility_threshold=1.5, combined_scaling_factor=1.5, max_momentum_value=10.0):
    # Calculate Price Momentum
    df['PriceMomentum'] = df['close'].pct_change(n)
    
    # Identify Volume Spikes
    df['AverageVolume'] = df['volume'].rolling(window=m).mean()
    df['VolumeSpike'] = (df['volume'] > df['AverageVolume'] * volume_threshold).astype(int)
    df['AdjustedMomentum'] = df['PriceMomentum'] * (1 + df['VolumeSpike'] * (combined_scaling_factor - 1))
    
    # Integrate Intraday Volatility and Trading Range
    df['IntradayRange'] = df['high'] - df['low']
    df['TrueRange'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    df['AverageTrueRange'] = df['TrueRange'].rolling(window=p).mean()
    
    # Adjust Momentum by Intraday Range
    df['IntradayAdjustedMomentum'] = df['AdjustedMomentum'] / df['IntradayRange']
    
    # Combine Volatility and Intraday Range
    df['VolatilityAboveThreshold'] = (df['AverageTrueRange'] > df['IntradayRange'] * volatility_threshold).astype(int)
    df['IntradayRangeAboveThreshold'] = (df['IntradayRange'] > df['IntradayRange'].rolling(window=p).mean() * intraday_range_threshold).astype(int)
    
    # Apply combined scaling factors
    df['CombinedScalingFactor'] = 1.0
    df.loc[df['VolatilityAboveThreshold'] == 1, 'CombinedScalingFactor'] = combined_scaling_factor
    df.loc[df['IntradayRangeAboveThreshold'] == 1, 'CombinedScalingFactor'] *= combined_scaling_factor
    df['CombinedAdjustedMomentum'] = df['IntradayAdjustedMomentum'] * df['CombinedScalingFactor']
    
    # Final Adjustment
    df['FinalMomentum'] = df['CombinedAdjustedMomentum'].clip(upper=max_momentum_value)
    
    return df['FinalMomentum']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# factor = heuristics_v2(df)
