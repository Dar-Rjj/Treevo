import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Simple Momentum
    df['SimpleMomentum'] = df['close'].pct_change(periods=1)
    
    # Volume Adjusted Component
    df['DailyVolumeChange'] = df['volume'].pct_change(periods=1)
    df['VolumeAdjustedMomentum'] = df['SimpleMomentum'] * df['DailyVolumeChange']
    
    # Enhanced Price Momentum
    df['Close5EMA'] = df['close'].ewm(span=5, adjust=False).mean()
    df['Close20EMA'] = df['close'].ewm(span=20, adjust=False).mean()
    df['High5EMA'] = df['high'].ewm(span=5, adjust=False).mean()
    df['High20EMA'] = df['high'].ewm(span=20, adjust=False).mean()
    df['Low5EMA'] = df['low'].ewm(span=5, adjust=False).mean()
    df['Low20EMA'] = df['low'].ewm(span=20, adjust=False).mean()
    
    df['CloseMomentum'] = df['Close5EMA'] - df['Close20EMA']
    df['HighMomentum'] = df['High5EMA'] - df['High20EMA']
    df['LowMomentum'] = df['Low5EMA'] - df['Low20EMA']
    df['CombinedMomentum'] = df['CloseMomentum'] + (df['HighMomentum'] - df['LowMomentum'])
    
    # Adjust for Volume
    df['Volume5EMA'] = df['volume'].ewm(span=5, adjust=False).mean()
    df['Volume20EMA'] = df['volume'].ewm(span=20, adjust=False).mean()
    
    volume_factor = np.where(df['Volume5EMA'] > df['Volume20EMA'], 1.2, 0.8)
    df['AdjustedMomentum'] = df['CombinedMomentum'] * volume_factor
    
    # Adjust Momentum by ATR and Volume Change
    df['TrueRange'] = np.maximum(
        np.maximum(df['high'] - df['low'], abs(df['high'] - df['close'].shift(1))),
        abs(df['low'] - df['close'].shift(1))
    )
    df['ATR'] = df['TrueRange'].ewm(span=14, adjust=False).mean()
    df['ATRAdjustedMomentum'] = df['AdjustedMomentum'] / df['ATR']
    
    df['VolumeAdjustedMomentum'] = df['ATRAdjustedMomentum'] * df['DailyVolumeChange'] * df['Volume5EMA']
    
    # Incorporate Price Reversal Sensitivity
    df['HighLowSpread'] = df['high'] - df['low']
    df['WeightedHighLowSpread'] = df['HighLowSpread'] * df['volume']
    
    # Incorporate Enhanced Price Gaps
    df['OpenToCloseGap'] = df['open'] - df['close'].shift(1)
    df['HighToLowGap'] = df['high'] - df['low']
    df['EnhancedPriceGaps'] = df['OpenToCloseGap'] + df['HighToLowGap']
    
    # Detect Volume Spikes
    df['VolumeSpike'] = df['volume'] - df['volume'].shift(1)
    
    # Combine Components
    df['FinalAlphaFactor'] = (df['VolumeAdjustedComponent'] + 
                              df['EnhancedPriceMomentum'] + 
                              df['VolumeAdjustedMomentum'] - 
                              df['WeightedHighLowSpread'] + 
                              df['EnhancedPriceGaps'] + 
                              df['VolumeSpike'])
    
    return df['FinalAlphaFactor']

# Example usage:
# df = pd.DataFrame(...)  # Your DataFrame with columns: date, open, high, low, close, amount, volume
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
