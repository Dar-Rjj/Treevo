import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily High-Low Spread
    df['DailyHighLowSpread'] = df['high'] - df['low']
    
    # Compute Weighted Sum of Recent Spreads
    decay_factor = 0.9
    df['HighLowSpreadMomentum'] = df['DailyHighLowSpread'].rolling(window=10).apply(
        lambda x: (x * decay_factor ** np.arange(10)).sum(), raw=True)
    
    # Calculate Short-Term Average Return
    df['ShortTermReturn'] = df['close'].rolling(window=5).mean()
    
    # Calculate Long-Term Average Return
    df['LongTermReturn'] = df['close'].rolling(window=20).mean()
    
    # Calculate Price and Volume Momentum
    df['PriceMomentum'] = df['close'] - df['close'].shift(10)
    df['VolumeMomentum'] = df['volume'] - df['volume'].rolling(window=5).mean().shift(5)
    
    # Introduce Volatility Measure
    true_range = df[['high', 'low', 'close']].rolling(window=10).apply(
        lambda x: np.max([np.abs(x[0] - x[1]), np.abs(x[0] - x[2].shift()), np.abs(x[1] - x[2].shift())], axis=0), raw=False)
    df['ATR'] = true_range.rolling(window=10).mean()
    
    # Combine Price, Volume, and Volatility
    combined_numerator = (df['PriceMomentum'] * df['VolumeMomentum']) + df['ATR'] + 1e-6
    df['CombinedFactor'] = combined_numerator / df['volume']
    
    # Combine Momentum Indicators
    df['CrossoverMomentum'] = (df['ShortTermReturn'] - df['LongTermReturn']) * df['HighLowSpreadMomentum']
    
    # Adjust by High-Low Spread and Volume
    df['HLSpreadVol'] = (df['high'] - df['low']) * df['volume']
    
    # Incorporate Lagged Close Price
    df['HLSpreadVolLagged'] = df['HLSpreadVol'] / df['close'].shift(1)
    
    # Add Momentum Component
    df['5DayMomentum'] = df['close'] - df['close'].shift(5)
    df['MomentumComponent'] = df['HLSpreadVolLagged'] + df['5DayMomentum']
    
    # Introduce Volatility Component
    df['GarmanKlassVol'] = np.sqrt(0.5 * (np.log(df['high'] / df['low'])**2) - (2 * np.log(2) - 1) * np.log(df['close'] / df['open'])**2)
    df['VolatilityComponent'] = df['MomentumComponent'] * df['GarmanKlassVol']
    
    # Final Combination
    df['AlphaFactor'] = df['CombinedFactor'] + df['CrossoverMomentum'] + df['VolatilityComponent']
    
    return df['AlphaFactor']

# Example usage:
# alpha_factor = heuristics_v2(df)
