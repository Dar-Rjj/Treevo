import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Weighted Price Returns
    df['NextDayOpen'] = df['open'].shift(-1)
    df['SimpleReturns'] = (df['NextDayOpen'] - df['close']) / df['close']
    df['VolumeWeightedReturns'] = df['SimpleReturns'] * df['volume']
    
    # Identify Volume Surge Days
    df['DailyVolumeChange'] = df['volume'] - df['volume'].shift(1)
    df['VolumeRollingMean'] = df['volume'].rolling(window=5).mean()
    df['VolumeSurge'] = (df['volume'] > df['VolumeRollingMean']).astype(int)
    
    # Calculate Volatility Using ATR
    df['TrueRange'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1
    )
    df['ATR'] = df['TrueRange'].rolling(window=14).mean()
    
    # Adjust Volume-Weighted Returns by ATR
    df['AdjustedReturns'] = df['VolumeWeightedReturns'] / df['ATR']
    
    # Incorporate Multi-Day Patterns
    df['ShortTermMomentum'] = df['close'].pct_change(5).rolling(window=5).mean()
    df['LongTermMomentum'] = df['close'].pct_change(20).rolling(window=20).mean()
    df['CombinedMomentum'] = 0.6 * df['ShortTermMomentum'] + 0.4 * df['LongTermMomentum']
    
    # Combine Adjusted Returns with Volume Surge Indicator
    surge_factor = 1.5
    df['FinalAlphaFactor'] = df['AdjustedReturns'] + df['VolumeSurge'] * (df['AdjustedReturns'] * (surge_factor - 1))
    df['FinalAlphaFactor'] += df['CombinedMomentum']
    
    return df['FinalAlphaFactor']

# Example usage:
# df = pd.read_csv('stock_data.csv', parse_dates=True, index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
