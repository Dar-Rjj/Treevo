import pandas as pd
import pandas as pd

def heuristics_v2(df, period=10, volume_ratio_threshold=1.5):
    # Calculate Daily Imbalance
    df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['NetChange'] = df['TypicalPrice'].diff()
    
    # Evaluate Volume Changes
    df['VolumeRatio'] = df['Volume'] / df['Volume'].shift(1)
    df['VolumeSurge'] = (df['VolumeRatio'] > volume_ratio_threshold).astype(int)
    
    # Integrate Imbalance and Volume Data
    df['PositiveImbalance'] = df.apply(lambda row: row['NetChange'] if row['NetChange'] > 0 else 0, axis=1)
    df['NegativeImbalance'] = df.apply(lambda row: row['NetChange'] if row['NetChange'] < 0 else 0, axis=1)
    
    df['WeightedPositiveImbalance'] = df['PositiveImbalance'] * df['VolumeRatio'] * df['VolumeSurge']
    df['WeightedNegativeImbalance'] = df['NegativeImbalance'] * df['VolumeRatio'] * df['VolumeSurge']
    
    # Composite Alpha Factor
    df['AlphaFactor'] = df['WeightedPositiveImbalance'].rolling(window=period).sum() - df['WeightedNegativeImbalance'].rolling(window=period).sum()
    
    return df['AlphaFactor']

# Example usage:
# df = pd.DataFrame(...)  # Your DataFrame here
# alpha_factor = heuristics_v2(df)
