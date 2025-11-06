import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Overnight Price Gap
    data['prev_close'] = data['close'].shift(1)
    data['gap_ratio'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Calculate True Range
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate 10-day Average True Range
    data['atr_10'] = data['true_range'].rolling(window=10, min_periods=1).mean()
    
    # Calculate Volatility Z-score relative to 20-day history
    data['volatility_mean'] = data['atr_10'].rolling(window=20, min_periods=1).mean()
    data['volatility_std'] = data['atr_10'].rolling(window=20, min_periods=1).std()
    data['volatility_zscore'] = (data['atr_10'] - data['volatility_mean']) / data['volatility_std']
    data['volatility_zscore'] = data['volatility_zscore'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Identify Extreme Gaps (top/bottom 10% of historical distribution)
    data['gap_rank'] = data['gap_ratio'].rolling(window=252, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    )
    data['extreme_gap'] = ((data['gap_rank'] >= 0.9) | (data['gap_rank'] <= 0.1)).astype(int)
    
    # Filter by Volatility Conditions
    data['high_volatility'] = (data['volatility_zscore'] > 1).astype(int)
    
    # Generate Gap Reversion Signal
    data['reversion_signal'] = 0
    # Negative gap (gap down) suggests potential reversion upward
    mask_negative_gap = (data['gap_ratio'] < 0) & (data['extreme_gap'] == 1) & (data['high_volatility'] == 1)
    # Positive gap (gap up) suggests potential reversion downward  
    mask_positive_gap = (data['gap_ratio'] > 0) & (data['extreme_gap'] == 1) & (data['high_volatility'] == 1)
    
    data.loc[mask_negative_gap, 'reversion_signal'] = -data.loc[mask_negative_gap, 'gap_ratio']
    data.loc[mask_positive_gap, 'reversion_signal'] = -data.loc[mask_positive_gap, 'gap_ratio']
    
    # Volume-Based Timing
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_spike'] = data['volume'] / data['volume_5d_avg']
    data['volume_spike'] = data['volume_spike'].replace([np.inf, -np.inf], 1).fillna(1)
    
    # Adjust Reversion Signal Strength by Volume Confirmation
    data['volume_adjustment'] = np.where(
        data['volume_spike'] > 1.2,
        np.minimum(data['volume_spike'], 2.0),  # Cap at 2x amplification
        0.5  # Reduce signal if volume is below threshold
    )
    
    # Final factor calculation
    data['factor'] = data['reversion_signal'] * data['volume_adjustment']
    
    return data['factor']
