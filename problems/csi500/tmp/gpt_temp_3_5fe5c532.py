import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Compute Volatility-Adjusted Price Divergence
    # Short-term volatility (3-day ATR)
    data['atr_3'] = data['true_range'].rolling(window=3, min_periods=3).mean()
    
    # Medium-term volatility (8-day ATR)
    data['atr_8'] = data['true_range'].rolling(window=8, min_periods=8).mean()
    
    # Volatility ratio
    data['vol_ratio'] = data['atr_3'] / data['atr_8']
    
    # Calculate Price-Volume Divergence
    # Daily returns
    data['daily_return'] = data['close'] / data['prev_close'] - 1
    
    # 5-day volume-weighted momentum
    data['vw_return'] = data['daily_return'] * data['volume']
    data['vw_momentum_5'] = data['vw_return'].rolling(window=5, min_periods=5).sum()
    
    # 10-day simple price momentum
    data['close_shift_9'] = data['close'].shift(9)
    data['simple_momentum_10'] = data['close'] / data['close_shift_9'] - 1
    
    # Price-volume divergence
    data['pv_divergence'] = data['vw_momentum_5'] - data['simple_momentum_10']
    
    # Volatility-adjusted divergence
    data['vol_adj_divergence'] = data['pv_divergence'] * data['vol_ratio']
    
    # Detect Regime-Switching Patterns
    # Volume regime transitions
    data['volume_shift_2'] = data['volume'].shift(2)
    data['volume_momentum_3'] = data['volume'] / data['volume_shift_2'] - 1
    
    # Volume persistence (variance over 8 days)
    data['volume_persistence'] = data['volume'].rolling(window=8, min_periods=8).var()
    
    # Regime transition indicator
    data['regime_transition'] = data['volume_momentum_3'] / (data['volume_persistence'] + 1e-8)
    
    # Apply threshold detection
    data['regime_multiplier'] = 1.0
    data.loc[data['regime_transition'] > 1.5, 'regime_multiplier'] = 1.5
    data.loc[data['regime_transition'] < 0.8, 'regime_multiplier'] = 0.5
    
    # Price regime consistency
    data['close_shift_2'] = data['close'].shift(2)
    data['close_shift_7'] = data['close'].shift(7)
    
    # 3-day price direction
    data['direction_3'] = np.where(data['close'] > data['close_shift_2'], 1, -1)
    
    # 8-day price direction
    data['direction_8'] = np.where(data['close'] > data['close_shift_7'], 1, -1)
    
    # Regime consistency score
    data['regime_consistency'] = np.where(data['direction_3'] == data['direction_8'], 1, -1)
    
    # Generate Dynamic Alpha Factor
    # Combine volatility-adjusted divergence with regime signals
    data['transition_enhanced'] = data['vol_adj_divergence'] * data['regime_multiplier']
    data['consistency_filtered'] = data['transition_enhanced'] * data['regime_consistency']
    
    # Implement adaptive smoothing
    data['smoothing_window'] = np.where(data['vol_ratio'] > 1.2, 2, 5)
    
    # Calculate smoothed factor based on adaptive window
    def adaptive_smooth(row, data):
        window = int(row['smoothing_window'])
        idx = data.index.get_loc(row.name)
        if idx >= window - 1:
            window_data = data.iloc[idx - window + 1:idx + 1]['consistency_filtered']
            return window_data.mean()
        else:
            return np.nan
    
    # Apply adaptive smoothing
    alpha_factor = pd.Series(index=data.index, dtype=float)
    for idx in data.index:
        if not pd.isna(data.loc[idx, 'smoothing_window']):
            alpha_factor.loc[idx] = adaptive_smooth(data.loc[idx], data)
    
    return alpha_factor
