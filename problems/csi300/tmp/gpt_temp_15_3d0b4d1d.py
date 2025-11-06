import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility-Regime Volume Analysis
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Rolling percentile of True Range (20-day window)
    data['tr_percentile'] = data['true_range'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 70)) if len(x) == 20 else np.nan
    )
    
    # Daily returns and absolute returns percentile
    data['returns'] = data['close'].pct_change()
    data['abs_returns_percentile'] = data['returns'].abs().rolling(window=20).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 70)) if len(x) == 20 else np.nan
    )
    
    # High volatility regime (top 30%)
    data['high_vol_regime'] = ((data['tr_percentile'] == 1) | (data['abs_returns_percentile'] == 1)).astype(int)
    
    # Volume momentum calculations
    data['volume_5d_mean'] = data['volume'].rolling(window=5).mean()
    data['volume_5d_std'] = data['volume'].rolling(window=5).std()
    
    # High volatility volume momentum
    data['high_vol_volume_momentum'] = data['volume'] / data['volume_5d_mean']
    data['high_vol_volume_persistence'] = data['high_vol_regime'].rolling(window=3).sum()
    
    # Low volatility regime volume acceleration
    data['low_vol_regime'] = ((data['tr_percentile'] == 0) & (data['abs_returns_percentile'] == 0)).astype(int)
    data['low_vol_volume_accel'] = (data['volume'] - data['volume_5d_mean']) / (data['volume_5d_std'] + 1e-8)
    
    # Price-Level Memory Effects
    # Identify local highs and lows (20-day window)
    data['local_high'] = data['high'].rolling(window=20, center=False).apply(
        lambda x: 1 if x.iloc[-1] == x.max() else 0, raw=False
    )
    data['local_low'] = data['low'].rolling(window=20, center=False).apply(
        lambda x: 1 if x.iloc[-1] == x.min() else 0, raw=False
    )
    
    # Previous significant levels
    data['prev_high_20d'] = data['high'].rolling(window=20).max().shift(1)
    data['prev_low_20d'] = data['low'].rolling(window=20).min().shift(1)
    
    # Price approaching previous highs/lows (within 2%)
    data['near_prev_high'] = (abs(data['close'] - data['prev_high_20d']) / data['prev_high_20d'] <= 0.02).astype(int)
    data['near_prev_low'] = (abs(data['close'] - data['prev_low_20d']) / data['prev_low_20d'] <= 0.02).astype(int)
    
    # Volume at key levels
    data['volume_at_resistance'] = data['volume'] * data['near_prev_high']
    data['volume_at_support'] = data['volume'] * data['near_prev_low']
    
    # Volume-to-Price-Range ratio
    data['daily_range'] = data['high'] - data['low']
    data['volume_range_ratio'] = data['volume'] / (data['daily_range'] + 1e-8)
    
    # Temporal Pattern Integration
    # Intraday position analysis
    data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['open_close_capture'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Consecutive day analysis
    data['price_up'] = (data['close'] > data['close'].shift(1)).astype(int)
    data['price_down'] = (data['close'] < data['close'].shift(1)).astype(int)
    
    data['consecutive_ups'] = data['price_up'].rolling(window=3).sum()
    data['consecutive_downs'] = data['price_down'].rolling(window=3).sum()
    
    # Volume on consecutive moves
    data['volume_on_up_trend'] = data['volume'] * (data['consecutive_ups'] >= 2).astype(int)
    data['volume_on_down_trend'] = data['volume'] * (data['consecutive_downs'] >= 2).astype(int)
    
    # Factor Synthesis
    # Regime-weighted signals
    high_vol_signal = data['high_vol_volume_momentum'] * data['high_vol_regime']
    low_vol_signal = data['low_vol_volume_accel'] * data['low_vol_regime']
    
    # Key level volume signals
    resistance_signal = data['volume_at_resistance'] * data['near_prev_high']
    support_signal = data['volume_at_support'] * data['near_prev_low']
    
    # Temporal pattern signals
    trend_volume_signal = (data['volume_on_up_trend'] - data['volume_on_down_trend']) / (data['volume_5d_mean'] + 1e-8)
    
    # Composite divergence factor
    data['regime_divergence_factor'] = (
        high_vol_signal.fillna(0) * 0.3 +
        low_vol_signal.fillna(0) * 0.25 +
        resistance_signal.fillna(0) * 0.15 +
        support_signal.fillna(0) * 0.15 +
        trend_volume_signal.fillna(0) * 0.15
    )
    
    # Normalize the factor
    data['regime_divergence_factor'] = (
        data['regime_divergence_factor'] - data['regime_divergence_factor'].rolling(window=20).mean()
    ) / (data['regime_divergence_factor'].rolling(window=20).std() + 1e-8)
    
    return data['regime_divergence_factor']
