import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Momentum Reversal Pressure with Volume Acceleration alpha factor
    """
    df = data.copy()
    
    # Asymmetric Momentum Divergence
    # Short-term pressure intensity
    df['upward_strength'] = (df['high'] - df['close'].shift(5)) / df['close'].shift(5)
    df['downward_strength'] = (df['close'].shift(5) - df['low']) / df['close'].shift(5)
    df['momentum_imbalance'] = df['upward_strength'] / (df['downward_strength'] + 1e-8)
    
    # Medium-term trend persistence
    df['trend_consistency'] = np.sign(df['close'] - df['close'].shift(20)) * np.sign(df['close'].shift(5) - df['close'].shift(20))
    df['trend_strength'] = np.abs(df['close'] - df['close'].shift(20)) / (df['close'].shift(20) + 1e-8)
    
    # 20-day high calculation
    df['high_20d'] = df['high'].rolling(window=20, min_periods=1).max()
    df['trend_exhaustion'] = ((df['high'] == df['high_20d']) & (df['close'] < df['open'])).astype(float)
    
    # Divergence pressure score
    df['momentum_gap'] = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 1e-8) - (df['close'] - df['close'].shift(20)) / (df['close'].shift(20) + 1e-8)
    df['divergence_pressure'] = df['momentum_gap'] * df['trend_consistency'] * (1 - df['trend_exhaustion'])
    
    # Volume-Accelerated Reversal Detection
    # Volume pressure asymmetry
    df['price_up'] = (df['close'] > df['close'].shift(1)).astype(float)
    df['price_down'] = (df['close'] < df['close'].shift(1)).astype(float)
    
    # Rolling calculations for volume acceleration
    up_volume_5d = []
    down_volume_5d = []
    
    for i in range(len(df)):
        if i < 5:
            up_volume_5d.append(np.nan)
            down_volume_5d.append(np.nan)
            continue
            
        window_data = df.iloc[i-4:i+1]
        up_vol = (window_data['volume'] * window_data['price_up']).sum()
        down_vol = (window_data['volume'] * window_data['price_down']).sum()
        up_volume_5d.append(up_vol)
        down_volume_5d.append(down_vol)
    
    df['up_volume_5d'] = up_volume_5d
    df['down_volume_5d'] = down_volume_5d
    df['volume_pressure_ratio'] = df['up_volume_5d'] / (df['down_volume_5d'] + 1e-8)
    
    # Volume climax momentum
    df['volume_median_20d'] = df['volume'].rolling(window=20, min_periods=1).median()
    df['volume_spike_intensity'] = df['volume'] / (df['volume_median_20d'] + 1e-8)
    
    df['price_reversal'] = (np.sign(df['close'] - df['close'].shift(1)) != np.sign(df['close'].shift(1) - df['close'].shift(2))).astype(float)
    df['volume_climax'] = ((df['volume_spike_intensity'] > 1.5) & (df['price_reversal'] == 1)).astype(float)
    df['climax_momentum'] = df['volume_spike_intensity'] * df['volume_climax']
    
    # Failed breakout volume patterns
    df['volume_avg_5d'] = df['volume'].rolling(window=5, min_periods=1).mean()
    df['high_rejection'] = ((df['high'] > df['high'].shift(1)) & (df['close'] < df['open'])).astype(float)
    df['low_absorption'] = ((df['low'] < df['low'].shift(1)) & (df['close'] > df['open'])).astype(float)
    
    df['high_rejection_pressure'] = (df['volume'] * df['high_rejection']) / (df['volume_avg_5d'] + 1e-8)
    df['low_absorption_pressure'] = (df['volume'] * df['low_absorption']) / (df['volume_avg_5d'] + 1e-8)
    df['volume_rejection_ratio'] = df['high_rejection_pressure'] / (df['low_absorption_pressure'] + 1e-8)
    
    # Regime-Adaptive Pressure Filtering
    # Volatility pressure regime
    df['price_range_20d'] = (df['high'].rolling(window=20, min_periods=1).max() - df['low'].rolling(window=20, min_periods=1).min()) / df['close'].shift(20)
    df['price_range_15d'] = (df['high'].rolling(window=15, min_periods=1).max() - df['low'].rolling(window=15, min_periods=1).min()) / df['close'].shift(15)
    
    df['high_vol_pressure'] = (df['price_range_20d'] > df['price_range_15d']).astype(float)
    df['low_vol_pressure'] = (df['price_range_20d'] < df['price_range_15d']).astype(float)
    df['volatility_regime'] = df['high_vol_pressure'] - df['low_vol_pressure']
    
    # Trend pressure regime
    df['high_20d_prev'] = df['high'].rolling(window=20, min_periods=1).max().shift(1)
    df['low_20d_prev'] = df['low'].rolling(window=20, min_periods=1).min().shift(1)
    
    df['bull_trend_pressure'] = (df['close'] > df['high_20d_prev']).astype(float)
    df['bear_trend_pressure'] = (df['close'] < df['low_20d_prev']).astype(float)
    df['trend_regime'] = df['bull_trend_pressure'] - df['bear_trend_pressure']
    
    # Regime pressure multiplier
    df['volatility_weight'] = 1 + 0.5 * df['volatility_regime']
    df['trend_confirmation'] = 1 + 0.3 * df['trend_regime']
    df['regime_multiplier'] = df['volatility_weight'] * df['trend_confirmation']
    
    # Composite Reversal Alpha
    # Momentum reversal pressure score
    df['momentum_reversal'] = df['divergence_pressure'] * df['volume_pressure_ratio'] * df['regime_multiplier']
    
    # Volume-confirmed reversal signals
    df['volume_confirmation'] = df['climax_momentum'] * (1 + df['volume_rejection_ratio'])
    
    # Final alpha factor
    alpha = df['momentum_reversal'] * df['volume_confirmation']
    
    # Clean up and return
    alpha = alpha.replace([np.inf, -np.inf], np.nan)
    alpha = alpha.fillna(method='ffill').fillna(0)
    
    return alpha
