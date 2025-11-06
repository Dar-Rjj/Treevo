import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Aware Multi-Timeframe Alpha Factor
    Combines short and medium-term momentum with volume confirmation and regime detection
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Alignment
    # Short-term Momentum (1-3 days)
    df['ret_1d'] = df['close'] / df['close'].shift(1) - 1
    df['ret_3d'] = df['close'] / df['close'].shift(3) - 1
    df['price_accel'] = (df['close'] / df['close'].shift(1)) - (df['close'].shift(1) / df['close'].shift(2))
    
    # Volume-confirmed Momentum
    df['volume_accel'] = (df['volume'] / df['volume'].shift(1)) - (df['volume'].shift(1) / df['volume'].shift(2))
    df['volume_weighted_ret'] = df['ret_1d'] * (df['volume'] / df['volume'].shift(1))
    
    # Short-term composite
    short_term_signal = (df['ret_1d'] + df['ret_3d'] + df['price_accel'] + df['volume_weighted_ret']) / 4
    
    # Medium-term Momentum (5-10 days)
    df['ret_5d'] = df['close'] / df['close'].shift(5) - 1
    df['ret_10d'] = df['close'] / df['close'].shift(10) - 1
    
    # Exponential decay weights
    weights = np.array([np.exp(-0.2 * i) for i in range(10)])
    weights = weights / weights.sum()
    
    # Decay-weighted momentum
    decay_returns = []
    for i in range(len(df)):
        if i >= 10:
            returns = [df['close'].iloc[i] / df['close'].iloc[i-j] - 1 for j in range(1, 11)]
            decay_returns.append(np.sum(np.array(returns) * weights))
        else:
            decay_returns.append(np.nan)
    df['decay_weighted_momentum'] = decay_returns
    
    # Volatility-normalized Momentum
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['volatility_10d'] = df['daily_range'].rolling(window=10, min_periods=5).mean()
    df['normalized_5d_ret'] = df['ret_5d'] / (df['volatility_10d'] + 1e-8)
    df['normalized_10d_ret'] = df['ret_10d'] / (df['volatility_10d'] + 1e-8)
    
    # Medium-term composite
    medium_term_signal = (df['ret_5d'] + df['ret_10d'] + df['decay_weighted_momentum'] + 
                         df['normalized_5d_ret'] + df['normalized_10d_ret']) / 5
    
    # Timeframe Alignment Score
    alignment_multiplier = 1 + np.sign(short_term_signal) * np.sign(medium_term_signal)
    timeframe_aligned_momentum = (short_term_signal + medium_term_signal) * alignment_multiplier
    
    # Regime Detection & Scaling
    # Volatility Regime
    df['current_vol'] = (df['high'] - df['low']) / df['close']
    df['avg_vol_20d'] = df['current_vol'].rolling(window=20, min_periods=10).mean()
    df['vol_ratio'] = df['current_vol'] / (df['avg_vol_20d'] + 1e-8)
    
    vol_regime = np.where(df['vol_ratio'] > 1.5, 1, 
                         np.where(df['vol_ratio'] < 0.7, -1, 0))
    
    # Volume Regime
    df['avg_volume_20d'] = df['volume'].rolling(window=20, min_periods=10).mean()
    df['std_volume_20d'] = df['volume'].rolling(window=20, min_periods=10).std()
    df['volume_zscore'] = (df['volume'] - df['avg_volume_20d']) / (df['std_volume_20d'] + 1e-8)
    
    volume_regime = np.where(df['volume_zscore'] > 1, 1,
                            np.where(df['volume_zscore'] < -1, -1, 0))
    
    # Regime-aware Scaling
    vol_scaling = 1 + 0.3 * vol_regime
    volume_scaling = 1 + 0.2 * volume_regime
    combined_regime_multiplier = vol_scaling * volume_scaling
    
    # Composite Alpha Generation
    # Signal Integration
    base_signal = timeframe_aligned_momentum
    volume_confirmed_signal = base_signal * (1 + df['volume_accel'])
    regime_adjusted_signal = volume_confirmed_signal * combined_regime_multiplier
    
    # Risk Adjustment
    volatility_dampened = regime_adjusted_signal / (1 + df['current_vol'])
    
    # Final alpha factor
    alpha = volatility_dampened
    
    return alpha
