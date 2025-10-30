import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Analysis
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_decay'] = data['momentum_3d'] - data['momentum_8d']
    
    # Volatility Context Classification
    data['daily_return'] = data['close'].pct_change()
    data['volatility_15d'] = data['daily_return'].rolling(window=15, min_periods=10).std()
    data['volatility_median_30d'] = data['volatility_15d'].rolling(window=30, min_periods=20).median()
    data['volatility_regime'] = np.where(data['volatility_15d'] > data['volatility_median_30d'], 1, 0)
    data['volatility_intensity'] = data['volatility_15d'] / data['volatility_median_30d']
    
    # Regime-Adjusted Momentum Quality
    data['momentum_quality'] = data['momentum_3d'] * (1 + data['volatility_intensity'] * np.where(data['volatility_regime'] == 1, -0.2, 0.1))
    
    # Intraday Range Efficiency
    data['range_ratio'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['efficiency_5d_persistence'] = data['range_ratio'].rolling(window=5, min_periods=3).mean()
    data['efficiency_10d_pattern'] = data['range_ratio'].rolling(window=10, min_periods=7).std()
    data['range_efficiency'] = data['efficiency_5d_persistence'] / (1 + data['efficiency_10d_pattern'])
    
    # Volume Asymmetry Assessment
    data['positive_return'] = data['daily_return'] > 0
    data['negative_return'] = data['daily_return'] < 0
    
    up_volume = []
    down_volume = []
    for i in range(len(data)):
        if i >= 10:
            up_mask = data['positive_return'].iloc[i-9:i+1]
            down_mask = data['negative_return'].iloc[i-9:i+1]
            up_vol = data['volume'].iloc[i-9:i+1][up_mask].mean() if up_mask.any() else np.nan
            down_vol = data['volume'].iloc[i-9:i+1][down_mask].mean() if down_mask.any() else np.nan
        else:
            up_vol = np.nan
            down_vol = np.nan
        up_volume.append(up_vol)
        down_volume.append(down_vol)
    
    data['up_day_volume'] = up_volume
    data['down_day_volume'] = down_volume
    data['volume_asymmetry'] = data['up_day_volume'] / data['down_day_volume']
    data['volume_asymmetry_trend'] = data['volume_asymmetry'].rolling(window=5, min_periods=3).mean()
    
    # Gap Opening Quality
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_extremeness'] = data['opening_gap'].abs() / data['opening_gap'].abs().rolling(window=20, min_periods=15).std()
    data['gap_absorption'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume-Momentum Divergence
    data['momentum_direction'] = np.sign(data['momentum_3d'])
    data['volume_trend_direction'] = np.sign(data['volume_asymmetry_trend'].diff())
    data['volume_momentum_divergence'] = np.where(
        data['momentum_direction'] != data['volume_trend_direction'], 
        -data['momentum_3d'].abs(), 
        data['momentum_3d'].abs()
    )
    
    # Range Breakout Quality
    data['high_15d'] = data['high'].rolling(window=15, min_periods=10).max()
    data['low_15d'] = data['low'].rolling(window=15, min_periods=10).min()
    data['range_breakout'] = np.where(
        (data['high'] > data['high_15d'].shift(1)) | (data['low'] < data['low_15d'].shift(1)), 1, 0
    )
    data['volume_10d_avg'] = data['volume'].rolling(window=10, min_periods=7).mean()
    data['breakout_volume_ratio'] = data['volume'] / data['volume_10d_avg']
    data['breakout_sustainability'] = data['range_breakout'].rolling(window=2, min_periods=2).sum()
    data['breakout_quality'] = data['range_breakout'] * data['breakout_volume_ratio'] * data['breakout_sustainability']
    
    # Failed Momentum Patterns
    data['high_momentum_low_efficiency'] = np.where(
        (data['momentum_3d'].abs() > data['momentum_3d'].abs().rolling(window=20, min_periods=15).quantile(0.7)) & 
        (data['range_efficiency'] < data['range_efficiency'].rolling(window=20, min_periods=15).quantile(0.3)),
        -1, 0
    )
    
    data['strong_gap_poor_absorption'] = np.where(
        (data['gap_extremeness'] > 1.5) & (data['gap_absorption'].abs() < 0.3),
        -1, 0
    )
    
    data['momentum_decay_volume_contradiction'] = np.where(
        (data['momentum_decay'] < -0.02) & (data['volume_momentum_divergence'] < 0),
        -1, 0
    )
    
    # Generate Adaptive Composite Alpha
    # Combine Regime-Aware Momentum with Efficiency
    base_signal = data['momentum_quality'] * data['range_efficiency']
    volume_adjusted = base_signal * (1 + 0.2 * np.tanh(data['volume_asymmetry'] - 1))
    volatility_scaled = volume_adjusted * (1 + 0.1 * (1 - data['volatility_intensity']))
    
    # Incorporate Divergence and Breakout Signals
    divergence_enhanced = volatility_scaled + 0.3 * data['volume_momentum_divergence']
    breakout_weighted = divergence_enhanced * (1 + 0.15 * data['breakout_quality'])
    
    # Penalize failed momentum patterns
    failed_pattern_penalty = (
        data['high_momentum_low_efficiency'] + 
        data['strong_gap_poor_absorption'] + 
        data['momentum_decay_volume_contradiction']
    )
    
    final_signal = breakout_weighted * (1 + 0.1 * failed_pattern_penalty)
    
    # Final Signal Refinement
    regime_adaptive_score = final_signal * np.where(
        data['volatility_regime'] == 1, 
        0.8,  # Reduce signal in high volatility
        1.2   # Enhance signal in low volatility
    )
    
    # Multi-timeframe confirmation
    momentum_confirmation = np.sign(data['momentum_3d']) == np.sign(data['momentum_8d'])
    confirmed_signal = regime_adaptive_score * np.where(momentum_confirmation, 1.1, 0.9)
    
    # Gap and absorption context
    gap_context = 1 + 0.1 * np.tanh(data['gap_absorption'] * data['gap_extremeness'])
    final_alpha = confirmed_signal * gap_context
    
    # Clean and return
    alpha_series = pd.Series(final_alpha, index=data.index)
    alpha_series = alpha_series.replace([np.inf, -np.inf], np.nan)
    
    return alpha_series
