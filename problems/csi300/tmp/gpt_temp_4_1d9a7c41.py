import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adaptive Acceleration Momentum factor
    """
    data = df.copy()
    
    # Price Acceleration Components
    # Ultra-Short (1-2 day) vs Short-Term (3-5 day) acceleration difference
    data['price_1d'] = data['close'].pct_change(1)
    data['price_2d'] = data['close'].pct_change(2)
    data['price_3d'] = data['close'].pct_change(3)
    data['price_5d'] = data['close'].pct_change(5)
    
    # Acceleration calculations
    data['accel_ultra_short'] = data['price_1d'] - data['price_2d'].shift(1)
    data['accel_short_term'] = data['price_3d'] - data['price_5d'].shift(2)
    data['accel_divergence'] = data['accel_ultra_short'] - data['accel_short_term']
    
    # Acceleration direction consistency
    data['accel_consistency'] = np.sign(data['accel_ultra_short']) * np.sign(data['accel_short_term'])
    
    # Gap Momentum Analysis
    data['overnight_gap'] = (data['open'] / data['close'].shift(1) - 1).abs()
    data['intraday_gap'] = data['close'] / data['open'] - 1
    data['gap_persistence'] = data['intraday_gap'].rolling(window=3, min_periods=2).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and np.std(x) > 0 else 0, raw=False
    )
    
    # Volume-Amount Asymmetry
    data['volume_change'] = data['volume'].pct_change(1)
    data['amount_change'] = data['amount'].pct_change(1)
    
    # Volume acceleration difference between up and down days
    up_days = data['price_1d'] > 0
    down_days = data['price_1d'] < 0
    
    data['volume_up_accel'] = data['volume_change'].where(up_days).rolling(window=5, min_periods=3).mean()
    data['volume_down_accel'] = data['volume_change'].where(down_days).rolling(window=5, min_periods=3).mean()
    data['volume_asymmetry'] = data['volume_up_accel'] - data['volume_down_accel']
    
    # Volume-amount flow imbalance
    data['volume_amount_imbalance'] = (data['volume_change'] - data['amount_change']).rolling(window=5, min_periods=3).mean()
    
    # Volatility Context Integration
    # Average True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_3d'] = data['true_range'].rolling(window=3, min_periods=3).mean()
    
    # Volatility trend
    data['volatility_trend'] = data['atr_3d'] / data['atr_3d'].shift(5)
    
    # Microstructure Efficiency
    data['range_utilization'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['volume_concentration'] = data['volume'].rolling(window=5, min_periods=5).mean() / \
                                 data['volume'].rolling(window=20, min_periods=20).mean()
    
    # Regime-Specific Synchronization
    data['price_volume_alignment'] = np.sign(data['price_1d']) * np.sign(data['volume_change'])
    data['gap_volume_coordination'] = data['overnight_gap'] * data['volume_change']
    
    # Multi-Dimensional Convergence
    # Bullish vs bearish acceleration pattern classification
    bullish_pattern = (data['accel_ultra_short'] > 0) & (data['accel_short_term'] > 0) & (data['accel_consistency'] > 0)
    bearish_pattern = (data['accel_ultra_short'] < 0) & (data['accel_short_term'] < 0) & (data['accel_consistency'] > 0)
    
    data['accel_pattern_strength'] = 0
    data.loc[bullish_pattern, 'accel_pattern_strength'] = 1
    data.loc[bearish_pattern, 'accel_pattern_strength'] = -1
    
    # Multi-dimensional acceleration persistence
    data['multi_accel_persistence'] = (data['accel_pattern_strength'].rolling(window=3, min_periods=2).sum() / 3.0)
    
    # Volatility-Adaptive Signal Processing
    high_vol = data['atr_3d'] > data['atr_3d'].rolling(window=20, min_periods=20).quantile(0.7)
    low_vol = data['atr_3d'] < data['atr_3d'].rolling(window=20, min_periods=20).quantile(0.3)
    
    # Signal components based on volatility regime
    data['signal_high_vol'] = data['accel_ultra_short'] * data['overnight_gap'] * data['volume_change']
    data['signal_low_vol'] = data['accel_divergence'] * data['gap_persistence']
    data['signal_normal_vol'] = (data['accel_divergence'] + data['price_volume_alignment'] + data['gap_volume_coordination']) / 3
    
    # Microstructure-Confirmed Signals
    data['volume_weighted_accel'] = data['accel_divergence'] * data['volume_concentration']
    data['efficiency_adjusted_accel'] = data['accel_divergence'] * data['range_utilization']
    
    # Final Alpha Generation
    # Component Weighting
    primary_component = data['accel_divergence']
    secondary_component = data['overnight_gap'] * data['gap_persistence']
    tertiary_component = data['volume_asymmetry'] + data['volume_amount_imbalance']
    
    # Volatility-Adaptive Scaling
    volatility_scale = data['atr_3d'] / data['atr_3d'].rolling(window=20, min_periods=20).mean()
    trend_adjustment = np.where(data['volatility_trend'] > 1, 1.2, 0.8)
    
    # Synchronization Enhancement
    volume_confirmation = data['price_volume_alignment'] * data['volume_concentration']
    efficiency_enhancement = data['range_utilization'].fillna(0)
    
    # Combine all components with volatility adaptation
    alpha = (
        primary_component * 0.4 +
        secondary_component * 0.3 +
        tertiary_component * 0.3
    ) * volatility_scale * trend_adjustment
    
    # Apply synchronization enhancements
    alpha = alpha * (1 + 0.2 * volume_confirmation) * (1 + 0.1 * efficiency_enhancement)
    
    # Final smoothing and normalization
    alpha = alpha.rolling(window=3, min_periods=3).mean()
    alpha = (alpha - alpha.rolling(window=20, min_periods=20).mean()) / alpha.rolling(window=20, min_periods=20).std()
    
    return alpha
