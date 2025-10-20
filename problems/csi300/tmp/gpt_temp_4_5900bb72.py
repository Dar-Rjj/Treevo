import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Momentum-Pressure Dynamics
    # Multi-Timeframe Momentum Components
    data['short_term_momentum'] = data['close'] / data['close'].shift(3) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_divergence'] = data['short_term_momentum'] - data['medium_term_momentum']
    data['price_acceleration'] = data['short_term_momentum'] - data['medium_term_momentum']
    
    # Momentum persistence: Count(Close_t > Close_{t-1} for past 5 days) / 5
    data['close_gt_prev'] = (data['close'] > data['close'].shift(1)).astype(int)
    data['momentum_persistence'] = data['close_gt_prev'].rolling(window=5).sum() / 5
    
    # Pressure Framework
    data['net_pressure'] = (2 * data['close'] - data['high'] - data['low']) / (data['high'] - data['low'])
    data['pressure_momentum'] = data['net_pressure'] - data['net_pressure'].shift(3)
    
    # Cumulative pressure: Σ[(Close_i - Open_i) / (High_i - Low_i)] for i=t-2:t
    data['daily_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['cumulative_pressure'] = data['daily_pressure'].rolling(window=3).sum()
    
    data['buying_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['selling_pressure'] = (data['high'] - data['close']) / (data['high'] - data['low'])
    
    # Momentum-Pressure Integration
    data['pressure_weighted_momentum'] = data['momentum_divergence'] * data['net_pressure']
    data['acceleration_pressure_alignment'] = data['price_acceleration'] * data['pressure_momentum']
    data['volume_pressure_momentum'] = data['volume'] * data['net_pressure'] * data['price_acceleration']
    data['momentum_persistence_pressure'] = data['momentum_persistence'] * data['cumulative_pressure']
    
    # Range-Efficiency Reversal System
    # Efficiency Components
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['cumulative_efficiency'] = data['range_efficiency'].rolling(window=3).sum()
    data['gap_efficiency'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Reversal Detection
    data['short_term_reversal'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Medium-term reversal: (Close_t - Close_{t-5}) / (Max(High_{t-5:t}) - Min(Low_{t-5:t}))
    data['high_5d'] = data['high'].rolling(window=5).max()
    data['low_5d'] = data['low'].rolling(window=5).min()
    data['medium_term_reversal'] = (data['close'] - data['close'].shift(5)) / (data['high_5d'] - data['low_5d'])
    
    # Recent high-low reversals
    data['high_gt_prev'] = (data['high'] > data['high'].shift(1)).astype(int)
    data['close_lt_prev'] = (data['close'] < data['close'].shift(1)).astype(int)
    data['high_low_reversal'] = (data['high_gt_prev'] & data['close_lt_prev']).astype(int)
    data['recent_high_low_reversals'] = data['high_low_reversal'].rolling(window=5).sum()
    
    # Failed breakouts
    data['close_lt_open'] = (data['close'] < data['open']).astype(int)
    data['failed_breakout'] = (data['high_gt_prev'] & data['close_lt_open']).astype(int)
    data['failed_breakouts'] = data['failed_breakout'].rolling(window=3).sum()
    
    data['reversal_convergence'] = data['short_term_reversal'] - data['medium_term_reversal']
    
    # Efficiency-Reversal Integration
    data['efficiency_weighted_reversal'] = data['reversal_convergence'] * data['range_efficiency']
    data['gap_recovery_momentum'] = data['price_acceleration'] * data['gap_efficiency']
    
    # Compression-reversal: (High_t - Low_t)/MA(High-Low, 5)_{t-1} × Recent high-low reversals
    data['high_low_range'] = data['high'] - data['low']
    data['ma_high_low_5'] = data['high_low_range'].rolling(window=5).mean().shift(1)
    data['compression_reversal'] = (data['high_low_range'] / data['ma_high_low_5']) * data['recent_high_low_reversals']
    
    data['volume_efficiency_reversal'] = data['volume'] * data['intraday_efficiency'] * data['reversal_convergence']
    
    # Volume-Acceleration Convergence
    # Volume Dynamics
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(5) - 1) - (data['volume'].shift(5) / data['volume'].shift(10) - 1)
    data['volume_reversal_ratio'] = (data['volume'] - data['volume'].shift(1)) / (data['volume'] + data['volume'].shift(1))
    
    # Volume trend reversal
    data['volume_trend_1'] = np.sign(data['volume'] - data['volume'].shift(1))
    data['volume_trend_2'] = np.sign(data['volume'].shift(1) - data['volume'].shift(2))
    data['volume_trend_reversal'] = data['volume_trend_1'] * data['volume_trend_2']
    data['volume_reversal_momentum'] = data['volume_reversal_ratio'] * data['volume_trend_reversal']
    
    # Acceleration Framework
    data['price_volume_acceleration'] = data['price_acceleration'] * data['volume_acceleration']
    
    # Volume breakout and contraction
    data['volume_breakout'] = (data['volume'] > 1.5 * data['volume'].shift(5)).astype(float)
    data['ma_volume_5'] = data['volume'].rolling(window=5).mean().shift(1)
    data['volume_contraction'] = (data['volume'] < 0.8 * data['ma_volume_5']).astype(float)
    data['volume_divergence'] = ((data['volume'] > 1.3 * data['volume'].shift(1)) & (data['range_efficiency'] < 0)).astype(float)
    
    # Volume-Acceleration Integration
    data['pressure_volume_acceleration'] = data['volume_acceleration'] * data['pressure_momentum']
    data['reversal_volume_convergence'] = data['volume_reversal_momentum'] * data['reversal_convergence']
    data['efficiency_volume_alignment'] = data['volume_momentum'] * data['range_efficiency'] * data['price_acceleration']
    data['volume_breakout_pressure'] = data['volume_breakout'] * data['cumulative_pressure'] * data['momentum_divergence']
    
    # Multi-Regime Context Processing
    # Volatility Regime Detection
    data['std_close_5'] = data['close'].rolling(window=5).std()
    data['std_close_10'] = data['close'].rolling(window=10).std()
    data['volatility_ratio'] = data['std_close_5'] / data['std_close_10']
    
    data['range_expansion'] = data['high_low_range'] / data['ma_high_low_5']
    data['daily_range'] = data['high_low_range'] / data['close']
    
    # Volatility expansion
    data['std_close_3'] = data['close'].rolling(window=3).std()
    data['std_close_10_shift'] = data['close'].rolling(window=10).std().shift(3)
    data['volatility_expansion'] = data['std_close_3'] / data['std_close_10_shift']
    
    data['high_volatility'] = ((data['volatility_ratio'] > 1) & (data['range_efficiency'] > 0.8)).astype(float)
    data['low_volatility'] = ((data['volatility_ratio'] <= 1) & (data['range_efficiency'] < 0.8)).astype(float)
    
    data['ma_daily_range_20'] = data['daily_range'].rolling(window=20).mean().shift(1)
    data['extreme_range'] = (data['daily_range'] > 2 * data['ma_daily_range_20']).astype(float)
    
    # Trend-Reversal Assessment
    data['trend_reversal_potential'] = (data['close'] - data['close'].shift(5)) * (data['close'].shift(5) - data['close'].shift(10))
    data['strong_reversal'] = (np.abs(data['trend_reversal_potential']) > 0.03 * data['close']).astype(float)
    data['reversal_regime'] = (data['strong_reversal'] & data['volume_divergence']).astype(float)
    
    # Volume Regime Classification
    data['volume_expansion'] = (data['volume'] > 1.2 * data['ma_volume_5']).astype(float)
    data['volume_contraction_regime'] = (data['volume'] < 0.8 * data['ma_volume_5']).astype(float)
    data['volume_stability'] = ((data['volume'] >= 0.8 * data['ma_volume_5']) & (data['volume'] <= 1.2 * data['ma_volume_5'])).astype(float)
    
    # Regime-Adaptive Signal Synthesis
    # High Volatility + Expansion Processing
    high_vol_factor = (data['pressure_weighted_momentum'] * data['volume_acceleration'] * data['volatility_expansion'] +
                      data['acceleration_pressure_alignment'] * data['range_expansion'] * data['volume_breakout'])
    
    # Low Volatility + Efficiency Processing
    low_vol_factor = (data['efficiency_weighted_reversal'] * data['cumulative_efficiency'] * data['momentum_persistence'] +
                     data['compression_reversal'] * data['volume_stability'] * data['gap_efficiency'])
    
    # Reversal + Volume Divergence Processing
    reversal_factor = (data['reversal_volume_convergence'] * data['trend_reversal_potential'] * data['volume_divergence'] +
                      data['gap_recovery_momentum'] * data['failed_breakouts'] * data['volume_reversal_momentum'])
    
    # Normal Regime Processing
    normal_factor = (data['momentum_persistence_pressure'] * data['efficiency_volume_alignment'] * 
                    data['pressure_volume_acceleration'])
    
    # Unified Alpha Generation - Regime-Specific Factor Selection
    alpha = (
        data['high_volatility'] * high_vol_factor +
        data['low_volatility'] * low_vol_factor +
        data['reversal_regime'] * reversal_factor +
        (1 - data['high_volatility'] - data['low_volatility'] - data['reversal_regime']) * normal_factor
    )
    
    return alpha
