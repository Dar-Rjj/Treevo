import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Helper functions
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    # Price-Momentum Acceleration Framework
    # Multi-Timeframe Price Acceleration
    data['short_term_accel'] = (data['close'] / data['close'].shift(2) - 1) - (data['close'].shift(2) / data['close'].shift(4) - 1)
    data['medium_term_accel'] = (data['close'] / data['close'].shift(5) - 1) - (data['close'].shift(5) / data['close'].shift(10) - 1)
    data['accel_divergence'] = data['short_term_accel'] - data['medium_term_accel']
    
    # Volume Acceleration Analysis
    data['volume_change_rate'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_accel'] = data['volume_change_rate'] - (data['volume'].shift(1) / data['volume'].shift(2) - 1)
    data['volume_momentum_alignment'] = np.sign(data['volume_accel']) * np.sign(data['short_term_accel'])
    
    # Asymmetric Response Detection
    data['is_up_day'] = data['close'] > data['close'].shift(1)
    data['is_down_day'] = data['close'] < data['close'].shift(1)
    
    # Rolling averages for up/down days
    up_volume_avg = []
    down_volume_avg = []
    for i in range(len(data)):
        if i < 9:
            up_volume_avg.append(np.nan)
            down_volume_avg.append(np.nan)
            continue
        window = data.iloc[i-9:i+1]
        up_volume = window.loc[window['is_up_day'], 'volume']
        down_volume = window.loc[window['is_down_day'], 'volume']
        up_volume_avg.append(up_volume.mean() if len(up_volume) > 0 else np.nan)
        down_volume_avg.append(down_volume.mean() if len(down_volume) > 0 else np.nan)
    
    data['up_volume_avg'] = up_volume_avg
    data['down_volume_avg'] = down_volume_avg
    data['up_move_volume_sensitivity'] = data['volume'] / data['up_volume_avg']
    data['down_move_volume_sensitivity'] = data['volume'] / data['down_volume_avg']
    data['volume_asymmetry_ratio'] = data['up_move_volume_sensitivity'] / data['down_move_volume_sensitivity']
    
    # Market Microstructure Quality Assessment
    # Price Efficiency Metrics
    data['opening_gap_efficiency'] = np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['intraday_return_consistency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['price_discovery_quality'] = 1 - (np.abs(data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low']))
    
    # Trading Activity Quality
    data['volume_concentration'] = data['volume'] / (data['high'] - data['low'])
    data['trade_size_proxy'] = data['amount'] / data['volume']
    
    # Activity Stability
    data['volume_concentration_std'] = data['volume_concentration'].rolling(window=5, min_periods=3).std()
    data['activity_stability'] = 1 / data['volume_concentration_std']
    
    # Microstructure Noise Filters
    data['efficiency_score'] = data['price_discovery_quality'] * (1 - data['opening_gap_efficiency'])
    data['activity_quality'] = data['volume_concentration'] * data['trade_size_proxy']
    data['microstructure_quality'] = data['efficiency_score'] * data['activity_quality'] * data['activity_stability']
    
    # Regime-Based Signal Enhancement
    # Volatility Regime Classification
    data['true_range'] = true_range(data['high'], data['low'], data['close'].shift(1))
    data['true_range_avg_20'] = data['true_range'].rolling(window=20, min_periods=10).mean()
    data['high_vol_regime'] = data['true_range'] > data['true_range_avg_20']
    data['low_vol_regime'] = data['true_range'] < data['true_range_avg_20']
    data['volatility_transition'] = data['true_range'] / data['true_range'].shift(1) - 1
    
    # Liquidity Regime Classification
    data['volume_avg_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['high_liquidity_regime'] = data['volume'] > data['volume_avg_20']
    data['low_liquidity_regime'] = data['volume'] < data['volume_avg_20']
    data['liquidity_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    
    # Regime-Specific Signal Weights
    data['regime_weight'] = 1.0
    hv_hl_mask = data['high_vol_regime'] & data['high_liquidity_regime']
    hv_ll_mask = data['high_vol_regime'] & data['low_liquidity_regime']
    lv_hl_mask = data['low_vol_regime'] & data['high_liquidity_regime']
    lv_ll_mask = data['low_vol_regime'] & data['low_liquidity_regime']
    
    data.loc[hv_hl_mask, 'regime_weight'] = 1.3 * (1 + data.loc[hv_hl_mask, 'volatility_transition'])
    data.loc[hv_ll_mask, 'regime_weight'] = 0.7 * (1 + data.loc[hv_ll_mask, 'volume_momentum_alignment'])
    data.loc[lv_hl_mask, 'regime_weight'] = 1.1 * (1 + data.loc[lv_hl_mask, 'liquidity_momentum'])
    data.loc[lv_ll_mask, 'regime_weight'] = 0.9 * data.loc[lv_ll_mask, 'microstructure_quality']
    
    # Core Factor Construction
    # Base Acceleration Factor
    data['raw_acceleration'] = data['accel_divergence'] * data['volume_asymmetry_ratio']
    data['quality_adjusted_acceleration'] = data['raw_acceleration'] * data['microstructure_quality']
    data['volume_confirmed_acceleration'] = data['quality_adjusted_acceleration'] * data['volume_momentum_alignment']
    
    # Regime-Adaptive Scaling
    data['volatility_scaling_factor'] = 1 / (1 + data['true_range'] / data['close'])
    data['liquidity_scaling_factor'] = data['volume'] / data['volume_avg_20']
    data['scaled_acceleration'] = data['volume_confirmed_acceleration'] * data['volatility_scaling_factor'] * data['liquidity_scaling_factor']
    
    # Signal Persistence Enhancement
    data['accel_sign_match'] = np.sign(data['short_term_accel']) == np.sign(data['medium_term_accel'])
    data['volume_increase'] = data['volume'] > data['volume'].shift(1)
    
    data['acceleration_persistence'] = data['accel_sign_match'].rolling(window=5, min_periods=3).sum()
    data['volume_persistence'] = data['volume_increase'].rolling(window=5, min_periods=3).sum()
    
    data['persistence_enhanced_factor'] = data['scaled_acceleration'] * (1 + data['acceleration_persistence']/5) * (1 + data['volume_persistence']/5)
    
    # Final Alpha Factor Synthesis
    # Regime-Weighted Final Factor
    data['base_factor'] = data['persistence_enhanced_factor']
    data['regime_adapted_factor'] = data['base_factor'] * data['regime_weight']
    
    # Risk-Adjusted Refinement
    data['max_high_10'] = data['high'].rolling(window=10, min_periods=5).max()
    data['drawdown_protection'] = 1 - (data['max_high_10'] - data['close']) / data['close']
    
    data['returns'] = data['close'].pct_change()
    data['returns_std_10'] = data['returns'].rolling(window=10, min_periods=5).std()
    data['volatility_adjustment'] = 1 / (1 + data['returns_std_10'])
    
    data['risk_adjusted_factor'] = data['regime_adapted_factor'] * data['drawdown_protection'] * data['volatility_adjustment']
    
    # Final Alpha Output
    data['signal_strength'] = np.abs(data['risk_adjusted_factor'])
    data['direction'] = np.sign(data['risk_adjusted_factor'])
    data['final_alpha'] = data['risk_adjusted_factor'] * (1 + data['signal_strength'])
    
    return data['final_alpha']
