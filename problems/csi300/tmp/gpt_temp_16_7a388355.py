import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Helper functions
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    def atr(high, low, close, window):
        tr = true_range(high, low, close.shift(1))
        return tr.rolling(window=window).mean()
    
    def vwap(close, volume, window):
        return (close * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
    
    def count_consistency(series, window):
        return series.rolling(window=window).apply(lambda x: len(set(np.sign(x.dropna()))) == 1 if len(x.dropna()) == window else np.nan, raw=False)
    
    # Multi-Scale Gap-Momentum Analysis
    data['gap_size'] = data['open'] / data['close'].shift(1) - 1
    data['gap_persistence'] = count_consistency(data['gap_size'], 5)
    data['gap_type'] = data['gap_size'].abs() / data['gap_size'].abs().rolling(window=20).mean()
    
    # Price Path Efficiency Calculation
    for window in [3, 5, 20]:
        price_changes = data['close'].diff().abs()
        cumulative_movement = price_changes.rolling(window=window).sum()
        total_movement = (data['close'] - data['close'].shift(window)).abs()
        data[f'efficiency_{window}day'] = total_movement / cumulative_movement
    
    data['efficiency_gradient'] = (data['efficiency_5day'] - data['efficiency_3day']) * np.sign(data['close'] - data['close'].shift(3))
    
    # Volume-Pressure Dynamics
    data['intraday_pressure'] = (data['close'] - data['open']) * data['volume']
    data['pressure_persistence'] = count_consistency(data['intraday_pressure'], 5)
    
    data['price_volume_association'] = np.sign(data['close'].diff()) * np.sign(data['volume'].diff())
    data['volume_price_correlation'] = data['close'].rolling(window=3).corr(data['volume'])
    data['association_persistence'] = count_consistency(data['price_volume_association'], 5)
    
    # Volatility Regime Classification
    data['atr_10'] = atr(data['high'], data['low'], data['close'], 10)
    data['atr_30'] = atr(data['high'], data['low'], data['close'], 30)
    data['volatility_ratio'] = data['atr_10'] / data['atr_30']
    
    data['daily_range'] = data['high'] - data['low']
    data['range_5day_avg'] = data['daily_range'].rolling(window=5).mean()
    data['range_20day_median'] = data['daily_range'].rolling(window=20).median()
    
    conditions = [
        (data['volatility_ratio'] > 1.2) | (data['range_5day_avg'] > 1.3 * data['range_20day_median']),
        (data['volatility_ratio'] < 0.8) | (data['range_5day_avg'] < 0.8 * data['range_20day_median'])
    ]
    choices = ['high', 'low']
    data['volatility_regime'] = np.select(conditions, choices, default='normal')
    
    # Microstructure-Liquidity Flow Integration
    data['bullish_pressure'] = (data['close'] - data['low']) * data['volume']
    data['bearish_pressure'] = (data['high'] - data['close']) * data['volume']
    data['flow_asymmetry_ratio'] = np.log(data['bullish_pressure'].rolling(window=3).sum() / data['bearish_pressure'].rolling(window=3).sum())
    
    data['trading_efficiency'] = (data['close'] - data['open']).abs() / data['daily_range']
    data['movement_efficiency'] = data['close'].diff().abs() / true_range(data['high'], data['low'], data['close'].shift(1))
    
    positive_return_volume = data['volume'].where(data['close'].diff() > 0, 0)
    data['up_volume_concentration'] = positive_return_volume.rolling(window=5).sum() / data['volume'].rolling(window=5).sum()
    
    # Regime-Adaptive Signal Construction
    data['short_term_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(20) - 1
    data['momentum_divergence'] = data['short_term_momentum'] - data['medium_term_momentum']
    
    data['base_momentum_signal'] = data['gap_persistence'] * data['pressure_persistence'] * data['efficiency_gradient'] * data['association_persistence']
    
    # Regime-Adaptive Momentum Acceleration
    data['momentum_acceleration'] = np.nan
    high_vol_mask = data['volatility_regime'] == 'high'
    low_vol_mask = data['volatility_regime'] == 'low'
    normal_vol_mask = data['volatility_regime'] == 'normal'
    
    data.loc[high_vol_mask, 'momentum_acceleration'] = (data['close'] - data['close'].shift(3)) - (data['close'].shift(3) - data['close'].shift(6))
    data.loc[low_vol_mask, 'momentum_acceleration'] = (data['close'] - data['close'].shift(8)) - (data['close'].shift(8) - data['close'].shift(16))
    data.loc[normal_vol_mask, 'momentum_acceleration'] = (data['close'] - data['close'].shift(5)) - (data['close'].shift(5) - data['close'].shift(10))
    
    # Volume Confirmation System
    data['volume_range_alignment'] = data['volume'].rolling(window=5).corr(data['daily_range'])
    data['vwap_8'] = vwap(data['close'], data['volume'], 8)
    data['atr_8'] = atr(data['high'], data['low'], data['close'], 8)
    data['volume_price_divergence'] = (data['close'] - data['vwap_8']) / data['atr_8']
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['volume_confirmation_score'] = data['volume_range_alignment'] * (1 - data['volume_price_divergence'].abs()) * data['volume_ratio']
    
    # Price-Volume-Liquidity Coherence Classification
    data['liquidity_divergence'] = (data['volume'] / data['volume'].shift(1) - 1) - (data['close'].pct_change() / data['close'].pct_change().shift(1) - 1)
    data['volume_ma_10'] = data['volume'].rolling(window=10).mean()
    
    coherence_conditions = [
        (data['liquidity_divergence'].abs() < 0.1) & (data['volume'] > data['volume_ma_10']),
        (data['liquidity_divergence'].abs() > 0.3) | (data['volume'] < 0.7 * data['volume_ma_10'])
    ]
    coherence_choices = ['high', 'low']
    data['coherence_state'] = np.select(coherence_conditions, coherence_choices, default='transition')
    
    # Efficiency-Weighted Momentum Integration
    data['multi_efficiency_score'] = (data['efficiency_3day'] + data['efficiency_5day'] + data['efficiency_20day']) / 3
    data['intraday_efficiency_score'] = (data['trading_efficiency'] + data['movement_efficiency']) / 2
    data['efficiency_weighted_momentum'] = data['momentum_divergence'] * data['multi_efficiency_score'] * data['intraday_efficiency_score']
    
    # Noise-Adjusted Signal Enhancement
    data['intraday_noise'] = (data['open'] - data['close'].shift(1)).abs() / data['daily_range']
    data['noise_to_signal_ratio'] = data['intraday_noise'] / (data['close'].diff(5).abs() / atr(data['high'], data['low'], data['close'], 5))
    
    # Adaptive Factor Integration
    data['core_momentum_efficiency_signal'] = data['base_momentum_signal'] * data['momentum_acceleration']
    data['volume_confirmed_signal'] = data['core_momentum_efficiency_signal'] * data['volume_confirmation_score'] * data['up_volume_concentration'] * data['flow_asymmetry_ratio']
    data['efficiency_enhanced_signal'] = data['volume_confirmed_signal'] * data['efficiency_weighted_momentum']
    
    # Regime-Based Filtering
    regime_filter_conditions = [
        data['coherence_state'] == 'high',
        data['coherence_state'] == 'low'
    ]
    regime_filter_choices = [1.0, 0.5]
    data['regime_filter'] = np.select(regime_filter_conditions, regime_filter_choices, default=0.75)
    
    data['momentum_persistence_filter'] = count_consistency(data['core_momentum_efficiency_signal'], 3)
    
    # Final Alpha Output
    alpha = (data['efficiency_enhanced_signal'] / (1 + data['noise_to_signal_ratio'])) * data['regime_filter'] * data['momentum_persistence_filter']
    
    return alpha
