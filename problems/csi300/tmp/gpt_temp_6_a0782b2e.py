import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Acceleration Framework
    # Price Acceleration Components
    data['short_term_momentum_change'] = (data['close'] / data['close'].shift(1) - 1) - (data['close'].shift(1) / data['close'].shift(2) - 1)
    data['medium_term_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['acceleration_spread'] = data['short_term_momentum_change'] - data['medium_term_momentum']
    
    # Volume Acceleration Components
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_acceleration'] = data['volume_momentum'] - (data['volume'].shift(1) / data['volume'].shift(2) - 1)
    data['volume_momentum_spread'] = data['volume'] / data['volume'].shift(5) - 1
    
    # Multi-Period Divergence Analysis
    # Price-Volume Correlation Framework
    data['pv_corr_5'] = data['close'].rolling(window=5).corr(data['volume'])
    data['pv_corr_10'] = data['close'].rolling(window=10).corr(data['volume'])
    data['correlation_spread'] = data['pv_corr_5'] - data['pv_corr_10']
    
    # Divergence Magnitude Assessment
    data['price_volume_momentum_divergence'] = data['medium_term_momentum'] - data['volume_momentum_spread']
    data['acceleration_divergence'] = data['acceleration_spread'] - data['volume_acceleration']
    data['combined_divergence_score'] = data['price_volume_momentum_divergence'] * data['acceleration_divergence']
    
    # Volatility Regime Adaptation
    # Multi-Scale Volatility Measurement
    data['short_term_volatility'] = (data['high'] - data['low']) / data['close']
    data['medium_term_volatility'] = data['close'].rolling(window=10).std()
    data['volatility_ratio'] = data['short_term_volatility'] / data['medium_term_volatility']
    
    # Regime Transition Framework
    data['volatility_momentum'] = data['medium_term_volatility'] / data['medium_term_volatility'].shift(5)
    data['range_breakout_signal'] = ((data['high'] > data['high'].shift(1)) & (data['low'] < data['low'].shift(1))).astype(float)
    data['regime_score'] = data['volatility_ratio'] * data['volatility_momentum'] * data['range_breakout_signal']
    
    # Intraday Efficiency & Microstructure
    # Gap Efficiency Analysis
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    data['gap_close_ratio'] = (data['close'] - data['open']) / abs(data['overnight_gap'] * data['close'].shift(1)).replace(0, np.nan)
    data['gap_efficiency_score'] = abs(data['gap_close_ratio'])
    
    # Session Quality Metrics
    data['high_low_capture'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['bid_ask_imbalance_proxy'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['flow_quality_score'] = data['high_low_capture'] * data['bid_ask_imbalance_proxy']
    
    # Volume Dynamics Integration
    # Volume Persistence Analysis
    volume_median_10 = data['volume'].rolling(window=10).median()
    data['volume_persistence'] = data['volume'].rolling(window=5).apply(
        lambda x: (x > volume_median_10.loc[x.index]).sum() / 5, raw=False
    )
    data['volume_concentration'] = data['volume'] / volume_median_10
    
    # Volume Surge Framework
    data['volume_surge_condition'] = (data['volume_concentration'] > 2.0).astype(float)
    data['volume_leadership'] = data['volume_concentration'] * ((data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan))
    data['surge_multiplier'] = data['volume_leadership'] * data['volume_persistence']
    
    # Adaptive Factor Construction
    # Core Divergence Component
    data['base_divergence'] = data['correlation_spread'] * data['combined_divergence_score']
    data['regime_enhanced_divergence'] = data['base_divergence'] * data['regime_score']
    
    # Efficiency Adjustment
    data['efficiency_multiplier'] = data['gap_efficiency_score'] * data['flow_quality_score']
    data['enhanced_factor'] = data['regime_enhanced_divergence'] * data['efficiency_multiplier']
    
    # Volume Adaptive Finalization
    data['primary_factor'] = data['enhanced_factor']
    data['volume_surge_factor'] = data['enhanced_factor'] * data['surge_multiplier']
    
    # Final Factor Selection
    data['final_factor'] = np.where(
        data['volume_surge_condition'] > 0,
        data['volume_surge_factor'],
        data['primary_factor']
    )
    
    return data['final_factor']
