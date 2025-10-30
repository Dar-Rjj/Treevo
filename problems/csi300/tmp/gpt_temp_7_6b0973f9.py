import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Adaptive Momentum Component
    # Multi-Timeframe Momentum
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['short_term_momentum'] = (data['close'] / data['close'].shift(5)) - 1
    data['medium_term_momentum'] = (data['close'] / data['close'].shift(20)) - 1
    data['momentum_acceleration'] = data['short_term_momentum'] - data['medium_term_momentum']
    
    # Volatility-Adjusted Momentum
    data['daily_returns'] = data['close'] / data['close'].shift(1) - 1
    data['realized_volatility'] = data['daily_returns'].rolling(window=5).std()
    data['intraday_volatility'] = (data['high'] - data['low']) / data['close']
    data['volatility_ratio'] = data['intraday_volatility'] / data['realized_volatility'].replace(0, np.nan)
    
    data['vol_scaled_short'] = data['short_term_momentum'] / data['realized_volatility'].replace(0, np.nan)
    data['vol_scaled_medium'] = data['medium_term_momentum'] / data['realized_volatility'].replace(0, np.nan)
    
    # Volume-Price Efficiency Component
    # Multi-Timeframe Volume Analysis
    data['volume_ma_3'] = data['volume'].rolling(window=3).mean()
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['volume_ma_7'] = data['volume'].rolling(window=7).mean()
    data['volume_ma_10'] = data['volume'].rolling(window=10).mean()
    
    data['short_volume_trend'] = (data['volume'] / data['volume_ma_3']) - 1
    data['medium_volume_trend'] = (data['volume'] / data['volume_ma_7']) - 1
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) - 1
    
    # Price Efficiency Metrics
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['abs_price_change'] = abs(data['close'] - data['close'].shift(1))
    data['efficiency_ratio'] = data['abs_price_change'] / data['true_range'].replace(0, np.nan)
    
    # Volume-Price Divergence Detection
    data['short_divergence'] = data['short_term_momentum'] * data['short_volume_trend']
    data['medium_divergence'] = data['medium_term_momentum'] * data['medium_volume_trend']
    
    # Divergence Persistence
    data['short_div_sign'] = np.sign(data['short_divergence'])
    data['medium_div_sign'] = np.sign(data['medium_divergence'])
    
    short_persistence = []
    medium_persistence = []
    for i in range(len(data)):
        if i < 5:
            short_persistence.append(np.nan)
            medium_persistence.append(np.nan)
        else:
            short_window = data['short_div_sign'].iloc[i-4:i+1]
            medium_window = data['medium_div_sign'].iloc[i-4:i+1]
            
            short_consistent = (short_window == short_window.iloc[-1]).sum()
            medium_consistent = (medium_window == medium_window.iloc[-1]).sum()
            
            short_persistence.append(short_consistent / 5)
            medium_persistence.append(medium_consistent / 5)
    
    data['short_persistence'] = short_persistence
    data['medium_persistence'] = medium_persistence
    data['persistence_score'] = (data['short_persistence'] + data['medium_persistence']) / 2
    
    # Range and Regime Analysis Component
    # Range Utilization Metrics
    data['intraday_range'] = (data['high'] - data['low']) / data['close']
    data['range_utilization'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['range_ma_3'] = data['intraday_range'].rolling(window=3).mean()
    data['range_expansion'] = data['intraday_range'] / data['range_ma_3'].shift(1).replace(0, np.nan)
    
    # Market Regime Classification
    data['volatility_environment'] = 'normal'
    data.loc[data['volatility_ratio'] > 1.2, 'volatility_environment'] = 'high'
    data.loc[data['volatility_ratio'] < 0.8, 'volatility_environment'] = 'low'
    
    data['volume_ratio_10'] = data['volume'] / data['volume_ma_10']
    data['volume_stability'] = 1 / (data['volume'] / data['volume_ma_5']).rolling(window=5).std()
    
    data['volume_environment'] = 'normal'
    data.loc[data['volume_ratio_10'] > 1.1, 'volume_environment'] = 'high'
    data.loc[data['volume_stability'] > 1.5, 'volume_environment'] = 'stable'
    
    # Signal Alignment and Confirmation
    # Momentum Direction Consistency
    data['intraday_vs_short'] = np.sign(data['intraday_momentum']) * np.sign(data['short_term_momentum'])
    data['short_vs_medium'] = np.sign(data['short_term_momentum']) * np.sign(data['medium_term_momentum'])
    data['direction_consistency'] = (data['intraday_vs_short'] + data['short_vs_medium']) / 2
    
    # Volume-Momentum Alignment
    data['short_alignment'] = data['short_volume_trend'] * abs(data['intraday_momentum'])
    data['medium_alignment'] = data['medium_volume_trend'] * abs(data['short_term_momentum'])
    data['alignment_score'] = (data['short_alignment'] + data['medium_alignment']) / 2
    
    # Divergence Confirmation
    data['short_strength'] = abs(data['short_divergence']) * data['persistence_score']
    data['medium_strength'] = abs(data['medium_divergence']) * data['volume_stability'].fillna(1)
    data['multi_timeframe_confirmation'] = data['short_strength'] - data['medium_strength']
    
    # Final Alpha Construction
    # Core Momentum Signal
    data['vol_adj_base'] = (data['vol_scaled_short'] + data['vol_scaled_medium']) / 2
    data['momentum_accel_factor'] = data['momentum_acceleration'] * data['efficiency_ratio']
    data['enhanced_momentum'] = data['vol_adj_base'] * (1 + data['momentum_accel_factor'])
    
    # Volume-Price Efficiency Multiplier
    data['divergence_component'] = (data['short_divergence'] + data['medium_divergence']) / 2
    data['persistence_adjustment'] = data['divergence_component'] * data['persistence_score']
    data['efficiency_multiplier'] = 1 + (data['persistence_adjustment'] * data['alignment_score'])
    
    # Range and Regime Adjustment
    data['range_utilization_factor'] = data['range_utilization'] * data['range_expansion']
    
    data['volatility_sensitivity'] = 1 / (1 + data['volatility_ratio'])
    data['volume_sensitivity'] = data['volume_ratio_10'] * data['volume_stability']
    data['regime_sensitivity'] = data['volatility_sensitivity'] * data['volume_sensitivity']
    
    data['regime_adaptive_range'] = data['range_utilization_factor'] * data['regime_sensitivity']
    
    # Alignment Confirmation Factor
    data['volume_momentum_alignment'] = data['alignment_score'] * data['volume_acceleration']
    data['confirmation_score'] = data['direction_consistency'] * data['volume_momentum_alignment']
    
    # Final Alpha Formula
    data['base_signal'] = data['enhanced_momentum'] * data['efficiency_multiplier']
    data['range_adjusted_signal'] = data['base_signal'] * data['regime_adaptive_range']
    data['final_alpha'] = data['range_adjusted_signal'] * data['confirmation_score']
    
    return data['final_alpha']
