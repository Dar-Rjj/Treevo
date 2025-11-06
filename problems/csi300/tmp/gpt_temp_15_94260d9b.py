import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Small epsilon to avoid division by zero
    eps = 1e-8
    
    # Multi-Timeframe Range Efficiency
    # Micro Range Efficiency
    data['micro_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + eps)
    
    # Meso Range Efficiency
    data['meso_high_max'] = data['high'].rolling(window=4, min_periods=4).max()
    data['meso_low_min'] = data['low'].rolling(window=4, min_periods=4).min()
    data['meso_efficiency'] = (data['close'] - data['close'].shift(3)) / (data['meso_high_max'] - data['meso_low_min'] + eps)
    
    # Macro Range Efficiency
    data['macro_high_max'] = data['high'].rolling(window=9, min_periods=9).max()
    data['macro_low_min'] = data['low'].rolling(window=9, min_periods=9).min()
    data['macro_efficiency'] = (data['close'] - data['close'].shift(8)) / (data['macro_high_max'] - data['macro_low_min'] + eps)
    
    # Fractal Efficiency Cascade
    data['fractal_efficiency_cascade'] = data['micro_efficiency'] * data['meso_efficiency'] * data['macro_efficiency']
    
    # Volume-Price Divergence Dynamics
    # Directional Divergence
    data['directional_divergence'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Magnitude Divergence
    data['magnitude_divergence'] = np.abs(data['close'] - data['close'].shift(1)) / (np.abs(data['volume'] - data['volume'].shift(1)) + eps)
    
    # Volume Fractal Cascade
    data['volume_fractal_cascade'] = (data['volume'] / (data['volume'].shift(1) + eps)) * \
                                    (data['volume'] / (data['volume'].shift(5) + eps)) * \
                                    (data['volume'] / (data['volume'].shift(13) + eps))
    
    # Volatility Regime Classification
    # True Range
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Average True Range
    data['atr'] = data['true_range'].rolling(window=5, min_periods=5).mean()
    
    # Regime Classification
    data['high_vol_regime'] = (data['true_range'] > data['atr']) & (data['volume'] > data['volume'].shift(1))
    data['low_vol_regime'] = (data['true_range'] < data['atr']) & (data['volume'] < data['volume'].shift(1))
    data['transition_regime'] = ~data['high_vol_regime'] & ~data['low_vol_regime']
    
    # Multi-Timeframe Divergence Momentum
    data['efficiency_divergence_momentum'] = data['fractal_efficiency_cascade'] - data['directional_divergence']
    data['volume_price_momentum'] = data['volume_fractal_cascade'] - data['magnitude_divergence']
    data['range_momentum_divergence'] = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + eps)) - 1
    
    # Fractal Pressure Dynamics
    # Intraday Pressure Components
    data['opening_momentum'] = (data['high'] - data['open']) / (data['open'] - data['low'] + eps)
    data['closing_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'] + eps)
    data['intraday_pressure_cascade'] = data['opening_momentum'] * data['closing_momentum'] * np.sign(data['opening_momentum'] - data['closing_momentum'])
    
    # Divergence Transmission Framework
    data['price_volume_divergence_efficiency'] = data['fractal_efficiency_cascade'] * data['directional_divergence']
    data['efficiency_divergence_momentum_2'] = data['fractal_efficiency_cascade'] - (data['micro_efficiency'] * data['meso_efficiency'])
    data['volume_divergence_momentum'] = data['volume_fractal_cascade'] - ((data['volume'] / (data['volume'].shift(1) + eps)) * (data['volume'] / (data['volume'].shift(5) + eps)))
    
    # Adaptive Signal Enhancement
    # Volatility-Weighted Adjustment
    data['recent_volatility_scaling'] = (data['high'] - data['low']).rolling(window=3, min_periods=3).mean() / (data['atr'] + eps)
    
    # Volatility Persistence
    data['volatility_persistence'] = (data['true_range'] > data['atr']).rolling(window=3, min_periods=3).sum() / 3
    
    # Volatility Weight
    data['volatility_weight'] = data['true_range'] / (data['true_range'] + data['atr'] + eps)
    
    # Divergence Convergence Detection
    # Multi-timeframe Alignment
    sign_alignment = (np.sign(data['micro_efficiency']) == np.sign(data['meso_efficiency'])).rolling(window=3, min_periods=3).sum()
    data['multi_timeframe_alignment'] = sign_alignment / 3
    
    # Volume-Price Correlation
    data['price_change'] = data['close'].diff()
    data['volume_change'] = data['volume'].diff()
    data['volume_price_correlation'] = data['price_change'].rolling(window=5, min_periods=5).corr(data['volume_change'])
    
    # Regime Persistence
    regime_persistence = []
    for i in range(len(data)):
        if i < 5:
            regime_persistence.append(0)
        else:
            current_regime = None
            if data['high_vol_regime'].iloc[i]:
                current_regime = 'high'
            elif data['low_vol_regime'].iloc[i]:
                current_regime = 'low'
            else:
                current_regime = 'transition'
            
            count = 0
            for j in range(1, 6):
                prev_regime = None
                if data['high_vol_regime'].iloc[i-j]:
                    prev_regime = 'high'
                elif data['low_vol_regime'].iloc[i-j]:
                    prev_regime = 'low'
                else:
                    prev_regime = 'transition'
                
                if prev_regime == current_regime:
                    count += 1
            
            regime_persistence.append(count / 5)
    
    data['regime_persistence'] = regime_persistence
    
    # Liquidity-Divergence Integration
    # Range Absorption Framework
    data['liquidity_absorption'] = ((data['high'] - data['low']) * data['volume']) / \
                                  (((data['high'] - data['low']) * data['volume']).rolling(window=6, min_periods=6).mean() + eps)
    
    # Trade Impact
    data['trade_impact'] = data['amount'] / (data['volume'] * (data['high'] - data['low']) + eps)
    
    # Resistance-Level Context
    data['medium_term_resistance'] = data['high'].rolling(window=5, min_periods=5).mean()
    data['range_vs_resistance'] = (data['high'] - data['medium_term_resistance']) / (data['high'] - data['low'] + eps)
    
    # Composite Alpha Synthesis
    # Regime-Adaptive Core Signals
    data['high_vol_core'] = data['fractal_efficiency_cascade'] * data['intraday_pressure_cascade'] * data['directional_divergence']
    data['low_vol_core'] = data['price_volume_divergence_efficiency'] * data['volume_divergence_momentum'] * (1 / (data['recent_volatility_scaling'] + eps))
    data['transition_core'] = data['efficiency_divergence_momentum'] * data['volume_fractal_cascade'] * data['multi_timeframe_alignment']
    
    # Volatility-Scaled Core
    data['volatility_scaled_core'] = np.where(
        data['high_vol_regime'], 
        data['high_vol_core'],
        np.where(
            data['low_vol_regime'],
            data['low_vol_core'],
            data['transition_core']
        )
    )
    
    # Divergence Confirmation
    data['divergence_confirmation'] = data['volatility_scaled_core'] * (data['efficiency_divergence_momentum'] + data['volume_price_momentum'])
    
    # Momentum Enhancement
    data['momentum_enhancement'] = data['divergence_confirmation'] * data['volume_price_correlation']
    
    # Adaptive Weighting Framework
    data['volatility_adjusted'] = data['momentum_enhancement'] / (data['true_range'] + eps)
    data['persistence_adjusted'] = data['volatility_adjusted'] * (data['volatility_persistence'] * data['regime_persistence'])
    data['liquidity_adjusted'] = data['persistence_adjusted'] * data['liquidity_absorption'] * data['range_vs_resistance']
    
    # Final Alpha Construction
    # Base Alpha
    data['base_alpha'] = data['liquidity_adjusted'] * np.sign(data['intraday_pressure_cascade'])
    
    # Enhanced Alpha
    sign_alignment_count = []
    for i in range(len(data)):
        if i < 3:
            sign_alignment_count.append(0)
        else:
            count = 0
            for j in range(1, 4):
                if np.sign(data['base_alpha'].iloc[i-j]) == np.sign(data['efficiency_divergence_momentum'].iloc[i-j]):
                    count += 1
            sign_alignment_count.append(count / 3)
    
    data['sign_alignment_ratio'] = sign_alignment_count
    data['enhanced_alpha'] = data['base_alpha'] * data['sign_alignment_ratio']
    
    return data['enhanced_alpha']
