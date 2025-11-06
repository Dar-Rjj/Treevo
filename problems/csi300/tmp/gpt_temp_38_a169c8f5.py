import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Momentum Acceleration Framework
    # Momentum Efficiency Components
    data['short_term_momentum'] = data['close'] / data['close'].shift(3) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_divergence'] = data['short_term_momentum'] - data['medium_term_momentum']
    
    # Acceleration Analysis
    data['intraday_acceleration'] = (data['high'] - data['close']) / data['close'] - (data['low'] - data['close']) / data['close']
    data['short_term_acceleration'] = (data['high'] - data['close'].shift(3)) / data['close'].shift(3) - (data['low'] - data['close'].shift(3)) / data['close'].shift(3)
    data['acceleration_divergence'] = data['intraday_acceleration'] - data['short_term_acceleration']
    
    # Efficiency Integration
    data['daily_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['range_5d_avg'] = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean()
    data['range_compression'] = (data['high'] - data['low']) / data['range_5d_avg'].replace(0, np.nan)
    data['efficiency_weighted_momentum'] = data['momentum_divergence'] * (1 - data['range_compression'])
    
    # Multi-Scale Volume-Price Anchoring System
    # Volume Momentum Analysis
    data['short_term_volume_momentum'] = data['volume'] / data['volume'].shift(3) - 1
    data['medium_term_volume_momentum'] = data['volume'] / data['volume'].shift(8) - 1
    data['volume_acceleration'] = data['short_term_volume_momentum'] - data['medium_term_volume_momentum']
    
    # Price Efficiency Analysis
    data['volume_efficiency'] = data['amount'] / data['volume'].replace(0, np.nan)
    data['price_acceleration'] = (data['close'] - 2 * data['close'].shift(1) + data['close'].shift(2)) / (data['high'] - data['low']).replace(0, np.nan)
    data['volume_price_convergence'] = data['price_acceleration'] * data['volume_efficiency']
    
    # Liquidity Anchoring Detection
    data['volume_amount_alignment'] = data['volume_acceleration'] * data['volume_efficiency']
    data['price_volume_confirmation'] = data['price_acceleration'] * data['short_term_volume_momentum']
    data['anchoring_strength'] = data['volume_amount_alignment'] + data['price_volume_confirmation']
    
    # Gap-Pressure Acceleration Integration
    # Multi-Scale Gap Analysis
    data['opening_gap'] = data['open'] / data['close'].shift(1) - 1
    data['opening_strength'] = (data['open'] - data['low'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    data['gap_fill_efficiency'] = ((data['close'] - data['open']) / data['open'].replace(0, np.nan)) / data['opening_gap'].replace(0, np.nan)
    
    # Acceleration-Gap Alignment
    data['gap_acceleration_correlation'] = data['acceleration_divergence'] * data['gap_fill_efficiency']
    
    # Microstructure Efficiency Anchoring
    data['intraday_recovery'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['recovery_efficiency'] = data['intraday_recovery'] * data['volume_efficiency']
    data['anchored_efficiency'] = data['opening_strength'] * data['intraday_recovery'] * data['volume_efficiency']
    
    # Volatility-Regime Adaptive Framework
    # Multi-scale Volatility Structure
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_6d'] = data['true_range'].rolling(window=6, min_periods=1).mean()
    data['volatility_regime'] = data['true_range'] / data['atr_6d'].replace(0, np.nan)
    
    # Range-Based Volatility Analysis
    data['daily_range_volatility'] = (data['high'] - data['low']) / data['close'].replace(0, np.nan)
    data['volatility_momentum'] = data['true_range'] / data['true_range'].shift(5).replace(0, np.nan) - 1
    
    # Regime-Adaptive Filtering
    data['volatility_adjusted_signal'] = data['efficiency_weighted_momentum'] * (1 + data['volatility_regime'])
    
    # Fractal Convergence Analysis
    # Momentum-Acceleration Alignment
    data['momentum_acceleration_convergence'] = data['efficiency_weighted_momentum'] * data['acceleration_divergence']
    
    # Anchoring-Efficiency Integration
    data['anchoring_efficiency_integration'] = data['anchored_efficiency'] * data['volume_price_convergence']
    
    # Gap-Pressure Confirmation
    data['gap_pressure_confirmation'] = data['gap_acceleration_correlation'] * data['anchoring_strength']
    
    # Composite Fractal Momentum Alpha
    # Core Signal Construction
    data['acceleration_momentum_core'] = data['efficiency_weighted_momentum'] * data['acceleration_divergence']
    data['volume_price_enhancement'] = data['acceleration_momentum_core'] * data['volume_price_convergence']
    data['gap_pressure_integration'] = data['volume_price_enhancement'] * data['gap_fill_efficiency']
    
    # Anchoring Quality Layer
    data['anchored_core_signal'] = data['gap_pressure_integration'] * data['anchored_efficiency']
    
    # Regime-Adaptive Integration
    high_vol_regime = data['volatility_regime'] > 1.0
    low_vol_regime = data['volatility_regime'] <= 1.0
    
    data['regime_adaptive_signal'] = np.where(
        high_vol_regime,
        data['anchored_core_signal'] * (1 + data['acceleration_divergence']),
        data['anchored_core_signal'] * (1 + data['efficiency_weighted_momentum'])
    )
    
    # Volume Flow Confirmation
    def calculate_up_volume(data, window):
        up_volume_sum = pd.Series(index=data.index, dtype=float)
        for i in range(len(data)):
            if i >= window - 1:
                window_data = data.iloc[i-window+1:i+1]
                up_volume = window_data['volume'][window_data['close'] > window_data['close'].shift(1).fillna(method='bfill')]
                up_volume_sum.iloc[i] = up_volume.sum()
            else:
                up_volume_sum.iloc[i] = np.nan
        return up_volume_sum
    
    data['up_volume_3d'] = calculate_up_volume(data, 3)
    data['up_volume_8d'] = calculate_up_volume(data, 8)
    data['volume_flow_ratio'] = data['up_volume_3d'] / data['up_volume_8d'].replace(0, np.nan)
    data['volume_flow_multiplier'] = 1 + (data['volume_flow_ratio'] - 1) * np.sign(data['acceleration_divergence'])
    
    # Final Alpha Synthesis
    data['fractal_momentum_alpha'] = data['regime_adaptive_signal'] * data['volume_flow_multiplier']
    
    # Clean up and return
    alpha_series = data['fractal_momentum_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series
