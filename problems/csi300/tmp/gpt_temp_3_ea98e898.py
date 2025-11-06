import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price and volume metrics
    data['price_change'] = data['close'] / data['close'].shift(1) - 1
    data['abs_price_change'] = np.abs(data['price_change'])
    data['volume_change'] = data['volume'] / data['volume'].shift(1) - 1
    data['abs_volume_change'] = np.abs(data['volume_change'])
    
    # Multi-Timeframe Price Momentum
    data['price_acceleration'] = data['close'] - 2 * data['close'].shift(1) + data['close'].shift(2)
    data['price_momentum_persistence'] = np.sign(data['price_acceleration']) * np.sign(data['price_change'])
    
    # Volume Momentum Dynamics
    data['volume_persistence'] = np.sign(data['volume_change']) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Generate Momentum Divergence Signal
    data['momentum_divergence'] = (data['price_acceleration'] * data['volume_change'] * 
                                  np.sign(data['price_change']) * data['abs_price_change'] * 
                                  data['abs_volume_change'] * data['price_momentum_persistence'] * 
                                  data['volume_persistence'])
    
    # Volatility State Metrics
    data['true_range'] = data['high'] - data['low']
    data['median_range_5d'] = data['true_range'].rolling(window=5, min_periods=3).median()
    data['dispersion_ratio'] = data['true_range'] / data['median_range_5d']
    
    # Regime Transition Intensity
    data['volatility_ratio'] = data['true_range'] / data['true_range'].shift(5)
    data['vol_transition_intensity'] = np.log(np.maximum(data['volatility_ratio'], 1e-6))
    
    # Volatility Persistence Dynamics
    data['range_autocorr'] = data['true_range'].rolling(window=5, min_periods=3).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 3 else 1.0, raw=False
    )
    data['persistence_factor'] = np.maximum(data['range_autocorr'], 0.1)
    
    # Combine volatility components
    data['regime_transition'] = (data['vol_transition_intensity'] * data['persistence_factor'] * 
                                data['dispersion_ratio'])
    
    # Volume Profile Analysis
    data['volume_to_amount'] = data['volume'] / data['amount']
    data['median_volume_amount_10d'] = data['volume_to_amount'].rolling(window=10, min_periods=5).median()
    data['volume_profile_deviation'] = (data['volume_to_amount'] / 
                                       data['median_volume_amount_10d'] - 1)
    
    # Hierarchical Signal Integration
    data['integrated_signal'] = (data['momentum_divergence'] * data['regime_transition'] * 
                                data['persistence_factor'] * data['volume_profile_deviation'] * 
                                data['dispersion_ratio'] * data['price_momentum_persistence'])
    
    # Adaptive Non-linear Transformation
    data['transformed_signal'] = np.tanh(data['integrated_signal'])
    
    # Hierarchical Momentum Enhancement
    data['prev_divergence'] = data['momentum_divergence'].shift(1)
    
    # Calculate adaptive decay based on volatility
    volatility_decay = 1 / (1 + np.abs(data['vol_transition_intensity']))
    data['smoothed_momentum'] = (volatility_decay * data['prev_divergence'] + 
                                (1 - volatility_decay) * data['momentum_divergence'])
    
    # Final Signal Optimization
    data['final_alpha'] = (data['transformed_signal'] * data['smoothed_momentum'] * 
                          data['true_range'] * np.sign(data['price_momentum_persistence']) * 
                          data['regime_transition'] * np.abs(data['volume_profile_deviation']))
    
    # Clean up and return
    alpha_series = data['final_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series
