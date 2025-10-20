import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Price Acceleration Dynamics
    data['short_term_momentum'] = data['close'] / data['close'].shift(3) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_convergence'] = (data['short_term_momentum'] - data['medium_term_momentum']) / (abs(data['medium_term_momentum']) + 1e-8)
    data['price_acceleration'] = (data['short_term_momentum'] - data['medium_term_momentum']) / 7
    
    # Volume Acceleration Framework
    data['short_term_volume_momentum'] = data['volume'] / data['volume'].shift(3) - 1
    data['medium_term_volume_momentum'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_acceleration'] = data['short_term_volume_momentum'] / (data['medium_term_volume_momentum'] + 1e-8)
    data['volume_breakout'] = (data['volume'] > 1.5 * data['volume'].shift(5)).astype(int)
    
    # Acceleration Divergence Core
    data['bullish_divergence'] = -data['price_acceleration'] * data['volume_acceleration']
    data['bearish_divergence'] = data['price_acceleration'] * data['volume_acceleration']
    data['divergence_magnitude'] = abs(data['price_acceleration']) * abs(data['volume_acceleration'])
    
    # Intraday Pressure Dynamics
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['directional_efficiency'] = (data['close'] - data['open']) / (data['true_range'] + 1e-8)
    data['3_day_cumulative_pressure'] = data['directional_efficiency'].rolling(window=3, min_periods=1).sum()
    
    # Volume-Pressure Alignment
    data['volume_price_correlation'] = data['volume'].rolling(window=5).corr(data['close'])
    data['volume_persistence'] = ((data['volume'] > data['volume'].shift(1)) & 
                                 (data['volume'].shift(1) > data['volume'].shift(2))).astype(int)
    data['pressure_volume_convergence'] = data['3_day_cumulative_pressure'] * data['volume_acceleration']
    
    # Reversal Pattern Detection
    # Recent high-low reversals
    high_low_reversals = []
    for i in range(len(data)):
        if i < 5:
            high_low_reversals.append(0)
            continue
        count = 0
        for j in range(1, 6):
            if (data['high'].iloc[i-j+1] > data['high'].iloc[i-j]) and (data['close'].iloc[i-j+1] < data['close'].iloc[i-j]):
                count += 1
        high_low_reversals.append(count)
    data['recent_high_low_reversals'] = high_low_reversals
    
    # Gap recovery strength
    data['gap_recovery_strength'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    
    # Failed breakouts
    failed_breakouts = []
    for i in range(len(data)):
        if i < 3:
            failed_breakouts.append(0)
            continue
        count = 0
        for j in range(1, 4):
            if (data['high'].iloc[i-j+1] > data['high'].iloc[i-j]) and (data['close'].iloc[i-j+1] < data['open'].iloc[i-j+1]):
                count += 1
        failed_breakouts.append(count)
    data['failed_breakouts'] = failed_breakouts
    
    # Volume Reversal Confirmation
    data['avg_volume_10'] = data['volume'].rolling(window=10, min_periods=1).mean().shift(1)
    data['abnormal_volume_spikes'] = data['volume'] / (data['avg_volume_10'] + 1e-8)
    data['volume_price_divergence'] = (data['volume'] / data['volume'].shift(1)) - (data['close'] / data['close'].shift(1))
    data['avg_volume_5'] = data['volume'].rolling(window=5, min_periods=1).mean().shift(1)
    data['volume_reversal_confirmation'] = (data['volume'] > 1.2 * data['avg_volume_5']).astype(int)
    
    # Reversal-Enhanced Pressure
    data['reversal_weighted_pressure'] = data['3_day_cumulative_pressure'] * data['recent_high_low_reversals']
    data['gap_recovery_pressure'] = data['3_day_cumulative_pressure'] * data['gap_recovery_strength']
    data['volume_spike_pressure'] = data['pressure_volume_convergence'] * data['abnormal_volume_spikes']
    
    # Multi-Dimensional Regime Classification
    # Volatility-Compression Framework
    data['volatility_ratio'] = data['close'].rolling(window=5).std() / (data['close'].rolling(window=20).std() + 1e-8)
    data['avg_true_range_5'] = data['true_range'].rolling(window=5, min_periods=1).mean()
    data['compression_ratio'] = data['true_range'] / (data['avg_true_range_5'] + 1e-8)
    data['high_volatility'] = (data['volatility_ratio'] > 1).astype(int)
    data['high_compression'] = (data['compression_ratio'] < 0.8).astype(int)
    
    # Trend-Strength Assessment
    data['trend_strength'] = abs(data['close'] / data['close'].shift(10) - 1)
    data['strong_trend'] = (data['trend_strength'] > 0.05).astype(int)
    
    # Volume-Regime Detection
    data['avg_volume_5_lag'] = data['volume'].rolling(window=5, min_periods=1).mean().shift(1)
    data['volume_surge'] = (data['volume'] > 1.5 * data['avg_volume_5_lag']).astype(int)
    data['volume_spike'] = (data['volume'] > 2.0 * data['volume'].shift(1)).astype(int)
    
    # Acceleration-Pressure Convergence with Reversal Enhancement
    # Core Convergence Framework
    data['price_volume_acceleration'] = data['momentum_convergence'] * data['volume_acceleration']
    data['pressure_efficiency_alignment'] = data['pressure_volume_convergence'] * data['volume_price_correlation']
    data['multi_dimensional_convergence'] = data['price_volume_acceleration'] * data['pressure_efficiency_alignment']
    
    # Reversal-Enhanced Convergence
    data['reversal_weighted_convergence'] = data['multi_dimensional_convergence'] * data['recent_high_low_reversals']
    data['gap_recovery_convergence'] = data['multi_dimensional_convergence'] * data['gap_recovery_strength']
    data['volume_spike_convergence'] = data['multi_dimensional_convergence'] * data['abnormal_volume_spikes']
    
    # Dynamic Regime Modulation
    data['volatility_adjustment'] = data['multi_dimensional_convergence'] * data['volatility_ratio']
    data['compression_sensitivity'] = data['multi_dimensional_convergence'] * data['compression_ratio']
    data['trend_amplification'] = data['multi_dimensional_convergence'] * data['trend_strength']
    
    # Regime-Adaptive Alpha Synthesis
    alpha_values = []
    
    for i in range(len(data)):
        if i < 20:  # Ensure enough data for calculations
            alpha_values.append(0)
            continue
            
        row = data.iloc[i]
        
        # High Volatility + Low Compression Regime
        if row['high_volatility'] == 1 and row['high_compression'] == 0:
            alpha = (row['price_acceleration'] * row['reversal_weighted_pressure'] * 
                    row['volume_persistence'] * row['volatility_ratio'])
        
        # Low Volatility + High Compression Regime
        elif row['high_volatility'] == 0 and row['high_compression'] == 1:
            alpha = (row['momentum_convergence'] * row['pressure_efficiency_alignment'] * 
                    row['gap_recovery_strength'] * row['compression_ratio'])
        
        # Strong Trend + Volume Surge Regime
        elif row['strong_trend'] == 1 and row['volume_surge'] == 1:
            alpha = (row['price_volume_acceleration'] * row['3_day_cumulative_pressure'] * 
                    row['trend_strength'] * (1 - row['failed_breakouts']/3))
        
        # Reversal-Dominant Regime (high reversal count)
        elif row['recent_high_low_reversals'] >= 3:
            alpha = (row['reversal_weighted_convergence'] * row['volume_spike_convergence'] * 
                    row['directional_efficiency'])
        
        # Normal Regime
        else:
            alpha = (row['multi_dimensional_convergence'] * row['directional_efficiency'] * 
                    row['volume_acceleration'] * row['recent_high_low_reversals'])
        
        alpha_values.append(alpha)
    
    data['alpha'] = alpha_values
    
    # Handle NaN values
    data['alpha'] = data['alpha'].fillna(0)
    
    return data['alpha']
