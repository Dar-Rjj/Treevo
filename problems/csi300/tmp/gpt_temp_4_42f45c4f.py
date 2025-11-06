import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Pressure Efficiency Analysis
    # Hierarchical Pressure Calculation
    data['intraday_buying_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['intraday_selling_pressure'] = (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    data['short_term_pressure'] = data['intraday_buying_pressure'].diff(5)
    data['long_term_pressure'] = data['intraday_buying_pressure'].diff(20)
    
    # Multi-Scale Efficiency Assessment
    data['daily_range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['short_term_efficiency'] = data['daily_range_efficiency'].rolling(window=3, min_periods=1).mean()
    data['long_term_efficiency'] = data['daily_range_efficiency'].rolling(window=8, min_periods=1).mean()
    
    # Pressure-Efficiency Divergence
    data['pressure_momentum'] = data['short_term_pressure'] - data['long_term_pressure']
    data['efficiency_momentum'] = (data['short_term_efficiency'] - data['long_term_efficiency']) / (abs(data['long_term_efficiency']) + 1e-8)
    
    # Calculate rolling correlation for pressure-efficiency alignment
    pressure_efficiency_corr = []
    for i in range(len(data)):
        if i >= 10:
            window_data = data.iloc[i-10:i]
            corr = window_data['pressure_momentum'].corr(window_data['efficiency_momentum'])
            pressure_efficiency_corr.append(corr if not np.isnan(corr) else 0)
        else:
            pressure_efficiency_corr.append(0)
    data['pressure_efficiency_alignment'] = pressure_efficiency_corr
    
    # Volume-Liquidity Asymmetry Detection
    # Volume Acceleration Patterns
    data['short_term_volume'] = data['volume'].diff(3) - data['volume'].diff(8)
    data['long_term_volume'] = data['volume'].diff(5) - data['volume'].diff(20)
    data['volume_acceleration'] = (data['short_term_volume'] - data['long_term_volume']) / (abs(data['long_term_volume']) + 1e-8)
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Liquidity Flow Analysis
    data['upward_liquidity_pressure'] = np.where(data['close'] > data['open'], data['amount'] / (data['volume'] + 1e-8), 0)
    data['downward_liquidity_pressure'] = np.where(data['close'] < data['open'], data['amount'] / (data['volume'] + 1e-8), 0)
    
    upward_pressure = data['upward_liquidity_pressure'].rolling(window=5, min_periods=1).mean()
    downward_pressure = data['downward_liquidity_pressure'].rolling(window=5, min_periods=1).mean()
    data['liquidity_pressure_asymmetry'] = upward_pressure - downward_pressure
    
    # Volume-Liquidity Synchronization
    volume_liquidity_corr = []
    for i in range(len(data)):
        if i >= 8:
            window_data = data.iloc[i-8:i]
            corr = window_data['volume_acceleration'].corr(window_data['liquidity_pressure_asymmetry'])
            volume_liquidity_corr.append(corr if not np.isnan(corr) else 0)
        else:
            volume_liquidity_corr.append(0)
    data['liquidity_velocity_alignment'] = volume_liquidity_corr
    
    # Range Dynamics & Volatility Context
    # Multi-Scale Range Expansion
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['short_term_range'] = data['daily_range'].rolling(window=5, min_periods=1).mean()
    data['long_term_range'] = data['daily_range'].rolling(window=20, min_periods=1).mean()
    data['range_expansion_ratio'] = data['daily_range'] / (data['long_term_range'] + 1e-8)
    
    # Volatility Regime Identification
    data['range_volatility'] = data['daily_range'].rolling(window=5, min_periods=1).std()
    data['efficiency_volatility'] = data['daily_range_efficiency'].rolling(window=5, min_periods=1).std()
    data['volatility_context_score'] = (data['range_volatility'] + data['efficiency_volatility']) / 2
    
    # Range-Pressure Interaction
    data['range_utilization_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    range_pressure_corr = []
    for i in range(len(data)):
        if i >= 8:
            window_data = data.iloc[i-8:i]
            corr = window_data['intraday_buying_pressure'].corr(window_data['range_utilization_efficiency'])
            range_pressure_corr.append(corr if not np.isnan(corr) else 0)
        else:
            range_pressure_corr.append(0)
    data['pressure_range_alignment'] = range_pressure_corr
    
    # Phase Synchronization & Momentum Structure
    # Multi-Timeframe Momentum Divergence
    data['short_term_momentum'] = (data['close'] - data['open']) / (data['open'] + 1e-8)
    data['medium_term_momentum'] = data['short_term_momentum'].rolling(window=13, min_periods=1).mean()
    data['long_term_momentum'] = data['short_term_momentum'].rolling(window=21, min_periods=1).mean()
    
    # Calculate momentum divergence as variance across timeframes
    momentum_data = data[['short_term_momentum', 'medium_term_momentum', 'long_term_momentum']].fillna(0)
    data['momentum_divergence_score'] = momentum_data.var(axis=1)
    
    # Phase Alignment Detection
    data['price_phase'] = np.sign(data['short_term_momentum'])
    data['volume_phase'] = np.sign(data['volume_acceleration'])
    data['pressure_phase'] = np.sign(data['pressure_momentum'])
    
    # Phase synchronization score (count of aligned phases)
    phase_alignment = (data['price_phase'] == data['volume_phase']).astype(int) + \
                     (data['price_phase'] == data['pressure_phase']).astype(int) + \
                     (data['volume_phase'] == data['pressure_phase']).astype(int)
    data['phase_synchronization_score'] = phase_alignment / 3.0
    
    # Adaptive Multi-Dimensional Integration
    # Core Divergence Generation
    data['pressure_efficiency_divergence'] = data['pressure_momentum'] * data['efficiency_momentum']
    data['volume_weighted_divergence'] = data['pressure_efficiency_divergence'] * data['volume_acceleration']
    data['liquidity_scaled_divergence'] = data['volume_weighted_divergence'] * data['liquidity_pressure_asymmetry']
    
    # Range & Volatility Adjustment
    data['volatility_scaled_signal'] = data['liquidity_scaled_divergence'] / (data['range_volatility'] + 1e-8)
    data['range_regime_filtered'] = data['volatility_scaled_signal'] * data['range_expansion_ratio']
    data['efficiency_volatility_context'] = data['range_regime_filtered'] / (data['efficiency_volatility'] + 1e-8)
    
    # Phase Synchronization Validation
    data['phase_aligned_signal'] = data['efficiency_volatility_context'] * data['phase_synchronization_score']
    
    # Multi-Timeframe Signal Integration
    data['short_term_signal'] = data['phase_aligned_signal'].rolling(window=5, min_periods=1).mean()
    data['medium_term_signal'] = data['phase_aligned_signal'].rolling(window=13, min_periods=1).mean()
    data['long_term_signal'] = data['phase_aligned_signal'].rolling(window=21, min_periods=1).mean()
    
    # Timeframe consistency scoring (weighted average)
    weights = [0.4, 0.35, 0.25]  # Short, medium, long term weights
    timeframe_signals = data[['short_term_signal', 'medium_term_signal', 'long_term_signal']].fillna(0)
    data['timeframe_consistency_score'] = (timeframe_signals * weights).sum(axis=1)
    
    # Dynamic Factor Calibration
    # Volume concentration adjustment
    volume_quantile = data['volume_efficiency'].rolling(window=20, min_periods=1).quantile(0.8)
    volume_concentration = data['volume_efficiency'] / (volume_quantile + 1e-8)
    
    # Efficiency persistence optimization
    efficiency_gradient = data['short_term_efficiency'] - data['long_term_efficiency']
    efficiency_strength = abs(efficiency_gradient) / (abs(data['long_term_efficiency']) + 1e-8)
    
    # Final calibrated factor
    final_factor = (data['timeframe_consistency_score'] * 
                   (1 + volume_concentration * 0.1) * 
                   (1 + efficiency_strength * 0.05) * 
                   (1 + data['liquidity_velocity_alignment'] * 0.15))
    
    return final_factor
