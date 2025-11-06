import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Multi-Scale Volatility Asymmetry
    # Short-Term Volatility Asymmetry (3-day)
    data['short_up_asym'] = ((data['high'] - data['close']) * data['volume']) / np.maximum(data['close'] - data['low'], 1e-6)
    data['short_down_asym'] = ((data['close'] - data['low']) * data['volume']) / np.maximum(data['high'] - data['close'], 1e-6)
    
    # Medium-Term Volatility Asymmetry (8-day)
    data['med_up_asym'] = ((data['high'] - data['close'].shift(3)) * data['volume']) / np.maximum(data['close'].shift(3) - data['low'], 1e-6)
    data['med_down_asym'] = ((data['close'] - data['low'].shift(3)) * data['volume']) / np.maximum(data['high'] - data['close'].shift(3), 1e-6)
    
    # Volatility Asymmetry Detection
    data['up_pressure_dom'] = (data['short_up_asym'] > data['med_up_asym']).astype(int)
    data['down_pressure_dom'] = (data['short_down_asym'] > data['med_down_asym']).astype(int)
    data['asym_convergence'] = data['up_pressure_dom'] + data['down_pressure_dom']
    
    # Calculate Price Pressure Gradient
    data['intraday_pressure'] = (data['close'] - data['open']) * data['amount']
    data['overnight_pressure'] = (data['open'] - data['close'].shift(1)) * data['amount']
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_concentration'] = data['volume'] / np.maximum(data['volume_5d_avg'], 1e-6)
    data['pressure_gradient'] = (data['intraday_pressure'] - data['overnight_pressure']) * data['volume_concentration']
    
    # Compute Volume-Pressure Confirmation
    data['bullish_pressure'] = ((data['high'] - data['open']) * data['amount']) / np.maximum(data['volume'], 1e-6)
    data['bearish_pressure'] = ((data['open'] - data['low']) * data['amount']) / np.maximum(data['volume'], 1e-6)
    data['net_volume_pressure'] = data['bullish_pressure'] - data['bearish_pressure']
    
    # Volume-Regime Synthesis
    data['volume_concentration_ratio'] = data['volume'] / np.maximum(data['amount'], 1e-6)
    data['volume_concentration_10d_median'] = data['volume_concentration_ratio'].rolling(window=10, min_periods=1).median()
    data['volume_5d_ma'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_12d_ma'] = data['volume'].rolling(window=12, min_periods=1).mean()
    data['volume_momentum'] = data['volume_5d_ma'] / np.maximum(data['volume_12d_ma'], 1e-6)
    
    # Volume-Asymmetry Alignment
    data['vol_asym_alignment'] = data['asym_convergence'] * data['volume_momentum']
    
    # Compute Market Efficiency Filter
    data['volatility_efficiency'] = np.abs(data['close'] - data['open']) / np.maximum(data['high'] - data['low'], 1e-6)
    data['pressure_density'] = data['amount'] / np.maximum(data['high'] - data['low'], 1e-6)
    
    # Integrate Asymmetry with Pressure Synthesis
    data['asym_pressure'] = data['asym_convergence'] * data['net_volume_pressure']
    data['efficiency_filtered'] = data['asym_pressure'] * data['volatility_efficiency']
    data['density_scaled'] = data['efficiency_filtered'] * data['pressure_density']
    data['volume_weighted'] = data['density_scaled'] * data['volume_momentum']
    
    # Compute Signal Persistence Adjustment
    data['signal_direction'] = np.sign(data['volume_weighted'])
    data['signal_consistency'] = data['signal_direction'].rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(x == x.iloc[-1]) / len(x) if len(x) > 0 else 1.0
    )
    data['persistence_adjusted'] = data['volume_weighted'] * data['signal_consistency']
    
    # Apply Volatility Context
    data['daily_range'] = data['high'] - data['low']
    data['volatility_stability'] = data['daily_range'].rolling(window=10, min_periods=1).std()
    data['volatility_context'] = data['persistence_adjusted'] * data['volatility_stability']
    
    # Volatility Expansion Enhancement
    data['current_volatility'] = data['daily_range']
    data['historical_volatility'] = data['daily_range'].rolling(window=15, min_periods=1).mean()
    data['vol_expansion_ratio'] = data['current_volatility'] / np.maximum(data['historical_volatility'], 1e-6)
    data['vol_expansion_multiplier'] = np.where(data['vol_expansion_ratio'] > 1, data['vol_expansion_ratio'], 1.0)
    data['expansion_enhanced'] = data['volatility_context'] * data['vol_expansion_multiplier']
    
    # Generate Final Alpha Factor with Cube Root Transformation
    data['final_signal'] = np.sign(data['expansion_enhanced']) * np.abs(data['expansion_enhanced']) ** (1/3)
    
    # Return the factor series
    return data['final_signal']
