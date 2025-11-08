import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_open'] = data['open'].shift(1)
    
    # 1. Gap-Based Momentum Signal
    # Opening Gap Magnitude
    data['abs_gap'] = np.abs(data['open'] / data['prev_close'] - 1)
    data['directional_gap'] = data['open'] / data['prev_close'] - 1
    
    # Gap Filling Behavior
    data['intraday_recovery'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['gap_fill_ratio'] = (data['close'] - data['open']) / (data['prev_close'] - data['open'] + 1e-8)
    
    # Gap Momentum Factor
    data['gap_momentum'] = data['directional_gap'] * (1 - np.abs(data['gap_fill_ratio']))
    
    # 2. Volume-Price Efficiency Metric
    # Price Movement Efficiency
    data['effective_price_move'] = np.abs(data['close'] - data['open'])
    data['total_price_range'] = data['high'] - data['low']
    data['efficiency_score'] = data['effective_price_move'] / (data['total_price_range'] + 1e-8)
    
    # Volume Efficiency Ratio
    data['volume_per_unit_move'] = data['volume'] / (data['effective_price_move'] + 1)
    
    # Normalize volume efficiency using rolling percentiles
    data['volume_efficiency_rank'] = data['volume_per_unit_move'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.3)) / (x.quantile(0.7) - x.quantile(0.3) + 1e-8)
    )
    
    # 3. Overnight Information Component
    # Overnight Return Momentum
    data['overnight_return'] = data['open'] / data['prev_close'] - 1
    data['prev_overnight_return'] = data['overnight_return'].shift(1)
    data['overnight_persistence'] = data['overnight_return'] / (data['prev_overnight_return'] + 1e-8)
    
    # Overnight-Intraday Continuity
    data['continuity_ratio'] = (data['close'] - data['open']) / (np.abs(data['open'] - data['prev_close']) + 1e-8)
    data['direction_alignment'] = np.sign(data['close'] - data['open']) * np.sign(data['open'] - data['prev_close'])
    
    # Overnight Information Weight
    data['overnight_weight'] = np.abs(data['overnight_persistence']) * data['direction_alignment']
    
    # 4. Dynamic Volatility Regime Adjustment
    # Regime-Based Volatility
    data['short_term_vol'] = np.abs(data['close'] - data['prev_close']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)
    data['medium_term_vol'] = data['short_term_vol'].rolling(window=10, min_periods=5).mean()
    data['volatility_regime'] = data['short_term_vol'] / (data['medium_term_vol'] + 1e-8)
    
    # Volatility Breakouts
    data['avg_range'] = (data['high'] - data['low']).rolling(window=20, min_periods=10).mean()
    data['volatility_spike'] = (data['high'] - data['low']) / (data['avg_range'] + 1e-8)
    data['regime_change'] = np.abs(data['volatility_regime'] - 1)
    
    # 5. Generate Composite Alpha Factor
    # Combine all components
    data['base_momentum'] = data['gap_momentum'] * data['efficiency_score']
    data['enhanced_momentum'] = data['base_momentum'] * (1 + data['overnight_weight'])
    data['final_factor'] = data['enhanced_momentum'] * (1 + data['regime_change'])
    
    # Apply momentum logic with gap persistence emphasis
    data['alpha_factor'] = data['final_factor'] * (1 - np.abs(data['gap_fill_ratio'])) * (1 + data['volume_efficiency_rank'])
    
    # Return the final factor series
    return data['alpha_factor']
