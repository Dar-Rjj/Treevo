import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Asymmetry Detection
    data['upside_move'] = np.maximum(0, data['high'] - data['close'])
    data['downside_move'] = np.maximum(0, data['close'] - data['low'])
    
    data['avg_upside_vol'] = data['upside_move'].rolling(window=10, min_periods=5).mean()
    data['avg_downside_vol'] = data['downside_move'].rolling(window=10, min_periods=5).mean()
    
    # Avoid division by zero
    data['vol_asymmetry_ratio'] = data['avg_upside_vol'] / (data['avg_downside_vol'] + 1e-8)
    
    # Momentum Quality Assessment
    data['returns'] = data['close'].pct_change()
    
    # Directional Persistence
    data['sign_returns'] = np.sign(data['returns'])
    data['sign_change'] = data['sign_returns'] != data['sign_returns'].shift(1)
    
    # Calculate consecutive same-sign returns
    data['consecutive_count'] = 0
    for i in range(1, len(data)):
        if data['sign_change'].iloc[i]:
            data['consecutive_count'].iloc[i] = 1
        else:
            data['consecutive_count'].iloc[i] = data['consecutive_count'].iloc[i-1] + 1
    
    # Return-to-Volatility Ratio
    data['total_return_5d'] = data['close'] / data['close'].shift(5) - 1
    data['volatility_5d'] = data['returns'].rolling(window=5, min_periods=3).std()
    data['return_to_vol_ratio'] = data['total_return_5d'] / (data['volatility_5d'] + 1e-8)
    
    # Momentum Quality Signal
    data['momentum_quality'] = data['consecutive_count'] * data['return_to_vol_ratio']
    
    # Price-Volume Efficiency Divergence
    data['intraday_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    data['daily_range'] = data['high'] - data['low']
    data['avg_range_5d'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['range_deviation'] = data['daily_range'] - data['avg_range_5d']
    
    # Efficiency Divergence
    data['efficiency_divergence'] = data['intraday_efficiency'] * data['range_deviation']
    
    # Microstructure Pressure Assessment
    data['price_rejection'] = (data['high'] - data['close']) - (data['close'] - data['low'])
    data['volume_concentration'] = data['volume'] / (data['daily_range'] + 1e-8)
    data['microstructure_pressure'] = data['price_rejection'] * data['volume_concentration']
    
    # Composite Alpha Factor
    # Base Momentum Efficiency
    data['base_momentum_efficiency'] = data['vol_asymmetry_ratio'] * data['efficiency_divergence']
    
    # Quality-Adjusted Factor
    data['quality_adjusted_factor'] = data['base_momentum_efficiency'] * data['momentum_quality']
    
    # Final Alpha with Microstructure Adjustment
    data['final_alpha'] = data['quality_adjusted_factor'] * (1 + data['microstructure_pressure'])
    
    # Return the final alpha factor series
    return data['final_alpha']
