import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Volatility Regime Detection
    # Daily Volatility Proxy
    data['normalized_range'] = (data['high'] - data['low']) / data['close']
    data['volatility_10d'] = data['normalized_range'].rolling(window=10).std()
    
    # Regime Classification
    data['volatility_percentile'] = data['volatility_10d'].rolling(window=30).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 80)) * 2 + 
                  (np.percentile(x, 20) <= x.iloc[-1] <= np.percentile(x, 80)) * 1, 
        raw=False
    )
    
    # Multi-Timeframe Momentum Construction
    # Short Horizon (3-day)
    data['return_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_short'] = data['return_3d'] / (data['volatility_10d'] + 0.0001)
    
    # Medium Horizon (8-day)
    data['return_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_medium'] = data['return_8d'] / (data['volatility_10d'] + 0.0001)
    
    # Long Horizon (21-day)
    data['return_21d'] = data['close'] / data['close'].shift(21) - 1
    data['momentum_long'] = data['return_21d'] / (data['volatility_10d'] + 0.0001)
    
    # Volume-Price Convergence Signal
    # Price Movement Quality
    data['directional_consistency'] = np.sign(data['return_3d']) * np.sign(data['return_8d'])
    data['magnitude_ratio'] = np.abs(data['return_3d']) / (np.abs(data['return_8d']) + 0.0001)
    
    # Volume Confirmation
    data['volume_trend'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_price_alignment'] = np.sign(data['volume_trend']) * np.sign(data['return_3d'])
    
    # Convergence Score
    data['raw_convergence'] = data['directional_consistency'] * data['magnitude_ratio'] * data['volume_price_alignment']
    data['convergence_score'] = np.tanh(data['raw_convergence'])
    
    # Adaptive Factor Integration
    # Momentum Combination
    data['weighted_product'] = (
        np.sign(data['momentum_short']) * np.abs(data['momentum_short'] + 1e-8) ** 0.4 *
        np.sign(data['momentum_medium']) * np.abs(data['momentum_medium'] + 1e-8) ** 0.35 *
        np.sign(data['momentum_long']) * np.abs(data['momentum_long'] + 1e-8) ** 0.25
    )
    data['momentum_combination'] = np.sign(data['weighted_product']) * np.abs(data['weighted_product']) ** (1/3)
    
    # Final Signal Construction
    data['core_factor'] = data['momentum_combination'] * data['convergence_score']
    data['volatility_adjusted'] = data['core_factor'] / (data['volatility_10d'] + 0.0001)
    
    # Regime-Specific Enhancement
    regime_multiplier = np.select(
        [data['volatility_percentile'] == 2, data['volatility_percentile'] == 1, data['volatility_percentile'] == 0],
        [0.7, 1.0, 1.3],
        default=1.0
    )
    
    data['final_factor'] = data['volatility_adjusted'] * regime_multiplier
    
    return data['final_factor']
