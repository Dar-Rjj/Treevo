import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Momentum Asymmetry
    # Range Skew Momentum
    close_to_close_vol_4d = data['close'].pct_change().rolling(window=4).std()
    range_skew_momentum = ((data['high'] - data['low']) / close_to_close_vol_4d) * (data['close'] / data['close'].shift(5) - 1)
    
    # Curvature Acceleration
    high_low_range = data['high'] - data['low']
    range_5d_curvature = high_low_range.rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0, raw=True)
    range_20d_curvature = high_low_range.rolling(window=20).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0, raw=True)
    curvature_acceleration = (range_5d_curvature - range_20d_curvature) / range_5d_curvature.replace(0, np.nan)
    
    # Liquidity-Efficiency Dynamics
    # Open-Close Reversal Efficiency
    volume_ratio = data['volume'] / data['volume'].shift(5) - 1
    open_close_reversal_eff = ((data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)) * np.sign(volume_ratio)
    
    # Volume-Price Divergence Speed
    volume_change = data['volume'] / data['volume'].shift(5) - 1
    price_change = data['close'] / data['close'].shift(5) - 1
    volume_price_divergence = volume_change / price_change.replace(0, np.nan)
    
    # Multi-Scale Information Hierarchy
    # Volume-Return Correlation Decay
    def rolling_corr_volume_return(window):
        if len(window) < 2:
            return np.nan
        volume_data = data['volume'].loc[window.index]
        returns = data['close'].loc[window.index] / data['close'].shift(1).loc[window.index] - 1
        return volume_data.corr(returns)
    
    volume_return_corr = data['close'].rolling(window=5).apply(rolling_corr_volume_return, raw=False)
    volume_return_corr_decay = volume_return_corr * (data['close'] / data['close'].shift(5) - 1)
    
    # Structural Break Efficiency Validation
    # Support/Resistance Penetration Momentum
    rolling_low_4d = data['low'].rolling(window=5).min()
    rolling_high_4d = data['high'].rolling(window=5).max()
    support_resistance_momentum = ((data['close'] - rolling_low_4d) / (rolling_high_4d - rolling_low_4d).replace(0, np.nan)) * data['volume']
    
    # Break Persistence Acceleration
    price_change_5d = data['close'] / data['close'].shift(5) - 1
    price_change_20d = data['close'] / data['close'].shift(20) - 1
    daily_direction = np.sign(data['close'] - data['close'].shift(1))
    break_persistence_acceleration = price_change_5d * price_change_20d * daily_direction
    
    # Combine all components with equal weights
    factor = (range_skew_momentum.fillna(0) + 
              curvature_acceleration.fillna(0) + 
              open_close_reversal_eff.fillna(0) + 
              volume_price_divergence.fillna(0) + 
              volume_return_corr_decay.fillna(0) + 
              support_resistance_momentum.fillna(0) + 
              break_persistence_acceleration.fillna(0))
    
    return factor
