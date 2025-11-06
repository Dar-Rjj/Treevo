import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Bidirectional Order Flow Imbalance
    # Micro-level pressure
    micro_pressure = (data['close'] - data['open']) * data['volume'] / (data['high'] - data['low'])
    micro_pressure = micro_pressure.replace([np.inf, -np.inf], np.nan)
    
    # Intraday absorption
    intraday_absorption = ((data['high'] - data['close']) * data['volume'] / (data['high'] - data['low']) - 
                          (data['close'] - data['low']) * data['volume'] / (data['high'] - data['low']))
    intraday_absorption = intraday_absorption.replace([np.inf, -np.inf], np.nan)
    
    # Flow imbalance
    flow_imbalance = micro_pressure + intraday_absorption
    
    # Multi-Timeframe Price Rejection
    # Short-term rejection
    short_term_rejection = ((data['close'] - data['low']) / (data['high'] - data['low']) - 
                           (data['close'].shift(1) - data['low'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)))
    short_term_rejection = short_term_rejection.replace([np.inf, -np.inf], np.nan)
    
    # Medium-term rejection
    # 5-day rolling windows
    high_5d = data['high'].rolling(window=5, min_periods=1).max()
    low_5d = data['low'].rolling(window=5, min_periods=1).min()
    high_5d_shifted = data['high'].shift(5).rolling(window=5, min_periods=1).max()
    low_5d_shifted = data['low'].shift(5).rolling(window=5, min_periods=1).min()
    
    medium_term_rejection = ((data['close'] - low_5d) / (high_5d - low_5d) - 
                            (data['close'].shift(5) - low_5d_shifted) / (high_5d_shifted - low_5d_shifted))
    medium_term_rejection = medium_term_rejection.replace([np.inf, -np.inf], np.nan)
    
    # Rejection momentum
    rejection_momentum = short_term_rejection * medium_term_rejection
    
    # Volume-Value Dislocation
    # Value density
    value_density = data['amount'] / data['volume']
    value_density = value_density.replace([np.inf, -np.inf], np.nan)
    
    # Volume acceleration
    volume_acceleration = (data['volume'] / data['volume'].shift(1) - 
                          data['volume'].shift(1) / data['volume'].shift(2))
    
    # Value momentum
    value_momentum = (value_density / value_density.shift(1)) - 1
    
    # Dislocation factor
    dislocation_factor = value_density * volume_acceleration * value_momentum
    
    # Asymmetric Volatility Response
    # Up-volatility sensitivity
    up_vol_sensitivity = (data['close'] > data['close'].shift(1)).astype(float) * (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    up_vol_sensitivity = up_vol_sensitivity.replace([np.inf, -np.inf], np.nan)
    
    # Down-volatility persistence
    down_vol_persistence = (data['close'] < data['close'].shift(1)).astype(float) * (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5))
    down_vol_persistence = down_vol_persistence.replace([np.inf, -np.inf], np.nan)
    
    # Asymmetric volatility
    asymmetric_volatility = up_vol_sensitivity - down_vol_persistence
    
    # Adaptive Microstructure Synthesis
    # Calculate flow regime based on volume percentile
    volume_percentile = data['volume'].rolling(window=20, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # Initialize final factor
    final_factor = pd.Series(index=data.index, dtype=float)
    
    # Apply different regimes
    high_flow_mask = volume_percentile > 0.7
    low_flow_mask = volume_percentile < 0.3
    normal_flow_mask = ~(high_flow_mask | low_flow_mask)
    
    # High flow regime
    final_factor[high_flow_mask] = (flow_imbalance[high_flow_mask] * rejection_momentum[high_flow_mask] + 
                                   dislocation_factor[high_flow_mask])
    
    # Low flow regime
    final_factor[low_flow_mask] = (flow_imbalance[low_flow_mask] * 0.7 + 
                                  rejection_momentum[low_flow_mask] * 1.3 + 
                                  dislocation_factor[low_flow_mask])
    
    # Normal flow regime
    final_factor[normal_flow_mask] = (flow_imbalance[normal_flow_mask] + 
                                     rejection_momentum[normal_flow_mask] + 
                                     dislocation_factor[normal_flow_mask] + 
                                     asymmetric_volatility[normal_flow_mask])
    
    return final_factor
