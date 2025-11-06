import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate intraday momentum components
    data['high_to_close_eff'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['range_utilization'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Calculate volume-price efficiency
    data['amount_per_share'] = data['amount'] / (data['volume'] + 1e-8)
    data['volume_weighted_eff'] = data['high_to_close_eff'] * np.log(data['volume'] + 1)
    data['volume_adjusted_signal'] = data['volume_weighted_eff'] * data['amount_per_share']
    
    # Calculate True Range for volatility regime
    prev_close = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - prev_close),
            np.abs(data['low'] - prev_close)
        )
    )
    
    # Volatility regime detection using rolling median
    data['vol_regime'] = data['true_range'] > data['true_range'].rolling(window=20, min_periods=10).median()
    
    # Detect divergence patterns
    data['volume_trend'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_abnormal'] = data['volume'] > (data['volume_trend'] * 1.5)
    
    # Calculate divergence magnitude and direction
    data['momentum_direction'] = np.sign(data['close'] - data['open'])
    data['divergence_signal'] = data['volume_adjusted_signal'] * data['momentum_direction']
    
    # Apply volume confirmation for divergence
    data['divergence_confirmed'] = data['divergence_signal'] * data['volume_abnormal'].astype(float)
    
    # Calculate pressure accumulation with time decay
    decay_factor = 0.9
    data['buying_pressure'] = np.where(data['close'] > data['open'], 
                                      (data['close'] - data['open']) * data['volume'], 0)
    data['selling_pressure'] = np.where(data['close'] < data['open'], 
                                       (data['open'] - data['close']) * data['volume'], 0)
    
    # Apply exponential decay to pressure accumulation
    data['cumulative_buy_pressure'] = 0.0
    data['cumulative_sell_pressure'] = 0.0
    
    for i in range(1, len(data)):
        data.loc[data.index[i], 'cumulative_buy_pressure'] = (
            data.loc[data.index[i-1], 'cumulative_buy_pressure'] * decay_factor + 
            data.loc[data.index[i], 'buying_pressure']
        )
        data.loc[data.index[i], 'cumulative_sell_pressure'] = (
            data.loc[data.index[i-1], 'cumulative_sell_pressure'] * decay_factor + 
            data.loc[data.index[i], 'selling_pressure']
        )
    
    # Net pressure accumulation
    data['net_pressure'] = data['cumulative_buy_pressure'] - data['cumulative_sell_pressure']
    
    # Generate adaptive signals combining all components
    data['volatility_adjusted_divergence'] = data['divergence_confirmed'] * np.where(
        data['vol_regime'], 0.7, 1.3  # Reduce weight in high volatility
    )
    
    # Final factor combining divergence with pressure accumulation
    data['factor'] = (
        data['volatility_adjusted_divergence'] * 0.6 + 
        np.tanh(data['net_pressure'] / (data['amount'].rolling(window=10, min_periods=5).mean() + 1e-8)) * 0.4
    )
    
    return data['factor']
