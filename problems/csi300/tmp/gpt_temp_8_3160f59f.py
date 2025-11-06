import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Scale Volatility Regime Classification
    # Compute fractal volatility ratios
    data['vol_3d'] = data['close'].pct_change().rolling(window=3).std()
    data['vol_10d'] = data['close'].pct_change().rolling(window=10).std()
    data['vol_5d'] = data['close'].pct_change().rolling(window=5).std()
    data['vol_20d'] = data['close'].pct_change().rolling(window=20).std()
    data['vol_1d'] = data['close'].pct_change().rolling(window=1).std()
    
    data['vol_ratio_3_10'] = data['vol_3d'] / data['vol_10d']
    data['vol_ratio_5_20'] = data['vol_5d'] / data['vol_20d']
    data['vol_ratio_1_5'] = data['vol_1d'] / data['vol_5d']
    
    # Define multi-horizon regimes
    def classify_regime(row):
        if (row['vol_ratio_3_10'] > 1.5 or 
            row['vol_ratio_5_20'] > 1.5 or 
            row['vol_ratio_1_5'] > 1.5):
            return 'high'
        elif (row['vol_ratio_3_10'] < 0.67 or 
              row['vol_ratio_5_20'] < 0.67 or 
              row['vol_ratio_1_5'] < 0.67):
            return 'low'
        else:
            return 'normal'
    
    data['vol_regime'] = data.apply(classify_regime, axis=1)
    
    # Fractal Momentum Divergence Analysis
    # Multi-timeframe momentum calculation
    data['mom_3d'] = data['close'] / data['close'].shift(3) - 1
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    data['mom_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Momentum divergence patterns
    data['mom_div_short'] = abs(data['mom_3d'] - data['mom_5d'])
    data['mom_div_medium'] = abs(data['mom_5d'] - data['mom_10d'])
    
    # Fractal divergence persistence
    data['mom_div_persistence'] = (data['mom_div_short'].rolling(window=3).mean() + 
                                  data['mom_div_medium'].rolling(window=3).mean()) / 2
    
    # Volume-Price Fractal Asymmetry
    # Volume concentration analysis
    def calculate_volume_ratios(window_data):
        up_volume = 0
        total_volume = 0
        for i in range(len(window_data)):
            if window_data['close'].iloc[i] > window_data['open'].iloc[i]:
                up_volume += window_data['volume'].iloc[i]
            total_volume += window_data['volume'].iloc[i]
        
        if total_volume == 0:
            return 0, 0, 0
        
        up_ratio = up_volume / total_volume
        down_ratio = 1 - up_ratio
        directional_bias = abs(up_ratio - down_ratio)
        return up_ratio, down_ratio, directional_bias
    
    # Calculate rolling volume ratios
    volume_metrics = []
    for i in range(len(data)):
        if i >= 2:
            window_data = data.iloc[i-2:i+1]
            up_ratio, down_ratio, directional_bias = calculate_volume_ratios(window_data)
            volume_metrics.append((up_ratio, down_ratio, directional_bias))
        else:
            volume_metrics.append((0, 0, 0))
    
    data[['up_volume_ratio', 'down_volume_ratio', 'volume_directional_bias']] = volume_metrics
    
    # Price efficiency divergence
    data['daily_range_completion'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['vwap'] = data['amount'] / (data['volume'] + 1e-8)
    data['price_amount_divergence'] = abs(data['close'] - data['vwap']) / (data['close'] + 1e-8)
    
    # Multi-day efficiency persistence
    data['efficiency_persistence'] = (data['daily_range_completion'].rolling(window=3).mean() + 
                                     data['price_amount_divergence'].rolling(window=3).mean()) / 2
    
    # Combined fractal asymmetry
    data['momentum_volume_asymmetry'] = data['mom_div_persistence'] * data['volume_directional_bias']
    data['efficiency_volume_asymmetry'] = data['efficiency_persistence'] * data['up_volume_ratio']
    
    # Fractal Regime-Momentum-Volume Alignment
    # Calculate momentum acceleration
    data['momentum_acceleration_1d'] = (data['close'] - data['close'].shift(1)) - (data['close'].shift(1) - data['close'].shift(2))
    data['momentum_acceleration_3d'] = (data['close'] - data['close'].shift(3)) - (data['close'].shift(3) - data['close'].shift(6))
    data['momentum_acceleration_5d'] = (data['close'] - data['close'].shift(5)) - (data['close'].shift(5) - data['close'].shift(10))
    
    # Regime-Adaptive Fractal Divergence Signals
    def calculate_regime_signal(row):
        if row['vol_regime'] == 'high':
            signal = (0.6 * row['momentum_volume_asymmetry'] + 
                     0.3 * row['price_amount_divergence'] + 
                     0.1 * row['momentum_acceleration_1d'])
        elif row['vol_regime'] == 'normal':
            signal = (0.4 * row['momentum_volume_asymmetry'] + 
                     0.4 * row['price_amount_divergence'] + 
                     0.2 * row['momentum_acceleration_3d'])
        else:  # low volatility
            signal = (0.2 * row['momentum_volume_asymmetry'] + 
                     0.5 * row['price_amount_divergence'] + 
                     0.3 * row['momentum_acceleration_5d'])
        return signal
    
    data['fractal_factor'] = data.apply(calculate_regime_signal, axis=1)
    
    # Return the factor series
    return data['fractal_factor']
