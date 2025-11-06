import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Divergence Momentum factor
    """
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Volatility Regime Detection
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Short-Term Volatility (5-day ATR)
    data['short_term_vol'] = data['true_range'].rolling(window=5, min_periods=3).mean()
    
    # Long-Term Volatility (20-day ATR)
    data['long_term_vol'] = data['true_range'].rolling(window=20, min_periods=10).mean()
    
    # Volatility Regime Classification
    data['vol_ratio'] = data['short_term_vol'] / data['long_term_vol']
    data['high_vol_regime'] = data['vol_ratio'] > 1.2
    
    # Price-Volume Momentum Components
    # Price Acceleration: (close_t / close_t-3 - 1) vs (close_t-3 / close_t-6 - 1)
    data['price_accel_short'] = data['close'] / data['close'].shift(3) - 1
    data['price_accel_long'] = data['close'].shift(3) / data['close'].shift(6) - 1
    data['price_acceleration'] = data['price_accel_short'] - data['price_accel_long']
    
    # Volume Acceleration: (volume_t / volume_t-3 - 1) vs (volume_t-3 / volume_t-6 - 1)
    data['volume_accel_short'] = data['volume'] / data['volume'].shift(3) - 1
    data['volume_accel_long'] = data['volume'].shift(3) / data['volume'].shift(6) - 1
    data['volume_acceleration'] = data['volume_accel_short'] - data['volume_accel_long']
    
    # Momentum Divergence
    data['momentum_divergence'] = np.sign(data['price_acceleration'] * data['volume_acceleration'])
    
    # Regime-Adaptive Signal Generation
    factor_values = []
    
    for i in range(len(data)):
        if i < 20:  # Need enough data for calculations
            factor_values.append(0)
            continue
            
        current_row = data.iloc[i]
        high_vol_regime = current_row['high_vol_regime']
        
        if high_vol_regime:
            # High Volatility regime - 3-day momentum with strong filters
            price_accel_threshold = 0.05
            volume_accel_threshold = 0.15
            
            if (abs(current_row['price_acceleration']) > price_accel_threshold and 
                abs(current_row['volume_acceleration']) > volume_accel_threshold):
                signal = current_row['momentum_divergence']
            else:
                signal = 0
        else:
            # Low Volatility regime - 10-day momentum with weaker filters
            # Calculate 10-day momentum components
            if i >= 30:  # Need enough data for 10-day lookback
                price_accel_10d_short = current_row['close'] / data.iloc[i-10]['close'] - 1
                price_accel_10d_long = data.iloc[i-10]['close'] / data.iloc[i-20]['close'] - 1
                price_accel_10d = price_accel_10d_short - price_accel_10d_long
                
                volume_accel_10d_short = current_row['volume'] / data.iloc[i-10]['volume'] - 1
                volume_accel_10d_long = data.iloc[i-10]['volume'] / data.iloc[i-20]['volume'] - 1
                volume_accel_10d = volume_accel_10d_short - volume_accel_10d_long
                
                price_accel_threshold = 0.02
                volume_accel_threshold = 0.08
                
                if (abs(price_accel_10d) > price_accel_threshold and 
                    abs(volume_accel_10d) > volume_accel_threshold):
                    signal = np.sign(price_accel_10d * volume_accel_10d)
                else:
                    signal = 0
            else:
                signal = 0
        
        # Volume-Volatility Weighting
        volume_vol_ratio = current_row['volume'] / current_row['short_term_vol'] if current_row['short_term_vol'] > 0 else 0
        weighted_signal = signal * np.log1p(volume_vol_ratio)  # Use log to handle scaling
        
        factor_values.append(weighted_signal)
    
    # Create output series
    factor_series = pd.Series(factor_values, index=data.index, name='regime_adaptive_divergence_momentum')
    
    # Clean up any infinite or NaN values
    factor_series = factor_series.replace([np.inf, -np.inf], 0).fillna(0)
    
    return factor_series
