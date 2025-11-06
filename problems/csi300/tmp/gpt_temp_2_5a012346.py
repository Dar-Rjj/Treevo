import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Momentum Divergence Factor
    Multi-scale fractal analysis combining price, volume, range, and amount dimensions
    """
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Multi-Fractal Price Momentum
    # Short Fractal (3-day)
    data['price_ret_3d'] = data['close'] / data['close'].shift(3) - 1
    data['price_accel_3d'] = ((data['close'] / data['close'].shift(1)) / 
                             (data['close'].shift(1) / data['close'].shift(2))) - 1
    
    # Medium Fractal (8-day)
    data['price_ret_8d'] = data['close'] / data['close'].shift(8) - 1
    data['price_accel_8d'] = ((data['close'] / data['close'].shift(4)) / 
                             (data['close'].shift(4) / data['close'].shift(8))) - 1
    
    # Long Fractal (21-day)
    data['price_ret_21d'] = data['close'] / data['close'].shift(21) - 1
    data['price_accel_21d'] = ((data['close'] / data['close'].shift(7)) / 
                              (data['close'].shift(7) / data['close'].shift(21))) - 1
    
    # Multi-Fractal Volume Momentum
    # Short Fractal Volume
    data['vol_ret_3d'] = data['volume'] / data['volume'].shift(3) - 1
    data['vol_accel_3d'] = ((data['volume'] / data['volume'].shift(1)) / 
                           (data['volume'].shift(1) / data['volume'].shift(2))) - 1
    
    # Medium Fractal Volume
    data['vol_ret_8d'] = data['volume'] / data['volume'].shift(8) - 1
    data['vol_accel_8d'] = ((data['volume'] / data['volume'].shift(4)) / 
                           (data['volume'].shift(4) / data['volume'].shift(8))) - 1
    
    # Long Fractal Volume
    data['vol_ret_21d'] = data['volume'] / data['volume'].shift(21) - 1
    data['vol_accel_21d'] = ((data['volume'] / data['volume'].shift(7)) / 
                            (data['volume'].shift(7) / data['volume'].shift(21))) - 1
    
    # Fractal Scale Correlation Analysis
    # Cross-Fractal Momentum Consistency
    data['accel_consistency'] = (
        (np.sign(data['price_accel_3d']) == np.sign(data['price_accel_8d'])).astype(int) * 0.4 +
        (np.sign(data['price_accel_3d']) == np.sign(data['price_accel_21d'])).astype(int) * 0.3 +
        (np.sign(data['price_accel_8d']) == np.sign(data['price_accel_21d'])).astype(int) * 0.3
    )
    
    # Fractal Volume-Price Divergence
    data['divergence_3d'] = np.sign(data['price_accel_3d']) != np.sign(data['vol_accel_3d'])
    data['divergence_8d'] = np.sign(data['price_accel_8d']) != np.sign(data['vol_accel_8d'])
    data['divergence_21d'] = np.sign(data['price_accel_21d']) != np.sign(data['vol_accel_21d'])
    
    # Range-Based Momentum Quality Assessment
    # True Range Calculation
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Fractal Range Momentum
    data['range_mom_3d'] = data['tr'] / data['tr'].shift(3) - 1
    data['range_mom_8d'] = data['tr'] / data['tr'].shift(8) - 1
    data['range_mom_21d'] = data['tr'] / data['tr'].shift(21) - 1
    
    # Volume-Range Efficiency
    data['eff_short'] = data['volume'] / data['tr']
    data['eff_medium'] = data['volume'].rolling(window=4).mean() / data['tr'].rolling(window=4).mean()
    data['eff_long'] = data['volume'].rolling(window=9).mean() / data['tr'].rolling(window=9).mean()
    
    # Amount-Volume Fractal Analysis
    data['av_ratio_short'] = data['amount'] / data['volume']
    data['av_ratio_med'] = data['amount'].rolling(window=4).mean() / data['volume'].rolling(window=4).mean()
    data['av_ratio_long'] = data['amount'].rolling(window=9).mean() / data['volume'].rolling(window=9).mean()
    
    data['av_mom_3d'] = data['av_ratio_short'] / data['av_ratio_short'].shift(3) - 1
    data['av_mom_8d'] = data['av_ratio_med'] / data['av_ratio_med'].shift(8) - 1
    data['av_mom_21d'] = data['av_ratio_long'] / data['av_ratio_long'].shift(21) - 1
    
    # Composite Fractal Momentum Factor
    # Multi-Fractal Divergence Synthesis
    data['fractal_divergence_score'] = (
        data['divergence_3d'].astype(float) * 0.5 +
        data['divergence_8d'].astype(float) * 0.3 +
        data['divergence_21d'].astype(float) * 0.2
    )
    
    # Fractal Acceleration Weighting
    data['accel_magnitude'] = (
        np.abs(data['price_accel_3d']) * 0.4 +
        np.abs(data['price_accel_8d']) * 0.35 +
        np.abs(data['price_accel_21d']) * 0.25
    )
    
    # Range-Quality Adjusted Signals
    data['range_efficiency_score'] = (
        (data['eff_short'] / data['eff_short'].rolling(window=10).mean()) * 0.4 +
        (data['eff_medium'] / data['eff_medium'].rolling(window=10).mean()) * 0.35 +
        (data['eff_long'] / data['eff_long'].rolling(window=10).mean()) * 0.25
    )
    
    # Amount-Volume Context Integration
    data['av_context_score'] = (
        data['av_mom_3d'] * 0.4 +
        data['av_mom_8d'] * 0.35 +
        data['av_mom_21d'] * 0.25
    )
    
    # Final Alpha Factor Generation
    # Multi-dimensional Fractal Integration
    for i in range(len(data)):
        if i < 21:  # Ensure enough data for calculations
            result.iloc[i] = 0
            continue
            
        # Dynamic Weighting Scheme
        recent_volatility = data['tr'].iloc[i-5:i+1].std()
        vol_weight = min(1.0, recent_volatility / data['tr'].iloc[i-20:i+1].std())
        
        # Composite factor calculation
        momentum_component = (
            data['price_ret_3d'].iloc[i] * 0.3 +
            data['price_ret_8d'].iloc[i] * 0.4 +
            data['price_ret_21d'].iloc[i] * 0.3
        )
        
        divergence_component = (
            data['fractal_divergence_score'].iloc[i] * 
            data['accel_magnitude'].iloc[i] * 
            data['accel_consistency'].iloc[i]
        )
        
        quality_component = (
            data['range_efficiency_score'].iloc[i] * 
            data['av_context_score'].iloc[i]
        )
        
        # Regime-Adaptive Signal Processing
        result.iloc[i] = (
            momentum_component * 0.4 * vol_weight +
            divergence_component * 0.35 +
            quality_component * 0.25
        )
    
    # Normalize the final factor
    result = (result - result.rolling(window=50, min_periods=21).mean()) / result.rolling(window=50, min_periods=21).std()
    
    return result.fillna(0)
