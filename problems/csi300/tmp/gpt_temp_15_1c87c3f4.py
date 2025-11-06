import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Price Fractal Efficiency Factor
    Combines price movement efficiency with volume pattern complexity using fractal analysis
    """
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # 5-day Cumulative True Range
    data['cumulative_tr_5d'] = data['true_range'].rolling(window=5).sum()
    
    # Price Efficiency Components
    data['net_price_change_5d'] = data['close'] - data['close'].shift(5)
    data['abs_price_movement_5d'] = abs(data['close'] - data['close'].shift(1)).rolling(window=5).sum()
    
    # Efficiency Ratio
    data['efficiency_ratio_5d'] = abs(data['net_price_change_5d']) / (data['abs_price_movement_5d'] + 1e-8)
    
    # Volume Analysis
    data['volume_range'] = data['volume'] - data['volume'].shift(1)
    data['volume_volatility_20d'] = data['volume'].rolling(window=20).std()
    
    # Simplified Volume Fractal Dimension Estimation
    def calculate_volume_fractal(volume_series, window=20):
        if len(volume_series) < window:
            return np.nan
        
        # Calculate rescaled range for different time scales
        scales = [5, 10, 15, 20]
        rs_values = []
        
        for scale in scales:
            if len(volume_series) >= scale:
                # Calculate mean and cumulative deviation
                mean_vol = volume_series[-scale:].mean()
                deviations = volume_series[-scale:] - mean_vol
                cumulative_dev = deviations.cumsum()
                
                # Range and standard deviation
                R = cumulative_dev.max() - cumulative_dev.min()
                S = volume_series[-scale:].std()
                
                if S > 0:
                    rs_values.append(R / S)
        
        if len(rs_values) >= 2:
            # Simple Hurst exponent estimation
            hurst = np.polyfit(np.log([5, 10, 15, 20][:len(rs_values)]), 
                              np.log(rs_values), 1)[0]
            fractal_dim = 2 - hurst
            return fractal_dim
        return np.nan
    
    # Calculate volume fractal dimension
    volume_fractal = []
    for i in range(len(data)):
        if i >= 20:
            vol_window = data['volume'].iloc[i-19:i+1]
            fractal_dim = calculate_volume_fractal(vol_window)
            volume_fractal.append(fractal_dim)
        else:
            volume_fractal.append(np.nan)
    
    data['volume_fractal_dim'] = volume_fractal
    
    # Multi-timeframe Efficiency Analysis
    data['efficiency_ratio_3d'] = abs(data['close'] - data['close'].shift(3)) / (
        abs(data['close'] - data['close'].shift(1)).rolling(window=3).sum() + 1e-8)
    
    data['efficiency_ratio_10d'] = abs(data['close'] - data['close'].shift(10)) / (
        abs(data['close'] - data['close'].shift(1)).rolling(window=10).sum() + 1e-8)
    
    data['efficiency_ratio_20d'] = abs(data['close'] - data['close'].shift(20)) / (
        abs(data['close'] - data['close'].shift(1)).rolling(window=20).sum() + 1e-8)
    
    # Volume-Price Fractal Correlation
    data['efficiency_trend'] = (data['efficiency_ratio_3d'] + 
                               data['efficiency_ratio_10d'] + 
                               data['efficiency_ratio_20d']) / 3
    
    # Fractal Efficiency Momentum
    data['fractal_efficiency_momentum'] = (
        data['efficiency_ratio_5d'] * (1 + data['volume_fractal_dim'])
    )
    
    # Regime Detection
    data['price_trend_strength'] = data['close'].rolling(window=10).std() / data['close'].rolling(window=10).mean()
    data['volume_trend_strength'] = data['volume'].rolling(window=10).std() / data['volume'].rolling(window=10).mean()
    
    # Signal Weighting based on Regime
    trending_regime = (data['price_trend_strength'] > data['price_trend_strength'].rolling(window=20).mean()) & \
                     (data['volume_trend_strength'] > data['volume_trend_strength'].rolling(window=20).mean())
    
    mean_reverting_regime = (data['price_trend_strength'] < data['price_trend_strength'].rolling(window=20).mean()) & \
                           (data['volume_trend_strength'] < data['volume_trend_strength'].rolling(window=20).mean())
    
    # Final Factor Calculation
    data['fractal_alpha'] = np.where(
        trending_regime,
        data['fractal_efficiency_momentum'] * 1.5,  # Emphasize efficiency in trending markets
        np.where(
            mean_reverting_regime,
            data['volume_fractal_dim'] * data['fractal_efficiency_momentum'],  # Balance both in mean-reverting
            data['fractal_efficiency_momentum']  # Neutral weighting in transition periods
        )
    )
    
    # Normalize the factor
    data['fractal_alpha_normalized'] = (
        data['fractal_alpha'] - data['fractal_alpha'].rolling(window=20).mean()
    ) / (data['fractal_alpha'].rolling(window=20).std() + 1e-8)
    
    return data['fractal_alpha_normalized']
