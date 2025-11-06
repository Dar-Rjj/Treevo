import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Volatility Adjusted Momentum
    intraday_momentum = (data['high'] - data['low']) / data['close']
    vol_5d = data['close'].rolling(window=5).std()
    factor1 = intraday_momentum / (vol_5d + 1e-8)
    
    # Volume-Scaled Price Reversal
    price_reversal = -data['close'].pct_change(1)
    volume_ratio = data['volume'] / data['volume'].rolling(window=20).mean()
    factor2 = price_reversal * volume_ratio
    
    # Amplitude-Weighted Trend Strength
    price_amplitude = (data['high'] - data['low']) / (data['open'] + 1e-8)
    
    def linear_regression_slope(series):
        if len(series) < 10:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    trend_10d = data['close'].rolling(window=10).apply(linear_regression_slope, raw=False)
    factor3 = trend_10d * price_amplitude
    
    # Volume-Price Divergence Factor
    volume_momentum = data['volume'].pct_change(5)
    price_momentum = data['close'].pct_change(5)
    factor4 = volume_momentum - price_momentum
    
    # Efficiency Ratio Adjusted Return
    net_change_10d = data['close'] - data['close'].shift(10)
    total_movement_10d = data['close'].pct_change(1).abs().rolling(window=10).sum() * data['close'].shift(10)
    efficiency_ratio = net_change_10d / (total_movement_10d + 1e-8)
    return_3d = data['close'].pct_change(3)
    factor5 = return_3d * efficiency_ratio
    
    # Pressure-Based Reversal Indicator
    buying_pressure = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    pressure_volume = buying_pressure * data['volume']
    price_reversal_1d = -data['close'].pct_change(1)
    factor6 = pressure_volume * price_reversal_1d
    
    # Range Breakout Confidence
    high_20d = data['high'].rolling(window=20).max()
    breakout_signal = (data['high'] > high_20d.shift(1)).astype(float)
    breakout_magnitude = (data['high'] - high_20d.shift(1)) / (high_20d.shift(1) + 1e-8)
    volume_confirmation = data['volume'] / data['volume'].rolling(window=20).mean()
    factor7 = breakout_signal * breakout_magnitude * volume_confirmation
    
    # Liquidity-Adjusted Momentum
    momentum_10d = data['close'].pct_change(10)
    liquidity_measure = data['volume'] / (data['amount'] + 1e-8)
    factor8 = momentum_10d * liquidity_measure
    
    # Volatility Regime Adaptive Factor
    atr_20d = ((data['high'] - data['low']).rolling(window=20).mean() + 
               (data['high'] - data['close'].shift(1)).abs().rolling(window=20).mean() + 
               (data['low'] - data['close'].shift(1)).abs().rolling(window=20).mean()) / 3
    
    volatility_threshold = atr_20d.median()
    ma_5d = data['close'].rolling(window=5).mean()
    deviation = (data['close'] - ma_5d) / (ma_5d + 1e-8)
    
    # High volatility: mean reversion (inverse relationship)
    high_vol_component = -deviation * (atr_20d > volatility_threshold)
    
    # Low volatility: momentum (direct relationship)
    price_acceleration = data['close'].pct_change(3) - data['close'].pct_change(6)
    low_vol_component = price_acceleration * (atr_20d <= volatility_threshold)
    
    factor9 = high_vol_component + low_vol_component
    
    # Volume-Weighted Price Levels
    def find_key_levels(high_series, low_series, volume_series, window=20):
        levels = pd.Series(index=high_series.index, dtype=float)
        
        for i in range(window, len(high_series)):
            window_high = high_series.iloc[i-window:i]
            window_low = low_series.iloc[i-window:i]
            window_volume = volume_series.iloc[i-window:i]
            
            # Find high volume days (top 20%)
            volume_threshold = window_volume.quantile(0.8)
            high_volume_mask = window_volume > volume_threshold
            
            if high_volume_mask.sum() > 0:
                # Key resistance from high prices on high volume days
                resistance_level = window_high[high_volume_mask].max()
                # Key support from low prices on high volume days
                support_level = window_low[high_volume_mask].min()
                
                current_price = high_series.iloc[i]
                distance_to_resistance = abs(current_price - resistance_level) / resistance_level
                distance_to_support = abs(current_price - support_level) / support_level
                
                # Use the closer level
                min_distance = min(distance_to_resistance, distance_to_support)
                levels.iloc[i] = min_distance
            else:
                levels.iloc[i] = np.nan
                
        return levels
    
    key_level_distance = find_key_levels(data['high'], data['low'], data['volume'])
    current_volume_activity = data['volume'] / data['volume'].rolling(window=20).mean()
    factor10 = -key_level_distance * current_volume_activity  # Negative because closer to key levels might mean reversal
    
    # Combine all factors (equal weighting for simplicity)
    combined_factor = (
        factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0) + 
        factor4.fillna(0) + factor5.fillna(0) + factor6.fillna(0) + 
        factor7.fillna(0) + factor8.fillna(0) + factor9.fillna(0) + 
        factor10.fillna(0)
    )
    
    return combined_factor
