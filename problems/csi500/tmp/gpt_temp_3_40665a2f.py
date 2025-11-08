import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Fractal Momentum Divergence Alpha Factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate Multi-Timeframe Fractal Momentum
    # Short-Term Fractal Momentum (5-day)
    def calculate_fractal_momentum_5d(data):
        if len(data) < 5:
            return np.nan
        # Price path efficiency: (close - open) / (sum of daily ranges)
        path_efficiency = (data['close'].iloc[-1] - data['close'].iloc[0]) / \
                         (data['high'] - data['low']).sum()
        
        # Fractal dimension change using high-low range complexity
        daily_ranges = data['high'] - data['low']
        range_std = daily_ranges.std()
        range_mean = daily_ranges.mean()
        fractal_complexity = range_std / range_mean if range_mean > 0 else 1.0
        
        return path_efficiency * fractal_complexity
    
    # Medium-Term Fractal Momentum (10-day)
    def calculate_fractal_momentum_10d(data):
        if len(data) < 10:
            return np.nan
        # Price range complexity using coefficient of variation
        daily_ranges = data['high'] - data['low']
        range_cv = daily_ranges.std() / daily_ranges.mean() if daily_ranges.mean() > 0 else 1.0
        
        # Fractal efficiency ratio: net price change / total price movement
        net_change = data['close'].iloc[-1] - data['close'].iloc[0]
        total_movement = abs(data['close'].diff()).sum()
        efficiency_ratio = net_change / total_movement if total_movement > 0 else 0
        
        return range_cv * efficiency_ratio
    
    # Calculate fractal momentum acceleration
    def calculate_acceleration(momentum_series, short_window=3, medium_window=5):
        if len(momentum_series) < max(short_window, medium_window) + 1:
            return np.nan, np.nan
        
        short_accel = momentum_series.diff(short_window).iloc[-1] if len(momentum_series) >= short_window + 1 else 0
        medium_accel = momentum_series.diff(medium_window).iloc[-1] if len(momentum_series) >= medium_window + 1 else 0
        
        return short_accel, medium_accel
    
    # Volume Fractal Characteristics
    def calculate_volume_fractal(data):
        if len(data) < 10:
            return np.nan
        
        volume_data = data['volume']
        
        # Volume clustering intensity using rolling z-score
        volume_mean = volume_data.rolling(window=5, min_periods=3).mean()
        volume_std = volume_data.rolling(window=5, min_periods=3).std()
        volume_zscore = (volume_data - volume_mean) / volume_std
        clustering_intensity = volume_zscore.abs().mean()
        
        # Volume burst persistence ratio
        volume_above_mean = (volume_data > volume_data.rolling(window=10, min_periods=5).mean()).astype(int)
        burst_persistence = volume_above_mean.rolling(window=5, min_periods=3).mean().iloc[-1]
        
        return clustering_intensity * burst_persistence
    
    # Volatility regime calculation
    def calculate_volatility_regime(data):
        if len(data) < 10:
            return 1.0
        
        # Average True Range
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - data['close'].shift(1))
        low_close_prev = abs(data['low'] - data['close'].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=10, min_periods=5).mean().iloc[-1]
        
        # Volatility regime classification
        avg_range = (data['high'] - data['low']).mean()
        volatility_ratio = atr / avg_range if avg_range > 0 else 1.0
        
        # Regime effectiveness weight (lower weight in extreme volatility)
        regime_weight = 1.0 / (1.0 + abs(volatility_ratio - 1.0))
        
        return regime_weight
    
    # Main calculation loop
    for i in range(len(df)):
        if i < 20:  # Need sufficient history
            alpha.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]
        
        # Calculate fractal momentums
        st_data = current_data.tail(5)
        mt_data = current_data.tail(10)
        
        st_momentum = calculate_fractal_momentum_5d(st_data)
        mt_momentum = calculate_fractal_momentum_10d(mt_data)
        
        if pd.isna(st_momentum) or pd.isna(mt_momentum):
            alpha.iloc[i] = 0
            continue
        
        # Create momentum series for acceleration calculation
        st_momentum_series = pd.Series([
            calculate_fractal_momentum_5d(current_data.iloc[j-4:j+1]) 
            for j in range(max(4, i-9), i+1) if j >= 4
        ])
        
        mt_momentum_series = pd.Series([
            calculate_fractal_momentum_10d(current_data.iloc[j-9:j+1]) 
            for j in range(max(9, i-14), i+1) if j >= 9
        ])
        
        # Calculate accelerations
        st_accel, mt_accel = calculate_acceleration(st_momentum_series, 3, 5)
        
        # Volume fractal weighting
        volume_data = current_data.tail(10)
        volume_fractal = calculate_volume_fractal(volume_data)
        
        if pd.isna(volume_fractal):
            volume_fractal = 1.0
        
        # Weight fractal acceleration signals
        st_weighted = st_accel * volume_fractal
        mt_weighted = mt_accel * volume_fractal
        
        # Detect fractal momentum divergence patterns
        price_momentum_trend = st_momentum - calculate_fractal_momentum_5d(current_data.iloc[i-5:i]) if i >= 5 else 0
        
        # Divergence conditions
        bullish_divergence = (price_momentum_trend < 0) and (st_weighted > 0)
        bearish_divergence = (price_momentum_trend > 0) and (st_weighted < 0)
        
        # Generate divergence signal strength
        divergence_strength = 0
        if bullish_divergence:
            divergence_strength = st_weighted * abs(price_momentum_trend)
        elif bearish_divergence:
            divergence_strength = st_weighted * abs(price_momentum_trend)
        
        # Volatility regime adaptation
        volatility_weight = calculate_volatility_regime(current_data.tail(15))
        
        # Fractal convergence strength (agreement between timeframes)
        convergence_strength = 1.0 if st_weighted * mt_weighted > 0 else 0.5
        
        # Final alpha construction
        final_signal = (st_weighted + 0.7 * mt_weighted) * divergence_strength * volatility_weight * convergence_strength
        
        alpha.iloc[i] = final_signal
    
    # Normalize the final alpha
    alpha = (alpha - alpha.rolling(window=20, min_periods=10).mean()) / \
            alpha.rolling(window=20, min_periods=10).std()
    
    return alpha.fillna(0)
