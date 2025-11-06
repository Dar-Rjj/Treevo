import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(data):
    # Volatility-Adjusted Trend Acceleration
    def volatility_adjusted_trend_acceleration(close, window=20):
        # Calculate rolling linear regression slope
        def rolling_slope(series, window):
            slopes = np.full(len(series), np.nan)
            x = np.arange(window).reshape(-1, 1)
            for i in range(window, len(series)+1):
                y = series.iloc[i-window:i].values
                if len(y) == window and not np.isnan(y).any():
                    model = LinearRegression()
                    model.fit(x, y)
                    slopes[i-1] = model.coef_[0]
            return pd.Series(slopes, index=series.index)
        
        slope = rolling_slope(close, window)
        # Calculate trend acceleration
        acceleration = slope.diff()
        # Normalize by absolute previous slope (avoid division by zero)
        prev_slope_abs = slope.shift(1).abs()
        normalized_acceleration = acceleration / np.where(prev_slope_abs > 0, prev_slope_abs, 1)
        
        # Calculate rolling volatility
        returns = close.pct_change()
        volatility = returns.rolling(window=window).std()
        
        # Scale trend acceleration by volatility
        volatility_adjusted = normalized_acceleration / np.where(volatility > 0, volatility, 1)
        return volatility_adjusted
    
    # Volume-Price Divergence Momentum
    def volume_price_divergence_momentum(close, high, low, volume, vwap_window=10, momentum_window=5):
        # Calculate VWAP
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(window=vwap_window).sum() / volume.rolling(window=vwap_window).sum()
        
        # Calculate VWAP slope
        def calculate_slope(series, window):
            slopes = np.full(len(series), np.nan)
            x = np.arange(window).reshape(-1, 1)
            for i in range(window, len(series)+1):
                y = series.iloc[i-window:i].values
                if len(y) == window and not np.isnan(y).any():
                    model = LinearRegression()
                    model.fit(x, y)
                    slopes[i-1] = model.coef_[0]
            return pd.Series(slopes, index=series.index)
        
        vwap_slope = calculate_slope(vwap, vwap_window)
        
        # Calculate price slope
        price_slope = calculate_slope(close, vwap_window)
        
        # Calculate divergence
        divergence = vwap_slope - price_slope
        
        # Apply momentum filter
        momentum = divergence - divergence.shift(momentum_window)
        # Multiply by sign of price trend
        result = momentum * np.sign(price_slope)
        return result
    
    # Intraday Pressure Accumulation
    def intraday_pressure_accumulation(open, high, low, close, volume, window=5):
        # Calculate buying and selling pressure
        mid_price = (high + low) / 2
        buying_pressure = np.where(close > mid_price, (close - mid_price) * volume, 0)
        selling_pressure = np.where(close < mid_price, (mid_price - close) * volume, 0)
        
        # Calculate net pressure
        net_pressure = buying_pressure - selling_pressure
        
        # Calculate cumulative net pressure with reset on zero crossing
        cumulative_net = np.zeros(len(net_pressure))
        current_accumulation = 0
        
        for i in range(len(net_pressure)):
            if i == 0:
                current_accumulation = net_pressure[i]
            else:
                # Reset if sign changes
                if np.sign(net_pressure[i]) != np.sign(net_pressure[i-1]) and net_pressure[i-1] != 0:
                    current_accumulation = net_pressure[i]
                else:
                    current_accumulation += net_pressure[i]
            
            cumulative_net[i] = current_accumulation
        
        # Scale by average daily volume
        avg_volume = volume.rolling(window=window).mean()
        scaled_pressure = cumulative_net / np.where(avg_volume > 0, avg_volume, 1)
        
        # Apply hyperbolic tangent for bounded output
        result = np.tanh(scaled_pressure)
        return pd.Series(result, index=close.index)
    
    # Range Efficiency Factor
    def range_efficiency_factor(open, high, low, close, volume, window=5):
        # Calculate true range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate price movement efficiency
        price_move = (close - open).abs()
        price_efficiency = price_move / np.where(true_range > 0, true_range, 1)
        
        # Calculate volume distribution efficiency
        range_mid = (high + low) / 2
        upper_volume = np.where(close > range_mid, volume, 0)
        lower_volume = np.where(close < range_mid, volume, 0)
        
        total_volume = upper_volume + lower_volume
        volume_concentration = np.abs(upper_volume - lower_volume) / np.where(total_volume > 0, total_volume, 1)
        
        # Create composite efficiency score
        composite_score = price_efficiency * volume_concentration
        
        # Apply exponential smoothing
        alpha = 2 / (window + 1)
        smoothed_score = composite_score.ewm(alpha=alpha, adjust=False).mean()
        
        return smoothed_score
    
    # Liquidity Barrier Break Detection
    def liquidity_barrier_break_detection(close, high, low, volume, barrier_window=20, volume_threshold=0.8):
        # Identify high volume days (top 20%)
        volume_quantile = volume.rolling(window=barrier_window).quantile(volume_threshold)
        high_volume_mask = volume > volume_quantile
        
        # Cluster high volume days by price (using close price as proxy)
        barrier_prices = close[high_volume_mask]
        barrier_volumes = volume[high_volume_mask]
        
        # Calculate barrier strength (simplified approach)
        barrier_strength = pd.Series(0.0, index=close.index)
        for i in range(len(close)):
            if i >= barrier_window:
                recent_barriers = barrier_prices.iloc[max(0, i-barrier_window):i]
                recent_volumes = barrier_volumes.iloc[max(0, i-barrier_window):i]
                
                if len(recent_barriers) > 0:
                    # Simple barrier strength based on recent high volume levels
                    current_price = close.iloc[i]
                    distances = np.abs(recent_barriers - current_price)
                    # Weight by volume and recency (simpler decay)
                    weights = recent_volumes.values * np.exp(-distances / (close.iloc[i] * 0.01))
                    barrier_strength.iloc[i] = weights.sum() / volume.iloc[i] if volume.iloc[i] > 0 else 0
        
        # Generate breakout score
        price_change = close.pct_change()
        volume_ratio = volume / volume.rolling(window=5).mean()
        
        # Combine factors for breakout detection
        breakout_score = (price_change * barrier_strength * volume_ratio).fillna(0)
        
        # Apply simple decay
        decayed_score = breakout_score.rolling(window=3).mean()
        
        return decayed_score
    
    # Combine all factors with equal weighting
    factor1 = volatility_adjusted_trend_acceleration(data['close'])
    factor2 = volume_price_divergence_momentum(data['close'], data['high'], data['low'], data['volume'])
    factor3 = intraday_pressure_accumulation(data['open'], data['high'], data['low'], data['close'], data['volume'])
    factor4 = range_efficiency_factor(data['open'], data['high'], data['low'], data['close'], data['volume'])
    factor5 = liquidity_barrier_break_detection(data['close'], data['high'], data['low'], data['volume'])
    
    # Normalize and combine factors
    factors = pd.DataFrame({
        'factor1': factor1,
        'factor2': factor2,
        'factor3': factor3,
        'factor4': factor4,
        'factor5': factor5
    })
    
    # Z-score normalization for each factor
    normalized_factors = factors.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x)
    
    # Equal weighted combination
    combined_factor = normalized_factors.mean(axis=1)
    
    return combined_factor
